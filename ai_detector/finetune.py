"""Sequential, resumable WildFC fine-tuning followed by held-out stacking."""

import argparse
from collections import Counter
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from .config import ROOT, CACHE, UP, MODELS, configure_environment
from .manifests import manifest_digest

DEFAULT_OUTPUT = ROOT / "reports/finetuned_meta"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def prepare_split(original, output):
    """Use only original training groups to reserve a separate stacking partition."""
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold

    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "split_summary.json"
    fingerprint = manifest_digest(original)
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary["original_manifest_sha256"] != fingerprint:
            raise ValueError("Original split changed; choose a new output directory")
        for name, digest in summary["manifests"].items():
            if manifest_digest(output / name) != digest:
                raise ValueError("Saved fine-tuning split was modified")
        return summary
    rows = read_rows(original)
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    assert len(train) + len(test) == len(rows) and train and test
    for key in ("sha256", "pixel_sha256", "leakage_group"):
        assert not {r[key] for r in train} & {r[key] for r in test}, key
    sources = np.array([r["group"] for r in train])
    groups = np.array([r["leakage_group"] for r in train])
    labels = np.array([r["label"] for r in train])
    source_groups = {s: set(groups[sources == s]) for s in sorted(set(sources))}
    group_sizes = Counter(groups)
    candidates = []
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for _, indices in StratifiedGroupKFold(5, shuffle=True, random_state=43).split(
            np.zeros(len(train)), sources, groups
        ):
            selected = set(groups[indices])
            missing = sum(
                not 0 < len(selected & gs) < len(gs)
                for gs in source_groups.values()
                if len(gs) > 1
            )
            imbalance = sum(
                abs(sum(labels[indices] == c) - 0.2 * sum(labels == c)) for c in (0, 1)
            )
            candidates.append((missing * 1e6 + imbalance, selected))
    selected = min(candidates, key=lambda x: x[0])[1]
    for source, gs in sorted(source_groups.items(), key=lambda x: (len(x[1]), x[0])):
        if len(gs) == 1:
            selected -= gs
        elif not gs & selected:
            options = [
                g
                for g in gs
                if all(
                    len(other - selected - {g}) >= 1
                    for other in source_groups.values()
                    if g in other
                )
            ]
            if not options:
                raise ValueError(f"Cannot split source safely: {source}")
            selected.add(min(options, key=lambda g: (group_sizes[g], g)))
        elif not gs - selected:
            options = [
                g
                for g in gs
                if all(
                    len((other & selected) - {g}) >= 1
                    for other in source_groups.values()
                    if g in other and len(other) > 1
                )
            ]
            if not options:
                raise ValueError(f"Cannot split source safely: {source}")
            selected.remove(min(options, key=lambda g: (group_sizes[g], g)))
    base = [
        dict(r, training_role="detector")
        for r in train
        if r["leakage_group"] not in selected
    ]
    stack = [
        dict(r, training_role="stacking")
        for r in train
        if r["leakage_group"] in selected
    ]
    for subset in (base, stack, test):
        assert {r["label"] for r in subset} == {0, 1}
    for key in ("sha256", "pixel_sha256", "leakage_group"):
        assert not {r[key] for r in base} & {r[key] for r in stack + test}
        assert not {r[key] for r in stack} & {r[key] for r in test}
    parts = {
        "detector_train.jsonl": base,
        "stacking_train.jsonl": stack,
        "test.jsonl": test,
        "stacking_evaluation.jsonl": stack + test,
    }
    hashes = {}
    for name, part in parts.items():
        path = output / name
        path.write_text("".join(json.dumps(r) + "\n" for r in part))
        hashes[name] = manifest_digest(path)
    summary = {
        "original_manifest_sha256": fingerprint,
        "manifests": hashes,
        "seed": 43,
        "counts": {n: len(r) for n, r in parts.items()},
        "labels": {n: dict(Counter(r["label"] for r in rs)) for n, rs in parts.items()},
        "policy": "Original test unchanged. Original training groups partitioned into detector fitting (~80%) and stacking fitting (~20%). No validation tuning.",
        "sources": {
            s: {
                "detector": sum(r["group"] == s for r in base),
                "stacking": sum(r["group"] == s for r in stack),
                "independent_groups": len(gs),
            }
            for s, gs in source_groups.items()
        },
    }
    write_json(summary_path, summary)
    return summary


def base_signature(name):
    from .config import COMMUNITY_REPO, COMMUNITY_REVISION

    if name == "community_forensics":
        from huggingface_hub import hf_hub_download

        paths = [
            Path(
                hf_hub_download(
                    COMMUNITY_REPO,
                    "model.safetensors",
                    revision=COMMUNITY_REVISION,
                    cache_dir=str(CACHE / "huggingface"),
                    local_files_only=True,
                )
            )
        ]
    else:
        paths = {
            "safe": [CACHE / "detectors/safe/checkpoint-best.pth"],
            "universalfakedetect": [
                CACHE / "clip/ViT-L-14.pt",
                CACHE / "detectors/universalfakedetect/fc_weights.pth",
            ],
            "dda": [CACHE / "detectors/dda/DDA_ckpt.pth"],
        }.get(name, [CACHE / f'detectors/aide/{name.removeprefix("aide_")}_train.pth'])
    result = {}
    for path in paths:
        with path.open("rb") as stream:
            result[str(path.relative_to(CACHE))] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
    return result


def updated_state(model):
    # Save trainable parameters and every buffer (including BatchNorm statistics).
    keys = {n for n, p in model.named_parameters() if p.requires_grad} | {
        n for n, _ in model.named_buffers()
    }
    return {
        k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in keys
    }


def apply_updates(model, checkpoint):
    state = checkpoint["updates"]
    current = model.state_dict()
    if not state or any(
        k not in current or current[k].shape != v.shape for k, v in state.items()
    ):
        raise ValueError("Fine-tuned checkpoint architecture mismatch")
    model.load_state_dict(state, strict=False)


def save_torch(path, data):
    import torch

    temporary = path.with_suffix(".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def seed_all(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def generated_image_limit(values):
    """Allow JPEG round-trips of already decoded, internally resized DDA views.

    Source files are still opened under Pillow's normal pixel limit. The loader
    runs with num_workers=0; restore this process-wide setting even on failure.
    """
    from PIL import Image

    previous = Image.MAX_IMAGE_PIXELS
    images = [im for value in values.values()
              for im in (value if isinstance(value, list) else [value])]
    if previous is not None:
        Image.MAX_IMAGE_PIXELS = max(previous, *(im.width * im.height for im in images))
    try:
        yield
    finally:
        Image.MAX_IMAGE_PIXELS = previous


class Images:
    def __init__(self, rows, transform, name):
        self.rows, self.transform, self.name = rows, transform, name
        self.epoch, self.draw = 0, 0
        self.real = [r for r in rows if r["label"] == 0]
        self.fake = [r for r in rows if r["label"] == 1]

    def __len__(self):
        return len(self.real) if self.name == "dda" else len(self.rows)

    def open(self, row):
        from PIL import Image, ImageOps

        with Image.open(ROOT / row["path"]) as im:
            if self.name == "community_forensics":
                im = ImageOps.exif_transpose(im)
            return im.convert("RGB")

    def __getitem__(self, index):
        import torch
        import numpy as np

        seed = 42 + self.epoch * 1000000 + self.draw
        self.draw += 1
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.name != "dda":
            r = self.rows[index]
            return self.transform(self.open(r)), r["label"]
        from PIL import Image
        from .detectors import module
        from .training_recipes import UP

        # Unpaired WildFC: do not manufacture matched reconstruction alignment.
        real = self.open(self.real[index])
        fake = self.open(random.choice(self.fake))
        from dda_training_data.custom_transforms import ComposedTransforms, apply_resize

        factors = [random.uniform(0.2, 1.0), random.uniform(1.0, 3.5)]
        methods = random.sample(
            [
                Image.Resampling.NEAREST,
                Image.Resampling.BOX,
                Image.Resampling.BILINEAR,
                Image.Resampling.HAMMING,
                Image.Resampling.BICUBIC,
                Image.Resampling.LANCZOS,
            ],
            2,
        )
        values = {
            "real": real,
            "fake": fake,
            "real_resized": [
                apply_resize(real, f, m) for f, m in zip(factors, methods)
            ],
            "fake_resized": [
                apply_resize(fake, f, m) for f, m in zip(factors, methods)
            ],
        }
        with generated_image_limit(values):
            values = ComposedTransforms(self.transform)(values)
        return torch.stack(
            [
                values["real"],
                *values["real_resized"],
                values["fake"],
                *values["fake_resized"],
            ]
        ), torch.tensor([0, 0, 0, 1, 1, 1])


def train(args):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from .detectors import build_detector
    from .training_recipes import (
        recipe,
        trainable_model,
        training_mode,
        train_transform,
        parameter_groups,
        loss_for,
    )

    output = args.output / "checkpoints" / args.model
    output.mkdir(parents=True, exist_ok=True)
    config = recipe(args.model, args.epochs)
    use_cuda = args.device.startswith("cuda")
    amp_dtype = (
        torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float16
    )
    config["precision"] = str(amp_dtype) if use_cuda else "torch.float32"
    if args.microbatch:
        config["microbatch"] = args.microbatch
    config.update(
        model=args.model,
        seed=42,
        input_manifest_sha256=manifest_digest(args.manifest),
        base_signature=base_signature(args.model),
        source_revisions={
            p.parent.name: p.read_text().strip() for p in UP.glob("*/REVISION")
        },
        adapter_sha256={
            p: manifest_digest(Path(__file__).parent / p)
            for p in ("finetune.py", "training_recipes.py", "detectors.py")
        },
        smoke_steps=args.smoke_steps,
        checkpoint_selection="fixed final epoch; no validation or test monitoring",
        sampling="inverse class frequency; DDA real/fake pairs",
    )
    protocol = output / "protocol.json"
    if protocol.exists() and json.loads(protocol.read_text()) != config:
        raise ValueError("Training protocol changed; choose a new output directory")
    write_json(protocol, config)
    final = output / "finetuned.pt"
    if final.exists():
        completed = torch.load(final, map_location="cpu", weights_only=True)
        if completed["config"] != config:
            raise ValueError("Checkpoint protocol mismatch")
        print(f"{args.model}: checkpoint already complete", flush=True)
        return
    rows = read_rows(args.manifest)
    if not rows or any(
        r["split"] != "train" or r.get("training_role") != "detector" for r in rows
    ):
        raise ValueError(
            "Detector training requires the detector-only training manifest"
        )
    for r in rows:
        if hashlib.sha256((ROOT / r["path"]).read_bytes()).hexdigest() != r["sha256"]:
            raise ValueError(f'Image changed: {r["path"]}')
    seed_all(42)
    model, _, _ = build_detector(args.model, args.device)
    trainable_model(model, args.model)
    transform = train_transform(args.model)
    dataset = Images(rows, transform, args.model)
    micro = config["microbatch"]
    accum = max(1, math.ceil(config["effective_batch"] / micro))
    optimizer = torch.optim.AdamW(
        parameter_groups(model, config),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_cuda and amp_dtype == torch.float16, init_scale=128.0
    )
    smoke_before = updated_state(model) if args.smoke_steps else None
    resume = output / "last.pt"
    start_epoch = 0
    start_batch = 0
    step = 0
    history = []
    resumed_loss, resumed_seen = 0.0, 0
    if resume.exists():
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved["config"] != config:
            raise ValueError("Resume config mismatch")
        apply_updates(model, saved)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        start_epoch = saved["epoch"]
        start_batch = saved["next_batch"]
        step = saved["steps"]
        history = saved["history"]
        resumed_loss = saved.get("epoch_loss_sum", 0.0)
        resumed_seen = saved.get("epoch_examples", 0)
        del saved

    def checkpoint(epoch, next_batch):
        state = {
            "format": "wildfc_trainable_delta_v1",
            "model": args.model,
            "updates": updated_state(model),
            "config": config,
            "epoch": epoch,
            "next_batch": next_batch,
            "steps": step,
            "history": history,
            "epoch_loss_sum": epoch_loss if next_batch else 0.0,
            "epoch_examples": seen if next_batch else 0,
        }
        save_torch(
            resume,
            {
                **state,
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
            },
        )
        return state

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f'{args.model}: {trainable:,} trainable parameters; microbatch={micro}, accumulation={accum}, epochs={config["epochs"]}',
        flush=True,
    )
    epoch_loss = 0.0
    start = time.monotonic()
    for epoch in range(start_epoch, config["epochs"]):
        seed_all(42 + epoch)
        training_mode(model, args.model)
        dataset.epoch = epoch
        dataset.draw = 0
        generator = torch.Generator().manual_seed(42 + epoch)
        if args.model == "dda":
            loader = DataLoader(
                dataset,
                batch_size=micro,
                shuffle=True,
                generator=generator,
                num_workers=0,
            )
        else:
            counts = Counter(r["label"] for r in rows)
            sampler = WeightedRandomSampler(
                [1 / counts[r["label"]] for r in rows],
                len(rows),
                replacement=True,
                generator=generator,
            )
            loader = DataLoader(
                dataset, batch_size=micro, sampler=sampler, num_workers=0
            )
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = resumed_loss if epoch == start_epoch and start_batch else 0.0
        seen = resumed_seen if epoch == start_epoch and start_batch else 0
        total_updates = math.ceil(len(loader) / accum) * config["epochs"]
        for batch, (x, y) in enumerate(loader):
            # Replaying augmentations from the epoch seed reconstructs interrupted RNG state.
            if epoch == start_epoch and batch < start_batch:
                continue
            if args.model == "dda":
                x = x.flatten(0, 1)
                y = y.flatten()
            if batch % accum == 0:
                if args.model == "dda":
                    factor = 0.001 + 0.999 * (1 + math.cos(math.pi * step / 1000)) / 2
                elif args.model == "universalfakedetect":
                    factor = 1.0
                else:
                    warm = round(config["warmup_fraction"] * total_updates)
                    factor = (
                        (step + 1) / max(1, warm)
                        if step < warm
                        else (
                            1
                            + math.cos(
                                math.pi * (step - warm) / max(1, total_updates - warm)
                            )
                        )
                        / 2
                    )
                lr = config["lr"] * factor
                if args.model in ("safe", "aide_GenImage", "aide_progan", "aide_sd14"):
                    lr = max(1e-6, lr)
                for group in optimizer.param_groups:
                    group["lr"] = lr
            seed_all(42 + epoch * 1000000 + batch)
            x = x.to(args.device)
            y = y.to(args.device)
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=use_cuda,
            ):
                loss = loss_for(model, x, y, args.model, config)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            # Correctly normalize the shorter final accumulation window.
            window = min(accum, len(loader) - (batch // accum) * accum)
            scaler.scale(loss / window).backward()
            epoch_loss += float(loss.detach()) * len(y)
            seen += len(y)
            if (batch + 1) % accum == 0 or batch + 1 == len(loader):
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                successful_update = scaler.get_scale() >= previous_scale
                optimizer.zero_grad(set_to_none=True)
                step += int(successful_update)
                progress = {
                    "status": "training",
                    "model": args.model,
                    "epoch": epoch + 1,
                    "epochs": config["epochs"],
                    "batch": batch + 1,
                    "batches": len(loader),
                    "steps": step,
                    "loss": float(loss.detach()),
                    "trainable_parameters": trainable,
                    "elapsed_seconds": time.monotonic() - start,
                }
                if args.device.startswith("cuda"):
                    progress["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
                write_json(output / "status.json", progress)
                print(json.dumps(progress), flush=True)
                if successful_update and step % 20 == 0:
                    checkpoint(epoch, batch + 1)
                if args.smoke_steps and step >= args.smoke_steps:
                    saved = checkpoint(epoch, batch + 1)
                    trainable_keys = {
                        n for n, p in model.named_parameters() if p.requires_grad
                    }
                    changed = [
                        k
                        for k in trainable_keys
                        if not torch.equal(smoke_before[k], saved["updates"][k])
                    ]
                    if not changed:
                        raise RuntimeError(
                            "Smoke test did not change any trainable parameter"
                        )
                    if not all(
                        torch.isfinite(v).all() for v in saved["updates"].values()
                    ):
                        raise RuntimeError("Nonfinite checkpoint weights")
                    saved["smoke_changed_parameters"] = changed
                    save_torch(output / "smoke.pt", saved)
                    return
        history.append({"epoch": epoch + 1, "loss": epoch_loss / max(1, seen)})
        saved = checkpoint(epoch + 1, 0)
        start_batch = 0
        write_json(output / "history.json", history)
    if args.smoke_steps:
        raise RuntimeError(
            "Smoke test did not reach the requested successful optimizer updates"
        )
    if step == 0:
        raise RuntimeError("No successful optimizer updates")
    if start_epoch >= config["epochs"]:
        saved = checkpoint(start_epoch, 0)
    save_torch(final, saved)
    write_json(
        output / "status.json",
        {
            "status": "complete",
            "checkpoint": str(final),
            "epochs": config["epochs"],
            "steps": step,
            "trainable_parameters": trainable,
        },
    )


def predict(args):
    import torch
    from PIL import Image, ImageOps
    from .detectors import build_detector

    checkpoint = args.output / "checkpoints" / args.model / "finetuned.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["config"]["base_signature"] != base_signature(args.model):
        raise ValueError("Original weights changed")
    if saved["model"] != args.model:
        raise ValueError("Wrong checkpoint model")
    digest = manifest_digest(checkpoint)
    fingerprint = manifest_digest(args.manifest)
    output = args.output / "base_predictions"
    output.mkdir(exist_ok=True)
    path = output / f"{args.model}.json"
    meta = {"checkpoint_sha256": digest, "input_manifest_sha256": fingerprint}
    rows = read_rows(args.manifest)
    if path.exists():
        d = json.loads(path.read_text())
        if all(d.get(k) == v for k, v in meta.items()) and len(d["results"]) == len(
            rows
        ):
            return
    progress = output / f"{args.model}.progress.jsonl"
    progress_meta = output / f"{args.model}.progress.meta.json"
    results = []
    if (
        progress.exists()
        and progress_meta.exists()
        and json.loads(progress_meta.read_text()) == meta
    ):
        content = progress.read_bytes()
        if content and not content.endswith(b"\n"):
            content = content[: content.rfind(b"\n") + 1]
        results = [json.loads(l) for l in content.splitlines()]
        if len(results) > len(rows) or any(
            (r["path"], r["sha256"]) != (s["path"], s["sha256"])
            for r, s in zip(rows, results)
        ):
            raise ValueError("Partial score alignment mismatch")
        progress.write_bytes(content)
    else:
        progress.write_text("")
        write_json(progress_meta, meta)
    model, transform, metadata = build_detector(args.model, args.device)
    apply_updates(model, saved)
    model.eval()
    del saved
    for i, r in enumerate(rows[len(results) :], start=len(results)):
        image_path = ROOT / r["path"]
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != r["sha256"]:
            raise ValueError("Image changed")
        with Image.open(image_path) as image:
            if args.model == "community_forensics":
                image = ImageOps.exif_transpose(image)
            x = transform(image.convert("RGB")).unsqueeze(0).to(args.device)
        with torch.inference_mode():
            logits = model(x)
            score = float(
                logits.sigmoid().item()
                if metadata["binary"]
                else logits.softmax(-1)[0, 1].item()
            )
        if not 0 <= score <= 1:
            raise ValueError("Invalid score")
        result = {**r, "score": score, "prediction": int(score > 0.5)}
        results.append(result)
        with progress.open("a") as f:
            f.write(json.dumps(result) + "\n")
        if (i + 1) % 100 == 0:
            print(args.model, "prediction", i + 1, "/", len(rows), flush=True)
    write_json(
        path, {**meta, "model": args.model, "metadata": metadata, "results": results}
    )
    progress.unlink(missing_ok=True)
    progress_meta.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest", type=Path, default=ROOT / "data/wildfc/ensemble_split/all.jsonl"
    )
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--device", choices=["cuda:0", "cpu"], default="cuda:0")
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument(
        "--phase", choices=["run", "prepare", "train", "predict"], default="run"
    )
    p.add_argument("--model", choices=MODELS)
    p.add_argument(
        "--epochs",
        type=int,
        help="Override each recipe epoch count (new output required)",
    )
    p.add_argument(
        "--microbatch", type=int, help="Override worker microbatch for memory testing"
    )
    p.add_argument(
        "--smoke-steps",
        type=int,
        default=0,
        help="Worker-only optimizer-step test; never publishes a final checkpoint",
    )
    args = p.parse_args()
    if (
        args.gpu_index < 0
        or (args.epochs is not None and args.epochs < 1)
        or (args.microbatch is not None and args.microbatch < 1)
        or args.smoke_steps < 0
    ):
        p.error("Invalid negative/zero option")
    if args.phase in ("train", "predict") and not args.model:
        p.error("Worker phase requires --model")
    if args.smoke_steps and args.phase != "train":
        p.error("--smoke-steps requires --phase train")
    args.output = args.output.resolve()
    args.manifest = args.manifest.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    configure_environment(offline=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = (
        str(args.gpu_index) if args.device.startswith("cuda") else ""
    )
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    if args.phase in ("train", "predict"):
        import fcntl
        import torch

        torch.set_num_threads(2)
        with (CACHE / f"detectors/gpu{args.gpu_index}_inference.lock").open(
            "w"
        ) as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if args.device.startswith("cuda"):
                if not torch.cuda.is_available():
                    raise RuntimeError("Selected GPU unavailable")
                total = torch.cuda.get_device_properties(0).total_memory
                torch.cuda.set_per_process_memory_fraction(
                    min(7_500_000_000, int(total * 0.8)) / total
                )
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            return train(args) if args.phase == "train" else predict(args)
    splits = args.output / "splits"
    summary = prepare_split(args.manifest, splits)
    print(json.dumps(summary["counts"]), flush=True)
    if args.phase == "prepare":
        return
    import fcntl

    with (args.output / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        selected = [args.model] if args.model else MODELS
        status = {
            "status": "running",
            "models": selected,
            "gpu_index": args.gpu_index,
            "splits": summary["counts"],
        }
        logs = args.output / "logs"
        logs.mkdir(exist_ok=True)
        try:
            for phase, manifest in [
                ("train", splits / "detector_train.jsonl"),
                ("predict", splits / "stacking_evaluation.jsonl"),
            ]:
                for name in selected:
                    status.update(phase=phase, model=name)
                    write_json(args.output / "status.json", status)
                    command = [
                        sys.executable,
                        "-m",
                        "ai_detector.finetune",
                        "--phase",
                        phase,
                        "--model",
                        name,
                        "--manifest",
                        str(manifest),
                        "--output",
                        str(args.output),
                        "--device",
                        args.device,
                        "--gpu-index",
                        str(args.gpu_index),
                    ]
                    if args.epochs:
                        command += ["--epochs", str(args.epochs)]
                    if args.microbatch:
                        command += ["--microbatch", str(args.microbatch)]
                    print(phase, name, flush=True)
                    with (logs / f"{name}_{phase}.log").open("a") as log:
                        subprocess.run(
                            command,
                            cwd=ROOT,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                        )
            if not args.model:
                status.update(phase="meta_training")
                write_json(args.output / "status.json", status)
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "ai_detector.classical_ensemble",
                        "--manifest",
                        str(splits / "stacking_evaluation.jsonl"),
                        "--predictions",
                        str(args.output / "base_predictions"),
                        "--output",
                        str(args.output),
                    ],
                    cwd=ROOT,
                    check=True,
                )
            status.update(status="complete")
            write_json(args.output / "status.json", status)
        except Exception as exc:
            status.update(status="failed", error=str(exc))
            write_json(args.output / "status.json", status)
            raise


if __name__ == "__main__":
    main()
