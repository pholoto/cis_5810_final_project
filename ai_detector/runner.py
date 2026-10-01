"""Compare downloaded detectors on a shared manifest using official eval transforms."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .config import ROOT, CACHE, MODELS, UP, configure_environment
from .detectors import make_detector
from .manifests import manifest_digest, matching_result, snapshot_manifest


def worker(args):
    import torch

    records = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    if args.device == "cuda:0":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"GPU {args.gpu_index} is unavailable; check nvidia-smi and the PyTorch CUDA installation, or use --device cpu"
            )
        # 7.5 GB decimal allocator ceiling, leaving >2 GB for CUDA overhead.
        total = torch.cuda.get_device_properties(0).total_memory
        allocator_limit = min(7_500_000_000, int(total * 0.8))
        torch.cuda.set_per_process_memory_fraction(allocator_limit / total, 0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()
    progress = args.output / f"{args.model}.progress.jsonl"
    progress_meta = args.output / f"{args.model}.progress.meta.json"
    digest = manifest_digest(args.manifest)
    results = []
    if (
        progress.exists()
        and progress_meta.exists()
        and json.loads(progress_meta.read_text()).get("input_manifest_sha256") == digest
    ):
        content = progress.read_bytes()
        if content and not content.endswith(b"\n"):
            content = content[: content.rfind(b"\n") + 1]
        results = [json.loads(line) for line in content.splitlines()]
        if len(results) > len(records) or any(
            (a["path"], a["sha256"], a["label"]) != (b["path"], b["sha256"], b["label"])
            for a, b in zip(results, records)
        ):
            raise ValueError("Partial predictions do not match the input manifest")
        progress.write_bytes(content)
    else:
        progress.write_text("")
        progress_meta.write_text(json.dumps({"input_manifest_sha256": digest}) + "\n")
    predict, metadata = make_detector(args.model, args.device)
    start_index = len(results)
    print(f"Resuming {args.model} at {start_index}/{len(records)}", flush=True)
    for i, row in enumerate(records[start_index:], start=start_index):
        path = ROOT / row["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        start = time.monotonic()
        score = predict(path)
        if not 0 <= score <= 1:
            raise ValueError(f"Invalid score: {score}")
        result = {
            "path": row["path"],
            "sha256": row["sha256"],
            "label": row["label"],
            "group": row["group"],
            "score": score,
            "prediction": int(score > 0.5),
            "seconds": time.monotonic() - start,
        }
        results.append(result)
        with progress.open("a") as saved:
            saved.write(json.dumps(result) + "\n")
        print(args.model, i + 1, len(records), result, flush=True)
    if args.device == "cuda:0":
        metadata.update(
            physical_gpu=args.gpu_index,
            allocator_limit_bytes=allocator_limit,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        try:
            usage = subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    str(args.gpu_index),
                    "--query-compute-apps=pid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            metadata["process_vram_mib_at_finish"] = [
                int(line.split(",")[1].strip())
                for line in usage.splitlines()
                if line.split(",")[0].strip() == str(os.getpid())
            ]
        except (OSError, ValueError, subprocess.CalledProcessError):
            metadata["process_vram_mib_at_finish"] = None
        print("GPU_MEMORY", json.dumps(metadata), flush=True)
    output = {
        "model": args.model,
        "input_manifest_sha256": manifest_digest(args.manifest),
        "threshold": 0.5,
        "score_caveat": "Uncalibrated model score, not a probability of authenticity",
        "metadata": metadata,
        "torch_version": torch.__version__,
        "source_revisions": {
            p.parent.name: p.read_text().strip() for p in UP.glob("*/REVISION")
        },
        "results": results,
    }
    temporary = args.output / f"{args.model}.json.tmp"
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    temporary.replace(args.output / f"{args.model}.json")
    progress.unlink(missing_ok=True)
    progress_meta.unlink(missing_ok=True)


def report(args):
    from ai_detector.reporting import render_report

    render_report(args.manifest, args.output, MODELS)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Image files or directories; omit when using --manifest",
    )
    p.add_argument("--manifest", type=Path)
    p.add_argument("--output", type=Path, default=ROOT / "reports/predictions")
    p.add_argument("--model", choices=MODELS)
    p.add_argument(
        "--device",
        choices=["cpu", "cuda:0"],
        default=None,
        help="GPU by default; cuda:0 uses the physical GPU selected by --gpu-index",
    )
    p.add_argument(
        "--gpu-index",
        type=int,
        default=None,
        help="Physical NVIDIA GPU index (default: saved setting or 0)",
    )
    p.add_argument("--report-only", action="store_true")
    p.add_argument(
        "--scores-only",
        action="store_true",
        help="Save numeric predictions without rendering image tables",
    )
    args = p.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.inputs and args.manifest:
        p.error("Use image inputs or --manifest, not both")
    if args.inputs:
        from PIL import Image

        paths = []
        for source in args.inputs:
            if not source.exists():
                p.error(f"Missing input: {source}")
            paths.extend(
                sorted(
                    x
                    for x in source.rglob("*")
                    if x.suffix.lower()
                    in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
                    and x.is_file()
                )
                if source.is_dir()
                else [source]
            )
        paths = list(dict.fromkeys(p.resolve() for p in paths))
        if not paths:
            p.error("No images found")
        records = []
        for path in paths:
            with Image.open(path) as image:
                image.verify()
            records.append(
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "label": None,
                    "group": "User image",
                }
            )
        args.manifest = args.output / "input_manifest.jsonl"
        content = "".join(json.dumps(r) + "\n" for r in records)
        if args.manifest.exists() and args.manifest.read_text() != content:
            p.error(
                "Output already belongs to different inputs; choose a new --output directory"
            )
        args.manifest.write_text(content)
    elif args.manifest:
        args.manifest = args.manifest.resolve()
    elif (args.output / "input_manifest.jsonl").exists():
        args.manifest = args.output / "input_manifest.jsonl"
    else:
        p.error("Supply image files/directories or --manifest")
    if not args.manifest.is_file():
        p.error(f"Manifest not found: {args.manifest}")
    config_path = args.output / "run_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    args.device = args.device or config.get("device", "cuda:0")
    if args.device not in ("cpu", "cuda:0"):
        raise ValueError(args.device)
    args.gpu_index = (
        args.gpu_index if args.gpu_index is not None else config.get("gpu_index", 0)
    )
    if args.gpu_index < 0:
        p.error("--gpu-index must be nonnegative")
    configure_environment(offline=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = (
        str(args.gpu_index) if args.device == "cuda:0" else ""
    )
    runtime = (
        CACHE / "runs" / hashlib.sha256(str(args.output).encode()).hexdigest()[:16]
    )
    runtime.mkdir(parents=True, exist_ok=True)
    if args.model:
        import fcntl

        with (runtime / f"{args.model}.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if matching_result(args.output / f"{args.model}.json", args.manifest):
                return
            if args.device == "cuda:0":
                with (CACHE / f"detectors/gpu{args.gpu_index}_inference.lock").open(
                    "w"
                ) as gpu_lock:
                    fcntl.flock(gpu_lock, fcntl.LOCK_EX)
                    return worker(args)
            return worker(args)
    import fcntl

    with (runtime / "run.lock").open("w") as run_lock:
        fcntl.flock(run_lock, fcntl.LOCK_EX)
        if args.report_only:
            # Render the last evaluated snapshot even if the source has grown.
            saved = args.output / "evaluated_manifest.jsonl"
            if saved.exists():
                args.manifest = saved
            report(args)
            return
        config_path.write_text(
            json.dumps({"device": args.device, "gpu_index": args.gpu_index}, indent=2)
            + "\n"
        )
        source_manifest = str(args.manifest)
        args.manifest, count, digest = snapshot_manifest(args.manifest, runtime)
        print(f"Evaluating fixed snapshot: {count} images, SHA256 {digest}", flush=True)
        status = {
            "status": "running",
            "source_manifest": source_manifest,
            "snapshot_manifest": str(args.manifest),
            "input_manifest_sha256": digest,
            "image_count": count,
            "device": args.device,
            "gpu_index": args.gpu_index,
            "runtime_logs": str(runtime),
        }
        (args.output / "run_status.json").write_text(
            json.dumps(status, indent=2) + "\n"
        )
        for name in MODELS:
            result = args.output / f"{name}.json"
            if matching_result(result, args.manifest):
                print(f"{name}: reusing matching predictions", flush=True)
                continue
            print(
                f"{name}: running (missing or stale predictions); log: {runtime/name}.log",
                flush=True,
            )
            with (runtime / f"{name}.log").open("w") as log:
                try:
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "ai_detector",
                            "--model",
                            name,
                            "--manifest",
                            str(args.manifest),
                            "--output",
                            str(args.output),
                            "--device",
                            args.device,
                            "--gpu-index",
                            str(args.gpu_index),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        cwd=ROOT,
                    )
                except subprocess.CalledProcessError as exc:
                    status.update(
                        status="failed",
                        failed_model=name,
                        error_log=str(runtime / f"{name}.log"),
                    )
                    (args.output / "run_status.json").write_text(
                        json.dumps(status, indent=2) + "\n"
                    )
                    raise RuntimeError(
                        f"{name} failed. Read {runtime/name}.log, fix the cause, and rerun the same command to resume."
                    ) from exc
        temporary = args.output / "evaluated_manifest.jsonl.tmp"
        temporary.write_bytes(args.manifest.read_bytes())
        temporary.replace(args.output / "evaluated_manifest.jsonl")
        if not args.scores_only:
            report(args)
        status["status"] = "complete"
        (args.output / "run_status.json").write_text(
            json.dumps(status, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
