"""Train a small stacking network and compare fixed voting baselines on a held-out split."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from .config import MODELS, ROOT
from .manifests import manifest_digest
from .reporting import font, text, save_png, NAMES
from PIL import Image, ImageDraw

DEFAULT_CONFIG = {
    "seed": 42,
    "epochs": 100,
    "batch_size": 256,
    "learning_rate": 0.001,
    "weight_decay": 0.001,
    "hidden_sizes": [16, 8],
    "threshold": 0.5,
    "feature_transform": "clipped score log-odds, standardized using training data only",
    "class_weighting": "BCE positive weight = train real / train AI",
    "checkpoint_selection": "final epoch; no held-out monitoring or hyperparameter selection",
}


class MetaClassifier(torch.nn.Module):
    def __init__(self, features=7):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(features, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)


def transform_scores(scores):
    scores = np.clip(np.asarray(scores, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(scores / (1 - scores)).astype(np.float32)


def fit_meta(train_scores, train_labels, config):
    """Receives TRAIN data only; the evaluation split cannot affect optimization."""
    torch.set_num_threads(2)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    x = transform_scores(train_scores)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.maximum(scale, 1e-6)
    x = torch.from_numpy((x - mean) / scale)
    y = torch.tensor(train_labels, dtype=torch.float32)
    if set(y.tolist()) != {0.0, 1.0}:
        raise ValueError("Both classes are required for meta training")
    model = MetaClassifier(x.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=(1 - y).sum() / y.sum())
    history = []
    for epoch in range(config["epochs"]):
        model.train()
        order = torch.randperm(len(x))
        total_loss = 0.0
        for indices in order.split(config["batch_size"]):
            optimizer.zero_grad()
            loss = criterion(model(x[indices]), y[indices])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)
        history.append(
            {"epoch": epoch + 1, "training_weighted_bce": total_loss / len(x)}
        )
    model.eval()
    return model, mean, scale, history


def predict_meta(model, mean, scale, scores):
    x = torch.from_numpy((transform_scores(scores) - mean) / scale)
    with torch.inference_mode():
        return model(x).sigmoid().numpy()


def votes(scores):
    hard = (scores > 0.5).astype(np.int32)
    # Three correlated AIDE variants otherwise count as three of seven votes.
    family_hard = np.column_stack(
        [hard[:, :4], (hard[:, 4:].sum(axis=1) >= 2).astype(np.int32)]
    )
    return {
        "majority_vote_7": hard.mean(axis=1),
        "family_vote_5": family_hard.mean(axis=1),
        "mean_score_7": scores.mean(axis=1),
    }


def metrics(labels, scores):
    predictions = (scores > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "real_recall": float(tn / (tn + fp)),
        "ai_recall": float(tp / (tp + fn)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "count": len(labels),
    }


def load_features(manifest, prediction_dir, models=MODELS):
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    fingerprint = manifest_digest(manifest)
    columns = []
    for name in models:
        d = json.loads((prediction_dir / f"{name}.json").read_text())
        if d.get("input_manifest_sha256") != fingerprint:
            raise ValueError(f"{name}: predictions from a different manifest")
        predictions = d["results"]
        if len(predictions) != len(rows):
            raise ValueError(f"{name}: incomplete predictions")
        for r, p in zip(rows, predictions):
            if (r["path"], r["sha256"], r["label"]) != (
                p["path"],
                p["sha256"],
                p["label"],
            ):
                raise ValueError(f"{name}: misaligned prediction")
        columns.append([p["score"] for p in predictions])
    x = np.asarray(columns, dtype=np.float64).T
    if not np.isfinite(x).all() or (x < 0).any() or (x > 1).any():
        raise ValueError("Invalid detector scores")
    return rows, x


def render_summary(results, path, train_count, test_count):
    names = {
        **NAMES,
        "majority_vote_7": "Majority vote (7)",
        "family_vote_5": "Family vote (5)",
        "mean_score_7": "Mean score (7)",
        "meta_mlp": "Meta-classifier MLP",
    }
    width = 1600
    im = Image.new("RGB", (width, 265 + 55 * len(results)), "#f3f6fa")
    d = ImageDraw.Draw(im)
    text(d, (35, 25), "WildFC · held-out ensemble comparison", 32, bold=True)
    text(
        d,
        (35, 72),
        f"Train {train_count:,} / evaluation {test_count:,} · source-stratified 80:20 · fixed threshold 0.5",
        21,
        fill="#52667a",
    )
    headers = [
        "Method",
        "Accuracy",
        "Balanced acc.",
        "Macro F1",
        "ROC AUC",
        "Real recall",
        "AI recall",
    ]
    xs = [35, 460, 640, 845, 1030, 1210, 1400]
    d.rectangle((25, 115, width - 25, 165), fill="#183b5b")
    for x, h in zip(xs, headers):
        text(d, (x, 129), h, 19, fill="white", bold=True)
    for i, (name, m) in enumerate(results.items()):
        y = 170 + i * 55
        d.rectangle(
            (25, y, width - 25, y + 51),
            fill=(
                "#e0f0e7"
                if name == "meta_mlp"
                else ("white" if i % 2 == 0 else "#e8eef5")
            ),
        )
        values = [names[name]] + [
            f"{m[k]*100:.2f}%"
            for k in (
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "roc_auc",
                "real_recall",
                "ai_recall",
            )
        ]
        for x, v in zip(xs, values):
            text(d, (x, y + 14), v, 19, bold=x == 35)
    text(
        d,
        (35, im.height - 78),
        "Base detectors frozen. Held-out evaluation; any meta-network is trained on training data only.",
        17,
        fill="#52667a",
    )
    text(
        d,
        (35, im.height - 49),
        "Sources are publishers/platforms, not known generator IDs. Article groups stay within one split.",
        17,
        fill="#52667a",
    )
    text(
        d,
        (35, im.height - 24),
        "Dataset: Pantsios et al., WildFC, MAD 2026. Results apply to this split; training overlap is not fully audited.",
        15,
        fill="#52667a",
    )
    save_png(im, path)


def run(manifest, prediction_dir, output, mode="all", model=None):
    if mode not in ("all", "voting", "ensemble", "model"):
        raise ValueError(f"Unknown mode: {mode}")
    if mode == "model" and model not in MODELS:
        raise ValueError("Single-model evaluation requires a valid model name")
    models = [model] if mode == "model" else MODELS
    train_meta = mode in ("all", "ensemble")
    output.mkdir(parents=True, exist_ok=True)
    config = dict(DEFAULT_CONFIG)
    config["model_order"] = models
    config["input_manifest_sha256"] = manifest_digest(manifest)
    config["comparison_methods"] = [
        "majority_vote_7",
        "family_vote_5",
        "mean_score_7",
        "meta_mlp",
    ]
    if mode != "all":
        config["comparison_methods"] = {
            "voting": ["majority_vote_7", "family_vote_5", "mean_score_7"],
            "ensemble": ["meta_mlp"],
            "model": [],
        }[mode]
        config["mode"] = mode
    protocol = output / "protocol.json"
    if protocol.exists() and json.loads(protocol.read_text()) != config:
        raise ValueError("Protocol differs; use a new experiment output directory")
    protocol.write_text(json.dumps(config, indent=2) + "\n")
    rows, x = load_features(manifest, prediction_dir, models)
    train = np.array([r["split"] == "train" for r in rows])
    test = np.array([r["split"] == "test" for r in rows])
    if not (train | test).all() or not train.any() or not test.any():
        raise ValueError("Invalid train/test split")
    for key in ("sha256", "pixel_sha256", "leakage_group"):
        if {r[key] for r in rows if r["split"] == "train"} & {
            r[key] for r in rows if r["split"] == "test"
        }:
            raise ValueError(f"Train/test overlap in {key}")
    y = np.array([r["label"] for r in rows])
    if train_meta:
        model, mean, scale, history = fit_meta(x[train], y[train], config)
        checkpoint = {
            "state_dict": model.state_dict(),
            "mean": torch.from_numpy(mean),
            "scale": torch.from_numpy(scale),
            "model_order": MODELS,
            "config": config,
        }
        torch.save(checkpoint, output / "meta_classifier.pt")
        with (output / "training_history.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "training_weighted_bce"])
            writer.writeheader()
            writer.writerows(history)
    # First held-out scoring happens only after training and checkpoint writing.
    scores = {name: x[test, i] for i, name in enumerate(models)}
    if mode in ("all", "voting"):
        scores.update(votes(x[test]))
    if train_meta:
        scores["meta_mlp"] = predict_meta(model, mean, scale, x[test])
    results = {name: metrics(y[test], p) for name, p in scores.items()}
    (output / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    with (output / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["method"] + list(next(iter(results.values())))
        )
        writer.writeheader()
        writer.writerows({"method": name, **m} for name, m in results.items())
    with (output / "heldout_predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "sha256", "source", "label"] + list(scores))
        for i, r in enumerate(r for r in rows if r["split"] == "test"):
            writer.writerow(
                [r["path"], r["sha256"], r["group"], r["label"]]
                + [float(p[i]) for p in scores.values()]
            )
    with (output / "features.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "sha256", "split", "source", "label"] + models)
        for r, features in zip(rows, x):
            writer.writerow(
                [r["path"], r["sha256"], r["split"], r["group"], r["label"]]
                + features.tolist()
            )
    per_source = {}
    test_rows = [r for r in rows if r["split"] == "test"]
    for source in sorted(set(r["group"] for r in test_rows)):
        mask = np.array([r["group"] == source for r in test_rows])
        per_source[source] = {
            name: {
                "count": int(mask.sum()),
                "accuracy": float(((p[mask] > 0.5) == y[test][mask]).mean()),
            }
            for name, p in scores.items()
        }
    (output / "per_source.json").write_text(json.dumps(per_source, indent=2) + "\n")
    render_summary(results, output / "summary.png", int(train.sum()), int(test.sum()))
    print(json.dumps(results, indent=2), flush=True)
    (output / "status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "train_count": int(train.sum()),
                "test_count": int(test.sum()),
                "input_manifest_sha256": config["input_manifest_sha256"],
            },
            indent=2,
        )
        + "\n"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest", type=Path, default=ROOT / "data/wildfc/ensemble_split/all.jsonl"
    )
    p.add_argument(
        "--predictions",
        type=Path,
        default=ROOT / "reports/wildfc_ensemble/base_predictions",
    )
    p.add_argument("--output", type=Path, default=ROOT / "reports/wildfc_ensemble")
    p.add_argument(
        "--mode", choices=["all", "model", "voting", "ensemble"], default="all"
    )
    p.add_argument("--model", choices=MODELS)
    a = p.parse_args()
    run(a.manifest, a.predictions, a.output, a.mode, a.model)


if __name__ == "__main__":
    main()
