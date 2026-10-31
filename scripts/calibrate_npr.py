
"""Calibrate NPR using stacking-training data, then evaluate on held-out test data."""

import json
import hashlib
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
)


ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_FILE = ROOT / "reports/wildfc_npr/npr.json"
ORIGINAL_MANIFEST = ROOT / "data/wildfc/ensemble_split/all.jsonl"

SPLIT_DIRECTORY = ROOT / "reports/npr_calibration/split"
VALIDATION_MANIFEST = SPLIT_DIRECTORY / "stacking_train.jsonl"
TEST_MANIFEST = SPLIT_DIRECTORY / "test.jsonl"


def load_manifest(path):
    with path.open() as file:
        return [json.loads(line) for line in file if(line.strip())]


def image_key(record):
    return (record["path"], record["sha256"], record["label"])


def get_predictions(manifest, cached_predictions):
    labels = []
    scores = []

    for record in manifest:
        key = image_key(record)

        if(key not in cached_predictions):
            raise ValueError(f"Missing prediction: {key}")

        prediction = cached_predictions[key]
        labels.append(record["label"])
        scores.append(prediction["score"])

    return np.asarray(labels), np.asarray(scores, dtype=float)


def evaluate(labels, scores, threshold):
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "auroc": roc_auc_score(labels, scores),
        "average_precision": average_precision_score(labels, scores),
        "real_recall": tn / (tn + fp),
        "ai_recall": tp / (tp + fn),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def print_results(title, labels, scores, threshold):
    metrics = evaluate(labels, scores, threshold)

    print(f"\n{title}")
    print(f"Images: {len(labels)}")
    print(f"Threshold: {threshold:.10g}")

    for name in (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "auroc",
        "average_precision",
        "real_recall",
        "ai_recall",
    ):
        print(f"{name}: {metrics[name] * 100:.2f}%")

    print(
        f"Confusion matrix (TN, FP, FN, TP): "
        f"{metrics['tn']}, {metrics['fp']}, "
        f"{metrics['fn']}, {metrics['tp']}"
    )


def main():
    cache = json.loads(PREDICTIONS_FILE.read_text())

    expected_hash = hashlib.sha256(
        ORIGINAL_MANIFEST.read_bytes()
    ).hexdigest()

    if(cache["input_manifest_sha256"] != expected_hash):
        raise ValueError("Cached predictions do not match the WildFC manifest")

    results = cache["results"]

    if(len(results) != len(load_manifest(ORIGINAL_MANIFEST))):
        raise ValueError("Prediction count does not match the manifest")

    cached_predictions = {
        image_key(record): record for record in results
    }

    if(len(cached_predictions) != len(results)):
        raise ValueError("Duplicate prediction keys detected")

    validation_manifest = load_manifest(VALIDATION_MANIFEST)
    test_manifest = load_manifest(TEST_MANIFEST)

    validation_keys = {image_key(record) for record in validation_manifest}
    test_keys = {image_key(record) for record in test_manifest}

    if(validation_keys & test_keys):
        raise ValueError("Validation/test image overlap detected")

    validation_labels, validation_scores = get_predictions(
        validation_manifest, cached_predictions
    )

    test_labels, test_scores = get_predictions(
        test_manifest, cached_predictions
    )

    if(
        not np.isfinite(validation_scores).all()
        or not np.isfinite(test_scores).all()
    ):
        raise ValueError("Non-finite scores detected")

    # Select threshold exclusively on validation data.
    false_positive_rates, true_positive_rates, thresholds = roc_curve(
        validation_labels,
        validation_scores,
        drop_intermediate=False,
    )

    validation_balanced_accuracy = (
        true_positive_rates + (1 - false_positive_rates)
    ) / 2

    # Exclude the infinite threshold, which predicts everything as real.
    valid_indices = np.flatnonzero(np.isfinite(thresholds))
    best_index = valid_indices[
        np.argmax(validation_balanced_accuracy[valid_indices])
    ]
    best_threshold = float(thresholds[best_index])

    print(f"Validation images: {len(validation_labels)}")
    print(f"Test images: {len(test_labels)}")
    print(f"Selected threshold: {best_threshold:.10g}")

    print_results(
        "VALIDATION: Default threshold",
        validation_labels,
        validation_scores,
        0.5,
    )

    print_results(
        "VALIDATION: Calibrated threshold",
        validation_labels,
        validation_scores,
        best_threshold,
    )

    print_results(
        "TEST: Default threshold",
        test_labels,
        test_scores,
        0.5,
    )

    print_results(
        "TEST: Calibrated threshold",
        test_labels,
        test_scores,
        best_threshold,
    )


if(__name__ == "__main__"):
    main()