
"""Compare seven- and eight-detector stacking on the same WildFC split."""

import csv
import json
from pathlib import Path

import numpy as np
import sklearn
import xgboost
from scipy.special import expit
from threadpoolctl import threadpool_limits

from ai_detector.config import ROOT, MODELS
from ai_detector.ensemble import (
    DEFAULT_CONFIG,
    fit_meta,
    load_features,
    metrics,
    predict_meta,
    transform_scores,
)
from ai_detector.classical_ensemble import classifiers


MANIFEST = ROOT / "data/wildfc/ensemble_split/all.jsonl"
SPLIT = ROOT / "reports/npr_calibration/split/stacking_evaluation.jsonl"
PREDICTIONS = ROOT / ".cache/predictions/eight_detector"
OUTPUT = ROOT / "reports/npr_ensemble_comparison"


def identity(row):
    return (row["path"], row["sha256"], row["label"])


def prepare_data():
    names = [*MODELS, "npr"]
    rows, scores = load_features(MANIFEST, PREDICTIONS, models=names)

    split_rows = [
        json.loads(line)
        for line in SPLIT.read_text().splitlines()
        if(line.strip())
    ]

    if(len(split_rows) != 3962):
        raise ValueError("Unexpected stacking evaluation size")

    keys = [identity(row) for row in split_rows]

    if(len(set(keys)) != len(keys)):
        raise ValueError("Duplicate split records")

    lookup = {identity(row): index for index, row in enumerate(rows)}

    if(len(lookup) != len(rows)):
        raise ValueError("Duplicate full-manifest records")

    if(not all(key in lookup for key in keys)):
        raise ValueError("Split contains images absent from prediction caches")

    indices = np.array([lookup[key] for key in keys])
    features = scores[indices]

    labels = np.array([row["label"] for row in split_rows])
    train = np.array([
        row["split"] == "train"
        and row.get("training_role") == "stacking"
        for row in split_rows
    ])
    test = np.array([row["split"] == "test" for row in split_rows])

    if(
        train.sum() != 1757
        or test.sum() != 2205
        or not (train | test).all()
    ):
        raise ValueError("Unexpected training or test partition")

    for field in ("sha256", "pixel_sha256", "leakage_group"):
        train_values = {row[field] for row, selected in zip(split_rows, train) if(selected)}
        test_values = {row[field] for row, selected in zip(split_rows, test) if(selected)}

        if(train_values & test_values):
            raise ValueError(f"Training/test leakage detected: {field}")

    if(set(labels[train]) != {0, 1} or set(labels[test]) != {0, 1}):
        raise ValueError("Both classes are required in each partition")

    return names, features, labels, train, test


def run_experiment(name, model_names, features, labels, train, test):
    output = OUTPUT / name
    output.mkdir(parents=True, exist_ok=True)

    train_features = features[train]
    test_features = features[test]
    train_labels = labels[train]
    test_labels = labels[test]

    transformed_train = transform_scores(train_features)
    transformed_test = transform_scores(test_features)

    results = {}
    predictions = {}

    for classifier_name, estimator in classifiers(train_labels).items():
        print(f"{name}: training {classifier_name}", flush=True)
        estimator.fit(transformed_train, train_labels)

        if(classifier_name.startswith("svm_")):
            # Original workflow uses sigmoid(margin) for score reporting.
            scores = expit(estimator.decision_function(transformed_test))
        else:
            scores = estimator.predict_proba(transformed_test)[:, 1]

        results[classifier_name] = metrics(test_labels, scores)
        predictions[classifier_name] = scores

    print(f"{name}: training MLP", flush=True)

    model, mean, scale, history = fit_meta(
        train_features,
        train_labels,
        DEFAULT_CONFIG,
    )

    mlp_scores = predict_meta(
        model,
        mean,
        scale,
        test_features,
    )

    results["meta_mlp"] = metrics(test_labels, mlp_scores)
    predictions["meta_mlp"] = mlp_scores

    (output / "metrics.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )

    with (output / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["method", *next(iter(results.values())).keys()],
        )
        writer.writeheader()
        writer.writerows(
            {"method": method, **values}
            for method, values in results.items()
        )

    (output / "protocol.json").write_text(
        json.dumps({
            "models": model_names,
            "train_count": int(train.sum()),
            "test_count": int(test.sum()),
            "split_manifest": str(SPLIT.relative_to(ROOT)),
            "prediction_manifest": str(MANIFEST.relative_to(ROOT)),
            "prediction_directory": str(PREDICTIONS.relative_to(ROOT)),
            "classifiers": "ai_detector.classical_ensemble.classifiers",
            "mlp_config": DEFAULT_CONFIG,
            "sklearn_version": sklearn.__version__,
            "xgboost_version": xgboost.__version__,
            "note": "Exploratory comparison; test labels previously inspected.",
        }, indent=2) + "\n"
    )

    print(f"\n{name} results:")
    for method, values in results.items():
        print(
            f"  {method}: "
            f"BAcc={values['balanced_accuracy'] * 100:.2f}%, "
            f"AUROC={values['roc_auc'] * 100:.2f}%, "
            f"AI recall={values['ai_recall'] * 100:.2f}%"
        )

    return results


def main():
    names, features, labels, train, test = prepare_data()

    print(f"Loaded feature matrix: {features.shape}")
    print(f"Stacking training: {train.sum()}")
    print(f"Held-out test: {test.sum()}")

    with threadpool_limits(limits=2):
        seven = run_experiment(
            "seven_detectors",
            names[:7],
            features[:, :7],
            labels,
            train,
            test,
        )

        eight = run_experiment(
            "eight_detectors",
            names,
            features,
            labels,
            train,
            test,
        )

    print("\nBalanced accuracy comparison:")
    for method in seven:
        before = seven[method]["balanced_accuracy"] * 100
        after = eight[method]["balanced_accuracy"] * 100
        print(
            f"{method}: "
            f"7={before:.2f}%, "
            f"8={after:.2f}%, "
            f"change={after - before:+.2f} percentage points"
        )


if(__name__ == "__main__"):
    main()
