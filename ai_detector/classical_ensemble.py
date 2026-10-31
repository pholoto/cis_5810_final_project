"""Compare CPU stacking classifiers using cached detector scores."""
import argparse
import csv
import json
from pathlib import Path
import joblib
import numpy as np
import sklearn
import xgboost
from scipy.special import expit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from threadpoolctl import threadpool_limits
from .config import ROOT, MODELS, configure_environment
from .manifests import manifest_digest
from .ensemble import (load_features, transform_scores, metrics, votes, fit_meta,
                       predict_meta, DEFAULT_CONFIG, render_summary)


def classifiers(y):
    """Fixed settings; no test-dependent search or threshold selection."""
    return {
        'logistic_l2': make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, solver='lbfgs', max_iter=5000, class_weight='balanced', random_state=42)),
        'svm_linear': make_pipeline(StandardScaler(), LinearSVC(
            C=1.0, class_weight='balanced', dual='auto', max_iter=20000, random_state=42)),
        'svm_rbf': make_pipeline(StandardScaler(), SVC(
            C=1.0, kernel='rbf', gamma='scale', class_weight='balanced', probability=False)),
        'random_forest': RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
            max_features='sqrt', class_weight='balanced', n_jobs=2, random_state=42),
        'xgboost': xgboost.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=1.0, reg_lambda=1.0, reg_alpha=0.0,
            scale_pos_weight=float((y == 0).sum() / (y == 1).sum()),
            objective='binary:logistic', eval_metric='logloss', tree_method='hist',
            n_jobs=2, random_state=42),
    }


def run(manifest, predictions, output):
    rows, x = load_features(manifest, predictions)
    train = np.array([r['split'] == 'train' for r in rows])
    test = np.array([r['split'] == 'test' for r in rows])
    if not (train | test).all() or not train.any() or not test.any():
        raise ValueError('Invalid train/test split')
    for key in ('sha256', 'pixel_sha256', 'leakage_group'):
        if {r[key] for r in rows if r['split'] == 'train'} & {r[key] for r in rows if r['split'] == 'test'}:
            raise ValueError(f'Train/test overlap: {key}')
    y = np.array([r['label'] for r in rows])
    if set(y[train]) != {0, 1} or set(y[test]) != {0, 1}:
        raise ValueError('Both classes required in each split')
    models = classifiers(y[train])
    checkpoint_hashes = {
        n: json.loads((predictions / f'{n}.json').read_text()).get('checkpoint_sha256')
        for n in MODELS
    }
    fine_tuned = any(checkpoint_hashes.values())
    if fine_tuned and (not all(checkpoint_hashes.values()) or any(
        r.get('training_role') != 'stacking' for r in rows if r['split'] == 'train'
    )):
        raise ValueError('Fine-tuned stacking requires seven checkpoint scores and reserved stacking training rows')
    protocol = {
        'manifest_sha256': manifest_digest(manifest), 'model_order': MODELS,
        'prediction_sha256': {n: manifest_digest(predictions / f'{n}.json') for n in MODELS},
        'train_count': int(train.sum()), 'test_count': int(test.sum()),
        'features': 'Seven frozen detector scores converted to clipped log-odds; linear/SVM scaling fitted on train only',
        'selection': 'Fixed settings; no validation, early stopping, test tuning or threshold search',
        'svm_scores': 'Sigmoid of decision margin for storage only, NOT calibrated probabilities; decision threshold is margin > 0',
        'test_note': 'Reused existing benchmark test split, previously inspected',
        'sklearn_version': sklearn.__version__, 'xgboost_version': xgboost.__version__,
        'parameters': {n: {k: v for k, v in m.get_params(deep=True).items()
                          if isinstance(v, (str, int, float, bool, type(None)))} for n, m in models.items()},
        'mlp_config': DEFAULT_CONFIG,
    }
    if fine_tuned:
        protocol['detector_checkpoint_sha256'] = checkpoint_hashes
        protocol['features'] = 'Seven fine-tuned detector scores converted to clipped log-odds; linear/SVM scaling fitted on reserved stacking train only'
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Protocol changed; choose a new output directory')
    path.write_text(json.dumps(protocol, indent=2) + '\n')
    status = output / 'status.json'
    status.write_text(json.dumps({'status': 'running'}))
    z = transform_scores(x)
    scores = {name: x[test, i] for i, name in enumerate(MODELS)}
    scores.update(votes(x[test]))
    results = {}
    for name, model in models.items():
        print('Training', name, flush=True)
        model.fit(z[train], y[train])
        joblib.dump({'estimator': model, 'model_order': MODELS,
                     'feature_transform': 'ai_detector.ensemble.transform_scores',
                     'protocol': protocol}, output / f'{name}.joblib')
        if name.startswith('svm_'):
            margin = model.decision_function(z[test])
            p = expit(margin)
        else:
            p = model.predict_proba(z[test])[:, list(model.classes_).index(1)]
        scores[name] = p
    print('Training reference MLP with original settings', flush=True)
    model, mean, scale, history = fit_meta(x[train], y[train], DEFAULT_CONFIG)
    import torch
    torch.save({'state_dict': model.state_dict(), 'mean': torch.from_numpy(mean),
                'scale': torch.from_numpy(scale), 'model_order': MODELS,
                'config': DEFAULT_CONFIG}, output / 'meta_classifier.pt')
    scores['meta_mlp'] = predict_meta(model, mean, scale, x[test])
    results = {name: metrics(y[test], p) for name, p in scores.items()}
    (output / 'metrics.json').write_text(json.dumps(results, indent=2) + '\n')
    with (output / 'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['method'] + list(next(iter(results.values()))))
        writer.writeheader()
        writer.writerows({'method': name, **m} for name, m in results.items())
    with (output / 'heldout_predictions.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'sha256', 'source', 'label'] + list(scores))
        for i, r in enumerate(r for r in rows if r['split'] == 'test'):
            writer.writerow([r['path'], r['sha256'], r['group'], r['label']] + [float(p[i]) for p in scores.values()])
    render_summary(results, output / 'summary.png', int(train.sum()), int(test.sum()))
    status.write_text(json.dumps({'status': 'complete', 'train_count': int(train.sum()), 'test_count': int(test.sum())}, indent=2))
    print(json.dumps(results, indent=2), flush=True)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=ROOT / 'data/wildfc/ensemble_split/all.jsonl')
    p.add_argument('--predictions', type=Path, default=ROOT / '.cache/predictions/pretrained')
    p.add_argument('--output', type=Path, default=ROOT / 'reports/pretrained_meta')
    args = p.parse_args()
    configure_environment(offline=True)
    with threadpool_limits(limits=2):
        run(args.manifest, args.predictions, args.output)


if __name__ == '__main__':
    main()
