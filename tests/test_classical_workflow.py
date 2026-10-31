"""Check stacking data boundaries, saved classifiers, and pipeline ordering."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from ai_detector import classical_ensemble as classical
from ai_detector.config import CACHE, MODELS
from ai_detector.ensemble import transform_scores
from scripts import run_wildfc as workflow


class ClassicalWorkflowTests(unittest.TestCase):
    def setUp(self):
        CACHE.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=CACHE)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_saved_classifier_uses_training_scaler_and_reproduces_scores(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(.1, .9, (60, 7))
        y = np.arange(60) % 2
        rows = [dict(path=str(i), sha256=str(i), pixel_sha256=str(i),
                     leakage_group=str(i), group='source', label=int(y[i]),
                     split='train' if i < 40 else 'test') for i in range(60)]
        manifest = self.root / 'manifest.jsonl'
        manifest.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        for name in MODELS:
            (self.root / f'{name}.json').write_text('{}')
        model = classical.classifiers(y[:40])['logistic_l2']
        out = self.root / 'output'
        with patch.object(classical, 'load_features', return_value=(rows, x)), \
             patch.object(classical, 'classifiers', return_value={'logistic_l2': model}), \
             patch.dict(classical.DEFAULT_CONFIG, epochs=2), \
             threadpool_limits(limits=2), contextlib.redirect_stdout(io.StringIO()):
            classical.run(manifest, self.root, out)
        saved = joblib.load(out / 'logistic_l2.joblib')['estimator']
        np.testing.assert_allclose(saved.named_steps['standardscaler'].mean_,
                                   transform_scores(x[:40]).mean(axis=0, dtype=np.float64))
        np.testing.assert_allclose(saved.predict_proba(transform_scores(x[40:])),
                                   model.predict_proba(transform_scores(x[40:])))
        self.assertTrue((out / 'summary.png').exists())
        # Fine-tuned scores must not accept detector-fitting images for stacking.
        for name in MODELS:
            (self.root / f'{name}.json').write_text('{"checkpoint_sha256":"test"}')
        with patch.object(classical, 'load_features', return_value=(rows, x)):
            with self.assertRaisesRegex(ValueError, 'reserved stacking'):
                classical.run(manifest, self.root, self.root / 'invalid')

    def test_pipeline_order_and_failure_stop(self):
        argv = ['run', '--mode', 'finetuned', '--gpu-index', '0']
        with patch('sys.argv', argv), patch.object(workflow.subprocess, 'run') as run, \
             contextlib.redirect_stdout(io.StringIO()):
            workflow.main()
        self.assertEqual(run.call_count, 2)
        first, second = [c.args[0] for c in run.call_args_list]
        self.assertIn('scripts/prepare_wildfc.py', first)
        self.assertIn('ai_detector.finetune', second)
        self.assertEqual(second[second.index('--gpu-index') + 1], '0')
        with patch('sys.argv', argv), patch.object(workflow.subprocess, 'run',
                side_effect=subprocess.CalledProcessError(1, first)) as run, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(subprocess.CalledProcessError):
                workflow.main()
        self.assertEqual(run.call_count, 1)

    def test_finetuned_pipeline_trains_all_stackers_in_same_folder(self):
        from ai_detector import finetune
        output = self.root / 'finetuned_meta'
        argv = ['finetune', '--output', str(output), '--gpu-index', '0']
        with patch('sys.argv', argv), patch.object(finetune, 'prepare_split',
                return_value={'counts': {}}), patch.object(finetune, 'configure_environment'), \
             patch.object(finetune.subprocess, 'run') as run, \
             contextlib.redirect_stdout(io.StringIO()):
            finetune.main()
        self.assertEqual(run.call_count, 15)  # Seven training + seven prediction workers + stacking.
        command = run.call_args_list[-1].args[0]
        self.assertIn('ai_detector.classical_ensemble', command)
        self.assertEqual(command[command.index('--output') + 1], str(output.resolve()))
        self.assertEqual(command[command.index('--manifest') + 1],
                         str(output.resolve() / 'splits/stacking_evaluation.jsonl'))
