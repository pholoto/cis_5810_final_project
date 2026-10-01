"""Regression tests for portability, resumability, extraction, and stacking."""

import contextlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ai_detector import runner
from ai_detector.config import CACHE, MODELS
from ai_detector.manifests import manifest_digest
from scripts.extract_wildfc import extract_archive


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        (CACHE / "tmp").mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=CACHE / "tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def manifest(self):
        rows = []
        import hashlib

        for i in range(2):
            image = self.root / f"image{i}.jpg"
            image.write_bytes(b"test image")
            rows.append(
                {
                    "path": str(image),
                    "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                    "label": i,
                    "group": "test",
                }
            )
        manifest = self.root / "manifest.jsonl"
        manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return manifest, rows

    def test_resume_incomplete_last_line(self):
        manifest, rows = self.manifest()
        out = self.root / "output"
        out.mkdir()
        first = {**rows[0], "score": 0.2, "prediction": 0, "seconds": 0}
        (out / "safe.progress.jsonl").write_text(json.dumps(first) + '\n{"partial":')
        (out / "safe.progress.meta.json").write_text(
            json.dumps({"input_manifest_sha256": manifest_digest(manifest)})
        )
        calls = []
        with patch.object(
            runner,
            "make_detector",
            return_value=(lambda path: calls.append(path) or 0.8, {}),
        ), contextlib.redirect_stdout(io.StringIO()):
            runner.worker(
                SimpleNamespace(
                    manifest=manifest, output=out, model="safe", device="cpu"
                )
            )
        self.assertEqual(len(calls), 1)
        self.assertTrue(runner.matching_result(out / "safe.json", manifest))
        rows.reverse()
        manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.assertFalse(runner.matching_result(out / "safe.json", manifest))

    def test_gpu_index_forwarded_to_every_sequential_worker(self):
        manifest, _ = self.manifest()
        out = self.root / "output"
        args = [
            "ai_detector",
            "--manifest",
            str(manifest),
            "--output",
            str(out),
            "--device",
            "cuda:0",
            "--gpu-index",
            "7",
            "--scores-only",
        ]
        with patch("sys.argv", args), patch.object(
            runner.subprocess, "run"
        ) as run, patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()):
            runner.main()
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "7")
        self.assertEqual(run.call_count, 7)
        for call in run.call_args_list:
            command = call.args[0]
            self.assertEqual(command[command.index("--gpu-index") + 1], "7")
        self.assertEqual(
            json.loads((out / "run_config.json").read_text())["gpu_index"], 7
        )

    def test_failure_records_actionable_status(self):
        import subprocess

        manifest, _ = self.manifest()
        out = self.root / "failed"
        args = [
            "ai_detector",
            "--manifest",
            str(manifest),
            "--output",
            str(out),
            "--device",
            "cpu",
            "--scores-only",
        ]
        with patch("sys.argv", args), patch.object(
            runner.subprocess,
            "run",
            side_effect=subprocess.CalledProcessError(1, ["worker"]),
        ), patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "rerun the same command"):
                runner.main()
        status = json.loads((out / "run_status.json").read_text())
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["failed_model"], "community_forensics")

    def test_python_wildfc_modes_default_to_gpu(self):
        from scripts.run_wildfc import main

        for mode in ("all", "model", "voting", "ensemble"):
            args = ["run_wildfc.py", "--mode", mode]
            if mode == "model":
                args += ["--model", "dda"]
            with patch("sys.argv", args), patch(
                "scripts.run_wildfc.subprocess.run"
            ) as execute, contextlib.redirect_stdout(io.StringIO()):
                main()
            self.assertEqual(execute.call_count, 3)
            inference = execute.call_args_list[1].args[0]
            self.assertEqual(inference[inference.index("--device") + 1], "cuda:0")
            self.assertEqual(inference[inference.index("--gpu-index") + 1], "0")
            self.assertEqual("--model" in inference, mode == "model")
            evaluation = execute.call_args_list[2].args[0]
            self.assertEqual(evaluation[evaluation.index("--mode") + 1], mode)

    def test_archive_rejects_traversal_and_links(self):
        for name, link in [("../escape", False), ("link", True)]:
            archive = self.root / "bad.tar.gz"
            with tarfile.open(archive, "w:gz") as f:
                member = tarfile.TarInfo(name)
                if link:
                    member.type = tarfile.SYMTYPE
                    member.linkname = "/tmp/outside"
                f.addfile(member)
            with self.assertRaises(ValueError):
                extract_archive(archive, self.root / "extracted")

    def test_archive_regular_files(self):
        archive = self.root / "good.tar.gz"
        with tarfile.open(archive, "w:gz") as f:
            member = tarfile.TarInfo("folder/image")
            member.size = 3
            f.addfile(member, io.BytesIO(b"abc"))
        for _ in range(2):
            extract_archive(archive, self.root / "extracted")
        self.assertEqual((self.root / "extracted/folder/image").read_bytes(), b"abc")

    def test_meta_checkpoint_and_report(self):
        import numpy as np
        import torch
        from ai_detector.ensemble import run, MetaClassifier, predict_meta, votes

        rows = [
            {
                "path": f"image{i}",
                "sha256": f"sha{i}",
                "pixel_sha256": f"pixel{i}",
                "leakage_group": f"article{i}",
                "label": i % 2,
                "group": f"source{i%4}",
                "split": "train" if i < 3200 else "test",
            }
            for i in range(4000)
        ]
        manifest = self.root / "ensemble.jsonl"
        manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
        base = self.root / "base"
        base.mkdir()
        out = self.root / "ensemble"
        for j, name in enumerate(MODELS):
            results = [
                {**r, "score": (0.8 if r["label"] else 0.2) + j * 0.005} for r in rows
            ]
            (base / f"{name}.json").write_text(
                json.dumps(
                    {
                        "input_manifest_sha256": manifest_digest(manifest),
                        "results": results,
                    }
                )
            )
        with contextlib.redirect_stdout(io.StringIO()):
            run(manifest, base, out)
        result = json.loads((out / "metrics.json").read_text())
        self.assertEqual(result["meta_mlp"]["accuracy"], 1)
        ck = torch.load(out / "meta_classifier.pt", weights_only=True)
        model = MetaClassifier()
        model.load_state_dict(ck["state_dict"])
        self.assertTrue(
            np.isfinite(
                predict_meta(
                    model, ck["mean"].numpy(), ck["scale"].numpy(), np.full((2, 7), 0.5)
                )
            ).all()
        )
        self.assertTrue((out / "summary.png").exists())
        v = votes(np.array([[1, 1, 0, 0, 1, 1, 1], [1, 1, 1, 0, 0, 0, 0]]))
        self.assertEqual(v["majority_vote_7"].tolist(), [5 / 7, 3 / 7])
        self.assertEqual(v["family_vote_5"].tolist(), [3 / 5, 3 / 5])
        # Voting and single-model modes must never train a meta-network.
        with patch(
            "ai_detector.ensemble.fit_meta",
            side_effect=AssertionError("Unexpected training"),
        ), contextlib.redirect_stdout(io.StringIO()):
            run(manifest, base, self.root / "voting", mode="voting")
            run(manifest, base, self.root / "single", mode="model", model="dda")
        self.assertFalse((self.root / "voting/meta_classifier.pt").exists())
        self.assertEqual(
            set(json.loads((self.root / "single/metrics.json").read_text())), {"dda"}
        )
        self.assertIn(
            "majority_vote_7",
            json.loads((self.root / "voting/metrics.json").read_text()),
        )
        bad = base / f"{MODELS[0]}.json"
        d = json.loads(bad.read_text())
        d["results"].reverse()
        bad.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError, "misaligned"):
            run(manifest, base, self.root / "bad")


if __name__ == "__main__":
    unittest.main()
