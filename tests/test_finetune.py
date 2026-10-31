"""Protect fine-tuning splits, checkpoint portability, and epoch recovery."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from PIL import Image
from torchvision import transforms as T

from ai_detector import finetune
from ai_detector.config import CACHE


class FineTuneTests(unittest.TestCase):
    def test_generated_views_allow_jpeg_and_restore_source_guard(self):
        # Reproduce the large-view failure cheaply with a reduced Pillow limit.
        view = Image.new("RGB", (30, 30))
        encoded = io.BytesIO()
        view.save(encoded, format="JPEG")
        data = encoded.getvalue()
        with patch.object(Image, "MAX_IMAGE_PIXELS", 100):
            with self.assertRaises(Image.DecompressionBombError):
                Image.open(io.BytesIO(data))
            with finetune.generated_image_limit({"real": view, "fake_resized": [view]}):
                with Image.open(io.BytesIO(data)) as decoded:
                    decoded.load()
                    self.assertEqual(decoded.size, view.size)
            self.assertEqual(Image.MAX_IMAGE_PIXELS, 100)
            with self.assertRaisesRegex(RuntimeError, "augmentation"):
                with finetune.generated_image_limit({"real": view}):
                    raise RuntimeError("augmentation")
            self.assertEqual(Image.MAX_IMAGE_PIXELS, 100)
            with self.assertRaises(Image.DecompressionBombError):
                Image.open(io.BytesIO(data))

    def setUp(self):
        (CACHE / "tmp").mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=CACHE / "tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_split_preserves_test_and_separates_articles(self):
        rows = [
            dict(
                path=f"p{i}",
                sha256=f"s{i}",
                pixel_sha256=f"px{i}",
                label=(i // 2) % 2,
                group=f"source{(i//2)%4}",
                leakage_group=f"article{i//2}",
                split="train" if i < 160 else "test",
            )
            for i in range(200)
        ]
        original = self.root / "original.jsonl"
        original.write_text("".join(json.dumps(r) + "\n" for r in rows))
        summary = finetune.prepare_split(original, self.root / "split")
        base = finetune.read_rows(self.root / "split/detector_train.jsonl")
        stack = finetune.read_rows(self.root / "split/stacking_train.jsonl")
        test = finetune.read_rows(self.root / "split/test.jsonl")
        self.assertEqual(test, rows[160:])
        self.assertFalse(
            {r["leakage_group"] for r in base}
            & {r["leakage_group"] for r in stack + test}
        )
        self.assertEqual(len(base) + len(stack), 160)
        self.assertEqual(
            json.loads(json.dumps(summary)),
            finetune.prepare_split(original, self.root / "split"),
        )
        for source in summary["sources"].values():
            if source["independent_groups"] > 1:
                self.assertGreater(source["detector"], 0)
                self.assertGreater(source["stacking"], 0)

    def test_partial_checkpoint_includes_buffers_and_preserves_frozen_weights(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3), torch.nn.Linear(3, 1)
        )
        model[0].requires_grad_(False)
        update = finetune.updated_state(model)
        self.assertNotIn("0.weight", update)
        self.assertIn("1.running_mean", update)
        self.assertIn("2.weight", update)
        before = model[0].weight.detach().clone()
        update["1.running_mean"].fill_(2)
        finetune.apply_updates(model, {"updates": update})
        torch.testing.assert_close(before, model[0].weight)
        self.assertEqual(float(model[1].running_mean[0]), 2.0)

    def test_resume_matches_uninterrupted_training(self):
        rows = []
        for i in range(48):
            path = self.root / f"image{i}.png"
            Image.new("RGB", (8, 8), (i * 30, 10, 20)).save(path)
            rows.append(
                dict(
                    path=str(path),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    label=i % 2,
                    split="train",
                    training_role="detector",
                    group="test",
                )
            )
        manifest = self.root / "train.jsonl"
        manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
        config = dict(
            epochs=2,
            microbatch=2,
            effective_batch=2,
            lr=0.01,
            weight_decay=0.0,
            smoothing=0.0,
            warmup_fraction=0.0,
            family="community_forensics",
        )

        def build(*args):
            return (
                torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(192, 1)),
                T.ToTensor(),
                {"binary": True},
            )

        args = SimpleNamespace(
            output=self.root / "full",
            model="community_forensics",
            epochs=None,
            microbatch=None,
            manifest=manifest,
            device="cpu",
            smoke_steps=0,
        )
        with patch("ai_detector.detectors.build_detector", side_effect=build), patch(
            "ai_detector.training_recipes.recipe", return_value=config
        ), patch(
            "ai_detector.training_recipes.train_transform", return_value=T.ToTensor()
        ), patch(
            "ai_detector.finetune.base_signature", return_value={}
        ), contextlib.redirect_stdout(
            io.StringIO()
        ):
            finetune.train(args)
            expected = torch.load(
                args.output / "checkpoints/community_forensics/finetuned.pt",
                weights_only=True,
            )
            args.output = self.root / "resumed"
            real_save = finetune.save_torch

            def interrupt(path, data):
                real_save(path, data)
                if (
                    path.name == "last.pt"
                    and data["epoch"] == 0
                    and data["next_batch"] == 20
                ):
                    raise RuntimeError("simulated interruption")

            with patch("ai_detector.finetune.save_torch", side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    finetune.train(args)
            finetune.train(args)
            actual = torch.load(
                args.output / "checkpoints/community_forensics/finetuned.pt",
                weights_only=True,
            )
        self.assertEqual(expected["steps"], actual["steps"])
        self.assertEqual(expected["history"], actual["history"])
        for name, value in expected["updates"].items():
            torch.testing.assert_close(value, actual["updates"][name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
