"""Regression coverage for detector adapters and ordered ensemble selection."""

import contextlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from ai_detector.config import EXTENDED_MODELS, MODELS, selected_models
from ai_detector.detectors import build_detector
from ai_detector.ensemble import votes
from ai_detector.finetune import collate_variable_images
from ai_detector.spai_adapter import SPAI, load_checkpoint_state, preprocess
from ai_detector.training_recipes import loss_for, recipe, train_transform, trainable_model, training_mode


class DetectorExtensionTests(unittest.TestCase):
    def test_aide_training_transform_returns_five_views(self):
        data = SimpleNamespace(
            DCT_base_Rec_Module=lambda: lambda x: [x + 1, x + 2, x + 3, x + 4],
            transform_before=lambda _: torch.zeros(3, 8, 8),
            transform_train=lambda x: x,
        )
        with patch("ai_detector.training_recipes.package"), patch(
            "ai_detector.training_recipes.module", return_value=data,
        ):
            transform = train_transform("aide_GenImage")
        output = transform(Image.new("RGB", (8, 8)))
        self.assertEqual(output.shape, (5, 3, 8, 8))
        self.assertEqual(output[:, 0, 0, 0].tolist(), [1, 2, 3, 4, 0])

    def test_npr_preprocessing_metadata_and_strict_checkpoint(self):
        def model(num_classes):
            return torch.nn.Linear(1, num_classes)

        fake_resnet = SimpleNamespace(resnet50=model)
        image = Image.new("RGB", (280, 300))
        state = {"model": {"module." + key: value for key, value in model(1).state_dict().items()}}
        with patch.dict(sys.modules, {"networks.resnet": fake_resnet}), patch(
            "sys.path", sys.path[:],
        ), patch("torch.load", return_value=state) as load:
            for name, size, description in (
                ("npr", 224, "center crop 224"), ("npr_no_crop", 256, "no crop"),
            ):
                loaded, transform, metadata = build_detector(name)
                self.assertTrue(torch.equal(loaded.weight, state["model"]["module.weight"]))
                self.assertEqual(transform(image).shape, (3, size, size))
                self.assertIn(description, metadata["transform"])
            self.assertTrue(load.call_args.kwargs["weights_only"])

    def test_family_votes_preserve_legacy_and_follow_reordered_names(self):
        self.assertEqual(selected_models(), MODELS)
        self.assertEqual(selected_models(include_npr=True), EXTENDED_MODELS)
        scores = np.array([[1, 0, 1, 0, 1, 1, 0, 0, 1]])
        names = [*EXTENDED_MODELS, "spai"]
        self.assertEqual(votes(scores[:, :7])["family_vote_5"].item(), 3 / 5)
        self.assertEqual(votes(scores[:, :8])["family_vote_6"].item(), 3 / 6)
        order = [8, 5, 0, 6, 1, 3, 2, 7, 4]
        result = votes(scores[:, order], [names[i] for i in order])
        self.assertEqual(result["family_vote_7"].item(), 4 / 7)
        self.assertEqual(result["majority_vote_9"].item(), 5 / 9)
        with self.assertRaisesRegex(ValueError, "duplicates"):
            selected_models(["spai", "spai"])

    def test_spai_variable_resolution_and_head_only_training(self):
        class PatchDetector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.features = torch.nn.Linear(1, 2)
                self.cls_head = torch.nn.Linear(2, 1)

            def forward(self, images, patch_batch):
                self.patch_batch = patch_batch
                self.shapes = [tuple(x.shape) for x in images]
                means = torch.stack([image.mean() for image in images]).unsqueeze(-1)
                return self.cls_head(self.features(means))

        base = PatchDetector()
        model = SPAI(base)
        trainable_model(model, "spai")
        training_mode(model, "spai")
        images, labels = collate_variable_images([
            (preprocess(Image.new("RGB", (13, 300), (255, 0, 0))), 1),
            (preprocess(Image.new("RGB", (450, 224))), 0),
        ])
        self.assertEqual(images[0][0].min().item(), 1)
        loss = loss_for(model, images, labels, "spai", recipe("spai"))
        loss.backward()
        self.assertEqual(base.shapes, [(1, 3, 300, 224), (1, 3, 224, 450)])
        self.assertEqual(base.patch_batch, 8)
        self.assertIsNone(base.features.weight.grad)
        self.assertIsNotNone(base.cls_head.weight.grad)
        self.assertTrue(base.cls_head.training)
        self.assertFalse(base.features.training)

    def test_spai_checkpoint_loads_weights_with_yacs_metadata(self):
        try:
            from yacs.config import CfgNode
        except ImportError:
            self.skipTest("Install optional SPAI dependencies to test YACS checkpoint metadata")
        checkpoint = io.BytesIO()
        torch.save({
            "model": {"weight": torch.ones(2)},
            "config": CfgNode({"MODEL": {"TYPE": "vit"}}),
        }, checkpoint)
        checkpoint.seek(0)
        self.assertTrue(torch.equal(load_checkpoint_state(checkpoint)["weight"], torch.ones(2)))

    def test_wildfc_forwards_explicit_roster_in_order(self):
        from scripts import run_wildfc

        argv = ["run_wildfc.py", "--mode", "voting", "--models", "spai", "npr", "safe"]
        with patch("sys.argv", argv), patch.object(run_wildfc.subprocess, "run") as execute, (
            contextlib.redirect_stdout(io.StringIO())
        ):
            run_wildfc.main()
        for call in execute.call_args_list[1:]:
            command = call.args[0]
            self.assertEqual(command[command.index("--models") + 1:], ["spai", "npr", "safe"])
        output = Path(execute.call_args.args[0][execute.call_args.args[0].index("--output") + 1])
        self.assertTrue(output.name.startswith("pretrained_voting_models_"))

    def test_spai_predict_rejects_changed_configuration(self):
        from ai_detector.finetune import predict

        saved = {
            "model": "spai",
            "config": {
                "base_signature": {},
                "spai_config_sha256": "intentionally-wrong",
            },
        }
        args = SimpleNamespace(
            model="spai",
            output=Path("unused"),
            manifest=Path("unused"),
            device="cpu",
        )
        with patch("torch.load", return_value=saved), patch(
            "ai_detector.finetune.base_signature", return_value={}
        ):
            with self.assertRaisesRegex(ValueError, "SPAI configuration changed"):
                predict(args)

    def test_spai_changed_inference_signature_invalidates_cache(self):
        import json
        import tempfile
        from ai_detector.config import SPAI_CONFIG
        from ai_detector.finetune import predict
        from ai_detector.manifests import manifest_digest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoints/spai/finetuned.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"test checkpoint")
            manifest = root / "manifest.jsonl"
            manifest.write_text("")
            cache = root / "base_predictions/spai.json"
            cache.parent.mkdir()
            cache.write_text(json.dumps({
                "checkpoint_sha256": manifest_digest(checkpoint),
                "input_manifest_sha256": manifest_digest(manifest),
                "inference_signature": {"revision": "old"},
                "results": [],
            }))
            saved = {
                "model": "spai",
                "config": {
                    "base_signature": {},
                    "spai_config_sha256": manifest_digest(SPAI_CONFIG),
                },
            }
            args = SimpleNamespace(model="spai", output=root, manifest=manifest, device="cpu")
            with patch("torch.load", return_value=saved), patch(
                "ai_detector.finetune.base_signature", return_value={}
            ), patch(
                "ai_detector.artifacts.inference_signature", return_value={"revision": "new"}
            ), patch("ai_detector.detectors.build_detector", side_effect=RuntimeError("cache invalidated")):
                with self.assertRaisesRegex(RuntimeError, "cache invalidated"):
                    predict(args)

    def test_spai_changed_signature_discards_partial_cache(self):
        import json
        import tempfile
        from ai_detector.config import SPAI_CONFIG
        from ai_detector.finetune import predict
        from ai_detector.manifests import manifest_digest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoints/spai/finetuned.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"test checkpoint")
            manifest = root / "manifest.jsonl"
            manifest.write_text("")
            output = root / "base_predictions"
            output.mkdir()
            progress = output / "spai.progress.jsonl"
            progress_meta = output / "spai.progress.meta.json"
            progress.write_text(json.dumps({"path": "stale-image"}) + "\n")
            progress_meta.write_text(json.dumps({
                "checkpoint_sha256": manifest_digest(checkpoint),
                "input_manifest_sha256": manifest_digest(manifest),
                "inference_signature": {"revision": "old"},
            }))
            saved = {
                "model": "spai",
                "config": {
                    "base_signature": {},
                    "spai_config_sha256": manifest_digest(SPAI_CONFIG),
                },
            }
            args = SimpleNamespace(model="spai", output=root, manifest=manifest, device="cpu")
            with patch("torch.load", return_value=saved), patch(
                "ai_detector.finetune.base_signature", return_value={}
            ), patch(
                "ai_detector.artifacts.inference_signature", return_value={"revision": "new"}
            ), patch(
                "ai_detector.detectors.build_detector", side_effect=RuntimeError("fresh inference")
            ):
                with self.assertRaisesRegex(RuntimeError, "fresh inference"):
                    predict(args)
            self.assertEqual(progress.read_text(), "")
            self.assertEqual(json.loads(progress_meta.read_text())["inference_signature"], {"revision": "new"})
