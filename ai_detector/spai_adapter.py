"""Adapt the pinned SPAI MFM detector to the shared logit interface."""

from collections import OrderedDict
from pathlib import Path

import torch

from .config import SPAI_CHECKPOINT, SPAI_CONFIG, SPAI_REVISION, UP


def preprocess(image, minimum=224):
    import cv2
    import numpy as np
    from torchvision.transforms.functional import to_tensor

    width, height = image.size
    horizontal, vertical = max(0, minimum - width), max(0, minimum - height)
    if horizontal or vertical:
        image = cv2.copyMakeBorder(
            np.asarray(image), vertical // 2, vertical - vertical // 2,
            horizontal // 2, horizontal - horizontal // 2, cv2.BORDER_REFLECT_101,
        )
    return to_tensor(image)


class SPAI(torch.nn.Module):
    def __init__(self, detector, patch_batch=8):
        super().__init__()
        self.detector = detector
        self.patch_batch = patch_batch

    def forward(self, images):
        return self.detector([image.unsqueeze(0) for image in images], self.patch_batch)


def load_checkpoint_state(checkpoint):
    # Treat saved YACS metadata as plain data. PyTorch's restricted unpickler
    # accepts OrderedDict entries but cannot populate a CfgNode directly.
    with torch.serialization.safe_globals([(OrderedDict, "yacs.config.CfgNode")]):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return saved.get("model", saved)


def build_spai(device, *, checkpoint=SPAI_CHECKPOINT):
    from functools import partial
    from .detectors import module, package

    source = UP / "spai"
    revision = source / "REVISION"
    if not revision.is_file() or revision.read_text().strip() != SPAI_REVISION:
        raise RuntimeError("Fetch pinned SPAI code with scripts/fetch_detector_sources.py --only spai")
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"SPAI checkpoint missing: {checkpoint}. See README.md.")
    try:
        # Isolate upstream packages and skip the unused OpenAI CLIP dependency.
        # The released model uses MFM and never instantiates CLIPBackbone.
        package("spai", source / "spai")
        package("spai.models", source / "spai/models")
        module("spai.models.backbones", source / "spai/models/backbones.py", remove_unused_clip=True)
        from spai.config import get_config
        from spai.models.sid import build_mf_vit
    except ImportError as exc:
        raise RuntimeError("Install the optional requirements-spai.txt dependencies") from exc

    config = get_config({"cfg": str(SPAI_CONFIG)})
    state = load_checkpoint_state(checkpoint)
    detector = build_mf_vit(config)
    detector.load_state_dict(
        {key.removeprefix("module."): value for key, value in state.items()}, strict=True,
    )
    model = SPAI(detector).eval().to(device)
    return (
        model,
        partial(preprocess, minimum=config.DATA.IMG_SIZE),
        {
            "transform": "original RGB resolution, reflect-pad sides to at least 224, tensor [0,1]",
            "score": "sigmoid(fake logit)",
            "binary": True,
            "device": device,
            "strict_state_dict": True,
            "precision": "float32",
            "feature_extraction_batch": model.patch_batch,
            "upstream_revision": SPAI_REVISION,
        },
    )
