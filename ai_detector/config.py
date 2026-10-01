"""Shared model identifiers and repository-local runtime paths."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
UP = CACHE / "upstream"
MODELS = [
    "community_forensics",
    "safe",
    "universalfakedetect",
    "dda",
    "aide_GenImage",
    "aide_progan",
    "aide_sd14",
]
COMMUNITY_REPO = "OwensLab/commfor-model-384"
COMMUNITY_REVISION = "6076002bf0d9dd37537f965ee2f06f826c333b61"


def configure_environment(*, offline=False):
    """Keep downloads and temporary files local; call before importing HF/Torch."""
    paths = {
        "HF_HOME": CACHE / "huggingface",
        "HF_XET_CACHE": CACHE / "huggingface/xet",
        "TORCH_HOME": CACHE / "torch",
        "XDG_CACHE_HOME": CACHE,
        "TMPDIR": CACHE / "tmp",
        "MPLCONFIGDIR": CACHE / "matplotlib",
        "PIP_CACHE_DIR": CACHE / "pip",
    }
    for key, value in paths.items():
        value.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(value)
    (CACHE / "detectors").mkdir(exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1" if offline else "0"
