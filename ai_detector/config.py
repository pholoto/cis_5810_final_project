"""Shared model identifiers and repository-local runtime paths."""

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
UP = CACHE / "upstream"


@dataclass(frozen=True)
class Detector:
    label: str
    family: str
    sources: tuple[str, ...] = ()
    default: bool = True
    trainable: bool = True
    ensemble: bool = True


DETECTORS = {
    "community_forensics": Detector("Community Forensics", "community_forensics"),
    "safe": Detector("SAFE", "safe", ("safe",)),
    "universalfakedetect": Detector("UniversalFakeDetect", "universalfakedetect", ("ufd",)),
    "dda": Detector("DDA", "dda", ("dda", "dinov2")),
    "aide_GenImage": Detector("AIDE GenImage", "aide", ("aide",)),
    "aide_progan": Detector("AIDE ProGAN", "aide", ("aide",)),
    "aide_sd14": Detector("AIDE SD1.4", "aide", ("aide",)),
    "npr": Detector("NPR", "npr", ("npr",), default=False),
    "npr_no_crop": Detector(
        "NPR (no crop)", "npr", ("npr",), default=False, trainable=False, ensemble=False,
    ),
    "spai": Detector("SPAI", "spai", ("spai",), default=False),
}
MODELS = [name for name, spec in DETECTORS.items() if spec.default]
EXTENDED_MODELS = [*MODELS, "npr"]
ENSEMBLE_MODELS = [name for name, spec in DETECTORS.items() if spec.ensemble]
TRAINABLE_MODELS = [name for name, spec in DETECTORS.items() if spec.trainable]
COMMUNITY_REPO = "OwensLab/commfor-model-384"
COMMUNITY_REVISION = "6076002bf0d9dd37537f965ee2f06f826c333b61"
NPR_REVISION = "781ced3f7ca2cdc69ec9dd4ef27e8d0b3c07752a"
SPAI_REVISION = "8ff7b3b6779b4fcb43cf313471d9cb1c62d129a4"
SPAI_CONFIG = UP / "spai/configs/spai.yaml"
SPAI_CHECKPOINT = CACHE / "detectors/spai/spai.pth"


def selected_models(models=None, include_npr=False, *, training=False):
    names = list(models if models is not None else (EXTENDED_MODELS if include_npr else MODELS))
    allowed = TRAINABLE_MODELS if training else ENSEMBLE_MODELS
    if not names or len(set(names)) != len(names):
        raise ValueError("Select at least one detector, without duplicates")
    if invalid := [name for name in names if name not in allowed]:
        capability = "fine-tuning" if training else "ensembles"
        raise ValueError(f"Unsupported detectors for {capability}: {', '.join(invalid)}")
    return names


def add_model_selection(parser, *, training=False):
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--include-npr", action="store_true", help="Append NPR to the original seven")
    options.add_argument(
        "--models", nargs="+", choices=TRAINABLE_MODELS if training else ENSEMBLE_MODELS,
        help="Run the specified detectors in the given order",
    )


def parse_model_selection(parser, args, *, training=False):
    try:
        return selected_models(args.models, args.include_npr, training=training)
    except ValueError as exc:
        parser.error(str(exc))


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
