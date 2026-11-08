"""Local checkpoint paths and prediction provenance."""

from pathlib import Path

from .config import CACHE, DETECTORS, UP, SPAI_CHECKPOINT, SPAI_CONFIG
from .manifests import manifest_digest


def base_signature(name):
    from .config import COMMUNITY_REPO, COMMUNITY_REVISION

    if name in ("npr", "npr_no_crop"):
        path = UP / "npr" / "NPR.pth"
        return {"npr/NPR.pth": manifest_digest(path)}

    if name == "community_forensics":
        from huggingface_hub import hf_hub_download

        paths = [
            Path(
                hf_hub_download(
                    COMMUNITY_REPO,
                    "model.safetensors",
                    revision=COMMUNITY_REVISION,
                    cache_dir=str(CACHE / "huggingface"),
                    local_files_only=True,
                )
            )
        ]
    elif name == "spai":
        paths = [SPAI_CHECKPOINT]
    elif name.startswith("aide_"):
        paths = [CACHE / f'detectors/aide/{name.removeprefix("aide_")}_train.pth']
    else:
        paths = {
            "safe": [CACHE / "detectors/safe/checkpoint-best.pth"],
            "universalfakedetect": [
                CACHE / "clip/ViT-L-14.pt",
                CACHE / "detectors/universalfakedetect/fc_weights.pth",
            ],
            "dda": [CACHE / "detectors/dda/DDA_ckpt.pth"],
        }[name]
    return {str(path.relative_to(CACHE)): manifest_digest(path) for path in paths}


def inference_signature(name, device):
    """Identify a prediction before loading the detector or reusing its scores."""
    inputs = [Path(__file__).with_name(filename) for filename in ("detectors.py", "runner.py")]
    revisions = (UP / source / "REVISION" for source in DETECTORS[name].sources)
    inputs.extend(path for path in revisions if path.is_file())
    if name == "spai":
        inputs.extend((Path(__file__).with_name("spai_adapter.py"), SPAI_CONFIG))
    return {
        "model": name,
        "device": device,
        "weights": base_signature(name),
        "code": {path.parent.name + "/" + path.name: manifest_digest(path) for path in inputs},
    }


