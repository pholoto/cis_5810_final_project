"""Check installed dependencies, pinned sources, and downloaded checkpoints offline."""

import importlib
import sys
from .config import CACHE, UP, COMMUNITY_REPO, COMMUNITY_REVISION, configure_environment


def main():
    configure_environment(offline=True)
    failures = []
    print(f"Python: {sys.version.split()[0]} (3.12 or 3.13 supported)")
    if not (3, 12) <= sys.version_info[:2] <= (3, 13):
        failures.append("Python 3.12 or 3.13")
    for name in (
        "pkg_resources",
        "torch",
        "torchvision",
        "timm",
        "huggingface_hub",
        "safetensors",
        "PIL",
        "numpy",
        "sklearn",
        "kornia",
        "pytorch_wavelets",
        "pywt",
        "open_clip",
        "requests",
        "bs4",
    ):
        try:
            importlib.import_module(name)
            print(f"OK dependency: {name}")
        except Exception as exc:
            failures.append(name)
            print(f"MISSING/BROKEN dependency: {name}: {exc}")
    required = [
        UP / "safe/models/resnet.py",
        UP / "ufd/models/clip/model.py",
        UP / "dda/Inference/models/lora.py",
        UP / "aide/models/AIDE.py",
        UP / "dinov2/dinov2/hub/backbones.py",
        CACHE / "clip/ViT-L-14.pt",
        CACHE / "detectors/safe/checkpoint-best.pth",
        CACHE / "detectors/universalfakedetect/fc_weights.pth",
        CACHE / "detectors/dda/DDA_ckpt.pth",
    ]
    required.extend(
        CACHE / f"detectors/aide/{n}_train.pth" for n in ("GenImage", "progan", "sd14")
    )
    for path in required:
        if path.is_file() and path.stat().st_size:
            print(f"OK file: {path.relative_to(CACHE)}")
        else:
            failures.append(str(path))
            print(f"MISSING: {path}")
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            COMMUNITY_REPO,
            "model.safetensors",
            revision=COMMUNITY_REVISION,
            cache_dir=str(CACHE / "huggingface"),
            local_files_only=True,
        )
        print("OK Community Forensics checkpoint")
    except Exception:
        failures.append("Community Forensics checkpoint")
        print("MISSING Community Forensics checkpoint")
    try:
        import torch

        print(
            f"CUDA available: {torch.cuda.is_available()} (not required for CPU mode)"
        )
    except ImportError:
        pass
    if failures:
        print(
            "Setup incomplete. Follow README steps 2 and 3; rerun this check afterward."
        )
        raise SystemExit(1)
    print(
        "Setup ready. File checks confirm presence, not a full checksum audit or inference test."
    )


if __name__ == "__main__":
    main()
