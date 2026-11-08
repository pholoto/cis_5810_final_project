"""Fetch pinned official source files; all writes stay under the repository."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ai_detector.config import NPR_REVISION, SPAI_REVISION

REPOS = {
    "community": (
        "JeongsooP/Community-Forensics",
        "ee5b71d43db0f3779e1edd64ee927b13f2dd6ad4",
    ),
    "safe": ("Ouxiang-Li/SAFE", "4e998724651b227def64f5be0cd60c0aa1552c35"),
    "ufd": (
        "WisconsinAIVision/UniversalFakeDetect",
        "030495aea3300a8b54c0ec37ec7fe1dd7e63c619",
    ),
    "dda": ("roy-ch/Dual-Data-Alignment", "8b9c06e75e63f4688bc25ac43a7e3412878cf67f"),
    "aide": ("shilinyan99/AIDE", "6725b710d5c437ab2f59792908ce0377dfc907de"),
    "dinov2": ("facebookresearch/dinov2", "7764ea0f912e53c92e82eb78a2a1631e92725fc8"),
    "npr": ("chuangchuangtan/NPR-DeepfakeDetection", NPR_REVISION),
    "spai": ("mever-team/spai", SPAI_REVISION),
}


def fetch(item):
    name, (repo, sha) = item
    base = ROOT / ".cache/upstream" / name
    response = requests.get(
        f"https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1", timeout=60
    )
    response.raise_for_status()
    for item in response.json()["tree"]:
        file = item["path"]
        if item["type"] != "blob":
            continue
        if name == "dinov2":
            keep = file in ("LICENSE", "hubconf.py", "dinov2/__init__.py") or (
                file.startswith(("dinov2/layers/", "dinov2/models/", "dinov2/hub/"))
                and file.endswith(".py")
            )

        elif name == "spai":
            keep = file in ("LICENSE", "configs/spai.yaml") or (
                file.startswith("spai/") and file.endswith(".py")
            )
        else:
            keep = (
                (name == "npr" and file == "NPR.pth")
                or file == "LICENSE"
                or file.endswith(
                    (".py", ".sh", ".json", ".yaml", ".yml", ".txt", ".md")
                )
            )

        if not keep:
            continue
        path = base / file
        if (
            path.exists()
            and (base / "REVISION").exists()
            and (base / "REVISION").read_text().strip() == sha
        ):
            continue
        response = requests.get(
            f"https://raw.githubusercontent.com/{repo}/{sha}/{file}", timeout=60
        )
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    (base / "REVISION").write_text(sha + "\n")
    print(name, sha, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=REPOS, help="Fetch one source (SPAI is opt-in)")
    args = parser.parse_args()
    sources = (
        [(args.only, REPOS[args.only])] if args.only
        else [(name, spec) for name, spec in REPOS.items() if name != "spai"]
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, sources))
