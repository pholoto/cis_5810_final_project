"""Verify and extract WildFC archives without links or paths outside the dataset folder."""

import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/wildfc"


def extract_archive(archive, destination):
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Unsafe archive path: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                if target.is_file() and target.stat().st_size == member.size:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".extracting")
                with source.extractfile(member) as src, temporary.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                temporary.replace(target)
            else:
                raise ValueError(f"Unsupported archive entry: {member.name}")


def main():
    manifest_path = BASE / "download_manifest.json"
    if not manifest_path.exists():
        raise SystemExit("Run python scripts/download_wildfc.py first.")
    manifest = json.loads(manifest_path.read_text())
    for record in manifest["archives"]:
        archive = BASE / record["filename"]
        with archive.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"Archive checksum mismatch: {archive}")
        print(f"Extracting {archive.name}", flush=True)
        extract_archive(archive, BASE / "extracted")
    manifest["extracted"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Ready. Next: python scripts/prepare_wildfc.py")


if __name__ == "__main__":
    main()
