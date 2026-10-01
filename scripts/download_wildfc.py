"""Download the full gated WildFC dataset into this repository and verify SHA256."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
DEST = ROOT / "data/wildfc"
for key, value in {
    "HF_HOME": CACHE / "huggingface",
    "HF_XET_CACHE": CACHE / "huggingface/xet",
    "XDG_CACHE_HOME": CACHE,
    "TMPDIR": CACHE / "tmp",
}.items():
    os.environ[key] = str(value)
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ.pop("HF_HUB_OFFLINE", None)
REPO = "pthan12/WildFC"
REVISION = "5f886904d97602ea44af8328177dc8cda292bd95"


def main():
    import fcntl
    from huggingface_hub import HfApi, hf_hub_download

    DEST.mkdir(parents=True, exist_ok=True)
    (CACHE / "tmp").mkdir(parents=True, exist_ok=True)
    with (CACHE / "wildfc_download.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        info = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
        files = {f.rfilename: f for f in info.siblings}
        names = ["wildfc.tar.gz", "real_data.tar.gz"]

        def download(name):
            remote = files[name]
            expected = remote.lfs.sha256
            print(f"START {name}: {remote.size} bytes", flush=True)
            for attempt in range(5):
                try:
                    path = Path(
                        hf_hub_download(
                            REPO,
                            name,
                            repo_type="dataset",
                            revision=REVISION,
                            local_dir=DEST,
                        )
                    )
                    break
                except Exception as error:
                    print(
                        f"RETRY {name}: attempt {attempt+1}, {type(error).__name__}",
                        flush=True,
                    )
                    if attempt == 4:
                        raise RuntimeError(f"Download failed: {name}") from None
                    time.sleep(15)
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if path.stat().st_size != remote.size or digest != expected:
                raise RuntimeError(f"Integrity check failed: {name}")
            record = {
                "filename": name,
                "bytes": remote.size,
                "sha256": digest,
                "verified_against": "Hugging Face upstream LFS SHA256",
            }
            print(f"VERIFIED {name}: {remote.size} bytes; SHA256 {digest}", flush=True)
            return record

        records = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in as_completed([pool.submit(download, name) for name in names]):
                records.append(future.result())
        hf_hub_download(
            REPO, "README.md", repo_type="dataset", revision=REVISION, local_dir=DEST
        )
        manifest = {
            "dataset": REPO,
            "revision": REVISION,
            "status": "complete",
            "archives": sorted(records, key=lambda r: r["filename"]),
            "total_archive_bytes": sum(r["bytes"] for r in records),
            "extracted": False,
        }
        temporary = DEST / "download_manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(DEST / "download_manifest.json")
        print(
            "DONE: both archives verified; total bytes =",
            manifest["total_archive_bytes"],
            flush=True,
        )


if __name__ == "__main__":
    main()
