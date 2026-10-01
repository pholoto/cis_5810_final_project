"""Resume official detector weights; store all artifacts inside this repository."""

import argparse
import sys
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import zipfile
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ai_detector.config import (
    CACHE,
    COMMUNITY_REPO,
    COMMUNITY_REVISION,
    configure_environment,
)

configure_environment(offline=False)
OUT = CACHE / "detectors"
CLIP_SHA = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
JOBS = [
    (
        "SAFE",
        OUT / "safe/checkpoint-best.pth",
        "https://raw.githubusercontent.com/Ouxiang-Li/SAFE/4e998724651b227def64f5be0cd60c0aa1552c35/checkpoint/checkpoint-best.pth",
        "b3f5ecfb46a154ed553aaaf4bf3ba59182310726ddb0cbb1fe42bd0e22d2f20e",
    ),
    (
        "UniversalFakeDetect head",
        OUT / "universalfakedetect/fc_weights.pth",
        "https://raw.githubusercontent.com/WisconsinAIVision/UniversalFakeDetect/030495aea3300a8b54c0ec37ec7fe1dd7e63c619/pretrained_weights/fc_weights.pth",
        "477100745713bcc957beb2b40859536859b6483fd6301b3b9293151b194c7847",
    ),
    (
        "UniversalFakeDetect backbone",
        CACHE / "clip/ViT-L-14.pt",
        f"https://openaipublic.azureedge.net/clip/models/{CLIP_SHA}/ViT-L-14.pt",
        CLIP_SHA,
    ),
    (
        "DDA",
        OUT / "dda/DDA_ckpt.pth",
        "https://huggingface.co/Junwei-Xi/Dual-Data-Alignment/resolve/4390d9023899196b437480bb6a441915ef5d816c/DDA_ckpt.pth",
        "b27a31d39374803ddeff02bfabb2be76e190b04300490cddfafb24f683f37e3e",
    ),
]
for name, file_id in [
    ("GenImage_train", "1ZJCJmzyIrbSOROS7bKTgSm-Fe6yHsVXz"),
    ("progan_train", "1mlLl9Ucjd4wTXWM-j0Zzu3ihvqL8uxzX"),
    ("sd14_train", "1JYD1QO62wHwrvh2MVWS1GPjV314WsHcx"),
]:
    JOBS.append(
        (
            f"AIDE {name}",
            OUT / f"aide/{name}.pth",
            f"https://drive.google.com/uc?export=download&id={file_id}",
            None,
        )
    )


def download(job):
    name, dest, source, expected = job
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    if not dest.exists():
        for attempt in range(5):
            try:
                session = requests.Session()
                offset = part.stat().st_size if part.exists() else 0
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                response = session.get(
                    source, headers=headers, stream=True, timeout=(30, 120)
                )
                response.raise_for_status()
                if "text/html" in response.headers.get("Content-Type", ""):
                    soup = BeautifulSoup(response.text, "html.parser")
                    form = soup.find("form", id="download-form")
                    if not form:
                        raise RuntimeError(
                            "Download returned HTML without a confirmation form: "
                            + soup.get_text(" ", strip=True)[:400]
                        )
                    params = {
                        i["name"]: i.get("value", "")
                        for i in form.find_all("input")
                        if i.get("name")
                    }
                    response.close()
                    response = session.get(
                        form["action"],
                        params=params,
                        headers=headers,
                        stream=True,
                        timeout=(30, 120),
                    )
                    response.raise_for_status()
                if "text/html" in response.headers.get("Content-Type", ""):
                    raise RuntimeError("Download returned HTML instead of weights")
                append = response.status_code == 206 and offset > 0
                if append and not response.headers.get("Content-Range", "").startswith(
                    f"bytes {offset}-"
                ):
                    raise RuntimeError("Unexpected resume range")
                total = int(response.headers.get("Content-Length", 0)) + (
                    offset if append else 0
                )
                print(
                    f"{name}: downloading, resume={offset if append else 0}, expected bytes={total}",
                    flush=True,
                )
                with part.open("ab" if append else "wb") as f:
                    for chunk in response.iter_content(8 * 1024 * 1024):
                        f.write(chunk)
                response.close()
                if total and part.stat().st_size != total:
                    raise RuntimeError("Incomplete download")
                break
            except Exception as exc:
                print(f"{name}: attempt {attempt + 1} failed: {exc}", flush=True)
                if attempt == 4:
                    raise
                time.sleep(5)
        candidate = part
    else:
        candidate = dest
    with candidate.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if expected and digest != expected:
        raise RuntimeError(
            f"{name}: SHA256 mismatch; partial file retained for inspection"
        )
    if not expected:
        if not zipfile.is_zipfile(candidate):
            raise RuntimeError(f"{name}: not a PyTorch ZIP checkpoint")
        with zipfile.ZipFile(candidate) as archive:
            bad = archive.testzip()
            if bad:
                raise RuntimeError(f"{name}: ZIP CRC failed: {bad}")
    if candidate == part:
        part.replace(dest)
    result = {
        "name": name,
        "path": str(dest.relative_to(ROOT)),
        "source": source,
        "bytes": dest.stat().st_size,
        "sha256": digest,
        "verification": (
            "pinned SHA256" if expected else "ZIP CRC; locally recorded SHA256"
        ),
    }
    dest.with_name(dest.name + ".json").write_text(json.dumps(result, indent=2) + "\n")
    print(f'{name}: VERIFIED {result["bytes"]} bytes, sha256={digest}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=["community_forensics", "safe", "universalfakedetect", "dda", "aide"],
        help="Download one detector family (default: all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Concurrent downloads, not inference workers",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    failures = []
    if args.only in (None, "community_forensics"):
        from huggingface_hub import hf_hub_download

        try:
            path = hf_hub_download(
                COMMUNITY_REPO,
                "model.safetensors",
                revision=COMMUNITY_REVISION,
                cache_dir=str(CACHE / "huggingface"),
            )
            print(f"Community Forensics: ready at {path}", flush=True)
        except Exception as exc:
            failures.append("Community Forensics")
            print(f"FAILED Community Forensics: {exc}", flush=True)
    family = lambda name: (
        "universalfakedetect"
        if name.startswith("Universal")
        else ("aide" if name.startswith("AIDE") else name.lower())
    )
    jobs = [job for job in JOBS if args.only is None or family(job[0]) == args.only]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(download, job): job[0] for job in jobs}
        for future in concurrent.futures.as_completed(pending):
            try:
                future.result()
            except Exception as exc:
                failures.append(pending[future])
                print(f"FAILED {pending[future]}: {exc}", flush=True)
    print("DONE; failures=" + repr(failures), flush=True)
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
