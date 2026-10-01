"""Snapshot input manifests and validate cached prediction alignment."""

import hashlib
import json


def manifest_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matching_result(path, manifest):
    """Only reuse results produced from the exact immutable input snapshot."""
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
        records = [json.loads(line) for line in manifest.read_text().splitlines()]
        return (
            data.get("input_manifest_sha256") == manifest_digest(manifest)
            and len(data["results"]) == len(records)
            and all(
                (a["path"], a.get("label"), a.get("sha256"))
                == (b["path"], b.get("label"), b.get("sha256"))
                for a, b in zip(data["results"], records)
            )
        )
    except (ValueError, KeyError, TypeError):
        return False


def snapshot_manifest(source, runtime):
    # The sampler appends complete JSONL records. Ignore an in-flight last line.
    content = source.read_bytes()
    if content and not content.endswith(b"\n"):
        content = content[: content.rfind(b"\n") + 1]
    records = [json.loads(line) for line in content.splitlines()]
    if not records:
        raise ValueError("No complete records in manifest")
    digest = hashlib.sha256(content).hexdigest()
    directory = runtime / "manifests"
    directory.mkdir(exist_ok=True)
    target = directory / f"{digest}.jsonl"
    if not target.exists():
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(content)
        temporary.replace(target)
    return target, len(records), digest
