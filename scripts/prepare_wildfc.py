"""Fallback split for this WildFC release, which supplies no per-image train/test assignments.

Deduplicate originals and create a source-diverse, approximately 80:20 grouped split.
Do not apply this fallback to a future release that supplies an official partition.
"""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import re
import warnings
import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/wildfc"
OUT = BASE / "ensemble_split"
SEED = 42


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not (BASE / "extracted/wildfc/fact_check_articles.csv").exists():
        raise SystemExit(
            "Run python scripts/download_wildfc.py and python scripts/extract_wildfc.py first."
        )
    if (OUT / "split_summary.json").exists():
        print((OUT / "split_summary.json").read_text())
        return
    articles = list(
        csv.DictReader((BASE / "extracted/wildfc/fact_check_articles.csv").open())
    )
    lookup = {(r["year_month"], r["article_id"]): r for r in articles}
    tasks = []
    for label, folder in [(1, "wildfc/images"), (0, "0_real")]:
        for p in sorted((BASE / "extracted" / folder).iterdir()):
            if not p.is_file():
                continue
            if label:
                m = re.match(r"^(\d{4}-\d{2})_article_(\d+)_", p.name)
                if not m:
                    raise ValueError(f"Unknown article naming: {p.name}")
                month, aid = m.groups()
                a = lookup.get((month, aid), {})
                publisher = (
                    " ".join(a.get("publisher", "").casefold().split()) or "unknown"
                )
                publisher = {"afp fact check": "afp"}.get(publisher, publisher)
                source = "ai/" + publisher
                group = f"article/{month}/{aid}"
            else:
                m = re.match(r"^(\d{4}-\d{2})_real_(sites|x|insta|fb)_", p.name)
                if not m:
                    raise ValueError(f"Unknown real source naming: {p.name}")
                month, platform = m.groups()
                source = "real/" + platform
                group = "real/" + p.stem
            tasks.append((p, label, month, source, group))

    def inspect(task):
        p, label, month, source, group = task
        try:
            raw = hashlib.sha256(p.read_bytes()).hexdigest()
            with warnings.catch_warnings(record=True) as caught, Image.open(p) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.load()
                w, h = image.size
                if min(w, h) < 32:
                    return None, {
                        "path": str(p.relative_to(ROOT)),
                        "reason": "smaller than 32 pixels",
                    }
                pixel = hashlib.sha256(
                    f"{w}x{h}:RGB:".encode() + image.tobytes()
                ).hexdigest()
            return {
                "path": str(p.relative_to(ROOT)),
                "sha256": raw,
                "pixel_sha256": pixel,
                "label": label,
                "group": source,
                "leakage_group": group,
                "year_month": month,
                "width": w,
                "height": h,
                "dataset": "pthan12/WildFC",
            }, None
        except Exception as exc:
            return None, {
                "path": str(p.relative_to(ROOT)),
                "reason": type(exc).__name__,
            }

    rows = []
    excluded = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i, (r, error) in enumerate(pool.map(inspect, tasks)):
            if r:
                rows.append(r)
            else:
                excluded.append(error)
            if (i + 1) % 1000 == 0:
                print("Inspected", i + 1, "/", len(tasks), flush=True)
    parent = {r["leakage_group"]: r["leakage_group"] for r in rows}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(b)] = find(a)

    by_pixel = defaultdict(list)
    for r in rows:
        by_pixel[r["pixel_sha256"]].append(r)
    kept = []
    for digest, items in by_pixel.items():
        for r in items[1:]:
            union(items[0]["leakage_group"], r["leakage_group"])
        if len({r["label"] for r in items}) > 1:
            excluded.extend(
                {
                    "path": r["path"],
                    "reason": "conflicting labels for identical decoded pixels",
                }
                for r in items
            )
            continue
        kept.append(items[0])
        excluded.extend(
            {
                "path": r["path"],
                "reason": "duplicate decoded pixels",
                "retained_path": items[0]["path"],
            }
            for r in items[1:]
        )
    rows = sorted(kept, key=lambda r: r["path"])
    for r in rows:
        r["leakage_group"] = find(r["leakage_group"])
    strata = np.array([r["group"] for r in rows])
    groups = np.array([r["leakage_group"] for r in rows])
    labels = np.array([r["label"] for r in rows])
    group_rows = defaultdict(list)
    for i, r in enumerate(rows):
        group_rows[r["leakage_group"]].append(i)
    source_groups = {s: set(groups[strata == s]) for s in sorted(set(strata))}
    targets = np.array([round(sum(labels == label) * 0.2) for label in (0, 1)])
    candidates = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for _, test in StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=SEED
        ).split(np.zeros(len(rows)), strata, groups):
            chosen = set(groups[test])
            counts = Counter(strata[test])
            missing = sum(
                not (0 < len(chosen & gs) < len(gs))
                for gs in source_groups.values()
                if len(gs) >= 2
            )
            score = missing * 1e6 + sum(
                abs(np.bincount(labels[test], minlength=2) - targets)
            )
            candidates.append((score, chosen))
    chosen = min(candidates, key=lambda x: x[0])[1]
    # Guarantee both splits contain each source with >=2 independent groups.
    for source, gs in sorted(source_groups.items(), key=lambda x: len(x[1])):
        if len(gs) == 1:
            chosen -= gs
            continue
        if not chosen & gs:
            options = [
                g
                for g in gs
                if all(
                    len(other - chosen - {g}) >= 1
                    for other in source_groups.values()
                    if g in other
                )
            ]
            if not options:
                raise ValueError(f"Cannot preserve source coverage: {source}")
            chosen.add(min(options, key=lambda g: (len(group_rows[g]), g)))
        elif not gs - chosen:
            options = [
                g
                for g in gs
                if all(
                    len((other & chosen) - {g}) >= 1
                    for other in source_groups.values()
                    if g in other and len(other) >= 2
                )
            ]
            if not options:
                raise ValueError(f"Cannot preserve training coverage: {source}")
            chosen.remove(min(options, key=lambda g: (len(group_rows[g]), g)))
    vectors = {
        g: np.bincount(labels[idx], minlength=2) for g, idx in group_rows.items()
    }
    counts = sum((vectors[g] for g in chosen), np.zeros(2, dtype=int))
    # Match 20% per label while retaining source and group guarantees.
    for _ in range(2000):
        loss = sum(abs(counts - targets))
        best = None
        if loss == 0:
            break
        for g in sorted(group_rows):
            remove = g in chosen
            relevant = [gs for gs in source_groups.values() if g in gs]
            if remove and any(
                len(gs) >= 2 and len(gs & chosen) <= 1 for gs in relevant
            ):
                continue
            if not remove and any(len(gs - chosen) <= 1 for gs in relevant):
                continue
            new = counts + (-vectors[g] if remove else vectors[g])
            newloss = sum(abs(new - targets))
            if newloss < loss and (
                best is None or (newloss, len(group_rows[g]), g) < best[:3]
            ):
                best = (newloss, len(group_rows[g]), g, new)
        if best is None:
            break
        _, _, g, counts = best
        if g in chosen:
            chosen.remove(g)
        else:
            chosen.add(g)
    train = [r for r in rows if r["leakage_group"] not in chosen]
    test = [r for r in rows if r["leakage_group"] in chosen]
    for r in train:
        r["split"] = "train"
    for r in test:
        r["split"] = "test"
    assert not {r["leakage_group"] for r in train} & {r["leakage_group"] for r in test}
    assert not {r["pixel_sha256"] for r in train} & {r["pixel_sha256"] for r in test}
    coverage = []
    for source, gs in source_groups.items():
        ntrain = sum(r["group"] == source for r in train)
        ntest = sum(r["group"] == source for r in test)
        if len(gs) >= 2:
            assert ntrain and ntest, source
        coverage.append(
            {
                "source": source,
                "train": ntrain,
                "test": ntest,
                "independent_groups": len(gs),
                "coverage_note": (
                    "train only: one independent group"
                    if len(gs) == 1
                    else "both splits"
                ),
            }
        )
    for name, content in [("train", train), ("test", test), ("all", train + test)]:
        p = OUT / f"{name}.jsonl"
        tmp = p.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in content))
        tmp.replace(p)
    atomic_json(OUT / "excluded.json", excluded)
    summary = {
        "seed": SEED,
        "dataset_revision": "5f886904d97602ea44af8328177dc8cda292bd95",
        "input_original_images": len(tasks),
        "retained": len(rows),
        "excluded": len(excluded),
        "segmented_crops_used": False,
        "train_count": len(train),
        "test_count": len(test),
        "test_fraction": len(test) / len(rows),
        "label_counts": {
            name: dict(Counter(r["label"] for r in items))
            for name, items in [("train", train), ("test", test)]
        },
        "sources": coverage,
        "split_policy": "80:20 by label; publisher/platform stratified; article and identical-pixel connected groups kept together; singleton sources train only",
        "limitations": [
            "Generator IDs are not supplied; publisher/platform diversity is a proxy, not verified generator diversity.",
            "Perceptual/near-duplicate and event-level overlap beyond article identity has not been audited.",
        ],
    }
    atomic_json(OUT / "split_summary.json", summary)
    print(
        json.dumps({k: v for k, v in summary.items() if k != "sources"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
