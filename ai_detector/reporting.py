"""Export per-model scores and readable PNG tables (no HTML)."""

import csv
import hashlib
import io
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    "community_forensics": "Community Forensics",
    "safe": "SAFE",
    "universalfakedetect": "UniversalFakeDetect",
    "dda": "DDA",
    "aide_GenImage": "AIDE GenImage",
    "aide_progan": "AIDE ProGAN",
    "aide_sd14": "AIDE SD1.4",
}
INK = "#172b43"
MUTED = "#52667a"
HEADER = "#183b5b"
BG = "#f3f6fa"


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default(size=size)


def text(draw, xy, value, size=20, fill=INK, bold=False):
    draw.text(xy, str(value), font=font(size, bold), fill=fill)


def fit(draw, value, width, size=17):
    value = str(value)
    while draw.textlength(value, font=font(size)) > width:
        value = value[:-2] + "…"
    return value


def save_png(image, path):
    # Palette conversion keeps even 1,000-image paginated reports within the
    # 50 MB allowance reserved above the 950 MB original-image budget.
    buffer = io.BytesIO()
    image.quantize(colors=128).save(buffer, format="PNG", optimize=True)
    if buffer.tell() > 500_000:
        raise RuntimeError(f"PNG page exceeds 500 KB report allowance: {path}")
    path.write_bytes(buffer.getvalue())


def render_report(manifest, output, models):
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    outputs = {
        name: json.loads((output / f"{name}.json").read_text()) for name in models
    }
    labeled = [r for r in records if r.get("label") in (0, 1)]
    summaries = []
    for name, data in outputs.items():
        fingerprint = data.get("input_manifest_sha256")
        if (
            fingerprint is not None
            and fingerprint != hashlib.sha256(manifest.read_bytes()).hexdigest()
        ):
            raise ValueError(
                f"Results for {name} belong to a different manifest snapshot; rerun inference"
            )
        results = data["results"]
        if len(results) != len(records):
            raise ValueError(f"Stale results for {name}: sample count differs")
        for r, s in zip(records, results):
            if r["path"] != s["path"] or r.get("label") != s.get("label"):
                raise ValueError(f"Stale results for {name}: manifest differs")
        known = [r for r in results if r.get("label") in (0, 1)]
        real = [r for r in known if r["label"] == 0]
        fake = [r for r in known if r["label"] == 1]
        summaries.append(
            {
                "model": name,
                "correct": sum(r["prediction"] == r["label"] for r in known),
                "total": len(known),
                "real_correct": sum(r["prediction"] == 0 for r in real),
                "real_total": len(real),
                "fake_correct": sum(r["prediction"] == 1 for r in fake),
                "fake_total": len(fake),
                "predicted_ai": sum(r["prediction"] == 1 for r in results),
                "images": len(results),
            }
        )
    (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    with (output / "predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "truth", "group"] + models)
        for i, r in enumerate(records):
            writer.writerow(
                [r["path"], r.get("label"), r.get("group", "")]
                + [outputs[n]["results"][i]["score"] for n in models]
            )
    real_count = sum(r.get("label") == 0 for r in records)
    fake_count = sum(r.get("label") == 1 for r in records)
    subtitle = f"{len(records)} images · {real_count} real / {fake_count} AI labeled · threshold 0.5"
    im = Image.new("RGB", (1500, 260 + 68 * len(models)), BG)
    d = ImageDraw.Draw(im)
    text(d, (40, 25), "Multi-model image detection", 34, bold=True)
    text(d, (40, 76), subtitle, 22, fill=MUTED)
    cols = [40, 480, 730, 970, 1230]
    d.rectangle((30, 123, 1470, 177), fill=HEADER)
    headers = [
        "Model",
        "Correct / labeled",
        "Real correct",
        "AI correct",
        "Predicted AI",
    ]
    for x, h in zip(cols, headers):
        text(d, (x, 138), h, 20, fill="white", bold=True)

    def ratio(n, total):
        return f"{n}/{total}" if total else "N/A"

    for i, s in enumerate(summaries):
        y = 180 + 68 * i
        d.rectangle((30, y, 1470, y + 64), fill="white" if i % 2 == 0 else "#e8eef5")
        values = [
            NAMES[s["model"]],
            ratio(s["correct"], s["total"]),
            ratio(s["real_correct"], s["real_total"]),
            ratio(s["fake_correct"], s["fake_total"]),
            f"{s['predicted_ai']}/{s['images']}",
        ]
        for x, v in zip(cols, values):
            text(d, (x, y + 19), v, 21, bold=x == 40)
    text(
        d,
        (40, im.height - 56),
        "Independent model outputs; no ensemble decision. Scores are uncalibrated.",
        18,
        fill=MUTED,
    )
    text(
        d,
        (40, im.height - 29),
        "Subset results are exploratory; cross-model training overlap has not been audited.",
        16,
        fill=MUTED,
    )
    save_png(im, output / "summary.png")
    page_size = 16
    pages = []
    for start in range(0, len(records), page_size):
        subset = records[start : start + page_size]
        width = 1920
        height = 215 + 94 * len(subset)
        im = Image.new("RGB", (width, height), BG)
        d = ImageDraw.Draw(im)
        text(
            d,
            (30, 22),
            f"Individual predictions · images {start+1}–{start+len(subset)} of {len(records)}",
            30,
            bold=True,
        )
        text(
            d,
            (30, 67),
            "Cell = prediction + AI score. Green: correct · red: incorrect · gray: ground truth unavailable.",
            19,
            fill=MUTED,
        )
        d.rectangle((20, 105, width - 20, 164), fill=HEADER)
        text(d, (30, 123), "Image / source", 20, fill="white", bold=True)
        text(d, (363, 123), "Truth", 20, fill="white", bold=True)
        for j, name in enumerate(models):
            label = (
                NAMES[name]
                .replace("Community Forensics", "Community\nForensics")
                .replace("UniversalFakeDetect", "Universal\nFakeDetect")
                .replace("AIDE ", "AIDE\n")
            )
            d.multiline_text(
                (478 + 202 * j, 114),
                label,
                font=font(18, True),
                fill="white",
                spacing=2,
            )
        for i, r in enumerate(subset):
            y = 169 + i * 94
            idx = start + i
            d.rectangle((20, y, width - 20, y + 89), fill="white")
            with Image.open(ROOT / r["path"]) as original:
                preview = ImageOps.exif_transpose(original).convert("RGB")
                preview.thumbnail((92, 78))
                im.paste(preview, (30, y + 5))
            text(
                d,
                (132, y + 9),
                f"{idx+1:04d}  " + fit(d, Path(r["path"]).name, 150),
                17,
                bold=True,
            )
            text(
                d,
                (132, y + 38),
                fit(d, r.get("group", "User image"), 215),
                16,
                fill=MUTED,
            )
            if "width" in r:
                text(d, (132, y + 62), f"{r['width']} × {r['height']}", 14, fill=MUTED)
            truth = r.get("label")
            text(d, (363, y + 31), {0: "Real", 1: "AI"}.get(truth, "—"), 21, bold=True)
            for j, name in enumerate(models):
                s = outputs[name]["results"][idx]
                x = 469 + 202 * j
                color = (
                    "#e8edf3"
                    if truth not in (0, 1)
                    else ("#e0f0e7" if s["prediction"] == truth else "#fae3e1")
                )
                d.rectangle((x, y + 3, x + 192, y + 85), fill=color)
                text(
                    d,
                    (x + 16, y + 14),
                    "AI" if s["prediction"] else "Real",
                    23,
                    bold=True,
                )
                text(d, (x + 16, y + 49), f"{s['score']:.4f}", 21, fill=MUTED)
        text(
            d,
            (30, height - 31),
            "Full filenames and numeric outputs: predictions.csv. AI is predicted when score > 0.5.",
            17,
            fill=MUTED,
        )
        path = output / f"predictions_{start//page_size+1:03d}.png"
        save_png(im, path)
        pages.append(path.name)
    notes = [
        f"# Results: {len(records)} images",
        "",
        subtitle,
        "",
        "- Summary: `summary.png`",
        "- Per-image tables: " + ", ".join(f"`{p}`" for p in pages),
        "- Scores: `predictions.csv`; per-model JSON includes preprocessing, timing, and device.",
        "",
        "Scores are uncalibrated. No voting or trained meta-classifier is applied.",
        "For CompEval subsets, sampling is storage-limited and unavailable viewer rows are skipped. Training overlap has not been audited.",
    ]
    (output / "summary.md").write_text("\n".join(notes) + "\n")
    print(
        f"Exported summary.png and {len(pages)} prediction table PNGs for {len(records)} images.",
        flush=True,
    )
