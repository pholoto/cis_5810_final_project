"""Prepare WildFC, compute detector scores, and evaluate the requested method."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ai_detector.config import MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["all", "model", "voting", "ensemble"], default="all"
    )
    parser.add_argument("--model", choices=MODELS, help="Required with --mode model")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--device", choices=["cuda:0", "cpu"], default="cuda:0")
    args = parser.parse_args()
    if args.mode == "model" and not args.model:
        parser.error("--mode model requires --model NAME")
    if args.model and args.mode != "model":
        parser.error("--model is only used with --mode model")
    if args.gpu_index < 0:
        parser.error("--gpu-index must be nonnegative")

    def execute(*command):
        subprocess.run([sys.executable, *map(str, command)], cwd=ROOT, check=True)

    execute("scripts/prepare_wildfc.py")
    manifest = ROOT / "data/wildfc/ensemble_split/all.jsonl"
    predictions = ROOT / "reports/wildfc_ensemble/base_predictions"
    command = [
        "-m",
        "ai_detector",
        "--manifest",
        manifest,
        "--output",
        predictions,
        "--device",
        args.device,
        "--gpu-index",
        args.gpu_index,
        "--scores-only",
    ]
    if args.mode == "model":
        command.extend(["--model", args.model])
    execute(*command)
    folder = {
        "all": "wildfc_ensemble",
        "voting": "wildfc_voting",
        "ensemble": "wildfc_meta",
        "model": f"wildfc_{args.model}",
    }[args.mode]
    command = [
        "-m",
        "ai_detector.ensemble",
        "--manifest",
        manifest,
        "--predictions",
        predictions,
        "--output",
        ROOT / "reports" / folder,
        "--mode",
        args.mode,
    ]
    if args.model:
        command.extend(["--model", args.model])
    execute(*command)
    print(f'Results: {ROOT / "reports" / folder / "summary.png"}')


if __name__ == "__main__":
    main()
