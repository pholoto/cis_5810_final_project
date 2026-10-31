"""Run pretrained voting, pretrained stacking, or fine-tuning plus stacking."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['voting', 'pretrained', 'finetuned'], default='pretrained')
    parser.add_argument('--gpu-index', type=int, default=0)
    args = parser.parse_args()
    if args.gpu_index < 0:
        parser.error('--gpu-index must be nonnegative')

    def execute(*command):
        subprocess.run([sys.executable, *map(str, command)], cwd=ROOT, check=True)

    execute('scripts/prepare_wildfc.py')
    folder = {'voting': 'pretrained_voting', 'pretrained': 'pretrained_meta',
              'finetuned': 'finetuned_meta'}[args.mode]
    output = ROOT / 'reports' / folder
    if args.mode == 'finetuned':
        execute('-m', 'ai_detector.finetune', '--gpu-index', args.gpu_index, '--output', output)
    else:
        manifest = ROOT / 'data/wildfc/ensemble_split/all.jsonl'
        predictions = ROOT / '.cache/predictions/pretrained'
        execute('-m', 'ai_detector', '--manifest', manifest, '--output', predictions,
                '--device', 'cuda:0', '--gpu-index', args.gpu_index, '--scores-only')
        module = 'ai_detector.ensemble' if args.mode == 'voting' else 'ai_detector.classical_ensemble'
        command = ['-m', module, '--manifest', manifest, '--predictions', predictions, '--output', output]
        if args.mode == 'voting':
            command += ['--mode', 'voting']
        execute(*command)
    print(f'Results: {output / "summary.png"}')


if __name__ == '__main__':
    main()
