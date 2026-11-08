"""Run pretrained voting, pretrained stacking, or fine-tuning plus stacking."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ai_detector.config import add_model_selection, parse_model_selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['voting', 'pretrained', 'finetuned'], default='pretrained')
    parser.add_argument('--gpu-index', type=int, default=0)
    parser.add_argument('--output', type=Path, help='Override the workflow report directory')
    add_model_selection(parser)
    args = parser.parse_args()
    if args.gpu_index < 0:
        parser.error('--gpu-index must be nonnegative')
    selection = parse_model_selection(parser, args, training=args.mode == 'finetuned')
    flags = []
    suffix = ''
    if args.models:
        flags = ['--models', *selection]
        digest = hashlib.sha256(','.join(selection).encode()).hexdigest()[:12]
        suffix = f'_models_{digest}'
    elif args.include_npr:
        flags = ['--include-npr']
        suffix = '_with_npr'

    def execute(*command):
        subprocess.run([sys.executable, *map(str, command)], cwd=ROOT, check=True)

    execute('scripts/prepare_wildfc.py')
    folder = {'voting': 'pretrained_voting', 'pretrained': 'pretrained_meta',
              'finetuned': 'finetuned_meta'}[args.mode]
    output = args.output.resolve() if args.output else ROOT / 'reports' / (folder + suffix)
    if args.mode == 'finetuned':
        command = ['-m', 'ai_detector.finetune', '--gpu-index', args.gpu_index, '--output', output]
        execute(*command, *flags)
    else:
        manifest = ROOT / 'data/wildfc/ensemble_split/all.jsonl'
        predictions = ROOT / '.cache/predictions' / ('pretrained' + suffix)
        inference_command = [
            '-m', 'ai_detector',
            '--manifest', manifest,
            '--output', predictions,
            '--device', 'cuda:0',
            '--gpu-index', args.gpu_index,
            '--scores-only',
        ]
        execute(*inference_command, *flags)
        module = 'ai_detector.ensemble' if args.mode == 'voting' else 'ai_detector.classical_ensemble'
        command = ['-m', module, '--manifest', manifest, '--predictions', predictions, '--output', output]
        if args.mode == 'voting':
            command += ['--mode', 'voting']
        execute(*command, *flags)
    print(f'Results: {output / "summary.png"}')


if __name__ == '__main__':
    main()
