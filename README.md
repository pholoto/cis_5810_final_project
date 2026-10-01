# AI image detector

Test seven pretrained image detectors on WildFC, compare voting methods, or train a
small neural ensemble from their output scores. Results are saved as PNG tables
and CSV/JSON files.

Use a Linux server with an NVIDIA GPU and **Python 3.12 or 3.13**. Run all commands
from the `AI_image_detector` folder. GPU **0** is the default; append
`--gpu-index N` to run commands to choose another GPU (if you have multiple GPUs).

## 1. Install requirements

Create and activate a Python environment:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip --cache-dir .cache/pip
```

Install GPU-enabled PyTorch, then the remaining requirements:

```sh
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126 --cache-dir .cache/pip
python -m pip install -r requirements.txt --cache-dir .cache/pip
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

The PyTorch command above uses CUDA 12.6. Also use miniconda if you prefer that over venv.

## 2. Download model code and checkpoints

```sh
python scripts/fetch_detector_sources.py
python scripts/download_detector_weights.py
python -m ai_detector.check_setup
```

These commands download the pinned model implementations and all checkpoints into
`.cache/` inside this project (approximately 13.1 GB of weights). Interrupted
checkpoint downloads can be resumed by rerunning the command.

The seven models are:

- `community_forensics`
- `safe`
- `universalfakedetect`
- `dda`
- `aide_GenImage`
- `aide_progan`
- `aide_sd14`

## 3. Download WildFC

Request access using your own account on the
[WildFC Hugging Face page](https://huggingface.co/datasets/pthan12/WildFC).
After approval, create a read-access token in your
[Hugging Face settings](https://huggingface.co/settings/tokens), then run:

```sh
python scripts/login_huggingface.py
python scripts/download_wildfc.py
python scripts/extract_wildfc.py
```

Paste the token at the login prompt. It is saved locally in `.cache/huggingface`;
do not share it. The dataset is downloaded and extracted under `data/wildfc`.
You can use another dataset if you want.

## 4. Run on WildFC

**Run everything: individual detectors, voting, and the neural ensemble.**

```sh
python scripts/run_wildfc.py
```

**Run just one detector**, for example DDA:

```sh
python scripts/run_wildfc.py --mode model --model dda
```

**Run voting**: majority vote, model-family vote, and mean-score averaging:

```sh
python scripts/run_wildfc.py --mode voting
```

**Train and evaluate the neural ensemble**:

```sh
python scripts/run_wildfc.py --mode ensemble
```

All commands prepare the same 80-20 train-test split and reuse saved detector predictions.
Voting and neural ensembles require scores from all seven configurations, so any
missing detector predictions are computed automatically. Detectors run sequentially
on the selected GPU. The small meta-network trains on CPU.

To select a different GPU, for example GPU 7:

```sh
python scripts/run_wildfc.py --gpu-index 7
```

The neural ensemble uses seven detector scores as inputs to a `7 → 16 → 8 → 1` MLP.
Only the meta-network is trained, the detectors weights remain frozen.
WildFC has no supplied per-image train/test split, so I use an approximately 80:20
split, keeping articles and exact image duplicates from crossing splits. No separate
validation set is used to tune the hyperparameters yet, but you can try.

### Open the outputs

| Command | Output folder |
| --- | --- |
| Everything | `reports/wildfc_ensemble/` |
| One model, e.g. DDA | `reports/wildfc_dda/` |
| Voting | `reports/wildfc_voting/` |
| Neural ensemble | `reports/wildfc_meta/` |

Open `summary.png` for the comparison table, `metrics.csv` for metrics, and
`heldout_predictions.csv` for test-image scores. Neural runs also save
`meta_classifier.pt`. Shared detector predictions are cached in
`reports/wildfc_ensemble/base_predictions/`.

Rerun the same command to resume interrupted detector inference. For advanced
manifest options, see [USAGE.md](USAGE.md).
