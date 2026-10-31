# AI image detector

Compare seven AI-image detectors on WildFC using pretrained voting, pretrained
stacking, or fine-tuning followed by stacking. Outputs are PNG tables and CSV/JSON
files. No experiment results or model weights are included in this repository.

Use Linux, an NVIDIA GPU, and **Python 3.12 or 3.13**. Run all commands from the
repository root. **GPU 0** is the default. Detectors run sequentially; meta-learners
train on CPU.

## 1. Install requirements

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip --cache-dir .cache/pip
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126 --cache-dir .cache/pip
python -m pip install -r requirements.txt --cache-dir .cache/pip
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

The PyTorch command uses CUDA 12.6. You can also use a Conda environment with the
same Python version.

## 2. Download model code and checkpoints

```sh
python scripts/fetch_detector_sources.py
python scripts/download_detector_weights.py
python -m ai_detector.check_setup
```

Pinned upstream source code and approximately 13.1 GB of pretrained weights are
stored in `.cache/`. Rerun the download command to resume an interrupted download.
The detectors are Community Forensics, SAFE, UniversalFakeDetect, DDA, and AIDE
with its GenImage, ProGAN, and SD14 checkpoints.

## 3. Download WildFC

Request access on the [WildFC dataset page](https://huggingface.co/datasets/pthan12/WildFC).
After approval, create a read-access token in your
[Hugging Face settings](https://huggingface.co/settings/tokens), then run:

```sh
python scripts/login_huggingface.py
python scripts/download_wildfc.py
python scripts/extract_wildfc.py
```

Paste the token at the login prompt. Credentials stay in `.cache/huggingface`;
do not share them. The dataset is stored in `data/wildfc/`.

## 4. Choose a workflow

Each command prepares the dataset split automatically. There are three result folders:

| Workflow | Data split | Output folder |
| --- | --- | --- |
| Pretrained voting | Evaluate on the shared 20% test set; no training | `reports/pretrained_voting/` |
| Pretrained detectors + meta-learners | 80% meta training / 20% test | `reports/pretrained_meta/` |
| Fine-tuned detectors + meta-learners | Approximately 64% detector training / 16% meta training / 20% test | `reports/finetuned_meta/` |

**Pretrained voting** — majority vote, model-family vote, and mean-score averaging:

```sh
python scripts/run_wildfc.py --mode voting --gpu-index 0
```

**Pretrained detectors + all meta-learners** — detector weights stay unchanged:

```sh
python scripts/run_wildfc.py --mode pretrained --gpu-index 0
```

**Fine-tune detectors, then train all meta-learners**:

```sh
python scripts/run_wildfc.py --mode finetuned --gpu-index 0
```

Both stacking workflows compare **L2 logistic regression, linear SVM, RBF SVM,
Random Forest, XGBoost, and an MLP (7 → 16 → 8 → 1)**. Their tables also include
individual detector and voting baselines. The input features are the seven final
detector scores converted to clipped log-odds, not backbone embeddings. Scaling
for logistic regression, SVMs, and the MLP is fitted on meta-training data only.
SVM scores are sigmoid-transformed decision margins, not calibrated probabilities.

WildFC does not supply the per-image split used here. The code creates a seeded,
approximately 80/20 split, keeping articles and exact duplicate images together.
Fine-tuning splits that 80% again into 80% detector fitting and 20% stacking fitting,
giving approximately **64/16/20 overall**. Sources are publishers/platforms, not
known generator identities. Source coverage is retained where independent groups
allow it; singleton sources stay in detector training. Exact counts may vary with
group sizes. No separate validation set or test-based hyperparameter tuning is used.

### Results and resuming

Open `summary.png` and `metrics.csv` in the selected output folder.
`heldout_predictions.csv` contains test-image scores. Stacking runs also save
`meta_classifier.pt`, one `.joblib` file per classical classifier, and a
`protocol.json` recording inputs and settings.

Both pretrained workflows share `.cache/predictions/pretrained/`, so running the
second workflow reuses detector predictions. Fine-tuned scores are stored in
`reports/finetuned_meta/base_predictions/`; its splits, logs, and detector
checkpoints are subfolders of the same result folder.

Rerun the same command after an interruption. Completed detector checkpoints and
matching predictions are reused; interrupted training resumes from its latest
optimizer-boundary save. Meta-learners are refitted when rerun. Keep the original
pretrained weights: fine-tuned checkpoints store parameter updates and model
buffers that are loaded on top of them.

### Fine-tuning details

| Detector | Updated parameters | Epochs |
| --- | --- | ---: |
| Community Forensics | ViT and classifier | 20 |
| SAFE | Detector parameters | 20 |
| UniversalFakeDetect | Linear head; CLIP frozen | 20 |
| DDA | LoRA and classifier | 1 |
| AIDE (each variant) | ResNet branches and classifier; ConvNeXt/HPF frozen | 20 |

Training uses pinned upstream augmentations with model-specific AdamW recipes,
class balancing, mixed precision, and gradient accumulation. The final epoch is
saved without validation-based selection. Detailed settings and upstream revisions
are recorded in `checkpoints/<model>/protocol.json`. Fine-tuning takes substantially
longer than inference, especially for AIDE. The current training worker caps the
PyTorch GPU allocator at 7.5 GB; gradient accumulation does not reproduce the
BatchNorm statistics of larger microbatches.

DDA's original training expects matched real/VAE-reconstructed pairs and JPEG
metadata, which WildFC lacks. This adaptation disables matched JPEG alignment,
pixel mixing, and frequency mixing, and uses unpaired real/fake images with
original/downscaled/upscaled views and BCE plus contrastive loss. It is not a
reproduction of the full original DDA recipe. UniversalFakeDetect uses a fixed
20-epoch schedule instead of validation-driven early stopping.

## Check the installation and tests

```sh
python -m ai_detector.check_setup
python -m unittest discover -s tests -v
```

Tests use synthetic data and do not require downloaded weights or a GPU.
Generated datasets, weights, credentials, caches, and reports are ignored by Git.
