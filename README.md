# AI image detector

Compare AI-image detectors on WildFC using pretrained voting, pretrained
stacking, or fine-tuning followed by stacking. The original seven detectors are
the default. NPR and SPAI are opt-in. Outputs are PNG tables and CSV/JSON
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

To add NPR, run `python scripts/fetch_detector_sources.py --only npr`. This
fetches both source and the released checkpoint. Check it with
`python -m ai_detector.check_setup --include-npr`.

To add SPAI, use the same project environment and run:

```sh
python -m pip install -r requirements-spai.txt --cache-dir .cache/pip
python scripts/fetch_detector_sources.py --only spai
python -m ai_detector.check_setup --include-spai
```

Before the setup check, place the [official SPAI checkpoint](https://github.com/mever-team/spai#weights-checkpoint)
at `.cache/detectors/spai/spai.pth`. The adapter pins
[SPAI source and config](https://github.com/mever-team/spai/tree/8ff7b3b6779b4fcb43cf313471d9cb1c62d129a4),
strictly loads the tensor state, and records the checkpoint SHA-256 in run
provenance. It uses the released MFM model with this project's existing
PyTorch, torchvision, and timm. Inference keeps the original image dimensions
except for reflection padding below 224 pixels on either side, and uses the
upstream patch aggregation and sigmoid score.

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

Use `--include-npr` to append NPR to the seven defaults. Use `--models`
to supply an explicit ordered list for any of the three workflows:

```sh
python scripts/run_wildfc.py --mode pretrained --gpu-index 0 --models community_forensics safe universalfakedetect dda aide_GenImage aide_progan aide_sd14 npr spai
```

The same selection works with `python -m ai_detector`,
`python -m ai_detector.ensemble`, `python -m ai_detector.classical_ensemble`,
and `python -m ai_detector.finetune`. For a single SPAI inference worker, use
`python -m ai_detector --model spai --manifest MANIFEST --output NEW_DIR --device cpu`
on a supported Linux environment. Family voting treats the selected AIDE
variants as one family and gives NPR and SPAI one family each.

Both stacking workflows compare **L2 logistic regression, linear SVM, RBF SVM,
Random Forest, XGBoost, and an MLP (number of selected detectors → 16 → 8 → 1)**. Their tables also include
individual detector and voting baselines. The input features are the seven final
detector scores by default. The stackers convert scores to clipped log-odds. Scaling
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

An explicit `--models` roster gets its own cache and report folder. Pretrained
reuse now verifies weights, device, adapter code, and pinned source/config
identity as well as manifest alignment. The first run against an older cache
will regenerate scores and archive the previous completed files. Use
`--output NEW_DIR` for a fresh workflow report
when an existing report or training protocol differs.

### Fine-tuning details

| Detector | Updated parameters | Epochs |
| --- | --- | ---: |
| Community Forensics | ViT and classifier | 20 |
| SAFE | Detector parameters | 20 |
| UniversalFakeDetect | Linear head; CLIP frozen | 20 |
| DDA | LoRA and classifier | 1 |
| AIDE (each variant) | ResNet branches and classifier; ConvNeXt/HPF frozen | 20 |
| NPR (opt-in) | Detector parameters | 1 |
| SPAI (opt-in) | Classifier head, with spectral backbone and patch aggregation frozen | 1 |

SPAI fine-tuning is a project adaptation of the released checkpoint. It uses
variable-size image batches, horizontal flips, BCE, and float32 training.
For a bounded server check, first prepare the detector split with
`python -m ai_detector.finetune --phase prepare --output NEW_DIR`, then run
`python -m ai_detector.finetune --phase train --model spai --manifest NEW_DIR/splits/detector_train.jsonl --output NEW_DIR --smoke-steps 1`.
Use a separate output directory for the subsequent full run.

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
