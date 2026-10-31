"""WildFC adapters for pinned upstream training recipes (see README.md)."""

import random
from types import SimpleNamespace
import torch
from torchvision import transforms as T
from .config import UP
from .detectors import module, package

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def recipe(name, epochs=None):
    family = "aide" if name.startswith("aide_") else name
    # Fine-tune from the released weights; never reinitialize a pretrained head.
    values = {
        "community_forensics": dict(
            epochs=20,
            microbatch=4,
            effective_batch=32,
            lr=2e-5,
            weight_decay=0.01,
            smoothing=0.0,
            warmup_fraction=0.2,
        ),
        "safe": dict(
            epochs=20,
            microbatch=8,
            effective_batch=32,
            lr=0.01 * 32 / 256,
            weight_decay=0.01,
            smoothing=0.1,
            warmup_fraction=0.05,
        ),
        "universalfakedetect": dict(
            epochs=20,
            microbatch=8,
            effective_batch=256,
            lr=1e-4,
            weight_decay=0.0,
            smoothing=0.0,
            warmup_fraction=0.0,
        ),
        "dda": dict(
            epochs=1,
            microbatch=1,
            effective_batch=64,
            lr=1e-4,
            weight_decay=0.0,
            smoothing=0.0,
            warmup_fraction=0.0,
        ),
        "aide": dict(
            epochs=20,
            microbatch=1,
            effective_batch=32,
            lr=1e-4 * 32 / 256,
            weight_decay=0.0,
            smoothing=0.1,
            warmup_fraction=0.0,
        ),
    }[family]
    values = dict(values, family=family)
    if epochs is not None:
        values["epochs"] = epochs
    return values


def trainable_model(model, name):
    if name == "universalfakedetect":
        model[0].requires_grad_(False)
        model[1].requires_grad_(True)
    elif name == "dda":
        for key, param in model.named_parameters():
            param.requires_grad_(".lora." in key or key.startswith("base_model.fc."))
        # Checkpoint activations, preserving architecture and all upstream LoRA targets.
        from torch.utils.checkpoint import checkpoint
        import functools

        for block in model.base_model.model.blocks:
            forward = block.forward
            block.forward = functools.partial(checkpoint, forward, use_reentrant=False)
    # AIDE constructor freezes its ConvNeXt and HPF. Its projection is inside
    # upstream's no_grad block and therefore also cannot receive gradients.
    elif name.startswith("aide_"):
        model.convnext_proj.requires_grad_(False)
    return model


def training_mode(model, name):
    model.train()
    if name == "universalfakedetect":
        model[0].eval()
    elif name.startswith("aide_"):
        model.openclip_convnext_xxl.eval()


def train_transform(name):
    if name == "community_forensics":
        ct = module(
            "commfor_training_transforms", UP / "community/custom_transforms.py"
        )
        ops = "JPEGinMemory,RandomResizeWithRandomIntpl,RandomCrop,RandomHorizontalFlip,RandomVerticalFlip,RRCWithRandomIntpl,RandomRotation,RandomTranslate,RandomShear,RandomPadding,RandomCutout"
        return T.Compose(
            [
                T.Resize(440),
                ct.RandomStateAugmentation(
                    resize_size=440,
                    crop_size=384,
                    auglist=ops,
                    min_augs="0",
                    max_augs="2",
                ),
                T.RandomCrop(384),
                ct.ToTensor_range(val_min=0, val_max=1),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    if name == "safe":
        data = module("safe_training_data", UP / "safe/data/datasets.py")
        return data.Get_Transforms(
            SimpleNamespace(
                input_size=256,
                transform_mode="crop",
                jpeg_factor=None,
                blur_sigma=None,
                mask_ratio=None,
                mask_patch_size=None,
            )
        )[0]
    if name == "universalfakedetect":
        data = module("ufd_training_data", UP / "ufd/data/datasets.py")
        options = SimpleNamespace(
            blur_prob=0.5,
            blur_sig=[0.0, 3.0],
            jpg_prob=0.5,
            jpg_method=["cv2", "pil"],
            jpg_qual=list(range(30, 101)),
        )
        return T.Compose(
            [
                T.Resize(256),
                T.Lambda(lambda image: data.data_augment(image, options)),
                T.RandomCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(CLIP_MEAN, CLIP_STD),
            ]
        )
    if name == "dda":
        package("dda_training_data", UP / "dda/Training/data")
        data = module(
            "dda_training_data.datasets", UP / "dda/Training/data/datasets.py"
        )
        return data.create_train_transforms(size=336, mean=CLIP_MEAN, std=CLIP_STD)
    if name.startswith("aide_"):
        package("aide_training_data", UP / "aide/data")
        data = module("aide_training_data.datasets", UP / "aide/data/datasets.py")
        dct = data.DCT_base_Rec_Module()

        def transform(image):
            tensor = data.transform_before(image)
            patches = dct(tensor)
            return torch.stack([data.transform_train(p) for p in (*patches, tensor)])

        return transform
    raise ValueError(name)


def parameter_groups(model, config):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if config["family"] not in ("safe", "aide"):
        return parameters
    # Upstream SAFE/AIDE exclude biases and normalization vectors from decay.
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (
            no_decay
            if p.ndim == 1 or name.endswith((".bias", ".gamma", ".beta"))
            else decay
        ).append(p)
    return [
        {"params": decay, "weight_decay": config["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def loss_for(model, x, y, name, config):
    if name == "dda":
        from pytorch_metric_learning.losses import ContrastiveLoss

        features = model.base_model.model.forward_features(x)["x_norm_clstoken"]
        logits = model.base_model.fc(features).flatten()
        cls = torch.nn.functional.binary_cross_entropy_with_logits(
            logits.float(), y.float()
        )
        contrast = ContrastiveLoss(pos_margin=0.0, neg_margin=1.0)(
            features.float(), y.long()
        )
        return 0.5 * cls + 0.5 * contrast
    logits = model(x)
    if name in ("community_forensics", "universalfakedetect"):
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits.flatten().float(), y.float()
        )
    return torch.nn.functional.cross_entropy(
        logits.float(), y.long(), label_smoothing=config["smoothing"]
    )
