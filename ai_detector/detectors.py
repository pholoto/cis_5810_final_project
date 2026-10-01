"""Load frozen detector architectures and their official evaluation transforms."""

import importlib.util
import sys
import types
from pathlib import Path
from .config import CACHE, UP


def module(name, path, remove_unused_clip=False):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    if remove_unused_clip:
        # Upstream AIDE imports OpenAI clip but never uses it; open_clip is used.
        source = Path(path).read_text().replace("import clip\n", "")
        exec(compile(source, str(path), "exec"), mod.__dict__)
    else:
        spec.loader.exec_module(mod)
    return mod


def package(name, path):
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]
    sys.modules[name] = pkg


def make_detector(name, device="cpu"):
    import torch
    from torchvision import transforms as T
    from PIL import Image

    torch.set_num_threads(2)
    torch.manual_seed(42)
    clip_norm = T.Normalize(
        [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
    )
    weights = CACHE / "detectors"
    if name == "community_forensics":
        from ai_detector.community import CommunityForensics

        model = CommunityForensics(device=device, cache_dir=str(CACHE / "huggingface"))
        return lambda p: model(p), {
            "transform": "resize short edge 440, center crop 384, ImageNet normalization",
            "score": "sigmoid(fake logit)",
            "device": device,
        }
    if name == "safe":
        mod = module("safe_resnet", UP / "safe/models/resnet.py")
        model = mod.resnet50(pretrained=False)
        checkpoint = weights / "safe/checkpoint-best.pth"
        transform = T.Compose([T.CenterCrop(256), T.ToTensor()])
        definition = "center crop 256; tensor [0,1]; official bior1.3 wavelet preprocessing inside model"
    elif name == "universalfakedetect":
        mod = module("ufd_clip_model", UP / "ufd/models/clip/model.py")
        model = mod.VisionTransformer(
            input_resolution=224,
            patch_size=14,
            width=1024,
            layers=24,
            heads=16,
            output_dim=768,
        )
        jit = torch.jit.load(str(CACHE / "clip/ViT-L-14.pt"), map_location="cpu")
        state = {
            k.removeprefix("visual."): v.float()
            for k, v in jit.state_dict().items()
            if k.startswith("visual.")
        }
        model.load_state_dict(state, strict=True)
        del jit, state
        head = torch.nn.Linear(768, 1)
        head.load_state_dict(
            torch.load(
                weights / "universalfakedetect/fc_weights.pth",
                map_location="cpu",
                weights_only=True,
            ),
            strict=True,
        )
        model = torch.nn.Sequential(model, head)
        checkpoint = None
        transform = T.Compose([T.CenterCrop(224), T.ToTensor(), clip_norm])
        definition = "center crop 224, no resize, CLIP normalization; raw CLIP visual features and linear head"
    elif name == "dda":
        sys.path.insert(0, str(UP / "dinov2"))
        from dinov2.hub.backbones import dinov2_vitl14

        lora = module("dda_lora", UP / "dda/Inference/models/lora.py")

        class DDA(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base_model = torch.nn.Module()
                self.base_model.model = dinov2_vitl14(pretrained=False)
                self.base_model.fc = torch.nn.Linear(1024, 1)
                lora.apply_lora_to_linear_layers(
                    self.base_model.model,
                    rank=8,
                    alpha=1.0,
                    target_modules=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
                    trainable_orig=False,
                )

            def forward(self, x):
                return self.base_model.fc(
                    self.base_model.model.forward_features(x)["x_norm_clstoken"]
                )

        model = DDA()
        checkpoint = weights / "dda/DDA_ckpt.pth"
        transform = T.Compose([T.CenterCrop(336), T.ToTensor(), clip_norm])
        definition = "center crop 336, no resize, CLIP normalization; official DINOv2 + LoRA rank 8 alpha 1"
    elif name.startswith("aide_"):
        package("aide_models", UP / "aide/models")
        mod = module(
            "aide_models.AIDE", UP / "aide/models/AIDE.py", remove_unused_clip=True
        )
        model = mod.AIDE(resnet_path=None, convnext_path=None)
        checkpoint = weights / f'aide/{name.removeprefix("aide_")}_train.pth'
        dct = module("aide_dct", UP / "aide/data/dct.py").DCT_base_Rec_Module()
        resize_norm = T.Compose(
            [
                T.Resize([256, 256]),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        def transform(image):
            tensor = T.ToTensor()(image)
            patches = dct(tensor)
            return torch.stack([resize_norm(p) for p in (*patches, tensor)])

        definition = "official DCT selection of 4 patches + full image; tensor resize 256x256 and ImageNet normalization; no test augmentation"
    else:
        raise ValueError(name)
    if checkpoint:
        state = torch.load(
            checkpoint, map_location="cpu", weights_only=True, mmap=True
        )["model"]
        model.load_state_dict(state, strict=True)
        del state
    model.eval().to(device)
    binary = name in ("universalfakedetect", "dda")

    def predict(path):
        with Image.open(path) as image:
            tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = model(tensor)
            score = (
                logits.sigmoid().item() if binary else logits.softmax(-1)[0, 1].item()
            )
        return float(score)

    return predict, {
        "transform": definition,
        "score": "sigmoid(fake logit)" if binary else "softmax(class 1 = fake)",
        "strict_state_dict": True,
        "device": device,
        "precision": "float32",
    }
