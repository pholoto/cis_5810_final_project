from pathlib import Path
from .config import COMMUNITY_REVISION


class CommunityForensics:
    def __init__(self, device="cpu", size=384, cache_dir=".cache/huggingface"):
        self.device, self.size, self.cache_dir = device, size, cache_dir
        self.model = None

    def __call__(self, path):
        import torch
        import timm
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download
        from torchvision import transforms as T
        from PIL import Image, ImageOps

        repo = f"OwensLab/commfor-model-{self.size}"
        if self.model is None:
            checkpoint = hf_hub_download(
                repo,
                "model.safetensors",
                revision=COMMUNITY_REVISION,
                cache_dir=self.cache_dir,
                local_files_only=True,
            )
            self.revision = Path(checkpoint).parent.name
            model = timm.create_model(
                f"vit_small_patch16_{self.size}.augreg_in21k_ft_in1k",
                pretrained=False,
                num_classes=1,
            )
            weights = load_file(checkpoint)
            model.load_state_dict(
                {k.removeprefix("vit."): v for k, v in weights.items()}, strict=True
            )
            self.model = model.eval().to(self.device)
        transform = T.Compose(
            [
                T.Resize(440 if self.size == 384 else 256),
                T.CenterCrop(self.size),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        with Image.open(path) as im:
            tensor = (
                transform(ImageOps.exif_transpose(im).convert("RGB"))
                .unsqueeze(0)
                .to(self.device)
            )
        with torch.inference_mode():
            score = float(self.model(tensor).sigmoid().item())
        return score
