from __future__ import annotations
import argparse
from pathlib import Path
import torch

from .model import build_densenet121, load_checkpoint
from .utils import (
    AREAS, DEFAULT_CKPT_NAME,
    get_default_cache_dir, build_preprocess, load_image, download_url
)

def parse_args():
    p = argparse.ArgumentParser(description="MURA fracture inference (DenseNet121).")
    p.add_argument("--image", required=True, help="Path to an image (png/jpg).")
    p.add_argument("--area", required=True, choices=AREAS, help="Anatomical area.")
    p.add_argument("--ckpt", default=None, help="Path to checkpoint .pt (optional).")

    # опционально: автоматом скачать веса
    p.add_argument("--weights-url", default=None, help="Base URL to download weights from (optional).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()

@torch.no_grad()
def predict_one(model, img_path: str, device: str = "cpu", img_size: int = 224) -> float:
    tfm = build_preprocess(img_size)
    img = load_image(img_path)
    x = tfm(img).unsqueeze(0).to(device)
    logit = model(x).squeeze(0).squeeze(0)
    prob = torch.sigmoid(logit).item()
    return float(prob)

def resolve_checkpoint(area: str, ckpt_arg: str | None, weights_url: str | None) -> Path:
    if ckpt_arg:
        return Path(ckpt_arg)

    cache = get_default_cache_dir()
    fname = DEFAULT_CKPT_NAME[area]
    local = cache / fname
    if local.exists():
        return local

    if not weights_url:
        raise FileNotFoundError(
            f"Checkpoint not found: {local}\n"
            f"Provide --ckpt PATH or --weights-url BASE_URL"
        )

    # weights_url = например: https://github.com/<you>/<repo>/releases/download/v0.1.0/
    url = weights_url.rstrip("/") + "/" + fname
    download_url(url, local)
    return local

def main():
    args = parse_args()
    device = args.device

    ckpt_path = resolve_checkpoint(args.area, args.ckpt, args.weights_url)

    model = build_densenet121(device=device)
    _ = load_checkpoint(model, str(ckpt_path), device=device)

    prob = predict_one(model, args.image, device=device)
    pred = int(prob >= args.threshold)

    print(f"area={args.area}")
    print(f"image={args.image}")
    print(f"ckpt={ckpt_path}")
    print(f"prob_fracture={prob:.4f}")
    print(f"pred={pred} (threshold={args.threshold})")

if __name__ == "__main__":
    main()
