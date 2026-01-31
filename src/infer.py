from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.model import load_model_from_checkpoint, ModelConfig, predict_proba
from src.data.datasets import ImageDataset
from src.data.transforms import build_transforms


def list_images(path: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if path.is_file():
        return [str(path)]
    imgs = []
    for p in path.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts and not p.name.startswith("._"):
            imgs.append(str(p))
    return sorted(imgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="path to .pt checkpoint")
    ap.add_argument("--input", type=str, required=True, help="image file OR folder")
    ap.add_argument("--out_csv", type=str, default="preds.csv")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    cfg = ModelConfig(img_size=args.img_size)
    model, ckpt = load_model_from_checkpoint(args.ckpt, device=device, cfg=cfg, strict=False)

    inp = Path(args.input)
    paths = list_images(inp)
    if not paths:
        raise SystemExit(f"No images found in: {inp}")

    tfm = build_transforms(img_size=args.img_size, train=False)
    ds = ImageDataset(paths, labels=None, transform=tfm, return_path=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    rows = []
    model.eval()
    with torch.no_grad():
        for xb, batch_paths in dl:
            xb = xb.to(device)
            probs = predict_proba(model, xb).detach().cpu().numpy().tolist()
            for p, pr in zip(batch_paths, probs):
                rows.append({"path": p, "prob_positive": float(pr)})

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(df.head(10).to_string(index=False))
    print(f"\nSaved: {args.out_csv} (n={len(df)})")


if __name__ == "__main__":
    main()
