from __future__ import annotations
import os
import sys
import urllib.request
from pathlib import Path
from typing import Dict

import torch
from PIL import Image
from torchvision import transforms

AREAS = ["XR_ELBOW","XR_FINGER","XR_FOREARM","XR_HAND","XR_HUMERUS","XR_SHOULDER","XR_WRIST"]

DEFAULT_CKPT_NAME = {a: f"{a}_FINAL_best.pt" for a in AREAS}

WEIGHTS_URL_BASE = "https://github.com/mariamvol/mura-model-classifier/releases/download/v0.1.0"

def get_default_cache_dir() -> Path:
    # ~/.cache/mura-model-classifier
    base = Path(os.path.expanduser("~")) / ".cache" / "mura-model-classifier"
    base.mkdir(parents=True, exist_ok=True)
    return base

def build_preprocess(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img

def download_url(url: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _progress(blocknum, blocksize, totalsize):
        read = blocknum * blocksize
        if totalsize > 0:
            pct = read / totalsize * 100
            sys.stdout.write(f"\rDownloading... {pct:5.1f}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, out_path, reporthook=_progress)
    print("\nDone.")
    return out_path

