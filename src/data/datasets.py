from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


def infer_label_from_path(path: str) -> int:
    p = path.replace("\\", "/").lower()
    if "positive" in p:
        return 1
    if "negative" in p:
        return 0
    raise ValueError(f"Cannot infer label from path: {path}")


def study_id_from_path(p: str) -> str:
    """
    .../XR_HAND/patientXXXX/studyY_positive/image.png -> patientXXXX/studyY_positive
    """
    parts = Path(p).as_posix().split("/")
    idx = None
    for i, s in enumerate(parts):
        if re.fullmatch(r"XR_[A-Z]+", s or ""):
            idx = i
            break
    if idx is None or idx + 2 >= len(parts):
        return "UNK/UNK"
    patient = parts[idx + 1]
    study = parts[idx + 2]
    return f"{patient}/{study}"


class ImageDataset(Dataset):
    """
    Generic dataset by list of image paths.
    If labels=None -> used for inference
    """
    def __init__(self, paths: List[str], labels: Optional[List[int]] = None, transform=None, return_path: bool = False):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.return_path = return_path

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        with Image.open(p) as img:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)

        if self.labels is None:
            return (img, p) if self.return_path else img

        y = int(self.labels[idx])
        return (img, y, p) if self.return_path else (img, y)
