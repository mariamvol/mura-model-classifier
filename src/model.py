from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models


@dataclass
class ModelConfig:
    arch: str = "densenet121"
    img_size: int = 224
    num_classes: int = 1  # binary -> 1 logit


def build_model(cfg: ModelConfig, device: torch.device) -> nn.Module:
    if cfg.arch != "densenet121":
        raise ValueError(f"Unsupported arch: {cfg.arch}")

    m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    in_feats = m.classifier.in_features
    m.classifier = nn.Linear(in_feats, cfg.num_classes)
    return m.to(device)


def load_checkpoint(path: str | Path, device: torch.device) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)

    if not isinstance(ckpt, dict):
        ckpt = {"model_state": ckpt}

    return ckpt


def load_model_from_checkpoint(
    ckpt_path: str | Path,
    device: torch.device,
    cfg: Optional[ModelConfig] = None,
    strict: bool = True,
) -> Tuple[nn.Module, dict]:
    ckpt = load_checkpoint(ckpt_path, device=device)

    img_size = int(ckpt.get("img_size", cfg.img_size if cfg else 224))
    cfg = cfg or ModelConfig(img_size=img_size)

    model = build_model(cfg, device=device)

    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=strict)
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_proba(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    x: (B,3,H,W)
    returns: (B,) probabilities in [0,1]
    """
    logits = model(x).squeeze(1)
    return torch.sigmoid(logits)
