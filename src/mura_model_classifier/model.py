from __future__ import annotations
import torch
import torch.nn as nn
from torchvision import models

def build_densenet121(device: torch.device | str = "cpu") -> nn.Module:
    # torchvision >=0.13 uses weights=
    m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    in_feats = m.classifier.in_features
    m.classifier = nn.Linear(in_feats, 1)  # binary logit
    return m.to(device)

def load_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device | str = "cpu") -> dict:
    ckpt = torch.load(ckpt_path, map_location=device)


    state = ckpt.get("model_state", ckpt)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:

        print(f"[WARN] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")

    model.eval()
    return ckpt
