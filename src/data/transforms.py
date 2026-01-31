from __future__ import annotations

from torchvision import transforms


def build_transforms(img_size: int = 224, train: bool = False):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.9, 1.1)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
