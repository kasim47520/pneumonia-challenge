"""
Data loading and preprocessing utilities for PneumoniaMNIST.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import medmnist
from medmnist import PneumoniaMNIST


def get_transforms(split="train", image_size=224):
    """
    Returns appropriate transforms for each split.
    Medical imaging augmentations: horizontal flips and small rotations are
    clinically reasonable for chest X-rays; vertical flips are NOT used as
    they would produce anatomically invalid images.
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])


def get_dataloaders(image_size=224, batch_size=32, num_workers=2, data_flag="pneumoniamnist"):
    """
    Load PneumoniaMNIST and return train/val/test DataLoaders.
    """
    train_dataset = PneumoniaMNIST(
        split="train", transform=get_transforms("train", image_size),
        download=True, as_rgb=True
    )
    val_dataset = PneumoniaMNIST(
        split="val", transform=get_transforms("val", image_size),
        download=True, as_rgb=True
    )
    test_dataset = PneumoniaMNIST(
        split="test", transform=get_transforms("test", image_size),
        download=True, as_rgb=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader


def get_raw_dataset(split="test", image_size=224):
    """Returns raw dataset (no augmentation) for analysis purposes."""
    return PneumoniaMNIST(
        split=split, transform=get_transforms(split="val", image_size=image_size),
        download=True, as_rgb=True
    )
