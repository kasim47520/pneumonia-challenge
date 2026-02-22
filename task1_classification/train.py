"""
Task 1: CNN/ViT Classification Training Script
Trains a model on PneumoniaMNIST and saves weights + metrics.

Usage:
    python train.py --model efficientnet_b0 --epochs 20 --lr 1e-4
"""

import argparse
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.dataset import get_dataloaders
from models.model import build_model, FocalLoss, get_optimizer, get_scheduler


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.squeeze().long().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.squeeze().long().to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Data
    train_loader, val_loader, _ = get_dataloaders(
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Model
    model = build_model(args.model, num_classes=2, pretrained=True).to(device)
    print(f"Model: {args.model} | Params: {sum(p.numel() for p in model.parameters()):,}")

    # Loss — Focal loss handles class imbalance in pneumonia data
    criterion = FocalLoss(gamma=2.0)
    optimizer = get_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer, args.epochs, warmup_epochs=3)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, "best_model.pth"))
            print(f"  ✓ Saved best model (val_acc={best_val_acc:.4f})")

    # Save training history
    history["best_val_acc"] = best_val_acc
    history["best_epoch"] = best_epoch
    history["model"] = args.model
    history["hyperparameters"] = vars(args)

    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PneumoniaMNIST Classifier")
    parser.add_argument("--model", default="efficientnet_b0",
                        choices=["efficientnet_b0", "efficientnet_b2", "vit_small_patch16_224",
                                 "resnet50", "densenet121"],
                        help="Model architecture")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_dir", default="outputs/task1")
    args = parser.parse_args()
    main(args)
