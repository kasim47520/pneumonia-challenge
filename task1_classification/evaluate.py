"""
Task 1: Comprehensive Evaluation Script
Generates all metrics, visualizations, and failure case analysis.

Usage:
    python evaluate.py --model efficientnet_b0 --weights outputs/task1/best_model.pth
"""

import argparse
import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.dataset import get_dataloaders, get_raw_dataset
from models.model import build_model

CLASS_NAMES = ["Normal", "Pneumonia"]


def get_predictions(model, loader, device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.squeeze().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_training_curves(history_path, output_dir):
    with open(history_path) as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#2196F3")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="#F44336")
    axes[0].axvline(history["best_epoch"], linestyle="--", color="gray", alpha=0.7, label="Best Epoch")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Acc", color="#2196F3")
    axes[1].plot(epochs, [a * 100 for a in history["val_acc"]], label="Val Acc", color="#F44336")
    axes[1].axvline(history["best_epoch"], linestyle="--", color="gray", alpha=0.7, label="Best Epoch")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training & Validation Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_confusion_matrix(labels, preds, output_dir):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return cm


def plot_roc_curve(labels, probs, auc, output_dir):
    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"ROC Curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2196F3")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Test Set"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_failure_cases(model, device, output_dir, n=16):
    """Visualize and save failure cases with predicted vs true labels."""
    dataset = get_raw_dataset(split="test", image_size=224)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    images_list, labels_list, probs_list, preds_list = [], [], [], []
    model.eval()
    with torch.no_grad():
        for imgs, lbls in loader:
            out = model(imgs.to(device))
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()
            images_list.append(imgs)
            labels_list.extend(lbls.squeeze().numpy())
            probs_list.extend(probs)
            preds_list.extend(preds)

    images_all = torch.cat(images_list, dim=0)
    labels_all = np.array(labels_list)
    preds_all  = np.array(preds_list)
    probs_all  = np.array(probs_list)

    fail_idx = np.where(labels_all != preds_all)[0]
    np.random.shuffle(fail_idx)
    selected = fail_idx[:n]

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle("Failure Cases — Misclassified Test Images", fontsize=14, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        if i >= len(selected):
            ax.axis("off"); continue
        idx = selected[i]
        img = images_all[idx].permute(1, 2, 0).numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)
        true_label = CLASS_NAMES[labels_all[idx]]
        pred_label = CLASS_NAMES[preds_all[idx]]
        conf = probs_all[idx] if preds_all[idx] == 1 else 1 - probs_all[idx]
        ax.imshow(img[:, :, 0], cmap="gray")
        ax.set_title(f"True: {true_label}\nPred: {pred_label} ({conf:.2f})",
                     fontsize=8, color="red")
        ax.axis("off")

    plt.tight_layout()
    path = os.path.join(output_dir, "failure_cases.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path} ({len(fail_idx)} total failures)")
    return len(fail_idx)


def plot_sample_predictions(model, device, output_dir, n=16):
    """Visualize random correct predictions."""
    dataset = get_raw_dataset(split="test", image_size=224)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    images_list, labels_list, probs_list, preds_list = [], [], [], []
    model.eval()
    with torch.no_grad():
        for imgs, lbls in loader:
            out = model(imgs.to(device))
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()
            images_list.append(imgs)
            labels_list.extend(lbls.squeeze().numpy())
            probs_list.extend(probs)
            preds_list.extend(preds)

    images_all = torch.cat(images_list, dim=0)
    labels_all = np.array(labels_list)
    preds_all  = np.array(preds_list)
    probs_all  = np.array(probs_list)

    correct_idx = np.where(labels_all == preds_all)[0]
    np.random.seed(42)
    np.random.shuffle(correct_idx)
    selected = correct_idx[:n]

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle("Sample Correct Predictions", fontsize=14, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        if i >= len(selected):
            ax.axis("off"); continue
        idx = selected[i]
        img = images_all[idx].permute(1, 2, 0).numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)
        true_label = CLASS_NAMES[labels_all[idx]]
        conf = probs_all[idx] if preds_all[idx] == 1 else 1 - probs_all[idx]
        ax.imshow(img[:, :, 0], cmap="gray")
        ax.set_title(f"{true_label} ({conf:.2f})", fontsize=8, color="green")
        ax.axis("off")

    plt.tight_layout()
    path = os.path.join(output_dir, "sample_predictions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model = build_model(args.model, num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model = model.to(device)
    print(f"Loaded weights from {args.weights}")

    # Get test predictions
    _, _, test_loader = get_dataloaders(
        image_size=args.image_size, batch_size=args.batch_size, num_workers=args.num_workers
    )
    labels, preds, probs = get_predictions(model, test_loader, device)

    # Metrics
    acc   = accuracy_score(labels, preds)
    prec  = precision_score(labels, preds, average="binary")
    rec   = recall_score(labels, preds, average="binary")
    f1    = f1_score(labels, preds, average="binary")
    auc   = roc_auc_score(labels, probs)

    print("\n" + "="*50)
    print("TEST SET METRICS")
    print("="*50)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"AUC-ROC  : {auc:.4f}")
    print()
    print(classification_report(labels, preds, target_names=CLASS_NAMES))

    # Save metrics JSON
    metrics = {"accuracy": acc, "precision": prec, "recall": rec,
               "f1": f1, "auc": auc}
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Plots
    if args.history_path and os.path.exists(args.history_path):
        plot_training_curves(args.history_path, args.output_dir)

    plot_confusion_matrix(labels, preds, args.output_dir)
    plot_roc_curve(labels, probs, auc, args.output_dir)
    num_failures = plot_failure_cases(model, device, args.output_dir)
    plot_sample_predictions(model, device, args.output_dir)

    print(f"\nTotal misclassified: {num_failures} / {len(labels)} ({100*num_failures/len(labels):.1f}%)")
    print("All outputs saved to:", args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="efficientnet_b0")
    parser.add_argument("--weights", default="outputs/task1/best_model.pth")
    parser.add_argument("--history_path", default="outputs/task1/history.json")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_dir", default="outputs/task1")
    args = parser.parse_args()
    main(args)
