"""
Model architectures for PneumoniaMNIST classification.
Uses timm library for pretrained ViT and EfficientNet models.
"""

import torch
import torch.nn as nn
import timm


def build_model(model_name="efficientnet_b0", num_classes=2, pretrained=True):
    """
    Build a classification model using timm.

    Args:
        model_name: timm model identifier. Recommended options:
            - 'efficientnet_b0'   : lightweight, fast
            - 'vit_small_patch16_224' : Vision Transformer
            - 'resnet50'          : classic baseline
        num_classes: number of output classes (2 for binary)
        pretrained: use ImageNet pretrained weights

    Returns:
        model (nn.Module)
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=3,
    )
    return model


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in medical datasets.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        return focal_loss.sum()


def get_optimizer(model, lr=1e-4, weight_decay=1e-4):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def get_scheduler(optimizer, num_epochs, warmup_epochs=5):
    """Cosine annealing with linear warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(num_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
