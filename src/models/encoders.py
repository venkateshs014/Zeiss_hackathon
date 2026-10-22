"""Torchvision encoder factory for representation learning."""

import torch.nn as nn
from torchvision import models


def build_encoder(name: str = "resnet50", pretrained: bool = False) -> tuple[nn.Module, int]:
    """Build a torchvision image encoder with its classification head removed."""
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    elif name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
    else:
        raise ValueError("Unsupported encoder. Choose 'resnet18' or 'resnet50'.")
    embedding_dim = model.fc.in_features
    model.fc = nn.Identity()
    return model, embedding_dim
