"""SimCLR model and NT-Xent objective."""

import torch
import torch.nn.functional as F
import torch.nn as nn

from src.models.encoders import build_encoder


class ProjectionHead(nn.Module):
    """Small MLP used to map encoder features into contrastive space."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("projection dimensions must be positive")
        hidden_dim = max(input_dim, output_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class SimCLR(nn.Module):
    """Encoder plus a two-layer projection head used only during pretraining."""

    def __init__(self, encoder_name: str = "resnet50", projection_dim: int = 128, pretrained: bool = False) -> None:
        super().__init__()
        self.encoder, self.embedding_dim = build_encoder(encoder_name, pretrained=pretrained)
        self.projector = ProjectionHead(self.embedding_dim, projection_dim)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encode(images))


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """Compute symmetric NT-Xent with self-similarities excluded from logits."""
    if z1.ndim != 2 or z2.ndim != 2 or z1.shape != z2.shape:
        raise ValueError("z1 and z2 must be rank-2 tensors with identical shapes")
    if z1.shape[0] == 0:
        raise ValueError("NT-Xent requires at least one positive pair")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    batch_size = z1.shape[0]
    representations = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    logits = representations @ representations.T / temperature
    self_mask = torch.eye(2 * batch_size, device=logits.device, dtype=torch.bool)
    logits = logits.masked_fill(self_mask, torch.finfo(logits.dtype).min)
    positives = torch.arange(2 * batch_size, device=logits.device)
    positives = (positives + batch_size) % (2 * batch_size)
    return F.cross_entropy(logits, positives)
