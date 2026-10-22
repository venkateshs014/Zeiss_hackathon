import torch
from PIL import Image

from src.data.transforms import build_contrastive_transform
from src.models.contrastive import ProjectionHead
from src.models.encoders import build_encoder


def test_transform_and_encoder_cpu_smoke() -> None:
    image = Image.new("RGB", (64, 64), color=(128, 64, 32))
    tensor = build_contrastive_transform(32)(image)
    assert tensor.shape == (3, 32, 32)
    encoder, embedding_dim = build_encoder("resnet18")
    encoder.eval()
    with torch.no_grad():
        features = encoder(tensor.unsqueeze(0))
    assert features.shape == (1, embedding_dim)


def test_projection_head_shape() -> None:
    head = ProjectionHead(8, 4)
    assert head(torch.randn(2, 8)).shape == (2, 4)
