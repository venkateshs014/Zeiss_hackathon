import pytest
import torch

from src.models.checkpoints import load_encoder_checkpoint
from src.models.encoders import build_encoder


def test_encoder_checkpoint_loader_accepts_export_and_rejects_projection_head(tmp_path) -> None:
    encoder, _ = build_encoder("resnet18")
    path = tmp_path / "encoder.pt"
    torch.save({"state_dict": encoder.state_dict(), "embedding_dim": 512}, path)
    loaded, _ = build_encoder("resnet18")
    load_encoder_checkpoint(loaded, path, torch.device("cpu"))
    bad = tmp_path / "full.pt"
    torch.save({"state_dict": {"encoder.conv1.weight": encoder.conv1.weight}}, bad)
    with pytest.raises(ValueError, match="encoder-only"):
        load_encoder_checkpoint(loaded, bad, torch.device("cpu"))
