import json
from pathlib import Path

import torch
from PIL import Image

from src.data.contrastive_dataset import UnlabeledFundusDataset
from src.models.contrastive import SimCLR, nt_xent_loss
from src.training.engine import load_checkpoint, save_checkpoint


def test_nt_xent_is_finite_and_has_gradients() -> None:
    z1 = torch.randn(4, 16, requires_grad=True)
    z2 = torch.randn(4, 16, requires_grad=True)
    loss = nt_xent_loss(z1, z2, temperature=0.2)
    assert torch.isfinite(loss)
    loss.backward()
    assert z1.grad is not None and torch.isfinite(z1.grad).all()
    assert z2.grad is not None and torch.isfinite(z2.grad).all()


def test_simclr_resnet50_feature_and_projection_shapes() -> None:
    model = SimCLR("resnet50", projection_dim=32)
    model.eval()
    with torch.no_grad():
        features = model.encode(torch.randn(2, 3, 32, 32))
        projected = model(torch.randn(2, 3, 32, 32))
    assert features.shape == (2, 2048)
    assert projected.shape == (2, 32)


def test_manifest_dataset_returns_two_views_and_rejects_labels(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    image = Image.new("RGB", (48, 48), (120, 80, 40))
    path = image_root / "one.png"
    image.save(path)
    manifest = tmp_path / "pretraining.jsonl"
    manifest.write_text(json.dumps({"stable_id": "abc", "dataset": "RFMiD", "split": "training", "relative_path": "images/one.png", "source_archive": None}) + "\n", encoding="utf-8")
    transform = lambda value: torch.from_numpy(__import__("numpy").asarray(value).copy()).permute(2, 0, 1).float()
    dataset = UnlabeledFundusDataset(manifest, transform)
    view_one, view_two = dataset[0]
    assert view_one.shape == view_two.shape == (3, 48, 48)

    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"stable_id": "abc", "dataset": "RFMiD", "split": "training", "relative_path": "images/one.png", "source_archive": None, "disease": 0}) + "\n", encoding="utf-8")
    try:
        UnlabeledFundusDataset(bad, transform)
    except ValueError as exc:
        assert "forbidden" in str(exc)
    else:
        raise AssertionError("label-bearing manifest was accepted")


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    model = SimCLR("resnet18", projection_dim=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    checkpoint = tmp_path / "latest.pt"
    save_checkpoint(checkpoint, model, optimizer, scheduler, scaler, 3, 7, {"seed": 1})
    restored = SimCLR("resnet18", projection_dim=8)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(restored_optimizer, T_max=2)
    next_epoch, step = load_checkpoint(checkpoint, restored, restored_optimizer, restored_scheduler, scaler, torch.device("cpu"))
    assert next_epoch == 4 and step == 7
    assert torch.equal(model.encoder.conv1.weight, restored.encoder.conv1.weight)
    assert checkpoint.with_name("latest.encoder.pt").is_file()
