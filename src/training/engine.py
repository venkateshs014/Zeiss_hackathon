"""Configurable SimCLR pretraining with resumable checkpoints."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.contrastive_dataset import UnlabeledFundusDataset
from src.data.transforms import build_contrastive_transform
from src.models.contrastive import SimCLR, nt_xent_loss
from src.utils.device import resolve_device
from src.utils.reproducibility import seed_everything


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: str | Path, model: SimCLR, optimizer: torch.optim.Optimizer, scheduler: Any, scaler: Any, epoch: int, step: int, config: dict[str, Any]) -> None:
    """Save complete resumable state and a separate encoder-only weight file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"epoch": epoch, "step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "rng": _rng_state(), "config": config}
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)
    torch.save({"state_dict": model.encoder.state_dict(), "embedding_dim": model.embedding_dim, "epoch": epoch}, target.with_name(target.stem + ".encoder.pt"))


def load_checkpoint(path: str | Path, model: SimCLR, optimizer: torch.optim.Optimizer, scheduler: Any, scaler: Any, device: torch.device) -> tuple[int, int]:
    """Restore checkpoint state and return the next epoch and global step."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    if "rng" in checkpoint:
        _restore_rng(checkpoint["rng"])
    return int(checkpoint["epoch"]) + 1, int(checkpoint.get("step", 0))


def run_pretraining(config: dict[str, Any]) -> None:
    """Run SimCLR; loss is logged but is not a representation-quality claim."""
    seed_everything(int(config["seed"]))
    device = resolve_device(str(config.get("device", "auto")))
    data_cfg, model_cfg, train_cfg = config["data"], config["model"], config["training"]
    dataset = UnlabeledFundusDataset(data_cfg["manifest"], build_contrastive_transform(int(data_cfg.get("image_size", 224))))
    loader = DataLoader(dataset, batch_size=int(train_cfg["batch_size"]), shuffle=True, num_workers=int(data_cfg.get("num_workers", 0)), pin_memory=bool(data_cfg.get("pin_memory", False)), drop_last=bool(data_cfg.get("drop_last", True)))
    model = SimCLR(model_cfg.get("name", "resnet50"), int(model_cfg.get("projection_dim", 128)), bool(model_cfg.get("pretrained", False))).to(device)
    configured_dim = model_cfg.get("embedding_dim")
    if configured_dim is not None and int(configured_dim) != model.embedding_dim:
        raise ValueError(f"Configured embedding_dim={configured_dim} does not match encoder output {model.embedding_dim}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg.get("weight_decay", 0.0)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(train_cfg["epochs"]))
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    start_epoch, global_step = 0, 0
    if train_cfg.get("resume"):
        start_epoch, global_step = load_checkpoint(train_cfg["resume"], model, optimizer, scheduler, scaler, device)
    update_probe_before = model.encoder.conv1.weight.detach().clone()
    log_path = output_dir / "training.jsonl"
    for epoch in range(start_epoch, int(train_cfg["epochs"])):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        total_loss, sample_count, batches = 0.0, 0, 0
        gradients_finite = True
        for view_one, view_two in loader:
            view_one, view_two = view_one.to(device, non_blocking=True), view_two.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = nt_xent_loss(model(view_one), model(view_two), float(train_cfg["temperature"]))
            scaler.scale(loss).backward()
            gradients_finite = gradients_finite and all(parameter.grad is None or torch.isfinite(parameter.grad).all().item() for parameter in model.parameters())
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu())
            sample_count += view_one.shape[0]
            batches += 1
            global_step += 1
        scheduler.step()
        elapsed = time.perf_counter() - started
        update_probe = float((model.encoder.conv1.weight.detach() - update_probe_before).abs().sum().cpu())
        record = {"epoch": epoch, "step": global_step, "loss": total_loss / max(1, batches), "seconds": elapsed, "samples": sample_count, "throughput_images_per_second": sample_count / max(elapsed, 1e-9), "learning_rate": optimizer.param_groups[0]["lr"], "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0, "gradients_finite": gradients_finite, "parameter_update_probe_l1": update_probe}
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        save_checkpoint(output_dir / "latest.pt", model, optimizer, scheduler, scaler, epoch, global_step, config)
