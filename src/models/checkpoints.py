"""Strict loading for exported encoder-only checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_encoder_checkpoint(encoder: torch.nn.Module, path: str | Path, device: torch.device) -> dict[str, Any]:
    """Load an encoder export and reject full SimCLR/projection-head checkpoints."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state_dict, dict):
        raise ValueError("Encoder checkpoint must contain a state_dict mapping")
    keys = list(state_dict)
    if any(key.startswith("encoder.") or key.startswith("projector.") for key in keys):
        raise ValueError("This is not an encoder-only export; it contains encoder/projector-prefixed full-model keys")
    encoder.load_state_dict(state_dict, strict=True)
    encoder.eval()
    return checkpoint
