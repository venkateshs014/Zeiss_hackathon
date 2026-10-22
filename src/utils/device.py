"""Device selection utilities."""

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve ``auto``, ``cpu``, or ``cuda`` with a useful CUDA error."""
    normalized = requested.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this environment")
        return torch.device("cuda")
    raise ValueError(f"Unsupported device '{requested}'. Expected auto, cpu, or cuda.")

