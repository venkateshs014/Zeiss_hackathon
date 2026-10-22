import pytest

from src.utils.device import resolve_device


def test_auto_device_is_valid() -> None:
    assert resolve_device("auto").type in {"cpu", "cuda"}


def test_invalid_device_has_actionable_error() -> None:
    with pytest.raises(ValueError, match="Unsupported device"):
        resolve_device("tpu")

