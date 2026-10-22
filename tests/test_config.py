from pathlib import Path

from src.utils.config import load_yaml


def test_initial_configs_load() -> None:
    for path in (Path("configs/pretrain.yaml"), Path("configs/evaluation.yaml")):
        config = load_yaml(path)
        assert isinstance(config, dict)
        assert "seed" in config

