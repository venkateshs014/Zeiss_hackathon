"""Run configurable SimCLR pretraining from an approved unlabeled manifest."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.engine import run_pretraining
from src.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain a retinal image encoder")
    parser.add_argument("--config", default="configs/pretrain.yaml")
    args = parser.parse_args()
    run_pretraining(load_yaml(args.config))


if __name__ == "__main__":
    main()
