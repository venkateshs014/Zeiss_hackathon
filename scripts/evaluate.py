"""Evaluate frozen ImageNet and SimCLR encoders with identical linear protocols."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.frozen import (
    evaluate_multilabel,
    evaluate_ordinal,
    extract_embeddings,
    labels_for_manifest,
    load_encoder,
    load_jsonl,
    make_budget_indices,
)
from src.utils.config import load_yaml
from src.utils.reproducibility import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen-encoder retinal representation evaluation")
    parser.add_argument("--config", default="configs/evaluation.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    seed_everything(int(config["seed"]))
    device = torch.device(config.get("device", "cpu") if config.get("device", "auto") != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    output = Path(config["evaluation"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    all_metrics, all_predictions = [], []
    for dataset_name, spec in config["datasets"].items():
        train_rows = load_jsonl(spec["train_manifest"])
        train_y, label_names, train_ids = labels_for_manifest(train_rows, spec["train_labels"], spec["task"])
        eval_specs = []
        if spec.get("validation_manifest"):
            eval_specs.append(("validation", spec["validation_manifest"], spec["validation_labels"]))
        if spec.get("test_manifest"):
            eval_specs.append(("test", spec["test_manifest"], spec["test_labels"]))
        if not eval_specs:
            raise ValueError(f"Dataset {dataset_name} has no validation_manifest or test_manifest")
        budget_indices_by_seed = {seed: make_budget_indices(len(train_rows), config["evaluation"]["label_budgets"], seed) for seed in config["evaluation"]["seeds"]}
        for model_name, model_spec in config["models"].items():
            if not model_spec.get("enabled", True):
                continue
            encoder = load_encoder(model_spec["type"], model_spec.get("checkpoint"), device, bool(model_spec.get("imagenet_pretrained", False)))
            train_x, _ = extract_embeddings(encoder, train_rows, int(config["data"]["image_size"]), int(config["data"]["batch_size"]), device)
            for split_name, eval_manifest, eval_label_spec in eval_specs:
                eval_rows = load_jsonl(eval_manifest)
                eval_y, _, eval_ids = labels_for_manifest(eval_rows, eval_label_spec, spec["task"])
                eval_x, _ = extract_embeddings(encoder, eval_rows, int(config["data"]["image_size"]), int(config["data"]["batch_size"]), device)
                for seed, budgets in budget_indices_by_seed.items():
                    for budget_name, indices in budgets.items():
                        metric_fn = evaluate_multilabel if spec["task"] == "multilabel" else evaluate_ordinal
                        metrics, predictions, probabilities = metric_fn(train_x[indices], train_y[indices], eval_x, eval_y, seed, float(config["evaluation"]["classifier_c"]))
                        record = {"dataset": dataset_name, "split": split_name, "model": model_name, "seed": seed, "label_budget": budget_name, "n_train": int(len(indices)), "n_eval": int(len(eval_y)), "metrics": metrics, "label_names": label_names}
                        all_metrics.append(record)
                        for row_id, true, predicted, scores in zip(eval_ids, eval_y, predictions, probabilities):
                            all_predictions.append({"dataset": dataset_name, "split": split_name, "model": model_name, "seed": seed, "label_budget": budget_name, "stable_id": row_id, "true": true.tolist() if hasattr(true, "tolist") else int(true), "predicted": predicted.tolist() if hasattr(predicted, "tolist") else int(predicted), "scores": scores.tolist()})
    (output / "metrics.json").write_text(json.dumps(all_metrics, indent=2, sort_keys=True), encoding="utf-8")
    grouped: dict[tuple[str, str, str, str], list[dict[str, float]]] = {}
    for record in all_metrics:
        key = (record["dataset"], record["split"], record["model"], record["label_budget"])
        grouped.setdefault(key, []).append(record["metrics"])
    summary = []
    for (dataset, split, model, budget), records in sorted(grouped.items()):
        metric_names = sorted({name for record in records for name, value in record.items() if isinstance(value, (int, float)) and value is not None})
        metrics = {}
        for name in metric_names:
            values = np.asarray([float(record[name]) for record in records if isinstance(record.get(name), (int, float))], dtype=float)
            if len(values):
                metrics[name] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0, "n": int(len(values))}
        summary.append({"dataset": dataset, "split": split, "model": model, "label_budget": budget, "metrics": metrics})
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_predictions:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
