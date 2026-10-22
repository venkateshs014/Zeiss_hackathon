"""Frozen-encoder linear evaluation for RFMiD and IDRiD."""

from __future__ import annotations

import csv
import io
import json
import random
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.models.encoders import build_encoder


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def _key(value: str) -> str:
    return Path(value).stem.lower().replace("_", "").replace("-", "")


def _read_csv(path: str | Path, member: str) -> tuple[list[str], list[dict[str, str]]]:
    with zipfile.ZipFile(path) as archive:
        text = archive.read(member).decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    return reader.fieldnames or [], list(reader)


def labels_for_manifest(rows: list[dict[str, Any]], csv_spec: dict[str, Any], task: str) -> tuple[np.ndarray, list[str], list[str]]:
    """Attach official CSV labels to one already-separated manifest split."""
    columns, csv_rows = _read_csv(csv_spec["archive"], csv_spec["member"])
    key_column = csv_spec["key_column"]
    by_key = {_key(str(row[key_column])): row for row in csv_rows}
    if task == "multilabel":
        label_columns = [column for column in columns[1:] if column and column.strip()]
        values: list[list[int]] = []
        for manifest_row in rows:
            source_key = _key(manifest_row["relative_path"])
            if source_key not in by_key:
                raise KeyError(f"No CSV label row for manifest image: {manifest_row['relative_path']}")
            values.append([int(float(by_key[source_key][column])) for column in label_columns])
        return np.asarray(values, dtype=np.int64), label_columns, [row["stable_id"] for row in rows]
    if task == "ordinal":
        column = csv_spec.get("target_column", "Retinopathy grade")
        values = []
        for manifest_row in rows:
            source_key = _key(manifest_row["relative_path"])
            if source_key not in by_key:
                raise KeyError(f"No CSV label row for manifest image: {manifest_row['relative_path']}")
            values.append(int(float(by_key[source_key][column])))
        return np.asarray(values, dtype=np.int64), ["0", "1", "2", "3", "4"], [row["stable_id"] for row in rows]
    raise ValueError(f"Unsupported evaluation task: {task}")


class ManifestImages(Dataset[tuple[torch.Tensor, str]]):
    def __init__(self, rows: list[dict[str, Any]], image_size: int) -> None:
        self.rows = rows
        self.transform = transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        row = self.rows[index]
        archive_path = Path(row["source_archive"])
        if not archive_path.is_file():
            raise FileNotFoundError(f"Manifest archive not found: {archive_path}")
        with zipfile.ZipFile(archive_path) as archive:
            raw = archive.read(row["relative_path"])
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        return self.transform(image), row["stable_id"]


@torch.no_grad()
def extract_embeddings(encoder: torch.nn.Module, rows: list[dict[str, Any]], image_size: int, batch_size: int, device: torch.device) -> tuple[np.ndarray, list[str]]:
    """Extract embeddings with the encoder in eval/no-grad mode; no updates occur."""
    encoder.eval()
    loader = DataLoader(ManifestImages(rows, image_size), batch_size=batch_size, shuffle=False, num_workers=0)
    features, identifiers = [], []
    for images, ids in loader:
        features.append(encoder(images.to(device)).detach().cpu().numpy())
        identifiers.extend(ids)
    return np.concatenate(features, axis=0), identifiers


def _fit_binary_models(train_x: np.ndarray, train_y: np.ndarray, seed: int, c_value: float) -> tuple[list[Any], np.ndarray, list[str]]:
    models, probabilities, statuses = [], [], []
    for column in range(train_y.shape[1]):
        labels = train_y[:, column]
        unique = np.unique(labels)
        if len(unique) < 2:
            model = DummyClassifier(strategy="constant", constant=int(unique[0]) if len(unique) else 0)
            status = f"constant_class_{int(unique[0]) if len(unique) else 0}"
        else:
            model = LogisticRegression(C=c_value, max_iter=1000, random_state=seed, solver="liblinear")
            status = "trained"
        model.fit(train_x, labels)
        models.append(model)
        statuses.append(status)
    return models, np.empty((train_x.shape[0], 0)), statuses


def evaluate_multilabel(train_x: np.ndarray, train_y: np.ndarray, eval_x: np.ndarray, eval_y: np.ndarray, seed: int, c_value: float) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    models, _, statuses = _fit_binary_models(train_x, train_y, seed, c_value)
    predictions = np.column_stack([model.predict(eval_x) for model in models]).astype(int)
    probabilities = np.column_stack([model.predict_proba(eval_x)[:, list(model.classes_).index(1)] if 1 in model.classes_ else np.full(len(eval_x), float(model.classes_[0])) for model in models])
    aurocs = [roc_auc_score(eval_y[:, j], probabilities[:, j]) for j in range(eval_y.shape[1]) if len(np.unique(eval_y[:, j])) == 2]
    aps = [average_precision_score(eval_y[:, j], probabilities[:, j]) for j in range(eval_y.shape[1]) if np.sum(eval_y[:, j]) > 0]
    balanced = [balanced_accuracy_score(eval_y[:, j], predictions[:, j]) for j in range(eval_y.shape[1])]
    metrics = {"macro_f1": float(f1_score(eval_y, predictions, average="macro", zero_division=0)), "balanced_accuracy": float(np.mean(balanced)), "macro_auroc": float(np.mean(aurocs)) if aurocs else None, "mean_average_precision": float(np.mean(aps)) if aps else None, "valid_auroc_classes": len(aurocs), "valid_average_precision_classes": len(aps), "class_status": statuses}
    return metrics, predictions, probabilities


def evaluate_ordinal(train_x: np.ndarray, train_y: np.ndarray, eval_x: np.ndarray, eval_y: np.ndarray, seed: int, c_value: float) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model = LogisticRegression(C=c_value, max_iter=1000, random_state=seed, solver="lbfgs", multi_class="auto")
    model.fit(train_x, train_y)
    predictions = model.predict(eval_x).astype(int)
    probabilities = np.zeros((len(eval_x), 5), dtype=np.float32)
    for index, label in enumerate(model.classes_):
        probabilities[:, int(label)] = model.predict_proba(eval_x)[:, index]
    auroc = None
    if len(np.unique(eval_y)) > 1 and len(model.classes_) > 1:
        try:
            auroc = float(roc_auc_score(eval_y, probabilities, multi_class="ovr", labels=[0, 1, 2, 3, 4]))
        except ValueError:
            auroc = None
    metrics = {"accuracy": float(accuracy_score(eval_y, predictions)), "macro_f1": float(f1_score(eval_y, predictions, labels=[0, 1, 2, 3, 4], average="macro", zero_division=0)), "balanced_accuracy": float(balanced_accuracy_score(eval_y, predictions)), "quadratic_weighted_kappa": float(cohen_kappa_score(eval_y, predictions, labels=[0, 1, 2, 3, 4], weights="quadratic")), "macro_auroc": auroc, "absent_train_classes": [label for label in range(5) if label not in set(train_y.tolist())], "official_labels": [0, 1, 2, 3, 4]}
    return metrics, predictions, probabilities


def make_budget_indices(n_samples: int, budgets: list[float], seed: int) -> dict[str, np.ndarray]:
    if n_samples == 0:
        raise ValueError("Cannot create label budgets from an empty training split")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_samples)
    result = {}
    for budget in budgets:
        if not 0 < budget <= 1:
            raise ValueError("label budgets must be fractions in (0, 1]")
        count = min(n_samples, max(1, int(np.ceil(n_samples * budget))))
        result[f"{budget:g}"] = np.sort(order[:count])
    return result


def load_encoder(kind: str, checkpoint: str | None, device: torch.device, imagenet_pretrained: bool = False) -> torch.nn.Module:
    encoder, _ = build_encoder("resnet50", pretrained=imagenet_pretrained)
    if kind in {"simclr", "finetuned_imagenet"}:
        if not checkpoint:
            raise ValueError(f"{kind} requires a local encoder checkpoint")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state_dict = payload.get("state_dict", payload)
        encoder.load_state_dict(state_dict)
    return encoder.to(device)
