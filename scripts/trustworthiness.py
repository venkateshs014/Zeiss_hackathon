"""Run corruption robustness, OOD scoring, UMAP, and retrieval checks."""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.frozen import ManifestImages, extract_embeddings, load_encoder, load_jsonl
from src.evaluation.trustworthiness import corruption_image, cosine_retrieval, detection_metrics, embedding_consistency, ood_scores, save_umap_plot
from src.utils.config import load_yaml
from src.utils.reproducibility import seed_everything


def _shifted_embeddings(encoder: torch.nn.Module, rows: list[dict], kind: str, severity: str, image_size: int, batch_size: int, device: torch.device) -> np.ndarray:
    base = ManifestImages(rows, image_size)
    images = []
    for index, row in enumerate(rows):
        with zipfile.ZipFile(Path(row["source_archive"])) as handle:
            raw = handle.read(row["relative_path"])
        shifted = corruption_image(Image.open(io.BytesIO(raw)), kind, severity)
        images.append(base.transform(shifted))
    tensor = torch.stack(images)
    outputs = []
    encoder.eval()
    with torch.no_grad():
        for start in range(0, len(tensor), batch_size):
            outputs.append(encoder(tensor[start:start + batch_size].to(device)).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trustworthiness evaluation for a frozen retinal encoder")
    parser.add_argument("--config", default="configs/trustworthiness.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    seed_everything(int(config["seed"]))
    device = torch.device("cuda" if config.get("device", "auto") == "auto" and torch.cuda.is_available() else config.get("device", "cpu"))
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    encoder = load_encoder("simclr", config["encoder_checkpoint"], device)
    in_rows = load_jsonl(config["manifests"]["in_domain"])
    ood_rows = load_jsonl(config["manifests"]["ood"])
    image_size, batch_size = int(config["image_size"]), int(config["batch_size"])
    in_embeddings, _ = extract_embeddings(encoder, in_rows, image_size, batch_size, device)
    ood_embeddings, _ = extract_embeddings(encoder, ood_rows, image_size, batch_size, device)
    report = {"warnings": ["UMAP geometry does not prove biological understanding; synthetic stress tests do not establish clinical reliability."], "synthetic_corruptions": {}, "ood": {"group_definition": config.get("ood_definition", "approved unseen-dataset manifest"), "clean_unseen": detection_metrics(ood_scores(in_embeddings, in_embeddings), ood_scores(in_embeddings, ood_embeddings))}}
    for kind in config["corruptions"]:
        for severity in config["severities"]:
            shifted = _shifted_embeddings(encoder, in_rows, kind, severity, image_size, batch_size, device)
            report["synthetic_corruptions"][f"{kind}_{severity}"] = embedding_consistency(in_embeddings, shifted)
    if config.get("umap", {}).get("enabled", True):
        sample_count = min(len(in_embeddings), int(config["umap"].get("max_points", 1000)))
        ood_count = min(len(ood_embeddings), sample_count)
        combined = np.concatenate([in_embeddings[:sample_count], ood_embeddings[:ood_count]])
        labels = ["in_domain"] * sample_count + ["unseen_ood"] * ood_count
        save_umap_plot(combined, labels, output / "umap_in_domain_vs_ood.png", int(config["seed"]), "Frozen retinal embeddings: diagnostic visualization")
    query_count = min(len(in_rows), int(config.get("retrieval", {}).get("query_count", 20)))
    retrieval = cosine_retrieval(in_embeddings[:query_count], in_embeddings, in_rows[:query_count], in_rows, int(config.get("retrieval", {}).get("top_k", 5)))
    with (output / "retrieval.jsonl").open("w", encoding="utf-8") as handle:
        for row in retrieval:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
