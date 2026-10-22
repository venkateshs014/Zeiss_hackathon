"""Trustworthiness metrics, retrieval, and seeded UMAP utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import average_precision_score, roc_auc_score


def corruption_image(image: Image.Image, kind: str, severity: str) -> Image.Image:
    """Apply deterministic, bounded image shifts; no anatomical geometry is changed."""
    factors = {"mild": 0.15, "moderate": 0.30}
    if severity not in factors:
        raise ValueError("severity must be mild or moderate")
    amount = factors[severity]
    image = image.convert("RGB")
    if kind == "brightness":
        return ImageEnhance.Brightness(image).enhance(1.0 + amount)
    if kind == "contrast":
        return ImageEnhance.Contrast(image).enhance(1.0 + amount)
    if kind == "saturation":
        return ImageEnhance.Color(image).enhance(1.0 + amount)
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=1.0 if severity == "mild" else 2.0))
    if kind == "resolution":
        scale = 0.5 if severity == "mild" else 0.25
        small = image.resize((max(8, int(image.width * scale)), max(8, int(image.height * scale))), Image.Resampling.BILINEAR)
        return small.resize(image.size, Image.Resampling.BILINEAR)
    raise ValueError(f"Unsupported corruption kind: {kind}")


def embedding_consistency(clean: np.ndarray, shifted: np.ndarray) -> dict[str, float]:
    """Measure per-image cosine stability under a synthetic shift."""
    if clean.shape != shifted.shape or clean.ndim != 2:
        raise ValueError("clean and shifted embeddings must have identical rank-2 shapes")
    clean_norm = clean / np.maximum(np.linalg.norm(clean, axis=1, keepdims=True), 1e-12)
    shifted_norm = shifted / np.maximum(np.linalg.norm(shifted, axis=1, keepdims=True), 1e-12)
    scores = np.sum(clean_norm * shifted_norm, axis=1)
    return {"mean_cosine_similarity": float(scores.mean()), "std_cosine_similarity": float(scores.std()), "min_cosine_similarity": float(scores.min())}


def ood_scores(reference: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Return distance-to-reference scores; larger values indicate less similarity."""
    if reference.ndim != 2 or queries.ndim != 2 or reference.shape[1] != queries.shape[1]:
        raise ValueError("reference and queries must be rank-2 with equal feature dimensions")
    ref = reference / np.maximum(np.linalg.norm(reference, axis=1, keepdims=True), 1e-12)
    qry = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
    return 1.0 - np.max(qry @ ref.T, axis=1)


def detection_metrics(in_domain_scores: np.ndarray, ood_scores_array: np.ndarray) -> dict[str, Any]:
    """Report AUROC/AUPR only when both groups are non-empty and finite."""
    scores = np.concatenate([in_domain_scores, ood_scores_array]).astype(float)
    labels = np.concatenate([np.zeros(len(in_domain_scores), dtype=int), np.ones(len(ood_scores_array), dtype=int)])
    valid = len(in_domain_scores) > 0 and len(ood_scores_array) > 0 and np.isfinite(scores).all()
    if not valid:
        return {"auroc": None, "aupr": None, "valid": False, "reason": "both non-empty finite in-domain and OOD groups are required"}
    return {"auroc": float(roc_auc_score(labels, scores)), "aupr": float(average_precision_score(labels, scores)), "valid": True, "positive_count": int(labels.sum()), "negative_count": int((labels == 0).sum())}


def cosine_retrieval(query_embeddings: np.ndarray, gallery_embeddings: np.ndarray, query_rows: list[dict[str, Any]], gallery_rows: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    """Deterministic cosine nearest-neighbor retrieval with stable-ID tie breaks."""
    if len(query_embeddings) != len(query_rows) or len(gallery_embeddings) != len(gallery_rows):
        raise ValueError("embedding and manifest row counts must match")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    query = query_embeddings / np.maximum(np.linalg.norm(query_embeddings, axis=1, keepdims=True), 1e-12)
    gallery = gallery_embeddings / np.maximum(np.linalg.norm(gallery_embeddings, axis=1, keepdims=True), 1e-12)
    similarities = query @ gallery.T
    output = []
    for index, scores in enumerate(similarities):
        order = sorted(range(len(scores)), key=lambda j: (-float(scores[j]), str(gallery_rows[j]["stable_id"])))[:top_k]
        output.append({"query_id": query_rows[index]["stable_id"], "query_path": query_rows[index]["relative_path"], "retrieved": [{"id": gallery_rows[j]["stable_id"], "relative_path": gallery_rows[j]["relative_path"], "similarity": float(scores[j])} for j in order]})
    return output


def save_umap_plot(embeddings: np.ndarray, labels: list[str], output_path: str | Path, seed: int, title: str) -> None:
    """Create a seeded UMAP plot; visualization is not evidence of biological meaning."""
    try:
        import matplotlib.pyplot as plt
        import umap
    except ImportError as exc:
        raise RuntimeError("UMAP plotting requires matplotlib and umap-learn") from exc
    if len(embeddings) != len(labels) or len(embeddings) < 3:
        raise ValueError("UMAP requires at least three embeddings and matching labels")
    reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=min(15, len(embeddings) - 1))
    coordinates = reducer.fit_transform(embeddings)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    for label in sorted(set(labels)):
        mask = np.asarray(labels) == label
        plt.scatter(coordinates[mask, 0], coordinates[mask, 1], s=12, alpha=0.7, label=label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
