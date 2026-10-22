import numpy as np
from PIL import Image

from src.evaluation.trustworthiness import corruption_image, cosine_retrieval, detection_metrics, ood_scores


def test_retrieval_is_deterministic_and_tie_breaks_by_id() -> None:
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    rows = [{"stable_id": "b", "relative_path": "b.png"}, {"stable_id": "a", "relative_path": "a.png"}, {"stable_id": "c", "relative_path": "c.png"}]
    first = cosine_retrieval(embeddings[:1], embeddings, rows[:1], rows, 3)
    second = cosine_retrieval(embeddings[:1], embeddings, rows[:1], rows, 3)
    assert first == second
    assert [item["id"] for item in first[0]["retrieved"]] == ["b", "c", "a"]


def test_ood_scores_and_validity_handling() -> None:
    scores = ood_scores(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]))
    assert np.isclose(scores[0], 1.0)
    metrics = detection_metrics(np.array([0.0, 0.1]), np.array([0.8, 0.9]))
    assert metrics["valid"] and metrics["auroc"] == 1.0
    invalid = detection_metrics(np.array([]), np.array([0.1]))
    assert invalid["auroc"] is None and not invalid["valid"]


def test_corruptions_are_bounded_and_non_geometric() -> None:
    image = Image.new("RGB", (32, 24), (100, 120, 140))
    for kind in ("brightness", "contrast", "saturation", "blur", "resolution"):
        shifted = corruption_image(image, kind, "moderate")
        assert shifted.size == image.size
