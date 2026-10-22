import numpy as np

from src.evaluation.frozen import evaluate_multilabel, evaluate_ordinal, make_budget_indices


def test_budget_indices_are_reproducible_and_shared() -> None:
    first = make_budget_indices(100, [0.01, 0.05, 1.0], seed=7)
    second = make_budget_indices(100, [0.01, 0.05, 1.0], seed=7)
    assert all(np.array_equal(first[key], second[key]) for key in first)
    assert [len(first[key]) for key in first] == [1, 5, 100]


def test_multilabel_metrics_handle_absent_training_class() -> None:
    rng = np.random.default_rng(4)
    train_x = rng.normal(size=(8, 4))
    train_y = np.column_stack([np.array([0, 1] * 4), np.zeros(8, dtype=int)])
    eval_x = rng.normal(size=(4, 4))
    eval_y = np.column_stack([np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1])])
    metrics, predictions, scores = evaluate_multilabel(train_x, train_y, eval_x, eval_y, seed=3, c_value=1.0)
    assert np.isfinite(metrics["macro_f1"])
    assert predictions.shape == eval_y.shape
    assert scores.shape == eval_y.shape
    assert metrics["class_status"][1].startswith("constant_class_")


def test_ordinal_metrics_use_official_zero_to_four_mapping() -> None:
    rng = np.random.default_rng(5)
    train_x = rng.normal(size=(10, 3))
    train_y = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    eval_x = rng.normal(size=(5, 3))
    eval_y = np.array([0, 1, 2, 3, 4])
    metrics, predictions, probabilities = evaluate_ordinal(train_x, train_y, eval_x, eval_y, seed=2, c_value=1.0)
    assert set(metrics["official_labels"]) == {0, 1, 2, 3, 4}
    assert predictions.shape == (5,)
    assert probabilities.shape == (5, 5)
    assert np.isfinite(metrics["quadratic_weighted_kappa"])
