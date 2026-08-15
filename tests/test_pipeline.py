from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from courier_action.evaluation import (
    deterministic_group_subsample,
    evaluate_dataframe,
    make_group_splits,
)
from courier_action.metrics import aggregate_metrics, compute_metrics
from courier_action.schema import SchemaError, target_array, validate_dataframe
from courier_action.synthetic import generate_synthetic_dataframe


def test_synthetic_frame_satisfies_schema() -> None:
    frame = generate_synthetic_dataframe(seed=7, n_groups=10, rows_per_group=16)
    validate_dataframe(frame)
    assert set(frame["action_type"]) == {"PICKUP", "DELIVERY"}


def test_schema_rejects_missing_column() -> None:
    frame = generate_synthetic_dataframe(seed=7, n_groups=10, rows_per_group=16)
    with pytest.raises(SchemaError, match="missing required columns"):
        validate_dataframe(frame.drop(columns=["action_type"]))


def test_generator_is_deterministic() -> None:
    first = generate_synthetic_dataframe(seed=42, n_groups=8, rows_per_group=16)
    second = generate_synthetic_dataframe(seed=42, n_groups=8, rows_per_group=16)
    pd.testing.assert_frame_equal(first, second)


def test_group_folds_are_disjoint_and_cover_rows() -> None:
    frame = generate_synthetic_dataframe(seed=11, n_groups=15, rows_per_group=16)
    y = target_array(frame)
    splits = make_group_splits(y, frame["courier_id"], n_splits=5, seed=42)
    validation_rows: list[int] = []
    for train_index, validation_index in splits:
        train_groups = set(frame.iloc[train_index]["courier_id"])
        validation_groups = set(frame.iloc[validation_index]["courier_id"])
        assert train_groups.isdisjoint(validation_groups)
        validation_rows.extend(validation_index.tolist())
    assert sorted(validation_rows) == list(range(len(frame)))


def test_metrics_use_fixed_threshold_and_aggregate() -> None:
    y = np.array([0, 1, 1, 0])
    probabilities = np.array([0.1, 0.9, 0.8, 0.2])
    metrics = compute_metrics(y, probabilities, threshold=0.5)
    assert metrics == {"f1": 1.0, "auc": 1.0, "accuracy": 1.0}
    aggregate = aggregate_metrics([metrics, metrics])
    assert aggregate["f1"] == {"mean": 1.0, "std": 0.0}


def test_subsample_is_deterministic_and_keeps_whole_groups() -> None:
    frame = generate_synthetic_dataframe(seed=3, n_groups=30, rows_per_group=20)
    first, first_meta = deterministic_group_subsample(frame, max_rows=220, seed=42)
    second, second_meta = deterministic_group_subsample(frame, max_rows=220, seed=42)
    pd.testing.assert_frame_equal(first, second)
    assert first_meta == second_meta
    for group_value, count in first["courier_id"].value_counts().items():
        original_count = int((frame["courier_id"] == group_value).sum())
        assert int(count) == original_count


def test_evaluation_result_is_aggregate_only() -> None:
    frame = generate_synthetic_dataframe(seed=5, n_groups=12, rows_per_group=16)
    protocol = {
        "name": "test",
        "label": "synthetic smoke — not a performance claim",
        "data_kind": "synthetic",
        "seed": 42,
        "n_splits": 3,
        "threshold": 0.5,
        "subsample": {"enabled": False},
        "models": {
            "logistic_regression": {"max_iter": 100},
            "lightgbm": {"n_estimators": 5, "num_leaves": 7},
        },
    }
    result = evaluate_dataframe(frame, protocol)
    assert result["aggregate_only"] is True
    assert result["row_level_predictions_written"] is False
    assert set(result["models"]) == {"logistic_regression", "lightgbm"}
    assert all("metrics" in model for model in result["models"].values())
    assert "predictions" not in result
