#!/usr/bin/env python3
"""Verify an aggregate result against the checked-in authorized artifact."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

MODEL_NAMES = ("logistic_regression", "lightgbm")
METRIC_NAMES = ("f1", "auc", "accuracy")


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def verify(expected_path: Path, actual_path: Path) -> None:
    expected = _load(expected_path)
    actual = _load(actual_path)
    for artifact in (expected, actual):
        if artifact.get("aggregate_only") is not True:
            raise ValueError("result is not marked aggregate_only")
        if artifact.get("row_level_predictions_written") is not False:
            raise ValueError("result claims to contain row-level predictions")
        if artifact.get("positive_label") != "PICKUP":
            raise ValueError("result positive_label must be PICKUP")

    if expected.get("positive_label") != actual.get("positive_label"):
        raise ValueError("result positive_label metadata differs")

    for key in ("label", "evaluation"):
        if expected.get(key) != actual.get(key):
            raise ValueError(f"result metadata differs for {key}")

    expected_data = expected.get("data", {})
    actual_data = actual.get("data", {})
    for key in ("kind", "rows_read", "rows_evaluated", "groups_evaluated", "class_counts", "subsampling"):
        if expected_data.get(key) != actual_data.get(key):
            raise ValueError(f"result data metadata differs for {key}")

    for model_name in MODEL_NAMES:
        expected_model = expected.get("models", {}).get(model_name, {})
        actual_model = actual.get("models", {}).get(model_name, {})
        if expected_model.get("folds") != actual_model.get("folds"):
            raise ValueError(f"fold count differs for {model_name}")
        for metric_name in METRIC_NAMES:
            for statistic in ("mean", "std"):
                expected_value = expected_model.get("metrics", {}).get(metric_name, {}).get(statistic)
                actual_value = actual_model.get("metrics", {}).get(metric_name, {}).get(statistic)
                if not isinstance(expected_value, (int, float)) or not isinstance(actual_value, (int, float)):
                    raise ValueError(f"missing {model_name}/{metric_name}/{statistic}")
                if not math.isclose(float(expected_value), float(actual_value), rel_tol=0.0, abs_tol=1e-7):
                    raise ValueError(
                        f"metric differs for {model_name}/{metric_name}/{statistic}: "
                        f"{expected_value} != {actual_value}"
                    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", required=True, type=Path)
    parser.add_argument("--actual", required=True, type=Path)
    args = parser.parse_args()
    verify(args.expected, args.actual)
    print("verified aggregate metrics for logistic_regression and lightgbm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
