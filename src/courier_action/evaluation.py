"""Fold-local grouped cross-validation for the courier action task."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .metrics import aggregate_metrics, compute_metrics
from .schema import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    GROUP_COLUMN,
    NUMERIC_FEATURES,
    aggregate_schema_metadata,
    load_training_csv,
    target_array,
    validate_dataframe,
)

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - exercised only with older scikit-learn
    StratifiedGroupKFold = None  # type: ignore[assignment,misc]


MODEL_NAMES = ("logistic_regression", "lightgbm")


def _package_versions() -> dict[str, str]:
    """Return the package versions that affect the recorded evaluation."""

    distributions = {
        "numpy": "numpy",
        "pandas": "pandas",
        "sklearn": "scikit-learn",
        "lightgbm": "lightgbm",
    }
    versions: dict[str, str] = {}
    for name, distribution in distributions.items():
        try:
            versions[name] = package_version(distribution)
        except PackageNotFoundError:  # pragma: no cover - defensive metadata path
            versions[name] = "unavailable"
    return versions


def _stable_group_key(value: object, seed: int) -> str:
    payload = f"{seed}:{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def deterministic_group_subsample(
    df: pd.DataFrame, *, max_rows: int, seed: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select whole groups in deterministic target-rate strata.

    The selection never slices a group.  Groups are ordered by a stable hash
    within target-rate strata and visited round-robin until the row budget is
    reached.  No group identifiers are returned in metadata.
    """

    validate_dataframe(df)
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")

    group_series = df[GROUP_COLUMN]
    y = target_array(df)
    if len(df) <= max_rows:
        return df.reset_index(drop=True).copy(), {
            "enabled": True,
            "method": "deterministic_stratified_by_group",
            "whole_groups": True,
            "requested_max_rows": int(max_rows),
            "selected_rows": int(len(df)),
            "selected_groups": int(group_series.nunique(dropna=False)),
            "selection_seed": int(seed),
            "truncated": False,
        }

    stats = pd.DataFrame(
        {GROUP_COLUMN: group_series.to_numpy(), "_positive": y}
    ).groupby(GROUP_COLUMN, sort=False, dropna=False).agg(
        rows=("_positive", "size"), positive=("_positive", "sum")
    )
    stats["positive_rate"] = stats["positive"] / stats["rows"]
    stats = stats.reset_index()
    stats["_group_text"] = stats[GROUP_COLUMN].astype("string")
    stats = stats.sort_values(["positive_rate", "_group_text"], kind="mergesort")
    n_strata = min(5, len(stats))
    positions = np.arange(len(stats))
    stats["_stratum"] = np.minimum(
        (positions * n_strata // len(stats)).astype(int), n_strata - 1
    )
    stats["_stable_key"] = stats[GROUP_COLUMN].map(
        lambda value: _stable_group_key(value, seed)
    )

    queues: dict[int, list[tuple[object, int]]] = {}
    for stratum in range(n_strata):
        part = stats[stats["_stratum"] == stratum].sort_values(
            "_stable_key", kind="mergesort"
        )
        queues[stratum] = [
            (row[GROUP_COLUMN], int(row["rows"]))
            for _, row in part.iterrows()
        ]

    selected: list[object] = []
    selected_rows = 0
    pointers = {stratum: 0 for stratum in range(n_strata)}
    while True:
        progressed = False
        for stratum in range(n_strata):
            queue = queues[stratum]
            while pointers[stratum] < len(queue):
                group_value, group_rows = queue[pointers[stratum]]
                pointers[stratum] += 1
                if selected_rows + group_rows <= max_rows:
                    selected.append(group_value)
                    selected_rows += group_rows
                    progressed = True
                    break
            if selected_rows >= max_rows:
                break
        if selected_rows >= max_rows or not progressed:
            break

    if not selected:
        raise ValueError("max_rows is smaller than every available group")

    selected_frame = df[group_series.isin(selected)].reset_index(drop=True).copy()
    metadata = {
        "enabled": True,
        "method": "deterministic_stratified_by_group",
        "whole_groups": True,
        "requested_max_rows": int(max_rows),
        "selected_rows": int(len(selected_frame)),
        "selected_groups": int(len(selected)),
        "selection_seed": int(seed),
        "truncated": True,
    }
    return selected_frame, metadata


def _build_group_splits(
    y: np.ndarray, groups: pd.Series | np.ndarray, *, n_splits: int, seed: int
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    dummy = np.zeros((len(y), 1), dtype=np.float32)
    if StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        splits = list(splitter.split(dummy, y, groups))
        splitter_name = "StratifiedGroupKFold"
    else:  # pragma: no cover - compatibility path
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(dummy, y, groups))
        splitter_name = "GroupKFold"

    for train_index, validation_index in splits:
        train_groups = set(pd.Series(groups).iloc[train_index].astype("string"))
        validation_groups = set(
            pd.Series(groups).iloc[validation_index].astype("string")
        )
        if train_groups & validation_groups:
            raise RuntimeError("group leakage detected between train and validation fold")
    return splits, splitter_name


def make_group_splits(
    y: np.ndarray, groups: pd.Series | np.ndarray, *, n_splits: int = 5, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return deterministic group-disjoint splits for tests and callers."""

    return _build_group_splits(y, groups, n_splits=n_splits, seed=seed)[0]


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # pragma: no cover - compatibility path
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("one_hot", _make_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, list(NUMERIC_FEATURES)),
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
    )


def _make_model(name: str, *, seed: int, config: dict[str, Any]):
    if name == "logistic_regression":
        return LogisticRegression(
            C=float(config.get("C", 1.0)),
            max_iter=int(config.get("max_iter", 250)),
            solver=str(config.get("solver", "liblinear")),
            class_weight=config.get("class_weight", "balanced"),
            random_state=seed,
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=int(config.get("n_estimators", 100)),
            learning_rate=float(config.get("learning_rate", 0.05)),
            num_leaves=int(config.get("num_leaves", 31)),
            max_depth=int(config.get("max_depth", -1)),
            min_child_samples=int(config.get("min_child_samples", 20)),
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
    raise ValueError(f"unsupported model: {name}")


def _make_pipeline(name: str, *, seed: int, config: dict[str, Any]) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", _make_preprocessor()),
            ("model", _make_model(name, seed=seed, config=config)),
        ]
    )


def _validate_protocol(protocol: dict[str, Any]) -> None:
    required = ("name", "seed", "n_splits", "threshold", "subsample", "models")
    missing = [key for key in required if key not in protocol]
    if missing:
        raise ValueError(f"protocol is missing keys: {', '.join(missing)}")
    if not isinstance(protocol["models"], dict):
        raise ValueError("protocol models must be an object")
    for name in MODEL_NAMES:
        if name not in protocol["models"]:
            raise ValueError(f"protocol must configure {name}")
    threshold = float(protocol["threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("protocol threshold must be between 0 and 1")
    if str(protocol.get("positive_label", "PICKUP")) != "PICKUP":
        raise ValueError("positive_label must be PICKUP")
    if int(protocol["n_splits"]) < 2:
        raise ValueError("protocol n_splits must be at least 2")
    subsample = protocol["subsample"]
    if not isinstance(subsample, dict):
        raise ValueError("protocol subsample must be an object")
    if bool(subsample.get("enabled", False)) and int(subsample.get("max_rows", 0)) <= 0:
        raise ValueError("enabled subsampling requires a positive max_rows")


def evaluate_dataframe(df: pd.DataFrame, protocol: dict[str, Any]) -> dict[str, Any]:
    """Evaluate both models and return aggregate-only JSON-compatible metadata."""

    _validate_protocol(protocol)
    validate_dataframe(df)
    seed = int(protocol["seed"])
    subsample_config = protocol["subsample"]
    if bool(subsample_config.get("enabled", False)):
        evaluated, subsample_metadata = deterministic_group_subsample(
            df, max_rows=int(subsample_config["max_rows"]), seed=seed
        )
    else:
        evaluated = df.reset_index(drop=True).copy()
        subsample_metadata = {
            "enabled": False,
            "method": "none",
            "whole_groups": True,
            "requested_max_rows": None,
            "selected_rows": int(len(evaluated)),
            "selected_groups": int(evaluated[GROUP_COLUMN].nunique(dropna=False)),
            "selection_seed": int(seed),
            "truncated": False,
        }

    y = target_array(evaluated)
    groups = evaluated[GROUP_COLUMN].reset_index(drop=True)
    X = evaluated.loc[:, list(FEATURE_COLUMNS)].reset_index(drop=True)
    splits, splitter_name = _build_group_splits(
        y,
        groups,
        n_splits=int(protocol["n_splits"]),
        seed=seed,
    )

    model_results: dict[str, Any] = {}
    for name in MODEL_NAMES:
        fold_metrics: list[dict[str, float]] = []
        model_config = dict(protocol["models"].get(name, {}))
        for train_index, validation_index in splits:
            pipeline = _make_pipeline(name, seed=seed, config=model_config)
            pipeline.fit(X.iloc[train_index], y[train_index])
            probabilities = pipeline.predict_proba(X.iloc[validation_index])
            model = pipeline.named_steps["model"]
            positive_index = int(np.flatnonzero(model.classes_ == 1)[0])
            fold_metrics.append(
                compute_metrics(
                    y[validation_index],
                    probabilities[:, positive_index],
                    threshold=float(protocol["threshold"]),
                )
            )
        model_results[name] = {
            "folds": int(len(fold_metrics)),
            "metrics": aggregate_metrics(fold_metrics),
        }

    schema_metadata = aggregate_schema_metadata(evaluated)
    return {
        "schema_version": "1.0",
        "label": str(protocol.get("label", protocol["name"])),
        "positive_label": str(protocol.get("positive_label", "PICKUP")),
        "aggregate_only": True,
        "row_level_predictions_written": False,
        "environment": {"package_versions": _package_versions()},
        "data": {
            "kind": str(protocol.get("data_kind", "external_csv")),
            "source": str(protocol.get("data_source", "caller-provided CSV")),
            "rows_read": int(len(df)),
            "rows_evaluated": schema_metadata["rows"],
            "groups_evaluated": schema_metadata["groups"],
            "class_counts": schema_metadata["class_counts"],
            "subsampling": subsample_metadata,
        },
        "evaluation": {
            "grouped_by": GROUP_COLUMN,
            "splitter": splitter_name,
            "n_splits": int(protocol["n_splits"]),
            "seed": seed,
            "threshold": float(protocol["threshold"]),
            "preprocessing": "fit separately inside each training fold",
            "feature_policy": "coordinates and entity/row identifiers excluded",
        },
        "models": model_results,
    }


def evaluate_csv(train_csv: str | Path, protocol: dict[str, Any]) -> dict[str, Any]:
    """Read one external training CSV and evaluate it without writing predictions."""

    return evaluate_dataframe(load_training_csv(train_csv), protocol)


def load_protocol(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    if not isinstance(protocol, dict):
        raise ValueError("protocol JSON must contain an object")
    _validate_protocol(protocol)
    return protocol


def write_result(result: dict[str, Any], output_dir: str | Path) -> Path:
    """Write only aggregate result metadata to ``results.json``."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / "results.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    return result_path
