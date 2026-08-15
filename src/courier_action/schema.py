"""Schema and feature-policy checks for the classification task."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

TARGET_COLUMN = "action_type"
GROUP_COLUMN = "courier_id"
VALID_LABELS = ("DELIVERY", "PICKUP")

REQUIRED_COLUMNS = (
    "courier_id",
    "wave_index",
    "tracking_id",
    "courier_wave_start_lng",
    "courier_wave_start_lat",
    "action_type",
    "date",
    "group",
    "level",
    "speed",
    "max_load",
    "weather_grade",
    "aoi_id",
    "shop_id",
    "id",
    "source_type",
    "source_tracking_id",
    "source_lng",
    "source_lat",
    "target_lng",
    "target_lat",
    "grid_distance",
    "expected_use_time",
    "urgency",
    "hour",
)

# These are the only columns exposed to a model.  Entity identifiers, row and
# tracking identifiers, and all coordinates are deliberately excluded.
FEATURE_COLUMNS = (
    "wave_index",
    "date",
    "group",
    "level",
    "speed",
    "max_load",
    "weather_grade",
    "source_type",
    "grid_distance",
    "expected_use_time",
    "urgency",
    "hour",
)

CATEGORICAL_FEATURES = ("group", "weather_grade", "source_type")
NUMERIC_FEATURES = tuple(c for c in FEATURE_COLUMNS if c not in CATEGORICAL_FEATURES)

IDENTIFIER_COLUMNS = (
    "courier_id",
    "tracking_id",
    "source_tracking_id",
    "aoi_id",
    "shop_id",
    "id",
)
COORDINATE_COLUMNS = (
    "courier_wave_start_lng",
    "courier_wave_start_lat",
    "source_lng",
    "source_lat",
    "target_lng",
    "target_lat",
)


class SchemaError(ValueError):
    """Raised when an input frame does not satisfy the task schema."""


def validate_dataframe(df: pd.DataFrame, *, require_target: bool = True) -> None:
    """Validate required columns and values without exposing row-level data."""

    if not isinstance(df, pd.DataFrame):
        raise SchemaError("input must be a pandas DataFrame")
    if not df.columns.is_unique:
        raise SchemaError("input columns must be unique")
    if df.empty:
        raise SchemaError("input must contain at least one row")

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise SchemaError(f"missing required columns: {', '.join(missing)}")

    group_values = df[GROUP_COLUMN]
    if group_values.isna().any():
        raise SchemaError("courier_id contains missing values")
    if group_values.astype("string").str.strip().eq("").any():
        raise SchemaError("courier_id contains blank values")

    if require_target:
        target = df[TARGET_COLUMN].astype("string")
        if target.isna().any():
            raise SchemaError("action_type contains missing values")
        unknown = sorted(set(target.dropna().tolist()) - set(VALID_LABELS))
        if unknown:
            raise SchemaError("action_type contains labels outside PICKUP/DELIVERY")
        if target.nunique(dropna=False) != len(VALID_LABELS):
            raise SchemaError("action_type must contain both PICKUP and DELIVERY")

    for column in NUMERIC_FEATURES:
        original = df[column]
        coerced = pd.to_numeric(original, errors="coerce")
        invalid = coerced.isna() & original.notna()
        if invalid.any():
            raise SchemaError(f"numeric feature {column!r} contains non-numeric values")


def load_training_csv(path: str | Path) -> pd.DataFrame:
    """Read and validate a training CSV; callers must keep the source external."""

    frame = pd.read_csv(path)
    validate_dataframe(frame)
    return frame


def target_array(df: pd.DataFrame):
    """Return the fixed binary target encoding, with PICKUP as positive."""

    validate_dataframe(df)
    return (df[TARGET_COLUMN].astype("string") == "PICKUP").astype("int8").to_numpy()


def aggregate_schema_metadata(df: pd.DataFrame) -> dict[str, Any]:
    """Return safe aggregate metadata for result files."""

    validate_dataframe(df)
    target = df[TARGET_COLUMN].astype("string")
    return {
        "rows": int(len(df)),
        "groups": int(df[GROUP_COLUMN].nunique(dropna=False)),
        "class_counts": {
            label: int((target == label).sum()) for label in VALID_LABELS
        },
    }
