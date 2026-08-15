"""Deterministic, unmistakably fabricated data for smoke tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import REQUIRED_COLUMNS


def generate_synthetic_dataframe(
    *, seed: int = 42, n_groups: int = 80, rows_per_group: int = 64
) -> pd.DataFrame:
    """Generate artificial records that satisfy the external task schema.

    The values use ``SYNTH_`` identifiers, future dates, and artificial feature
    patterns.  They are only for software smoke tests and do not represent the
    distribution of any courier, order, or course dataset.
    """

    if n_groups < 5:
        raise ValueError("n_groups must be at least 5 for grouped cross-validation")
    if rows_per_group < 8 or rows_per_group % 2:
        raise ValueError("rows_per_group must be an even number of at least 8")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for group_index in range(n_groups):
        # Keep each group balanced and make group sizes deterministic but varied.
        variation = int(rng.integers(-3, 4)) * 2
        group_rows = rows_per_group + variation
        labels = np.array(["PICKUP"] * (group_rows // 2) + ["DELIVERY"] * (group_rows // 2))
        rng.shuffle(labels)
        group_id = f"SYNTH_COURIER_{group_index:04d}"

        for row_index, label in enumerate(labels):
            is_pickup = label == "PICKUP"
            signal = 1.0 if is_pickup else -1.0
            weather = rng.choice(
                ["SYNTH_CLEAR", "SYNTH_RAIN", "SYNTH_WIND"],
                p=[0.55, 0.30, 0.15] if is_pickup else [0.25, 0.45, 0.30],
            )
            source_type = rng.choice(
                ["SYNTH_SHOP", "SYNTH_HUB", "SYNTH_LOCKER"],
                p=[0.55, 0.25, 0.20] if is_pickup else [0.25, 0.45, 0.30],
            )
            rows.append(
                {
                    "courier_id": group_id,
                    "wave_index": int(group_index % 5),
                    "tracking_id": f"SYNTH_TRACKING_{group_index:04d}_{row_index:04d}",
                    "courier_wave_start_lng": round(0.001 * group_index, 6),
                    "courier_wave_start_lat": round(0.002 * group_index, 6),
                    "action_type": label,
                    "date": 20990101 + int(group_index % 27),
                    "group": float(group_index % 3),
                    "level": int(2 + (group_index % 4)),
                    "speed": round(float(25.0 + signal * 4.0 + rng.normal(0, 1.5)), 5),
                    "max_load": int(10 + (group_index % 6)),
                    "weather_grade": weather,
                    "aoi_id": f"SYNTH_AOI_{group_index % 9:03d}",
                    "shop_id": f"SYNTH_SHOP_{group_index % 13:03d}",
                    "id": f"SYNTH_ROW_{group_index:04d}_{row_index:04d}",
                    "source_type": source_type,
                    "source_tracking_id": f"SYNTH_SOURCE_{group_index:04d}_{row_index:04d}",
                    "source_lng": round(0.010 + 0.001 * group_index, 6),
                    "source_lat": round(0.020 + 0.001 * group_index, 6),
                    "target_lng": round(0.030 + 0.001 * group_index, 6),
                    "target_lat": round(0.040 + 0.001 * group_index, 6),
                    "grid_distance": round(float(8.0 - signal * 1.5 + rng.normal(0, 0.8)), 5),
                    "expected_use_time": int(max(1, 30 - 5 * signal + rng.normal(0, 3))),
                    "urgency": round(float(50.0 + 15.0 * signal + rng.normal(0, 8.0)), 5),
                    "hour": int(rng.integers(0, 24)),
                }
            )

    frame = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    permutation = rng.permutation(len(frame))
    return frame.iloc[permutation].reset_index(drop=True)
