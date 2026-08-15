# Data policy

The training data are not included in this repository. The expected training
CSV has these required columns:

| Column | Role |
| --- | --- |
| `courier_id` | group key for grouped cross-validation |
| `wave_index` | input feature |
| `tracking_id` | source identifier; excluded from features |
| `courier_wave_start_lng` | coordinate; excluded from features |
| `courier_wave_start_lat` | coordinate; excluded from features |
| `action_type` | target: `PICKUP` or `DELIVERY` |
| `date` | input feature |
| `group` | input feature |
| `level` | input feature |
| `speed` | input feature |
| `max_load` | input feature |
| `weather_grade` | input feature |
| `aoi_id` | entity identifier; excluded from features |
| `shop_id` | entity identifier; excluded from features |
| `id` | row identifier; excluded from features |
| `source_type` | input feature |
| `source_tracking_id` | source identifier; excluded from features |
| `source_lng` | coordinate; excluded from features |
| `source_lat` | coordinate; excluded from features |
| `target_lng` | coordinate; excluded from features |
| `target_lat` | coordinate; excluded from features |
| `grid_distance` | input feature |
| `expected_use_time` | input feature |
| `urgency` | input feature |
| `hour` | input feature |

Authorized access is through the CUHK course Kaggle competition
`cuhkml-2025-classification`. Redistribution rights were not established for
the course data, so it remains outside this repository. Do not commit source
CSVs, submissions, notebooks, PDFs, identifiers, or coordinates.

The included `make-synthetic` command creates fabricated `SYNTH_` records for
software checks. Those records are not a substitute for authorized data and do
not reflect its distributions.
