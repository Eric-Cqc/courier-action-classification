# Courier action classification

This repository is a code-only, leakage-audited reconstruction of a binary
classification workflow. The target is `action_type`: `PICKUP` versus
`DELIVERY`. It compares logistic regression with LightGBM and emits aggregate
cross-validation metrics only.

> Reconstruction of a CUHK course classification task hosted on Kaggle (cuhkml-2025-classification).

The assignment frames the records in a food-delivery courier context. The
underlying dataset's origin and redistribution licence were not independently
established; the course data is not included.

No leaderboard placement or rank is claimed.

## Quick start

The package is designed for the supplied offline virtual environment:

```bash
. .venv/bin/activate
python -m pip install -e . --no-deps --no-build-isolation
python -m pytest -q
```

Create a fabricated smoke dataset and evaluate it:

```bash
courier-action make-synthetic --output-dir /tmp/ca-synth --seed 42
courier-action evaluate \
  --train-csv /tmp/ca-synth/train.csv \
  --protocol configs/smoke.json \
  --output-dir /tmp/ca-smoke
```

The synthetic output is deliberately artificial: identifiers begin with
`SYNTH_`, dates are future placeholders, and the feature relationships are
hand-built for software testing. It does not reflect real distributions and
its metrics are labelled `synthetic smoke — not a performance claim`.

## Evaluation protocol

The headline protocol is deterministic grouped cross-validation:

- The grouping key is `courier_id`; `StratifiedGroupKFold` is used when the
  installed scikit-learn provides it, with a `GroupKFold` compatibility path.
- All imputers, scaling, and one-hot encoding are fitted inside each training
  fold. Validation rows are transformed only after that fit; training and
  validation data are never concatenated for preprocessing.
- The decision threshold is fixed at `0.5`. It is not selected from the same
  out-of-fold predictions used for reporting.
- Seeds are explicit and model fitting uses one worker for deterministic
  smoke/rebuild runs.
- The evaluator writes no row-level predictions, identifiers, or coordinates.
  Result JSON contains only aggregate counts and fold mean/std metrics.

The authorized configuration uses a deterministic stratified-by-group
subsample with a maximum of 160,000 rows for runtime. Whole courier groups are
selected, target-rate strata are visited round-robin, and the seed is recorded
in the result metadata. The selected row count can be below the cap because a
group is never split. The synthetic smoke configuration evaluates its complete
small fabricated dataset.

## Differences from the legacy coursework notebook

The legacy notebook is treated as a read-only reference and is not shipped.
Its workflow used a random stratified split for intermediate checks, fitted
some transformations before validation, combined train and test frames for
feature work, and used non-grouped cross-validation for the final classifier.
It also searched thresholds on the same out-of-fold probabilities that were
then reported. The legacy CV F1 was inflated by leaky evaluation; this
repository does not republish the legacy number.

This reconstruction keeps the target task and model comparison while making
the leakage controls explicit: courier-disjoint folds, fold-local
preprocessing, a fixed threshold, deterministic selection, and aggregate-only
artifacts. Regression is outside scope.

## Model card

### Intended use

This is an educational and research reproducibility scaffold for auditing a
binary courier-action classifier. It is not an operational dispatch system,
does not make decisions for couriers, and should not be used as a production
service without a separate safety, privacy, and data-governance review.

### Inputs and exclusions

The schema is documented in [data/README.md](data/README.md). The model uses
non-identifier context fields such as wave, time, load, weather, source type,
distance, expected use time, urgency, and hour. Courier, tracking, shop, AOI,
row identifiers, and all longitude/latitude fields are excluded from model
features. `courier_id` is used only to form disjoint evaluation groups.

### Leakage controls

The preprocessing pipeline is refit for every training fold. The validation
fold is kept group-disjoint from training, no train/test concatenation is used,
and threshold selection is not performed on reported predictions. These
controls estimate generalization to unseen courier groups rather than
memorization of repeated entities.

### Privacy and data governance

No source CSV, submission, notebook, PDF, row-level prediction, courier/order
identifier, or real coordinate is included. The authorized CSV must stay
outside the repository. Redistribution rights were not independently
established, so users must obtain their own authorized access.

### Limitations

The binary label and available columns may not represent operational practice.
The authorized result uses a whole-group subsample for runtime, so it is not a
full-data estimate. Results cannot be reproduced without authorized access to
the external course CSV and the same protocol; the synthetic smoke result is
only a software check. The course-data origin and redistribution licence have
not been independently established, and no claim is made about deployment
utility.

## Results and verification

`results/authorized-evaluation.json` contains the regenerated clean-protocol
real-data aggregate result. `results/synthetic-smoke.json` is an aggregate
smoke artifact explicitly labelled `synthetic smoke — not a performance claim`.
Both files contain model-level mean and standard deviation for F1, AUC, and
accuracy, plus protocol metadata; neither contains row-level output.

Verify a fresh authorized run against the checked-in aggregate artifact:

```bash
python scripts/verify_results.py \
  --expected results/authorized-evaluation.json \
  --actual /tmp/ca-verify/results.json
python scripts/audit_public_tree.py
```

The project is distributed under the MIT License for the original code and
documentation only. See [data/README.md](data/README.md) for the separate data
policy.
