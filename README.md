# GDG Network Intrusion Detection Challenge — Baseline

## Objective
Predict one of four network traffic classes (`Benign`, `Availability`,
`Reconnaissance`, `Spoofing`) for each row in `test.csv`. The competition metric
is Macro F1.

## Files
Place the following files in this directory before running the code:
- `train.csv`
- `test.csv`
- `sample_submission.csv`

The script creates:
- `submission.csv`

## Baseline methodology
- Excludes `id` from model features because it is a record identifier.
- Separates numerical and categorical columns based on their data types.
- Uses median imputation for numerical features and most-frequent imputation
  plus one-hot encoding for categorical features.
- Trains a class-weighted Extra Trees classifier to account for severe class
  imbalance.
- Reports Macro F1 and per-class metrics on a stratified random validation split.
- Retrains on all labeled rows and writes a CSV with exactly `id,Target`.

## Important validation limitation
The competition description says that test traffic behavior profiles are
separated from the training data. A random validation split may therefore be
too optimistic and may not estimate leaderboard performance reliably. This
baseline is a starting point, not proof of strong generalization. Before final
submission, investigate whether the dataset has a defensible profile/group
variable and compare a group-aware validation strategy if possible. Do not
invent a grouping variable or claim that a validation method represents the
hidden test split unless the data supports that claim.

## Run
Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Place the three CSV files beside `train.py`, then run:

```bash
python train.py
```

## Results
Record the actual validation Macro F1 and per-class precision/recall printed by
your run here. Do not report a score until you have run the code.

## Interview preparation
Be prepared to explain:
- Why Macro F1 is the main metric.
- Why class imbalance matters.
- Why `id` is excluded from features.
- Why categorical features are one-hot encoded.
- What class weighting does and what its limitations are.
- Why random validation may not match the challenge's separated traffic profiles.
- What further experiments you performed and how you chose the final model.
