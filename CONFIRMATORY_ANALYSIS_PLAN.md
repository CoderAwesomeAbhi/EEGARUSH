# DS007262 Confirmatory Analysis Plan

## Frozen Exploratory Choice

The exploratory MAT/STEW report (`THEORY_VALIDATION_RESULTS.md`) is the only source used for confirmatory model selection. The locked model is `zscore + logistic_l2` with fixed `C=1.0`, because it has the highest mean window ROC-AUC across MAT and STEW among the tested single-model candidates:

| Candidate | MAT ROC-AUC | STEW ROC-AUC | Mean ROC-AUC |
| --- | ---: | ---: | ---: |
| zscore + logistic_l2 | 0.865995 | 0.791446 | 0.8287205 |
| zscore + linear_svm | 0.858511 | 0.791896 | 0.8252035 |

No DS007262 labels, difficulty levels, or scores are used to choose this model, its hyperparameters, its feature set, or its preprocessing pipeline.

## Frozen Training Pipeline

The training data are the exploratory MAT and STEW feature tables only:

| Dataset | Feature table | Calibration rows | Scored training rows |
| --- | --- | --- | --- |
| MAT | `outputs_reproduced/features/eeg_features.csv` | First 60 seconds of resting rows | Workload rows plus rest rows after 60 seconds |
| STEW | `results/multi_dataset/stew_features.parquet` | First 50% of cached rest rows per subject | Workload rows plus remaining rest rows |

For each training subject, the 200 selected EEG features are z-scored relative to that subject's calibration rows. The combined MAT/STEW scored rows are then passed through median imputation, `StandardScaler`, and L2-regularized logistic regression with `C=1.0`, `solver="liblinear"`, `class_weight="balanced"`, `max_iter=5000`, and `random_state=20260602`.

## Frozen Feature Set

The selected features are the Cartesian product of the eight common EEG channels (`F3`, `F4`, `F7`, `F8`, `O1`, `O2`, `T3`, `T4`) and the 25 per-channel feature templates listed in `FROZEN_CONFIG.yaml`, producing exactly 200 features. No DS007262-specific feature selection is allowed.

## DS007262 External Test

DS007262 is used only for external scoring. The runner materializes missing BrainVision files from OpenNeuro S3 when `--download-missing` is passed, because the local checkout stores annex/LFS pointer stubs rather than raw EEG payloads.

For each DS007262 subject:

1. Read the subject's BrainVision EEG recording.
2. Locate `started_tutorial_artihmetic` and the first numeric difficulty trial in `*_events.tsv`.
3. Use the first 60 seconds after `started_tutorial_artihmetic` as the subject's baseline-calibration epoch, split into non-overlapping 6-second windows.
4. Score every non-tutorial arithmetic trial with numeric `difficulty_range`, positive duration, and no dropped-samples event row.
5. Apply the frozen subject z-score calibration, the MAT/STEW-fitted imputer and scaler, and the MAT/STEW-fitted logistic regression decision function.

The local DS007262 event files expose seven non-tutorial difficulty ranges (`0.6-1.5` through `6.0-6.9`), not eight. The ordered ranges are mapped to ordinal levels 1-7. If a future materialized DS007262 snapshot contains an eighth range, the same sorting rule extends the mapping to level 8 without changing the model.

## Primary Confirmatory Statistic

The primary statistic is Spearman rank correlation between ordinal workload level and the frozen model's continuous decision score, computed on subject-level means for each available subject-level difficulty bin. This aggregation prevents subjects with more usable epochs from dominating the correlation.

Secondary descriptive outputs include trial-level Spearman correlation, per-subject Spearman correlations where computable, a level-summary CSV, and a score-by-level figure. These secondary outputs cannot override the primary success/failure rule.
