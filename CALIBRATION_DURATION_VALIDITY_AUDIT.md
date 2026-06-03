# Calibration-Duration Validity Audit

The previously reported MAT curve changed both calibration duration and the set of scored rest windows because the scoring rule was `rest start >= calibration_seconds`.
That confounds calibration length with evaluation-set composition.

A corrected feature-table-level design is specified by holding scoring windows fixed to workload rows plus rest rows with `start_sec >= 60`, while varying only the calibration rows (`rest start < duration`).
By default, this audit writes the fixed-evaluation design manifest without running another expensive refit curve. Set `RUN_CORRECTED_DURATION_CURVE=1` to compute corrected AUCs.

## Corrected Fixed-Evaluation Curve

| baseline_seconds | fixed_eval_windows | calibration_windows | calibration_scoring_overlap | valid_subjects_fixed_eval | window_auc | n_predictions | status                                     |
| ---------------- | ------------------ | ------------------- | --------------------------- | ------------------------- | ---------- | ------------- | ------------------------------------------ |
| 1                | 3186               | 36                  | 0                           | 36                        | nan        | 0             | not_run_set_RUN_CORRECTED_DURATION_CURVE_1 |
| 3                | 3186               | 72                  | 0                           | 36                        | nan        | 0             | not_run_set_RUN_CORRECTED_DURATION_CURVE_1 |
| 5                | 3186               | 108                 | 0                           | 36                        | nan        | 0             | not_run_set_RUN_CORRECTED_DURATION_CURVE_1 |
| 7                | 3186               | 144                 | 0                           | 36                        | nan        | 0             | not_run_set_RUN_CORRECTED_DURATION_CURVE_1 |
| 9                | 3186               | 180                 | 0                           | 36                        | nan        | 0             | not_run_set_RUN_CORRECTED_DURATION_CURVE_1 |
| 11               | 3186               | 216                 | 0                           | 36                        | nan        | 0             | not_run_set_RUN_CORRECTED_DURATION_CURVE_1 |
| 13               | 3186               | 252                 | 0                           | 36                        | nan        | 0             | not_run_set_RUN_CORRECTED_DURATION_CURVE_1 |
| 15               | 3186               | 288                 | 0                           | 36                        | nan        | 0             | not_run_set_RUN_CORRECTED_DURATION_CURVE_1 |
| 17               | 3186               | 324                 | 0                           | 36                        | nan        | 0             | not_run_set_RUN_CORRECTED_DURATION_CURVE_1 |
| 19               | 3186               | 360                 | 0                           | 36                        | nan        | 0             | not_run_set_RUN_CORRECTED_DURATION_CURVE_1 |
| 21               | 3186               | 396                 | 0                           | 36                        | nan        | 0             | not_run_set_RUN_CORRECTED_DURATION_CURVE_1 |
| 23               | 3186               | 432                 | 0                           | 36                        | nan        | 0             | not_run_set_RUN_CORRECTED_DURATION_CURVE_1 |

## Recommendation

The old duration figure should be removed or demoted. If a duration curve is retained, use the fixed-evaluation curve and explicitly state that it is feature-table-level because raw EDF header provenance was unavailable.
