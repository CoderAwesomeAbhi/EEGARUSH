# MAT Raw Rebuild Decision

Final verdict: `MAT_CACHED_SIGNAL_REPRODUCES_BUT_BALANCED_SIGNAL_WEAK`

## Raw Metadata Summary

- MAT raw files are usable with raw-header-driven splitting: 72 EDF files, 36 subjects, 500 Hz, stable 21-channel identity.
- Duration variability did not invalidate analysis; it required exact header-driven window provenance.
- Primary feature set is locked as `no_gamma_184` for MAT.

## Cached Reproduction

- Cached result reproduced from raw EDF data: `True`.

## Corrected Balanced Primary Metrics

| model       | calibration      | window_auc | subject_auc_mean | subject_auc_ci95_low | subject_auc_ci95_high |
| ----------- | ---------------- | ---------- | ---------------- | -------------------- | --------------------- |
| logistic_l2 | absolute         | 0.770963   | 0.841553         | 0.78344              | 0.891443              |
| logistic_l2 | mean_subtraction | 0.858588   | 0.880102         | 0.834605             | 0.920351              |
| logistic_l2 | zscore           | 0.801985   | 0.816327         | 0.754815             | 0.86976               |
| linear_svm  | absolute         | 0.771034   | 0.84042          | 0.785994             | 0.887897              |
| linear_svm  | mean_subtraction | 0.838349   | 0.863662         | 0.817319             | 0.903348              |
| linear_svm  | zscore           | 0.789163   | 0.803713         | 0.741628             | 0.862837              |

## Balanced Delta Inference

| comparison                                | n_subjects | mean_delta_subject_auc | median_delta_subject_auc | ci95_low  | ci95_high   | bootstrap_resamples |
| ----------------------------------------- | ---------- | ---------------------- | ------------------------ | --------- | ----------- | ------------------- |
| logistic_l2_zscore_minus_absolute         | 36         | -0.0252268             | -0.00255102              | -0.09893  | 0.0506023   | 2000                |
| logistic_l2_zscore_minus_mean_subtraction | 36         | -0.0637755             | -0.0331633               | -0.11735  | -0.00906675 | 2000                |
| linear_svm_zscore_minus_absolute          | 36         | -0.0367063             | 0.00510204               | -0.112408 | 0.0396861   | 2000                |
| linear_svm_zscore_minus_mean_subtraction  | 36         | -0.059949              | -0.0204082               | -0.11452  | -0.00707908 | 2000                |

## Full-Pipeline Null

- Strongest balanced configuration: `logistic_l2` / `mean_subtraction`.
- Completed permutations: `200`.
- Observed AUC: `0.858588`.
- Null mean: `0.499037`.
- Null 95% interval: `[0.443191, 0.554170]`.
- Empirical p-value: `0.004975`.

## STEW Gate

- Cleared to proceed to STEW reconstruction: `False`.
