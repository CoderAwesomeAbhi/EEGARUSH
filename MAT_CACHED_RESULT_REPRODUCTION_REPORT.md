# MAT Cached Result Reproduction Report

Verdict: `MAT_CACHED_SIGNAL_REPRODUCED`

The reproduction protocol uses raw EDF-derived no-gamma features with the cached split geometry: rest `<60s` for calibration, rest `>=60s` plus all arithmetic windows for scoring.

## Raw Reproduction Metrics

| model       | calibration      | window_auc | subject_auc_mean | subject_auc_ci95_low | subject_auc_ci95_high | n_predictions |
| ----------- | ---------------- | ---------- | ---------------- | -------------------- | --------------------- | ------------- |
| logistic_l2 | absolute         | 0.716073   | 0.776762         | 0.723067             | 0.826821              | 3186          |
| logistic_l2 | mean_subtraction | 0.822098   | 0.835301         | 0.78359              | 0.879608              | 3186          |
| logistic_l2 | zscore           | 0.873201   | 0.879315         | 0.844676             | 0.910963              | 3186          |
| linear_svm  | absolute         | 0.711653   | 0.770616         | 0.717662             | 0.822377              | 3186          |
| linear_svm  | mean_subtraction | 0.824031   | 0.835938         | 0.786747             | 0.878771              | 3186          |
| linear_svm  | zscore           | 0.867735   | 0.875075         | 0.83958              | 0.909635              | 3186          |

## Cached-vs-Raw Comparison

| model       | calibration      | window_auc_raw_no_gamma | window_auc_cached_all_200 | window_auc_delta_raw_minus_cached | reproduction_status              |
| ----------- | ---------------- | ----------------------- | ------------------------- | --------------------------------- | -------------------------------- |
| logistic_l2 | absolute         | 0.716073                | 0.736025                  | -0.0199523                        | metric_agreement_within_0.03_auc |
| logistic_l2 | mean_subtraction | 0.822098                | 0.826001                  | -0.00390287                       | metric_agreement_within_0.03_auc |
| logistic_l2 | zscore           | 0.873201                | 0.865995                  | 0.00720604                        | metric_agreement_within_0.03_auc |
| linear_svm  | absolute         | 0.711653                | 0.736328                  | -0.0246751                        | metric_agreement_within_0.03_auc |
| linear_svm  | mean_subtraction | 0.824031                | 0.820951                  | 0.00307982                        | metric_agreement_within_0.03_auc |
| linear_svm  | zscore           | 0.867735                | 0.858511                  | 0.00922409                        | metric_agreement_within_0.03_auc |
