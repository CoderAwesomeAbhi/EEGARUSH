# MAT Metric Alignment Audit

Verdict: `no_metric_or_alignment_bug_detected`

## Alignment Findings

- Prediction/label/alignment bug exists: `False`.
- Evaluated window counts balanced across subjects: `True`.
- Every paired comparison uses the same sorted `row_id`, label vector, and subject vector across configurations.
- Pooled and macro subject-level metrics legitimately differ because AUC is nonlinear; even with equal per-subject window counts, each subject's within-subject separability can differ from pooled ranking across all held-out scores.
- Previous z-score paired deltas were computed from subject-level AUCs and are directionally consistent with this audit.

## Issues

_None._

## Subject Window Counts

| subject_id | scored_rest_windows | scored_task_windows |
| ---------- | ------------------- | ------------------- |
| Subject00  | 14                  | 14                  |
| Subject01  | 14                  | 14                  |
| Subject02  | 14                  | 14                  |
| Subject03  | 14                  | 14                  |
| Subject04  | 14                  | 14                  |
| Subject05  | 14                  | 14                  |
| Subject06  | 14                  | 14                  |
| Subject07  | 14                  | 14                  |
| Subject08  | 14                  | 14                  |
| Subject09  | 14                  | 14                  |
| Subject10  | 14                  | 14                  |
| Subject11  | 14                  | 14                  |
| Subject12  | 14                  | 14                  |
| Subject13  | 14                  | 14                  |
| Subject14  | 14                  | 14                  |
| Subject15  | 14                  | 14                  |
| Subject16  | 14                  | 14                  |
| Subject17  | 14                  | 14                  |
| Subject18  | 14                  | 14                  |
| Subject19  | 14                  | 14                  |
| Subject20  | 14                  | 14                  |
| Subject21  | 14                  | 14                  |
| Subject22  | 14                  | 14                  |
| Subject23  | 14                  | 14                  |
| Subject24  | 14                  | 14                  |
| Subject25  | 14                  | 14                  |
| Subject26  | 14                  | 14                  |
| Subject27  | 14                  | 14                  |
| Subject28  | 14                  | 14                  |
| Subject29  | 14                  | 14                  |
| Subject30  | 14                  | 14                  |
| Subject31  | 14                  | 14                  |
| Subject32  | 14                  | 14                  |
| Subject33  | 14                  | 14                  |
| Subject34  | 14                  | 14                  |
| Subject35  | 14                  | 14                  |

## Recomputed Metrics

| model       | calibration      | pooled_window_auc | macro_subject_mean_auc | macro_subject_median_auc | subject_auc_sd | subject_auc_ci95_low | subject_auc_ci95_high | n_subjects | n_predictions |
| ----------- | ---------------- | ----------------- | ---------------------- | ------------------------ | -------------- | -------------------- | --------------------- | ---------- | ------------- |
| linear_svm  | absolute         | 0.771034          | 0.84042                | 0.897959                 | 0.1561         | 0.786848             | 0.886905              | 36         | 1008          |
| linear_svm  | mean_subtraction | 0.838349          | 0.863662               | 0.887755                 | 0.133316       | 0.81746              | 0.903345              | 36         | 1008          |
| linear_svm  | zscore           | 0.789163          | 0.803713               | 0.84949                  | 0.193818       | 0.738088             | 0.86352               | 36         | 1008          |
| logistic_l2 | absolute         | 0.770963          | 0.841553               | 0.885204                 | 0.166267       | 0.785998             | 0.89229               | 36         | 1008          |
| logistic_l2 | mean_subtraction | 0.858588          | 0.880102               | 0.903061                 | 0.136791       | 0.832766             | 0.920638              | 36         | 1008          |
| logistic_l2 | zscore           | 0.801985          | 0.816327               | 0.852041                 | 0.185129       | 0.751842             | 0.871032              | 36         | 1008          |

## Mean-Subtraction Paired Bootstrap

| model       | comparison                      | n_subjects | mean_delta_auc | median_delta_auc | ci95_low   | ci95_high | ci_excludes_zero | positive_ci_excludes_zero | bootstrap_resamples |
| ----------- | ------------------------------- | ---------- | -------------- | ---------------- | ---------- | --------- | ---------------- | ------------------------- | ------------------- |
| logistic_l2 | mean_subtraction_minus_absolute | 36         | 0.0385488      | 0.0357143        | -0.0185658 | 0.0931122 | False            | False                     | 10000               |
| logistic_l2 | mean_subtraction_minus_zscore   | 36         | 0.0637755      | 0.0331633        | 0.0119048  | 0.117489  | True             | True                      | 10000               |
| linear_svm  | mean_subtraction_minus_absolute | 36         | 0.0232426      | 0.0204082        | -0.0290533 | 0.0717156 | False            | False                     | 10000               |
| linear_svm  | mean_subtraction_minus_zscore   | 36         | 0.059949       | 0.0204082        | 0.00708617 | 0.116071  | True             | True                      | 10000               |
