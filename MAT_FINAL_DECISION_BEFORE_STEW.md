# MAT Final Decision Before STEW

## Configuration

- Protocol: corrected balanced raw MAT.
- Feature set: no-gamma 184.
- Candidate primary method: logistic L2 mean subtraction.
- Primary statistic: macro subject-level mean ROC-AUC.

## Metric Alignment Verdict

- Alignment passed: `True`.

## Subject-Level Metrics

| model       | calibration      | pooled_window_auc | macro_subject_mean_auc | macro_subject_median_auc | subject_auc_sd | subject_auc_ci95_low | subject_auc_ci95_high | n_subjects | n_predictions |
| ----------- | ---------------- | ----------------- | ---------------------- | ------------------------ | -------------- | -------------------- | --------------------- | ---------- | ------------- |
| linear_svm  | absolute         | 0.771034          | 0.84042                | 0.897959                 | 0.1561         | 0.786848             | 0.886905              | 36         | 1008          |
| linear_svm  | mean_subtraction | 0.838349          | 0.863662               | 0.887755                 | 0.133316       | 0.81746              | 0.903345              | 36         | 1008          |
| linear_svm  | zscore           | 0.789163          | 0.803713               | 0.84949                  | 0.193818       | 0.738088             | 0.86352               | 36         | 1008          |
| logistic_l2 | absolute         | 0.770963          | 0.841553               | 0.885204                 | 0.166267       | 0.785998             | 0.89229               | 36         | 1008          |
| logistic_l2 | mean_subtraction | 0.858588          | 0.880102               | 0.903061                 | 0.136791       | 0.832766             | 0.920638              | 36         | 1008          |
| logistic_l2 | zscore           | 0.801985          | 0.816327               | 0.852041                 | 0.185129       | 0.751842             | 0.871032              | 36         | 1008          |

## Paired Bootstrap Differences

| model       | comparison                      | n_subjects | mean_delta_auc | median_delta_auc | ci95_low   | ci95_high | ci_excludes_zero | positive_ci_excludes_zero | bootstrap_resamples |
| ----------- | ------------------------------- | ---------- | -------------- | ---------------- | ---------- | --------- | ---------------- | ------------------------- | ------------------- |
| logistic_l2 | mean_subtraction_minus_absolute | 36         | 0.0385488      | 0.0357143        | -0.0185658 | 0.0931122 | False            | False                     | 10000               |
| logistic_l2 | mean_subtraction_minus_zscore   | 36         | 0.0637755      | 0.0331633        | 0.0119048  | 0.117489  | True             | True                      | 10000               |
| linear_svm  | mean_subtraction_minus_absolute | 36         | 0.0232426      | 0.0204082        | -0.0290533 | 0.0717156 | False            | False                     | 10000               |
| linear_svm  | mean_subtraction_minus_zscore   | 36         | 0.059949       | 0.0204082        | 0.00708617 | 0.116071  | True             | True                      | 10000               |

## Existing Permutation Statistic

- Earlier permutation unit: `pooled_window_auc`.

## Subject-Level Null

- Subject-level full-pipeline null ran: `False`.
- Null not run because conditional criteria were not all satisfied.

## STEW Gate

- Proceeding to STEW is scientifically justified: `False`.

Final verdict: `MAT_MEAN_SUBTRACTION_PROMISING_BUT_NOT_PROVEN_DO_NOT_PROCEED`
