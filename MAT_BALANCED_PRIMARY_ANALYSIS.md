# MAT Balanced Primary Analysis

This is the scientifically preferred MAT design: every subject contributes 30 s rest calibration, 30 s scored rest, and 30 s scored arithmetic using no-gamma 184 features.

## Metrics

| model       | calibration      | window_auc | subject_auc_mean | subject_auc_ci95_low | subject_auc_ci95_high | n_predictions |
| ----------- | ---------------- | ---------- | ---------------- | -------------------- | --------------------- | ------------- |
| logistic_l2 | absolute         | 0.770963   | 0.841553         | 0.78344              | 0.891443              | 1008          |
| logistic_l2 | mean_subtraction | 0.858588   | 0.880102         | 0.834605             | 0.920351              | 1008          |
| logistic_l2 | zscore           | 0.801985   | 0.816327         | 0.754815             | 0.86976               | 1008          |
| linear_svm  | absolute         | 0.771034   | 0.84042          | 0.785994             | 0.887897              | 1008          |
| linear_svm  | mean_subtraction | 0.838349   | 0.863662         | 0.817319             | 0.903348              | 1008          |
| linear_svm  | zscore           | 0.789163   | 0.803713         | 0.741628             | 0.862837              | 1008          |

## Paired Subject Bootstrap Deltas

| comparison                                | n_subjects | mean_delta_subject_auc | median_delta_subject_auc | ci95_low  | ci95_high   | bootstrap_resamples |
| ----------------------------------------- | ---------- | ---------------------- | ------------------------ | --------- | ----------- | ------------------- |
| logistic_l2_zscore_minus_absolute         | 36         | -0.0252268             | -0.00255102              | -0.09893  | 0.0506023   | 2000                |
| logistic_l2_zscore_minus_mean_subtraction | 36         | -0.0637755             | -0.0331633               | -0.11735  | -0.00906675 | 2000                |
| linear_svm_zscore_minus_absolute          | 36         | -0.0367063             | 0.00510204               | -0.112408 | 0.0396861   | 2000                |
| linear_svm_zscore_minus_mean_subtraction  | 36         | -0.059949              | -0.0204082               | -0.11452  | -0.00707908 | 2000                |
