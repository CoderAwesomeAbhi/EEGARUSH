# Calibration Improvement Inference

Subject-level ROC-AUC was used as the inferential unit. For each dataset/model pair, subject AUCs were paired across calibration methods and bootstrapped by resampling subjects with replacement.

| dataset | model       | comparison                    | n_subjects | mean_delta_subject_auc | median_delta_subject_auc | ci95_low   | ci95_high | bootstrap_resamples |
| ------- | ----------- | ----------------------------- | ---------- | ---------------------- | ------------------------ | ---------- | --------- | ------------------- |
| MAT     | logistic_l2 | zscore_minus_absolute         | 36         | 0.0823476              | 0.0658333                | 0.0265124  | 0.139735  | 2000                |
| MAT     | logistic_l2 | zscore_minus_mean_subtraction | 36         | 0.0305725              | 0.0216667                | -0.0149991 | 0.0853988 | 2000                |
| MAT     | linear_svm  | zscore_minus_absolute         | 36         | 0.076783               | 0.0575                   | 0.0190942  | 0.138514  | 2000                |
| MAT     | linear_svm  | zscore_minus_mean_subtraction | 36         | 0.0256604              | 0.0108333                | -0.0230703 | 0.0814025 | 2000                |
| STEW    | logistic_l2 | zscore_minus_absolute         | 48         | 0.0693545              | 0.0430554                | 0.0118078  | 0.138027  | 2000                |
| STEW    | logistic_l2 | zscore_minus_mean_subtraction | 48         | 0.0648068              | 0.0485052                | 0.0120201  | 0.122942  | 2000                |
| STEW    | linear_svm  | zscore_minus_absolute         | 48         | 0.0757837              | 0.0550505                | 0.0217122  | 0.139771  | 2000                |
| STEW    | linear_svm  | zscore_minus_mean_subtraction | 48         | 0.059061               | 0.0618056                | 0.00538003 | 0.117521  | 2000                |

Numerical AUC differences should be interpreted through these paired subject-cluster intervals, not as window-level independent evidence.
