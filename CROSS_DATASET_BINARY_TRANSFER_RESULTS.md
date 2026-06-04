# Cross-Dataset Binary Transfer Results

These experiments train on one exploratory dataset and test on the other with the same 200-feature intersection. Target workload labels are not used for fitting or calibration-method selection.

Important caveat: MAT raw EDF headers remain unavailable locally, so this is a feature-table-level transfer audit rather than a raw-provenance-complete result.

## Transfer Metrics

| source_dataset | target_dataset | model       | calibration      | target_mode                   | window_auc | subject_auc_mean | subject_auc_sd | n_subjects | n_predictions |
| -------------- | -------------- | ----------- | ---------------- | ----------------------------- | ---------- | ---------------- | -------------- | ---------- | ------------- |
| MAT            | STEW           | logistic_l2 | absolute         | zero_shot_absolute            | 0.522111   | 0.51728          | 0.239199       | 48         | 1519          |
| MAT            | STEW           | logistic_l2 | mean_subtraction | zero_shot_absolute            | 0.498642   | 0.470954         | 0.278709       | 48         | 1519          |
| MAT            | STEW           | logistic_l2 | mean_subtraction | unlabeled_baseline_calibrated | 0.514147   | 0.470954         | 0.278709       | 48         | 1519          |
| MAT            | STEW           | logistic_l2 | zscore           | zero_shot_absolute            | 0.532238   | 0.555046         | 0.277512       | 48         | 1519          |
| MAT            | STEW           | logistic_l2 | zscore           | unlabeled_baseline_calibrated | 0.60148    | 0.608549         | 0.236033       | 48         | 1519          |
| STEW           | MAT            | logistic_l2 | absolute         | zero_shot_absolute            | 0.618885   | 0.63956          | 0.243775       | 36         | 3186          |
| STEW           | MAT            | logistic_l2 | mean_subtraction | zero_shot_absolute            | 0.618191   | 0.640709         | 0.220438       | 36         | 3186          |
| STEW           | MAT            | logistic_l2 | mean_subtraction | unlabeled_baseline_calibrated | 0.632215   | 0.640709         | 0.220438       | 36         | 3186          |
| STEW           | MAT            | logistic_l2 | zscore           | zero_shot_absolute            | 0.516409   | 0.520144         | 0.245206       | 36         | 3186          |
| STEW           | MAT            | logistic_l2 | zscore           | unlabeled_baseline_calibrated | 0.649916   | 0.650657         | 0.218787       | 36         | 3186          |

## Within-Dataset LOSO Reference

| dataset | model       | calibration      | window_auc | subject_auc_mean | n_subjects |
| ------- | ----------- | ---------------- | ---------- | ---------------- | ---------- |
| MAT     | logistic_l2 | absolute         | 0.736025   | 0.792007         | 36         |
| MAT     | linear_svm  | absolute         | 0.736328   | 0.788841         | 36         |
| MAT     | logistic_l2 | mean_subtraction | 0.826001   | 0.843782         | 36         |
| MAT     | linear_svm  | mean_subtraction | 0.820951   | 0.839964         | 36         |
| MAT     | logistic_l2 | zscore           | 0.865995   | 0.874355         | 36         |
| MAT     | linear_svm  | zscore           | 0.858511   | 0.865624         | 36         |
| STEW    | logistic_l2 | absolute         | 0.723308   | 0.737614         | 48         |
| STEW    | linear_svm  | absolute         | 0.722913   | 0.734801         | 48         |
| STEW    | logistic_l2 | mean_subtraction | 0.694381   | 0.742161         | 48         |
| STEW    | linear_svm  | mean_subtraction | 0.699872   | 0.751524         | 48         |
| STEW    | logistic_l2 | zscore           | 0.791446   | 0.806968         | 48         |
| STEW    | linear_svm  | zscore           | 0.791896   | 0.810585         | 48         |
