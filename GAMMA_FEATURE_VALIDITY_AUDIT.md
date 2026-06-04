# Gamma Feature Validity Audit

Code audit: gamma features are defined as 30--45 Hz absolute and relative bandpower (`band_abs_{ch}_gamma`, `band_rel_{ch}_gamma`).
MAT raw acquisition/header metadata were not available locally, STEW is a 128 Hz Emotiv-derived dataset, and DS007262 is 250 Hz BrainVision. This heterogeneity prevents strong biological interpretation of gamma features across datasets.

## No-Gamma Sensitivity

| dataset | model       | calibration | feature_set  | n_features | window_auc | subject_auc_mean | subject_auc_sd | n_predictions |
| ------- | ----------- | ----------- | ------------ | ---------- | ---------- | ---------------- | -------------- | ------------- |
| MAT     | logistic_l2 | zscore      | all_200      | 200        | 0.865995   | 0.874355         | 0.122731       | 3186          |
| MAT     | logistic_l2 | zscore      | no_gamma_184 | 184        | 0.873199   | 0.879315         | 0.102848       | 3186          |
| STEW    | linear_svm  | zscore      | all_200      | 200        | 0.791896   | 0.810585         | 0.185704       | 1519          |
| STEW    | linear_svm  | zscore      | no_gamma_184 | 184        | 0.794195   | 0.814947         | 0.179087       | 1519          |

## Recommendation

Gamma features may remain as predictive engineering features only if sensitivity results are reported. They should not be interpreted biologically without dataset-compatible acquisition/filter audits and source/artifact controls.
