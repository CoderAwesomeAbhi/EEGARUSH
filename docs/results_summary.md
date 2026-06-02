# Results Summary

This document summarizes the generated outputs used by the manuscript.

## Feature Table

- Subjects: 36
- Windows: 4,266
- Rest windows: 3,186
- Workload windows: 1,080
- Columns in feature table: 812

## Leave-One-Subject-Out Metrics

| Model | Accuracy | Sensitivity | Specificity | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| SVM RBF | 0.734 | 0.567 | 0.791 | 0.519 | 0.796 | 0.545 |
| Gradient Boosting | 0.727 | 0.453 | 0.820 | 0.456 | 0.764 | 0.426 |
| Logistic Regression | 0.719 | 0.631 | 0.749 | 0.533 | 0.763 | 0.466 |
| XGBoost | 0.723 | 0.483 | 0.805 | 0.469 | 0.761 | 0.430 |
| Random Forest | 0.741 | 0.173 | 0.934 | 0.253 | 0.760 | 0.451 |

## Grouped Subject-Holdout Metrics

| Model | Accuracy | Sensitivity | Specificity | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.716 | 0.511 | 0.789 | 0.486 | 0.750 | 0.433 |
| Random Forest | 0.735 | 0.152 | 0.942 | 0.231 | 0.749 | 0.451 |
| SVM RBF | 0.696 | 0.619 | 0.723 | 0.516 | 0.734 | 0.490 |
| Gradient Boosting | 0.694 | 0.348 | 0.817 | 0.374 | 0.722 | 0.435 |
| Logistic Regression | 0.651 | 0.585 | 0.675 | 0.468 | 0.680 | 0.420 |

## Top Significant Paired Features

| Feature | Workload-Rest Difference | Absolute dz | t-test q | Wilcoxon q |
|---|---:|---:|---:|---:|
| stat_O1_skew | -0.108 | 1.001 | 0.000564 | 0.0000506 |
| band_abs_Cz_beta | -3.321 | 0.933 | 0.000564 | 0.00000593 |
| ratio_Pz_beta_alpha | 0.245 | 0.929 | 0.000564 | 0.0000919 |
| stat_P3_ptp | -9.141 | 0.929 | 0.000564 | 0.0000919 |
| band_abs_T5_alpha | -7.994 | 0.886 | 0.000664 | 0.0000719 |
| ratio_P3_beta_alpha | 0.236 | 0.880 | 0.000664 | 0.000117 |
| band_rel_Pz_gamma | 0.00491 | 0.869 | 0.000664 | 0.0000558 |
| stat_P3_skew | -0.0815 | 0.859 | 0.000664 | 0.000361 |
| stat_O1_ptp | -10.983 | 0.858 | 0.000664 | 0.000253 |
| band_abs_F4_beta | -2.469 | 0.853 | 0.000664 | 0.000142 |
