# MAT Macro Subject Null Results

- Configuration: corrected balanced raw MAT / no-gamma 184 / `mean_subtraction` / `logistic_l2`.
- Statistic: macro subject-level mean ROC-AUC.
- Label permutation: labels are permuted within subject before rerunning LOSO model fitting and evaluation; calibration masks, evaluation masks, imputation, scaling, and model fitting are recomputed within each permutation fold.
- Completed permutations: `200`.
- Observed macro subject-level mean ROC-AUC: `0.880102`.
- Null mean: `0.500600`.
- Null 95% interval: `[0.441057, 0.553729]`.
- Empirical p-value: `0.004975`.
- Runtime seconds: `474.20`.
