# Full-Pipeline Null Audit

Verdict: `not_completed_computationally_blocked`.

I attempted full LOSO retraining permutation runs at 100, 20, and 5 permutations. Even the 5-permutation run did not complete after several CPU minutes in this environment before reaching downstream audit stages.
The prior score-shuffling control therefore remains insufficient as a full pipeline null, and the manuscript should not present it as validating the complete modeling pipeline.

## Required Follow-Up

Run the full retraining permutation audit on a less constrained machine, or optimize the implementation to persist per-permutation results incrementally.

## Previous Score-Shuffling Control

| dataset | calibration | model       | observed_auc | n_permutations_requested | n_permutations_valid | permutation_auc_mean | permutation_auc_sd | permutation_p_value | note                                                                                                            |
| ------- | ----------- | ----------- | ------------ | ------------------------ | -------------------- | -------------------- | ------------------ | ------------------- | --------------------------------------------------------------------------------------------------------------- |
| MAT     | zscore      | logistic_l2 | 0.865995     | 1000                     | 1000                 | 0.508327             | 0.0107226          | 0.000999001         | Labels were permuted within subject on nested out-of-fold scores; models were not retrained for each null draw. |
| STEW    | zscore      | linear_svm  | 0.791896     | 1000                     | 1000                 | 0.500369             | 0.0141408          | 0.000999001         | Labels were permuted within subject on nested out-of-fold scores; models were not retrained for each null draw. |
