# MAT Existing Permutation Unit Audit

Verdict: `pooled_window_auc`

The prior full-pipeline null in `scripts/rebuild_mat_from_raw.py` aggregated all held-out permuted labels and scores into `y_all` and `score_all`, then computed `roc_auc_score(y_all, score_all)`. That is a pooled window-level ROC-AUC statistic, not macro subject-level mean ROC-AUC.

Therefore, the earlier permutation result does not validate the macro subject-level primary statistic requested in this final gate.
