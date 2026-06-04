# STEW Within-Dataset Exploratory Sensitivity Results (Phase 2)

**Exploratory cross-task / cross-device sensitivity only.** STEW is non-comparable;
this is **not** replication or confirmation. Predeclared by
`STEW_EXPLORATORY_SENSITIVITY_PROTOCOL_BEFORE_MODELING.md`. Within-STEW analysis
trains and tests inside the same device/unit system, so it uses the full locked
`no_gamma_184` 8-channel family (see `TRANSFER_FEATURE_COMPATIBILITY_AUDIT.md`).

## Design (as predeclared)

- Official IEEE STEW source, 48 subjects, native 128 Hz.
- Locked 8 channels: F3, F4, F7, F8, O1, O2, T3↔T7, T4↔T8.
- Balanced per subject: calibration = first 30 s of `_lo`; scored-rest = 30–60 s
  of `_lo`; scored-task = first 30 s of `_hi`. 4 s / 50 % windows ⇒ **14/14/14**
  windows per subject (2016 windows total). No subject excluded (all yielded full
  budgets; no QC failure).
- Model: L2 logistic regression (C=1.0), leave-one-subject-out.
- Primary unit: macro subject-level ROC-AUC.

## Results (logistic L2)

| Calibration | Pooled window AUC | Macro subject **mean** AUC | Macro subject median | Subject SD | Subject-bootstrap 95% CI |
|---|---|---|---|---|---|
| absolute | 0.800352 | **0.815795** | 0.869898 | 0.197616 | [0.758588, 0.867350] |
| **mean_subtraction** (candidate) | 0.804063 | **0.839498** | 0.905612 | 0.178023 | [0.788478, 0.887970] |
| zscore (diagnostic only) | 0.890714 | **0.898703** | 0.928571 | 0.127987 | [0.857355, 0.931973] |

### Paired subject-bootstrap deltas (2000 resamples)

| Comparison | Mean Δ | Median Δ | 95% CI |
|---|---|---|---|
| mean_subtraction − absolute | +0.023703 | +0.010204 | **[−0.013180, +0.060717]** (includes 0) |
| mean_subtraction − zscore | −0.059205 | −0.020408 | [−0.104698, −0.017001] |

### Full-pipeline macro subject-level permutation (mean_subtraction, 200 perms)

- Observed macro subject AUC: **0.839498**
- Null mean: 0.502071; null 95% interval: [0.458830, 0.539908]
- **Empirical p = 0.004975** (200 within-subject label permutations, full LOSO refit each).

## Interpretation (predeclared discipline)

- The MAT-selected **mean-subtraction** method gives **credible above-chance
  within-STEW decoding** (macro subject AUC 0.840; permutation p ≈ 0.005; CI well
  above 0.5). This is **exploratory cross-task/cross-device sensitivity support**,
  not replication or confirmation.
- Consistent with the MAT finding, mean subtraction is **not** proven superior to
  absolute features here — the paired CI crosses zero ([−0.013, +0.061]).
- z-scoring scores highest within STEW, but it is a **secondary diagnostic only**;
  per protocol the candidate method is not switched because STEW favors z-scoring.
- The MAT→STEW transport stress test is reported separately in
  `MAT_TO_STEW_EXPLORATORY_TRANSFER_RESULTS.md`.

Outputs: `results/stew_sensitivity/stew_window_manifest.csv`,
`stew_within_metrics.csv`, `stew_subject_metrics.csv`, `stew_paired_bootstrap.csv`,
`stew_macro_subject_permutation.csv`, `stew_permutation_summary.csv`,
`figures/stew_within_macro_auc.png`.
