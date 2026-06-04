# MAT → STEW Exploratory Transport Stress Test Results (Phase 3)

**Exploratory cross-task / cross-device transport only.** Not replication, not
confirmation. Frozen pipeline per `TRANSFER_FEATURE_COMPATIBILITY_AUDIT.md` and
`results/stew_sensitivity/transport_compatible_feature_spec.yaml`, defined before
any transfer result was viewed.

## Why this pipeline differs from within-STEW

MAT amplitudes are µV; STEW are undocumented raw 16-bit Emotiv counts (no verified
unit conversion exists — archive, IEEE page, and publication all checked).
Therefore transport uses **only the 96 features proven invariant under
`x → a·x + b` (a>0)** — `skew, kurtosis, hjorth_mobility, hjorth_complexity,
spectral_entropy, band_rel_{δ,θ,α,β}, ratio_{θ/α,β/α,θ/β}` × 8 channels.
Amplitude/variance/abs-power/Hjorth-activity **and shannon entropy** are excluded.

## Design (as predeclared)

- Train on **transport-compatible MAT only** (500→128 Hz anti-aliased resample,
  36 subjects); test on **official STEW only** (48 subjects, native 128 Hz).
- 8 locked channels; 96 invariant features; L2 logistic regression (C=1.0).
- Imputer + scaler + model fit on **MAT training data only**.
- Target (STEW) per-subject calibration computed from **unlabeled STEW baseline**
  (first 30 s of `_lo`) only. **STEW workload labels never used** for training,
  selection, preprocessing, or tuning.
- Direction: MAT→STEW only (predeclared, because MAT selected the candidate).
- Primary unit: macro subject-level ROC-AUC.

## Results (MAT128 → STEW, logistic L2)

| Calibration | Pooled window AUC | Macro subject **mean** AUC | Median | Subject SD | 95% CI |
|---|---|---|---|---|---|
| absolute | 0.475508 | **0.472045** | 0.528061 | 0.275176 | [0.396359, 0.547837] |
| **mean_subtraction** (candidate) | 0.432057 | **0.447598** | 0.446429 | 0.273269 | [0.374575, 0.525308] |
| zscore (diagnostic only) | 0.692215 | **0.682823** | 0.727041 | 0.232065 | [0.614150, 0.747462] |

### Paired subject-bootstrap deltas (2000 resamples)

| Comparison | Mean Δ | Median Δ | 95% CI |
|---|---|---|---|
| mean_subtraction − absolute | −0.024447 | −0.015306 | **[−0.070057, +0.022646]** (includes 0) |
| mean_subtraction − zscore | −0.235225 | −0.178571 | [−0.335462, −0.140614] |

## Interpretation (predeclared discipline)

- **The locked candidate (mean subtraction) FAILS to transport.** Macro subject
  AUC = **0.448**, **below chance**, with a 95% CI ([0.375, 0.525]) straddling /
  below 0.5.
- Mean subtraction is **not** better than absolute in transport — the paired CI
  crosses zero ([−0.070, +0.023]); the point estimate is slightly negative.
- Only **z-scoring** transports above chance (0.683), but z-scoring is a
  **secondary diagnostic, not the candidate method**. Per the frozen protocol the
  candidate is **not** switched because transport favors z-scoring. (Mechanistic
  note: of the three modes, only z-scoring removes per-subject multiplicative
  scale, which plausibly matters most under cross-device shift — but this is a
  diagnostic observation, not a confirmatory result, and is not pursued here.)
- Failure is reported, not hidden. The locked baseline-relative (mean-subtraction)
  hypothesis **does not transport** from MAT to official STEW under the frozen,
  unit-invariant, exploratory pipeline.

Outputs: `results/stew_sensitivity/mat_to_stew_transfer_metrics.csv`,
`mat_to_stew_transfer_bootstrap.csv`, `mat_to_stew_transfer_predictions.csv`,
`figures/mat_to_stew_transfer_macro_auc.png`.
