# STEW Sensitivity & Transfer Decision (Phase 4)

Inputs: `STEW_EXPLORATORY_SENSITIVITY_PROTOCOL_BEFORE_MODELING.md`,
`TRANSFER_FEATURE_COMPATIBILITY_AUDIT.md`,
`results/stew_sensitivity/transport_compatible_feature_spec.yaml`,
`STEW_EXPLORATORY_SENSITIVITY_RESULTS.md`,
`MAT_TO_STEW_EXPLORATORY_TRANSFER_RESULTS.md`, and the CSVs under
`results/stew_sensitivity/`.

## Pipeline validity (precondition)

- MAT raw EDFs downloaded and **integrity-verified**: 72/72 SHA-256 match the
  validated MAT provenance manifest byte-for-byte.
- Measurement-unit problem resolved correctly: **no verified STEW µV conversion
  exists**, so transport used only the **96 features proven invariant under
  `x→a·x+b`** (frozen before any transfer result).
- Calibration/scored-rest/scored-task non-overlap held (14/14/14 per subject);
  8-channel alignment defensible; STEW labels never used in transport training.
- The pipeline executed validly → `STEW_PIPELINE_INVALID_STOP` does **not** apply.

## Evidence summary

| Result | Value |
|---|---|
| Within-STEW absolute — macro subject AUC | 0.815795 |
| Within-STEW **mean_subtraction** — macro subject AUC | **0.839498** |
| Within-STEW zscore (diagnostic) — macro subject AUC | 0.898703 |
| Within-STEW mean_sub − absolute paired 95% CI | [−0.013180, +0.060717] (incl. 0) |
| Within-STEW mean_subtraction permutation | observed 0.839498, null mean 0.502071, 95% [0.458830, 0.539908], **p = 0.004975** |
| MAT→STEW absolute — macro subject AUC | 0.472045 |
| MAT→STEW **mean_subtraction** — macro subject AUC | **0.447598** (below chance) |
| MAT→STEW zscore (diagnostic) — macro subject AUC | 0.682823 |
| MAT→STEW mean_sub − absolute paired 95% CI | [−0.070057, +0.022646] (incl. 0) |

## Reasoning

1. **Within STEW, the candidate mean-subtraction method shows credible
   above-chance decoding** (macro subject AUC 0.840; permutation p ≈ 0.005; CI
   above 0.5). So the method does **not** fail the cross-task/device sensitivity
   test → `MAT_METHOD_FAILS_CROSS_TASK_SENSITIVITY_DOWNGRADE_PROJECT` does **not**
   apply.
2. **The locked MAT→STEW mean-subtraction transport fails**: macro subject AUC
   0.448 (below chance), and not better than absolute (paired CI crosses zero).
   The only above-chance transport comes from **z-scoring**, which is a
   **secondary diagnostic, not the candidate**, and the protocol forbids switching
   methods because transport favors it. So the locked candidate transport result
   is **not** scientifically credible →
   `MAT_TO_STEW_EXPLORATORY_TRANSPORT_SIGNAL_SURVIVES_…` does **not** apply.
3. Therefore: credible within-STEW signal for the candidate method, but transport
   of the locked method fails.

## VERDICT

```
STEW_WITHIN_DATASET_SIGNAL_ONLY_TRANSFER_FAILS_NEGATIVE_METHODS_DIRECTION
```

Mean subtraction produces credible above-chance within-STEW decoding, but the
MAT→STEW transport of the locked baseline-relative method fails.

## Required statements

- **STEW remains exploratory and non-comparable.** Different task (SIMKAP vs
  arithmetic), device (Emotiv 14ch vs 10-20 21ch), sampling rate (128 vs 500 Hz),
  reference, and undocumented amplitude units. Nothing here is a strict
  replication or untouched confirmation.
- **STEW cannot alone confirm the baseline-relative hypothesis.** The within-STEW
  signal is supportive sensitivity evidence for the method *inside* STEW only; the
  cross-dataset transport of the locked method did not survive.
- **A new untouched confirmation dataset search is NOT justified at this gate.**
  That step is reserved for the `…TRANSPORT_SIGNAL_SURVIVES…` verdict, which was
  not reached. Any eventual confirmation dataset must be **untouched** (not
  previously inspected/modeled here) and must contain a **genuine resting
  baseline**, ideally with reduced device/task/montage shift relative to MAT.

## What this does and does not change

- Does **not** revive any rejected claim (no universal axis, no z-scoring
  superiority claim, no PAC/gamma/source/clinical/confirmed-transfer claims).
- Does **not** alter the locked MAT within-dataset evidence
  (`mean-subtraction macro subject AUC = 0.880102`, permutation p = 0.004975 at
  500 Hz); the 128 Hz transport re-extraction was a separate exploratory pipeline.
- The actionable next step is **not** automatic: either accept the negative
  transport result as a bound on transportability, or (if pursued later) seek a
  genuinely construct-matched, baseline-bearing, untouched dataset with less
  device shift — to be decided by the user, not triggered here.
