# STEW Exploratory Sensitivity Protocol (Predeclared, Before Modeling)

**Status:** Written and frozen **before** any STEW model performance was computed.
No model metrics were inspected when authoring this protocol.

## Nature of this analysis (non-negotiable framing)

STEW (official IEEE DataPort source) is a **non-comparable, exploratory
cross-task / cross-device sensitivity dataset**. It is **not** a strict
same-paradigm replication and **not** an untouched confirmation. Per
`STEW_IEEE_CONSTRUCT_MATCH_DECISION.md` the verdict is
`STEW_IEEE_USABLE_ONLY_AS_NONCOMPARABLE_SENSITIVITY`. Any positive result is
"exploratory cross-task sensitivity support," never "replication" or
"confirmation."

## Primary question

Does the MAT-selected **resting-baseline mean-subtraction** method retain any
credible advantage or above-chance decoding in official-source STEW, and does a
**locked MAT→STEW transfer** test show transportability across substantial domain
shift (different task, device, sampling rate, montage, reference)?

## Locked analysis choices

- **Primary candidate method:** subject resting-baseline **mean subtraction**.
- **Primary comparator:** **absolute** features (no per-subject calibration).
- **Secondary diagnostic only:** **z-scoring** (rest mean/std). Not a candidate;
  reported for diagnosis only.
- **Primary model:** L2 logistic regression with the existing fixed parameters
  (`make_model("logistic_l2", C=1.0)`: `solver=liblinear`, `class_weight=balanced`,
  `max_iter=5000`). No hyperparameter sweep.
- **Primary statistical unit:** **macro subject-level ROC-AUC**.
- **Primary channel set:** the locked **eight** channels only —
  `F3, F4, F7, F8, O1, O2, T3(MAT)↔T7(STEW), T4(MAT)↔T8(STEW)`.
  `P7/P8/T5/T6` are **excluded** from the primary analysis (mentioned only as
  unused harmonizable channels in limitations).
- **Primary features:** **no-gamma only** (`no_gamma_184`: the 184 per-channel
  features from `eeg_cogstates`, gamma band excluded). Nyquist at 128 Hz (64 Hz)
  safely covers δ/θ/α/β.

## STEW balanced segment design

For each subject, using the official `subNN_lo.txt` (rest/no-task) and
`subNN_hi.txt` (high-workload) files:

- **calibration baseline:** first **30 s** of `_lo`;
- **scored-rest:** next non-overlapping **30 s** of `_lo` (30–60 s);
- **scored-task:** first **30 s** of `_hi`;
- **windowing:** identical to corrected balanced MAT — **4.0 s windows, 50 %
  overlap** (2.0 s step). 30 s ⇒ floor((30−4)/2)+1 = **14 windows** per segment;
- **balance requirement:** equal scored-rest and scored-task window counts for
  every subject (14 and 14);
- **calibration/scoring separation:** calibration windows (0–30 s of `_lo`) never
  overlap scored-rest windows (30–60 s of `_lo`); scored-task comes from `_hi`;
- **exclusion rule (predeclared):** a subject is excluded only for an explicit
  quality-control failure — a file that does not yield the full 14/14/14 windows,
  a non-finite/flat channel across a whole segment, or a missing `_lo`/`_hi` file.
  Any exclusion is documented **before** results are inspected.

## Statistical plan

For each calibration mode (absolute, mean_subtraction, zscore) with L2 logistic
regression under leave-one-subject-out:
- pooled window ROC-AUC;
- macro subject-level **mean** ROC-AUC (primary);
- macro subject-level **median** ROC-AUC;
- subject-level standard deviation;
- subject-bootstrap 95 % CI (2000 resamples);
- paired subject-bootstrap delta and 95 % CI:
  - mean_subtraction − absolute;
  - mean_subtraction − zscore.
- a true **full-pipeline macro subject-level permutation test** for the locked
  mean-subtraction STEW configuration, **200 permutations** initially
  (labels permuted within subject, full LOSO refit per permutation).

## Reporting discipline (predeclared)

- Report the result **even if poor**.
- Do **not** switch the candidate method because STEW favors a different one.
- Do **not** tune the method based on STEW results.
- Success is "exploratory cross-task sensitivity support," not replication.

## Transfer direction

Only **MAT→STEW** is run in this task (MAT selected the candidate method).
STEW→MAT is **not** run here. See the transfer requirements in
`TRANSFER_FEATURE_COMPATIBILITY_AUDIT.md` and
`results/stew_sensitivity/transport_compatible_feature_spec.yaml`.
