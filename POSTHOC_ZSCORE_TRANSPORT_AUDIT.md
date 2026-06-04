# Post-Hoc Z-Scoring Transport Observation — Validity Audit (Task 1)

**Goal:** Determine, using only already-generated code, reports, and result files,
whether the MAT→STEW z-scoring macro subject AUC = **0.682823** is technically
interpretable as a *hypothesis-generating* observation (not as confirmed evidence).

**Sources audited (no new models run):**
- `scripts/run_stew_sensitivity_and_transfer.py` (`transfer()`, lines ~287–321;
  `apply_baseline_calibration` in `src/eeg_cogstates/theory_validation.py`;
  `subject_aucs`/`boot_ci`/`paired_delta`).
- `results/stew_sensitivity/transport_compatible_feature_spec.yaml` (frozen spec).
- `results/stew_sensitivity/mat_to_stew_transfer_metrics.csv`,
  `mat_to_stew_transfer_bootstrap.csv`, `mat_to_stew_transfer_predictions.csv`.
- `TRANSFER_FEATURE_COMPATIBILITY_AUDIT.md`, `MAT_TO_STEW_EXPLORATORY_TRANSFER_RESULTS.md`.

## Audit findings

### 1. Did z-scoring use only unlabeled STEW baseline-calibration data?
**Yes.** In `transfer()`, `x_stew = apply_baseline_calibration(stew, stew_eval,
stew_calib, TRANSPORT_COLS, "zscore")`. `stew_calib = segment_type=="calibration"`
= first 30 s of each subject's `_lo` (rest) file. `apply_baseline_calibration`
derives per-subject mean/std via `_subject_stats`, which reads **only `calib_mask`
rows** and computes `nanmean`/`nanstd` of features — **no labels involved**. The
z-score of each scored row is `(x − baseline_mean)/baseline_std`. Calibration
(0–30 s `_lo`), scored-rest (30–60 s `_lo`), and scored-task (0–30 s `_hi`) are
disjoint (14/14/14), so baseline stats come from a non-overlapping segment.

### 2. Were STEW workload labels excluded from fitting/preprocessing/selection/tuning?
**Yes.** `model.fit(xtr, y_mat)` uses **MAT** labels only. `y_stew` is used solely
to *score* (`roc_auc_score`) after prediction, never to fit. Imputer/scaler are fit
on MAT. Features were frozen in the YAML **before** results (no STEW-driven feature
selection). C is fixed at 1.0 (no tuning, no inner CV). Calibration mode is not
chosen by any STEW-label criterion — all three modes are computed and reported.

### 3. Was MAT-only fitting preserved for imputation, scaling, and the L2 model?
**Yes.** `imp.fit_transform(x_mat)`, `sc.fit_transform(...x_mat...)`,
`model.fit(xtr, y_mat)` all fit on MAT. STEW is only `transform`ed and scored via
`decision_function`. Identical for all three calibration modes.

### 4. Were the 96 frozen unit-compatible transport features used for z-scoring too?
**Yes.** The `for cal in ["absolute","mean_subtraction","zscore"]` loop passes the
same `TRANSPORT_COLS` (asserted length 96) to every mode. Z-scoring used the exact
same frozen scale/offset-invariant feature subset as mean subtraction — not a
different or expanded feature space.

### 5. Were subject-level macro AUC and bootstrap aligned correctly?
**Yes.** `subject_aucs` groups predictions by `subject_id` and computes
`roc_auc_score` per subject (each STEW subject has 14 rest + 14 task eval rows →
both classes present). Macro mean = `np.mean(subject_aucs)`; `boot_ci` resamples
the 48 subject AUCs. The z-scoring value (0.682823) was produced by the **identical**
metric/bootstrap code used for absolute (0.472045) and mean subtraction (0.447598).

### 6. Any leakage, coding bug, metric inconsistency, or post-result feature change?
**None affecting z-scoring.**
- *Leakage:* none — STEW labels never reach fitting; z-score uses only unlabeled,
  non-overlapping baseline. Target-domain standardization from unlabeled baseline
  is standard domain adaptation, not leakage.
- *Coding bug:* the run ended with a `NameError` on `summary.json` assembly, which
  occurs **after** all transfer CSVs are written; it did not touch feature
  extraction, fitting, or the z-scoring metric. (Fixed for reproducibility.)
- *Metric inconsistency:* none — same macro-subject-AUC + 2000× subject bootstrap.
- *Post-result feature change:* none — the 96 features were frozen in the YAML
  before any transfer result was viewed; z-scoring reused them unchanged.

## Critical framing (why this is hypothesis-generating, not confirmatory)

Z-scoring was a **predeclared secondary diagnostic** (in `CALIBRATION_MODES` from
the outset), so its *computation* is clean and predeclared. What is **post-hoc** is
the human **elevation/interpretation** of z-scoring as a method of interest *after*
seeing it outperform the locked candidate on STEW. The predeclared transport
hypothesis was **mean subtraction**, and it **failed** (0.448, below chance;
paired vs absolute CI [−0.070, +0.023] includes zero). Therefore the z-scoring
result:
- is **technically valid** as a clean, leakage-free, predeclared-computation
  observation, and
- is **not** confirmed/replicated/successful transport evidence — it is a **new
  hypothesis** that must be frozen and tested on an untouched dataset.

## VERDICT

```
ZSCORE_OBSERVATION_VALID_FOR_NEW_HYPOTHESIS_GENERATION
```
