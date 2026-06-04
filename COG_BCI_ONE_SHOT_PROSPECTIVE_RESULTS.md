# COG-BCI One-Shot Prospective Test — Results

**This is the single, consumed, one-shot prospective evaluation of the post-hoc
z-scoring transport hypothesis.** It was run exactly once, under the frozen
executable config (`COG_BCI_EXECUTABLE_CONFIG.yaml`,
`COG_BCI_EXECUTABLE_CONFIG_FROZEN_READY_FOR_SINGLE_PROSPECTIVE_RUN`), after all
pre-run gates passed. The result is reported exactly as observed, regardless of
direction. **The test has not been and will not be rerun.**

- Run marker: `results/cog_bci_one_shot/RUN_MARKER.json`
  (`executed_utc 2026-06-04T19:26:22Z`, `single_one_shot: true`).
- Frozen materials verified at run start: config SHA-256
  `0273e308…c263c0`, script SHA-256 `679d183f…a3ca2` (match
  `results/cog_bci_provenance/cog_bci_frozen_run_materials_checksums.json`).
- Raw results: `results/cog_bci_one_shot/cog_bci_primary_results.json`.

## Evidence roles (locked, honest framing)

- **MAT and STEW were development evidence only.** They generated the hypothesis and
  may not count as prospective validation.
- **COG-BCI was the single untouched prospective test dataset.**
- **Mean-subtraction transfer had already failed** (MAT→STEW mean-subtraction macro
  subject AUC 0.447598, below chance; not better than absolute).
- **z-scoring was elevated only *after* the STEW result** (MAT→STEW z-scoring 0.682823,
  a post-hoc secondary diagnostic), and was tested prospectively **exactly once here**.
- **No alternative target, task, session, model, feature set, channel set, resampling,
  endpoint, or comparator was tried after COG-BCI outcomes were accessed.** N-back was
  not run; alternative sessions were not run.

## Frozen configuration actually executed

- Train: MAT only (rest vs arithmetic), imputer + scaler + L2 logistic fit on MAT only.
- Test: COG-BCI `ses-S1`, eyes-open rest (`RS_End_EO`, first 30 s, label 0) vs MATB
  difficult (`MATBdiff`, first 30 s, label 1); calibration baseline `RS_Beg_EO`
  (first 30 s, unlabeled).
- Sampling: both MAT and COG-BCI represented at **128 Hz** via
  `scipy.signal.resample_poly(up=32, down=125)` before feature extraction.
- Channels: 8 (F3,F4,F7,F8,O1,O2,T7→T3,T8→T4). Features: frozen 96. Windows: 4 s,
  50 % overlap, 14 rest + 14 task per subject. Subjects: all 29 (no pre-outcome
  exclusion). Model: L2 logistic (C=1.0, liblinear, balanced, max_iter=5000).

## Primary endpoint — macro subject-level ROC-AUC (MAT→COG-BCI transport)

| Method | Role | Macro subject AUC | Subject SD | Subject-bootstrap 95% CI | Above chance? |
|---|---|---|---|---|---|
| **z-scoring** | **primary** | **0.435961** | 0.200017 | [0.359602, 0.509338] | **No** |
| mean subtraction | primary comparator | 0.392153 | 0.234233 | [0.310697, 0.476258] | No |
| absolute | secondary comparator | 0.367875 | 0.242358 | [0.286590, 0.456725] | No |

n = 29 target subjects for every method.

## Primary superiority test — paired subject-bootstrap ΔAUC

| Comparison | Mean Δ | Median Δ | 95% CI | Excludes zero positively? |
|---|---|---|---|---|
| **z-scoring − mean subtraction** (primary) | +0.043807 | +0.010204 | **[−0.076359, +0.175233]** | **No** |
| z-scoring − absolute (secondary) | +0.068086 | +0.091837 | [−0.060173, +0.190715] | No |

## Classification (locked rule)

- **Full success** requires z-scoring **above chance** AND paired 95 % CI
  (z − mean subtraction) excluding zero positively.
- **Partial** requires z-scoring above chance but superiority CI including zero.
- **Failure**: z-scoring **at/below chance**, or it does not outperform mean subtraction.

**Observed:** z-scoring macro subject AUC = **0.435961 < 0.5 (below chance)**, and the
primary superiority CI **[−0.076359, +0.175233] includes zero**.

➡ **FAILURE.** z-scoring is below chance, so the above-chance precondition is not met.
Although z-scoring is numerically the highest of the three methods, all three transport
below chance, and the z-scoring−mean-subtraction advantage is not statistically
distinguishable from zero.

## Interpretation (honest)

- The post-hoc z-scoring transport hypothesis **did not validate prospectively** on
  the untouched COG-BCI dataset. The MAT→STEW z-scoring elevation (0.683) **did not
  replicate**; on COG-BCI all three calibration strategies transport **below chance**.
- This is consistent with — and extends — the earlier failure of mean-subtraction
  cross-device transfer. Cross-dataset/cross-task workload **transfer** of a
  MAT-trained decoder is **not** supported, under any of absolute, mean-subtraction,
  or z-scoring calibration.
- **Unchanged and still valid:** the within-dataset findings (MAT mean-subtraction
  macro subject AUC 0.880102, p = 0.004975; within-STEW mean-subtraction 0.839498,
  p = 0.004975). These are within-dataset evidence only and are not affected by this
  prospective transfer result.
- **No rejected claim is revived** (no universal/invariant axis, no PAC/gamma/source/
  clinical, no confirmed transfer). z-scoring transport is now **prospectively
  rejected**, not merely unconfirmed.

## Note on the permutation diagnostic

The frozen one-shot script computes the locked primary endpoint and the locked primary
and secondary paired-bootstrap comparisons. The optional "also report" full-pipeline
permutation diagnostic for z-scoring was not implemented in the frozen script and was
**not** added post-hoc (modifying the checksum-locked script after freezing is
prohibited). It is not needed for the verdict: the primary endpoint (z-scoring below
chance) already determines failure under the locked rule.
