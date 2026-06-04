# EEGARUSH Scientific Checkpoint (Locked)

This document records the validated project state after the Codex audit/rebuild
phase. Read it in full before any scientific work. The operating contract and
safety rules are summarized in the repository-root `CLAUDE.md`.

## Safety Rules

- Work only on the current branch: `claude/stew-ieee-source-audit`.
- Do not edit `main.tex` or any manuscript PDF.
- Do not analyze DS007262.
- Do not use the Hugging Face/MONSTER STEW bundle for primary modeling.
- Do not search for a final confirmation dataset.
- Do not add raw data under `data/raw/` to Git.
- Do not commit, push, or create a PR yet.
- Do not revive claims about an invariant/universal workload axis, z-scoring
  superiority, PAC, gamma mechanisms, source-localized neural mechanisms,
  clinical readiness, or confirmed cross-dataset transfer.

## Locked Scientific Checkpoint

### MAT findings that are valid

- Official raw MAT EDF files were audited.
- Verified: 36 subjects, 72 EDF files, 500 Hz sampling rate.
- Verified condition mapping: `_1 = rest/background`, `_2 = arithmetic`.
- Raw durations vary; actual EDF provenance replaces simplified duration assumptions.
- A corrected balanced no-gamma MAT protocol was run with no calibration/scoring overlap.
- Each subject had equal scored-rest and scored-task window counts.
- The old z-scoring-centered theory did not survive the corrected raw reconstruction.
- Current MAT-selected candidate method: subject resting-baseline mean subtraction.
- Locked candidate model for replication: L2 logistic regression with fixed existing settings.
- Primary metric: macro subject-level ROC-AUC.
- MAT mean-subtraction macro subject AUC = 0.880102.
- Valid full-pipeline macro subject-level MAT null test: observed AUC = 0.880102,
  null mean = 0.500600, null 95% interval = [0.441057, 0.553729],
  empirical p = 0.004975 across 200 permutations.

### MAT limits

- Mean subtraction is not proven superior to absolute features in MAT because the
  paired mean-subtraction-minus-absolute 95% CI crossed zero: [-0.018566, 0.093112].
- MAT is valid within-dataset evidence only.
- No cross-dataset transfer claim is yet justified.

### STEW findings that are valid

- The Hugging Face/MONSTER STEW bundle was audited.
- It contains time-resolved signal windows: `STEW_X.npy` with shape `(28512, 14, 256)`.
- It identifies 48 subjects and binary low/high workload labels.
- It does not identify genuine resting baseline data.
- It does not contain original temporal/start-time provenance.
- It does not encode channel labels sufficiently for defensible MAT/STEW harmonization.
- Therefore the MONSTER bundle is insufficient for the baseline-relative replication test.
- Official IEEE DataPort STEW source/archive data are required next.

## Actual Current Research Question

Does subtracting each subject's resting EEG baseline produce a reproducible and
transferable workload-decoding signal across provenance-valid EEG datasets?

## Current Next Gate

Before any STEW modeling, the official IEEE DataPort STEW archive must be manually
downloaded and placed locally under:

`data/raw/stew/ieee_dataport_stew/`

After it is present, the next scientific task will be provenance audit only:

- verify source files;
- verify true rest/baseline condition;
- verify task/workload condition;
- verify subject IDs;
- verify sampling rate and channel names;
- determine whether a non-overlapping calibration/scored-rest design is possible;
- determine whether MAT/STEW channel harmonization is defensible.

No STEW models may run until that provenance audit passes.

## Known Reproducibility Gap: Generated MAT Feature Parquet

- `results/raw_rebuilt/mat_no_gamma_features.parquet` is not present in the GitHub
  checkpoint.
- It is a generated intermediate artifact required by one existing test
  (`tests/test_mat_raw_rebuild.py::test_primary_feature_file_excludes_gamma_features`).
- Its absence does not invalidate the already recorded MAT summary results, audit
  reports, or permutation output. However, it prevents a clean clone from passing
  the complete raw-rebuild test suite without first regenerating the artifact.
- It must later be regenerated from provenance-verified raw MAT data, **or** the
  testing/reproducibility design must be revised to use a committed safe fixture
  or a documented regeneration command.
- It is **not** regenerated in this setup task (no scientific models are run, and
  regeneration would require raw MAT data that is intentionally git-ignored).

## Pivot After Failed Centering Transfer (post-STEW)

Result of the exploratory within-STEW sensitivity + MAT→STEW transport stage
(`STEW_SENSITIVITY_AND_TRANSFER_DECISION.md`,
`MAT_TO_STEW_EXPLORATORY_TRANSFER_RESULTS.md`,
`POSTHOC_ZSCORE_TRANSPORT_AUDIT.md`).

### Proven

- The **mean-subtraction** configuration decodes workload **above chance within
  MAT** (macro subject AUC 0.880102, permutation p = 0.004975, 500 Hz) **and
  within STEW** (macro subject AUC 0.839498, permutation p = 0.004975, 128 Hz).

### Rejected

- **Mean subtraction as a supported cross-device transfer method.** The
  predeclared MAT→STEW mean-subtraction transport **failed** (macro subject AUC
  0.447598, below chance; vs absolute 0.472045; paired CI [−0.070, +0.023]
  includes zero).
- Any claim of **confirmed generalizable workload transfer**.
- Any **invariant / universal workload axis** claim (still rejected).

### Newly generated hypothesis (post-hoc, frozen for prospective test)

- Across device/task domains with severe unit and baseline-scale differences,
  **subject-level baseline standardization (z-scoring) may transport better than
  baseline centering (mean subtraction) alone**. Observed only as a predeclared
  *secondary diagnostic* (MAT→STEW z-scoring macro subject AUC 0.682823) and
  **elevated post-hoc** after viewing transfer results.
- This is **not** confirmed/replicated/successful transport evidence. The z-scoring
  computation is technically valid and leakage-free
  (`ZSCORE_OBSERVATION_VALID_FOR_NEW_HYPOTHESIS_GENERATION`), but because it was
  generated after observing MAT→STEW results it **requires an untouched
  prospective evaluation dataset**. MAT and STEW count only as development
  evidence; neither may serve as prospective validation.
- The frozen prospective plan is `REVISED_ZSCORE_PROSPECTIVE_EVALUATION_PROTOCOL.md`.
- A **metadata-only** screen for an untouched, baseline-bearing, construct-matched
  dataset is now sanctioned (`UNTOUCHED_DATASET_ELIGIBILITY_SCREEN.md`); **no
  signal/label download or analysis** of any candidate is permitted until a
  dataset is selected and a fresh provenance audit passes.

### COG-BCI prospective target (conditionally selected, pre-download)

- **COG-BCI (Hinss et al. 2023; Zenodo 10.5281/zenodo.6874128)** is selected as the
  single prospective evaluation dataset, **conditionally** — metadata eligibility
  passed (`COG_BCI_METADATA_ELIGIBILITY_AUDIT.md` →
  `COG_BCI_METADATA_ELIGIBLE_FOR_PRETEST_LOCK`), but selection remains pending a
  later **raw-data provenance audit** that must pass before any modeling.
- **No COG-BCI signal files may be downloaded or analyzed** until the one-shot test
  protocol (`COG_BCI_ONE_SHOT_PROSPECTIVE_TEST_PROTOCOL.md`) is committed and the
  provenance audit passes.
- The **primary paradigm is predeclared as MATB** (highest level = MATB
  **difficult**), chosen by construct match, **not** after outcomes.
- **N-back cannot influence the primary verdict** — it is secondary/exploratory only.
- Primary endpoint: macro subject-level ROC-AUC for **z-scoring** on **eyes-open
  rest vs MATB-difficult**; primary comparison z-scoring − mean subtraction (paired
  subject-bootstrap CI). Eyes-closed rest is excluded from the primary model.
- Pre-download decision: `COG_BCI_PRE_DOWNLOAD_DECISION.md`.

### COG-BCI status update — source verified, sampling corrected, executable freeze ready

- Source provenance audit **passed** (29/29 subjects readable, 8 locked channels,
  500 Hz, eyes-open rest + MATB-difficult present; `COG_BCI_SOURCE_PROVENANCE_*`).
- Executable freeze first returned `BLOCKED_PROTOCOL_AMBIGUITY` because the
  protocol's "500 Hz native — no resample" statement conflicted with the frozen
  MAT→STEW transport pipeline, which trained the source model at **128 Hz**.
- **Sampling representation corrected** (pre-outcome, consistency-only):
  `COG_BCI_PROTOCOL_AMENDMENT_SAMPLING_REPRESENTATION.md`. Both source MAT **and**
  target COG-BCI are resampled 500 → 128 Hz via `scipy.signal.resample_poly`,
  `up=32`, `down=125`, before extracting the frozen **96** transport features. The
  96 features are scale/offset-invariant but **not** sampling-rate-invariant.
  Native-500-Hz COG-BCI analysis is forbidden from the primary verdict.
- All 29 subjects' exact locked `ses-S1` inputs (`RS_Beg_EO`, `RS_End_EO`,
  `MATBdiff` × `.set`/`.fdt`) are materialized locally and SHA-256-recorded
  (`results/cog_bci_provenance/cog_bci_execution_input_hash_manifest.csv`).
- Executable config was **frozen ready for a single prospective run**
  (`COG_BCI_EXECUTABLE_CONFIG.yaml`, `COG_BCI_RUN_ONCE_CHECKLIST.md`,
  `COG_BCI_EXECUTABLE_FREEZE_DECISION.md`,
  `scripts/run_cog_bci_one_shot_prospective.py`).

### COG-BCI prospective test — executed once, FAILED (final)

- The one-shot test was run **exactly once** (2026-06-04) and is now **consumed**.
  See `COG_BCI_ONE_SHOT_PROSPECTIVE_RESULTS.md`,
  `COG_BCI_FINAL_PROJECT_DECISION.md`, `results/cog_bci_one_shot/`.
- MAT→COG-BCI transport macro subject-level ROC-AUC: **z-scoring 0.435961**
  (below chance), mean subtraction 0.392153, absolute 0.367875 (n = 29).
- Primary superiority ΔAUC (z-scoring − mean subtraction): mean +0.043807, paired
  subject-bootstrap 95% CI **[−0.076359, +0.175233]** (includes zero).
- **Classification: FAILURE** — z-scoring is at/below chance. The MAT→STEW z-scoring
  elevation did **not** replicate; all three calibration strategies transport below
  chance. Verdict: `PROSPECTIVE_ZSCORE_TRANSPORT_FAILED_WRITE_NEGATIVE_METHODS_PAPER`.
- z-scoring transport is now **prospectively rejected**. Cross-device/cross-task
  **transfer** of the MAT-trained decoder is **not supported** (extends the earlier
  mean-subtraction transfer failure). Within-dataset MAT/STEW results are unchanged
  and remain within-dataset evidence only. No rejected claim is revived.
