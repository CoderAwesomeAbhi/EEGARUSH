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
