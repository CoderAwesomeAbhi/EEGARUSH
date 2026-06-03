# Scientific Rescue Decision

Verdict: `EXPLORATORY_RESULTS_SALVAGEABLE_NEW_CONFIRMATION_REQUIRED`.

## Basis

- MAT feature-table calibration/scoring overlap count: `0`.
- MAT later rest scoring windows available: `2106`.
- MAT raw EDF files found locally: `0`.
- DS007262 construct verdict: `task_anchored_sensitivity_only`.

## Decision Logic

The MAT feature table supports a non-overlapping 60-second calibration split, but raw EDF header provenance is unavailable locally and should be verified before making strong acquisition-duration claims.
DS007262 is not a construct-matched resting-baseline confirmation because the analyzed event files do not contain a genuine rest/neutral baseline. The DS result is therefore a negative task-anchored sensitivity test, not a clean confirmatory test of the baseline-relative transfer hypothesis.

## Required Before Manuscript Revision

- Revise DS007262 language from confirmatory resting-baseline transfer to task-anchored sensitivity / negative external stress test.
- Demote or replace the old calibration-duration curve with the fixed-evaluation audit result.
- Report paired subject-level uncertainty for calibration-method improvements.
- Report full-pipeline permutation nulls separately from score-shuffling controls.
- Avoid biological gamma interpretation unless no-gamma sensitivity and metadata constraints are included.
- Select a new untouched construct-matched external dataset using `NEW_EXTERNAL_CONFIRMATION_REQUIREMENTS.md`.
