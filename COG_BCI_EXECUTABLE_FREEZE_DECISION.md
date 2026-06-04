# COG-BCI Executable Freeze Decision

**No-results freeze.** No AUC, model prediction, separability statistic,
permutation statistic, feature importance, bootstrap interval, or any
outcome-dependent result was computed in this task. No COG-BCI raw/source files
are committed (all live under git-ignored `data/raw/`).

This decision finalizes whether the COG-BCI one-shot prospective z-scoring
transport test is cleared for its single execution. It supersedes the prior
`COG_BCI_EXECUTABLE_CONFIG_BLOCKED_PROTOCOL_AMBIGUITY_DO_NOT_RUN` verdict, whose
single blocker — the sampling representation — has now been resolved by an explicit,
pre-outcome protocol amendment.

Inputs:
- `COG_BCI_PROTOCOL_AMENDMENT_SAMPLING_REPRESENTATION.md` (the resolution)
- `COG_BCI_ONE_SHOT_PROSPECTIVE_TEST_PROTOCOL.md` (corrected)
- `REVISED_ZSCORE_PROSPECTIVE_EVALUATION_PROTOCOL.md`
- `TRANSFER_FEATURE_COMPATIBILITY_AUDIT.md`
- `results/stew_sensitivity/transport_compatible_feature_spec.yaml`
- `COG_BCI_SOURCE_PROVENANCE_AUDIT.md` / `COG_BCI_SOURCE_PROVENANCE_DECISION.md`
- `results/cog_bci_provenance/` CSVs + the input hash manifest.

---

## 1. The single prior blocker is resolved

The executable freeze previously blocked because two committed, pre-data freezes
conflicted on the sampling representation:

- the frozen MAT→STEW transport pipeline trained the source model at **128 Hz**
  (`transport_compatible_feature_spec.yaml`:
  `mat_resample: resample_poly, up=32, down=125, target_hz=128.0`;
  `scripts/run_stew_sensitivity_and_transfer.py`: `TARGET_SFREQ = 128.0`); but
- `COG_BCI_ONE_SHOT_PROSPECTIVE_TEST_PROTOCOL.md` asserted "500 Hz native — no
  resample needed."

Because the 96 transport features are scale/offset-invariant but **not**
sampling-rate-invariant, the 128-Hz-trained model must be applied to 128-Hz target
features. The conflict has been resolved — **before any COG-BCI predictive metric**
— by `COG_BCI_PROTOCOL_AMENDMENT_SAMPLING_REPRESENTATION.md`, which:

- restores the frozen representation: **both** source MAT and target COG-BCI are
  resampled 500 → 128 Hz via `scipy.signal.resample_poly(up=32, down=125)` before
  extracting the frozen 96 features;
- forbids native-500-Hz COG-BCI analysis from influencing the primary verdict;
- is justified by internal pipeline consistency only (no COG-BCI performance), and
  reopens **no** other degree of freedom.

The corrected statement is now reflected in the protocol, `docs/EEGARUSH_CHECKPOINT.md`,
`CLAUDE.md`, and `COG_BCI_EXECUTABLE_CONFIG.yaml`
(`sampling_pipeline.resolved: true`, `blocking: false`).

## 2. Source input integrity — complete

All **29** subjects' exact locked `ses-S1` inputs are now materialized locally and
SHA-256-recorded (`results/cog_bci_provenance/cog_bci_execution_input_hash_manifest.csv`,
174/174 rows `YES_local_exact_input`):

- `sub-01`…`sub-10`: materialized from local full ZIPs, **full-archive MD5 verified**
  byte-for-byte against the Zenodo manifest.
- `sub-11`…`sub-29`: materialized via official **Zenodo HTTP range reads** of exactly
  the locked members (per-member CRC32 validated on extract); labeled as exact-input
  SHA-256 hashing rather than full-archive MD5.

Each subject has all six locked files (`RS_Beg_EO`, `RS_End_EO`, `MATBdiff` ×
`.set`/`.fdt`). The final run will not depend on any live remote range read; all
inputs are local. No subject required a pre-outcome exclusion (all readable, 8
locked channels present, 500 Hz, all three locked conditions present).

## 3. Everything else remains as locked (unchanged)

| Element | Frozen value |
|---|---|
| Subjects | all 29 (`sub-01`…`sub-29`); pre-outcome objective exclusions only |
| Session | `ses-S1` only |
| Calibration | `RS_Beg_EO`, first 30 s |
| Scored-rest (label 0) | `RS_End_EO`, first 30 s |
| Scored-task (label 1) | `MATBdiff` (MATB difficult), first 30 s |
| Windows | 4 s, 50 % overlap → **14 rest + 14 task** scored windows per subject |
| Channels | 8: F3,F4,F7,F8,O1,O2,T7,T8 → MAT F3,F4,F7,F8,O1,O2,T3,T4 |
| Features | frozen 96 transport-compatible |
| Sampling | **128 Hz for both MAT and COG-BCI** (resample_poly up=32, down=125) |
| Model | L2 logistic, `C=1.0`, `liblinear`, `class_weight=balanced`, `max_iter=5000`, fit on MAT only |
| Primary method | z-scoring |
| Primary comparator | mean subtraction |
| Secondary comparator | absolute |
| Endpoint | macro subject-level ROC-AUC |
| Primary comparison | paired subject-bootstrap ΔAUC (z-scoring − mean subtraction) |
| Full success | z-scoring above chance AND paired 95 % CI excludes zero positively |
| N-back | excluded from primary |

Window counts were re-confirmed from the audited durations
(`cog_bci_locked_channel_verification.csv`; all segments ≥ 30 s) →
`floor((30−4)/2)+1 = 14`, equal for scored-rest and scored-task, all 29 subjects
(`results/cog_bci_provenance/cog_bci_primary_session_window_plan.csv`). Window
counts are time-defined and unaffected by the sampling rate.

## 4. One-shot script readiness (prepared, not executed)

`scripts/run_cog_bci_one_shot_prospective.py`:
- reads only the frozen config;
- resamples **both** source MAT and target COG-BCI to the frozen 128 Hz
  representation, extracts only the frozen 96 features on the 8 locked channels;
- refuses to execute without `--execute-locked-one-shot`;
- refuses unless config status is `FROZEN_READY_*` and the sampling rule is the
  resolved 128 Hz `resample_poly(up=32, down=125)` rule for both datasets;
- refuses unless config & script SHA-256 match the frozen run-materials checksums
  (`results/cog_bci_provenance/cog_bci_frozen_run_materials_checksums.json`);
- refuses if any result output already exists; verifies every input SHA-256;
- writes a run marker + checksums on eventual execution.

It was syntax-checked and the guards were verified. **No prediction/AUC step was
run.**

## 5. Run-once procedure

Follow `COG_BCI_RUN_ONCE_CHECKLIST.md`; run exactly once:

```
python scripts/run_cog_bci_one_shot_prospective.py --execute-locked-one-shot
```

Then report success / partial / failure honestly, regardless of direction.

---

## VERDICT

```
COG_BCI_EXECUTABLE_CONFIG_FROZEN_READY_FOR_SINGLE_PROSPECTIVE_RUN
```

The sampling-representation ambiguity is resolved by an explicit pre-outcome
amendment; all 29 subjects' locked inputs are local and SHA-256-recorded; the
config, checklist, and one-shot script implement the corrected 128 Hz representation
exactly; and all other locked design choices are unchanged. No COG-BCI predictive
metric has been computed. The one-shot prospective test is cleared for a single
execution and has **not** been run.
