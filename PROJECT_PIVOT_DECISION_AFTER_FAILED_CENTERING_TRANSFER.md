# Project Pivot Decision After Failed Centering Transfer (Task 5)

Synthesizes: `STEW_SENSITIVITY_AND_TRANSFER_DECISION.md`,
`MAT_TO_STEW_EXPLORATORY_TRANSFER_RESULTS.md`,
`POSTHOC_ZSCORE_TRANSPORT_AUDIT.md`,
`REVISED_ZSCORE_PROSPECTIVE_EVALUATION_PROTOCOL.md`,
`UNTOUCHED_DATASET_ELIGIBILITY_SCREEN.md`.

## Where the science stands

- **Within-dataset:** mean subtraction decodes workload above chance within MAT
  (AUC 0.880102, p=0.004975) and within STEW (AUC 0.839498, p=0.004975).
- **Predeclared cross-device transfer FAILED:** MAT→STEW mean-subtraction macro
  subject AUC = 0.447598 (below chance), not better than absolute (0.472045;
  paired CI [−0.070, +0.023] includes zero).
- **Post-hoc observation:** MAT→STEW **z-scoring** macro subject AUC = 0.682823
  (above chance) — a predeclared *secondary diagnostic*, elevated only after
  viewing results.

## Gate inputs

1. **Z-scoring audit verdict:** `ZSCORE_OBSERVATION_VALID_FOR_NEW_HYPOTHESIS_GENERATION`
   — leakage-free, MAT-only fitting, unlabeled-baseline calibration, frozen 96
   features, aligned metrics; valid as a *hypothesis*, not as confirmation.
2. **Revised prospective protocol:** **frozen** before any new data
   (`REVISED_ZSCORE_PROSPECTIVE_EVALUATION_PROTOCOL.md`).
3. **Untouched dataset screen:** **≥1 eligible** candidate found — **COG-BCI
   (Hinss et al. 2023)**: genuine separate eyes-open/closed resting baseline,
   64-ch 10-20 (all 8 locked channels present), 500 Hz, µV units, BIDS, CC-BY 4.0,
   untouched in this repo.

## Reasoning over the verdict options

- `…ZSCORE_HYPOTHESIS_INVALID_STOP_ESCALATION` — would require the z-scoring audit
  to be invalid. It is **valid**. → does not apply.
- `NO_ELIGIBLE_UNTOUCHED_DATASET_FOUND_NEGATIVE_METHODS_PAPER_ONLY` — would require
  zero eligible untouched datasets. COG-BCI is **eligible**. → does not apply.
- `…ZSCORE_HYPOTHESIS_READY_FOR_UNTOUCHED_TEST` — z-scoring hypothesis is valid for
  generation, protocol is frozen, and an eligible untouched dataset exists. → **applies.**

## VERDICT

```
CENTERING_TRANSFER_FAILED_ZSCORE_HYPOTHESIS_READY_FOR_UNTOUCHED_TEST
```

## Binding conditions on what "ready" means

- The centering (mean-subtraction) transfer hypothesis is **rejected** as a
  supported cross-device method; this is not reopened.
- The z-scoring transport result is **not** confirmed, replicated, or successful —
  it is a hypothesis awaiting a **single** prospective test on the untouched
  dataset under the frozen protocol (MAT = development; target = validation).
- **Before any COG-BCI modeling:** a fresh full provenance audit must pass (files,
  subject IDs, true rest vs workload, channel names/order, sampling rate,
  harmonization, non-overlapping calibration/scored design) — as done for MAT/STEW.
- No model tuning, feature reselection, channel-space expansion, or endpoint
  switching after viewing COG-BCI outcomes. A single untouched test; no dataset
  shopping.
- Still binding: no `main.tex` edits, no DS007262 re-analysis, no MONSTER bundle
  modeling, nothing committed under `data/raw/`, no revival of rejected claims.
