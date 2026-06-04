# COG-BCI Source Provenance Gate Decision

Based on `COG_BCI_SOURCE_PROVENANCE_AUDIT.md` and the four CSVs in
`results/cog_bci_provenance/`. Documentation/structure/header inspection only — **no
model, no AUC, no result-dependent statistic computed; no raw data committed.**

## Gate inputs

- Official source obtained and **readable: 29/29** subjects (10 local MD5-verified,
  19 verified via authenticated Zenodo range reads; full-archive MD5 on record).
- **Eyes-open `RS_Beg_EO` and `RS_End_EO` present for all 29 subjects × 3 sessions.**
- **MATB-difficult (`MATBdiff`) present for all 29 subjects × 3 sessions**, identified
  by filename/condition only (no outcome/performance used).
- **500 Hz** and **all eight locked channels (F3,F4,F7,F8,O1,O2,T7,T8)** present in
  every checked recording (`locked_present = 8`, `sfreq = 500` for all 87 rows).
- **Units are physical voltage (µV)** — no STEW-style unit ambiguity.
- The locked **non-overlapping** calibration (`RS_Beg_EO`) / scored-rest
  (`RS_End_EO`) / scored-task (`MATBdiff`) design is **implementable as written**;
  equal evaluated window counts achievable by capping scored-task to the
  scored-rest count.
- **No subject exclusions required** for the locked conditions.

## Reasoning over the verdict options

- `COG_BCI_SOURCE_UNAVAILABLE_OR_UNREADABLE_DO_NOT_TEST` — files obtained and
  readable (29/29). → does not apply.
- `COG_BCI_SOURCE_REQUIRES_PROTOCOL_BREAK_DO_NOT_TEST` — the locked baseline
  (eyes-open), task (MATB-difficult), eight channels, endpoint, and non-overlapping
  design are all satisfiable **without changing any locked rule**. The only
  remaining choice — selecting one of the three sessions — is a **config
  specialization** (predeclared `ses-S1`, first session, by a non-performance
  rule), **not** a change to baseline/task/channels/features/model/endpoint/success.
  → does not apply.
- `COG_BCI_SOURCE_VALID_FREEZE_EXECUTABLE_CONFIG_BEFORE_ONE_SHOT_TEST` — official
  data support the locked baseline, MATB-difficult condition, eight channels, and
  non-overlapping design. → **applies.**

## VERDICT

```
COG_BCI_SOURCE_VALID_FREEZE_EXECUTABLE_CONFIG_BEFORE_ONE_SHOT_TEST
```

## Binding conditions for the next stage (before any AUC)

- Freeze the **executable config**, fixing only undetermined config details that do
  **not** alter locked rules:
  - **session = `ses-S1`** (first session; non-performance rule);
  - **equal evaluated window counts** = min(scored-rest, scored-task) per subject at
    4 s / 50 % overlap;
  - calibration from `RS_Beg_EO`, scored-rest from `RS_End_EO`, scored-task from
    `MATBdiff`; channels = the 8; features = the frozen 96; model = fixed L2 logistic.
- For execution, the 19 range-audited subjects must be **fully downloaded and
  MD5-verified** before use.
- Produce a **run-once checklist**; then run the one-shot z-scoring prospective test
  **exactly once**; report success / partial / failure honestly.
- **Critical stopping rule honored:** the one-shot test is **NOT** run in this task.
- Still binding: no `main.tex` edits, no DS007262 re-analysis, no MONSTER modeling,
  nothing committed under `data/raw/`, no revival of rejected claims; z-scoring
  transport remains unconfirmed until the one-shot test reports.
