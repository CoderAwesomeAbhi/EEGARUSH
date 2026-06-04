# COG-BCI Pre-Download Decision (Task 4)

Synthesizes `COG_BCI_METADATA_ELIGIBILITY_AUDIT.md` and
`COG_BCI_ONE_SHOT_PROSPECTIVE_TEST_PROTOCOL.md`. Documentation/metadata only — **no
COG-BCI signals, labels, or outcomes were downloaded or inspected.**

## Gate inputs

1. **Metadata eligibility:** `COG_BCI_METADATA_ELIGIBLE_FOR_PRETEST_LOCK` — 29
   subjects; 500 Hz; 64-ch 10-20 with all eight locked channels
   (F3,F4,F7,F8,O1,O2,T7≡T3,T8≡T4); genuine separate eyes-open & eyes-closed
   resting; MATB (easy/medium/difficult) and N-back present; `.set/.fdt` BIDS raw;
   CC-BY 4.0; untouched relative to this repo.
2. **Primary paradigm validity:** MATB is present with a documented highest level
   ("difficult"), giving a valid fixed workload contrast → the
   `COG_BCI_PRIMARY_PARADIGM_INVALID_DO_NOT_DOWNLOAD` contingency does **not** apply.
3. **Channel availability:** all eight harmonized channels documented → no
   channel-incompatibility stop.
4. **One-shot protocol:** frozen before download
   (`COG_BCI_ONE_SHOT_PROSPECTIVE_TEST_PROTOCOL.md`) — primary paradigm MATB,
   baseline eyes-open only, endpoint macro subject ROC-AUC for z-scoring on
   eyes-open-rest vs MATB-difficult, primary comparison z-scoring − mean subtraction,
   frozen 96 features, fixed L2 model, single run, locked analysis order.

## Reasoning over the verdict options

- `COG_BCI_METADATA_INSUFFICIENT_DO_NOT_DOWNLOAD` — metadata is sufficient and
  complete. → does not apply.
- `COG_BCI_PRIMARY_PARADIGM_INVALID_DO_NOT_DOWNLOAD` — MATB is present and provides a
  valid fixed contrast. → does not apply.
- `COG_BCI_NOT_ELIGIBLE_REPLACE_DATASET` — dataset is eligible and construct-matched.
  → does not apply.
- `COG_BCI_ONE_SHOT_PROTOCOL_LOCKED_READY_FOR_DOWNLOAD_AND_PROVENANCE_AUDIT` —
  eligible + paradigm valid + channels present + one-shot protocol frozen. → **applies.**

## VERDICT

```
COG_BCI_ONE_SHOT_PROTOCOL_LOCKED_READY_FOR_DOWNLOAD_AND_PROVENANCE_AUDIT
```

## Binding conditions

- The next stage may **download COG-BCI** and run a **raw-data provenance audit
  only** — no modeling until that audit passes and the executable config is frozen.
- The one-shot prospective test runs **exactly once** under the frozen protocol;
  the primary verdict uses **MATB only**, eyes-open rest only, the 8 locked
  channels, and the frozen 96 features.
- Still binding: no `main.tex` edits, no DS007262 re-analysis, no MONSTER modeling,
  nothing committed under `data/raw/`, no revival of rejected claims, and the
  z-scoring hypothesis remains unconfirmed until the one-shot test reports.
