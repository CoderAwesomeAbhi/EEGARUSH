# COG-BCI One-Shot Prospective Test — Run-Once Checklist

**Purpose:** enforce that the COG-BCI z-scoring transport prospective test runs
**exactly once**, with all inputs verified and all degrees of freedom frozen.

> **CURRENT STATE: FROZEN — READY FOR A SINGLE PROSPECTIVE RUN.**
> `COG_BCI_EXECUTABLE_CONFIG.yaml` has `status: FROZEN_READY_FOR_SINGLE_PROSPECTIVE_RUN`.
> The sampling-representation ambiguity is resolved
> (`COG_BCI_PROTOCOL_AMENDMENT_SAMPLING_REPRESENTATION.md`). **No COG-BCI predictive
> metric has been computed.** The one-shot test has **not** been run.

---

## Gate 0 — Protocol ambiguity resolved

- [x] Sampling conflict resolved by an explicit committed amendment
      (`COG_BCI_PROTOCOL_AMENDMENT_SAMPLING_REPRESENTATION.md`): both source MAT
      and target COG-BCI resampled 500→128 Hz via
      `scipy.signal.resample_poly(up=32, down=125)` before feature extraction.
- [x] `COG_BCI_EXECUTABLE_CONFIG.yaml`: `sampling_pipeline.resolved: true`,
      `blocking: false`, `status: FROZEN_READY_FOR_SINGLE_PROSPECTIVE_RUN`.
- [x] Native-500-Hz COG-BCI analysis prohibited from the primary verdict.
- [x] No COG-BCI predictive metric was computed at or before this point.

## Gate 1 — Source input integrity

- [x] All 29 subject locked `ses-S1` inputs materialized locally (10 from local
      full ZIPs with full-archive MD5 verified; 19 via official Zenodo HTTP range
      reads, per-member CRC32 validated on extract). No live remote reads at run time.
- [x] Exact locked inputs present for every subject: `RS_Beg_EO`, `RS_End_EO`,
      `MATBdiff` × `.set`/`.fdt` = 6 files × 29 = 174 files.
- [x] SHA-256 of every locked input recorded in
      `results/cog_bci_provenance/cog_bci_execution_input_hash_manifest.csv`
      (174/174 rows `YES_local_exact_input`).
- [ ] **At run start:** re-verify every input SHA-256 matches the manifest
      (the script enforces this and aborts on any mismatch).
- [ ] No locked input is non-finite / corrupt; no locked channel missing; no locked
      condition missing. Any such failure is a **pre-outcome** exclusion only.

## Gate 2 — No prior result exists

- [x] No COG-BCI AUC, bootstrap, permutation, or any predictive metric exists.
- [ ] **At run start:** `results/cog_bci_one_shot/` contains no result file and no
      `RUN_MARKER.json` (the script aborts if any exist).

## Gate 3 — Configuration & script integrity

- [ ] `COG_BCI_EXECUTABLE_CONFIG.yaml` SHA-256 matches the recorded frozen value in
      `results/cog_bci_provenance/cog_bci_frozen_run_materials_checksums.json`.
- [ ] `scripts/run_cog_bci_one_shot_prospective.py` SHA-256 matches the recorded
      frozen value (the script aborts on any mismatch).
- [x] Frozen config matches locked rules: subjects=29, session=ses-S1,
      calibration=RS_Beg_EO, scored-rest=RS_End_EO, scored-task=MATBdiff,
      channels=8 (F3,F4,F7,F8,O1,O2,T7→T3,T8→T4), features=96, model=L2 logistic
      (C=1.0, liblinear, balanced, max_iter=5000), endpoint=macro subject ROC-AUC,
      primary comparison=paired bootstrap ΔAUC (z-scoring − mean subtraction),
      secondary=z-scoring − absolute, windows=14 rest + 14 task per subject,
      sampling=128 Hz for both datasets.

## Gate 4 — Single execution discipline

- [ ] Script invoked with the explicit `--execute-locked-one-shot` flag (it refuses
      otherwise).
- [ ] Exactly **one** execution.
- [ ] A run marker is written on execution.
- [ ] **No rerun** after outcomes are accessed — unless the run is explicitly
      labeled invalid due to a **pre-result software crash** (crash before any
      predictive metric was produced).

## Gate 5 — Post-run honesty

- [ ] Report success / partial / failure honestly, regardless of direction
      (per the locked decision rules in the config and protocol).
- [ ] No post-outcome retuning, task switching (N-back stays secondary/exploratory
      and cannot change the primary verdict), channel expansion, feature
      reselection, endpoint change, or subject cherry-picking.
- [ ] MAT/STEW results unchanged; no rejected claim revived; `main.tex` untouched.

---

**Run command (only after the unchecked run-start gates pass):**

```
python scripts/run_cog_bci_one_shot_prospective.py --execute-locked-one-shot
```
