# COG-BCI One-Shot Prospective Test — Run-Once Checklist

**Purpose:** enforce that the COG-BCI z-scoring transport prospective test runs
**exactly once**, with all inputs verified and all degrees of freedom frozen, and
**only after** the sampling-representation protocol ambiguity is resolved.

> **CURRENT STATE: BLOCKED.**
> `COG_BCI_EXECUTABLE_CONFIG.yaml` has `status: BLOCKED_PROTOCOL_AMBIGUITY_DO_NOT_RUN`.
> Do **not** run any item past Gate 0 until Gate 0 passes. See
> `COG_BCI_EXECUTABLE_FREEZE_DECISION.md`.

---

## Gate 0 — Protocol ambiguity resolved (HARD BLOCKER)

- [ ] The sampling-rate conflict between
      `results/stew_sensitivity/transport_compatible_feature_spec.yaml` (128 Hz)
      and `COG_BCI_ONE_SHOT_PROSPECTIVE_TEST_PROTOCOL.md` (500 Hz native, no
      resample) is resolved by an **explicit, committed protocol amendment**.
- [ ] The amendment names the exact sampling rule (expected: resample COG-BCI
      500→128 Hz via `scipy.signal.resample_poly(up=32, down=125)` before feature
      extraction, matching the frozen MAT model).
- [ ] `COG_BCI_EXECUTABLE_CONFIG.yaml`: `sampling_pipeline.resolved: true` and
      `status:` changed from `BLOCKED_PROTOCOL_AMBIGUITY_DO_NOT_RUN` to a
      ready state.
- [ ] No COG-BCI predictive metric was computed at or before this point.

**If Gate 0 is not fully checked, STOP. The one-shot test must not run.**

---

## Gate 1 — Source input integrity

- [ ] All 29 subject archives downloaded locally (no live remote range reads at run time).
- [ ] All 29 per-subject archive MD5s verified byte-for-byte against the Zenodo manifest.
- [ ] All exact locked `ses-S1` input files (`RS_Beg_EO`, `RS_End_EO`, `MATBdiff`
      × `.set`/`.fdt`) materialized locally for all 29 subjects.
- [ ] SHA-256 of every locked input file recorded in
      `results/cog_bci_provenance/cog_bci_execution_input_hash_manifest.csv`
      (and re-verified to match at run start).
- [ ] No locked input is non-finite / corrupt; no locked channel missing; no locked
      condition missing. Any such failure is a **pre-outcome** exclusion only.

*(As of this freeze: 10/29 subjects materialized + hashed; 19/29 pending — Gate 1
not yet satisfiable.)*

---

## Gate 2 — No prior result exists

- [ ] No COG-BCI AUC, bootstrap, permutation, or any predictive metric exists.
- [ ] The results directory `results/cog_bci_one_shot/` does **not** contain any
      result file (script aborts if any exist).
- [ ] No `RUN_MARKER.json` exists.

---

## Gate 3 — Configuration & script integrity

- [ ] `COG_BCI_EXECUTABLE_CONFIG.yaml` SHA-256 recorded.
- [ ] `scripts/run_cog_bci_one_shot_prospective.py` SHA-256 recorded.
- [ ] Both checksums match the values frozen at the moment of (eventual) execution,
      written to `results/cog_bci_one_shot/config_and_script_checksums.json`.
- [ ] Frozen config matches locked rules: subjects=29, session=ses-S1,
      calibration=RS_Beg_EO, scored-rest=RS_End_EO, scored-task=MATBdiff,
      channels=8, features=96, model=L2 logistic (C=1.0, liblinear, balanced,
      max_iter=5000), endpoint=macro subject ROC-AUC, comparison=paired bootstrap
      ΔAUC (z − mean-subtraction), windows=14/14/14.

---

## Gate 4 — Single execution discipline

- [ ] Script invoked with the explicit `--execute-locked-one-shot` flag (it refuses
      otherwise).
- [ ] Exactly **one** execution.
- [ ] A run marker is written on execution.
- [ ] **No rerun** after outcomes are accessed — unless the run is explicitly
      labeled invalid due to a **pre-result software crash** (i.e., the crash
      occurred before any predictive metric was produced). Any rerun must document
      that the prior attempt produced no metric.

---

## Gate 5 — Post-run honesty

- [ ] Report success / partial / failure honestly, regardless of direction
      (per the locked decision rules in the config and protocol).
- [ ] No post-outcome retuning, task switching (N-back stays secondary/exploratory
      and cannot change the primary verdict), channel expansion, feature
      reselection, endpoint change, or subject cherry-picking.
- [ ] MAT/STEW results unchanged; no rejected claim revived; `main.tex` untouched.

---

**Run command (only after every gate above passes):**

```
python scripts/run_cog_bci_one_shot_prospective.py --execute-locked-one-shot
```
