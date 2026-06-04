# COG-BCI Metadata Eligibility Audit (Task 1 — Documentation Only)

**Method:** Official paper / repository / Zenodo metadata and documentation only.
**No COG-BCI signal files, outcome arrays, or raw EEG were downloaded or inspected.**

**Primary sources**
- Hinss, M.F. et al. *Open multi-session and multi-task EEG cognitive dataset for
  passive brain–computer interface applications*, **Nature Scientific Data** (2023),
  `10.1038/s41597-022-01898-y` (open mirror: PMC9918545).
- Repository: **Zenodo `10.5281/zenodo.6874128`** (CC-BY 4.0).

## Verified items

| Requirement | Finding (from official metadata) | Met? |
|---|---|---|
| Dataset title / source | COG-BCI database (Hinss et al. 2023, Nature Sci. Data); Zenodo 10.5281/zenodo.6874128 | ✓ |
| Licensing / accessibility | Creative Commons Attribution 4.0 International; open Zenodo download | ✓ |
| Number of subjects | **29 participants**, **3 sessions** (≈1 week apart) | ✓ |
| EEG sampling rate | **500 Hz** (24-bit, 0.05 µV resolution) — matches MAT's 500 Hz | ✓ |
| Electrode montage | **64** active Ag-AgCl electrodes, **standard/extended 10-20**, Fpz reference | ✓ |
| Locked 8 channels present | F3, F4, F7, F8, O1, O2, **T7 (≡MAT T3)**, **T8 (≡MAT T4)** all in the 64-ch 10-20 montage | ✓ (all 8) |
| True resting baseline | **Yes, separate** — 1 min **eyes-open** + 1 min **eyes-closed** resting at session **beginning (RS_Beg)** and **end (RS_End)**, distinct from task blocks | ✓ |
| Workload paradigms incl. MATB / N-back | **MATB present** (operational multitasking) and **N-back present** | ✓ |
| Documented workload levels | **MATB: easy / medium / difficult** (3 discrete levels); **N-back: 0-/1-/2-back** (easy/medium/high) | ✓ |
| Raw / time-resolved EEG | EEGLAB **`.set`/`.fdt`** in **BIDS** structure (continuous/time-resolved) | ✓ |
| Untouched vs this repository | **Yes** — no reference to COG-BCI / Hinss / MATB / Flanker / PVT anywhere in repo | ✓ |

### MATB difficulty levels (exact wording)
- **Easy:** "participants only engaged in the system monitoring and the tracking tasks"
- **Medium:** "participants engaged in both tasks as well as the resource management task"
- **Difficult:** "the communications task was added, as well as the tracking task was made more difficult"

This confirms a documented **highest workload level = MATB "difficult,"** usable as a
single predeclared primary task condition.

## Compatibility notes (not blockers)

- **Units:** research-grade BrainProducts in **µV** → the STEW raw-count unit
  problem does **not** recur. The frozen **96 scale/offset-invariant** transport
  features are still used (no feature-space expansion).
- **Sampling rate:** 500 Hz matches MAT (no resampling of MAT needed for this target).
- **Reference:** Fpz (vs MAT A2-A1) — a re-referencing consideration to handle in
  the later provenance audit, not an eligibility blocker.
- **Construct:** MATB operational multitasking vs MAT serial arithmetic — both are
  workload manipulations; cross-task shift is expected and is the point of the test.

## VERDICT

```
COG_BCI_METADATA_ELIGIBLE_FOR_PRETEST_LOCK
```

All eligibility criteria are satisfied from official documentation alone. A fresh
full **raw-data provenance audit** (files, subject IDs, true rest vs MATB, channel
names/order, sampling rate, non-overlapping calibration/scored design,
harmonization) must still pass **before any modeling**, exactly as for MAT and STEW.
No COG-BCI signals or labels were downloaded or inspected.
