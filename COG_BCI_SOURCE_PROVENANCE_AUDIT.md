# COG-BCI Source Provenance & Construct Audit

**Audit only.** No predictive model was run; **no AUC, separability, correlation,
feature-importance, or any result-dependent statistic was computed.** Outcome
(workload) labels were not used beyond confirming condition file naming. No COG-BCI
signal files are committed (all under git-ignored `data/raw/`).

**Source:** COG-BCI database (Hinss et al. 2023), Zenodo record **7413650** (concept
DOI `10.5281/zenodo.6874128`), CC-BY 4.0. Manifest: 29 per-subject zips + 5 docs,
**31.69 GB**, MD5s on record. Audit script: `scripts/audit_cog_bci_provenance.py`.

**Retrieval & integrity:** 10 subject zips downloaded locally and **MD5-verified
byte-for-byte** against the Zenodo manifest; the remaining 19 verified via
**authenticated HTTPS range reads** (`remotezip`) of just the needed members
(namelist + ses-S1 headers), with full-archive MD5 on record (full download +
MD5 deferred to execution time). One initially-partial local `sub-11.zip` (28 MB of
1127 MB) was detected by MD5 mismatch, discarded, and re-read cleanly from source.

## Audit questions (Task 2)

1. **Files readable:** Yes — **29/29** subject archives readable.
2. **Subjects / sessions present:** **29 subjects** (`sub-01`…`sub-29`), **3
   sessions each** (`ses-S1`, `ses-S2`, `ses-S3`).
3. **Eyes-open resting identifiable:** Yes — dedicated `RS_Beg_EO` and `RS_End_EO`
   recordings (eyes-closed `RS_Beg_EC`/`RS_End_Ec` also exist and are **excluded**
   from the primary model by the locked protocol).
4. **RS_Beg_EO and RS_End_EO both available per subject/session:** **Yes for all
   87 subject×session combinations** (29×3) — no missing condition in the
   session/condition summary.
5. **MATB-difficult identifiable without performance:** Yes — a dedicated
   `MATBdiff.set/.fdt` recording per session, identified by **filename/condition
   naming only** (no behavioral/outcome data used).
6. **Sampling frequency encoded in files:** **500 Hz** in all 87 ses-S1 condition
   recordings checked.
7. **Channel names/order + 8 locked channels:** All eight locked channels
   (**F3, F4, F7, F8, O1, O2, T7≡MAT T3, T8≡MAT T4**) present in **every** checked
   recording (`locked_present = 8` for all 87 rows). Total channel count is **63 or
   64** depending on subject (extra channel e.g. ECG/extra reference); the locked 8
   are always present. Channel order is recorded per file in
   `cog_bci_locked_channel_verification.csv`.
8. **Units:** physical **voltage (µV / EEGLAB-BIDS convention)** — research-grade
   BrainProducts, **not** arbitrary device counts (the STEW raw-count unit problem
   does **not** recur). Per-recording DC offset is irrelevant to the locked pipeline
   (96 features invariant under `x→a·x+b`, plus per-subject calibration).
9. **Non-overlapping calibration / scored-rest / scored-task design — exactly as
   locked:** **Implementable.** `RS_Beg_EO` (≈60 s), `RS_End_EO` (≈60 s), and
   `MATBdiff` (≈299 s) are **three separate recordings**, so calibration
   (`RS_Beg_EO`), scored-rest (`RS_End_EO`), and scored-task (`MATBdiff`) are
   non-overlapping by construction. Equal fixed evaluated window counts per subject
   are achievable at 4 s / 50 % overlap: `RS_End_EO` ≈60 s → ≈28 windows;
   `MATBdiff` ≈299 s → ≈148 windows; cap scored-task to the scored-rest count
   (≈28) to enforce equality. Calibration stats come from `RS_Beg_EO`.
10. **Exclusions required:** **None** for the locked conditions — every subject has
    all three locked recordings with 8 channels at 500 Hz in `ses-S1` (and the
    files are present in all three sessions). (The transient `sub-11` partial was a
    download artifact, not a dataset defect.)
11. **Single clean one-shot execution without modifying the locked protocol:**
    **Yes**, with one **config specialization** (not a protocol change): because
    **3 sessions** exist, the single session must be predeclared. The first session
    **`ses-S1`** is selected by a fixed, non-performance rule. This does **not**
    alter any locked rule (baseline = eyes-open, task = MATB-difficult, channels =
    8, features = 96, model = L2 logistic, endpoint = macro subject ROC-AUC, success
    rule). It is to be frozen in the next executable-config stage.

## Outputs (committable)

- `results/cog_bci_provenance/cog_bci_source_manifest.csv` — per archive: source
  (local/remote), size, expected MD5, MD5/integrity status, entry count.
- `results/cog_bci_provenance/cog_bci_subject_session_condition_summary.csv` —
  per subject×session presence of the three locked conditions (87 rows).
- `results/cog_bci_provenance/cog_bci_locked_channel_verification.csv` — per
  subject ses-S1 × condition: sfreq, total channels, locked-channels-present (8),
  missing list, duration.
- `results/cog_bci_provenance/cog_bci_protocol_feasibility.csv` — per subject:
  presence of calibration/scored-rest/scored-task and `locked_design_executable`.

## Construct/compatibility notes (not blockers)

- Task construct: MATB operational multitasking vs MAT serial arithmetic (both
  workload; expected cross-task shift — the point of the prospective test).
- Reference scheme differs (COG-BCI Fpz vs MAT A2-A1); irrelevant to the 96
  scale/offset-invariant transport features.
- Sampling rate **500 Hz matches MAT** (no MAT resample needed for this target).

## Stopping rule honored

Even though provenance passes, the one-shot z-scoring test is **NOT** run here. The
next task must freeze the exact executable config (incl. the `ses-S1` session
choice and window-count rule) and produce a run-once checklist **before any AUC is
computed**.
