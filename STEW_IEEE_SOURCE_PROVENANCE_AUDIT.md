# STEW (Official IEEE DataPort) Source Provenance Audit

**Scope:** Provenance audit only. No models were trained. No raw or extracted
STEW data is committed to Git (`data/raw/` remains git-ignored).

**Source archive:** `STEW Dataset.zip` (54,811,358 bytes) downloaded by the user
from IEEE DataPort and copied (still git-ignored) to
`data/raw/stew/ieee_dataport_stew/`, then extracted to
`data/raw/stew/ieee_dataport_stew/STEW Dataset/`.

**Generated summaries (committable):**
- `results/stew_ieee_provenance/stew_ieee_source_manifest.csv` (96 rows)
- `results/stew_ieee_provenance/stew_condition_subject_summary.csv` (48 rows)
- `results/stew_ieee_provenance/mat_stew_verified_channel_harmonization.csv` (14 rows)
- Audit script: `scripts/audit_stew_ieee_source_provenance.py` (stdlib-only)

---

## 1. Source files

- Archive contents: `STEW Dataset/` containing `ratings.txt` plus per-subject EEG
  text files. Total 97 files (1 ratings file + 96 EEG files).
- EEG files follow the pattern `subNN_lo.txt` and `subNN_hi.txt`.
- All 96 EEG files verified: **19200 samples × 14 columns**, 4,339,200 bytes each
  (uniform — no truncated or malformed files).

## 2. Subjects

- **48 subjects**, IDs `sub01` … `sub48`.
- **Every subject has both** a `_lo` (rest/no-task) and a `_hi` (high-workload)
  recording (48/48 complete pairs).

## 3. Conditions (rest/baseline vs workload)

- `_lo` → **rest / no-task** baseline recording (label 0).
- `_hi` → **high-workload** recording during the SIMKAP simultaneous multitasking
  test (label 1).
- This is a genuine resting/no-task baseline embedded in the official source —
  the specific element the Hugging Face/MONSTER bundle lacked.
- **Corroboration from `ratings.txt`:** subjective workload ratings (1–9) are
  present for 45/48 subjects (missing: sub05, sub24, sub42). For **all 45 rated
  subjects the rest rating is strictly lower than the task rating** (lo<hi
  monotonic 45/45; lo range ≈ 1–5, hi range ≈ 4–9). This independently supports
  the `_lo = low/rest`, `_hi = high-workload` condition labels.

## 4. Sampling rate and channels

- **Sampling rate: 128 Hz (documented Emotiv EPOC rate).** It is **not embedded**
  in the `.txt` files but is corroborated by the uniform sample count:
  19200 samples ÷ 128 Hz = **150.0 s** per recording (the documented retained
  2.5-minute segment).
- **Channels: 14 columns**, in the documented fixed Emotiv EPOC order
  `AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4`.
- **Channel labels are NOT embedded** in the source files — column identity relies
  on the documented Emotiv EPOC hardware order, not on per-file metadata.

## 5. Non-overlapping calibration / scored-rest / scored-task feasibility

Constructable **per subject**:
- **Calibration baseline** and **scored-rest** can both be drawn from disjoint
  time windows of the single `_lo` (rest) recording (19200 samples / 150 s is
  ample to split into non-overlapping calibration and scored-rest segments).
- **Scored-task** windows come from the `_hi` recording.
- Because calibration and scored-rest are taken from disjoint windows of the same
  rest file, and scored-task is a separate file, a **non-overlapping** three-way
  segmentation is feasible (parallel to the MAT `_1`/`_2` two-file design).

## 6. MAT/STEW channel harmonization

MAT scalp EEG channels (19, from `results/raw_provenance/mat_raw_edf_manifest.csv`,
old 10-20 nomenclature): `Fp1, Fp2, F3, F4, F7, F8, T3, T4, C3, C4, T5, T6, P3,
P4, O1, O2, Fz, Cz, Pz` (plus the `A2-A1` reference and an `ECG` channel, excluded).

Harmonizable STEW→MAT channels (**10 of 14**):

| STEW (documented) | MAT channel | basis |
|---|---|---|
| F7 | F7 | direct |
| F3 | F3 | direct |
| F4 | F4 | direct |
| F8 | F8 | direct |
| O1 | O1 | direct |
| O2 | O2 | direct |
| T7 | T3 | documented 10-20 equivalence |
| T8 | T4 | documented 10-20 equivalence |
| P7 | T5 | documented 10-20 equivalence |
| P8 | T6 | documented 10-20 equivalence |

STEW channels with **no MAT counterpart** (dropped): `AF3, AF4, FC5, FC6`.

**Defensibility:** harmonization to ~10 common channels is achievable, but it
relies on two *documented external conventions* rather than file-embedded metadata:
(a) the fixed Emotiv EPOC channel order, and (b) old↔new 10-20 name equivalences
(T7≡T3, T8≡T4, P7≡T5, P8≡T6). It is defensible as an approximate harmonization,
not as a verified-from-source label match.

---

## Provenance summary

| Property | MAT (locked) | STEW (official IEEE) |
|---|---|---|
| Subjects | 36 | 48 |
| Genuine resting baseline | yes (`_1`) | **yes (`_lo` / no-task)** |
| Workload task | mental arithmetic | SIMKAP multitasking |
| Sampling rate | 500 Hz (EDF header) | 128 Hz (documented, corroborated) |
| Channels | 21 (19 scalp), 10-20 | 14, Emotiv EPOC |
| Channel labels in source | yes (EDF headers) | **no (documented order only)** |
| Temporal provenance | yes | yes (150 s retained segments) |
| Subject IDs | yes | yes (`sub01`–`sub48`) |
| Non-overlapping cal/rest/task design | yes | yes (constructable) |
| Harmonizable channels with the other set | — | 10 |

The official IEEE source **restores** the three things the MONSTER bundle lacked
(genuine resting baseline, temporal provenance, subject IDs). Remaining structural
differences vs MAT are: sampling rate (128 vs 500 Hz), montage/channel set
(14 Emotiv vs 19 scalp 10-20; only 10 harmonizable), absence of source-embedded
channel labels, different referencing, and a different workload task construct
(multitasking vs arithmetic).

**Construct-match verdict is recorded in
`STEW_IEEE_CONSTRUCT_MATCH_DECISION.md`.**
