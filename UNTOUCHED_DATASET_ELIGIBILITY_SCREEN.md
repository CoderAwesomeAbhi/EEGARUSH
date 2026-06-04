# Untouched Dataset Eligibility Screen (Metadata Only — Task 4)

**Method:** Metadata/provenance screening only. **No signal files, labels, or
outcomes were downloaded, inspected, or analyzed** for any candidate. Selection is
on construct match + provenance quality, **not** anticipated performance. Performed
only after `REVISED_ZSCORE_PROSPECTIVE_EVALUATION_PROTOCOL.md` was frozen.

## Eligibility requirements (from task)

genuine separate resting/baseline EEG · raw/time-resolved EEG · subject IDs ·
channel names/order · documented sampling rate & provenance · enough channels for
defensible harmonization with the frozen 8-channel / 96-feature transport space ·
workload (task or graded) outcome · no prior use anywhere in this repo · accessible
licensing/source.

## Already-touched datasets (EXCLUDED — not untouched)

- **PhysioNet EEGMAT** (mental arithmetic) — primary MAT development set. Excluded.
- **STEW (IEEE DataPort + MONSTER/HF bundle)** — development sensitivity set. Excluded.
- **DS007262** — previously analyzed in this repo. Excluded.

## Candidate 1 — COG-BCI (Hinss et al. 2023) **[ELIGIBLE]**

- **Source:** *Open multi-session and multi-task EEG cognitive dataset for passive
  BCI*, Nature Scientific Data (2023); Zenodo `10.5281/zenodo.6874128`.
- **Untouched relative to this repo:** Yes (no reference to COG-BCI/MATB/Hinss/
  Flanker/PVT anywhere in the repo).
- **Baseline availability:** **Yes, genuine & separate** — 2-min resting state
  (1 min **eyes open** + 1 min **eyes closed**) at session **start and end**
  (RS_Beg/RS_End, EO/EC), distinct from the task blocks.
- **Workload condition:** **MATB** and **N-Back** elicit graded mental workload
  (3 levels); PVT/Flanker also present. Workload-vs-rest contrast available.
- **Channels / sampling rate:** **64** active Ag-AgCl electrodes (ActiCap,
  BrainProducts), **standard 10-20** montage (Fpz reference); **500 Hz**, 24-bit,
  0.05 µV resolution. The locked 8 channels (F3,F4,F7,F8,O1,O2,T7≡T3,T8≡T4) are all
  present in a 64-ch 10-20 cap.
- **Raw-data accessibility:** Raw/time-resolved EEG in **BIDS**, EEGLAB `.set/.fdt`.
  Subject IDs present. CC-BY 4.0 (open).
- **Compatibility risks:** (a) task construct = multitasking / working memory vs
  MAT serial arithmetic (both workload, moderate match); (b) reference scheme
  differs (Fpz vs MAT A2-A1) — re-referencing consideration; (c) **units are µV**
  (research-grade), so the STEW-style raw-count unit problem does **not** recur —
  this *reduces* transport risk; (d) sampling rate **500 Hz matches MAT** (no
  resample needed). Per the frozen protocol we still use only the 96
  scale/offset-invariant features and do **not** expand the feature space.
- **Eligibility verdict:** **ELIGIBLE** — strongest construct/provenance match;
  genuine separate baseline; clean 10-20 harmonization superset; documented µV
  units; open license.

## Candidate 2 — Shin et al. 2018 EEG+NIRS cognitive dataset **[PARTIAL / UNCERTAIN]**

- **Source:** *Simultaneous acquisition of EEG and NIRS during cognitive tasks*,
  Nature Scientific Data 5:180003 (2018); GitHub `JaeyoungShin/simultaneous_EEG-NIRS`.
- **Untouched relative to this repo:** Yes.
- **Baseline availability:** **Uncertain from metadata** — tasks are n-back (0/2/3),
  DSR, and word generation with inter-trial rest; a *genuine separate* resting
  baseline block is not clearly documented in metadata (would require inspection,
  which is not permitted at this stage). Flagged as a requirement to confirm before
  selection.
- **Workload condition:** n-back provides graded working-memory workload.
- **Channels / sampling rate:** **32** electrodes, **10-5** system, **1000 Hz**
  (resample to MAT rate needed). Locked 8 channels available in a 32-ch 10-5 cap.
- **Raw-data accessibility:** Open (Creative Commons), MATLAB/raw available.
- **Compatibility risks:** 1000 Hz (resample); separate-baseline availability
  unconfirmed; EEG+NIRS hybrid montage.
- **Eligibility verdict:** **PARTIAL** — promising but the genuine-separate-baseline
  requirement is not confirmable from metadata alone; kept as a backup behind
  Candidate 1.

## Note — related Shin 2017 arithmetic+baseline set

The earlier Shin 2016/2017 EEG+NIRS *mental-arithmetic vs baseline* set exists and
explicitly contrasts arithmetic with a baseline, but its construct (arithmetic)
overlaps MAT closely and its separate-resting provenance should be confirmed; not
needed given Candidate 1. Listed for completeness only.

## Screen outcome

- **≥1 eligible untouched candidate found:** **Yes — COG-BCI (Hinss et al. 2023).**
- Selection is provisional and provenance-based; a **fresh full provenance audit**
  (verify files, subject IDs, true rest vs workload, channel names/order, sampling
  rate, harmonization, non-overlapping calibration/scored design) must pass
  **before any modeling**, exactly as was done for MAT and STEW. No COG-BCI signals
  or labels have been downloaded or inspected.

## Sources

- COG-BCI: <https://www.nature.com/articles/s41597-022-01898-y> ·
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC9918545/> ·
  <https://zenodo.org/records/6874128>
- Shin 2018 EEG+NIRS: <https://www.nature.com/articles/sdata20183> ·
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC5810421/>
