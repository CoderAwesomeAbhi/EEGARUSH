# STEW Source Retrieval Report

## Verdict

Final verdict: `STEW_SOURCE_FILES_OBTAINED_READY_FOR_PROVENANCE_AUDIT`

## Official Dataset Identity

- Full dataset name: STEW: Simultaneous Task EEG Workload Dataset.
- Original publication: Wei Lun Lim, Olga Sourina, and Lipo Wang. 2018. "STEW: Simultaneous Task EEG Workload Data Set." IEEE Transactions on Neural Systems and Rehabilitation Engineering, 26(11), 2106-2114. DOI: `10.1109/TNSRE.2018.2872924`.
- Official dataset DOI: `10.21227/44r8-ya50`.
- Official hosting location: IEEE DataPort open-access dataset page, `https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset`.
- Repository/manuscript source currently referenced: MONSTER/Hugging Face processed time-series mirror, `https://huggingface.co/datasets/monster-monash/STEW`.

## Source Evidence

- PubMed and the accepted manuscript identify STEW as an open-access EEG dataset for a SIMKAP multitasking workload experiment with 48 subjects.
- The Hugging Face MONSTER dataset card reports 48 participants, 14 EEG channels, 128 Hz sampling, 2.5 minutes of EEG per case, baseline rest plus workload recordings, and 28,512 processed multivariate time-series examples of length 256.
- The Hugging Face card states that STEW can be accessed upon request through IEEE DataPort. Therefore, the original IEEE DataPort archive may require account/request workflow even though the MONSTER processed time-series files are directly downloadable.

## Download Requirements Assessment

| Source | Role | Raw/time-resolved EEG offered? | Retrieval status | Access requirements |
| --- | --- | --- | --- | --- |
| IEEE DataPort STEW page | Original official dataset host | Reported by literature as raw EEG / original STEW dataset | Not downloaded in this task | Appears request/account-mediated based on the Hugging Face card's "upon request" wording; exact DataPort download controls must be handled by a human if the original archive is required |
| MONSTER/Hugging Face `monster-monash/STEW` | Repository-compatible processed time-series source | Yes, processed time-resolved windows: `STEW_X.npy`, labels, subject IDs, metadata, folds | Downloaded into Git-ignored local storage | Direct unauthenticated HTTP download worked in this environment |

## Local Repository Search

Before retrieval, the local search found no overlooked STEW raw/source files. Existing local STEW artifacts were limited to cached features and result tables:

- `results/multi_dataset/stew_features.parquet`
- `results/multi_dataset/predictions_stew.csv`
- `results/multi_dataset/biological_stew.csv`
- `results/multi_dataset/metrics_stew.csv`
- prior audit/result files under `results/stew_provenance`, `results/theory_validation`, and `outputs_phd_revision`

No local `STEW_X.npy`, `STEW_y.npy`, `STEW_subject_id.csv`, raw continuous signal archive, MATLAB file, EDF file, EEGLAB `.set/.fdt`, or source dataset folder was present before this retrieval.

## Files Obtained

The directly downloadable MONSTER/Hugging Face STEW processed time-series bundle was retrieved into:

`data/raw/stew/monster-monash_STEW/`

This folder is under `data/raw/`, which is already excluded by `.gitignore`; the raw/source files must not be committed.

Retrieved files:

- `README.md`
- `STEW.py`
- `STEW_X.npy`
- `STEW_y.npy`
- `STEW_subject_id.csv`
- `STEW_metadata.npy`
- `test_indices_fold_0.txt`
- `test_indices_fold_1.txt`
- `test_indices_fold_2.txt`
- `test_indices_fold_3.txt`
- `test_indices_fold_4.txt`

The file manifest with sizes, file types, and SHA-256 hashes was written to:

`results/stew_provenance/stew_source_file_manifest.csv`

## Interpretation For Next Audit

The obtained files are not the original continuous IEEE DataPort archive. They are the repository-compatible MONSTER/Hugging Face processed time-series representation. They appear time-resolved and may be sufficient for a second-dataset processed-window provenance audit, but they still require a separate audit before any STEW reconstruction, transfer test, or manuscript claim.

Do not run feature extraction, modeling, transfer testing, or manuscript editing until the retrieved source bundle is audited.
