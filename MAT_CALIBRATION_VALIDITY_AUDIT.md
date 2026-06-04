# MAT Calibration Validity Audit

Verdict: `feature_table_split_nonoverlapping_but_raw_header_unverified`.

## Findings

- MAT feature table: `outputs_reproduced/features/eeg_features.csv`.
- Split rule audited: `timed_first_60s`.
- Raw EDF files found locally: `0`.
- Rest calibration windows: `1080`.
- Rest scoring windows: `2106`.
- Task scoring windows: `1080`.
- Calibration/scoring overlap windows: `0`.
- Recomputed MAT zscore logistic_l2 AUC from saved predictions: `0.865995`.

## Interpretation

The executed feature table contains rest windows extending beyond 60 seconds, so the split used by the code is not internally impossible at the feature-table level.
However, no raw PhysioNet MAT EDF files are present in this repository, so EDF-header duration, sampling frequency, and acquisition metadata cannot be independently verified here.
MAT headline result must be treated as provenance-limited until raw EDF headers are audited.

## Subject-Level Provenance Summary

| subject_id | rest_calibration_windows | scored_windows | calibration_scoring_overlap | max_rest_end_sec |
| ---------- | ------------------------ | -------------- | --------------------------- | ---------------- |
| Subject00  | 30                       | 90             | 0                           | 182              |
| Subject01  | 30                       | 90             | 0                           | 182              |
| Subject02  | 30                       | 90             | 0                           | 182              |
| Subject03  | 30                       | 90             | 0                           | 182              |
| Subject04  | 30                       | 84             | 0                           | 170              |
| Subject05  | 30                       | 90             | 0                           | 182              |
| Subject06  | 30                       | 90             | 0                           | 182              |
| Subject07  | 30                       | 90             | 0                           | 182              |
| Subject08  | 30                       | 90             | 0                           | 182              |
| Subject09  | 30                       | 90             | 0                           | 182              |
| Subject10  | 30                       | 93             | 0                           | 188              |
| Subject11  | 30                       | 90             | 0                           | 182              |

Full manifests: `results/audit/mat_file_header_manifest.csv` and `results/audit/mat_window_provenance.csv`.
