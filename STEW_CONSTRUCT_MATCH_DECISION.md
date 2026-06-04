# STEW Construct Match Decision

## Decision Inputs

Validated MAT target construct:

- subject baseline mean subtraction;
- explicit baseline/rest calibration;
- separate scored-rest and scored-task windows;
- non-overlapping calibration and scoring data;
- fixed no-gamma shared-channel features;
- L2 logistic regression fixed from MAT;
- macro subject-level ROC-AUC.

Audited MONSTER/Hugging Face STEW bundle:

- time-resolved EEG windows exist: yes;
- engineered features only: no;
- subject identifier for every example: yes;
- number of subjects: `48`;
- binary labels: `0` and `1`;
- documented label meaning: low/high workload ratings, not verified rest/workload;
- explicit baseline/rest label: absent;
- window start times/source indices: absent;
- original chronological order by subject and condition: not documented;
- overlap/stride: not documented;
- channel labels/order encoded in files: absent;
- sampling frequency documented in README: `128 Hz`;
- window length: `256` samples.

## Pass/Fail Against Required Criteria

| Criterion | Status | Reason |
| --- | --- | --- |
| Contains time-resolved EEG signal windows | Pass | `STEW_X.npy` is `(28512, 14, 256)` float32. |
| Every usable example has reliable subject ID | Pass | `STEW_subject_id.csv` and `STEW_metadata.npy` provide one subject ID per example. |
| True baseline/rest and workload/task data distinguishable | Fail | `STEW_y.npy` is documented as low/high workload ratings, not rest/workload; no baseline/rest field exists. |
| Can reserve baseline calibration and separate scored-rest windows | Fail | No explicit baseline/rest examples are identifiable. |
| Calibration/scoring separation does not depend on invented temporal order | Fail | No start times, source sample indices, source files, or chronological condition order are present. |
| Channel harmonization with MAT is defensible from bundle alone | Partial fail | Existing repository code uses the Emotiv EPOC channel order, but the MONSTER files do not encode channel labels/order. |
| Preprocessing provenance sufficient for responsible interpretation | Fail | Filtering, normalization, overlap/stride, artifact handling, and raw-to-window mapping are not documented locally. |

## Processed-Window Sensitivity Assessment

A seed-locked split of baseline windows into calibration and held-out scored-rest subsets is not valid from this MONSTER bundle because explicit baseline/rest windows are not identifiable. The labels available in `STEW_y.npy` are low/high workload labels derived from workload ratings, not a verified rest/task condition variable.

Therefore, the bundle is not even sufficient for a baseline-partition sensitivity analysis unless additional source metadata are obtained that identify which rows correspond to baseline/rest recordings.

## Required Next Data Source

To support valid STEW baseline-relative testing, obtain the original IEEE DataPort STEW archive or equivalent source metadata that provides:

- raw or source time-resolved EEG recordings;
- subject IDs;
- explicit rest/baseline and workload/task condition labels;
- channel names/order;
- sampling rate;
- enough recording order or start-time provenance to create non-overlapping calibration and scored-rest windows;
- preprocessing/segmentation documentation if using processed windows.

## Protocol Decision

`STEW_LOCKED_REPLICATION_PROTOCOL_BEFORE_MODELING.md` was not created because the MONSTER bundle does not pass the provenance and construct-match gate for locked replication.

Final verdict: `STEW_MONSTER_INSUFFICIENT_REQUIRE_IEEE_DATAPORT_RAW_ARCHIVE`
