# Raw MAT Provenance Report

Verdict: `RAW_MAT_PROVENANCE_FAILURE_REBUILD_REQUIRED`

## Scope

- Dataset: PhysioNet EEG During Mental Arithmetic Tasks v1.0.0.
- Raw input directory: `data/raw/eegmat/`.
- Manuscript file was not modified.
- DS007262 and external confirmation search were not used.

## Official Expectations Tested

- Subjects: `36`.
- EDF files: `72`.
- Sampling frequency: `500.0` Hz.
- Descriptor durations: rest `180.0` s, arithmetic `60.0` s.
- Header filter metadata: high-pass `0.5` Hz, low-pass `45.0` Hz, notch `50` Hz if present in EDF metadata.
- Condition mapping: `_1` = rest/background, `_2` = mental arithmetic.
- Channel identity: expected 21-channel EDF header set used by the cached feature table.

## Header Summary

### Sampling Frequency

| condition  | count | min | max |
| ---------- | ----- | --- | --- |
| arithmetic | 36    | 500 | 500 |
| rest       | 36    | 500 | 500 |

### Duration

| condition  | count | min | median | max |
| ---------- | ----- | --- | ------ | --- |
| arithmetic | 36    | 62  | 62     | 62  |
| rest       | 36    | 80  | 182    | 188 |

### EDF Header Filter Metadata

| condition  | highpass_min_hz | highpass_max_hz | lowpass_min_hz | lowpass_max_hz |
| ---------- | --------------- | --------------- | -------------- | -------------- |
| arithmetic | 0               | 0.5             | 45             | 250            |
| rest       | 0               | 0.5             | 45             | 250            |

### Duration Mismatches

| subject_id | condition  | filename        | duration_sec | expected_duration_sec | expected_4s_50pct_windows_from_header |
| ---------- | ---------- | --------------- | ------------ | --------------------- | ------------------------------------- |
| Subject00  | rest       | Subject00_1.edf | 182          | 180                   | 90                                    |
| Subject00  | arithmetic | Subject00_2.edf | 62           | 60                    | 30                                    |
| Subject01  | rest       | Subject01_1.edf | 182          | 180                   | 90                                    |
| Subject01  | arithmetic | Subject01_2.edf | 62           | 60                    | 30                                    |
| Subject02  | rest       | Subject02_1.edf | 182          | 180                   | 90                                    |
| Subject02  | arithmetic | Subject02_2.edf | 62           | 60                    | 30                                    |
| Subject03  | rest       | Subject03_1.edf | 182          | 180                   | 90                                    |
| Subject03  | arithmetic | Subject03_2.edf | 62           | 60                    | 30                                    |
| Subject04  | rest       | Subject04_1.edf | 170          | 180                   | 84                                    |
| Subject04  | arithmetic | Subject04_2.edf | 62           | 60                    | 30                                    |
| Subject05  | rest       | Subject05_1.edf | 182          | 180                   | 90                                    |
| Subject05  | arithmetic | Subject05_2.edf | 62           | 60                    | 30                                    |
| Subject06  | rest       | Subject06_1.edf | 182          | 180                   | 90                                    |
| Subject06  | arithmetic | Subject06_2.edf | 62           | 60                    | 30                                    |
| Subject07  | rest       | Subject07_1.edf | 182          | 180                   | 90                                    |
| Subject07  | arithmetic | Subject07_2.edf | 62           | 60                    | 30                                    |
| Subject08  | rest       | Subject08_1.edf | 182          | 180                   | 90                                    |
| Subject08  | arithmetic | Subject08_2.edf | 62           | 60                    | 30                                    |
| Subject09  | rest       | Subject09_1.edf | 182          | 180                   | 90                                    |
| Subject09  | arithmetic | Subject09_2.edf | 62           | 60                    | 30                                    |
| Subject10  | rest       | Subject10_1.edf | 188          | 180                   | 93                                    |
| Subject10  | arithmetic | Subject10_2.edf | 62           | 60                    | 30                                    |
| Subject11  | rest       | Subject11_1.edf | 182          | 180                   | 90                                    |
| Subject11  | arithmetic | Subject11_2.edf | 62           | 60                    | 30                                    |
| Subject12  | rest       | Subject12_1.edf | 182          | 180                   | 90                                    |
| Subject12  | arithmetic | Subject12_2.edf | 62           | 60                    | 30                                    |
| Subject13  | rest       | Subject13_1.edf | 182          | 180                   | 90                                    |
| Subject13  | arithmetic | Subject13_2.edf | 62           | 60                    | 30                                    |
| Subject14  | rest       | Subject14_1.edf | 182          | 180                   | 90                                    |
| Subject14  | arithmetic | Subject14_2.edf | 62           | 60                    | 30                                    |
| Subject15  | rest       | Subject15_1.edf | 182          | 180                   | 90                                    |
| Subject15  | arithmetic | Subject15_2.edf | 62           | 60                    | 30                                    |
| Subject16  | rest       | Subject16_1.edf | 182          | 180                   | 90                                    |
| Subject16  | arithmetic | Subject16_2.edf | 62           | 60                    | 30                                    |
| Subject17  | rest       | Subject17_1.edf | 182          | 180                   | 90                                    |
| Subject17  | arithmetic | Subject17_2.edf | 62           | 60                    | 30                                    |
| Subject18  | rest       | Subject18_1.edf | 182          | 180                   | 90                                    |
| Subject18  | arithmetic | Subject18_2.edf | 62           | 60                    | 30                                    |
| Subject19  | rest       | Subject19_1.edf | 182          | 180                   | 90                                    |
| Subject19  | arithmetic | Subject19_2.edf | 62           | 60                    | 30                                    |
| Subject20  | rest       | Subject20_1.edf | 182          | 180                   | 90                                    |
| Subject20  | arithmetic | Subject20_2.edf | 62           | 60                    | 30                                    |
| Subject21  | rest       | Subject21_1.edf | 182          | 180                   | 90                                    |
| Subject21  | arithmetic | Subject21_2.edf | 62           | 60                    | 30                                    |
| Subject22  | rest       | Subject22_1.edf | 182          | 180                   | 90                                    |
| Subject22  | arithmetic | Subject22_2.edf | 62           | 60                    | 30                                    |
| Subject23  | rest       | Subject23_1.edf | 182          | 180                   | 90                                    |
| Subject23  | arithmetic | Subject23_2.edf | 62           | 60                    | 30                                    |
| Subject24  | rest       | Subject24_1.edf | 182          | 180                   | 90                                    |
| Subject24  | arithmetic | Subject24_2.edf | 62           | 60                    | 30                                    |
| Subject25  | rest       | Subject25_1.edf | 182          | 180                   | 90                                    |
| Subject25  | arithmetic | Subject25_2.edf | 62           | 60                    | 30                                    |
| Subject26  | rest       | Subject26_1.edf | 182          | 180                   | 90                                    |
| Subject26  | arithmetic | Subject26_2.edf | 62           | 60                    | 30                                    |
| Subject27  | rest       | Subject27_1.edf | 182          | 180                   | 90                                    |
| Subject27  | arithmetic | Subject27_2.edf | 62           | 60                    | 30                                    |
| Subject28  | rest       | Subject28_1.edf | 182          | 180                   | 90                                    |
| Subject28  | arithmetic | Subject28_2.edf | 62           | 60                    | 30                                    |
| Subject29  | rest       | Subject29_1.edf | 182          | 180                   | 90                                    |
| Subject29  | arithmetic | Subject29_2.edf | 62           | 60                    | 30                                    |
| Subject30  | rest       | Subject30_1.edf | 182          | 180                   | 90                                    |
| Subject30  | arithmetic | Subject30_2.edf | 62           | 60                    | 30                                    |
| Subject31  | rest       | Subject31_1.edf | 80           | 180                   | 39                                    |
| Subject31  | arithmetic | Subject31_2.edf | 62           | 60                    | 30                                    |
| Subject32  | rest       | Subject32_1.edf | 182          | 180                   | 90                                    |
| Subject32  | arithmetic | Subject32_2.edf | 62           | 60                    | 30                                    |
| Subject33  | rest       | Subject33_1.edf | 182          | 180                   | 90                                    |
| Subject33  | arithmetic | Subject33_2.edf | 62           | 60                    | 30                                    |
| Subject34  | rest       | Subject34_1.edf | 182          | 180                   | 90                                    |
| Subject34  | arithmetic | Subject34_2.edf | 62           | 60                    | 30                                    |
| Subject35  | rest       | Subject35_1.edf | 182          | 180                   | 90                                    |
| Subject35  | arithmetic | Subject35_2.edf | 62           | 60                    | 30                                    |

## Cached Feature-Table Consistency

- Cached window counts match raw-header-implied 4 s / 50% overlap windows: `True`.
- This checks count consistency only; it does not prove the cached numerical features were regenerated from raw data.

| filename        | duration_sec | expected_4s_50pct_windows_from_header | cached_windows | cached_max_end_sec | cached_matches_raw_header_window_count |
| --------------- | ------------ | ------------------------------------- | -------------- | ------------------ | -------------------------------------- |
| Subject00_1.edf | 182          | 90                                    | 90             | 182                | True                                   |
| Subject00_2.edf | 62           | 30                                    | 30             | 62                 | True                                   |
| Subject01_1.edf | 182          | 90                                    | 90             | 182                | True                                   |
| Subject01_2.edf | 62           | 30                                    | 30             | 62                 | True                                   |
| Subject02_1.edf | 182          | 90                                    | 90             | 182                | True                                   |
| Subject02_2.edf | 62           | 30                                    | 30             | 62                 | True                                   |
| Subject03_1.edf | 182          | 90                                    | 90             | 182                | True                                   |
| Subject03_2.edf | 62           | 30                                    | 30             | 62                 | True                                   |
| Subject04_1.edf | 170          | 84                                    | 84             | 170                | True                                   |
| Subject04_2.edf | 62           | 30                                    | 30             | 62                 | True                                   |
| Subject05_1.edf | 182          | 90                                    | 90             | 182                | True                                   |
| Subject05_2.edf | 62           | 30                                    | 30             | 62                 | True                                   |
| Subject06_1.edf | 182          | 90                                    | 90             | 182                | True                                   |
| Subject06_2.edf | 62           | 30                                    | 30             | 62                 | True                                   |
| Subject07_1.edf | 182          | 90                                    | 90             | 182                | True                                   |
| Subject07_2.edf | 62           | 30                                    | 30             | 62                 | True                                   |
| Subject08_1.edf | 182          | 90                                    | 90             | 182                | True                                   |
| Subject08_2.edf | 62           | 30                                    | 30             | 62                 | True                                   |
| Subject09_1.edf | 182          | 90                                    | 90             | 182                | True                                   |
| Subject09_2.edf | 62           | 30                                    | 30             | 62                 | True                                   |
| Subject10_1.edf | 188          | 93                                    | 93             | 188                | True                                   |
| Subject10_2.edf | 62           | 30                                    | 30             | 62                 | True                                   |
| Subject11_1.edf | 182          | 90                                    | 90             | 182                | True                                   |
| Subject11_2.edf | 62           | 30                                    | 30             | 62                 | True                                   |
| Subject12_1.edf | 182          | 90                                    | 90             | 182                | True                                   |
| Subject12_2.edf | 62           | 30                                    | 30             | 62                 | True                                   |
| Subject13_1.edf | 182          | 90                                    | 90             | 182                | True                                   |
| Subject13_2.edf | 62           | 30                                    | 30             | 62                 | True                                   |
| Subject14_1.edf | 182          | 90                                    | 90             | 182                | True                                   |
| Subject14_2.edf | 62           | 30                                    | 30             | 62                 | True                                   |
| Subject15_1.edf | 182          | 90                                    | 90             | 182                | True                                   |
| Subject15_2.edf | 62           | 30                                    | 30             | 62                 | True                                   |
| Subject16_1.edf | 182          | 90                                    | 90             | 182                | True                                   |
| Subject16_2.edf | 62           | 30                                    | 30             | 62                 | True                                   |
| Subject17_1.edf | 182          | 90                                    | 90             | 182                | True                                   |
| Subject17_2.edf | 62           | 30                                    | 30             | 62                 | True                                   |
| Subject18_1.edf | 182          | 90                                    | 90             | 182                | True                                   |
| Subject18_2.edf | 62           | 30                                    | 30             | 62                 | True                                   |
| Subject19_1.edf | 182          | 90                                    | 90             | 182                | True                                   |
| Subject19_2.edf | 62           | 30                                    | 30             | 62                 | True                                   |
| Subject20_1.edf | 182          | 90                                    | 90             | 182                | True                                   |
| Subject20_2.edf | 62           | 30                                    | 30             | 62                 | True                                   |
| Subject21_1.edf | 182          | 90                                    | 90             | 182                | True                                   |
| Subject21_2.edf | 62           | 30                                    | 30             | 62                 | True                                   |
| Subject22_1.edf | 182          | 90                                    | 90             | 182                | True                                   |
| Subject22_2.edf | 62           | 30                                    | 30             | 62                 | True                                   |
| Subject23_1.edf | 182          | 90                                    | 90             | 182                | True                                   |
| Subject23_2.edf | 62           | 30                                    | 30             | 62                 | True                                   |
| Subject24_1.edf | 182          | 90                                    | 90             | 182                | True                                   |
| Subject24_2.edf | 62           | 30                                    | 30             | 62                 | True                                   |
| Subject25_1.edf | 182          | 90                                    | 90             | 182                | True                                   |
| Subject25_2.edf | 62           | 30                                    | 30             | 62                 | True                                   |
| Subject26_1.edf | 182          | 90                                    | 90             | 182                | True                                   |
| Subject26_2.edf | 62           | 30                                    | 30             | 62                 | True                                   |
| Subject27_1.edf | 182          | 90                                    | 90             | 182                | True                                   |
| Subject27_2.edf | 62           | 30                                    | 30             | 62                 | True                                   |
| Subject28_1.edf | 182          | 90                                    | 90             | 182                | True                                   |
| Subject28_2.edf | 62           | 30                                    | 30             | 62                 | True                                   |
| Subject29_1.edf | 182          | 90                                    | 90             | 182                | True                                   |
| Subject29_2.edf | 62           | 30                                    | 30             | 62                 | True                                   |
| Subject30_1.edf | 182          | 90                                    | 90             | 182                | True                                   |
| Subject30_2.edf | 62           | 30                                    | 30             | 62                 | True                                   |
| Subject31_1.edf | 80           | 39                                    | 39             | 80                 | True                                   |
| Subject31_2.edf | 62           | 30                                    | 30             | 62                 | True                                   |
| Subject32_1.edf | 182          | 90                                    | 90             | 182                | True                                   |
| Subject32_2.edf | 62           | 30                                    | 30             | 62                 | True                                   |
| Subject33_1.edf | 182          | 90                                    | 90             | 182                | True                                   |
| Subject33_2.edf | 62           | 30                                    | 30             | 62                 | True                                   |
| Subject34_1.edf | 182          | 90                                    | 90             | 182                | True                                   |
| Subject34_2.edf | 62           | 30                                    | 30             | 62                 | True                                   |
| Subject35_1.edf | 182          | 90                                    | 90             | 182                | True                                   |
| Subject35_2.edf | 62           | 30                                    | 30             | 62                 | True                                   |

## Gate Decision

`RAW_MAT_PROVENANCE_FAILURE_REBUILD_REQUIRED`

The raw EDF files did not exactly reproduce every expected descriptor duration. Per the hard-stop gate, modeling/rebuild phases must not proceed in this run. The cached feature table is count-consistent with the raw EDF header durations, but the old headline MAT result remains unsuitable for final manuscript claims until the duration discrepancy is explicitly handled in a rebuilt exploratory pipeline.
