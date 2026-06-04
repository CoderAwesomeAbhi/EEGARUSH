# Raw MAT Duration Interpretation

Verdict: `raw_duration_variability_usable_with_header_driven_windowing`

The simplified descriptor expectation of uniform 180 s rest and 60 s arithmetic is not the exact EDF-header reality. The raw-aware reconstruction therefore uses verified EDF durations as ground truth.

## Verified EDF Facts

### Durations

| index | condition  | count | min | median | max |
| ----- | ---------- | ----- | --- | ------ | --- |
| 0     | arithmetic | 36    | 62  | 62     | 62  |
| 1     | rest       | 36    | 80  | 182    | 188 |

### Sampling Frequency

| index | condition  | count | min | max |
| ----- | ---------- | ----- | --- | --- |
| 0     | arithmetic | 36    | 500 | 500 |
| 1     | rest       | 36    | 500 | 500 |

### Channel Counts

| index | condition  | count | min | max |
| ----- | ---------- | ----- | --- | --- |
| 0     | arithmetic | 36    | 21  | 21  |
| 1     | rest       | 36    | 21  | 21  |

### Header Filter Metadata

| condition  | highpass_values | lowpass_values |
| ---------- | --------------- | -------------- |
| arithmetic | [0.0, 0.5]      | [45.0, 250.0]  |
| rest       | [0.0, 0.5]      | [45.0, 250.0]  |

## Cached Count Explanation

`72` of `72` cached file-level window counts match raw-header-implied 4 s windows with 50% overlap. The unequal scored-rest counts are therefore plausibly explained by genuine/raw-header duration variation rather than by an impossible cached split.

## Signal Quality Inspection

| n_annotations | trailing_flatline_channels | global_flatline_fraction | subject_id | condition  | file            |
| ------------- | -------------------------- | ------------------------ | ---------- | ---------- | --------------- |
| 0             | 0                          | 0.0113916                | Subject00  | rest       | Subject00_1.edf |
| 0             | 0                          | 0.0324204                | Subject00  | arithmetic | Subject00_2.edf |
| 0             | 0                          | 0.0117158                | Subject01  | rest       | Subject01_1.edf |
| 0             | 0                          | 0.0330938                | Subject01  | arithmetic | Subject01_2.edf |
| 0             | 0                          | 0.011676                 | Subject02  | rest       | Subject02_1.edf |
| 0             | 0                          | 0.0322349                | Subject02  | arithmetic | Subject02_2.edf |
| 0             | 0                          | 0.0116979                | Subject03  | rest       | Subject03_1.edf |
| 0             | 0                          | 0.0326986                | Subject03  | arithmetic | Subject03_2.edf |
| 0             | 0                          | 0.0118913                | Subject04  | rest       | Subject04_1.edf |
| 0             | 0                          | 0.0323075                | Subject04  | arithmetic | Subject04_2.edf |
| 0             | 0                          | 0.0115798                | Subject05  | rest       | Subject05_1.edf |
| 0             | 0                          | 0.032489                 | Subject05  | arithmetic | Subject05_2.edf |
| 0             | 0                          | 0.0114823                | Subject06  | rest       | Subject06_1.edf |
| 0             | 0                          | 0.0325736                | Subject06  | arithmetic | Subject06_2.edf |
| 0             | 0                          | 0.0115235                | Subject07  | rest       | Subject07_1.edf |
| 0             | 0                          | 0.0325091                | Subject07  | arithmetic | Subject07_2.edf |
| 0             | 0                          | 0.0113944                | Subject08  | rest       | Subject08_1.edf |
| 0             | 0                          | 0.032614                 | Subject08  | arithmetic | Subject08_2.edf |
| 0             | 0                          | 0.0116924                | Subject09  | rest       | Subject09_1.edf |
| 0             | 0                          | 0.0322994                | Subject09  | arithmetic | Subject09_2.edf |
| 0             | 0                          | 0.00864105               | Subject10  | rest       | Subject10_1.edf |
| 0             | 0                          | 0.032622                 | Subject10  | arithmetic | Subject10_2.edf |
| 0             | 0                          | 0.0116389                | Subject11  | rest       | Subject11_1.edf |
| 0             | 0                          | 0.0326059                | Subject11  | arithmetic | Subject11_2.edf |
| 0             | 0                          | 0.0115413                | Subject12  | rest       | Subject12_1.edf |
| 0             | 0                          | 0.0325252                | Subject12  | arithmetic | Subject12_2.edf |
| 0             | 0                          | 0.0117075                | Subject13  | rest       | Subject13_1.edf |
| 0             | 0                          | 0.0326341                | Subject13  | arithmetic | Subject13_2.edf |
| 0             | 0                          | 0.0117927                | Subject14  | rest       | Subject14_1.edf |
| 0             | 0                          | 0.032868                 | Subject14  | arithmetic | Subject14_2.edf |
| 0             | 0                          | 0.0115042                | Subject15  | rest       | Subject15_1.edf |
| 0             | 0                          | 0.0325333                | Subject15  | arithmetic | Subject15_2.edf |
| 0             | 0                          | 0.011485                 | Subject16  | rest       | Subject16_1.edf |
| 0             | 0                          | 0.032489                 | Subject16  | arithmetic | Subject16_2.edf |
| 0             | 0                          | 0.0114589                | Subject17  | rest       | Subject17_1.edf |
| 0             | 0                          | 0.0326341                | Subject17  | arithmetic | Subject17_2.edf |
| 0             | 0                          | 0.0115633                | Subject18  | rest       | Subject18_1.edf |
| 0             | 0                          | 0.0323277                | Subject18  | arithmetic | Subject18_2.edf |
| 0             | 0                          | 0.0113161                | Subject19  | rest       | Subject19_1.edf |
| 0             | 0                          | 0.032489                 | Subject19  | arithmetic | Subject19_2.edf |
| 0             | 0                          | 0.011452                 | Subject20  | rest       | Subject20_1.edf |
| 0             | 0                          | 0.0323599                | Subject20  | arithmetic | Subject20_2.edf |
| 0             | 0                          | 0.0114891                | Subject21  | rest       | Subject21_1.edf |
| 0             | 0                          | 0.032368                 | Subject21  | arithmetic | Subject21_2.edf |
| 0             | 0                          | 0.0113916                | Subject22  | rest       | Subject22_1.edf |
| 0             | 0                          | 0.0325252                | Subject22  | arithmetic | Subject22_2.edf |
| 0             | 0                          | 0.0112707                | Subject23  | rest       | Subject23_1.edf |
| 0             | 0                          | 0.0326986                | Subject23  | arithmetic | Subject23_2.edf |
| 0             | 0                          | 0.0115125                | Subject24  | rest       | Subject24_1.edf |
| 0             | 0                          | 0.0324365                | Subject24  | arithmetic | Subject24_2.edf |
| 0             | 0                          | 0.0113147                | Subject25  | rest       | Subject25_1.edf |
| 0             | 0                          | 0.0327269                | Subject25  | arithmetic | Subject25_2.edf |
| 0             | 0                          | 0.0115345                | Subject26  | rest       | Subject26_1.edf |
| 0             | 0                          | 0.0328882                | Subject26  | arithmetic | Subject26_2.edf |
| 0             | 0                          | 0.0114411                | Subject27  | rest       | Subject27_1.edf |
| 0             | 0                          | 0.0324486                | Subject27  | arithmetic | Subject27_2.edf |
| 0             | 0                          | 0.0114768                | Subject28  | rest       | Subject28_1.edf |
| 0             | 0                          | 0.0325091                | Subject28  | arithmetic | Subject28_2.edf |
| 0             | 0                          | 0.0113779                | Subject29  | rest       | Subject29_1.edf |
| 0             | 0                          | 0.0324648                | Subject29  | arithmetic | Subject29_2.edf |
| 0             | 0                          | 0.0114383                | Subject30  | rest       | Subject30_1.edf |
| 0             | 0                          | 0.0324607                | Subject30  | arithmetic | Subject30_2.edf |
| 0             | 0                          | 0.0176754                | Subject31  | rest       | Subject31_1.edf |
| 0             | 0                          | 0.0330051                | Subject31  | arithmetic | Subject31_2.edf |
| 0             | 0                          | 0.0114919                | Subject32  | rest       | Subject32_1.edf |
| 0             | 0                          | 0.0326059                | Subject32  | arithmetic | Subject32_2.edf |
| 0             | 0                          | 0.0113669                | Subject33  | rest       | Subject33_1.edf |
| 0             | 0                          | 0.032372                 | Subject33  | arithmetic | Subject33_2.edf |
| 0             | 0                          | 0.0113765                | Subject34  | rest       | Subject34_1.edf |
| 0             | 0                          | 0.0325293                | Subject34  | arithmetic | Subject34_2.edf |
| 0             | 0                          | 0.0116031                | Subject35  | rest       | Subject35_1.edf |
| 0             | 0                          | 0.0324849                | Subject35  | arithmetic | Subject35_2.edf |

## Usability

The raw EDF files remain usable for rebuilding analyses if all splits are driven by exact EDF durations and every window is provenance-labeled. Duration variability is not treated as fatal, but it prevents using descriptor-level durations as ground truth.
