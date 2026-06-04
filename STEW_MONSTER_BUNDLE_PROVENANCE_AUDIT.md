# STEW MONSTER Bundle Provenance Audit

## Scope

Audited local bundle:

`data/raw/stew/monster-monash_STEW/`

No models were run. No feature extraction was performed. No manuscript files were edited.

## File Structure And Formats

The bundle contains 11 files:

- `README.md`: Hugging Face dataset card.
- `STEW.py`: Hugging Face loading script.
- `STEW_X.npy`: NumPy array of EEG time-series windows.
- `STEW_y.npy`: NumPy array of binary labels.
- `STEW_subject_id.csv`: one subject identifier per example.
- `STEW_metadata.npy`: object dictionary containing `subject_id`.
- `test_indices_fold_0.txt` through `test_indices_fold_4.txt`: fold index files.

Detailed filenames, sizes, hashes, file types, and roles are in:

`results/stew_provenance/stew_monster_bundle_manifest.csv`

## Signal Versus Features

The bundle contains time-resolved signal windows, not engineered features.

- `STEW_X.npy` shape: `(28512, 14, 256)`.
- `STEW_X.npy` dtype: `float32`.
- Interpretation: examples x channels x samples.
- Window duration implied by README: 256 samples / 128 Hz = 2 seconds.

This is a processed-window representation, not the original continuous IEEE DataPort recording archive.

## Tensor And Table Dimensions

- `STEW_X.npy`: `(28512, 14, 256)`.
- `STEW_y.npy`: `(28512,)`.
- `STEW_subject_id.csv`: `(28512, 1)`.
- `STEW_metadata.npy`: zero-dimensional object array containing a dictionary with one key, `subject_id`, whose value has shape `(28512,)`.

Schema details are in:

`results/stew_provenance/stew_example_schema.csv`

## Channel Names And Ordering

The MONSTER bundle documents:

- number of channels: `14`;
- sampling frequency: `128 Hz`;
- window length: `256`.

The downloaded bundle does not encode channel names or channel order inside `STEW_X.npy`, `STEW_metadata.npy`, `STEW_subject_id.csv`, or `STEW_y.npy`.

The known STEW/Emotiv EPOC montage used by existing repository code is:

`AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4`

However, because the MONSTER bundle itself does not store channel labels/order per array axis, this must be treated as documentation-dependent rather than self-verifying provenance.

## Sampling Frequency

Sampling frequency is documented in `README.md` as `128 Hz`.

It is not encoded as machine-readable metadata in the NumPy arrays.

## Window Length And Overlap

- Window length: `256` samples.
- Implied duration: `2` seconds at `128 Hz`.
- Window overlap: not documented in the downloaded bundle.
- Source start times: not present.
- Trial IDs/session IDs: not present.

The README states that the processed dataset consists of 28,512 multivariate time series, each length 256. It does not document whether these windows overlap or how they were segmented from continuous EEG.

## Subject Identifiers

Subject identity exists for every example.

- Number of examples: `28512`.
- Number of subjects: `48`.
- Rows per subject: `594`.
- Label rows per subject: `297` for label `0` and `297` for label `1`.

The subject identifiers in `STEW_subject_id.csv` and `STEW_metadata.npy` are consistent in shape and purpose.

Subject/label details are in:

`results/stew_provenance/stew_label_subject_condition_summary.csv`

## Condition Labels

The bundle has binary labels in `STEW_y.npy`:

- label `0`: `14256` examples;
- label `1`: `14256` examples.

The README defines these labels as workload-rating classes:

- workload ratings above `4` are assigned to "high";
- workload ratings below or equal to `4` are assigned to "low".

Therefore, `STEW_y.npy` is not documented as a rest/workload label. The bundle documentation says baseline rest was recorded in the original STEW dataset, but the processed MONSTER files do not expose an explicit baseline/rest condition field.

## Temporal Provenance

Temporal provenance is insufficient for a MAT-comparable baseline-relative protocol.

Missing from the bundle:

- original chronological order within each subject and condition;
- window start time;
- sample start/end index;
- source recording filename;
- trial/session ID;
- explicit baseline/rest versus task/workload condition field;
- segmentation or overlap metadata.

Because these fields are absent, calibration windows cannot be separated from scored-rest windows without inventing temporal order or condition structure.

## Processing Provenance

The MONSTER README documents that the data are a processed multivariate time-series dataset. The local bundle does not document enough preprocessing detail to determine:

- filtering;
- artifact cleaning;
- normalization or standardization;
- overlap/stride;
- segmentation rule;
- exact mapping from original IEEE recordings to processed windows;
- whether baseline windows were retained, relabeled, balanced, shuffled, or mixed with workload windows.

These gaps prevent a responsible claim that MONSTER STEW is equivalent to the original raw IEEE DataPort STEW archive for MAT-comparable baseline-relative testing.

## Licensing And Source Relationship

- MONSTER/Hugging Face card license: CC BY 4.0.
- Official STEW dataset citation: Lim, Sourina, Wang, IEEE TNSRE 2018.
- Official host: IEEE DataPort, DOI `10.21227/44r8-ya50`.
- The MONSTER bundle cites IEEE DataPort and says STEW can be accessed upon request through IEEE DataPort.

The MONSTER bundle is useful as a processed time-series source, but it is not automatically equivalent to the original continuous/raw source archive.

## Audit Conclusion

The obtained MONSTER/Hugging Face STEW bundle contains time-resolved EEG windows with valid per-example subject IDs and binary low/high workload labels. It does not expose verified baseline/rest labels, original temporal order, source-window start times, channel labels encoded in the files, or sufficient preprocessing provenance.

It cannot support a locked MAT-style baseline-relative STEW replication without the original IEEE DataPort raw/source archive or additional provenance that maps processed windows to rest/task source segments.
