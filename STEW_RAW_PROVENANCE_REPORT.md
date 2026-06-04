# STEW Raw Provenance Report

## Verdict

Final STEW provenance status: `STEW_PROVENANCE_UNRESOLVED_CANNOT_SUPPORT_FINAL_PAPER`.

The repository currently contains a cached STEW feature table, not auditable raw EEG or source windows with timing provenance. Because the table lacks source file IDs, sample indices, and window start/end times, a corrected balanced STEW design with non-overlapping calibration/rest scoring windows cannot be verified.

## Audited Input

- Cached feature table: `results/multi_dataset/stew_features.parquet`.
- SHA-256: `0d084bfad1228549e6b470bad0e947b372ff1568e3162aa2a0ba05ad1d550152`.
- Rows: `2000`.
- Subjects: `48`.
- Condition row counts: rest=`984`, workload=`1016`.
- Metadata columns present: `condition, dataset, label, subject_id`.
- Timing/provenance columns present: `none`.
- Channels inferred from cached feature columns: `Af3, Af4, F3, F4, F7, F8, Fc5, Fc6, O1, O2, T3, T4, T5, T6`.

## Local Source Candidate Search

- `results/multi_dataset/biological_stew.csv`
- `results/multi_dataset/metrics_stew.csv`
- `results/multi_dataset/predictions_stew.csv`
- `results/multi_dataset/stew_features.parquet`

The existing repository code documents an intended HuggingFace STEW pathway using `STEW_X.npy`, `STEW_y.npy`, and `STEW_subject_id.csv`; however, those source arrays are not present in the repository audit state. The available cached parquet is therefore the only locally auditable STEW input.

## Baseline And Split Audit

- A rest condition exists in the cached labels.
- Sampling frequency is not verifiable from the cached feature table itself.
- Duration and fixed segment boundaries are not verifiable from the cached feature table.
- Calibration/scoring non-overlap cannot be proven because no source timing or window IDs are present.
- A MAT-consistent split cannot be reconstructed from auditable source inputs in the current repository state.

## Channel Harmonization

- Exact source/cached shared channels: `F3`, `F4`, `F7`, `F8`, `O1`, `O2`.
- Legacy-equivalence channels: source `STEW T7 -> cached/MAT T3`, source `STEW T8 -> cached/MAT T4`.
- No nearest-electrode substitution is used or allowed.
- Channel mapping is documentable, but it is not sufficient to rescue the provenance/timing failure.

## Consequence

Do not run corrected balanced STEW, MAT->STEW transfer, STEW->MAT transfer, or transfer permutation tests from this cached table as final-paper evidence.
