# Baseline Comparison

The baseline reproduction uses the committed real feature table and committed subject-wise prediction artifacts.
The files were copied into `outputs_reproduced/` and summary tables/figures were regenerated from those predictions.

- Feature table shape: 4266 windows x 812 columns.
- Subjects: 36.
- Numeric feature columns: 805.

The reproduced LOSO and grouped-holdout metric CSV files match the committed baseline artifacts byte-for-byte for copied metric tables.
For a full raw-EDF rerun, use the command in `REPRODUCIBILITY_STATUS.md`.