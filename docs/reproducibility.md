# Reproducibility Notes

## Dataset

The analysis uses the PhysioNet EEG During Mental Arithmetic Tasks v1.0.0
dataset. Raw EDF files are not stored in this repository. Download them from:

https://physionet.org/content/eegmat/1.0.0/

The pipeline expects files named like `Subject00_1.edf` and `Subject00_2.edf`.
The `_1` files are baseline/rest recordings and the `_2` files are mental
arithmetic workload recordings.

## Exact command used for the committed outputs

```powershell
python run_pipeline.py `
  --data_dir "C:\Users\abhij\Downloads\eeg-during-mental-arithmetic-tasks-1.0.0\eeg-during-mental-arithmetic-tasks-1.0.0" `
  --output_dir outputs `
  --window_seconds 4 `
  --overlap 0.5 `
  --n_boot 500
```

## Generated analysis artifacts

- `outputs/features/eeg_features.csv`: extracted window-level feature table.
- `outputs/statistics/feature_stat_tests.csv`: paired feature-level statistics.
- `outputs/statistics/top_significant_features.csv`: top FDR-ranked features.
- `outputs/models/metrics_holdout.csv`: subject-group holdout metrics.
- `outputs/models/metrics_loso.csv`: leave-one-subject-out metrics.
- `outputs/figures/`: figures used by the manuscript.

## Primary result

The primary generalization estimate is leave-one-subject-out cross-validation,
because random window-level splits can leak subject-specific EEG structure
between training and test sets.
