# Reproducibility Status

## Current Status

The committed feature table and baseline outputs are present. The raw PhysioNet EDF files are not committed, but a local EDF dataset copy was detected during this audit.

## Reproduce Baseline From Raw EDF Files

```powershell
pip install -r requirements.txt
python run_pipeline.py --data_dir C:\Users\abhij\Downloads\eeg-during-mental-arithmetic-tasks-1.0.0 --output_dir outputs_reproduced_raw --window_seconds 4 --overlap 0.5 --n_boot 500
```

If the dataset is nested one level deeper after unzipping, use the nested folder containing `Subject00_1.edf` and `Subject00_2.edf`.

## Reproduce Journal/ISEF Upgrade Outputs

```powershell
python run_all_journal_upgrade.py
```

This command uses `outputs/features/eeg_features.csv` and writes outputs to `outputs_reproduced/`, `outputs_journal_upgrade/`, and named markdown reports in the repository root.

## Exact Feature Table Used

- `outputs/features/eeg_features.csv`: 4266 rows, 812 columns, 805 numeric features.

## What Is Not Automatically Reproduced

External validation is documented as a future protocol unless a second public EEG dataset is downloaded and mapped into this repository.