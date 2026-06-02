# Audit Report

Repository: `C:\Users\abhij\Downloads\bioarxivarjun\scripts`
Audit time: 2026-05-28 23:09:32

## Existing Structure

- Feature CSV: `outputs/features/eeg_features.csv`
- Existing model outputs: `outputs/models/`
- Existing statistics outputs: `outputs/statistics/`
- Existing figures: `outputs/figures/`
- Manuscript files: `paper/main.tex`, `paper/main.pdf`
- Main baseline command: `python run_pipeline.py --data_dir <dataset> --output_dir outputs --window_seconds 4 --overlap 0.5 --n_boot 500`

## Data Inventory

- Feature table: 4266 windows x 812 columns.
- Numeric model features: 805.
- Subjects in feature table: 36.
- Class balance: {0: 3186, 1: 1080} where 0=rest and 1=workload.
- EDF files found near the repo: 0.

## Python Environment

- Python: 3.11.9
- numpy: 2.4.1
- pandas: 3.0.2
- sklearn: 1.8.0
- scipy: 1.17.1
- matplotlib: 3.10.9
- mne: 1.12.1

## Command Checks

- `C:\Users\abhij\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe --version` -> exit code 0
  Output tail:
  ```
Python 3.11.9
  ```
- `C:\Users\abhij\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe scripts/smoke_test_synthetic.py` -> exit code 2
  Output tail:
  ```
C:\Users\abhij\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe: can't open file 'C:\\Users\\abhij\\Downloads\\bioarxivarjun\\scripts\\scripts\\smoke_test_synthetic.py': [Errno 2] No such file or directory
  ```

## Missing or Weak Items Found

- The original project did not include a formal leakage theorem or empirical leakage probability table.
- The original project reported useful LOSO metrics but did not yet include nested feature stability, SNWA, ablations, negative controls, or calibration reliability.
- External validation is not automatically run in this repository because no second dataset is bundled.

## Fixes Added by the Journal Upgrade

- Added `run_all_journal_upgrade.py` as a single command to regenerate the new tables, figures, and reports from the committed feature table.
- Added leakage probability analysis, random-window cautionary comparison, SNWA, nested stability, ablations, negative controls, calibration, confidence intervals, ISEF materials, and a journal-level manuscript draft.