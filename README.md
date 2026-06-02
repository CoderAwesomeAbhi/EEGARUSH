# EEG Cognitive State Classification — Multi-Dataset Replication

Reproducible multi-dataset machine-learning analysis of rest versus mental-arithmetic
workload using three public EEG datasets: **PhysioNet MAT** (N=36), **STEW** (N=48),
and **OpenNeuro DS007262** (N=18), totaling 102 participants.

This repository contains:

- Full EEG feature-extraction and modeling pipeline (MAT-only and multi-dataset).
- Subject-Normalized Workload Axis (SNWA) — an interpretable one-dimensional score.
- Generated features, statistics, models, figures, and tables from all analyses.
- PhD-level statistical audit (DeLong tests, permutation null, bootstrap CIs,
  Bayes factors, power analysis, Brier skill scores, calibration).
- Journal-grade upgrade outputs (ablation, negative controls, calibration,
  leakage analysis, nested feature selection).
- A bioRxiv-oriented manuscript in `paper/tex/main.tex`.

The project is intentionally conservative: it evaluates subject-wise
generalization and does **not** claim diagnosis, mind reading, clinical triage,
or deployment readiness.

## Main Result

The primary evaluation is leave-one-subject-out cross-validation (LOSO), which
tests generalization to unseen participants.

| Model | Accuracy | Sensitivity | Specificity | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| SVM RBF | 0.734 | 0.567 | 0.791 | 0.519 | 0.796 | 0.545 |
| Logistic Regression | 0.719 | 0.631 | 0.749 | 0.533 | 0.763 | 0.466 |
| XGBoost | 0.723 | 0.483 | 0.805 | 0.469 | 0.761 | 0.430 |
| Gradient Boosting | 0.727 | 0.453 | 0.820 | 0.456 | 0.764 | 0.426 |
| Random Forest | 0.741 | 0.173 | 0.934 | 0.253 | 0.760 | 0.451 |

SNWA with K=8 features achieves ROC-AUC 0.761, approaching full SVM performance
with only 8 interpretable features.

## Repository Map

```text
.
├── requirements.txt                # Pinned Python dependencies
├── environment.yml                 # Conda environment
├── Dockerfile / Makefile           # Containerized reproduction
├── SHA256SUMS.txt                  # File integrity checksums
├── paper/
│   ├── tex/main.tex                # LaTeX manuscript source
│   ├── pdf/main.pdf                # Compiled manuscript
│   ├── figures/                    # Publication figures
│   └── archive/                    # Manuscript drafts
├── src/eeg_cogstates/              # Core library
├── scripts/
│   ├── run_pipeline.py             # End-to-end analysis
│   ├── run_all_phd_revision_tests.py   # PhD statistical audit
│   ├── run_all_journal_upgrade.py      # Journal-grade outputs
│   ├── finish_everything.py        # Completes remaining tasks
│   ├── download_physionet_eegmat.py
│   ├── multi_dataset_pipeline.py
│   ├── smoke_test_synthetic.py
│   └── realtime_bci_demo.py
├── outputs/                        # Original MAT-only run
├── outputs_reproduced/             # Reproduced baseline + features
│   ├── features/eeg_features.csv
│   ├── models/                     # LOSO + holdout predictions
│   ├── statistics/
│   └── figures/
├── outputs_journal_upgrade/        # Ablation, SNWA, negative controls
│   ├── tables/                     # 19 analysis tables
│   └── figures/                    # 15 publication figures
├── outputs_phd_revision/           # Statistical audit outputs
│   ├── tables/                     # CF2, CF3, SR1-SR12, EV8, FE6
│   └── figures/                    # Calibration, ROC, scatter
├── results/multi_dataset/          # MAT vs STEW vs DS007262
├── external_validation_ds007262/   # DS007262 external preds
├── external_data/                  # Raw external datasets
├── docs/                           # Reproducibility, pre-analysis plan
├── report_templates/
├── tests/                          # Unit tests for feature extraction
└── notebooks/                      # Jupyter exploration
```

## Dataset

Raw EEG files are not committed to this repository. Datasets are available at:
- **PhysioNet MAT**: https://physionet.org/content/eegmat/1.0.0/
- **STEW**: https://huggingface.co/datasets/monster-monash/STEW
- **OpenNeuro DS007262**: https://doi.org/10.18112/openneuro.ds007262.v1.0.6

## Installation

```powershell
pip install -r requirements.txt
# or
conda env create -f environment.yml
```

## Reproduce Analyses

```powershell
# Full PhD revision test suite (statistical audit)
python scripts/run_all_phd_revision_tests.py

# Journal-grade upgrade (ablation, SNWA, controls)
python scripts/run_all_journal_upgrade.py

# Finish all remaining tasks
python scripts/finish_everything.py
```

## Manuscript

Build the manuscript at `paper/tex/main.tex`:

```powershell
cd paper
pdflatex tex/main.tex
pdflatex tex/main.tex
```

Output: `paper/pdf/main.pdf`

## Citation

If you use this repository, cite the code and original datasets:

Zyma I, Tukaev S, Seleznov I, Kiyono K, Popov A, Chernykh M, Shpenkov O.
Electroencephalograms during Mental Arithmetic Task Performance.
*Data*. 2019;4(1):14. doi:10.3390/data4010014.

## License

MIT License. Raw datasets are governed by their own licenses.
