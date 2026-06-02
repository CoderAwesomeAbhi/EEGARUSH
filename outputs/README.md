# Output Artifacts

These outputs were generated from the full PhysioNet EEGMAT run using
4-second windows, 50% overlap, connectivity features enabled, and 2000
subject-bootstrap repetitions.

**Hardware:** Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz, 16 GB RAM
**Runtime:** ~90 minutes for full pipeline
**Python:** 3.11.9, scikit-learn 1.5.1, MNE 1.8.0

Primary files:

- `features/eeg_features.csv`: 4266 windows x 812 feature columns
  - SHA-256: `7cd2cf43edea9ca145c2ca4532712da37a299ef1a67852f1cf9cf5733127233c`
- `models/metrics_loso.csv`: main subject-wise model evaluation with 95% CIs.
- `models/metrics_holdout.csv`: grouped subject holdout evaluation.
- `statistics/top_significant_features.csv`: FDR-ranked paired feature tests.
- `figures/`: manuscript-ready result figures (300 DPI).

Raw EDF files are intentionally excluded from the repository. Reproduce these
outputs by downloading the dataset from PhysioNet and running `run_pipeline.py`
as described in `docs/reproducibility.md`.
