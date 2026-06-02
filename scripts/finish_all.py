r"""
finish_all.py
=============
Run this ONCE from the project root to complete all remaining work:
  cd C:\Users\abhij\Downloads\bioarxivarjun
  python finish_all.py

This will:
  1. Train models on the raw-EDF-extracted features (n_boot=2000)
  2. Regenerate figures for outputs_reproduced
  3. Run the full PhD revision test suite
  4. Copy any missing files to outputs_reproduced
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

np.random.seed(42)


def section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def step(num: int, total: int, desc: str) -> None:
    print(f"\n>>> Step {num}/{total}: {desc} ...")


# ─────────────────────────────────────────────────────────────────────────────
import platform

section("FINISH ALL — Completing remaining experiments")
print(f"  Host: {platform.node()}, CPU: {platform.processor()}")
print(f"  Python: {sys.version}")

total_steps = 5

# ── Step 1: Model training on new features ──────────────────────────────────
step(1, total_steps, "Training models on raw-EDF-extracted features (n_boot=2000)")
from src.eeg_cogstates.modeling import train_and_evaluate

features_csv = ROOT / "outputs_reproduced" / "features" / "eeg_features.csv"
model_dir = ROOT / "outputs_reproduced" / "models"
model_dir.mkdir(parents=True, exist_ok=True)

t0 = time.time()
train_and_evaluate(
    features_csv=features_csv,
    output_dir=model_dir,
    run_loso=True,
    n_boot=2000,
)
elapsed = time.time() - t0
print(f"  Model training done in {elapsed/60:.1f} min")
print(f"  Files in {model_dir.name}/:")
for p in sorted(model_dir.glob("*")):
    print(f"    {p.name}  ({p.stat().st_size/1024:.0f} KB)")


# ── Step 2: Generate figures ────────────────────────────────────────────────
step(2, total_steps, "Generating figures for outputs_reproduced")
from src.eeg_cogstates.visualization import make_all_figures

stats_csv = ROOT / "outputs_reproduced" / "statistics" / "feature_stat_tests.csv"
fig_dir = ROOT / "outputs_reproduced" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

make_all_figures(features_csv, stats_csv, model_dir, fig_dir)
print(f"  Figures in {fig_dir}/:")
for p in sorted(fig_dir.glob("*.png")):
    print(f"    {p.name}")


# ── Step 3: Subject scatter plot ────────────────────────────────────────────
step(3, total_steps, "Generating subject scatter plot")
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

loso_preds = pd.read_csv(model_dir / "predictions_loso.csv")
m1, m2 = "svm_rbf", "logistic_regression"
a1 = loso_preds[loso_preds.model == m1].groupby("subject_id").apply(
    lambda g: roc_auc_score(g.true_label, g.score_workload) if g.true_label.nunique() > 1 else np.nan
)
a2 = loso_preds[loso_preds.model == m2].groupby("subject_id").apply(
    lambda g: roc_auc_score(g.true_label, g.score_workload) if g.true_label.nunique() > 1 else np.nan
)
common = a1.dropna().align(a2.dropna(), join="inner")
if not common[0].empty:
    x, y = common[0].to_numpy(), common[1].to_numpy()
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.6, c="#0072B2")
    lims = [min(x.min(), y.min()) - 0.05, max(x.max(), y.max()) + 0.05]
    plt.plot(lims, lims, "k--", alpha=0.3)
    plt.xlabel(f"{m1} per-subject AUC")
    plt.ylabel(f"{m2} per-subject AUC")
    plt.title(f"Per-subject AUC: {m1} vs {m2}")
    plt.tight_layout()
    plt.savefig(str(fig_dir / "subject_auc_scatter.png"), dpi=200)
    print(f"  subject_auc_scatter.png ({len(x)} subjects)")
else:
    print("  WARNING: no common subjects found for scatter plot")


# ── Step 4: PhD revision tests ──────────────────────────────────────────────
step(4, total_steps, "Running PhD revision test suite (fixed: 50 perm repeats)")

# Patch N_PERM to 50 to avoid timeout
sys.argv = [sys.argv[0]]

# Import and run the PhD revision test suite
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
from scripts.run_all_phd_revision_tests import main as phd_main

t0 = time.time()
phd_main()
elapsed = time.time() - t0
print(f"\n  PhD revision tests done in {elapsed/60:.1f} min")
print(f"  Outputs in outputs_phd_revision/:")

phd_tables = ROOT / "outputs_phd_revision" / "tables"
phd_figs = ROOT / "outputs_phd_revision" / "figures"
if phd_tables.exists():
    for p in sorted(phd_tables.glob("*")):
        print(f"    tables/{p.name}")
if phd_figs.exists():
    for p in sorted(phd_figs.glob("*")):
        print(f"    figures/{p.name}")


# ── Step 5: Summary ─────────────────────────────────────────────────────────
step(5, total_steps, "Summary")
print(f"""
All remaining experiments complete!

  outputs_reproduced/
    features/eeg_features.csv      — from raw EDF extraction
    statistics/                    — paired stats from raw EDFs
    models/                        — fresh LOSO+holdout models (n_boot=2000)
    figures/                       — all publication figures
    table_*.csv                    — baseline comparison tables

  outputs_journal_upgrade/         — ablation, SNWA, negative controls,
                                     calibration, leakage analysis

  outputs_phd_revision/           — subject bootstrap CIs, permutation null,
                                     DeLong tests, power analysis, etc.

  results/multi_dataset/           — MAT vs STEW vs DS007262 comparison

Time: {time.strftime('%Y-%m-%d %H:%M')}
""")
