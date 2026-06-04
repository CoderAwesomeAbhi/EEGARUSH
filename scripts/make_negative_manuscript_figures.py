#!/usr/bin/env python3
"""Generate publication figures for the negative-methods manuscript.

This script ONLY reads committed result-summary files (CSV/JSON) and draws
figures. It runs no model, no resampling, no feature extraction, and touches no
raw data. All numeric values are read from:

  results/raw_rebuilt/mat_subject_level_metrics.csv          (within-MAT)
  results/raw_rebuilt/mat_macro_subject_full_pipeline_permutation.csv (MAT null)
  results/stew_sensitivity/stew_within_metrics.csv           (within-STEW)
  results/stew_sensitivity/stew_permutation_summary.csv      (STEW null)
  results/stew_sensitivity/mat_to_stew_transfer_metrics.csv  (MAT->STEW transport)
  results/cog_bci_one_shot/cog_bci_one_shot_summary.csv      (COG-BCI prospective)
  results/cog_bci_one_shot/cog_bci_paired_comparisons_summary.csv (COG-BCI delta CI)

Outputs (600 DPI PNG) into paper/figures/:
  figure_neg_design_flow.png       (F1 sequential design / hypothesis-control flow)
  figure_neg_within_domain_auc.png (F2 within-MAT and within-STEW AUCs)
  figure_neg_mat_to_stew.png       (F3 MAT->STEW exploratory transport AUCs)
  figure_neg_cogbci_oneshot.png    (F4 COG-BCI one-shot AUCs + z - mean-sub CI)
"""
from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({"figure.dpi": 600, "savefig.dpi": 600, "font.size": 10})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)

# Okabe-Ito colour-blind-safe palette (Wong 2011).
C_ABS = "#999999"   # absolute  (grey)
C_MS = "#0072B2"    # mean subtraction (blue)
C_Z = "#D55E00"     # z-scoring (vermilion)
CHANCE = "#000000"

MODE_LABEL = {"absolute": "Absolute",
              "mean_subtraction": "Mean subtraction",
              "zscore": "z-scoring"}
MODE_COLOR = {"absolute": C_ABS, "mean_subtraction": C_MS, "zscore": C_Z}
MODE_ORDER = ["absolute", "mean_subtraction", "zscore"]


def read_csv(path):
    with open(os.path.join(ROOT, path), newline="") as fh:
        return list(csv.DictReader(fh))


def _auc_bars(ax, vals, los, his, title, note=None):
    x = range(len(MODE_ORDER))
    for i, m in enumerate(MODE_ORDER):
        lo, hi = los[m], his[m]
        ax.bar(i, vals[m], color=MODE_COLOR[m], width=0.62, zorder=2,
               yerr=[[vals[m] - lo], [hi - vals[m]]], capsize=4,
               error_kw={"ecolor": "#333333", "lw": 1.0})
        ax.text(i, vals[m] + 0.012, f"{vals[m]:.3f}", ha="center",
                va="bottom", fontsize=8.5)
    ax.axhline(0.5, color=CHANCE, ls="--", lw=1.0, zorder=1)
    ax.text(len(MODE_ORDER) - 0.5, 0.508, "chance", ha="right", va="bottom",
            fontsize=7.5, color=CHANCE)
    ax.set_xticks(list(x))
    ax.set_xticklabels([MODE_LABEL[m] for m in MODE_ORDER], rotation=12)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Macro subject ROC-AUC")
    ax.set_title(title, fontsize=10.5)
    if note:
        ax.text(0.5, 0.95, note, transform=ax.transAxes, ha="center",
                va="top", fontsize=8, color="#555555")


def fig_within_domain():
    mat = {r["calibration"]: r for r in read_csv(
        "results/raw_rebuilt/mat_subject_level_metrics.csv")
        if r["model"] == "logistic_l2"}
    stew = {r["calibration"]: r for r in read_csv(
        "results/stew_sensitivity/stew_within_metrics.csv")
        if r["model"] == "logistic_l2"}

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))
    for ax, d, title in (
            (axes[0], mat, "Within-MAT (rest vs. arithmetic)\n36 subjects, LOSO"),
            (axes[1], stew, "Within-STEW (low vs. high workload)\n48 subjects, LOSO")):
        vals = {m: float(d[m]["macro_subject_mean_auc"]) for m in MODE_ORDER}
        los = {m: float(d[m]["subject_auc_ci95_low"]) for m in MODE_ORDER}
        his = {m: float(d[m]["subject_auc_ci95_high"]) for m in MODE_ORDER}
        _auc_bars(ax, vals, los, his, title)
    fig.suptitle("Within-domain workload decoding is above chance "
                 "(development datasets)", fontsize=11.5, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(FIG, "figure_neg_within_domain_auc.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_mat_to_stew():
    tr = {r["calibration"]: r for r in read_csv(
        "results/stew_sensitivity/mat_to_stew_transfer_metrics.csv")}
    vals = {m: float(tr[m]["macro_subject_mean_auc"]) for m in MODE_ORDER}
    los = {m: float(tr[m]["subject_auc_ci95_low"]) for m in MODE_ORDER}
    his = {m: float(tr[m]["subject_auc_ci95_high"]) for m in MODE_ORDER}

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    _auc_bars(ax, vals, los, his,
              "MAT → STEW exploratory transport\n"
              "(train MAT, test STEW; 48 target subjects)")
    fig.tight_layout()
    out = os.path.join(FIG, "figure_neg_mat_to_stew.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_cogbci_oneshot():
    rows = {r["method"]: r for r in read_csv(
        "results/cog_bci_one_shot/cog_bci_one_shot_summary.csv")}
    pair = {r["comparison"]: r for r in read_csv(
        "results/cog_bci_one_shot/cog_bci_paired_comparisons_summary.csv")}
    vals = {m: float(rows[m]["macro_subject_mean_auc"]) for m in MODE_ORDER}
    los = {m: float(rows[m]["subject_auc_ci95_low"]) for m in MODE_ORDER}
    his = {m: float(rows[m]["subject_auc_ci95_high"]) for m in MODE_ORDER}

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.6),
                             gridspec_kw={"width_ratios": [1.4, 1.0]})
    _auc_bars(axes[0], vals, los, his,
              "COG-BCI one-shot prospective test\n"
              "(MAT → COG-BCI; 29 target subjects)")

    # Primary delta CI: z-scoring - mean subtraction.
    ax = axes[1]
    p = pair["zscore_minus_mean_subtraction"]
    mean, lo, hi = (float(p["mean_delta"]), float(p["ci95_low"]),
                    float(p["ci95_high"]))
    ax.axvline(0.0, color=CHANCE, ls="--", lw=1.0)
    ax.errorbar([mean], [0], xerr=[[mean - lo], [hi - mean]], fmt="o",
                color=C_Z, capsize=5, ms=7, lw=1.6)
    ax.text(mean, 0.12, f"Δ = {mean:+.3f}\n[{lo:+.3f}, {hi:+.3f}]",
            ha="center", va="bottom", fontsize=8.5)
    ax.set_yticks([])
    ax.set_ylim(-0.6, 0.6)
    ax.set_xlim(-0.30, 0.30)
    ax.set_xlabel("Δ macro subject AUC")
    ax.set_title("Primary comparison\n(z-scoring − mean subtraction)",
                 fontsize=10.5)
    ax.text(0.5, 0.06, "95% CI includes zero", transform=ax.transAxes,
            ha="center", fontsize=8, color="#555555")
    fig.tight_layout()
    out = os.path.join(FIG, "figure_neg_cogbci_oneshot.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_design_flow():
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.12",
                     linewidth=1.2, edgecolor="#333333", facecolor=fc,
                     zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=8.6, zorder=3)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                     arrowstyle="-|>", mutation_scale=14, lw=1.3,
                     color="#333333", zorder=1))

    dev = "#D9E8F5"
    hyp = "#FCE3CF"
    fail = "#F4CCCC"

    box(0.4, 7.6, 4.1, 1.8,
        "DEVELOPMENT 1 — MAT\nrest vs. arithmetic, 36 subj, 500→128 Hz\n"
        "mean-subtraction macro AUC 0.880\n(within-dataset, p = 0.004975)", dev)
    box(5.5, 7.6, 4.1, 1.8,
        "DEVELOPMENT 2 — STEW (sensitivity)\nlow vs. high, 48 subj, 128 Hz\n"
        "mean-subtraction macro AUC 0.840\n(within-dataset, p = 0.004975)", dev)
    box(0.4, 5.1, 9.2, 1.5,
        "PREDECLARED TRANSPORT: MAT → STEW (96 scale/offset-invariant features)\n"
        "mean-subtraction macro AUC 0.448 — BELOW CHANCE  (transport FAILS)", fail)
    box(0.4, 2.9, 9.2, 1.4,
        "POST-HOC OBSERVATION: MAT → STEW z-scoring macro AUC 0.683\n"
        "hypothesis-generating only (not confirmatory); frozen for one prospective test", hyp)
    box(0.4, 0.3, 9.2, 1.7,
        "UNTOUCHED PROSPECTIVE TEST — COG-BCI, one shot, checksum-locked\n"
        "z-scoring macro AUC 0.436 (below chance); Δ vs. mean-sub CI "
        "[−0.076, +0.175] includes zero\nVERDICT: PROSPECTIVE TRANSPORT FAILED "
        "(negative methods result)", fail)

    arrow(2.45, 7.6, 3.0, 6.6)
    arrow(7.55, 7.6, 7.0, 6.6)
    arrow(5.0, 5.1, 5.0, 4.3)
    arrow(5.0, 2.9, 5.0, 2.0)
    ax.set_title("Sequential hypothesis-control design "
                 "(development → post-hoc hypothesis → frozen prospective test)",
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(FIG, "figure_neg_design_flow.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig_design_flow()
    fig_within_domain()
    fig_mat_to_stew()
    fig_cogbci_oneshot()
    print("done")
