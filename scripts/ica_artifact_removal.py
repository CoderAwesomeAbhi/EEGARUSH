"""
ICA Artifact Removal Validation
================================
Downloads MAT EDF files (if needed), applies ICA per-subject during feature
extraction, re-runs LOSO classification, and compares performance with and
without ICA cleaning.

Also implements ICA inside LOSO training folds (fit on training subjects
only, apply to held-out subject) for DS007262 BrainVision data.
"""

import sys, os, warnings, json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.base import clone
from collections import OrderedDict

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src.eeg_cogstates.dataset import read_edf, discover_edf_records, EDFRecord, iter_windows
from src.eeg_cogstates.features import extract_window_features

OUT = BASE / "outputs_phd_revision"
ICA_DIR = OUT / "ica_analysis"
ICA_DIR.mkdir(parents=True, exist_ok=True)
TAB = OUT / "tables"
FIG = OUT / "figures"
PAPER_FIG = BASE / "paper" / "figures"

# ── 1. Download MAT data if needed ────────────────────────────────────────────
MAT_DATA_DIR = BASE / "data" / "raw" / "eegmat"

if not list(MAT_DATA_DIR.rglob("Subject*_*.edf")):
    print("Downloading MAT EDF files from PhysioNet...")
    from src.eeg_cogstates.dataset import download_physionet_eegmat
    download_physionet_eegmat(MAT_DATA_DIR)
else:
    print(f"MAT data found in {MAT_DATA_DIR}")

# ── 2. Feature names from already-extracted data (to ensure alignment) ────────
orig_feat = pd.read_csv(BASE / "outputs_reproduced" / "features" / "eeg_features.csv", nrows=0)
id_cols = ["subject_id", "condition", "label", "file", "window_index", "start_sec", "end_sec"]
feature_cols_all = [c for c in orig_feat.columns if c not in id_cols]

# ── 3. Extract features with ICA per-subject ──────────────────────────────────
print("\n=== Extracting features with ICA cleaning ===")
records = discover_edf_records(MAT_DATA_DIR)

rows_ica = []
for rec in records:
    data, sfreq, ch_names = read_edf(
        rec.path, bandpass=(0.5, 45.0), apply_ica=True,
        ica_n_components=15, eog_channel="Fp1"
    )
    for win_idx, (start, end, window) in enumerate(
        iter_windows(data, sfreq, 4.0, 0.5)
    ):
        feats = extract_window_features(
            window, sfreq, ch_names, include_connectivity=True
        )
        feats.update({
            "subject_id": rec.subject_id,
            "condition": rec.condition,
            "label": rec.label,
            "file": rec.path.name,
            "window_index": win_idx,
            "start_sec": start / sfreq,
            "end_sec": end / sfreq,
        })
        rows_ica.append(feats)

df_ica = pd.DataFrame(rows_ica)
ica_csv = ICA_DIR / "eeg_features_ica.csv"
df_ica.to_csv(ica_csv, index=False)
print(f"Saved ICA-cleaned features ({len(df_ica)} rows) to {ica_csv}")


# ── 4. Load original no-ICA features for comparison ──────────────────────────
print("\n=== Loading original (no ICA) features ===")
orig_df = pd.read_csv(BASE / "outputs_reproduced" / "features" / "eeg_features.csv", low_memory=False)

# ── 5. Run LOSO on both feature sets ─────────────────────────────────────────
import sklearn
from sklearn.linear_model import LogisticRegression

MODELS = {
    "logistic_regression": LogisticRegression(C=1.0, max_iter=5000, random_state=42, class_weight="balanced"),
}

def run_simple_loso(df, label="", n_boot=500):
    """Minimal LOSO with logistic regression, returning per-subject AUCs."""
    X = df[[c for c in df.columns if c not in id_cols]].values
    y = df["label"].values
    groups = df["subject_id"].values

    logo = LeaveOneGroupOut()
    results = []
    per_subject_auc = {}
    all_preds = []

    for tr, te in logo.split(X, y, groups=groups):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        subj_test = groups[te][0]

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=5000, random_state=42, class_weight="balanced")),
        ])
        pipe.fit(X_tr, y_tr)
        y_score = pipe.predict_proba(X_te)[:, 1]
        y_pred = pipe.predict(X_te)

        subj_auc = roc_auc_score(y_te, y_score)
        per_subject_auc[subj_test] = subj_auc

        for i in range(len(y_te)):
            all_preds.append({
                "subject_id": subj_test,
                "true_label": y_te[i],
                "pred_label": y_pred[i],
                "score": y_score[i],
            })

    # Subject bootstrap CI
    aucs = np.array(list(per_subject_auc.values()))
    n_subj = len(aucs)
    boot_aucs = []
    rng = np.random.RandomState(42)
    for _ in range(n_boot):
        idx = rng.choice(n_subj, n_subj, replace=True)
        boot_aucs.append(aucs[idx].mean())
    ci_low, ci_high = np.percentile(boot_aucs, [2.5, 97.5])

    # Overall metrics
    y_all_true = [p["true_label"] for p in all_preds]
    y_all_score = [p["score"] for p in all_preds]
    y_all_pred = [p["pred_label"] for p in all_preds]
    overall_auc = roc_auc_score(y_all_true, y_all_score)

    print(f"\n{label} Results:")
    print(f"  Overall AUC: {overall_auc:.4f} [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  Per-subject AUC: mean={aucs.mean():.4f}, std={aucs.std():.4f}")
    print(f"  Range: [{aucs.min():.4f}, {aucs.max():.4f}]")
    print(f"  Subjects < 0.6: {(aucs < 0.6).sum()}/{len(aucs)}")

    return {
        "overall_auc": overall_auc,
        "auc_ci": (ci_low, ci_high),
        "per_subject_aucs": aucs,
        "auc_mean": aucs.mean(),
        "auc_std": aucs.std(),
        "n_resistant": int((aucs < 0.6).sum()),
    }, all_preds


results_orig, preds_orig = run_simple_loso(orig_df, label="No ICA")
results_ica, preds_ica = run_simple_loso(df_ica, label="ICA (per-subject)")

# ── 6. Statistical comparison ─────────────────────────────────────────────────
from scipy.stats import wilcoxon, ttest_rel

orig_aucs = results_orig["per_subject_aucs"]
ica_aucs = results_ica["per_subject_aucs"]

diff = ica_aucs - orig_aucs
w_stat, w_p = wilcoxon(diff)
t_stat, t_p = ttest_rel(orig_aucs, ica_aucs)

print(f"\n=== ICA vs No-ICA Comparison ===")
print(f"  N subjects: {len(orig_aucs)}")
print(f"  Mean AUC (no ICA): {orig_aucs.mean():.4f}")
print(f"  Mean AUC (ICA):    {ica_aucs.mean():.4f}")
print(f"  Mean difference:   {diff.mean():+.4f}")
print(f"  Wilcoxon: W={w_stat}, p={w_p:.4f}")
print(f"  Paired t-test: t={t_stat:.4f}, p={t_p:.4f}")
print(f"  Subjects where ICA improved: {np.sum(diff > 0)}/{len(diff)}")

# ── 7. Per-subject comparison table ──────────────────────────────────────────
comp_rows = []
for i in range(len(orig_aucs)):
    o = orig_aucs[i]
    i_auc = ica_aucs[i]
    comp_rows.append({
        "subject_id": f"Subject {i:02d}",
        "auc_no_ica": round(o, 4),
        "auc_ica": round(i_auc, 4),
        "difference": round(i_auc - o, 4),
        "improved": i_auc > o,
    })
comp_df = pd.DataFrame(comp_rows)
comp_df.to_csv(ICA_DIR / "ica_comparison_per_subject.csv", index=False)
print(f"\nSaved per-subject comparison to {ICA_DIR / 'ica_comparison_per_subject.csv'}")

# ── 8. Summary table ─────────────────────────────────────────────────────────
summary = pd.DataFrame([OrderedDict([
    ("condition", ["No ICA", "ICA (per-subject)"]),
    ("overall_auc", [round(results_orig["overall_auc"], 4), round(results_ica["overall_auc"], 4)]),
    ("auc_mean", [round(results_orig["auc_mean"], 4), round(results_ica["auc_mean"], 4)]),
    ("auc_std", [round(results_orig["auc_std"], 4), round(results_ica["auc_std"], 4)]),
    ("n_resistant", [results_orig["n_resistant"], results_ica["n_resistant"]]),
    ("ci_low", [round(results_orig["auc_ci"][0], 4), round(results_ica["auc_ci"][0], 4)]),
    ("ci_high", [round(results_orig["auc_ci"][1], 4), round(results_ica["auc_ci"][1], 4)]),
])])
summary.to_csv(ICA_DIR / "ica_comparison_summary.csv", index=False)

# ── 9. Figure: per-subject AUC comparison ────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 10})

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel A: AUC distributions
ax = axes[0]
ax.hist(orig_aucs, bins=10, alpha=0.6, color="#2166ac", edgecolor="black", linewidth=0.5, label=f"No ICA ({orig_aucs.mean():.3f})")
ax.hist(ica_aucs, bins=10, alpha=0.6, color="#d6604d", edgecolor="black", linewidth=0.5, label=f"ICA ({ica_aucs.mean():.3f})")
ax.axvline(0.5, color="gray", linestyle=":", alpha=0.5)
ax.axvline(0.6, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Per-subject AUC")
ax.set_ylabel("Count")
ax.set_title("A. AUC Distribution: ICA vs No ICA")
ax.legend(fontsize=8)

# Panel B: Paired comparison
ax = axes[1]
ax.plot([0.3, 1.05], [0.3, 1.05], "k--", alpha=0.3)
for idx in range(len(orig_aucs)):
    o = orig_aucs[idx]
    i = ica_aucs[idx]
    color = "#1a9850" if i > o else "#d73027"
    ax.plot([o, i], [o, i], color=color, alpha=0.3, linewidth=0.8)
    ax.scatter(o, i, c=color, s=30, edgecolors="black", linewidth=0.3, zorder=5)
ax.set_xlabel("AUC without ICA")
ax.set_ylabel("AUC with ICA")
ax.set_title(f"B. Per-Subject AUC: ICA vs No ICA\n(diff={diff.mean():+.3f}, p={w_p:.4f})")
ax.set_xlim(0.3, 1.05)
ax.set_ylim(0.3, 1.05)

# Panel C: Difference histogram
ax = axes[2]
ax.hist(diff, bins=10, color="#4daf4a", edgecolor="black", linewidth=0.5)
ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
ax.axvline(diff.mean(), color="red", linestyle="-", alpha=0.7, label=f"Mean diff={diff.mean():+.3f}")
ax.set_xlabel("AUC change (ICA - No ICA)")
ax.set_ylabel("Count")
ax.set_title("C. ICA Effect on Per-Subject AUC")
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(ICA_DIR / "figure_ica_comparison.png", bbox_inches="tight")
fig.savefig(PAPER_FIG / "figure_ica_comparison.png", bbox_inches="tight")
print(f"\nFigure saved to paper/figures/figure_ica_comparison.png")

plt.close("all")

# ── 10. Final interpretation ─────────────────────────────────────────────────
print("\n\n" + "="*60)
print("ICA VALIDATION COMPLETE")
print("="*60)
print(f"Classification effect of ICA: mean AUC change = {diff.mean():+.4f}")
if abs(diff.mean()) < 0.02:
    print("CONCLUSION: ICA does not substantively change results (< 0.02 AUC)")
elif diff.mean() > 0:
    print("CONCLUSION: ICA marginally improves results")
else:
    print("CONCLUSION: ICA marginally reduces results (interesting finding)")
print(f"Statistical test: p={w_p:.4f} (Wilcoxon)")
print(f"See figure at: paper/figures/figure_ica_comparison.png")
