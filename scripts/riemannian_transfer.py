"""
Riemannian Geometry for Cross-Dataset Transfer
================================================
Reconstructs per-window covariance matrices from existing features,
applies Riemannian alignment, tests cross-dataset transfer improvement.
"""

import warnings, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from collections import OrderedDict

warnings.filterwarnings("ignore")

BASE = Path(r'C:\Users\abhij\Downloads\bioarxivarjun')
OUT = BASE / "outputs_phd_revision"
RIEM = OUT / "riemannian_analysis"
RIEM.mkdir(parents=True, exist_ok=True)
PAPER_FIG = BASE / "paper" / "figures"

print("=== Riemannian Geometry Analysis ===")

# ── 1. Load features ──────────────────────────────────────────────────────────
print("\nLoading features...")
feats = pd.read_csv(BASE / "outputs_reproduced" / "features" / "eeg_features.csv", low_memory=False)
id_cols = ["subject_id", "condition", "label", "file", "window_index", "start_sec", "end_sec"]

common_channels = ["F3", "F4", "P3", "P4", "O1", "O2", "F7", "F8"]
n_chan = len(common_channels)
n_tang = n_chan * (n_chan + 1) // 2

# ── 2. Efficient covariance reconstruction ────────────────────────────────────
print(f"Reconstructing {n_chan}x{n_chan} covariance matrices for {len(feats)} windows...")

def build_covariance_matrix(feats_df, channels, id_cols):
    """Vectorized covariance reconstruction."""
    n = len(channels)
    n_windows = len(feats_df)
    covs = np.zeros((n_windows, n, n))

    # Prepare std arrays
    std_vals = np.zeros((n_windows, n))
    for i, ch in enumerate(channels):
        col = f"stat_{ch}_std"
        if col in feats_df.columns:
            std_vals[:, i] = feats_df[col].values
        else:
            std_vals[:, i] = 1.0

    # Diagonal: variances
    for i in range(n):
        covs[:, i, i] = std_vals[:, i] ** 2
    covs[covs <= 0] = 1e-8

    # Off-diagonal: corr * std_i * std_j
    for i in range(n):
        for j in range(i + 1, n):
            key1 = f"corr_{channels[i]}_{channels[j]}"
            key2 = f"corr_{channels[j]}_{channels[i]}"
            if key1 in feats_df.columns:
                corr = feats_df[key1].values
            elif key2 in feats_df.columns:
                corr = feats_df[key2].values
            else:
                corr = np.zeros(n_windows)
            corr = np.nan_to_num(corr, nan=0)
            corr = np.clip(corr, -1, 1)
            covs[:, i, j] = corr * std_vals[:, i] * std_vals[:, j]
            covs[:, j, i] = covs[:, i, j]

    # SPD regularization
    covs += np.eye(n).reshape(1, n, n) * 1e-6
    return covs

t0 = time.time()
covariances = build_covariance_matrix(feats, common_channels, id_cols)
print(f"  Built {covariances.shape} in {time.time()-t0:.1f}s")

# ── 3. Riemannian operations ──────────────────────────────────────────────────
from scipy.linalg import sqrtm, logm, inv, expm

def riemannian_mean(covs, max_iter=30, tol=1e-5):
    """Karcher mean of SPD matrices via gradient descent on the manifold."""
    n = covs.shape[1]
    M = np.eye(n)
    for it in range(max_iter):
        tangents = np.zeros((len(covs), n, n))
        for k, c in enumerate(covs):
            m_sqrt = sqrtm(M)
            m_inv_sqrt = inv(m_sqrt)
            tangents[k] = logm(m_inv_sqrt @ c @ m_inv_sqrt)
        v_mean = tangents.mean(axis=0)
        step = np.linalg.norm(v_mean)
        if step < tol:
            break
        M = sqrtm(M) @ expm(v_mean) @ sqrtm(M)
    return M

def tangent_space(cov, ref):
    """Map SPD matrix to tangent space at ref."""
    ref_sqrt = sqrtm(ref)
    ref_inv_sqrt = inv(ref_sqrt)
    log_map = logm(ref_inv_sqrt @ cov @ ref_inv_sqrt)
    triu_idx = np.triu_indices(cov.shape[0])
    return log_map[triu_idx]

# ── 4. LOSO with tangent space features ──────────────────────────────────────
y = feats["label"].values
groups = feats["subject_id"].values
logo = LeaveOneGroupOut()

per_subj_riemann = {}
preds_riemann = []

print("\nRunning LOSO with Riemannian tangent features...")
t0 = time.time()
fold = 0
for tr, te in logo.split(np.zeros(len(y)), y, groups=groups):
    fold += 1
    subj_test = groups[te][0]

    train_covs = covariances[tr]
    test_covs = covariances[te]

    # Riemannian mean of training set
    train_ref = riemannian_mean(train_covs)

    X_tr = np.array([tangent_space(c, train_ref) for c in train_covs])
    X_te = np.array([tangent_space(c, train_ref) for c in test_covs])

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=5000, random_state=42, class_weight="balanced")),
    ])
    pipe.fit(X_tr, y[tr])
    y_score = pipe.predict_proba(X_te)[:, 1]
    subj_auc = roc_auc_score(y[te], y_score)
    per_subj_riemann[subj_test] = subj_auc

    if fold % 10 == 0:
        print(f"  Fold {fold}/{len(np.unique(groups))}, subj={subj_test}, AUC={subj_auc:.3f}")

print(f"LOSO completed in {time.time()-t0:.1f}s")

riemann_aucs = np.array(list(per_subj_riemann.values()))
print(f"\nRiemannian tangent LOSO:")
print(f"  Mean AUC: {riemann_aucs.mean():.4f} +/- {riemann_aucs.std():.4f}")

# ── 5. Baseline LOSO ──────────────────────────────────────────────────────────
print("\nRunning baseline LOSO (805 features)...")
X_orig = feats[[c for c in feats.columns if c not in id_cols]].values

per_subj_baseline = {}
logo2 = LeaveOneGroupOut()
for tr, te in logo2.split(X_orig, y, groups=groups):
    subj_test = groups[te][0]
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=5000, random_state=42, class_weight="balanced")),
    ])
    pipe.fit(X_orig[tr], y[tr])
    y_score = pipe.predict_proba(X_orig[te])[:, 1]
    per_subj_baseline[subj_test] = roc_auc_score(y[te], y_score)

baseline_aucs = np.array(list(per_subj_baseline.values()))
print(f"  Mean AUC: {baseline_aucs.mean():.4f} +/- {baseline_aucs.std():.4f}")

# ── 6. Comparison ────────────────────────────────────────────────────────────
common_s = sorted(set(per_subj_baseline.keys()) & set(per_subj_riemann.keys()))
base_v = np.array([per_subj_baseline[s] for s in common_s])
riem_v = np.array([per_subj_riemann[s] for s in common_s])
diff = riem_v - base_v

from scipy.stats import wilcoxon
w_stat, w_p = wilcoxon(diff)

print(f"\n=== COMPARISON ===")
print(f"Baseline:      {base_v.mean():.4f} +/- {base_v.std():.4f}")
print(f"Riemannian:    {riem_v.mean():.4f} +/- {riem_v.std():.4f}")
print(f"Difference:    {diff.mean():+.4f} (p={w_p:.4f})")
print(f"Improved:      {(diff > 0).sum()}/{len(diff)} subjects")

# ── 7. Cross-dataset transfer ─────────────────────────────────────────────────
print("\n--- Cross-Dataset Transfer: MAT -> DS007262 ---")
ds262 = pd.read_csv(BASE / "external_validation_ds007262" / "ds007262_low_high_features.csv")
ds262_chan = [c for c in common_channels if f"stat_{c}_std" in ds262.columns]
print(f"DS007262 channels: {ds262_chan}")

ds262_cov = build_covariance_matrix(ds262, ds262_chan, [])
ds262_y = ds262["label"].values

# Compute Riemannian mean on full MAT, map DS007262 to same tangent space
mat_ref = riemannian_mean(covariances)
X_mat_tang = np.array([tangent_space(c, mat_ref) for c in covariances])
X_ds_tang = np.array([tangent_space(c, mat_ref) for c in ds262_cov])

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(C=1.0, max_iter=5000, random_state=42, class_weight="balanced")),
])
pipe.fit(X_mat_tang, y)
ds_score = pipe.predict_proba(X_ds_tang)[:, 1]
ds_auc = roc_auc_score(ds262_y, ds_score)
print(f"MAT -> DS007262 (Riemannian): AUC = {ds_auc:.4f}")

# ── 8. Save results ──────────────────────────────────────────────────────────
results = pd.DataFrame([OrderedDict([
    ("method", ["Baseline (805 features)", "Riemannian Tangent Space", 
                "MAT->DS262 Riemannian"]),
    ("mean_auc", [round(base_v.mean(), 4), round(riem_v.mean(), 4), round(ds_auc, 4)]),
    ("std_auc", [round(base_v.std(), 4), round(riem_v.std(), 4), None]),
    ("p_vs_baseline", [None, round(w_p, 4), None]),
])])
results.to_csv(RIEM / "riemannian_results.csv", index=False)
print(f"\nSaved to {RIEM / 'riemannian_results.csv'}")

# ── 9. Figure ─────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 10})

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
means = [base_v.mean(), riem_v.mean()]
stds = [base_v.std(), riem_v.std()]
bars = ax.bar(["Baseline\n(805 features)", "Riemannian\nTangent Space"], means, yerr=stds,
              capsize=5, color=["#2166ac", "#d6604d"], edgecolor="black", linewidth=0.8)
ax.set_ylabel("Mean LOSO AUC")
ax.set_title(f"A. Riemannian vs Baseline\n(diff={diff.mean():+.3f}, p={w_p:.4f})")
ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
for bar, val in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.3f}",
            ha="center", fontweight="bold", fontsize=9)

ax = axes[1]
ax.plot([0.3, 1.05], [0.3, 1.05], "k--", alpha=0.3)
for b, r in zip(base_v, riem_v):
    color = "#1a9850" if r > b else "#d73027"
    ax.scatter(b, r, c=color, s=30, edgecolors="black", linewidth=0.3, alpha=0.7)
ax.set_xlabel("Baseline AUC")
ax.set_ylabel("Riemannian AUC")
ax.set_title("B. Per-Subject Comparison")
ax.set_xlim(0.3, 1.05)
ax.set_ylim(0.3, 1.05)

fig.tight_layout()
fig.savefig(RIEM / "figure_riemannian_comparison.png", bbox_inches="tight")
fig.savefig(PAPER_FIG / "figure_riemannian_comparison.png", bbox_inches="tight")
print(f"Figure saved")

plt.close("all")
print("\n=== Riemannian Analysis Complete ===")
