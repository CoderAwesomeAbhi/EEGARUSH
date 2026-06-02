"""
Riemannian Geometry Transfer (Vectorized)
============================================
Fast implementation using batched eigendecomposition for Riemannian mean.
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

print("=== Riemannian Geometry Analysis (Vectorized) ===")
t_start = time.time()

channels = ["F3", "F4", "P3", "P4", "O1", "O2", "F7", "F8"]
n_chan = len(channels)

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("Loading features...")
feats = pd.read_csv(BASE / "outputs_reproduced" / "features" / "eeg_features.csv", low_memory=False)
id_cols = ["subject_id", "condition", "label", "file", "window_index", "start_sec", "end_sec"]
y = feats["label"].values
groups = feats["subject_id"].values

# ── 2. Vectorized covariance reconstruction ────────────────────────────────────
print("Building covariance matrices...")
n_w = len(feats)

std_vals = np.zeros((n_w, n_chan))
for i, ch in enumerate(channels):
    col = f"stat_{ch}_std"
    if col in feats.columns:
        std_vals[:, i] = feats[col].values.astype(float)
    else:
        std_vals[:, i] = 1.0

covs = np.zeros((n_w, n_chan, n_chan))
for i in range(n_chan):
    covs[:, i, i] = std_vals[:, i] ** 2
covs[covs <= 0] = 1e-8

for i in range(n_chan):
    for j in range(i + 1, n_chan):
        k1 = f"corr_{channels[i]}_{channels[j]}"
        k2 = f"corr_{channels[j]}_{channels[i]}"
        if k1 in feats.columns:
            corr = feats[k1].values.astype(float)
        elif k2 in feats.columns:
            corr = feats[k2].values.astype(float)
        else:
            corr = np.zeros(n_w)
        corr = np.nan_to_num(corr, nan=0)
        corr = np.clip(corr, -1, 1)
        covs[:, i, j] = corr * std_vals[:, i] * std_vals[:, j]
        covs[:, j, i] = covs[:, i, j]
covs += np.eye(n_chan).reshape(1, n_chan, n_chan) * 1e-8

print(f"  Built {covs.shape} in {time.time()-t_start:.1f}s")

# ── 3. Vectorized Riemannian mean ──────────────────────────────────────────────
def riemannian_mean_vec(covs_batch, max_iter=30, tol=1e-6):
    """Karcher mean via batched eigendecomposition."""
    n = covs_batch.shape[1]
    n_cov = len(covs_batch)
    M = np.eye(n)
    for it in range(max_iter):
        m_eig, m_eigvec = np.linalg.eigh(M)
        m_sqrt = m_eigvec @ np.diag(np.sqrt(m_eig)) @ m_eigvec.T
        m_inv_sqrt = m_eigvec @ np.diag(1.0 / np.sqrt(m_eig)) @ m_eigvec.T

        C_trans = m_inv_sqrt @ covs_batch @ m_inv_sqrt.T
        c_eig, c_eigvec = np.linalg.eigh(C_trans)
        c_eig = np.maximum(c_eig, 1e-10)
        log_C = np.einsum('nij,nj,nkj->nik', c_eigvec, np.log(c_eig), c_eigvec)
        v_mean = log_C.mean(axis=0)
        step = np.linalg.norm(v_mean)
        if step < tol:
            break
        v_eig, v_eigvec = np.linalg.eigh(v_mean)
        exp_v = v_eigvec @ np.diag(np.exp(v_eig)) @ v_eigvec.T
        M = m_sqrt @ exp_v @ m_sqrt
    return M, it + 1

print("Computing Riemannian mean (full dataset)...")
t0 = time.time()
mat_ref, n_iter = riemannian_mean_vec(covs)
print(f"  Converged in {n_iter} iterations, {time.time()-t0:.1f}s")

# ── 4. Tangent space features (vectorized) ─────────────────────────────────────
print("Extracting tangent space features...")
t0 = time.time()

m_eig, m_eigvec = np.linalg.eigh(mat_ref)
m_inv_sqrt = m_eigvec @ np.diag(1.0 / np.sqrt(m_eig)) @ m_eigvec.T

C_trans = m_inv_sqrt @ covs @ m_inv_sqrt.T
c_eig, c_eigvec = np.linalg.eigh(C_trans)
c_eig = np.maximum(c_eig, 1e-10)
log_C = np.einsum('nij,nj,nkj->nik', c_eigvec, np.log(c_eig), c_eigvec)

triu = np.triu_indices(n_chan)
X_tangent = log_C[:, triu[0], triu[1]]

print(f"  Tangent features: {X_tangent.shape} in {time.time()-t0:.1f}s")

# ── 5. LOSO with tangent features ──────────────────────────────────────────────
print("Running LOSO (tangent space)...")
logo = LeaveOneGroupOut()
per_subj_tangent = {}
t0 = time.time()

for tr, te in logo.split(X_tangent, y, groups=groups):
    subj = groups[te][0]
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=5000, random_state=42, class_weight="balanced")),
    ])
    pipe.fit(X_tangent[tr], y[tr])
    y_score = pipe.predict_proba(X_tangent[te])[:, 1]
    per_subj_tangent[subj] = roc_auc_score(y[te], y_score)

tangent_aucs = np.array(list(per_subj_tangent.values()))
print(f"  Done in {time.time()-t0:.1f}s")
print(f"  Mean AUC: {tangent_aucs.mean():.4f} +/- {tangent_aucs.std():.4f}")

# ── 6. Baseline LOSO ──────────────────────────────────────────────────────────
print("Running baseline LOSO (805 features)...")
feat_cols = [c for c in feats.columns if c not in id_cols]
X_orig = feats[feat_cols].values
per_subj_base = {}

for tr, te in logo.split(X_orig, y, groups=groups):
    subj = groups[te][0]
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=5000, random_state=42, class_weight="balanced")),
    ])
    pipe.fit(X_orig[tr], y[tr])
    y_score = pipe.predict_proba(X_orig[te])[:, 1]
    per_subj_base[subj] = roc_auc_score(y[te], y_score)

base_aucs = np.array(list(per_subj_base.values()))
print(f"  Mean AUC: {base_aucs.mean():.4f} +/- {base_aucs.std():.4f}")

# ── 7. Comparison ─────────────────────────────────────────────────────────────
common = sorted(set(per_subj_base) & set(per_subj_tangent))
base_v = np.array([per_subj_base[s] for s in common])
tang_v = np.array([per_subj_tangent[s] for s in common])
diff = tang_v - base_v

from scipy.stats import wilcoxon
try:
    w_stat, w_p = wilcoxon(diff)
except ValueError:
    w_stat, w_p = 0, 1.0

print(f"\n=== Comparison ===")
print(f"Baseline:   {base_v.mean():.4f} +/- {base_v.std():.4f}")
print(f"Tangent:    {tang_v.mean():.4f} +/- {tang_v.std():.4f}")
print(f"Difference: {diff.mean():+.4f} (p={w_p:.4f})")
print(f"Improved:   {(diff > 0).sum()}/{len(diff)}")
print(f"Total time: {time.time()-t_start:.1f}s")

# ── 8. Cross-dataset: proper Riemannian transfer with target recentering ──
print("\n--- Cross-dataset transfer: MAT -> DS007262 (proper Riemannian) ---")
ds262 = pd.read_csv(BASE / "external_validation_ds007262" / "ds007262_low_high_features.csv")
ds262_y = ds262["label"].values
n_ds = len(ds262)

std_ds = np.zeros((n_ds, n_chan))
for i, ch in enumerate(channels):
    col = f"stat_{ch}_std"
    std_ds[:, i] = ds262[col].values.astype(float) if col in ds262.columns else 1.0

cov_ds = np.zeros((n_ds, n_chan, n_chan))
for i in range(n_chan):
    cov_ds[:, i, i] = std_ds[:, i] ** 2
cov_ds[cov_ds <= 0] = 1e-8
for i in range(n_chan):
    for j in range(i+1, n_chan):
        k1 = f"corr_{channels[i]}_{channels[j]}"
        k2 = f"corr_{channels[j]}_{channels[i]}"
        if k1 in ds262.columns:
            corr = ds262[k1].values.astype(float)
        elif k2 in ds262.columns:
            corr = ds262[k2].values.astype(float)
        else:
            corr = np.zeros(n_ds)
        corr = np.nan_to_num(corr, nan=0)
        corr = np.clip(corr, -1, 1)
        cov_ds[:, i, j] = corr * std_ds[:, i] * std_ds[:, j]
        cov_ds[:, j, i] = cov_ds[:, i, j]
cov_ds += np.eye(n_chan).reshape(1, n_chan, n_chan) * 1e-8

# Compute DS007262 Riemannian mean via batch gradient descent
mat_ds_mean, _ = riemannian_mean_vec(cov_ds)
print(f"  MAT reference mean: ref")
print(f"  DS007262 reference mean computed")

# Option A: Project both into shared mean (no recentering, original approach)
C_ds_t_orig = m_inv_sqrt @ cov_ds @ m_inv_sqrt.T
c_eig_ds, c_eigvec_ds = np.linalg.eigh(C_ds_t_orig)
c_eig_ds = np.maximum(c_eig_ds, 1e-10)
log_C_ds_orig = np.einsum('nij,nj,nkj->nik', c_eigvec_ds, np.log(c_eig_ds), c_eigvec_ds)
X_ds_tang_orig = log_C_ds_orig[:, triu[0], triu[1]]

# Option B: Recenter DS007262 to MAT's mean (Zanini et al. 2018 approach)
# Parallel transport: align target covariances to source mean
m_ds_eig, m_ds_eigvec = np.linalg.eigh(mat_ds_mean)
m_ds_sqrt = m_ds_eigvec @ np.diag(np.sqrt(m_ds_eig)) @ m_ds_eigvec.T
m_ds_inv_sqrt = m_ds_eigvec @ np.diag(1.0 / np.sqrt(m_ds_eig)) @ m_ds_eigvec.T

# Transport: T = M_s^{1/2} @ M_ds^{-1/2} (transport from target to source)
m_sqrt = m_eigvec @ np.diag(np.sqrt(m_eig)) @ m_eigvec.T
transport = m_sqrt @ m_ds_inv_sqrt
cov_ds_aligned = transport @ cov_ds @ transport.T

# Project aligned target into source tangent space
C_ds_t_aligned = m_inv_sqrt @ cov_ds_aligned @ m_inv_sqrt.T
c_eig_ds_a, c_eigvec_ds_a = np.linalg.eigh(C_ds_t_aligned)
c_eig_ds_a = np.maximum(c_eig_ds_a, 1e-10)
log_C_ds_aligned = np.einsum('nij,nj,nkj->nik', c_eigvec_ds_a, np.log(c_eig_ds_a), c_eigvec_ds_a)
X_ds_tang_aligned = log_C_ds_aligned[:, triu[0], triu[1]]

# Option C: Riemannian alignment - jointly recenter both
# Compute mean of both datasets pooled
all_covs = np.concatenate([covs, cov_ds], axis=0)
pooled_mean, _ = riemannian_mean_vec(all_covs)
pm_eig, pm_eigvec = np.linalg.eigh(pooled_mean)
pm_inv_sqrt = pm_eigvec @ np.diag(1.0 / np.sqrt(pm_eig)) @ pm_eigvec.T

C_t_pool_mat = pm_inv_sqrt @ covs @ pm_inv_sqrt.T
c_mp, c_ev_mp = np.linalg.eigh(C_t_pool_mat)
c_mp = np.maximum(c_mp, 1e-10)
log_C_pool_mat = np.einsum('nij,nj,nkj->nik', c_ev_mp, np.log(c_mp), c_ev_mp)
X_tang_pool_mat = log_C_pool_mat[:, triu[0], triu[1]]

C_t_pool_ds = pm_inv_sqrt @ cov_ds @ pm_inv_sqrt.T
c_dp, c_ev_dp = np.linalg.eigh(C_t_pool_ds)
c_dp = np.maximum(c_dp, 1e-10)
log_C_pool_ds = np.einsum('nij,nj,nkj->nik', c_ev_dp, np.log(c_dp), c_ev_dp)
X_tang_pool_ds = log_C_pool_ds[:, triu[0], triu[1]]

# Evaluate all three approaches + baseline
pipe_orig = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(C=1.0, max_iter=5000, random_state=42, class_weight="balanced")),
])
pipe_orig.fit(X_tangent, y)
ds_auc_tang_orig = roc_auc_score(ds262_y, pipe_orig.predict_proba(X_ds_tang_orig)[:, 1])
print(f"  Original (MAT tangent space): AUC = {ds_auc_tang_orig:.4f}")

pipe_aligned = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(C=1.0, max_iter=5000, random_state=42, class_weight="balanced")),
])
pipe_aligned.fit(X_tangent, y)
ds_auc_tang_aligned = roc_auc_score(ds262_y, pipe_aligned.predict_proba(X_ds_tang_aligned)[:, 1])
print(f"  Recentered (Zanini 2018):     AUC = {ds_auc_tang_aligned:.4f}")

pipe_pool = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(C=1.0, max_iter=5000, random_state=42, class_weight="balanced")),
])
pipe_pool.fit(X_tang_pool_mat, y)
ds_auc_tang_pool = roc_auc_score(ds262_y, pipe_pool.predict_proba(X_tang_pool_ds)[:, 1])
print(f"  Pooled mean recenter:         AUC = {ds_auc_tang_pool:.4f}")

# Baseline transfer (align features between MAT and DS262)
common_feat = [c for c in feat_cols if c in ds262.columns]
X_orig_ds = ds262[common_feat].values
X_orig_common = feats[common_feat].values
pipe2 = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(C=1.0, max_iter=5000, random_state=42, class_weight="balanced")),
])
pipe2.fit(X_orig_common, y)
ds_auc_base = roc_auc_score(ds262_y, pipe2.predict_proba(X_orig_ds)[:, 1])
print(f"  Baseline (aligned features): AUC = {ds_auc_base:.4f}")

# ── 9. Save ──────────────────────────────────────────────────────────────────
results = pd.DataFrame([OrderedDict([
    ("method", [
        "Baseline (805 features, MAT LOSO)",
        "Riemannian Tangent (MAT LOSO)", 
        "Baseline (MAT->DS262 transfer)",
        "Riemannian Original (MAT->DS262)",
        "Riemannian Recentered (MAT->DS262)",
        "Riemannian Pooled Mean (MAT->DS262)",
    ]),
    ("mean_auc", [
        round(base_v.mean(), 4), round(tang_v.mean(), 4),
        round(ds_auc_base, 4), round(ds_auc_tang_orig, 4),
        round(ds_auc_tang_aligned, 4), round(ds_auc_tang_pool, 4),
    ]),
    ("std_auc", [
        round(base_v.std(), 4), round(tang_v.std(), 4),
        None, None, None, None,
    ]),
    ("p_vs_baseline", [None, round(w_p, 4), None, None, None, None]),
])])
results.to_csv(RIEM / "riemannian_results.csv", index=False)
print(f"\nSaved to {RIEM / 'riemannian_results.csv'}")

# ── 10. Figure ────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 10})

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
means = [base_v.mean(), tang_v.mean()]
stds = [base_v.std(), tang_v.std()]
bars = ax.bar(["Baseline\n(805 features)", "Riemannian\nTangent Space"], means, yerr=stds,
              capsize=5, color=["#2166ac", "#d6604d"], edgecolor="black", linewidth=0.8)
ax.set_ylabel("Mean LOSO AUC")
ax.set_title(f"A. Riemannian vs Baseline\n(diff={diff.mean():+.3f}, p={w_p:.4f})")
ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
for b, v in zip(bars, means):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{v:.3f}", 
            ha="center", fontweight="bold", fontsize=9)

ax = axes[1]
ax.plot([0.3, 1.05], [0.3, 1.05], "k--", alpha=0.3)
for b, t in zip(base_v, tang_v):
    ax.scatter(b, t, c="#1a9850" if t > b else "#d73027", s=30, edgecolors="black", linewidth=0.3, alpha=0.7)
ax.set_xlabel("Baseline AUC"); ax.set_ylabel("Riemannian AUC")
ax.set_title("B. Per-Subject Comparison")
ax.set_xlim(0.3, 1.05); ax.set_ylim(0.3, 1.05)

fig.tight_layout()
fig.savefig(RIEM / "figure_riemannian_comparison.png", bbox_inches="tight")
fig.savefig(PAPER_FIG / "figure_riemannian_comparison.png", bbox_inches="tight")
print("Figure saved to paper/figures/")
plt.close("all")
print(f"\n=== Complete in {time.time()-t_start:.1f}s ===")
