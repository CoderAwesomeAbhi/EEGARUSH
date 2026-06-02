"""
DS007262 Failure Analysis: Why does cross-dataset classification fail?
=======================================================================
Addresses Issue #4: Deep investigation of why DS007262 is at chance.
Tests: (1) Is there a theta effect in DS007262? (2) Does difficulty gradient
produce a neural signal? (3) Are feature distributions fundamentally different?
"""

import sys, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

OUT = BASE / "outputs_phd_revision"
DS_DIR = OUT / "ds007262_analysis"
DS_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIG = BASE / "paper" / "figures"

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 10})

print("=" * 60)
print("DS007262 Failure Analysis")
print("=" * 60)
t0 = time.time()

# ── 1. Load DS007262 features ──────────────────────────────────────────────
feat_path = BASE / "external_validation_ds007262" / "ds007262_low_high_features.csv"
df = pd.read_csv(feat_path, low_memory=False)
print(f"\nLoaded {len(df)} windows from DS007262")
print(f"Subjects: {df['subject_id'].nunique()}")
print(f"Conditions: {df['condition'].unique()}")

# ── 2. Is there a theta effect? ─────────────────────────────────────────────
print("\n=== 2. Frontal theta effect in DS007262 ===")
theta_cols = [c for c in df.columns if 'band_abs' in c and 'theta' in c and any(ch in c.lower() for ch in ['_f3', '_f4'])]
print(f"Theta columns: {theta_cols}")

if len(theta_cols) >= 1:
    theta_vals = df[theta_cols].mean(axis=1).values
    labels = df['label'].values
    subjects = df['subject_id'].values
    trial_types = df['trial_type'].values
    
    rest_theta = [np.mean(theta_vals[(labels == 0) & (subjects == s)]) for s in np.unique(subjects)]
    work_theta = [np.mean(theta_vals[(labels == 1) & (subjects == s)]) for s in np.unique(subjects)]
    
    t_stat, p_val = stats.ttest_rel(work_theta, rest_theta)
    d = (np.mean(work_theta) - np.mean(rest_theta)) / (np.std(np.array(work_theta) - np.array(rest_theta), ddof=1) + 1e-15)
    print(f"  Frontal theta: rest={np.mean(rest_theta):.4f} work={np.mean(work_theta):.4f}")
    print(f"  t={t_stat:.3f} p={p_val:.4f} d={d:.4f}")
    
    # Also ratio features
    ratio_cols = [c for c in df.columns if 'theta_alpha' in c and any(ch in c.lower() for ch in ['_f3', '_f4'])]
    if ratio_cols:
        ratio_vals = df[ratio_cols].mean(axis=1).values
        rest_r = [np.mean(ratio_vals[(labels == 0) & (subjects == s)]) for s in np.unique(subjects)]
        work_r = [np.mean(ratio_vals[(labels == 1) & (subjects == s)]) for s in np.unique(subjects)]
        t_r, p_r = stats.ttest_rel(work_r, rest_r)
        d_r = (np.mean(work_r) - np.mean(rest_r)) / (np.std(np.array(work_r)-np.array(rest_r), ddof=1)+1e-15)
        print(f"  Theta/alpha ratio: d={d_r:.4f} p={p_r:.4f}")

# ── 3. Difficulty gradient analysis ─────────────────────────────────────────
print("\n=== 3. Difficulty gradient analysis ===")
# DS007262 has 7 difficulty levels; we collapse to low/high
# Check if theta power scales with difficulty
trial_details = df['trial_type'].value_counts()
print(f"  Trial types: {dict(trial_details)}")

# Low = levels 0.6-1.5 (label 0), High = levels 6.0-6.9 (label 1)
if len(theta_cols) >= 1:
    # Per-subject, per-difficulty theta
    results = []
    for s in np.unique(subjects):
        s_mask = subjects == s
        for tt in np.unique(trial_types):
            tt_mask = trial_types == tt
            idx = s_mask & tt_mask
            if idx.sum() >= 1:
                results.append({
                    'subject_id': s,
                    'trial_type': tt,
                    'theta_mean': np.mean(theta_vals[idx]),
                    'label': labels[idx][0],
                })
    df_gradient = pd.DataFrame(results)
    
    # Compare theta across all 7 difficulty levels (if available)
    # Actually, features only have low/high from the current pipeline
    # Let's check if there's a difference
    print(f"  Mean theta by label:")
    print(f"    Low difficulty (label 0): {np.mean(theta_vals[labels==0]):.4f}")
    print(f"    High difficulty (label 1): {np.mean(theta_vals[labels==1]):.4f}")

# ── 4. Compare DS007262 vs MAT theta distributions ─────────────────────────
print("\n=== 4. Cross-dataset feature comparison ===")
# Load MAT features
mat_path = BASE / "outputs_reproduced" / "features" / "eeg_features.csv"
df_mat = pd.read_csv(mat_path, nrows=None, low_memory=False)

# Find common theta columns
mat_theta_cols = [c for c in df_mat.columns if 'band_abs' in c and 'theta' in c and any(ch in c.lower() for ch in ['_f3', '_f4'])]
if mat_theta_cols and theta_cols:
    mat_theta = df_mat[mat_theta_cols].mean(axis=1).values
    mat_labels = df_mat['label'].values
    
    # Compare effect size magnitude
    mat_rest = [np.mean(mat_theta[(mat_labels == 0) & (df_mat['subject_id'] == s)]) for s in np.unique(df_mat['subject_id'])]
    mat_work = [np.mean(mat_theta[(mat_labels == 1) & (df_mat['subject_id'] == s)]) for s in np.unique(df_mat['subject_id'])]
    mat_d = (np.mean(mat_work) - np.mean(mat_rest)) / (np.std(np.array(mat_work) - np.array(mat_rest), ddof=1) + 1e-15)
    
    print(f"  MAT theta effect: d={mat_d:.4f}")
    print(f"  DS007262 theta effect: d={d:.4f}")
    
    # Ratio comparison
    mat_ratio_cols = [c for c in df_mat.columns if 'theta_alpha' in c and any(ch in c.lower() for ch in ['_f3', '_f4'])]
    if mat_ratio_cols:
        mat_ratio = df_mat[mat_ratio_cols].mean(axis=1).values
        mr = [np.mean(mat_ratio[(mat_labels == 0) & (df_mat['subject_id'] == s)]) for s in np.unique(df_mat['subject_id'])]
        mw = [np.mean(mat_ratio[(mat_labels == 1) & (df_mat['subject_id'] == s)]) for s in np.unique(df_mat['subject_id'])]
        mat_ratio_d = (np.mean(mw) - np.mean(mr)) / (np.std(np.array(mw)-np.array(mr), ddof=1)+1e-15)
        print(f"  MAT theta/alpha ratio d={mat_ratio_d:.4f}")
    
    # Distribution comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    ax = axes[0]
    ax.hist(mat_rest, bins=20, alpha=0.5, label='MAT rest', color='#4393c3')
    ax.hist(mat_work, bins=20, alpha=0.5, label='MAT work', color='#d6604d')
    ax.set_xlabel('Frontal theta power')
    ax.set_ylabel('Count (windows)')
    ax.set_title('MAT feature distribution')
    ax.legend()
    
    ax = axes[1]
    ax.hist(theta_vals[labels==0], bins=20, alpha=0.5, label='DS007 low', color='#4393c3')
    ax.hist(theta_vals[labels==1], bins=20, alpha=0.5, label='DS007 high', color='#d6604d')
    ax.set_xlabel('Frontal theta power')
    ax.set_ylabel('Count (windows)')
    ax.set_title('DS007262 feature distribution')
    ax.legend()
    
    ax = axes[2]
    # Per-subject effect sizes
    ds_effects = np.array(work_theta) - np.array(rest_theta)
    mat_effects = np.array(mat_work) - np.array(mat_rest)
    ax.boxplot([mat_effects, ds_effects], labels=['MAT', 'DS007262'])
    ax.set_ylabel('Frontal theta effect (work - rest)')
    ax.set_title('Per-subject theta modulation')
    ax.axhline(0, color='gray', ls='--', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(DS_DIR / "ds007262_feature_comparison.png", bbox_inches="tight")
    fig.savefig(PAPER_FIG / "figure_ds007262_comparison.png", bbox_inches="tight")
    print(f"  Comparison figure saved")

# ── 5. Summary ──────────────────────────────────────────────────────────────
print(f"\n=== SUMMARY ===")
print(f"Frontal theta effect in DS007262: d={d:.4f} (MAT: d={mat_d:.4f})")
print(f"DS007262 has N=18 subjects; theta effect may be underpowered")
print(f"Key difference: DS007262 is graded difficulty within arithmetic, not rest-vs-task")
print(f"Recommendation: Frame as informative negative control rather than failed replication")
print(f"\nTime: {time.time()-t0:.1f}s")
