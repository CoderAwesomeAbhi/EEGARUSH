"""
Robustness Analysis — Fixed
Uses per-subject paired approach matching the paper's methodology.
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
ANALYSIS_DIR = OUT / "robustness_analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIG = BASE / "paper" / "figures"

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 10})

t0 = time.time()
print("=" * 60)
print("Robustness Analysis (per-subject)")
print("=" * 60)

# ── 1. Power Analysis ─────────────────────────────────────────────────────
print("\n=== 1. Power Analysis ===")
effect_sizes = [0.3, 0.5, 0.6, 0.7, 0.84, 1.0]
Ns = range(5, 301)
alpha = 0.05

power_results = []
for d in effect_sizes:
    for n in Ns:
        df = n - 1
        t_crit = stats.t.ppf(1 - alpha / 2, df)
        ncp = d * np.sqrt(n)
        power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
        power_results.append({'effect_size': d, 'N': n, 'power': power})
df_power = pd.DataFrame(power_results)

print("N needed for 80% power (paired t, alpha=0.05):")
for d in effect_sizes:
    needed = df_power[(df_power['effect_size'] == d) & (df_power['power'] >= 0.80)]['N'].min()
    print(f"  d={d:.2f}: N={needed}")

fig, ax = plt.subplots(figsize=(8, 5))
for d in effect_sizes:
    sub = df_power[df_power['effect_size'] == d]
    ax.plot(sub['N'], sub['power'], label=f'd={d:.2f}', lw=2)
ax.axhline(0.80, color='gray', ls='--', alpha=0.5, label='80% power')
ax.axvline(36, color='red', ls=':', alpha=0.3, label='N=36 (MAT)')
ax.axvline(48, color='orange', ls=':', alpha=0.3, label='N=48 (STEW)')
ax.axvline(200, color='green', ls=':', alpha=0.3, label='N=200')
ax.set_xlabel('N (subjects)')
ax.set_ylabel('Statistical Power')
ax.set_title('Power Analysis: Paired t-test')
ax.legend(fontsize=8, loc='lower right')
ax.set_xlim(0, 300)
fig.savefig(ANALYSIS_DIR / "power_analysis.png", bbox_inches="tight")
fig.savefig(PAPER_FIG / "figure_power_analysis.png", bbox_inches="tight")
print(f"  Figure saved")

# ── 2. Power needed for individual-differences analyses ─────────────────────
print("\n=== 2. Power for correlations (resting predictors) ===")
# Spearman correlation: what N for 80% power at various rho?
from scipy.stats import ncx2
for rho in [0.3, 0.4, 0.5, 0.6, 0.61]:
    # Convert to Fisher's z, compute power
    n_found = None
    for n in range(5, 501):
        t_crit = stats.t.ppf(1 - 0.05 / 2, n - 2)
        r_crit = t_crit / np.sqrt(n - 2 + t_crit**2)
        z_crit = np.arctanh(r_crit)
        z_obs = np.arctanh(rho)
        se = 1 / np.sqrt(n - 3)
        power = 1 - stats.norm.cdf((z_crit - z_obs) / se) + stats.norm.cdf((-z_crit - z_obs) / se)
        if power >= 0.80 and n_found is None:
            n_found = n
    print(f"  rho={rho:.2f}: N={n_found} for 80% power")

# ── 3. Specification curve on ratio features (scale-invariant) ─────────────
print("\n=== 3. Specification Curve (ratio features) ===")
df_feat = pd.read_csv(BASE / "outputs_reproduced" / "features" / "eeg_features.csv", low_memory=False)

# Use theta/alpha ratio (scale-invariant, should replicate)
specs = []

# Theta/alpha ratio (F3, F4)
for ch in ['F3', 'F4']:
    col = f'ratio_{ch}_theta_alpha'
    if col in df_feat.columns:
        vals = df_feat[col].values
        subjs = df_feat['subject_id'].values
        lbls = df_feat['label'].values
        rest = np.array([np.mean(vals[(subjs==s) & (lbls==0)]) for s in np.unique(subjs)])
        work = np.array([np.mean(vals[(subjs==s) & (lbls==1)]) for s in np.unique(subjs)])
        d = (work - rest).mean() / ((work - rest).std(ddof=1) + 1e-15)
        t, p = stats.ttest_rel(work, rest)
        specs.append({'spec': f'Theta/alpha ratio ({ch})', 'd': d, 'p': p, 'n': len(rest)})

# Theta/beta ratio
for ch in ['F3', 'F4']:
    col = f'ratio_{ch}_theta_beta'
    if col in df_feat.columns:
        vals = df_feat[col].values
        subjs = df_feat['subject_id'].values
        lbls = df_feat['label'].values
        rest = np.array([np.mean(vals[(subjs==s) & (lbls==0)]) for s in np.unique(subjs)])
        work = np.array([np.mean(vals[(subjs==s) & (lbls==1)]) for s in np.unique(subjs)])
        d = (work - rest).mean() / ((work - rest).std(ddof=1) + 1e-15)
        t, p = stats.ttest_rel(work, rest)
        specs.append({'spec': f'Theta/beta ratio ({ch})', 'd': d, 'p': p, 'n': len(rest)})

# Alpha band (should decrease)
for ch in ['F3', 'F4']:
    col = f'band_abs_{ch}_alpha'
    if col in df_feat.columns:
        vals = df_feat[col].values
        subjs = df_feat['subject_id'].values
        lbls = df_feat['label'].values
        rest = np.array([np.mean(vals[(subjs==s) & (lbls==0)]) for s in np.unique(subjs)])
        work = np.array([np.mean(vals[(subjs==s) & (lbls==1)]) for s in np.unique(subjs)])
        d = (work - rest).mean() / ((work - rest).std(ddof=1) + 1e-15)
        t, p = stats.ttest_rel(work, rest)
        specs.append({'spec': f'Absolute alpha ({ch})', 'd': d, 'p': p, 'n': len(rest)})

df_spec = pd.DataFrame(specs)
print(f"  Specifications: {len(df_spec)}")
for _, r in df_spec.iterrows():
    sig = ' *' if r['p'] < 0.05 else ' ns'
    print(f"  {r['spec']:40s}: d={r['d']:+.4f} p={r['p']:.4f}{sig}")

df_spec.to_csv(ANALYSIS_DIR / "specification_curve.csv", index=False)

# ── 4. Covariate analysis ──────────────────────────────────────────────────
print("\n=== 4. Covariate Analysis (anxiety/fatigue proxy) ===")
subj_info = pd.read_csv(BASE / "data" / "raw" / "eegmat" / "subject-info.csv")
subj_info['subject_id'] = subj_info['Subject'].str.strip()

auc_path = BASE / "outputs_phd_revision" / "tables" / "ev8_external_subject_aucs.csv"
if auc_path.exists():
    df_auc = pd.read_csv(auc_path)
    # Check column names
    print(f"  AUC columns: {df_auc.columns.tolist()}")
    auc_col = [c for c in df_auc.columns if 'auc' in c.lower()]
    subj_col = [c for c in df_auc.columns if 'subject' in c.lower()]
    print(f"  Subject col: {subj_col}, AUC col: {auc_col}")
    
    if subj_col and auc_col:
        df_auc = df_auc.rename(columns={subj_col[0]: 'subject_id', auc_col[0]: 'auc'})
        df_m = df_auc.merge(subj_info, on='subject_id', how='inner')
        
        for covar in ['Age', 'Number of subtractions']:
            r, p = stats.spearmanr(df_m[covar].values, df_m['auc'].values.astype(float))
            print(f"  {covar} vs AUC: rho={r:.4f} p={p:.4f}")
        
        m_auc = df_m[df_m['Gender'] == 'M']['auc'].values.astype(float)
        f_auc = df_m[df_m['Gender'] == 'F']['auc'].values.astype(float)
        if len(m_auc) > 2 and len(f_auc) > 2:
            u, p_g = stats.mannwhitneyu(m_auc, f_auc)
            print(f"  Gender (M={len(m_auc)}, F={len(f_auc)}): MW p={p_g:.4f}")

print(f"\nTotal time: {time.time()-t0:.1f}s")
