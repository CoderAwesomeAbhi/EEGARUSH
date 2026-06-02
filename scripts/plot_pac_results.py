"""
Theta-Gamma Coupling Figures
"""
import sys, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

OUT = BASE / "outputs_phd_revision"
PAC_DIR = OUT / "pac_analysis"
FIG = OUT / "figures"
PAPER_FIG = BASE / "paper" / "figures"

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 10})

# Load data
df_pac = pd.read_csv(PAC_DIR / "pac_data.csv")
df_stats = pd.read_csv(PAC_DIR / "pac_statistics.csv")
df_gamma = pd.read_csv(PAC_DIR / "gamma_amplitude_statistics.csv")

# ── Panel A: PAC bar chart ──────────────────────────────────────────────────
pac_cols = [c for c in df_pac.columns if c.startswith('pac_')]
ch_labels = [c.replace('pac_EEG ', '') for c in pac_cols]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

ax = axes[0]
means = []
sems = []
p_vals = []
for c in pac_cols:
    rest = df_pac[df_pac['label']==0].groupby('subject_id')[c].mean()
    work = df_pac[df_pac['label']==1].groupby('subject_id')[c].mean()
    common = sorted(set(rest.index) & set(work.index))
    means.append(work[common].mean())
    sems.append(work[common].std() / np.sqrt(len(common)))
    _, p = stats.ttest_rel(work[common], rest[common])
    p_vals.append(p)

x = np.arange(len(ch_labels))
ax.bar(x, means, yerr=sems, color='#d6604d', edgecolor='black', capsize=3)
ax.set_xticks(x)
ax.set_xticklabels(ch_labels, rotation=45, ha='right')
ax.set_ylabel('Modulation Index (MI)')
ax.set_title('A. Theta-Gamma PAC During Arithmetic')

for i, pv in enumerate(p_vals):
    sig = '**' if pv < 0.001 else '*' if pv < 0.05 else 'ns'
    ax.text(x[i], means[i] + sems[i] + 0.00005, sig, ha='center', va='bottom', fontsize=7)

# ── Panel B: Paired PAC comparison ──────────────────────────────────────────
ax = axes[1]
for c in pac_cols:
    rest = df_pac[df_pac['label']==0].groupby('subject_id')[c].mean()
    work = df_pac[df_pac['label']==1].groupby('subject_id')[c].mean()
    common = sorted(set(rest.index) & set(work.index))
    ax.plot([rest[common].values, work[common].values],
            color='gray', alpha=0.15, lw=0.5)
    ax.plot(rest[common].mean(), work[common].mean(), 'o',
            color='#2166ac', markersize=8, markeredgecolor='black')

ax.plot([0.0012, 0.0020], [0.0012, 0.0020], 'k--', alpha=0.3)
ax.set_xlabel('Rest PAC (mean MI)')
ax.set_ylabel('Arithmetic PAC (mean MI)')
ax.set_title('B. PAC: Rest vs Arithmetic')
ax.set_xlim(0.0012, 0.0020)
ax.set_ylim(0.0012, 0.0020)

# ── Panel C: Gamma amplitude comparison ─────────────────────────────────────
ax = axes[2]
gamma_cols = [c for c in df_pac.columns if c.startswith('gamma_amp_')]
gamma_ch_labels = [c.replace('gamma_amp_EEG ', '') for c in gamma_cols]

g_means = []
g_sems = []
for c in gamma_cols:
    rest = df_pac[df_pac['label']==0].groupby('subject_id')[c].mean()
    work = df_pac[df_pac['label']==1].groupby('subject_id')[c].mean()
    common = sorted(set(rest.index) & set(work.index))
    diff = work[common] - rest[common]
    g_means.append(diff.mean())
    g_sems.append(diff.std() / np.sqrt(len(common)))

x = np.arange(len(gamma_ch_labels))
colors = ['#d6604d' if m > 0 else '#2166ac' for m in g_means]
ax.bar(x, g_means, yerr=g_sems, color=colors, edgecolor='black', capsize=3)
ax.axhline(0, color='gray', ls='--', alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(gamma_ch_labels, rotation=45, ha='right')
ax.set_ylabel('$\Delta$ Gamma Amplitude ($\\mu$V)')
ax.set_title('C. Gamma Amplitude: Arithmetic - Rest')

fig.tight_layout()
fig.savefig(PAC_DIR / "figure_pac_results.png", bbox_inches="tight")
fig.savefig(PAPER_FIG / "figure_pac_results.png", bbox_inches="tight")
print(f"Figure saved to paper/figures/figure_pac_results.png")
plt.close("all")

# ── Summary text ────────────────────────────────────────────────────────────
print("\n=== PAC Analysis Summary ===")
n_sig_fdr = (df_stats['p_fdr'] < 0.05).sum()
n_sig = (df_stats['p_value'] < 0.05).sum()
print(f"Channels with significant PAC increase (p<0.05): {n_sig}/{len(df_stats)}")
print(f"Channels with significant PAC increase (FDR): {n_sig_fdr}/{len(df_stats)}")
print(f"Mean Cohen's d across channels: {df_stats['cohen_d'].mean():.3f}")
print(f"Strongest: {df_stats.iloc[0]['channel']} (d={df_stats.iloc[0]['cohen_d']:.3f})")

n_gamma_fdr = (df_gamma['p_fdr'] < 0.05).sum()
print(f"Channels with significant gamma increase (FDR): {n_gamma_fdr}/{len(df_gamma)}")
print(f"Mean Cohen's d for gamma: {df_gamma['cohen_d'].mean():.3f}")
