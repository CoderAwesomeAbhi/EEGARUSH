"""
Improved Theta-Gamma PAC Analysis with Surrogate Testing
=========================================================
Addresses Issue #7: surrogate testing, concatenated epochs, 
comodulograms, permutation-based statistics.
"""

import sys, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal, stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from src.eeg_cogstates.dataset import read_edf, discover_edf_records

OUT = BASE / "outputs_phd_revision"
PAC_DIR = OUT / "pac_analysis_v2"
PAC_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIG = BASE / "paper" / "figures"

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 10})

print("=" * 60)
print("Improved Theta-Gamma PAC with Surrogate Testing")
print("=" * 60)

# ── Parameters ─────────────────────────────────────────────────────────────
THETA_BAND = (4, 8)
GAMMA_BAND = (30, 45)
N_BINS = 18
SFREQ = 256

FRONTAL_CHANS = ['EEG Fp1', 'EEG Fp2', 'EEG F3', 'EEG F4', 'EEG F7', 'EEG F8', 'EEG Fz']
CENTRAL_CHANS = ['EEG C3', 'EEG C4', 'EEG Cz']
ALL_CHANS = FRONTAL_CHANS + CENTRAL_CHANS
N_SURROGATES = 200

# ── Load records ──────────────────────────────────────────────────────────
MAT_DATA_DIR = BASE / "data" / "raw" / "eegmat"
records = discover_edf_records(MAT_DATA_DIR)
print(f"\nFound {len(records)} EDF records ({len(records)//2} subjects)")

# ── PAC functions ─────────────────────────────────────────────────────────
def hilbert_phase(signal_data, fs, band):
    b, a = signal.butter(4, [band[0] / (fs/2), band[1] / (fs/2)], btype='band')
    filtered = signal.filtfilt(b, a, signal_data)
    return np.angle(signal.hilbert(filtered))

def hilbert_amplitude(signal_data, fs, band):
    b, a = signal.butter(4, [band[0] / (fs/2), band[1] / (fs/2)], btype='band')
    filtered = signal.filtfilt(b, a, signal_data)
    return np.abs(signal.hilbert(filtered))

def modulation_index(phase, amplitude, n_bins=18):
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    amp_per_bin = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
        amp_per_bin[i] = np.mean(amplitude[mask])
    if np.sum(amp_per_bin) == 0:
        return 0.0
    p = amp_per_bin / np.sum(amp_per_bin)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    h = -np.sum(p * np.log(p))
    return (np.log(n_bins) - h) / np.log(n_bins)

def compute_pac_for_window(data, fs, theta_band, gamma_band, n_bins=18):
    n_chans = data.shape[0]
    mi_values = np.zeros(n_chans)
    for ch in range(n_chans):
        theta_phase = hilbert_phase(data[ch], fs, theta_band)
        gamma_amp = hilbert_amplitude(data[ch], fs, gamma_band)
        mi_values[ch] = modulation_index(theta_phase, gamma_amp, n_bins)
    return mi_values

def compute_surrogate_mi(phase, amplitude, n_surrogates=200, n_bins=18):
    """Generate surrogate MI distribution by time-shifting amplitude."""
    n = len(amplitude)
    surr_mi = np.zeros(n_surrogates)
    for s in range(n_surrogates):
        shift = np.random.randint(n // 4, 3 * n // 4)
        shifted_amp = np.roll(amplitude, shift)
        surr_mi[s] = modulation_index(phase, shifted_amp, n_bins)
    return surr_mi

# ── Per-subject concatenated PAC ──────────────────────────────────────────
print("\n=== Computing PAC with concatenated epochs + surrogates ===")

subject_results = {}
t0 = time.time()

for subj_id in sorted(set(r.subject_id for r in records)):
    subj_recs = [r for r in records if r.subject_id == subj_id]
    rest_data = []
    work_data = []
    
    for rec in subj_recs:
        try:
            data, fs, ch_names = read_edf(
                rec.path, bandpass=(0.5, 45.0), apply_ica=False,
                rereference="average"
            )
        except Exception as e:
            print(f"  ERROR {rec.path.name}: {e}")
            continue
        
        # Use full recording (no windowing) for longer epochs
        if rec.label == 0:
            rest_data.append(data)
        else:
            work_data.append(data)
    
    if len(rest_data) == 0 or len(work_data) == 0:
        continue
    
    # Concatenate within condition
    rest_concat = np.hstack(rest_data)
    work_concat = np.hstack(work_data)
    
    # Validate concatenated lengths
    for cond_name, cond_data in [("rest", rest_concat), ("work", work_concat)]:
        duration = cond_data.shape[1] / fs
        if duration < 10:
            print(f"  WARNING {subj_id} {cond_name}: only {duration:.1f}s")
    
    # Compute PAC on concatenated data
    ch_indices = [i for i, ch in enumerate(ch_names) if ch in ALL_CHANS]
    
    for cond_name, cond_data in [("rest", rest_concat), ("arithmetic", work_concat)]:
        mi_real = np.zeros(len(ch_indices))
        mi_surr_mean = np.zeros(len(ch_indices))
        mi_surr_std = np.zeros(len(ch_indices))
        mi_z = np.zeros(len(ch_indices))
        mi_p = np.ones(len(ch_indices))
        
        for idx, ch_i in enumerate(ch_indices):
            theta_phase = hilbert_phase(cond_data[ch_i], fs, THETA_BAND)
            gamma_amp = hilbert_amplitude(cond_data[ch_i], fs, GAMMA_BAND)
            mi_real[idx] = modulation_index(theta_phase, gamma_amp, N_BINS)
            
            # Surrogate distribution
            surr_mi = compute_surrogate_mi(theta_phase, gamma_amp, N_SURROGATES, N_BINS)
            mi_surr_mean[idx] = np.mean(surr_mi)
            mi_surr_std[idx] = np.std(surr_mi)
            mi_z[idx] = (mi_real[idx] - mi_surr_mean[idx]) / (mi_surr_std[idx] + 1e-15)
            mi_p[idx] = np.mean(surr_mi >= mi_real[idx]) + 1e-15
        
        key = f"{subj_id}_{cond_name}"
        subject_results[key] = {
            "subject_id": subj_id,
            "condition": cond_name,
            "duration_sec": cond_data.shape[1] / fs,
            "n_samples": cond_data.shape[1],
            "mi_real": mi_real,
            "mi_surr_mean": mi_surr_mean,
            "mi_surr_std": mi_surr_std,
            "mi_z": mi_z,
            "mi_p": mi_p,
            "ch_names": [ch_names[i] for i in ch_indices],
        }

print(f"  Done in {time.time()-t0:.1f}s, {len(subject_results)} condition-recordings")

# ── Statistical analysis ──────────────────────────────────────────────────
print("\n=== Statistical Analysis ===")

stat_rows = []
for ch_idx, ch_name in enumerate(ALL_CHANS):
    rest_mi = []
    work_mi = []
    rest_z = []
    work_z = []
    rest_p = []
    work_p = []
    
    for subj_id in sorted(set(r.subject_id for r in records)):
        rest_key = f"{subj_id}_rest"
        work_key = f"{subj_id}_arithmetic"
        if rest_key in subject_results and work_key in subject_results:
            sr = subject_results[rest_key]
            sw = subject_results[work_key]
            # Find channel index in the subject's channel list
            if ch_name in sr["ch_names"]:
                ci = sr["ch_names"].index(ch_name)
                rest_mi.append(sr["mi_real"][ci])
                rest_z.append(sr["mi_z"][ci])
                rest_p.append(sr["mi_p"][ci])
                work_mi.append(sw["mi_real"][sr["ch_names"].index(ch_name)])
                work_z.append(sw["mi_z"][sr["ch_names"].index(ch_name)])
                work_p.append(sw["mi_p"][sr["ch_names"].index(ch_name)])
    
    rest_mi = np.array(rest_mi)
    work_mi = np.array(work_mi)
    
    if len(rest_mi) < 5:
        continue
    
    diff = work_mi - rest_mi
    t_stat, p_val = stats.ttest_rel(work_mi, rest_mi)
    d_cohen = np.mean(diff) / (np.std(diff, ddof=1) + 1e-15)
    w_stat, wilcoxon_p = stats.wilcoxon(work_mi, rest_mi, alternative="two-sided")
    
    # Proportion of subjects with surrogate-significant PAC
    rest_sig = np.mean([p < 0.05 for p in rest_p])
    work_sig = np.mean([p < 0.05 for p in work_p])
    
    stat_rows.append({
        "channel": ch_name,
        "rest_mi_mean": np.mean(rest_mi),
        "work_mi_mean": np.mean(work_mi),
        "mean_diff": np.mean(diff),
        "t_stat": t_stat,
        "p_value": p_val,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": wilcoxon_p,
        "cohen_d": d_cohen,
        "n_subjects": len(rest_mi),
        "rest_sig_proportion": rest_sig,
        "work_sig_proportion": work_sig,
        "rest_mean_z": np.mean(rest_z),
        "work_mean_z": np.mean(work_z),
    })

df_stats = pd.DataFrame(stat_rows)
df_stats["p_fdr"] = stats.false_discovery_control(df_stats["p_value"].values)
df_stats["wilcoxon_fdr"] = stats.false_discovery_control(df_stats["wilcoxon_p"].values)
df_stats = df_stats.sort_values("p_value")

print("\nConcatenated PAC Results (average reference, surrogates):")
for _, r in df_stats.iterrows():
    sig = " **" if r["p_fdr"] < 0.05 else " *" if r["p_value"] < 0.05 else ""
    print(f"  {r['channel']:12s}: rest={r['rest_mi_mean']:.6f} work={r['work_mi_mean']:.6f} "
          f"d={r['cohen_d']:+.4f} p={r['p_value']:.4f}{sig} "
          f"wilcoxon_p={r['wilcoxon_p']:.4f} rest_sig={r['rest_sig_proportion']:.2f} "
          f"work_sig={r['work_sig_proportion']:.2f}")

df_stats.to_csv(PAC_DIR / "pac_statistics_v2.csv", index=False)

# ── Figure: PAC results ───────────────────────────────────────────────────
print("\n=== Generating Figures ===")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel A: MI bar chart
ax = axes[0]
ch_labels = [c.replace("EEG ", "") for c in df_stats["channel"]]
means = df_stats["work_mi_mean"].values
sems = np.array([df_stats.loc[i, "work_mi_mean"] for i in df_stats.index]) * 0  # placeholder
# Compute SEM from subject-level data
sem_vals = []
for ch_name in df_stats["channel"]:
    work_vals = []
    for subj_id in sorted(set(r.subject_id for r in records)):
        wk = f"{subj_id}_arithmetic"
        if wk in subject_results and ch_name in subject_results[wk]["ch_names"]:
            ci = subject_results[wk]["ch_names"].index(ch_name)
            work_vals.append(subject_results[wk]["mi_real"][ci])
    sem_vals.append(np.std(work_vals) / np.sqrt(len(work_vals)))
sem_vals = np.array(sem_vals)

x = np.arange(len(ch_labels))
colors = ["#d6604d" if p < 0.05 else "#4393c3" for p in df_stats["p_fdr"].values]
ax.bar(x, means, color=colors, edgecolor="black", capsize=3)
ax.errorbar(x, means, yerr=sem_vals, fmt="none", ecolor="black", capsize=3)
ax.set_xticks(x)
ax.set_xticklabels(ch_labels, rotation=45, ha="right")
ax.set_ylabel("Modulation Index (MI)")
ax.set_title("A. Theta-Gamma PAC During Arithmetic")

for i, pv in enumerate(df_stats["p_fdr"].values):
    sig = "**" if pv < 0.001 else "*" if pv < 0.01 else "†" if pv < 0.05 else ""
    ax.text(x[i], means[i] + sem_vals[i] + 0.00003, sig, ha="center", va="bottom", fontsize=8)

# Panel B: Paired comparison
ax = axes[1]
for ch_name in df_stats["channel"]:
    rest_v = []
    work_v = []
    for subj_id in sorted(set(r.subject_id for r in records)):
        rk = f"{subj_id}_rest"
        wk = f"{subj_id}_arithmetic"
        if rk in subject_results and wk in subject_results and ch_name in subject_results[rk]["ch_names"]:
            ci = subject_results[rk]["ch_names"].index(ch_name)
            rest_v.append(subject_results[rk]["mi_real"][ci])
            work_v.append(subject_results[wk]["mi_real"][ci])
    ax.plot([rest_v, work_v], color="gray", alpha=0.1, lw=0.5)
    ax.plot(np.mean(rest_v), np.mean(work_v), "o", color="#2166ac", markersize=8, markeredgecolor="black")

ax.plot([0.001, 0.003], [0.001, 0.003], "k--", alpha=0.3)
ax.set_xlabel("Rest MI (concatenated)")
ax.set_ylabel("Arithmetic MI (concatenated)")
ax.set_title("B. Rest vs Arithmetic")
min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.3)

# Panel C: Z-score (surrogate-normalized)
ax = axes[2]
z_means = df_stats["work_mean_z"].values
z_sems = []
for ch_name in df_stats["channel"]:
    z_vals = []
    for subj_id in sorted(set(r.subject_id for r in records)):
        wk = f"{subj_id}_arithmetic"
        if wk in subject_results and ch_name in subject_results[wk]["ch_names"]:
            ci = subject_results[wk]["ch_names"].index(ch_name)
            z_vals.append(subject_results[wk]["mi_z"][ci])
    z_sems.append(np.std(z_vals) / np.sqrt(len(z_vals)))
z_sems = np.array(z_sems)

ax.bar(x, z_means, yerr=z_sems, color="#d6604d", edgecolor="black", capsize=3)
ax.axhline(1.96, color="gray", ls="--", alpha=0.5, label="p<0.05 (one-tailed)")
ax.set_xticks(x)
ax.set_xticklabels(ch_labels, rotation=45, ha="right")
ax.set_ylabel("PAC Z-score (vs surrogates)")
ax.set_title("C. PAC Effect Size (Surrogate-Normalized)")
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(PAC_DIR / "figure_pac_v2.png", bbox_inches="tight")
fig.savefig(PAPER_FIG / "figure_pac_results.png", bbox_inches="tight")
print(f"  Figure saved")
plt.close("all")

# ── Comodulogram ──────────────────────────────────────────────────────────
print("\n=== Comodulogram: Fz (frontal midline) ===")

phase_freqs = np.arange(2, 14, 2)
amp_freqs = np.arange(20, 50, 4)
fz_idx = ALL_CHANS.index("EEG Fz")

comod_rest = np.zeros((len(phase_freqs), len(amp_freqs)))
comod_work = np.zeros((len(phase_freqs), len(amp_freqs)))

for subj_id in sorted(set(r.subject_id for r in records)):
    wk = f"{subj_id}_arithmetic"
    rk = f"{subj_id}_rest"
    if wk not in subject_results or rk not in subject_results:
        continue
    
    for cond_name, cond_data, comod_mat in [
        ("rest", None, comod_rest), ("arithmetic", None, comod_work)
    ]:
        key = f"{subj_id}_{cond_name}"
        if key not in subject_results:
            continue
        # Need raw data for comodulogram
        subj_recs = [r for r in records if r.subject_id == subj_id]
        concat_data = []
        for rec in subj_recs:
            if rec.label == (0 if cond_name == "rest" else 1):
                data, fs, ch_names = read_edf(rec.path, bandpass=(0.5, 45.0), apply_ica=False, rereference="average")
                concat_data.append(data)
        if not concat_data:
            continue
        concat = np.hstack(concat_data)
        fz_signal = concat[ch_names.index("EEG Fz")]
        
        for pi, pf in enumerate(phase_freqs):
            for ai, af in enumerate(amp_freqs):
                theta_phase = hilbert_phase(fz_signal, fs, (pf, pf+2))
                gamma_amp = hilbert_amplitude(fz_signal, fs, (af, af+4))
                mi = modulation_index(theta_phase, gamma_amp, N_BINS)
                comod_mat[pi, ai] += mi

# Average across subjects
n_subj = len(subject_results) // 2
comod_rest /= n_subj
comod_work /= n_subj

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, comod, title in [
    (axes[0], comod_rest, "Rest"), (axes[1], comod_work, "Arithmetic")
]:
    im = ax.pcolormesh(amp_freqs, phase_freqs, comod, shading="auto",
                       cmap="viridis", vmin=min(comod_rest.min(), comod_work.min()),
                       vmax=max(comod_rest.max(), comod_work.max()))
    ax.set_xlabel("Amplitude Frequency (Hz)")
    ax.set_ylabel("Phase Frequency (Hz)")
    ax.set_title(f"Fz: {title}")
    plt.colorbar(im, ax=ax, label="MI")

fig.suptitle("Comodulogram (Fz)")
fig.tight_layout()
fig.savefig(PAC_DIR / "comodulogram_fz.png", bbox_inches="tight")
fig.savefig(PAPER_FIG / "comodulogram_fz.png", bbox_inches="tight")
print(f"  Comodulogram saved")
plt.close("all")

# ── Summary ────────────────────────────────────────────────────────────────
print(f"\n\n=== SUMMARY ===")
print(f"N subjects: {df_stats['n_subjects'].iloc[0]}")
print(f"Significant channels (paired t, FDR): {(df_stats['p_fdr'] < 0.05).sum()}/{len(df_stats)}")
print(f"Significant channels (Wilcoxon, FDR): {(df_stats['wilcoxon_fdr'] < 0.05).sum()}/{len(df_stats)}")
print(f"Mean Cohen's d: {df_stats['cohen_d'].mean():.3f}")
print(f"Strongest d: {df_stats.iloc[0]['channel']} (d={df_stats.iloc[0]['cohen_d']:.3f})")
print(f"Mean surrogate Z (arithmetic): {df_stats['work_mean_z'].mean():.2f}")
print(f"Proportion surr-sig (arithmetic): {df_stats['work_sig_proportion'].mean():.2f}")
print(f"Mean concat duration: {np.mean([v['duration_sec'] for v in subject_results.values()]):.0f}s")
print(f"\nTotal time: {time.time()-t0:.1f}s")
