"""
Theta-Gamma Phase-Amplitude Coupling Analysis
==============================================
Computes PAC between theta phase (4-8 Hz) and gamma amplitude (30-45 Hz)
on MAT EDF recordings, comparing rest vs arithmetic conditions.

Uses the Modulation Index (MI) from Tort et al. 2010 (J Neurophysiol).
"""

import sys, os, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal, stats
from collections import OrderedDict

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src.eeg_cogstates.dataset import read_edf, discover_edf_records

OUT = BASE / "outputs_phd_revision"
PAC_DIR = OUT / "pac_analysis"
PAC_DIR.mkdir(parents=True, exist_ok=True)
TAB = OUT / "tables"
FIG = OUT / "figures"
PAPER_FIG = BASE / "paper" / "figures"

print("=" * 60)
print("Theta-Gamma Phase-Amplitude Coupling Analysis")
print("=" * 60)

# ── 1. Parameters ──────────────────────────────────────────────────────────
THETA_BAND = (4, 8)
GAMMA_BAND = (30, 45)
N_BINS = 18  # number of phase bins for MI computation
SFREQ = 256  # MAT sampling rate

# Frontal channels for PAC analysis (theta-generating regions)
FRONTAL_CHANS = ['EEG Fp1', 'EEG Fp2', 'EEG F3', 'EEG F4', 'EEG F7', 'EEG F8',
                 'EEG Fz']
CENTRAL_CHANS = ['EEG C3', 'EEG C4', 'EEG Cz']
ALL_CHANS = FRONTAL_CHANS + CENTRAL_CHANS

# ── 2. Load records ────────────────────────────────────────────────────────
MAT_DATA_DIR = BASE / "data" / "raw" / "eegmat"
records = discover_edf_records(MAT_DATA_DIR)
print(f"\nFound {len(records)} EDF records ({len(records)//2} subjects)")

# ── 3. PAC computation functions ───────────────────────────────────────────

def hilbert_phase(signal_data, fs, band):
    """Extract instantaneous phase via Hilbert transform in a frequency band."""
    b, a = signal.butter(4, [band[0] / (fs/2), band[1] / (fs/2)], btype='band')
    filtered = signal.filtfilt(b, a, signal_data)
    return np.angle(signal.hilbert(filtered))

def hilbert_amplitude(signal_data, fs, band):
    """Extract instantaneous amplitude envelope via Hilbert transform."""
    b, a = signal.butter(4, [band[0] / (fs/2), band[1] / (fs/2)], btype='band')
    filtered = signal.filtfilt(b, a, signal_data)
    return np.abs(signal.hilbert(filtered))

def modulation_index(phase, amplitude, n_bins=18):
    """
    Compute Modulation Index (Tort et al. 2010).
    Measures how non-uniformly gamma amplitude is distributed across theta phase.
    MI = (log(N) - H(P)) / log(N) where H(P) is entropy of the amplitude distribution.
    """
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
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
    mi = (np.log(n_bins) - h) / np.log(n_bins)
    return mi

def compute_pac_for_window(data, fs, theta_band, gamma_band, n_bins=18):
    """Compute PAC across all channels for one window."""
    n_chans = data.shape[0]
    mi_values = np.zeros(n_chans)
    
    for ch in range(n_chans):
        theta_phase = hilbert_phase(data[ch], fs, theta_band)
        gamma_amp = hilbert_amplitude(data[ch], fs, gamma_band)
        mi_values[ch] = modulation_index(theta_phase, gamma_amp, n_bins)
    
    return mi_values

def compute_mean_gamma_amp(data, fs, gamma_band):
    """Compute mean gamma amplitude for a window."""
    n_chans = data.shape[0]
    gamma_amp = np.zeros(n_chans)
    for ch in range(n_chans):
        amp = hilbert_amplitude(data[ch], fs, gamma_band)
        gamma_amp[ch] = np.mean(amp)
    return gamma_amp

# ── 4. Per-subject PAC extraction ───────────────────────────────────────────
print("\n=== Computing theta-gamma PAC per window ===")

results = []
t0 = time.time()

for rec in records:
    try:
        data, fs, ch_names = read_edf(
            rec.path, bandpass=(0.5, 45.0), apply_ica=False
        )
    except Exception as e:
        print(f"  ERROR reading {rec.path.name}: {e}")
        continue
    
    # 4-second windows, 50% overlap
    window_samples = int(4.0 * fs)
    step = window_samples // 2
    
    for start in range(0, data.shape[1] - window_samples + 1, step):
        end = start + window_samples
        window = data[:, start:end]
        
        # Compute PAC for frontal+central channels
        mi = compute_pac_for_window(window, fs, THETA_BAND, GAMMA_BAND, N_BINS)
        gamma_amp = compute_mean_gamma_amp(window, fs, GAMMA_BAND)
        
        row = {
            'subject_id': rec.subject_id,
            'condition': rec.condition,
            'label': rec.label,
            'file': rec.path.name,
            'window_start': start / fs,
        }
        
        for i, ch in enumerate(ch_names):
            if ch in ALL_CHANS:
                row[f'pac_{ch}'] = mi[i]
                row[f'gamma_amp_{ch}'] = gamma_amp[i]
        
        results.append(row)

df_pac = pd.DataFrame(results)
print(f"  Computed PAC for {len(df_pac)} windows in {time.time()-t0:.1f}s")
print(f"  Columns: {[c for c in df_pac.columns if 'pac_' in c]}")

# ── 5. Statistical analysis: rest vs arithmetic ────────────────────────────
print("\n=== Rest vs Arithmetic PAC Comparison ===")

pac_cols = [c for c in df_pac.columns if c.startswith('pac_')]
gamma_cols = [c for c in df_pac.columns if c.startswith('gamma_amp_')]

stat_rows = []
for ch_col in pac_cols:
    ch_name = ch_col.replace('pac_', '')
    
    rest_vals = df_pac[df_pac['label'] == 0][ch_col].values
    work_vals = df_pac[df_pac['label'] == 1][ch_col].values
    
    if len(rest_vals) < 5 or len(work_vals) < 5:
        continue
    
    # Subject-level averages
    rest_subj = df_pac[df_pac['label'] == 0].groupby('subject_id')[ch_col].mean()
    work_subj = df_pac[df_pac['label'] == 1].groupby('subject_id')[ch_col].mean()
    common_subj = sorted(set(rest_subj.index) & set(work_subj.index))
    
    if len(common_subj) < 5:
        continue
    
    rest_subj_vals = rest_subj[common_subj].values
    work_subj_vals = work_subj[common_subj].values
    
    diff = work_subj_vals - rest_subj_vals
    t_stat, p_val = stats.ttest_rel(work_subj_vals, rest_subj_vals)
    d_cohen = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)
    
    stat_rows.append({
        'channel': ch_name,
        'rest_mean': np.mean(rest_subj_vals),
        'work_mean': np.mean(work_subj_vals),
        'mean_diff': np.mean(diff),
        't_stat': t_stat,
        'p_value': p_val,
        'cohen_d': d_cohen,
        'n_subjects': len(common_subj),
        'n_windows_rest': len(rest_vals),
        'n_windows_work': len(work_vals),
    })

df_stats = pd.DataFrame(stat_rows)
df_stats['p_fdr'] = stats.false_discovery_control(df_stats['p_value'].values)
df_stats = df_stats.sort_values('p_value')

print("\nPAC Results (sorted by p-value):")
for _, r in df_stats.iterrows():
    sig = ' **' if r['p_fdr'] < 0.05 else ' *' if r['p_value'] < 0.05 else ''
    print(f"  {r['channel']:12s}: rest={r['rest_mean']:.6f} work={r['work_mean']:.6f} "
          f"d={r['cohen_d']:+.4f} p={r['p_value']:.4f}{sig}")

df_stats.to_csv(PAC_DIR / "pac_statistics.csv", index=False)

# ── 6. Gamma amplitude analysis ────────────────────────────────────────────
print("\n=== Gamma Amplitude: Rest vs Arithmetic ===")

gamma_stat_rows = []
for ch_col in gamma_cols:
    ch_name = ch_col.replace('gamma_amp_', '')
    
    rest_subj = df_pac[df_pac['label'] == 0].groupby('subject_id')[ch_col].mean()
    work_subj = df_pac[df_pac['label'] == 1].groupby('subject_id')[ch_col].mean()
    common_subj = sorted(set(rest_subj.index) & set(work_subj.index))
    
    if len(common_subj) < 5:
        continue
    
    rest_vals = rest_subj[common_subj].values
    work_vals = work_subj[common_subj].values
    
    diff = work_vals - rest_vals
    t_stat, p_val = stats.ttest_rel(work_vals, rest_vals)
    d_cohen = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)
    
    gamma_stat_rows.append({
        'channel': ch_name,
        'rest_mean': np.mean(rest_vals),
        'work_mean': np.mean(work_vals),
        'mean_diff': np.mean(diff),
        't_stat': t_stat,
        'p_value': p_val,
        'cohen_d': d_cohen,
        'n_subjects': len(common_subj),
    })

df_gamma = pd.DataFrame(gamma_stat_rows)
df_gamma['p_fdr'] = stats.false_discovery_control(df_gamma['p_value'].values)
df_gamma = df_gamma.sort_values('p_value')

print("\nGamma Amplitude Results (sorted by p-value):")
for _, r in df_gamma.iterrows():
    sig = ' **' if r['p_fdr'] < 0.05 else ' *' if r['p_value'] < 0.05 else ''
    print(f"  {r['channel']:12s}: rest={r['rest_mean']:.4f} work={r['work_mean']:.4f} "
          f"d={r['cohen_d']:+.4f} p={r['p_value']:.4f}{sig}")

df_gamma.to_csv(PAC_DIR / "gamma_amplitude_statistics.csv", index=False)

# ── 7. Summary ──────────────────────────────────────────────────────────────
print(f"\n\n=== SUMMARY ===")
print(f"Files processed: {len(records)}")
print(f"Total windows: {len(df_pac)}")
print(f"Significant PAC channels (p<0.05): {(df_stats['p_value'] < 0.05).sum()}/{len(df_stats)}")
print(f"Significant PAC channels (FDR): {(df_stats['p_fdr'] < 0.05).sum()}/{len(df_stats)}")
print(f"Significant gamma channels (p<0.05): {(df_gamma['p_value'] < 0.05).sum()}/{len(df_gamma)}")
print(f"Significant gamma channels (FDR): {(df_gamma['p_fdr'] < 0.05).sum()}/{len(df_gamma)}")

# Save full PAC data
df_pac.to_csv(PAC_DIR / "pac_data.csv", index=False)
print(f"\nSaved to {PAC_DIR}/")

print(f"\nTotal time: {time.time()-t0:.1f}s")
