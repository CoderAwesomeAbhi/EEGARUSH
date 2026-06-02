from __future__ import annotations

import math
import re
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis


BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

EPS = 1e-12


def clean_channel_name(name: str) -> str:
    name = str(name)
    name = re.sub(r"^EEG\s+", "", name, flags=re.IGNORECASE)
    name = name.replace("-REF", "").replace(".", "").replace(" ", "")
    name = re.sub(r"[^A-Za-z0-9]+", "", name)
    return name or "ch"


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def safe_entropy_from_values(x: np.ndarray, bins: int = 20) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3 or np.nanstd(x) < EPS:
        return 0.0
    counts, _ = np.histogram(x, bins=bins)
    p = counts.astype(float)
    p = p[p > 0]
    p = p / np.sum(p)
    return float(-np.sum(p * np.log2(p + EPS)))


def spectral_entropy(psd: np.ndarray) -> float:
    psd = np.asarray(psd, dtype=float)
    psd = np.maximum(psd, 0)
    total = float(np.sum(psd))
    if total <= EPS or psd.size <= 1:
        return 0.0
    p = psd / total
    h = -np.sum(p * np.log2(p + EPS))
    return float(h / np.log2(psd.size))


def hjorth_parameters(x: np.ndarray) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return 0.0, 0.0, 0.0

    dx = np.diff(x)
    ddx = np.diff(dx)

    var0 = float(np.var(x))
    var1 = float(np.var(dx))
    var2 = float(np.var(ddx))

    activity = var0
    mobility = math.sqrt(var1 / (var0 + EPS))
    complexity = math.sqrt(var2 / (var1 + EPS)) / (mobility + EPS)
    return activity, mobility, complexity


def compute_psd(x: np.ndarray, sfreq: float) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    nperseg = int(min(max(32, sfreq * 2), x.size))
    freqs, psd = welch(
        x,
        fs=float(sfreq),
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )
    return freqs, psd


def bandpower(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return 0.0
    return _trapz(psd[mask], freqs[mask])


def infer_region(channel: str) -> str:
    c = clean_channel_name(channel).upper()
    if c.startswith(("FP", "AF", "F")):
        return "frontal"
    if c.startswith("C"):
        return "central"
    if c.startswith("P"):
        return "parietal"
    if c.startswith("O"):
        return "occipital"
    if c.startswith(("T", "FT", "TP")):
        return "temporal"
    return "other"


def infer_hemisphere(channel: str) -> str:
    c = clean_channel_name(channel).upper()
    nums = re.findall(r"\d+", c)
    if not nums:
        return "midline" if c.endswith("Z") else "unknown"
    last_digit = int(nums[-1][-1])
    return "left" if last_digit % 2 == 1 else "right"


def extract_channel_features(x: np.ndarray, sfreq: float, prefix: str) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    feats: Dict[str, float] = {}

    feats[f"stat_{prefix}_mean"] = float(np.mean(x))
    feats[f"stat_{prefix}_std"] = float(np.std(x))
    feats[f"stat_{prefix}_var"] = float(np.var(x))
    feats[f"stat_{prefix}_rms"] = float(np.sqrt(np.mean(x**2)))
    feats[f"stat_{prefix}_ptp"] = float(np.ptp(x))
    feats[f"stat_{prefix}_skew"] = float(skew(x, bias=False)) if x.size > 2 and np.std(x) > EPS else 0.0
    feats[f"stat_{prefix}_kurtosis"] = float(kurtosis(x, bias=False)) if x.size > 3 and np.std(x) > EPS else 0.0
    feats[f"stat_{prefix}_shannon_entropy"] = safe_entropy_from_values(x)

    activity, mobility, complexity = hjorth_parameters(x)
    feats[f"hjorth_{prefix}_activity"] = activity
    feats[f"hjorth_{prefix}_mobility"] = mobility
    feats[f"hjorth_{prefix}_complexity"] = complexity

    freqs, psd = compute_psd(x, sfreq)
    total_power = bandpower(freqs, psd, 0.5, 45.0)
    feats[f"spectral_{prefix}_entropy"] = spectral_entropy(psd)

    absolute_bandpowers = {}
    for band, (low, high) in BANDS.items():
        bp = bandpower(freqs, psd, low, high)
        absolute_bandpowers[band] = bp
        feats[f"band_abs_{prefix}_{band}"] = bp
        feats[f"band_rel_{prefix}_{band}"] = bp / (total_power + EPS)

    feats[f"ratio_{prefix}_theta_alpha"] = absolute_bandpowers["theta"] / (absolute_bandpowers["alpha"] + EPS)
    feats[f"ratio_{prefix}_beta_alpha"] = absolute_bandpowers["beta"] / (absolute_bandpowers["alpha"] + EPS)
    feats[f"ratio_{prefix}_theta_beta"] = absolute_bandpowers["theta"] / (absolute_bandpowers["beta"] + EPS)

    return feats


def extract_window_features(
    window: np.ndarray,
    sfreq: float,
    channel_names: Iterable[str],
    include_connectivity: bool = True,
) -> Dict[str, float]:
    window = np.asarray(window, dtype=float)
    channel_names = [clean_channel_name(ch) for ch in channel_names]

    if window.ndim != 2:
        raise ValueError(f"Expected window shape (channels, samples), got {window.shape}")
    if len(channel_names) != window.shape[0]:
        channel_names = [f"ch{i:02d}" for i in range(window.shape[0])]

    feats: Dict[str, float] = {}

    per_channel_features = []
    for idx, ch in enumerate(channel_names):
        ch_feats = extract_channel_features(window[idx], sfreq, ch)
        feats.update(ch_feats)
        per_channel_features.append((ch, ch_feats))

    for band in BANDS:
        abs_vals = [feats[f"band_abs_{ch}_{band}"] for ch, _ in per_channel_features]
        rel_vals = [feats[f"band_rel_{ch}_{band}"] for ch, _ in per_channel_features]
        feats[f"global_band_abs_mean_{band}"] = float(np.mean(abs_vals))
        feats[f"global_band_abs_std_{band}"] = float(np.std(abs_vals))
        feats[f"global_band_rel_mean_{band}"] = float(np.mean(rel_vals))
        feats[f"global_band_rel_std_{band}"] = float(np.std(rel_vals))

    regions = sorted(set(infer_region(ch) for ch in channel_names))
    for region in regions:
        idxs = [i for i, ch in enumerate(channel_names) if infer_region(ch) == region]
        if not idxs:
            continue
        for band in BANDS:
            vals = [feats[f"band_rel_{channel_names[i]}_{band}"] for i in idxs]
            feats[f"region_{region}_band_rel_mean_{band}"] = float(np.mean(vals))

    for hemi in ["left", "right", "midline"]:
        idxs = [i for i, ch in enumerate(channel_names) if infer_hemisphere(ch) == hemi]
        if not idxs:
            continue
        for band in BANDS:
            vals = [feats[f"band_rel_{channel_names[i]}_{band}"] for i in idxs]
            feats[f"hemisphere_{hemi}_band_rel_mean_{band}"] = float(np.mean(vals))

    if include_connectivity and window.shape[0] > 1:
        corr = np.corrcoef(window)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        upper_vals = []
        left_right_vals = []
        for i in range(window.shape[0]):
            for j in range(i + 1, window.shape[0]):
                value = float(corr[i, j])
                upper_vals.append(value)
                ch_i, ch_j = channel_names[i], channel_names[j]
                feats[f"corr_{ch_i}_{ch_j}"] = value
                if {infer_hemisphere(ch_i), infer_hemisphere(ch_j)} == {"left", "right"}:
                    left_right_vals.append(value)
        if upper_vals:
            feats["connectivity_corr_mean"] = float(np.mean(upper_vals))
            feats["connectivity_corr_std"] = float(np.std(upper_vals))
            feats["connectivity_corr_abs_mean"] = float(np.mean(np.abs(upper_vals)))
        if left_right_vals:
            feats["connectivity_left_right_corr_mean"] = float(np.mean(left_right_vals))
            feats["connectivity_left_right_corr_abs_mean"] = float(np.mean(np.abs(left_right_vals)))

    for key, value in list(feats.items()):
        if not np.isfinite(value):
            feats[key] = 0.0

    return feats
