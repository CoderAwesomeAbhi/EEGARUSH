#!/usr/bin/env python3
"""
realtime_bci_demo.py
====================
Real-time BCI demo for ISEF: live EEG workload classification
with streaming visualization dashboard.

Simulates a real-time stream from pre-recorded EEG or synthetic data,
applies the SNWA model in real-time, and displays a live dashboard.

Usage:
    python scripts/realtime_bci_demo.py                      # synthetic data
    python scripts/realtime_bci_demo.py --eeg_file data/sample_eeg.npy
    python scripts/realtime_bci_demo.py --fake  --sleep 0.2  # fast synthetic
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
import traceback
import warnings
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ── Conditional imports ──────────────────────────────────────────────────────

MNE_AVAIL = False
MPL_AVAIL = False
SNS_AVAIL = False
SKLEARN_AVAIL = False
NUMPY_AVAIL = PANDAS_AVAIL = SCIPY_AVAIL = False

try:
    import numpy as np
    NUMPY_AVAIL = True
except ImportError:
    raise SystemExit("pip install numpy")

try:
    import pandas as pd
    PANDAS_AVAIL = True
except ImportError:
    raise SystemExit("pip install pandas")

try:
    from scipy.signal import welch
    from scipy.stats import skew, kurtosis
    SCIPY_AVAIL = True
except ImportError:
    raise SystemExit("pip install scipy")

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAIL = True
except ImportError:
    raise SystemExit("pip install scikit-learn")

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import FancyBboxPatch
    MPL_AVAIL = True
except ImportError:
    MPL_AVAIL = False

try:
    import seaborn as sns
    SNS_AVAIL = True
except ImportError:
    SNS_AVAIL = False

# ── Constants ─────────────────────────────────────────────────────────────────

BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0), "gamma": (30.0, 45.0),
}
EPS = 1e-12
FS = 256.0
WINDOW_SEC = 4.0
CHANNELS_19 = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
]

STYLE_DASHBOARD = {
    "bg": "#1a1a2e",
    "card_bg": "#16213e",
    "accent": "#e94560",
    "text": "#eaeaea",
    "green": "#0f9b58",
    "yellow": "#f4a261",
    "blue": "#0072B2",
    "fontsize": 11,
}


# ── Feature extraction ───────────────────────────────────────────────────────

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
        x, fs=float(sfreq), nperseg=nperseg,
        noverlap=nperseg // 2, detrend="constant", scaling="density",
    )
    return freqs, psd


def bandpower(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return 0.0
    return _trapz(psd[mask], freqs[mask])


def extract_features_from_window(data_2d: np.ndarray, sfreq: float, channels: List[str]) -> Dict[str, float]:
    n_ch = min(data_2d.shape[0], len(channels))
    feats: Dict[str, float] = {}
    for i in range(n_ch):
        ch = channels[i]
        x = np.asarray(data_2d[i], dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 0:
            continue
        feats[f"stat_{ch}_mean"] = float(np.mean(x))
        feats[f"stat_{ch}_std"] = float(np.std(x))
        feats[f"stat_{ch}_rms"] = float(np.sqrt(np.mean(x ** 2)))
        feats[f"stat_{ch}_skew"] = float(skew(x, bias=False)) if x.size > 2 and np.std(x) > EPS else 0.0
        feats[f"stat_{ch}_kurtosis"] = float(kurtosis(x, bias=False)) if x.size > 3 and np.std(x) > EPS else 0.0
        h_act, h_mob, h_cmp = hjorth_parameters(x)
        feats[f"hjorth_{ch}_activity"] = h_act
        feats[f"hjorth_{ch}_mobility"] = h_mob
        feats[f"hjorth_{ch}_complexity"] = h_cmp
        freqs, psd = compute_psd(x, sfreq)
        total_power = bandpower(freqs, psd, 0.5, 45.0)
        feats[f"spectral_{ch}_entropy"] = spectral_entropy(psd)
        abs_bps = {}
        for band, (low, high) in BANDS.items():
            bp = bandpower(freqs, psd, low, high)
            abs_bps[band] = bp
            feats[f"band_abs_{ch}_{band}"] = bp
            feats[f"band_rel_{ch}_{band}"] = bp / (total_power + EPS)
        feats[f"ratio_{ch}_theta_alpha"] = abs_bps["theta"] / (abs_bps["alpha"] + EPS)
        feats[f"ratio_{ch}_beta_alpha"] = abs_bps["beta"] / (abs_bps["alpha"] + EPS)
        feats[f"ratio_{ch}_theta_beta"] = abs_bps["theta"] / (abs_bps["beta"] + EPS)
    return feats


# ── SNWA Model ────────────────────────────────────────────────────────────────

class SNWAClassifier:
    def __init__(self, k: int = 8, random_state: int = 42):
        self.k = k
        self.random_state = random_state
        self.selected_features: List[str] = []
        self.feature_weights: Dict[str, float] = {}
        self.rest_median: Dict[str, float] = {}
        self.rest_mad: Dict[str, float] = {}
        self.calibrator: Optional[LogisticRegression] = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "SNWAClassifier":
        rest_mask = y == 0
        work_mask = y == 1
        X_rest = X[rest_mask]
        X_work = X[work_mask]
        self.rest_median = X_rest.median().to_dict()
        self.rest_mad = X_rest.subtract(X_rest.mean()).abs().mean().clip(lower=EPS).to_dict()
        paired_diffs: Dict[str, float] = {}
        for col in X.columns:
            rest_vals = X_rest[col].dropna()
            work_vals = X_work[col].dropna()
            if len(rest_vals) < 3 or len(work_vals) < 3:
                continue
            d = float(work_vals.mean() - rest_vals.mean())
            s = float(np.sqrt((np.var(work_vals) + np.var(rest_vals)) / 2))
            paired_diffs[col] = d / (s + EPS)
        ranked = sorted(paired_diffs.items(), key=lambda x: abs(x[1]), reverse=True)
        self.selected_features = [f for f, _ in ranked[:self.k]]
        self.feature_weights = {f: w for f, w in ranked[:self.k]}
        X_norm = self._normalize(X)
        X_subset = X_norm[self.selected_features]
        scores = X_subset @ np.array([self.feature_weights[f] for f in self.selected_features])
        scores = np.asarray(scores).flatten()
        self.calibrator = LogisticRegression(max_iter=1000, random_state=self.random_state)
        self.calibrator.fit(scores.reshape(-1, 1), y)
        return self

    def _normalize(self, X: pd.DataFrame) -> pd.DataFrame:
        X_norm = X.copy()
        for col in X_norm.columns:
            if col in self.rest_median and col in self.rest_mad:
                X_norm[col] = (X_norm[col] - self.rest_median[col]) / self.rest_mad[col]
        return X_norm

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_norm = self._normalize(X)
        X_subset = X_norm[[c for c in self.selected_features if c in X_norm.columns]]
        missing = set(self.selected_features) - set(X_subset.columns)
        for m in missing:
            X_subset[m] = 0.0
        X_subset = X_subset[self.selected_features]
        scores = X_subset @ np.array([self.feature_weights[f] for f in self.selected_features])
        scores = np.asarray(scores).flatten()
        return self.calibrator.predict_proba(scores.reshape(-1, 1))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ── Simulated EEG stream ──────────────────────────────────────────────────────

class EEGStreamSimulator:
    def __init__(self, n_channels: int = 19, fs: float = FS, seed: int = 42):
        self.n_channels = n_channels
        self.fs = fs
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.state = "rest"
        self.state_duration = 0.0

    def read_window(self, duration: float = WINDOW_SEC) -> Tuple[np.ndarray, int]:
        n_samples = int(duration * self.fs)
        self.state_duration += duration
        if self.state_duration > 8.0 + self.rng.uniform(0, 4):
            self.state = "workload" if self.state == "rest" else "rest"
            self.state_duration = 0.0
        label = 1 if self.state == "workload" else 0
        data = np.zeros((self.n_channels, n_samples), dtype=np.float64)
        for ch in range(self.n_channels):
            t_arr = self.t + np.arange(n_samples) / self.fs
            noise = self.rng.normal(0, 1.0, n_samples) * 0.5
            alpha = 0.5 * np.sin(2 * np.pi * 10 * t_arr) if label == 0 else 0.2 * np.sin(2 * np.pi * 10 * t_arr)
            theta = 0.3 * np.sin(2 * np.pi * 6 * t_arr) if label == 1 else 0.15 * np.sin(2 * np.pi * 6 * t_arr)
            data[ch] = alpha + theta + noise
            if ch in (2, 3, 16):
                data[ch] += theta * 2.0
        self.t += duration
        return data, label


class EEGFilePlayer:
    def __init__(self, filepath: str, fs: float = FS):
        self.data = np.load(filepath)
        if self.data.ndim == 1:
            self.data = self.data[np.newaxis, :]
        self.fs = fs
        self.pos = 0
        self.n_channels, self.n_samples = self.data.shape

    def read_window(self, duration: float = WINDOW_SEC) -> Tuple[np.ndarray, int]:
        n = int(duration * self.fs)
        if self.pos + n > self.n_samples:
            self.pos = 0
        chunk = self.data[:, self.pos:self.pos + n]
        self.pos += n
        return chunk, -1


# ── Live Dashboard ────────────────────────────────────────────────────────────

class RealtimeDashboard:
    def __init__(self, model: SNWAClassifier, stream, channels: List[str]):
        self.model = model
        self.stream = stream
        self.channels = channels
        self.max_points = 100
        self.timeline = deque(maxlen=self.max_points)
        self.scores = deque(maxlen=self.max_points)
        self.predictions = deque(maxlen=self.max_points)
        self.true_labels = deque(maxlen=self.max_points)
        self.theta_powers = deque(maxlen=self.max_points)
        self.time_elapsed = 0.0
        self.correct_count = 0
        self.total_count = 0
        self.is_running = True

        if SNS_AVAIL:
            sns.set_style("darkgrid")
        self.fig = plt.figure(figsize=(16, 9), facecolor=STYLE_DASHBOARD["bg"])
        self._build_layout()

    def _build_layout(self):
        gs = self.fig.add_gridspec(4, 4, hspace=0.35, wspace=0.30)
        self.ax_status = self.fig.add_subplot(gs[0, :])
        self.ax_score = self.fig.add_subplot(gs[1, 0:2])
        self.ax_theta = self.fig.add_subplot(gs[1, 2:])
        self.ax_confusion = self.fig.add_subplot(gs[2, 0])
        self.ax_bandpower = self.fig.add_subplot(gs[2, 1:3])
        self.ax_roc = self.fig.add_subplot(gs[2, 3])
        self.ax_channel = self.fig.add_subplot(gs[3, :])

        for ax in [self.ax_status, self.ax_score, self.ax_theta, self.ax_confusion,
                   self.ax_bandpower, self.ax_roc, self.ax_channel]:
            ax.set_facecolor(STYLE_DASHBOARD["card_bg"])
            ax.tick_params(colors=STYLE_DASHBOARD["text"], labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(STYLE_DASHBOARD["text"])
            ax.title.set_color(STYLE_DASHBOARD["text"])

    def update(self, frame):
        if not self.is_running:
            return

        try:
            data, true_label = self.stream.read_window()
        except Exception:
            self.is_running = False
            return

        feats = extract_features_from_window(data, FS, self.channels)
        X = pd.DataFrame([feats])
        missing_cols = set(self.model.rest_median.keys()) - set(X.columns)
        for c in missing_cols:
            X[c] = 0.0
        X = X[[c for c in self.model.rest_median.keys() if c in X.columns]]

        proba = self.model.predict_proba(X)[0, 1]
        pred = int(proba >= 0.5)

        frontal_ch = "F3"
        theta_key = f"band_abs_{frontal_ch}_theta"
        theta_val = feats.get(theta_key, 0.0)

        self.time_elapsed += WINDOW_SEC
        self.timeline.append(self.time_elapsed)
        self.scores.append(proba)
        self.predictions.append(pred)
        self.true_labels.append(true_label)
        self.theta_powers.append(theta_val)

        if true_label >= 0:
            self.total_count += 1
            if pred == true_label:
                self.correct_count += 1

        accuracy = self.correct_count / max(self.total_count, 1)

        for ax in self.fig.axes:
            ax.clear()
            ax.set_facecolor(STYLE_DASHBOARD["card_bg"])
            ax.tick_params(colors=STYLE_DASHBOARD["text"], labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(STYLE_DASHBOARD["text"])
            ax.title.set_color(STYLE_DASHBOARD["text"])

        state_str = "WORKLOAD" if pred == 1 else "REST"
        state_color = STYLE_DASHBOARD["green"] if pred == 1 else STYLE_DASHBOARD["blue"]

        self.ax_status.set_xlim(0, 1)
        self.ax_status.set_ylim(0, 1)
        self.ax_status.axis("off")
        self.ax_status.text(
            0.5, 0.65, state_str, fontsize=42, fontweight="bold",
            color=state_color, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=STYLE_DASHBOARD["card_bg"],
                      edgecolor=state_color, linewidth=3),
        )
        self.ax_status.text(
            0.22, 0.25, f"Confidence: {proba:.1%}", fontsize=16,
            color=STYLE_DASHBOARD["text"], ha="center", va="center",
        )
        acc_color = STYLE_DASHBOARD["green"] if accuracy >= 0.7 else STYLE_DASHBOARD["yellow"]
        self.ax_status.text(
            0.78, 0.25, f"Accuracy: {accuracy:.0%}", fontsize=16,
            color=acc_color, ha="center", va="center",
        )

        if len(self.timeline) > 1:
            times = list(self.timeline)
            self.ax_score.plot(times, list(self.scores), color=STYLE_DASHBOARD["accent"],
                               linewidth=2, alpha=0.8)
            self.ax_score.axhline(0.5, color="white", linestyle="--", alpha=0.4, linewidth=1)
            self.ax_score.set_ylabel("Workload Score", color=STYLE_DASHBOARD["text"])
            self.ax_score.set_xlabel("Time (s)", color=STYLE_DASHBOARD["text"])
            self.ax_score.set_title("Real-Time SNWA Score", color=STYLE_DASHBOARD["text"])
            self.ax_score.set_ylim(-0.05, 1.05)

            self.ax_theta.plot(times, list(self.theta_powers), color=STYLE_DASHBOARD["green"],
                               linewidth=2, alpha=0.8)
            self.ax_theta.set_ylabel("Frontal Theta Power", color=STYLE_DASHBOARD["text"])
            self.ax_theta.set_xlabel("Time (s)", color=STYLE_DASHBOARD["text"])
            self.ax_theta.set_title("Frontal \u03b8 Power (F3)", color=STYLE_DASHBOARD["text"])

        confusion_data = np.zeros((2, 2), dtype=int)
        for tl, pl in zip(self.true_labels, self.predictions):
            if tl >= 0:
                confusion_data[tl, pl] += 1
        if SNS_AVAIL:
            sns.heatmap(confusion_data, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Pred Rest", "Pred Work"],
                        yticklabels=["True Rest", "True Work"],
                        ax=self.ax_confusion, cbar=False)
        else:
            self.ax_confusion.imshow(confusion_data, cmap="Blues", aspect="auto")
            for (i, j), v in np.ndenumerate(confusion_data):
                self.ax_confusion.text(j, i, str(v), ha="center", va="center",
                                       color="white" if v > confusion_data.max()/2 else "black")
            self.ax_confusion.set_xticks([0, 1])
            self.ax_confusion.set_yticks([0, 1])
            self.ax_confusion.set_xticklabels(["Pred Rest", "Pred Work"])
            self.ax_confusion.set_yticklabels(["True Rest", "True Work"])
        self.ax_confusion.set_title("Confusion Matrix", color=STYLE_DASHBOARD["text"])

        if len(self.true_labels) > 1:
            tl_arr = np.array(self.true_labels)
            sc_arr = np.array(self.scores)
            if len(np.unique(tl_arr)) >= 2:
                from sklearn.metrics import roc_curve, roc_auc_score
                fpr, tpr, _ = roc_curve(tl_arr, sc_arr)
                auc_val = roc_auc_score(tl_arr, sc_arr)
                self.ax_roc.plot(fpr, tpr, color=STYLE_DASHBOARD["accent"], linewidth=2,
                                 label=f"AUC={auc_val:.3f}")
                self.ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.3, color="white")
                self.ax_roc.set_xlabel("FPR", color=STYLE_DASHBOARD["text"])
                self.ax_roc.set_ylabel("TPR", color=STYLE_DASHBOARD["text"])
                self.ax_roc.set_title("Cumulative ROC", color=STYLE_DASHBOARD["text"])
                self.ax_roc.legend(fontsize=9, loc="lower right")

        band_labels = []
        band_vals = []
        for band in BANDS:
            key = f"band_rel_F3_{band}"
            if key in feats:
                band_labels.append(band)
                band_vals.append(feats[key])
        if band_vals:
            colors = ["#0072B2", "#E69F00", "#009E73", "#F0E442", "#D55E00"]
            self.ax_bandpower.barh(band_labels, band_vals, color=colors[:len(band_vals)])
            self.ax_bandpower.set_xlabel("Relative Power", color=STYLE_DASHBOARD["text"])
            self.ax_bandpower.set_title("F3 Bandpower Profile", color=STYLE_DASHBOARD["text"])

        ch_powers = {}
        for ch in self.channels[:8]:
            key = f"band_rel_{ch}_theta"
            if key in feats:
                ch_powers[ch] = feats[key]
        if ch_powers:
            ch_names = list(ch_powers.keys())
            ch_vals = list(ch_powers.values())
            self.ax_channel.bar(ch_names, ch_vals, color=[STYLE_DASHBOARD["accent"],
                               STYLE_DASHBOARD["blue"], STYLE_DASHBOARD["green"],
                               STYLE_DASHBOARD["yellow"], "#CC79A7", "#56B4E9",
                               "#F0E442", "#D55E00"])
            self.ax_channel.set_ylabel("Frontal \u03b8 Power", color=STYLE_DASHBOARD["text"])
            self.ax_channel.set_title("Topography Snapshot", color=STYLE_DASHBOARD["text"])
            self.ax_channel.tick_params(axis="x", rotation=45, labelsize=7)

        plt.tight_layout(pad=2.0)
        return

    def show(self):
        ani = animation.FuncAnimation(self.fig, self.update, interval=WINDOW_SEC * 500,
                                      cache_frame_data=False, blit=False)
        try:
            plt.show()
        except KeyboardInterrupt:
            pass
        self.is_running = False


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time BCI Demo — EEG Workload Classification Dashboard"
    )
    parser.add_argument("--eeg_file", type=str, default=None,
                        help="Path to .npy EEG file (channels x time)")
    parser.add_argument("--fake", action="store_true", default=False,
                        help="Use simulated EEG data (for demo without hardware)")
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="Seconds between windows (speed control)")
    parser.add_argument("--channels", type=int, default=19,
                        help="Number of EEG channels")
    parser.add_argument("--k", type=int, default=8,
                        help="Number of SNWA features")
    parser.add_argument("--no-display", action="store_true",
                        help="Run headless (CLI output only, no GUI)")
    parser.add_argument("--training-data", type=str, default=None,
                        help="CSV of pre-extracted features for model fitting")
    return parser.parse_args()


def train_model_on_data(csv_path: str, k: int) -> SNWAClassifier:
    print(f"  Loading training data from {csv_path}")
    df = pd.read_csv(csv_path)
    label_col = "label" if "label" in df.columns else "condition"
    if label_col == "condition":
        df["label"] = (df["condition"].str.lower().str.contains("work|load|task|math")).astype(int)
    feature_cols = [c for c in df.columns if c not in ("subject_id", "label", "condition",
                                                       "dataset", "file", "window_index")]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df["label"].astype(int).to_numpy()
    model = SNWAClassifier(k=k)
    model.fit(X, y)
    print(f"  Model trained on {len(df)} windows with {k} features")
    return model


def train_model_synthetic(k: int, seed: int = 42) -> SNWAClassifier:
    print("  Generating synthetic training data...")
    rng = np.random.default_rng(seed)
    n_subjects = 20
    n_windows = 100
    n_channels = 19
    chan_list = CHANNELS_19[:n_channels]
    all_rows = []
    for subj in range(n_subjects):
        for _ in range(n_windows):
            label = rng.integers(0, 2)
            baseline_noise = rng.normal(0, 0.8, (n_channels, int(FS * WINDOW_SEC)))
            data = baseline_noise.copy()
            t = np.arange(data.shape[1]) / FS
            if label == 1:
                for ch in (2, 3, 16):
                    theta_amp = 0.4 + rng.uniform(0, 0.2)
                    data[ch] += theta_amp * np.sin(2 * np.pi * (5.5 + rng.uniform(0, 1)) * t)
                for ch in (6, 7):
                    data[ch] -= 0.2 * np.sin(2 * np.pi * 10 * t)
            else:
                for ch in (0, 1, 6, 7):
                    alpha_amp = 0.3 + rng.uniform(0, 0.2)
                    data[ch] += alpha_amp * np.sin(2 * np.pi * (9.5 + rng.uniform(0, 1)) * t)
            feats = extract_features_from_window(data, FS, chan_list)
            feats["subject_id"] = f"syn_{subj}"
            feats["label"] = label
            all_rows.append(feats)
        if subj % 5 == 0:
            print(f"    Generated {subj * n_windows} windows...")
    df = pd.DataFrame(all_rows)
    X = df[[c for c in df.columns if c not in ("subject_id", "label")]].fillna(0)
    y = df["label"].to_numpy()
    model = SNWAClassifier(k=k)
    model.fit(X, y)
    print(f"  Synthetic model trained ({len(df)} windows)")
    return model


def main():
    args = parse_args()
    print("=" * 60)
    print("  EEG Workload Classification — Real-Time Demo")
    print("=" * 60)

    if args.training_data and Path(args.training_data).exists():
        model = train_model_on_data(args.training_data, args.k)
    else:
        model = train_model_synthetic(args.k)

    print("\n  Top 8 SNWA features:")
    for f in model.selected_features[:8]:
        w = model.feature_weights.get(f, 0)
        print(f"    {f:45s}  weight={w:+.3f}")

    if args.eeg_file:
        stream = EEGFilePlayer(args.eeg_file)
    else:
        stream = EEGStreamSimulator(n_channels=args.channels)

    print(f"\n  Stream type: {'file' if args.eeg_file else 'synthetic'}")
    print(f"  SNWA features: {args.k}")
    print(f"  Window: {WINDOW_SEC}s at {FS}Hz")
    print(f"  Sleep between windows: {args.sleep}s\n")

    theta_label = "Frontal " + "\u03b8"
    header = f"{'Window':>8s} | {'Score':>7s} | {'Pred':>7s} | {'True':>5s} | {'Correct':>7s} | {'Accum Acc':>9s} | {theta_label:>10s}"
    print(header)
    print("-" * len(header))

    correct = 0
    total = 0

    try:
        while True:
            data, true_label = stream.read_window()
            feats = extract_features_from_window(data, FS, CHANNELS_19[:args.channels])
            X = pd.DataFrame([feats])
            missing = set(model.rest_median.keys()) - set(X.columns)
            for c in missing:
                X[c] = 0.0
            common = [c for c in model.rest_median.keys() if c in X.columns]
            X = X[common]

            proba = model.predict_proba(X)[0, 1]
            pred = int(proba >= 0.5)

            if true_label >= 0:
                total += 1
                if pred == true_label:
                    correct += 1
                is_correct = "YES" if pred == true_label else "NO"
            else:
                is_correct = "N/A"

            accum_acc = correct / max(total, 1)

            theta_key = "band_abs_F3_theta"
            theta_val = feats.get(theta_key, 0.0)

            print(
                f"{total:>8d} | {proba:>7.3f} | {'WORK' if pred else 'REST':>7s} | "
                f"{str(true_label):>5s} | {is_correct:>7s} | {accum_acc:>8.1%} | {theta_val:>10.4f}"
            )

            if not MPL_AVAIL or args.no_display:
                time.sleep(args.sleep)
                continue

            if total == 1 and args.no_display:
                pass
            elif total == 1 and MPL_AVAIL and not args.no_display:
                print("\n  Launching dashboard... (close plot window to exit)")
                dashboard = RealtimeDashboard(model, stream, CHANNELS_19[:args.channels])
                dashboard.show()
                break

            time.sleep(args.sleep)

    except KeyboardInterrupt:
        print("\n\nDemo stopped by user.")
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"  Final: {correct}/{total} correct ({correct / max(total, 1):.1%})")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
