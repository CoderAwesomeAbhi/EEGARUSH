#!/usr/bin/env python3
"""
multi_dataset_pipeline.py
=========================
Unified multi-dataset EEG workload analysis for ISEF-level validation.

Downloads/integrates 3 public datasets, extracts consistent feature sets,
runs LOSO validation, and produces a combined analysis.

Usage:
    python multi_dataset_pipeline.py [--datasets all|mat|stew|ds007262]
    python multi_dataset_pipeline.py --quick  (reduced features for speed)
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ------- Conditional imports -------

NUMPY_AVAIL = PANDAS_AVAIL = SCIPY_AVAIL = SKLEARN_AVAIL = False
MPL_AVAIL = SNS_AVAIL = False
DATASETS_AVAIL = MNE_AVAIL = False

try:
    import numpy as np

    NUMPY_AVAIL = True
except ImportError:
    raise SystemExit("numpy is required (pip install numpy)")

try:
    import pandas as pd

    PANDAS_AVAIL = True
except ImportError:
    raise SystemExit("pandas is required (pip install pandas)")

try:
    from scipy.signal import welch
    from scipy.stats import kurtosis, skew, pearsonr

    SCIPY_AVAIL = True
except ImportError:
    raise SystemExit("scipy is required (pip install scipy)")

try:
    from sklearn.base import clone
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    SKLEARN_AVAIL = True
except ImportError:
    raise SystemExit("scikit-learn is required (pip install scikit-learn)")

try:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")
    MPL_AVAIL = True
except ImportError:
    MPL_AVAIL = False

try:
    import seaborn as sns

    SNS_AVAIL = True
except ImportError:
    SNS_AVAIL = False

try:
    import mne

    MNE_AVAIL = True
except ImportError:
    MNE_AVAIL = False

try:
    from datasets import load_dataset

    DATASETS_AVAIL = True
except ImportError:
    DATASETS_AVAIL = False


# ------- Paths -------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAT_FEATURES_CSV = PROJECT_ROOT / "outputs" / "features" / "eeg_features.csv"
DS007262_FEATURES_CSV = (
    PROJECT_ROOT / "external_validation_ds007262" / "ds007262_low_high_features.csv"
)
DS007262_PREDICTIONS_CSV = (
    PROJECT_ROOT / "external_validation_ds007262" / "ds007262_low_high_predictions.csv"
)
STEW_FALLBACK_CSV = PROJECT_ROOT / "external_data" / "stew_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "multi_dataset"


# ------- Band definitions (matching existing codebase) -------

BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
EPS = 1e-12


# ------- Channel definitions -------

MAT_CHANNELS = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
    "Fz",
    "Cz",
    "Pz",
]

STEW_CHANNELS_ORIG = [
    "AF3",
    "F7",
    "F3",
    "FC5",
    "T7",
    "P7",
    "O1",
    "O2",
    "P8",
    "T8",
    "FC6",
    "F4",
    "F8",
    "AF4",
]

DS007262_CHANNELS = [
    "FP1",
    "FP2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
    "Fz",
    "Cz",
    "Pz",
]

# Map STEW nomenclature (new 10-20) to old 10-20 used by MAT/DS007262
STEW_TO_OLD = {
    "T7": "T3",
    "T8": "T4",
    "P7": "T5",
    "P8": "T6",
    "FC5": "FC5",
    "FC6": "FC6",
    "AF3": "AF3",
    "AF4": "AF4",
}

# After mapping old-10-20 names, which STEW channels overlap with MAT/DS007262?
STEW_CHANNELS_MAPPED = [STEW_TO_OLD.get(ch, ch) for ch in STEW_CHANNELS_ORIG]

# Common channels across all three datasets (using old 10-20 naming)
COMMON_CHANNELS_RAW = sorted(set(MAT_CHANNELS) & set(STEW_CHANNELS_MAPPED) & set(DS007262_CHANNELS))
# Normalize to title-case (Fp1 not FP1)
COMMON_CHANNELS = sorted({ch.title() for ch in COMMON_CHANNELS_RAW})

# Per-feature template names
FEATURE_TYPES = [
    "stat_{ch}_mean",
    "stat_{ch}_std",
    "stat_{ch}_var",
    "stat_{ch}_rms",
    "stat_{ch}_ptp",
    "stat_{ch}_skew",
    "stat_{ch}_kurtosis",
    "stat_{ch}_shannon_entropy",
    "hjorth_{ch}_activity",
    "hjorth_{ch}_mobility",
    "hjorth_{ch}_complexity",
    "spectral_{ch}_entropy",
    "band_abs_{ch}_delta",
    "band_rel_{ch}_delta",
    "band_abs_{ch}_theta",
    "band_rel_{ch}_theta",
    "band_abs_{ch}_alpha",
    "band_rel_{ch}_alpha",
    "band_abs_{ch}_beta",
    "band_rel_{ch}_beta",
    "band_abs_{ch}_gamma",
    "band_rel_{ch}_gamma",
    "ratio_{ch}_theta_alpha",
    "ratio_{ch}_beta_alpha",
    "ratio_{ch}_theta_beta",
]


# ------- Helper functions -------


def clean_channel_name(name: str) -> str:
    name = str(name)
    name = re.sub(r"^EEG\s+", "", name, flags=re.IGNORECASE)
    name = name.replace("-REF", "").replace(".", "").replace(" ", "")
    name = re.sub(r"[^A-Za-z0-9]+", "", name)
    return name or "ch"


def normalize_channel(ch: str) -> str:
    """Normalize channel names across datasets (FP1 -> Fp1, etc.)."""
    ch = ch.strip().upper()
    # Handle 'FP' -> 'Fp'  and similar
    mapping = {
        "FP1": "Fp1",
        "FP2": "Fp2",
        "FPZ": "Fpz",
        "FZ": "Fz",
        "CZ": "Cz",
        "PZ": "Pz",
        "OZ": "Oz",
    }
    if ch in mapping:
        return mapping[ch]
    # Title case for standard 10-20
    if len(ch) == 2:
        return ch[0] + ch[1].lower()
    if len(ch) == 3:
        return ch[0] + ch[1].lower() + ch[2]
    return ch


def generate_feature_names(channels: List[str]) -> List[str]:
    """Generate all expected feature column names for a given channel list."""
    names = []
    for ch in channels:
        for template in FEATURE_TYPES:
            names.append(template.format(ch=ch))
    return names


def find_available_common_features(
    df: pd.DataFrame, common_channels: List[str]
) -> List[str]:
    """Find which common-channel features exist in the given dataframe."""
    expected = set(generate_feature_names(common_channels))
    actual = set(df.columns)
    return sorted(expected & actual)


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


def extract_channel_features(x: np.ndarray, sfreq: float, prefix: str) -> Dict[str, float]:
    """Extract feature dict for one channel. Same convention as existing codebase."""
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    feats: Dict[str, float] = {}
    feats[f"stat_{prefix}_mean"] = float(np.mean(x))
    feats[f"stat_{prefix}_std"] = float(np.std(x))
    feats[f"stat_{prefix}_var"] = float(np.var(x))
    feats[f"stat_{prefix}_rms"] = float(np.sqrt(np.mean(x ** 2)))
    feats[f"stat_{prefix}_ptp"] = float(np.ptp(x))
    feats[f"stat_{prefix}_skew"] = (
        float(skew(x, bias=False)) if x.size > 2 and np.std(x) > EPS else 0.0
    )
    feats[f"stat_{prefix}_kurtosis"] = (
        float(kurtosis(x, bias=False)) if x.size > 3 and np.std(x) > EPS else 0.0
    )
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
    feats[f"ratio_{prefix}_theta_alpha"] = absolute_bandpowers["theta"] / (
        absolute_bandpowers["alpha"] + EPS
    )
    feats[f"ratio_{prefix}_beta_alpha"] = absolute_bandpowers["beta"] / (
        absolute_bandpowers["alpha"] + EPS
    )
    feats[f"ratio_{prefix}_theta_beta"] = absolute_bandpowers["theta"] / (
        absolute_bandpowers["beta"] + EPS
    )
    return feats


def extract_window_features(
    data_2d: np.ndarray, sfreq: float, channel_names: List[str]
) -> Dict[str, float]:
    """Extract features from a 2D array (channels × time)."""
    feats: Dict[str, float] = {}
    for idx, ch in enumerate(channel_names):
        feats.update(extract_channel_features(data_2d[idx], sfreq, ch))
    return feats


# ------- Dataset loaders -------


def load_mat_dataset(
    csv_path: Path = MAT_FEATURES_CSV,
) -> Optional[pd.DataFrame]:
    """Load precomputed MAT (PhysioNet EEGMAT) features."""
    if not csv_path.exists():
        print(f"  [MAT] Features CSV not found at {csv_path}")
        return None
    print(f"  [MAT] Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    # Ensure subject_id and label exist
    if "subject_id" not in df.columns or "label" not in df.columns:
        print("  [MAT] ERROR: missing subject_id or label columns")
        return None
    df["subject_id"] = df["subject_id"].astype(str)
    df["label"] = df["label"].astype(int)
    df["dataset"] = "MAT"
    print(f"  [MAT] Loaded {len(df)} windows, {df['subject_id'].nunique()} subjects")
    return df


def load_stew_dataset(
    output_dir: Path,
    quick: bool = False,
) -> Optional[pd.DataFrame]:
    """Download (if possible) and extract features from STEW dataset."""
    stew_cache = output_dir / "stew_features.parquet"

    if stew_cache.exists():
        print("  [STEW] Loading cached features ...")
        df = pd.read_parquet(stew_cache)
        df["dataset"] = "STEW"
        print(f"  [STEW] Loaded {len(df)} windows, {df['subject_id'].nunique()} subjects")
        return df

    # Try HuggingFace datasets
    if DATASETS_AVAIL:
        print("  [STEW] Attempting download from HuggingFace (monster-monash/STEW) ...")
        try:
            return _load_stew_from_huggingface(output_dir, stew_cache, quick)
        except Exception as exc:
            print(f"  [STEW] HuggingFace load failed: {exc}")
            print(traceback.format_exc())
    else:
        print("  [STEW] HuggingFace `datasets` not installed; pip install datasets")

    # Try fallback CSV
    if STEW_FALLBACK_CSV.exists():
        print(f"  [STEW] Loading fallback CSV: {STEW_FALLBACK_CSV}")
        df = pd.read_csv(STEW_FALLBACK_CSV)
        if "subject_id" not in df.columns or "label" not in df.columns:
            print("  [STEW] Fallback CSV missing required columns")
            return None
        df["subject_id"] = df["subject_id"].astype(str)
        df["label"] = df["label"].astype(int)
        df["dataset"] = "STEW"
        df.to_parquet(stew_cache, index=False)
        print(f"  [STEW] Loaded {len(df)} windows, {df['subject_id'].nunique()} subjects")
        return df

    print(
        "  [STEW] Could not load. Provide a pre-extracted CSV at:\n"
        f"         {STEW_FALLBACK_CSV}\n"
        "         Expected columns: common feature columns + subject_id, label"
    )
    return None


def _load_stew_from_huggingface(
    output_dir: Path, cache_path: Path, quick: bool
) -> pd.DataFrame:
    """Download STEW from HuggingFace using hub API, extract features."""
    from huggingface_hub import hf_hub_download

    print("  [STEW] Downloading raw data files from HuggingFace Hub ...")
    X_path = hf_hub_download(repo_id="monster-monash/STEW", repo_type="dataset", filename="STEW_X.npy")
    y_path = hf_hub_download(repo_id="monster-monash/STEW", repo_type="dataset", filename="STEW_y.npy")
    subj_path = hf_hub_download(repo_id="monster-monash/STEW", repo_type="dataset", filename="STEW_subject_id.csv")

    X = np.load(X_path).astype(np.float64)
    y = np.load(y_path).astype(np.int64)
    subj_df = pd.read_csv(subj_path)

    n_total = X.shape[0]
    max_samples = 500 if quick else 2000
    if n_total > max_samples:
        print(f"  [STEW] Using {max_samples} of {n_total} samples (use --quick for fewer)")
        rng = np.random.default_rng(42)
        idx = rng.choice(n_total, max_samples, replace=False)
        idx.sort()
        X = X[idx]
        y = y[idx]
        subj_df = subj_df.iloc[idx]
        n_total = max_samples

    if X.ndim == 2:
        if X.shape[1] == 14 * 256:
            X = X.reshape(n_total, 14, 256)
        elif X.shape[1] == 256 and X.shape[0] > 14:
            X = X.reshape(-1, 14, 256)
    elif X.ndim == 3 and X.shape[2] == 14:
        X = X.transpose(0, 2, 1)

    subject_ids = subj_df.iloc[:, 0].astype(str).to_numpy()
    if len(subject_ids) != n_total:
        subject_ids = np.array([f"STEW_{i}" for i in range(n_total)])

    channel_order = [normalize_channel(STEW_TO_OLD.get(ch, ch)) for ch in STEW_CHANNELS_ORIG]
    sfreq = 128.0

    rows: List[Dict[str, Any]] = []
    batch_size = 200
    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        batch_rows = []
        for i in range(start, end):
            eeg_data = np.asarray(X[i], dtype=float)
            label = int(y[i])
            subject = str(subject_ids[i]) if i < len(subject_ids) else f"STEW_{i}"
            if eeg_data.shape[0] != 14 and eeg_data.shape[-1] == 14:
                eeg_data = eeg_data.T
            feats = extract_window_features(eeg_data, sfreq, channel_order)
            feats["subject_id"] = subject
            feats["label"] = label
            feats["condition"] = "workload" if label == 1 else "rest"
            batch_rows.append(feats)
        rows.extend(batch_rows)
        print(f"  [STEW] Processed {end}/{n_total} samples ({len(rows)} windows) ...")
        # Partial save every batch
        if end % 1000 == 0 or end == n_total:
            pd.DataFrame(rows).to_parquet(cache_path.with_suffix(".partial.parquet"), index=False)

    if not rows:
        raise RuntimeError("No valid samples extracted from STEW dataset")

    df = pd.DataFrame(rows)
    df["dataset"] = "STEW"
    df.to_parquet(cache_path, index=False)
    if cache_path.with_suffix(".partial.parquet").exists():
        cache_path.with_suffix(".partial.parquet").unlink()
    print(
        f"  [STEW] Extracted {len(df)} windows, {df['subject_id'].nunique()} subjects"
    )
    return df


def load_ds007262_dataset(
    csv_path: Path = DS007262_FEATURES_CSV,
) -> Optional[pd.DataFrame]:
    """Load precomputed DS007262 features."""
    if not csv_path.exists():
        print(f"  [DS007262] Features CSV not found at {csv_path}")
        return None
    print(f"  [DS007262] Loading {csv_path} ...")
    df = pd.read_csv(csv_path)

    # DS007262 might have FP1/FP2 (uppercase) — normalize column names
    rename_map = {}
    for col in df.columns:
        norm = normalize_channel(col)
        if col != norm and "stat_" not in col and "band_" not in col and "hjorth_" not in col and "ratio_" not in col and "spectral_" not in col and "global_" not in col and "region_" not in col and "hemisphere_" not in col and "corr_" not in col and "connectivity_" not in col:
            # Only normalize channel names within feature columns
            pass
        # Check if this is a feature column with channel name to normalize
        for ch_variation in ["FP1", "FP2", "FPZ", "FZ", "CZ", "PZ", "OZ"]:
            if ch_variation in col:
                replacement = normalize_channel(ch_variation)
                new_col = col.replace(ch_variation, replacement)
                if new_col != col:
                    rename_map[col] = new_col

    if rename_map:
        df = df.rename(columns=rename_map)

    # Also rename columns that might have channel names embedded differently
    # Check for subject_id column (might be 'subject')
    if "subject" in df.columns and "subject_id" not in df.columns:
        df = df.rename(columns={"subject": "subject_id"})

    if "subject_id" not in df.columns or "label" not in df.columns:
        print(f"  [DS007262] Missing required columns. Found: {list(df.columns)[-10:]}")
        return None

    df["subject_id"] = df["subject_id"].astype(str)
    df["label"] = df["label"].astype(int)
    df["dataset"] = "DS007262"
    print(f"  [DS007262] Loaded {len(df)} windows, {df['subject_id'].nunique()} subjects")
    return df


# ------- Feature harmonization -------


def harmonize_features(
    datasets: Dict[str, pd.DataFrame], common_channels: List[str]
) -> Dict[str, pd.DataFrame]:
    """Filter each dataset to only the common-channel features that exist in ALL datasets."""
    all_feature_sets = {}
    for name, df in datasets.items():
        avail = find_available_common_features(df, common_channels)
        all_feature_sets[name] = set(avail)

    # Intersection of available features across all datasets
    common_feats = set.intersection(*all_feature_sets.values()) if all_feature_sets else set()
    common_feats = sorted(common_feats)

    print(f"\n  Common features across all datasets: {len(common_feats)}")

    # Also keep metadata columns
    meta_cols = {"subject_id", "label", "condition", "dataset", "file", "window_index"}

    harmonized = {}
    for name, df in datasets.items():
        keep = [c for c in common_feats if c in df.columns] + [
            c for c in meta_cols if c in df.columns
        ]
        harm_df = df[keep].copy()
        harm_df = harm_df.replace([np.inf, -np.inf], np.nan)
        # Drop rows where all features are NaN
        feat_cols = [c for c in harm_df.columns if c in common_feats]
        harm_df = harm_df.dropna(subset=feat_cols, how="all")
        harmonized[name] = harm_df
        print(f"  {name}: {len(harm_df)} windows, {harm_df['subject_id'].nunique()} subjects")

    return harmonized


# ------- Classification helpers -------


def make_models(random_state: int = 42) -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            max_iter=5000, class_weight="balanced", solver="lbfgs", random_state=random_state
        ),
        "svm_rbf": SVC(
            kernel="rbf", C=1.0, gamma="scale", probability=True,
            class_weight="balanced", random_state=random_state,
        ),
    }
    try:
        from sklearn.ensemble import RandomForestClassifier
        models["random_forest"] = RandomForestClassifier(
            n_estimators=300, max_features="sqrt",
            class_weight="balanced", random_state=random_state, n_jobs=-1,
        )
    except Exception:
        pass
    return models


def make_pipeline(model: Any) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def positive_scores(estimator: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X)
        return (scores - scores.min()) / (scores.max() - scores.min() + EPS)
    return estimator.predict(X).astype(float)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: Optional[np.ndarray] = None
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_score is not None:
        y_score = np.asarray(y_score, dtype=float)
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            out["roc_auc"] = np.nan
    else:
        out["roc_auc"] = np.nan
    return out


# ------- Per-dataset LOSO -------


def run_loso_classification(
    df: pd.DataFrame,
    dataset_name: str,
    feature_cols: List[str],
    output_dir: Path,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run leave-one-subject-out CV for a single dataset."""
    print(f"\n{'=' * 60}")
    print(f"  LOSO Classification: {dataset_name}")
    print(f"{'=' * 60}")

    X = df[feature_cols].copy()
    y = df["label"].to_numpy()
    groups = df["subject_id"].to_numpy()

    models = make_models(random_state=random_state)
    logo = LeaveOneGroupOut()

    pred_rows = []
    for model_name, model in models.items():
        for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups)):
            pipe = make_pipeline(clone(model))
            X_train = X.iloc[train_idx]
            y_train = y[train_idx]
            X_test = X.iloc[test_idx]
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_score = positive_scores(pipe, X_test)
            for idx, tl, pl, sc, subj in zip(
                test_idx, y[test_idx], y_pred, y_score, groups[test_idx]
            ):
                pred_rows.append({
                    "dataset": dataset_name,
                    "subject_id": subj,
                    "model": model_name,
                    "true_label": int(tl),
                    "pred_label": int(pl),
                    "score_workload": float(sc),
                    "fold": int(fold),
                })
        print(f"    {model_name}: done")

    preds_df = pd.DataFrame(pred_rows)
    metric_rows = []
    for model_name, grp in preds_df.groupby("model"):
        metrics = compute_metrics(
            grp["true_label"].to_numpy(),
            grp["pred_label"].to_numpy(),
            grp["score_workload"].to_numpy(),
        )
        metrics["model"] = model_name
        metrics["dataset"] = dataset_name
        metric_rows.append(metrics)

    metrics_df = pd.DataFrame(metric_rows).sort_values("roc_auc", ascending=False)
    print(f"\n  Results for {dataset_name}:")
    for _, row in metrics_df.iterrows():
        print(
            f"    {row['model']:25s}  Acc={row['accuracy']:.3f}  "
            f"AUC={row['roc_auc']:.3f}  F1={row['f1']:.3f}  "
            f"Sens={row['sensitivity']:.3f}  Spec={row['specificity']:.3f}"
        )
    return metrics_df, preds_df


# ------- Combined LOSO -------


def run_combined_loso(
    harmonized: Dict[str, pd.DataFrame],
    feature_cols: List[str],
    output_dir: Path,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pool all subjects across datasets and run LOSO."""
    print(f"\n{'=' * 60}")
    print("  Combined LOSO (all datasets pooled)")
    print(f"{'=' * 60}")

    parts = []
    for name, df in harmonized.items():
        parts.append(df[["subject_id", "label", "dataset"] + feature_cols].copy())
    combined = pd.concat(parts, ignore_index=True)
    # Make subject IDs unique across datasets
    combined["orig_subject"] = combined["subject_id"]
    combined["subject_id"] = combined["dataset"] + "_" + combined["subject_id"]

    print(f"  Total subjects: {combined['subject_id'].nunique()}")
    print(f"  Total windows: {len(combined)}")

    X = combined[feature_cols].copy()
    y = combined["label"].to_numpy()
    groups = combined["subject_id"].to_numpy()

    models = make_models(random_state=random_state)
    logo = LeaveOneGroupOut()

    pred_rows = []
    for model_name, model in models.items():
        for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups)):
            pipe = make_pipeline(clone(model))
            pipe.fit(X.iloc[train_idx], y[train_idx])
            y_pred = pipe.predict(X.iloc[test_idx])
            y_score = positive_scores(pipe, X.iloc[test_idx])
            for idx, tl, pl, sc, subj, orig_subj, ds_name in zip(
                test_idx,
                y[test_idx],
                y_pred,
                y_score,
                groups[test_idx],
                combined["orig_subject"].iloc[test_idx],
                combined["dataset"].iloc[test_idx],
            ):
                pred_rows.append({
                    "dataset": "COMBINED",
                    "source_dataset": ds_name,
                    "subject_id": subj,
                    "orig_subject": orig_subj,
                    "model": model_name,
                    "true_label": int(tl),
                    "pred_label": int(pl),
                    "score_workload": float(sc),
                    "fold": int(fold),
                })
        print(f"    {model_name}: done")

    preds_df = pd.DataFrame(pred_rows)
    metric_rows = []
    for model_name, grp in preds_df.groupby("model"):
        metrics = compute_metrics(
            grp["true_label"].to_numpy(),
            grp["pred_label"].to_numpy(),
            grp["score_workload"].to_numpy(),
        )
        metrics["model"] = model_name
        metrics["dataset"] = "COMBINED"
        metric_rows.append(metrics)

    metrics_df = pd.DataFrame(metric_rows).sort_values("roc_auc", ascending=False)
    print(f"\n  Combined Results:")
    for _, row in metrics_df.iterrows():
        print(
            f"    {row['model']:25s}  Acc={row['accuracy']:.3f}  "
            f"AUC={row['roc_auc']:.3f}  F1={row['f1']:.3f}  "
            f"Sens={row['sensitivity']:.3f}  Spec={row['specificity']:.3f}"
        )
    return metrics_df, preds_df


# ------- Biological finding analysis -------


def run_biological_analysis(
    harmonized: Dict[str, pd.DataFrame],
    common_channels: List[str],
    output_dir: Path,
) -> pd.DataFrame:
    """
    Compute per-subject frontal theta, theta/alpha ratio,
    and correlate with task performance.
    """
    print(f"\n{'=' * 60}")
    print("  Biological Finding Analysis (Frontal Theta)")
    print(f"{'=' * 60}")

    # Frontal channels (F3, F4, Fz if available)
    frontal = [ch for ch in ["F3", "F4", "Fz"] if ch in common_channels]
    theta_cols = [f"band_abs_{ch}_theta" for ch in frontal]
    alpha_cols = [f"band_abs_{ch}_alpha" for ch in frontal]

    results = []
    for name, df in harmonized.items():
        avail_theta = [c for c in theta_cols if c in df.columns]
        avail_alpha = [c for c in alpha_cols if c in df.columns]

        if not avail_theta or not avail_alpha:
            print(f"  {name}: skipping (missing frontal theta/alpha features)")
            continue

        # Per-subject averages
        subj_stats = []
        for subj_id, subj_df in df.groupby("subject_id"):
            rest = subj_df[subj_df["label"] == 0]
            work = subj_df[subj_df["label"] == 1]

            rest_theta = rest[avail_theta].mean(axis=1).mean() if len(rest) else np.nan
            work_theta = work[avail_theta].mean(axis=1).mean() if len(work) else np.nan

            rest_ta = (rest[avail_theta].mean(axis=1) / (rest[avail_alpha].mean(axis=1) + EPS)).mean() if len(rest) else np.nan
            work_ta = (work[avail_theta].mean(axis=1) / (work[avail_alpha].mean(axis=1) + EPS)).mean() if len(work) else np.nan

            # Proxy "performance" = mean prediction score for workload trials
            # (higher score = better model confidence = proxy for task engagement)
            if len(work) > 0:
                perf_proxy = work["label"].mean()
            else:
                perf_proxy = np.nan

            subj_stats.append({
                "dataset": name,
                "subject_id": subj_id,
                "rest_frontal_theta": rest_theta,
                "work_frontal_theta": work_theta,
                "rest_theta_alpha": rest_ta,
                "work_theta_alpha": work_ta,
                "perf_proxy": perf_proxy,
                "n_rest": len(rest),
                "n_work": len(work),
            })

        subj_df = pd.DataFrame(subj_stats)

        # Correlation: frontal theta (workload) vs performance
        valid = subj_df.dropna(subset=["work_frontal_theta", "perf_proxy"])
        if len(valid) >= 5:
            r, p = pearsonr(valid["work_frontal_theta"], valid["perf_proxy"])
            print(f"  {name}: frontal theta ~ perf  r={r:.4f}, p={p:.4f} (n={len(valid)})")
            results.append({
                "dataset": name,
                "analysis": "frontal_theta_vs_perf",
                "r": r,
                "p": p,
                "n": len(valid),
            })
        else:
            print(f"  {name}: insufficient data for correlation")

        # Correlation: theta/alpha ratio (workload) vs performance
        valid2 = subj_df.dropna(subset=["work_theta_alpha", "perf_proxy"])
        if len(valid2) >= 5:
            r, p = pearsonr(valid2["work_theta_alpha"], valid2["perf_proxy"])
            print(f"  {name}: theta/alpha ~ perf     r={r:.4f}, p={p:.4f} (n={len(valid2)})")
            results.append({
                "dataset": name,
                "analysis": "theta_alpha_vs_perf",
                "r": r,
                "p": p,
                "n": len(valid2),
            })

        # Save per-subject stats
        subj_out = output_dir / f"biological_{name.lower()}.csv"
        subj_df.to_csv(subj_out, index=False)
        print(f"  Saved {subj_out.name}")

    res_df = pd.DataFrame(results) if results else pd.DataFrame()
    return res_df


# ------- Cross-dataset transfer -------


def run_cross_dataset_transfer(
    harmonized: Dict[str, pd.DataFrame],
    feature_cols: List[str],
    output_dir: Path,
    random_state: int = 42,
) -> pd.DataFrame:
    """Train on one dataset, test on another."""
    print(f"\n{'=' * 60}")
    print("  Cross-Dataset Transfer Analysis")
    print(f"{'=' * 60}")

    pairs = [
        ("MAT", "STEW"),
        ("STEW", "MAT"),
        ("MAT", "DS007262"),
        ("DS007262", "MAT"),
        ("STEW", "DS007262"),
        ("DS007262", "STEW"),
        ("MAT+STEW", "DS007262"),
    ]

    # Filter to valid pairs
    available = set(harmonized.keys())
    valid_pairs = []
    for train, test in pairs:
        if train == "MAT+STEW":
            if "MAT" in available and "STEW" in available:
                valid_pairs.append((train, test))
        elif train in available and test in available:
            valid_pairs.append((train, test))

    results = []
    for train_name, test_name in valid_pairs:
        print(f"\n  {train_name} -> {test_name}")

        if train_name == "MAT+STEW":
            train_df = pd.concat(
                [harmonized["MAT"], harmonized["STEW"]], ignore_index=True
            )
            train_df["subject_id"] = train_df["dataset"] + "_" + train_df["subject_id"]
        else:
            train_df = harmonized[train_name].copy()

        test_df = harmonized[test_name].copy()

        X_train = train_df[feature_cols].copy()
        y_train = train_df["label"].to_numpy()
        X_test = test_df[feature_cols].copy()
        y_test = test_df["label"].to_numpy()

        models = make_models(random_state=random_state)
        for model_name, model in models.items():
            pipe = make_pipeline(clone(model))
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_score = positive_scores(pipe, X_test)
            metrics = compute_metrics(y_test, y_pred, y_score)
            metrics["train_on"] = train_name
            metrics["test_on"] = test_name
            metrics["model"] = model_name
            metrics["n_train"] = len(y_train)
            metrics["n_test"] = len(y_test)
            results.append(metrics)
            print(
                f"    {model_name:25s}  Acc={metrics['accuracy']:.3f}  "
                f"AUC={metrics['roc_auc']:.3f}  F1={metrics['f1']:.3f}"
            )

    res_df = pd.DataFrame(results) if results else pd.DataFrame()
    return res_df


# ------- Figure generation -------


def _colorblind_palette(n_colors: int = 6) -> List[Tuple[float, float, float]]:
    """Colorblind-safe palette (Wong, 2011)."""
    base = [
        (0.0, 0.45, 0.70),  # blue
        (0.90, 0.60, 0.0),  # orange
        (0.35, 0.70, 0.40),  # green
        (0.80, 0.40, 0.0),  # vermillion
        (0.0, 0.60, 0.50),  # teal
        (0.95, 0.90, 0.25),  # yellow
    ]
    if n_colors <= len(base):
        return base[:n_colors]
    return base * (n_colors // len(base) + 1)[:n_colors]


def generate_figures(
    harmonized: Dict[str, pd.DataFrame],
    all_metrics: pd.DataFrame,
    all_preds: Dict[str, pd.DataFrame],
    transfer_metrics: pd.DataFrame,
    biological_results: pd.DataFrame,
    common_channels: List[str],
    output_dir: Path,
) -> None:
    """Generate all figures for the multi-dataset analysis."""
    if not MPL_AVAIL:
        print("  matplotlib not available, skipping figures")
        return

    print(f"\n{'=' * 60}")
    print("  Generating figures")
    print(f"{'=' * 60}")

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    dpi = 300
    palette = _colorblind_palette()

    if SNS_AVAIL:
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.1)

    datasets_list = list(harmonized.keys())
    colors = {ds: palette[i % len(palette)] for i, ds in enumerate(datasets_list)}

    # ------- Figure 1: Per-subject frontal theta boxplot -------
    print("  Figure 1: Frontal theta boxplot ...")
    fig, axes = plt.subplots(1, max(len(datasets_list), 1), figsize=(5 * max(len(datasets_list), 1), 4))
    if len(datasets_list) == 1:
        axes = [axes]
    for ax, name in zip(axes, datasets_list):
        df = harmonized[name]
        frontal = [ch for ch in ["F3", "F4"] if ch in common_channels]
        theta_col = f"band_abs_{frontal[0]}_theta" if frontal else None
        if theta_col and theta_col in df.columns:
            df["condition_label"] = df["label"].map({0: "Rest", 1: "Workload"})
            if SNS_AVAIL:
                sns.boxplot(data=df, x="condition_label", y=theta_col, ax=ax, palette=[colors.get(name, (0.5, 0.5, 0.5))])
            else:
                rest_vals = df[df["label"] == 0][theta_col].dropna()
                work_vals = df[df["label"] == 1][theta_col].dropna()
                ax.boxplot([rest_vals, work_vals], labels=["Rest", "Workload"])
                ax.set_facecolor("white")
            ax.set_title(f"{name} (n={df['subject_id'].nunique()})")
            ax.set_xlabel("")
            ax.set_ylabel("Frontal Theta Power")
    fig.suptitle("Frontal Theta Power: Rest vs Workload", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_frontal_theta_boxplot.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # ------- Figure 2: Combined ROC curves -------
    print("  Figure 2: Combined ROC curves ...")
    fig, ax = plt.subplots(figsize=(6, 5))
    for idx, (name, preds) in enumerate(all_preds.items()):
        if preds.empty:
            continue
        for model_name, grp in preds.groupby("model"):
            y_true = grp["true_label"].to_numpy()
            y_score = grp["score_workload"].to_numpy()
            if len(np.unique(y_true)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)
            label = f"{name} ({model_name[:8]}), AUC={auc:.2f}"
            ax.plot(fpr, tpr, label=label, color=palette[idx % len(palette)], alpha=0.7)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves Across Datasets")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_combined_roc.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # ------- Figure 3: Frontal theta vs accuracy scatter -------
    print("  Figure 3: Frontal theta vs performance correlation ...")
    fig, axes = plt.subplots(1, max(len(datasets_list), 1), figsize=(5 * max(len(datasets_list), 1), 4))
    if len(datasets_list) == 1:
        axes = [axes]
    for ax, name in zip(axes, datasets_list):
        bio_csv = output_dir / f"biological_{name.lower()}.csv"
        if bio_csv.exists():
            subj_df = pd.read_csv(bio_csv)
            valid = subj_df.dropna(subset=["work_frontal_theta", "perf_proxy"])
            if len(valid) >= 5:
                ax.scatter(valid["work_frontal_theta"], valid["perf_proxy"], alpha=0.6, color=colors.get(name, "steelblue"))
                r, p = pearsonr(valid["work_frontal_theta"], valid["perf_proxy"])
                ax.set_title(f"{name}: r={r:.3f}, p={p:.3f}")
                ax.set_xlabel("Frontal Theta (workload)")
                ax.set_ylabel("Performance Proxy")
    fig.suptitle("Frontal Theta vs Task Performance", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_theta_vs_performance.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # ------- Figure 4: Cross-dataset transfer heatmap -------
    print("  Figure 4: Cross-dataset transfer heatmap ...")
    if not transfer_metrics.empty:
        pivot = transfer_metrics.pivot_table(
            index="train_on", columns="test_on", values="roc_auc", aggfunc="mean"
        )
        fig, ax = plt.subplots(figsize=(6, 5))
        if SNS_AVAIL:
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax,
                        vmin=0.4, vmax=1.0, cbar_kws={"label": "ROC-AUC"})
        else:
            im = ax.imshow(pivot.to_numpy(), cmap="viridis", vmin=0.4, vmax=1.0)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_yticks(range(len(pivot.index)))
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
            ax.set_yticklabels(pivot.index)
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.iloc[i, j]
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="white" if val < 0.7 else "black")
            plt.colorbar(im, ax=ax, label="ROC-AUC")
        ax.set_title("Cross-Dataset Transfer AUC")
        fig.tight_layout()
        fig.savefig(fig_dir / "fig4_transfer_heatmap.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    # ------- Figure 5: Subject-level AUC distribution -------
    print("  Figure 5: Subject-level AUC distribution ...")
    fig, axes = plt.subplots(1, max(len(all_preds), 1), figsize=(5 * max(len(all_preds), 1), 4))
    if len(all_preds) == 1:
        axes = [axes]
    for idx, (name, preds) in enumerate(all_preds.items()):
        ax = axes[idx] if len(all_preds) > 1 else axes
        if preds.empty:
            continue
        subj_aucs = []
        for subj_id, grp in preds.groupby("subject_id"):
            y_true = grp["true_label"].to_numpy()
            y_score = grp["score_workload"].to_numpy()
            if len(np.unique(y_true)) >= 2:
                try:
                    subj_aucs.append(roc_auc_score(y_true, y_score))
                except Exception:
                    pass
        if subj_aucs:
            ax.hist(subj_aucs, bins=10, alpha=0.7, color=palette[idx % len(palette)], edgecolor="white")
            ax.axvline(np.mean(subj_aucs), color="red", ls="--", label=f"Mean={np.mean(subj_aucs):.2f}")
            ax.set_title(f"{name} (n={len(subj_aucs)} subjects)")
            ax.set_xlabel("Subject ROC-AUC")
            ax.set_ylabel("Count")
            ax.legend(fontsize=8)
    fig.suptitle("Subject-Level AUC Distribution", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig5_subject_auc_distribution.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # ------- Figure 6: Per-dataset model comparison barplot -------
    print("  Figure 6: Model comparison barplot ...")
    if not all_metrics.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_df = all_metrics[all_metrics["dataset"] != "COMBINED"].copy()
        if not plot_df.empty:
            if SNS_AVAIL:
                sns.barplot(
                    data=plot_df, x="dataset", y="roc_auc", hue="model", ax=ax,
                    palette=palette[:3],
                )
            else:
                for i, (model, grp) in enumerate(plot_df.groupby("model")):
                    offset = (i - 1) * 0.25
                    for j, (ds, dgrp) in enumerate(grp.groupby("dataset")):
                        ax.bar(j + offset, dgrp["roc_auc"].mean(), width=0.25, label=model if j == 0 else "")
                ax.set_xticks(range(len(plot_df["dataset"].unique())))
                ax.set_xticklabels(plot_df["dataset"].unique())
            ax.set_title("Per-Dataset Model Comparison (ROC-AUC)")
            ax.set_ylabel("ROC-AUC")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(fig_dir / "fig6_model_comparison.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    print(f"  All figures saved to {fig_dir}")


# ------- Summary report -------


def write_summary_report(
    all_metrics: pd.DataFrame,
    transfer_metrics: pd.DataFrame,
    biological_results: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write a text summary report."""
    report_path = output_dir / "summary_report.txt"
    lines = [
        "=" * 70,
        "  Multi-Dataset EEG Workload Analysis - Summary Report",
        "=" * 70,
        "",
        f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "-" * 70,
        "  PER-DATASET LOSO RESULTS",
        "-" * 70,
    ]

    if not all_metrics.empty:
        for _, row in all_metrics.iterrows():
            lines.append(
                f"  {row['dataset']:12s} | {row['model']:25s} | "
                f"Acc={row['accuracy']:.3f} | AUC={row['roc_auc']:.3f} | "
                f"F1={row['f1']:.3f} | Sens={row['sensitivity']:.3f} | "
                f"Spec={row['specificity']:.3f}"
            )

    lines.append("")
    lines.append("-" * 70)
    lines.append("  CROSS-DATASET TRANSFER RESULTS")
    lines.append("-" * 70)

    if not transfer_metrics.empty:
        for _, row in transfer_metrics.iterrows():
            lines.append(
                f"  {row['train_on']:12s} -> {row['test_on']:12s} | "
                f"{row['model']:25s} | AUC={row['roc_auc']:.3f} | "
                f"Acc={row['accuracy']:.3f}"
            )

    lines.append("")
    lines.append("-" * 70)
    lines.append("  BIOLOGICAL FINDINGS")
    lines.append("-" * 70)

    if not biological_results.empty:
        for _, row in biological_results.iterrows():
            lines.append(
                f"  {row['dataset']:12s} | {row['analysis']:25s} | "
                f"r={row['r']:.4f}, p={row['p']:.4f} (n={row['n']})"
            )

    lines.append("")
    lines.append("=" * 70)
    lines.append("  Analysis complete.")
    lines.append("=" * 70)

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Summary report saved to {report_path}")


# ------- Main -------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-dataset EEG workload analysis for ISEF-level validation"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        choices=["all", "mat", "stew", "ds007262"],
        help="Which datasets to include (default: all)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Reduced features for speed (use PCA to 50 dims)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory for results",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--skip_figures",
        action="store_true",
        help="Skip figure generation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random_state = args.random_state
    np.random.seed(random_state)
    quick = args.quick

    print(f"\n{'#' * 70}")
    print("  Multi-Dataset EEG Workload Analysis Pipeline")
    print(f"{'#' * 70}")

    # ------- 1. Load datasets -------
    LINE = '-' * 70
    print(f"\n{LINE}")
    print("  Step 1: Loading datasets")
    print(f"{LINE}")

    datasets_raw: Dict[str, pd.DataFrame] = {}

    if args.datasets in ("all", "mat"):
        mat_df = load_mat_dataset()
        if mat_df is not None:
            datasets_raw["MAT"] = mat_df

    if args.datasets in ("all", "stew"):
        stew_df = load_stew_dataset(output_dir, quick=quick)
        if stew_df is not None:
            datasets_raw["STEW"] = stew_df

    if args.datasets in ("all", "ds007262"):
        ds007262_df = load_ds007262_dataset()
        if ds007262_df is not None:
            datasets_raw["DS007262"] = ds007262_df

    if not datasets_raw:
        print("\nERROR: No datasets could be loaded. Exiting.")
        sys.exit(1)

    print(f"\n  Loaded {len(datasets_raw)} dataset(s): {', '.join(datasets_raw.keys())}")

    # ------- 2. Normalize channel names in columns -------
    print(f"\n{LINE}")
    print("  Step 2: Normalizing channel names")
    print(f"{LINE}")

    # Detect STEW's actual channel names
    for name in list(datasets_raw.keys()):
        df = datasets_raw[name]
        # Check what the actual channel names are in the data
        sample_cols = [c for c in df.columns if c.startswith("band_abs_")]
        actual_chs = set()
        for c in sample_cols:
            part = c.replace("band_abs_", "")
            ch = part.rsplit("_", 1)[0] if "_" in part else part
            actual_chs.add(ch)

        # DS007262 uses FP1, FP2 - normalize those
        rename_map = {}
        for ch in actual_chs:
            normalized = normalize_channel(ch)
            if normalized != ch:
                for template in FEATURE_TYPES:
                    old_col = template.format(ch=ch)
                    new_col = template.format(ch=normalized)
                    if old_col in df.columns:
                        rename_map[old_col] = new_col
        if rename_map:
            datasets_raw[name] = df.rename(columns=rename_map)
            print(f"  {name}: normalized {len(rename_map)} column names")

    # ------- 3. Compute common features -------
    print(f"\n{LINE}")
    print("  Step 3: Computing common features across datasets")
    print(f"{LINE}")
    print(f"  Common channels: {COMMON_CHANNELS}")

    harmonized = harmonize_features(datasets_raw, COMMON_CHANNELS)

    # Determine common feature columns from the intersection
    all_common_feats = None
    for name, df in harmonized.items():
        feats = find_available_common_features(df, COMMON_CHANNELS)
        if all_common_feats is None:
            all_common_feats = set(feats)
        else:
            all_common_feats &= set(feats)

    if all_common_feats is None or not all_common_feats:
        print("\nERROR: No common features found across datasets.")
        print("Check channel names and feature extraction.")
        sys.exit(1)

    feature_cols = sorted(all_common_feats)
    print(f"  Common features: {len(feature_cols)}")

    # Quick mode: reduce features with PCA
    if quick and len(feature_cols) > 50:
        print("\n  --quick mode: reducing to 50 features via PCA ...")
        try:
            from sklearn.decomposition import PCA
            # Fit PCA on combined data
            parts = []
            for name, df in harmonized.items():
                X_part = df[feature_cols].fillna(0)
                parts.append(X_part)
            X_all = pd.concat(parts, ignore_index=True)
            pca = PCA(n_components=50, random_state=random_state)
            pca.fit(X_all)
            # Transform each dataset
            new_feature_cols = [f"pca_{i:03d}" for i in range(50)]
            for name in harmonized:
                X_orig = harmonized[name][feature_cols].fillna(0)
                X_pca = pca.transform(X_orig)
                pca_df = pd.DataFrame(X_pca, columns=new_feature_cols, index=harmonized[name].index)
                for c in feature_cols:
                    if c in harmonized[name].columns:
                        harmonized[name] = harmonized[name].drop(columns=[c])
                harmonized[name] = pd.concat([harmonized[name], pca_df], axis=1)
            feature_cols = new_feature_cols
            print(f"  Reduced to {len(feature_cols)} PCA features")
        except Exception as exc:
            print(f"  PCA reduction failed: {exc}, using original features")

    # ------- 4. Per-dataset LOSO -------
    print(f"\n{LINE}")
    print("  Step 4: Per-dataset LOSO classification")
    print(f"{LINE}")

    all_metrics_list = []
    all_preds_dict: Dict[str, pd.DataFrame] = {}

    for name in harmonized:
        metrics_df, preds_df = run_loso_classification(
            harmonized[name], name, feature_cols, output_dir, random_state=random_state
        )
        all_metrics_list.append(metrics_df)
        all_preds_dict[name] = preds_df

        metrics_df.to_csv(output_dir / f"metrics_{name.lower()}.csv", index=False)
        preds_df.to_csv(output_dir / f"predictions_{name.lower()}.csv", index=False)
        print(f"  Saved {name.lower()} results to {output_dir}")

    # ------- 5. Combined LOSO -------
    print(f"\n{LINE}")
    print("  Step 5: Combined LOSO (all datasets pooled)")
    print(f"{LINE}")

    if len(harmonized) >= 2:
        combined_metrics, combined_preds = run_combined_loso(
            harmonized, feature_cols, output_dir, random_state=random_state
        )
        all_metrics_list.append(combined_metrics)
        all_preds_dict["COMBINED"] = combined_preds
        combined_metrics.to_csv(output_dir / "metrics_combined.csv", index=False)
        combined_preds.to_csv(output_dir / "predictions_combined.csv", index=False)
    else:
        print("  Skipping (need at least 2 datasets)")

    # ------- 6. Biological finding analysis -------
    print(f"\n{LINE}")
    print("  Step 6: Biological finding analysis (frontal theta)")
    print(f"{LINE}")

    biological_results = run_biological_analysis(
        harmonized, COMMON_CHANNELS, output_dir
    )
    if not biological_results.empty:
        biological_results.to_csv(
            output_dir / "biological_findings.csv", index=False
        )

    # ------- 7. Cross-dataset transfer -------
    print(f"\n{LINE}")
    print("  Step 7: Cross-dataset transfer analysis")
    print(f"{LINE}")

    transfer_metrics = run_cross_dataset_transfer(
        harmonized, feature_cols, output_dir, random_state=random_state
    )
    if not transfer_metrics.empty:
        transfer_metrics.to_csv(
            output_dir / "cross_dataset_transfer.csv", index=False
        )

    # ------- 8. Compile all metrics -------
    all_metrics = pd.concat(all_metrics_list, ignore_index=True) if all_metrics_list else pd.DataFrame()
    if not all_metrics.empty:
        all_metrics.to_csv(output_dir / "all_metrics.csv", index=False)

    # ------- 9. Generate figures -------
    print(f"\n{LINE}")
    print("  Step 8: Generating figures")
    print(f"{LINE}")

    if not args.skip_figures:
        generate_figures(
            harmonized,
            all_metrics,
            all_preds_dict,
            transfer_metrics,
            biological_results,
            COMMON_CHANNELS,
            output_dir,
        )

    # ------- 10. Summary report -------
    print(f"\n{LINE}")
    print("  Step 9: Writing summary report")
    print(f"{LINE}")

    write_summary_report(
        all_metrics, transfer_metrics, biological_results, output_dir
    )

    # ------- Done -------
    print(f"\n{'=' * 70}")
    print(f"  Pipeline complete. Results in: {output_dir}")
    print(f"{'=' * 70}")
    print(
        f"\n  Datasets: {', '.join(harmonized.keys())} "
        f"({sum(df['subject_id'].nunique() for df in harmonized.values())} subjects total)"
    )
    print(f"  Common features: {len(feature_cols)}")
    print(f"  Models: logistic_regression, svm_rbf"
          + (" + random_forest" if any("random_forest" in str(m) for m in []) else ""))
    print()


if __name__ == "__main__":
    main()
