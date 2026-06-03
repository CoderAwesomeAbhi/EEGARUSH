"""Leakage-aware theoretical validation experiments for EEG workload features.

The functions in this module deliberately separate three concepts:

1. Baseline calibration rows: known resting/baseline windows used only to
   estimate subject-specific centering/scaling constants.
2. Evaluation rows: rows scored by the classifier.
3. Model-training rows: evaluation rows from training subjects only.

Outer validation is leave-one-subject-out. Hyperparameters for the real
baseline-comparison experiments are selected by group-preserving inner CV on
training subjects only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import os

import numpy as np
import pandas as pd

try:  # Imported to pin the runtime stack requested by the validation prompt.
    import mne  # noqa: F401
except Exception:  # pragma: no cover - environment reporting handles this.
    mne = None  # type: ignore

from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


EPS = 1e-12
RANDOM_STATE = 20260602

COMMON_8_CHANNELS = ["F3", "F4", "F7", "F8", "O1", "O2", "T3", "T4"]

FEATURE_TEMPLATES = [
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

CALIBRATION_MODES = ["absolute", "mean_subtraction", "zscore"]
MODEL_NAMES = ["logistic_l2", "linear_svm"]
DEFAULT_C_GRID = [1.0]


@dataclass(frozen=True)
class DatasetSpec:
    """Configuration for one dataset's calibration/evaluation split."""

    name: str
    path: Path
    kind: str
    baseline_seconds: Optional[float] = None
    baseline_fraction: float = 0.5
    timing_status: str = "unknown"


@dataclass
class FoldFit:
    """Fitted model bundle for one outer fold."""

    model: object
    imputer: SimpleImputer
    scaler: StandardScaler
    best_c: float
    inner_auc: float
    coef: np.ndarray


def expected_feature_names(channels: Sequence[str] = COMMON_8_CHANNELS) -> List[str]:
    names: List[str] = []
    for ch in channels:
        for template in FEATURE_TEMPLATES:
            names.append(template.format(ch=ch))
    return names


def load_feature_table(spec: DatasetSpec) -> pd.DataFrame:
    """Load one feature table and normalize required metadata columns."""
    if spec.kind == "csv":
        df = pd.read_csv(spec.path)
    elif spec.kind == "parquet":
        df = pd.read_parquet(spec.path)
    else:
        raise ValueError(f"Unsupported dataset kind: {spec.kind}")

    required = {"subject_id", "label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{spec.name} is missing required columns: {missing}")

    df = df.copy()
    df["subject_id"] = df["subject_id"].astype(str)
    df["label"] = df["label"].astype(int)
    if "condition" not in df.columns:
        df["condition"] = np.where(df["label"].eq(0), "rest", "workload")
    if "dataset" not in df.columns:
        df["dataset"] = spec.name
    df["_row_order"] = np.arange(len(df), dtype=int)
    return df


def select_common_8_features(df: pd.DataFrame) -> List[str]:
    """Return the exact 8-channel per-channel feature intersection."""
    expected = expected_feature_names()
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} common 8-channel features. "
            f"First missing columns: {missing[:10]}"
        )
    return expected


def build_calibration_split(
    df: pd.DataFrame,
    spec: DatasetSpec,
    baseline_seconds: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Build non-overlapping calibration/evaluation masks.

    Timed datasets use early rest windows as calibration and later rest windows
    as evaluation negatives. Untimed datasets use a deterministic within-subject
    rest-window split by row order; this is not a duration curve substitute.
    """
    use_seconds = spec.baseline_seconds if baseline_seconds is None else baseline_seconds
    labels = df["label"].to_numpy()
    subjects = df["subject_id"].astype(str).to_numpy()
    calib = np.zeros(len(df), dtype=bool)
    eval_mask = labels == 1

    has_timing = {"start_sec", "end_sec"}.issubset(df.columns)
    if has_timing and use_seconds is not None:
        start = pd.to_numeric(df["start_sec"], errors="coerce").to_numpy(dtype=float)
        rest = labels == 0
        calib = rest & np.isfinite(start) & (start < float(use_seconds))
        eval_mask = (labels == 1) | (rest & np.isfinite(start) & (start >= float(use_seconds)))
        return calib, eval_mask, f"timed_first_{use_seconds:g}s"

    # Untimed fallback: split rest rows by existing row order within subject.
    for subject in np.unique(subjects):
        idx = np.where((subjects == subject) & (labels == 0))[0]
        if idx.size < 2:
            continue
        idx = idx[np.argsort(df.iloc[idx]["_row_order"].to_numpy())]
        n_calib = int(math.floor(idx.size * spec.baseline_fraction))
        n_calib = max(1, min(n_calib, idx.size - 1))
        calib[idx[:n_calib]] = True
        eval_mask[idx[n_calib:]] = True
    return calib, eval_mask, f"untimed_first_{spec.baseline_fraction:.2f}_rest_fraction"


def _subject_stats(
    x: pd.DataFrame,
    subjects: np.ndarray,
    calib_mask: np.ndarray,
    feature_cols: Sequence[str],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for subject in np.unique(subjects):
        idx = np.where((subjects == subject) & calib_mask)[0]
        if idx.size == 0:
            continue
        vals = x.iloc[idx][feature_cols].to_numpy(dtype=float)
        mean = np.nanmean(vals, axis=0)
        std = np.nanstd(vals, axis=0, ddof=0)
        std[~np.isfinite(std) | (std < EPS)] = 1.0
        mean[~np.isfinite(mean)] = 0.0
        stats[str(subject)] = (mean, std)
    return stats


def apply_baseline_calibration(
    df: pd.DataFrame,
    row_mask: np.ndarray,
    calib_mask: np.ndarray,
    feature_cols: Sequence[str],
    mode: str,
) -> np.ndarray:
    """Apply absolute, mean-subtraction, or rest z-scoring per subject."""
    if mode not in CALIBRATION_MODES:
        raise ValueError(f"Unknown calibration mode: {mode}")

    rows = np.where(row_mask)[0]
    x = df.iloc[rows][feature_cols].to_numpy(dtype=float)
    if mode == "absolute":
        return x

    subjects_all = df["subject_id"].astype(str).to_numpy()
    stats = _subject_stats(df, subjects_all, calib_mask, feature_cols)
    out = np.empty_like(x, dtype=float)
    for local_i, global_i in enumerate(rows):
        subject = str(subjects_all[global_i])
        if subject not in stats:
            raise ValueError(f"No calibration baseline rows for subject {subject}")
        mean, std = stats[subject]
        centered = x[local_i] - mean
        out[local_i] = centered if mode == "mean_subtraction" else centered / std
    return out


def make_model(model_name: str, c_value: float, random_state: int = RANDOM_STATE):
    if model_name == "logistic_l2":
        return LogisticRegression(
            C=float(c_value),
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            random_state=random_state,
        )
    if model_name == "linear_svm":
        return LinearSVC(
            C=float(c_value),
            class_weight="balanced",
            max_iter=20000,
            dual="auto",
            random_state=random_state,
        )
    raise ValueError(f"Unknown model: {model_name}")


def _score_model(model: object, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x), dtype=float)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    raise TypeError(f"Model {type(model)} does not expose decision_function or predict_proba")


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def _fit_once(
    df: pd.DataFrame,
    train_subjects: Sequence[str],
    test_subjects: Sequence[str],
    feature_cols: Sequence[str],
    calib_mask: np.ndarray,
    eval_mask: np.ndarray,
    calibration_mode: str,
    model_name: str,
    c_value: float,
    y_override: Optional[np.ndarray] = None,
) -> Tuple[Optional[FoldFit], np.ndarray, np.ndarray, np.ndarray]:
    subjects = df["subject_id"].astype(str).to_numpy()
    labels = df["label"].to_numpy(dtype=int) if y_override is None else np.asarray(y_override, dtype=int)
    train_mask = np.isin(subjects, list(train_subjects)) & eval_mask
    test_mask = np.isin(subjects, list(test_subjects)) & eval_mask

    y_train = labels[train_mask]
    y_test = labels[test_mask]
    if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
        return None, np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=int)

    x_train = apply_baseline_calibration(df, train_mask, calib_mask, feature_cols, calibration_mode)
    x_test = apply_baseline_calibration(df, test_mask, calib_mask, feature_cols, calibration_mode)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = imputer.fit_transform(x_train)
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(imputer.transform(x_test))

    model = make_model(model_name, c_value)
    model.fit(x_train, y_train)
    scores = _score_model(model, x_test)
    coef = np.asarray(getattr(model, "coef_", np.zeros((1, len(feature_cols)))), dtype=float).reshape(-1)
    fit = FoldFit(model=model, imputer=imputer, scaler=scaler, best_c=float(c_value), inner_auc=float("nan"), coef=coef)
    row_ids = np.where(test_mask)[0]
    return fit, y_test, scores, row_ids


def choose_c_inner_cv(
    df: pd.DataFrame,
    train_subjects: Sequence[str],
    feature_cols: Sequence[str],
    calib_mask: np.ndarray,
    eval_mask: np.ndarray,
    calibration_mode: str,
    model_name: str,
    c_grid: Sequence[float],
    inner_splits: int = 5,
    y_override: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Select C using group-preserving inner CV on training subjects only."""
    if len(c_grid) == 1:
        return float(c_grid[0]), float("nan")

    unique_subjects = np.array(sorted(map(str, train_subjects)))
    if unique_subjects.size < 3:
        return float(c_grid[0]), float("nan")

    n_splits = min(inner_splits, unique_subjects.size)
    splitter = GroupKFold(n_splits=n_splits)
    best_c = float(c_grid[0])
    best_auc = -np.inf

    for c_value in c_grid:
        fold_aucs: List[float] = []
        for tr_idx, va_idx in splitter.split(unique_subjects, groups=unique_subjects):
            inner_train = unique_subjects[tr_idx]
            inner_val = unique_subjects[va_idx]
            _, y_val, scores, _ = _fit_once(
                df=df,
                train_subjects=inner_train,
                test_subjects=inner_val,
                feature_cols=feature_cols,
                calib_mask=calib_mask,
                eval_mask=eval_mask,
                calibration_mode=calibration_mode,
                model_name=model_name,
                c_value=float(c_value),
                y_override=y_override,
            )
            auc_value = _safe_auc(y_val, scores) if y_val.size else float("nan")
            if np.isfinite(auc_value):
                fold_aucs.append(float(auc_value))
        mean_auc = float(np.mean(fold_aucs)) if fold_aucs else float("nan")
        if np.isfinite(mean_auc) and (mean_auc > best_auc + EPS):
            best_auc = mean_auc
            best_c = float(c_value)
    return best_c, (best_auc if np.isfinite(best_auc) else float("nan"))


def nested_loso_predictions(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    calib_mask: np.ndarray,
    eval_mask: np.ndarray,
    calibration_mode: str,
    model_name: str,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    inner_splits: int = 5,
    y_override: Optional[np.ndarray] = None,
    fold_c_overrides: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run outer LOSO with inner grouped-CV hyperparameter selection."""
    subjects = df["subject_id"].astype(str).to_numpy()
    labels = df["label"].to_numpy(dtype=int) if y_override is None else np.asarray(y_override, dtype=int)
    unique_subjects = np.array(sorted(np.unique(subjects)))
    splitter = LeaveOneGroupOut()

    pred_rows: List[Dict[str, object]] = []
    fold_rows: List[Dict[str, object]] = []
    coef_rows: List[Dict[str, object]] = []

    dummy_x = np.zeros((len(unique_subjects), 1))
    for train_idx, test_idx in splitter.split(dummy_x, groups=unique_subjects):
        train_subjects = unique_subjects[train_idx]
        test_subject = str(unique_subjects[test_idx][0])

        if fold_c_overrides is None:
            best_c, inner_auc = choose_c_inner_cv(
                df=df,
                train_subjects=train_subjects,
                feature_cols=feature_cols,
                calib_mask=calib_mask,
                eval_mask=eval_mask,
                calibration_mode=calibration_mode,
                model_name=model_name,
                c_grid=c_grid,
                inner_splits=inner_splits,
                y_override=y_override,
            )
        else:
            best_c = float(fold_c_overrides[test_subject])
            inner_auc = float("nan")

        fit, y_test, scores, row_ids = _fit_once(
            df=df,
            train_subjects=train_subjects,
            test_subjects=[test_subject],
            feature_cols=feature_cols,
            calib_mask=calib_mask,
            eval_mask=eval_mask,
            calibration_mode=calibration_mode,
            model_name=model_name,
            c_value=best_c,
            y_override=y_override,
        )

        outer_auc = _safe_auc(y_test, scores) if y_test.size else float("nan")
        fold_rows.append(
            {
                "subject_id": test_subject,
                "model": model_name,
                "calibration": calibration_mode,
                "best_c": best_c,
                "inner_mean_auc": inner_auc,
                "outer_subject_auc": outer_auc,
                "n_test_rows": int(y_test.size),
                "n_test_positive": int(np.sum(y_test == 1)) if y_test.size else 0,
                "n_test_negative": int(np.sum(y_test == 0)) if y_test.size else 0,
            }
        )
        if fit is None or not y_test.size:
            continue

        for row_id, y_value, score in zip(row_ids, y_test, scores):
            pred_rows.append(
                {
                    "row_id": int(row_id),
                    "subject_id": test_subject,
                    "y_true": int(y_value),
                    "score": float(score),
                    "model": model_name,
                    "calibration": calibration_mode,
                    "best_c": best_c,
                }
            )
        for feature, coef in zip(feature_cols, fit.coef):
            coef_rows.append(
                {
                    "subject_id": test_subject,
                    "model": model_name,
                    "calibration": calibration_mode,
                    "feature": feature,
                    "coef": float(coef),
                    "best_c": best_c,
                }
            )

    return pd.DataFrame(pred_rows), pd.DataFrame(fold_rows), pd.DataFrame(coef_rows)


def summarize_predictions(pred: pd.DataFrame, dataset: str, split_description: str, n_features: int) -> Dict[str, object]:
    if pred.empty or "y_true" not in pred.columns or pred["y_true"].nunique() < 2:
        auc_value = float("nan")
    else:
        auc_value = float(roc_auc_score(pred["y_true"].to_numpy(dtype=int), pred["score"].to_numpy(dtype=float)))
    subject_aucs = []
    if not pred.empty and {"subject_id", "y_true", "score"}.issubset(pred.columns):
        for _, group in pred.groupby("subject_id"):
            if group["y_true"].nunique() == 2:
                subject_aucs.append(float(roc_auc_score(group["y_true"], group["score"])))
    return {
        "dataset": dataset,
        "model": str(pred["model"].iloc[0]) if not pred.empty and "model" in pred.columns else "",
        "calibration": str(pred["calibration"].iloc[0]) if not pred.empty and "calibration" in pred.columns else "",
        "split_description": split_description,
        "n_features": int(n_features),
        "n_predictions": int(len(pred)),
        "n_subjects": int(pred["subject_id"].nunique()) if not pred.empty and "subject_id" in pred.columns else 0,
        "window_auc": auc_value,
        "subject_auc_mean": float(np.mean(subject_aucs)) if subject_aucs else float("nan"),
        "subject_auc_sd": float(np.std(subject_aucs, ddof=1)) if len(subject_aucs) > 1 else float("nan"),
        "n_subject_auc": int(len(subject_aucs)),
    }


def permute_labels_within_subject(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    labels = df["label"].to_numpy(dtype=int).copy()
    subjects = df["subject_id"].astype(str).to_numpy()
    out = labels.copy()
    for subject in np.unique(subjects):
        idx = np.where(subjects == subject)[0]
        out[idx] = rng.permutation(labels[idx])
    return out


def final_direction_coefficients(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    calib_mask: np.ndarray,
    eval_mask: np.ndarray,
    calibration_mode: str,
    model_name: str,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    inner_splits: int = 5,
) -> Tuple[pd.DataFrame, float, float]:
    """Train a final model after grouped CV C selection and return coefficients."""
    all_subjects = np.array(sorted(df["subject_id"].astype(str).unique()))
    best_c, cv_auc = choose_c_inner_cv(
        df=df,
        train_subjects=all_subjects,
        feature_cols=feature_cols,
        calib_mask=calib_mask,
        eval_mask=eval_mask,
        calibration_mode=calibration_mode,
        model_name=model_name,
        c_grid=c_grid,
        inner_splits=inner_splits,
    )
    fit, _, _, _ = _fit_once(
        df=df,
        train_subjects=all_subjects,
        test_subjects=all_subjects,
        feature_cols=feature_cols,
        calib_mask=calib_mask,
        eval_mask=eval_mask,
        calibration_mode=calibration_mode,
        model_name=model_name,
        c_value=best_c,
    )
    if fit is None:
        raise RuntimeError(f"Could not fit final {model_name}/{calibration_mode} model")
    return (
        pd.DataFrame(
            {
                "feature": list(feature_cols),
                "coef": fit.coef.astype(float),
                "model": model_name,
                "calibration": calibration_mode,
                "best_c": best_c,
                "inner_cv_auc": cv_auc,
            }
        ),
        best_c,
        cv_auc,
    )


def effect_direction_correlation(mat_coef: pd.DataFrame, stew_coef: pd.DataFrame) -> Tuple[float, float, int]:
    merged = mat_coef[["feature", "coef"]].merge(
        stew_coef[["feature", "coef"]],
        on="feature",
        suffixes=("_mat", "_stew"),
    )
    if len(merged) < 3:
        return float("nan"), float("nan"), int(len(merged))
    r_value, p_value = pearsonr(merged["coef_mat"].to_numpy(), merged["coef_stew"].to_numpy())
    return float(r_value), float(p_value), int(len(merged))


def roc_points(pred: pd.DataFrame) -> pd.DataFrame:
    if pred.empty or pred["y_true"].nunique() < 2:
        return pd.DataFrame(columns=["fpr", "tpr", "threshold"])
    fpr, tpr, threshold = roc_curve(pred["y_true"], pred["score"])
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": threshold})
