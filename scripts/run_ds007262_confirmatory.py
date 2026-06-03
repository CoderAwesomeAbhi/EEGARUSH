#!/usr/bin/env python3
"""Run the frozen DS007262 confirmatory workload-scaling test.

This script trains the locked MAT/STEW model and applies it to DS007262.
DS007262 is never used for model fitting, feature selection, calibration-mode
selection, or hyperparameter selection.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "ds007262_confirmatory"
RAW_CACHE = RESULTS_DIR / "raw_cache"
os.environ.setdefault("MPLCONFIGDIR", str(RESULTS_DIR / "mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from eeg_cogstates.theory_validation import (  # noqa: E402
    COMMON_8_CHANNELS,
    DatasetSpec,
    apply_baseline_calibration,
    build_calibration_split,
    expected_feature_names,
    load_feature_table,
)
from scripts.multi_dataset_pipeline import extract_window_features  # noqa: E402


RANDOM_STATE = 20260602
EPS = 1e-12
OPENNEURO_S3_PREFIX = "https://s3.amazonaws.com/openneuro.org/ds007262"
DS_ROOT = PROJECT_ROOT / "external_data" / "ds007262_git"
TRAIN_SPECS = [
    DatasetSpec(
        name="MAT",
        path=PROJECT_ROOT / "outputs_reproduced" / "features" / "eeg_features.csv",
        kind="csv",
        baseline_seconds=60.0,
        timing_status="timed_first_60s",
    ),
    DatasetSpec(
        name="STEW",
        path=PROJECT_ROOT / "results" / "multi_dataset" / "stew_features.parquet",
        kind="parquet",
        baseline_fraction=0.5,
        timing_status="untimed_first_0.50_rest_fraction",
    ),
]
NUMERIC_RANGE_RE = re.compile(r"^\d+(?:\.\d+)?-\d+(?:\.\d+)?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-missing", action="store_true", help="Fetch missing BrainVision raw files from OpenNeuro S3 into the runtime cache.")
    parser.add_argument("--recompute-features", action="store_true", help="Ignore cached DS007262 graded features and re-extract from raw EEG.")
    parser.add_argument("--max-subjects", type=int, default=None, help="Optional smoke-test limit; do not use for confirmatory reporting.")
    return parser.parse_args()


def setup_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_CACHE.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "mplconfig").mkdir(parents=True, exist_ok=True)


def is_pointer_stub(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return True
    try:
        head = path.read_text(errors="ignore")[:160]
    except UnicodeDecodeError:
        return False
    return head.startswith("version https://git-annex") or head.startswith("version https://git-lfs")


def cached_raw_path(source_path: Path) -> Path:
    rel = source_path.relative_to(DS_ROOT)
    return RAW_CACHE / rel


def fetch_openneuro_file(rel_path: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{OPENNEURO_S3_PREFIX}/{rel_path.as_posix()}"
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp.replace(dest)


def materialize_raw_files(subject_dir: Path, download_missing: bool) -> Dict[str, Path]:
    eeg_dir = subject_dir / "eeg"
    subject = subject_dir.name
    files = {
        "vhdr": eeg_dir / f"{subject}_task-arithmetic_eeg.vhdr",
        "eeg": eeg_dir / f"{subject}_task-arithmetic_eeg.eeg",
        "vmrk": eeg_dir / f"{subject}_task-arithmetic_eeg.vmrk",
    }
    out: Dict[str, Path] = {}
    for key, source in files.items():
        cache_path = cached_raw_path(source)
        if cache_path.exists() and not is_pointer_stub(cache_path):
            out[key] = cache_path
            continue
        if source.exists() and not is_pointer_stub(source):
            out[key] = source
            continue
        if not download_missing:
            raise FileNotFoundError(f"{source} is an annex/LFS pointer or is missing; rerun with --download-missing")
        fetch_openneuro_file(source.relative_to(DS_ROOT), cache_path)
        out[key] = cache_path
    return out


def train_frozen_model() -> Tuple[SimpleImputer, StandardScaler, LogisticRegression, List[str]]:
    feature_cols = expected_feature_names(COMMON_8_CHANNELS)
    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []

    for spec in TRAIN_SPECS:
        df = load_feature_table(spec)
        missing = [col for col in feature_cols if col not in df.columns]
        if missing:
            raise ValueError(f"{spec.name} is missing frozen feature columns: {missing[:10]}")
        calib_mask, eval_mask, _ = build_calibration_split(df, spec)
        x_parts.append(apply_baseline_calibration(df, eval_mask, calib_mask, feature_cols, "zscore"))
        y_parts.append(df.loc[eval_mask, "label"].to_numpy(dtype=int))

    x_train = np.vstack(x_parts)
    y_train = np.concatenate(y_parts)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = imputer.fit_transform(x_train)
    x_train = scaler.fit_transform(x_train)
    model = LogisticRegression(
        C=1.0,
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, y_train)
    return imputer, scaler, model, feature_cols


def range_bounds(value: str) -> Tuple[float, float]:
    low, high = value.split("-", 1)
    return float(low), float(high)


def read_events(subject_dir: Path) -> pd.DataFrame:
    subject = subject_dir.name
    events_path = subject_dir / "eeg" / f"{subject}_task-arithmetic_events.tsv"
    df = pd.read_csv(events_path, sep="\t")
    df["subject_id"] = subject
    return df


def find_baseline_interval(events: pd.DataFrame) -> Tuple[float, float]:
    trial_type = events["trial_type"].astype(str)
    starts = events.loc[trial_type.str.contains("started_tutorial", case=False, na=False), "onset"]
    numeric = events.loc[events["difficulty_range"].astype(str).str.match(NUMERIC_RANGE_RE), "onset"]
    if starts.empty or numeric.empty:
        raise ValueError("Cannot locate started_tutorial marker and first numeric difficulty trial")
    start = float(starts.iloc[0])
    first_trial = float(numeric.min())
    end = min(start + 60.0, first_trial)
    if end - start < 30.0:
        raise ValueError(f"Baseline interval too short: {end - start:.3f} seconds")
    return start, end


def task_events(events: pd.DataFrame) -> pd.DataFrame:
    difficulty = events["difficulty_range"].astype(str)
    mask = difficulty.str.match(NUMERIC_RANGE_RE)
    duration = pd.to_numeric(events["duration"], errors="coerce")
    tutorial = events.get("istutorial", pd.Series(False, index=events.index)).astype(str).str.lower().eq("true")
    dropped = events["trial_type"].astype(str).eq("dropped_samples")
    keep = mask & duration.gt(0) & ~tutorial & ~dropped
    out = events.loc[keep].copy()
    out["duration"] = duration.loc[keep].astype(float)
    out["difficulty_range"] = out["difficulty_range"].astype(str)
    bounds = out["difficulty_range"].map(range_bounds)
    out["difficulty_low"] = [low for low, _ in bounds]
    out["difficulty_high"] = [high for _, high in bounds]
    out["difficulty_midpoint"] = (out["difficulty_low"] + out["difficulty_high"]) / 2.0
    return out


def pick_common_channels(raw: mne.io.BaseRaw) -> Tuple[List[str], List[str]]:
    lookup = {name.lower(): name for name in raw.ch_names}
    actual = []
    for channel in COMMON_8_CHANNELS:
        if channel.lower() not in lookup:
            raise ValueError(f"Raw recording is missing frozen channel {channel}")
        actual.append(lookup[channel.lower()])
    return actual, list(COMMON_8_CHANNELS)


def extract_epoch_features(raw: mne.io.BaseRaw, tmin: float, duration: float, actual_channels: List[str], canonical_channels: List[str]) -> Dict[str, float]:
    tmax = min(tmin + duration, float(raw.times[-1]))
    if tmax - tmin <= 0:
        raise ValueError("Epoch lies outside raw recording")
    segment = raw.copy().pick(actual_channels).crop(tmin=tmin, tmax=tmax, include_tmax=False)
    data = segment.get_data()
    return extract_window_features(data, float(raw.info["sfreq"]), canonical_channels)


def extract_subject_features(subject_dir: Path, download_missing: bool) -> pd.DataFrame:
    subject = subject_dir.name
    raw_paths = materialize_raw_files(subject_dir, download_missing)
    events = read_events(subject_dir)
    baseline_start, baseline_end = find_baseline_interval(events)
    trials = task_events(events)

    raw = mne.io.read_raw_brainvision(raw_paths["vhdr"], preload=True, verbose="ERROR")
    actual_channels, canonical_channels = pick_common_channels(raw)

    rows: List[Dict[str, object]] = []
    window_start = baseline_start
    baseline_index = 0
    while window_start + 6.0 <= baseline_end + EPS:
        feats = extract_epoch_features(raw, window_start, 6.0, actual_channels, canonical_channels)
        feats.update(
            {
                "subject_id": subject,
                "epoch_role": "baseline",
                "onset": window_start,
                "duration": 6.0,
                "baseline_window_index": baseline_index,
                "difficulty_range": "",
                "difficulty_low": np.nan,
                "difficulty_high": np.nan,
                "difficulty_midpoint": np.nan,
            }
        )
        rows.append(feats)
        baseline_index += 1
        window_start += 6.0

    for trial_index, trial in enumerate(trials.itertuples(index=False)):
        feats = extract_epoch_features(raw, float(trial.onset), float(trial.duration), actual_channels, canonical_channels)
        feats.update(
            {
                "subject_id": subject,
                "epoch_role": "task",
                "onset": float(trial.onset),
                "duration": float(trial.duration),
                "trial_index": trial_index,
                "difficulty_range": str(trial.difficulty_range),
                "difficulty_low": float(trial.difficulty_low),
                "difficulty_high": float(trial.difficulty_high),
                "difficulty_midpoint": float(trial.difficulty_midpoint),
                "response_accuracy": getattr(trial, "response_accuracy", np.nan),
                "outcome": getattr(trial, "outcome", ""),
            }
        )
        rows.append(feats)

    return pd.DataFrame(rows)


def extract_ds_features(download_missing: bool, recompute: bool, max_subjects: int | None) -> pd.DataFrame:
    cache_path = RESULTS_DIR / "ds007262_graded_features.csv"
    if cache_path.exists() and not recompute:
        return pd.read_csv(cache_path)

    subject_dirs = sorted(path for path in DS_ROOT.glob("sub-*") if path.is_dir())
    if max_subjects is not None:
        subject_dirs = subject_dirs[:max_subjects]

    frames = []
    failures = []
    for subject_dir in subject_dirs:
        try:
            print(f"[ds007262] extracting {subject_dir.name}")
            frames.append(extract_subject_features(subject_dir, download_missing))
        except Exception as exc:  # Keep subject failures auditable.
            print(f"[ds007262] failed {subject_dir.name}: {exc}")
            failures.append({"subject_id": subject_dir.name, "failure": str(exc)})

    pd.DataFrame(failures).to_csv(RESULTS_DIR / "subject_extraction_failures.csv", index=False)
    if not frames:
        raise RuntimeError("No DS007262 subjects could be extracted")

    df = pd.concat(frames, ignore_index=True)
    task = df["epoch_role"].eq("task")
    ranges = sorted(df.loc[task, "difficulty_range"].dropna().unique(), key=lambda value: range_bounds(str(value))[0])
    level_map = {value: index + 1 for index, value in enumerate(ranges)}
    df["workload_level"] = df["difficulty_range"].map(level_map)
    pd.DataFrame({"difficulty_range": list(level_map), "workload_level": list(level_map.values())}).to_csv(
        RESULTS_DIR / "difficulty_level_mapping.csv", index=False
    )
    df.to_csv(cache_path, index=False)
    return df


def subject_zscore_ds(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, pd.DataFrame]:
    baseline = df["epoch_role"].eq("baseline").to_numpy()
    task = df["epoch_role"].eq("task").to_numpy()
    subjects = df["subject_id"].astype(str).to_numpy()
    out = np.empty((int(task.sum()), len(feature_cols)), dtype=float)
    task_rows = np.where(task)[0]

    for local_i, row_i in enumerate(task_rows):
        subject = subjects[row_i]
        base_idx = np.where((subjects == subject) & baseline)[0]
        if base_idx.size < 2:
            raise ValueError(f"Subject {subject} has fewer than two baseline calibration windows")
        base = df.iloc[base_idx][feature_cols].to_numpy(dtype=float)
        mean = np.nanmean(base, axis=0)
        std = np.nanstd(base, axis=0, ddof=0)
        std[~np.isfinite(std) | (std < EPS)] = 1.0
        x = df.iloc[[row_i]][feature_cols].to_numpy(dtype=float)[0]
        out[local_i] = (x - mean) / std

    return out, df.iloc[task_rows].copy()


def one_sided_spearman(x: Iterable[float], y: Iterable[float]) -> Tuple[float, float, float]:
    rho, p_two_sided = spearmanr(list(x), list(y))
    if not np.isfinite(rho):
        return float("nan"), float("nan"), float("nan")
    p_one_sided_positive = p_two_sided / 2.0 if rho > 0 else 1.0 - (p_two_sided / 2.0)
    return float(rho), float(p_two_sided), float(p_one_sided_positive)


def score_and_summarize(df: pd.DataFrame, imputer: SimpleImputer, scaler: StandardScaler, model: LogisticRegression, feature_cols: List[str]) -> pd.DataFrame:
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"DS007262 extracted features are missing frozen columns: {missing[:10]}")

    x_ds, scored = subject_zscore_ds(df, feature_cols)
    x_ds = scaler.transform(imputer.transform(x_ds))
    scored["frozen_score"] = model.decision_function(x_ds)
    scored["frozen_probability"] = model.predict_proba(x_ds)[:, 1]
    scored.to_csv(RESULTS_DIR / "ds007262_frozen_predictions.csv", index=False)
    return scored


def write_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    scored = scored.copy()
    scored["workload_level"] = pd.to_numeric(scored["workload_level"], errors="coerce")
    scored = scored.dropna(subset=["workload_level", "frozen_score"])

    level_summary = (
        scored.groupby("workload_level", as_index=False)
        .agg(
            difficulty_range=("difficulty_range", "first"),
            mean_score=("frozen_score", "mean"),
            median_score=("frozen_score", "median"),
            sd_score=("frozen_score", "std"),
            n_trials=("frozen_score", "size"),
            n_subjects=("subject_id", "nunique"),
        )
        .sort_values("workload_level")
    )
    level_summary.to_csv(RESULTS_DIR / "ds007262_level_summary.csv", index=False)

    subject_level = (
        scored.groupby(["subject_id", "workload_level"], as_index=False)
        .agg(mean_score=("frozen_score", "mean"), n_trials=("frozen_score", "size"))
        .sort_values(["subject_id", "workload_level"])
    )
    subject_level.to_csv(RESULTS_DIR / "ds007262_subject_level_scores.csv", index=False)

    rho_primary, p_primary_two, p_primary_one = one_sided_spearman(subject_level["workload_level"], subject_level["mean_score"])
    rho_trial, p_trial_two, p_trial_one = one_sided_spearman(scored["workload_level"], scored["frozen_score"])

    per_subject_rows = []
    for subject, group in subject_level.groupby("subject_id"):
        if group["workload_level"].nunique() >= 3:
            rho, p_two, p_one = one_sided_spearman(group["workload_level"], group["mean_score"])
            per_subject_rows.append({"subject_id": subject, "spearman_rho": rho, "p_two_sided": p_two, "p_one_sided_positive": p_one, "n_levels": group["workload_level"].nunique()})
    per_subject = pd.DataFrame(per_subject_rows)
    per_subject.to_csv(RESULTS_DIR / "ds007262_per_subject_spearman.csv", index=False)

    n_levels = int(scored["workload_level"].nunique())
    n_subjects = int(scored["subject_id"].nunique())
    success = (
        n_subjects >= 12
        and n_levels >= 6
        and np.isfinite(rho_primary)
        and rho_primary >= 0.20
        and p_primary_one < 0.05
    )
    status = "success" if success else "failure"
    metrics = pd.DataFrame(
        [
            {
                "status": status,
                "primary_statistic": "subject_level_spearman",
                "primary_spearman_rho": rho_primary,
                "primary_p_two_sided": p_primary_two,
                "primary_p_one_sided_positive": p_primary_one,
                "trial_level_spearman_rho": rho_trial,
                "trial_level_p_two_sided": p_trial_two,
                "trial_level_p_one_sided_positive": p_trial_one,
                "available_level_count": n_levels,
                "available_levels": ",".join(map(str, sorted(scored["workload_level"].dropna().astype(int).unique()))),
                "n_subjects": n_subjects,
                "n_trials": int(len(scored)),
                "n_subject_level_rows": int(len(subject_level)),
                "per_subject_rho_mean": float(per_subject["spearman_rho"].mean()) if not per_subject.empty else np.nan,
                "per_subject_rho_median": float(per_subject["spearman_rho"].median()) if not per_subject.empty else np.nan,
                "per_subject_positive_count": int(per_subject["spearman_rho"].gt(0).sum()) if not per_subject.empty else 0,
            }
        ]
    )
    metrics.to_csv(RESULTS_DIR / "ds007262_confirmatory_metrics.csv", index=False)
    write_markdown_summary(metrics, level_summary, per_subject)
    plot_level_curve(level_summary)
    return metrics


def plot_level_curve(level_summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    yerr = level_summary["sd_score"].fillna(0.0)
    ax.errorbar(level_summary["workload_level"], level_summary["mean_score"], yerr=yerr, marker="o", linewidth=2)
    ax.set_xlabel("DS007262 ordinal workload level")
    ax.set_ylabel("Frozen logistic decision score")
    ax.set_title("Frozen DS007262 workload scaling")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figures" / "ds007262_level_curve.png", dpi=180)
    plt.close(fig)


def write_markdown_summary(metrics: pd.DataFrame, level_summary: pd.DataFrame, per_subject: pd.DataFrame) -> None:
    row = metrics.iloc[0]
    lines = [
        "# DS007262 Confirmatory Results",
        "",
        "The frozen MAT/STEW model was applied to DS007262 without retraining on DS007262.",
        "",
        "## Primary Metric",
        "",
        f"- Status: `{row['status']}`",
        f"- Primary subject-level Spearman rho: `{row['primary_spearman_rho']:.6g}`",
        f"- Primary one-sided p-value for positive monotonicity: `{row['primary_p_one_sided_positive']:.6g}`",
        f"- Available difficulty levels: `{row['available_levels']}` (`{row['available_level_count']}` levels)",
        f"- Subjects: `{row['n_subjects']}`",
        f"- Scored task trials: `{row['n_trials']}`",
        "",
        "## Level Means",
        "",
        markdown_table(level_summary),
        "",
        "## Per-Subject Direction Summary",
        "",
        f"- Mean per-subject rho: `{row['per_subject_rho_mean']:.6g}`",
        f"- Median per-subject rho: `{row['per_subject_rho_median']:.6g}`",
        f"- Subjects with positive rho: `{row['per_subject_positive_count']}`",
    ]
    if per_subject.empty:
        lines.append("- Per-subject Spearman correlations were not computable.")
    (RESULTS_DIR / "DS007262_CONFIRMATORY_RESULTS.md").write_text("\n".join(lines) + "\n")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"

    def fmt(value: object) -> str:
        if pd.isna(value):
            return "nan"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        return str(value)

    cols = list(df.columns)
    rows = [[fmt(row[col]) for col in cols] for _, row in df.iterrows()]
    widths = [max(len(str(col)), *(len(row[i]) for row in rows)) for i, col in enumerate(cols)]

    def render(values: List[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(values)) + " |"

    lines = [render([str(col) for col in cols])]
    lines.append("| " + " | ".join("-" * width for width in widths) + " |")
    lines.extend(render(row) for row in rows)
    return "\n".join(lines)


def write_not_run_status(reason: str) -> None:
    metrics = pd.DataFrame(
        [
            {
                "status": "not_run_raw_data_unavailable",
                "reason": reason,
                "primary_spearman_rho": np.nan,
                "primary_p_one_sided_positive": np.nan,
                "available_level_count": np.nan,
                "n_subjects": 0,
                "n_trials": 0,
            }
        ]
    )
    metrics.to_csv(RESULTS_DIR / "ds007262_confirmatory_metrics.csv", index=False)
    (RESULTS_DIR / "DS007262_CONFIRMATORY_RESULTS.md").write_text(
        "# DS007262 Confirmatory Results\n\n"
        "Status: `not_run_raw_data_unavailable`\n\n"
        f"Reason: {reason}\n"
    )


def main() -> int:
    args = parse_args()
    setup_dirs()
    print("[frozen] training locked MAT/STEW logistic_l2 zscore model")
    imputer, scaler, model, feature_cols = train_frozen_model()
    try:
        features = extract_ds_features(args.download_missing, args.recompute_features, args.max_subjects)
        scored = score_and_summarize(features, imputer, scaler, model, feature_cols)
        metrics = write_metrics(scored)
    except Exception as exc:
        write_not_run_status(str(exc))
        print(f"[ds007262] NOT RUN: {exc}")
        return 2

    print(metrics.to_string(index=False))
    print(f"[ds007262] wrote outputs to {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
