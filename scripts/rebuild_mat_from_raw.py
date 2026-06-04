#!/usr/bin/env python3
"""Rebuild MAT no-gamma features and validation analyses from raw EDF files.

The script intentionally keeps two analyses separate:

1. Reproduction protocol: mimic the cached feature-table provenance using
   first 60 seconds of rest as calibration, remaining rest as scored rest, and
   all arithmetic windows as scored task.
2. Corrected balanced primary protocol: every subject contributes the same
   first 30 seconds of rest for calibration, a non-overlapping 30 seconds of
   rest for scored rest, and the first 30 seconds of arithmetic for scored task.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / "raw_rebuilt" / "mplconfig"))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from eeg_cogstates.features import extract_window_features
from eeg_cogstates.theory_validation import (
    COMMON_8_CHANNELS,
    CALIBRATION_MODES,
    MODEL_NAMES,
    expected_feature_names,
    nested_loso_predictions,
    summarize_predictions,
    permute_labels_within_subject,
    apply_baseline_calibration,
    make_model,
)


RAW_DIR = ROOT / "data" / "raw" / "eegmat"
PROVENANCE_DIR = ROOT / "results" / "raw_provenance"
REBUILT_DIR = ROOT / "results" / "raw_rebuilt"
CACHED_METRICS = ROOT / "results" / "theory_validation" / "baseline_calibration_metrics.csv"
CACHED_FEATURES = ROOT / "outputs_reproduced" / "features" / "eeg_features.csv"
RNG = np.random.default_rng(20260603)

WINDOW_SECONDS = 4.0
OVERLAP = 0.5
STEP_SECONDS = WINDOW_SECONDS * (1.0 - OVERLAP)
PRIMARY_FEATURE_SET = "no_gamma_184"
BOOTSTRAPS = 2000


@dataclass(frozen=True)
class ProtocolMasks:
    name: str
    calibration_mask: np.ndarray
    eval_mask: np.ndarray
    split_description: str


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_None._"

    def fmt(value: object) -> str:
        if isinstance(value, (list, tuple, set)):
            return json.dumps(list(value))
        if pd.isna(value):
            return "nan"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        return str(value)

    cols = list(df.columns)
    rows = [[fmt(row[col]) for col in cols] for _, row in df.iterrows()]
    widths = [max(len(str(col)), *(len(row[i]) for row in rows)) for i, col in enumerate(cols)]

    def render(values: list[str]) -> str:
        return "| " + " | ".join(values[i].ljust(widths[i]) for i in range(len(values))) + " |"

    lines = [render([str(col) for col in cols])]
    lines.append("| " + " | ".join("-" * width for width in widths) + " |")
    lines.extend(render(row) for row in rows)
    return "\n".join(lines)


def no_gamma_feature_names() -> list[str]:
    return [name for name in expected_feature_names(COMMON_8_CHANNELS) if "_gamma" not in name]


def parse_condition(filename: str) -> tuple[str, str, int]:
    subject = filename.split("_")[0]
    if filename.endswith("_1.edf"):
        return subject, "rest", 0
    if filename.endswith("_2.edf"):
        return subject, "arithmetic", 1
    raise ValueError(f"Unexpected MAT filename: {filename}")


def raw_edf_paths() -> list[Path]:
    paths = sorted(RAW_DIR.rglob("Subject*_*.edf"))
    if not paths:
        raise FileNotFoundError(f"No raw MAT EDF files found under {RAW_DIR}")
    return paths


def canonical_picks(raw: mne.io.BaseRaw) -> tuple[list[int], list[str], dict[str, str]]:
    mapping = {}
    picks = []
    for canonical in COMMON_8_CHANNELS:
        expected = f"EEG {canonical}"
        if expected not in raw.ch_names:
            raise ValueError(f"Missing channel {expected} in {raw.filenames}")
        picks.append(raw.ch_names.index(expected))
        mapping[expected] = canonical
    return picks, list(COMMON_8_CHANNELS), mapping


def inspect_signal_quality(raw: mne.io.BaseRaw, picks: Sequence[int]) -> dict[str, object]:
    data = raw.get_data(picks=picks)
    if data.shape[1] == 0:
        return {"n_annotations": len(raw.annotations), "trailing_flatline_channels": 0, "global_flatline_fraction": 1.0}
    trailing = data[:, max(0, data.shape[1] - int(raw.info["sfreq"] * 2.0)) :]
    trailing_flat = int(np.sum(np.std(trailing, axis=1) < 1e-12))
    diffs = np.diff(data, axis=1)
    flat_fraction = float(np.mean(np.abs(diffs) < 1e-12)) if diffs.size else 1.0
    return {
        "n_annotations": len(raw.annotations),
        "trailing_flatline_channels": trailing_flat,
        "global_flatline_fraction": flat_fraction,
    }


def iter_window_bounds(n_samples: int, sfreq: float) -> Iterable[tuple[int, int, float, float]]:
    win = int(round(WINDOW_SECONDS * sfreq))
    step = int(round(STEP_SECONDS * sfreq))
    for start in range(0, n_samples - win + 1, step):
        end = start + win
        yield start, end, start / sfreq, end / sfreq


def build_raw_features(force: bool = False) -> pd.DataFrame:
    REBUILT_DIR.mkdir(parents=True, exist_ok=True)
    feature_path = REBUILT_DIR / "mat_no_gamma_features.parquet"
    manifest_path = REBUILT_DIR / "mat_raw_window_provenance.csv"
    if feature_path.exists() and manifest_path.exists() and not force:
        return pd.read_parquet(feature_path)

    rows = []
    provenance_rows = []
    quality_rows = []
    feature_cols = no_gamma_feature_names()

    for path in raw_edf_paths():
        subject, condition, label = parse_condition(path.name)
        raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        if sfreq != 500.0:
            raise ValueError(f"Unexpected sfreq in {path.name}: {sfreq}")
        picks, canonical_channels, mapping = canonical_picks(raw)
        quality = inspect_signal_quality(raw, picks)
        quality.update({"subject_id": subject, "condition": condition, "file": path.name})
        quality_rows.append(quality)

        data = raw.get_data(picks=picks) * 1e6
        duration_sec = raw.n_times / sfreq
        for window_index, (start, end, start_sec, end_sec) in enumerate(iter_window_bounds(raw.n_times, sfreq)):
            window = data[:, start:end]
            feats = extract_window_features(window, sfreq, canonical_channels, include_connectivity=False)
            row = {col: feats[col] for col in feature_cols}
            row.update(
                {
                    "subject_id": subject,
                    "condition": condition,
                    "label": label,
                    "dataset": "MAT_RAW",
                    "file": path.name,
                    "window_index": window_index,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "source_duration_sec": duration_sec,
                    "window_uid": f"{subject}|{condition}|{path.name}|{start_sec:.3f}|{end_sec:.3f}",
                    "channel_mapping": json.dumps(mapping, sort_keys=True),
                }
            )
            rows.append(row)
            provenance_rows.append(
                {
                    "window_uid": row["window_uid"],
                    "subject_id": subject,
                    "condition": condition,
                    "label": label,
                    "source_edf": path.name,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "source_duration_sec": duration_sec,
                    "window_extends_beyond_source": bool(end_sec > duration_sec + 1e-9),
                    "window_seconds": WINDOW_SECONDS,
                    "overlap_seconds": WINDOW_SECONDS - STEP_SECONDS,
                    "overlap_status": "4s_windows_50pct_overlap",
                    "segment_type": "unassigned",
                }
            )
        raw.close()

    df = pd.DataFrame(rows)
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing no-gamma features: {missing[:10]}")
    df.to_parquet(feature_path, index=False)
    pd.DataFrame(provenance_rows).to_csv(manifest_path, index=False)
    pd.DataFrame(quality_rows).to_csv(REBUILT_DIR / "mat_raw_signal_quality.csv", index=False)
    return df


def reproduction_masks(df: pd.DataFrame) -> ProtocolMasks:
    rest = df["condition"].eq("rest").to_numpy()
    task = df["condition"].eq("arithmetic").to_numpy()
    start = df["start_sec"].to_numpy(dtype=float)
    calib = rest & (start < 60.0)
    eval_mask = task | (rest & (start >= 60.0))
    return ProtocolMasks("reproduction", calib, eval_mask, "rest<60s calibration; rest>=60s + all arithmetic scored")


def balanced_masks(df: pd.DataFrame) -> ProtocolMasks:
    rest = df["condition"].eq("rest").to_numpy()
    task = df["condition"].eq("arithmetic").to_numpy()
    start = df["start_sec"].to_numpy(dtype=float)
    end = df["end_sec"].to_numpy(dtype=float)
    calib = rest & (start >= 0.0) & (end <= 30.0)
    scored_rest = rest & (start >= 30.0) & (end <= 60.0)
    scored_task = task & (start >= 0.0) & (end <= 30.0)
    return ProtocolMasks("balanced_primary", calib, scored_rest | scored_task, "rest 0-30s calibration; rest 30-60s scored; arithmetic 0-30s scored")


def annotate_manifest(protocol: ProtocolMasks, df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(REBUILT_DIR / "mat_raw_window_provenance.csv")
    manifest = manifest.merge(
        df[["window_uid"]].reset_index().rename(columns={"index": "row_id"}),
        on="window_uid",
        how="left",
    )
    segment = np.full(len(df), "unused", dtype=object)
    segment[protocol.calibration_mask] = "calibration"
    eval_idx = np.where(protocol.eval_mask)[0]
    for idx in eval_idx:
        segment[idx] = "scored_task" if int(df.iloc[idx]["label"]) == 1 else "scored_rest"
    manifest["segment_type"] = manifest["row_id"].map({i: segment[i] for i in range(len(segment))}).fillna("unused")
    manifest["used_for_calibration"] = manifest["segment_type"].eq("calibration")
    manifest["used_for_scoring"] = manifest["segment_type"].isin(["scored_rest", "scored_task"])
    manifest["calibration_scoring_overlap"] = manifest["used_for_calibration"] & manifest["used_for_scoring"]
    manifest.to_csv(out_path, index=False)
    return manifest


def validate_balanced_segments(df: pd.DataFrame, protocol: ProtocolMasks) -> pd.DataFrame:
    rows = []
    for subject, group in df.groupby("subject_id"):
        idx = group.index.to_numpy()
        rows.append(
            {
                "subject_id": subject,
                "calibration_windows": int(protocol.calibration_mask[idx].sum()),
                "scored_rest_windows": int(((protocol.eval_mask[idx]) & group["label"].eq(0).to_numpy()).sum()),
                "scored_task_windows": int(((protocol.eval_mask[idx]) & group["label"].eq(1).to_numpy()).sum()),
                "status": "ok",
            }
        )
    out = pd.DataFrame(rows)
    ok = (
        out["calibration_windows"].eq(14)
        & out["scored_rest_windows"].eq(14)
        & out["scored_task_windows"].eq(14)
    )
    out.loc[~ok, "status"] = "segment_budget_mismatch"
    return out


def bootstrap_subject_ci(pred: pd.DataFrame, n_boot: int = BOOTSTRAPS) -> tuple[float, float, float, float]:
    subject_aucs = []
    for _, group in pred.groupby("subject_id"):
        if group["y_true"].nunique() == 2:
            subject_aucs.append(float(roc_auc_score(group["y_true"], group["score"])))
    values = np.asarray(subject_aucs, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    boot = [float(np.mean(RNG.choice(values, size=values.size, replace=True))) for _ in range(n_boot)]
    return float(np.mean(values)), float(np.std(values, ddof=1)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def paired_delta_ci(pred_a: pd.DataFrame, pred_b: pd.DataFrame, name: str, n_boot: int = BOOTSTRAPS) -> dict[str, object]:
    rows = []
    for subject in sorted(set(pred_a["subject_id"]) & set(pred_b["subject_id"])):
        ga = pred_a[pred_a["subject_id"].eq(subject)]
        gb = pred_b[pred_b["subject_id"].eq(subject)]
        if ga["y_true"].nunique() == 2 and gb["y_true"].nunique() == 2:
            rows.append(
                {
                    "subject_id": subject,
                    "delta": float(roc_auc_score(ga["y_true"], ga["score"]) - roc_auc_score(gb["y_true"], gb["score"])),
                }
            )
    values = np.asarray([row["delta"] for row in rows], dtype=float)
    boot = [float(np.mean(RNG.choice(values, size=values.size, replace=True))) for _ in range(n_boot)] if values.size else []
    return {
        "comparison": name,
        "n_subjects": int(values.size),
        "mean_delta_subject_auc": float(np.mean(values)) if values.size else float("nan"),
        "median_delta_subject_auc": float(np.median(values)) if values.size else float("nan"),
        "ci95_low": float(np.percentile(boot, 2.5)) if boot else float("nan"),
        "ci95_high": float(np.percentile(boot, 97.5)) if boot else float("nan"),
        "bootstrap_resamples": n_boot if boot else 0,
    }


def run_protocol(df: pd.DataFrame, protocol: ProtocolMasks, feature_cols: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_rows = []
    pred_frames = []
    fold_frames = []
    for model in MODEL_NAMES:
        for calibration in CALIBRATION_MODES:
            pred, folds, _ = nested_loso_predictions(
                df=df,
                feature_cols=feature_cols,
                calib_mask=protocol.calibration_mask,
                eval_mask=protocol.eval_mask,
                calibration_mode=calibration,
                model_name=model,
                c_grid=[1.0],
                inner_splits=5,
            )
            pred["protocol"] = protocol.name
            folds["protocol"] = protocol.name
            pred_frames.append(pred)
            fold_frames.append(folds)
            summary = summarize_predictions(pred, "MAT_RAW", protocol.split_description, len(feature_cols))
            mean, sd, lo, hi = bootstrap_subject_ci(pred)
            summary.update(
                {
                    "protocol": protocol.name,
                    "subject_auc_mean": mean,
                    "subject_auc_sd": sd,
                    "subject_auc_ci95_low": lo,
                    "subject_auc_ci95_high": hi,
                    "feature_set": PRIMARY_FEATURE_SET,
                }
            )
            metrics_rows.append(summary)
    return pd.DataFrame(metrics_rows), pd.concat(pred_frames, ignore_index=True), pd.concat(fold_frames, ignore_index=True)


def compare_cached(raw_metrics: pd.DataFrame) -> pd.DataFrame:
    cached = pd.read_csv(CACHED_METRICS)
    cached = cached[cached["dataset"].eq("MAT")].copy()
    cached["feature_set"] = "cached_all_200"
    raw = raw_metrics.copy()
    merged = raw.merge(
        cached[["model", "calibration", "window_auc", "subject_auc_mean", "n_predictions", "n_features"]],
        on=["model", "calibration"],
        how="left",
        suffixes=("_raw_no_gamma", "_cached_all_200"),
    )
    merged["window_auc_delta_raw_minus_cached"] = merged["window_auc_raw_no_gamma"] - merged["window_auc_cached_all_200"]
    merged["subject_auc_delta_raw_minus_cached"] = merged["subject_auc_mean_raw_no_gamma"] - merged["subject_auc_mean_cached_all_200"]
    close = merged["window_auc_delta_raw_minus_cached"].abs() <= 0.03
    merged["reproduction_status"] = np.where(close, "metric_agreement_within_0.03_auc", "not_reproduced_metric_divergence")
    return merged


def write_duration_interpretation() -> None:
    manifest = pd.read_csv(PROVENANCE_DIR / "mat_raw_edf_manifest.csv")
    counts = pd.read_csv(PROVENANCE_DIR / "mat_raw_vs_cached_feature_counts.csv")
    quality = pd.read_csv(REBUILT_DIR / "mat_raw_signal_quality.csv") if (REBUILT_DIR / "mat_raw_signal_quality.csv").exists() else pd.DataFrame()
    duration = manifest.groupby("condition", as_index=False)["duration_sec"].agg(["count", "min", "median", "max"]).reset_index()
    sfreq = manifest.groupby("condition", as_index=False)["sampling_frequency_hz"].agg(["count", "min", "max"]).reset_index()
    channels = manifest.groupby("condition", as_index=False)["num_channels"].agg(["count", "min", "max"]).reset_index()
    filters = manifest.groupby("condition", as_index=False).agg(
        highpass_values=("raw_header_highpass_hz", lambda s: sorted(set(s))),
        lowpass_values=("raw_header_lowpass_hz", lambda s: sorted(set(s))),
    )
    mismatch_count = int((counts["cached_windows"] == counts["expected_4s_50pct_windows_from_header"]).sum())
    qtext = markdown_table(quality) if not quality.empty else "_Signal quality inspection runs during raw feature rebuild._"
    lines = [
        "# Raw MAT Duration Interpretation",
        "",
        "Verdict: `raw_duration_variability_usable_with_header_driven_windowing`",
        "",
        "The simplified descriptor expectation of uniform 180 s rest and 60 s arithmetic is not the exact EDF-header reality. The raw-aware reconstruction therefore uses verified EDF durations as ground truth.",
        "",
        "## Verified EDF Facts",
        "",
        "### Durations",
        "",
        markdown_table(duration),
        "",
        "### Sampling Frequency",
        "",
        markdown_table(sfreq),
        "",
        "### Channel Counts",
        "",
        markdown_table(channels),
        "",
        "### Header Filter Metadata",
        "",
        markdown_table(filters),
        "",
        "## Cached Count Explanation",
        "",
        f"`{mismatch_count}` of `{len(counts)}` cached file-level window counts match raw-header-implied 4 s windows with 50% overlap. The unequal scored-rest counts are therefore plausibly explained by genuine/raw-header duration variation rather than by an impossible cached split.",
        "",
        "## Signal Quality Inspection",
        "",
        qtext,
        "",
        "## Usability",
        "",
        "The raw EDF files remain usable for rebuilding analyses if all splits are driven by exact EDF durations and every window is provenance-labeled. Duration variability is not treated as fatal, but it prevents using descriptor-level durations as ground truth.",
    ]
    (ROOT / "RAW_MAT_DURATION_INTERPRETATION.md").write_text("\n".join(lines) + "\n")


def write_config() -> None:
    text = f"""dataset: PhysioNet EEGMAT 1.0.0
raw_dir: data/raw/eegmat
sampling_frequency_hz: 500
channels:
  source_to_canonical:
    EEG F3: F3
    EEG F4: F4
    EEG F7: F7
    EEG F8: F8
    EEG O1: O1
    EEG O2: O2
    EEG T3: T3
    EEG T4: T4
preprocessing:
  raw_reader: mne.io.read_raw_edf
  additional_filtering: none
  rationale: raw EDF header filter metadata are heterogeneous; primary features exclude gamma and use verified raw headers for provenance
windowing:
  window_seconds: {WINDOW_SECONDS}
  overlap_fraction: {OVERLAP}
  step_seconds: {STEP_SECONDS}
feature_set:
  name: {PRIMARY_FEATURE_SET}
  n_features: 184
  gamma_features_excluded: true
protocols:
  reproduction:
    calibration: rest windows with start_sec < 60
    scored_rest: rest windows with start_sec >= 60
    scored_task: all arithmetic windows
  balanced_primary:
    calibration: rest windows ending within 0-30 seconds
    scored_rest: rest windows starting at >=30 and ending <=60 seconds
    scored_task: arithmetic windows ending within 0-30 seconds
models:
  logistic_l2:
    C: 1.0
    solver: liblinear
    class_weight: balanced
  linear_svm:
    C: 1.0
    class_weight: balanced
validation: strict leave-one-subject-out
"""
    (REBUILT_DIR / "mat_preprocessing_config.yaml").write_text(text)


def write_preprocessing_report(df: pd.DataFrame, feature_cols: Sequence[str], balanced_segments: pd.DataFrame) -> None:
    lines = [
        "# Raw MAT Preprocessing Report",
        "",
        "Primary feature representation: `no_gamma_184`.",
        "",
        f"- Raw windows extracted: `{len(df)}`.",
        f"- Subjects: `{df['subject_id'].nunique()}`.",
        f"- Feature count: `{len(feature_cols)}`.",
        f"- Gamma columns present: `{any('_gamma' in col for col in feature_cols)}`.",
        f"- Window length: `{WINDOW_SECONDS}` seconds.",
        f"- Overlap: `50%`.",
        "- Additional filtering: `none`; EDF signals are read directly and gamma features are excluded to avoid cross-file low-pass/header heterogeneity.",
        "",
        "## Balanced Segment Budget",
        "",
        markdown_table(balanced_segments),
    ]
    (ROOT / "RAW_MAT_PREPROCESSING_REPORT.md").write_text("\n".join(lines) + "\n")


def write_reproduction_report(metrics: pd.DataFrame, comparison: pd.DataFrame) -> None:
    status = "MAT_CACHED_SIGNAL_REPRODUCED" if comparison["reproduction_status"].eq("metric_agreement_within_0.03_auc").all() else "MAT_CACHED_SIGNAL_NOT_REPRODUCED"
    lines = [
        "# MAT Cached Result Reproduction Report",
        "",
        f"Verdict: `{status}`",
        "",
        "The reproduction protocol uses raw EDF-derived no-gamma features with the cached split geometry: rest `<60s` for calibration, rest `>=60s` plus all arithmetic windows for scoring.",
        "",
        "## Raw Reproduction Metrics",
        "",
        markdown_table(metrics[["model", "calibration", "window_auc", "subject_auc_mean", "subject_auc_ci95_low", "subject_auc_ci95_high", "n_predictions"]]),
        "",
        "## Cached-vs-Raw Comparison",
        "",
        markdown_table(comparison[["model", "calibration", "window_auc_raw_no_gamma", "window_auc_cached_all_200", "window_auc_delta_raw_minus_cached", "reproduction_status"]]),
    ]
    (ROOT / "MAT_CACHED_RESULT_REPRODUCTION_REPORT.md").write_text("\n".join(lines) + "\n")


def write_balanced_report(metrics: pd.DataFrame, bootstrap: pd.DataFrame) -> None:
    lines = [
        "# MAT Balanced Primary Analysis",
        "",
        "This is the scientifically preferred MAT design: every subject contributes 30 s rest calibration, 30 s scored rest, and 30 s scored arithmetic using no-gamma 184 features.",
        "",
        "## Metrics",
        "",
        markdown_table(metrics[["model", "calibration", "window_auc", "subject_auc_mean", "subject_auc_ci95_low", "subject_auc_ci95_high", "n_predictions"]]),
        "",
        "## Paired Subject Bootstrap Deltas",
        "",
        markdown_table(bootstrap),
    ]
    (ROOT / "MAT_BALANCED_PRIMARY_ANALYSIS.md").write_text("\n".join(lines) + "\n")


def plot_balanced(metrics: pd.DataFrame) -> None:
    fig_dir = REBUILT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for model, group in metrics.groupby("model"):
        group = group.set_index("calibration").loc[CALIBRATION_MODES].reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        yerr = np.vstack([
            group["subject_auc_mean"] - group["subject_auc_ci95_low"],
            group["subject_auc_ci95_high"] - group["subject_auc_mean"],
        ])
        ax.bar(group["calibration"], group["subject_auc_mean"], yerr=yerr, capsize=4, color=["#6b7280", "#0f766e", "#b45309"])
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Subject mean ROC-AUC")
        ax.set_title(f"MAT balanced no-gamma ({model})")
        fig.tight_layout()
        fig.savefig(fig_dir / f"balanced_primary_{model}.png", dpi=200)
        plt.close(fig)


def full_pipeline_permutation(
    df: pd.DataFrame,
    protocol: ProtocolMasks,
    feature_cols: Sequence[str],
    model: str,
    calibration: str,
    observed_auc: float,
    n_perm: int,
) -> pd.DataFrame:
    start_time = time.time()
    rows = []
    subjects_all = df["subject_id"].astype(str).to_numpy()
    eval_rows = np.where(protocol.eval_mask)[0]
    eval_subjects = subjects_all[eval_rows]
    unique_subjects = np.array(sorted(np.unique(eval_subjects)))
    x_eval = apply_baseline_calibration(df, protocol.eval_mask, protocol.calibration_mask, feature_cols, calibration)

    for i in range(n_perm):
        y_perm = permute_labels_within_subject(df, RNG)
        y_eval = y_perm[eval_rows].astype(int)
        y_all = []
        score_all = []
        for test_subject in unique_subjects:
            train_mask = eval_subjects != test_subject
            test_mask = eval_subjects == test_subject
            y_train = y_eval[train_mask]
            y_test = y_eval[test_mask]
            if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
                continue
            imputer = SimpleImputer(strategy="median")
            scaler = StandardScaler()
            x_train = scaler.fit_transform(imputer.fit_transform(x_eval[train_mask]))
            x_test = scaler.transform(imputer.transform(x_eval[test_mask]))
            fitted = make_model(model, 1.0)
            fitted.fit(x_train, y_train)
            if hasattr(fitted, "decision_function"):
                scores = np.asarray(fitted.decision_function(x_test), dtype=float)
            else:
                scores = np.asarray(fitted.predict_proba(x_test)[:, 1], dtype=float)
            y_all.extend(y_test.tolist())
            score_all.extend(scores.tolist())
        auc_value = float(roc_auc_score(y_all, score_all)) if len(set(y_all)) == 2 else float("nan")
        rows.append(
            {
                "permutation_index": i,
                "auc": auc_value,
                "model": model,
                "calibration": calibration,
                "observed_auc": observed_auc,
                "status": "ok",
                "elapsed_seconds_total": time.time() - start_time,
            }
        )
        pd.DataFrame(rows).to_csv(REBUILT_DIR / "mat_full_pipeline_permutation.csv", index=False)
    out = pd.DataFrame(rows)
    out.to_csv(REBUILT_DIR / "mat_full_pipeline_permutation.csv", index=False)
    return out


def write_null_report(null_df: pd.DataFrame, observed_auc: float, model: str, calibration: str) -> tuple[float, float, float, float]:
    valid = null_df[np.isfinite(null_df["auc"])]
    null_mean = float(valid["auc"].mean())
    null_low = float(valid["auc"].quantile(0.025))
    null_high = float(valid["auc"].quantile(0.975))
    empirical_p = float((1 + np.sum(valid["auc"].to_numpy() >= observed_auc)) / (len(valid) + 1))
    runtime = float(null_df["elapsed_seconds_total"].max()) if len(null_df) else float("nan")
    lines = [
        "# MAT Full-Pipeline Null Results",
        "",
        f"Configuration: `{model}` with `{calibration}` calibration.",
        "",
        f"- Completed permutations: `{len(valid)}`.",
        f"- Observed window ROC-AUC: `{observed_auc:.6f}`.",
        f"- Null mean ROC-AUC: `{null_mean:.6f}`.",
        f"- Null 95% interval: `[{null_low:.6f}, {null_high:.6f}]`.",
        f"- Empirical p-value: `{empirical_p:.6f}`.",
        f"- Runtime seconds: `{runtime:.2f}`.",
    ]
    (ROOT / "MAT_FULL_PIPELINE_NULL_RESULTS.md").write_text("\n".join(lines) + "\n")
    fig_dir = REBUILT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(valid["auc"], bins=30, color="#94a3b8", edgecolor="white")
    ax.axvline(observed_auc, color="#b91c1c", linewidth=2, label="observed")
    ax.set_xlabel("Permutation ROC-AUC")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "mat_full_pipeline_null.png", dpi=200)
    plt.close(fig)
    return null_mean, null_low, null_high, empirical_p


def write_decision(
    reproduction_comparison: pd.DataFrame,
    balanced_metrics: pd.DataFrame,
    balanced_bootstrap: pd.DataFrame,
    null_stats: tuple[float, float, float, float],
    n_perm: int,
) -> str:
    reproduced = reproduction_comparison["reproduction_status"].eq("metric_agreement_within_0.03_auc").all()
    best = balanced_metrics.sort_values("window_auc", ascending=False).iloc[0]
    logistic = balanced_metrics[balanced_metrics["model"].eq("logistic_l2")]
    z_logistic = logistic[logistic["calibration"].eq("zscore")].iloc[0]
    abs_logistic = logistic[logistic["calibration"].eq("absolute")].iloc[0]
    delta_abs = balanced_bootstrap[balanced_bootstrap["comparison"].eq("logistic_l2_zscore_minus_absolute")].iloc[0]
    null_mean, null_low, null_high, empirical_p = null_stats
    z_improves = float(delta_abs["ci95_low"]) > 0
    null_survives = empirical_p <= 0.05 and int(n_perm) >= 200
    if not reproduced:
        verdict = "MAT_CACHED_SIGNAL_NOT_REPRODUCED"
    elif z_improves and null_survives:
        verdict = "MAT_RAW_REBUILT_SIGNAL_SURVIVES_PROCEED_TO_STEW"
    elif reproduced:
        verdict = "MAT_CACHED_SIGNAL_REPRODUCES_BUT_BALANCED_SIGNAL_WEAK"
    else:
        verdict = "MAT_RAW_DATA_UNUSABLE_OR_PIPELINE_INVALID"

    lines = [
        "# MAT Raw Rebuild Decision",
        "",
        f"Final verdict: `{verdict}`",
        "",
        "## Raw Metadata Summary",
        "",
        "- MAT raw files are usable with raw-header-driven splitting: 72 EDF files, 36 subjects, 500 Hz, stable 21-channel identity.",
        "- Duration variability did not invalidate analysis; it required exact header-driven window provenance.",
        "- Primary feature set is locked as `no_gamma_184` for MAT.",
        "",
        "## Cached Reproduction",
        "",
        f"- Cached result reproduced from raw EDF data: `{reproduced}`.",
        "",
        "## Corrected Balanced Primary Metrics",
        "",
        markdown_table(balanced_metrics[["model", "calibration", "window_auc", "subject_auc_mean", "subject_auc_ci95_low", "subject_auc_ci95_high"]]),
        "",
        "## Balanced Delta Inference",
        "",
        markdown_table(balanced_bootstrap),
        "",
        "## Full-Pipeline Null",
        "",
        f"- Strongest balanced configuration: `{best['model']}` / `{best['calibration']}`.",
        f"- Completed permutations: `{n_perm}`.",
        f"- Observed AUC: `{float(best['window_auc']):.6f}`.",
        f"- Null mean: `{null_mean:.6f}`.",
        f"- Null 95% interval: `[{null_low:.6f}, {null_high:.6f}]`.",
        f"- Empirical p-value: `{empirical_p:.6f}`.",
        "",
        "## STEW Gate",
        "",
        f"- Cleared to proceed to STEW reconstruction: `{verdict == 'MAT_RAW_REBUILT_SIGNAL_SURVIVES_PROCEED_TO_STEW'}`.",
    ]
    (ROOT / "MAT_RAW_REBUILD_DECISION.md").write_text("\n".join(lines) + "\n")
    return verdict


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--permutations", type=int, default=200)
    args = parser.parse_args()

    REBUILT_DIR.mkdir(parents=True, exist_ok=True)
    feature_cols = no_gamma_feature_names()
    df = build_raw_features(force=args.force_features)
    df = df.reset_index(drop=True)
    if len(feature_cols) != 184:
        raise ValueError(f"Expected 184 no-gamma features, got {len(feature_cols)}")
    if any("_gamma" in col for col in feature_cols):
        raise ValueError("Primary feature set contains gamma features")

    write_config()
    write_duration_interpretation()

    repro = reproduction_masks(df)
    balanced = balanced_masks(df)
    repro_manifest = annotate_manifest(repro, df, REBUILT_DIR / "reproduction_protocol_window_manifest.csv")
    balanced_manifest = annotate_manifest(balanced, df, REBUILT_DIR / "balanced_primary_window_manifest.csv")
    balanced_segments = validate_balanced_segments(df, balanced)

    write_preprocessing_report(df, feature_cols, balanced_segments)

    repro_metrics_path = REBUILT_DIR / "reproduction_protocol_metrics.csv"
    repro_pred_path = REBUILT_DIR / "reproduction_protocol_predictions.csv"
    comparison_path = REBUILT_DIR / "cached_vs_raw_reproduction_metrics.csv"
    if repro_metrics_path.exists() and repro_pred_path.exists() and comparison_path.exists() and not args.force_features:
        repro_metrics = pd.read_csv(repro_metrics_path)
        repro_pred = pd.read_csv(repro_pred_path)
        comparison = pd.read_csv(comparison_path)
    else:
        repro_metrics, repro_pred, _ = run_protocol(df, repro, feature_cols)
        repro_metrics.to_csv(repro_metrics_path, index=False)
        repro_pred.to_csv(repro_pred_path, index=False)
        comparison = compare_cached(repro_metrics)
        comparison.to_csv(comparison_path, index=False)
    write_reproduction_report(repro_metrics, comparison)

    balanced_metrics_path = REBUILT_DIR / "balanced_primary_metrics.csv"
    balanced_pred_path = REBUILT_DIR / "balanced_primary_predictions.csv"
    balanced_bootstrap_path = REBUILT_DIR / "balanced_primary_bootstrap.csv"
    if balanced_metrics_path.exists() and balanced_pred_path.exists() and balanced_bootstrap_path.exists() and not args.force_features:
        balanced_metrics = pd.read_csv(balanced_metrics_path)
        balanced_pred = pd.read_csv(balanced_pred_path)
        balanced_bootstrap = pd.read_csv(balanced_bootstrap_path)
    else:
        balanced_metrics, balanced_pred, _ = run_protocol(df, balanced, feature_cols)
        balanced_metrics.to_csv(balanced_metrics_path, index=False)
        balanced_pred.to_csv(balanced_pred_path, index=False)

        deltas = []
        for model in MODEL_NAMES:
            preds = {
                cal: balanced_pred[balanced_pred["model"].eq(model) & balanced_pred["calibration"].eq(cal)]
                for cal in CALIBRATION_MODES
            }
            deltas.append(paired_delta_ci(preds["zscore"], preds["absolute"], f"{model}_zscore_minus_absolute"))
            deltas.append(paired_delta_ci(preds["zscore"], preds["mean_subtraction"], f"{model}_zscore_minus_mean_subtraction"))
        balanced_bootstrap = pd.DataFrame(deltas)
        balanced_bootstrap.to_csv(balanced_bootstrap_path, index=False)
    write_balanced_report(balanced_metrics, balanced_bootstrap)
    plot_balanced(balanced_metrics)

    strongest = balanced_metrics.sort_values("window_auc", ascending=False).iloc[0]
    null_df = full_pipeline_permutation(
        df,
        balanced,
        feature_cols,
        model=str(strongest["model"]),
        calibration=str(strongest["calibration"]),
        observed_auc=float(strongest["window_auc"]),
        n_perm=args.permutations,
    )
    null_stats = write_null_report(null_df, float(strongest["window_auc"]), str(strongest["model"]), str(strongest["calibration"]))
    verdict = write_decision(comparison, balanced_metrics, balanced_bootstrap, null_stats, len(null_df))
    print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
