#!/usr/bin/env python3
"""Scientific rescue audits for the EEG workload manuscript.

The script intentionally does not edit the manuscript. It writes audit
manifests, corrected sensitivity analyses, and decision documents that
distinguish executed evidence from unsupported claims.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / "audit" / "mplconfig"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from eeg_cogstates.theory_validation import (
    COMMON_8_CHANNELS,
    DatasetSpec,
    apply_baseline_calibration,
    build_calibration_split,
    expected_feature_names,
    load_feature_table,
    nested_loso_predictions,
)


RNG = np.random.default_rng(20260603)
AUDIT_DIR = ROOT / "results" / "audit"
STAT_DIR = ROOT / "results" / "statistical_comparisons"
PERM_DIR = ROOT / "results" / "permutation"
XFER_DIR = ROOT / "results" / "cross_dataset_binary_transfer"
SENS_DIR = ROOT / "results" / "sensitivity"

MAT_SPEC = DatasetSpec(
    name="MAT",
    path=ROOT / "outputs_reproduced" / "features" / "eeg_features.csv",
    kind="csv",
    baseline_seconds=60.0,
)
STEW_SPEC = DatasetSpec(
    name="STEW",
    path=ROOT / "results" / "multi_dataset" / "stew_features.parquet",
    kind="parquet",
    baseline_fraction=0.5,
)


def setup() -> None:
    for path in [AUDIT_DIR, STAT_DIR, PERM_DIR, XFER_DIR, SENS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def safe_auc(y: Sequence[int], scores: Sequence[float]) -> float:
    y_arr = np.asarray(y, dtype=int)
    if np.unique(y_arr).size < 2:
        return float("nan")
    return float(roc_auc_score(y_arr, np.asarray(scores, dtype=float)))


def model_for(name: str):
    if name == "logistic_l2":
        return LogisticRegression(
            C=1.0,
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            random_state=20260602,
        )
    if name == "linear_svm":
        return LinearSVC(
            C=1.0,
            class_weight="balanced",
            max_iter=20000,
            dual="auto",
            random_state=20260602,
        )
    raise ValueError(name)


def score_model(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x), dtype=float)
    return np.asarray(model.predict_proba(x)[:, 1], dtype=float)


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

    def render(vals: List[str]) -> str:
        return "| " + " | ".join(vals[i].ljust(widths[i]) for i in range(len(vals))) + " |"

    lines = [render([str(c) for c in cols])]
    lines.append("| " + " | ".join("-" * w for w in widths) + " |")
    lines.extend(render(row) for row in rows)
    return "\n".join(lines)


def find_mat_raw_files() -> List[Path]:
    candidates: List[Path] = []
    for base in [ROOT, ROOT.parent, ROOT / "external_data", ROOT / "outputs", ROOT / "outputs_reproduced"]:
        if base.exists():
            candidates.extend(base.rglob("Subject*_*.edf"))
            candidates.extend(base.rglob("Subject*_*.EDF"))
    return sorted(set(candidates))


def audit_mat_calibration() -> Dict[str, object]:
    df = pd.read_csv(MAT_SPEC.path)
    calib, scored, split = build_calibration_split(load_feature_table(MAT_SPEC), MAT_SPEC)
    raw_files = find_mat_raw_files()

    provenance = df[["subject_id", "condition", "label", "file", "window_index", "start_sec", "end_sec"]].copy()
    provenance["used_for_calibration"] = calib
    provenance["used_for_scoring"] = scored
    provenance["calibration_scoring_overlap"] = provenance["used_for_calibration"] & provenance["used_for_scoring"]
    provenance["source"] = "outputs_reproduced/features/eeg_features.csv"
    provenance.to_csv(AUDIT_DIR / "mat_window_provenance.csv", index=False)

    header_rows = []
    grouped = df.groupby("file", as_index=False).agg(
        n_windows=("file", "size"),
        min_start_sec=("start_sec", "min"),
        max_end_sec=("end_sec", "max"),
        label=("label", "first"),
        condition=("condition", "first"),
    )
    raw_lookup = {p.name: p for p in raw_files}
    for row in grouped.itertuples(index=False):
        raw_path = raw_lookup.get(row.file)
        header_rows.append(
            {
                "file": row.file,
                "condition": row.condition,
                "label": int(row.label),
                "raw_file_found": raw_path is not None,
                "raw_path": str(raw_path) if raw_path else "",
                "raw_header_sfreq_hz": np.nan,
                "raw_header_duration_sec": np.nan,
                "feature_table_inferred_duration_sec": float(row.max_end_sec),
                "feature_table_min_start_sec": float(row.min_start_sec),
                "feature_table_n_windows": int(row.n_windows),
                "header_read_status": "raw_edf_not_found_in_repository",
            }
        )
    manifest = pd.DataFrame(header_rows)
    manifest.to_csv(AUDIT_DIR / "mat_file_header_manifest.csv", index=False)

    overlap_count = int(provenance["calibration_scoring_overlap"].sum())
    rest_calib = int(((provenance["label"] == 0) & provenance["used_for_calibration"]).sum())
    rest_scored = int(((provenance["label"] == 0) & provenance["used_for_scoring"]).sum())
    task_scored = int(((provenance["label"] == 1) & provenance["used_for_scoring"]).sum())
    reproduced_auc = safe_auc(
        pd.read_csv(ROOT / "results/theory_validation/baseline_calibration_predictions.csv").query(
            "dataset == 'MAT' and calibration == 'zscore' and model == 'logistic_l2'"
        )["y_true"],
        pd.read_csv(ROOT / "results/theory_validation/baseline_calibration_predictions.csv").query(
            "dataset == 'MAT' and calibration == 'zscore' and model == 'logistic_l2'"
        )["score"],
    )

    subject_counts = (
        provenance.groupby("subject_id")
        .agg(
            rest_calibration_windows=("used_for_calibration", "sum"),
            scored_windows=("used_for_scoring", "sum"),
            calibration_scoring_overlap=("calibration_scoring_overlap", "sum"),
            max_rest_end_sec=("end_sec", lambda s: float(s[provenance.loc[s.index, "label"].eq(0)].max())),
        )
        .reset_index()
    )

    verdict = (
        "feature_table_split_nonoverlapping_but_raw_header_unverified"
        if overlap_count == 0 and rest_scored > 0
        else "invalid_overlap_or_missing_scored_rest"
    )
    if not raw_files:
        headline = "MAT headline result must be treated as provenance-limited until raw EDF headers are audited."
    else:
        headline = "MAT raw EDF files were found, but this script does not yet parse headers."

    lines = [
        "# MAT Calibration Validity Audit",
        "",
        f"Verdict: `{verdict}`.",
        "",
        "## Findings",
        "",
        f"- MAT feature table: `{MAT_SPEC.path.relative_to(ROOT)}`.",
        f"- Split rule audited: `{split}`.",
        f"- Raw EDF files found locally: `{len(raw_files)}`.",
        f"- Rest calibration windows: `{rest_calib}`.",
        f"- Rest scoring windows: `{rest_scored}`.",
        f"- Task scoring windows: `{task_scored}`.",
        f"- Calibration/scoring overlap windows: `{overlap_count}`.",
        f"- Recomputed MAT zscore logistic_l2 AUC from saved predictions: `{reproduced_auc:.6f}`.",
        "",
        "## Interpretation",
        "",
        "The executed feature table contains rest windows extending beyond 60 seconds, so the split used by the code is not internally impossible at the feature-table level.",
        "However, no raw PhysioNet MAT EDF files are present in this repository, so EDF-header duration, sampling frequency, and acquisition metadata cannot be independently verified here.",
        headline,
        "",
        "## Subject-Level Provenance Summary",
        "",
        markdown_table(subject_counts.head(12)),
        "",
        "Full manifests: `results/audit/mat_file_header_manifest.csv` and `results/audit/mat_window_provenance.csv`.",
    ]
    (ROOT / "MAT_CALIBRATION_VALIDITY_AUDIT.md").write_text("\n".join(lines) + "\n")
    return {
        "verdict": verdict,
        "raw_files_found": len(raw_files),
        "overlap_count": overlap_count,
        "rest_scored": rest_scored,
        "task_scored": task_scored,
    }


def audit_ds007262_construct() -> Dict[str, object]:
    roots = [ROOT / "external_data" / "ds007262_validation_raw", ROOT / "external_data" / "ds007262_git"]
    files = sorted({p for root in roots if root.exists() for p in root.glob("sub-*/eeg/*_events.tsv")})
    rows = []
    rest_like = []
    for path in files:
        df = pd.read_csv(path, sep="\t")
        subject = path.parts[-3]
        for row in df.itertuples(index=False):
            trial_type = str(getattr(row, "trial_type", ""))
            marker = str(getattr(row, "marker", ""))
            difficulty = str(getattr(row, "difficulty_range", ""))
            marker_stream = str(getattr(row, "marker_stream", ""))
            is_rest_like = bool(re.search(r"rest|baseline|neutral|fixation", " ".join([trial_type, marker, difficulty]), re.I))
            if is_rest_like:
                rest_like.append((subject, trial_type, marker, difficulty))
            rows.append(
                {
                    "subject_id": subject,
                    "events_file": str(path.relative_to(ROOT)),
                    "trial_type": trial_type,
                    "marker": marker,
                    "difficulty_range": difficulty,
                    "marker_stream": marker_stream,
                    "duration": getattr(row, "duration", np.nan),
                    "istutorial": getattr(row, "istutorial", np.nan),
                    "is_numeric_difficulty": bool(re.match(r"^\d+(?:\.\d+)?-\d+(?:\.\d+)?$", difficulty)),
                    "rest_like_text_match": is_rest_like,
                }
            )
    manifest = pd.DataFrame(rows)
    summary = (
        manifest.groupby(["trial_type", "difficulty_range", "marker_stream", "rest_like_text_match"], dropna=False)
        .size()
        .reset_index(name="n_events")
        .sort_values("n_events", ascending=False)
    )
    summary.to_csv(AUDIT_DIR / "ds007262_event_condition_manifest.csv", index=False)

    numeric_non_tutorial = manifest[
        manifest["is_numeric_difficulty"] & ~manifest["istutorial"].astype(str).str.lower().eq("true")
    ]
    levels = sorted(numeric_non_tutorial["difficulty_range"].dropna().unique(), key=lambda x: float(str(x).split("-")[0]))
    verdict = "valid_rest_baseline_confirmation" if rest_like else "task_anchored_sensitivity_only"
    lines = [
        "# DS007262 Construct-Match Audit",
        "",
        f"Verdict: `{verdict}`.",
        "",
        "## Findings",
        "",
        f"- Event files inspected: `{len(files)}`.",
        f"- Text matches for genuine rest/baseline/neutral/fixation events: `{len(rest_like)}`.",
        f"- Available non-tutorial numeric difficulty ranges: `{', '.join(map(str, levels))}`.",
        "",
        "The event files contain tutorial and arithmetic markers, numeric difficulty ranges, and dropped-sample annotations.",
        "They do not expose a genuine pre-task resting or neutral-baseline condition that matches the MAT/STEW resting-baseline construct.",
        "",
        "## Manuscript Correction Recommendation",
        "",
        "The executed DS007262 result should be described as a `task-anchored calibration sensitivity test`, not a confirmatory resting-baseline transfer test.",
        "Its failure remains informative as a negative external stress test, but it does not cleanly falsify the baseline-relative transfer hypothesis as defined for MAT/STEW rest calibration.",
    ]
    (ROOT / "DS007262_CONSTRUCT_MATCH_AUDIT.md").write_text("\n".join(lines) + "\n")
    return {"verdict": verdict, "event_files": len(files), "rest_like": len(rest_like)}


def subject_auc_table() -> pd.DataFrame:
    pred = pd.read_csv(ROOT / "results/theory_validation/baseline_calibration_predictions.csv")
    rows = []
    for keys, group in pred.groupby(["dataset", "model", "calibration", "subject_id"]):
        dataset, model, calibration, subject = keys
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "calibration": calibration,
                "subject_id": subject,
                "subject_auc": safe_auc(group["y_true"], group["score"]),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_calibration_deltas(n_boot: int = 2000) -> None:
    aucs = subject_auc_table()
    rows = []
    boot_rows = []
    for dataset in ["MAT", "STEW"]:
        for model in ["logistic_l2", "linear_svm"]:
            wide = aucs[(aucs["dataset"] == dataset) & (aucs["model"] == model)].pivot(
                index="subject_id", columns="calibration", values="subject_auc"
            )
            for comparator in ["absolute", "mean_subtraction"]:
                pair = wide[["zscore", comparator]].dropna()
                deltas = pair["zscore"] - pair[comparator]
                values = pair[["zscore", comparator]].to_numpy(dtype=float)
                n_pair = values.shape[0]
                boot = []
                for b in range(n_boot):
                    sample = RNG.integers(0, n_pair, size=n_pair)
                    value = float((values[sample, 0] - values[sample, 1]).mean())
                    boot.append(value)
                    boot_rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "comparison": f"zscore_minus_{comparator}",
                            "bootstrap_index": b,
                            "delta_subject_auc": value,
                        }
                    )
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "comparison": f"zscore_minus_{comparator}",
                        "n_subjects": len(pair),
                        "mean_delta_subject_auc": float(deltas.mean()),
                        "median_delta_subject_auc": float(deltas.median()),
                        "ci95_low": float(np.quantile(boot, 0.025)),
                        "ci95_high": float(np.quantile(boot, 0.975)),
                        "bootstrap_resamples": n_boot,
                    }
                )
    out = pd.DataFrame(rows)
    boots = pd.DataFrame(boot_rows)
    out.to_csv(STAT_DIR / "calibration_method_deltas.csv", index=False)
    boots.to_csv(STAT_DIR / "calibration_method_bootstrap.csv", index=False)
    lines = [
        "# Calibration Improvement Inference",
        "",
        "Subject-level ROC-AUC was used as the inferential unit. For each dataset/model pair, subject AUCs were paired across calibration methods and bootstrapped by resampling subjects with replacement.",
        "",
        markdown_table(out),
        "",
        "Numerical AUC differences should be interpreted through these paired subject-cluster intervals, not as window-level independent evidence.",
    ]
    (ROOT / "CALIBRATION_IMPROVEMENT_INFERENCE.md").write_text("\n".join(lines) + "\n")


def permute_eval_labels_within_subject(df: pd.DataFrame, eval_mask: np.ndarray) -> np.ndarray:
    labels = df["label"].to_numpy(dtype=int).copy()
    subjects = df["subject_id"].astype(str).to_numpy()
    for subject in np.unique(subjects):
        idx = np.where((subjects == subject) & eval_mask)[0]
        labels[idx] = RNG.permutation(labels[idx])
    return labels


def full_pipeline_permutations(n_perm: int = 100) -> None:
    if n_perm <= 0:
        out = pd.DataFrame(
            [
                {
                    "dataset": "MAT",
                    "model": "logistic_l2",
                    "calibration": "zscore",
                    "permutation_index": np.nan,
                    "auc": np.nan,
                    "status": "not_completed_computationally_blocked",
                },
                {
                    "dataset": "STEW",
                    "model": "linear_svm",
                    "calibration": "zscore",
                    "permutation_index": np.nan,
                    "auc": np.nan,
                    "status": "not_completed_computationally_blocked",
                },
            ]
        )
        out.to_csv(PERM_DIR / "full_pipeline_permutation_results.csv", index=False)
        score_shuffle = pd.read_csv(ROOT / "results/theory_validation/negative_control_summary.csv")
        lines = [
            "# Full-Pipeline Null Audit",
            "",
            "Verdict: `not_completed_computationally_blocked`.",
            "",
            "I attempted full LOSO retraining permutation runs at 100, 20, and 5 permutations. Even the 5-permutation run did not complete after several CPU minutes in this environment before reaching downstream audit stages.",
            "The prior score-shuffling control therefore remains insufficient as a full pipeline null, and the manuscript should not present it as validating the complete modeling pipeline.",
            "",
            "## Required Follow-Up",
            "",
            "Run the full retraining permutation audit on a less constrained machine, or optimize the implementation to persist per-permutation results incrementally.",
            "",
            "## Previous Score-Shuffling Control",
            "",
            markdown_table(score_shuffle),
        ]
        (ROOT / "FULL_PIPELINE_NULL_AUDIT.md").write_text("\n".join(lines) + "\n")
        return

    configs = [
        ("MAT", MAT_SPEC, "zscore", "logistic_l2"),
        ("STEW", STEW_SPEC, "zscore", "linear_svm"),
    ]
    rows = []
    for dataset, spec, calibration, model in configs:
        df = load_feature_table(spec)
        features = expected_feature_names()
        calib, eval_mask, _ = build_calibration_split(df, spec)
        true_pred, _, _ = nested_loso_predictions(
            df, features, calib, eval_mask, calibration, model, c_grid=[1.0], inner_splits=3
        )
        observed = safe_auc(true_pred["y_true"], true_pred["score"])
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "calibration": calibration,
                "permutation_index": -1,
                "auc": observed,
                "status": "observed",
            }
        )
        for i in range(n_perm):
            y_perm = permute_eval_labels_within_subject(df, eval_mask)
            pred, _, _ = nested_loso_predictions(
                df,
                features,
                calib,
                eval_mask,
                calibration,
                model,
                c_grid=[1.0],
                inner_splits=3,
                y_override=y_perm,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "calibration": calibration,
                    "permutation_index": i,
                    "auc": safe_auc(pred["y_true"], pred["score"]),
                    "status": "permutation",
                }
            )
            if (i + 1) % 10 == 0:
                print(f"[permutation] {dataset} {i + 1}/{n_perm}")
    out = pd.DataFrame(rows)
    out.to_csv(PERM_DIR / "full_pipeline_permutation_results.csv", index=False)
    score_shuffle = pd.read_csv(ROOT / "results/theory_validation/negative_control_summary.csv")
    summary_rows = []
    for (dataset, model, calibration), group in out.groupby(["dataset", "model", "calibration"]):
        observed = float(group[group["status"] == "observed"]["auc"].iloc[0])
        null = group[group["status"] == "permutation"]["auc"].dropna()
        p = (1 + float((null >= observed).sum())) / (1 + len(null))
        summary_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "calibration": calibration,
                "observed_auc": observed,
                "n_full_pipeline_permutations": int(len(null)),
                "full_pipeline_null_mean": float(null.mean()),
                "full_pipeline_null_sd": float(null.std(ddof=1)),
                "full_pipeline_empirical_p": p,
            }
        )
    summary = pd.DataFrame(summary_rows)
    lines = [
        "# Full-Pipeline Null Audit",
        "",
        "This audit retrained the complete LOSO calibration, imputation, scaling, model-fitting, and evaluation pipeline under within-subject label permutations on evaluation rows.",
        f"Completed permutations per primary configuration: `{n_perm}`.",
        "",
        "## Full-Pipeline Results",
        "",
        markdown_table(summary),
        "",
        "## Previous Score-Shuffling Control",
        "",
        markdown_table(score_shuffle),
        "",
        "The previous control permuted labels against already-generated out-of-fold scores; the full-pipeline null is stricter but was run with fewer permutations because it retrains all folds.",
    ]
    (ROOT / "FULL_PIPELINE_NULL_AUDIT.md").write_text("\n".join(lines) + "\n")


def fit_source_apply_target(
    source: pd.DataFrame,
    target: pd.DataFrame,
    source_spec: DatasetSpec,
    target_spec: DatasetSpec,
    feature_cols: List[str],
    calibration: str,
    model_name: str,
    target_calibration_mode: str,
) -> pd.DataFrame:
    src_calib, src_eval, _ = build_calibration_split(source, source_spec)
    tgt_calib, tgt_eval, _ = build_calibration_split(target, target_spec)
    x_train = apply_baseline_calibration(source, src_eval, src_calib, feature_cols, calibration)
    y_train = source.loc[src_eval, "label"].to_numpy(dtype=int)
    if target_calibration_mode == "zero_shot_absolute":
        x_test = target.loc[tgt_eval, feature_cols].to_numpy(dtype=float)
    else:
        x_test = apply_baseline_calibration(target, tgt_eval, tgt_calib, feature_cols, calibration)
    y_test = target.loc[tgt_eval, "label"].to_numpy(dtype=int)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(x_train))
    x_test = scaler.transform(imputer.transform(x_test))
    model = model_for(model_name)
    model.fit(x_train, y_train)
    scores = score_model(model, x_test)
    out = target.loc[tgt_eval, ["subject_id", "label"]].copy()
    out["score"] = scores
    out["y_true"] = y_test
    return out


def cross_dataset_transfer() -> None:
    mat = load_feature_table(MAT_SPEC)
    stew = load_feature_table(STEW_SPEC)
    features = expected_feature_names()
    rows = []
    pred_rows = []
    for source_name, source, source_spec, target_name, target, target_spec in [
        ("MAT", mat, MAT_SPEC, "STEW", stew, STEW_SPEC),
        ("STEW", stew, STEW_SPEC, "MAT", mat, MAT_SPEC),
    ]:
        for calibration in ["absolute", "mean_subtraction", "zscore"]:
            for target_mode in ["zero_shot_absolute", "unlabeled_baseline_calibrated"]:
                if calibration == "absolute" and target_mode == "unlabeled_baseline_calibrated":
                    continue
                pred = fit_source_apply_target(
                    source,
                    target,
                    source_spec,
                    target_spec,
                    features,
                    calibration,
                    "logistic_l2",
                    target_mode,
                )
                auc = safe_auc(pred["y_true"], pred["score"])
                subj_aucs = pred.groupby("subject_id").apply(lambda g: safe_auc(g["y_true"], g["score"]), include_groups=False)
                rows.append(
                    {
                        "source_dataset": source_name,
                        "target_dataset": target_name,
                        "model": "logistic_l2",
                        "calibration": calibration,
                        "target_mode": target_mode,
                        "window_auc": auc,
                        "subject_auc_mean": float(subj_aucs.mean()),
                        "subject_auc_sd": float(subj_aucs.std(ddof=1)),
                        "n_subjects": int(pred["subject_id"].nunique()),
                        "n_predictions": int(len(pred)),
                    }
                )
                pred["source_dataset"] = source_name
                pred["target_dataset"] = target_name
                pred["calibration"] = calibration
                pred["target_mode"] = target_mode
                pred_rows.append(pred)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(XFER_DIR / "binary_transfer_metrics.csv", index=False)
    pd.concat(pred_rows, ignore_index=True).to_csv(XFER_DIR / "binary_transfer_predictions.csv", index=False)
    loso = pd.read_csv(ROOT / "results/theory_validation/baseline_calibration_metrics.csv")
    lines = [
        "# Cross-Dataset Binary Transfer Results",
        "",
        "These experiments train on one exploratory dataset and test on the other with the same 200-feature intersection. Target workload labels are not used for fitting or calibration-method selection.",
        "",
        "Important caveat: MAT raw EDF headers remain unavailable locally, so this is a feature-table-level transfer audit rather than a raw-provenance-complete result.",
        "",
        "## Transfer Metrics",
        "",
        markdown_table(metrics),
        "",
        "## Within-Dataset LOSO Reference",
        "",
        markdown_table(loso[["dataset", "model", "calibration", "window_auc", "subject_auc_mean", "n_subjects"]]),
    ]
    (ROOT / "CROSS_DATASET_BINARY_TRANSFER_RESULTS.md").write_text("\n".join(lines) + "\n")


def calibration_duration_validity() -> None:
    df = load_feature_table(MAT_SPEC)
    features = expected_feature_names()
    fixed_eval = (df["label"].to_numpy(dtype=int) == 1) | (
        (df["label"].to_numpy(dtype=int) == 0)
        & (pd.to_numeric(df["start_sec"], errors="coerce").to_numpy(dtype=float) >= 60.0)
    )
    subjects = df["subject_id"].astype(str).to_numpy()
    labels = df["label"].to_numpy(dtype=int)
    valid_fixed_subjects = sum(np.unique(labels[(subjects == s) & fixed_eval]).size == 2 for s in np.unique(subjects))
    rows = []
    start_sec = pd.to_numeric(df["start_sec"], errors="coerce").to_numpy(dtype=float)
    run_corrected = os.environ.get("RUN_CORRECTED_DURATION_CURVE", "0") == "1"
    for seconds in range(1, 61, 2):
        calib = (labels == 0) & (start_sec < float(seconds))
        overlap = int((calib & fixed_eval).sum())
        row = {
            "baseline_seconds": seconds,
            "fixed_eval_windows": int(fixed_eval.sum()),
            "calibration_windows": int(calib.sum()),
            "calibration_scoring_overlap": overlap,
            "valid_subjects_fixed_eval": int(valid_fixed_subjects),
            "window_auc": np.nan,
            "n_predictions": 0,
            "status": "not_run_set_RUN_CORRECTED_DURATION_CURVE_1" if not run_corrected else "pending",
        }
        if run_corrected:
            pred, _, _ = nested_loso_predictions(df, features, calib, fixed_eval, "zscore", "logistic_l2", c_grid=[1.0], inner_splits=3)
            row["window_auc"] = safe_auc(pred["y_true"], pred["score"])
            row["n_predictions"] = int(len(pred))
            row["status"] = "ok" if overlap == 0 and len(pred) else "invalid"
        rows.append(row)
    corrected = pd.DataFrame(rows)
    corrected.to_csv(AUDIT_DIR / "mat_calibration_duration_fixed_eval_curve.csv", index=False)
    old = pd.read_csv(ROOT / "results/theory_validation/calibration_duration_curve.csv")
    lines = [
        "# Calibration-Duration Validity Audit",
        "",
        "The previously reported MAT curve changed both calibration duration and the set of scored rest windows because the scoring rule was `rest start >= calibration_seconds`.",
        "That confounds calibration length with evaluation-set composition.",
        "",
        "A corrected feature-table-level design is specified by holding scoring windows fixed to workload rows plus rest rows with `start_sec >= 60`, while varying only the calibration rows (`rest start < duration`).",
        "By default, this audit writes the fixed-evaluation design manifest without running another expensive refit curve. Set `RUN_CORRECTED_DURATION_CURVE=1` to compute corrected AUCs.",
        "",
        "## Corrected Fixed-Evaluation Curve",
        "",
        markdown_table(corrected.head(12)),
        "",
        "## Recommendation",
        "",
        "The old duration figure should be removed or demoted. If a duration curve is retained, use the fixed-evaluation curve and explicitly state that it is feature-table-level because raw EDF header provenance was unavailable.",
    ]
    (ROOT / "CALIBRATION_DURATION_VALIDITY_AUDIT.md").write_text("\n".join(lines) + "\n")


def no_gamma_sensitivity() -> None:
    rows = []
    for dataset, spec, model in [("MAT", MAT_SPEC, "logistic_l2"), ("STEW", STEW_SPEC, "linear_svm")]:
        df = load_feature_table(spec)
        all_features = expected_feature_names()
        no_gamma = [f for f in all_features if "_gamma" not in f]
        calib, eval_mask, _ = build_calibration_split(df, spec)
        for feature_set, features in [("all_200", all_features), ("no_gamma_184", no_gamma)]:
            pred, _, _ = nested_loso_predictions(df, features, calib, eval_mask, "zscore", model, c_grid=[1.0], inner_splits=3)
            subj = pred.groupby("subject_id").apply(lambda g: safe_auc(g["y_true"], g["score"]), include_groups=False)
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "calibration": "zscore",
                    "feature_set": feature_set,
                    "n_features": len(features),
                    "window_auc": safe_auc(pred["y_true"], pred["score"]),
                    "subject_auc_mean": float(subj.mean()),
                    "subject_auc_sd": float(subj.std(ddof=1)),
                    "n_predictions": int(len(pred)),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(SENS_DIR / "no_gamma_comparison.csv", index=False)
    lines = [
        "# Gamma Feature Validity Audit",
        "",
        "Code audit: gamma features are defined as 30--45 Hz absolute and relative bandpower (`band_abs_{ch}_gamma`, `band_rel_{ch}_gamma`).",
        "MAT raw acquisition/header metadata were not available locally, STEW is a 128 Hz Emotiv-derived dataset, and DS007262 is 250 Hz BrainVision. This heterogeneity prevents strong biological interpretation of gamma features across datasets.",
        "",
        "## No-Gamma Sensitivity",
        "",
        markdown_table(out),
        "",
        "## Recommendation",
        "",
        "Gamma features may remain as predictive engineering features only if sensitivity results are reported. They should not be interpreted biologically without dataset-compatible acquisition/filter audits and source/artifact controls.",
    ]
    (ROOT / "GAMMA_FEATURE_VALIDITY_AUDIT.md").write_text("\n".join(lines) + "\n")


def external_dataset_requirements() -> None:
    (ROOT / "NEW_EXTERNAL_CONFIRMATION_REQUIREMENTS.md").write_text(
        "# New External Confirmation Requirements\n\n"
        "DS007262 has already been inspected and does not expose a genuine resting-baseline condition in the analyzed event files. The next confirmation dataset must be selected before viewing target outcomes.\n\n"
        "Required properties:\n\n"
        "- Raw EEG access, not only precomputed features.\n"
        "- A genuine resting, neutral, or fixation baseline segment before scored workload trials.\n"
        "- Graded workload levels, continuous workload ratings, or validated subjective workload scores.\n"
        "- Behavioral outcomes and/or subjective workload ratings such as NASA-TLX.\n"
        "- Sufficient channel compatibility with MAT/STEW or a predeclared harmonization strategy.\n"
        "- Clear acquisition metadata including sampling rate, references, filters, and artifact handling.\n"
        "- Clear licensing, provenance, and citation metadata.\n"
        "- No prior use in this repository for model selection, calibration-method selection, or manuscript claim selection.\n"
        "- A plan for freezing preprocessing, feature definitions, calibration method, model class, and success criteria before target labels are scored.\n"
    )
    (ROOT / "EXTERNAL_DATASET_ELIGIBILITY_TEMPLATE.md").write_text(
        "# External Dataset Eligibility Template\n\n"
        "| Criterion | Candidate dataset answer | Pass/Fail | Notes |\n"
        "| --- | --- | --- | --- |\n"
        "| Dataset name and DOI/URL |  |  |  |\n"
        "| Raw EEG available |  |  |  |\n"
        "| Genuine rest/neutral baseline before workload |  |  |  |\n"
        "| Graded workload or continuous ratings |  |  |  |\n"
        "| Behavioral outcomes available |  |  |  |\n"
        "| Subjective workload/fatigue/anxiety ratings |  |  |  |\n"
        "| Sampling rate and filters documented |  |  |  |\n"
        "| Reference montage documented |  |  |  |\n"
        "| Channel overlap/harmonization defensible |  |  |  |\n"
        "| License permits analysis |  |  |  |\n"
        "| Not previously used for model selection in this repo |  |  |  |\n"
        "| Frozen analysis plan written before outcome inspection |  |  |  |\n\n"
        "Decision: `eligible`, `eligible_with_constraints`, or `not_eligible`.\n"
    )


def rescue_decision(mat: Dict[str, object], ds: Dict[str, object]) -> None:
    if mat["overlap_count"] != 0 or mat["rest_scored"] == 0:
        verdict = "RESULTS_INVALID_REBUILD_REQUIRED"
    elif ds["verdict"] != "valid_rest_baseline_confirmation":
        verdict = "EXPLORATORY_RESULTS_SALVAGEABLE_NEW_CONFIRMATION_REQUIRED"
    else:
        verdict = "NEGATIVE_METHODS_PAPER_DEFENSIBLE"
    lines = [
        "# Scientific Rescue Decision",
        "",
        f"Verdict: `{verdict}`.",
        "",
        "## Basis",
        "",
        f"- MAT feature-table calibration/scoring overlap count: `{mat['overlap_count']}`.",
        f"- MAT later rest scoring windows available: `{mat['rest_scored']}`.",
        f"- MAT raw EDF files found locally: `{mat['raw_files_found']}`.",
        f"- DS007262 construct verdict: `{ds['verdict']}`.",
        "",
        "## Decision Logic",
        "",
        "The MAT feature table supports a non-overlapping 60-second calibration split, but raw EDF header provenance is unavailable locally and should be verified before making strong acquisition-duration claims.",
        "DS007262 is not a construct-matched resting-baseline confirmation because the analyzed event files do not contain a genuine rest/neutral baseline. The DS result is therefore a negative task-anchored sensitivity test, not a clean confirmatory test of the baseline-relative transfer hypothesis.",
        "",
        "## Required Before Manuscript Revision",
        "",
        "- Revise DS007262 language from confirmatory resting-baseline transfer to task-anchored sensitivity / negative external stress test.",
        "- Demote or replace the old calibration-duration curve with the fixed-evaluation audit result.",
        "- Report paired subject-level uncertainty for calibration-method improvements.",
        "- Report full-pipeline permutation nulls separately from score-shuffling controls.",
        "- Avoid biological gamma interpretation unless no-gamma sensitivity and metadata constraints are included.",
        "- Select a new untouched construct-matched external dataset using `NEW_EXTERNAL_CONFIRMATION_REQUIREMENTS.md`.",
    ]
    (ROOT / "SCIENTIFIC_RESCUE_DECISION.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    setup()
    mat = audit_mat_calibration()
    ds = audit_ds007262_construct()
    bootstrap_calibration_deltas(n_boot=2000)
    full_pipeline_permutations(n_perm=int(os.environ.get("FULL_PIPELINE_PERMUTATIONS", "0")))
    cross_dataset_transfer()
    calibration_duration_validity()
    no_gamma_sensitivity()
    external_dataset_requirements()
    rescue_decision(mat, ds)


if __name__ == "__main__":
    main()
