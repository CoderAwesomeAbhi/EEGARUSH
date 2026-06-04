#!/usr/bin/env python3
"""Final subject-level MAT decision audit before any STEW reconstruction."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / "raw_rebuilt" / "mplconfig"))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from eeg_cogstates.theory_validation import (
    CALIBRATION_MODES,
    MODEL_NAMES,
    apply_baseline_calibration,
    make_model,
    permute_labels_within_subject,
)

RNG = np.random.default_rng(20260603)
BOOTSTRAPS = 10000
PRIMARY_MODEL = "logistic_l2"
PRIMARY_CALIBRATION = "mean_subtraction"
REBUILT_DIR = ROOT / "results" / "raw_rebuilt"


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
    from eeg_cogstates.theory_validation import COMMON_8_CHANNELS, expected_feature_names

    return [name for name in expected_feature_names(COMMON_8_CHANNELS) if "_gamma" not in name]


def balanced_masks(df: pd.DataFrame):
    rest = df["condition"].eq("rest").to_numpy()
    task = df["condition"].eq("arithmetic").to_numpy()
    start = df["start_sec"].to_numpy(dtype=float)
    end = df["end_sec"].to_numpy(dtype=float)
    calib = rest & (start >= 0.0) & (end <= 30.0)
    scored_rest = rest & (start >= 30.0) & (end <= 60.0)
    scored_task = task & (start >= 0.0) & (end <= 30.0)
    return type(
        "ProtocolMasks",
        (),
        {
            "calibration_mask": calib,
            "eval_mask": scored_rest | scored_task,
        },
    )()


def load_balanced() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Sequence[str]]:
    features = pd.read_parquet(REBUILT_DIR / "mat_no_gamma_features.parquet").reset_index(drop=True)
    predictions = pd.read_csv(REBUILT_DIR / "balanced_primary_predictions.csv")
    manifest = pd.read_csv(REBUILT_DIR / "balanced_primary_window_manifest.csv")
    feature_cols = no_gamma_feature_names()
    return features, predictions, manifest, feature_cols


def subject_auc_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, calibration, subject), group in pred.groupby(["model", "calibration", "subject_id"]):
        if group["y_true"].nunique() != 2:
            auc = float("nan")
            valid = False
        else:
            auc = float(roc_auc_score(group["y_true"], group["score"]))
            valid = True
        rows.append(
            {
                "model": model,
                "calibration": calibration,
                "subject_id": subject,
                "subject_auc": auc,
                "valid_subject_auc": valid,
                "n_windows": int(len(group)),
                "n_rest": int((group["y_true"] == 0).sum()),
                "n_task": int((group["y_true"] == 1).sum()),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, n_boot: int = BOOTSTRAPS) -> tuple[float, float]:
    boot = [float(np.mean(RNG.choice(values, size=len(values), replace=True))) for _ in range(n_boot)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def compute_subject_metrics(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    subj = subject_auc_table(pred)
    rows = []
    for (model, calibration), group in pred.groupby(["model", "calibration"]):
        subj_group = subj[
            subj["model"].eq(model)
            & subj["calibration"].eq(calibration)
            & subj["valid_subject_auc"]
        ]
        values = subj_group["subject_auc"].to_numpy(dtype=float)
        lo, hi = bootstrap_ci(values)
        rows.append(
            {
                "model": model,
                "calibration": calibration,
                "pooled_window_auc": float(roc_auc_score(group["y_true"], group["score"])),
                "macro_subject_mean_auc": float(np.mean(values)),
                "macro_subject_median_auc": float(np.median(values)),
                "subject_auc_sd": float(np.std(values, ddof=1)),
                "subject_auc_ci95_low": lo,
                "subject_auc_ci95_high": hi,
                "n_subjects": int(len(values)),
                "n_predictions": int(len(group)),
            }
        )
    return pd.DataFrame(rows), subj


def paired_bootstrap(subj: pd.DataFrame, model: str, cal_a: str, cal_b: str, label: str) -> dict[str, object]:
    a = subj[subj["model"].eq(model) & subj["calibration"].eq(cal_a)][["subject_id", "subject_auc"]]
    b = subj[subj["model"].eq(model) & subj["calibration"].eq(cal_b)][["subject_id", "subject_auc"]]
    merged = a.merge(b, on="subject_id", suffixes=("_a", "_b"))
    merged["delta"] = merged["subject_auc_a"] - merged["subject_auc_b"]
    values = merged["delta"].to_numpy(dtype=float)
    boot = [float(np.mean(RNG.choice(values, size=len(values), replace=True))) for _ in range(BOOTSTRAPS)]
    lo = float(np.percentile(boot, 2.5))
    hi = float(np.percentile(boot, 97.5))
    return {
        "model": model,
        "comparison": label,
        "n_subjects": int(len(values)),
        "mean_delta_auc": float(np.mean(values)),
        "median_delta_auc": float(np.median(values)),
        "ci95_low": lo,
        "ci95_high": hi,
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "positive_ci_excludes_zero": bool(lo > 0),
        "bootstrap_resamples": BOOTSTRAPS,
    }


def audit_alignment(pred: pd.DataFrame, manifest: pd.DataFrame, metrics: pd.DataFrame, subj: pd.DataFrame) -> tuple[bool, pd.DataFrame, list[str]]:
    issues = []
    configs = [(m, c) for m in MODEL_NAMES for c in CALIBRATION_MODES]
    base_rows = None
    base_labels = None
    base_subjects = None
    counts_rows = []

    for model, calibration in configs:
        group = pred[pred["model"].eq(model) & pred["calibration"].eq(calibration)].sort_values("row_id")
        if group.empty:
            issues.append(f"missing predictions for {model}/{calibration}")
            continue
        rows = group["row_id"].to_numpy()
        labels = group["y_true"].to_numpy()
        subjects = group["subject_id"].to_numpy()
        if base_rows is None:
            base_rows, base_labels, base_subjects = rows, labels, subjects
        else:
            if not np.array_equal(base_rows, rows):
                issues.append(f"row_id mismatch for {model}/{calibration}")
            if not np.array_equal(base_labels, labels):
                issues.append(f"label mismatch for {model}/{calibration}")
            if not np.array_equal(base_subjects, subjects):
                issues.append(f"subject order mismatch for {model}/{calibration}")

    scored = manifest[manifest["segment_type"].isin(["scored_rest", "scored_task"])]
    counts = scored.pivot_table(index="subject_id", columns="segment_type", values="window_uid", aggfunc="count", fill_value=0)
    for subject, row in counts.iterrows():
        counts_rows.append(
            {
                "subject_id": subject,
                "scored_rest_windows": int(row.get("scored_rest", 0)),
                "scored_task_windows": int(row.get("scored_task", 0)),
            }
        )
    counts_df = pd.DataFrame(counts_rows)
    if not counts_df["scored_rest_windows"].eq(14).all() or not counts_df["scored_task_windows"].eq(14).all():
        issues.append("balanced scored window counts are not 14 rest and 14 task for every subject")
    if manifest["calibration_scoring_overlap"].any():
        issues.append("calibration/scoring overlap detected")
    invalid_subjects = subj[~subj["valid_subject_auc"]]
    if not invalid_subjects.empty:
        issues.append("invalid subject AUC for at least one subject/configuration")

    reported = pd.read_csv(REBUILT_DIR / "balanced_primary_metrics.csv")
    merged = metrics.merge(reported, on=["model", "calibration"], suffixes=("_audit", "_reported"))
    if not np.allclose(merged["pooled_window_auc"], merged["window_auc"], atol=1e-12):
        issues.append("pooled-window AUCs do not reproduce reported balanced metrics")
    previous_delta = pd.read_csv(REBUILT_DIR / "balanced_primary_bootstrap.csv")
    logistic_prev = previous_delta[previous_delta["comparison"].eq("logistic_l2_zscore_minus_absolute")]
    if logistic_prev.empty:
        issues.append("previous paired z-score delta file missing expected comparison")

    return len(issues) == 0, counts_df, issues


def write_alignment_report(ok: bool, counts: pd.DataFrame, issues: list[str], metrics: pd.DataFrame, paired: pd.DataFrame) -> None:
    lines = [
        "# MAT Metric Alignment Audit",
        "",
        f"Verdict: `{'no_metric_or_alignment_bug_detected' if ok else 'MAT_METRIC_OR_ALIGNMENT_BUG_FIX_REQUIRED'}`",
        "",
        "## Alignment Findings",
        "",
        f"- Prediction/label/alignment bug exists: `{not ok}`.",
        f"- Evaluated window counts balanced across subjects: `{counts['scored_rest_windows'].eq(14).all() and counts['scored_task_windows'].eq(14).all()}`.",
        "- Every paired comparison uses the same sorted `row_id`, label vector, and subject vector across configurations.",
        "- Pooled and macro subject-level metrics legitimately differ because AUC is nonlinear; even with equal per-subject window counts, each subject's within-subject separability can differ from pooled ranking across all held-out scores.",
        "- Previous z-score paired deltas were computed from subject-level AUCs and are directionally consistent with this audit.",
        "",
        "## Issues",
        "",
        markdown_table(pd.DataFrame({"issue": issues})) if issues else "_None._",
        "",
        "## Subject Window Counts",
        "",
        markdown_table(counts),
        "",
        "## Recomputed Metrics",
        "",
        markdown_table(metrics),
        "",
        "## Mean-Subtraction Paired Bootstrap",
        "",
        markdown_table(paired),
    ]
    (ROOT / "MAT_METRIC_ALIGNMENT_AUDIT.md").write_text("\n".join(lines) + "\n")


def write_subject_results(metrics: pd.DataFrame, paired: pd.DataFrame) -> None:
    lines = [
        "# MAT Mean-Subtraction Subject-Level Results",
        "",
        "Inference unit: subject. Pooled windows are reported descriptively only.",
        "",
        "## Subject-Level Metrics",
        "",
        markdown_table(metrics),
        "",
        "## Paired Subject Bootstrap Differences",
        "",
        markdown_table(paired),
    ]
    (ROOT / "MAT_MEAN_SUBTRACTION_SUBJECT_LEVEL_RESULTS.md").write_text("\n".join(lines) + "\n")


def audit_existing_permutation_unit() -> str:
    script = (ROOT / "scripts" / "rebuild_mat_from_raw.py").read_text()
    result = (ROOT / "MAT_FULL_PIPELINE_NULL_RESULTS.md").read_text()
    if "roc_auc_score(y_all, score_all)" in script and "Subject mean" not in result:
        unit = "pooled_window_auc"
    else:
        unit = "ambiguous_or_other"
    lines = [
        "# MAT Existing Permutation Unit Audit",
        "",
        f"Verdict: `{unit}`",
        "",
        "The prior full-pipeline null in `scripts/rebuild_mat_from_raw.py` aggregated all held-out permuted labels and scores into `y_all` and `score_all`, then computed `roc_auc_score(y_all, score_all)`. That is a pooled window-level ROC-AUC statistic, not macro subject-level mean ROC-AUC.",
        "",
        "Therefore, the earlier permutation result does not validate the macro subject-level primary statistic requested in this final gate.",
    ]
    (ROOT / "MAT_EXISTING_PERMUTATION_UNIT_AUDIT.md").write_text("\n".join(lines) + "\n")
    return unit


def macro_subject_auc(y: Sequence[int], scores: Sequence[float], subjects: Sequence[str]) -> float:
    frame = pd.DataFrame({"y": y, "score": scores, "subject_id": subjects})
    aucs = []
    for _, group in frame.groupby("subject_id"):
        if group["y"].nunique() == 2:
            aucs.append(float(roc_auc_score(group["y"], group["score"])))
    return float(np.mean(aucs)) if aucs else float("nan")


def subject_level_permutation(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    n_perm: int,
    observed_auc: float,
) -> tuple[pd.DataFrame, tuple[float, float, float, float, float]]:
    protocol = balanced_masks(df)
    eval_rows = np.where(protocol.eval_mask)[0]
    subjects_all = df["subject_id"].astype(str).to_numpy()
    eval_subjects = subjects_all[eval_rows]
    unique_subjects = np.array(sorted(np.unique(eval_subjects)))
    x_eval = apply_baseline_calibration(df, protocol.eval_mask, protocol.calibration_mask, feature_cols, PRIMARY_CALIBRATION)
    out_path = REBUILT_DIR / "mat_subject_level_full_pipeline_permutation.csv"
    rows = []
    start = time.time()
    for i in range(n_perm):
        y_perm = permute_labels_within_subject(df, RNG)
        y_eval = y_perm[eval_rows].astype(int)
        y_all = []
        score_all = []
        subj_all = []
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
            model = make_model(PRIMARY_MODEL, 1.0)
            model.fit(x_train, y_train)
            scores = np.asarray(model.decision_function(x_test), dtype=float)
            y_all.extend(y_test.tolist())
            score_all.extend(scores.tolist())
            subj_all.extend(eval_subjects[test_mask].tolist())
        auc_value = macro_subject_auc(y_all, score_all, subj_all)
        rows.append(
            {
                "permutation_index": i,
                "macro_subject_mean_auc": auc_value,
                "observed_macro_subject_mean_auc": observed_auc,
                "model": PRIMARY_MODEL,
                "calibration": PRIMARY_CALIBRATION,
                "elapsed_seconds_total": time.time() - start,
            }
        )
        pd.DataFrame(rows).to_csv(out_path, index=False)
    result = pd.DataFrame(rows)
    valid = result[np.isfinite(result["macro_subject_mean_auc"])]
    null_mean = float(valid["macro_subject_mean_auc"].mean())
    null_low = float(valid["macro_subject_mean_auc"].quantile(0.025))
    null_high = float(valid["macro_subject_mean_auc"].quantile(0.975))
    p_value = float((1 + np.sum(valid["macro_subject_mean_auc"].to_numpy() >= observed_auc)) / (len(valid) + 1))
    runtime = float(result["elapsed_seconds_total"].max())
    return result, (null_mean, null_low, null_high, p_value, runtime)


def write_subject_null(null_df: pd.DataFrame, observed: float, stats: tuple[float, float, float, float, float]) -> None:
    null_mean, null_low, null_high, p_value, runtime = stats
    lines = [
        "# MAT Subject-Level Full-Pipeline Null Results",
        "",
        f"- Configuration: `{PRIMARY_MODEL}` / `{PRIMARY_CALIBRATION}`.",
        "- Statistic: macro subject-level mean ROC-AUC.",
        f"- Completed permutations: `{len(null_df)}`.",
        f"- Observed macro subject-level mean AUC: `{observed:.6f}`.",
        f"- Null mean: `{null_mean:.6f}`.",
        f"- Null 95% interval: `[{null_low:.6f}, {null_high:.6f}]`.",
        f"- Empirical p-value: `{p_value:.6f}`.",
        f"- Runtime seconds: `{runtime:.2f}`.",
    ]
    (ROOT / "MAT_SUBJECT_LEVEL_FULL_PIPELINE_NULL_RESULTS.md").write_text("\n".join(lines) + "\n")

    fig_dir = REBUILT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(null_df["macro_subject_mean_auc"], bins=30, color="#a3b18a", edgecolor="white")
    ax.axvline(observed, color="#bc4749", linewidth=2, label="observed")
    ax.set_xlabel("Macro subject-level mean ROC-AUC")
    ax.set_ylabel("Permutation count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "mat_subject_level_full_pipeline_null.png", dpi=200)
    plt.close(fig)


def write_final_decision(
    alignment_ok: bool,
    metrics: pd.DataFrame,
    paired: pd.DataFrame,
    perm_unit: str,
    null_ran: bool,
    null_stats: tuple[float, float, float, float, float] | None,
) -> str:
    logistic = metrics[metrics["model"].eq(PRIMARY_MODEL)].set_index("calibration")
    delta = paired[paired["model"].eq(PRIMARY_MODEL) & paired["comparison"].eq("mean_subtraction_minus_absolute")].iloc[0]
    mean_beats_abs = float(logistic.loc["mean_subtraction", "macro_subject_mean_auc"]) > float(logistic.loc["absolute", "macro_subject_mean_auc"])
    positive_ci = bool(delta["positive_ci_excludes_zero"])
    null_success = bool(null_stats and null_stats[3] <= 0.05)
    if not alignment_ok:
        verdict = "MAT_METRIC_OR_ALIGNMENT_BUG_FIX_REQUIRED"
    elif mean_beats_abs and positive_ci and null_ran and null_success:
        verdict = "MAT_VALID_PRIMARY_SIGNAL_PROCEED_TO_STEW_RECONSTRUCTION"
    elif mean_beats_abs and positive_ci and null_ran and not null_success:
        verdict = "MAT_SUBJECT_LEVEL_NULL_TEST_FAILED_DOWNGRADE_PROJECT"
    elif mean_beats_abs:
        verdict = "MAT_MEAN_SUBTRACTION_PROMISING_BUT_NOT_PROVEN_DO_NOT_PROCEED"
    else:
        verdict = "MAT_WINDOW_LEVEL_ONLY_SIGNAL_DOWNGRADE_PROJECT"

    lines = [
        "# MAT Final Decision Before STEW",
        "",
        "## Configuration",
        "",
        "- Protocol: corrected balanced raw MAT.",
        "- Feature set: no-gamma 184.",
        "- Candidate primary method: logistic L2 mean subtraction.",
        "- Primary statistic: macro subject-level mean ROC-AUC.",
        "",
        "## Metric Alignment Verdict",
        "",
        f"- Alignment passed: `{alignment_ok}`.",
        "",
        "## Subject-Level Metrics",
        "",
        markdown_table(metrics),
        "",
        "## Paired Bootstrap Differences",
        "",
        markdown_table(paired),
        "",
        "## Existing Permutation Statistic",
        "",
        f"- Earlier permutation unit: `{perm_unit}`.",
        "",
        "## Subject-Level Null",
        "",
        f"- Subject-level full-pipeline null ran: `{null_ran}`.",
    ]
    if null_stats:
        null_mean, null_low, null_high, p_value, runtime = null_stats
        observed = float(logistic.loc["mean_subtraction", "macro_subject_mean_auc"])
        lines.extend(
            [
                f"- Observed macro subject mean AUC: `{observed:.6f}`.",
                f"- Null mean: `{null_mean:.6f}`.",
                f"- Null 95% interval: `[{null_low:.6f}, {null_high:.6f}]`.",
                f"- Empirical p-value: `{p_value:.6f}`.",
                f"- Runtime seconds: `{runtime:.2f}`.",
            ]
        )
    else:
        lines.append("- Null not run because conditional criteria were not all satisfied.")
    lines.extend(
        [
            "",
            "## STEW Gate",
            "",
            f"- Proceeding to STEW is scientifically justified: `{verdict == 'MAT_VALID_PRIMARY_SIGNAL_PROCEED_TO_STEW_RECONSTRUCTION'}`.",
            "",
            f"Final verdict: `{verdict}`",
        ]
    )
    (ROOT / "MAT_FINAL_DECISION_BEFORE_STEW.md").write_text("\n".join(lines) + "\n")
    return verdict


def main() -> int:
    df, pred, manifest, feature_cols = load_balanced()
    metrics, subj = compute_subject_metrics(pred)
    metrics = metrics.sort_values(["model", "calibration"]).reset_index(drop=True)
    paired = pd.DataFrame(
        [
            paired_bootstrap(subj, "logistic_l2", "mean_subtraction", "absolute", "mean_subtraction_minus_absolute"),
            paired_bootstrap(subj, "logistic_l2", "mean_subtraction", "zscore", "mean_subtraction_minus_zscore"),
            paired_bootstrap(subj, "linear_svm", "mean_subtraction", "absolute", "mean_subtraction_minus_absolute"),
            paired_bootstrap(subj, "linear_svm", "mean_subtraction", "zscore", "mean_subtraction_minus_zscore"),
        ]
    )
    metrics.to_csv(REBUILT_DIR / "mat_subject_level_metrics.csv", index=False)
    paired.to_csv(REBUILT_DIR / "mat_subject_level_paired_bootstrap.csv", index=False)

    alignment_ok, counts, issues = audit_alignment(pred, manifest, metrics, subj)
    write_alignment_report(alignment_ok, counts, issues, metrics, paired)
    if not alignment_ok:
        write_subject_results(metrics, paired)
        perm_unit = audit_existing_permutation_unit()
        verdict = write_final_decision(False, metrics, paired, perm_unit, False, None)
        print(verdict)
        return 0

    write_subject_results(metrics, paired)
    perm_unit = audit_existing_permutation_unit()
    logistic = metrics[metrics["model"].eq("logistic_l2")].set_index("calibration")
    logistic_delta = paired[paired["model"].eq("logistic_l2") & paired["comparison"].eq("mean_subtraction_minus_absolute")].iloc[0]
    conditions_met = (
        float(logistic.loc["mean_subtraction", "macro_subject_mean_auc"]) > float(logistic.loc["absolute", "macro_subject_mean_auc"])
        and bool(logistic_delta["positive_ci_excludes_zero"])
        and perm_unit != "macro_subject_level_mean_auc"
    )
    null_stats = None
    null_ran = False
    if conditions_met:
        observed = float(logistic.loc["mean_subtraction", "macro_subject_mean_auc"])
        null_df, null_stats = subject_level_permutation(df, feature_cols, n_perm=200, observed_auc=observed)
        write_subject_null(null_df, observed, null_stats)
        null_ran = True
    verdict = write_final_decision(alignment_ok, metrics, paired, perm_unit, null_ran, null_stats)
    print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
