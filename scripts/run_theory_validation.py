#!/usr/bin/env python3
"""Run leakage-aware theory validation experiments.

Outputs are written to results/theory_validation/ and a dense root-level
THEORY_VALIDATION_RESULTS.md report is regenerated on every run.
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "theory_validation"
FIGURES_DIR = RESULTS_DIR / "figures"
os.environ.setdefault("MPLCONFIGDIR", str(RESULTS_DIR / "mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import sys

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_cogstates.theory_validation import (  # noqa: E402
    CALIBRATION_MODES,
    COMMON_8_CHANNELS,
    DEFAULT_C_GRID,
    MODEL_NAMES,
    DatasetSpec,
    build_calibration_split,
    effect_direction_correlation,
    expected_feature_names,
    final_direction_coefficients,
    load_feature_table,
    nested_loso_predictions,
    permute_labels_within_subject,
    roc_points,
    select_common_8_features,
    summarize_predictions,
)

DURATION_MODEL_NAMES = ["logistic_l2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--mat-baseline-seconds", type=float, default=60.0)
    parser.add_argument("--stew-baseline-fraction", type=float, default=0.5)
    parser.add_argument("--duration-step", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=20260602)
    parser.add_argument("--summary-only", action="store_true", help="Regenerate THEORY_VALIDATION_RESULTS.md from existing CSV outputs.")
    return parser.parse_args()


def setup_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "mplconfig").mkdir(parents=True, exist_ok=True)


def markdown_table(df: pd.DataFrame, floatfmt: str = ".6f") -> str:
    """Render a small DataFrame as markdown without the optional tabulate package."""
    if df.empty:
        return "_No rows._"

    def fmt(value: object) -> str:
        if pd.isna(value):
            return "nan"
        if isinstance(value, (float, np.floating)):
            return format(float(value), floatfmt)
        return str(value)

    cols = list(df.columns)
    rows = [[fmt(row[col]) for col in cols] for _, row in df.iterrows()]
    widths = []
    for i, col in enumerate(cols):
        widths.append(max(len(str(col)), *(len(row[i]) for row in rows)))

    def render_row(values: List[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(values)) + " |"

    lines = [render_row([str(c) for c in cols])]
    lines.append("| " + " | ".join("-" * width for width in widths) + " |")
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)


def dataset_specs(args: argparse.Namespace) -> List[DatasetSpec]:
    return [
        DatasetSpec(
            name="MAT",
            path=PROJECT_ROOT / "outputs_reproduced" / "features" / "eeg_features.csv",
            kind="csv",
            baseline_seconds=args.mat_baseline_seconds,
            timing_status="timed_start_sec_end_sec",
        ),
        DatasetSpec(
            name="STEW",
            path=PROJECT_ROOT / "results" / "multi_dataset" / "stew_features.parquet",
            kind="parquet",
            baseline_fraction=args.stew_baseline_fraction,
            timing_status="untimed_cached_parquet_no_duration_curve",
        ),
    ]


def run_baseline_comparison(
    datasets: Dict[str, Tuple[pd.DataFrame, DatasetSpec, List[str]]],
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[Tuple[str, str, str], Dict[str, float]]]:
    metrics_rows = []
    predictions = []
    fold_details = []
    coefficients = []
    fold_c_map: Dict[Tuple[str, str, str], Dict[str, float]] = {}

    for dataset, (df, spec, feature_cols) in datasets.items():
        calib_mask, eval_mask, split_desc = build_calibration_split(df, spec)
        for calibration in CALIBRATION_MODES:
            for model in MODEL_NAMES:
                print(f"[baseline] {dataset} {calibration} {model}")
                pred, folds, coefs = nested_loso_predictions(
                    df=df,
                    feature_cols=feature_cols,
                    calib_mask=calib_mask,
                    eval_mask=eval_mask,
                    calibration_mode=calibration,
                    model_name=model,
                    c_grid=DEFAULT_C_GRID,
                    inner_splits=args.inner_splits,
                )
                pred["dataset"] = dataset
                folds["dataset"] = dataset
                coefs["dataset"] = dataset
                metrics_rows.append(summarize_predictions(pred, dataset, split_desc, len(feature_cols)))
                predictions.append(pred)
                fold_details.append(folds)
                coefficients.append(coefs)
                fold_c_map[(dataset, calibration, model)] = {
                    str(row.subject_id): float(row.best_c) for row in folds.itertuples(index=False)
                }

    metrics = pd.DataFrame(metrics_rows)
    pred_all = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    folds_all = pd.concat(fold_details, ignore_index=True) if fold_details else pd.DataFrame()
    coefs_all = pd.concat(coefficients, ignore_index=True) if coefficients else pd.DataFrame()
    metrics.to_csv(RESULTS_DIR / "baseline_calibration_metrics.csv", index=False)
    pred_all.to_csv(RESULTS_DIR / "baseline_calibration_predictions.csv", index=False)
    folds_all.to_csv(RESULTS_DIR / "baseline_calibration_fold_details.csv", index=False)
    coefs_all.to_csv(RESULTS_DIR / "baseline_calibration_fold_coefficients.csv", index=False)
    return metrics, pred_all, folds_all, fold_c_map


def run_duration_curve(
    datasets: Dict[str, Tuple[pd.DataFrame, DatasetSpec, List[str]]],
    fold_c_map: Dict[Tuple[str, str, str], Dict[str, float]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows = []
    for dataset, (df, spec, feature_cols) in datasets.items():
        has_timing = {"start_sec", "end_sec"}.issubset(df.columns)
        if not has_timing:
            rows.append(
                {
                    "dataset": dataset,
                    "model": "",
                    "calibration": "zscore",
                    "baseline_seconds": np.nan,
                    "window_auc": np.nan,
                    "subject_auc_mean": np.nan,
                    "n_predictions": 0,
                    "n_subjects": int(df["subject_id"].nunique()),
                    "n_valid_subject_aucs": 0,
                    "status": "not_run_missing_timing_metadata",
                }
            )
            continue

        rest = df[df["label"].eq(0)]
        max_seconds = int(np.floor(float(rest["end_sec"].max())))
        durations = list(range(1, max_seconds + 1, args.duration_step))
        for seconds in durations:
            calib_mask, eval_mask, _ = build_calibration_split(df, spec, baseline_seconds=float(seconds))
            valid_subjects = 0
            subjects = df["subject_id"].astype(str).to_numpy()
            labels = df["label"].to_numpy(dtype=int)
            for subject in np.unique(subjects):
                y_eval = labels[(subjects == subject) & eval_mask]
                if np.unique(y_eval).size == 2:
                    valid_subjects += 1
            if valid_subjects < int(df["subject_id"].nunique()):
                rows.append(
                    {
                        "dataset": dataset,
                        "model": "logistic_l2",
                        "calibration": "zscore",
                        "baseline_seconds": seconds,
                        "window_auc": np.nan,
                        "subject_auc_mean": np.nan,
                        "n_predictions": 0,
                        "n_subjects": int(df["subject_id"].nunique()),
                        "n_valid_subject_aucs": valid_subjects,
                        "status": "not_run_insufficient_post_calibration_rest",
                    }
                )
                continue
            for model in DURATION_MODEL_NAMES:
                print(f"[duration] {dataset} zscore {model} {seconds}s")
                pred, _, _ = nested_loso_predictions(
                    df=df,
                    feature_cols=feature_cols,
                    calib_mask=calib_mask,
                    eval_mask=eval_mask,
                    calibration_mode="zscore",
                    model_name=model,
                    c_grid=DEFAULT_C_GRID,
                    inner_splits=args.inner_splits,
                    fold_c_overrides=fold_c_map[(dataset, "zscore", model)],
                )
                pred["dataset"] = dataset
                summary = summarize_predictions(pred, dataset, f"timed_first_{seconds}s", len(feature_cols))
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "calibration": "zscore",
                        "baseline_seconds": seconds,
                        "window_auc": summary["window_auc"],
                        "subject_auc_mean": summary["subject_auc_mean"],
                        "n_predictions": summary["n_predictions"],
                        "n_subjects": summary["n_subjects"],
                        "n_valid_subject_aucs": summary["n_subject_auc"],
                        "status": "ok" if summary["n_subject_auc"] == df["subject_id"].nunique() else "partial_or_invalid_folds",
                    }
                )

    curve = pd.DataFrame(rows)
    curve.to_csv(RESULTS_DIR / "calibration_duration_curve.csv", index=False)
    return curve


def run_effect_direction(
    datasets: Dict[str, Tuple[pd.DataFrame, DatasetSpec, List[str]]],
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    coef_frames = []
    correlation_rows = []
    final_coefs: Dict[Tuple[str, str, str], pd.DataFrame] = {}

    for dataset, (df, spec, feature_cols) in datasets.items():
        calib_mask, eval_mask, split_desc = build_calibration_split(df, spec)
        for calibration in CALIBRATION_MODES:
            for model in MODEL_NAMES:
                print(f"[direction] {dataset} {calibration} {model}")
                coefs, best_c, inner_auc = final_direction_coefficients(
                    df=df,
                    feature_cols=feature_cols,
                    calib_mask=calib_mask,
                    eval_mask=eval_mask,
                    calibration_mode=calibration,
                    model_name=model,
                    c_grid=DEFAULT_C_GRID,
                    inner_splits=args.inner_splits,
                )
                coefs["dataset"] = dataset
                coefs["split_description"] = split_desc
                coefs["best_c"] = best_c
                coefs["inner_cv_auc"] = inner_auc
                coef_frames.append(coefs)
                final_coefs[(dataset, calibration, model)] = coefs

    for calibration in CALIBRATION_MODES:
        for model in MODEL_NAMES:
            mat = final_coefs[("MAT", calibration, model)]
            stew = final_coefs[("STEW", calibration, model)]
            r_value, p_value, n_features = effect_direction_correlation(mat, stew)
            correlation_rows.append(
                {
                    "calibration": calibration,
                    "model": model,
                    "pearson_r": r_value,
                    "p_value": p_value,
                    "n_features": n_features,
                }
            )

    coef_all = pd.concat(coef_frames, ignore_index=True)
    corrs = pd.DataFrame(correlation_rows)
    coef_all.to_csv(RESULTS_DIR / "effect_direction_coefficients.csv", index=False)
    corrs.to_csv(RESULTS_DIR / "effect_direction_correlations.csv", index=False)
    return coef_all, corrs


def select_negative_control_targets(metrics: pd.DataFrame) -> pd.DataFrame:
    targets = []
    for dataset, group in metrics.groupby("dataset"):
        valid = group[np.isfinite(group["window_auc"])]
        if valid.empty:
            continue
        best = valid.sort_values("window_auc", ascending=False).iloc[0]
        targets.append(best)
    return pd.DataFrame(targets)


def run_negative_controls(
    datasets: Dict[str, Tuple[pd.DataFrame, DatasetSpec, List[str]]],
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(args.random_state)
    target_rows = select_negative_control_targets(metrics)
    perm_rows = []
    summary_rows = []

    for target in target_rows.itertuples(index=False):
        dataset = str(target.dataset)
        calibration = str(target.calibration)
        model = str(target.model)
        observed_auc = float(target.window_auc)
        observed = predictions[
            predictions["dataset"].eq(dataset)
            & predictions["calibration"].eq(calibration)
            & predictions["model"].eq(model)
        ].copy()
        if observed.empty or observed["y_true"].nunique() < 2:
            continue

        print(f"[negative] {dataset} {calibration} {model} permutations={args.permutations}")
        for i in range(args.permutations):
            y_perm = observed["y_true"].to_numpy(dtype=int).copy()
            subjects = observed["subject_id"].astype(str).to_numpy()
            for subject in np.unique(subjects):
                idx = np.where(subjects == subject)[0]
                y_perm[idx] = rng.permutation(y_perm[idx])
            if np.unique(y_perm).size < 2:
                perm_auc = np.nan
            else:
                perm_auc = float(roc_auc_score(y_perm, observed["score"].to_numpy(dtype=float)))
            perm_rows.append(
                {
                    "dataset": dataset,
                    "calibration": calibration,
                    "model": model,
                    "permutation": i,
                    "permutation_auc": perm_auc,
                    "observed_auc": observed_auc,
                    "note": "within_subject_labels_permuted_on_nested_out_of_fold_scores",
                }
            )
            if (i + 1) % 100 == 0:
                print(f"  completed {i + 1}/{args.permutations}")

        this = pd.DataFrame([r for r in perm_rows if r["dataset"] == dataset and r["calibration"] == calibration and r["model"] == model])
        valid = this["permutation_auc"].dropna().to_numpy(dtype=float)
        p_value = (1.0 + float(np.sum(valid >= observed_auc))) / (1.0 + float(valid.size)) if valid.size else np.nan
        summary_rows.append(
            {
                "dataset": dataset,
                "calibration": calibration,
                "model": model,
                "observed_auc": observed_auc,
                "n_permutations_requested": int(args.permutations),
                "n_permutations_valid": int(valid.size),
                "permutation_auc_mean": float(np.mean(valid)) if valid.size else np.nan,
                "permutation_auc_sd": float(np.std(valid, ddof=1)) if valid.size > 1 else np.nan,
                "permutation_p_value": p_value,
                "note": "Labels were permuted within subject on nested out-of-fold scores; models were not retrained for each null draw.",
            }
        )

    perm = pd.DataFrame(perm_rows)
    summary = pd.DataFrame(summary_rows)
    perm.to_csv(RESULTS_DIR / "negative_control_permutations.csv", index=False)
    summary.to_csv(RESULTS_DIR / "negative_control_summary.csv", index=False)
    return perm, summary


def plot_rocs(predictions: pd.DataFrame) -> None:
    if predictions.empty:
        return
    for dataset, df_dataset in predictions.groupby("dataset"):
        fig, ax = plt.subplots(figsize=(8, 6))
        for (calibration, model), group in df_dataset.groupby(["calibration", "model"]):
            if group["y_true"].nunique() < 2:
                continue
            roc = roc_points(group)
            auc_value = roc_auc_score(group["y_true"], group["score"])
            ax.plot(roc["fpr"], roc["tpr"], linewidth=1.8, label=f"{calibration}/{model} AUC={auc_value:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1)
        ax.set_title(f"{dataset} LOSO ROC Curves")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"roc_curves_{dataset.lower()}.png", dpi=180)
        plt.close(fig)


def plot_duration(curve: pd.DataFrame) -> None:
    valid = curve[(curve["status"] == "ok") & np.isfinite(curve["window_auc"])]
    if valid.empty:
        return
    for dataset, df_dataset in valid.groupby("dataset"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for model, group in df_dataset.groupby("model"):
            group = group.sort_values("baseline_seconds")
            ax.plot(group["baseline_seconds"], group["window_auc"], marker="o", markersize=2.5, linewidth=1.4, label=model)
        ax.set_title(f"{dataset} Baseline Calibration-Duration Curve")
        ax.set_xlabel("Baseline seconds used for calibration")
        ax.set_ylabel("LOSO window ROC-AUC")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"calibration_duration_curve_{dataset.lower()}.png", dpi=180)
        plt.close(fig)


def plot_negative_controls(permutations: pd.DataFrame) -> None:
    if permutations.empty:
        return
    for (dataset, calibration, model), group in permutations.groupby(["dataset", "calibration", "model"]):
        valid = group["permutation_auc"].dropna()
        if valid.empty:
            continue
        observed = float(group["observed_auc"].iloc[0])
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(valid, bins=35, color="#6b8fb3", alpha=0.85, edgecolor="white")
        ax.axvline(observed, color="#b33b2e", linewidth=2, label=f"observed AUC={observed:.3f}")
        ax.set_title(f"{dataset} Negative Control: {calibration}/{model}")
        ax.set_xlabel("Permutation ROC-AUC")
        ax.set_ylabel("Count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"negative_control_{dataset.lower()}_{calibration}_{model}.png", dpi=180)
        plt.close(fig)


def plateau_rows(curve: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = curve[(curve["status"] == "ok") & np.isfinite(curve["window_auc"])]
    for (dataset, model), group in valid.groupby(["dataset", "model"]):
        group = group.sort_values("baseline_seconds")
        max_auc = float(group["window_auc"].max())
        threshold = max_auc - 0.01
        plateau = group[group["window_auc"] >= threshold].iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "max_auc": max_auc,
                "plateau_definition": "first_duration_within_0.01_auc_of_max_valid_auc",
                "plateau_seconds": float(plateau["baseline_seconds"]),
                "plateau_auc": float(plateau["window_auc"]),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "calibration_duration_plateaus.csv", index=False)
    return out


def write_summary(
    metrics: pd.DataFrame,
    curve: pd.DataFrame,
    plateaus: pd.DataFrame,
    correlations: pd.DataFrame,
    negative_summary: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines: List[str] = []
    lines.append("# Theory Validation Results")
    lines.append("")
    lines.append("Generated by `scripts/run_theory_validation.py`. No manuscript text was modified by this run.")
    lines.append("")
    lines.append("## Execution Contract")
    lines.append("")
    lines.append("- Outer validation: leave-one-subject-out.")
    if len(DEFAULT_C_GRID) == 1:
        lines.append(f"- Hyperparameter protocol: fixed preregistered C={DEFAULT_C_GRID[0]}; no label-dependent hyperparameter search was performed.")
    else:
        lines.append(f"- Inner model selection: grouped {args.inner_splits}-fold CV on training subjects only with C grid {DEFAULT_C_GRID}.")
    lines.append("- Feature set: 200 per-channel features from the requested 8-channel intersection: " + ", ".join(COMMON_8_CHANNELS) + ".")
    lines.append("- MAT calibration split: first 60 s of resting windows used for calibration; those windows are excluded from scoring.")
    lines.append("- STEW calibration split: first 50% of each subject's cached rest rows by row order used for calibration because the parquet has no timing metadata.")
    lines.append("- Duration curves: outer LOSO using the same fixed-C protocol; only logistic_l2 is used for the duration curve to isolate baseline length.")
    lines.append("- Negative controls: 1000 within-subject label permutations on nested out-of-fold scores for the best observed MAT and STEW configurations.")
    lines.append("")
    lines.append("## Baseline-Calibration Comparison")
    lines.append("")
    lines.append(markdown_table(metrics.sort_values(["dataset", "window_auc"], ascending=[True, False]), floatfmt=".6f"))
    lines.append("")
    lines.append("## Calibration-Duration Curve")
    lines.append("")
    if plateaus.empty:
        lines.append("No valid full-LOSO duration plateau could be estimated.")
    else:
        lines.append(markdown_table(plateaus, floatfmt=".6f"))
    skipped = curve[curve["status"].ne("ok")]
    if not skipped.empty:
        lines.append("")
        lines.append("Duration-curve failures or partial rows:")
        lines.append(markdown_table(skipped[["dataset", "model", "baseline_seconds", "status", "n_valid_subject_aucs"]].head(20), floatfmt=".6f"))
    lines.append("")
    lines.append("## Effect-Direction Analysis")
    lines.append("")
    lines.append(markdown_table(correlations.sort_values(["calibration", "model"]), floatfmt=".6g"))
    lines.append("")
    lines.append("## Negative Controls")
    lines.append("")
    if negative_summary.empty:
        lines.append("No valid negative-control permutations were produced.")
    else:
        lines.append(markdown_table(negative_summary, floatfmt=".6f"))
    lines.append("")
    lines.append("## Files Written")
    lines.append("")
    for path in sorted(RESULTS_DIR.glob("*.csv")):
        lines.append(f"- `{path.relative_to(PROJECT_ROOT)}`")
    for path in sorted(FIGURES_DIR.glob("*.png")):
        lines.append(f"- `{path.relative_to(PROJECT_ROOT)}`")
    lines.append("")
    lines.append("## Non-Hallucination Notes")
    lines.append("")
    lines.append("- STEW calibration-duration analysis was not run because the cached STEW parquet contains no timing columns.")
    lines.append("- Any poor AUCs in the tables above are retained as failures rather than filtered out.")
    lines.append("- The permutation p-values are empirical p = (1 + count(null AUC >= observed AUC)) / (1 + valid permutations).")
    lines.append("- Negative-control permutations shuffle labels on already nested out-of-fold scores; they test score-label association, not retrained-model instability.")

    (PROJECT_ROOT / "THEORY_VALIDATION_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_environment_manifest() -> None:
    packages = ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "mne", "pyarrow"]
    rows = []
    for package in packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "unknown")
        except Exception as exc:
            version = f"import_failed: {exc}"
        rows.append({"package": package, "version": version})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "environment_versions.csv", index=False)


def main() -> None:
    args = parse_args()
    setup_dirs()
    write_environment_manifest()

    if args.summary_only:
        metrics = pd.read_csv(RESULTS_DIR / "baseline_calibration_metrics.csv")
        curve = pd.read_csv(RESULTS_DIR / "calibration_duration_curve.csv")
        plateaus = pd.read_csv(RESULTS_DIR / "calibration_duration_plateaus.csv")
        correlations = pd.read_csv(RESULTS_DIR / "effect_direction_correlations.csv")
        negative_summary = pd.read_csv(RESULTS_DIR / "negative_control_summary.csv")
        write_summary(metrics, curve, plateaus, correlations, negative_summary, args)
        print("[done] regenerated THEORY_VALIDATION_RESULTS.md from existing CSV outputs")
        return

    datasets: Dict[str, Tuple[pd.DataFrame, DatasetSpec, List[str]]] = {}
    for spec in dataset_specs(args):
        print(f"[load] {spec.name}: {spec.path}")
        df = load_feature_table(spec)
        feature_cols = select_common_8_features(df)
        datasets[spec.name] = (df, spec, feature_cols)
        pd.DataFrame(
            [
                {
                    "dataset": spec.name,
                    "path": str(spec.path.relative_to(PROJECT_ROOT)),
                    "n_rows": len(df),
                    "n_subjects": df["subject_id"].nunique(),
                    "n_features_8ch": len(feature_cols),
                    "timing_status": spec.timing_status,
                }
            ]
        ).to_csv(RESULTS_DIR / f"dataset_manifest_{spec.name.lower()}.csv", index=False)

    metrics, predictions, _, fold_c_map = run_baseline_comparison(datasets, args)
    curve = run_duration_curve(datasets, fold_c_map, args)
    _, correlations = run_effect_direction(datasets, args)
    permutations, negative_summary = run_negative_controls(datasets, metrics, predictions, args)
    plateaus = plateau_rows(curve)

    plot_rocs(predictions)
    plot_duration(curve)
    plot_negative_controls(permutations)
    write_summary(metrics, curve, plateaus, correlations, negative_summary, args)
    print(f"[done] wrote {RESULTS_DIR.relative_to(PROJECT_ROOT)} and THEORY_VALIDATION_RESULTS.md")


if __name__ == "__main__":
    main()
