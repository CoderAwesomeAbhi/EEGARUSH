from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)


CB_PALETTE = {
    "svm_rbf": "#0072B2",
    "logistic_regression": "#E69F00",
    "random_forest": "#009E73",
    "gradient_boosting": "#F0E442",
    "xgboost": "#D55E00",
    "SNWA_K8": "#56B4E9",
    "rest": "#0072B2",
    "workload": "#E69F00",
}


def _savefig(path: Path, dpi: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def _panel_label(ax, label: str, x: float = -0.1, y: float = 1.05) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="right")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrices(predictions_csv: str | Path, output_dir: str | Path) -> None:
    predictions_csv = Path(predictions_csv)
    output_dir = Path(output_dir)
    if not predictions_csv.exists():
        return

    preds = pd.read_csv(predictions_csv)
    for model_name, g in preds.groupby("model"):
        cm = confusion_matrix(g["true_label"], g["pred_label"], labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["rest", "workload"])
        disp.plot(values_format="d")
        plt.title(f"Confusion Matrix: {model_name}")
        _savefig(output_dir / f"confusion_matrix_{model_name}.png")


def plot_roc_curves(predictions_csv: str | Path, output_path: str | Path) -> None:
    predictions_csv = Path(predictions_csv)
    if not predictions_csv.exists():
        return
    preds = pd.read_csv(predictions_csv)

    plt.figure()
    for model_name, g in preds.groupby("model"):
        if g["true_label"].nunique() < 2:
            continue
        color = CB_PALETTE.get(model_name, None)
        RocCurveDisplay.from_predictions(
            g["true_label"],
            g["score_workload"],
            name=model_name,
            color=color,
            ax=plt.gca(),
        )
    plt.title("ROC Curves: Subject-wise Holdout")
    _savefig(Path(output_path))


def plot_precision_recall_curves(predictions_csv: str | Path, output_path: str | Path) -> None:
    predictions_csv = Path(predictions_csv)
    if not predictions_csv.exists():
        return
    preds = pd.read_csv(predictions_csv)

    plt.figure()
    for model_name, g in preds.groupby("model"):
        if g["true_label"].nunique() < 2:
            continue
        PrecisionRecallDisplay.from_predictions(
            g["true_label"],
            g["score_workload"],
            name=model_name,
            ax=plt.gca(),
        )
    plt.title("Precision-Recall Curves: Subject-wise Holdout")
    _savefig(Path(output_path))


def plot_global_bandpower(features_csv: str | Path, output_path: str | Path) -> None:
    features_csv = Path(features_csv)
    if not features_csv.exists():
        return
    df = pd.read_csv(features_csv)
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    cols = [f"global_band_rel_mean_{band}" for band in bands if f"global_band_rel_mean_{band}" in df.columns]
    if not cols:
        return

    subject_condition = df.groupby(["subject_id", "condition"], as_index=False)[cols].mean(numeric_only=True)

    plot_rows = []
    for condition, g in subject_condition.groupby("condition"):
        for col in cols:
            vals = g[col].dropna().to_numpy()
            band = col.replace("global_band_rel_mean_", "")
            if len(vals) == 0:
                continue
            se = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            plot_rows.append(
                {
                    "condition": condition,
                    "band": band,
                    "mean": float(np.mean(vals)),
                    "ci95": float(1.96 * se),
                }
            )
    plot_df = pd.DataFrame(plot_rows)
    if plot_df.empty:
        return

    x = np.arange(len(bands))
    width = 0.35

    plt.figure(figsize=(9, 5))
    for offset, condition in zip([-width / 2, width / 2], ["rest", "workload"]):
        g = plot_df[plot_df["condition"] == condition].set_index("band").reindex(bands)
        color = CB_PALETTE.get(condition, None)
        plt.bar(x + offset, g["mean"], width=width, yerr=g["ci95"], capsize=3,
                label=condition, color=color)
    plt.xticks(x, bands)
    plt.ylabel("Relative bandpower")
    plt.title("Global Relative EEG Bandpower: Rest vs Workload")
    plt.legend()
    _savefig(Path(output_path))


def plot_top_feature_importances(model_dir: str | Path, output_dir: str | Path, top_n: int = 25) -> None:
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    for path in model_dir.glob("feature_importance_*.csv"):
        df = pd.read_csv(path).head(top_n).iloc[::-1]
        if df.empty:
            continue
        model_name = path.stem.replace("feature_importance_", "")
        plt.figure(figsize=(10, max(5, 0.25 * len(df))))
        plt.barh(df["feature"], df["importance"])
        plt.xlabel("Importance")
        plt.title(f"Top Feature Importances: {model_name}")
        _savefig(output_dir / f"top_feature_importance_{model_name}.png")


def plot_statistics_volcano(stats_csv: str | Path, output_path: str | Path) -> None:
    stats_csv = Path(stats_csv)
    if not stats_csv.exists():
        return
    df = pd.read_csv(stats_csv)
    required = {"cohens_dz", "paired_t_q_fdr"}
    if not required.issubset(df.columns):
        return

    q = df["paired_t_q_fdr"].replace(0, np.nan)
    y = -np.log10(q)
    x = df["cohens_dz"]

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=12, alpha=0.7)
    plt.axhline(-np.log10(0.05), linestyle="--")
    plt.xlabel("Cohen's dz effect size")
    plt.ylabel("-log10(FDR-adjusted p-value)")
    plt.title("Feature-Level Statistics: Workload vs Rest")
    _savefig(Path(output_path))


def plot_subject_scatter(
    predictions_csv: str | Path,
    output_path: str | Path,
    model_1: str = "svm_rbf",
    model_2: str = "SNWA_K8",
) -> None:
    predictions_csv = Path(predictions_csv)
    output_path = Path(output_path)
    if not predictions_csv.exists():
        return
    df = pd.read_csv(predictions_csv)
    sub_auc_1 = df[df["model"] == model_1].groupby("subject_id").apply(
        lambda g: roc_auc_score(g["true_label"], g["score_workload"]) if g["true_label"].nunique() > 1 else np.nan
    )
    sub_auc_2 = df[df["model"] == model_2].groupby("subject_id").apply(
        lambda g: roc_auc_score(g["true_label"], g["score_workload"]) if g["true_label"].nunique() > 1 else np.nan
    )
    common = sub_auc_1.dropna().align(sub_auc_2.dropna(), join="inner")
    if common[0].empty:
        return
    x, y = common[0].to_numpy(), common[1].to_numpy()
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.6, c=CB_PALETTE.get("svm_rbf", "#0072B2"))
    lims = [min(x.min(), y.min()) - 0.05, max(x.max(), y.max()) + 0.05]
    plt.plot(lims, lims, "k--", alpha=0.3, label="Equality")
    plt.xlabel(f"{model_1} per-subject AUC")
    plt.ylabel(f"{model_2} per-subject AUC")
    plt.title("Per-Subject AUC Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_correlation_heatmap(
    features_csv: str | Path,
    output_path: str | Path,
    top_n: int = 20,
) -> None:
    features_csv = Path(features_csv)
    output_path = Path(output_path)
    if not features_csv.exists():
        return
    df = pd.read_csv(features_csv)
    META = {"subject_id", "condition", "label", "file", "window_index", "start_sec", "end_sec"}
    feat_cols = [c for c in df.columns if c not in META and pd.api.types.is_numeric_dtype(df[c])]
    if len(feat_cols) < top_n:
        return
    corr = df[feat_cols].corr(method="spearman")
    top_features = (
        df[feat_cols].var().sort_values(ascending=False).head(top_n).index.tolist()
    )
    sub_corr = corr.loc[top_features, top_features]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sub_corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(top_features)))
    ax.set_yticks(range(len(top_features)))
    ax.set_xticklabels(top_features, rotation=90, fontsize=6)
    ax.set_yticklabels(top_features, fontsize=6)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.title(f"Spearman Correlation: Top {top_n} Features by Variance")
    _savefig(output_path)


def plot_combined_roc(predictions_csv: str | Path, output_path: str | Path) -> None:
    predictions_csv = Path(predictions_csv)
    if not predictions_csv.exists():
        return
    preds = pd.read_csv(predictions_csv)

    plt.figure(figsize=(7, 6))
    for model_name, g in preds.groupby("model"):
        if g["true_label"].nunique() < 2:
            continue
        color = CB_PALETTE.get(model_name, "#333333")
        RocCurveDisplay.from_predictions(
            g["true_label"],
            g["score_workload"],
            name=model_name,
            color=color,
            ax=plt.gca(),
        )
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Chance")
    plt.title("Combined ROC Curves — All Models (LOSO)")
    _savefig(Path(output_path))


def make_all_figures(
    features_csv: str | Path,
    stats_csv: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    holdout_predictions = Path(model_dir) / "predictions_holdout.csv"
    loso_predictions = Path(model_dir) / "predictions_loso.csv"
    plot_confusion_matrices(holdout_predictions, output_dir)
    plot_roc_curves(holdout_predictions, output_dir / "roc_curves_holdout.png")
    plot_precision_recall_curves(holdout_predictions, output_dir / "precision_recall_holdout.png")
    plot_global_bandpower(features_csv, output_dir / "global_relative_bandpower.png")
    plot_top_feature_importances(model_dir, output_dir)
    plot_statistics_volcano(stats_csv, output_dir / "statistics_volcano.png")
    if loso_predictions.exists():
        plot_subject_scatter(loso_predictions, output_dir / "subject_auc_scatter.png")
    plot_feature_correlation_heatmap(features_csv, output_dir / "feature_correlation_heatmap.png")
    if loso_predictions.exists():
        plot_combined_roc(loso_predictions, output_dir / "combined_roc_loso.png")
