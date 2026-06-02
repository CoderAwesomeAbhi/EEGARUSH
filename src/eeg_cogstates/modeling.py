from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


METADATA_COLUMNS = {
    "subject_id",
    "condition",
    "label",
    "file",
    "window_index",
    "start_sec",
    "end_sec",
}


def get_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    y = df["label"].astype(int).to_numpy()
    groups = df["subject_id"].astype(str).to_numpy()

    feature_cols = [
        c
        for c in df.columns
        if c not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    ]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    return X, y, groups, feature_cols


def make_models(random_state: int = 42) -> Dict[str, object]:
    models: Dict[str, object] = {
        "logistic_regression": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "svm_rbf": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
    }

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
        )
    except Exception:
        pass

    return models


def make_pipeline(model: object) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def positive_scores(estimator: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X)
        min_s, max_s = float(np.min(scores)), float(np.max(scores))
        return (scores - min_s) / (max_s - min_s + 1e-12)
    return estimator.predict(X).astype(float)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: Optional[np.ndarray] = None) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_score is not None:
        y_score = np.asarray(y_score, dtype=float)

    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0

    out = {
        "n": int(len(y_true)),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity_recall": float(sensitivity),
        "specificity": float(specificity),
        "PPV_precision": float(ppv),
        "NPV": float(npv),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            out["roc_auc"] = np.nan
        try:
            out["pr_auc_average_precision"] = float(average_precision_score(y_true, y_score))
        except Exception:
            out["pr_auc_average_precision"] = np.nan
    else:
        out["roc_auc"] = np.nan
        out["pr_auc_average_precision"] = np.nan

    return out


def subject_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 500,
    random_state: int = 42,
) -> Dict[str, float]:
    rng = np.random.default_rng(random_state)
    unique_groups = np.unique(groups)
    metric_samples: Dict[str, List[float]] = {}

    for _ in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idxs = np.concatenate([np.where(groups == g)[0] for g in sampled_groups])
        if len(np.unique(y_true[idxs])) < 2:
            continue
        m = compute_metrics(y_true[idxs], y_pred[idxs], y_score[idxs])
        for k, v in m.items():
            if k in {"n", "TP", "FP", "FN", "TN"}:
                continue
            if np.isfinite(v):
                metric_samples.setdefault(k, []).append(float(v))

    ci = {}
    for metric, values in metric_samples.items():
        arr = np.asarray(values, dtype=float)
        if arr.size:
            ci[f"{metric}_ci_low"] = float(np.percentile(arr, 2.5))
            ci[f"{metric}_ci_high"] = float(np.percentile(arr, 97.5))
    return ci


def fit_predict_one(
    model_name: str,
    base_model: object,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
) -> Tuple[Pipeline, np.ndarray, np.ndarray]:
    pipe = make_pipeline(clone(base_model))
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_score = positive_scores(pipe, X_test)
    return pipe, y_pred, y_score


def run_holdout(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    models: Dict[str, object],
    output_dir: Path,
    n_boot: int = 500,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Pipeline]]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_test = groups[test_idx]

    metric_rows = []
    pred_rows = []
    fitted = {}

    for model_name, model in models.items():
        estimator, y_pred, y_score = fit_predict_one(model_name, model, X_train, y_train, X_test)
        fitted[model_name] = estimator

        metrics = compute_metrics(y_test, y_pred, y_score)
        metrics.update(subject_bootstrap_ci(y_test, y_pred, y_score, groups_test, n_boot=n_boot, random_state=random_state))
        metrics.update({"model": model_name, "evaluation": "subject_group_holdout"})
        metric_rows.append(metrics)

        for idx, true_label, pred_label, score, subject in zip(test_idx, y_test, y_pred, y_score, groups_test):
            pred_rows.append(
                {
                    "row_index": int(idx),
                    "subject_id": subject,
                    "true_label": int(true_label),
                    "pred_label": int(pred_label),
                    "score_workload": float(score),
                    "model": model_name,
                    "evaluation": "subject_group_holdout",
                }
            )

    metrics_df = pd.DataFrame(metric_rows).sort_values(["roc_auc", "f1", "accuracy"], ascending=False)
    preds_df = pd.DataFrame(pred_rows)
    metrics_df.to_csv(output_dir / "metrics_holdout.csv", index=False)
    preds_df.to_csv(output_dir / "predictions_holdout.csv", index=False)
    return metrics_df, preds_df, fitted


def run_loso_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    models: Dict[str, object],
    output_dir: Path,
    n_boot: int = 500,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logo = LeaveOneGroupOut()
    pred_rows = []

    for model_name, model in models.items():
        for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups)):
            estimator, y_pred, y_score = fit_predict_one(
                model_name,
                model,
                X.iloc[train_idx],
                y[train_idx],
                X.iloc[test_idx],
            )
            for idx, true_label, pred_label, score, subject in zip(test_idx, y[test_idx], y_pred, y_score, groups[test_idx]):
                pred_rows.append(
                    {
                        "row_index": int(idx),
                        "subject_id": subject,
                        "true_label": int(true_label),
                        "pred_label": int(pred_label),
                        "score_workload": float(score),
                        "model": model_name,
                        "fold": int(fold),
                        "evaluation": "leave_one_subject_out",
                    }
                )

    preds_df = pd.DataFrame(pred_rows)
    metric_rows = []
    for model_name, group_df in preds_df.groupby("model"):
        y_true = group_df["true_label"].to_numpy()
        y_pred = group_df["pred_label"].to_numpy()
        y_score = group_df["score_workload"].to_numpy()
        fold_groups = group_df["subject_id"].to_numpy()
        metrics = compute_metrics(y_true, y_pred, y_score)
        metrics.update(subject_bootstrap_ci(y_true, y_pred, y_score, fold_groups, n_boot=n_boot, random_state=random_state))
        metrics.update({"model": model_name, "evaluation": "leave_one_subject_out"})
        metric_rows.append(metrics)

    metrics_df = pd.DataFrame(metric_rows).sort_values(["roc_auc", "f1", "accuracy"], ascending=False)
    metrics_df.to_csv(output_dir / "metrics_loso.csv", index=False)
    preds_df.to_csv(output_dir / "predictions_loso.csv", index=False)
    return metrics_df, preds_df


def save_feature_importance(
    fitted: Dict[str, Pipeline],
    feature_names: List[str],
    output_dir: Path,
) -> None:
    for model_name, pipe in fitted.items():
        model = pipe.named_steps["model"]
        importances = None

        if hasattr(model, "feature_importances_"):
            importances = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            importances = np.ravel(np.abs(model.coef_)).astype(float)

        if importances is None or len(importances) != len(feature_names):
            continue

        out = pd.DataFrame({"feature": feature_names, "importance": importances})
        out = out.sort_values("importance", ascending=False)
        out.to_csv(output_dir / f"feature_importance_{model_name}.csv", index=False)


def train_and_evaluate(
    features_csv: str | Path,
    output_dir: str | Path,
    run_loso: bool = True,
    n_boot: int = 500,
    random_state: int = 42,
) -> Dict[str, Path]:
    features_csv = Path(features_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(features_csv)
    X, y, groups, feature_names = get_feature_matrix(df)
    models = make_models(random_state=random_state)

    holdout_metrics, holdout_preds, fitted = run_holdout(
        X, y, groups, models, output_dir=output_dir, n_boot=n_boot, random_state=random_state
    )

    if run_loso:
        run_loso_cv(X, y, groups, models, output_dir=output_dir, n_boot=n_boot, random_state=random_state)

    save_feature_importance(fitted, feature_names, output_dir)

    best_model_name = str(holdout_metrics.iloc[0]["model"])
    best_model = fitted[best_model_name]
    joblib.dump(
        {
            "model_name": best_model_name,
            "pipeline": best_model,
            "feature_names": feature_names,
        },
        output_dir / "best_model.joblib",
    )

    return {
        "holdout_metrics": output_dir / "metrics_holdout.csv",
        "holdout_predictions": output_dir / "predictions_holdout.csv",
        "best_model": output_dir / "best_model.joblib",
    }
