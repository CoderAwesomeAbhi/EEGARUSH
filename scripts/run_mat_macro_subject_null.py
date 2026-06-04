#!/usr/bin/env python3
"""Run MAT macro subject-level full-pipeline null for the locked primary candidate."""

from __future__ import annotations

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
    COMMON_8_CHANNELS,
    expected_feature_names,
    apply_baseline_calibration,
    make_model,
    permute_labels_within_subject,
)


REBUILT_DIR = ROOT / "results" / "raw_rebuilt"
RNG = np.random.default_rng(20260603)
PRIMARY_MODEL = "logistic_l2"
PRIMARY_CALIBRATION = "mean_subtraction"


def no_gamma_feature_names() -> list[str]:
    return [name for name in expected_feature_names(COMMON_8_CHANNELS) if "_gamma" not in name]


def balanced_masks(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    rest = df["condition"].eq("rest").to_numpy()
    task = df["condition"].eq("arithmetic").to_numpy()
    start = df["start_sec"].to_numpy(dtype=float)
    end = df["end_sec"].to_numpy(dtype=float)
    calib = rest & (start >= 0.0) & (end <= 30.0)
    scored_rest = rest & (start >= 30.0) & (end <= 60.0)
    scored_task = task & (start >= 0.0) & (end <= 30.0)
    return calib, scored_rest | scored_task


def macro_subject_auc(y: Sequence[int], scores: Sequence[float], subjects: Sequence[str]) -> float:
    frame = pd.DataFrame({"y": y, "score": scores, "subject_id": subjects})
    aucs = []
    for _, group in frame.groupby("subject_id"):
        if group["y"].nunique() == 2:
            aucs.append(float(roc_auc_score(group["y"], group["score"])))
    return float(np.mean(aucs)) if aucs else float("nan")


def run_permutations(df: pd.DataFrame, feature_cols: Sequence[str], observed_auc: float, n_perm: int) -> pd.DataFrame:
    calib_mask, eval_mask = balanced_masks(df)
    eval_rows = np.where(eval_mask)[0]
    subjects_all = df["subject_id"].astype(str).to_numpy()
    eval_subjects = subjects_all[eval_rows]
    unique_subjects = np.array(sorted(np.unique(eval_subjects)))
    x_eval = apply_baseline_calibration(df, eval_mask, calib_mask, feature_cols, PRIMARY_CALIBRATION)
    out_path = REBUILT_DIR / "mat_macro_subject_full_pipeline_permutation.csv"
    rows = []
    start_time = time.time()
    for i in range(n_perm):
        y_perm = permute_labels_within_subject(df, RNG)
        y_eval = y_perm[eval_rows].astype(int)
        y_all: list[int] = []
        score_all: list[float] = []
        subj_all: list[str] = []
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
        rows.append(
            {
                "permutation_index": i,
                "macro_subject_mean_auc": macro_subject_auc(y_all, score_all, subj_all),
                "observed_macro_subject_mean_auc": observed_auc,
                "model": PRIMARY_MODEL,
                "calibration": PRIMARY_CALIBRATION,
                "elapsed_seconds_total": time.time() - start_time,
            }
        )
        pd.DataFrame(rows).to_csv(out_path, index=False)
    return pd.DataFrame(rows)


def write_report(null_df: pd.DataFrame, observed: float) -> tuple[float, float, float, float, float]:
    valid = null_df[np.isfinite(null_df["macro_subject_mean_auc"])]
    null_mean = float(valid["macro_subject_mean_auc"].mean())
    null_low = float(valid["macro_subject_mean_auc"].quantile(0.025))
    null_high = float(valid["macro_subject_mean_auc"].quantile(0.975))
    p_value = float((1 + np.sum(valid["macro_subject_mean_auc"].to_numpy() >= observed)) / (len(valid) + 1))
    runtime = float(null_df["elapsed_seconds_total"].max())
    lines = [
        "# MAT Macro Subject Null Results",
        "",
        f"- Configuration: corrected balanced raw MAT / no-gamma 184 / `{PRIMARY_CALIBRATION}` / `{PRIMARY_MODEL}`.",
        "- Statistic: macro subject-level mean ROC-AUC.",
        "- Label permutation: labels are permuted within subject before rerunning LOSO model fitting and evaluation; calibration masks, evaluation masks, imputation, scaling, and model fitting are recomputed within each permutation fold.",
        f"- Completed permutations: `{len(valid)}`.",
        f"- Observed macro subject-level mean ROC-AUC: `{observed:.6f}`.",
        f"- Null mean: `{null_mean:.6f}`.",
        f"- Null 95% interval: `[{null_low:.6f}, {null_high:.6f}]`.",
        f"- Empirical p-value: `{p_value:.6f}`.",
        f"- Runtime seconds: `{runtime:.2f}`.",
    ]
    (ROOT / "MAT_MACRO_SUBJECT_NULL_RESULTS.md").write_text("\n".join(lines) + "\n")
    fig_dir = REBUILT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(valid["macro_subject_mean_auc"], bins=30, color="#93a8ac", edgecolor="white")
    ax.axvline(observed, color="#a4161a", linewidth=2, label="observed")
    ax.set_xlabel("Macro subject-level mean ROC-AUC")
    ax.set_ylabel("Permutation count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "mat_macro_subject_full_pipeline_null.png", dpi=200)
    plt.close(fig)
    return null_mean, null_low, null_high, p_value, runtime


def main() -> int:
    metrics = pd.read_csv(REBUILT_DIR / "mat_subject_level_metrics.csv")
    row = metrics[metrics["model"].eq(PRIMARY_MODEL) & metrics["calibration"].eq(PRIMARY_CALIBRATION)].iloc[0]
    observed = float(row["macro_subject_mean_auc"])
    df = pd.read_parquet(REBUILT_DIR / "mat_no_gamma_features.parquet").reset_index(drop=True)
    feature_cols = no_gamma_feature_names()
    null_df = run_permutations(df, feature_cols, observed, n_perm=200)
    _, _, _, p_value, _ = write_report(null_df, observed)
    print("MAT_MACRO_SIGNAL_PASSED" if p_value <= 0.05 else "MAT_MACRO_SIGNAL_FAILED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
