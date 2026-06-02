from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


METADATA_COLUMNS = {
    "subject_id",
    "condition",
    "label",
    "file",
    "window_index",
    "start_sec",
    "end_sec",
}


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if valid.sum() == 0:
        return q

    p_valid = p[valid]
    order = np.argsort(p_valid)
    ranked = p_valid[order]
    m = len(ranked)

    adjusted = ranked * m / (np.arange(1, m + 1))
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    q_valid = np.empty_like(adjusted)
    q_valid[order] = adjusted
    q[valid] = q_valid
    return q


def numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for col in df.columns:
        if col in METADATA_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def paired_feature_tests(features_csv: str | Path, output_dir: str | Path) -> pd.DataFrame:
    features_csv = Path(features_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(features_csv)
    feature_cols = numeric_feature_columns(df)

    subject_condition = df.groupby(["subject_id", "condition"], as_index=False)[feature_cols].mean(numeric_only=True)

    rows = []
    for feature in feature_cols:
        pivot = subject_condition.pivot(index="subject_id", columns="condition", values=feature)
        if not {"rest", "workload"}.issubset(set(pivot.columns)):
            continue

        pair_df = pivot[["rest", "workload"]].dropna()
        if len(pair_df) < 5:
            continue

        rest = pair_df["rest"].to_numpy(dtype=float)
        workload = pair_df["workload"].to_numpy(dtype=float)
        diff = workload - rest

        if np.nanstd(diff) <= 1e-12:
            p_t = 1.0
            t_stat = 0.0
            cohens_dz = 0.0
        else:
            t_stat, p_t = ttest_rel(workload, rest, nan_policy="omit")
            cohens_dz = float(np.nanmean(diff) / (np.nanstd(diff, ddof=1) + 1e-12))

        try:
            if np.allclose(diff, 0):
                w_stat, p_w = 0.0, 1.0
            else:
                w_stat, p_w = wilcoxon(workload, rest, zero_method="wilcox")
        except Exception:
            w_stat, p_w = np.nan, np.nan

        rows.append(
            {
                "feature": feature,
                "n_subject_pairs": int(len(pair_df)),
                "rest_mean": float(np.nanmean(rest)),
                "workload_mean": float(np.nanmean(workload)),
                "mean_difference_workload_minus_rest": float(np.nanmean(diff)),
                "paired_t_statistic": float(t_stat),
                "paired_t_p": float(p_t),
                "wilcoxon_statistic": float(w_stat) if np.isfinite(w_stat) else np.nan,
                "wilcoxon_p": float(p_w) if np.isfinite(p_w) else np.nan,
                "cohens_dz": float(cohens_dz),
                "abs_cohens_dz": float(abs(cohens_dz)),
            }
        )

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No statistical tests could be run. Check the feature table.")

    results["paired_t_q_fdr"] = benjamini_hochberg(results["paired_t_p"])
    results["wilcoxon_q_fdr"] = benjamini_hochberg(results["wilcoxon_p"])
    results = results.sort_values(["paired_t_q_fdr", "abs_cohens_dz"], ascending=[True, False])

    results.to_csv(output_dir / "feature_stat_tests.csv", index=False)
    results.head(100).to_csv(output_dir / "top_significant_features.csv", index=False)
    return results
