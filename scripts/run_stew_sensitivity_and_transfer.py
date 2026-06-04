#!/usr/bin/env python3
"""STEW exploratory sensitivity + locked MAT->STEW transfer stress test.

Predeclared by:
  * STEW_EXPLORATORY_SENSITIVITY_PROTOCOL_BEFORE_MODELING.md
  * TRANSFER_FEATURE_COMPATIBILITY_AUDIT.md

Exploratory cross-task/cross-device sensitivity ONLY. STEW is non-comparable and
is never a strict replication or confirmation. No raw data is committed.

Pipeline:
  * STEW native 128 Hz; MAT raw resampled 500 -> 128 Hz (anti-aliased polyphase).
  * Locked 8 channels: F3,F4,F7,F8,O1,O2,T3(MAT)/T7(STEW),T4(MAT)/T8(STEW).
  * Locked no_gamma_184 features, 4 s / 50% windows.
  * Balanced segments: calib = first 30 s rest; scored-rest = next 30 s rest;
    scored-task = first 30 s workload. 14/14/14 windows per subject.
  * Model: L2 logistic regression (C=1.0, liblinear, balanced).
  * Primary unit: macro subject-level ROC-AUC.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / "stew_sensitivity" / "mplconfig"))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from scipy.signal import resample_poly
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from eeg_cogstates.features import extract_window_features
from eeg_cogstates.theory_validation import (
    COMMON_8_CHANNELS,
    apply_baseline_calibration,
    expected_feature_names,
    make_model,
    nested_loso_predictions,
)

# ---- locked constants ----
WINDOW_SECONDS = 4.0
OVERLAP = 0.5
STEP_SECONDS = WINDOW_SECONDS * (1.0 - OVERLAP)
TARGET_SFREQ = 128.0
SEG_SECONDS = 30.0
BOOTSTRAPS = 2000
N_PERM = 200
RNG = np.random.default_rng(20260604)

STEW_DIR = ROOT / "data" / "raw" / "stew" / "ieee_dataport_stew" / "STEW Dataset"
MAT_DIR = ROOT / "data" / "raw" / "eegmat"
OUT_DIR = ROOT / "results" / "stew_sensitivity"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Documented Emotiv EPOC channel order (columns of STEW .txt files).
STEW_EMOTIV_ORDER = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
                     "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
# Map locked COMMON_8 (MAT naming) -> STEW Emotiv column name.
COMMON8_TO_STEW = {"F3": "F3", "F4": "F4", "F7": "F7", "F8": "F8",
                   "O1": "O1", "O2": "O2", "T3": "T7", "T4": "T8"}
STEW_COL_IDX = [STEW_EMOTIV_ORDER.index(COMMON8_TO_STEW[c]) for c in COMMON_8_CHANNELS]

# Within-STEW (same device/unit system): full locked no-gamma 184 family.
FEATURE_COLS = [n for n in expected_feature_names(COMMON_8_CHANNELS) if "_gamma" not in n]
assert len(FEATURE_COLS) == 184, len(FEATURE_COLS)

# MAT->STEW TRANSPORT: ONLY features empirically proven invariant under the
# arbitrary device map x -> a*x + b (a>0). See TRANSFER_FEATURE_COMPATIBILITY_AUDIT.md.
# Verified at machine precision (<1e-6 max relative change over 42 random (a,b) draws);
# stat_shannon_entropy and all amplitude/variance/abs-power/Hjorth-activity terms FAIL
# and are excluded.
TRANSPORT_TEMPLATES = [
    "stat_{ch}_skew", "stat_{ch}_kurtosis",
    "hjorth_{ch}_mobility", "hjorth_{ch}_complexity",
    "spectral_{ch}_entropy",
    "band_rel_{ch}_delta", "band_rel_{ch}_theta",
    "band_rel_{ch}_alpha", "band_rel_{ch}_beta",
    "ratio_{ch}_theta_alpha", "ratio_{ch}_beta_alpha", "ratio_{ch}_theta_beta",
]
TRANSPORT_COLS = [t.format(ch=ch) for ch in COMMON_8_CHANNELS for t in TRANSPORT_TEMPLATES]
assert len(TRANSPORT_COLS) == 96, len(TRANSPORT_COLS)
assert all(c in FEATURE_COLS for c in TRANSPORT_COLS)


def iter_windows(n_samples: int, sfreq: float):
    win = int(round(WINDOW_SECONDS * sfreq))
    step = int(round(STEP_SECONDS * sfreq))
    for start in range(0, n_samples - win + 1, step):
        end = start + win
        yield start, end, start / sfreq, end / sfreq


def windows_in_segment(data, sfreq, seg_start_s, seg_end_s):
    """Yield 8xN windows whose [start,end] fall within [seg_start_s, seg_end_s]."""
    s0 = int(round(seg_start_s * sfreq))
    s1 = int(round(seg_end_s * sfreq))
    seg = data[:, s0:s1]
    for st, en, st_s, en_s in iter_windows(seg.shape[1], sfreq):
        yield seg[:, st:en], seg_start_s + st_s, seg_start_s + en_s


def feats_for_segment(data, sfreq, subject, dataset, segment_type, label, condition):
    rows = []
    for win, st_s, en_s in windows_in_segment(
        data, sfreq, *( (0.0, SEG_SECONDS) if segment_type in ("calibration", "scored_task")
                        else (SEG_SECONDS, 2 * SEG_SECONDS) )):
        feats = extract_window_features(win, sfreq, COMMON_8_CHANNELS, include_connectivity=False)
        row = {c: feats[c] for c in FEATURE_COLS}
        row.update({
            "subject_id": subject, "dataset": dataset, "condition": condition,
            "segment_type": segment_type, "label": int(label),
            "start_sec": round(st_s, 4), "end_sec": round(en_s, 4),
            "window_uid": f"{dataset}|{subject}|{segment_type}|{st_s:.3f}|{en_s:.3f}",
        })
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# STEW feature build (native 128 Hz)
# ---------------------------------------------------------------------------
def build_stew() -> pd.DataFrame:
    rows = []
    subs = sorted({p.name[:5] for p in STEW_DIR.glob("sub*_*.txt")})
    for sub in subs:
        lo = STEW_DIR / f"{sub}_lo.txt"
        hi = STEW_DIR / f"{sub}_hi.txt"
        if not (lo.exists() and hi.exists()):
            continue
        lo_arr = np.loadtxt(lo).T[STEW_COL_IDX, :]   # 8 x N
        hi_arr = np.loadtxt(hi).T[STEW_COL_IDX, :]
        rows += feats_for_segment(lo_arr, TARGET_SFREQ, sub, "STEW", "calibration", 0, "rest_no_task")
        rows += feats_for_segment(lo_arr, TARGET_SFREQ, sub, "STEW", "scored_rest", 0, "rest_no_task")
        rows += feats_for_segment(hi_arr, TARGET_SFREQ, sub, "STEW", "scored_task", 1, "high_workload")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MAT transport-compatible feature build (resample 500 -> 128 Hz)
# ---------------------------------------------------------------------------
def build_mat() -> pd.DataFrame:
    rows = []
    paths = sorted(MAT_DIR.rglob("Subject*_*.edf"))
    if not paths:
        raise FileNotFoundError(f"No MAT EDF under {MAT_DIR}")
    for path in paths:
        sub = path.name.split("_")[0]
        if path.name.endswith("_1.edf"):
            condition, segments = "rest", [("calibration", 0), ("scored_rest", 0)]
        elif path.name.endswith("_2.edf"):
            condition, segments = "arithmetic", [("scored_task", 1)]
        else:
            continue
        raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        assert sfreq == 500.0, f"unexpected sfreq {sfreq} in {path.name}"
        picks = [raw.ch_names.index(f"EEG {c}") for c in COMMON_8_CHANNELS]
        data = raw.get_data(picks=picks) * 1e6  # -> microvolts
        raw.close()
        # deterministic anti-aliased resample 500 -> 128 Hz (up=32, down=125)
        data128 = resample_poly(data, up=32, down=125, axis=1)
        for seg_type, label in segments:
            rows += feats_for_segment(data128, TARGET_SFREQ, sub, "MAT128", seg_type, label, condition)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# metrics helpers
# ---------------------------------------------------------------------------
def subject_aucs(pred: pd.DataFrame) -> np.ndarray:
    vals = []
    for _, g in pred.groupby("subject_id"):
        if g["y_true"].nunique() == 2:
            vals.append(roc_auc_score(g["y_true"], g["score"]))
    return np.asarray(vals, dtype=float)


def boot_ci(values: np.ndarray, n=BOOTSTRAPS):
    if values.size == 0:
        return (np.nan,) * 4
    boot = [np.mean(RNG.choice(values, values.size, replace=True)) for _ in range(n)]
    return (float(np.mean(values)), float(np.std(values, ddof=1) if values.size > 1 else np.nan),
            float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


def paired_delta(a: pd.DataFrame, b: pd.DataFrame, name: str, n=BOOTSTRAPS):
    deltas = []
    for s in sorted(set(a["subject_id"]) & set(b["subject_id"])):
        ga, gb = a[a.subject_id == s], b[b.subject_id == s]
        if ga["y_true"].nunique() == 2 and gb["y_true"].nunique() == 2:
            deltas.append(roc_auc_score(ga.y_true, ga.score) - roc_auc_score(gb.y_true, gb.score))
    v = np.asarray(deltas, float)
    boot = [np.mean(RNG.choice(v, v.size, replace=True)) for _ in range(n)] if v.size else []
    return {"comparison": name, "n_subjects": int(v.size),
            "mean_delta": float(np.mean(v)) if v.size else np.nan,
            "median_delta": float(np.median(v)) if v.size else np.nan,
            "ci95_low": float(np.percentile(boot, 2.5)) if boot else np.nan,
            "ci95_high": float(np.percentile(boot, 97.5)) if boot else np.nan,
            "bootstraps": n if boot else 0}


# ---------------------------------------------------------------------------
# Phase 2: within-STEW
# ---------------------------------------------------------------------------
def within_stew(stew: pd.DataFrame):
    calib_mask = stew["segment_type"].eq("calibration").to_numpy()
    eval_mask = stew["segment_type"].isin(["scored_rest", "scored_task"]).to_numpy()
    metrics, preds, subj_rows = [], {}, []
    for cal in ["absolute", "mean_subtraction", "zscore"]:
        pred, folds, _ = nested_loso_predictions(
            df=stew, feature_cols=FEATURE_COLS, calib_mask=calib_mask, eval_mask=eval_mask,
            calibration_mode=cal, model_name="logistic_l2", c_grid=[1.0], inner_splits=5)
        preds[cal] = pred
        s_aucs = subject_aucs(pred)
        mean, sd, lo, hi = boot_ci(s_aucs)
        pooled = roc_auc_score(pred.y_true, pred.score) if pred.y_true.nunique() == 2 else np.nan
        metrics.append({"calibration": cal, "model": "logistic_l2", "n_subjects": int(s_aucs.size),
                        "pooled_window_auc": float(pooled), "macro_subject_mean_auc": mean,
                        "macro_subject_median_auc": float(np.median(s_aucs)) if s_aucs.size else np.nan,
                        "subject_auc_sd": sd, "subject_auc_ci95_low": lo, "subject_auc_ci95_high": hi})
        for _, g in pred.groupby("subject_id"):
            if g["y_true"].nunique() == 2:
                subj_rows.append({"subject_id": g.subject_id.iloc[0], "calibration": cal,
                                  "subject_auc": float(roc_auc_score(g.y_true, g.score)),
                                  "n_rest": int((g.y_true == 0).sum()), "n_task": int((g.y_true == 1).sum())})
    paired = [paired_delta(preds["mean_subtraction"], preds["absolute"], "mean_subtraction_minus_absolute"),
              paired_delta(preds["mean_subtraction"], preds["zscore"], "mean_subtraction_minus_zscore")]
    return pd.DataFrame(metrics), pd.DataFrame(subj_rows), pd.DataFrame(paired), preds


def stew_permutation(stew: pd.DataFrame):
    calib_mask = stew["segment_type"].eq("calibration").to_numpy()
    eval_mask = stew["segment_type"].isin(["scored_rest", "scored_task"]).to_numpy()
    subjects = stew["subject_id"].astype(str).to_numpy()
    base_labels = stew["label"].to_numpy(dtype=int)

    # observed macro subject mean AUC for mean_subtraction
    pred, _, _ = nested_loso_predictions(
        df=stew, feature_cols=FEATURE_COLS, calib_mask=calib_mask, eval_mask=eval_mask,
        calibration_mode="mean_subtraction", model_name="logistic_l2", c_grid=[1.0], inner_splits=5)
    observed = float(np.mean(subject_aucs(pred)))

    null = []
    for i in range(N_PERM):
        y = base_labels.copy()
        for s in np.unique(subjects):
            idx = np.where((subjects == s) & eval_mask)[0]
            y[idx] = RNG.permutation(base_labels[idx])
        p, _, _ = nested_loso_predictions(
            df=stew, feature_cols=FEATURE_COLS, calib_mask=calib_mask, eval_mask=eval_mask,
            calibration_mode="mean_subtraction", model_name="logistic_l2", c_grid=[1.0],
            inner_splits=5, y_override=y)
        null.append(float(np.mean(subject_aucs(p))))
    null = np.asarray(null, float)
    valid = null[np.isfinite(null)]
    emp_p = float((1 + np.sum(valid >= observed)) / (valid.size + 1))
    df = pd.DataFrame({"permutation_index": np.arange(len(null)), "macro_subject_auc": null})
    df.attrs = {}
    summary = {"observed_macro_subject_auc": observed, "n_perm": int(valid.size),
               "null_mean": float(valid.mean()), "null_ci95_low": float(np.percentile(valid, 2.5)),
               "null_ci95_high": float(np.percentile(valid, 97.5)), "empirical_p": emp_p}
    return df, summary


# ---------------------------------------------------------------------------
# Phase 3: MAT -> STEW transfer
# ---------------------------------------------------------------------------
def transfer(mat: pd.DataFrame, stew: pd.DataFrame):
    mat_calib = mat["segment_type"].eq("calibration").to_numpy()
    mat_eval = mat["segment_type"].isin(["scored_rest", "scored_task"]).to_numpy()
    stew_calib = stew["segment_type"].eq("calibration").to_numpy()
    stew_eval = stew["segment_type"].isin(["scored_rest", "scored_task"]).to_numpy()

    metrics, boots, pred_frames, per_subj = [], [], [], {}
    for cal in ["absolute", "mean_subtraction", "zscore"]:
        # TRANSPORT pipeline uses ONLY the unit-invariant feature subset.
        x_mat = apply_baseline_calibration(mat, mat_eval, mat_calib, TRANSPORT_COLS, cal)
        y_mat = mat.loc[mat_eval, "label"].to_numpy(int)
        x_stew = apply_baseline_calibration(stew, stew_eval, stew_calib, TRANSPORT_COLS, cal)
        y_stew = stew.loc[stew_eval, "label"].to_numpy(int)
        stew_subj = stew.loc[stew_eval, "subject_id"].to_numpy()

        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        xtr = sc.fit_transform(imp.fit_transform(x_mat))      # fit on MAT only
        xte = sc.transform(imp.transform(x_stew))
        model = make_model("logistic_l2", 1.0)
        model.fit(xtr, y_mat)
        scores = np.asarray(model.decision_function(xte), float)

        pred = pd.DataFrame({"subject_id": stew_subj, "y_true": y_stew, "score": scores,
                             "calibration": cal})
        pred_frames.append(pred)
        per_subj[cal] = pred
        s_aucs = subject_aucs(pred)
        mean, sd, lo, hi = boot_ci(s_aucs)
        pooled = roc_auc_score(y_stew, scores) if len(set(y_stew)) == 2 else np.nan
        metrics.append({"calibration": cal, "direction": "MAT128_to_STEW", "model": "logistic_l2",
                        "n_target_subjects": int(s_aucs.size), "pooled_window_auc": float(pooled),
                        "macro_subject_mean_auc": mean, "macro_subject_median_auc":
                        float(np.median(s_aucs)) if s_aucs.size else np.nan,
                        "subject_auc_sd": sd, "subject_auc_ci95_low": lo, "subject_auc_ci95_high": hi})
    boots = [paired_delta(per_subj["mean_subtraction"], per_subj["absolute"],
                          "mean_subtraction_minus_absolute"),
             paired_delta(per_subj["mean_subtraction"], per_subj["zscore"],
                          "mean_subtraction_minus_zscore")]
    preds_all = pd.concat(pred_frames, ignore_index=True)
    return pd.DataFrame(metrics), pd.DataFrame(boots), preds_all


def write_yaml_spec(mat, stew):
    excluded = [c for c in FEATURE_COLS if c not in TRANSPORT_COLS]
    spec = f"""# Transport-compatible feature specification
# FROZEN BEFORE any transfer/AUC/permutation result was computed.
measurement_unit_problem:
  mat_units: microvolts_from_EDF
  stew_units: raw_16bit_emotiv_epoc_adc_counts_with_dc_offset_~4200
  official_unit_conversion_found: false
  searched: [official_zip_archive_no_readme, ieee_dataport_page, stew_publication_metadata]
  conclusion: no_verified_microvolt_conversion_exists_for_STEW
  resampling_note: MAT_500_to_128_resolves_sampling_rate_only_NOT_amplitude_units

two_separate_pipelines:
  within_stew:
    rationale: train_and_test_within_same_device_unit_system
    feature_set: no_gamma_184
    n_features: {len(FEATURE_COLS)}
    candidate_method: mean_subtraction
    comparator: absolute
    secondary_diagnostic: zscore
    labeling: cross_task_device_sensitivity_only_not_replication_or_confirmation
  mat_to_stew_transport:
    rationale: amplitude_units_not_comparable_so_use_only_scale_offset_invariant_features
    invariance_property: invariant_under_x_to_a_x_plus_b_with_a_gt_0
    empirical_verification: max_relative_change_lt_1e-6_over_42_random_(a,b)_draws
    feature_set: transport_invariant_96
    n_features: {len(TRANSPORT_COLS)}
    mat_resample: {{method: scipy.signal.resample_poly, up: 32, down: 125, native_hz: 500, target_hz: {TARGET_SFREQ}}}
    candidate_method: mean_subtraction
    comparator: absolute
    secondary_diagnostic: zscore
    imputation_scaling_fit_on: MAT_training_only
    target_calibration_source: STEW_unlabeled_baseline_first_30s_lo
    target_labels_used_for_training_or_tuning: false

transport_invariant_templates_per_channel:
{chr(10).join('  - ' + t.replace('{ch}_', '') for t in TRANSPORT_TEMPLATES)}
transport_excluded_templates_reason:
  amplitude_or_variance_in_device_units: [stat_mean, stat_std, stat_var, stat_rms, stat_ptp, hjorth_activity, band_abs_delta, band_abs_theta, band_abs_alpha, band_abs_beta]
  value_histogram_entropy_numerically_noninvariant: [stat_shannon_entropy]

transport_invariant_feature_list: {json.dumps(TRANSPORT_COLS)}
transport_excluded_feature_count: {len(excluded)}

locked_channels: {COMMON_8_CHANNELS}
channel_mapping_mat_to_stew: {json.dumps(COMMON8_TO_STEW)}
window_seconds: {WINDOW_SECONDS}
overlap_fraction: {OVERLAP}
segment_seconds: {SEG_SECONDS}
balanced_design: {{calibration: first_30s_rest, scored_rest: 30_60s_rest, scored_task: first_30s_workload}}
model: {{name: logistic_l2, C: 1.0, solver: liblinear, class_weight: balanced}}
mat_windows: {len(mat)}
stew_windows: {len(stew)}
status: exploratory_frozen_before_results
"""
    (OUT_DIR / "transport_compatible_feature_spec.yaml").write_text(spec)


def main():
    print("Building STEW features (128 Hz native)...")
    stew = build_stew().reset_index(drop=True)
    print(f"  STEW windows: {len(stew)} ; subjects: {stew.subject_id.nunique()}")
    print("Building MAT transport-compatible features (500->128 Hz)...")
    mat = build_mat().reset_index(drop=True)
    print(f"  MAT128 windows: {len(mat)} ; subjects: {mat.subject_id.nunique()}")

    # feature validity check (predeclared exclusion rule: drop only degenerate cols)
    transport_degenerate = [c for c in TRANSPORT_COLS
                            if not np.isfinite(stew[c]).any() or not np.isfinite(mat[c]).any()]
    if transport_degenerate:
        raise RuntimeError(f"Degenerate transport features: {transport_degenerate}")
    write_yaml_spec(mat, stew)

    # window manifest (committable)
    man = pd.concat([
        stew[["dataset", "subject_id", "segment_type", "label", "condition", "start_sec", "end_sec", "window_uid"]],
        mat[["dataset", "subject_id", "segment_type", "label", "condition", "start_sec", "end_sec", "window_uid"]],
    ], ignore_index=True)
    man.to_csv(OUT_DIR / "stew_window_manifest.csv", index=False)

    # save feature tables locally (parquet not committed by sync)
    stew.to_parquet(OUT_DIR / "stew_features_128.parquet", index=False)
    mat.to_parquet(OUT_DIR / "mat128_features.parquet", index=False)

    # ---- Phase 2 ----
    print("Phase 2: within-STEW ...")
    w_metrics, w_subj, w_paired, _ = within_stew(stew)
    w_metrics.to_csv(OUT_DIR / "stew_within_metrics.csv", index=False)
    w_subj.to_csv(OUT_DIR / "stew_subject_metrics.csv", index=False)
    w_paired.to_csv(OUT_DIR / "stew_paired_bootstrap.csv", index=False)
    print(w_metrics.to_string(index=False))

    print("Phase 2: STEW permutation (mean_subtraction, 200) ...")
    perm_df, perm_sum = stew_permutation(stew)
    perm_df.to_csv(OUT_DIR / "stew_macro_subject_permutation.csv", index=False)
    pd.DataFrame([perm_sum]).to_csv(OUT_DIR / "stew_permutation_summary.csv", index=False)
    print("  ", perm_sum)

    # figure: within-STEW bars
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(w_metrics.calibration, w_metrics.macro_subject_mean_auc,
           yerr=[w_metrics.macro_subject_mean_auc - w_metrics.subject_auc_ci95_low,
                 w_metrics.subject_auc_ci95_high - w_metrics.macro_subject_mean_auc],
           capsize=4, color=["#6b7280", "#0f766e", "#b45309"])
    ax.axhline(0.5, ls="--", c="k", lw=1)
    ax.set_ylim(0, 1); ax.set_ylabel("Macro subject ROC-AUC")
    ax.set_title("Within-STEW exploratory sensitivity")
    fig.tight_layout(); fig.savefig(FIG_DIR / "stew_within_macro_auc.png", dpi=200); plt.close(fig)

    # ---- Phase 3 ----
    print("Phase 3: MAT128 -> STEW transfer ...")
    t_metrics, t_boot, t_pred = transfer(mat, stew)
    t_metrics.to_csv(OUT_DIR / "mat_to_stew_transfer_metrics.csv", index=False)
    t_boot.to_csv(OUT_DIR / "mat_to_stew_transfer_bootstrap.csv", index=False)
    t_pred.to_csv(OUT_DIR / "mat_to_stew_transfer_predictions.csv", index=False)
    print(t_metrics.to_string(index=False))
    print(t_boot.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(t_metrics.calibration, t_metrics.macro_subject_mean_auc,
           yerr=[t_metrics.macro_subject_mean_auc - t_metrics.subject_auc_ci95_low,
                 t_metrics.subject_auc_ci95_high - t_metrics.macro_subject_mean_auc],
           capsize=4, color=["#6b7280", "#0f766e", "#b45309"])
    ax.axhline(0.5, ls="--", c="k", lw=1)
    ax.set_ylim(0, 1); ax.set_ylabel("Macro subject ROC-AUC")
    ax.set_title("MAT(128Hz) -> STEW exploratory transfer")
    fig.tight_layout(); fig.savefig(FIG_DIR / "mat_to_stew_transfer_macro_auc.png", dpi=200); plt.close(fig)

    # consolidated machine-readable summary for the decision report
    summary = {
        "within_stew": w_metrics.to_dict(orient="records"),
        "within_paired": w_paired.to_dict(orient="records"),
        "permutation": perm_sum,
        "transfer": t_metrics.to_dict(orient="records"),
        "transfer_paired": t_boot.to_dict(orient="records"),
        "dropped_feature_families": transport_degenerate if transport_degenerate else "none",
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print("WROTE summary.json")


if __name__ == "__main__":
    raise SystemExit(main())
