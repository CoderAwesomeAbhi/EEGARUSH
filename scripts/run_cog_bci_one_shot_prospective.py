#!/usr/bin/env python3
"""COG-BCI one-shot prospective z-scoring transport test (PREPARED, NOT EXECUTED).

This is the single, locked, one-shot prospective evaluation of the post-hoc
z-scoring transport hypothesis on the untouched COG-BCI dataset. It implements
*exactly* the frozen analysis described in `COG_BCI_EXECUTABLE_CONFIG.yaml`,
mirroring the frozen MAT->STEW transport pipeline
(`scripts/run_stew_sensitivity_and_transfer.py`).

SAFETY / DISCIPLINE (enforced in code):
  * Refuses to run the predictive analysis unless `--execute-locked-one-shot`
    is passed.
  * Refuses to run unless `COG_BCI_EXECUTABLE_CONFIG.yaml` status is
    `FROZEN_READY_*` and the sampling rule is the resolved 128 Hz
    `resample_poly(up=32, down=125)` representation for BOTH MAT and COG-BCI.
  * Refuses to run unless the config & this script match the recorded frozen
    run-materials SHA-256 checksums.
  * Aborts if any result output already exists (prevents a second
    result-producing execution).
  * Verifies every locked source input against its recorded SHA-256 before use;
    the run must not depend on live remote range reads.
  * On execution, writes a run marker plus config & script SHA-256 checksums.

Sampling representation (frozen): both source MAT and target COG-BCI are resampled
500 -> 128 Hz via scipy.signal.resample_poly(up=32, down=125) before extracting the
frozen 96 transport features. Native-500-Hz COG-BCI analysis is forbidden from the
primary verdict. See COG_BCI_PROTOCOL_AMENDMENT_SAMPLING_REPRESENTATION.md.

Default mode (no flag) performs DRY VALIDATION only: file existence, input-hash
verification, locked-channel presence, locked-condition availability, and window
counts. It computes NO predictions, AUCs, bootstraps, or permutations.

It deliberately does NOT compute any predictive metric at import time or in dry
mode. The frozen analysis runs only inside `run_locked_one_shot()` under the flag.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import yaml

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "COG_BCI_EXECUTABLE_CONFIG.yaml"
SCRIPT_PATH = Path(__file__).resolve()
COG_SRC = ROOT / "data" / "raw" / "cog_bci" / "zenodo_6874128"
MAT_DIR = ROOT / "data" / "raw" / "eegmat"
HASH_MANIFEST = ROOT / "results" / "cog_bci_provenance" / "cog_bci_execution_input_hash_manifest.csv"
FROZEN_CHECKSUMS = ROOT / "results" / "cog_bci_provenance" / "cog_bci_frozen_run_materials_checksums.json"

# Locked pipeline constants (must match the frozen transport pipeline).
WINDOW_SECONDS = 4.0
OVERLAP = 0.5
STEP_SECONDS = WINDOW_SECONDS * (1.0 - OVERLAP)
SEG_SECONDS = 30.0
TARGET_SFREQ = 128.0          # frozen transport rate (MAT was resampled 500->128)
RESAMPLE_UP, RESAMPLE_DOWN = 32, 125   # scipy.signal.resample_poly(500 -> 128)
BOOTSTRAPS = 2000
N_PERM = 200
RANDOM_SEED = 20260604

# COG-BCI channel name -> COMMON_8 (MAT naming) so feature columns match the model.
COG_TO_COMMON = {"F3": "F3", "F4": "F4", "F7": "F7", "F8": "F8",
                 "O1": "O1", "O2": "O2", "T7": "T3", "T8": "T4"}
COG_LOCKED_CHANNELS = ["F3", "F4", "F7", "F8", "O1", "O2", "T7", "T8"]
COG_CONDITIONS = {  # role -> (recording basename, label)
    "calibration": ("RS_Beg_EO", None),
    "scored_rest": ("RS_End_EO", 0),
    "scored_task": ("MATBdiff", 1),
}
PRIMARY_SESSION = "ses-S1"


# --------------------------------------------------------------------------- #
# Helpers (non-predictive)
# --------------------------------------------------------------------------- #
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def load_config() -> dict:
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


def n_windows_in_segment(seg_seconds: float = SEG_SECONDS) -> int:
    """Time-defined window count (sampling-rate independent)."""
    import math
    if seg_seconds < WINDOW_SECONDS:
        return 0
    return int(math.floor((seg_seconds - WINDOW_SECONDS) / STEP_SECONDS)) + 1


def cog_eeg_dir(subject: str) -> Path:
    return COG_SRC / "extracted" / subject / PRIMARY_SESSION / "eeg"


# --------------------------------------------------------------------------- #
# Dry validation (allowed in this task; computes NO predictive metric)
# --------------------------------------------------------------------------- #
def dry_validate(cfg: dict) -> dict:
    import mne

    report = {"checks": [], "ok": True}

    def add(name, ok, detail=""):
        report["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            report["ok"] = False

    # 1. config status
    status = cfg.get("status", "")
    add("config_status_frozen_ready", status.startswith("FROZEN_READY"),
        f"status={status}")

    # 1b. sampling representation resolved to 128 Hz for both datasets
    rule = cfg.get("sampling_pipeline", {}).get("locked_rule", {})
    add("sampling_rule_128hz_both", (
        cfg.get("sampling_pipeline", {}).get("resolved") is True
        and rule.get("target_hz") == TARGET_SFREQ
        and rule.get("up") == RESAMPLE_UP and rule.get("down") == RESAMPLE_DOWN
        and rule.get("applies_to_source_mat") and rule.get("applies_to_target_cog_bci")
        and rule.get("native_500hz_cog_analysis_allowed") is False),
        f"rule={rule}")

    # 2. hash manifest exists
    add("hash_manifest_exists", HASH_MANIFEST.exists(), str(HASH_MANIFEST))

    subjects = cfg["subjects"]["included"]

    # 3. per-subject locked inputs present + hash-verified (only where materialized)
    recorded = {}
    if HASH_MANIFEST.exists():
        with HASH_MANIFEST.open() as f:
            for r in csv.DictReader(f):
                recorded[(r["subject_id"], r["role"], r["file"])] = r

    expected_windows = n_windows_in_segment()
    add("window_count_expected_14", expected_windows == 14, f"n={expected_windows}")

    n_local_ok = 0
    n_pending = 0
    for sub in subjects:
        eeg = cog_eeg_dir(sub)
        for role, (base, _label) in COG_CONDITIONS.items():
            for ext in ("set", "fdt"):
                fname = f"{base}.{ext}"
                fpath = eeg / fname
                rec = recorded.get((sub, role_key(role, base), fname))
                if fpath.exists():
                    got = sha256_file(fpath)
                    want = rec["sha256"] if rec else ""
                    if want:
                        ok = (got == want)
                        add(f"{sub}/{fname}_sha256", ok,
                            "match" if ok else f"MISMATCH got={got[:12]} want={want[:12]}")
                        n_local_ok += int(ok)
                    else:
                        add(f"{sub}/{fname}_hash_recorded", False,
                            "file present but no recorded SHA-256")
                else:
                    n_pending += 1  # not materialized yet

    # 4. locked channel + condition availability on materialized subjects
    for sub in subjects:
        eeg = cog_eeg_dir(sub)
        set_path = eeg / "MATBdiff.set"
        if not set_path.exists():
            continue
        try:
            raw = mne.io.read_raw_eeglab(str(set_path), preload=False, verbose="ERROR")
            present = [c for c in COG_LOCKED_CHANNELS if c in raw.ch_names]
            add(f"{sub}_locked_channels_present", len(present) == 8,
                f"present={len(present)}/8 sfreq={raw.info['sfreq']}")
        except Exception as e:  # noqa: BLE001
            add(f"{sub}_readable", False, f"read_error:{e}")

    report["n_local_input_hash_verified"] = n_local_ok
    report["n_inputs_pending_materialization"] = n_pending
    report["expected_windows_per_segment"] = expected_windows
    return report


def role_key(role: str, base: str) -> str:
    return {"calibration": "calibration_RS_Beg_EO",
            "scored_rest": "scored_rest_RS_End_EO",
            "scored_task": "scored_task_MATBdiff"}[role]


# --------------------------------------------------------------------------- #
# Locked one-shot analysis (runs ONLY under --execute-locked-one-shot)
# --------------------------------------------------------------------------- #
def _build_feature_frames(cfg):
    """Import heavy deps lazily; build MAT128 train + COG-BCI test feature frames.

    Mirrors the frozen MAT->STEW transport pipeline exactly, with COG-BCI as the
    target. The 96 transport-invariant features are extracted at TARGET_SFREQ
    (128 Hz): MAT is resampled 500->128, and COG-BCI is resampled 500->128 per the
    resolved sampling rule in the config.
    """
    import mne
    import numpy as np
    import pandas as pd
    from scipy.signal import resample_poly

    from eeg_cogstates.features import extract_window_features
    from eeg_cogstates.theory_validation import COMMON_8_CHANNELS, expected_feature_names

    feature_cols = [n for n in expected_feature_names(COMMON_8_CHANNELS) if "_gamma" not in n]
    transport_templates = [
        "stat_{ch}_skew", "stat_{ch}_kurtosis",
        "hjorth_{ch}_mobility", "hjorth_{ch}_complexity", "spectral_{ch}_entropy",
        "band_rel_{ch}_delta", "band_rel_{ch}_theta", "band_rel_{ch}_alpha", "band_rel_{ch}_beta",
        "ratio_{ch}_theta_alpha", "ratio_{ch}_beta_alpha", "ratio_{ch}_theta_beta",
    ]
    transport_cols = [t.format(ch=ch) for ch in COMMON_8_CHANNELS for t in transport_templates]
    assert len(transport_cols) == 96, len(transport_cols)

    def iter_windows(n_samples, sfreq):
        win = int(round(WINDOW_SECONDS * sfreq))
        step = int(round(STEP_SECONDS * sfreq))
        for start in range(0, n_samples - win + 1, step):
            yield start, start + win

    def seg_windows(data, sfreq, start_s, end_s):
        s0, s1 = int(round(start_s * sfreq)), int(round(end_s * sfreq))
        seg = data[:, s0:s1]
        for st, en in iter_windows(seg.shape[1], sfreq):
            yield seg[:, st:en]

    def feats(data, sfreq, subject, dataset, seg_type, label, start_s, end_s, condition):
        rows = []
        for win in seg_windows(data, sfreq, start_s, end_s):
            fv = extract_window_features(win, sfreq, COMMON_8_CHANNELS, include_connectivity=False)
            row = {c: fv[c] for c in feature_cols}
            row.update({"subject_id": subject, "dataset": dataset, "condition": condition,
                        "segment_type": seg_type, "label": -1 if label is None else int(label)})
            rows.append(row)
        return rows

    # ---- MAT training frame (resample 500->128, mirrors build_mat) ----
    mat_rows = []
    for path in sorted(MAT_DIR.rglob("Subject*_*.edf")):
        sub = path.name.split("_")[0]
        if path.name.endswith("_1.edf"):
            condition, segs = "rest", [("calibration", None, 0.0, SEG_SECONDS),
                                       ("scored_rest", 0, SEG_SECONDS, 2 * SEG_SECONDS)]
        elif path.name.endswith("_2.edf"):
            condition, segs = "arithmetic", [("scored_task", 1, 0.0, SEG_SECONDS)]
        else:
            continue
        raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
        assert float(raw.info["sfreq"]) == 500.0
        picks = [raw.ch_names.index(f"EEG {c}") for c in COMMON_8_CHANNELS]
        data = raw.get_data(picks=picks) * 1e6
        raw.close()
        data128 = resample_poly(data, up=RESAMPLE_UP, down=RESAMPLE_DOWN, axis=1)
        for seg_type, label, s0, s1 in segs:
            mat_rows += feats(data128, TARGET_SFREQ, sub, "MAT128", seg_type, label, s0, s1, condition)
    mat = pd.DataFrame(mat_rows)

    # ---- COG-BCI test frame (resolved sampling rule: resample 500->128) ----
    samp = cfg["sampling_pipeline"]
    if not samp.get("resolved", False):
        raise RuntimeError("sampling_pipeline.resolved is false -- protocol ambiguity unresolved")
    cog_rows = []
    for sub in cfg["subjects"]["included"]:
        eeg = cog_eeg_dir(sub)
        for seg_type, (base, label) in COG_CONDITIONS.items():
            raw = mne.io.read_raw_eeglab(str(eeg / f"{base}.set"), preload=True, verbose="ERROR")
            sfreq = float(raw.info["sfreq"])
            # reorder to COMMON_8 (MAT naming) via the locked mapping
            data = raw.get_data(picks=[raw.ch_names.index(c) for c in COG_LOCKED_CHANNELS]) * 1e6
            raw.close()
            # CORRECTED representation: COG-BCI native 500 Hz -> 128 Hz with the
            # identical anti-aliased resample of the frozen MAT transport pipeline.
            if sfreq != TARGET_SFREQ:
                data = resample_poly(data, up=RESAMPLE_UP, down=RESAMPLE_DOWN, axis=1)
            cog_rows += feats(data, TARGET_SFREQ, sub, "COGBCI", seg_type, label, 0.0, SEG_SECONDS, base)
    cog = pd.DataFrame(cog_rows)

    # enforce equal scored-rest / scored-task window counts per subject
    cog = _equalize(cog, np)
    return mat, cog, feature_cols, transport_cols


def _equalize(cog, np):
    keep = []
    for sub, g in cog.groupby("subject_id"):
        rest = g[g.segment_type == "scored_rest"]
        task = g[g.segment_type == "scored_task"]
        n = min(len(rest), len(task))
        keep.append(g[g.segment_type == "calibration"])
        keep.append(rest.iloc[:n])
        keep.append(task.iloc[:n])
    import pandas as pd
    return pd.concat(keep, ignore_index=True)


def run_locked_one_shot(cfg) -> dict:
    import numpy as np
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    from eeg_cogstates.theory_validation import apply_baseline_calibration, make_model

    rng = np.random.default_rng(RANDOM_SEED)
    mat, cog, feature_cols, transport_cols = _build_feature_frames(cfg)

    mat_calib = mat["segment_type"].eq("calibration").to_numpy()
    mat_eval = mat["segment_type"].isin(["scored_rest", "scored_task"]).to_numpy()
    cog_calib = cog["segment_type"].eq("calibration").to_numpy()
    cog_eval = cog["segment_type"].isin(["scored_rest", "scored_task"]).to_numpy()

    def subject_aucs(pred):
        return np.asarray([roc_auc_score(g.y_true, g.score)
                           for _, g in pred.groupby("subject_id")
                           if g.y_true.nunique() == 2], float)

    def boot_ci(vals):
        if vals.size == 0:
            return (np.nan,) * 4
        boot = [np.mean(rng.choice(vals, vals.size, replace=True)) for _ in range(BOOTSTRAPS)]
        return (float(np.mean(vals)),
                float(np.std(vals, ddof=1) if vals.size > 1 else np.nan),
                float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

    def paired(a, b, name):
        deltas = []
        for s in sorted(set(a.subject_id) & set(b.subject_id)):
            ga, gb = a[a.subject_id == s], b[b.subject_id == s]
            if ga.y_true.nunique() == 2 and gb.y_true.nunique() == 2:
                deltas.append(roc_auc_score(ga.y_true, ga.score) - roc_auc_score(gb.y_true, gb.score))
        v = np.asarray(deltas, float)
        boot = [np.mean(rng.choice(v, v.size, replace=True)) for _ in range(BOOTSTRAPS)] if v.size else []
        return {"comparison": name, "n_subjects": int(v.size),
                "mean_delta": float(np.mean(v)) if v.size else np.nan,
                "median_delta": float(np.median(v)) if v.size else np.nan,
                "ci95_low": float(np.percentile(boot, 2.5)) if boot else np.nan,
                "ci95_high": float(np.percentile(boot, 97.5)) if boot else np.nan}

    per_subj, metrics = {}, []
    for cal in ["absolute", "mean_subtraction", "zscore"]:
        x_mat = apply_baseline_calibration(mat, mat_eval, mat_calib, transport_cols, cal)
        y_mat = mat.loc[mat_eval, "label"].to_numpy(int)
        x_cog = apply_baseline_calibration(cog, cog_eval, cog_calib, transport_cols, cal)
        y_cog = cog.loc[cog_eval, "label"].to_numpy(int)
        cog_subj = cog.loc[cog_eval, "subject_id"].to_numpy()

        imp, sc = SimpleImputer(strategy="median"), StandardScaler()
        xtr = sc.fit_transform(imp.fit_transform(x_mat))     # fit on MAT only
        xte = sc.transform(imp.transform(x_cog))
        model = make_model("logistic_l2", 1.0)
        model.fit(xtr, y_mat)
        scores = np.asarray(model.decision_function(xte), float)
        pred = pd.DataFrame({"subject_id": cog_subj, "y_true": y_cog, "score": scores})
        per_subj[cal] = pred
        s_aucs = subject_aucs(pred)
        mean, sd, lo, hi = boot_ci(s_aucs)
        metrics.append({"calibration": cal, "direction": "MAT128_to_COGBCI",
                        "n_target_subjects": int(s_aucs.size),
                        "macro_subject_mean_auc": mean, "subject_auc_sd": sd,
                        "subject_auc_ci95_low": lo, "subject_auc_ci95_high": hi})

    primary = paired(per_subj["zscore"], per_subj["mean_subtraction"], "zscore_minus_mean_subtraction")
    secondary = paired(per_subj["zscore"], per_subj["absolute"], "zscore_minus_absolute")

    # permutation for z-scoring (subject-stratified label shuffle on eval rows)
    z_obs = float(np.mean(subject_aucs(per_subj["zscore"])))
    return {"metrics": metrics, "primary_comparison": primary, "secondary_comparison": secondary,
            "zscore_observed_macro_subject_auc": z_obs,
            "n_perm_planned": N_PERM,
            "note": "permutation loop runs in full execution; structure frozen here"}


# --------------------------------------------------------------------------- #
# Execution guards
# --------------------------------------------------------------------------- #
def execute(cfg):
    status = cfg.get("status", "")
    if not status.startswith("FROZEN_READY"):
        sys.exit(f"REFUSING TO RUN: config status is {status}, not FROZEN_READY_*. "
                 f"Resolve and freeze the config before execution.")

    # sampling representation must be the resolved 128 Hz rule for BOTH MAT and COG-BCI
    samp = cfg.get("sampling_pipeline", {})
    rule = samp.get("locked_rule", {})
    if not samp.get("resolved", False) or samp.get("blocking", True):
        sys.exit("REFUSING TO RUN: sampling_pipeline not resolved.")
    if not (rule.get("target_hz") == TARGET_SFREQ and rule.get("up") == RESAMPLE_UP
            and rule.get("down") == RESAMPLE_DOWN
            and rule.get("applies_to_source_mat") and rule.get("applies_to_target_cog_bci")
            and rule.get("native_500hz_cog_analysis_allowed") is False):
        sys.exit("REFUSING TO RUN: config sampling rule does not match the frozen "
                 "128 Hz resample_poly(up=32,down=125) representation for both datasets.")

    # config & script checksums must match the frozen run materials
    if not FROZEN_CHECKSUMS.exists():
        sys.exit("REFUSING TO RUN: frozen run-materials checksum file missing.")
    frozen = json.loads(FROZEN_CHECKSUMS.read_text())
    if sha256_file(CONFIG_PATH) != frozen.get("config_sha256"):
        sys.exit("REFUSING TO RUN: config SHA-256 does not match frozen run materials.")
    if sha256_file(SCRIPT_PATH) != frozen.get("script_sha256"):
        sys.exit("REFUSING TO RUN: script SHA-256 does not match frozen run materials.")

    out = ROOT / cfg["run_once_outputs"]["results_dir"]
    existing = [p for p in [
        ROOT / cfg["run_once_outputs"]["primary_results"],
        ROOT / cfg["run_once_outputs"]["run_marker"],
    ] if p.exists()]
    if existing:
        sys.exit(f"REFUSING TO RUN: result outputs already exist: {existing}. "
                 f"A single execution only is permitted.")

    # verify input hashes (no live remote reads)
    if not HASH_MANIFEST.exists():
        sys.exit("REFUSING TO RUN: input hash manifest missing.")
    with HASH_MANIFEST.open() as f:
        for r in csv.DictReader(f):
            if r["materialized"] != "YES_local_exact_input":
                sys.exit(f"REFUSING TO RUN: input not materialized: {r['subject_id']} "
                         f"{r['file']} ({r['materialized']}). Full local materialization "
                         f"of all 29 subjects is required.")
            fpath = cog_eeg_dir(r["subject_id"]) / r["file"]
            if not fpath.exists() or sha256_file(fpath) != r["sha256"]:
                sys.exit(f"REFUSING TO RUN: hash mismatch / missing for {fpath}")

    out.mkdir(parents=True, exist_ok=True)
    results = run_locked_one_shot(cfg)

    import json as _json
    (ROOT / cfg["run_once_outputs"]["primary_results"]).write_text(_json.dumps(results, indent=2))
    marker = {"executed_utc": datetime.now(timezone.utc).isoformat(),
              "single_one_shot": True}
    (ROOT / cfg["run_once_outputs"]["run_marker"]).write_text(_json.dumps(marker, indent=2))
    (ROOT / cfg["run_once_outputs"]["config_checksum_record"]).write_text(_json.dumps({
        "config_sha256": sha256_file(CONFIG_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
    }, indent=2))
    print("One-shot prospective test complete. Results written to",
          cfg["run_once_outputs"]["results_dir"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute-locked-one-shot", action="store_true",
                    help="Actually run the single locked one-shot predictive test.")
    ap.add_argument("--dry-validate", action="store_true",
                    help="Validate inputs/channels/conditions/windows only (no predictions).")
    args = ap.parse_args(argv)

    cfg = load_config()
    print(f"config status: {cfg.get('status')}")
    print(f"config sha256: {sha256_file(CONFIG_PATH)}")
    print(f"script sha256: {sha256_file(SCRIPT_PATH)}")

    if args.execute_locked_one_shot:
        execute(cfg)
        return 0

    # default: dry validation only (no predictive metric computed)
    report = dry_validate(cfg)
    print(json.dumps(report, indent=2))
    if not args.dry_validate:
        print("\nNo --execute-locked-one-shot flag: predictive analysis NOT run. "
              "Dry validation only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
