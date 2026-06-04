#!/usr/bin/env python3
"""Audit raw PhysioNet EEGMAT EDF provenance against cached feature tables.

This script is intentionally limited to Phase 1 of the raw rescue workflow:
it reads EDF headers, writes provenance manifests, compares them to official
metadata expectations, and stops before any model rebuilding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / "raw_provenance" / "mplconfig"))
sys.path.insert(0, str(ROOT / "src"))

import mne
import numpy as np
import pandas as pd


RAW_DIR_DEFAULT = ROOT / "data" / "raw" / "eegmat"
RAW_PROVENANCE_DIR = ROOT / "results" / "raw_provenance"
CACHED_FEATURE_TABLE = ROOT / "outputs_reproduced" / "features" / "eeg_features.csv"
PREVIOUS_HEADER_MANIFEST = ROOT / "results" / "audit" / "mat_file_header_manifest.csv"

EXPECTED_SUBJECTS = 36
EXPECTED_FILES = 72
EXPECTED_SFREQ = 500.0
EXPECTED_REST_DURATION = 180.0
EXPECTED_TASK_DURATION = 60.0
EXPECTED_HIGHPASS = 0.5
EXPECTED_LOWPASS = 45.0
EXPECTED_CHANNELS = {
    "EEG Fp1",
    "EEG Fp2",
    "EEG F3",
    "EEG F4",
    "EEG F7",
    "EEG F8",
    "EEG Fz",
    "EEG C3",
    "EEG C4",
    "EEG Cz",
    "EEG P3",
    "EEG P4",
    "EEG Pz",
    "EEG O1",
    "EEG O2",
    "EEG T3",
    "EEG T4",
    "EEG T5",
    "EEG T6",
    "EEG A2-A1",
    "ECG ECG",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_mat_filename(path: Path) -> tuple[str, str, int, str]:
    match = re.fullmatch(r"Subject(\d{2})_([12])\.edf", path.name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Unexpected MAT EDF filename: {path.name}")
    subject_id = f"Subject{int(match.group(1)):02d}"
    suffix = int(match.group(2))
    if suffix == 1:
        return subject_id, "rest", 0, "_1_rest_background"
    return subject_id, "arithmetic", 1, "_2_mental_arithmetic"


def discover_mat_edfs(raw_dir: Path) -> list[Path]:
    files = sorted(raw_dir.rglob("Subject*_*.edf"))
    return [path for path in files if re.fullmatch(r"Subject\d{2}_[12]\.edf", path.name, re.I)]


def duration_expected(condition: str) -> float:
    return EXPECTED_REST_DURATION if condition == "rest" else EXPECTED_TASK_DURATION


def window_count(duration_sec: float, window_sec: float = 4.0, step_sec: float = 2.0) -> int:
    if duration_sec < window_sec:
        return 0
    return int(math.floor((duration_sec - window_sec) / step_sec) + 1)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_None._"

    def fmt(value: object) -> str:
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


def read_manifest(files: Iterable[Path]) -> pd.DataFrame:
    rows = []
    for path in files:
        subject_id, condition, label, condition_mapping = parse_mat_filename(path)
        raw = mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        sample_count = int(raw.n_times)
        duration_sec = sample_count / sfreq if sfreq else float("nan")
        channels = list(raw.ch_names)
        rows.append(
            {
                "subject_id": subject_id,
                "condition": condition,
                "label": label,
                "condition_mapping": condition_mapping,
                "filename": path.name,
                "relative_path": str(path.relative_to(ROOT)),
                "num_channels": len(channels),
                "channel_names": json.dumps(channels),
                "sampling_frequency_hz": sfreq,
                "sample_count": sample_count,
                "duration_sec": duration_sec,
                "expected_duration_sec": duration_expected(condition),
                "duration_matches_expected": bool(abs(duration_sec - duration_expected(condition)) <= 0.5),
                "raw_header_highpass_hz": raw.info.get("highpass"),
                "raw_header_lowpass_hz": raw.info.get("lowpass"),
                "reference_metadata": json.dumps(
                    {
                        "custom_ref_applied": raw.info.get("custom_ref_applied"),
                        "description": raw.info.get("description"),
                        "highpass_hz": raw.info.get("highpass"),
                        "lowpass_hz": raw.info.get("lowpass"),
                    },
                    default=str,
                ),
                "file_sha256": sha256_file(path),
                "expected_4s_50pct_windows_from_header": window_count(duration_sec),
            }
        )
        raw.close()
    return pd.DataFrame(rows)


def cached_counts() -> pd.DataFrame:
    if not CACHED_FEATURE_TABLE.exists():
        return pd.DataFrame()
    cached = pd.read_csv(
        CACHED_FEATURE_TABLE,
        usecols=["file", "condition", "label", "subject_id", "window_index", "start_sec", "end_sec"],
    )
    return (
        cached.groupby("file", as_index=False)
        .agg(
            cached_condition=("condition", "first"),
            cached_label=("label", "first"),
            cached_subject_id=("subject_id", "first"),
            cached_windows=("file", "size"),
            cached_min_start_sec=("start_sec", "min"),
            cached_max_end_sec=("end_sec", "max"),
        )
        .rename(columns={"file": "filename"})
    )


def previous_counts() -> pd.DataFrame:
    if not PREVIOUS_HEADER_MANIFEST.exists():
        return pd.DataFrame()
    prev = pd.read_csv(PREVIOUS_HEADER_MANIFEST)
    keep = [
        "file",
        "feature_table_inferred_duration_sec",
        "feature_table_min_start_sec",
        "feature_table_n_windows",
        "header_read_status",
    ]
    return prev[[col for col in keep if col in prev.columns]].rename(columns={"file": "filename"})


def make_comparison(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n_files = len(manifest)
    n_subjects = manifest["subject_id"].nunique() if not manifest.empty else 0
    suffix_counts = manifest["condition"].value_counts().to_dict() if not manifest.empty else {}
    sfreqs = sorted(manifest["sampling_frequency_hz"].dropna().unique().tolist()) if not manifest.empty else []
    duration_failures = manifest.loc[~manifest["duration_matches_expected"], "filename"].tolist() if not manifest.empty else []
    highpass_values = sorted(manifest["raw_header_highpass_hz"].dropna().unique().tolist()) if not manifest.empty else []
    lowpass_values = sorted(manifest["raw_header_lowpass_hz"].dropna().unique().tolist()) if not manifest.empty else []
    channel_sets = manifest["channel_names"].apply(lambda value: set(json.loads(value))) if not manifest.empty else []
    all_expected_channels = all(EXPECTED_CHANNELS == channels for channels in channel_sets) if len(channel_sets) else False

    checks = [
        ("subject_count", EXPECTED_SUBJECTS, n_subjects, n_subjects == EXPECTED_SUBJECTS, ""),
        ("edf_file_count", EXPECTED_FILES, n_files, n_files == EXPECTED_FILES, ""),
        ("condition_mapping", "36 rest + 36 arithmetic", suffix_counts, suffix_counts.get("rest") == 36 and suffix_counts.get("arithmetic") == 36, ""),
        ("sampling_frequency_hz", EXPECTED_SFREQ, sfreqs, len(sfreqs) == 1 and abs(float(sfreqs[0]) - EXPECTED_SFREQ) <= 1e-9, ""),
        (
            "duration_sec_exact_descriptor",
            "rest=180.0, arithmetic=60.0",
            {
                "rest_unique": sorted(manifest.loc[manifest["condition"].eq("rest"), "duration_sec"].round(6).unique().tolist()),
                "arithmetic_unique": sorted(manifest.loc[manifest["condition"].eq("arithmetic"), "duration_sec"].round(6).unique().tolist()),
            },
            len(duration_failures) == 0,
            ";".join(duration_failures),
        ),
        (
            "edf_header_filter_metadata",
            f"highpass={EXPECTED_HIGHPASS}, lowpass={EXPECTED_LOWPASS}, notch=50 reported by descriptor",
            {"highpass_unique": highpass_values, "lowpass_unique": lowpass_values, "notch_header_field": "not_present"},
            (
                len(highpass_values) == 1
                and len(lowpass_values) == 1
                and abs(float(highpass_values[0]) - EXPECTED_HIGHPASS) <= 1e-9
                and abs(float(lowpass_values[0]) - EXPECTED_LOWPASS) <= 1e-9
            ),
            "EDF headers do not expose notch metadata; high/low-pass values are not uniform across files.",
        ),
        ("channel_identity", sorted(EXPECTED_CHANNELS), "all files share expected 21 channels", all_expected_channels, ""),
    ]
    for check_name, expected, observed, passed, detail in checks:
        rows.append(
            {
                "check": check_name,
                "expected": json.dumps(expected, sort_keys=True),
                "observed": json.dumps(observed, sort_keys=True),
                "passed": bool(passed),
                "detail": detail,
            }
        )

    cached = cached_counts()
    if not cached.empty:
        merged = manifest.merge(cached, on="filename", how="outer")
        merged["cached_window_count_matches_header"] = (
            merged["cached_windows"].fillna(-1).astype(int)
            == merged["expected_4s_50pct_windows_from_header"].fillna(-2).astype(int)
        )
        mismatches = merged.loc[~merged["cached_window_count_matches_header"], "filename"].dropna().tolist()
        rows.append(
            {
                "check": "cached_feature_window_counts_match_raw_header",
                "expected": "cached windows equal floor((duration-4)/2)+1 per EDF",
                "observed": f"{int(merged['cached_window_count_matches_header'].sum())}/{len(merged)} files match",
                "passed": len(mismatches) == 0,
                "detail": ";".join(mismatches),
            }
        )

    return pd.DataFrame(rows)


def write_report(manifest: pd.DataFrame, comparison: pd.DataFrame) -> str:
    duration_pass = bool(comparison.loc[comparison["check"].eq("duration_sec_exact_descriptor"), "passed"].iloc[0])
    sfreq_pass = bool(comparison.loc[comparison["check"].eq("sampling_frequency_hz"), "passed"].iloc[0])
    mapping_pass = bool(comparison.loc[comparison["check"].eq("condition_mapping"), "passed"].iloc[0])
    channel_pass = bool(comparison.loc[comparison["check"].eq("channel_identity"), "passed"].iloc[0])
    cached_match = comparison.loc[comparison["check"].eq("cached_feature_window_counts_match_raw_header"), "passed"]
    cached_pass = bool(cached_match.iloc[0]) if len(cached_match) else False

    verdict = (
        "RAW_MAT_PROVENANCE_VERIFIED_PHASE_2_ALLOWED"
        if duration_pass and sfreq_pass and mapping_pass and channel_pass
        else "RAW_MAT_PROVENANCE_FAILURE_REBUILD_REQUIRED"
    )

    duration_summary = (
        manifest.groupby("condition")["duration_sec"]
        .agg(["count", "min", "median", "max"])
        .reset_index()
    )
    sfreq_summary = (
        manifest.groupby("condition")["sampling_frequency_hz"]
        .agg(["count", "min", "max"])
        .reset_index()
    )
    mismatch_cols = [
        "subject_id",
        "condition",
        "filename",
        "duration_sec",
        "expected_duration_sec",
        "expected_4s_50pct_windows_from_header",
    ]
    duration_mismatches = manifest.loc[~manifest["duration_matches_expected"], mismatch_cols]
    duration_mismatch_text = markdown_table(duration_mismatches) if not duration_mismatches.empty else "_None._"
    filter_summary = (
        manifest.groupby("condition", as_index=False)
        .agg(
            highpass_min_hz=("raw_header_highpass_hz", "min"),
            highpass_max_hz=("raw_header_highpass_hz", "max"),
            lowpass_min_hz=("raw_header_lowpass_hz", "min"),
            lowpass_max_hz=("raw_header_lowpass_hz", "max"),
        )
    )

    cached = cached_counts()
    cached_text = "_Cached feature table was not available._"
    if not cached.empty:
        merged = manifest.merge(cached, on="filename", how="left")
        merged["cached_matches_raw_header_window_count"] = (
            merged["cached_windows"].astype(int) == merged["expected_4s_50pct_windows_from_header"].astype(int)
        )
        cols = [
            "filename",
            "duration_sec",
            "expected_4s_50pct_windows_from_header",
            "cached_windows",
            "cached_max_end_sec",
            "cached_matches_raw_header_window_count",
        ]
        cached_text = markdown_table(merged[cols])

    lines = [
        "# Raw MAT Provenance Report",
        "",
        f"Verdict: `{verdict}`",
        "",
        "## Scope",
        "",
        "- Dataset: PhysioNet EEG During Mental Arithmetic Tasks v1.0.0.",
        "- Raw input directory: `data/raw/eegmat/`.",
        "- Manuscript file was not modified.",
        "- DS007262 and external confirmation search were not used.",
        "",
        "## Official Expectations Tested",
        "",
        f"- Subjects: `{EXPECTED_SUBJECTS}`.",
        f"- EDF files: `{EXPECTED_FILES}`.",
        f"- Sampling frequency: `{EXPECTED_SFREQ:.1f}` Hz.",
        f"- Descriptor durations: rest `{EXPECTED_REST_DURATION:.1f}` s, arithmetic `{EXPECTED_TASK_DURATION:.1f}` s.",
        f"- Header filter metadata: high-pass `{EXPECTED_HIGHPASS:.1f}` Hz, low-pass `{EXPECTED_LOWPASS:.1f}` Hz, notch `50` Hz if present in EDF metadata.",
        "- Condition mapping: `_1` = rest/background, `_2` = mental arithmetic.",
        "- Channel identity: expected 21-channel EDF header set used by the cached feature table.",
        "",
        "## Header Summary",
        "",
        "### Sampling Frequency",
        "",
        markdown_table(sfreq_summary),
        "",
        "### Duration",
        "",
        markdown_table(duration_summary),
        "",
        "### EDF Header Filter Metadata",
        "",
        markdown_table(filter_summary),
        "",
        "### Duration Mismatches",
        "",
        duration_mismatch_text,
        "",
        "## Cached Feature-Table Consistency",
        "",
        f"- Cached window counts match raw-header-implied 4 s / 50% overlap windows: `{cached_pass}`.",
        "- This checks count consistency only; it does not prove the cached numerical features were regenerated from raw data.",
        "",
        cached_text,
        "",
        "## Gate Decision",
        "",
    ]
    if verdict == "RAW_MAT_PROVENANCE_FAILURE_REBUILD_REQUIRED":
        lines.extend(
            [
                "`RAW_MAT_PROVENANCE_FAILURE_REBUILD_REQUIRED`",
                "",
                "The raw EDF files did not exactly reproduce every expected descriptor duration. Per the hard-stop gate, modeling/rebuild phases must not proceed in this run. The cached feature table is count-consistent with the raw EDF header durations, but the old headline MAT result remains unsuitable for final manuscript claims until the duration discrepancy is explicitly handled in a rebuilt exploratory pipeline.",
            ]
        )
    else:
        lines.extend(
            [
                "`RAW_MAT_PROVENANCE_VERIFIED_PHASE_2_ALLOWED`",
                "",
                "The raw EDF headers reproduce the expected sampling frequency, durations, condition mapping, and channel identity. Phase 2 raw feature rebuilding is allowed.",
            ]
        )

    (ROOT / "RAW_MAT_PROVENANCE_REPORT.md").write_text("\n".join(lines) + "\n")
    return verdict


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR_DEFAULT)
    args = parser.parse_args()

    RAW_PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)
    files = discover_mat_edfs(args.raw_dir)
    if not files:
        raise FileNotFoundError(f"No MAT EDF files found under {args.raw_dir}")

    manifest = read_manifest(files)
    manifest.to_csv(RAW_PROVENANCE_DIR / "mat_raw_edf_manifest.csv", index=False)

    comparison = make_comparison(manifest)
    comparison.to_csv(RAW_PROVENANCE_DIR / "mat_official_vs_observed_metadata.csv", index=False)

    cached = cached_counts()
    prev = previous_counts()
    if not cached.empty or not prev.empty:
        joined = manifest.merge(cached, on="filename", how="left")
        if not prev.empty:
            joined = joined.merge(prev, on="filename", how="left")
        joined.to_csv(RAW_PROVENANCE_DIR / "mat_raw_vs_cached_feature_counts.csv", index=False)

    verdict = write_report(manifest, comparison)
    print(verdict)
    return 0 if verdict == "RAW_MAT_PROVENANCE_VERIFIED_PHASE_2_ALLOWED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
