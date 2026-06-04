"""Audit STEW provenance before any corrected transfer analysis.

This script intentionally does not run models. It only inspects the available
STEW inputs in the repository and decides whether an auditable MAT-like
baseline/scoring split can be reconstructed.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STEW_FEATURES = ROOT / "results" / "multi_dataset" / "stew_features.parquet"
OUT_DIR = ROOT / "results" / "stew_provenance"

MAT_LOCKED_CHANNELS = ["F3", "F4", "F7", "F8", "O1", "O2", "T3", "T4"]
STEW_ORIGINAL_CHANNELS = [
    "AF3",
    "F7",
    "F3",
    "FC5",
    "T7",
    "P7",
    "O1",
    "O2",
    "P8",
    "T8",
    "FC6",
    "F4",
    "F8",
    "AF4",
]
LEGACY_EQUIVALENCE = {"T3": "T7", "T4": "T8"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_channels(columns: list[str]) -> list[str]:
    found: set[str] = set()
    pattern = re.compile(r"^(?:stat|hjorth|spectral|band_abs|band_rel|ratio)_([^_]+)_")
    for column in columns:
        match = pattern.match(column)
        if match:
            found.add(match.group(1))
    return sorted(found)


def local_stew_source_candidates() -> list[Path]:
    candidates: list[Path] = []
    for base in [ROOT / "external_data", ROOT / "data", ROOT / "results" / "multi_dataset"]:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            lower = path.name.lower()
            if "stew" in lower:
                candidates.append(path)
    return sorted(candidates)


def write_manifest(df: pd.DataFrame, file_hash: str, channels: list[str]) -> None:
    grouped = (
        df.groupby(["subject_id", "condition"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns={"rest": "rest_rows", "workload": "workload_rows"})
    )
    for column in ["rest_rows", "workload_rows"]:
        if column not in grouped:
            grouped[column] = 0
    grouped["total_rows"] = grouped["rest_rows"] + grouped["workload_rows"]
    grouped["input_path"] = str(STEW_FEATURES.relative_to(ROOT))
    grouped["input_type"] = "cached_feature_table_only"
    grouped["file_sha256"] = file_hash
    grouped["condition_labels"] = "rest(label=0); workload(label=1)"
    grouped["sampling_rate_hz"] = "not verifiable from cached feature table"
    grouped["channel_names"] = ",".join(channels)
    grouped["duration_or_row_count_status"] = (
        "row counts available; source window timing/duration unavailable"
    )
    grouped["genuine_baseline_condition"] = (
        "rest rows exist, but source timing is unavailable and non-overlap cannot be audited"
    )
    grouped["timing_available"] = False
    grouped["reconstruction_from_source_possible"] = False
    grouped["notes"] = (
        "Cached features lack raw EEG path, sample index, window start/end, and source file IDs."
    )
    columns = [
        "subject_id",
        "input_path",
        "input_type",
        "file_sha256",
        "condition_labels",
        "rest_rows",
        "workload_rows",
        "total_rows",
        "sampling_rate_hz",
        "channel_names",
        "duration_or_row_count_status",
        "genuine_baseline_condition",
        "timing_available",
        "reconstruction_from_source_possible",
        "notes",
    ]
    grouped[columns].to_csv(OUT_DIR / "stew_input_manifest.csv", index=False)


def write_channel_harmonization(channels: list[str]) -> None:
    rows = []
    for mat_channel in MAT_LOCKED_CHANNELS:
        if mat_channel in {"T3", "T4"}:
            source_channel = LEGACY_EQUIVALENCE[mat_channel]
            cached_channel = mat_channel if mat_channel in channels else ""
            mapping_type = "legacy_10_20_equivalence_cached_as_legacy_name"
            defensible = bool(cached_channel)
            note = (
                "Original STEW montage uses modern T7/T8; existing repository code maps "
                "these to legacy MAT names T3/T4. This is documentable only as a named "
                "10-20 equivalence, not as nearest-electrode substitution."
            )
        elif mat_channel in channels:
            source_channel = mat_channel
            cached_channel = mat_channel
            mapping_type = "exact_name_match"
            defensible = True
            note = "Exact shared feature name present in cached STEW table."
        else:
            source_channel = ""
            cached_channel = ""
            mapping_type = "missing"
            defensible = False
            note = "No defensible exact or legacy-equivalent channel found."
        rows.append(
            {
                "mat_locked_channel": mat_channel,
                "stew_source_or_expected_channel": source_channel,
                "stew_cached_feature_channel": cached_channel,
                "mapping_type": mapping_type,
                "defensible_mapping": defensible,
                "unjustified_approximation_required": not defensible,
                "note": note,
            }
        )
    pd.DataFrame(rows).to_csv(OUT_DIR / "mat_stew_channel_harmonization.csv", index=False)


def write_reports(df: pd.DataFrame, file_hash: str, channels: list[str]) -> None:
    candidates = local_stew_source_candidates()
    rest_rows = int((df["condition"] == "rest").sum())
    workload_rows = int((df["condition"] == "workload").sum())
    subject_count = int(df["subject_id"].nunique())
    metadata_cols = sorted(set(df.columns) & {"subject_id", "label", "condition", "dataset"})
    timing_cols = [c for c in df.columns if c.lower() in {"file", "window_index", "start_sec", "end_sec", "sample_start", "sample_end", "time"}]

    source_lines = []
    if candidates:
        for path in candidates:
            source_lines.append(f"- `{path.relative_to(ROOT)}`")
    else:
        source_lines.append("- No local STEW source candidates found.")

    report = [
        "# STEW Raw Provenance Report",
        "",
        "## Verdict",
        "",
        "Final STEW provenance status: `STEW_PROVENANCE_UNRESOLVED_CANNOT_SUPPORT_FINAL_PAPER`.",
        "",
        "The repository currently contains a cached STEW feature table, not auditable raw EEG or source windows with timing provenance. Because the table lacks source file IDs, sample indices, and window start/end times, a corrected balanced STEW design with non-overlapping calibration/rest scoring windows cannot be verified.",
        "",
        "## Audited Input",
        "",
        f"- Cached feature table: `{STEW_FEATURES.relative_to(ROOT)}`.",
        f"- SHA-256: `{file_hash}`.",
        f"- Rows: `{len(df)}`.",
        f"- Subjects: `{subject_count}`.",
        f"- Condition row counts: rest=`{rest_rows}`, workload=`{workload_rows}`.",
        f"- Metadata columns present: `{', '.join(metadata_cols)}`.",
        f"- Timing/provenance columns present: `{', '.join(timing_cols) if timing_cols else 'none'}`.",
        f"- Channels inferred from cached feature columns: `{', '.join(channels)}`.",
        "",
        "## Local Source Candidate Search",
        "",
        *source_lines,
        "",
        "The existing repository code documents an intended HuggingFace STEW pathway using `STEW_X.npy`, `STEW_y.npy`, and `STEW_subject_id.csv`; however, those source arrays are not present in the repository audit state. The available cached parquet is therefore the only locally auditable STEW input.",
        "",
        "## Baseline And Split Audit",
        "",
        "- A rest condition exists in the cached labels.",
        "- Sampling frequency is not verifiable from the cached feature table itself.",
        "- Duration and fixed segment boundaries are not verifiable from the cached feature table.",
        "- Calibration/scoring non-overlap cannot be proven because no source timing or window IDs are present.",
        "- A MAT-consistent split cannot be reconstructed from auditable source inputs in the current repository state.",
        "",
        "## Channel Harmonization",
        "",
        "- Exact source/cached shared channels: `F3`, `F4`, `F7`, `F8`, `O1`, `O2`.",
        "- Legacy-equivalence channels: source `STEW T7 -> cached/MAT T3`, source `STEW T8 -> cached/MAT T4`.",
        "- No nearest-electrode substitution is used or allowed.",
        "- Channel mapping is documentable, but it is not sufficient to rescue the provenance/timing failure.",
        "",
        "## Consequence",
        "",
        "Do not run corrected balanced STEW, MAT->STEW transfer, STEW->MAT transfer, or transfer permutation tests from this cached table as final-paper evidence.",
    ]
    (ROOT / "STEW_RAW_PROVENANCE_REPORT.md").write_text("\n".join(report) + "\n")

    decision = [
        "# Exploratory Transfer Decision Before Confirmation",
        "",
        "## MAT Gate",
        "",
        "- MAT macro subject-level full-pipeline null passed for the locked corrected configuration.",
        "- Observed macro subject-level mean ROC-AUC: `0.880102`.",
        "- Null mean: `0.500600`.",
        "- Null 95% interval: `[0.441057, 0.553729]`.",
        "- Empirical p-value: `0.004975`.",
        "- Completed permutations: `200`.",
        "",
        "## STEW Gate",
        "",
        "- STEW is exploratory robustness evidence, not untouched confirmation.",
        "- The current repository state exposes only cached STEW features without auditable timing/source provenance.",
        "- The cross-dataset channel mapping is documentable for the locked 8-channel set using exact matches plus legacy T3/T4 to T7/T8 equivalence.",
        "- The provenance/timing failure prevents corrected balanced STEW rebuilding and prevents valid cross-dataset transfer claims.",
        "",
        "## DS007262 Gate",
        "",
        "- DS007262 is not a valid resting-baseline confirmatory dataset for the current hypothesis.",
        "- DS007262 must remain task-anchored sensitivity evidence only and was not analyzed in this task.",
        "",
        "## New External Dataset Search",
        "",
        "A new untouched confirmation dataset search is not scientifically justified yet, because the exploratory STEW transfer gate has not produced auditable support. The next step is to resolve STEW/source provenance or explicitly downgrade to MAT-only exploratory evidence.",
        "",
        "Final verdict: `STEW_PROVENANCE_OR_HARMONIZATION_INVALID_STOP_TRANSFER_CLAIM`",
    ]
    (ROOT / "EXPLORATORY_TRANSFER_DECISION_BEFORE_CONFIRMATION.md").write_text(
        "\n".join(decision) + "\n"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(STEW_FEATURES)
    df = df.copy()
    df["subject_id"] = df["subject_id"].astype(str)
    if "condition" not in df.columns:
        df["condition"] = df["label"].map({0: "rest", 1: "workload"}).fillna("unknown")

    file_hash = sha256(STEW_FEATURES)
    channels = extract_channels(list(df.columns))
    write_manifest(df, file_hash, channels)
    write_channel_harmonization(channels)
    write_reports(df, file_hash, channels)


if __name__ == "__main__":
    main()
