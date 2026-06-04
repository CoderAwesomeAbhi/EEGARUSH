#!/usr/bin/env python3
"""Provenance audit of the OFFICIAL IEEE DataPort STEW archive.

Audit-only. No models are trained. No raw data is committed.

Reads the locally-extracted official archive under
    data/raw/stew/ieee_dataport_stew/STEW Dataset/
and emits committable provenance summaries under
    results/stew_ieee_provenance/

Outputs:
  * stew_ieee_source_manifest.csv          (one row per source file)
  * stew_condition_subject_summary.csv     (one row per subject)
  * mat_stew_verified_channel_harmonization.csv  (MAT<->STEW channel mapping)

Sampling rate and channel labels are NOT embedded in the STEW .txt files;
they are taken from the documented Emotiv EPOC specification and corroborated
here against the verified sample counts (19200 samples / 128 Hz = 150 s).
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STEW_DIR = ROOT / "data" / "raw" / "stew" / "ieee_dataport_stew" / "STEW Dataset"
MAT_MANIFEST = ROOT / "results" / "raw_provenance" / "mat_raw_edf_manifest.csv"
OUT_DIR = ROOT / "results" / "stew_ieee_provenance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Documented Emotiv EPOC (14ch) channel order — NOT embedded in the .txt files.
STEW_EMOTIV_ORDER = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]
DOCUMENTED_SFREQ_HZ = 128.0  # Emotiv EPOC documented rate (corroborated below)

# Old<->new 10-20 nomenclature equivalences used for harmonization.
TEN_TWENTY_EQUIV = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}

SUB_RE = re.compile(r"^sub(\d+)_(lo|hi)\.txt$")
CONDITION_MAP = {"lo": "rest_no_task", "hi": "high_workload_simkap"}
LABEL_MAP = {"lo": 0, "hi": 1}


def scan_file(path: Path) -> tuple[int, int]:
    """Return (n_rows, n_cols_first_row) reading the whitespace-delimited file."""
    n_rows = 0
    n_cols = -1
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            if n_cols < 0:
                n_cols = len(line.split())
            n_rows += 1
    return n_rows, n_cols


def load_ratings() -> dict[int, dict[str, int]]:
    ratings: dict[int, dict[str, int]] = {}
    rpath = STEW_DIR / "ratings.txt"
    if not rpath.exists():
        return ratings
    with rpath.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                sid = int(parts[0])
                ratings[sid] = {"rating_lo": int(parts[1]), "rating_hi": int(parts[2])}
    return ratings


def main() -> None:
    assert STEW_DIR.exists(), f"STEW archive not found at {STEW_DIR}"
    ratings = load_ratings()

    data_files = sorted(
        p for p in STEW_DIR.glob("*.txt") if SUB_RE.match(p.name)
    )

    manifest_rows = []
    per_subject: dict[int, dict] = {}
    for path in data_files:
        m = SUB_RE.match(path.name)
        sid = int(m.group(1))
        cond_key = m.group(2)
        n_rows, n_cols = scan_file(path)
        duration_sec = n_rows / DOCUMENTED_SFREQ_HZ
        manifest_rows.append({
            "filename": path.name,
            "subject_id": f"sub{sid:02d}",
            "subject_num": sid,
            "condition_key": cond_key,
            "condition": CONDITION_MAP[cond_key],
            "label": LABEL_MAP[cond_key],
            "n_samples": n_rows,
            "n_channels": n_cols,
            "documented_sfreq_hz": DOCUMENTED_SFREQ_HZ,
            "implied_duration_sec": round(duration_sec, 4),
            "channel_labels_embedded": False,
            "sfreq_embedded": False,
            "bytes": path.stat().st_size,
        })
        s = per_subject.setdefault(sid, {"subject_num": sid})
        s[f"{cond_key}_present"] = True
        s[f"{cond_key}_samples"] = n_rows
        s[f"{cond_key}_channels"] = n_cols

    # ---- manifest CSV ----
    man_cols = list(manifest_rows[0].keys())
    with (OUT_DIR / "stew_ieee_source_manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=man_cols)
        w.writeheader()
        w.writerows(manifest_rows)

    # ---- per-subject / condition summary CSV ----
    sub_rows = []
    for sid in sorted(per_subject):
        s = per_subject[sid]
        r = ratings.get(sid, {})
        sub_rows.append({
            "subject_id": f"sub{sid:02d}",
            "subject_num": sid,
            "rest_no_task_file": f"sub{sid:02d}_lo.txt" if s.get("lo_present") else "",
            "high_workload_file": f"sub{sid:02d}_hi.txt" if s.get("hi_present") else "",
            "rest_present": bool(s.get("lo_present")),
            "task_present": bool(s.get("hi_present")),
            "rest_samples": s.get("lo_samples", 0),
            "task_samples": s.get("hi_samples", 0),
            "n_channels": s.get("lo_channels") or s.get("hi_channels"),
            "rating_lo_rest": r.get("rating_lo", ""),
            "rating_hi_task": r.get("rating_hi", ""),
            "rating_present": sid in ratings,
            "rating_monotonic_lo_lt_hi": (
                (r["rating_lo"] < r["rating_hi"]) if r else ""
            ),
        })
    with (OUT_DIR / "stew_condition_subject_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sub_rows[0].keys()))
        w.writeheader()
        w.writerows(sub_rows)

    # ---- MAT/STEW channel harmonization CSV ----
    with MAT_MANIFEST.open() as f:
        mat0 = next(csv.DictReader(f))
    mat_channels_raw = json.loads(mat0["channel_names"])
    # Normalise MAT EEG labels: strip "EEG " prefix, drop non-scalp.
    mat_scalp = []
    for ch in mat_channels_raw:
        name = ch.replace("EEG ", "").strip()
        if name in ("A2-A1",) or name.startswith("ECG"):
            continue
        mat_scalp.append(name)
    mat_set = set(mat_scalp)

    harm_rows = []
    for ch in STEW_EMOTIV_ORDER:
        mapped = ch
        basis = "direct_name_match"
        if ch not in mat_set and ch in TEN_TWENTY_EQUIV:
            mapped = TEN_TWENTY_EQUIV[ch]
            basis = "documented_10-20_equivalence"
        in_mat = mapped in mat_set
        harm_rows.append({
            "stew_channel_documented": ch,
            "mapped_mat_channel": mapped if in_mat else "",
            "mapping_basis": basis if in_mat else "no_mat_counterpart",
            "present_in_mat": in_mat,
            "harmonizable": in_mat,
            "stew_label_embedded_in_source": False,
        })
    harmonizable = [r for r in harm_rows if r["harmonizable"]]
    with (OUT_DIR / "mat_stew_verified_channel_harmonization.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(harm_rows[0].keys()))
        w.writeheader()
        w.writerows(harm_rows)

    # ---- console summary ----
    n_sub = len(per_subject)
    both = sum(1 for s in per_subject.values() if s.get("lo_present") and s.get("hi_present"))
    all_19200 = all(r["n_samples"] == 19200 for r in manifest_rows)
    all_14ch = all(r["n_channels"] == 14 for r in manifest_rows)
    mono = [r for r in sub_rows if r["rating_monotonic_lo_lt_hi"] is True]
    print("STEW IEEE PROVENANCE AUDIT")
    print(f"  subjects: {n_sub}; with both rest+task: {both}")
    print(f"  data files: {len(manifest_rows)} (rest+task)")
    print(f"  all files 19200 samples: {all_19200}; all 14 channels: {all_14ch}")
    print(f"  documented sfreq {DOCUMENTED_SFREQ_HZ} Hz -> {19200/DOCUMENTED_SFREQ_HZ:.1f} s per file")
    print(f"  ratings present: {sum(1 for s in sub_rows if s['rating_present'])}/{n_sub}; "
          f"lo<hi monotonic: {len(mono)}/{sum(1 for s in sub_rows if s['rating_present'])}")
    print(f"  MAT scalp EEG channels: {len(mat_scalp)} -> {mat_scalp}")
    print(f"  STEW channels: {len(STEW_EMOTIV_ORDER)}; harmonizable with MAT: {len(harmonizable)}")
    print(f"  harmonizable channels: {[r['mapped_mat_channel'] for r in harmonizable]}")
    print(f"  STEW-only (no MAT counterpart): "
          f"{[r['stew_channel_documented'] for r in harm_rows if not r['harmonizable']]}")
    print("  wrote 3 CSVs to results/stew_ieee_provenance/")


if __name__ == "__main__":
    main()
