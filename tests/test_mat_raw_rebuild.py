from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RAW_MANIFEST = ROOT / "results" / "raw_provenance" / "mat_raw_edf_manifest.csv"
RAW_WINDOWS = ROOT / "results" / "raw_rebuilt" / "mat_raw_window_provenance.csv"
BALANCED_WINDOWS = ROOT / "results" / "raw_rebuilt" / "balanced_primary_window_manifest.csv"
FEATURES = ROOT / "results" / "raw_rebuilt" / "mat_no_gamma_features.parquet"


def test_condition_mapping_is_correct():
    manifest = pd.read_csv(RAW_MANIFEST)
    assert set(manifest.loc[manifest["filename"].str.endswith("_1.edf"), "condition"]) == {"rest"}
    assert set(manifest.loc[manifest["filename"].str.endswith("_2.edf"), "condition"]) == {"arithmetic"}
    assert set(manifest.loc[manifest["filename"].str.endswith("_1.edf"), "label"]) == {0}
    assert set(manifest.loc[manifest["filename"].str.endswith("_2.edf"), "label"]) == {1}


def test_sampling_frequency_is_500_hz():
    manifest = pd.read_csv(RAW_MANIFEST)
    assert manifest["sampling_frequency_hz"].nunique() == 1
    assert float(manifest["sampling_frequency_hz"].iloc[0]) == 500.0


def test_raw_windows_never_extend_beyond_source_duration():
    windows = pd.read_csv(RAW_WINDOWS)
    assert not windows["window_extends_beyond_source"].any()
    assert (windows["end_sec"] <= windows["source_duration_sec"] + 1e-9).all()


def test_balanced_calibration_never_overlaps_scored_windows():
    windows = pd.read_csv(BALANCED_WINDOWS)
    assert not windows["calibration_scoring_overlap"].any()
    for _, group in windows.groupby("subject_id"):
        calibration = group[group["segment_type"].eq("calibration")]
        scored = group[group["segment_type"].isin(["scored_rest", "scored_task"])]
        assert set(calibration["window_uid"]).isdisjoint(set(scored["window_uid"]))


def test_every_subject_has_required_balanced_segments():
    windows = pd.read_csv(BALANCED_WINDOWS)
    counts = windows.pivot_table(
        index="subject_id",
        columns="segment_type",
        values="window_uid",
        aggfunc="count",
        fill_value=0,
    )
    assert counts["calibration"].eq(14).all()
    assert counts["scored_rest"].eq(14).all()
    assert counts["scored_task"].eq(14).all()
    assert len(counts) == 36


def test_primary_feature_file_excludes_gamma_features():
    df = pd.read_parquet(FEATURES)
    feature_cols = [
        col
        for col in df.columns
        if col
        not in {
            "subject_id",
            "condition",
            "label",
            "dataset",
            "file",
            "window_index",
            "start_sec",
            "end_sec",
            "source_duration_sec",
            "window_uid",
            "channel_mapping",
        }
    ]
    assert len(feature_cols) == 184
    assert not any("_gamma" in col for col in feature_cols)
