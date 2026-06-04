from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "results" / "raw_provenance" / "mat_raw_edf_manifest.csv"
COMPARISON = ROOT / "results" / "raw_provenance" / "mat_official_vs_observed_metadata.csv"
REPORT = ROOT / "RAW_MAT_PROVENANCE_REPORT.md"


def _require_phase1_outputs() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    assert MANIFEST.exists(), "Run scripts/run_raw_mat_provenance.py before this test."
    assert COMPARISON.exists(), "Run scripts/run_raw_mat_provenance.py before this test."
    assert REPORT.exists(), "Run scripts/run_raw_mat_provenance.py before this test."
    return pd.read_csv(MANIFEST), pd.read_csv(COMPARISON), REPORT.read_text()


def _comparison_passed(comparison: pd.DataFrame, check: str) -> bool:
    row = comparison.loc[comparison["check"].eq(check)]
    assert len(row) == 1, f"Missing comparison check: {check}"
    return bool(row["passed"].iloc[0])


def test_raw_mat_sampling_rate_is_verified_from_edf_headers():
    manifest, comparison, _ = _require_phase1_outputs()
    assert _comparison_passed(comparison, "sampling_frequency_hz")
    assert manifest["sampling_frequency_hz"].nunique() == 1
    assert float(manifest["sampling_frequency_hz"].iloc[0]) == 500.0


def test_raw_mat_condition_mapping_is_verified_from_filenames():
    manifest, comparison, _ = _require_phase1_outputs()
    assert _comparison_passed(comparison, "condition_mapping")
    assert set(manifest.loc[manifest["filename"].str.contains("_1.edf"), "condition"]) == {"rest"}
    assert set(manifest.loc[manifest["filename"].str.contains("_2.edf"), "condition"]) == {"arithmetic"}


def test_raw_mat_channel_identity_is_stable_across_edf_headers():
    manifest, comparison, _ = _require_phase1_outputs()
    assert _comparison_passed(comparison, "channel_identity")
    channel_sets = manifest["channel_names"].apply(lambda value: tuple(json.loads(value)))
    assert channel_sets.nunique() == 1
    assert manifest["num_channels"].nunique() == 1
    assert int(manifest["num_channels"].iloc[0]) == 21


def test_raw_mat_duration_mismatches_trigger_hard_stop_report():
    manifest, comparison, report = _require_phase1_outputs()
    duration_passed = _comparison_passed(comparison, "duration_sec_exact_descriptor")
    if duration_passed:
        assert manifest["duration_matches_expected"].all()
        assert "RAW_MAT_PROVENANCE_VERIFIED_PHASE_2_ALLOWED" in report
    else:
        assert (~manifest["duration_matches_expected"]).any()
        assert "RAW_MAT_PROVENANCE_FAILURE_REBUILD_REQUIRED" in report
