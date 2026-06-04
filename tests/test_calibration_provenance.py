from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_mat_calibration_and_scoring_windows_do_not_overlap():
    provenance = pd.read_csv(ROOT / "results" / "audit" / "mat_window_provenance.csv")
    assert not provenance["calibration_scoring_overlap"].any()


def test_mat_scored_set_contains_rest_and_task_windows():
    provenance = pd.read_csv(ROOT / "results" / "audit" / "mat_window_provenance.csv")
    scored = provenance[provenance["used_for_scoring"]]
    assert set(scored["label"].unique()) == {0, 1}


def test_mat_calibration_uses_rest_windows_only():
    provenance = pd.read_csv(ROOT / "results" / "audit" / "mat_window_provenance.csv")
    calibration = provenance[provenance["used_for_calibration"]]
    assert set(calibration["label"].unique()) == {0}
