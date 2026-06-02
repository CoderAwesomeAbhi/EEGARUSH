from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy.testing import assert_almost_equal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eeg_cogstates.features import (
    BANDS,
    compute_psd,
    hjorth_parameters,
    bandpower,
    spectral_entropy,
    safe_entropy_from_values,
    extract_channel_features,
)


FS = 256.0


def test_band_definitions():
    """EP7: Verify frequency band boundaries match paper specification."""
    assert BANDS["delta"] == (0.5, 4.0)
    assert BANDS["theta"] == (4.0, 8.0)
    assert BANDS["alpha"] == (8.0, 13.0)
    assert BANDS["beta"] == (13.0, 30.0)
    assert BANDS["gamma"] == (30.0, 45.0)


def test_compute_psd_shape():
    """PSD computation returns expected output length."""
    x = np.random.randn(int(FS * 4))
    freqs, psd = compute_psd(x, sfreq=FS)
    assert len(freqs) > 1
    assert len(psd) == len(freqs)


def test_hjorth_parameters_pure_sine():
    """Hjorth parameters for a pure sine wave: mobility ~ frequency, complexity ~ 1."""
    t = np.linspace(0, 2, int(FS * 2), endpoint=False)
    freq_hz = 10.0
    x = np.sin(2 * np.pi * freq_hz * t)
    activity, mobility, complexity = hjorth_parameters(x)
    assert activity > 0
    assert mobility > 0
    assert 0.9 < complexity < 1.1


def test_hjorth_parameters_constant():
    """Constant signal: activity ~ 0, mobility ~ 0, complexity ~ 0."""
    x = np.ones(int(FS))
    activity, mobility, complexity = hjorth_parameters(x)
    assert_almost_equal(activity, 0.0, decimal=6)
    assert_almost_equal(mobility, 0.0, decimal=6)
    assert_almost_equal(complexity, 0.0, decimal=6)


def test_bandpower_total():
    """Sum of bandpowers across all bands should approximately equal total PSD power."""
    x = np.random.randn(int(FS * 4))
    freqs, psd = compute_psd(x, sfreq=FS)
    total = 0.0
    for band_name, (lo, hi) in BANDS.items():
        total += bandpower(freqs, psd, lo, hi)
    # Total power in [0.5, 45] should be less than full PSD integral (includes DC)
    full_low = min(freqs[freqs >= 0.5][0] if any(freqs >= 0.5) else freqs[0], freqs[0])
    full_high = max(freqs[freqs <= 45.0][-1] if any(freqs <= 45.0) else freqs[-1], freqs[-1])
    band_idx = (freqs >= 0.5) & (freqs <= 45.0)
    psd_total = np.trapezoid(psd[band_idx], freqs[band_idx])
    assert total <= psd_total * 1.01  # allow 1% numerical error


def test_theta_bandpower():
    """Theta bandpower from a pure 6 Hz signal concentrates in the theta band."""
    t = np.linspace(0, 4, int(FS * 4), endpoint=False)
    x = np.sin(2 * np.pi * 6.0 * t)
    freqs, psd = compute_psd(x, sfreq=FS)
    theta_power = bandpower(freqs, psd, 4.0, 8.0)
    total_power = bandpower(freqs, psd, 0.5, 45.0)
    ratio = theta_power / (total_power + 1e-12)
    assert ratio > 0.5, f"Theta ratio {ratio:.3f} should dominate for 6 Hz input"


def test_spectral_entropy_uniform():
    """Uniform PSD gives spectral entropy near 1.0 (normalized)."""
    psd = np.ones(100)
    ent = spectral_entropy(psd)
    assert ent > 0.99, f"Entropy {ent:.3f} too low for uniform PSD (should be ~1.0)"


def test_spectral_entropy_delta():
    """Single non-zero frequency gives near-zero entropy."""
    psd = np.zeros(100)
    psd[50] = 1.0
    ent = spectral_entropy(psd)
    assert ent < 0.5, f"Entropy {ent:.3f} should be low for delta PSD"


def test_extract_channel_features_output_keys():
    """Channel feature extraction returns expected feature types."""
    x = np.random.randn(int(FS * 4))
    features = extract_channel_features(x, sfreq=FS, prefix="Fz")
    assert "stat_Fz_mean" in features
    assert "stat_Fz_std" in features
    assert "stat_Fz_var" in features
    assert "stat_Fz_rms" in features
    assert "stat_Fz_ptp" in features
    assert "stat_Fz_skew" in features
    assert "stat_Fz_kurtosis" in features
    assert "stat_Fz_shannon_entropy" in features
    assert "hjorth_Fz_activity" in features
    assert "hjorth_Fz_mobility" in features
    assert "hjorth_Fz_complexity" in features
    assert "spectral_Fz_entropy" in features
    for band in ["delta", "theta", "alpha", "beta", "gamma"]:
        assert f"band_abs_Fz_{band}" in features
        assert f"band_rel_Fz_{band}" in features
    assert "ratio_Fz_theta_alpha" in features
    assert "ratio_Fz_beta_alpha" in features
    assert "ratio_Fz_theta_beta" in features


def test_mean_computation():
    """Mean of random signal should be close to zero."""
    x = np.random.randn(int(FS * 4))
    features = extract_channel_features(x, sfreq=FS, prefix="Cz")
    assert abs(features["stat_Cz_mean"]) < 0.5


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
