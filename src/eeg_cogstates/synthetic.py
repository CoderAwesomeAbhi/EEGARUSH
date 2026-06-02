from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .features import extract_window_features


def synth_signal(
    sfreq: float,
    seconds: float,
    condition: str,
    rng: np.random.Generator,
    n_channels: int = 8,
) -> np.ndarray:
    t = np.arange(0, seconds, 1 / sfreq)
    data = []
    for ch in range(n_channels):
        noise = rng.normal(0, 0.6, size=t.size)
        phase = rng.uniform(0, 2 * np.pi)

        alpha_amp = 2.4 if condition == "rest" else 1.3
        theta_amp = 1.0 if condition == "rest" else 2.0
        beta_amp = 0.8 if condition == "rest" else 1.6

        x = (
            alpha_amp * np.sin(2 * np.pi * 10 * t + phase)
            + theta_amp * np.sin(2 * np.pi * 6 * t + phase / 2)
            + beta_amp * np.sin(2 * np.pi * 18 * t + phase / 3)
            + noise
        )
        data.append(x)
    return np.asarray(data)


def create_synthetic_feature_csv(
    output_csv: str | Path,
    n_subjects: int = 10,
    sfreq: float = 128.0,
    seconds: float = 8.0,
    windows_per_subject_condition: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    channel_names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4"]

    rows = []
    for subject in tqdm(range(n_subjects), desc="Creating synthetic EEG features"):
        for condition, label in [("rest", 0), ("workload", 1)]:
            for window_idx in range(windows_per_subject_condition):
                data = synth_signal(sfreq, seconds, condition, rng, n_channels=len(channel_names))
                feats = extract_window_features(data, sfreq, channel_names, include_connectivity=True)
                feats.update(
                    {
                        "subject_id": f"Synthetic{subject:02d}",
                        "condition": condition,
                        "label": label,
                        "file": "synthetic",
                        "window_index": window_idx,
                        "start_sec": 0.0,
                        "end_sec": seconds,
                    }
                )
                rows.append(feats)

    df = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df
