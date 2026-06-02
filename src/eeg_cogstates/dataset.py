from __future__ import annotations

import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .features import extract_window_features


PHYSIONET_BASE_URL = "https://physionet.org/files/eegmat/1.0.0/"


@dataclass
class EDFRecord:
    path: Path
    subject_id: str
    condition: str
    label: int


def download_file(url: str, dest: Path, retries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            urllib.request.urlretrieve(url, dest)
            return
        except Exception as exc:
            last_error = exc
            time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def get_physionet_records() -> List[str]:
    try:
        with urllib.request.urlopen(PHYSIONET_BASE_URL + "RECORDS", timeout=20) as response:
            text = response.read().decode("utf-8")
        records = [line.strip() for line in text.splitlines() if line.strip()]
        return records
    except Exception:
        return [f"Subject{i:02d}_{suffix}" for i in range(36) for suffix in (1, 2)]


def download_physionet_eegmat(data_dir: str | Path) -> Path:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    records = get_physionet_records()

    metadata_files = ["README.txt", "RECORDS", "SHA256SUMS.txt", "subject-info.csv"]
    for fname in metadata_files:
        download_file(PHYSIONET_BASE_URL + fname, data_dir / fname)

    for rec in tqdm(records, desc="Downloading EDF files"):
        fname = rec if rec.endswith(".edf") else rec + ".edf"
        download_file(PHYSIONET_BASE_URL + fname, data_dir / fname)

    return data_dir


def parse_edf_record(path: Path) -> Optional[EDFRecord]:
    match = re.search(r"Subject(\d+)_(\d)\.edf$", path.name, flags=re.IGNORECASE)
    if not match:
        return None
    subject_id = f"Subject{int(match.group(1)):02d}"
    suffix = int(match.group(2))
    if suffix == 1:
        condition, label = "rest", 0
    elif suffix == 2:
        condition, label = "workload", 1
    else:
        return None
    return EDFRecord(path=path, subject_id=subject_id, condition=condition, label=label)


def discover_edf_records(data_dir: str | Path, max_subjects: Optional[int] = None) -> List[EDFRecord]:
    data_dir = Path(data_dir)
    records = []
    for path in sorted(data_dir.rglob("Subject*_*.edf")):
        rec = parse_edf_record(path)
        if rec is not None:
            records.append(rec)

    if max_subjects is not None:
        keep = {f"Subject{i:02d}" for i in range(max_subjects)}
        records = [rec for rec in records if rec.subject_id in keep]

    if not records:
        raise FileNotFoundError(
            f"No EDF files found under {data_dir}. "
            "Run with --download or download the PhysioNet EEGMAT dataset first."
        )
    return records


def read_edf(
    path: str | Path,
    bandpass: Optional[Tuple[float, float]] = None,
    apply_ica: bool = False,
    ica_n_components: Optional[int] = None,
    eog_channel: Optional[str] = None,
    rereference: Optional[str] = None,
) -> Tuple[np.ndarray, float, List[str]]:
    try:
        import mne
    except ImportError as exc:
        raise ImportError("Install MNE first: pip install mne") from exc

    raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
    if bandpass is not None:
        low, high = bandpass
        raw.filter(l_freq=low, h_freq=high, verbose="ERROR")

    if rereference is not None:
        raw.set_eeg_reference(rereference, verbose="ERROR")

    if apply_ica:
        from mne.preprocessing import ICA
        n_comp = ica_n_components or min(15, len(raw.ch_names) - 1)
        ica = ICA(n_components=n_comp, method="fastica", random_state=42, max_iter="auto")
        ica.fit(raw, verbose="ERROR")
        if eog_channel is not None and eog_channel in raw.ch_names:
            eog_indices, eog_scores = ica.find_bads_eog(
                raw, ch_name=eog_channel, threshold=2.0, verbose="ERROR"
            )
            ica.exclude = eog_indices
        muscle_idx, muscle_scores = ica.find_bads_muscle(raw, verbose="ERROR")
        ica.exclude.extend(muscle_idx)
        raw = ica.apply(raw, verbose="ERROR")

    data = raw.get_data() * 1e6
    sfreq = float(raw.info["sfreq"])
    channel_names = list(raw.ch_names)
    return data, sfreq, channel_names


def apply_ica_to_subjects(
    subject_raw_data: Dict[str, mne.io.Raw],
    test_subject_id: str,
    n_components: int = 15,
    eog_channel: Optional[str] = None,
) -> Dict[str, mne.io.Raw]:
    """Fit ICA on all training subjects' concatenated data, apply to each subject."""
    import mne
    from mne.preprocessing import ICA

    train_ids = [s for s in subject_raw_data if s != test_subject_id]
    if len(train_ids) == 0:
        return subject_raw_data

    concatenated = mne.concatenate_raws(
        [subject_raw_data[sid].copy() for sid in train_ids]
    )

    ica = ICA(n_components=n_components, method="fastica", random_state=42, max_iter="auto")
    ica.fit(concatenated, verbose="ERROR")

    if eog_channel is not None:
        for sid, raw in subject_raw_data.items():
            if eog_channel in raw.ch_names:
                eog_idx, _ = ica.find_bads_eog(ch_name=eog_channel, threshold=2.0, verbose="ERROR")
                ica.exclude = eog_idx
                break

    cleaned = {}
    for sid, raw in subject_raw_data.items():
        cleaned[sid] = ica.apply(raw.copy(), verbose="ERROR")
    return cleaned


def iter_windows(data: np.ndarray, sfreq: float, window_seconds: float, overlap: float) -> Iterator[Tuple[int, int, np.ndarray]]:
    if not (0 <= overlap < 1):
        raise ValueError("overlap must be in [0, 1).")
    n_samples = data.shape[1]
    window_size = int(round(window_seconds * sfreq))
    if window_size <= 1:
        raise ValueError("window_seconds is too small.")
    step = max(1, int(round(window_size * (1 - overlap))))
    for start in range(0, n_samples - window_size + 1, step):
        end = start + window_size
        yield start, end, data[:, start:end]


def build_feature_table(
    data_dir: str | Path,
    output_csv: str | Path,
    window_seconds: float = 4.0,
    overlap: float = 0.5,
    include_connectivity: bool = True,
    bandpass: Optional[Tuple[float, float]] = None,
    max_subjects: Optional[int] = None,
) -> pd.DataFrame:
    records = discover_edf_records(data_dir, max_subjects=max_subjects)

    rows: List[Dict[str, float]] = []
    for rec in tqdm(records, desc="Extracting EEG features"):
        data, sfreq, channel_names = read_edf(rec.path, bandpass=bandpass)

        for window_index, (start, end, window) in enumerate(iter_windows(data, sfreq, window_seconds, overlap)):
            feats = extract_window_features(
                window=window,
                sfreq=sfreq,
                channel_names=channel_names,
                include_connectivity=include_connectivity,
            )
            feats.update(
                {
                    "subject_id": rec.subject_id,
                    "condition": rec.condition,
                    "label": rec.label,
                    "file": rec.path.name,
                    "window_index": window_index,
                    "start_sec": start / sfreq,
                    "end_sec": end / sfreq,
                }
            )
            rows.append(feats)

    df = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df
