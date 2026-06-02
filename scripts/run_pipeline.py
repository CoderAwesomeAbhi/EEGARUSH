from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from src.eeg_cogstates.dataset import build_feature_table, download_physionet_eegmat
from src.eeg_cogstates.modeling import train_and_evaluate
from src.eeg_cogstates.statistics import paired_feature_tests
from src.eeg_cogstates.visualization import make_all_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EEG cognitive state ML pipeline")
    parser.add_argument("--download", action="store_true", help="Download PhysioNet EEGMAT dataset")
    parser.add_argument("--data_dir", type=str, default="data/raw/eegmat", help="Directory containing EDF files")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--window_seconds", type=float, default=4.0, help="EEG window length in seconds")
    parser.add_argument("--overlap", type=float, default=0.5, help="Window overlap fraction")
    parser.add_argument("--no_connectivity", action="store_true", help="Disable pairwise channel-correlation features")
    parser.add_argument("--bandpass", action="store_true", help="Apply optional 0.5-45 Hz bandpass filter")
    parser.add_argument("--max_subjects", type=int, default=None, help="Debug option: use only first N subjects")
    parser.add_argument("--skip_loso", action="store_true", help="Skip leave-one-subject-out CV")
    parser.add_argument("--n_boot", type=int, default=500, help="Subject bootstrap repetitions")
    return parser.parse_args()


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    features_dir = output_dir / "features"
    stats_dir = output_dir / "statistics"
    model_dir = output_dir / "models"
    fig_dir = output_dir / "figures"

    features_csv = features_dir / "eeg_features.csv"

    if args.download:
        print("Downloading dataset...")
        download_physionet_eegmat(data_dir)

    print("Extracting features...")
    build_feature_table(
        data_dir=data_dir,
        output_csv=features_csv,
        window_seconds=args.window_seconds,
        overlap=args.overlap,
        include_connectivity=not args.no_connectivity,
        bandpass=(0.5, 45.0) if args.bandpass else None,
        max_subjects=args.max_subjects,
    )

    print("Running paired statistical tests...")
    paired_feature_tests(features_csv, stats_dir)
    stats_csv = stats_dir / "feature_stat_tests.csv"

    print("Training and evaluating subject-wise ML models...")
    train_and_evaluate(
        features_csv=features_csv,
        output_dir=model_dir,
        run_loso=not args.skip_loso,
        n_boot=args.n_boot,
    )

    print("Creating figures...")
    make_all_figures(
        features_csv=features_csv,
        stats_csv=stats_csv,
        model_dir=model_dir,
        output_dir=fig_dir,
    )

    print("\nDone.")
    print(f"Feature table: {features_csv}")
    print(f"Statistics: {stats_csv}")
    print(f"Model metrics: {model_dir}")
    print(f"Figures: {fig_dir}")


if __name__ == "__main__":
    main()
