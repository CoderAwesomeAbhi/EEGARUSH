from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eeg_cogstates.synthetic import create_synthetic_feature_csv
from src.eeg_cogstates.statistics import paired_feature_tests
from src.eeg_cogstates.modeling import train_and_evaluate
from src.eeg_cogstates.visualization import make_all_figures


def main() -> None:
    output_dir = PROJECT_ROOT / "outputs" / "synthetic_smoke_test"
    features_csv = output_dir / "features" / "synthetic_features.csv"

    print("Creating synthetic EEG-like feature table...")
    create_synthetic_feature_csv(features_csv, n_subjects=10, windows_per_subject_condition=4)

    print("Running stats...")
    paired_feature_tests(features_csv, output_dir / "statistics")

    print("Training models...")
    train_and_evaluate(
        features_csv=features_csv,
        output_dir=output_dir / "models",
        run_loso=False,
        n_boot=100,
    )

    print("Making figures...")
    make_all_figures(
        features_csv=features_csv,
        stats_csv=output_dir / "statistics" / "feature_stat_tests.csv",
        model_dir=output_dir / "models",
        output_dir=output_dir / "figures",
    )

    print(f"Smoke test complete. Check: {output_dir}")


if __name__ == "__main__":
    main()
