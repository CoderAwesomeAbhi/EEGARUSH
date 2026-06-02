from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eeg_cogstates.dataset import download_physionet_eegmat


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PhysioNet EEGMAT dataset")
    parser.add_argument("--data_dir", type=str, default="data/raw/eegmat")
    args = parser.parse_args()

    path = download_physionet_eegmat(args.data_dir)
    print(f"Downloaded dataset to: {path}")


if __name__ == "__main__":
    main()
