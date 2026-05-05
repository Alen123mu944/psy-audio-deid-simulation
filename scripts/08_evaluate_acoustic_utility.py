from __future__ import annotations

import _bootstrap  # noqa: F401

from src.acoustic_metrics import evaluate_feature_preservation
from src.io_utils import read_csv_dicts, write_csv


def main() -> None:
    feature_rows = read_csv_dicts("data/results/tables/acoustic_features_librosa.csv")
    if not feature_rows:
        raise SystemExit("acoustic_features_librosa.csv not found. Run 07_extract_acoustic_features.py first.")
    rows = evaluate_feature_preservation(feature_rows)
    write_csv("data/results/tables/table_voice_utility.csv", rows, ["feature", "pearson_r", "mean_absolute_error", "relative_change_mean", "preservation_interpretation"])
    print("Wrote voice acoustic utility table.")


if __name__ == "__main__":
    main()
