from __future__ import annotations

import _bootstrap  # noqa: F401

from src.acoustic_metrics import feature_rows_for_pair
from src.io_utils import load_config, project_path, read_csv_dicts, write_csv


def main() -> None:
    cfg = load_config()
    mapping = read_csv_dicts("data/processed/mappings/voice_deid_mapping.csv")
    if not mapping:
        raise SystemExit("voice_deid_mapping.csv not found. Run 04_voice_deidentify.py first.")

    rows = []
    for item in mapping:
        rows.extend(
            feature_rows_for_pair(
                item["audio_id"],
                project_path(item["original_path"]),
                project_path(item["deidentified_path"]),
                int(cfg["audio"]["target_sample_rate"]),
            )
        )
    fieldnames = [
        "audio_id",
        "condition",
        "duration",
        "RMS_energy",
        "zero_crossing_rate",
        "spectral_centroid",
        "spectral_bandwidth",
        "F0_mean",
        "F0_std",
        "pause_ratio",
    ]
    write_csv("data/results/tables/acoustic_features_librosa.csv", rows, fieldnames)
    print(f"Wrote librosa acoustic features for {len(rows)} audio-condition rows.")


if __name__ == "__main__":
    main()
