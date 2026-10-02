from __future__ import annotations

import _bootstrap  # noqa: F401

from src.acoustic_metrics import extract_librosa_features
from src.io_utils import load_config, project_path, read_csv_dicts, write_csv


def feature_row(audio_id: str, condition: str, path: str, target_sample_rate: int) -> dict:
    features = extract_librosa_features(project_path(path), target_sample_rate)
    row = {"audio_id": audio_id, "condition": condition}
    row.update(features)
    return row


def main() -> None:
    cfg = load_config()
    mapping = read_csv_dicts("data/processed/mappings/voice_deid_mapping.csv")
    if not mapping:
        raise SystemExit("voice_deid_mapping.csv not found. Run 04_voice_deidentify.py first.")
    manifest = read_csv_dicts("data/processed/audio_original/audio_manifest.csv")
    target_sample_rate = int(cfg["audio"]["target_sample_rate"])

    rows = []
    for item in manifest:
        rows.append(feature_row(item["audio_id"], "original", item["original_path"], target_sample_rate))
    for item in mapping:
        rows.append(feature_row(item["audio_id"], item.get("condition", "voice_deidentified"), item["deidentified_path"], target_sample_rate))

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
