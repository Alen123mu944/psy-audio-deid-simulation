from __future__ import annotations

import _bootstrap  # noqa: F401

from src.io_utils import load_config, read_csv_dicts, write_csv
from src.voice_transform import deidentify_manifest


def main() -> None:
    cfg = load_config()
    manifest = read_csv_dicts("data/processed/audio_original/audio_manifest.csv")
    if not manifest:
        raise SystemExit("audio_manifest.csv not found. Run 01_sample_audio_clips.py first.")
    rows = deidentify_manifest(manifest, cfg["voice_deid"]["output_dir"], float(cfg["voice_deid"]["pitch_shift_semitones"]))
    write_csv("data/processed/mappings/voice_deid_mapping.csv", rows, ["audio_id", "original_path", "deidentified_path", "voice_deid_method", "pitch_shift"])
    print(f"Wrote {len(rows)} de-identified audio mappings.")


if __name__ == "__main__":
    main()
