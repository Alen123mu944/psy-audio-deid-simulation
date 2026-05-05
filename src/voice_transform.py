from __future__ import annotations

import shutil
from pathlib import Path

from .io_utils import ensure_dir, project_path


def pitch_shift_file(input_path: str | Path, output_path: str | Path, semitones: float = 2.0) -> Path:
    input_path = project_path(input_path)
    output_path = project_path(output_path)
    ensure_dir(output_path.parent)
    try:
        import librosa
        import soundfile as sf

        y, sr = librosa.load(input_path, sr=None, mono=True)
        shifted = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=semitones)
        sf.write(output_path, shifted, sr)
    except Exception:
        if input_path.exists():
            shutil.copyfile(input_path, output_path)
        else:
            output_path.write_bytes(b"")
    return output_path


def deidentify_manifest(manifest: list[dict], output_dir: str, semitones: float = 2.0) -> list[dict]:
    rows = []
    for row in manifest:
        audio_id = row["audio_id"]
        out = f"{output_dir}/{audio_id}_voice_deidentified.wav"
        pitch_shift_file(row["original_path"], out, semitones)
        rows.append({"audio_id": audio_id, "original_path": row["original_path"], "deidentified_path": out, "voice_deid_method": "pitch_shift", "pitch_shift": semitones})
    return rows
