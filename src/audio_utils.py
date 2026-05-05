from __future__ import annotations

from pathlib import Path

import numpy as np

from .io_utils import ensure_dir, project_path


def discover_librispeech_files(root: str | Path) -> list[Path]:
    root = project_path(root)
    if not root.exists():
        return []
    return sorted(root.rglob("*.flac")) + sorted(root.rglob("*.wav"))


def write_synthetic_tone(path: str | Path, duration_sec: float = 8.0, sample_rate: int = 16000, frequency: float = 220.0) -> Path:
    path = project_path(path)
    ensure_dir(path.parent)
    try:
        import soundfile as sf

        t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
        signal = 0.2 * np.sin(2 * np.pi * frequency * t)
        sf.write(path, signal, sample_rate)
    except Exception:
        path.write_bytes(b"")
    return path


def make_placeholder_audio_manifest(n_speakers: int = 20, clips_per_speaker: int = 5) -> list[dict]:
    rows = []
    for sidx in range(1, n_speakers + 1):
        for cidx in range(1, clips_per_speaker + 1):
            audio_id = f"S{sidx:03d}_C{cidx:03d}"
            path = f"data/processed/audio_original/{audio_id}_original.wav"
            write_synthetic_tone(path, frequency=180 + sidx * 3 + cidx)
            rows.append({"audio_id": audio_id, "speaker_id": f"S{sidx:03d}", "original_path": path, "duration_sec": 8.0, "sample_rate": 16000})
    return rows
