from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except Exception:  # pragma: no cover - fallback for minimal Python environments.
    yaml = None


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG: dict[str, Any] = {
    "project": {"random_seed": 42, "output_dir": "data/results"},
    "audio": {"librispeech_dir": "data/raw/librispeech_test_clean/LibriSpeech/test-clean", "sample_speakers": 30, "clips_per_speaker": 5, "target_sample_rate": 16000},
    "voice_deid": {"pitch_shift_semitones": 2, "output_dir": "data/processed/audio_voice_deidentified"},
    "semantic_generation": {"n_transcripts": 150, "output_file": "data/raw/synthetic_generated/transcripts_raw.jsonl"},
    "metadata_generation": {"n_records": 150, "output_file": "data/raw/synthetic_generated/metadata_raw.jsonl"},
    "metadata_sanitization": {"target_sample_rate": 16000, "target_channels": "mono"},
}


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def ensure_dir(path: str | Path) -> Path:
    path = project_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    if yaml is None:
        return DEFAULT_CONFIG
    with project_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = project_path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    path = project_path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> Path:
    rows = list(rows)
    path = project_path(path)
    ensure_dir(path.parent)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_csv_dicts(path: str | Path) -> list[dict[str, str]]:
    path = project_path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def copy_to_manuscript(src: str | Path, dst_name: str | None = None) -> Path:
    import shutil

    src = project_path(src)
    dst = ensure_dir("outputs_for_manuscript") / (dst_name or src.name)
    shutil.copyfile(src, dst)
    return dst
