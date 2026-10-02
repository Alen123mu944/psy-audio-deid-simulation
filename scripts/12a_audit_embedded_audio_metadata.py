from __future__ import annotations

import _bootstrap  # noqa: F401

import re
from pathlib import Path
from typing import Any

import soundfile as sf

from src.io_utils import copy_to_manuscript, write_csv, write_jsonl


SENSITIVE_RE = re.compile(r"\b(patient|clinician|doctor|dr[_\s.-]|hospital|clinic|ward|room|\d{4}-\d{2}-\d{2}|EC_\d{3}|LM_\d{3}|OK_\d{3}|PV_\d{3}|RA_\d{3})\b", re.I)
ALLOWED_WAV_CHUNKS = {"RIFF", "WAVE", "fmt", "data", "End"}


def _bit_depth(info: sf.SoundFile) -> str:
    subtype = str(info.subtype or "")
    match = re.search(r"(\d+)", subtype)
    return match.group(1) if match else subtype


def _has_nonstandard_wav_chunks(extra_info: str) -> bool:
    chunks = []
    for line in extra_info.splitlines():
        stripped = line.strip()
        if not stripped or ":" in stripped or stripped.startswith("File") or stripped.startswith("Length"):
            continue
        token = stripped.split()[0]
        if token in {"Format", "Channels", "Sample", "Block", "Bit", "Bytes/sec"}:
            continue
        if token.isalpha() or token in {"RIFF", "WAVE"}:
            chunks.append(token)
    return any(chunk not in ALLOWED_WAV_CHUNKS for chunk in chunks)


def _audit_files(file_group: str, paths: list[Path]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    formats = set()
    sample_rates = set()
    channels = set()
    bit_depths = set()
    has_comment_block = False
    has_non_audio_chunks = False
    embedded_sensitive = False
    samples = []
    readable = 0
    for path in paths:
        try:
            info = sf.info(str(path))
        except Exception as exc:
            samples.append({"file_group": file_group, "path": str(path), "status": "error", "error": str(exc)[:300]})
            continue
        readable += 1
        extra = getattr(info, "extra_info", "") or ""
        formats.add(str(info.format))
        sample_rates.add(str(info.samplerate))
        channels.add(str(info.channels))
        bit_depths.add(_bit_depth(info))
        has_comment_block = has_comment_block or "Comment Metadata" in extra or "LIST" in extra or "INFO" in extra
        has_non_audio_chunks = has_non_audio_chunks or _has_nonstandard_wav_chunks(extra)
        embedded_sensitive = embedded_sensitive or bool(SENSITIVE_RE.search(extra))
        if len(samples) < 5:
            samples.append(
                {
                    "file_group": file_group,
                    "path": str(path),
                    "status": "ok",
                    "format": str(info.format),
                    "subtype": str(info.subtype),
                    "samplerate": info.samplerate,
                    "channels": info.channels,
                    "duration_sec": round(float(info.duration), 3),
                    "extra_info_excerpt": extra[:1200],
                }
            )
    row = {
        "file_group": file_group,
        "n_files_scanned": len(paths),
        "n_files_readable": readable,
        "formats": ";".join(sorted(formats)),
        "sample_rate_set": ";".join(sorted(sample_rates)),
        "channel_set": ";".join(sorted(channels)),
        "bit_depth_set": ";".join(sorted(bit_depths)),
        "has_comment_block": has_comment_block,
        "has_non_audio_chunks": has_non_audio_chunks,
        "embedded_sensitive_metadata_present": embedded_sensitive,
        "technical_metadata_only": readable > 0 and not embedded_sensitive,
    }
    return row, samples


def main() -> None:
    groups = {
        "raw_librispeech_flac": sorted(Path("data/raw/librispeech_test_clean/LibriSpeech/test-clean").glob("**/*.flac")),
        "processed_original_wav": sorted(Path("data/processed/audio_original").glob("*_original.wav")),
        "voice_deidentified_pitch_s4_wav": sorted(Path("data/processed/audio_voice_deidentified/pitch_s4").glob("*.wav")),
    }
    rows = []
    samples = []
    for group, paths in groups.items():
        row, sample = _audit_files(group, paths)
        rows.append(row)
        samples.extend(sample)
    write_csv(
        "data/results/tables/table_embedded_metadata_audit.csv",
        rows,
        [
            "file_group",
            "n_files_scanned",
            "n_files_readable",
            "formats",
            "sample_rate_set",
            "channel_set",
            "bit_depth_set",
            "has_comment_block",
            "has_non_audio_chunks",
            "embedded_sensitive_metadata_present",
            "technical_metadata_only",
        ],
    )
    write_jsonl("data/results/logs/embedded_metadata_audit_sample.jsonl", samples)
    copy_to_manuscript("data/results/tables/table_embedded_metadata_audit.csv", "table_embedded_metadata_audit.csv")
    print("Wrote embedded audio metadata audit table and sample log.")


if __name__ == "__main__":
    main()
