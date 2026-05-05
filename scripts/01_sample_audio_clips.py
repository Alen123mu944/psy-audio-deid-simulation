from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401

from src.audio_utils import discover_librispeech_files, make_placeholder_audio_manifest
from src.io_utils import ensure_dir, load_config, project_path, write_csv


def get_duration_sec(path: Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def convert_to_wav(source_path: Path, output_path: Path, target_sample_rate: int) -> float:
    import librosa
    import soundfile as sf

    y, sr = librosa.load(source_path, sr=target_sample_rate, mono=True)
    sf.write(output_path, y, sr)
    return len(y) / float(sr)


def speaker_id_from_librispeech_path(path: Path) -> str:
    # LibriSpeech layout: test-clean/<speaker_id>/<chapter_id>/<utterance>.flac
    return path.parent.parent.name


def sample_real_librispeech(cfg: dict) -> list[dict]:
    audio_cfg = cfg["audio"]
    rng = random.Random(cfg["project"]["random_seed"])
    files = discover_librispeech_files(audio_cfg["librispeech_dir"])
    min_duration = float(audio_cfg["min_duration_sec"])
    max_duration = float(audio_cfg["max_duration_sec"])
    clips_per_speaker = int(audio_cfg["clips_per_speaker"])
    sample_speakers = int(audio_cfg["sample_speakers"])
    target_sample_rate = int(audio_cfg["target_sample_rate"])

    eligible_by_speaker: dict[str, list[tuple[Path, float]]] = defaultdict(list)
    for path in files:
        try:
            duration = get_duration_sec(path)
        except Exception as exc:
            print(f"Skipping unreadable file {path}: {exc}")
            continue
        if min_duration <= duration <= max_duration:
            eligible_by_speaker[speaker_id_from_librispeech_path(path)].append((path, duration))

    eligible_speakers = sorted(
        speaker for speaker, speaker_files in eligible_by_speaker.items() if len(speaker_files) >= clips_per_speaker
    )
    if len(eligible_speakers) < sample_speakers:
        raise RuntimeError(
            f"Only {len(eligible_speakers)} speakers have at least {clips_per_speaker} clips "
            f"within {min_duration}-{max_duration} seconds; requested {sample_speakers}."
        )

    selected_speakers = rng.sample(eligible_speakers, sample_speakers)
    output_dir = ensure_dir("data/processed/audio_original")
    for old_wav in output_dir.glob("*_original.wav"):
        old_wav.unlink()

    rows = []
    for speaker_index, speaker_id in enumerate(sorted(selected_speakers), start=1):
        selected_clips = rng.sample(sorted(eligible_by_speaker[speaker_id], key=lambda item: str(item[0])), clips_per_speaker)
        for clip_index, (source_path, source_duration) in enumerate(selected_clips, start=1):
            audio_id = f"S{speaker_index:03d}_C{clip_index:03d}"
            relative_output = Path("data/processed/audio_original") / f"{audio_id}_original.wav"
            output_path = project_path(relative_output)
            converted_duration = convert_to_wav(source_path, output_path, target_sample_rate)
            rows.append(
                {
                    "audio_id": audio_id,
                    "speaker_id": speaker_id,
                    "original_path": str(relative_output).replace("\\", "/"),
                    "source_path": str(source_path).replace("\\", "/"),
                    "duration_sec": round(converted_duration, 3),
                    "source_duration_sec": round(source_duration, 3),
                    "sample_rate": target_sample_rate,
                }
            )
    return rows


def main() -> None:
    cfg = load_config()
    files = discover_librispeech_files(cfg["audio"]["librispeech_dir"])
    if not files:
        rows = make_placeholder_audio_manifest(cfg["audio"]["sample_speakers"], cfg["audio"]["clips_per_speaker"])
    else:
        rows = sample_real_librispeech(cfg)
    write_csv(
        "data/processed/audio_original/audio_manifest.csv",
        rows,
        ["audio_id", "speaker_id", "original_path", "source_path", "duration_sec", "source_duration_sec", "sample_rate"],
    )
    print(f"Wrote {len(rows)} manifest rows.")


if __name__ == "__main__":
    main()
