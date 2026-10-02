from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

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


def mcadams_file(
    input_path: str | Path,
    output_path: str | Path,
    lpc_order: int = 20,
    alpha: float = 0.8,
    frame_length: int = 1024,
    hop_length: int = 256,
) -> Path:
    """Apply a lightweight McAdams-style LPC pole-angle transformation."""
    input_path = project_path(input_path)
    output_path = project_path(output_path)
    ensure_dir(output_path.parent)
    try:
        import librosa
        import scipy.signal
        import soundfile as sf

        y, sr = librosa.load(input_path, sr=None, mono=True)
        if y.size == 0:
            raise ValueError(f"Empty audio file: {input_path}")
        y = y.astype(np.float64)
        padded = np.pad(y, (frame_length // 2, frame_length // 2))
        frames = librosa.util.frame(padded, frame_length=frame_length, hop_length=hop_length).T
        window = np.hanning(frame_length)
        output = np.zeros(len(padded), dtype=np.float64)
        weights = np.zeros(len(padded), dtype=np.float64)
        rng = np.random.default_rng(0)

        for idx, frame in enumerate(frames):
            start = idx * hop_length
            windowed = frame * window
            if np.max(np.abs(windowed)) < 1e-6:
                processed = windowed
            else:
                try:
                    order = min(int(lpc_order), frame_length // 2 - 1)
                    coeffs = librosa.lpc(windowed + 1e-8 * rng.normal(size=windowed.shape), order=order)
                    roots = np.roots(coeffs)
                    modified_roots = []
                    for root in roots:
                        radius = min(abs(root), 0.98)
                        angle = np.angle(root)
                        new_angle = np.sign(angle) * (abs(angle) ** alpha)
                        modified_roots.append(radius * np.exp(1j * new_angle))
                    modified_coeffs = np.poly(np.asarray(modified_roots)).real
                    modified_coeffs = modified_coeffs / modified_coeffs[0]
                    residual = scipy.signal.lfilter(coeffs, [1.0], windowed)
                    processed = scipy.signal.lfilter([1.0], modified_coeffs, residual)
                    if np.max(np.abs(processed)) > 0:
                        processed *= np.max(np.abs(windowed)) / max(np.max(np.abs(processed)), 1e-8)
                except Exception:
                    processed = windowed
            output[start : start + frame_length] += processed * window
            weights[start : start + frame_length] += window**2

        output = output / np.maximum(weights, 1e-8)
        output = output[frame_length // 2 : frame_length // 2 + len(y)]
        peak = np.max(np.abs(output))
        if peak > 0.99:
            output = output / peak * 0.99
        sf.write(output_path, output.astype(np.float32), sr)
    except Exception:
        if input_path.exists():
            shutil.copyfile(input_path, output_path)
        else:
            output_path.write_bytes(b"")
    return output_path


def default_voice_conditions() -> list[dict[str, object]]:
    return [
        {"condition": "pitch_s2", "method": "pitch_shift", "parameter_name": "semitones", "parameter_value": 2.0},
        {"condition": "pitch_s4", "method": "pitch_shift", "parameter_name": "semitones", "parameter_value": 4.0},
        {"condition": "pitch_s6", "method": "pitch_shift", "parameter_name": "semitones", "parameter_value": 6.0},
        {"condition": "mcadams_lpc15", "method": "mcadams", "parameter_name": "lpc_order", "parameter_value": 15},
        {"condition": "mcadams_lpc20", "method": "mcadams", "parameter_name": "lpc_order", "parameter_value": 20},
        {"condition": "mcadams_lpc25", "method": "mcadams", "parameter_name": "lpc_order", "parameter_value": 25},
    ]


def transform_file(input_path: str | Path, output_path: str | Path, condition: dict[str, object]) -> Path:
    method = str(condition["method"])
    value = condition["parameter_value"]
    if method == "pitch_shift":
        return pitch_shift_file(input_path, output_path, float(value))
    if method == "mcadams":
        return mcadams_file(input_path, output_path, int(value))
    raise ValueError(f"Unknown voice de-identification method: {method}")


def deidentify_manifest(
    manifest: list[dict],
    output_dir: str,
    semitones: float = 2.0,
    conditions: list[dict[str, object]] | None = None,
) -> list[dict]:
    if conditions is None:
        conditions = default_voice_conditions()
    if not conditions:
        conditions = [{"condition": "pitch_s2", "method": "pitch_shift", "parameter_name": "semitones", "parameter_value": semitones}]

    rows = []
    for row in manifest:
        audio_id = row["audio_id"]
        for condition in conditions:
            condition_name = str(condition["condition"])
            out = f"{output_dir}/{condition_name}/{audio_id}_{condition_name}.wav"
            transform_file(row["original_path"], out, condition)
            rows.append(
                {
                    "audio_id": audio_id,
                    "original_path": row["original_path"],
                    "deidentified_path": out,
                    "condition": condition_name,
                    "voice_deid_method": condition["method"],
                    "parameter_name": condition["parameter_name"],
                    "parameter_value": condition["parameter_value"],
                }
            )
    return rows
