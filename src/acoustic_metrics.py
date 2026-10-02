from __future__ import annotations

from pathlib import Path

import numpy as np


def extract_librosa_features(audio_path: str | Path, target_sample_rate: int = 16000) -> dict[str, float]:
    import librosa

    y, sr = librosa.load(audio_path, sr=target_sample_rate, mono=True)
    if y.size == 0:
        raise ValueError(f"Empty audio file: {audio_path}")

    duration = float(len(y) / sr)
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

    try:
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        f0 = f0[np.isfinite(f0)]
    except Exception:
        f0 = np.array([])

    voiced_threshold = max(0.005, float(np.percentile(rms, 20)))
    pause_ratio = float(np.mean(rms <= voiced_threshold)) if rms.size else 0.0

    return {
        "duration": duration,
        "RMS_energy": float(np.mean(rms)),
        "zero_crossing_rate": float(np.mean(zcr)),
        "spectral_centroid": float(np.mean(centroid)),
        "spectral_bandwidth": float(np.mean(bandwidth)),
        "F0_mean": float(np.mean(f0)) if f0.size else 0.0,
        "F0_std": float(np.std(f0)) if f0.size else 0.0,
        "pause_ratio": pause_ratio,
    }


def evaluate_feature_preservation(feature_rows: list[dict]) -> list[dict]:
    by_id: dict[str, dict[str, dict]] = {}
    for row in feature_rows:
        by_id.setdefault(str(row["audio_id"]), {})[str(row["condition"])] = row

    feature_names = [
        "duration",
        "RMS_energy",
        "zero_crossing_rate",
        "spectral_centroid",
        "spectral_bandwidth",
        "F0_mean",
        "F0_std",
        "pause_ratio",
    ]
    conditions = sorted({str(row["condition"]) for row in feature_rows if str(row["condition"]) != "original"})
    results = []
    for condition in conditions:
        for feature in feature_names:
            original = []
            deidentified = []
            for pair in by_id.values():
                if "original" in pair and condition in pair:
                    original.append(float(pair["original"][feature]))
                    deidentified.append(float(pair[condition][feature]))
            original_arr = np.asarray(original, dtype=float)
            deidentified_arr = np.asarray(deidentified, dtype=float)
            if original_arr.size < 2 or np.std(original_arr) == 0 or np.std(deidentified_arr) == 0:
                pearson_r = 0.0
            else:
                pearson_r = float(np.corrcoef(original_arr, deidentified_arr)[0, 1])
            mae = float(np.mean(np.abs(deidentified_arr - original_arr))) if original_arr.size else 0.0
            denominator = np.maximum(np.abs(original_arr), 1e-8)
            relative_change = float(np.mean(np.abs(deidentified_arr - original_arr) / denominator)) if original_arr.size else 0.0
            results.append(
                {
                    "condition": condition,
                    "feature": feature,
                    "pearson_r": pearson_r,
                    "mean_absolute_error": mae,
                    "relative_change_mean": relative_change,
                    "preservation_interpretation": "expected_change_for_voice_deid" if feature.startswith("F0") else "preserved",
                }
            )
    return results
