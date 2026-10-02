from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401

from src.io_utils import ensure_dir, load_config, project_path, read_csv_dicts, write_csv


def load_encoder(model_source: str):
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    return EncoderClassifier.from_hparams(
        source=model_source,
        savedir=str(project_path("pretrained_models/spkrec-ecapa-voxceleb")),
        local_strategy=LocalStrategy.COPY,
        run_opts={"device": "cpu"},
    )


def encode_file(classifier, audio_path: str) -> np.ndarray:
    import soundfile as sf

    signal, sr = sf.read(str(project_path(audio_path)), dtype="float32")
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    wav = torch.tensor(signal).unsqueeze(0)
    with torch.no_grad():
        embedding = classifier.encode_batch(wav)
    return embedding.squeeze().detach().cpu().numpy()


def save_embedding(classifier, audio_id: str, condition: str, audio_path: str, embeddings_dir: Path) -> str:
    relative_path = Path("data/processed/mappings/embeddings") / f"{audio_id}_{condition}.npy"
    embedding_path = project_path(relative_path)
    audio_file = project_path(audio_path)
    should_encode = not embedding_path.exists() or audio_file.stat().st_mtime > embedding_path.stat().st_mtime
    if should_encode:
        embedding = encode_file(classifier, audio_path)
        np.save(embedding_path, embedding)
    return str(relative_path).replace("\\", "/")


def main() -> None:
    cfg = load_config()
    mapping = read_csv_dicts("data/processed/mappings/voice_deid_mapping.csv")
    if not mapping:
        raise SystemExit("voice_deid_mapping.csv not found. Run 04_voice_deidentify.py first.")
    manifest_rows = read_csv_dicts("data/processed/audio_original/audio_manifest.csv")
    manifest = {row["audio_id"]: row for row in manifest_rows}
    embeddings_dir = ensure_dir("data/processed/mappings/embeddings")
    classifier = load_encoder(cfg["speaker_embedding"]["model_source"])

    rows = []
    for audio_id, item in manifest.items():
        embedding_path = save_embedding(classifier, audio_id, "original", item["original_path"], embeddings_dir)
        rows.append(
            {
                "audio_id": audio_id,
                "speaker_id": item.get("speaker_id", ""),
                "condition": "original",
                "voice_deid_method": "original",
                "parameter": "-",
                "audio_path": item["original_path"],
                "embedding_path": embedding_path,
            }
        )

    for row in mapping:
        audio_id = row["audio_id"]
        condition = row.get("condition", "voice_deidentified")
        parameter = f"{row.get('parameter_name', '')}={row.get('parameter_value', '')}".strip("=")
        embedding_path = save_embedding(classifier, audio_id, condition, row["deidentified_path"], embeddings_dir)
        rows.append(
            {
                "audio_id": audio_id,
                "speaker_id": manifest.get(audio_id, {}).get("speaker_id", ""),
                "condition": condition,
                "voice_deid_method": row.get("voice_deid_method", ""),
                "parameter": parameter,
                "audio_path": row["deidentified_path"],
                "embedding_path": embedding_path,
            }
        )
    write_csv(
        "data/processed/mappings/speaker_embedding_manifest.csv",
        rows,
        ["audio_id", "speaker_id", "condition", "voice_deid_method", "parameter", "audio_path", "embedding_path"],
    )
    print(f"Wrote speaker embeddings and manifest for {len(rows)} audio-condition rows.")


if __name__ == "__main__":
    main()
