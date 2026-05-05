from __future__ import annotations

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


def main() -> None:
    cfg = load_config()
    mapping = read_csv_dicts("data/processed/mappings/voice_deid_mapping.csv")
    if not mapping:
        raise SystemExit("voice_deid_mapping.csv not found. Run 04_voice_deidentify.py first.")
    manifest = {row["audio_id"]: row for row in read_csv_dicts("data/processed/audio_original/audio_manifest.csv")}
    embeddings_dir = ensure_dir("data/processed/mappings/embeddings")
    classifier = load_encoder(cfg["speaker_embedding"]["model_source"])

    rows = []
    for row in mapping:
        speaker_id = manifest.get(row["audio_id"], {}).get("speaker_id", "")
        for condition, audio_path in [("original", row["original_path"]), ("voice_deidentified", row["deidentified_path"])]:
            embedding_path = embeddings_dir / f"{row['audio_id']}_{condition}.npy"
            if not embedding_path.exists():
                embedding = encode_file(classifier, audio_path)
                np.save(embedding_path, embedding)
            rows.append(
                {
                    "audio_id": row["audio_id"],
                    "speaker_id": speaker_id,
                    "condition": condition,
                    "audio_path": audio_path,
                    "embedding_path": str(embedding_path),
                }
            )
    write_csv("data/processed/mappings/speaker_embedding_manifest.csv", rows, ["audio_id", "speaker_id", "condition", "audio_path", "embedding_path"])
    print(f"Wrote speaker embeddings and manifest for {len(rows)} audio-condition rows.")


if __name__ == "__main__":
    main()
