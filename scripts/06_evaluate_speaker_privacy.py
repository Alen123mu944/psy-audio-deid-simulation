from __future__ import annotations

import _bootstrap  # noqa: F401

from src.io_utils import read_csv_dicts, write_csv
from src.speaker_metrics import evaluate_embeddings


def main() -> None:
    rows = read_csv_dicts("data/processed/mappings/speaker_embedding_manifest.csv")
    if not rows:
        raise SystemExit("speaker_embedding_manifest.csv not found. Run 05_extract_speaker_embeddings.py first.")
    summary, similarities = evaluate_embeddings(rows)
    write_csv(
        "data/results/tables/table_voice_privacy.csv",
        summary,
        ["condition", "method", "parameter", "mean_cosine_similarity", "top1_speaker_id_accuracy", "linkage_success_rate", "speaker_verification_eer", "n_enrollment_clips", "n_test_clips", "n_genuine_trials", "n_impostor_trials"],
    )
    write_csv("data/results/tables/speaker_similarity_distributions.csv", similarities, ["condition", "cosine_similarity"])
    print("Wrote real speaker privacy tables from SpeechBrain embeddings, including EER.")


if __name__ == "__main__":
    main()
