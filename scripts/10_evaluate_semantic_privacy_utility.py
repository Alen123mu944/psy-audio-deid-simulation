from __future__ import annotations

import _bootstrap  # noqa: F401

from src.io_utils import read_jsonl, write_csv
from src.semantic_metrics import evaluate_semantic


def main() -> None:
    rows = read_jsonl("data/processed/transcripts_deidentified/transcripts_deidentified.jsonl")
    if not rows:
        raise SystemExit("No deidentified transcript JSONL found. Run 09_semantic_deidentify_presidio.py first.")
    summary, by_type = evaluate_semantic(rows)
    write_csv("data/results/tables/table_semantic_layer.csv", summary, ["metric", "value"])
    write_csv("data/results/tables/table_semantic_by_type.csv", by_type, ["identifier_type", "precision", "recall", "f1", "n_gold", "n_detected", "n_true_positive"])
    print("Wrote semantic privacy and utility tables.")


if __name__ == "__main__":
    main()
