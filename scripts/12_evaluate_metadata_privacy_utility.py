from __future__ import annotations

import _bootstrap  # noqa: F401

from src.io_utils import read_jsonl, write_csv
from src.metadata_metrics import evaluate_metadata


def main() -> None:
    raw = read_jsonl("data/raw/synthetic_generated/metadata_reviewed.jsonl") or read_jsonl("data/raw/synthetic_generated/metadata_raw.jsonl")
    sanitized = read_jsonl("data/processed/metadata_sanitized/metadata_sanitized.jsonl")
    if not raw or not sanitized:
        raise SystemExit("Metadata inputs missing. Run 03_generate_synthetic_metadata_openai.py and 11_metadata_sanitize.py first.")
    rows = evaluate_metadata(raw, sanitized)
    write_csv("data/results/tables/table_metadata_layer.csv", rows, ["metric", "value"])
    print("Wrote metadata privacy and utility table.")


if __name__ == "__main__":
    main()
