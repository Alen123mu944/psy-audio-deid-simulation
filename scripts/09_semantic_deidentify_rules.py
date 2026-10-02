from __future__ import annotations

import _bootstrap  # noqa: F401

from src.io_utils import read_jsonl, write_jsonl
from src.semantic_deid import deidentify_record


def main() -> None:
    rows = read_jsonl("data/raw/synthetic_generated/transcripts_reviewed.jsonl") or read_jsonl("data/raw/synthetic_generated/transcripts_raw.jsonl")
    if not rows:
        raise SystemExit("No transcript JSONL found. Run 02_generate_synthetic_transcripts_openrouter.py first.")

    direct = [deidentify_record(row, mode="direct_only_baseline") for row in rows]
    full = [deidentify_record(row, mode="full_semantic_layer") for row in rows]

    write_jsonl("data/processed/transcripts_deidentified/transcripts_direct_only_baseline.jsonl", direct)
    write_jsonl("data/processed/transcripts_deidentified/transcripts_full_semantic_layer.jsonl", full)
    write_jsonl("data/processed/transcripts_deidentified/transcripts_deidentified.jsonl", full)
    print(f"Wrote {len(rows)} records for direct-only baseline and full semantic layer.")


if __name__ == "__main__":
    main()
