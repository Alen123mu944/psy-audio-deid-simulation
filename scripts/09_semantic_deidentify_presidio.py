from __future__ import annotations

import _bootstrap  # noqa: F401

from src.io_utils import read_jsonl, write_jsonl
from src.semantic_deid import deidentify_record


def main() -> None:
    rows = read_jsonl("data/raw/synthetic_generated/transcripts_reviewed.jsonl") or read_jsonl("data/raw/synthetic_generated/transcripts_raw.jsonl")
    if not rows:
        raise SystemExit("No transcript JSONL found. Run 02_generate_synthetic_transcripts_openai.py first.")
    out = [deidentify_record(row) for row in rows]
    write_jsonl("data/processed/transcripts_deidentified/transcripts_deidentified.jsonl", out)
    print(f"Wrote {len(out)} semantic de-identification records.")


if __name__ == "__main__":
    main()
