from __future__ import annotations

import _bootstrap  # noqa: F401

from src.io_utils import load_config, read_jsonl, write_jsonl
from src.metadata_deid import sanitize_records


def main() -> None:
    cfg = load_config()
    rows = read_jsonl("data/raw/synthetic_generated/metadata_reviewed.jsonl") or read_jsonl("data/raw/synthetic_generated/metadata_raw.jsonl")
    if not rows:
        raise SystemExit("No metadata JSONL found. Run 03_generate_synthetic_metadata.py first.")
    sanitized = sanitize_records(rows, cfg["metadata_sanitization"]["target_sample_rate"], cfg["metadata_sanitization"]["target_channels"])
    write_jsonl("data/processed/metadata_sanitized/metadata_sanitized.jsonl", sanitized)
    print(f"Wrote {len(sanitized)} sanitized metadata records.")


if __name__ == "__main__":
    main()
