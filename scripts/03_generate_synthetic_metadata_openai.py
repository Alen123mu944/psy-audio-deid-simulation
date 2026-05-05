from __future__ import annotations

import _bootstrap  # noqa: F401

from src.io_utils import load_config, write_jsonl
from src.metadata_generation import generate_fallback_metadata


PROMPT = """You are assisting with a privacy-preserving simulation study.

Generate synthetic metadata records for fictional psychiatric audio recordings.
Do not use real patient information, real hospitals, real clinicians, real addresses, or real institutions.
Return valid JSON only.
"""


def main() -> None:
    cfg = load_config()
    rows = generate_fallback_metadata(int(cfg["metadata_generation"]["n_records"]), cfg["project"]["random_seed"])
    write_jsonl(cfg["metadata_generation"]["output_file"], rows)
    write_jsonl("data/raw/synthetic_generated/metadata_reviewed.jsonl", rows)
    print(f"Wrote {len(rows)} synthetic metadata records using deterministic fallback generation.")


if __name__ == "__main__":
    main()
