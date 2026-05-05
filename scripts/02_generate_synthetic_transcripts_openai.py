from __future__ import annotations

import _bootstrap  # noqa: F401

from src.io_utils import load_config, write_csv, write_jsonl
from src.semantic_generation import generate_fallback_transcripts, quality_report


PROMPT = """You are assisting with a privacy-preserving simulation study.

Generate synthetic psychiatric interview-like utterances for de-identification research.
Do not use real patient information, real hospitals, real clinicians, real addresses, or real institutions.
Return valid JSON only with utterance, identifier annotations, clinical concept annotations, and a deidentified reference version.
"""


def main() -> None:
    cfg = load_config()
    n = int(cfg["semantic_generation"]["n_transcripts"])
    rows = generate_fallback_transcripts(n, cfg["project"]["random_seed"])
    write_jsonl(cfg["semantic_generation"]["output_file"], rows)
    write_jsonl("data/raw/synthetic_generated/transcripts_reviewed.jsonl", rows)
    write_csv(
        "data/results/logs/transcript_generation_quality_report.csv",
        quality_report(rows),
        ["id", "is_valid_json", "n_identifiers", "n_clinical_concepts", "missing_identifier_spans", "missing_concept_spans", "flag_real_world_entity", "flag_extreme_content", "needs_review"],
    )
    print(f"Wrote {len(rows)} synthetic transcripts using deterministic fallback generation.")


if __name__ == "__main__":
    main()
