from __future__ import annotations

import _bootstrap  # noqa: F401

from src.io_utils import copy_to_manuscript, read_jsonl, write_csv
from src.metadata_metrics import evaluate_metadata


SUMMARY_FIELDS = ["metric", "value"]
BY_SUBSET_FIELDS = [
    "metadata_subset",
    "n_records",
    "direct_metadata_leakage_before",
    "direct_metadata_leakage_after",
    "filename_path_leakage_before",
    "filename_path_leakage_after",
    "free_text_context_leakage_before",
    "free_text_context_leakage_after",
    "linkage_risk_score_before",
    "linkage_risk_score_after",
    "longitudinal_utility_preservation",
    "audio_standardization_success",
]


def main() -> None:
    raw = read_jsonl("data/raw/synthetic_generated/metadata_reviewed.jsonl") or read_jsonl("data/raw/synthetic_generated/metadata_raw.jsonl")
    sanitized = read_jsonl("data/processed/metadata_sanitized/metadata_sanitized.jsonl")
    if not raw or not sanitized:
        raise SystemExit("Metadata inputs missing. Run 03_generate_synthetic_metadata.py and 11_metadata_sanitize.py first.")
    summary, by_subset = evaluate_metadata(raw, sanitized)
    write_csv("data/results/tables/table_metadata_layer.csv", summary, SUMMARY_FIELDS)
    write_csv("data/results/tables/table_metadata_by_subset.csv", by_subset, BY_SUBSET_FIELDS)
    copy_to_manuscript("data/results/tables/table_metadata_layer.csv", "table_metadata_layer.csv")
    copy_to_manuscript("data/results/tables/table_metadata_by_subset.csv", "table_metadata_by_subset.csv")
    print("Wrote metadata privacy and utility tables.")


if __name__ == "__main__":
    main()
