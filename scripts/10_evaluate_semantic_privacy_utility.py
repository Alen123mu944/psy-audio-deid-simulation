from __future__ import annotations

import _bootstrap  # noqa: F401

from src.io_utils import read_jsonl, write_csv
from src.semantic_metrics import evaluate_semantic_wide


SUMMARY_FIELDS = [
    "condition",
    "n_records",
    "n_gold_identifiers",
    "n_predicted_identifiers",
    "overall_identifier_precision",
    "overall_identifier_recall",
    "overall_identifier_f1",
    "direct_identifier_recall",
    "contextual_identifier_recall",
    "residual_identifier_rate",
    "high_risk_residual_identifier_rate",
    "clinical_concept_preservation_rate",
    "over_redaction_rate",
]



def main() -> None:
    inputs = [
        ("direct_only_baseline", "data/processed/transcripts_deidentified/transcripts_direct_only_baseline.jsonl"),
        ("full_semantic_layer", "data/processed/transcripts_deidentified/transcripts_full_semantic_layer.jsonl"),
    ]
    summary_rows = []
    for condition, path in inputs:
        rows = read_jsonl(path)
        if not rows:
            raise SystemExit(f"No deidentified transcript JSONL found at {path}. Run 09_semantic_deidentify_rules.py first.")
        summary, _ = evaluate_semantic_wide(rows, condition=condition)
        summary_rows.append(summary)

    write_csv("data/results/tables/table_semantic_comparison_with_qwen.csv", summary_rows, SUMMARY_FIELDS)

    print("Wrote rule-based rows for the semantic comparison table.")


if __name__ == "__main__":
    main()
