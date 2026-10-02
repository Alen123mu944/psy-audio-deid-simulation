from __future__ import annotations

import _bootstrap  # noqa: F401

from src.io_utils import copy_to_manuscript, read_csv_dicts, read_jsonl, write_csv
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
BY_TYPE_FIELDS = ["condition", "identifier_type", "precision", "recall", "f1", "n_gold", "n_detected", "n_true_positive"]
BY_SUBSET_FIELDS = ["condition", "subset"] + [field for field in SUMMARY_FIELDS if field != "condition"]


def main() -> None:
    rows = read_jsonl("data/processed/transcripts_deidentified/transcripts_qwen_llm_semantic_layer.jsonl")
    if not rows:
        raise SystemExit("No Qwen deidentified transcript JSONL found. Run 09b_semantic_deidentify_siliconflow_qwen.py first.")
    summary, by_type, by_subset = evaluate_semantic_wide(rows, "qwen_llm_semantic_layer")
    write_csv("data/results/tables/table_semantic_qwen_layer.csv", [summary], SUMMARY_FIELDS)
    write_csv("data/results/tables/table_semantic_qwen_by_type.csv", by_type, BY_TYPE_FIELDS)
    write_csv("data/results/tables/table_semantic_qwen_by_subset.csv", by_subset, BY_SUBSET_FIELDS)

    comparison = read_csv_dicts("data/results/tables/table_semantic_comparison.csv")
    merged = [row for row in comparison if row.get("condition") != "qwen_llm_semantic_layer"]
    merged.append({key: str(summary.get(key, "")) for key in SUMMARY_FIELDS})
    write_csv("data/results/tables/table_semantic_comparison_with_qwen.csv", merged, SUMMARY_FIELDS)
    copy_to_manuscript("data/results/tables/table_semantic_qwen_layer.csv", "table_semantic_qwen_layer.csv")
    copy_to_manuscript("data/results/tables/table_semantic_qwen_by_subset.csv", "table_semantic_qwen_by_subset.csv")
    copy_to_manuscript("data/results/tables/table_semantic_comparison_with_qwen.csv", "table_semantic_comparison_with_qwen.csv")
    print("Wrote Qwen semantic evaluation tables.")


if __name__ == "__main__":
    main()
