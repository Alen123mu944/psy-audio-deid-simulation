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

BY_TYPE_FIELDS = ["condition", "identifier_type", "precision", "recall", "f1", "n_gold", "n_detected", "n_true_positive"]
BY_SUBSET_FIELDS = ["condition", "subset"] + [field for field in SUMMARY_FIELDS if field != "condition"]


def main() -> None:
    inputs = [
        ("direct_only_baseline", "data/processed/transcripts_deidentified/transcripts_direct_only_baseline.jsonl"),
        ("full_semantic_layer", "data/processed/transcripts_deidentified/transcripts_full_semantic_layer.jsonl"),
    ]
    summary_rows = []
    by_type_rows = []
    by_subset_rows = []
    for condition, path in inputs:
        rows = read_jsonl(path)
        if not rows:
            raise SystemExit(f"No deidentified transcript JSONL found at {path}. Run 09_semantic_deidentify_rules.py first.")
        summary, by_type, by_subset = evaluate_semantic_wide(rows, condition=condition)
        summary_rows.append(summary)
        by_type_rows.extend(by_type)
        by_subset_rows.extend(by_subset)

    write_csv("data/results/tables/table_semantic_comparison.csv", summary_rows, SUMMARY_FIELDS)
    write_csv("data/results/tables/table_semantic_by_type.csv", by_type_rows, BY_TYPE_FIELDS)
    write_csv("data/results/tables/table_semantic_by_subset.csv", by_subset_rows, BY_SUBSET_FIELDS)

    full_summary = next(row for row in summary_rows if row["condition"] == "full_semantic_layer")
    metric_rows = [{"metric": key, "value": value} for key, value in full_summary.items() if key != "condition"]
    write_csv("data/results/tables/table_semantic_layer.csv", metric_rows, ["metric", "value"])
    print("Wrote semantic comparison, by-type, by-subset, and manuscript compatibility tables.")


if __name__ == "__main__":
    main()
