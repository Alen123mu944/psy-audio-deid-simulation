"""Offline verification of the locked 744-gold submission (standard library only)."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import _bootstrap  # noqa: F401
from src.metadata_metrics import evaluate_metadata
from src.semantic_metrics import evaluate_semantic_wide, match_annotations

ROOT = Path(__file__).resolve().parents[1]
RAW = "data/raw/synthetic_generated/transcripts_reviewed.jsonl"
OUTPUT = "data/processed/transcripts_deidentified"

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def jsonl(path):
    return [json.loads(line) for line in (ROOT / path).read_text(encoding="utf-8").splitlines() if line.strip()]

def csv_rows(path):
    with (ROOT / path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))

def compare_rows(actual, recorded, label):
    require(len(actual) == len(recorded), f"{label}: row count differs")
    for index, (computed, saved) in enumerate(zip(actual, recorded), 1):
        require(set(computed) == set(saved), f"{label}: columns differ")
        for key, value in computed.items():
            if isinstance(value, (float, int)):
                require(math.isclose(float(value), float(saved[key]), rel_tol=1e-12, abs_tol=1e-12), f"{label} row {index}: {key} differs")
            else:
                require(str(value) == saved[key], f"{label} row {index}: {key} differs")

def prediction_signature(rows):
    payload = [{k:r[k] for k in ("id", "predicted_identifier_annotations", "automated_deidentified_utterance")} for r in sorted(rows, key=lambda r:r["id"])]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def semantic_tables():
    """Compute every active semantic table from record-level outputs, not saved CSVs."""
    computed = {}
    all_summary, all_types, all_subsets = [], [], []
    manifest = json.loads((ROOT/"FROZEN_MANIFEST.json").read_text())
    for method in manifest["methods"]:
        rows = jsonl(f"{OUTPUT}/transcripts_{method}.jsonl")
        summary, types, subsets = evaluate_semantic_wide(rows, method)
        all_summary.append(summary)
        if method == "qwen_llm_semantic_layer":
            computed["table_semantic_qwen_layer.csv"] = [summary]
            computed["table_semantic_qwen_by_type.csv"] = types
            computed["table_semantic_qwen_by_subset.csv"] = subsets
        else:
            all_types.extend(types)
            all_subsets.extend(subsets)
    computed["table_semantic_comparison.csv"] = all_summary[:2]
    computed["table_semantic_comparison_with_qwen.csv"] = all_summary
    computed["table_semantic_by_type.csv"] = all_types
    computed["table_semantic_by_subset.csv"] = all_subsets
    return computed

def main():
    manifest = json.loads((ROOT/"FROZEN_MANIFEST.json").read_text())
    require(manifest["n_gold_identifiers"] == 744, "wrong locked version")
    for path, expected in manifest["sha256"].items():
        require(hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == expected, f"frozen file changed: {path}")
    generated = jsonl(RAW)
    require(len(generated) == len({r["id"] for r in generated}) == 150, "raw IDs must be 150 unique records")
    require(dict(Counter(r["subset"] for r in generated)) == manifest["records_by_subset"], "subset record counts differ")
    gold_counts = Counter()
    for row in generated:
        gold_counts[row["subset"]] += len(row["identifier_annotations"])
        for a in row["identifier_annotations"]:
            require(0 <= a["start"] < a["end"] <= len(row["utterance"]), f"gold offsets invalid: {row['id']}")
            require(row["utterance"][a["start"]:a["end"]] == a["span"], f"gold span differs: {row['id']}")
    require(dict(gold_counts) == manifest["gold_by_subset"] and sum(gold_counts.values()) == 744, "gold subset counts differ")
    by_id = {r["id"]:r for r in generated}
    additions = []
    previous = jsonl("provenance/transcripts_pre_correction_742.jsonl")
    require(len(previous) == 150 and sum(len(r["identifier_annotations"]) for r in previous) == 742, "pre-correction provenance differs")
    for old in previous:
        new = by_id[old["id"]]
        require({k:v for k,v in old.items() if k != "identifier_annotations"} == {k:v for k,v in new.items() if k != "identifier_annotations"}, "raw text or clinical/reference fields changed")
        require(all(a in new["identifier_annotations"] for a in old["identifier_annotations"]), "old gold removed")
        additions.extend({"id":new["id"], **a} for a in new["identifier_annotations"] if a not in old["identifier_annotations"])
    require(additions == manifest["gold_additions"], "review corrections differ")
    corrections = csv_rows("review/gold_corrections.csv")
    require(len(corrections) == len(additions) == 2, "expected exactly two recorded corrections")
    for recorded, actual in zip(corrections, additions):
        for key in ("id", "span", "type", "start", "end", "risk_level", "recommended_action"):
            require(recorded[key] == str(actual[key]), f"review record mismatch: {key}")
    review_rows = csv_rows("review/locked_review.csv")
    require(len(review_rows) == len({r['id'] for r in review_rows}) == 150, "review export IDs differ")
    for reviewed in review_rows:
        raw = by_id[reviewed['id']]
        require(reviewed['utterance'] == raw['utterance'] and reviewed['subset'] == raw['subset'], "review export input mismatch")
        require(json.loads(reviewed['gold_annotations']) == raw['identifier_annotations'], "review export gold mismatch")
        require(reviewed['reference_deidentified_utterance'] == raw['deidentified_utterance'], "review export reference mismatch")
    for method in manifest["methods"]:
        rows = jsonl(f"{OUTPUT}/transcripts_{method}.jsonl")
        require(len(rows) == len({r["id"] for r in rows}) == 150 and {r["id"] for r in rows} == set(by_id), f"{method}: missing/duplicate IDs")
        require(prediction_signature(rows) == manifest["prediction_sha256"][method], f"{method}: predictions or outputs changed")
        counts = [0,0,0]
        for row in rows:
            raw = by_id[row["id"]]
            for key in ("utterance", "subset", "clinical_concept_annotations"):
                require(row[key] == raw[key], f"{method}: {key} mismatch")
            require(row["reference_deidentified_utterance"] == raw["deidentified_utterance"], f"{method}: reference mismatch")
            require(row["gold_identifier_annotations"] == raw["identifier_annotations"], f"{method}: gold mismatch")
            for a in row["predicted_identifier_annotations"]:
                require(0 <= a["start"] < a["end"] <= len(row["utterance"]), f"{method}: prediction offsets invalid")
                require(row["utterance"][a["start"]:a["end"]] == a["span"], f"{method}: prediction span mismatch")
            result = match_annotations(row["gold_identifier_annotations"], row["predicted_identifier_annotations"])
            counts = [a+b for a,b in zip(counts,result[:3])]
        require(counts == manifest["expected_tp_fp_fn"][method], f"{method}: TP/FP/FN differ")
        print(f"PASS {method}: TP/FP/FN={counts}")
    tables = semantic_tables()
    for filename, actual in tables.items():
        compare_rows(actual, csv_rows(f"outputs_for_manuscript/{filename}"), filename)
    compare_rows(tables["table_semantic_comparison_with_qwen.csv"], csv_rows("manuscript_tables/S2.csv"), "S2")
    compare_rows(tables["table_semantic_qwen_by_subset.csv"], csv_rows("manuscript_tables/S3.csv"), "S3 keyed subsets")
    logs = csv_rows("data/results/logs/qwen_semantic_deid_openrouter_log.csv")
    require(len(logs) == 150 and all(r["status"] == "ok" for r in logs), "Qwen has failed records")
    raw_metadata = jsonl("data/raw/synthetic_generated/metadata_reviewed.jsonl")
    clean_metadata = jsonl("data/processed/metadata_sanitized/metadata_sanitized.jsonl")
    require(len(raw_metadata) == len(clean_metadata) == 150, "metadata count differs")
    metadata, subsets = evaluate_metadata(raw_metadata, clean_metadata)
    compare_rows(metadata, csv_rows("outputs_for_manuscript/table_metadata_layer.csv"), "metadata summary")
    compare_rows(subsets, csv_rows("outputs_for_manuscript/table_metadata_by_subset.csv"), "metadata subsets")
    qwen = tables["table_semantic_qwen_layer.csv"][0]
    matrix = csv_rows("outputs_for_manuscript/table_risk_utility_matrix.csv")
    require({r["condition"] for r in matrix} == {"Original","Voice-only","Semantic-only","Metadata-only","Full_framework"}, "matrix conditions differ")
    for row in matrix:
        has_semantic = row["condition"] in {"Semantic-only","Full_framework"}
        for key,source in [("semantic_residual_identifier_rate","residual_identifier_rate"),("clinical_concept_preservation","clinical_concept_preservation_rate")]:
            expected = qwen[source] if has_semantic else 1.0
            require(math.isclose(float(row[key]), expected, abs_tol=1e-12), f"matrix {row['condition']}: {key} differs")
    audit = json.loads((ROOT/'manuscript_tables/manuscript_alignment_audit.json').read_text())
    observed_main = audit['main_Table5_Table6_observed']
    semantic = observed_main['5'][2]
    for label, metric, decimals in [('Precision','overall_identifier_precision',3),('recall','overall_identifier_recall',3),('F1','overall_identifier_f1',3),('residual identifier rate','residual_identifier_rate',4),('high-risk residual rate','high_risk_residual_identifier_rate',3)]:
        require(f'{label} = {qwen[metric]:.{decimals}f}' in semantic[2], f'main Table 5: {label} differs')
    require(f"Clinical concept preservation = {qwen['clinical_concept_preservation_rate']:.3f}" == semantic[3], 'main Table 5 utility differs')
    matrix_by_name = {r['condition'].replace('_',' '):r for r in matrix}
    for row in observed_main['6'][1:]:
        computed = matrix_by_name[row[0]]
        keys = ['speaker_id_accuracy','semantic_residual_identifier_rate','metadata_linkage_risk_score','clinical_concept_preservation','acoustic_feature_preservation']
        require(row[1:] == [f'{float(computed[k]):.3f}' for k in keys], f'main Table 6 differs: {row[0]}')
    print("PASS: 150 records; 744 gold (clean=250, contextual=251, ASR=243); unchanged predictions.")
    print("PASS: canonical semantic CSVs, S2/S3, metadata, review corrections and frozen hashes.")
    print(f"Qwen: P={qwen['overall_identifier_precision']}, R={qwen['overall_identifier_recall']}, F1={qwen['overall_identifier_f1']}")
    print("Scope: frozen semantic/metadata results; not a new live-model run or raw-audio reproduction.")

if __name__ == "__main__":
    main()
