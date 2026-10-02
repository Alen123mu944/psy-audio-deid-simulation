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
    additions = manifest["gold_additions"]
    for addition in additions:
        require(addition["id"] in by_id, "review correction ID missing")
        annotation = {k:v for k,v in addition.items() if k != "id"}
        require(by_id[addition["id"]]["identifier_annotations"].count(annotation) == 1, "confirmed correction missing or duplicated in locked gold")
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
    metadata_values = {r["metric"]:r["value"] for r in metadata}
    voice_values = {r["metric"]:r["value"] for r in csv_rows("outputs_for_manuscript/table_voice_layer.csv")}
    selected = voice_values["selected_voice_condition"]
    privacy = {r["condition"]:r for r in csv_rows("data/results/tables/table_voice_privacy.csv")}
    utility = csv_rows("data/results/tables/table_voice_utility.csv")
    acoustic_features = {"duration","RMS_energy","zero_crossing_rate","spectral_centroid","spectral_bandwidth","pause_ratio"}
    acoustic = [float(r["pearson_r"]) for r in utility if r["condition"] == selected and r["feature"] in acoustic_features]
    require(len(acoustic) == len(acoustic_features), "voice utility features missing")
    for row in matrix:
        has_voice = row["condition"] in {"Voice-only","Full_framework"}
        voice_condition = selected if has_voice else "original"
        expected = {
            "speaker_id_accuracy":float(privacy[voice_condition]["top1_speaker_id_accuracy"]),
            "acoustic_feature_preservation":sum(acoustic)/len(acoustic) if has_voice else 1.0,
            "metadata_linkage_risk_score":metadata_values["linkage_risk_score_after"] if row["condition"] in {"Metadata-only","Full_framework"} else 1.0,
        }
        for key, value in expected.items():
            require(math.isclose(float(row[key]), float(value), rel_tol=1e-12, abs_tol=1e-12), f"matrix {row['condition']}: {key} differs")
    print("PASS: 150 records; 744 gold (clean=250, contextual=251, ASR=243); unchanged predictions.")
    print("PASS: canonical result tables, integrated matrix and frozen hashes.")
    print(f"Qwen: P={qwen['overall_identifier_precision']}, R={qwen['overall_identifier_recall']}, F1={qwen['overall_identifier_f1']}")
    print("Scope: frozen semantic/metadata results; not a new live-model run or raw-audio reproduction.")

if __name__ == "__main__":
    main()
