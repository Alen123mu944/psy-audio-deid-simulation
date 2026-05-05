from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


DIRECT_TYPES = {"PERSON", "CONTACT", "ADDRESS", "MEDICAL_RECORD", "CLINICIAN_NAME"}
CONTEXTUAL_TYPES = {"LOCATION", "ORGANIZATION", "HEALTHCARE_ORGANIZATION", "SCHOOL_WORKPLACE", "FAMILY_NAME", "RARE_EVENT", "SESSION_PATTERN", "DATE", "DATE_TIME"}
COMPATIBLE_TYPES = {
    "DATE": {"DATE", "DATE_TIME"},
    "DATE_TIME": {"DATE", "DATE_TIME"},
    "ORGANIZATION": {"ORGANIZATION", "HEALTHCARE_ORGANIZATION"},
    "HEALTHCARE_ORGANIZATION": {"ORGANIZATION", "HEALTHCARE_ORGANIZATION"},
    "PERSON": {"PERSON", "CLINICIAN_NAME", "FAMILY_NAME"},
    "CLINICIAN_NAME": {"PERSON", "CLINICIAN_NAME"},
    "LOCATION": {"LOCATION", "ADDRESS"},
    "ADDRESS": {"LOCATION", "ADDRESS"},
}


def _compatible(a: str, b: str) -> bool:
    return b in COMPATIBLE_TYPES.get(a, {a})


def _overlaps(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return int(a.get("start", -1)) < int(b.get("end", -2)) and int(b.get("start", -1)) < int(a.get("end", -2))


def match_annotations(gold: list[dict[str, Any]], pred: list[dict[str, Any]]) -> tuple[int, int, int, list[dict[str, Any]]]:
    used = set()
    matches = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            if pi in used:
                continue
            if _overlaps(g, p) and _compatible(str(g.get("type")), str(p.get("type"))):
                used.add(pi)
                matches.append({"gold_index": gi, "pred_index": pi, "type": g.get("type")})
                break
    tp = len(matches)
    fp = len(pred) - tp
    fn = len(gold) - tp
    return tp, fp, fn, matches


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def clinical_concept_preservation(row: dict[str, Any]) -> float:
    concepts = row.get("clinical_concept_annotations", [])
    if not concepts:
        return 0.0
    text = row.get("automated_deidentified_utterance", "").lower()
    kept = sum(1 for concept in concepts if str(concept.get("span", "")).lower() in text)
    return kept / len(concepts)


def evaluate_semantic(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    total_tp = total_fp = total_fn = 0
    direct_tp = direct_fn = contextual_tp = contextual_fn = 0
    type_counts = defaultdict(lambda: Counter(tp=0, fp=0, fn=0, gold=0, detected=0))
    residual = []
    preservation_scores = []

    for row in rows:
        gold = row.get("gold_identifier_annotations", [])
        pred = row.get("predicted_identifier_annotations", [])
        tp, fp, fn, matches = match_annotations(gold, pred)
        matched_gold = {m["gold_index"] for m in matches}
        total_tp += tp
        total_fp += fp
        total_fn += fn
        preservation_scores.append(clinical_concept_preservation(row))
        for idx, g in enumerate(gold):
            typ = str(g.get("type"))
            type_counts[typ]["gold"] += 1
            if idx in matched_gold:
                type_counts[typ]["tp"] += 1
                if typ in DIRECT_TYPES:
                    direct_tp += 1
                if typ in CONTEXTUAL_TYPES:
                    contextual_tp += 1
            else:
                type_counts[typ]["fn"] += 1
                residual.append(g)
            if typ in DIRECT_TYPES:
                direct_fn += 0 if idx in matched_gold else 1
            if typ in CONTEXTUAL_TYPES:
                contextual_fn += 0 if idx in matched_gold else 1
        for p in pred:
            type_counts[str(p.get("type"))]["detected"] += 1
        for pidx, p in enumerate(pred):
            if pidx not in {m["pred_index"] for m in matches}:
                type_counts[str(p.get("type"))]["fp"] += 1

    precision, recall, f1 = prf(total_tp, total_fp, total_fn)
    direct_recall = direct_tp / (direct_tp + direct_fn) if direct_tp + direct_fn else 0.0
    contextual_recall = contextual_tp / (contextual_tp + contextual_fn) if contextual_tp + contextual_fn else 0.0
    residual_rate = total_fn / (total_tp + total_fn) if total_tp + total_fn else 0.0
    high_risk_residual = [r for r in residual if r.get("type") in DIRECT_TYPES]
    summary = [
        {"metric": "overall_identifier_precision", "value": precision},
        {"metric": "overall_identifier_recall", "value": recall},
        {"metric": "overall_identifier_f1", "value": f1},
        {"metric": "direct_identifier_recall", "value": direct_recall},
        {"metric": "contextual_identifier_recall", "value": contextual_recall},
        {"metric": "residual_identifier_rate", "value": residual_rate},
        {"metric": "high_risk_residual_identifier_rate", "value": len(high_risk_residual) / (total_tp + total_fn) if total_tp + total_fn else 0.0},
        {"metric": "clinical_concept_preservation_rate", "value": sum(preservation_scores) / len(preservation_scores) if preservation_scores else 0.0},
    ]
    by_type = []
    for typ, counts in sorted(type_counts.items()):
        p, r, f = prf(counts["tp"], counts["fp"], counts["fn"])
        by_type.append({"identifier_type": typ, "precision": p, "recall": r, "f1": f, "n_gold": counts["gold"], "n_detected": counts["detected"], "n_true_positive": counts["tp"]})
    return summary, by_type
