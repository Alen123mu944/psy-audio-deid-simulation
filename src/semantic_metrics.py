from __future__ import annotations

from typing import Any


DIRECT_TYPES = {"PERSON", "CONTACT", "ADDRESS", "MEDICAL_RECORD", "CLINICIAN_NAME", "DATE", "DATE_TIME"}
CONTEXTUAL_TYPES = {"LOCATION", "ORGANIZATION", "HEALTHCARE_ORGANIZATION", "SCHOOL_WORKPLACE", "FAMILY_NAME", "RARE_EVENT", "SESSION_PATTERN"}
COMPATIBLE_TYPES = {
    "DATE": {"DATE", "DATE_TIME"},
    "DATE_TIME": {"DATE", "DATE_TIME"},
    "ORGANIZATION": {"ORGANIZATION", "HEALTHCARE_ORGANIZATION"},
    "HEALTHCARE_ORGANIZATION": {"ORGANIZATION", "HEALTHCARE_ORGANIZATION"},
    "PERSON": {"PERSON", "CLINICIAN_NAME", "FAMILY_NAME"},
    "CLINICIAN_NAME": {"PERSON", "CLINICIAN_NAME"},
    "FAMILY_NAME": {"PERSON", "FAMILY_NAME"},
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


def _evaluate_counts(rows: list[dict[str, Any]]) -> dict[str, float]:
    total_tp = total_fp = total_fn = 0
    direct_tp = direct_fn = contextual_tp = contextual_fn = 0
    high_risk_gold = high_risk_fn = 0
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
            is_matched = idx in matched_gold
            if is_matched:
                if typ in DIRECT_TYPES:
                    direct_tp += 1
                if typ in CONTEXTUAL_TYPES:
                    contextual_tp += 1
            else:
                if typ in DIRECT_TYPES:
                    direct_fn += 1
                if typ in CONTEXTUAL_TYPES:
                    contextual_fn += 1
            if str(g.get("risk_level", "")).lower() == "high" or typ in {"CONTACT", "ADDRESS", "MEDICAL_RECORD", "CLINICIAN_NAME", "PERSON"}:
                high_risk_gold += 1
                if not is_matched:
                    high_risk_fn += 1

    precision, recall, f1 = prf(total_tp, total_fp, total_fn)
    direct_recall = direct_tp / (direct_tp + direct_fn) if direct_tp + direct_fn else 0.0
    contextual_recall = contextual_tp / (contextual_tp + contextual_fn) if contextual_tp + contextual_fn else 0.0
    residual_rate = total_fn / (total_tp + total_fn) if total_tp + total_fn else 0.0
    concept_preservation = sum(preservation_scores) / len(preservation_scores) if preservation_scores else 0.0
    summary = {
        "n_records": float(len(rows)),
        "n_gold_identifiers": float(total_tp + total_fn),
        "n_predicted_identifiers": float(total_tp + total_fp),
        "overall_identifier_precision": precision,
        "overall_identifier_recall": recall,
        "overall_identifier_f1": f1,
        "direct_identifier_recall": direct_recall,
        "contextual_identifier_recall": contextual_recall,
        "residual_identifier_rate": residual_rate,
        "high_risk_residual_identifier_rate": high_risk_fn / high_risk_gold if high_risk_gold else 0.0,
        "clinical_concept_preservation_rate": concept_preservation,
        "over_redaction_rate": 1.0 - concept_preservation,
    }
    return summary


def evaluate_semantic_wide(
    rows: list[dict[str, Any]], condition: str, *, include_subsets: bool = False
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = _evaluate_counts(rows)
    summary_row: dict[str, Any] = {"condition": condition}
    summary_row.update(summary)

    by_subset = []
    subsets = sorted({str(row.get("subset", "")) for row in rows}) if include_subsets else []
    for subset in subsets:
        subset_rows = [row for row in rows if str(row.get("subset", "")) == subset]
        subset_summary = _evaluate_counts(subset_rows)
        subset_row: dict[str, Any] = {"condition": condition, "subset": subset}
        subset_row.update(subset_summary)
        by_subset.append(subset_row)
    return summary_row, by_subset
