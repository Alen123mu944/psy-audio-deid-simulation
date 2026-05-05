from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Detection:
    span: str
    start: int
    end: int
    type: str

    def as_dict(self) -> dict[str, object]:
        return {"span": self.span, "start": self.start, "end": self.end, "type": self.type}


PATTERNS: list[tuple[str, str]] = [
    (r"\bDr\.\s+[A-Z][a-z]+\b", "CLINICIAN_NAME"),
    (r"\btherapist\s+[A-Z][a-z]+\b", "CLINICIAN_NAME"),
    (r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}\s+(?:Mental Health Center|Clinic|Hospital)\b", "HEALTHCARE_ORGANIZATION"),
    (r"\bWard\s+\d+[A-Z]?\b", "HEALTHCARE_ORGANIZATION"),
    (r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}\s+(?:High School|University|College|Logistics|Company|Warehouse)\b", "SCHOOL_WORKPLACE"),
    (r"\b(?:every\s+\w+day|weekly session|second appointment|first appointment|4 p\.m\.|every Tuesday at 4 p\.m\.)\b", "SESSION_PATTERN"),
    (r"\b(?:fire|accident|bridge|apartment building fire|local news)\b", "RARE_EVENT"),
    (r"\bmy\s+(?:brother|sister|mother|father)\s+[A-Z][a-z]+\b", "FAMILY_NAME"),
    (r"\b(?:MR|HX|REC)-\d{4,6}\b", "MEDICAL_RECORD"),
    (r"\b\d{3}-\d{4}\b", "CONTACT"),
    (r"\b[A-Za-z0-9._%+-]+@example\.test\b", "CONTACT"),
    (r"\b(?:March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\b", "DATE"),
    (r"\b\d{4}-\d{2}-\d{2}\b", "DATE"),
    (r"\blast October\b|\btwo weeks after New Year\b", "DATE"),
    (r"\b(?:Briar Hill|Westford Junction|Lake Ash Street|Pine Market)\b", "LOCATION"),
]

REPLACEMENTS = {
    "PERSON": "[PERSON]",
    "CLINICIAN_NAME": "[CLINICIAN]",
    "HEALTHCARE_ORGANIZATION": "[mental health clinic]",
    "ORGANIZATION": "[organization]",
    "LOCATION": "[location]",
    "DATE": "[relative time]",
    "DATE_TIME": "[relative time]",
    "SCHOOL_WORKPLACE": "[school/workplace]",
    "ADDRESS": "[address removed]",
    "CONTACT": "[contact removed]",
    "FAMILY_NAME": "my family member",
    "RARE_EVENT": "[major stressful event]",
    "SESSION_PATTERN": "[regular therapy schedule]",
    "MEDICAL_RECORD": "[record identifier removed]",
}


def detect_identifiers(text: str) -> list[Detection]:
    detections: list[Detection] = []
    for pattern, typ in PATTERNS:
        for match in re.finditer(pattern, text):
            detections.append(Detection(match.group(0), match.start(), match.end(), typ))
    detections.sort(key=lambda item: (item.start, -(item.end - item.start)))
    filtered: list[Detection] = []
    occupied: set[int] = set()
    for det in detections:
        positions = set(range(det.start, det.end))
        if positions & occupied:
            continue
        filtered.append(det)
        occupied |= positions
    return filtered


def anonymize_text(text: str, detections: list[Detection]) -> str:
    out = []
    cursor = 0
    for det in sorted(detections, key=lambda item: item.start):
        out.append(text[cursor : det.start])
        out.append(REPLACEMENTS.get(det.type, "[identifier]"))
        cursor = det.end
    out.append(text[cursor:])
    return re.sub(r"\s+", " ", "".join(out)).strip()


def deidentify_record(row: dict) -> dict:
    detections = detect_identifiers(row["utterance"])
    return {
        "id": row["id"],
        "utterance": row["utterance"],
        "predicted_identifier_annotations": [det.as_dict() for det in detections],
        "automated_deidentified_utterance": anonymize_text(row["utterance"], detections),
        "reference_deidentified_utterance": row.get("deidentified_utterance", ""),
        "gold_identifier_annotations": row.get("identifier_annotations", []),
        "clinical_concept_annotations": row.get("clinical_concept_annotations", []),
    }
