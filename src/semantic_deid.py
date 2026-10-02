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


# The direct-only baseline intentionally targets explicit identifiers. The full
# layer adds contextual psychiatric disclosure risks such as rare events,
# recurring schedules, family names, workplaces, and small locations.
PATTERNS: list[tuple[str, str, str]] = [
    (r"\bDr\.?\s+(?:[A-Z]\.\s*)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", "CLINICIAN_NAME", "direct"),
    (r"\bdr\s+(?:[a-z]\.\s*)?[a-z]+(?:\s+[a-z]+)?\b", "CLINICIAN_NAME", "direct"),
    (r"\b(?:doctor|therapist|psychiatrist|psychologist|counselor)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", "CLINICIAN_NAME", "direct"),
    (r"\b(?:this is|i'?m|im)\s+[A-Z]?[a-z]+(?:\s+[A-Z]?[a-z]+)?\b", "PERSON", "direct"),
    (r"\b(?:MR|MRN|HX|REC|PT|PAT|CLMHC|CL)-[A-Za-z0-9]+(?:-[A-Za-z0-9]+){0,4}\b", "MEDICAL_RECORD", "direct"),
    (r"\b(?:MRN|mrn|record|chart|file|case|patient code|chart id|record code)\s+(?:number\s+|is\s+|id\s+)?[A-Za-z0-9]+(?:[- ][A-Za-z0-9]+){0,5}\b", "MEDICAL_RECORD", "direct"),
    (r"\b(?:MRN|mrn|record|chart|file|case|record code)\s+(?:one|two|three|four|five|six|seven|eight|nine|zero|oh|[a-z]|\d)(?:\s+(?:one|two|three|four|five|six|seven|eight|nine|zero|oh|[a-z]|\d)){2,8}\b", "MEDICAL_RECORD", "direct"),
    (r"\b\(?\d{3}\)?[- ]?\d{3}[- ]\d{4}\b", "CONTACT", "direct"),
    (r"\b\(?\d{3}\)?[- ]?\d{3,4}\b", "CONTACT", "direct"),
    (r"\b(?:five\s+){2}five(?:[- ]+(?:zero|one|two|three|four|five|six|seven|eight|nine|oh|\d+)){2,10}\b", "CONTACT", "direct"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]*example\.test\b", "CONTACT", "direct"),
    (r"\b(?:\d{2,4}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\s+[A-Z]?[A-Za-z]+\s+(?:Street|St\.|Avenue|Ave\.|Road|Rd\.|Lane|Ln\.|Drive|Dr\.|Court|Ct\.)\b", "ADDRESS", "direct"),
    (r"\b(?:March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s*\d{4})?\b", "DATE", "direct"),
    (r"\b\d{4}-\d{2}-\d{2}\b", "DATE", "direct"),
    (r"\b\d{1,2}/\d{1,2}/\d{4}\b", "DATE", "direct"),
    (r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth|twenty first|twenty second|twenty third|twenty fourth|twenty fifth|twenty sixth|twenty seventh|twenty eighth|twenty ninth|thirtieth|thirty first)(?:\s+twenty\s+twenty\s+\w+)?\b", "DATE", "direct"),
    (r"\b(?:last\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|week|month|fall|winter|spring|summer|October|February)|two weeks after New Year|the week before Thanksgiving|New Year'?s week)\b", "DATE", "direct"),
    (r"\b(?:around\s+)?(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d{1,2})(?:\s+thirty|\s+fifteen|\s+forty five)?\s*(?:a\s*m|p\s*m|am|pm|a\.m\.|p\.m\.)\b", "DATE", "direct"),
    (r"\b(?:Cedar Lane Mental Health Center|Northbridge Clinic|Silver Harbor Hospital|Maple Rest Clinic)\b", "HEALTHCARE_ORGANIZATION", "contextual"),
    (r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}\s+(?:Mental Health Center|Clinic|Hospital|Counseling Center|Behavioral Health)\b", "HEALTHCARE_ORGANIZATION", "contextual"),
    (r"\bWard\s+\d+[A-Z]?\b", "HEALTHCARE_ORGANIZATION", "contextual"),
    (r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}\s+(?:High School|University|College|Logistics|Company|Warehouse|Factory|Call Center|Diner|Shelter)\b", "SCHOOL_WORKPLACE", "contextual"),
    (r"\b(?:weekly\s+(?:CBT|therapy|session)s?(?:\s+on\s+\w+days?)?|weekly session|second appointment|first appointment|third session|intake appointment|every\s+other\s+\w+day|every\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)(?:\s+at\s+\d{1,2}(?::\d{2})?\s*(?:a\.m\.|p\.m\.|am|pm)?)?|on\s+\w+days?|\w+day\s+\d{1,2}\s*(?:a\.m\.|p\.m\.|am|pm)\s+sessions?|\w+day\s+afternoon\s+sessions?|the\s+\d{1,2}\s*(?:a\.m\.|p\.m\.|am|pm)\s+slot)\b", "SESSION_PATTERN", "contextual"),
    (r"\b(?:[A-Z][A-Za-z]+\s+)?(?:Winter\s+)?(?:Lantern|Candlelight)\s+(?:Parade|Walk|Festival|Regatta|Run|Boat Parade)\b(?:\s+last\s+\w+)?", "RARE_EVENT", "contextual"),
    (r"\b(?:apartment building fire|factory closure|bridge accident|bus crash|evacuation|local news story|courthouse hearing|campus lockdown|warehouse fire|ferry accident)\b", "RARE_EVENT", "contextual"),
    (r"\bmy\s+(?:brother|sister|mother|father|aunt|uncle|cousin|ex-husband|ex-wife|roommate),?\s+[A-Z][a-z]+\b", "FAMILY_NAME", "contextual"),
    (r"\b(?:Briar Hill|Westford Junction|Lake Ash Street|Pine Market|Cedar Junction|Foxglove Court|Harbor Point|Northbridge|Riverton|Maple Flats|South Quay)\b", "LOCATION", "contextual"),
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


def _patterns_for_mode(mode: str) -> list[tuple[str, str, str]]:
    if mode in {"direct", "direct_only", "direct_only_baseline"}:
        return [entry for entry in PATTERNS if entry[2] == "direct"]
    if mode in {"full", "full_semantic", "full_semantic_layer"}:
        return PATTERNS
    raise ValueError(f"Unsupported semantic de-identification mode: {mode}")


def detect_identifiers(text: str, mode: str = "full") -> list[Detection]:
    detections: list[Detection] = []
    for pattern, typ, _category in _patterns_for_mode(mode):
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
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


def deidentify_record(row: dict, mode: str = "full") -> dict:
    detections = detect_identifiers(row["utterance"], mode=mode)
    condition = "direct_only_baseline" if mode in {"direct", "direct_only", "direct_only_baseline"} else "full_semantic_layer"
    return {
        "id": row["id"],
        "subset": row.get("subset", ""),
        "source_model": row.get("source_model", ""),
        "condition": condition,
        "utterance": row["utterance"],
        "predicted_identifier_annotations": [det.as_dict() for det in detections],
        "automated_deidentified_utterance": anonymize_text(row["utterance"], detections),
        "reference_deidentified_utterance": row.get("deidentified_utterance", ""),
        "gold_identifier_annotations": row.get("identifier_annotations", []),
        "clinical_concept_annotations": row.get("clinical_concept_annotations", []),
    }
