from __future__ import annotations

import random
from typing import Any


IDENTIFIER_TYPES = [
    "PERSON",
    "DATE",
    "LOCATION",
    "ORGANIZATION",
    "FAMILY_NAME",
    "SCHOOL_WORKPLACE",
    "ADDRESS",
    "CONTACT",
    "RARE_EVENT",
    "SESSION_PATTERN",
    "MEDICAL_RECORD",
    "CLINICIAN_NAME",
    "HEALTHCARE_ORGANIZATION",
]

CLINICAL_TYPES = [
    "SYMPTOM",
    "RISK_BEHAVIOR",
    "MEDICATION",
    "TREATMENT",
    "FUNCTIONAL_IMPAIRMENT",
    "FAMILY_STRESS",
    "TRAUMA",
    "SUBSTANCE_USE",
    "SOCIAL_WITHDRAWAL",
    "SLEEP_PROBLEM",
]

_TEMPLATES = [
    {
        "utterance": "I told Dr. {doctor} at {clinic} that since {date} I have had insomnia and depressed mood, and my brother {relative} keeps calling me at {contact}.",
        "ids": [("Dr. {doctor}", "CLINICIAN_NAME"), ("{clinic}", "HEALTHCARE_ORGANIZATION"), ("{date}", "DATE"), ("my brother {relative}", "FAMILY_NAME"), ("{contact}", "CONTACT")],
        "concepts": [("insomnia", "SLEEP_PROBLEM"), ("depressed mood", "SYMPTOM")],
        "deid": "I told [CLINICIAN] at [mental health clinic] that since [relative time] I have had insomnia and depressed mood, and my brother keeps contacting me.",
    },
    {
        "utterance": "After the apartment building fire near {location}, I stopped going to {school} and started drinking most nights because of anxiety.",
        "ids": [("apartment building fire", "RARE_EVENT"), ("{location}", "LOCATION"), ("{school}", "SCHOOL_WORKPLACE")],
        "concepts": [("drinking most nights", "SUBSTANCE_USE"), ("anxiety", "SYMPTOM")],
        "deid": "After [major stressful event] near [location], I stopped going to [school/workplace] and started drinking most nights because of anxiety.",
    },
    {
        "utterance": "My second appointment is every Tuesday at 4 p.m. with therapist {doctor}, and I still feel socially withdrawn at {workplace}.",
        "ids": [("second appointment", "SESSION_PATTERN"), ("every Tuesday at 4 p.m.", "SESSION_PATTERN"), ("therapist {doctor}", "CLINICIAN_NAME"), ("{workplace}", "SCHOOL_WORKPLACE")],
        "concepts": [("socially withdrawn", "SOCIAL_WITHDRAWAL"), ("feel socially withdrawn", "SYMPTOM")],
        "deid": "My [regular therapy schedule] is with [CLINICIAN], and I still feel socially withdrawn at [school/workplace].",
    },
    {
        "utterance": "The nurse wrote record {record} after I said the medication helped my panic attacks, but I missed work at {workplace} again.",
        "ids": [("record {record}", "MEDICAL_RECORD"), ("{workplace}", "SCHOOL_WORKPLACE")],
        "concepts": [("medication", "MEDICATION"), ("panic attacks", "SYMPTOM"), ("missed work", "FUNCTIONAL_IMPAIRMENT")],
        "deid": "The nurse wrote [record identifier removed] after I said the medication helped my panic attacks, but I missed work at [school/workplace] again.",
    },
]

_VALUES = {
    "doctor": ["Marlow", "Benton", "Keller", "Novak", "Rivers"],
    "clinic": ["Cedar Lane Mental Health Center", "Northbridge Clinic", "Silver Harbor Hospital", "Maple Rest Clinic"],
    "date": ["March 12", "last October", "2025-02-18", "two weeks after New Year"],
    "relative": ["Lena", "Owen", "Mira", "Caleb"],
    "contact": ["555-0138", "patient77@example.test", "555-0194"],
    "location": ["Briar Hill", "Westford Junction", "Lake Ash Street", "Pine Market"],
    "school": ["Riverview High School", "Oak Valley University", "North Gate College"],
    "workplace": ["Harbor Box Logistics", "Blue Finch Company", "Linden Market Warehouse"],
    "record": ["MR-88421", "HX-19022", "REC-55204"],
}


def _fill(template: str, rng: random.Random) -> str:
    values = {key: rng.choice(items) for key, items in _VALUES.items()}
    return template.format(**values)


def _annotate(text: str, patterns: list[tuple[str, str]], rng: random.Random) -> list[dict[str, Any]]:
    annotations = []
    for pattern, typ in patterns:
        span = _fill(pattern, rng)
        start = text.find(span)
        if start >= 0:
            annotations.append(
                {
                    "span": span,
                    "start": start,
                    "end": start + len(span),
                    "type": typ,
                    "risk_level": "high" if typ in {"PERSON", "CONTACT", "ADDRESS", "MEDICAL_RECORD", "CLINICIAN_NAME"} else "medium",
                    "recommended_action": "remove" if typ in {"CONTACT", "ADDRESS", "MEDICAL_RECORD"} else "generalize",
                }
            )
    return annotations


def generate_fallback_transcripts(n: int, seed: int = 42) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    for idx in range(1, n + 1):
        template = _TEMPLATES[(idx - 1) % len(_TEMPLATES)]
        # Use one value map per row so annotation spans match the utterance.
        values = {key: rng.choice(items) for key, items in _VALUES.items()}
        utterance = template["utterance"].format(**values)
        identifiers = []
        for pattern, typ in template["ids"]:
            span = pattern.format(**values)
            start = utterance.find(span)
            if start >= 0:
                identifiers.append(
                    {
                        "span": span,
                        "start": start,
                        "end": start + len(span),
                        "type": typ,
                        "risk_level": "high" if typ in {"CONTACT", "MEDICAL_RECORD", "CLINICIAN_NAME"} else "medium",
                        "recommended_action": "remove" if typ in {"CONTACT", "MEDICAL_RECORD"} else "generalize",
                    }
                )
        concepts = []
        for span, typ in template["concepts"]:
            start = utterance.find(span)
            if start >= 0:
                concepts.append({"span": span, "start": start, "end": start + len(span), "type": typ, "should_preserve": True})
        rows.append(
            {
                "id": f"SIM{idx:03d}",
                "utterance": utterance,
                "identifier_annotations": identifiers,
                "clinical_concept_annotations": concepts,
                "deidentified_utterance": template["deid"],
            }
        )
    return rows


def quality_report(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    real_markers = ["Mayo Clinic", "Harvard", "NHS", "Kaiser"]
    extreme_markers = ["graphic", "murder", "torture"]
    report = []
    for row in rows:
        text = row.get("utterance", "")
        ids = row.get("identifier_annotations", [])
        concepts = row.get("clinical_concept_annotations", [])
        missing_ids = [ann.get("span", "") for ann in ids if ann.get("span", "") not in text]
        missing_concepts = [ann.get("span", "") for ann in concepts if ann.get("span", "") not in text]
        flag_real = any(marker.lower() in text.lower() for marker in real_markers)
        flag_extreme = any(marker.lower() in text.lower() for marker in extreme_markers)
        report.append(
            {
                "id": row.get("id", ""),
                "is_valid_json": True,
                "n_identifiers": len(ids),
                "n_clinical_concepts": len(concepts),
                "missing_identifier_spans": ";".join(missing_ids),
                "missing_concept_spans": ";".join(missing_concepts),
                "flag_real_world_entity": flag_real,
                "flag_extreme_content": flag_extreme,
                "needs_review": len(ids) < 2 or len(concepts) < 1 or bool(missing_ids) or bool(missing_concepts) or flag_real or flag_extreme,
            }
        )
    return report
