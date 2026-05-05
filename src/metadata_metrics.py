from __future__ import annotations


IDENTIFYING_FIELDS = [
    "filename",
    "creation_time",
    "device_model",
    "software_version",
    "location_tag",
    "clinician_id",
    "patient_code",
    "sample_rate",
    "bit_depth",
    "channels",
    "background_context",
    "duration_seconds",
]

UTILITY_FIELDS = ["study_subject_id", "session_id", "relative_time", "broad_setting", "duration_bin", "sample_rate", "channels"]


def linkage_risk_score(record: dict) -> float:
    has_patient_or_clinician = bool(record.get("patient_code") or record.get("clinician_id"))
    has_exact_location = bool(record.get("location_tag"))
    has_exact_timestamp = bool(record.get("creation_time"))
    has_device = bool(record.get("device_model") or record.get("software_version") or record.get("bit_depth") not in (None, 16) or record.get("channels") not in (None, "mono"))
    has_unique_file = bool(record.get("filename"))
    return (
        0.30 * has_patient_or_clinician
        + 0.20 * has_exact_location
        + 0.20 * has_exact_timestamp
        + 0.15 * has_device
        + 0.15 * has_unique_file
    )


def sanitized_linkage_risk(record: dict) -> float:
    has_exact_location = record.get("broad_setting") not in (None, "", "clinic", "home", "remote", "inpatient")
    has_exact_timestamp = bool(record.get("creation_time"))
    has_device = record.get("sample_rate") not in (None, 16000) or record.get("channels") not in (None, "mono")
    has_unique_file = bool(record.get("filename"))
    has_patient_or_clinician = bool(record.get("patient_code") or record.get("clinician_id"))
    return (
        0.30 * has_patient_or_clinician
        + 0.20 * has_exact_location
        + 0.20 * has_exact_timestamp
        + 0.15 * has_device
        + 0.15 * has_unique_file
    )


def evaluate_metadata(raw: list[dict], sanitized: list[dict]) -> list[dict[str, float | str]]:
    before_identifying = sum(1 for row in raw for field in IDENTIFYING_FIELDS if row.get(field) not in (None, ""))
    residual_identifying = 0
    for row in sanitized:
        for field in IDENTIFYING_FIELDS:
            value = row.get(field)
            if value in (None, ""):
                continue
            # Standardized technical fields are retained for utility but no longer act as device fingerprints.
            if field == "sample_rate" and int(value) == 16000:
                continue
            if field == "bit_depth" and int(value) == 16:
                continue
            if field == "channels" and value == "mono":
                continue
            if field == "background_context" and value == "non_speech_context_removed":
                continue
            residual_identifying += 1
    utility_retained = sum(1 for row in sanitized for field in UTILITY_FIELDS if row.get(field) not in (None, ""))
    utility_total = len(sanitized) * len(UTILITY_FIELDS)
    before_risk = sum(linkage_risk_score(row) for row in raw) / len(raw) if raw else 0.0
    after_risk = sum(sanitized_linkage_risk(row) for row in sanitized) / len(sanitized) if sanitized else 0.0
    leakage_before = 1.0 if before_identifying else 0.0
    leakage_after = residual_identifying / before_identifying if before_identifying else 0.0
    return [
        {"metric": "metadata_leakage_rate_before", "value": leakage_before},
        {"metric": "metadata_leakage_rate_after", "value": leakage_after},
        {"metric": "field_sanitization_rate", "value": 1 - leakage_after},
        {"metric": "linkage_risk_score_before", "value": before_risk},
        {"metric": "linkage_risk_score_after", "value": after_risk},
        {"metric": "longitudinal_utility_preservation", "value": utility_retained / utility_total if utility_total else 0.0},
    ]
