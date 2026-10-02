from __future__ import annotations

import re
from typing import Any


DIRECT_FIELDS = ["patient_code", "clinician_id", "location_tag", "room_id"]
TEMPORAL_FIELDS = ["creation_time", "upload_time"]
DEVICE_FIELDS = ["device_model", "software_version"]
FILENAME_PATH_FIELDS = ["filename", "file_path"]
FREE_TEXT_FIELDS = ["background_context", "sidecar_note", "calendar_label"]
UTILITY_FIELDS = ["study_subject_id", "session_id", "relative_time", "broad_setting", "duration_bin", "sample_rate", "channels"]
ALLOWED_CONTEXT_PLACEHOLDERS = {"", "non_speech_context_removed", "context_note_removed", "calendar_label_removed", None}
ALLOWED_BROAD_SETTINGS = {"clinic", "home", "remote", "inpatient"}

SENSITIVE_PATTERNS = [
    re.compile(r"\b[A-Z]{2}_\d{3}\b"),
    re.compile(r"\bDr[_\.\s-]?[A-Z][A-Za-z]+\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b(?:Green Valley|Cedar Lane|Northbridge|Silver Harbor|Ward\s+\d+[A-Z]?|Room\s+\d+[A-Z]?)\b", re.I),
    re.compile(r"\b(?:Zoom|Tascam|Blue Yeti|Audacity|RecorderSuite|WaveLab|ClinicCapture)\b", re.I),
]

FREE_TEXT_SENSITIVE_PATTERNS = [
    *SENSITIVE_PATTERNS,
    re.compile(r"\b(?:caregiver|mother|father|sister|brother|husband|wife|partner)\b", re.I),
    re.compile(r"\b(?:appointment|follow[-\s]?up|discharge|admission|intake)\b", re.I),
    re.compile(r"\b(?:patient|subject)\s*(?:code|id|identifier)\b", re.I),
]


def _has_value(record: dict, fields: list[str]) -> bool:
    return any(record.get(field) not in (None, "") for field in fields)


def _contains_sensitive_text(value: Any) -> bool:
    text = str(value or "")
    return any(pattern.search(text) for pattern in SENSITIVE_PATTERNS)


def _text_fields_leak(record: dict, fields: list[str]) -> bool:
    return any(_contains_sensitive_text(record.get(field, "")) for field in fields)


def _filename_path_leakage(record: dict) -> bool:
    return _text_fields_leak(record, FILENAME_PATH_FIELDS)


def _free_text_leakage(record: dict) -> bool:
    for field in FREE_TEXT_FIELDS:
        value = record.get(field)
        if value in ALLOWED_CONTEXT_PLACEHOLDERS:
            continue
        text = str(value or "")
        if any(pattern.search(text) for pattern in FREE_TEXT_SENSITIVE_PATTERNS):
            return True
    return False


def _direct_metadata_leakage(record: dict) -> bool:
    return _has_value(record, DIRECT_FIELDS)


def _temporal_uniqueness_risk(record: dict) -> bool:
    return _has_value(record, TEMPORAL_FIELDS) or _text_fields_leak(record, ["filename", "file_path", "calendar_label"])


def _location_specificity_risk(record: dict) -> bool:
    if _has_value(record, ["location_tag", "room_id"]):
        return True
    broad = record.get("broad_setting")
    if broad not in (None, "") and broad not in ALLOWED_BROAD_SETTINGS:
        return True
    return _text_fields_leak(record, ["filename", "file_path", "background_context", "sidecar_note", "calendar_label"])


def _device_fingerprinting_risk(record: dict, target_sample_rate: int = 16000, target_channels: str = "mono") -> bool:
    if _has_value(record, DEVICE_FIELDS):
        return True
    if record.get("sample_rate") not in (None, target_sample_rate):
        return True
    if record.get("bit_depth") not in (None, 16):
        return True
    if record.get("channels") not in (None, target_channels):
        return True
    return False


def _audio_standardized(record: dict, target_sample_rate: int = 16000, target_channels: str = "mono") -> bool:
    return record.get("sample_rate") == target_sample_rate and record.get("bit_depth") == 16 and record.get("channels") == target_channels


def linkage_risk_score(record: dict, sanitized: bool = False) -> float:
    score = 0.0
    score += 0.20 * _direct_metadata_leakage(record)
    score += 0.15 * _temporal_uniqueness_risk(record)
    score += 0.15 * _location_specificity_risk(record)
    score += 0.15 * _device_fingerprinting_risk(record)
    score += 0.20 * _filename_path_leakage(record)
    score += 0.15 * _free_text_leakage(record)
    if sanitized:
        # Generalized longitudinal fields are useful but still retain low-level quasi-information.
        score += 0.03 * bool(record.get("broad_setting"))
        score += 0.02 * bool(record.get("duration_bin"))
        score += 0.02 * bool(record.get("session_id") or record.get("relative_time"))
    return min(score, 1.0)


def _row_metrics(record: dict, sanitized: bool = False) -> dict[str, bool | float]:
    return {
        "direct_metadata_leakage": _direct_metadata_leakage(record),
        "filename_path_leakage": _filename_path_leakage(record),
        "free_text_context_leakage": _free_text_leakage(record),
        "device_fingerprinting_risk": _device_fingerprinting_risk(record),
        "temporal_uniqueness_risk": _temporal_uniqueness_risk(record),
        "location_specificity_risk": _location_specificity_risk(record),
        "audio_standardized": _audio_standardized(record),
        "linkage_risk_score": linkage_risk_score(record, sanitized=sanitized),
    }


def _rate(rows: list[dict], key: str) -> float:
    return sum(1 for row in rows if row.get(key)) / len(rows) if rows else 0.0


def _mean(rows: list[dict], key: str) -> float:
    return sum(float(row.get(key, 0.0)) for row in rows) / len(rows) if rows else 0.0


def _utility_rate(sanitized: list[dict]) -> float:
    total = len(sanitized) * len(UTILITY_FIELDS)
    retained = sum(1 for row in sanitized for field in UTILITY_FIELDS if row.get(field) not in (None, ""))
    return retained / total if total else 0.0


def evaluate_metadata(raw: list[dict], sanitized: list[dict]) -> tuple[list[dict[str, float | str]], list[dict[str, float | str]]]:
    raw_metrics = [_row_metrics(row, sanitized=False) for row in raw]
    sanitized_metrics = [_row_metrics(row, sanitized=True) for row in sanitized]
    summary = [
        {"metric": "direct_metadata_leakage_rate_before", "value": _rate(raw_metrics, "direct_metadata_leakage")},
        {"metric": "direct_metadata_leakage_rate_after", "value": _rate(sanitized_metrics, "direct_metadata_leakage")},
        {"metric": "filename_path_leakage_rate_before", "value": _rate(raw_metrics, "filename_path_leakage")},
        {"metric": "filename_path_leakage_rate_after", "value": _rate(sanitized_metrics, "filename_path_leakage")},
        {"metric": "free_text_context_leakage_rate_before", "value": _rate(raw_metrics, "free_text_context_leakage")},
        {"metric": "free_text_context_leakage_rate_after", "value": _rate(sanitized_metrics, "free_text_context_leakage")},
        {"metric": "device_fingerprinting_risk_before", "value": _rate(raw_metrics, "device_fingerprinting_risk")},
        {"metric": "device_fingerprinting_risk_after", "value": _rate(sanitized_metrics, "device_fingerprinting_risk")},
        {"metric": "temporal_uniqueness_risk_before", "value": _rate(raw_metrics, "temporal_uniqueness_risk")},
        {"metric": "temporal_uniqueness_risk_after", "value": _rate(sanitized_metrics, "temporal_uniqueness_risk")},
        {"metric": "location_specificity_risk_before", "value": _rate(raw_metrics, "location_specificity_risk")},
        {"metric": "location_specificity_risk_after", "value": _rate(sanitized_metrics, "location_specificity_risk")},
        {"metric": "linkage_risk_score_before", "value": _mean(raw_metrics, "linkage_risk_score")},
        {"metric": "linkage_risk_score_after", "value": _mean(sanitized_metrics, "linkage_risk_score")},
        {"metric": "longitudinal_utility_preservation", "value": _utility_rate(sanitized)},
        {"metric": "audio_standardization_success", "value": _rate(sanitized_metrics, "audio_standardized")},
    ]

    by_subset = []
    subsets = sorted({row.get("metadata_subset", "unspecified") for row in raw})
    sanitized_by_id = {row.get("id"): row for row in sanitized}
    for subset in subsets:
        raw_subset = [row for row in raw if row.get("metadata_subset", "unspecified") == subset]
        san_subset = [sanitized_by_id[row.get("id")] for row in raw_subset if row.get("id") in sanitized_by_id]
        raw_subset_metrics = [_row_metrics(row, sanitized=False) for row in raw_subset]
        san_subset_metrics = [_row_metrics(row, sanitized=True) for row in san_subset]
        by_subset.append(
            {
                "metadata_subset": subset,
                "n_records": len(raw_subset),
                "direct_metadata_leakage_before": _rate(raw_subset_metrics, "direct_metadata_leakage"),
                "direct_metadata_leakage_after": _rate(san_subset_metrics, "direct_metadata_leakage"),
                "filename_path_leakage_before": _rate(raw_subset_metrics, "filename_path_leakage"),
                "filename_path_leakage_after": _rate(san_subset_metrics, "filename_path_leakage"),
                "free_text_context_leakage_before": _rate(raw_subset_metrics, "free_text_context_leakage"),
                "free_text_context_leakage_after": _rate(san_subset_metrics, "free_text_context_leakage"),
                "linkage_risk_score_before": _mean(raw_subset_metrics, "linkage_risk_score"),
                "linkage_risk_score_after": _mean(san_subset_metrics, "linkage_risk_score"),
                "longitudinal_utility_preservation": _utility_rate(san_subset),
                "audio_standardization_success": _rate(san_subset_metrics, "audio_standardized"),
            }
        )
    return summary, by_subset
