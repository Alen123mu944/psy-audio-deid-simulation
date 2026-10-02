from __future__ import annotations

from hashlib import sha256

from .metadata_generation import duration_bin


def _stable_number(value: str, modulo: int = 9999) -> int:
    digest = sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo + 1


def broad_setting(record: dict) -> str:
    text = " ".join(
        str(record.get(key, "")).lower()
        for key in ["location_tag", "background_context", "sidecar_note", "file_path", "broad_setting_hint"]
    )
    if "home" in text:
        return "home"
    if "remote" in text or "call" in text or "telehealth" in text:
        return "remote"
    if "inpatient" in text or "hospital" in text or "ward" in text or "hallway" in text:
        return "inpatient"
    return "clinic"


def sanitize_record(record: dict, target_sample_rate: int = 16000, target_channels: str = "mono") -> dict:
    patient_code = str(record.get("patient_code", record.get("id", "")))
    session_number = int(record.get("session_number", 1))
    subject_num = _stable_number(patient_code)
    file_source = "|".join(str(record.get(key, "")) for key in ["filename", "file_path", "id"])
    return {
        "id": record.get("id"),
        "metadata_subset": record.get("metadata_subset", "unspecified"),
        "file_id": f"AUDIO_{_stable_number(file_source):04d}",
        "study_subject_id": f"SUBJ_{subject_num:04d}",
        "session_id": f"SESS_{session_number:02d}",
        "relative_time": f"session_{session_number:02d}",
        "broad_setting": broad_setting(record),
        "sample_rate": target_sample_rate,
        "bit_depth": 16,
        "channels": target_channels,
        "duration_bin": duration_bin(record.get("duration_seconds", 0)),
        "background_context": "non_speech_context_removed",
        "sidecar_note": "context_note_removed",
        "calendar_label": "calendar_label_removed",
        "linkage_key_stored_separately": True,
        "metadata_policy_version": "metadata_sanitization_v2",
    }


def sanitize_records(records: list[dict], target_sample_rate: int = 16000, target_channels: str = "mono") -> list[dict]:
    return [sanitize_record(record, target_sample_rate, target_channels) for record in records]
