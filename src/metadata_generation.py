from __future__ import annotations

import random
from datetime import datetime, timedelta


METADATA_SUBSETS = ["structured_standard", "filename_path_leakage", "free_text_context"]
SETTINGS = ["clinic", "home", "remote", "inpatient"]
DEVICES = ["Zoom H4n Pro", "Tascam DR-05X", "Blue Yeti", "ClinicRecorder X2"]
SOFTWARE = ["Audacity 3.4.2", "RecorderSuite 2.1", "WaveLab 12", "ClinicCapture 5.0"]
LOCATIONS = ["Green Valley Mental Health Center", "Cedar Lane Clinic", "Northbridge Hospital", "Silver Harbor Clinic"]
ROOMS = ["Room 3B", "Consultation Room 12", "Ward 7A", "Telehealth Booth 2"]
CLINICIANS = ["Dr_Wilson", "Dr_Marlow", "Dr_Benton", "Dr_Rivers"]
PATIENTS = ["EC_042", "LM_117", "OK_203", "PV_314", "RA_502"]
CONTEXTS = [
    "clinic waiting room announcement",
    "home television audible",
    "remote call notification",
    "inpatient hallway noise",
]
FREE_TEXT_CONTEXTS = [
    "mother's voice says the patient nickname during the first minute",
    "clinic announcement mentions Cedar Lane intake desk",
    "remote call notification shows Dr_Benton calendar reminder",
    "hallway conversation mentions Ward 7A and discharge paperwork",
]


def duration_bin(seconds: int | float) -> str:
    minutes = float(seconds) / 60
    if minutes < 5:
        return "<5_min"
    if minutes < 10:
        return "5-10_min"
    if minutes < 20:
        return "10-20_min"
    return "20+_min"


def _subset_targets(n: int) -> list[str]:
    base = n // len(METADATA_SUBSETS)
    remainder = n % len(METADATA_SUBSETS)
    labels: list[str] = []
    for idx, subset in enumerate(METADATA_SUBSETS):
        labels.extend([subset] * (base + (1 if idx < remainder else 0)))
    return labels


def _safe_location(location: str) -> str:
    return location.replace(" ", "")


def _standard_filename(record_id: str, session: int) -> str:
    return f"audio_{record_id.lower()}_session{session:02d}.wav"


def _leaky_filename(creation: datetime, location: str, patient: str, clinician: str, session: int) -> str:
    clinician_token = clinician.replace("_", "")
    return f"{creation.date()}_{_safe_location(location)}_{patient}_{clinician_token}_session{session:02d}.wav"


def _leaky_path(location: str, patient: str, clinician: str, session: int) -> str:
    return f"/recordings/{_safe_location(location)}/{clinician}/{patient}/session_{session:02d}/raw_audio.wav"


def generate_fallback_metadata(n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    base = datetime(2025, 1, 6, 9, 0, 0)
    subsets = _subset_targets(n)
    rows = []
    for idx, subset in enumerate(subsets, start=1):
        patient = PATIENTS[(idx - 1) % len(PATIENTS)]
        session = ((idx - 1) // len(PATIENTS)) + 1
        location = rng.choice(LOCATIONS)
        clinician = rng.choice(CLINICIANS)
        creation = base + timedelta(days=7 * session, minutes=rng.randint(0, 420))
        upload = creation + timedelta(hours=rng.randint(1, 30), minutes=rng.randint(0, 59))
        duration = rng.randint(240, 1500)
        setting = rng.choice(SETTINGS)
        room = rng.choice(ROOMS)
        record_id = f"META{idx:03d}"

        filename = _standard_filename(record_id, session)
        file_path = f"/recordings/study_audio/session_{session:02d}/{filename}"
        background_context = rng.choice(CONTEXTS)
        sidecar_note = "routine recording note removed from public release"
        calendar_label = f"study_session_{session:02d}"
        expected_risk_categories = ["direct_fields", "temporal", "device_fingerprint", "location_specificity"]

        if subset == "filename_path_leakage":
            filename = _leaky_filename(creation, location, patient, clinician, session)
            file_path = _leaky_path(location, patient, clinician, session)
            expected_risk_categories += ["filename_leakage", "path_leakage"]
        elif subset == "free_text_context":
            background_context = rng.choice(FREE_TEXT_CONTEXTS)
            sidecar_note = (
                f"Recorded after appointment with {clinician} at {location}; "
                f"patient code {patient}; room tag {room}; caregiver voice audible."
            )
            calendar_label = f"{patient}_{clinician}_{_safe_location(location)}_followup_{creation.date()}"
            expected_risk_categories += ["free_text_context", "calendar_label"]

        rows.append(
            {
                "id": record_id,
                "metadata_subset": subset,
                "filename": filename,
                "file_path": file_path,
                "creation_time": creation.isoformat(),
                "upload_time": upload.isoformat(),
                "device_model": rng.choice(DEVICES),
                "software_version": rng.choice(SOFTWARE),
                "location_tag": location,
                "room_id": room,
                "clinician_id": clinician,
                "patient_code": patient,
                "session_number": session,
                "sample_rate": rng.choice([16000, 44100, 48000]),
                "bit_depth": rng.choice([16, 24, 32]),
                "channels": rng.choice(["mono", "stereo"]),
                "duration_seconds": duration,
                "background_context": background_context,
                "sidecar_note": sidecar_note,
                "calendar_label": calendar_label,
                "broad_setting_hint": setting,
                "original_risk_fields": [
                    "filename",
                    "file_path",
                    "creation_time",
                    "upload_time",
                    "device_model",
                    "software_version",
                    "location_tag",
                    "room_id",
                    "clinician_id",
                    "patient_code",
                    "background_context",
                    "sidecar_note",
                    "calendar_label",
                ],
                "expected_risk_categories": expected_risk_categories,
            }
        )
    return rows
