from __future__ import annotations

import random
from datetime import datetime, timedelta


SETTINGS = ["clinic", "home", "remote", "inpatient"]
DEVICES = ["Zoom H4n Pro", "Tascam DR-05X", "Blue Yeti", "ClinicRecorder X2"]
SOFTWARE = ["Audacity 3.4.2", "RecorderSuite 2.1", "WaveLab 12", "ClinicCapture 5.0"]
LOCATIONS = ["Green Valley Mental Health Center", "Cedar Lane Clinic", "Northbridge Hospital", "Silver Harbor Clinic"]
CLINICIANS = ["Dr_Wilson", "Dr_Marlow", "Dr_Benton", "Dr_Rivers"]
PATIENTS = ["EC_042", "LM_117", "OK_203", "PV_314", "RA_502"]
CONTEXTS = ["clinic waiting room announcement", "home television audible", "remote call notification", "inpatient hallway noise"]


def duration_bin(seconds: int | float) -> str:
    minutes = float(seconds) / 60
    if minutes < 5:
        return "<5_min"
    if minutes < 10:
        return "5-10_min"
    if minutes < 20:
        return "10-20_min"
    return "20+_min"


def generate_fallback_metadata(n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    base = datetime(2025, 1, 6, 9, 0, 0)
    rows = []
    for idx in range(1, n + 1):
        patient = PATIENTS[(idx - 1) % len(PATIENTS)]
        session = ((idx - 1) // len(PATIENTS)) + 1
        location = rng.choice(LOCATIONS)
        creation = base + timedelta(days=7 * session, minutes=rng.randint(0, 420))
        duration = rng.randint(240, 1500)
        setting = rng.choice(SETTINGS)
        rows.append(
            {
                "id": f"META{idx:03d}",
                "filename": f"{creation.date()}_{location.replace(' ', '')}_{patient}_session{session:02d}.wav",
                "creation_time": creation.isoformat(),
                "device_model": rng.choice(DEVICES),
                "software_version": rng.choice(SOFTWARE),
                "location_tag": location,
                "clinician_id": rng.choice(CLINICIANS),
                "patient_code": patient,
                "session_number": session,
                "sample_rate": rng.choice([16000, 44100, 48000]),
                "bit_depth": rng.choice([16, 24, 32]),
                "channels": rng.choice(["mono", "stereo"]),
                "duration_seconds": duration,
                "background_context": rng.choice(CONTEXTS),
                "broad_setting_hint": setting,
                "original_risk_fields": ["filename", "creation_time", "device_model", "software_version", "location_tag", "clinician_id", "patient_code", "background_context"],
            }
        )
    return rows
