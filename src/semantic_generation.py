from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
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

DIRECT_IDENTIFIER_TYPES = {"PERSON", "CONTACT", "ADDRESS", "MEDICAL_RECORD", "CLINICIAN_NAME", "DATE"}
CONTEXTUAL_IDENTIFIER_TYPES = {
    "LOCATION",
    "ORGANIZATION",
    "HEALTHCARE_ORGANIZATION",
    "SCHOOL_WORKPLACE",
    "FAMILY_NAME",
    "RARE_EVENT",
    "SESSION_PATTERN",
}

CLINICAL_TYPES = [
    "SYMPTOM",
    "RISK_IDEATION",
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

SUBSETS = ["clean_direct", "contextual_paraphrased", "asr_noisy"]

SUBSET_INSTRUCTIONS = {
    "clean_direct": "Use clear direct identifiers such as a fictional clinician name, contact detail, record code, date, or address. Also include one or two contextual identifiers.",
    "contextual_paraphrased": "Use indirect or contextual identifiers, such as a unique local event, family role plus first name, school/workplace, therapy schedule, or small fictional location. Avoid obvious phone numbers unless useful.",
    "asr_noisy": "Write the utterance as if it came from imperfect ASR: light disfluency, missing punctuation, spelled-out times, or minor casing variation. Annotation spans must still copy the exact noisy text.",
}

FICTIONAL_ENTITY_HINTS = [
    "Cedar Lane Mental Health Center",
    "Northbridge Clinic",
    "Silver Harbor Hospital",
    "Maple Rest Clinic",
    "Riverview High School",
    "Oak Valley College",
    "Harbor Box Logistics",
    "Blue Finch Company",
    "Linden Market Warehouse",
    "Briar Hill",
    "Westford Junction",
    "Lake Ash Street",
    "Pine Market",
]

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


@dataclass
class GenerationResult:
    records: list[dict[str, Any]]
    raw_response: str
    error: str = ""


def _risk_level(identifier_type: str) -> str:
    if identifier_type in {"CONTACT", "ADDRESS", "MEDICAL_RECORD", "CLINICIAN_NAME", "PERSON"}:
        return "high"
    if identifier_type in {"RARE_EVENT", "SESSION_PATTERN", "SCHOOL_WORKPLACE", "FAMILY_NAME"}:
        return "medium"
    return "medium"


def _recommended_action(identifier_type: str) -> str:
    if identifier_type in {"CONTACT", "ADDRESS", "MEDICAL_RECORD"}:
        return "remove"
    return "generalize"



def _find_span(text: str, span: str) -> tuple[int, int]:
    start = text.find(span)
    if start >= 0:
        return start, start + len(span)
    lowered = text.lower()
    lowered_span = span.lower()
    start = lowered.find(lowered_span)
    if start >= 0:
        return start, start + len(span)
    return -1, -1


def _normalize_annotations(text: str, annotations: Any, allowed_types: set[str]) -> tuple[list[dict[str, Any]], list[str]]:
    normalized: list[dict[str, Any]] = []
    problems: list[str] = []
    if not isinstance(annotations, list):
        return normalized, ["annotations_not_list"]
    for idx, ann in enumerate(annotations):
        if not isinstance(ann, dict):
            problems.append(f"annotation_{idx}_not_object")
            continue
        span = str(ann.get("span", "")).strip()
        typ = str(ann.get("type", "")).strip().upper()
        if not span or typ not in allowed_types:
            problems.append(f"annotation_{idx}_missing_span_or_type")
            continue
        start, end = _find_span(text, span)
        if start < 0:
            problems.append(f"annotation_{idx}_span_not_found:{span}")
            continue
        normalized.append(
            {
                "span": text[start:end],
                "start": start,
                "end": end,
                "type": typ,
                "risk_level": str(ann.get("risk_level") or _risk_level(typ)).lower(),
                "recommended_action": str(ann.get("recommended_action") or _recommended_action(typ)).lower(),
            }
        )
    normalized.sort(key=lambda item: (item["start"], item["end"]))
    return normalized, problems


def normalize_transcript_record(raw: dict[str, Any], row_id: str, subset: str, source_model: str) -> tuple[dict[str, Any] | None, list[str]]:
    problems: list[str] = []
    utterance = str(raw.get("utterance", "")).strip()
    if not utterance:
        return None, ["missing_utterance"]
    identifiers, id_problems = _normalize_annotations(utterance, raw.get("identifier_annotations", []), set(IDENTIFIER_TYPES))
    concepts, concept_problems = _normalize_annotations(utterance, raw.get("clinical_concept_annotations", []), set(CLINICAL_TYPES))
    problems.extend(id_problems)
    problems.extend(concept_problems)
    if len(identifiers) < 2:
        problems.append("too_few_identifiers")
    if len(concepts) < 1:
        problems.append("too_few_clinical_concepts")
    if problems:
        return None, problems
    return (
        {
            "id": row_id,
            "subset": subset,
            "source_model": source_model,
            "utterance": utterance,
            "identifier_annotations": identifiers,
            "clinical_concept_annotations": concepts,
            "deidentified_utterance": str(raw.get("deidentified_utterance", "")).strip(),
        },
        [],
    )


def build_generation_prompt(batch_size: int, subset: str, batch_index: int, seed: int) -> str:
    fiction_hint = ", ".join(FICTIONAL_ENTITY_HINTS)
    return f"""You are generating fully synthetic data for a privacy-preserving psychiatric audio de-identification simulation.

Return JSON only. Do not include markdown, explanation, or hidden reasoning.
Create exactly {batch_size} record(s) for subset: {subset}.
Batch index: {batch_index}. Random seed label: {seed}.

Purpose:
- Stress-test semantic de-identification of psychiatric interview transcripts.
- Include both privacy-sensitive identifiers and clinically meaningful concepts.
- Use only fictional people, places, institutions, events, record codes, addresses, emails, and phone numbers.
- Do not use real hospitals, real universities, real clinicians, real patients, real addresses, real phone numbers, or real institutions.
- For contact details, use 555-style numbers or example.test emails only.
- Keep self-harm or suicide-risk content non-graphic and clinically phrased; do not include instructions or details.

Subset requirement:
{SUBSET_INSTRUCTIONS.get(subset, '')}

Allowed identifier types:
{', '.join(IDENTIFIER_TYPES)}

Allowed clinical concept types:
{', '.join(CLINICAL_TYPES)}

Fictional entity examples you may reuse or vary:
{fiction_hint}

Output schema:
{{
  "records": [
    {{
      "utterance": "one psychiatric interview-like patient utterance, 1-2 sentences",
      "identifier_annotations": [
        {{"span": "exact substring copied from utterance", "type": "IDENTIFIER_TYPE", "risk_level": "high|medium", "recommended_action": "remove|generalize"}}
      ],
      "clinical_concept_annotations": [
        {{"span": "exact substring copied from utterance", "type": "CLINICAL_TYPE"}}
      ],
      "deidentified_utterance": "same clinical meaning with identifiers removed or generalized"
    }}
  ]
}}

Strict requirements:
- Each record must contain 2-5 identifier annotations.
- Each record must contain 1-4 clinical concept annotations.
- Every annotation span must exactly appear in the utterance.
- Clinical concept spans should preserve symptoms, treatment, medication, risk ideation, functioning, sleep, substance use, trauma, or social withdrawal.
- Do not mark ordinary clinical concepts as identifiers.
- Do not put placeholder tokens like [NAME] in the original utterance.
"""



def clean_model_response(text: str) -> str:
    text = _ANSI_RE.sub("", text)
    text = _THINK_RE.sub("", text)
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_response(text: str) -> dict[str, Any]:
    cleaned = clean_model_response(text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(cleaned[start : end + 1])
    if isinstance(parsed, list):
        return {"records": parsed}
    if not isinstance(parsed, dict):
        raise ValueError("Model response was not a JSON object or list.")
    return parsed



def call_openrouter(prompt: str, model: str, api_key: str, timeout_sec: int = 120, max_tokens: int = 900) -> str:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You return compact valid JSON only. Do not include markdown or explanations."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + api_key.strip(),
            "Content-Type": "application/json",
            "HTTP-Referer": "https://local.codex",
            "X-Title": "psy-audio-deid-simulation",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenRouter returned no choices: {json.dumps(payload)[:1000]}")
    return str(choices[0].get("message", {}).get("content", ""))


def generate_openrouter_batch(
    batch_size: int,
    subset: str,
    batch_index: int,
    seed: int,
    model: str,
    api_key: str,
    timeout_sec: int = 120,
    max_tokens: int = 900,
) -> GenerationResult:
    prompt = build_generation_prompt(batch_size=batch_size, subset=subset, batch_index=batch_index, seed=seed)
    try:
        raw = call_openrouter(prompt=prompt, model=model, api_key=api_key, timeout_sec=timeout_sec, max_tokens=max_tokens)
        parsed = parse_json_response(raw)
        records = parsed.get("records", [])
        if not isinstance(records, list):
            return GenerationResult([], raw, "records_not_list")
        return GenerationResult(records, raw, "")
    except Exception as exc:
        return GenerationResult([], "", str(exc))



def quality_report(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    real_markers = ["Mayo Clinic", "Harvard", "NHS", "Kaiser", "Mass General", "Stanford"]
    extreme_markers = ["graphic", "murder", "torture", "how to"]
    report = []
    for row in rows:
        text = row.get("utterance", "")
        ids = row.get("identifier_annotations", [])
        concepts = row.get("clinical_concept_annotations", [])
        missing_ids = [ann.get("span", "") for ann in ids if ann.get("span", "") not in text]
        missing_concepts = [ann.get("span", "") for ann in concepts if ann.get("span", "") not in text]
        flag_real = any(marker.lower() in text.lower() for marker in real_markers)
        flag_extreme = any(marker.lower() in text.lower() for marker in extreme_markers)
        direct_count = sum(1 for ann in ids if ann.get("type") in DIRECT_IDENTIFIER_TYPES)
        contextual_count = sum(1 for ann in ids if ann.get("type") in CONTEXTUAL_IDENTIFIER_TYPES)
        report.append(
            {
                "id": row.get("id", ""),
                "subset": row.get("subset", ""),
                "source_model": row.get("source_model", ""),
                "is_valid_json": True,
                "n_identifiers": len(ids),
                "n_direct_identifiers": direct_count,
                "n_contextual_identifiers": contextual_count,
                "n_clinical_concepts": len(concepts),
                "missing_identifier_spans": ";".join(missing_ids),
                "missing_concept_spans": ";".join(missing_concepts),
                "flag_real_world_entity": flag_real,
                "flag_extreme_content": flag_extreme,
                "needs_review": len(ids) < 2 or len(concepts) < 1 or bool(missing_ids) or bool(missing_concepts) or flag_real or flag_extreme,
            }
        )
    return report
