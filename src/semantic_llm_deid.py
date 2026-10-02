from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from src.semantic_generation import IDENTIFIER_TYPES


SILICONFLOW_CHAT_COMPLETIONS_URL = "https://api.siliconflow.cn/v1/chat/completions"
DEFAULT_MODEL = "Qwen/Qwen3.6-27B"

REPLACEMENTS = {
    "PERSON": "[PERSON]",
    "CLINICIAN_NAME": "[CLINICIAN]",
    "DATE": "[relative time]",
    "LOCATION": "[location]",
    "ORGANIZATION": "[organization]",
    "HEALTHCARE_ORGANIZATION": "[mental health clinic]",
    "SCHOOL_WORKPLACE": "[school/workplace]",
    "ADDRESS": "[address removed]",
    "CONTACT": "[contact removed]",
    "FAMILY_NAME": "my family member",
    "RARE_EVENT": "[major stressful event]",
    "SESSION_PATTERN": "[regular therapy schedule]",
    "MEDICAL_RECORD": "[record identifier removed]",
}

SYSTEM_PROMPT = """You are a clinical text de-identification engine for synthetic psychiatric interview transcripts.
Return valid compact JSON only. Do not include markdown, explanations, or chain-of-thought.
Your goal is to protect privacy while preserving clinically meaningful psychiatric content."""


def build_deid_prompt(row: dict[str, Any]) -> str:
    replacements = "\n".join(f"- {key} -> {value}" for key, value in REPLACEMENTS.items())
    allowed_types = ", ".join(IDENTIFIER_TYPES)
    return f"""De-identify the following synthetic psychiatric transcript.

Input record:
- id: {row.get('id', '')}
- utterance: {row.get('utterance', '')}

Allowed identifier types:
{allowed_types}

Replacement policy:
{replacements}

Identifier guidance:
- Direct identifiers include patient names, clinician names, contact details, addresses, record numbers, exact dates, and exact times.
- Contextual identifiers include healthcare organizations, school/workplace names, small locations, family names/relationships, rare local events, and recurring therapy schedules.
- Preserve clinical concepts such as symptoms, sleep problems, medication, treatment, functional impairment, substance use, trauma, social withdrawal, and non-graphic risk ideation.
- Do not invent new facts.
- Every predicted span must be copied exactly from the input utterance.
- If an identifier appears in noisy ASR style, still annotate the exact noisy substring.

Return JSON using this schema:
{{
  "id": "{row.get('id', '')}",
  "predicted_identifier_annotations": [
    {{
      "span": "exact substring from the input utterance",
      "type": "one allowed identifier type",
      "risk_level": "high or medium",
      "action": "remove or generalize",
      "replacement": "replacement string"
    }}
  ],
  "automated_deidentified_utterance": "de-identified utterance with identifiers removed or generalized, while preserving clinical meaning"
}}
"""


@dataclass
class LLMDeidResult:
    record: dict[str, Any]
    log: dict[str, Any]


def _clean_json_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = _clean_json_text(text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("response was not a JSON object")
    return parsed


def _find_span(text: str, span: str) -> tuple[int, int]:
    start = text.find(span)
    if start >= 0:
        return start, start + len(span)
    start = text.lower().find(span.lower())
    if start >= 0:
        return start, start + len(span)
    return -1, -1


def normalize_predictions(utterance: str, predictions: Any) -> tuple[list[dict[str, Any]], list[str]]:
    normalized: list[dict[str, Any]] = []
    problems: list[str] = []
    if not isinstance(predictions, list):
        return normalized, ["predicted_identifier_annotations_not_list"]
    allowed_types = set(IDENTIFIER_TYPES)
    occupied: set[int] = set()
    for idx, pred in enumerate(predictions):
        if not isinstance(pred, dict):
            problems.append(f"prediction_{idx}_not_object")
            continue
        span = str(pred.get("span", "")).strip()
        typ = str(pred.get("type", "")).strip().upper()
        if not span or typ not in allowed_types:
            problems.append(f"prediction_{idx}_missing_span_or_type")
            continue
        start, end = _find_span(utterance, span)
        if start < 0:
            problems.append(f"prediction_{idx}_span_not_found:{span[:80]}")
            continue
        positions = set(range(start, end))
        if positions & occupied:
            problems.append(f"prediction_{idx}_overlap_dropped:{span[:80]}")
            continue
        occupied |= positions
        normalized.append(
            {
                "span": utterance[start:end],
                "start": start,
                "end": end,
                "type": typ,
                "risk_level": str(pred.get("risk_level") or "medium").lower(),
                "action": str(pred.get("action") or "generalize").lower(),
                "replacement": str(pred.get("replacement") or REPLACEMENTS.get(typ, "[identifier]")),
            }
        )
    normalized.sort(key=lambda item: (item["start"], item["end"]))
    return normalized, problems


def call_siliconflow_chat(
    prompt: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    timeout_sec: int = 120,
    max_tokens: int = 2000,
) -> tuple[str, dict[str, Any]]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "top_p": 0.7,
        "max_tokens": max_tokens,
        "stream": False,
        "enable_thinking": False,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        SILICONFLOW_CHAT_COMPLETIONS_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": "Bearer " + api_key.strip(), "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"SiliconFlow HTTP {exc.code}: {detail}") from exc
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"SiliconFlow returned no choices: {json.dumps(payload)[:1000]}")
    message = choices[0].get("message", {})
    return str(message.get("content", "")), payload.get("usage", {})


def deidentify_with_qwen(
    row: dict[str, Any],
    api_key: str,
    model: str = DEFAULT_MODEL,
    timeout_sec: int = 120,
    max_tokens: int = 2000,
    max_attempts: int = 2,
) -> LLMDeidResult:
    started = time.time()
    utterance = row.get("utterance", "")
    prompt = build_deid_prompt(row)
    last_error = ""
    usage: dict[str, Any] = {}
    raw_content = ""
    parsed: dict[str, Any] = {}
    for attempt in range(1, max_attempts + 1):
        try:
            raw_content, usage = call_siliconflow_chat(prompt, api_key=api_key, model=model, timeout_sec=timeout_sec, max_tokens=max_tokens)
            parsed = parse_json_object(raw_content)
            predictions, normalization_problems = normalize_predictions(utterance, parsed.get("predicted_identifier_annotations", []))
            deidentified = str(parsed.get("automated_deidentified_utterance", "")).strip() or utterance
            record = {
                "id": row.get("id", ""),
                "subset": row.get("subset", ""),
                "source_model": row.get("source_model", ""),
                "condition": "qwen_llm_semantic_layer",
                "llm_model": model,
                "utterance": utterance,
                "predicted_identifier_annotations": predictions,
                "automated_deidentified_utterance": deidentified,
                "reference_deidentified_utterance": row.get("deidentified_utterance", ""),
                "gold_identifier_annotations": row.get("identifier_annotations", []),
                "clinical_concept_annotations": row.get("clinical_concept_annotations", []),
            }
            log = {
                "id": row.get("id", ""),
                "subset": row.get("subset", ""),
                "condition": "qwen_llm_semantic_layer",
                "model": model,
                "attempts": attempt,
                "status": "ok",
                "n_predicted_raw": len(parsed.get("predicted_identifier_annotations", []) or []),
                "n_predicted_valid": len(predictions),
                "normalization_problems": " ; ".join(normalization_problems[:8]),
                "error": "",
                "elapsed_sec": round(time.time() - started, 3),
                "prompt_tokens": usage.get("prompt_tokens", ""),
                "completion_tokens": usage.get("completion_tokens", ""),
                "total_tokens": usage.get("total_tokens", ""),
            }
            return LLMDeidResult(record, log)
        except Exception as exc:
            last_error = str(exc)[:1000]
            time.sleep(min(2 * attempt, 5))
    record = {
        "id": row.get("id", ""),
        "subset": row.get("subset", ""),
        "source_model": row.get("source_model", ""),
        "condition": "qwen_llm_semantic_layer",
        "llm_model": model,
        "utterance": utterance,
        "predicted_identifier_annotations": [],
        "automated_deidentified_utterance": utterance,
        "reference_deidentified_utterance": row.get("deidentified_utterance", ""),
        "gold_identifier_annotations": row.get("identifier_annotations", []),
        "clinical_concept_annotations": row.get("clinical_concept_annotations", []),
    }
    log = {
        "id": row.get("id", ""),
        "subset": row.get("subset", ""),
        "condition": "qwen_llm_semantic_layer",
        "model": model,
        "attempts": max_attempts,
        "status": "error",
        "n_predicted_raw": 0,
        "n_predicted_valid": 0,
        "normalization_problems": "",
        "error": last_error,
        "elapsed_sec": round(time.time() - started, 3),
        "prompt_tokens": "",
        "completion_tokens": "",
        "total_tokens": "",
    }
    return LLMDeidResult(record, log)
