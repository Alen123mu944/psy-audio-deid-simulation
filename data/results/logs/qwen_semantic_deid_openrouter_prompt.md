# Qwen Semantic De-identification Prompt Record

Provider: OpenRouter chat completions, pinned to SiliconFlow
Endpoint: https://openrouter.ai/api/v1/chat/completions
Model: `qwen/qwen3.6-27b`
Condition: `qwen_llm_semantic_layer`
Parallel requests: 6
Timeout seconds: 120
Max tokens: 2000
Max attempts per record: 2
Temperature: 0
Top-p: 0.7
Response format: JSON object

## System Prompt

```text
You are a clinical text de-identification engine for synthetic psychiatric interview transcripts.
Return valid compact JSON only. Do not include markdown, explanations, or chain-of-thought.
Your goal is to protect privacy while preserving clinically meaningful psychiatric content.
```

## User Prompt Template Example

```text
De-identify the following synthetic psychiatric transcript.

Input record:
- id: SIM001
- utterance: I met with Dr. Tessa Rowland at Cedar Lane Mental Health Center on 06/14/2026, and she said to call (555) 014-7721 about case CLMHC-56-2048. Lately I've had panic attacks and insomnia, and I keep thinking I'd be better off dead but I haven't made any plans.

Allowed identifier types:
PERSON, DATE, LOCATION, ORGANIZATION, FAMILY_NAME, SCHOOL_WORKPLACE, ADDRESS, CONTACT, RARE_EVENT, SESSION_PATTERN, MEDICAL_RECORD, CLINICIAN_NAME, HEALTHCARE_ORGANIZATION

Replacement policy:
- PERSON -> [PERSON]
- CLINICIAN_NAME -> [CLINICIAN]
- DATE -> [relative time]
- LOCATION -> [location]
- ORGANIZATION -> [organization]
- HEALTHCARE_ORGANIZATION -> [mental health clinic]
- SCHOOL_WORKPLACE -> [school/workplace]
- ADDRESS -> [address removed]
- CONTACT -> [contact removed]
- FAMILY_NAME -> my family member
- RARE_EVENT -> [major stressful event]
- SESSION_PATTERN -> [regular therapy schedule]
- MEDICAL_RECORD -> [record identifier removed]

Identifier guidance:
- Direct identifiers include patient names, clinician names, contact details, addresses, record numbers, exact dates, and exact times.
- Contextual identifiers include healthcare organizations, school/workplace names, small locations, family names/relationships, rare local events, and recurring therapy schedules.
- Preserve clinical concepts such as symptoms, sleep problems, medication, treatment, functional impairment, substance use, trauma, social withdrawal, and non-graphic risk ideation.
- Do not invent new facts.
- Every predicted span must be copied exactly from the input utterance.
- If an identifier appears in noisy ASR style, still annotate the exact noisy substring.

Return JSON using this schema:
{
  "id": "SIM001",
  "predicted_identifier_annotations": [
    {
      "span": "exact substring from the input utterance",
      "type": "one allowed identifier type",
      "risk_level": "high or medium",
      "action": "remove or generalize",
      "replacement": "replacement string"
    }
  ],
  "automated_deidentified_utterance": "de-identified utterance with identifiers removed or generalized, while preserving clinical meaning"
}

```
