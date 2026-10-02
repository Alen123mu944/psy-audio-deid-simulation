from __future__ import annotations

import _bootstrap  # noqa: F401

import getpass
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.io_utils import ensure_dir, load_config, read_jsonl, write_csv, write_jsonl
from src.semantic_llm_deid import DEFAULT_MODEL, SYSTEM_PROMPT, build_deid_prompt, deidentify_with_qwen

OUTPUT_FILE = "data/processed/transcripts_deidentified/transcripts_qwen_llm_semantic_layer.jsonl"
CHECKPOINT_FILE = "data/processed/transcripts_deidentified/transcripts_qwen_llm_checkpoint.jsonl"
LOG_FILE = "data/results/logs/qwen_semantic_deid_log.csv"

LOG_FIELDS = [
    "id",
    "subset",
    "condition",
    "model",
    "attempts",
    "status",
    "n_predicted_raw",
    "n_predicted_valid",
    "normalization_problems",
    "error",
    "elapsed_sec",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
]


def _get_api_key() -> str:
    key = os.environ.get("SILICONFLOW_API_KEY", "").strip()
    if key:
        return key
    return getpass.getpass("SiliconFlow API key: ").strip()


def _load_checkpoint(model: str) -> list[dict]:
    rows = read_jsonl(CHECKPOINT_FILE)
    if not rows:
        return []
    return [row for row in rows if row.get("llm_model") == model]


def _write_prompt_record(model: str, parallel_requests: int, timeout_sec: int, max_tokens: int, max_attempts: int, sample_row: dict) -> None:
    path = ensure_dir("data/results/logs") / "qwen_semantic_deid_prompt.md"
    text = f"""# Qwen Semantic De-identification Prompt Record

Provider: SiliconFlow chat completions
Endpoint: https://api.siliconflow.cn/v1/chat/completions
Model: `{model}`
Condition: `qwen_llm_semantic_layer`
Parallel requests: {parallel_requests}
Timeout seconds: {timeout_sec}
Max tokens: {max_tokens}
Max attempts per record: {max_attempts}
Temperature: 0
Top-p: 0.7
Enable thinking: false
Response format: JSON object

## System Prompt

```text
{SYSTEM_PROMPT}
```

## User Prompt Template Example

```text
{build_deid_prompt(sample_row)}
```
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    cfg = load_config()
    qwen_cfg = cfg.get("qwen_semantic_deid", {})
    model = str(qwen_cfg.get("model", DEFAULT_MODEL))
    parallel_requests = max(1, int(qwen_cfg.get("parallel_requests", 2)))
    timeout_sec = int(qwen_cfg.get("timeout_sec", 120))
    max_tokens = int(qwen_cfg.get("max_tokens", 2000))
    max_attempts = int(qwen_cfg.get("max_attempts", 2))
    limit = qwen_cfg.get("limit")
    limit = int(limit) if limit is not None else None

    api_key = _get_api_key()
    if not api_key:
        raise SystemExit("SILICONFLOW_API_KEY was not provided.")

    rows = read_jsonl("data/raw/synthetic_generated/transcripts_reviewed.jsonl") or read_jsonl("data/raw/synthetic_generated/transcripts_raw.jsonl")
    if not rows:
        raise SystemExit("No transcript JSONL found. Run transcript generation first.")
    if limit is not None:
        rows = rows[:limit]

    _write_prompt_record(model, parallel_requests, timeout_sec, max_tokens, max_attempts, rows[0])

    completed = _load_checkpoint(model)
    completed_by_id = {row["id"]: row for row in completed}
    logs = []
    pending = [row for row in rows if row.get("id") not in completed_by_id]
    print(f"Using model={model}; target={len(rows)}; completed={len(completed_by_id)}; pending={len(pending)}; parallel={parallel_requests}", flush=True)

    with ThreadPoolExecutor(max_workers=parallel_requests) as executor:
        futures = [executor.submit(deidentify_with_qwen, row, api_key, model, timeout_sec, max_tokens, max_attempts) for row in pending]
        for idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            completed_by_id[result.record["id"]] = result.record
            logs.append(result.log)
            ordered = [completed_by_id[row["id"]] for row in rows if row.get("id") in completed_by_id]
            write_jsonl(CHECKPOINT_FILE, ordered)
            write_jsonl(OUTPUT_FILE, ordered)
            write_csv(LOG_FILE, logs, LOG_FIELDS)
            print(
                f"{result.record['id']} {result.log['status']}: valid_spans={result.log['n_predicted_valid']}; completed={len(ordered)}/{len(rows)}",
                flush=True,
            )

    ordered = [completed_by_id[row["id"]] for row in rows]
    write_jsonl(CHECKPOINT_FILE, ordered)
    write_jsonl(OUTPUT_FILE, ordered)
    write_csv(LOG_FILE, logs, LOG_FIELDS)
    print(f"Wrote {len(ordered)} Qwen semantic de-identification records to {OUTPUT_FILE}.", flush=True)


if __name__ == "__main__":
    main()
