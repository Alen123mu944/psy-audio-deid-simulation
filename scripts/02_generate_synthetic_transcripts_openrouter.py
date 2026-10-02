from __future__ import annotations

import _bootstrap  # noqa: F401

import getpass
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.io_utils import load_config, read_jsonl, write_csv, write_jsonl
from src.semantic_generation import (
    SUBSETS,
    generate_openrouter_batch,
    normalize_transcript_record,
    quality_report,
)

CHECKPOINT_FILE = "data/raw/synthetic_generated/transcripts_openrouter_checkpoint.jsonl"
LOG_FILE = "data/results/logs/transcript_generation_openrouter_log.csv"
QUALITY_FILE = "data/results/logs/transcript_generation_quality_report.csv"
REVIEWED_FILE = "data/raw/synthetic_generated/transcripts_reviewed.jsonl"

LOG_FIELDS = [
    "subset",
    "attempt",
    "requested",
    "raw_records_returned",
    "accepted",
    "rejected",
    "total_accepted",
    "model",
    "error_or_rejection_notes",
]

QUALITY_FIELDS = [
    "id",
    "subset",
    "source_model",
    "is_valid_json",
    "n_identifiers",
    "n_direct_identifiers",
    "n_contextual_identifiers",
    "n_clinical_concepts",
    "missing_identifier_spans",
    "missing_concept_spans",
    "flag_real_world_entity",
    "flag_extreme_content",
    "needs_review",
]


def _subset_targets(n: int) -> dict[str, int]:
    base = n // len(SUBSETS)
    remainder = n % len(SUBSETS)
    return {subset: base + (1 if idx < remainder else 0) for idx, subset in enumerate(SUBSETS)}


def _write_progress(rows: list[dict], logs: list[dict], output_file: str) -> None:
    write_jsonl(output_file, rows)
    write_jsonl(REVIEWED_FILE, rows)
    write_jsonl(CHECKPOINT_FILE, rows)
    write_csv(LOG_FILE, logs, LOG_FIELDS)
    write_csv(QUALITY_FILE, quality_report(rows), QUALITY_FIELDS)


def _load_checkpoint(model: str, n: int) -> list[dict]:
    rows = read_jsonl(CHECKPOINT_FILE)
    if not rows:
        return []
    if any(row.get("source_model") != model for row in rows):
        return []
    return rows[:n]


def _get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    return getpass.getpass("OpenRouter API key: ").strip()


def _request_one(subset: str, attempt: int, seed: int, model: str, api_key: str, timeout_sec: int, max_tokens: int):
    result = generate_openrouter_batch(
        batch_size=1,
        subset=subset,
        batch_index=attempt,
        seed=seed,
        model=model,
        api_key=api_key,
        timeout_sec=timeout_sec,
        max_tokens=max_tokens,
    )
    return subset, attempt, result


def main() -> None:
    cfg = load_config()
    semantic_cfg = cfg.get("semantic_generation", {})
    n = int(semantic_cfg.get("n_transcripts", 150))
    model = str(semantic_cfg.get("model", "openai/gpt-5.2"))
    seed = int(cfg.get("project", {}).get("random_seed", 42))
    timeout_sec = int(semantic_cfg.get("timeout_sec", 180))
    max_tokens = int(semantic_cfg.get("max_tokens", 1000))
    attempts_per_record = int(semantic_cfg.get("max_attempts_per_record", 4))
    parallel_requests = max(1, int(semantic_cfg.get("parallel_requests", 4)))
    output_file = str(semantic_cfg.get("output_file", "data/raw/synthetic_generated/transcripts_raw.jsonl"))
    api_key = _get_api_key()
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY was not provided.")

    targets = _subset_targets(n)
    rows = _load_checkpoint(model, n)
    logs = []
    row_index = len(rows) + 1

    if rows:
        print(f"Resuming from checkpoint with {len(rows)} accepted records.", flush=True)
    print(f"Using model={model}; parallel_requests={parallel_requests}; target={n}", flush=True)

    with ThreadPoolExecutor(max_workers=parallel_requests) as executor:
        for subset, target in targets.items():
            accepted_for_subset = sum(1 for row in rows if row.get("subset") == subset)
            attempts_submitted = 0
            max_attempts = max(target * attempts_per_record, target + parallel_requests)
            while accepted_for_subset < target and attempts_submitted < max_attempts:
                remaining = target - accepted_for_subset
                wave_size = min(parallel_requests, remaining, max_attempts - attempts_submitted)
                futures = []
                for _ in range(wave_size):
                    attempts_submitted += 1
                    attempt = attempts_submitted
                    task_seed = seed + len(rows) + attempt + len(subset)
                    futures.append(
                        executor.submit(
                            _request_one,
                            subset,
                            attempt,
                            task_seed,
                            model,
                            api_key,
                            timeout_sec,
                            max_tokens,
                        )
                    )

                for future in as_completed(futures):
                    subset_name, attempt, result = future.result()
                    accepted_this_attempt = 0
                    rejected_this_attempt = 0
                    rejection_notes: list[str] = []
                    if result.error:
                        rejection_notes.append(result.error[:240])
                    for raw in result.records[:1]:
                        if accepted_for_subset >= target:
                            break
                        row_id = f"SIM{row_index:03d}"
                        normalized, problems = normalize_transcript_record(raw, row_id=row_id, subset=subset_name, source_model=model)
                        if normalized is None:
                            rejected_this_attempt += 1
                            rejection_notes.append("|".join(problems)[:240])
                            continue
                        rows.append(normalized)
                        row_index += 1
                        accepted_for_subset += 1
                        accepted_this_attempt += 1
                    logs.append(
                        {
                            "subset": subset_name,
                            "attempt": attempt,
                            "requested": 1,
                            "raw_records_returned": len(result.records),
                            "accepted": accepted_this_attempt,
                            "rejected": rejected_this_attempt,
                            "total_accepted": len(rows),
                            "model": model,
                            "error_or_rejection_notes": " ; ".join(rejection_notes[:5]),
                        }
                    )
                    _write_progress(rows, logs, output_file)
                    status = "accepted" if accepted_this_attempt else "rejected"
                    print(
                        f"{subset_name} attempt {attempt}: {status}; subset {accepted_for_subset}/{target}; total {len(rows)}/{n}",
                        flush=True,
                    )

    if len(rows) != n:
        raise SystemExit(f"Generated {len(rows)} valid transcripts, expected {n}. See {LOG_FILE}; checkpoint saved at {CHECKPOINT_FILE}")

    _write_progress(rows, logs, output_file)
    subset_counts = Counter(row["subset"] for row in rows)
    print(f"Wrote {len(rows)} OpenRouter-generated synthetic transcripts using {model}.", flush=True)
    print(f"Subset counts: {dict(subset_counts)}", flush=True)


if __name__ == "__main__":
    main()
