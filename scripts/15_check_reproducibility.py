from __future__ import annotations

import csv
import re
from pathlib import Path

import _bootstrap  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_MANUSCRIPT_OUTPUTS = [
    "outputs_for_manuscript/table_voice_layer.csv",
    "outputs_for_manuscript/table_voice_method_comparison.csv",
    "outputs_for_manuscript/table_semantic_comparison_with_qwen.csv",
    "outputs_for_manuscript/table_semantic_qwen_layer.csv",
    "outputs_for_manuscript/table_semantic_qwen_by_subset.csv",
    "outputs_for_manuscript/table_metadata_layer.csv",
    "outputs_for_manuscript/table_metadata_by_subset.csv",
    "outputs_for_manuscript/table_embedded_metadata_audit.csv",
    "outputs_for_manuscript/table_risk_utility_matrix.csv",
    "outputs_for_manuscript/figure_simulation_workflow.png",
    "outputs_for_manuscript/figure_speaker_similarity.png",
    "outputs_for_manuscript/figure_risk_reduction.png",
    "outputs_for_manuscript/manuscript_section4_draft.md",
]

CORE_INTERMEDIATES = [
    "config.yaml",
    "requirements.txt",
    "data/processed/audio_original/audio_manifest.csv",
    "data/processed/mappings/voice_deid_mapping.csv",
    "data/results/tables/table_voice_privacy.csv",
    "data/results/tables/table_voice_utility.csv",
    "data/raw/synthetic_generated/transcripts_reviewed.jsonl",
    "data/processed/transcripts_deidentified/transcripts_qwen_llm_semantic_layer.jsonl",
    "data/raw/synthetic_generated/metadata_reviewed.jsonl",
    "data/processed/metadata_sanitized/metadata_sanitized.jsonl",
]

SECRET_PATTERNS = [
    re.compile(r"sk-or-v1-[A-Za-z0-9_-]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
]

SKIP_DIRS = {".git", ".venv", "data", "pretrained_models", "__pycache__"}
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".txt", ".csv", ".json", ".jsonl"}


def rel(path: str | Path) -> Path:
    return ROOT / path


def csv_rows(path: str | Path) -> list[dict[str, str]]:
    p = rel(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def metric(rows: list[dict[str, str]], name: str, default: str = "missing") -> str:
    for row in rows:
        if row.get("metric") == name:
            return row.get("value", default)
    if rows and name in rows[0]:
        return rows[0].get(name, default)
    return default


def check_files(paths: list[str], label: str, required: bool) -> bool:
    ok = True
    print(f"\n{label}")
    for path in paths:
        exists = rel(path).exists()
        status = "OK" if exists else ("MISSING" if required else "optional-missing")
        print(f"[{status}] {path}")
        if required and not exists:
            ok = False
    return ok


def check_tables() -> bool:
    ok = True
    voice = csv_rows("outputs_for_manuscript/table_voice_layer.csv")
    semantic = csv_rows("outputs_for_manuscript/table_semantic_qwen_layer.csv")
    metadata = csv_rows("outputs_for_manuscript/table_metadata_layer.csv")
    matrix = csv_rows("outputs_for_manuscript/table_risk_utility_matrix.csv")

    print("\nKey result values")
    print(f"selected_voice_condition: {metric(voice, 'selected_voice_condition')}")
    print(f"qwen_residual_identifier_rate: {metric(semantic, 'residual_identifier_rate')}")
    print(f"metadata_linkage_risk_score_after: {metric(metadata, 'linkage_risk_score_after')}")
    full = next((row for row in matrix if row.get("condition") == "Full_framework"), None)
    if full:
        print(f"full_framework_speaker_id_accuracy: {full.get('speaker_id_accuracy')}")
        print(f"full_framework_semantic_residual: {full.get('semantic_residual_identifier_rate')}")
        print(f"full_framework_metadata_risk: {full.get('metadata_linkage_risk_score')}")
    else:
        print("[MISSING] Full_framework row in table_risk_utility_matrix.csv")
        ok = False

    if not voice or not semantic or not metadata or not matrix:
        ok = False
    return ok


def check_no_secrets() -> bool:
    print("\nSecret scan")
    hits: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                hits.append(str(path.relative_to(ROOT)))
    if hits:
        for hit in hits:
            print(f"[POSSIBLE SECRET] {hit}")
        return False
    print("[OK] no obvious API keys found in tracked text-like project files")
    return True


def main() -> None:
    ok = True
    ok &= check_files(REQUIRED_MANUSCRIPT_OUTPUTS, "Required manuscript outputs", required=True)
    ok &= check_files(CORE_INTERMEDIATES, "Core intermediate files", required=False)
    ok &= check_tables()
    ok &= check_no_secrets()
    if not ok:
        raise SystemExit("Reproducibility check found missing or inconsistent items.")
    print("\nReproducibility check passed.")


if __name__ == "__main__":
    main()
