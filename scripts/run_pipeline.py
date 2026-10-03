from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)

STAGES: dict[str, list[str]] = {
    "voice": [
        "scripts/00_download_or_prepare_librispeech.py",
        "scripts/01_sample_audio_clips.py",
        "scripts/04_voice_deidentify.py",
        "scripts/05_extract_speaker_embeddings.py",
        "scripts/06_evaluate_speaker_privacy.py",
        "scripts/07_extract_acoustic_features.py",
        "scripts/08_evaluate_acoustic_utility.py",
    ],
    "semantic-baseline": [
        "scripts/02_generate_synthetic_transcripts_openrouter.py",
        "scripts/09_semantic_deidentify_rules.py",
        "scripts/10_evaluate_semantic_privacy_utility.py",
    ],
    "semantic-qwen": [
        "scripts/09c_semantic_deidentify_openrouter_qwen.py",
        "scripts/10b_evaluate_qwen_semantic.py",
    ],
    "metadata": [
        "scripts/03_generate_synthetic_metadata.py",
        "scripts/11_metadata_sanitize.py",
        "scripts/12_evaluate_metadata_privacy_utility.py",
        "scripts/12a_audit_embedded_audio_metadata.py",
    ],
    "manuscript": [
        "scripts/13_generate_summary_tables.py",
    ],
}
STAGES["semantic"] = STAGES["semantic-baseline"] + STAGES["semantic-qwen"]
STAGES["all"] = STAGES["voice"] + STAGES["semantic"] + STAGES["metadata"] + STAGES["manuscript"]

REQUIRED_ENV = {
    "scripts/02_generate_synthetic_transcripts_openrouter.py": "OPENROUTER_API_KEY",
    "scripts/09c_semantic_deidentify_openrouter_qwen.py": "OPENROUTER_API_KEY",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible simulation pipeline stages.")
    parser.add_argument("--stage", choices=sorted(STAGES), default="manuscript", help="Pipeline stage to run.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue after a failed script and report all failures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scripts = STAGES[args.stage]
    failures: list[str] = []

    print(f"Stage: {args.stage}")
    for script in scripts:
        env_name = REQUIRED_ENV.get(script)
        if env_name and not os.environ.get(env_name):
            print(f"Warning: {script} needs {env_name}. The script will prompt interactively if run.")
        cmd = [str(PYTHON), script]
        print("$ " + " ".join(cmd))
        if args.dry_run:
            continue
        try:
            subprocess.run(cmd, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            failures.append(f"{script} exited with {exc.returncode}")
            if not args.continue_on_error:
                raise SystemExit("\n".join(failures)) from exc

    if failures:
        raise SystemExit("Pipeline completed with failures:\n" + "\n".join(failures))
    if args.dry_run:
        print("Dry run complete.")
    else:
        print("Pipeline stage completed successfully.")


if __name__ == "__main__":
    main()
