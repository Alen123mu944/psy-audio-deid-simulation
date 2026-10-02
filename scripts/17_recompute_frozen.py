"""Recompute frozen semantic/metadata tables offline; preserve raw inputs and predictions."""
import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verify_frozen", ROOT/"scripts/16_verify_submission.py")
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    import hashlib
    import json
    manifest = json.loads((ROOT/"FROZEN_MANIFEST.json").read_text())
    for path, expected in manifest["sha256"].items():
        verify.require(hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == expected, f"frozen file changed: {path}")
    tables = verify.semantic_tables()
    for name, rows in tables.items():
        for folder in ("data/results/tables", "outputs_for_manuscript"):
            write_csv(ROOT/folder/name, rows)
    write_csv(ROOT/"manuscript_tables/S2.csv", tables["table_semantic_comparison_with_qwen.csv"])
    write_csv(ROOT/"manuscript_tables/S3.csv", tables["table_semantic_qwen_by_subset.csv"])
    for script in ("12_evaluate_metadata_privacy_utility.py", "13_generate_summary_tables.py"):
        subprocess.run([sys.executable, "-B", str(ROOT/"scripts"/script)], check=True, cwd=ROOT)
    verify.main()

if __name__ == "__main__":
    main()
