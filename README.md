# Psychiatric audio de-identification — locked gold744 submission

Current evaluation version: **gold744-author-corrected-2026-10-03**.
The active files contain 150 synthetic records and 744 gold identifiers:
clean_direct 250, contextual_paraphrased 251, asr_noisy 243 (50 records each).
No patient data or API credentials are bundled.

## Verify offline

Python 3.11 or later; no third-party packages, API key, or network required:

```bash
python3 -B scripts/16_verify_submission.py
```

The verifier checks frozen input/code hashes, record IDs and annotation offsets,
the two recorded author corrections, all three sets of predictions, semantic
summary/by-type/by-subset tables in both result folders, keyed S2/S3 exports,
metadata results, and semantic values in the integrated risk-utility matrix.
It independently recomputes metrics from JSONL records using the unchanged
evaluation logic. A PASS does not certify clinical validity or exhaustive
human review, and does not rerun the raw-audio experiments.
GitHub Actions runs this verification and offline table recomputation on pushes
and pull requests. Numerical table comparisons allow a tolerance of 1e-12.

## Recompute tables offline

```bash
python3 -B scripts/17_recompute_frozen.py
```

This regenerates all active semantic tables, S2/S3 exports, metadata tables,
integrated summaries and the Section 4 draft, then invokes verification.
It preserves frozen inputs and predictions and refuses changed source files.
To regenerate figures as well, install matplotlib (tested dependencies are
listed in requirements-verified.txt), then run:

```bash
python3 -B scripts/14_generate_figures.py
```

## Locked semantic results

| Method | Gold | Predictions | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct-only baseline | 744 | 471 | 394 | 77 | 350 | 0.836518 | 0.529570 | 0.648560 |
| Expanded rule-based layer | 744 | 788 | 673 | 115 | 71 | 0.854061 | 0.904570 | 0.878590 |
| Qwen LLM layer | 744 | 777 | 742 | 35 | 2 | 0.954955 | 0.997312 | 0.975674 |

Qwen residual identifier rate is 0.002688172043010753; high-risk residual
rate is 0; clinical concept preservation is 0.9827777777777778.
Rounded to three decimal places these are 0.003, 0.000 and 0.983.

## Included evidence

- data/raw/synthetic_generated/: final transcript and metadata inputs.
- data/processed/transcripts_deidentified/: all three methods, each using
  the same locked gold annotations; saved predictions are unchanged.
- data/results/tables/ and outputs_for_manuscript/: current results.
- manuscript_tables/S2.csv and S3.csv: keyed, full-precision manuscript exports.
- review/: locked review export, author-confirmed correction log and rescore notes.
- FROZEN_MANIFEST.json: version, hashes, counts, correction and prediction signatures.
- provenance/transcripts_pre_correction_742.jsonl: historical input solely for
  verifying the two additions; not an active evaluation input.
- src/, scripts/, config.yaml, data/results/logs/: implementation and run records.

The old historical batch also had 744 labels, but it is NOT this corrected batch.
See REPRODUCIBILITY_STATEMENT.md for provenance and reproducibility boundaries.
The author-review workbook and historical aggregate duplicates remain in the
local submission package; this repository includes the CSV correction and review
records needed for offline verification. Full manuscript Word files are excluded.

## New API runs and audio reproduction

The full pipeline and generation scripts can call paid APIs and create a NEW
stochastic batch. Run them only in a separate copy. Do not overwrite this frozen
release to compare with the revised manuscript. Generation scripts obtain keys
from environment variables or interactive input; never place keys in the archive.

The frozen GPT-5.2 inputs and Qwen predictions were produced in September 2026
via OpenRouter; Qwen routing was pinned to SiliconFlow. A live rerun is not an
exact replay of archived responses. The frozen command above needs no API.

Voice results and embedded audio metadata audit are retained from the supplied
package. LibriSpeech audio and model weights are not bundled; reproducing those
stages from raw signals requires the public corpus, model downloads and dependencies.
This synchronization rescored semantic/metadata records, not raw audio.
