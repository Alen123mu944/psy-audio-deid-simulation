# Reproducibility statement — locked author-corrected 744-label version

## Provenance

This release uses the September 2026 reconstructed synthetic batch of 150
psychiatric interview-like records. GPT-5.2 inputs were generated via OpenRouter;
Qwen3.6-27B predictions were obtained via OpenRouter with routing pinned to
SiliconFlow. The saved Qwen log contains 150 successful records after retrying
a transient failure. No model was called during this October 2026 synchronization.

The pre-correction batch had 742 gold annotations. The existing author-confirmed
correction record adds PERSON "Dana" to SIM059 and SESSION_PATTERN
"every other Tuesday" to SIM082. These exact changes give 744 annotations:
250 clean, 251 contextual and 243 ASR-like. This release uses that already locked
version without further annotation changes. Dates remain included in scoring.

For every method, the input text, subset assignments, clinical annotations,
reference de-identified text, predicted annotations and automated output text
are unchanged. Only evaluation gold was synchronized. All methods were rescored
using the same unchanged matching and utility code. Hashes and the historical
pre-correction input make the two changes independently inspectable.

## Current results and manuscript alignment

| Measure | Locked current result |
|---|---:|
| Records | 150 |
| Gold identifiers | 744 |
| Qwen predicted identifiers | 777 |
| True positives / false positives / false negatives | 742 / 35 / 2 |
| Qwen precision | 0.954954954954955 |
| Qwen recall | 0.9973118279569892 |
| Qwen F1 | 0.9756738987508218 |
| Residual identifier rate | 0.002688172043010753 |
| High-risk residual rate | 0 |
| Clinical concept preservation | 0.9827777777777778 |

The current revised manuscript's Table 5 semantic values and Supplementary
Table S2 match this version after rounding. The supplied Supplementary Table S3
had a cyclic misalignment between three row labels and their numeric values.
The corrected S3 is exported by subset key in manuscript_tables/S3.csv.
The accompanying corrected Supplementary Word file fixes only those S3 values.

The original manuscript-era batch also reported 744 annotations but different
predictions (757) and F1 (0.9766822118587609). Its record-level inputs and
responses remain unavailable. Those historical aggregates are preserved in the
local submission package. Equality of label totals does not establish batch identity.
This release supports the revised manuscript using the corrected reconstructed
batch, not exact recovery of missing historical model responses.

## Review scope and evaluation limitations

The existing author-review workbook is retained in the local submission package;
review/gold_corrections.csv documents the two confirmed corrections, and
review/locked_review.csv exposes the final gold annotations and reference texts.
This packaging operation did not perform or certify a complete independent
clinical or annotation review of all 150 records. Author statements about review
should describe the work actually performed.

The locked annotations and reference texts are retained as supplied, including
known review candidates. In particular, the reference text for SIM082 still
contains "every other Tuesday"; this is not the Qwen automated output used for
scoring. Potential further gold omissions previously noted in SIM054 and SIM093
were not silently added to this locked release. Residual-identifier rates are
defined against the locked gold annotation set, not a guarantee that every
possible identifier has been detected. Clinical preservation uses the existing
lexical concept-matching measure, not independent clinical adjudication.

## Offline verification

python3 -B scripts/16_verify_submission.py checks frozen SHA-256 hashes,
150 unique records, 50/50/50 subset sizes, exact gold counts, annotation offsets,
two review additions, unchanged prediction signatures, TP/FP/FN for all three
methods, all active semantic tables, S2/S3 keyed exports, metadata results and
semantic entries in the integrated matrix. scripts/17_recompute_frozen.py
regenerates the tables offline and then runs the same checks.

Metadata results are recomputed from the included 150 raw and sanitized
synthetic records. Voice results and the embedded-audio audit are retained;
audio and model weights are excluded for size. They require separate public
downloads for raw-signal reproduction. The offline PASS does not certify a new
audio run, exhaustive author review or an exact replay of live model generation.
