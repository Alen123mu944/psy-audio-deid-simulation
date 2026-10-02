# Current manuscript table exports

S2.csv: three methods evaluated against the same locked 744 gold annotations.
S3.csv: Qwen subsets keyed as asr_noisy, clean_direct, contextual_paraphrased.
These are full-precision exports; the Word tables display rounded values.

S3 correction confirmed on 2026-10-03:

| Word row label | Incorrect gold/prediction counts | Correct gold/prediction counts |
|---|---|---|
| ASR-like noisy transcripts | 250 / 254 (clean values) | 243 / 255 |
| Clean direct identifiers | 251 / 268 (contextual values) | 250 / 254 |
| Contextual paraphrased identifiers | 243 / 255 (ASR values) | 251 / 268 |

The entire numeric row, not just the counts, must follow its subset key.
No record was reassigned to a different subset.
