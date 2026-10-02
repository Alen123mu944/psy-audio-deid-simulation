# Psychiatric audio de-identification simulation

Code and frozen synthetic data supporting the manuscript.
The semantic dataset contains **150 records and 744 gold identifiers**:
clean 250, contextual 251, ASR-like 243 (50 records per subset).

Verify results offline with Python 3.11+; no API key or dependencies required:

```bash
python3 scripts/16_verify_submission.py
```

Recompute the tables offline:

```bash
python3 scripts/17_recompute_frozen.py
```

Data and predictions: `data/`. Tables and figures: `outputs_for_manuscript/`.
S2/S3 exports: `manuscript_tables/`. Review records: `review/`.
Qwen F1: **0.975674**.

See [REPRODUCIBILITY_STATEMENT.md](REPRODUCIBILITY_STATEMENT.md) for provenance.
Live API runs produce a new batch; use a separate copy. Audio reproduction requires
LibriSpeech and the dependencies in `requirements.txt`.
