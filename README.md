# Psychiatric audio de-identification simulation

Paper: *A Multi-layer De-identification Framework for Psychiatric Audio Recordings: Balancing Clinical Utility and Privacy Protection*.

Code and synthetic data supporting the manuscript: **150 transcripts, 744 gold labels**.

Verify and recompute results offline with Python 3.11+:

```bash
python3 scripts/16_verify_submission.py
python3 scripts/17_recompute_frozen.py
```

Inputs and predictions: `data/`. Results: `outputs_for_manuscript/`.
Offline checks require no API key or third-party packages. Full experiments use
`requirements.txt`, LibriSpeech and `OPENROUTER_API_KEY`; run new batches in a separate copy.

Provenance: [REPRODUCIBILITY_STATEMENT.md](REPRODUCIBILITY_STATEMENT.md).
