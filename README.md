# Psychiatric audio de-identification simulation

Paper: *A Multi-layer De-identification Framework for Psychiatric Audio Recordings: Balancing Clinical Utility and Privacy Protection*.

Code and synthetic data supporting the manuscript.

Offline verification and table recomputation (Python 3.11+; no API key or third-party packages):

```bash
python3 scripts/16_verify_submission.py
python3 scripts/17_recompute_frozen.py
```

Data and predictions: `data/`. Tables and figures: `outputs_for_manuscript/`.
Full experiments require `requirements.txt`, public LibriSpeech audio and `OPENROUTER_API_KEY`.
Run new experiments in a separate copy; live model outputs may vary.
