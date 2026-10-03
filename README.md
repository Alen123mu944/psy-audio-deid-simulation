# Psychiatric audio de-identification simulation

Paper: *A Multi-layer De-identification Framework for Psychiatric Audio Recordings: Balancing Clinical Utility and Privacy Protection*.

Code and synthetic data supporting the manuscript.

Offline verification and semantic/metadata table recomputation (Python 3.11+; no API key or third-party packages):

```bash
python3 scripts/16_verify_submission.py
python3 scripts/17_recompute_frozen.py
```

Data and predictions: `data/`. Tables and figures: `outputs_for_manuscript/`.
Speaker identification and EER use C1–C3 for enrollment (90 clips) and the same C4/C5 held-out set (60 clips) for every condition.
Paired cosine similarity and acoustic utility describe all 150 clips.
Full experiments require `requirements.txt`, public LibriSpeech audio and `OPENROUTER_API_KEY`.
Run new experiments in a separate copy; live model outputs may vary.
