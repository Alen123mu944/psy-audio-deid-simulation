# Reproducibility

This release contains the September 2026 reconstructed synthetic batch:
150 transcripts and 744 final gold annotations. GPT-5.2 generated the inputs;
Qwen3.6-27B predictions used OpenRouter with SiliconFlow routing.
Gold includes the author-confirmed additions PERSON Dana (SIM059) and
SESSION_PATTERN every other Tuesday (SIM082). Saved predictions are unchanged.
This batch supports the revised manuscript; the original manuscript-era
record-level responses remain unavailable.

Offline scripts verify frozen hashes and recompute semantic and metadata results.
Clinical preservation uses lexical concept matching against the locked annotations.
Voice results are retained as aggregates; raw-audio reproduction requires public
LibriSpeech audio and model downloads. Live API generation creates a new batch.
