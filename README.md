# Psychiatric Audio De-identification Simulation

This project implements a proof-of-concept simulation for a three-layer de-identification framework for psychiatric audio recordings.

The simulation is not clinical validation. It is designed to show how privacy protection and clinical utility can be evaluated without using real psychiatric recordings or real patient data.

## Streams

1. Voice layer: public or synthetic audio, transparent pitch perturbation, speaker privacy metrics, and acoustic utility metrics.
2. Semantic layer: fictional psychiatric interview-like transcripts, identifier detection/de-identification, and clinical concept preservation.
3. Metadata layer: fictional recording metadata, metadata sanitization, linkage-risk scoring, and longitudinal utility preservation.

## Quick Start

```bash
python scripts/01_sample_audio_clips.py
python scripts/04_voice_deidentify.py
python scripts/05_extract_speaker_embeddings.py
python scripts/06_evaluate_speaker_privacy.py
python scripts/07_extract_acoustic_features.py
python scripts/08_evaluate_acoustic_utility.py
python scripts/02_generate_synthetic_transcripts_openai.py
python scripts/03_generate_synthetic_metadata_openai.py
python scripts/09_semantic_deidentify_presidio.py
python scripts/10_evaluate_semantic_privacy_utility.py
python scripts/11_metadata_sanitize.py
python scripts/12_evaluate_metadata_privacy_utility.py
python scripts/13_generate_summary_tables.py
python scripts/14_generate_figures.py
```

The project includes deterministic template-based generation, so the semantic and metadata streams can run without an OpenAI API key. In the current proof-of-concept run, the voice stream uses real LibriSpeech test-clean clips, SpeechBrain ECAPA-TDNN speaker embeddings, and librosa acoustic features.

## Expected Manuscript Outputs

```text
outputs_for_manuscript/
├── table_voice_layer.csv
├── table_semantic_layer.csv
├── table_metadata_layer.csv
├── table_risk_utility_matrix.csv
├── figure_simulation_workflow.png
├── figure_speaker_similarity.png
├── figure_risk_reduction.png
└── manuscript_section4_draft.md
```

## Data Ethics

No real patient data should be used by the synthetic transcript or metadata generators. If OpenAI API generation is enabled, the prompt instructs the model to produce only fictional names, institutions, addresses, and events.
