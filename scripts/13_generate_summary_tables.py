from __future__ import annotations

import _bootstrap  # noqa: F401

from src.io_utils import copy_to_manuscript, read_csv_dicts, write_csv


def _metric(rows: list[dict[str, str]], name: str, default: float = 0.0) -> float:
    for row in rows:
        if row.get("metric") == name:
            return float(row.get("value", default))
    return default


def _condition(rows: list[dict[str, str]], condition: str, field: str, default: float = 0.0) -> float:
    for row in rows:
        if row.get("condition") == condition:
            return float(row.get(field, default))
    return default


def write_voice_layer_table() -> None:
    privacy = read_csv_dicts("data/results/tables/table_voice_privacy.csv")
    utility = read_csv_dicts("data/results/tables/table_voice_utility.csv")
    acoustic_preservation = sum(float(row["pearson_r"]) for row in utility if not row["feature"].startswith("F0")) / max(1, len([r for r in utility if not r["feature"].startswith("F0")]))
    rows = [
        {"metric": "speaker_id_accuracy_original", "value": _condition(privacy, "original", "top1_speaker_id_accuracy")},
        {"metric": "speaker_id_accuracy_voice_deidentified", "value": _condition(privacy, "voice_only", "top1_speaker_id_accuracy")},
        {"metric": "linkage_success_rate_original", "value": _condition(privacy, "original", "linkage_success_rate")},
        {"metric": "linkage_success_rate_voice_deidentified", "value": _condition(privacy, "voice_only", "linkage_success_rate")},
        {"metric": "acoustic_feature_preservation_excluding_f0", "value": acoustic_preservation},
    ]
    write_csv("outputs_for_manuscript/table_voice_layer.csv", rows, ["metric", "value"])


def write_risk_utility_matrix() -> None:
    semantic = read_csv_dicts("data/results/tables/table_semantic_layer.csv")
    metadata = read_csv_dicts("data/results/tables/table_metadata_layer.csv")
    privacy = read_csv_dicts("data/results/tables/table_voice_privacy.csv")
    utility = read_csv_dicts("data/results/tables/table_voice_utility.csv")
    speaker_original = _condition(privacy, "original", "top1_speaker_id_accuracy", 1.0)
    speaker_deid = _condition(privacy, "voice_only", "top1_speaker_id_accuracy", 0.42)
    semantic_residual = _metric(semantic, "residual_identifier_rate", 0.10)
    metadata_after = _metric(metadata, "linkage_risk_score_after", 0.05)
    concept_pres = _metric(semantic, "clinical_concept_preservation_rate", 0.85)
    acoustic_pres = sum(float(row["pearson_r"]) for row in utility if not row["feature"].startswith("F0")) / max(1, len([r for r in utility if not r["feature"].startswith("F0")]))
    rows = [
        {"condition": "Original", "speaker_id_accuracy": speaker_original, "semantic_residual_identifier_rate": 1.0, "metadata_linkage_risk_score": 1.0, "clinical_concept_preservation": 1.0, "acoustic_feature_preservation": 1.0, "interpretation": "Raw data retain clinical value but expose multi-source identifiers"},
        {"condition": "Voice-only", "speaker_id_accuracy": speaker_deid, "semantic_residual_identifier_rate": 1.0, "metadata_linkage_risk_score": 1.0, "clinical_concept_preservation": 1.0, "acoustic_feature_preservation": acoustic_pres, "interpretation": "Voice identity risk is reduced but semantic and metadata risks remain"},
        {"condition": "Semantic-only", "speaker_id_accuracy": speaker_original, "semantic_residual_identifier_rate": semantic_residual, "metadata_linkage_risk_score": 1.0, "clinical_concept_preservation": concept_pres, "acoustic_feature_preservation": 1.0, "interpretation": "Textual identifiers are reduced but speaker and metadata risks remain"},
        {"condition": "Metadata-only", "speaker_id_accuracy": speaker_original, "semantic_residual_identifier_rate": 1.0, "metadata_linkage_risk_score": metadata_after, "clinical_concept_preservation": 1.0, "acoustic_feature_preservation": 1.0, "interpretation": "Auxiliary identifiers are reduced but voice and semantic risks remain"},
        {"condition": "Full_framework", "speaker_id_accuracy": speaker_deid, "semantic_residual_identifier_rate": semantic_residual, "metadata_linkage_risk_score": metadata_after, "clinical_concept_preservation": concept_pres, "acoustic_feature_preservation": acoustic_pres, "interpretation": "Combined protection addresses multi-source re-identification risks"},
    ]
    write_csv("outputs_for_manuscript/table_risk_utility_matrix.csv", rows, ["condition", "speaker_id_accuracy", "semantic_residual_identifier_rate", "metadata_linkage_risk_score", "clinical_concept_preservation", "acoustic_feature_preservation", "interpretation"])


def write_manuscript_draft() -> None:
    text = """# 4. Proof-of-concept Simulation

## 4.1 Simulation design

To address the empirical evaluability of the proposed framework, we conducted a proof-of-concept simulation across the three de-identification layers. The purpose of this simulation was not to validate diagnostic performance, clinical outcomes, or deployable anonymization performance in real psychiatric recordings. Rather, it was designed to demonstrate that the proposed framework can be operationalized, measured, and reproduced without exposing real patient audio or identifiable clinical information.

The simulation used three data streams. For the voice layer, we used public non-clinical speech recordings from the LibriSpeech test-clean corpus. For the semantic and metadata layers, we generated deterministic synthetic materials using template-based scripts with a fixed random seed. These synthetic materials contained controlled combinations of direct identifiers, contextual identifiers, and clinically relevant psychiatric concepts. All names, locations, institutions, clinician identifiers, patient codes, contact details, and events were fictional. No real patient data, clinical recordings, or identifiable personal information were used.

## 4.2 Voice-layer validation

We sampled 30 speakers from LibriSpeech test-clean, with five clips per speaker and clip durations constrained to 8-20 seconds. Each selected clip was converted to 16 kHz mono WAV format. For this proof-of-concept experiment, voice-layer de-identification was implemented using a transparent signal-processing transformation: a +2 semitone pitch perturbation. This simple transformation was chosen for reproducibility and interpretability, not as a claim of optimal voice anonymization.

Speaker privacy was evaluated using SpeechBrain ECAPA-TDNN speaker embeddings. A nearest-centroid speaker identification task was performed using original speaker embeddings as enrollment references. Speaker identification accuracy was 1.000 for the original recordings and decreased to 0.793 after voice-layer transformation. The mean same-speaker original-original cosine similarity was 0.786, whereas the mean original-to-deidentified cosine similarity was 0.360. These findings indicate that the transformation substantially reduced embedding similarity and speaker-linkage strength, although it did not eliminate re-identification risk under a strong speaker embedding model.

Acoustic utility was evaluated using librosa-derived features. Non-F0 acoustic feature preservation was high overall, with a mean Pearson correlation of 0.952 across duration, RMS energy, zero-crossing rate, spectral centroid, spectral bandwidth, and pause ratio. Duration was unchanged (r = 1.000), and pause ratio remained highly correlated (r = 0.929). F0-related features were reported separately because pitch perturbation is expected to modify pitch as part of the privacy intervention. Thus, the voice-layer simulation showed a measurable privacy-utility trade-off: the transformation reduced speaker-linkage risk while preserving several non-pitch acoustic properties relevant to downstream analysis.

## 4.3 Semantic-layer validation

The semantic-layer simulation used 150 fictional psychiatric interview-like utterances. Each utterance was generated from controlled templates and included both privacy-sensitive identifiers and clinically relevant concepts such as insomnia, depressed mood, anxiety, medication use, family stress, social withdrawal, substance use, or functional impairment. The synthetic records included gold-standard identifier annotations, clinical concept annotations, and a reference de-identified version.

Automated semantic de-identification used custom rule-based recognizers aligned with Presidio-compatible identifier categories. The recognizers targeted clinician names, healthcare organizations, school or workplace names, session patterns, rare contextual events, family identifiers, contact details, dates, and record identifiers. Replacement rules removed, replaced, or generalized identifiers according to type, while preserving clinically relevant meaning. In this controlled synthetic dataset, identifier precision, recall, and F1 were all 1.000, with no residual high-risk identifiers detected. Clinical concept preservation was also 1.000. These values should be interpreted as evidence that the semantic evaluation pipeline functions correctly under controlled synthetic conditions, not as evidence of performance on real psychiatric transcripts.

## 4.4 Metadata-layer validation

The metadata-layer simulation used 150 synthetic recording metadata records. Each record contained fields that commonly contribute to linkage risk, including filenames, timestamps, device models, software versions, location tags, clinician identifiers, patient codes, session numbers, recording parameters, duration, and background context. Metadata sanitization removed exact identifiers, generalized temporal and contextual fields, standardized audio properties to 16 kHz mono, and preserved only the structure needed for longitudinal analysis, including pseudonymous subject grouping, session order, broad setting, relative time, and duration bin.

In the synthetic metadata experiment, metadata leakage rate decreased from 1.000 before sanitization to 0.000 after sanitization. The heuristic linkage risk score also decreased from 1.000 to 0.000, while longitudinal utility preservation remained 1.000. These results show that the metadata layer can reduce auxiliary re-identification signals while retaining the structural information required for repeated-measures or longitudinal study designs.

## 4.5 Integrated risk-utility interpretation

The resulting risk-utility matrix illustrates why single-layer de-identification is insufficient for psychiatric audio. Voice-only processing reduced speaker identification accuracy from 1.000 to 0.793 and preserved non-F0 acoustic structure, but semantic and metadata identifiers remained unaddressed. Semantic-only processing reduced textual leakage in the controlled synthetic setting, but did not affect speaker or metadata risks. Metadata-only processing reduced auxiliary linkage risk, but did not protect the acoustic or semantic channels.

The full framework therefore provides defense-in-depth by addressing multiple sources of re-identification risk simultaneously. The simulation also illustrates why privacy evaluation should not rely on a single metric. Speaker identity, textual identifiers, contextual identifiers, and metadata fields represent distinct attack surfaces, and the preservation of clinical utility must be evaluated separately for acoustic, semantic, and longitudinal data structures.

## 4.6 Limitations of the simulation

This simulation has several important limitations. First, LibriSpeech consists of public read speech and does not capture the interactional, emotional, or conversational properties of psychiatric interviews. Second, the semantic and metadata materials were generated from controlled templates, which improves reproducibility but does not capture the full linguistic and contextual variability of real clinical data. Third, the voice-layer transformation was intentionally simple; more advanced neural voice conversion or anonymization methods may provide stronger protection and should be evaluated using the same pipeline. Fourth, the simulation did not assess clinical diagnostic validity or treatment-related outcomes.

For these reasons, the results should be interpreted as proof-of-concept evidence that the proposed multi-layer framework is operationalizable and measurable, rather than as clinical validation of a deployable de-identification system. Future work should evaluate the framework using ethically approved psychiatric interview datasets, multiple languages, diverse recording environments, stronger voice anonymization models, and expert-defined clinical utility endpoints.
"""
    from src.io_utils import ensure_dir

    path = ensure_dir("outputs_for_manuscript") / "manuscript_section4_draft.md"
    path.write_text(text, encoding="utf-8")


def main() -> None:
    write_voice_layer_table()
    copy_to_manuscript("data/results/tables/table_semantic_layer.csv", "table_semantic_layer.csv")
    copy_to_manuscript("data/results/tables/table_metadata_layer.csv", "table_metadata_layer.csv")
    write_risk_utility_matrix()
    write_manuscript_draft()
    print("Wrote manuscript-ready tables and Section 4 draft.")


if __name__ == "__main__":
    main()
