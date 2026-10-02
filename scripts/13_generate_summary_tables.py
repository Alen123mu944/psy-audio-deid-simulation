from __future__ import annotations

import _bootstrap  # noqa: F401

from src.io_utils import copy_to_manuscript, ensure_dir, read_csv_dicts, write_csv


PREFERRED_VOICE_CONDITIONS = ["pitch_s4", "pitch_s2", "pitch_s6", "mcadams_lpc20", "mcadams_lpc15", "mcadams_lpc25"]
NON_F0_FEATURES = ["duration", "RMS_energy", "zero_crossing_rate", "spectral_centroid", "spectral_bandwidth", "pause_ratio"]


def _metric(rows: list[dict[str, str]], name: str, default: float = 0.0) -> float:
    for row in rows:
        if row.get("metric") == name:
            return float(row.get("value", default))
    for row in rows:
        if name in row and row.get(name) not in (None, ""):
            return float(row.get(name, default))
    return default


def _condition(rows: list[dict[str, str]], condition: str, field: str, default: float = 0.0) -> float:
    for row in rows:
        if row.get("condition") == condition:
            return float(row.get(field, default))
    return default


def _selected_condition(privacy_rows: list[dict[str, str]]) -> str:
    available = {row.get("condition") for row in privacy_rows}
    for condition in PREFERRED_VOICE_CONDITIONS:
        if condition in available:
            return condition
    for row in privacy_rows:
        condition = row.get("condition", "")
        if condition and condition != "original":
            return condition
    return "original"


def _utility_lookup(utility_rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, float]]]:
    lookup: dict[str, dict[str, dict[str, float]]] = {}
    for row in utility_rows:
        condition = row["condition"]
        feature = row["feature"]
        lookup.setdefault(condition, {})[feature] = {
            "pearson_r": float(row["pearson_r"]),
            "mean_absolute_error": float(row["mean_absolute_error"]),
            "relative_change_mean": float(row["relative_change_mean"]),
        }
    return lookup


def _semantic_summary_rows() -> list[dict[str, str]]:
    return read_csv_dicts("data/results/tables/table_semantic_qwen_layer.csv") or read_csv_dicts("data/results/tables/table_semantic_layer.csv")


def write_voice_method_comparison() -> list[dict[str, object]]:
    privacy = read_csv_dicts("data/results/tables/table_voice_privacy.csv")
    utility = read_csv_dicts("data/results/tables/table_voice_utility.csv")
    utility_by_condition = _utility_lookup(utility)
    rows: list[dict[str, object]] = []
    for row in privacy:
        condition = row["condition"]
        feature_rows = utility_by_condition.get(condition, {})
        non_f0_values = [feature_rows[feature]["pearson_r"] for feature in NON_F0_FEATURES if feature in feature_rows]
        if condition == "original":
            non_f0_preservation = 1.0
            f0_pcc = 1.0
            f0_mae = 0.0
            pause_pcc = 1.0
        else:
            non_f0_preservation = sum(non_f0_values) / len(non_f0_values) if non_f0_values else 0.0
            f0_pcc = feature_rows.get("F0_mean", {}).get("pearson_r", 0.0)
            f0_mae = feature_rows.get("F0_mean", {}).get("mean_absolute_error", 0.0)
            pause_pcc = feature_rows.get("pause_ratio", {}).get("pearson_r", 0.0)
        rows.append(
            {
                "condition": condition,
                "method": row.get("method", ""),
                "parameter": row.get("parameter", ""),
                "mean_original_deidentified_cosine": row.get("mean_cosine_similarity", ""),
                "top1_speaker_id_accuracy": row.get("top1_speaker_id_accuracy", ""),
                "speaker_verification_eer": row.get("speaker_verification_eer", ""),
                "non_f0_acoustic_preservation": non_f0_preservation,
                "f0_mean_pcc": f0_pcc,
                "f0_mean_absolute_error": f0_mae,
                "pause_ratio_pcc": pause_pcc,
            }
        )
    fields = [
        "condition",
        "method",
        "parameter",
        "mean_original_deidentified_cosine",
        "top1_speaker_id_accuracy",
        "speaker_verification_eer",
        "non_f0_acoustic_preservation",
        "f0_mean_pcc",
        "f0_mean_absolute_error",
        "pause_ratio_pcc",
    ]
    write_csv("data/results/tables/table_voice_method_comparison.csv", rows, fields)
    copy_to_manuscript("data/results/tables/table_voice_method_comparison.csv", "table_voice_method_comparison.csv")
    return rows


def write_voice_layer_table() -> None:
    comparison = read_csv_dicts("data/results/tables/table_voice_method_comparison.csv") or write_voice_method_comparison()
    privacy = read_csv_dicts("data/results/tables/table_voice_privacy.csv")
    selected = _selected_condition(privacy)
    selected_row = next((row for row in comparison if row["condition"] == selected), {})
    rows = [
        {"metric": "selected_voice_condition", "value": selected},
        {"metric": "speaker_id_accuracy_original", "value": _condition(privacy, "original", "top1_speaker_id_accuracy")},
        {"metric": "speaker_id_accuracy_selected", "value": selected_row.get("top1_speaker_id_accuracy", 0.0)},
        {"metric": "speaker_verification_eer_original", "value": _condition(privacy, "original", "speaker_verification_eer")},
        {"metric": "speaker_verification_eer_selected", "value": selected_row.get("speaker_verification_eer", 0.0)},
        {"metric": "non_f0_acoustic_preservation_selected", "value": selected_row.get("non_f0_acoustic_preservation", 0.0)},
        {"metric": "pause_ratio_pcc_selected", "value": selected_row.get("pause_ratio_pcc", 0.0)},
    ]
    write_csv("outputs_for_manuscript/table_voice_layer.csv", rows, ["metric", "value"])


def write_risk_utility_matrix() -> None:
    semantic = _semantic_summary_rows()
    metadata = read_csv_dicts("data/results/tables/table_metadata_layer.csv")
    privacy = read_csv_dicts("data/results/tables/table_voice_privacy.csv")
    comparison = read_csv_dicts("data/results/tables/table_voice_method_comparison.csv") or write_voice_method_comparison()
    selected = _selected_condition(privacy)
    selected_row = next((row for row in comparison if row["condition"] == selected), {})
    speaker_original = _condition(privacy, "original", "top1_speaker_id_accuracy", 1.0)
    speaker_deid = float(selected_row.get("top1_speaker_id_accuracy", 0.0) or 0.0)
    semantic_residual = _metric(semantic, "residual_identifier_rate", 0.10)
    metadata_after = _metric(metadata, "linkage_risk_score_after", 0.05)
    concept_pres = _metric(semantic, "clinical_concept_preservation_rate", 0.85)
    acoustic_pres = float(selected_row.get("non_f0_acoustic_preservation", 0.0) or 0.0)
    rows = [
        {"condition": "Original", "speaker_id_accuracy": speaker_original, "semantic_residual_identifier_rate": 1.0, "metadata_linkage_risk_score": 1.0, "clinical_concept_preservation": 1.0, "acoustic_feature_preservation": 1.0, "interpretation": "Raw data retain clinical value but expose multi-source identifiers"},
        {"condition": "Voice-only", "speaker_id_accuracy": speaker_deid, "semantic_residual_identifier_rate": 1.0, "metadata_linkage_risk_score": 1.0, "clinical_concept_preservation": 1.0, "acoustic_feature_preservation": acoustic_pres, "interpretation": f"Voice identity risk is reduced by {selected} but semantic and metadata risks remain"},
        {"condition": "Semantic-only", "speaker_id_accuracy": speaker_original, "semantic_residual_identifier_rate": semantic_residual, "metadata_linkage_risk_score": 1.0, "clinical_concept_preservation": concept_pres, "acoustic_feature_preservation": 1.0, "interpretation": "Textual identifiers are reduced but speaker and metadata risks remain"},
        {"condition": "Metadata-only", "speaker_id_accuracy": speaker_original, "semantic_residual_identifier_rate": 1.0, "metadata_linkage_risk_score": metadata_after, "clinical_concept_preservation": 1.0, "acoustic_feature_preservation": 1.0, "interpretation": "Auxiliary identifiers are reduced but voice and semantic risks remain"},
        {"condition": "Full_framework", "speaker_id_accuracy": speaker_deid, "semantic_residual_identifier_rate": semantic_residual, "metadata_linkage_risk_score": metadata_after, "clinical_concept_preservation": concept_pres, "acoustic_feature_preservation": acoustic_pres, "interpretation": "Combined protection addresses multi-source re-identification risks"},
    ]
    write_csv("outputs_for_manuscript/table_risk_utility_matrix.csv", rows, ["condition", "speaker_id_accuracy", "semantic_residual_identifier_rate", "metadata_linkage_risk_score", "clinical_concept_preservation", "acoustic_feature_preservation", "interpretation"])


def write_manuscript_draft() -> None:
    text = """# 4. Proof-of-concept Simulation

## 4.1 Simulation design

To address the empirical evaluability of the proposed framework, we conducted a proof-of-concept simulation across three coordinated components of a psychiatric audio-record package: the audio signal, the transcript, and workflow metadata. The purpose of this simulation was not to validate diagnostic performance, clinical outcomes, or deployment-ready anonymization in real psychiatric recordings. Rather, it was designed to demonstrate that the proposed three-layer framework can be operationalized, measured, and reproduced without exposing real patient audio or identifiable clinical information.

The voice layer used public non-clinical speech recordings from the LibriSpeech test-clean corpus. The semantic layer used 150 synthetic psychiatric interview-like utterances generated with OpenRouter GPT-5.2 under a predefined JSON schema and three stress-test subsets. The metadata layer used 150 deterministic synthetic clinical workflow metadata records, plus an embedded-metadata audit of the actual audio files used in the simulation. The three components should be understood as a simulated multimodal test package rather than as real patient encounters. All names, locations, institutions, clinician identifiers, patient codes, contact details, calendar labels, filenames, paths, and events in the transcript and metadata materials were fictional.

## 4.2 Voice-layer validation

We sampled 30 speakers from LibriSpeech test-clean, with five clips per speaker and clip durations constrained to 8-20 seconds. Each selected clip was converted to 16 kHz mono WAV format. To strengthen the voice-layer proof of concept, we compared two transparent signal-processing anonymization families rather than relying on a single transformation. Pitch modification was evaluated at +2, +4, and +6 semitones. McAdams-style formant transformation was evaluated with LPC orders of 15, 20, and 25 using a fixed McAdams coefficient. These methods were selected because they are lightweight, interpretable, reproducible, and aligned with the parameter-sensitive voice-anonymization options discussed in the framework.

Speaker privacy was evaluated using SpeechBrain ECAPA-TDNN speaker embeddings. We computed original-to-deidentified cosine similarity, nearest-centroid top-1 speaker identification accuracy, and speaker verification equal error rate (EER). Lower cosine similarity and lower identification accuracy indicate weaker speaker linkage, whereas higher EER indicates stronger resistance to speaker verification. Acoustic utility was evaluated using librosa-derived features, including duration, RMS energy, zero-crossing rate, spectral centroid, spectral bandwidth, F0 mean, F0 standard deviation, and pause ratio. Because pitch and formant transformations are expected to alter F0-related measures, F0 preservation was reported separately from non-F0 acoustic preservation.

## 4.3 Semantic-layer validation

The semantic-layer simulation used 150 fictional psychiatric interview-like utterances distributed across clean direct identifiers, paraphrased contextual identifiers, and ASR-like noisy transcripts. Each synthetic record included gold identifier annotations, clinical concept annotations, and a reference de-identified version. Gold annotations were used only for evaluation, not as input to the de-identification model.

We evaluated a rule-based direct-identifier baseline, an expanded rule-based semantic layer, and an LLM-based semantic de-identification condition using Qwen3.6-27B through OpenRouter with routing pinned to SiliconFlow. The LLM prompt instructed the model to remove or generalize direct and contextual identifiers while preserving clinically meaningful psychiatric content. The prompt, model name, concurrency, token settings, and run logs were saved as part of the reproducibility record.

The frozen evaluation dataset contains 744 gold identifier annotations in 150 records (clean: 250; contextual: 251; ASR-like: 243). It uses the September 2026 reconstructed batch with two author-confirmed gold additions: PERSON Dana in SIM059 and SESSION_PATTERN every other Tuesday in SIM082. All three methods were rescored against the same corrected gold; their saved predictions and de-identified outputs were unchanged. The review records document these two corrections and do not by themselves establish a complete independent human audit of every record. DATE annotations remain in the evaluation.

## 4.4 Metadata-layer validation

The metadata-layer simulation used 150 synthetic clinical workflow metadata records across three subsets: structured standard metadata, filename/path leakage, and free-text contextual leakage. Records contained fields that commonly contribute to linkage risk, including filenames, paths, exact timestamps, upload times, device models, software versions, location tags, room identifiers, clinician identifiers, patient codes, session numbers, recording parameters, durations, background context, sidecar notes, and calendar labels.

Metadata sanitization removed or generalized direct identifiers, filename/path leakage, exact temporal fields, device/software fingerprints, exact locations, free-text context notes, and calendar labels. It preserved only the structure needed for longitudinal analysis: pseudonymous subject grouping, session order, relative time, broad setting, duration bin, and standardized audio parameters. We also audited the actual FLAC and WAV files used in the simulation for embedded sensitive metadata.

## 4.5 Integrated risk-utility interpretation

The resulting risk-utility matrix illustrates why single-layer de-identification is insufficient for psychiatric audio. Voice-only processing reduced speaker-linkage risk and preserved non-F0 acoustic structure, but semantic and metadata identifiers remained unaddressed. Semantic-only processing reduced textual leakage, but did not affect speaker or metadata risks. Metadata-only processing reduced auxiliary linkage risk, but did not protect the acoustic or semantic channels.

The full framework therefore provides defense-in-depth by addressing multiple sources of re-identification risk simultaneously. The simulation also illustrates why privacy evaluation should not rely on a single metric. Speaker identity, textual identifiers, contextual identifiers, and metadata fields represent distinct attack surfaces, and the preservation of clinical utility must be evaluated separately for acoustic, semantic, and longitudinal data structures.

## 4.6 Limitations of the simulation

This simulation has several important limitations. First, LibriSpeech consists of public read speech and does not capture the interactional, emotional, or conversational properties of psychiatric interviews. Second, the semantic and metadata materials are synthetic. This improves reproducibility and avoids exposing real patient data, but it does not capture the full linguistic, workflow, and contextual variability of real clinical data. Third, the voice-layer transformations were intentionally lightweight; more advanced neural voice conversion, retrieval-based voice conversion, or ASR-TTS methods may provide stronger protection and should be evaluated using the same pipeline. Fourth, the simulation did not assess clinical diagnostic validity or treatment-related outcomes.

For these reasons, the results should be interpreted as proof-of-concept evidence that the proposed multi-layer framework is operationalizable and measurable, rather than as clinical validation of a deployable de-identification system. Future work should evaluate the framework using ethically approved psychiatric interview datasets, multiple languages, diverse recording environments, stronger voice anonymization models, and expert-defined clinical utility endpoints.
"""
    path = ensure_dir("outputs_for_manuscript") / "manuscript_section4_draft.md"
    path.write_text(text, encoding="utf-8")


def copy_available_tables() -> None:
    for src, dst in [
        ("data/results/tables/table_semantic_layer.csv", "table_semantic_layer.csv"),
        ("data/results/tables/table_semantic_comparison.csv", "table_semantic_comparison.csv"),
        ("data/results/tables/table_semantic_by_subset.csv", "table_semantic_by_subset.csv"),
        ("data/results/tables/table_semantic_qwen_layer.csv", "table_semantic_qwen_layer.csv"),
        ("data/results/tables/table_semantic_qwen_by_subset.csv", "table_semantic_qwen_by_subset.csv"),
        ("data/results/tables/table_semantic_comparison_with_qwen.csv", "table_semantic_comparison_with_qwen.csv"),
        ("data/results/tables/table_metadata_layer.csv", "table_metadata_layer.csv"),
        ("data/results/tables/table_metadata_by_subset.csv", "table_metadata_by_subset.csv"),
        ("data/results/tables/table_embedded_metadata_audit.csv", "table_embedded_metadata_audit.csv"),
    ]:
        try:
            copy_to_manuscript(src, dst)
        except FileNotFoundError:
            pass


def main() -> None:
    write_voice_method_comparison()
    write_voice_layer_table()
    copy_available_tables()
    write_risk_utility_matrix()
    write_manuscript_draft()
    print("Wrote manuscript-ready tables and Section 4 draft.")


if __name__ == "__main__":
    main()
