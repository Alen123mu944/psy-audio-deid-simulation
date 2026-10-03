from __future__ import annotations

import _bootstrap  # noqa: F401

from src.io_utils import copy_to_manuscript, read_csv_dicts, write_csv


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
    qwen = read_csv_dicts("data/results/tables/table_semantic_qwen_layer.csv") or read_csv_dicts("outputs_for_manuscript/table_semantic_qwen_layer.csv")
    if qwen:
        return qwen
    comparison = read_csv_dicts("data/results/tables/table_semantic_comparison_with_qwen.csv") or read_csv_dicts("outputs_for_manuscript/table_semantic_comparison_with_qwen.csv")
    return [row for row in comparison if row.get("condition") == "full_semantic_layer"]


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
                "n_enrollment_clips": row.get("n_enrollment_clips", ""),
                "n_test_clips": row.get("n_test_clips", ""),
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
        "n_enrollment_clips",
        "n_test_clips",
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
    metadata = read_csv_dicts("data/results/tables/table_metadata_layer.csv") or read_csv_dicts("outputs_for_manuscript/table_metadata_layer.csv")
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


def copy_available_tables() -> None:
    for src, dst in [
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
    print("Wrote manuscript-ready tables.")


if __name__ == "__main__":
    main()
