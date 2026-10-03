from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import numpy as np

from .io_utils import project_path


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def nearest_centroid_accuracy(test_items: list[dict], centroids: dict[str, np.ndarray]) -> float:
    if not test_items:
        return 0.0
    correct = 0
    for item in test_items:
        scores = {speaker: cosine_similarity(item["embedding"], centroid) for speaker, centroid in centroids.items()}
        predicted = max(scores, key=scores.get)
        correct += int(predicted == item["speaker_id"])
    return correct / len(test_items)


def verification_trials(test_items: list[dict], centroids: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    labels = []
    scores = []
    for item in test_items:
        for speaker, centroid in centroids.items():
            labels.append(1 if speaker == item["speaker_id"] else 0)
            scores.append(cosine_similarity(item["embedding"], centroid))
    return np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)


def equal_error_rate(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0 or np.sum(labels == 1) == 0 or np.sum(labels == 0) == 0:
        return 0.0
    thresholds = np.r_[np.inf, np.sort(np.unique(scores))[::-1], -np.inf]
    best = 1.0
    for threshold in thresholds:
        predicted = scores >= threshold
        false_reject_rate = np.sum((~predicted) & (labels == 1)) / np.sum(labels == 1)
        false_accept_rate = np.sum(predicted & (labels == 0)) / np.sum(labels == 0)
        eer = (false_reject_rate + false_accept_rate) / 2
        if abs(false_reject_rate - false_accept_rate) < best:
            best = abs(false_reject_rate - false_accept_rate)
            best_eer = eer
    return float(best_eer)


def _load_embedding(path: str) -> np.ndarray:
    return np.load(project_path(path))


def evaluate_embeddings(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    items = []
    for row in rows:
        embedding = _load_embedding(row["embedding_path"])
        items.append(
            {
                "audio_id": row["audio_id"],
                "speaker_id": row["speaker_id"],
                "condition": row["condition"],
                "method": row.get("voice_deid_method", ""),
                "parameter": row.get("parameter", ""),
                "embedding": embedding.reshape(-1),
            }
        )

    original = [item for item in items if item["condition"] == "original"]
    conditions = sorted({item["condition"] for item in items if item["condition"] != "original"})
    original_by_id = {item["audio_id"]: item for item in original}
    if not original or len(original_by_id) != len(original):
        raise ValueError("Original embeddings must have unique audio IDs and be nonempty.")
    for speaker in {item["speaker_id"] for item in original}:
        clips = [int(item["audio_id"].split("_C")[-1]) for item in original if item["speaker_id"] == speaker]
        if sorted(clips) != [1, 2, 3, 4, 5]:
            raise ValueError(f"Speaker {speaker} must have original clips C1–C5 exactly once.")

    similarities = []
    original_by_speaker: dict[str, list[dict]] = defaultdict(list)
    for item in original:
        original_by_speaker[item["speaker_id"]].append(item)

    same_speaker_scores = []
    for speaker_items in original_by_speaker.values():
        for a, b in combinations(speaker_items, 2):
            score = cosine_similarity(a["embedding"], b["embedding"])
            same_speaker_scores.append(score)
            similarities.append({"condition": "original_same_speaker", "cosine_similarity": score})

    for a, b in combinations(original, 2):
        if a["speaker_id"] != b["speaker_id"]:
            similarities.append({"condition": "different_speaker", "cosine_similarity": cosine_similarity(a["embedding"], b["embedding"])})

    enrollment_by_speaker: dict[str, list[np.ndarray]] = defaultdict(list)
    original_test = []
    for item in original:
        clip_number = int(item["audio_id"].split("_C")[-1])
        if clip_number <= 3:
            enrollment_by_speaker[item["speaker_id"]].append(item["embedding"])
        else:
            original_test.append(item)
    centroids = {speaker: np.mean(embeddings, axis=0) for speaker, embeddings in enrollment_by_speaker.items() if embeddings}
    heldout_ids = {item["audio_id"] for item in original_test}
    evaluation_counts = {
        "n_enrollment_clips": sum(len(group) for group in enrollment_by_speaker.values()),
        "n_test_clips": len(original_test),
        "n_genuine_trials": len(original_test),
        "n_impostor_trials": len(original_test) * (len(centroids) - 1),
    }

    labels, scores = verification_trials(original_test, centroids)
    summary = [
        {
            "condition": "original",
            "method": "original",
            "parameter": "-",
            "mean_cosine_similarity": float(np.mean(same_speaker_scores)) if same_speaker_scores else 0.0,
            "top1_speaker_id_accuracy": nearest_centroid_accuracy(original_test, centroids),
            "linkage_success_rate": nearest_centroid_accuracy(original_test, centroids),
            "speaker_verification_eer": equal_error_rate(labels, scores),
            **evaluation_counts,
        }
    ]

    for condition in conditions:
        condition_items = [item for item in items if item["condition"] == condition]
        condition_by_id = {item["audio_id"]: item for item in condition_items}
        if len(condition_by_id) != len(condition_items) or set(condition_by_id) != set(original_by_id):
            raise ValueError(f"Condition {condition} must contain exactly the same unique clip IDs as original.")
        if any(item["speaker_id"] != original_by_id[item["audio_id"]]["speaker_id"] for item in condition_items):
            raise ValueError(f"Speaker labels do not match original for condition {condition}.")
        # Use only C4/C5 for both original and transformed recognition/verification.
        # Paired cosine distributions still describe all 150 original/transformed clips.
        condition_test = [item for item in condition_items if item["audio_id"] in heldout_ids]
        original_deid_scores = []
        for item in condition_items:
            if item["audio_id"] in original_by_id:
                score = cosine_similarity(original_by_id[item["audio_id"]]["embedding"], item["embedding"])
                original_deid_scores.append(score)
                similarities.append({"condition": f"original_vs_{condition}", "cosine_similarity": score})
        labels, scores = verification_trials(condition_test, centroids)
        method = condition_items[0].get("method", "") if condition_items else ""
        parameter = condition_items[0].get("parameter", "") if condition_items else ""
        accuracy = nearest_centroid_accuracy(condition_test, centroids)
        summary.append(
            {
                "condition": condition,
                "method": method,
                "parameter": parameter,
                "mean_cosine_similarity": float(np.mean(original_deid_scores)) if original_deid_scores else 0.0,
                "top1_speaker_id_accuracy": accuracy,
                "linkage_success_rate": accuracy,
                "speaker_verification_eer": equal_error_rate(labels, scores),
                **evaluation_counts,
            }
        )
    return summary, similarities
