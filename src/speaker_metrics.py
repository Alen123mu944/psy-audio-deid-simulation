from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import random

import numpy as np


def simulate_speaker_privacy(seed: int = 42) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    similarities = []
    for _ in range(200):
        similarities.append({"condition": "original_same_speaker", "cosine_similarity": rng.uniform(0.72, 0.92)})
        similarities.append({"condition": "original_vs_deidentified", "cosine_similarity": rng.uniform(0.38, 0.68)})
        similarities.append({"condition": "different_speaker", "cosine_similarity": rng.uniform(0.05, 0.35)})
    summary = [
        {"condition": "original", "mean_cosine_similarity": 0.82, "top1_speaker_id_accuracy": 0.91, "linkage_success_rate": 0.88},
        {"condition": "voice_only", "mean_cosine_similarity": 0.53, "top1_speaker_id_accuracy": 0.42, "linkage_success_rate": 0.39},
        {"condition": "full_framework", "mean_cosine_similarity": 0.53, "top1_speaker_id_accuracy": 0.42, "linkage_success_rate": 0.39},
    ]
    return summary, similarities


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


def evaluate_embeddings(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    items = []
    for row in rows:
        embedding = np.load(row["embedding_path"])
        items.append(
            {
                "audio_id": row["audio_id"],
                "speaker_id": row["speaker_id"],
                "condition": row["condition"],
                "embedding": embedding.reshape(-1),
            }
        )

    original = [item for item in items if item["condition"] == "original"]
    deidentified = [item for item in items if item["condition"] == "voice_deidentified"]
    original_by_id = {item["audio_id"]: item for item in original}
    deid_by_id = {item["audio_id"]: item for item in deidentified}

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

    original_deid_scores = []
    for audio_id, original_item in original_by_id.items():
        if audio_id in deid_by_id:
            score = cosine_similarity(original_item["embedding"], deid_by_id[audio_id]["embedding"])
            original_deid_scores.append(score)
            similarities.append({"condition": "original_vs_deidentified", "cosine_similarity": score})

    different_speaker_scores = []
    for a, b in combinations(original, 2):
        if a["speaker_id"] != b["speaker_id"]:
            score = cosine_similarity(a["embedding"], b["embedding"])
            different_speaker_scores.append(score)
            similarities.append({"condition": "different_speaker", "cosine_similarity": score})

    enrollment_by_speaker: dict[str, list[np.ndarray]] = defaultdict(list)
    original_test = []
    deid_test = []
    for item in original:
        clip_number = int(item["audio_id"].split("_C")[-1])
        if clip_number <= 3:
            enrollment_by_speaker[item["speaker_id"]].append(item["embedding"])
        else:
            original_test.append(item)
    for item in deidentified:
        deid_test.append(item)

    centroids = {speaker: np.mean(embeddings, axis=0) for speaker, embeddings in enrollment_by_speaker.items() if embeddings}
    original_accuracy = nearest_centroid_accuracy(original_test, centroids)
    deid_accuracy = nearest_centroid_accuracy(deid_test, centroids)

    summary = [
        {
            "condition": "original",
            "mean_cosine_similarity": float(np.mean(same_speaker_scores)) if same_speaker_scores else 0.0,
            "top1_speaker_id_accuracy": original_accuracy,
            "linkage_success_rate": original_accuracy,
        },
        {
            "condition": "voice_only",
            "mean_cosine_similarity": float(np.mean(original_deid_scores)) if original_deid_scores else 0.0,
            "top1_speaker_id_accuracy": deid_accuracy,
            "linkage_success_rate": deid_accuracy,
        },
        {
            "condition": "full_framework",
            "mean_cosine_similarity": float(np.mean(original_deid_scores)) if original_deid_scores else 0.0,
            "top1_speaker_id_accuracy": deid_accuracy,
            "linkage_success_rate": deid_accuracy,
        },
    ]
    return summary, similarities
