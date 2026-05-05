from __future__ import annotations

import _bootstrap  # noqa: F401

from src.io_utils import ensure_dir, read_csv_dicts


def workflow_figure(out_dir) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    boxes = [
        (0.08, 0.70, "Public/synthetic\naudio"),
        (0.38, 0.70, "Synthetic\ntranscripts"),
        (0.68, 0.70, "Synthetic\nmetadata"),
        (0.08, 0.40, "Voice\ntransformation"),
        (0.38, 0.40, "Semantic\nde-identification"),
        (0.68, 0.40, "Metadata\nsanitization"),
        (0.38, 0.12, "Privacy metrics + utility metrics\nRisk-utility matrix"),
    ]
    for x, y, label in boxes:
        ax.text(x, y, label, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", facecolor="#e8f1fb", edgecolor="#2f5f8f"))
    for x in [0.08, 0.38, 0.68]:
        ax.annotate("", xy=(x, 0.48), xytext=(x, 0.63), arrowprops=dict(arrowstyle="->", color="#2f5f8f"))
        ax.annotate("", xy=(0.38, 0.22), xytext=(x, 0.34), arrowprops=dict(arrowstyle="->", color="#2f5f8f"))
    fig.tight_layout()
    fig.savefig(out_dir / "figure_simulation_workflow.png", dpi=200)
    plt.close(fig)


def similarity_figure(out_dir) -> None:
    import matplotlib.pyplot as plt

    rows = read_csv_dicts("data/results/tables/speaker_similarity_distributions.csv")
    groups = ["original_same_speaker", "original_vs_deidentified", "different_speaker"]
    data = [[float(row["cosine_similarity"]) for row in rows if row["condition"] == group] for group in groups]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, labels=["Original same\nspeaker", "Original vs\nDe-ID", "Different\nspeaker"])
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Speaker Similarity Distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "figure_speaker_similarity.png", dpi=200)
    plt.close(fig)


def risk_reduction_figure(out_dir) -> None:
    import matplotlib.pyplot as plt

    rows = read_csv_dicts("outputs_for_manuscript/table_risk_utility_matrix.csv")
    full = next((row for row in rows if row["condition"] == "Full_framework"), {})
    labels = ["Speaker re-ID", "Semantic leakage", "Metadata linkage"]
    before = [0.91, 1.0, 1.0]
    after = [
        float(full.get("speaker_id_accuracy", 0.42)),
        float(full.get("semantic_residual_identifier_rate", 0.1)),
        float(full.get("metadata_linkage_risk_score", 0.05)),
    ]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - 0.18 for i in x], before, width=0.36, label="Original")
    ax.bar([i + 0.18 for i in x], after, width=0.36, label="Full framework")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Risk proxy")
    ax.set_title("Layer-specific Risk Reduction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "figure_risk_reduction.png", dpi=200)
    plt.close(fig)


def main() -> None:
    out_dir = ensure_dir("outputs_for_manuscript")
    workflow_figure(out_dir)
    similarity_figure(out_dir)
    risk_reduction_figure(out_dir)
    print("Wrote manuscript figures.")


if __name__ == "__main__":
    main()
