from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from .llm_utils import load_json

RUBRIC_SCORE_ORDER = [
    "relevance",
    "clarity",
    "calibration",
    "current_drivers",
    "evidence_use",
    "base_rates",
    "quantitative_reasoning",
    "counterarguments",
]
RUBRIC_SCORE_LABELS = {
    "relevance": "Relevance",
    "clarity": "Clarity",
    "calibration": "Calibration",
    "current_drivers": "Current drivers",
    "evidence_use": "Evidence use",
    "base_rates": "Base rates",
    "quantitative_reasoning": "Quantitative reasoning",
    "counterarguments": "Counterarguments",
}


def plot_overall_score_distribution(rows: list[dict], output_path: Path) -> None:
    counts = Counter(
        row["overall_score"]
        for row in rows
        if isinstance(row.get("overall_score"), int) and 0 <= row["overall_score"] <= 5
    )
    labels = [str(score) for score in range(6)]
    values = [counts.get(score, 0) for score in range(6)]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(
        labels,
        values,
        color="#4c78a8",
        edgecolor="#222222",
        linewidth=0.8,
    )
    ax.set_title("Gemini Flash Overall Score Distribution")
    ax.set_xlabel("Overall score")
    ax.set_ylabel("Number of rationales")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.8)

    label_offset = max(values) * 0.012 if values and max(values) > 0 else 1
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + (bar.get_width() / 2),
            value + label_offset,
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_dimension_means(rows: list[dict], output_path: Path) -> None:
    labels = []
    values = []
    for key in RUBRIC_SCORE_ORDER:
        dimension_scores = [
            row.get("scores", {}).get(key)
            for row in rows
            if isinstance(row.get("scores"), dict)
            and isinstance(row.get("scores", {}).get(key), (int, float))
        ]
        labels.append(RUBRIC_SCORE_LABELS[key])
        values.append(sum(dimension_scores) / len(dimension_scores))

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    bars = ax.barh(
        labels,
        values,
        color="#4c78a8",
        edgecolor="#222222",
        linewidth=0.8,
    )
    ax.invert_yaxis()
    ax.set_title("Average Flash Score by Rubric Dimension")
    ax.set_xlabel("Average dimension score (0-5)")
    ax.set_xlim(0, 5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#e6e6e6", linewidth=0.8)

    for bar, value in zip(bars, values):
        ax.text(
            value + 0.06,
            bar.get_y() + (bar.get_height() / 2),
            f"{value:.2f}",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    score_data = load_json(Path("workspace/outputs/rationale_scores.json"))
    rows = [row for row in score_data["scores"] if row.get("is_valid") is True]
    output_dir = Path("workspace/outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_score_path = output_dir / "flash_score_distribution.png"
    dimension_means_path = output_dir / "flash_dimension_means.png"

    plot_overall_score_distribution(rows, overall_score_path)
    plot_dimension_means(rows, dimension_means_path)


if __name__ == "__main__":
    main()
