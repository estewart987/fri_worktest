"""Select questions for top-rationale review and synthesis.

Step 2 of the pipeline. Reads rationale_scores.json (output from score_rationales.py),
filters to questions with enough high-quality non-redacted rationales, and writes
the top 5 to five_selected_questions.json for use by analyze_questions.py.
"""

import argparse
from pathlib import Path

from .llm_utils import (
    is_any_redacted,
    is_exact_redacted,
    is_post_resolution,
    load_json,
    normalized_reasoning,
    reasoning_text,
    write_json,
)


DEFAULT_SCORES = Path("workspace/outputs/rationale_scores.json")
DEFAULT_OUTPUT = Path("workspace/outputs/five_selected_questions.json")

# A question should have at least this many total rationales.
MIN_RATIONALE_COUNT = 10

# Rationales scoring below this are excluded from those sent to the Pro model.
# Score 3 = "adequate" on the 0-5 rubric.
MIN_SCORE = 3

# A question should have at least this many unique, non-redacted, adequately-scored rationales.
# Below 3 there's not enough material to compare.
MIN_UNIQUE_COUNT = 3

# Number of top-ranked rows to inspect for redaction / post-resolution flags.
TOP_N_TO_INSPECT = 3

# Exclude a question if this many of the top-N rows hit a disqualifying flag.
MOSTLY_THRESHOLD = 2


def question_summary(question_id: str, rows: list[dict]) -> dict:
    """Determines if a question has enough good rationales to include and summarizes score distribution for that question."""
    question_text = rows[0].get("question") or ""

    # Sort rationales from best to worst.
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            -(row.get("overall_score") or -999),
            -sum((row.get("scores") or {}).values()),
            row.get("row_number") or 0,
        ),
    )
    top_rows = sorted_rows[:TOP_N_TO_INSPECT]
    non_redacted = [row for row in rows if not is_any_redacted(reasoning_text(row))]

    # Filter for rationales that are scoreable, non-redacted, and meet the minimum eligibility criteria.
    non_redacted_ge_min = [
        row
        for row in non_redacted
        if isinstance(row.get("overall_score"), int)
        and row["overall_score"] >= MIN_SCORE
    ]

    # Remove duplicate rationales
    unique_ge_min: list[dict] = []
    seen: set[str] = set()
    for row in sorted_rows:
        if row not in non_redacted_ge_min:
            continue
        key = normalized_reasoning(reasoning_text(row))
        if key in seen:
            continue
        seen.add(key)
        unique_ge_min.append(row)

    scores = [
        row["overall_score"]
        for row in rows
        if isinstance(row.get("overall_score"), int)
    ]
    score_range = [min(scores), max(scores)] if scores else [None, None]
    score_span = score_range[1] - score_range[0] if score_range[0] is not None else None

    # If too many top-ranked rationales are redacted or have post-resolution, don't select the question.
    top_exact_redacted = sum(is_exact_redacted(reasoning_text(row)) for row in top_rows)
    top_post_resolution = sum(
        is_post_resolution(reasoning_text(row)) for row in top_rows
    )

    eligible = (
        len(rows) >= MIN_RATIONALE_COUNT
        and len(unique_ge_min) >= MIN_UNIQUE_COUNT
        and top_exact_redacted < MOSTLY_THRESHOLD
        and top_post_resolution < MOSTLY_THRESHOLD
    )

    return {
        "question_id": question_id,
        "source": rows[0].get("source"),
        "question": question_text,
        "rationale_count": len(rows),
        "non_redacted_count": len(non_redacted),
        "unique_non_redacted_score_ge_threshold_count": len(unique_ge_min),
        "score_range": score_range,
        "score_span": score_span,
        "score_mean": round(sum(scores) / len(scores), 3) if scores else None,
        "eligible": eligible,
        "eligibility_checks": {
            "enough_rationales": len(rows) >= MIN_RATIONALE_COUNT,
            "enough_scoreable_unique": len(unique_ge_min) >= MIN_UNIQUE_COUNT,
            "top_not_mostly_redacted": top_exact_redacted < MOSTLY_THRESHOLD,
            "top_not_mostly_post_resolution": top_post_resolution < MOSTLY_THRESHOLD,
        },
    }


def export_top_questions(
    output_path: Path,
    score_data: dict,
    eligible_summaries: list[dict],
) -> None:
    """Writes the top 5 eligible questions with their scored rationales to the output file."""
    selected = eligible_summaries[:5]
    selected_ids = {q["question_id"] for q in selected}
    rows_by_question: dict[str, list[dict]] = {}
    for row in score_data.get("scores", []):
        if row.get("question_id") in selected_ids:
            rows_by_question.setdefault(row["question_id"], []).append(row)

    exported_questions = []
    for question in selected:
        question_rows = sorted(
            [
                row
                for row in rows_by_question.get(question["question_id"], [])
                if isinstance(row.get("overall_score"), int)
                and row["overall_score"] >= MIN_SCORE
            ],
            key=lambda row: (
                -(row.get("overall_score") or -999),
                -sum((row.get("scores") or {}).values()),
                row.get("row_number") or 0,
            ),
        )
        exported_questions.append(
            {
                "question_id": question["question_id"],
                "source": question.get("source"),
                "question": question.get("question"),
                "flash_rationale_count": len(question_rows),
                "flash_rationales": [
                    {
                        "row_number": row.get("row_number"),
                        "forecast_key": row.get("forecast_key"),
                        "user_id": row.get("user_id"),
                        "forecast": row.get("forecast"),
                        "reasoning": row.get("reasoning"),
                        "source": row.get("source"),
                        "question_id": row.get("question_id"),
                        "question": row.get("question"),
                        "flash_evaluation": {
                            "model": row.get("model"),
                            "scores": row.get("scores"),
                            "overall_score": row.get("overall_score"),
                            "summary": row.get("summary"),
                            "main_strengths": row.get("main_strengths"),
                            "main_weaknesses": row.get("main_weaknesses"),
                        },
                    }
                    for row in question_rows
                ],
            }
        )

    write_json(output_path, {"questions": exported_questions})


def select_questions(scores_path: Path, output_path: Path) -> None:
    score_data = load_json(scores_path)
    rows = [
        row
        for row in score_data.get("scores", [])
        if row.get("is_valid") is True and row.get("question_id")
    ]

    by_question: dict[str, list[dict]] = {}
    for row in rows:
        by_question.setdefault(row["question_id"], []).append(row)

    summaries = [question_summary(qid, qrows) for qid, qrows in by_question.items()]
    summaries.sort(
        key=lambda item: (
            not item["eligible"],
            -(item["unique_non_redacted_score_ge_threshold_count"]),
            -(item["score_mean"] or -999),
            -(item["score_span"] or -999),
            -(item["rationale_count"]),
            item["question_id"],
        )
    )

    eligible = [item for item in summaries if item["eligible"]]
    export_top_questions(output_path, score_data, eligible)
    print(f"Wrote top 5 questions to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select top questions for Pro model analysis."
    )
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    select_questions(args.scores, args.output)


if __name__ == "__main__":
    main()
