"""Rank top 3 rationales per question and generate an overall synthesis summary.

Step 3 of the pipeline. Reads five_selected_questions.json (output from select_questions.py)
and runs two Pro model tasks:
  1. Per-question top 3 ranking - one call per question, results are written incrementally
     so a quota failure mid-run doesn't lose completed questions.
  2. Cross-question summary - one call total.
"""

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from .llm_utils import (
    CompletionExhaustedError,
    TerminalQuotaError,
    load_json,
    run_validated_json_completion,
    write_json,
)
from .prompts import build_overall_summary_prompt, build_question_top_3_ranking_prompt


DEFAULT_INPUT = Path("workspace/outputs/five_selected_questions.json")
DEFAULT_TOP3_OUTPUT = Path("workspace/outputs/five_selected_questions_top_3.json")
DEFAULT_SUMMARY_OUTPUT = Path(
    "workspace/outputs/five_selected_questions_overall_summary.json"
)
DEFAULT_MODEL = "gemini/gemini-3.1-pro-preview"


def validate_ranking(response: dict, valid_row_numbers: set[int]) -> dict:
    """Verifies the model returned exactly 3 ranked items pointing to valid unique row numbers from the input."""
    if not isinstance(response, dict):
        raise ValueError("response must be a JSON object")

    top_3 = response.get("top_3")
    if not isinstance(top_3, list) or len(top_3) != 3:
        raise ValueError("top_3 must be a list of exactly 3 items")

    seen_ranks: set[int] = set()
    seen_row_numbers: set[int] = set()
    normalized: list[dict] = []

    for item in top_3:
        if not isinstance(item, dict):
            raise ValueError("each top_3 item must be an object")

        rank = item.get("rank")
        if not isinstance(rank, int) or rank not in (1, 2, 3):
            raise ValueError("top_3.rank must be 1, 2, or 3")
        if rank in seen_ranks:
            raise ValueError("top_3 ranks must be unique")
        seen_ranks.add(rank)

        # Validate row_number to be able to link back to original rationale.
        row_number = item.get("row_number")
        if not isinstance(row_number, int) or row_number not in valid_row_numbers:
            raise ValueError(
                f"top_3.row_number {row_number!r} is not in the provided rationales"
            )
        if row_number in seen_row_numbers:
            raise ValueError("top_3 row numbers must be unique")
        seen_row_numbers.add(row_number)

        normalized.append(
            {
                "rank": rank,
                "row_number": row_number,
                "forecast": item.get("forecast"),
                "reasoning": (item.get("reasoning") or "").strip(),
                "why_selected": (item.get("why_selected") or "").strip(),
                "strengths": item.get("strengths") or [],
                "weaknesses": item.get("weaknesses") or [],
            }
        )

    normalized.sort(key=lambda item: item["rank"])
    return {
        "question": (response.get("question") or "").strip(),
        "top_3": normalized,
        "selection_notes": (response.get("selection_notes") or "").strip(),
    }


def rank_question(*, model: str, question: dict, max_retries: int) -> dict:
    """Asks the Pro model to pick and explain the top 3 rationales for a single question."""
    prompt = build_question_top_3_ranking_prompt(
        question_text=question["question"],
        rationales=question["flash_rationales"],
    )
    valid_row_numbers = {
        r["row_number"]
        for r in question["flash_rationales"]
        if isinstance(r.get("row_number"), int)
    }
    validator = lambda resp: validate_ranking(resp, valid_row_numbers)

    metadata = {
        "model": model,
        "question_id": question["question_id"],
        "source": question["source"],
        "question": question["question"],
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        result, raw_response, attempts = run_validated_json_completion(
            model=model,
            prompt=prompt,
            validator=validator,
            max_retries=max_retries,
        )
        return {
            **result,
            **metadata,
            "attempts": attempts,
            "is_valid": True,
            "validation_status": "valid",
            "raw_model_response": raw_response,
        }
    except CompletionExhaustedError as error:
        return {
            **metadata,
            "attempts": error.attempts,
            "is_valid": False,
            "validation_status": "failed",
            "error_type": (
                type(error.last_error).__name__ if error.last_error else "UnknownError"
            ),
            "error_message": (
                str(error.last_error) if error.last_error else "Unknown error"
            ),
            "raw_model_response": error.raw_response,
        }


def write_rankings(
    output_path: Path, input_path: Path, model: str, rankings: list[dict]
) -> None:
    valid = sum(1 for r in rankings if r.get("is_valid") is True)
    failed = sum(1 for r in rankings if r.get("is_valid") is False)
    write_json(
        output_path,
        {
            "input_path": str(input_path),
            "model": model,
            "question_count": len(rankings),
            "valid_question_count": valid,
            "failed_question_count": failed,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "questions": rankings,
        },
    )


def generate_summary(questions: list[dict], model: str, max_retries: int) -> dict:
    """Asks the Pro model for an overall synthesis of the strongest rationales across all 5 questions."""
    prompt = build_overall_summary_prompt(questions)
    metadata = {
        "model": model,
        "summarized_at": datetime.now(timezone.utc).isoformat(),
    }
    try:

        def validate_summary(response: dict) -> dict:
            if not isinstance(response, dict):
                raise ValueError("response must be a JSON object")
            s = response.get("overall_summary")
            if not isinstance(s, str) or not s.strip():
                raise ValueError("overall_summary must be a non-empty string")
            return {"overall_summary": s.strip()}

        result, raw_response, attempts = run_validated_json_completion(
            model=model,
            prompt=prompt,
            validator=validate_summary,
            max_retries=max_retries,
        )
        return {
            **result,
            **metadata,
            "attempts": attempts,
            "is_valid": True,
            "validation_status": "valid",
            "raw_model_response": raw_response,
        }
    except CompletionExhaustedError as error:
        return {
            **metadata,
            "attempts": error.attempts,
            "is_valid": False,
            "validation_status": "failed",
            "error_type": (
                type(error.last_error).__name__ if error.last_error else "UnknownError"
            ),
            "error_message": (
                str(error.last_error) if error.last_error else "Unknown error"
            ),
            "raw_model_response": error.raw_response,
        }


def analyze_questions(
    input_path: Path,
    top3_output: Path,
    summary_output: Path,
    model: str,
    dry_run: bool,
    max_retries: int,
) -> None:
    """Runs top 3 ranking for each question then generates a cross-question synthesis summary."""
    load_dotenv()
    if max_retries < 0:
        raise ValueError("--max-retries must be non-negative.")

    input_data = load_json(input_path)
    questions = input_data.get("questions", [])
    if not isinstance(questions, list) or not questions:
        raise ValueError("input file must contain a non-empty 'questions' list")

    if dry_run:
        first = questions[0]
        print(
            build_question_top_3_ranking_prompt(
                question_text=first["question"],
                rationales=first["flash_rationales"],
            )
        )
        return

    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY must be set in the environment.")

    # First, get the top 3 rankings per question.
    rankings = []
    try:
        for question in questions:
            ranking = rank_question(
                model=model, question=question, max_retries=max_retries
            )
            rankings.append(ranking)
            if not ranking.get("is_valid"):
                print(
                    f"Failed to rank {question['source']}/{question['question_id']}: "
                    f"{ranking.get('error_message')}"
                )
    except TerminalQuotaError as error:
        print(f"Quota limit hit after {len(rankings)}/{len(questions)} questions.")
        raise SystemExit(2) from error
    finally:
        if rankings:
            write_rankings(top3_output, input_path, model, rankings)

    valid = sum(1 for r in rankings if r.get("is_valid") is True)
    print(f"Top 3 ranking complete: {valid}/{len(questions)} succeeded.")

    # Second, generate the cross-question summary
    print("Generating overall summary")
    try:
        summary = generate_summary(questions, model, max_retries)
    except TerminalQuotaError as error:
        print(f"Quota error on summary: {error}")
        raise SystemExit(2) from error

    write_json(
        summary_output,
        {
            **summary,
            "input_path": str(input_path),
            "question_count": len(questions),
        },
    )
    status = "succeeded" if summary.get("is_valid") else "failed"
    print(f"Overall summary {status}. Written to {summary_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank top 3 rationales per question and generate an overall summary."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--top3-output", type=Path, default=DEFAULT_TOP3_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first ranking prompt without making an API call.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Retry malformed model responses this many times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze_questions(
        input_path=args.input,
        top3_output=args.top3_output,
        summary_output=args.summary_output,
        model=args.model,
        dry_run=args.dry_run,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()
