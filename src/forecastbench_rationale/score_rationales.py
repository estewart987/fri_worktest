"""Score ForecastBench rationales with a Gemini model.

Step 1 of the pipeline. Reads the forecasts file, passes every rationale
to an LLM which scores it against the 8-dimension rubric, and writes results to rationale_scores.json.
Writes after every row so a failure preserves all completed scores.
"""

import argparse
import hashlib
import json
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
from .prompts import build_rationale_scoring_prompt


DEFAULT_FORECASTS = Path(
    "workspace/forecastbench_data/2024-07-21.ForecastBench.human_super_individual.json"
)
DEFAULT_QUESTIONS = Path("workspace/forecastbench_data/2024-07-21-human.json")
DEFAULT_OUTPUT = Path("workspace/outputs/rationale_scores.json")
PROGRESS_EVERY = 100
RUBRIC_SCORE_KEYS = {
    "relevance",
    "evidence_use",
    "base_rates",
    "current_drivers",
    "calibration",
    "counterarguments",
    "quantitative_reasoning",
    "clarity",
}


def question_key(row: dict) -> tuple[str, str]:
    return row["source"], row["id"]


def forecast_key(row_number: int, forecast: dict) -> str:
    """Creates a content hash for a forecast so we can tell which rows were already scored in a previous run."""
    fingerprint_input = json.dumps(
        {
            "source": forecast.get("source"),
            "id": forecast.get("id"),
            "user_id": forecast.get("user_id"),
            "forecast": forecast.get("forecast"),
            "reasoning": forecast.get("reasoning"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    fingerprint = hashlib.sha256(fingerprint_input.encode("utf-8")).hexdigest()[:16]
    return f"{row_number}:{forecast.get('source')}:{forecast.get('id')}:{fingerprint}"


def validate_forecast_questions(
    forecasts: list[dict], question_index: dict[tuple[str, str], dict]
) -> None:
    """Throws error if any forecast points to a question ID not in the question file."""
    missing = sorted(
        {
            question_key(row)
            for row in forecasts
            if question_key(row) not in question_index
        }
    )
    if not missing:
        return
    examples = ", ".join(f"{s}/{q}" for s, q in missing[:10])
    suffix = "" if len(missing) <= 10 else f", plus {len(missing) - 10} more"
    raise ValueError(
        f"Forecasts reference question IDs not in the question file: {examples}{suffix}"
    )


def validate_score(score: dict) -> dict:
    """Validate and normalize the model's output."""
    if not isinstance(score, dict):
        raise ValueError("response must be a JSON object")

    scores = score.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("scores must be a JSON object")

    score_keys = set(scores)
    if score_keys != RUBRIC_SCORE_KEYS:
        missing = sorted(RUBRIC_SCORE_KEYS - score_keys)
        extra = sorted(score_keys - RUBRIC_SCORE_KEYS)
        raise ValueError(f"scores keys mismatch; missing={missing}, extra={extra}")

    for key, value in scores.items():
        if not isinstance(value, int) or not 0 <= value <= 5:
            raise ValueError(f"scores.{key} must be an integer from 0 to 5")

    overall_score = score.get("overall_score")
    if not isinstance(overall_score, int) or not 0 <= overall_score <= 5:
        raise ValueError("overall_score must be an integer from 0 to 5")

    summary = score.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("summary must be a non-empty string")

    for field in ("main_strengths", "main_weaknesses"):
        values = score.get(field)
        if not isinstance(values, list) or not values:
            raise ValueError(f"{field} must be a non-empty list")
        if not all(isinstance(value, str) and value.strip() for value in values):
            raise ValueError(f"{field} must contain only non-empty strings")

    return {
        "scores": scores,
        "overall_score": overall_score,
        "summary": summary.strip(),
        "main_strengths": [value.strip() for value in score["main_strengths"]],
        "main_weaknesses": [value.strip() for value in score["main_weaknesses"]],
    }


def score_metadata(
    row_number: int, forecast: dict, question: dict, model: str, attempts: int
) -> dict:
    return {
        "forecast_key": forecast_key(row_number, forecast),
        "model": model,
        "row_number": row_number,
        "source": forecast["source"],
        "question_id": forecast["id"],
        "question": question.get("question"),
        "user_id": forecast.get("user_id"),
        "forecast": forecast.get("forecast"),
        "reasoning": forecast.get("reasoning"),
        "attempts": attempts,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }


def score_forecast(
    model: str,
    question: dict,
    forecast: dict,
    row_number: int,
    max_retries: int,
) -> dict:
    """Scores one rationale by calling the model. Returns a failure record if all retries are used up."""
    prompt = build_rationale_scoring_prompt(question, forecast)
    try:
        # Make the API call and validate the response, retrying with feedback if it fails validation.
        score, raw_response, attempts = run_validated_json_completion(
            model=model,
            prompt=prompt,
            validator=validate_score,
            max_retries=max_retries,
            reasoning_effort="low",
        )
        score.update(
            score_metadata(
                row_number=row_number,
                forecast=forecast,
                question=question,
                model=model,
                attempts=attempts,
            )
        )
        score.update(
            {
                "is_valid": True,
                "validation_status": "valid",
                "raw_model_response": raw_response,
            }
        )
        return score
    except CompletionExhaustedError as error:
        # Keep failed rows in the output so the run is auditable and resumable.
        failure = score_metadata(
            row_number=row_number,
            forecast=forecast,
            question=question,
            model=model,
            attempts=error.attempts,
        )
        failure.update(
            {
                "is_valid": False,
                "validation_status": "failed",
                "error_type": (
                    type(error.last_error).__name__
                    if error.last_error
                    else "UnknownError"
                ),
                "error_message": (
                    str(error.last_error)
                    if error.last_error
                    else "Unknown scoring error"
                ),
                "raw_model_response": error.raw_response,
            }
        )
        return failure


def load_resumable_scores(
    output_path: Path, model: str, eligible_keys: set[str]
) -> list[dict]:
    """Loads valid scores from a previous run that are included in the current forecast set."""
    if not output_path.exists():
        return []

    existing_output = load_json(output_path)
    scores = existing_output.get("scores", [])
    return [
        score
        for score in scores
        if score.get("is_valid") is True
        and score.get("model") == model
        and score.get("forecast_key") in eligible_keys
    ]


def write_scores(
    output_path: Path,
    model: str,
    forecasts_path: Path,
    questions_path: Path,
    forecast_count: int,
    scores: list[dict],
) -> None:
    valid_count = sum(1 for score in scores if score.get("is_valid") is True)
    failed_count = sum(1 for score in scores if score.get("is_valid") is False)
    write_json(
        output_path,
        {
            "model": model,
            "forecasts_path": str(forecasts_path),
            "questions_path": str(questions_path),
            "forecast_count": forecast_count,
            "score_count": len(scores),
            "valid_score_count": valid_count,
            "failed_score_count": failed_count,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "scores": scores,
        },
    )


def score_rationales(
    forecasts_path: Path,
    questions_path: Path,
    output_path: Path,
    model: str,
    dry_run: bool,
    resume: bool,
    max_retries: int,
) -> None:
    """Loads the forecasts, scores each one against the rubric, and saves results after every row so a failure doesn't lose progress."""
    load_dotenv()
    if max_retries < 0:
        raise ValueError("--max-retries must be non-negative.")
    # Load forecasts and questions, validate, and build an index for matching forecasts to questions.
    forecast_data = load_json(forecasts_path)
    question_index = {
        question_key(row): row for row in load_json(questions_path)["questions"]
    }
    all_forecasts = forecast_data["forecasts"]
    validate_forecast_questions(all_forecasts, question_index)
    rows = list(enumerate(all_forecasts, start=1))

    if dry_run:
        _, first_forecast = rows[0]
        print(
            build_rationale_scoring_prompt(
                question_index[question_key(first_forecast)], first_forecast
            )
        )
        return

    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY must be set in the environment.")

    # Build a set of stable keys for all forecasts so resume logic can
    # identify which forecasts were already scored in a previous run.
    eligible_keys = {
        forecast_key(row_number, forecast) for row_number, forecast in rows
    }
    scores = load_resumable_scores(output_path, model, eligible_keys) if resume else []
    completed_keys = {score["forecast_key"] for score in scores}
    total_rows = len(rows)
    print(
        f"Starting: {len(completed_keys)}/{total_rows} already scored, {total_rows - len(completed_keys)} remaining."
    )

    # Score each rationale, skipping previously completed ones.
    for row_number, forecast in rows:
        key = forecast_key(row_number, forecast)
        if key in completed_keys:
            continue

        question = question_index[question_key(forecast)]
        try:
            score = score_forecast(
                model=model,
                question=question,
                forecast=forecast,
                row_number=row_number,
                max_retries=max_retries,
            )
        except TerminalQuotaError as error:
            write_scores(
                output_path,
                model,
                forecasts_path,
                questions_path,
                len(all_forecasts),
                scores,
            )
            print(f"Quota limit hit; saved {len(scores)} scores.")
            raise SystemExit(2) from error

        scores.append(score)
        completed_keys.add(key)

        if score.get("is_valid") is False:
            print(
                f"Score failed for row {row_number} ({forecast.get('source')}/{forecast.get('id')}): {score.get('error_message')}"
            )

        # Write after every row so an error doesn't lose completed requests.
        write_scores(
            output_path,
            model,
            forecasts_path,
            questions_path,
            len(all_forecasts),
            scores,
        )
        # Print a progress update every PROGRESS_EVERY rows or at the end, showing valid vs failed counts.
        if (
            len(completed_keys) % PROGRESS_EVERY == 0
            or len(completed_keys) == total_rows
        ):
            valid = sum(1 for s in scores if s.get("is_valid") is True)
            print(
                f"Progress: {len(completed_keys)}/{total_rows} ({valid} valid, {len(scores) - valid} failed)"
            )

    valid = sum(1 for s in scores if s.get("is_valid") is True)
    print(
        f"Done: {len(scores)}/{total_rows} scored ({valid} valid, {len(scores) - valid} failed)"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score ForecastBench rationales with Gemini."
    )
    parser.add_argument("--forecasts", type=Path, default=DEFAULT_FORECASTS)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="gemini/gemini-3-flash-preview")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first prompt without calling the API.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing output and start from scratch.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Retry malformed responses this many times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score_rationales(
        forecasts_path=args.forecasts,
        questions_path=args.questions,
        output_path=args.output,
        model=args.model,
        dry_run=args.dry_run,
        resume=not args.no_resume,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()
