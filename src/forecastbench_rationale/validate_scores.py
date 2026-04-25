"""Validate Flash scoring quality by comparing it against the Pro model on a stratified sample.

Step 4 of the pipeline. Draws a 200 row stratified sample from the Flash scores,
covering redacted rationales, post-resolution language, and the full score range.
Once that sample has been scored by Pro (via score_rationales.py), pass --pro-scores
to calculate Pearson correlations, mean score differences, and a list
of the largest disagreements for manual review.
"""

import argparse
import math
import random
import re
from collections import Counter
from pathlib import Path
from statistics import mean

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
DEFAULT_SAMPLE_OUTPUT = Path("workspace/outputs/rationale_audit_sample_200.json")
DEFAULT_COMPARISON_OUTPUT = Path(
    "workspace/outputs/rationale_audit_model_comparison.json"
)

RUBRIC_SCORE_KEYS = [
    "relevance",
    "evidence_use",
    "base_rates",
    "current_drivers",
    "calibration",
    "counterarguments",
    "quantitative_reasoning",
    "clarity",
]

# Hardcoded seed for reproducibility.
SAMPLE_SEED = 254

# Stratified bucket targets, total = 200.
# Oversamples known problem cases, but the sampler will fall back to the
# remaining pool if a bucket doesn't have enough candidates.
BUCKET_SIZES = {
    "exact_redacted": 30,
    "partial_redacted": 20,
    "post_resolution_or_confirmation": 20,
    "flash_top_non_redacted": 35,
    "flash_mid_non_redacted": 45,
    "flash_low_non_redacted": 35,
    "random_remaining": 15,
}


def word_count_of(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text))


def select_sample(scores_path: Path, output_path: Path) -> None:
    """Draws a 200 row stratified sample from Flash scores, covering different score ranges and known problem cases like redacted rationales."""
    score_data = load_json(scores_path)
    rows = [
        row
        for row in score_data.get("scores", [])
        if row.get("is_valid") is True and isinstance(row.get("forecast_key"), str)
    ]

    # Calculate duplicate counts of each sampled rationale
    dup_counts = Counter(
        normalized_reasoning(reasoning_text(row))
        for row in rows
        if normalized_reasoning(reasoning_text(row))
    )

    rng = random.Random(SAMPLE_SEED)
    selected_keys: set[str] = set()

    # Each bucket predicate defines which rows are eligible for that bucket.
    # Buckets are sampled in order and rows already selected are excluded from later buckets.
    bucket_predicates = {
        "exact_redacted": lambda row: is_exact_redacted(reasoning_text(row)),
        "partial_redacted": lambda row: (
            is_any_redacted(reasoning_text(row))
            and not is_exact_redacted(reasoning_text(row))
        ),
        "post_resolution_or_confirmation": lambda row: (
            not is_any_redacted(reasoning_text(row))
            and is_post_resolution(reasoning_text(row))
        ),
        "flash_top_non_redacted": lambda row: (
            not is_any_redacted(reasoning_text(row))
            and not is_post_resolution(reasoning_text(row))
            and row.get("overall_score") in (4, 5)
        ),
        "flash_mid_non_redacted": lambda row: (
            not is_any_redacted(reasoning_text(row))
            and not is_post_resolution(reasoning_text(row))
            and row.get("overall_score") in (2, 3)
        ),
        "flash_low_non_redacted": lambda row: (
            not is_any_redacted(reasoning_text(row))
            and not is_post_resolution(reasoning_text(row))
            and row.get("overall_score") in (0, 1)
        ),
        "random_remaining": lambda row: True,
    }

    forecasts = []
    bucket_availability = {}
    bucket_selected = {}
    for bucket_name, target_size in BUCKET_SIZES.items():
        candidates = [
            row
            for row in rows
            if row.get("forecast_key") not in selected_keys
            and bucket_predicates[bucket_name](row)
        ]
        bucket_availability[bucket_name] = len(candidates)
        take = min(len(candidates), target_size)
        bucket_selected[bucket_name] = take
        sampled = rng.sample(candidates, take) if take else []
        for row in sampled:
            selected_keys.add(row["forecast_key"])
            text = reasoning_text(row)
            norm = normalized_reasoning(text)
            forecasts.append(
                {
                    "source": row["source"],
                    "id": row["question_id"],
                    "user_id": row.get("user_id"),
                    "forecast": row.get("forecast"),
                    "reasoning": row.get("reasoning"),
                    "audit_bucket": bucket_name,
                    "audit_original_row_number": row.get("row_number"),
                    "audit_original_forecast_key": row.get("forecast_key"),
                    "audit_flash_model": row.get("model"),
                    "audit_flash_overall_score": row.get("overall_score"),
                    "audit_flash_dimension_scores": row.get("scores"),
                    "audit_flags": {
                        "exact_redacted": is_exact_redacted(text),
                        "any_redacted": is_any_redacted(text),
                        "post_resolution_or_confirmation": is_post_resolution(text),
                        "word_count": word_count_of(text),
                        "duplicate_rationale_count": dup_counts.get(norm, 0),
                    },
                }
            )

    remaining_needed = sum(BUCKET_SIZES.values()) - len(forecasts)
    if remaining_needed > 0:
        remaining_candidates = [
            row for row in rows if row.get("forecast_key") not in selected_keys
        ]
        if len(remaining_candidates) < remaining_needed:
            raise ValueError(
                f"Need {remaining_needed} additional rows to fill the audit sample, "
                f"but only {len(remaining_candidates)} remain."
            )
        for row in rng.sample(remaining_candidates, remaining_needed):
            selected_keys.add(row["forecast_key"])
            text = reasoning_text(row)
            norm = normalized_reasoning(text)
            forecasts.append(
                {
                    "source": row["source"],
                    "id": row["question_id"],
                    "user_id": row.get("user_id"),
                    "forecast": row.get("forecast"),
                    "reasoning": row.get("reasoning"),
                    "audit_bucket": "random_remaining",
                    "audit_original_row_number": row.get("row_number"),
                    "audit_original_forecast_key": row.get("forecast_key"),
                    "audit_flash_model": row.get("model"),
                    "audit_flash_overall_score": row.get("overall_score"),
                    "audit_flash_dimension_scores": row.get("scores"),
                    "audit_flags": {
                        "exact_redacted": is_exact_redacted(text),
                        "any_redacted": is_any_redacted(text),
                        "post_resolution_or_confirmation": is_post_resolution(text),
                        "word_count": word_count_of(text),
                        "duplicate_rationale_count": dup_counts.get(norm, 0),
                    },
                }
            )

    bucket_selected["random_remaining"] = sum(
        1
        for forecast in forecasts
        if forecast.get("audit_bucket") == "random_remaining"
    )

    write_json(
        output_path,
        {
            "metadata": {
                "scores_path": str(scores_path),
                "source_model": score_data.get("model"),
                "sample_seed": SAMPLE_SEED,
                "bucket_targets": BUCKET_SIZES,
                "bucket_availability": bucket_availability,
                "bucket_selected": bucket_selected,
                "sample_size": len(forecasts),
            },
            "forecasts": forecasts,
        },
    )
    print(f"Sample: wrote {len(forecasts)} rows to {output_path}")


def pearson(xs: list[float], ys: list[float]) -> float | None:
    """Calculates Pearson r for two lists of numbers. Returns None if the result is undefined."""
    if len(xs) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    xd = [x - mx for x in xs]
    yd = [y - my for y in ys]
    denom = math.sqrt(sum(x**2 for x in xd) * sum(y**2 for y in yd))
    if denom == 0:
        return None
    return round(sum(x * y for x, y in zip(xd, yd)) / denom, 4)


def compare_flash_pro(
    sample_path: Path, pro_scores_path: Path, output_path: Path
) -> None:
    """Matches Flash and Pro scores for each row in the sample and calculates agreements across the rubric dimensions."""
    sample_data = load_json(sample_path)
    pro_data = load_json(pro_scores_path)

    # Index Pro scores by content so we can match to Flash without relying on row numbers.
    pro_index = {
        (
            row.get("source"),
            row.get("question_id"),
            row.get("user_id"),
            row.get("forecast"),
            row.get("reasoning"),
        ): row
        for row in pro_data.get("scores", [])
    }

    rows = []
    for forecast in sample_data.get("forecasts", []):
        key = (
            forecast.get("source"),
            forecast.get("id"),
            forecast.get("user_id"),
            forecast.get("forecast"),
            forecast.get("reasoning"),
        )
        pro_row = pro_index.get(key)
        if pro_row is None:
            raise ValueError(
                f"No Pro score found for {forecast.get('source')}/{forecast.get('id')} "
                f"user={forecast.get('user_id')!r}"
            )

        flash_score = forecast.get("audit_flash_overall_score")
        pro_score = pro_row.get("overall_score")
        flash_dims = forecast.get("audit_flash_dimension_scores") or {}
        pro_dims = pro_row.get("scores") or {}
        flags = forecast.get("audit_flags") or {}

        row = {
            "source": forecast.get("source"),
            "question_id": forecast.get("id"),
            "audit_bucket": forecast.get("audit_bucket"),
            "flash_overall": flash_score,
            "pro_overall": pro_score,
            "diff": (
                pro_score - flash_score
                if isinstance(flash_score, int) and isinstance(pro_score, int)
                else None
            ),
            "word_count": flags.get("word_count"),
            "exact_redacted": flags.get("exact_redacted"),
            "post_resolution": flags.get("post_resolution_or_confirmation"),
            "reasoning_preview": (forecast.get("reasoning") or "")[:200],
            "pro_summary": pro_row.get("summary"),
        }
        for dim in RUBRIC_SCORE_KEYS:
            row[f"flash_{dim}"] = flash_dims.get(dim)
            row[f"pro_{dim}"] = pro_dims.get(dim)
        rows.append(row)

    # Filter to rows where both models returned a valid score.
    scored = [
        r
        for r in rows
        if isinstance(r["flash_overall"], int) and isinstance(r["pro_overall"], int)
    ]
    diffs = [r["diff"] for r in scored]
    flash_overall = [r["flash_overall"] for r in scored]
    pro_overall = [r["pro_overall"] for r in scored]

    overall_agreement = {
        "n": len(scored),
        "mean_flash": round(mean(flash_overall), 3),
        "mean_pro": round(mean(pro_overall), 3),
        "mean_diff_pro_minus_flash": round(mean(diffs), 3),
        "pearson_r": pearson(flash_overall, pro_overall),
        "exact_match_rate": round(sum(d == 0 for d in diffs) / len(diffs), 4),
        "within_one_rate": round(sum(abs(d) <= 1 for d in diffs) / len(diffs), 4),
        "diff_distribution": dict(sorted(Counter(diffs).items())),
    }

    # Calculate correlations for each rubric dimension.
    dimension_correlations = {}
    for dim in RUBRIC_SCORE_KEYS:
        paired = [
            (r[f"flash_{dim}"], r[f"pro_{dim}"])
            for r in scored
            if isinstance(r.get(f"flash_{dim}"), int)
            and isinstance(r.get(f"pro_{dim}"), int)
        ]
        if paired:
            flash_vals, pro_vals = zip(*paired)
            diffs_dim = [p - f for f, p in paired]
            dimension_correlations[dim] = {
                "pearson_r": pearson(list(flash_vals), list(pro_vals)),
                "mean_diff_pro_minus_flash": round(mean(diffs_dim), 3),
            }
        else:
            dimension_correlations[dim] = {
                "pearson_r": None,
                "mean_diff_pro_minus_flash": None,
            }

    # Identify largest disagreements.
    large_disagreements = sorted(
        [r for r in scored if abs(r["diff"]) >= 2],
        key=lambda r: (-abs(r["diff"]), r.get("source") or ""),
    )[:15]

    write_json(
        output_path,
        {
            "flash_model": sample_data.get("metadata", {}).get("source_model"),
            "pro_model": pro_data.get("model"),
            "overall_agreement": overall_agreement,
            "dimension_correlations": dimension_correlations,
            "large_disagreements_diff_ge_2": large_disagreements,
        },
    )
    print(
        f"Comparison: {len(scored)} rows, Pearson r = {overall_agreement['pearson_r']}. "
        f"Written to {output_path}"
    )


def validate_scores(
    scores_path: Path,
    sample_output: Path,
    pro_scores_path: Path | None,
    comparison_output: Path,
) -> None:
    """Draws the sample and runs the Flash vs Pro comparison if Pro scores are provided."""
    print("Creating audit sample")
    select_sample(scores_path, sample_output)

    if pro_scores_path is not None:
        print("Comparing Flash vs Pro scores")
        compare_flash_pro(sample_output, pro_scores_path, comparison_output)
    else:
        print(
            "Skipping comparison (no --pro-scores). "
            "Score the sample with score_rationales.py, then rerun with --pro-scores."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a validation sample and compare Flash vs Pro scoring."
    )
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--sample-output", type=Path, default=DEFAULT_SAMPLE_OUTPUT)
    parser.add_argument(
        "--pro-scores",
        type=Path,
        default=None,
        help="Pro model scores file. If provided, runs the Flash vs Pro comparison.",
    )
    parser.add_argument(
        "--comparison-output", type=Path, default=DEFAULT_COMPARISON_OUTPUT
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_scores(
        scores_path=args.scores,
        sample_output=args.sample_output,
        pro_scores_path=args.pro_scores,
        comparison_output=args.comparison_output,
    )


if __name__ == "__main__":
    main()
