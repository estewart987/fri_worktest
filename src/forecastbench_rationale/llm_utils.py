"""LiteLLM wrapper and JSON helpers shared across pipeline scripts."""

import json
import os
import re
import time
from pathlib import Path
from typing import Callable

# Must be set before importing litellm to avoid a network call for model pricing.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from litellm import completion


# These strings indicate the quota is met for the day/billing period
# and there's no point retrying, so we should save progress and exit cleanly.
TERMINAL_QUOTA_PATTERNS = (
    "daily",
    "quota exceeded",
    "resource exhausted",
    "billing",
    "insufficient quota",
    "exceeded your current quota",
    "free tier",
    "paid tier",
    "generate requests per day",
    "requests per day",
)

# These indicate a transient rate limit, like per-minute, that's worth retrying after a pause.
TRANSIENT_QUOTA_PATTERNS = (
    "per minute",
    "requests per minute",
    "rate limit",
    "rate_limit",
    "rate limits",
    "too many requests",
    "429",
)


class TerminalQuotaError(RuntimeError):
    """Raised when the API quota is met and the run should stop."""


class CompletionExhaustedError(RuntimeError):
    """Raised when retries run out without getting a valid structured response."""

    def __init__(self, last_error: Exception | None, raw_response: str, attempts: int):
        super().__init__(str(last_error) if last_error else "Unknown completion error")
        self.last_error = last_error
        self.raw_response = raw_response
        self.attempts = attempts


REDACTION_TEXT = "[redacted to maintain anonymity]"

# Matches rationale text that was clearly written after the question resolved.
# Used in both question selection and audit sampling.
POST_RESOLUTION_PATTERN = re.compile(
    r"(?:already resolved|question resolved|it just happened|just happened|"
    r"officially confirmed|already confirmed|happened today|"
    r"resolved as (?:yes|no|a yes|a no)|already resolved as|updating resolved|"
    r"updating based on resolution|(?:updating|updated).{0,20}(?:yes|no) resolution|"
    r"has (?:already )?resolved)",
    re.IGNORECASE,
)


def reasoning_text(row: dict) -> str:
    """Pulls the reasoning field out of a row, returning an empty string if it's missing."""
    value = row.get("reasoning")
    return value if isinstance(value, str) else ""


def is_exact_redacted(text: str) -> bool:
    return text.strip().lower() == REDACTION_TEXT.lower()


def is_any_redacted(text: str) -> bool:
    return REDACTION_TEXT.lower() in text.lower()


def is_post_resolution(text: str) -> bool:
    """True if the text looks like it was written after the question already resolved."""
    return bool(POST_RESOLUTION_PATTERN.search(text))


def normalized_reasoning(text: str) -> str:
    """Lowercased with white space removed version of text, used to identify duplicate rationales."""
    return re.sub(r"\s+", " ", text.strip().lower())


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    """Writes data as JSON and creates parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def parse_json_response(text: str) -> dict:
    """Parses JSON from the model response, stripping markdown code fences if present."""
    if not isinstance(text, str):
        raise ValueError("model response content must be a string")

    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
        if text.startswith("json"):
            text = text[4:].strip()
    return json.loads(text)


def repair_prompt(error_message: str) -> str:
    """Returns a follow-up message telling the model what was wrong with its previous JSON response."""
    return (
        "Your previous response was not valid for the required schema. "
        f"Validation error: {error_message}\n\n"
        "Return only the corrected JSON object. Do not include Markdown, prose, "
        "or any fields outside the requested schema."
    )


def is_terminal_quota_error(error: Exception) -> bool:
    """Returns True if daily API quota is met and retrying won't help."""
    message = str(error).lower()
    if not any(pattern in message for pattern in TERMINAL_QUOTA_PATTERNS):
        return False
    return not any(pattern in message for pattern in TRANSIENT_QUOTA_PATTERNS)


def completion_content(
    model: str,
    messages: list[dict],
    *,
    reasoning_effort: str | None = None,
) -> str:
    """Calls LiteLLM and returns the text content of the model response."""
    completion_kwargs = {"model": model, "messages": messages}
    if reasoning_effort is not None:
        completion_kwargs["reasoning_effort"] = reasoning_effort
    response = completion(**completion_kwargs)
    return response.choices[0].message.content  # type: ignore


def run_validated_json_completion(
    *,
    model: str,
    prompt: str,
    validator: Callable[[dict], dict],
    max_retries: int,
    reasoning_effort: str | None = None,
) -> tuple[dict, str, int]:
    """Sends a prompt and retries with feedback if the model response doesn't pass validation."""
    messages = [{"role": "user", "content": prompt}]
    raw_response = ""
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 2):
        try:
            raw_response = completion_content(
                model=model,
                messages=messages,
                reasoning_effort=reasoning_effort,
            )
            parsed = parse_json_response(raw_response)
            validated = validator(parsed)
            return validated, raw_response, attempt
        except (json.JSONDecodeError, ValueError) as error:
            last_error = error
            # Show the model its bad response and ask for a corrected version.
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": raw_response},
                {"role": "user", "content": repair_prompt(str(error))},
            ]
        except Exception as error:
            if is_terminal_quota_error(error):
                raise TerminalQuotaError(str(error)) from error
            last_error = error
            if attempt <= max_retries:
                time.sleep(min(2**attempt, 30))

    raise CompletionExhaustedError(last_error, raw_response, max_retries + 1)
