#!/usr/bin/env bash
# Run the full rationale scoring and analysis pipeline.
# See README.md for setup instructions and required environment variables.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
FLASH_MODEL="${FLASH_MODEL:-gemini/gemini-3-flash-preview}"
PRO_MODEL="${PRO_MODEL:-gemini/gemini-3.1-pro-preview}"
CLEAN=0
SKIP_INSTALL=0

usage() {
  cat <<'EOF'
Usage: bash run_pipeline.sh [--clean] [--skip-install]

Options:
  --clean         Remove existing outputs before running.
  --skip-install  Reuse the existing virtual environment without reinstalling.

Environment variables:
  GEMINI_API_KEY  Required (or set in .env).
  FLASH_MODEL     Override the Flash scoring model.
  PRO_MODEL       Override the Pro model used for analysis and audit rescoring.
  VENV_DIR        Override the virtual environment path.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean) CLEAN=1; shift ;;
    --skip-install) SKIP_INSTALL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ ! -d "$VENV_DIR" ]]; then
  python3.12 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python3 -m pip install --upgrade pip

if [[ $SKIP_INSTALL -eq 0 ]]; then
  python3 -m pip install -e "$ROOT_DIR"
fi

mkdir -p "$ROOT_DIR/workspace/outputs"

if [[ $CLEAN -eq 1 ]]; then
  rm -rf "$ROOT_DIR/workspace/outputs"
  mkdir -p "$ROOT_DIR/workspace/outputs"
fi

if [[ -z "${GEMINI_API_KEY:-}" && ! -f "$ROOT_DIR/.env" ]]; then
  echo "Error: GEMINI_API_KEY is not set and .env was not found." >&2
  exit 1
fi

OUTPUTS="$ROOT_DIR/workspace/outputs"

step() { echo; echo "==> $1"; shift; "$@"; }

step "Step 1: Score all rationales with Flash" \
  score-rationales --model "$FLASH_MODEL"

step "Step 2: Select top 5 questions" \
  select-questions

step "Step 3: Rank top-3 rationales and generate synthesis" \
  analyze-questions --model "$PRO_MODEL"

# Audit phases 1 and 2 (no API calls needed)
step "Step 4a: Audit Flash scores and create validation sample" \
  validate-scores

# Score the 200-row sample with Pro for model comparison
step "Step 4b: Score validation sample with Pro" \
  score-rationales \
    --forecasts "$OUTPUTS/rationale_audit_sample_200.json" \
    --questions "$ROOT_DIR/workspace/forecastbench_data/2024-07-21-human.json" \
    --output "$OUTPUTS/rationale_audit_sample_200_pro_scores.json" \
    --model "$PRO_MODEL"

# Audit phase 3: Flash vs Pro comparison
step "Step 4c: Compare Flash and Pro scores" \
  validate-scores \
    --pro-scores "$OUTPUTS/rationale_audit_sample_200_pro_scores.json"

echo
echo "Pipeline complete. Outputs are in $OUTPUTS"
