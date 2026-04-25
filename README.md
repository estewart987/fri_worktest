# ForecastBench Rationale Scoring

This repo scores ForecastBench forecasting rationales with an LLM, selects 5 questions for closer review, ranks the top 3 rationales per selected question, generates a cross-question synthesis summary, and audits the scoring behavior by comparing Flash and Pro on a stratified sample.

## Quick Start

Add your Gemini API key to the environment, then run the pipeline script to execute the full workflow:

```bash
export GEMINI_API_KEY="your_key_here"
bash run_pipeline.sh
```

## Setup

If you would rather run each step manually, follow the steps below. All commands provided are the exact commands I ran with the corresponding arguments.

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the package and its dependencies:

```bash
pip install -e .
```

Set your Gemini API key:

```bash
export GEMINI_API_KEY="your_key_here"
```

You can also place it in a `.env` file as `GEMINI_API_KEY=your_key_here`.

## Step 1. Score All Rationales

Runs every rationale in the dataset through a Flash model using an 8-dimension rubric, then assigns each an overall score from 0 to 5.

```bash
score-rationales
```

Writes:

```
workspace/outputs/rationale_scores.json
```

Optional arguments:
- `--model` changes the scoring model. Default is `gemini/gemini-3-flash-preview`.
- `--forecasts` scores a different forecasts file.
- `--questions` points to a different questions file for validation and prompt context.
- `--output` writes to a different output file.
- `--no-resume` starts fresh instead of continuing a previous run.
- `--max-retries` retries malformed model responses before recording a failed row.
- `--dry-run` prints the first scoring prompt without making an API call.

## Step 2. Select Questions for Analysis

Filters questions that have enough high-scoring, non-redacted, non-duplicate rationales to be worth closer analysis, and exports the top 5.

```bash
select-questions
```

Writes:

```
workspace/outputs/five_selected_questions.json
```

Optional arguments:
- `--scores` reads scores from a different input file.
- `--output` writes the selected questions to a different path.

## Step 3. Rank Top 3 Rationales and Generate Overall Summary

Uses a Pro model for two tasks:
1. Rank the top 3 rationales for each selected question.
2. Generate a cross-question summary of common patterns in the stronger rationales.

```bash
analyze-questions
```

Writes:

```
workspace/outputs/five_selected_questions_top_3.json
workspace/outputs/five_selected_questions_overall_summary.json
```

Optional arguments:
- `--input` points to a different selected-question file.
- `--top3-output` changes the ranked top-3 output path.
- `--summary-output` changes the overall summary output path.
- `--model` changes the analysis model. Default is `gemini/gemini-3.1-pro-preview`.
- `--max-retries` retries malformed model responses.
- `--dry-run` prints the first ranking prompt without making an API call.

## Step 4a. Create the Audit Sample

Draws a reproducible 200 rationale sample that oversamples known failure modes such as redaction and post-resolution language. If a bucket doesn't have enough candidates, the sampler takes as many rationales as are available and fills the remaining slots from the unselected pool.

```bash
validate-scores
```

Writes:

```
workspace/outputs/rationale_audit_sample_200.json
```

Optional arguments:
- `--scores` points to a different Flash scoring output.
- `--sample-output` changes the sample output path.

The sample metadata records each bucket's target, availability, and selected count.

If `--pro-scores` is not provided, this step only creates the sample and skips model comparison.

## Step 4b. Score the Audit Sample with Pro

Rescores the 200 rationale sample using a Pro model with the same rubric, allowing for a direct Flash vs. Pro comparison.

```bash
score-rationales \
  --forecasts workspace/outputs/rationale_audit_sample_200.json \
  --questions workspace/forecastbench_data/2024-07-21-human.json \
  --output workspace/outputs/rationale_audit_sample_200_pro_scores.json \
  --model gemini/gemini-3.1-pro-preview
```

Writes:

```
workspace/outputs/rationale_audit_sample_200_pro_scores.json
```

Optional arguments:
- `--model` changes the audit model.
- `--forecasts` changes which sample is rescored.
- `--questions` changes the question file used for validation and prompt context.
- `--output` changes the output file.
- `--no-resume` starts fresh instead of continuing a previous run.

## Step 4c. Compare Flash and Pro on the Audit Sample

Matches the Flash and Pro scores for each sampled rationale, then calculates agreement statistics and identifies the largest disagreements.

```bash
validate-scores \
  --pro-scores workspace/outputs/rationale_audit_sample_200_pro_scores.json
```

Writes:

```
workspace/outputs/rationale_audit_model_comparison.json
```

Optional arguments:
- `--scores` points to a different Flash scoring file used to recreate the sample.
- `--sample-output` changes the regenerated sample path.
- `--pro-scores` points to a different Pro scoring file.
- `--comparison-output` changes the comparison output path.

## Run the Full Pipeline

`run_pipeline.sh` creates or reuses a virtual environment, installs dependencies, runs Steps 1 through 4c, and writes outputs under `workspace/outputs`.

```bash
bash run_pipeline.sh
```

Optional flags:
- `--clean` removes existing outputs before running.
- `--skip-install` reuses the existing virtual environment without reinstalling dependencies.

Environment variables:
- `GEMINI_API_KEY` is required unless set in `.env`.
- `FLASH_MODEL` overrides the Step 1 scoring model.
- `PRO_MODEL` overrides the Pro model used for analysis and audit rescoring.
- `VENV_DIR` overrides the virtual environment path.

## Key Files

Input data:

```
workspace/forecastbench_data/2024-07-21-human.json
workspace/forecastbench_data/2024-07-21.ForecastBench.human_super_individual.json
```

Main outputs:

```
workspace/outputs/rationale_scores.json
workspace/outputs/five_selected_questions.json
workspace/outputs/five_selected_questions_top_3.json
workspace/outputs/five_selected_questions_overall_summary.json
workspace/outputs/rationale_audit_sample_200.json
workspace/outputs/rationale_audit_sample_200_pro_scores.json
workspace/outputs/rationale_audit_model_comparison.json
```
