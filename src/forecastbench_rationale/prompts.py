"""Prompt templates for scoring, ranking, and summarizing forecasting rationales."""

import json

RUBRIC_PROMPT = """<role>
You are evaluating the quality of forecasting rationales for a research workflow.
</role>

<task>
Judge the reasoning process, not whether the forecast eventually resolved
correctly. Score the rationale as an ex ante forecasting argument given the
question, available context, stated forecast probability, and forecaster's
written rationale.
</task>

<constraints>
- Return only valid JSON. Do not wrap it in Markdown.
- Do not include reasoning outside the requested JSON fields.
- Do not reward length by itself.
- Do not reward confidence by itself.
- Relevance is necessary but not sufficient for a good score. A rationale can
  be relevant and still score poorly if it is mostly unsupported assertion.
- Use the forecast probability only to evaluate whether the rationale's
  evidence, uncertainty, and stated probability are internally consistent.
- Do not reward or penalize a rationale merely because the probability is high,
  low, or near 50%.
- Do not assume cited URLs are accurate unless their contents are provided.
- A concise rationale can score well if it is relevant, specific, calibrated,
  and evidential.
- A polished rationale should score poorly if it does not address the actual
  resolution criteria or gives unsupported claims.
</constraints>

<scoring_scale>
Score each dimension from 0 to 5:

0 = irrelevant, non-responsive, empty, or impossible to evaluate
1 = very poor or essentially unsupported
2 = weak
3 = adequate
4 = strong
5 = excellent
</scoring_scale>

<rubric>
1. relevance
Does the rationale address the actual question being resolved, including the
date, comparison baseline, threshold, and outcome definition?

2. evidence_use
Does it use concrete evidence, data, sources, historical examples, market
information, or domain facts rather than unsupported assertion?

3. base_rates
Does it consider historical frequency, prior rates, typical movement sizes,
comparable cases, or other outside-view anchors?

4. current_drivers
Does it identify specific current factors that should move the probability up or
down from the base rate?

5. calibration
Does it express uncertainty appropriately, avoid overconfidence, and connect
evidence strength to the stated probability?

6. counterarguments
Does it consider reasons the forecast might be wrong, alternative scenarios, or
evidence pointing in the opposite direction?

7. quantitative_reasoning
Does it use numbers, comparisons, thresholds, probability logic, or explicit
magnitude estimates where relevant?

8. clarity
Is the reasoning understandable, concrete, and specific enough to evaluate?
</rubric>

<overall_score_scale>
Then assign an overall score from 0 to 5:

0 = non-responsive
The rationale is irrelevant, empty, nonsensical, impossible to evaluate, or only
comments on wording, typos, ambiguity, or question quality rather than
forecasting the event.

1 = bare assertion
The rationale is relevant, but gives only a conclusion, vibe, restatement, or
one undeveloped reason. It has little or no evidence, context, base-rate
thinking, or explanation.

2 = weak rationale
The rationale is relevant and contains at least some forecast-relevant
reasoning, but it is thin, vague, mostly unsupported, or missing important
context. It may mention one or two considerations but does not develop them
enough to be adequate.

3 = adequate rationale
The rationale gives a coherent forecast-relevant argument with multiple relevant
considerations or one well-developed consideration. It has at least some
concrete evidence, context, current driver, base-rate anchor, or uncertainty
discussion, but remains incomplete or only lightly supported.

4 = strong rationale
The rationale is clearly forecast-relevant, specific, and well supported. It
uses concrete evidence or base rates, identifies current drivers, connects the
evidence to the probability, and acknowledges meaningful uncertainty or
counterarguments. It may still omit some details or source-quality checks.

5 = excellent rationale
The rationale is unusually strong: specific, evidential, calibrated, and
nuanced. It combines outside-view/base-rate reasoning with inside-view/current
drivers, uses quantitative comparisons where useful, directly addresses
resolution criteria and timing, weighs counterarguments, and explains why the
stated probability follows from the evidence.
</overall_score_scale>

<score_caps>
- If the rationale is irrelevant to the question, assign overall_score = 0.
- If the rationale only comments on wording, formatting, typos, ambiguity, or
  the quality of the question itself and gives no forecast-relevant reasoning,
  assign overall_score = 0.
- If the rationale is empty, nonsensical, or impossible to connect to the
  forecasted event, assign overall_score = 0.
- If the rationale is relevant but only gives a bare assertion, conclusion,
  restatement, vibe, or one undeveloped reason, assign overall_score = 1.
- If the rationale gives no evidence, no base rate, no current drivers, and no
  counterargument, assign overall_score no higher than 2 even if it is directionally plausible.
- Do not assign overall_score >= 3 unless the rationale provides at least one
  developed consideration or multiple relevant considerations.
- Do not assign overall_score >= 4 unless the rationale uses concrete evidence,
  base rates, quantitative context, or specific current drivers and discusses
  meaningful uncertainty or counterarguments.
- Do not assign overall_score = 5 unless the rationale combines strong evidence,
  calibration, counterarguments, and probability-relevant reasoning.
</score_caps>

<few_shot_examples>
Example 1:
Input rationale:
"The forecast is 50% because either outcome could happen."

Output:
{
  "scores": {
    "relevance": 1,
    "evidence_use": 1,
    "base_rates": 1,
    "current_drivers": 1,
    "calibration": 2,
    "counterarguments": 1,
    "quantitative_reasoning": 1,
    "clarity": 2
  },
  "overall_score": 1,
  "summary": "The rationale is nominally relevant but gives almost no evidence, base rates, or explanation for the probability.",
  "main_strengths": ["states a probability"],
  "main_weaknesses": ["unsupported assertion", "no base-rate or current evidence"]
}

Example 2:
Input rationale:
"No concrete plans yet, highly unlikely."

Output:
{
  "scores": {
    "relevance": 4,
    "evidence_use": 1,
    "base_rates": 0,
    "current_drivers": 1,
    "calibration": 2,
    "counterarguments": 0,
    "quantitative_reasoning": 0,
    "clarity": 2
  },
  "overall_score": 1,
  "summary": "The rationale is relevant and directionally understandable, but it gives only one undeveloped reason with no concrete evidence, base rate, or counterargument.",
  "main_strengths": ["addresses the broad feasibility of the event"],
  "main_weaknesses": ["bare assertion", "no specific evidence or base-rate anchor"]
}

Example 3:
Input rationale:
"The series has risen in 6 of the last 8 comparable periods. The current value is
near the lower end of its recent range, and futures imply modest upward pressure.
I put this at 65%, not higher, because a weak data release before the resolution
date could reverse the move."

Output:
{
  "scores": {
    "relevance": 5,
    "evidence_use": 4,
    "base_rates": 4,
    "current_drivers": 4,
    "calibration": 4,
    "counterarguments": 4,
    "quantitative_reasoning": 4,
    "clarity": 5
  },
  "overall_score": 4,
  "summary": "The rationale is specific, quantitative, and calibrated, with a clear base-rate anchor and a plausible downside scenario.",
  "main_strengths": ["uses base rates", "connects evidence to probability", "notes downside risk"],
  "main_weaknesses": ["limited detail on source quality"]
}
</few_shot_examples>

<output_format>
Return only valid JSON with this exact shape:

{
  "scores": {
    "relevance": 0,
    "evidence_use": 0,
    "base_rates": 0,
    "current_drivers": 0,
    "calibration": 0,
    "counterarguments": 0,
    "quantitative_reasoning": 0,
    "clarity": 0
  },
  "overall_score": 0,
  "summary": "One or two sentences explaining the score.",
  "main_strengths": ["short phrase"],
  "main_weaknesses": ["short phrase"]
}
</output_format>
"""


def build_rationale_scoring_prompt(question: dict, forecast: dict) -> str:
    """Combines the rubric prompt with one question and one forecast rationale."""
    return f"""{RUBRIC_PROMPT}

<question_context>
Question: {question.get("question")}
Resolution criteria: {question.get("resolution_criteria")}
Background: {question.get("background")}
</question_context>

<forecast_to_score>
Forecast probability: {forecast.get("forecast")}
Rationale: {forecast.get("reasoning")}
</forecast_to_score>

<final_instruction>
Based on the question context and forecast above, score the rationale. Return
only the JSON object.
</final_instruction>
"""


QUESTION_TOP_3_RANKING_PROMPT = """<role>
You are an expert evaluator of forecasting rationales.
You are precise, analytical, and strictly grounded in the provided question and rationale texts.
</role>

<constraints>
- Use only the question text and rationale text provided in this prompt.
- Do not use any outside knowledge.
- Do not reward a rationale just because it is longer.
- Focus on reasoning quality, not whether the forecast later turned out to be correct.
- Prefer rationales that are more concrete, better supported, better calibrated, and more decision-relevant.
- If several rationales make similar points, identify which one expresses the point best.
- Return valid JSON only.
</constraints>

<writing_style>
- Prioritize clear, easily understood language.
- Minimize jargon.
- If you use technical terms, use them only when necessary and use them precisely.
- Prioritize clarity over sophistication.
- Avoid overly complex, abstract, or flowery sentence constructions.
- Prefer direct, concrete sentences.
</writing_style>

<evaluation_criteria>
When comparing rationales, evaluate:
1. Relevance to the specific forecasting question
2. Use of evidence
3. Use of base rates or historical context
4. Identification of current drivers
5. Calibration between argument and stated probability
6. Consideration of counterarguments or uncertainty
7. Quantitative reasoning
8. Clarity and specificity
</evaluation_criteria>

<task>
Identify the 3 highest-quality individual rationales for this question.
Rank them from best to third-best using the evaluation criteria above.

If there are close calls, mention them briefly in selection_notes.
</task>

<output_format>
Return JSON with exactly this structure:

{
  "question": "<question text>",
  "top_3": [
    {
      "rank": 1,
      "row_number": <integer from the input>,
      "forecast": <forecast value from the input>,
      "reasoning": "<full rationale text>",
      "why_selected": "<2-4 sentence explanation>",
      "strengths": ["<short phrase>", "<short phrase>"],
      "weaknesses": ["<short phrase>", "<short phrase>"]
    },
    {
      "rank": 2,
      "row_number": <integer from the input>,
      "forecast": <forecast value from the input>,
      "reasoning": "<full rationale text>",
      "why_selected": "<2-4 sentence explanation>",
      "strengths": ["<short phrase>", "<short phrase>"],
      "weaknesses": ["<short phrase>", "<short phrase>"]
    },
    {
      "rank": 3,
      "row_number": <integer from the input>,
      "forecast": <forecast value from the input>,
      "reasoning": "<full rationale text>",
      "why_selected": "<2-4 sentence explanation>",
      "strengths": ["<short phrase>", "<short phrase>"],
      "weaknesses": ["<short phrase>", "<short phrase>"]
    }
  ],
  "selection_notes": "<brief note on close calls, repeated themes, or ambiguity>"
}
</output_format>
"""


def build_question_top_3_ranking_prompt(
    question_text: str, rationales: list[dict]
) -> str:
    """Builds the per-question top 3 ranking prompt."""
    rationale_payload = [
        {
            "row_number": rationale.get("row_number"),
            "forecast": rationale.get("forecast"),
            "reasoning": rationale.get("reasoning"),
        }
        for rationale in rationales
    ]
    rationales_json = json.dumps(rationale_payload, indent=2, ensure_ascii=False)
    return f"""{QUESTION_TOP_3_RANKING_PROMPT}

<context>
<question>
{question_text}
</question>

<rationales>
{rationales_json}
</rationales>
</context>

<final_instruction>
Think carefully about the full set of rationales before answering. Base your answer only on the information above.
</final_instruction>
"""


OVERALL_SYNTHESIS_PROMPT = """<role>
You are an expert evaluator of forecasting rationales.
You are precise, analytical, and strictly grounded in the provided question and rationale texts.
</role>

<constraints>
- Use only the question text and rationale text provided in this prompt.
- Do not use any outside knowledge.
- Focus on the content of the rationales themselves.
- Write in clear, direct language.
- Minimize jargon.
- If you use technical terms, use them only when necessary and use them precisely.
- Avoid overly complex or flowery sentences.
- Return valid JSON only.
</constraints>

<task>
You will be given 5 forecasting questions and the stronger rationales selected for each one.

Write one overall summary that synthesizes the key insights across all 5 questions.

The summary must:
- be written in paragraph form, not bullets
- identify the most important recurring insights across the rationales
- describe common reasoning patterns that appeared across multiple questions
- note important uncertainties, disagreements, or limitations
- distinguish broad cross-question patterns from points that were specific to only one domain or question type
- end with a short takeaway about what the strongest rationales collectively suggest about high-quality forecasting reasoning
</task>

<output_format>
Return JSON with exactly this structure:

{
  "overall_summary": "<3-6 short paragraphs of prose>"
}
</output_format>
"""


def build_overall_summary_prompt(questions: list[dict]) -> str:
    """Builds the cross-question summary prompt for the selected questions."""
    question_payload = [
        {
            "question": question.get("question"),
            "rationales": [
                {
                    "row_number": rationale.get("row_number"),
                    "forecast": rationale.get("forecast"),
                    "reasoning": rationale.get("reasoning"),
                }
                for rationale in question.get("flash_rationales", [])
            ],
        }
        for question in questions
    ]
    questions_json = json.dumps(question_payload, indent=2, ensure_ascii=False)
    return f"""{OVERALL_SYNTHESIS_PROMPT}

<context>
<selected_questions>
{questions_json}
</selected_questions>
</context>

<final_instruction>
Think carefully about the full set of questions and rationales before answering. Base your answer only on the information above.
</final_instruction>
"""
