"""Scores interview answers across multiple dimensions using NVIDIA NIM."""

import json
import logging

from src.llm_client import LLMClient, parse_json_object, strip_fences

logger = logging.getLogger("interviewer.scorer")


class AnswerScorer:
    """Evaluates each answer on accuracy, depth, communication, and ownership."""

    DIMENSIONS = ["accuracy", "depth", "communication", "ownership"]

    def __init__(self, persona: dict):
        self.llm = LLMClient()
        self.persona = persona

    def score_all(self, responses: list[dict], profile: dict, cross_answer_context: str = "") -> list[dict]:
        """Score every non-skipped response. Returns the list with scores attached."""
        scored = []
        for response in responses:
            if response["skipped"]:
                response["scores"] = {d: 0 for d in self.DIMENSIONS}
                response["score_total"] = 0
                response["feedback"] = "Skipped — no answer provided."
                scored.append(response)
                continue

            try:
                evaluation = self._evaluate_answer(response, profile, cross_answer_context)
            except Exception:
                evaluation = {
                    "scores": {d: 5 for d in self.DIMENSIONS},
                    "feedback": "Scoring unavailable for this answer — default scores applied.",
                }

            response["scores"] = evaluation["scores"]
            response["score_total"] = sum(evaluation["scores"].values())
            response["feedback"] = evaluation["feedback"]
            scored.append(response)

        return scored

    def _evaluate_answer(self, response: dict, profile: dict, cross_answer_context: str = "") -> dict:
        """Ask the LLM to score a single answer with persona-aware rubric."""
        ideal = "\n".join(f"  - {s}" for s in response.get("ideal_signals", []))
        languages = ", ".join(profile.get("skills", list(profile.get("languages", {}).keys())))

        cross_section = ""
        if cross_answer_context:
            cross_section = f"\n{cross_answer_context}\n"

        prompt = f"""You are a senior technical interview evaluator.

INTERVIEWER PERSONA: {self.persona['name']}
SCORING GUIDANCE: {self.persona.get('scoring_bias', 'Score all dimensions equally.')}
{cross_section}
QUESTION ({response['category']}, {response['difficulty']}):
{response['question']}

CANDIDATE'S ANSWER:
{response['answer']}

CONTEXT: This question targets {response['context']}.
CANDIDATE'S TECH STACK: {languages}
TIME TAKEN: {response['time_seconds']}s (timing: {response.get('timing_status', 'early')} — early=submitted before limit, full=used most of time, timeout=ran out of 100s limit)

IDEAL ANSWER SIGNALS:
{ideal}

Score the answer on these 4 dimensions (1-10 each):
1. ACCURACY — Is the answer technically correct?
2. DEPTH — Does it go beyond surface level with specifics and tradeoffs?
3. COMMUNICATION — Is it clear, structured, and well-articulated?
4. OWNERSHIP — Does the candidate show genuine understanding vs. memorized content?

Return ONLY valid JSON:
{{
  "scores": {{
    "accuracy": <int 1-10>,
    "depth": <int 1-10>,
    "communication": <int 1-10>,
    "ownership": <int 1-10>
  }},
  "feedback": "<2-3 sentence evaluation>"
}}

Raw JSON only. No markdown fences."""

        text = self.llm.generate(prompt)
        return self._parse_scores(text)

    def _parse_scores(self, text: str) -> dict:
        """Extract scores from the LLM response with robust fallback."""
        try:
            data = parse_json_object(text)
        except (ValueError, json.JSONDecodeError):
            return self._fallback()

        scores = data.get("scores", {})
        for dim in self.DIMENSIONS:
            val = scores.get(dim, 5)
            try:
                scores[dim] = max(1, min(10, int(val)))
            except (ValueError, TypeError):
                scores[dim] = 5

        return {
            "scores": scores,
            "feedback": data.get("feedback", "No feedback generated."),
        }

    def compute_summary(self, scored_responses: list[dict]) -> dict:
        """Aggregate scores into an overall summary."""
        answered = [r for r in scored_responses if not r["skipped"]]
        if not answered:
            return {
                "overall": 0,
                "by_dimension": {d: 0 for d in self.DIMENSIONS},
                "by_category": {},
                "answered_count": 0,
                "skipped_count": len(scored_responses),
            }

        by_dimension = {}
        for dim in self.DIMENSIONS:
            vals = [r["scores"][dim] for r in answered]
            by_dimension[dim] = round(sum(vals) / len(vals), 1)

        by_category: dict[str, list] = {}
        for r in answered:
            cat = r["category"]
            by_category.setdefault(cat, []).append(r["score_total"])

        category_avgs = {
            cat: round(sum(vals) / len(vals), 1) for cat, vals in by_category.items()
        }

        overall = round(sum(by_dimension.values()) / len(by_dimension), 1)

        return {
            "overall": overall,
            "by_dimension": by_dimension,
            "by_category": category_avgs,
            "answered_count": len(answered),
            "skipped_count": len(scored_responses) - len(answered),
        }

    def _fallback(self) -> dict:
        return {
            "scores": {d: 5 for d in self.DIMENSIONS},
            "feedback": "Could not parse evaluation — default scores applied.",
        }
