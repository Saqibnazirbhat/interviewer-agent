"""Follow-up question generator — turns the interview from a quiz into a conversation.

After each answer, the FollowUpGenerator decides:
  1. MOVE_ON   — the answer was complete, proceed to the next question
  2. CLARIFY   — the answer was vague or incomplete, ask a targeted follow-up
  3. DIG_DEEPER — the candidate made an interesting claim worth exploring

This creates the dynamic, conversational feel of a real interview.
"""

import json
import logging

from src.llm_client import LLMClient, parse_json_object

logger = logging.getLogger("interviewer.followup")


class FollowUpGenerator:
    """Analyzes answers and generates contextual follow-up questions."""

    # Maximum follow-ups per question to prevent infinite loops
    MAX_FOLLOWUPS_PER_QUESTION = 2
    # Maximum total follow-ups across the entire interview
    MAX_FOLLOWUPS_PER_INTERVIEW = 5

    def __init__(self, persona: dict):
        self.llm = LLMClient()
        self.persona = persona
        self._followup_counts: dict[int, int] = {}  # question_id -> count
        self._total_followups = 0

    def should_followup(
        self,
        question: dict,
        answer: str,
        score: dict | None = None,
        time_seconds: float = 0,
        cross_answer_context: str = "",
    ) -> dict:
        """Decide whether to follow up on this answer.

        Args:
            question: The original question dict (id, question, category, difficulty, ideal_signals).
            answer: The candidate's answer text.
            score: Optional score dict with accuracy/depth/communication/ownership (1-10 each).
            time_seconds: How long the candidate took to answer.

        Returns:
            dict with:
                "action": "move_on" | "clarify" | "dig_deeper"
                "followup_question": str (empty if move_on)
                "reason": str (brief explanation of the decision)
        """
        q_id = question.get("id", 0)

        # Don't follow up if interview-wide cap is reached
        if self._total_followups >= self.MAX_FOLLOWUPS_PER_INTERVIEW:
            return {"action": "move_on", "followup_question": "", "reason": "Interview follow-up cap reached"}

        # Don't follow up if already asked max follow-ups for this question
        if self._followup_counts.get(q_id, 0) >= self.MAX_FOLLOWUPS_PER_QUESTION:
            return {"action": "move_on", "followup_question": "", "reason": "Max follow-ups reached"}

        # Skip follow-up for very short answers (likely skip-adjacent)
        if len(answer.strip()) < 20:
            return {"action": "move_on", "followup_question": "", "reason": "Answer too short to follow up on"}

        # Use heuristic pre-filter: if scores are all high, likely no follow-up needed
        if score:
            avg = sum(score.values()) / len(score) if score else 0
            # Very strong answers rarely need follow-up
            if avg >= 8.5 and question.get("difficulty") != "hard":
                return {"action": "move_on", "followup_question": "", "reason": "Strong answer"}

        # Ask the LLM to decide
        try:
            return self._generate_followup(question, answer, score, time_seconds, cross_answer_context)
        except Exception as exc:
            logger.warning("Follow-up generation failed: %s", exc)
            return {"action": "move_on", "followup_question": "", "reason": "Generation failed"}

    def _generate_followup(
        self, question: dict, answer: str, score: dict | None, time_seconds: float,
        cross_answer_context: str = "",
    ) -> dict:
        """Use the LLM to analyze the answer and decide on a follow-up."""
        ideal = "\n".join(f"  - {s}" for s in question.get("ideal_signals", []))
        score_str = json.dumps(score) if score else "not yet scored"

        cross_section = ""
        if cross_answer_context:
            cross_section = f"\n{cross_answer_context}\n"

        prompt = f"""You are a {self.persona['name']} conducting an interview.

{self.persona.get('description', '')}
{cross_section}

You just asked this question ({question.get('category', '')}, {question.get('difficulty', '')}):
"{question.get('question', '')}"

The candidate answered (took {time_seconds:.0f}s):
"{answer}"

Ideal answer signals:
{ideal}

Current scores: {score_str}

Decide what to do next. You have THREE options:

1. MOVE_ON — The answer adequately addresses the question. No follow-up needed.
   Use this when: the answer is complete, covers the key signals, and shows understanding.

2. CLARIFY — The answer is vague, incomplete, or misses key aspects.
   Use this when: important signals are missing, the answer is generic, or claims are unsubstantiated.
   Generate a SHORT, SPECIFIC follow-up that targets what was missed.

3. DIG_DEEPER — The candidate made an interesting claim or showed expertise worth exploring.
   Use this when: the candidate mentioned a specific project, technique, or result that deserves probing.
   Generate a follow-up that pushes them to elaborate on the most interesting part.

IMPORTANT RULES:
- Stay in character as {self.persona.get('short_name', self.persona['name'])}
- Follow-up questions must be SHORT (1-2 sentences max)
- Follow-ups must be SPECIFIC — reference something from their answer
- Prefer DIG_DEEPER over CLARIFY when the answer shows genuine knowledge
- MOVE_ON is the right choice ~50% of the time — don't force follow-ups

Return ONLY valid JSON:
{{
  "action": "move_on" or "clarify" or "dig_deeper",
  "followup_question": "the follow-up question text or empty string if move_on",
  "reason": "1-sentence explanation of your decision"
}}

Raw JSON only. No markdown fences."""

        data = self.llm.generate_json(prompt, mode="object")

        action = data.get("action", "move_on")
        if action not in ("move_on", "clarify", "dig_deeper"):
            action = "move_on"

        followup = data.get("followup_question", "")
        reason = data.get("reason", "")

        # Track follow-up count
        if action != "move_on" and followup:
            q_id = question.get("id", 0)
            self._followup_counts[q_id] = self._followup_counts.get(q_id, 0) + 1
            self._total_followups += 1

        return {
            "action": action,
            "followup_question": followup if action != "move_on" else "",
            "reason": reason,
        }

    def reset(self):
        """Reset follow-up counts (e.g., for a new interview)."""
        self._followup_counts.clear()
        self._total_followups = 0
