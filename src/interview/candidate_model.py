"""Cross-answer reasoning — maintains a running memory of the entire interview.

The CandidateModel tracks:
  - Every claim the candidate makes across all answers
  - Contradictions between answers (e.g., "I built it alone" vs "my team built it")
  - Skills actually demonstrated vs. just claimed on resume
  - Confidence patterns (strong on X, weak on Y)

This growing context feeds into scoring and follow-up generation so the
interview becomes a coherent conversation rather than isolated Q&A.
"""

import json
import logging
from dataclasses import dataclass, field

from src.llm_client import LLMClient, parse_json_object

logger = logging.getLogger("interviewer.candidate_model")


@dataclass
class Claim:
    """A factual claim the candidate made during the interview."""
    text: str
    source_question_id: int
    category: str = ""
    confidence: str = "medium"  # low, medium, high


@dataclass
class Contradiction:
    """A detected contradiction between two claims."""
    claim_a: str
    claim_b: str
    question_id_a: int
    question_id_b: int
    explanation: str = ""


class CandidateModel:
    """Builds a running profile of the candidate throughout the interview.

    JSON-serializable — no live objects, safe for session store persistence.
    """

    def __init__(self):
        self.claims: list[dict] = []
        self.contradictions: list[dict] = []
        self.demonstrated_skills: list[str] = []
        self.claimed_skills: list[str] = []
        self.confidence_pattern: dict[str, str] = {}  # category -> "strong"/"weak"/"neutral"
        self.answer_history: list[dict] = []  # condensed per-answer summaries
        self._llm = LLMClient()

    def record_answer(
        self,
        question: dict,
        answer: str,
        scores: dict | None = None,
        skipped: bool = False,
    ):
        """Record an answer and extract claims, skills, and confidence signals.

        Uses a fast LLM call to extract structured data from the answer.
        Falls back gracefully if the LLM call fails.
        """
        q_id = question.get("id", 0)
        category = question.get("category", "general")

        entry = {
            "question_id": q_id,
            "category": category,
            "question_text": question.get("question", ""),
            "skipped": skipped,
        }

        if skipped or not answer.strip():
            entry["summary"] = "Skipped"
            entry["claims"] = []
            entry["skills_demonstrated"] = []
            self.answer_history.append(entry)
            # Record weak confidence for skipped categories
            self.confidence_pattern[category] = "weak"
            return

        # Extract claims and skills via LLM
        try:
            extraction = self._extract_from_answer(question, answer, scores)
            entry["summary"] = extraction.get("summary", "")
            entry["claims"] = extraction.get("claims", [])
            entry["skills_demonstrated"] = extraction.get("skills_demonstrated", [])

            # Store claims
            for claim_text in entry["claims"]:
                self.claims.append({
                    "text": claim_text,
                    "source_question_id": q_id,
                    "category": category,
                })

            # Track demonstrated skills
            for skill in entry["skills_demonstrated"]:
                if skill not in self.demonstrated_skills:
                    self.demonstrated_skills.append(skill)

            # Detect contradictions against prior claims
            new_contradictions = extraction.get("contradictions", [])
            for c in new_contradictions:
                self.contradictions.append({
                    "claim_a": c.get("earlier_claim", ""),
                    "claim_b": c.get("new_claim", ""),
                    "question_id_a": c.get("earlier_question_id", 0),
                    "question_id_b": q_id,
                    "explanation": c.get("explanation", ""),
                })

            # Update confidence pattern
            if scores:
                avg = sum(scores.values()) / len(scores) if scores else 5
                if avg >= 7.5:
                    self.confidence_pattern[category] = "strong"
                elif avg <= 4:
                    self.confidence_pattern[category] = "weak"
                else:
                    self.confidence_pattern.setdefault(category, "neutral")

        except Exception as exc:
            logger.warning("CandidateModel extraction failed: %s", exc)
            entry["summary"] = answer[:100]
            entry["claims"] = []
            entry["skills_demonstrated"] = []

        self.answer_history.append(entry)

    def _extract_from_answer(
        self, question: dict, answer: str, scores: dict | None
    ) -> dict:
        """Use the LLM to extract claims, skills, and contradictions from an answer."""
        # Build prior claims context for contradiction detection
        prior_claims = []
        for c in self.claims[-20:]:  # last 20 claims for context
            prior_claims.append(f"  - [Q{c['source_question_id']}] {c['text']}")
        prior_str = "\n".join(prior_claims) if prior_claims else "  (none yet)"

        prompt = f"""Analyze this interview answer and extract structured data.

QUESTION ({question.get('category', '')}, {question.get('difficulty', '')}):
"{question.get('question', '')}"

CANDIDATE'S ANSWER:
"{answer}"

PRIOR CLAIMS BY THIS CANDIDATE:
{prior_str}

Extract the following in JSON:
1. "summary" — A 1-sentence summary of what the candidate said
2. "claims" — Array of specific factual claims (e.g., "Led a team of 5", "Built the auth system from scratch", "Has 3 years of React experience"). Only concrete, verifiable claims — not opinions.
3. "skills_demonstrated" — Array of technical/soft skills the candidate actually demonstrated knowledge of in this answer (not just mentioned)
4. "contradictions" — Array of contradictions with prior claims. Each: {{"earlier_claim": "...", "earlier_question_id": N, "new_claim": "...", "explanation": "..."}}. Empty array if no contradictions.

Return ONLY valid JSON. No markdown fences.
{{
  "summary": "...",
  "claims": ["..."],
  "skills_demonstrated": ["..."],
  "contradictions": []
}}"""

        return self._llm.generate_json(prompt, mode="object")

    def get_context_for_scoring(self) -> str:
        """Build a context string to inject into the scoring prompt.

        Gives the scorer awareness of the full interview trajectory.
        """
        if not self.answer_history:
            return ""

        parts = ["CROSS-ANSWER CONTEXT:"]

        # Summarize prior answers
        parts.append("Prior answer summaries:")
        for entry in self.answer_history[-5:]:  # last 5
            status = "SKIPPED" if entry["skipped"] else entry.get("summary", "")
            parts.append(f"  Q{entry['question_id']} ({entry['category']}): {status}")

        # Contradictions
        if self.contradictions:
            parts.append(f"\nCONTRADICTIONS DETECTED ({len(self.contradictions)}):")
            for c in self.contradictions[-3:]:
                parts.append(
                    f"  - Q{c['question_id_a']}: \"{c['claim_a']}\" vs "
                    f"Q{c['question_id_b']}: \"{c['claim_b']}\" — {c['explanation']}"
                )

        # Confidence pattern
        if self.confidence_pattern:
            parts.append("\nCONFIDENCE PATTERN:")
            for cat, level in self.confidence_pattern.items():
                parts.append(f"  {cat}: {level}")

        # Skills demonstrated
        if self.demonstrated_skills:
            parts.append(f"\nSKILLS DEMONSTRATED: {', '.join(self.demonstrated_skills[:10])}")

        return "\n".join(parts)

    def get_context_for_followup(self) -> str:
        """Build a context string for the follow-up generator.

        Helps generate follow-ups that reference earlier answers.
        """
        if not self.answer_history:
            return ""

        parts = ["INTERVIEW HISTORY (for contextual follow-ups):"]

        # Recent answers
        for entry in self.answer_history[-3:]:
            if not entry["skipped"]:
                parts.append(f"  Q{entry['question_id']}: {entry.get('summary', 'N/A')}")

        # Unverified claims worth probing
        if self.claims:
            recent_claims = [c["text"] for c in self.claims[-5:]]
            parts.append(f"\nRecent claims to potentially probe: {', '.join(recent_claims)}")

        # Contradictions to explore
        if self.contradictions:
            latest = self.contradictions[-1]
            parts.append(
                f"\nLatest contradiction: \"{latest['claim_a']}\" vs \"{latest['claim_b']}\" "
                f"— consider probing this."
            )

        return "\n".join(parts)

    def get_summary(self) -> dict:
        """Return a full summary of the candidate model for reports."""
        return {
            "total_claims": len(self.claims),
            "contradictions": self.contradictions,
            "contradiction_count": len(self.contradictions),
            "demonstrated_skills": self.demonstrated_skills,
            "claimed_skills": self.claimed_skills,
            "confidence_pattern": self.confidence_pattern,
            "answer_count": len(self.answer_history),
            "skipped_count": sum(1 for a in self.answer_history if a["skipped"]),
        }

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict for session storage."""
        return {
            "claims": self.claims,
            "contradictions": self.contradictions,
            "demonstrated_skills": self.demonstrated_skills,
            "claimed_skills": self.claimed_skills,
            "confidence_pattern": self.confidence_pattern,
            "answer_history": self.answer_history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CandidateModel":
        """Reconstruct from a serialized dict."""
        model = cls.__new__(cls)
        model._llm = LLMClient()
        model.claims = data.get("claims", [])
        model.contradictions = data.get("contradictions", [])
        model.demonstrated_skills = data.get("demonstrated_skills", [])
        model.claimed_skills = data.get("claimed_skills", [])
        model.confidence_pattern = data.get("confidence_pattern", {})
        model.answer_history = data.get("answer_history", [])
        return model
