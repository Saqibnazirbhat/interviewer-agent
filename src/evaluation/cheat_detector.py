"""Detects dishonest or suspicious patterns in interview responses."""

import json
import logging

from src.llm_client import LLMClient, parse_json_object

logger = logging.getLogger("interviewer.cheat_detector")


class CheatDetector:
    """Flags answers that appear copy-pasted, suspiciously fast, or inconsistent with the candidate's profile."""

    TIMING_THRESHOLDS = {"easy": 5, "medium": 10, "hard": 15}

    def __init__(self):
        self.llm = LLMClient()

    def check_all(self, responses: list[dict], profile: dict) -> list[dict]:
        """Run integrity checks on every response."""
        for response in responses:
            if response["skipped"]:
                response["integrity"] = {"verdict": "skipped", "flags": []}
                continue
            try:
                response["integrity"] = self._check_single(response, profile)
            except Exception:
                response["integrity"] = {"verdict": "clean", "flags": []}
        return responses

    def _check_single(self, response: dict, profile: dict) -> dict:
        """Run all checks on one response."""
        flags = []

        timing_flag = self._check_timing(response)
        if timing_flag:
            flags.append(timing_flag)

        ai_flags = self._ai_analysis(response, profile)
        flags.extend(ai_flags)

        verdict = "clean"
        if len(flags) >= 2:
            verdict = "flagged"
        elif len(flags) == 1:
            verdict = "suspicious"

        return {"verdict": verdict, "flags": flags}

    def _check_timing(self, response: dict) -> str | None:
        """Flag impossibly fast answers."""
        elapsed = response["time_seconds"]
        answer_len = len(response["answer"])
        difficulty = response.get("difficulty", "medium")
        min_time = self.TIMING_THRESHOLDS.get(difficulty, 10)

        if answer_len > 80 and elapsed < min_time:
            chars_per_sec = answer_len / max(elapsed, 0.1)
            return f"Speed anomaly: {answer_len} chars in {elapsed:.0f}s ({chars_per_sec:.1f} c/s, {difficulty} question)"
        return None

    def _ai_analysis(self, response: dict, profile: dict) -> list[str]:
        """Use the LLM to detect generic/rehearsed answers and profile contradictions."""
        repo_names = [r["name"] for r in profile.get("top_repos", [])]
        languages = list(profile.get("languages", {}).keys())
        skills = profile.get("skills", [])
        expertise = ", ".join(languages) if languages else ", ".join(skills[:10])
        identity = profile.get("username", profile.get("name", "Candidate"))

        prompt = f"""You are an interview integrity analyst. Detect dishonesty in this answer.

CANDIDATE PROFILE:
  Identity: {identity}
  Expertise: {expertise}
  Projects: {', '.join(repo_names) if repo_names else ', '.join(p.get('name', '') for p in profile.get('projects', [])[:5])}

QUESTION ({response['category']}, {response['difficulty']}):
{response['question']}

ANSWER (took {response['time_seconds']}s):
{response['answer']}

Check for:
1. GENERIC — Does the answer lack any personal specifics? Could anyone have written it?
2. CONTRADICTION — Does the answer claim experience the GitHub profile doesn't support?
3. REHEARSED — Does the answer sound like a memorized script?

Return ONLY valid JSON:
{{
  "is_generic": true/false,
  "contradicts_profile": true/false,
  "contradiction_detail": "explanation or empty string",
  "sounds_rehearsed": true/false,
  "notes": "brief explanation"
}}

Raw JSON only. No markdown."""

        try:
            text = self.llm.generate(prompt)
            analysis = self._parse(text)
        except Exception:
            return []

        flags = []
        if analysis.get("is_generic"):
            flags.append("Generic response — lacks personal specifics")
        if analysis.get("contradicts_profile"):
            detail = analysis.get("contradiction_detail", "unspecified")
            flags.append(f"Profile contradiction: {detail}")
        if analysis.get("sounds_rehearsed"):
            flags.append("Answer sounds rehearsed or templated")
        return flags

    def _parse(self, text: str) -> dict:
        """Extract JSON from the LLM response."""
        try:
            return parse_json_object(text)
        except (ValueError, json.JSONDecodeError):
            return {}

    def summarize_flags(self, responses: list[dict]) -> dict:
        """Produce an aggregate integrity report."""
        total = len(responses)
        counts = {"flagged": 0, "suspicious": 0, "clean": 0, "skipped": 0}
        all_flags = []

        for idx, r in enumerate(responses):
            verdict = r.get("integrity", {}).get("verdict", "clean")
            counts[verdict] = counts.get(verdict, 0) + 1
            q_num = idx + 1  # 1-based display number
            flags_for_q = r.get("integrity", {}).get("flags", [])
            if flags_for_q:
                # Group all flags for this question into a single entry
                all_flags.append({
                    "question_id": q_num,
                    "flag": "; ".join(flags_for_q),
                })

        answerable = total - counts["skipped"]
        integrity_score = round(counts["clean"] / max(answerable, 1) * 10, 1)

        return {
            "total_questions": total,
            "clean_count": counts["clean"],
            "suspicious_count": counts["suspicious"],
            "flagged_count": counts["flagged"],
            "all_flags": all_flags,
            "integrity_score": integrity_score,
        }
