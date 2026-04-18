"""Tests for follow-up generator caps (per-question and per-interview)."""

from unittest.mock import MagicMock, patch

from src.interview.followup import FollowUpGenerator


def _persona():
    return {
        "name": "Technical Expert",
        "short_name": "Technical Expert",
        "description": "Deep technical interviewer.",
    }


def _question(qid: int = 1):
    return {
        "id": qid,
        "category": "experience",
        "difficulty": "medium",
        "question": "Describe your last project.",
        "ideal_signals": ["specific tech", "impact"],
    }


def _make_gen():
    """Build a FollowUpGenerator without touching the network."""
    with patch("src.interview.followup.LLMClient") as Client:
        instance = MagicMock()
        # Always return a valid follow-up — the cap logic should still kick in
        instance.generate_json.return_value = {
            "action": "dig_deeper",
            "followup_question": "Why that approach?",
            "reason": "wants depth",
        }
        Client.return_value = instance
        return FollowUpGenerator(_persona())


def test_per_interview_cap_is_five():
    gen = _make_gen()
    answer = "I built a service that scaled to 10k QPS with caching." * 2
    score = {"accuracy": 6, "depth": 6, "communication": 6, "ownership": 6}

    issued = 0
    # Try 8 different questions — each should be eligible until the cap fires
    for qid in range(1, 9):
        result = gen.should_followup(_question(qid), answer, score, time_seconds=40)
        if result["action"] != "move_on":
            issued += 1
    assert issued == 5, f"Expected exactly 5 follow-ups, got {issued}"


def test_per_question_cap_still_enforced():
    """Even within the interview budget, a single question shouldn't get >2 follow-ups."""
    gen = _make_gen()
    answer = "I migrated a legacy monolith to microservices over six months." * 2
    score = {"accuracy": 6, "depth": 6, "communication": 6, "ownership": 6}

    issued = 0
    for _ in range(5):
        result = gen.should_followup(_question(qid=1), answer, score, time_seconds=40)
        if result["action"] != "move_on":
            issued += 1
    assert issued == 2, f"Per-question cap should fire at 2, got {issued}"


def test_reset_clears_both_counters():
    gen = _make_gen()
    answer = "I led a team of 5 engineers building a recommendation system." * 2
    score = {"accuracy": 6, "depth": 6, "communication": 6, "ownership": 6}

    # Burn the per-interview budget across 5 questions
    for qid in range(1, 6):
        gen.should_followup(_question(qid), answer, score, time_seconds=40)
    assert gen._total_followups == 5

    gen.reset()
    assert gen._total_followups == 0
    assert gen._followup_counts == {}

    # After reset, follow-ups should fire again
    result = gen.should_followup(_question(qid=99), answer, score, time_seconds=40)
    assert result["action"] != "move_on"


def test_short_answer_skips_followup():
    """Sanity check — pre-filter on answer length still works."""
    gen = _make_gen()
    result = gen.should_followup(_question(), "yes", None, time_seconds=5)
    assert result["action"] == "move_on"
    assert "too short" in result["reason"].lower()


def test_strong_score_skips_followup_on_easy():
    """Sanity check — the heuristic skip on high-score easy questions still works."""
    gen = _make_gen()
    q = _question()
    q["difficulty"] = "easy"
    score = {"accuracy": 9, "depth": 9, "communication": 9, "ownership": 9}
    long_answer = "This is a thoughtful, complete answer with concrete details." * 3
    result = gen.should_followup(q, long_answer, score, time_seconds=40)
    assert result["action"] == "move_on"
