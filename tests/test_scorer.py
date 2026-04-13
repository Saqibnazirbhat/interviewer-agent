"""Unit tests for AnswerScorer.compute_summary and _parse_scores."""

import json
from unittest.mock import MagicMock, patch

from src.evaluation.scorer import AnswerScorer


def _make_scorer():
    """Create a scorer without calling Gemini API."""
    with patch.object(AnswerScorer, "__init__", lambda self, persona: None):
        scorer = AnswerScorer.__new__(AnswerScorer)
        scorer.persona = {"name": "Test", "scoring_bias": "balanced"}
        scorer.llm = MagicMock()
        return scorer


class TestComputeSummary:
    def test_all_answered(self):
        scorer = _make_scorer()
        responses = [
            {"skipped": False, "category": "skill", "scores": {"accuracy": 8, "depth": 6, "communication": 7, "ownership": 9}, "score_total": 30},
            {"skipped": False, "category": "experience", "scores": {"accuracy": 6, "depth": 8, "communication": 5, "ownership": 7}, "score_total": 26},
        ]
        summary = scorer.compute_summary(responses)
        assert summary["answered_count"] == 2
        assert summary["skipped_count"] == 0
        assert summary["by_dimension"]["accuracy"] == 7.0  # (8+6)/2
        assert summary["by_dimension"]["depth"] == 7.0
        assert summary["by_dimension"]["communication"] == 6.0
        assert summary["by_dimension"]["ownership"] == 8.0
        assert summary["overall"] == 7.0  # (7+7+6+8)/4

    def test_all_skipped(self):
        scorer = _make_scorer()
        responses = [
            {"skipped": True, "category": "skill", "scores": {"accuracy": 0, "depth": 0, "communication": 0, "ownership": 0}, "score_total": 0},
        ]
        summary = scorer.compute_summary(responses)
        assert summary["answered_count"] == 0
        assert summary["skipped_count"] == 1
        assert summary["overall"] == 0

    def test_mixed_skip_and_answered(self):
        scorer = _make_scorer()
        responses = [
            {"skipped": False, "category": "skill", "scores": {"accuracy": 10, "depth": 10, "communication": 10, "ownership": 10}, "score_total": 40},
            {"skipped": True, "category": "project", "scores": {"accuracy": 0, "depth": 0, "communication": 0, "ownership": 0}, "score_total": 0},
        ]
        summary = scorer.compute_summary(responses)
        assert summary["answered_count"] == 1
        assert summary["skipped_count"] == 1
        assert summary["overall"] == 10.0

    def test_by_category(self):
        scorer = _make_scorer()
        responses = [
            {"skipped": False, "category": "skill", "scores": {"accuracy": 8, "depth": 8, "communication": 8, "ownership": 8}, "score_total": 32},
            {"skipped": False, "category": "skill", "scores": {"accuracy": 6, "depth": 6, "communication": 6, "ownership": 6}, "score_total": 24},
            {"skipped": False, "category": "experience", "scores": {"accuracy": 10, "depth": 10, "communication": 10, "ownership": 10}, "score_total": 40},
        ]
        summary = scorer.compute_summary(responses)
        assert summary["by_category"]["skill"] == 28.0  # (32+24)/2
        assert summary["by_category"]["experience"] == 40.0

    def test_empty_responses(self):
        scorer = _make_scorer()
        summary = scorer.compute_summary([])
        assert summary["answered_count"] == 0
        assert summary["overall"] == 0


class TestParseScores:
    def test_valid_json(self):
        scorer = _make_scorer()
        text = json.dumps({"scores": {"accuracy": 8, "depth": 7, "communication": 9, "ownership": 6}, "feedback": "Good."})
        result = scorer._parse_scores(text)
        assert result["scores"]["accuracy"] == 8
        assert result["feedback"] == "Good."

    def test_json_with_fences(self):
        scorer = _make_scorer()
        text = '```json\n{"scores": {"accuracy": 5, "depth": 5, "communication": 5, "ownership": 5}, "feedback": "OK."}\n```'
        result = scorer._parse_scores(text)
        assert result["scores"]["accuracy"] == 5

    def test_json_with_extra_text(self):
        scorer = _make_scorer()
        text = 'Here is the score:\n{"scores": {"accuracy": 7, "depth": 7, "communication": 7, "ownership": 7}, "feedback": "Solid."}\nEnd.'
        result = scorer._parse_scores(text)
        assert result["scores"]["accuracy"] == 7

    def test_clamps_scores_to_range(self):
        scorer = _make_scorer()
        text = json.dumps({"scores": {"accuracy": 15, "depth": -3, "communication": 5, "ownership": 5}, "feedback": "X"})
        result = scorer._parse_scores(text)
        assert result["scores"]["accuracy"] == 10
        assert result["scores"]["depth"] == 1

    def test_missing_dimension_defaults_to_5(self):
        scorer = _make_scorer()
        text = json.dumps({"scores": {"accuracy": 8}, "feedback": "Partial."})
        result = scorer._parse_scores(text)
        assert result["scores"]["depth"] == 5
        assert result["scores"]["communication"] == 5
        assert result["scores"]["ownership"] == 5

    def test_garbage_returns_fallback(self):
        scorer = _make_scorer()
        result = scorer._parse_scores("this is not json at all")
        assert result["scores"]["accuracy"] == 5
        assert "default scores" in result["feedback"].lower() or "could not parse" in result["feedback"].lower()
