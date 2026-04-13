"""Unit tests for QuestionGenerator._parse_response."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.interview.question_generator import QuestionGenerator


def _make_generator():
    """Create a generator without calling Gemini API."""
    with patch.object(QuestionGenerator, "__init__", lambda self: None):
        gen = QuestionGenerator.__new__(QuestionGenerator)
        gen.llm = MagicMock()
        return gen


class TestParseResponse:
    def test_valid_json_array(self):
        gen = _make_generator()
        questions = [
            {"id": 1, "category": "skill", "question": "Q1", "context": "ctx", "difficulty": "easy", "ideal_signals": ["s1"]},
            {"id": 2, "category": "experience", "question": "Q2", "context": "ctx", "difficulty": "medium", "ideal_signals": ["s2"]},
        ] * 3  # 6 questions
        text = json.dumps(questions)
        result = gen._parse_response(text)
        assert len(result) == 6
        assert result[0]["question"] == "Q1"

    def test_json_with_markdown_fences(self):
        gen = _make_generator()
        questions = [{"id": i, "category": "skill", "question": f"Q{i}", "context": "", "difficulty": "easy", "ideal_signals": []} for i in range(1, 8)]
        text = f"```json\n{json.dumps(questions)}\n```"
        result = gen._parse_response(text)
        assert len(result) == 7

    def test_json_with_trailing_text(self):
        gen = _make_generator()
        questions = [{"id": i, "category": "skill", "question": f"Q{i}", "context": "", "difficulty": "easy", "ideal_signals": []} for i in range(1, 8)]
        text = f"Here are your questions:\n{json.dumps(questions)}\n\nI hope these help!"
        result = gen._parse_response(text)
        assert len(result) == 7

    def test_missing_fields_get_defaults(self):
        gen = _make_generator()
        questions = [{"question": "What is X?"} for _ in range(6)]
        text = json.dumps(questions)
        result = gen._parse_response(text)
        assert result[0]["id"] == 1
        assert result[0]["category"] == "experience"
        assert result[0]["difficulty"] == "medium"
        assert result[0]["ideal_signals"] == []
        assert result[0]["context"] == ""

    def test_too_few_questions_raises(self):
        gen = _make_generator()
        questions = [{"id": 1, "question": "Q1"}]
        text = json.dumps(questions)
        with pytest.raises(ValueError, match="Expected ~15 questions"):
            gen._parse_response(text)

    def test_no_array_raises(self):
        gen = _make_generator()
        with pytest.raises(ValueError, match="No JSON array"):
            gen._parse_response("This is just text with no array.")

    def test_unclosed_array_raises(self):
        gen = _make_generator()
        with pytest.raises(ValueError, match="Unclosed JSON array"):
            gen._parse_response('[{"id": 1, "question": "Q1"')

    def test_invalid_json_raises(self):
        gen = _make_generator()
        with pytest.raises(ValueError):
            gen._parse_response('[{invalid json}]')

    def test_non_array_type_raises(self):
        gen = _make_generator()
        # Wrap in brackets to look like array but return a dict
        with pytest.raises(ValueError):
            gen._parse_response('{"not": "an array"}')

    def test_sequential_ids_assigned(self):
        gen = _make_generator()
        questions = [{"question": f"Q{i}"} for i in range(6)]
        text = json.dumps(questions)
        result = gen._parse_response(text)
        # IDs should be 1-indexed
        assert [q["id"] for q in result] == [1, 2, 3, 4, 5, 6]

    def test_preserves_existing_ids(self):
        gen = _make_generator()
        questions = [{"id": 42, "question": "Q1"}] + [{"question": f"Q{i}"} for i in range(2, 7)]
        text = json.dumps(questions)
        result = gen._parse_response(text)
        assert result[0]["id"] == 42


class TestParseResponseEdgeCases:
    def test_nested_arrays_in_ideal_signals(self):
        """Questions with nested arrays should parse correctly."""
        gen = _make_generator()
        questions = [
            {"id": i, "category": "skill", "question": f"Q{i}", "context": "",
             "difficulty": "easy", "ideal_signals": ["signal [with brackets]", "another"]}
            for i in range(1, 8)
        ]
        text = json.dumps(questions)
        result = gen._parse_response(text)
        assert len(result) == 7
        assert "brackets" in result[0]["ideal_signals"][0]

    def test_escaped_quotes_in_questions(self):
        gen = _make_generator()
        questions = [
            {"id": i, "category": "skill", "question": f'Explain the "SOLID" principles for Q{i}',
             "context": "", "difficulty": "easy", "ideal_signals": []}
            for i in range(1, 8)
        ]
        text = json.dumps(questions)
        result = gen._parse_response(text)
        assert '"SOLID"' in result[0]["question"]
