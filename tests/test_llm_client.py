"""Unit tests for LLMClient JSON parsing helpers and TokenUsage."""

import pytest

from src.llm_client import (
    TokenUsage,
    parse_json_array,
    parse_json_object,
    strip_fences,
)


class TestStripFences:
    def test_no_fences(self):
        assert strip_fences('{"a": 1}') == '{"a": 1}'

    def test_json_fences(self):
        assert strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_plain_fences(self):
        assert strip_fences('```\n{"a": 1}\n```') == '{"a": 1}'

    def test_whitespace(self):
        assert strip_fences('  \n```json\n{"a": 1}\n```\n  ') == '{"a": 1}'


class TestParseJsonObject:
    def test_clean_json(self):
        assert parse_json_object('{"key": "value"}') == {"key": "value"}

    def test_json_in_text(self):
        assert parse_json_object('Result: {"x": 1} done') == {"x": 1}

    def test_fenced_json(self):
        assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}

    def test_nested_objects(self):
        text = '{"scores": {"a": 1, "b": 2}, "note": "ok"}'
        result = parse_json_object(text)
        assert result["scores"]["a"] == 1

    def test_no_object_raises(self):
        with pytest.raises(ValueError, match="Could not parse"):
            parse_json_object("just plain text")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            parse_json_object("")


class TestParseJsonArray:
    def test_clean_array(self):
        assert parse_json_array('[1, 2, 3]') == [1, 2, 3]

    def test_array_in_text(self):
        result = parse_json_array('Questions: [{"id": 1}] end')
        assert result == [{"id": 1}]

    def test_nested_brackets(self):
        text = '[{"a": [1, 2]}, {"b": [3]}]'
        result = parse_json_array(text)
        assert len(result) == 2
        assert result[0]["a"] == [1, 2]

    def test_strings_with_brackets(self):
        text = '[{"q": "What is [x]?"}]'
        result = parse_json_array(text)
        assert result[0]["q"] == "What is [x]?"

    def test_no_array_raises(self):
        with pytest.raises(ValueError, match="No JSON array"):
            parse_json_array("no array here")

    def test_unclosed_array_raises(self):
        with pytest.raises(ValueError, match="Unclosed JSON array"):
            parse_json_array('[{"id": 1}')


class TestTokenUsage:
    def test_initial_state(self):
        usage = TokenUsage()
        assert usage.total_tokens == 0
        assert usage.call_count == 0

    def test_record(self):
        usage = TokenUsage()
        usage.record(100, 50)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.call_count == 1

    def test_cumulative(self):
        usage = TokenUsage()
        usage.record(100, 50)
        usage.record(200, 100)
        assert usage.total_tokens == 450
        assert usage.call_count == 2

    def test_summary(self):
        usage = TokenUsage()
        usage.record(100, 50)
        usage.errors = 2
        usage.retries = 1
        s = usage.summary()
        assert s["prompt_tokens"] == 100
        assert s["errors"] == 2
        assert s["retries"] == 1
