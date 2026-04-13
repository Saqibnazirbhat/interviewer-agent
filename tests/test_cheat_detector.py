"""Unit tests for CheatDetector._check_timing and summarize_flags."""

from unittest.mock import MagicMock, patch

from src.evaluation.cheat_detector import CheatDetector


def _make_detector():
    """Create a detector without calling Gemini API."""
    with patch.object(CheatDetector, "__init__", lambda self: None):
        detector = CheatDetector.__new__(CheatDetector)
        detector.llm = MagicMock()
        detector.TIMING_THRESHOLDS = {"easy": 5, "medium": 10, "hard": 15}
        return detector


class TestCheckTiming:
    def test_normal_speed_no_flag(self):
        detector = _make_detector()
        response = {"time_seconds": 30, "answer": "A reasonable answer here.", "difficulty": "medium"}
        assert detector._check_timing(response) is None

    def test_fast_but_short_answer_no_flag(self):
        """Short answers typed quickly are fine — only long answers flag."""
        detector = _make_detector()
        response = {"time_seconds": 3, "answer": "yes", "difficulty": "easy"}
        assert detector._check_timing(response) is None

    def test_impossibly_fast_long_answer_flags(self):
        detector = _make_detector()
        # 200 chars in 3 seconds on a medium question (threshold=10s)
        response = {"time_seconds": 3, "answer": "x" * 200, "difficulty": "medium"}
        flag = detector._check_timing(response)
        assert flag is not None
        assert "Speed anomaly" in flag

    def test_fast_easy_question_flags(self):
        detector = _make_detector()
        response = {"time_seconds": 2, "answer": "x" * 100, "difficulty": "easy"}
        flag = detector._check_timing(response)
        assert flag is not None

    def test_fast_hard_question_flags(self):
        detector = _make_detector()
        response = {"time_seconds": 8, "answer": "x" * 150, "difficulty": "hard"}
        flag = detector._check_timing(response)
        assert flag is not None

    def test_just_above_threshold_no_flag(self):
        detector = _make_detector()
        response = {"time_seconds": 11, "answer": "x" * 100, "difficulty": "medium"}
        assert detector._check_timing(response) is None

    def test_zero_time_flags(self):
        detector = _make_detector()
        response = {"time_seconds": 0, "answer": "x" * 100, "difficulty": "easy"}
        flag = detector._check_timing(response)
        assert flag is not None


class TestSummarizeFlags:
    def test_all_clean(self):
        detector = _make_detector()
        responses = [
            {"question_id": 1, "integrity": {"verdict": "clean", "flags": []}},
            {"question_id": 2, "integrity": {"verdict": "clean", "flags": []}},
        ]
        summary = detector.summarize_flags(responses)
        assert summary["clean_count"] == 2
        assert summary["flagged_count"] == 0
        assert summary["suspicious_count"] == 0
        assert summary["integrity_score"] == 10.0

    def test_one_flagged(self):
        detector = _make_detector()
        responses = [
            {"question_id": 1, "integrity": {"verdict": "flagged", "flags": ["Speed anomaly", "Generic response"]}},
            {"question_id": 2, "integrity": {"verdict": "clean", "flags": []}},
        ]
        summary = detector.summarize_flags(responses)
        assert summary["flagged_count"] == 1
        assert summary["clean_count"] == 1
        assert len(summary["all_flags"]) == 2
        assert summary["integrity_score"] == 5.0  # 1 clean out of 2

    def test_mixed_verdicts(self):
        detector = _make_detector()
        responses = [
            {"question_id": 1, "integrity": {"verdict": "clean", "flags": []}},
            {"question_id": 2, "integrity": {"verdict": "suspicious", "flags": ["Generic response"]}},
            {"question_id": 3, "integrity": {"verdict": "flagged", "flags": ["Speed anomaly", "Contradiction"]}},
            {"question_id": 4, "integrity": {"verdict": "skipped", "flags": []}},
        ]
        summary = detector.summarize_flags(responses)
        assert summary["total_questions"] == 4
        assert summary["clean_count"] == 1
        assert summary["suspicious_count"] == 1
        assert summary["flagged_count"] == 1
        # Integrity based on non-skipped: 1 clean / 3 answerable
        assert abs(summary["integrity_score"] - 3.3) < 0.1

    def test_empty_responses(self):
        detector = _make_detector()
        summary = detector.summarize_flags([])
        assert summary["total_questions"] == 0
        assert summary["all_flags"] == []

    def test_missing_integrity_field(self):
        detector = _make_detector()
        responses = [{"question_id": 1}]
        # Should handle gracefully
        summary = detector.summarize_flags(responses)
        assert summary["total_questions"] == 1
        assert summary["clean_count"] == 1  # defaults to "clean"
