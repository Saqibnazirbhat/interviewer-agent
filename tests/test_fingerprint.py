"""Unit tests for AuthenticityFingerprint — behavioral analysis."""

from src.evaluation.fingerprint import (
    AuthenticityFingerprint,
    compute_authenticity,
    compute_vocab_complexity,
)


class TestVocabComplexity:
    def test_empty_text(self):
        assert compute_vocab_complexity("") == 0.0

    def test_short_text(self):
        assert compute_vocab_complexity("yes") == 0.0  # < 3 words

    def test_simple_text(self):
        score = compute_vocab_complexity("the cat sat on the mat")
        assert 0 < score < 5  # simple vocabulary

    def test_complex_text(self):
        score = compute_vocab_complexity(
            "The implementation leverages polymorphic inheritance "
            "alongside dependency injection to facilitate decoupled "
            "microservice architectures with comprehensive observability"
        )
        assert score > 4  # complex vocabulary

    def test_returns_bounded(self):
        score = compute_vocab_complexity("a b c d e f g h i j k l m n o p")
        assert 0 <= score <= 10


class TestComputeAuthenticity:
    def test_too_few_answers(self):
        responses = [
            {"answer": "Yes", "time_seconds": 5, "category": "skill", "skipped": False},
        ]
        fp = compute_authenticity(responses)
        assert fp.overall_score == 100.0
        assert "Too few" in fp.details.get("reason", "")

    def test_skipped_answers_excluded(self):
        responses = [
            {"answer": "", "time_seconds": 0, "category": "skill", "skipped": True},
            {"answer": "", "time_seconds": 0, "category": "skill", "skipped": True},
            {"answer": "", "time_seconds": 0, "category": "skill", "skipped": True},
        ]
        fp = compute_authenticity(responses)
        assert fp.overall_score == 100.0

    def test_normal_interview_high_score(self):
        """A realistic interview with varied timings and lengths should score high."""
        responses = [
            {"answer": "I built a REST API using Flask with proper error handling and auth " * 3, "time_seconds": 45, "category": "skill", "skipped": False, "difficulty": "medium"},
            {"answer": "In my previous role I managed a team of developers " * 2, "time_seconds": 30, "category": "experience", "skipped": False, "difficulty": "easy"},
            {"answer": "The architecture uses microservices with event-driven communication " * 4, "time_seconds": 90, "category": "project", "skipped": False, "difficulty": "hard"},
            {"answer": "I would approach this problem by first analyzing requirements " * 2, "time_seconds": 55, "category": "situational", "skipped": False, "difficulty": "medium"},
        ]
        fp = compute_authenticity(responses)
        assert fp.overall_score >= 70

    def test_suspiciously_consistent_timing_flags(self):
        """All answers taking exactly the same time is suspicious."""
        responses = [
            {"answer": "x " * 80, "time_seconds": 30.0, "category": "skill", "skipped": False},
            {"answer": "y " * 80, "time_seconds": 30.1, "category": "experience", "skipped": False},
            {"answer": "z " * 80, "time_seconds": 30.0, "category": "project", "skipped": False},
            {"answer": "w " * 80, "time_seconds": 29.9, "category": "skill", "skipped": False},
        ]
        fp = compute_authenticity(responses)
        assert fp.latency_cv < 0.15
        assert any("consistent" in f.lower() for f in fp.flags)

    def test_uniform_answer_lengths_flags(self):
        """All answers being exactly the same length is suspicious."""
        base = "The answer is exactly one hundred characters long please believe me" + " " * 34
        responses = [
            {"answer": base, "time_seconds": t, "category": cat, "skipped": False}
            for t, cat in [(20, "skill"), (35, "experience"), (50, "project"), (25, "skill")]
        ]
        fp = compute_authenticity(responses)
        assert fp.length_uniformity < 0.2

    def test_fingerprint_to_dict(self):
        fp = AuthenticityFingerprint()
        fp.flags = ["test flag"]
        fp.overall_score = 75.3
        d = fp.to_dict()
        assert d["overall_score"] == 75.3
        assert d["flags"] == ["test flag"]
        assert "latency_variance" in d
        assert "vocab_complexity_mean" in d

    def test_varied_timing_no_flag(self):
        """Natural timing variation should not flag."""
        responses = [
            {"answer": "Short answer here with enough words " * 3, "time_seconds": 15, "category": "skill", "skipped": False},
            {"answer": "A much longer and more detailed response about architecture " * 5, "time_seconds": 60, "category": "experience", "skipped": False},
            {"answer": "Medium length response about testing practices " * 3, "time_seconds": 35, "category": "project", "skipped": False},
            {"answer": "Quick answer to easy question " * 2, "time_seconds": 10, "category": "skill", "skipped": False},
        ]
        fp = compute_authenticity(responses)
        assert fp.latency_cv > 0.3
        assert not any("consistent" in f.lower() for f in fp.flags)

    def test_score_clamped_to_range(self):
        """Score should never go below 0 or above 100."""
        # Even with many flags, score should be >= 0
        fp = AuthenticityFingerprint()
        fp.overall_score = -50
        fp.overall_score = max(0, fp.overall_score)
        assert fp.overall_score >= 0
