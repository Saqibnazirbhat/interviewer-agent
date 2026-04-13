"""Unit tests for the AdaptiveEngine."""

from src.interview.adaptive import AdaptiveEngine, DIFFICULTY_ORDER


def _make_questions(n=6):
    """Create a simple set of test questions with mixed difficulties."""
    diffs = ["easy", "easy", "medium", "medium", "hard", "hard"]
    cats = ["experience", "skill", "project", "situational", "curveball", "experience"]
    return [
        {
            "id": i + 1,
            "category": cats[i % len(cats)],
            "difficulty": diffs[i % len(diffs)],
            "question": f"Test question {i + 1}",
            "context": "test",
            "ideal_signals": [f"signal_{i}"],
        }
        for i in range(n)
    ]


class TestAdaptiveEngineInit:
    def test_initial_state(self):
        qs = _make_questions()
        engine = AdaptiveEngine(qs)
        assert engine.performance == 5.0
        assert engine.target_difficulty == "medium"
        assert len(engine.remaining) == 6
        assert not engine.is_done

    def test_empty_pool(self):
        engine = AdaptiveEngine([])
        assert engine.is_done
        assert engine.remaining == []


class TestPickFirst:
    def test_picks_easy_or_medium(self):
        qs = _make_questions()
        engine = AdaptiveEngine(qs)
        first = engine.pick_first()
        assert first["difficulty"] in ("easy", "medium")
        assert first["adaptive_index"] == 0
        assert len(engine.remaining) == 5

    def test_falls_back_to_any_if_no_easy_medium(self):
        qs = [{"id": 1, "category": "skill", "difficulty": "hard", "question": "Q", "context": "", "ideal_signals": []}]
        engine = AdaptiveEngine(qs)
        first = engine.pick_first()
        assert first["difficulty"] == "hard"

    def test_marks_question_as_asked(self):
        qs = _make_questions()
        engine = AdaptiveEngine(qs)
        first = engine.pick_first()
        assert first["id"] in engine.asked_ids


class TestRecordScore:
    def test_normalizes_to_10(self):
        engine = AdaptiveEngine(_make_questions())
        engine.record_score(30, max_total=40)
        assert engine.scores == [7.5]

    def test_clamps_to_range(self):
        engine = AdaptiveEngine(_make_questions())
        engine.record_score(100, max_total=40)
        assert engine.scores == [10.0]
        engine.record_score(-5, max_total=40)
        assert engine.scores[-1] == 0.0

    def test_skip_records_low_score(self):
        engine = AdaptiveEngine(_make_questions())
        engine.record_skip()
        assert engine.scores == [2.0]


class TestPerformance:
    def test_neutral_start(self):
        engine = AdaptiveEngine(_make_questions())
        assert engine.performance == 5.0

    def test_single_score(self):
        engine = AdaptiveEngine(_make_questions())
        engine.scores = [8.0]
        assert engine.performance == 8.0

    def test_three_scores_simple_average(self):
        engine = AdaptiveEngine(_make_questions())
        engine.scores = [6.0, 7.0, 8.0]
        assert engine.performance == 7.0

    def test_four_scores_weighted(self):
        engine = AdaptiveEngine(_make_questions())
        engine.scores = [2.0, 8.0, 8.0, 8.0]
        # older: [2.0], recent: [8.0, 8.0, 8.0]
        # weighted = 2.0 + (8+8+8)*2 = 2+48 = 50
        # count = 1 + 3*2 = 7
        expected = 50 / 7
        assert abs(engine.performance - expected) < 0.01


class TestTargetDifficulty:
    def test_high_performance_targets_hard(self):
        engine = AdaptiveEngine(_make_questions())
        engine.scores = [9.0, 9.0, 9.0]
        assert engine.target_difficulty == "hard"

    def test_medium_performance_targets_medium(self):
        engine = AdaptiveEngine(_make_questions())
        engine.scores = [5.0, 6.0, 5.0]
        assert engine.target_difficulty == "medium"

    def test_low_performance_targets_easy(self):
        engine = AdaptiveEngine(_make_questions())
        engine.scores = [2.0, 3.0, 1.0]
        assert engine.target_difficulty == "easy"

    def test_boundary_7_5_is_hard(self):
        engine = AdaptiveEngine(_make_questions())
        engine.scores = [7.5]
        assert engine.target_difficulty == "hard"

    def test_boundary_4_5_is_medium(self):
        engine = AdaptiveEngine(_make_questions())
        engine.scores = [4.5]
        assert engine.target_difficulty == "medium"


class TestPickNext:
    def test_picks_matching_difficulty(self):
        qs = _make_questions()
        engine = AdaptiveEngine(qs)
        engine.pick_first()
        engine.scores = [9.0, 9.0, 9.0]  # high perf -> target hard
        next_q = engine.pick_next()
        assert next_q is not None
        # Should prefer hard questions
        assert next_q["difficulty"] == "hard"

    def test_avoids_same_category_twice(self):
        qs = [
            {"id": 1, "category": "skill", "difficulty": "medium", "question": "Q1", "context": "", "ideal_signals": []},
            {"id": 2, "category": "skill", "difficulty": "medium", "question": "Q2", "context": "", "ideal_signals": []},
            {"id": 3, "category": "experience", "difficulty": "medium", "question": "Q3", "context": "", "ideal_signals": []},
        ]
        engine = AdaptiveEngine(qs)
        first = engine.pick_first()
        assert first["category"] == "skill"
        engine.scores = [5.0]
        next_q = engine.pick_next()
        # Should prefer experience over skill (variety bonus)
        assert next_q["category"] == "experience"

    def test_returns_none_when_empty(self):
        qs = [{"id": 1, "category": "skill", "difficulty": "easy", "question": "Q", "context": "", "ideal_signals": []}]
        engine = AdaptiveEngine(qs)
        engine.pick_first()
        assert engine.pick_next() is None

    def test_exhausts_all_questions(self):
        qs = _make_questions(4)
        engine = AdaptiveEngine(qs)
        engine.pick_first()
        engine.scores = [5.0]
        engine.pick_next()
        engine.scores.append(5.0)
        engine.pick_next()
        engine.scores.append(5.0)
        last = engine.pick_next()
        assert last is not None
        assert engine.is_done


class TestGetStatus:
    def test_status_structure(self):
        engine = AdaptiveEngine(_make_questions())
        engine.pick_first()
        engine.record_score(20)
        status = engine.get_status()
        assert "questions_asked" in status
        assert "questions_remaining" in status
        assert "rolling_performance" in status
        assert "target_difficulty" in status
        assert "scores" in status
        assert status["questions_asked"] == 1
        assert status["questions_remaining"] == 5
