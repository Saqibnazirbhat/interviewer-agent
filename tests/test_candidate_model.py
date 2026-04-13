"""Unit tests for CandidateModel — cross-answer reasoning."""

from unittest.mock import MagicMock, patch

from src.interview.candidate_model import CandidateModel


def _make_model():
    """Create a CandidateModel with mocked LLM."""
    model = CandidateModel()
    model._llm = MagicMock()
    return model


class TestRecordAnswer:
    def test_skipped_answer(self):
        model = _make_model()
        q = {"id": 1, "category": "skill", "question": "Q1"}
        model.record_answer(q, "", skipped=True)
        assert len(model.answer_history) == 1
        assert model.answer_history[0]["skipped"] is True
        assert model.confidence_pattern["skill"] == "weak"

    def test_successful_extraction(self):
        model = _make_model()
        model._llm.generate_json.return_value = {
            "summary": "Candidate described API design",
            "claims": ["Led a team of 5", "Built the auth system"],
            "skills_demonstrated": ["API design", "authentication"],
            "contradictions": [],
        }
        q = {"id": 1, "category": "skill", "question": "Describe your work"}
        model.record_answer(q, "I led a team of 5 building an auth system", scores={"accuracy": 8, "depth": 7, "communication": 8, "ownership": 9})

        assert len(model.answer_history) == 1
        assert len(model.claims) == 2
        assert "API design" in model.demonstrated_skills
        assert model.confidence_pattern["skill"] == "strong"

    def test_contradiction_detected(self):
        model = _make_model()
        # First answer
        model._llm.generate_json.return_value = {
            "summary": "Built it alone",
            "claims": ["Built the entire system alone"],
            "skills_demonstrated": [],
            "contradictions": [],
        }
        model.record_answer({"id": 1, "category": "experience", "question": "Q1"}, "I built it alone")

        # Second answer with contradiction
        model._llm.generate_json.return_value = {
            "summary": "Team effort",
            "claims": ["My team of 3 built it"],
            "skills_demonstrated": [],
            "contradictions": [{
                "earlier_claim": "Built the entire system alone",
                "earlier_question_id": 1,
                "new_claim": "My team of 3 built it",
                "explanation": "Contradicts solo claim",
            }],
        }
        model.record_answer({"id": 2, "category": "project", "question": "Q2"}, "My team of 3 built it")

        assert len(model.contradictions) == 1
        assert model.contradictions[0]["question_id_a"] == 1
        assert model.contradictions[0]["question_id_b"] == 2

    def test_llm_failure_graceful(self):
        model = _make_model()
        model._llm.generate_json.side_effect = Exception("API error")
        q = {"id": 1, "category": "skill", "question": "Q1"}
        model.record_answer(q, "Some answer text here")
        # Should not crash, should still record
        assert len(model.answer_history) == 1
        assert model.answer_history[0]["summary"] == "Some answer text here"[:100]

    def test_weak_confidence_on_low_scores(self):
        model = _make_model()
        model._llm.generate_json.return_value = {
            "summary": "Weak answer",
            "claims": [],
            "skills_demonstrated": [],
            "contradictions": [],
        }
        q = {"id": 1, "category": "skill", "question": "Q1"}
        model.record_answer(q, "Uh I think maybe...", scores={"accuracy": 3, "depth": 2, "communication": 3, "ownership": 2})
        assert model.confidence_pattern["skill"] == "weak"


class TestContextGeneration:
    def test_empty_context(self):
        model = _make_model()
        assert model.get_context_for_scoring() == ""
        assert model.get_context_for_followup() == ""

    def test_scoring_context_includes_history(self):
        model = _make_model()
        model.answer_history = [
            {"question_id": 1, "category": "skill", "skipped": False, "summary": "Described API work"},
        ]
        model.demonstrated_skills = ["Python", "REST APIs"]
        ctx = model.get_context_for_scoring()
        assert "CROSS-ANSWER CONTEXT" in ctx
        assert "Described API work" in ctx
        assert "Python" in ctx

    def test_scoring_context_includes_contradictions(self):
        model = _make_model()
        model.answer_history = [{"question_id": 1, "category": "skill", "skipped": False, "summary": "X"}]
        model.contradictions = [{
            "claim_a": "I built it alone",
            "claim_b": "My team built it",
            "question_id_a": 1,
            "question_id_b": 2,
            "explanation": "Contradicts solo claim",
        }]
        ctx = model.get_context_for_scoring()
        assert "CONTRADICTIONS" in ctx
        assert "I built it alone" in ctx

    def test_followup_context_includes_claims(self):
        model = _make_model()
        model.answer_history = [
            {"question_id": 1, "category": "skill", "skipped": False, "summary": "API work"},
        ]
        model.claims = [{"text": "Has 5 years React experience", "source_question_id": 1, "category": "skill"}]
        ctx = model.get_context_for_followup()
        assert "INTERVIEW HISTORY" in ctx
        assert "5 years React" in ctx


class TestSerialization:
    def test_to_dict_and_from_dict(self):
        model = _make_model()
        model.claims = [{"text": "Claim 1", "source_question_id": 1, "category": "skill"}]
        model.contradictions = [{"claim_a": "A", "claim_b": "B", "question_id_a": 1, "question_id_b": 2, "explanation": ""}]
        model.demonstrated_skills = ["Python"]
        model.confidence_pattern = {"skill": "strong"}
        model.answer_history = [{"question_id": 1, "category": "skill", "skipped": False, "summary": "Test"}]

        data = model.to_dict()
        restored = CandidateModel.from_dict(data)

        assert len(restored.claims) == 1
        assert len(restored.contradictions) == 1
        assert "Python" in restored.demonstrated_skills
        assert restored.confidence_pattern["skill"] == "strong"
        assert len(restored.answer_history) == 1

    def test_from_dict_empty(self):
        restored = CandidateModel.from_dict({})
        assert restored.claims == []
        assert restored.demonstrated_skills == []
        assert restored.answer_history == []


class TestGetSummary:
    def test_summary_structure(self):
        model = _make_model()
        model.claims = [{"text": "C1"}, {"text": "C2"}]
        model.contradictions = [{"claim_a": "A", "claim_b": "B"}]
        model.demonstrated_skills = ["Python"]
        model.answer_history = [
            {"skipped": False},
            {"skipped": True},
        ]

        summary = model.get_summary()
        assert summary["total_claims"] == 2
        assert summary["contradiction_count"] == 1
        assert summary["demonstrated_skills"] == ["Python"]
        assert summary["answer_count"] == 2
        assert summary["skipped_count"] == 1
