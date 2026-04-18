"""Integration tests for the /answer endpoint slot-counting flow.

Verifies that:
  * follow-ups consume question slots
  * the interview ends at TOTAL_QUESTION_SLOTS regardless of follow-up frequency
  * generated questions never asked are NOT padded as skipped when the cap is hit
"""

import os
import secrets

# Encryption key must be set BEFORE importing app modules
os.environ.setdefault("DATA_ENCRYPTION_KEY", secrets.token_urlsafe(32))
os.environ.setdefault("OWNER_PASSWORD", "test-owner-pwd")
os.environ.setdefault("NVIDIA_API_KEY", "test-key-not-used")

from unittest.mock import MagicMock, patch

import pytest


def _make_session(question_count: int = 20):
    """Build a fake interview session ready for /answer to consume."""
    questions = [
        {
            "id": i + 1,
            "category": "experience",
            "question": f"Question {i + 1}?",
            "context": "test",
            "difficulty": "medium",
            "ideal_signals": ["s"],
        }
        for i in range(question_count)
    ]

    from src.interview.adaptive import AdaptiveEngine
    engine = AdaptiveEngine(questions)
    engine.pick_first()  # mark Q1 as asked

    return {
        "questions": questions,
        "adaptive_engine": engine,
        "responses": [],
        "state": "interviewing",
        "profile": {"name": "Test", "username": "test"},
        "persona": {
            "name": "Technical Expert",
            "short_name": "Technical Expert",
            "description": "test",
            "scoring_bias": "test",
        },
        "session_token_hash": "x" * 64,
    }


@pytest.fixture
def client_and_store():
    """Spin up the FastAPI TestClient with all LLM-touching paths mocked."""
    # Patch the scorer + cheat detector + follow-up generator so /answer never hits the network
    with patch("src.evaluation.scorer.AnswerScorer") as Scorer, \
         patch("src.evaluation.cheat_detector.CheatDetector") as Detector, \
         patch("src.interview.followup.FollowUpGenerator") as FollowupCls:

        scorer_inst = MagicMock()
        scorer_inst.score_all.side_effect = lambda responses, *_a, **_kw: [
            dict(r, scores={"accuracy": 7, "depth": 7, "communication": 7, "ownership": 7},
                 score_total=28, feedback="ok") for r in responses
        ]
        Scorer.return_value = scorer_inst

        detector_inst = MagicMock()
        detector_inst.check_all.side_effect = lambda responses, *_a, **_kw: [
            r.update({"integrity": {"verdict": "clean", "flags": []}}) or r for r in responses
        ]
        Detector.return_value = detector_inst

        # Follow-up generator: ALWAYS issues a follow-up (we test the cap holds)
        fg_inst = MagicMock()
        fg_inst.should_followup.return_value = {
            "action": "dig_deeper",
            "followup_question": "Tell me more about that.",
            "reason": "depth probe",
        }
        FollowupCls.return_value = fg_inst

        from fastapi.testclient import TestClient
        from src.web.app import app, store

        client = TestClient(app)
        yield client, store


def _run(coro):
    """Run a coroutine in a fresh event loop (Py3.14 doesn't auto-create one)."""
    import asyncio
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _seed_session(store, sid: str, session_dict):
    """Drop a pre-built session into the store synchronously."""
    _run(store.put(sid, session_dict))


def _bypass_token(monkeypatch):
    """Make _require_session_token a no-op for these tests."""
    from src.web import app as app_mod
    monkeypatch.setattr(app_mod, "_require_session_token", lambda *a, **kw: None)


def test_followups_consume_slots(client_and_store, monkeypatch):
    """20 saved responses (mix of main + follow-ups) ends the interview."""
    client, store = client_and_store
    _bypass_token(monkeypatch)

    sid = "test-session-1"
    sess = _make_session(question_count=20)
    _seed_session(store, sid, sess)

    payload_template = {
        "session_id": sid,
        "session_token": "irrelevant",
        "answer": "A reasonably long answer with concrete details about my work.",
        "time_seconds": 40.0,
        "skipped": False,
        "used_hint": False,
        "is_followup": False,
        "followup_question_text": "",
        "timing_status": "early",
        "question_id": 1,
        "question_index": 0,
    }

    seen_slots = []
    done = False

    for step in range(25):
        resp = client.post("/answer", json=payload_template)
        assert resp.status_code == 200, f"step {step}: {resp.text}"
        data = resp.json()
        if data.get("done"):
            done = True
            break

        if "followup" in data:
            payload_template["is_followup"] = True
            payload_template["followup_question_text"] = data["followup"]["question"]
            payload_template["question_id"] = data["followup"]["original_question_id"]
            seen_slots.append(("fu", data["followup"]["index"]))
        elif "next_question" in data:
            payload_template["is_followup"] = False
            payload_template["followup_question_text"] = ""
            payload_template["question_id"] = data["next_question"]["id"]
            payload_template["question_index"] = data["next_question"]["index"]
            seen_slots.append(("main", data["next_question"]["index"]))

    assert done, "Interview never ended despite 25 /answer calls"

    final_session = _run(store.get(sid))
    assert len(final_session["responses"]) == 20, \
        f"Expected exactly 20 saved responses (slot cap), got {len(final_session['responses'])}"

    # Slot indices returned should be strictly increasing and contiguous from 1
    indices = [s[1] for s in seen_slots]
    assert indices == sorted(indices), "Slot indices must be monotonically increasing"
    assert indices[0] == 1, f"First served slot after Q1 should be 1 (0-based), got {indices[0]}"


def test_no_followup_when_at_cap(client_and_store, monkeypatch):
    """At slot 20, no follow-up should be issued — interview just ends."""
    client, store = client_and_store
    _bypass_token(monkeypatch)

    sid = "test-session-2"
    sess = _make_session(question_count=20)
    # Pre-fill 19 responses so the next /answer hits the cap
    sess["responses"] = [
        {"question_id": i + 1, "category": "experience", "question": "q", "answer": "a",
         "skipped": False, "time_seconds": 30, "scores": {"accuracy": 7, "depth": 7,
         "communication": 7, "ownership": 7}, "score_total": 28}
        for i in range(19)
    ]
    _seed_session(store, sid, sess)

    payload = {
        "session_id": sid,
        "session_token": "irrelevant",
        "answer": "Final answer with substance.",
        "time_seconds": 40.0,
        "skipped": False,
        "used_hint": False,
        "is_followup": False,
        "followup_question_text": "",
        "timing_status": "early",
        "question_id": 1,
        "question_index": 0,
    }
    resp = client.post("/answer", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("done") is True, f"Should be done at slot 20, got {data}"
    assert "followup" not in data, "Must not issue follow-up at the cap"
    assert "next_question" not in data
