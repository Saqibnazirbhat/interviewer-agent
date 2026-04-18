"""End-to-end driver: walk through a complete 20-slot interview against the
running FastAPI app with all LLM calls mocked. Run with:
    python tests/drive_test_interview.py
Prints a per-step trace showing which slots were main questions vs follow-ups,
scores, and final completion state.
"""

import os
import secrets
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("DATA_ENCRYPTION_KEY", secrets.token_urlsafe(32))
os.environ.setdefault("OWNER_PASSWORD", "test-owner-pwd")
os.environ.setdefault("NVIDIA_API_KEY", "test-key")

import asyncio
import random
from unittest.mock import MagicMock, patch


SAMPLE_ANSWERS = [
    "I built a REST API in Flask that handled 5k QPS by adding Redis caching and async workers. The hardest part was getting the cache invalidation right when multiple writers hit the same key.",
    "At my last role I led a team of four engineers migrating a legacy Django monolith to FastAPI microservices over six months. We cut p99 latency from 800ms to 140ms.",
    "For that project I used Postgres with logical replication for the read replicas, and we added pgBouncer in front to handle connection pressure during peak hours.",
    "I would start by profiling — usually with py-spy or cProfile — then look at the hot path. In one case I found a quadratic loop that was killing us under load.",
    "Honestly the trickiest part was deciding what NOT to build. We deferred the analytics rewrite for two quarters because the data pipeline was already paying down debt elsewhere.",
    "Sure — I'd push back gently and ask for the specific use case. Most performance complaints are either measurement noise or a single bad query, and I want to see numbers first.",
    "I used pytest with parametrize for the table-driven cases and a fixture that spun up a real Postgres in a container. Mocking the DB caught fewer bugs than the slower real-DB tests.",
    "We chose Kafka over RabbitMQ because the consumer rebalance semantics fit our partitioned workload better and we already had Confluent on staff.",
    "Yeah — I once shipped a bug that double-charged about 200 users. Caught it within an hour from our metrics, rolled back, refunded everyone, and wrote a post-mortem the same day.",
    "I'd say the biggest growth was learning to write smaller PRs. Earlier I'd send 800-line diffs that no one could review well.",
    "For state machines I usually reach for python-statemachine or just a plain dict-of-dicts if the graph is small. The library's overhead is rarely worth it for less than ten states.",
    "I'd interview the on-call engineers first — they know where the pain is. Then look at the postmortem archive for repeated themes.",
    "If forced to pick one: observability. You can recover from almost any other gap if you can see what's happening, and you can't fix what you can't see.",
    "I prefer Go for that kind of CLI tool — fast startup, single binary, the standard library has everything you need for argparse-style work.",
    "A teammate disagreed strongly on using gRPC vs REST. We benchmarked both, found the difference was 8ms, and went with REST because debugging was easier for the team.",
    "I usually write the README first — if I can't explain it cleanly to a future reader, the design probably isn't ready.",
    "Probably the time I refactored a class hierarchy three times in one week before realizing the original was fine. I learned to write down the WHY before touching the WHAT.",
    "Sure — I'd suggest pair programming for two weeks, then revisit. Most onboarding friction comes from missing context, not missing skill.",
    "I'd default to a feature flag and a canary deploy for any change touching billing. Cost of a bad rollout there is too high to push it directly.",
    "If I had a free week? Probably build out our chaos testing — we have unit + integration but nothing that exercises real failure modes.",
]

INDEX_OF_FOLLOWUP_ANSWERS = [
    "Specifically, I tracked the cache hit rate by partition and added a per-key TTL jitter — without that we got a thundering herd every five minutes.",
    "The team structure mattered too — I paired a senior with each junior for the first month, which cut the ramp-up time roughly in half.",
    "We picked logical over streaming replication because of the schema-evolution flexibility — we ship migrations weekly and didn't want replica lag during DDL.",
    "Right — I picked py-spy because it sampled at 1ms granularity without needing code changes, which was important since the bug was in production traffic.",
    "Concretely we decided based on payback period — anything over 18 months we deferred unless legal forced our hand.",
]


def build_session(question_count: int = 20):
    questions = [
        {"id": i + 1, "category": "experience", "question": f"Question {i+1}: tell me about your work.",
         "context": "test", "difficulty": "medium", "ideal_signals": ["concrete example"]}
        for i in range(question_count)
    ]
    from src.interview.adaptive import AdaptiveEngine
    engine = AdaptiveEngine(questions)
    engine.pick_first()
    return {
        "questions": questions,
        "adaptive_engine": engine,
        "responses": [],
        "state": "interviewing",
        "profile": {"name": "Test Candidate", "username": "testuser", "skills": ["Python", "Go"]},
        "persona": {
            "name": "Technical Expert",
            "short_name": "Technical Expert",
            "description": "Deep technical interviewer.",
            "scoring_bias": "Reward concrete examples.",
        },
        "_token_hash": "x" * 64,
    }


def run():
    random.seed(42)

    # Capture the real AnswerScorer class before patching so we can compute
    # the summary with the real (non-mocked) implementation at the end.
    from src.evaluation.scorer import AnswerScorer as RealScorer

    # Mock anything that would call the LLM
    with patch("src.evaluation.scorer.AnswerScorer") as Scorer, \
         patch("src.evaluation.cheat_detector.CheatDetector") as Detector, \
         patch("src.interview.followup.FollowUpGenerator") as FollowupCls, \
         patch("src.interview.candidate_model.CandidateModel") as CandidateCls:

        # CandidateModel stub: no-op record_answer, empty context strings
        cm_inst = MagicMock()
        cm_inst.record_answer.return_value = None
        cm_inst.get_context_for_scoring.return_value = ""
        cm_inst.get_context_for_followup.return_value = ""
        cm_inst.to_dict.return_value = {}
        CandidateCls.return_value = cm_inst
        CandidateCls.from_dict.return_value = cm_inst

        # Realistic varied scoring: 5-9 per dimension
        def score_side_effect(responses, *_a, **_kw):
            for r in responses:
                if r.get("skipped"):
                    r["scores"] = {"accuracy": 0, "depth": 0, "communication": 0, "ownership": 0}
                    r["score_total"] = 0
                    r["feedback"] = "Skipped."
                else:
                    s = {
                        "accuracy": random.randint(6, 9),
                        "depth": random.randint(5, 8),
                        "communication": random.randint(6, 9),
                        "ownership": random.randint(5, 9),
                    }
                    r["scores"] = s
                    r["score_total"] = sum(s.values())
                    r["feedback"] = "Solid concrete example."
            return responses

        scorer_inst = MagicMock()
        scorer_inst.score_all.side_effect = score_side_effect
        Scorer.return_value = scorer_inst

        detector_inst = MagicMock()
        def detect_side_effect(responses, *_a, **_kw):
            for r in responses:
                r["integrity"] = {"verdict": "clean", "flags": []}
            return responses
        detector_inst.check_all.side_effect = detect_side_effect
        Detector.return_value = detector_inst

        # Follow-up generator: fires roughly 1 in 3 to mix things up
        fg_inst = MagicMock()
        fu_counter = {"n": 0}
        def fu_side_effect(question, answer, score, time_seconds, cross_answer_context=""):
            fu_counter["n"] += 1
            if fu_counter["n"] % 3 == 0:
                return {"action": "dig_deeper",
                        "followup_question": "Can you give me one specific detail you didn't mention?",
                        "reason": "wants depth"}
            return {"action": "move_on", "followup_question": "", "reason": "answer was complete"}
        fg_inst.should_followup.side_effect = fu_side_effect
        FollowupCls.return_value = fg_inst

        from fastapi.testclient import TestClient
        from src.web import app as app_mod
        # Bypass token auth for the simulation
        app_mod._require_session_token = lambda *a, **kw: None

        client = TestClient(app_mod.app)
        sid = "drive-test-session"
        sess = build_session(20)
        asyncio.new_event_loop().run_until_complete(app_mod.store.put(sid, sess))

        print(f"\n{'='*70}")
        print(f"  SIMULATED INTERVIEW — 20 slots, follow-ups counted against cap")
        print(f"{'='*70}\n")

        payload = {
            "session_id": sid, "session_token": "n/a",
            "answer": SAMPLE_ANSWERS[0],
            "time_seconds": 55.0, "skipped": False, "used_hint": False,
            "is_followup": False, "followup_question_text": "",
            "timing_status": "early", "question_id": 1, "question_index": 0,
        }

        main_count = 0
        fu_count = 0
        next_main_idx = 1  # next answer index from SAMPLE_ANSWERS for main questions
        next_fu_idx = 0    # next answer index from INDEX_OF_FOLLOWUP_ANSWERS for follow-ups

        for step in range(40):
            ctx = "FOLLOW-UP" if payload["is_followup"] else "MAIN"
            print(f"  Step {step+1:2d}  [{ctx:9s}]  Q-id={payload['question_id']:2d}  "
                  f"answer_len={len(payload['answer']):3d}  time={payload['time_seconds']:.0f}s")
            r = client.post("/answer", json=payload)
            assert r.status_code == 200, f"FAILED: {r.text}"
            data = r.json()

            if payload["is_followup"]:
                fu_count += 1
            else:
                main_count += 1

            if data.get("done"):
                print(f"\n  >>> Interview ENDED. main={main_count}, followups={fu_count}, "
                      f"total={main_count+fu_count}")
                break

            if "followup" in data:
                fu = data["followup"]
                payload["is_followup"] = True
                payload["followup_question_text"] = fu["question"]
                payload["question_id"] = fu["original_question_id"]
                payload["answer"] = INDEX_OF_FOLLOWUP_ANSWERS[next_fu_idx % len(INDEX_OF_FOLLOWUP_ANSWERS)]
                payload["time_seconds"] = float(random.randint(35, 80))
                next_fu_idx += 1
                print(f"     -> FOLLOW-UP issued, slot index {fu['index']}")
            elif "next_question" in data:
                nq = data["next_question"]
                payload["is_followup"] = False
                payload["followup_question_text"] = ""
                payload["question_id"] = nq["id"]
                payload["question_index"] = nq["index"]
                payload["answer"] = SAMPLE_ANSWERS[next_main_idx % len(SAMPLE_ANSWERS)]
                payload["time_seconds"] = float(random.randint(40, 95))
                next_main_idx += 1
                print(f"     -> NEXT MAIN Q{nq['id']}, slot index {nq['index']}")

        # Read the final session and report scores
        final = asyncio.new_event_loop().run_until_complete(app_mod.store.get(sid))
        responses = final["responses"]

        print(f"\n  {'-'*68}")
        print(f"  RESPONSES SAVED: {len(responses)}")
        main_only = [r for r in responses if not r.get("is_followup")]
        fu_only   = [r for r in responses if r.get("is_followup")]
        print(f"     main: {len(main_only)}   followups: {len(fu_only)}")

        # Use the captured real scorer class; compute_summary is pure math.
        with patch("src.evaluation.scorer.LLMClient"):
            real = RealScorer({"name": "Technical Expert", "scoring_bias": ""})
            summary = real.compute_summary(responses)
        print(f"\n  SCORES")
        print(f"     answered:   {summary['answered_count']}")
        print(f"     skipped:    {summary['skipped_count']}")
        print(f"     raw quality:    {summary.get('raw_quality')}")
        print(f"     participation:  {summary.get('participation_rate')}")
        print(f"     OVERALL:        {summary['overall']}/10")
        print(f"     by dimension:   {summary['by_dimension']}")
        print(f"\n  {'='*68}\n")

        # Assertions — the contract
        assert len(responses) == 20, f"slot cap broken: got {len(responses)} responses"
        assert fu_count >= 1, "no follow-ups fired during simulation"
        assert main_count + fu_count == 20, f"step count != 20: main={main_count} fu={fu_count}"
        print("  PASSED: 20 slots used, follow-ups counted, scoring computed.")


if __name__ == "__main__":
    run()
