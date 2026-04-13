"""FastAPI backend for the Interviewer Agent browser UI."""

import asyncio
import logging
import logging.handlers
import os
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import hashlib
import hmac
import secrets

from fastapi import Cookie, Response

from src.web.models import (
    EnrichGitHubRequest,
    GitHubRequest,
    OwnerLoginRequest,
    ReportRequest,
    StartInterviewRequest,
    SubmitAnswerRequest,
)
from src.llm_client import get_active_model, set_active_model
from src.web.results_store import ResultsStore
from src.web.session_store import SessionStore

load_dotenv()

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            LOG_DIR / "interviewer.log",
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3,
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("interviewer.web")

store = SessionStore()
results_store = ResultsStore()

# ---------------------------------------------------------------------------
# Rate limiting — simple in-memory token bucket per IP
# ---------------------------------------------------------------------------
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB

_rate_buckets: dict[str, list] = defaultdict(list)
RATE_LIMIT = 30  # max requests per window
RATE_WINDOW = 60  # seconds


_rate_cleanup_counter = 0

def _check_rate_limit(client_ip: str):
    """Raise 429 if the client exceeds the rate limit."""
    global _rate_cleanup_counter
    now = time.time()
    bucket = _rate_buckets[client_ip]
    # Prune old entries for this IP
    _rate_buckets[client_ip] = [t for t in bucket if now - t < RATE_WINDOW]
    if len(_rate_buckets[client_ip]) >= RATE_LIMIT:
        raise HTTPException(429, "Too many requests. Please slow down.")
    _rate_buckets[client_ip].append(now)
    # Periodically purge stale IPs to prevent unbounded growth
    _rate_cleanup_counter += 1
    if _rate_cleanup_counter >= 100:
        _rate_cleanup_counter = 0
        stale = [ip for ip, ts in _rate_buckets.items() if not ts or now - ts[-1] > RATE_WINDOW]
        for ip in stale:
            del _rate_buckets[ip]


# ---------------------------------------------------------------------------
# Owner authentication — cookie-based session with 24h expiry
# ---------------------------------------------------------------------------
OWNER_PASSWORD = os.getenv("OWNER_PASSWORD", "")
OWNER_COOKIE_NAME = "owner_session"
OWNER_SESSION_TTL = 24 * 60 * 60  # 24 hours

# In-memory store for valid owner session tokens: token -> expiry timestamp
_owner_sessions: dict[str, float] = {}


def _create_owner_session() -> str:
    """Create a new owner session token."""
    token = secrets.token_urlsafe(32)
    _owner_sessions[token] = time.time() + OWNER_SESSION_TTL
    return token


def _verify_owner_session(token: str | None) -> bool:
    """Check if an owner session token is valid and not expired."""
    if not token or token not in _owner_sessions:
        return False
    now = time.time()
    if now > _owner_sessions[token]:
        _owner_sessions.pop(token, None)
        return False
    # Periodically purge all expired tokens (every ~20 checks)
    if len(_owner_sessions) > 5:
        expired = [t for t, exp in _owner_sessions.items() if now > exp]
        for t in expired:
            _owner_sessions.pop(t, None)
    return True


def _require_owner(owner_session: str | None):
    """Raise 401 if the owner session is invalid."""
    if not _verify_owner_session(owner_session):
        raise HTTPException(401, "Owner authentication required.")


# ---------------------------------------------------------------------------
# Async wrapper for blocking calls (LLM, GitHub API, file parsing)
# ---------------------------------------------------------------------------
async def run_sync(fn, *args, **kwargs):
    """Run a blocking function in the thread pool to avoid blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(fn, *args, **kwargs))


async def _periodic_cleanup():
    """Background task that cleans expired sessions and uploads every hour."""
    while True:
        await asyncio.sleep(3600)  # every hour
        try:
            expired = await store.cleanup_expired()
            if expired:
                logger.info("Periodic cleanup: removed %d expired sessions", expired)
            # Clean old uploaded files (older than 24h)
            cutoff = time.time() - 24 * 60 * 60
            for f in UPLOAD_DIR.iterdir():
                if f.is_file() and f.stat().st_mtime < cutoff:
                    f.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("Periodic cleanup error: %s", exc)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Startup/shutdown lifecycle — clean expired sessions on boot."""
    expired = await store.cleanup_expired()
    if expired:
        logger.info("Cleaned %d expired sessions on startup", expired)
    logger.info("Interviewer Agent started — listening on http://localhost:8000")
    cleanup_task = asyncio.create_task(_periodic_cleanup())
    yield
    cleanup_task.cancel()
    logger.info("Interviewer Agent shutting down")


app = FastAPI(title="Interviewer Agent", version="2.0", lifespan=lifespan)

# CORS — locked to same-origin and localhost dev
ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    allow_credentials=True,
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting and request logging."""
    client_ip = request.client.host if request.client else "unknown"
    # Skip rate limiting for static files
    if not request.url.path.startswith("/static"):
        try:
            _check_rate_limit(client_ip)
        except HTTPException as exc:
            logger.warning("Rate limited %s on %s", client_ip, request.url.path)
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start

    # Log non-static requests
    if not request.url.path.startswith("/static"):
        logger.info(
            "%s %s %d %.1fms [%s]",
            request.method, request.url.path, response.status_code,
            elapsed * 1000, client_ip,
        )
    return response

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Mount static files
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    """Serve the main HTML page."""
    index_path = STATIC_DIR / "index.html"
    return FileResponse(str(index_path))


@app.get("/health")
async def health():
    """Simple health check."""
    active = await store.count()
    return {"status": "ok", "service": "interviewer-agent", "active_sessions": active}


@app.post("/upload")
async def upload_resume(file: UploadFile = File(...)):
    """Accept a resume file, parse it, detect role, return candidate profile."""
    try:
        # Validate file extension
        filename = file.filename or "resume.txt"
        ext = Path(filename).suffix.lower()
        if ext not in {".pdf", ".docx", ".txt"}:
            raise HTTPException(400, f"Unsupported file type '{ext}'. Use PDF, DOCX, or TXT.")

        # Read and validate file size
        content = await file.read()
        await file.close()
        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(400, f"File too large ({len(content) // 1024 // 1024}MB). Maximum is 10MB.")

        # Save uploaded file
        save_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{filename}"
        save_path.write_bytes(content)

        # Parse resume (blocking I/O → run in thread pool)
        from src.ingestion.resume_parser import ResumeParser
        parser = ResumeParser()
        profile = await run_sync(parser.parse_file, str(save_path))

        # Remove large fields from response
        response_profile = {k: v for k, v in profile.items() if k != "resume_text"}
        response_profile["_resume_text_length"] = len(profile.get("resume_text", ""))

        # Store full profile in session
        session_id = uuid.uuid4().hex
        await store.put(session_id, {"profile": profile, "state": "profile_ready"})
        response_profile["session_id"] = session_id

        return JSONResponse(response_profile)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Failed to process resume: {str(exc)}")


@app.post("/github")
async def github_profile(data: GitHubRequest):
    """Fetch GitHub profile and return enriched profile JSON."""
    try:
        from src.ingestion.github_fetcher import GitHubFetcher
        fetcher = GitHubFetcher()
        profile = await run_sync(fetcher.fetch_profile, data.username)
        profile["source"] = "github"

        session_id = uuid.uuid4().hex
        await store.put(session_id, {"profile": profile, "state": "profile_ready"})

        response = dict(profile)
        response["session_id"] = session_id
        # Map GitHub profile fields to match the UI expectations
        response.setdefault("detected_role", "Software Engineer")
        response.setdefault("industry", "Technology")
        # Derive seniority from account age
        years = profile.get("account_age_years", 0)
        if years >= 10:
            seniority = "Senior / Staff"
        elif years >= 5:
            seniority = "Mid Level"
        elif years >= 2:
            seniority = "Junior"
        else:
            seniority = "Entry Level"
        response.setdefault("seniority", seniority)
        response.setdefault("years_of_experience", profile.get("account_age_years", 0))
        response.setdefault("skills", list(profile.get("languages", {}).keys()))

        # Top projects from repos
        projects = []
        for repo in profile.get("top_repos", [])[:3]:
            projects.append({
                "name": repo["name"],
                "description": repo.get("description", ""),
                "technologies_or_tools": list(repo.get("languages", {}).keys()),
            })
        response.setdefault("projects", projects)

        return JSONResponse(response)

    except (EnvironmentError, ValueError) as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Failed to fetch GitHub profile: {str(exc)}")


@app.post("/enrich")
async def enrich_with_github(data: EnrichGitHubRequest):
    """Merge GitHub data into an existing resume-based session profile."""
    try:
        session = await store.get(data.session_id)
        if session is None:
            raise HTTPException(400, "Invalid or expired session.")

        from src.ingestion.github_fetcher import GitHubFetcher
        fetcher = GitHubFetcher()
        github_data = await run_sync(fetcher.fetch_profile, data.username)

        profile = session["profile"]

        # Store full GitHub profile for the question generator's github_context block
        profile["github_profile"] = github_data

        # Merge GitHub repos into projects if resume had few/none
        if len(profile.get("projects", [])) < 2:
            for repo in github_data.get("top_repos", [])[:3]:
                profile.setdefault("projects", []).append({
                    "name": repo["name"],
                    "description": repo.get("description", ""),
                    "technologies_or_tools": list(repo.get("languages", {}).keys()),
                })

        # Merge GitHub languages into skills
        gh_languages = list(github_data.get("languages", {}).keys())
        existing_skills = [s.lower() for s in profile.get("skills", [])]
        for lang in gh_languages:
            if lang.lower() not in existing_skills:
                profile.setdefault("skills", []).append(lang)

        # Store username for reference
        profile["github_username"] = data.username

        session["profile"] = profile
        await store.put(data.session_id, session)

        return JSONResponse({
            "status": "enriched",
            "github_username": data.username,
            "repos_added": len(github_data.get("top_repos", [])),
            "languages": gh_languages,
        })

    except HTTPException:
        raise
    except (EnvironmentError, ValueError) as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Failed to enrich with GitHub: {str(exc)}")


@app.post("/start")
async def start_interview(data: StartInterviewRequest):
    """Start an interview session."""
    try:
        session = await store.get(data.session_id)
        if session is None:
            raise HTTPException(400, "Invalid or expired session. Please start over.")

        profile = session["profile"]

        # Apply any profile overrides from the user
        if data.detected_role:
            profile["detected_role"] = data.detected_role
        if data.industry:
            profile["industry"] = data.industry
        if data.achievement_description:
            profile["achievement_description"] = data.achievement_description

        # Get persona adapted to industry
        persona_id = data.persona_id
        industry = profile.get("industry", "")
        detected_role = profile.get("detected_role", "")

        from src.interview.personas import get_persona
        persona = get_persona(persona_id, industry=industry, detected_role=detected_role)

        # Generate questions (blocking LLM call → thread pool)
        from src.interview.question_generator import QuestionGenerator
        generator = QuestionGenerator()
        questions = await run_sync(generator.generate, profile, persona)

        # Generate intro
        intro = await run_sync(generator.generate_intro, profile, persona)

        # Initialize adaptive engine
        from src.interview.adaptive import AdaptiveEngine
        engine = AdaptiveEngine(questions)
        first_q = engine.pick_first()

        # Store session state
        session["persona"] = persona
        session["questions"] = questions
        session["adaptive_engine"] = engine
        session["responses"] = []
        session["state"] = "interviewing"
        await store.put(data.session_id, session)

        return JSONResponse({
            "intro": intro,
            "total_questions": len(questions),
            "first_question": {
                "index": first_q["adaptive_index"],
                "id": first_q["id"],
                "category": first_q["category"],
                "question": first_q["question"],
                "context": first_q["context"],
                "difficulty": first_q["difficulty"],
            },
            "persona_name": persona.get("short_name", persona["name"]),
            "adaptive": engine.get_status(),
        })

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Failed to start interview: {str(exc)}")


@app.post("/answer")
async def submit_answer(data: SubmitAnswerRequest):
    """Score an answer and return the next question."""
    try:
        session = await store.get(data.session_id)
        if session is None:
            raise HTTPException(400, "Invalid or expired session.")

        if session["state"] != "interviewing":
            raise HTTPException(400, "Interview not in progress.")

        question_id = data.question_id
        question_index = data.question_index
        answer_text = data.answer.strip()
        time_seconds = data.time_seconds
        skipped = data.skipped or not answer_text
        used_hint = data.used_hint

        engine = session.get("adaptive_engine")
        questions = session["questions"]

        # Find the question by ID (adaptive order) or fall back to index
        question = None
        if question_id:
            question = next((q for q in questions if q.get("id") == question_id), None)
        if question is None and question_index < len(questions):
            question = questions[question_index]
        if question is None:
            raise HTTPException(400, "Invalid question.")

        timing_status = data.timing_status  # early | full | timeout

        # Build response record — use actual follow-up question text if this is a follow-up
        actual_question_text = data.followup_question_text if data.is_followup and data.followup_question_text else question["question"]

        response = {
            "question_id": question.get("id", question_index + 1),
            "category": question.get("category", "general"),
            "question": actual_question_text,
            "answer": answer_text,
            "skipped": skipped,
            "time_seconds": time_seconds,
            "timing_status": timing_status,
            "context": question.get("context", ""),
            "difficulty": question.get("difficulty", "medium"),
            "ideal_signals": question.get("ideal_signals", []),
            "used_hint": used_hint,
            "is_followup": data.is_followup,
        }

        # Score the answer
        profile = session["profile"]
        persona = session["persona"]

        # Get or create the candidate model for cross-answer reasoning
        from src.interview.candidate_model import CandidateModel
        if "candidate_model" in session and isinstance(session["candidate_model"], dict):
            candidate_model = CandidateModel.from_dict(session["candidate_model"])
        elif "candidate_model" in session and isinstance(session["candidate_model"], CandidateModel):
            candidate_model = session["candidate_model"]
        else:
            candidate_model = CandidateModel()

        cross_context = candidate_model.get_context_for_scoring()

        if not skipped:
            try:
                from src.evaluation.scorer import AnswerScorer
                scorer = AnswerScorer(persona)
                scored = await run_sync(scorer.score_all, [response], profile, cross_context)
                response = scored[0]
            except Exception as exc:
                logger.warning("Scoring failed for Q%s: %s", question.get("id"), exc)
                response["scores"] = {"accuracy": 5, "depth": 5, "communication": 5, "ownership": 5}
                response["score_total"] = 20
                response["feedback"] = "Scoring unavailable -- default scores applied."

            # Cheat detection
            try:
                from src.evaluation.cheat_detector import CheatDetector
                detector = CheatDetector()
                await run_sync(detector.check_all, [response], profile)
            except Exception as exc:
                logger.warning("Cheat detection failed for Q%s: %s", question.get("id"), exc)
                response["integrity"] = {"verdict": "clean", "flags": []}
        else:
            response["scores"] = {"accuracy": 0, "depth": 0, "communication": 0, "ownership": 0}
            response["score_total"] = 0
            response["feedback"] = "Skipped."
            response["integrity"] = {"verdict": "skipped", "flags": []}

        # Apply hint penalty
        if used_hint and not skipped:
            for dim in response.get("scores", {}):
                response["scores"][dim] = max(1, response["scores"][dim] - 1)
            response["score_total"] = sum(response["scores"].values())
            response["feedback"] += " (Hint penalty applied: -1 per dimension)"

        # Record answer in candidate model for cross-answer reasoning
        try:
            await run_sync(
                candidate_model.record_answer,
                question=question,
                answer=answer_text,
                scores=response.get("scores"),
                skipped=skipped,
            )
        except Exception as exc:
            logger.warning("Candidate model record failed: %s", exc)
        session["candidate_model"] = candidate_model.to_dict()

        session["responses"].append(response)

        # Feed score to adaptive engine
        if engine:
            if skipped:
                engine.record_skip()
            else:
                engine.record_score(response.get("score_total", 20))

        # Follow-up decision: should we probe deeper before moving on?
        # Don't generate follow-ups for answers that are themselves follow-ups
        followup_data = None
        if not skipped and answer_text and not data.is_followup:
            try:
                from src.interview.followup import FollowUpGenerator
                # Reuse or create follow-up generator on the session
                if "followup_gen" not in session:
                    session["followup_gen"] = FollowUpGenerator(persona)
                fg = session["followup_gen"]
                followup_context = candidate_model.get_context_for_followup()
                followup_data = await run_sync(
                    fg.should_followup,
                    question=question,
                    answer=answer_text,
                    score=response.get("scores"),
                    time_seconds=time_seconds,
                    cross_answer_context=followup_context,
                )
            except Exception as exc:
                logger.warning("Follow-up generation failed: %s", exc)
                followup_data = None

        # Build result with next question or done flag
        result = {"status": "ok"}

        # If there's a follow-up, return it instead of the next question
        if followup_data and followup_data.get("action") != "move_on" and followup_data.get("followup_question"):
            result["followup"] = {
                "action": followup_data["action"],
                "question": followup_data["followup_question"],
                "reason": followup_data.get("reason", ""),
                "original_question_id": question.get("id"),
            }
            result["done"] = False
            # Persist and return early — don't advance to next question
            await store.put(data.session_id, session)
            return JSONResponse(result)

        if engine and not engine.is_done:
            next_q = engine.pick_next()
            if next_q:
                result["next_question"] = {
                    "index": next_q["adaptive_index"],
                    "id": next_q["id"],
                    "category": next_q["category"],
                    "question": next_q["question"],
                    "context": next_q["context"],
                    "difficulty": next_q["difficulty"],
                }
                result["done"] = False
            else:
                session["state"] = "completed"
                result["done"] = True
        elif not engine:
            next_index = question_index + 1
            if next_index < len(questions):
                next_q = questions[next_index]
                result["next_question"] = {
                    "index": next_index,
                    "id": next_q["id"],
                    "category": next_q["category"],
                    "question": next_q["question"],
                    "context": next_q["context"],
                    "difficulty": next_q["difficulty"],
                }
                result["done"] = False
            else:
                session["state"] = "completed"
                result["done"] = True
        else:
            session["state"] = "completed"
            result["done"] = True

        # Persist session updates
        await store.put(data.session_id, session)

        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Failed to process answer: {str(exc)}")


@app.post("/complete")
async def complete_interview(data: ReportRequest):
    """Candidate-facing endpoint: finalize interview and save results without exposing scores."""
    try:
        session = await store.get(data.session_id)
        if session is None:
            raise HTTPException(400, "Invalid or expired session.")

        responses = session["responses"]
        profile = session["profile"]
        persona = session["persona"]

        if not responses:
            raise HTTPException(400, "No responses to evaluate.")

        # Run scoring and report generation in background (owner will see it later)
        from src.evaluation.scorer import AnswerScorer
        scorer = AnswerScorer(persona)
        for r in responses:
            if "scores" not in r:
                r["scores"] = {"accuracy": 0, "depth": 0, "communication": 0, "ownership": 0}
                r["score_total"] = 0
        score_summary = scorer.compute_summary(responses)

        from src.evaluation.cheat_detector import CheatDetector
        detector = CheatDetector()
        for r in responses:
            if "integrity" not in r:
                r["integrity"] = {"verdict": "clean", "flags": []}
        integrity_summary = detector.summarize_flags(responses)

        # Generate report files
        try:
            from src.report.generator import ReportGenerator
            reporter = ReportGenerator(persona)
            await run_sync(reporter.generate, profile, responses, score_summary, integrity_summary)
        except Exception as exc:
            logger.warning("Report generation failed: %s", exc)

        # Save result for owner dashboard
        overall_pct = round(score_summary["overall"] * 10)
        rec = _compute_recommendation(overall_pct, integrity_summary)
        try:
            cm_data = session.get("candidate_model", {})
            if isinstance(cm_data, dict):
                from src.interview.candidate_model import CandidateModel
                cm = CandidateModel.from_dict(cm_data)
                cm_summary = cm.get_summary()
            else:
                cm_summary = {}
            dims = score_summary.get("by_dimension", {})
            await results_store.save(data.session_id, {
                "candidate_name": profile.get("name", profile.get("login", "Unknown")),
                "role": profile.get("detected_role", ""),
                "industry": profile.get("industry", ""),
                "persona_name": persona.get("short_name", persona.get("name", "")),
                "overall_score": score_summary.get("overall", 0),
                "accuracy": dims.get("accuracy", 0),
                "depth": dims.get("depth", 0),
                "communication": dims.get("communication", 0),
                "ownership": dims.get("ownership", 0),
                "answered_count": score_summary.get("answered_count", 0),
                "skipped_count": score_summary.get("skipped_count", 0),
                "total_questions": len(responses),
                "integrity_score": integrity_summary.get("integrity_score", 10),
                "flagged_count": integrity_summary.get("flagged_count", 0),
                "recommendation": rec.get("label", ""),
                "by_category": score_summary.get("by_category", {}),
                "skills_demonstrated": cm_summary.get("demonstrated_skills", []),
                "contradiction_count": cm_summary.get("contradiction_count", 0),
            })
        except Exception as exc:
            logger.warning("Failed to save result: %s", exc)

        session["state"] = "completed"
        await store.put(data.session_id, session)

        # Clean up live cache for this session (AdaptiveEngine, FollowUpGenerator)
        from src.web.session_store import _LIVE_CACHE
        _LIVE_CACHE.pop(data.session_id, None)

        # Return NOTHING about scores — just confirmation
        return JSONResponse({"status": "completed", "message": "Interview recorded successfully."})

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Failed to complete interview: {str(exc)}")


# ---------------------------------------------------------------------------
# Owner routes — all require valid owner session cookie
# ---------------------------------------------------------------------------

@app.get("/owner")
async def owner_page(owner_session: str | None = Cookie(default=None)):
    """Serve owner portal. Shows login if not authenticated, dashboard if authenticated."""
    owner_path = STATIC_DIR / "owner.html"
    if not owner_path.exists():
        raise HTTPException(404, "Owner page not found.")
    return FileResponse(str(owner_path))


@app.post("/owner/login")
async def owner_login(data: OwnerLoginRequest, response: Response):
    """Validate password and set session cookie."""
    if not OWNER_PASSWORD:
        raise HTTPException(500, "OWNER_PASSWORD not configured in .env")
    if not hmac.compare_digest(data.password, OWNER_PASSWORD):
        logger.warning("Failed owner login attempt")
        raise HTTPException(401, "Incorrect password.")
    token = _create_owner_session()
    logger.info("Owner logged in successfully")
    resp = JSONResponse({"status": "ok"})
    resp.set_cookie(
        key=OWNER_COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="strict",
        max_age=OWNER_SESSION_TTL,
        path="/",
    )
    return resp


@app.post("/owner/logout")
async def owner_logout(owner_session: str | None = Cookie(default=None)):
    """Clear owner session."""
    if owner_session:
        _owner_sessions.pop(owner_session, None)
    resp = JSONResponse({"status": "ok"})
    resp.delete_cookie(OWNER_COOKIE_NAME, path="/")
    return resp


@app.get("/owner/auth-check")
async def owner_auth_check(owner_session: str | None = Cookie(default=None)):
    """Check if current owner session is valid."""
    if _verify_owner_session(owner_session):
        return JSONResponse({"authenticated": True})
    return JSONResponse({"authenticated": False}, status_code=401)


@app.get("/api/model")
async def get_model(owner_session: str | None = Cookie(default=None)):
    """Return the current active model. OWNER ONLY."""
    _require_owner(owner_session)
    return JSONResponse({"model": get_active_model()})


@app.post("/api/model")
async def change_model(request: Request, owner_session: str | None = Cookie(default=None)):
    """Change the active model. OWNER ONLY. Takes effect for the next interview."""
    _require_owner(owner_session)
    body = await request.json()
    model_id = body.get("model", "").strip()
    if not model_id:
        raise HTTPException(400, "Model ID is required.")
    set_active_model(model_id)
    return JSONResponse({"status": "ok", "model": model_id})


@app.post("/report")
async def generate_report(data: ReportRequest, owner_session: str | None = Cookie(default=None)):
    """Generate the final evaluation report. OWNER ONLY."""
    _require_owner(owner_session)
    try:
        session = await store.get(data.session_id)
        if session is None:
            raise HTTPException(400, "Invalid or expired session.")

        responses = session["responses"]
        profile = session["profile"]
        persona = session["persona"]

        if not responses:
            raise HTTPException(400, "No responses to evaluate.")

        # Compute score summary
        from src.evaluation.scorer import AnswerScorer
        scorer = AnswerScorer(persona)

        # Ensure all responses have scores
        for r in responses:
            if "scores" not in r:
                r["scores"] = {"accuracy": 0, "depth": 0, "communication": 0, "ownership": 0}
                r["score_total"] = 0

        score_summary = scorer.compute_summary(responses)

        # Compute integrity summary
        from src.evaluation.cheat_detector import CheatDetector
        detector = CheatDetector()

        for r in responses:
            if "integrity" not in r:
                r["integrity"] = {"verdict": "clean", "flags": []}

        integrity_summary = detector.summarize_flags(responses)

        # Compute authenticity fingerprint
        fingerprint_data = {}
        try:
            from src.evaluation.fingerprint import compute_authenticity
            fp = compute_authenticity(responses)
            fingerprint_data = fp.to_dict()
        except Exception as exc:
            logger.warning("Authenticity fingerprint failed: %s", exc)

        # Generate report (blocking LLM + file I/O → thread pool)
        from src.report.generator import ReportGenerator
        reporter = ReportGenerator(persona)
        paths = await run_sync(reporter.generate, profile, responses, score_summary, integrity_summary)

        # Read the markdown report for display
        md_content = ""
        try:
            md_content = Path(paths["markdown"]).read_text(encoding="utf-8")
        except Exception:
            pass

        # Build response with all data needed for the UI
        overall_pct = round(score_summary["overall"] * 10)
        rec = _compute_recommendation(overall_pct, integrity_summary)

        # Save result for comparison dashboard
        try:
            cm_data = session.get("candidate_model", {})
            if isinstance(cm_data, dict):
                from src.interview.candidate_model import CandidateModel
                cm = CandidateModel.from_dict(cm_data)
                cm_summary = cm.get_summary()
            else:
                cm_summary = {}

            dims = score_summary.get("by_dimension", {})
            await results_store.save(data.session_id, {
                "candidate_name": profile.get("name", profile.get("login", "Unknown")),
                "role": profile.get("detected_role", ""),
                "industry": profile.get("industry", ""),
                "persona_name": persona.get("short_name", persona.get("name", "")),
                "overall_score": score_summary.get("overall", 0),
                "accuracy": dims.get("accuracy", 0),
                "depth": dims.get("depth", 0),
                "communication": dims.get("communication", 0),
                "ownership": dims.get("ownership", 0),
                "answered_count": score_summary.get("answered_count", 0),
                "skipped_count": score_summary.get("skipped_count", 0),
                "total_questions": len(responses),
                "integrity_score": integrity_summary.get("integrity_score", 10),
                "flagged_count": integrity_summary.get("flagged_count", 0),
                "recommendation": rec.get("label", ""),
                "by_category": score_summary.get("by_category", {}),
                "skills_demonstrated": cm_summary.get("demonstrated_skills", []),
                "contradiction_count": cm_summary.get("contradiction_count", 0),
            })
        except Exception as exc:
            logger.warning("Failed to save result for dashboard: %s", exc)

        return JSONResponse({
            "score_summary": score_summary,
            "integrity_summary": integrity_summary,
            "overall_percentage": overall_pct,
            "recommendation": rec,
            "dimensions": score_summary.get("by_dimension", {}),
            "strengths": _extract_strengths(responses, score_summary),
            "weaknesses": _extract_weaknesses(responses, score_summary),
            "red_flags": integrity_summary.get("all_flags", []),
            "markdown_report": md_content,
            "pdf_path": paths.get("pdf", ""),
            "md_path": paths.get("markdown", ""),
            "authenticity": fingerprint_data,
        })

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Failed to generate report: {str(exc)}")


@app.get("/replay/{session_id}")
async def replay_page(session_id: str, owner_session: str | None = Cookie(default=None)):
    """Serve the interview replay page. OWNER ONLY."""
    _require_owner(owner_session)
    replay_path = STATIC_DIR / "replay.html"
    if not replay_path.exists():
        raise HTTPException(404, "Replay page not found.")
    return FileResponse(str(replay_path))


@app.get("/api/replay/{session_id}")
async def replay_data(session_id: str, owner_session: str | None = Cookie(default=None)):
    """Return the full interview timeline with AI commentary for replay. OWNER ONLY."""
    _require_owner(owner_session)
    try:
        session = await store.get(session_id)
        if session is None:
            raise HTTPException(404, "Session not found or expired.")

        responses = session.get("responses", [])
        profile = session.get("profile", {})
        persona = session.get("persona", {})

        if not responses:
            raise HTTPException(400, "No responses to replay.")

        # Build timeline entries with AI commentary
        from src.llm_client import LLMClient
        llm = LLMClient()

        timeline = []
        for i, r in enumerate(responses):
            score_total = r.get("score_total", 0)
            max_score = 40
            pct = round((score_total / max_score) * 100) if max_score else 0

            # Color coding based on score
            if r.get("skipped"):
                color = "gray"
                moment_type = "skipped"
            elif pct >= 75:
                color = "green"
                moment_type = "strong"
            elif pct >= 50:
                color = "amber"
                moment_type = "adequate"
            else:
                color = "red"
                moment_type = "weak"

            # Check integrity flags
            integrity = r.get("integrity", {})
            flags = integrity.get("flags", [])
            if flags:
                color = "red"
                moment_type = "flagged"

            entry = {
                "index": i,
                "question_id": r.get("question_id"),
                "category": r.get("category", ""),
                "difficulty": r.get("difficulty", ""),
                "question": r.get("question", ""),
                "answer": r.get("answer", ""),
                "skipped": r.get("skipped", False),
                "time_seconds": r.get("time_seconds", 0),
                "scores": r.get("scores", {}),
                "score_total": score_total,
                "score_pct": pct,
                "feedback": r.get("feedback", ""),
                "integrity": integrity,
                "color": color,
                "moment_type": moment_type,
                "is_followup": r.get("is_followup", False),
                "followup_type": r.get("followup_type", ""),
            }
            timeline.append(entry)

        # Generate AI commentary for key moments
        try:
            commentary = await run_sync(_generate_replay_commentary, llm, timeline, persona, profile)
        except Exception as exc:
            logger.warning("Replay commentary generation failed: %s", exc)
            commentary = {}

        # Build candidate model summary if available
        candidate_summary = {}
        cm_data = session.get("candidate_model")
        if cm_data and isinstance(cm_data, dict):
            from src.interview.candidate_model import CandidateModel
            cm = CandidateModel.from_dict(cm_data)
            candidate_summary = cm.get_summary()

        return JSONResponse({
            "session_id": session_id,
            "candidate_name": profile.get("name", profile.get("login", "Unknown")),
            "persona": {
                "name": persona.get("name", ""),
                "short_name": persona.get("short_name", ""),
                "color": persona.get("color", "cyan"),
            },
            "timeline": timeline,
            "commentary": commentary,
            "candidate_model": candidate_summary,
            "total_questions": len(timeline),
        })

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Failed to build replay: {str(exc)}")


def _generate_replay_commentary(llm, timeline: list, persona: dict, profile: dict) -> dict:
    """Generate AI commentary for notable moments in the interview."""
    from src.llm_client import parse_json_object

    # Select key moments: best, worst, flagged, and overall
    moments = []
    for entry in timeline:
        if entry["skipped"]:
            continue
        moments.append({
            "index": entry["index"],
            "question": entry["question"][:100],
            "score_pct": entry["score_pct"],
            "category": entry["category"],
            "moment_type": entry["moment_type"],
            "flags": entry["integrity"].get("flags", []),
        })

    if not moments:
        return {}

    moments_str = "\n".join(
        f"  Q{m['index']+1} ({m['category']}, {m['moment_type']}): {m['question'][:80]}... — {m['score_pct']}%"
        + (f" FLAGS: {m['flags']}" if m['flags'] else "")
        for m in moments[:12]  # limit to 12 for prompt size
    )

    prompt = f"""You are {persona.get('name', 'an interviewer')} reviewing an interview replay.

Candidate: {profile.get('name', profile.get('login', 'Unknown'))}
Role: {profile.get('detected_role', 'Software Engineer')}

Interview moments:
{moments_str}

For each notable moment (best answer, worst answer, any flagged moment, and overall impression), write a brief AI commentary (1-2 sentences each) as if narrating the replay.

Return JSON:
{{
  "overall": "1-2 sentence overall impression of the interview",
  "highlights": [
    {{"index": <moment index>, "comment": "1-sentence commentary"}},
    ...
  ]
}}

Return at most 5 highlights. Raw JSON only, no fences."""

    try:
        data = parse_json_object(llm.generate(prompt))
        return data
    except Exception:
        return {"overall": "Interview replay available.", "highlights": []}


@app.get("/dashboard")
async def dashboard_page(owner_session: str | None = Cookie(default=None)):
    """Serve the comparison dashboard page. OWNER ONLY."""
    _require_owner(owner_session)
    dash_path = STATIC_DIR / "dashboard.html"
    if not dash_path.exists():
        raise HTTPException(404, "Dashboard page not found.")
    return FileResponse(str(dash_path))


@app.get("/api/dashboard")
async def dashboard_data(owner_session: str | None = Cookie(default=None)):
    """Return all completed interview results for the dashboard. OWNER ONLY."""
    _require_owner(owner_session)
    try:
        results = await results_store.get_all(limit=200)
        total = await results_store.count()
        return JSONResponse({"results": results, "total": total})
    except Exception as exc:
        raise HTTPException(500, f"Failed to load dashboard: {str(exc)}")


@app.get("/api/compare/{session_a}/{session_b}")
async def compare_candidates(session_a: str, session_b: str, owner_session: str | None = Cookie(default=None)):
    """Return side-by-side comparison data for two candidates. OWNER ONLY."""
    _require_owner(owner_session)
    try:
        a = await results_store.get_by_id(session_a)
        b = await results_store.get_by_id(session_b)
        if not a or not b:
            raise HTTPException(404, "One or both sessions not found.")
        return JSONResponse({"candidate_a": a, "candidate_b": b})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Comparison failed: {str(exc)}")


@app.get("/download/{filename}")
async def download_file(filename: str, owner_session: str | None = Cookie(default=None)):
    """Download a generated report file. OWNER ONLY."""
    _require_owner(owner_session)
    # Sanitize filename to prevent path traversal
    safe_name = Path(filename).name  # strips any directory components
    if safe_name != filename or ".." in filename:
        raise HTTPException(400, "Invalid filename.")
    filepath = (Path("outputs") / safe_name).resolve()
    # Double-check resolved path is inside outputs/
    if not str(filepath).startswith(str(Path("outputs").resolve())):
        raise HTTPException(400, "Invalid filename.")
    if not filepath.exists():
        raise HTTPException(404, "File not found.")
    return FileResponse(str(filepath), filename=safe_name)


def _compute_recommendation(overall_pct: int, integrity: dict) -> dict:
    """Compute hiring recommendation from scores."""
    flagged = integrity.get("flagged_count", 0)

    if flagged >= 3:
        return {"label": "No", "color": "red", "detail": "Multiple integrity flags"}

    if overall_pct >= 80:
        label = "Strong Yes" if flagged == 0 else "Yes"
    elif overall_pct >= 65:
        label = "Yes" if flagged == 0 else "Maybe"
    elif overall_pct >= 50:
        label = "Maybe"
    else:
        label = "No"

    color_map = {"Strong Yes": "green", "Yes": "green", "Maybe": "yellow", "No": "red"}
    return {"label": label, "color": color_map.get(label, "gray"), "detail": ""}


def _extract_strengths(responses: list, summary: dict) -> list[str]:
    """Extract top 3 strengths from scores."""
    strengths = []
    dims = summary.get("by_dimension", {})
    sorted_dims = sorted(dims.items(), key=lambda x: x[1], reverse=True)
    for dim, val in sorted_dims[:3]:
        if val >= 6:
            strengths.append(f"Strong {dim} -- scored {val}/10 on average")
    if not strengths:
        strengths.append("Completed the interview")
    return strengths[:3]


def _extract_weaknesses(responses: list, summary: dict) -> list[str]:
    """Extract top 3 weaknesses from scores."""
    weaknesses = []
    dims = summary.get("by_dimension", {})
    sorted_dims = sorted(dims.items(), key=lambda x: x[1])
    for dim, val in sorted_dims[:3]:
        if val < 7:
            weaknesses.append(f"Room for improvement in {dim} -- scored {val}/10")
    skipped = summary.get("skipped_count", 0)
    if skipped > 0:
        weaknesses.append(f"Skipped {skipped} question(s)")
    return weaknesses[:3]
