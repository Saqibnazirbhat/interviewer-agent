"""Pydantic request/response models for all FastAPI endpoints."""

from pydantic import BaseModel, Field


class GitHubRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=39, pattern=r"^[a-zA-Z0-9](?:[a-zA-Z0-9]|-(?=[a-zA-Z0-9])){0,38}$")


class StartInterviewRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    session_token: str = Field(..., min_length=1, max_length=200, description="Session auth token returned at session creation.")
    persona_id: str = Field(default="technical_expert", max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    detected_role: str | None = Field(default=None, max_length=200)
    industry: str | None = Field(default=None, max_length=200)
    achievement_description: str | None = Field(default=None, max_length=2000)


class SubmitAnswerRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    session_token: str = Field(..., min_length=1, max_length=200, description="Session auth token returned at session creation.")
    question_id: int | None = Field(default=None, ge=0, le=10000)
    question_index: int = Field(default=0, ge=0, le=1000)
    answer: str = Field(default="", max_length=20000)
    time_seconds: float = Field(default=0, ge=0, le=7200)
    skipped: bool = False
    used_hint: bool = False
    is_followup: bool = False
    followup_question_text: str = Field(default="", max_length=5000)
    timing_status: str = Field(default="early", pattern="^(early|full|timeout)$")


class ReportRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    session_token: str = Field(default="", max_length=200, description="Session auth token. Required for candidate /complete, optional for owner /report.")


class HintRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    session_token: str = Field(..., min_length=1, max_length=200, description="Session auth token returned at session creation.")
    question_id: int | None = Field(default=None, ge=0, le=10000)
    question_index: int = Field(default=0, ge=0, le=1000)


class EnrichGitHubRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    session_token: str = Field(..., min_length=1, max_length=200, description="Session auth token returned at session creation.")
    username: str = Field(..., min_length=1, max_length=39, pattern=r"^[a-zA-Z0-9](?:[a-zA-Z0-9]|-(?=[a-zA-Z0-9])){0,38}$")


class OwnerLoginRequest(BaseModel):
    password: str = Field(..., min_length=1, max_length=256)
