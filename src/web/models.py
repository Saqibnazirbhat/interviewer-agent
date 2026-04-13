"""Pydantic request/response models for all FastAPI endpoints."""

from pydantic import BaseModel, Field


class GitHubRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=100)


class StartInterviewRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    persona_id: str = Field(default="technical_expert")
    detected_role: str | None = None
    industry: str | None = None
    achievement_description: str | None = None


class SubmitAnswerRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    question_id: int | None = None
    question_index: int = Field(default=0, ge=0)
    answer: str = Field(default="", max_length=50000)
    time_seconds: float = Field(default=0, ge=0, le=7200)
    skipped: bool = False
    used_hint: bool = False
    is_followup: bool = False
    followup_question_text: str = Field(default="", max_length=5000)
    timing_status: str = Field(default="early", pattern="^(early|full|timeout)$")


class ReportRequest(BaseModel):
    session_id: str = Field(..., min_length=1)


class HintRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    question_id: int | None = None
    question_index: int = Field(default=0, ge=0)


class EnrichGitHubRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    username: str = Field(..., min_length=1, max_length=100)


class OwnerLoginRequest(BaseModel):
    password: str = Field(..., min_length=1)
