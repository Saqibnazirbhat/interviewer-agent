# Interviewer Agent — Codex Instructions

## Project Goal
Build a world-class AI interviewer agent in Python that:
1. Takes a candidate's GitHub username or resume as input
2. Analyzes their projects, skills, and experience
3. Conducts a dynamic, adaptive interview session via browser UI
4. Scores answers in real-time
5. Generates a full evaluation report

## Architecture
- src/llm_client.py   → NVIDIA NIM client (OpenAI-compatible SDK)
- src/ingestion/       → fetch and parse candidate data (GitHub + resume)
- src/interview/       → interview engine, personas, question generator, adaptive difficulty, follow-ups
- src/evaluation/      → scoring, cheat detection, authenticity fingerprinting
- src/report/          → markdown and PDF report generation
- src/web/             → FastAPI backend + static HTML/JS frontend
- data/                → candidate JSON storage + model config
- outputs/             → final reports

## Tech Stack
- Python 3.10+
- openai SDK (calling NVIDIA NIM API — OpenAI-compatible)
- NVIDIA NIM (default model: meta/llama-3.3-70b-instruct)
- PyGithub (GitHub API)
- python-dotenv (env management)
- rich (terminal UI)
- reportlab (PDF generation)
- FastAPI + uvicorn (browser UI backend)
- PyPDF2 + python-docx (resume parsing)

## Rules
- Always load API keys from .env using dotenv
- Use openai SDK pointed at NVIDIA NIM (https://integrate.api.nvidia.com/v1) as the AI backbone
- All interview logic must be modular — one class per file
- Questions must be personalized to the candidate's actual projects
- Never hardcode API keys
- Use streaming for LLM responses so it feels like a real interviewer typing
- Every module must have clear docstrings
- Two-role access: candidates see no scores; owner gets full dashboard, reports, replay, comparison
- Owner portal is protected by OWNER_PASSWORD from .env
