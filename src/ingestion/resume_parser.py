"""Parses resumes (PDF, DOCX, TXT) and uses NVIDIA NIM to extract candidate profile data."""

import json
import re
from pathlib import Path

from src.llm_client import LLMClient, parse_json_object


def _sanitize_filename(name: str) -> str:
    """Strip a string down to a safe filesystem name — no path traversal possible.

    Only alphanumerics, underscores, and hyphens survive. Leading dots/dashes
    are stripped. Result is truncated to 80 characters. Falls back to
    'candidate' if nothing remains.
    """
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", name)
    safe = safe.lstrip("_-.").rstrip("_-.")
    safe = re.sub(r"_+", "_", safe)  # collapse runs of underscores
    safe = safe[:80]
    return safe or "candidate"


class ResumeParser:
    """Extracts structured candidate data from resume files using LLM for intelligent parsing."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

    def __init__(self):
        self.llm = LLMClient()

    def parse_file(self, file_path: str) -> dict:
        """Parse a resume file and return a structured candidate profile."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Resume file not found: {file_path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format '{ext}'. Supported: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )

        raw_text = self._extract_text(path, ext)
        if not raw_text.strip():
            raise ValueError("Resume file appears to be empty or unreadable.")

        profile = self._analyze_with_llm(raw_text)
        profile["resume_text"] = raw_text
        profile["source"] = "resume"
        profile["resume_file"] = str(path.name)

        self._save_profile(profile)
        return profile

    def parse_text(self, text: str) -> dict:
        """Parse raw resume text (for web uploads where text is already extracted)."""
        if not text.strip():
            raise ValueError("Resume text is empty.")

        profile = self._analyze_with_llm(text)
        profile["resume_text"] = text
        profile["source"] = "resume"
        profile["resume_file"] = "uploaded"

        self._save_profile(profile)
        return profile

    def _extract_text(self, path: Path, ext: str) -> str:
        """Extract plain text from the resume file based on format."""
        if ext == ".txt":
            return path.read_text(encoding="utf-8", errors="replace")
        elif ext == ".pdf":
            return self._extract_pdf(path)
        elif ext == ".docx":
            return self._extract_docx(path)
        return ""

    def _extract_pdf(self, path: Path) -> str:
        """Extract text from a PDF file using PyPDF2."""
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            raise ImportError(
                "PyPDF2 is required for PDF parsing. Install it: pip install PyPDF2"
            )

        reader = PdfReader(str(path))
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n".join(text_parts)

    def _extract_docx(self, path: Path) -> str:
        """Extract text from a DOCX file using python-docx."""
        try:
            from docx import Document
        except ImportError:
            raise ImportError(
                "python-docx is required for DOCX parsing. Install it: pip install python-docx"
            )

        doc = Document(str(path))
        return "\n".join(para.text for para in doc.paragraphs if para.text.strip())

    def _analyze_with_llm(self, resume_text: str) -> dict:
        """Use the LLM to extract structured profile data from resume text."""
        prompt = f"""You are an expert resume analyzer. Analyze the following resume and extract structured data.
This resume could be for ANY profession — doctor, lawyer, engineer, teacher, designer, marketer, MBA, nurse, chef, etc.

RESUME TEXT:
{resume_text[:8000]}

Extract the following information and return as a JSON object:
{{
  "name": "candidate's full name",
  "username": "name in lowercase with no spaces (for file naming)",
  "bio": "one-line professional summary",
  "detected_role": "their current or most recent job title (e.g., Software Engineer, Registered Nurse, Marketing Manager, Corporate Lawyer)",
  "industry": "the industry they work in (e.g., Technology, Healthcare, Finance, Education, Legal, Design, Marketing, Hospitality)",
  "seniority": "one of: Entry Level, Junior, Mid Level, Senior, Lead, Manager, Director, VP, Executive, C-Suite",
  "years_of_experience": "calculate ONLY from actual paid work experience dates listed under Professional/Work Experience sections. Internships count. Do NOT count education, personal projects, or academic coursework. If total is less than 1 year, use a decimal (e.g., 0.25 for 3 months). Return as a number.",
  "skills": ["list", "of", "key", "skills", "from", "resume"],
  "work_experience": [
    {{
      "title": "job title",
      "company": "company name",
      "duration": "time period",
      "highlights": ["key achievement 1", "key achievement 2"]
    }}
  ],
  "projects": [
    {{
      "name": "project or achievement name",
      "description": "brief description",
      "technologies_or_tools": ["relevant tools or methods used"]
    }}
  ],
  "education": [
    {{
      "degree": "degree name",
      "institution": "school name",
      "year": "graduation year or period"
    }}
  ],
  "certifications": ["list of certifications if any"]
}}

Be thorough. If a field is not found in the resume, use reasonable defaults:
- name: "Candidate" if not found
- username: derive from name
- For projects: if no explicit projects section, extract notable achievements from work experience
- For skills: extract both explicit skills and implied skills from experience

Return ONLY valid JSON. No markdown fences, no commentary."""

        try:
            text = self.llm.generate(prompt)
            return self._parse_response(text)
        except Exception as exc:
            raise RuntimeError(
                f"Resume analysis failed: {exc}\n"
                "Check your NVIDIA_API_KEY and network connection."
            ) from exc

    def _parse_response(self, text: str) -> dict:
        """Parse the JSON response from the LLM."""
        cleaned = text.strip()

        # Strip markdown code fences if present
        if cleaned.startswith("```"):
            first_newline = cleaned.find("\n")
            if first_newline != -1:
                cleaned = cleaned[first_newline + 1:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        # Find JSON object boundaries
        start = cleaned.find("{")
        if start == -1:
            raise ValueError(f"No JSON object found in LLM response:\n{text[:300]}")

        end = cleaned.rfind("}")
        if end == -1 or end <= start:
            raise ValueError(f"Unclosed JSON object in LLM response:\n{text[:300]}")

        try:
            profile = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in response: {exc}") from exc

        # Ensure required fields have defaults
        profile.setdefault("name", "Candidate")
        # Sanitize username to prevent path traversal — this is LLM-derived input
        raw_username = profile.get("username", profile["name"].lower().replace(" ", "_"))
        profile["username"] = _sanitize_filename(raw_username)
        profile.setdefault("bio", "")
        profile.setdefault("detected_role", "Professional")
        profile.setdefault("industry", "General")
        profile.setdefault("seniority", "Mid Level")
        profile.setdefault("years_of_experience", 0)
        profile.setdefault("skills", [])
        profile.setdefault("work_experience", [])
        profile.setdefault("projects", [])
        profile.setdefault("education", [])
        profile.setdefault("certifications", [])

        return profile

    def _save_profile(self, profile: dict) -> Path:
        """Persist the profile to data/ — encrypted at rest."""
        from src.web.session_store import encrypt_bytes

        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        username = _sanitize_filename(profile.get("username", "candidate"))
        filepath = (data_dir / f"{username}.json.enc").resolve()
        # Verify resolved path stays inside data/
        if not str(filepath).startswith(str(data_dir.resolve())):
            raise ValueError("Invalid username produced an unsafe file path.")
        # Don't save the full resume text to the JSON (can be very large)
        save_data = {k: v for k, v in profile.items() if k != "resume_text"}
        save_data["resume_excerpt"] = profile.get("resume_text", "")[:1000]
        plaintext = json.dumps(save_data, indent=2, default=str).encode("utf-8")
        filepath.write_bytes(encrypt_bytes(plaintext))
        return filepath
