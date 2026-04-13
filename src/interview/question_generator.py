"""Generates personalized interview questions from a candidate's profile using NVIDIA NIM."""

import json

from src.llm_client import LLMClient, parse_json_array


class QuestionGenerator:
    """Creates a tailored question set shaped by the candidate's profile and the interviewer persona."""

    def __init__(self):
        self.llm = LLMClient()

    def generate(self, profile: dict, persona: dict) -> list[dict]:
        """Produce 15 interview questions personalized to the candidate and persona."""
        source = profile.get("source", "github")
        if source == "resume":
            prompt = self._build_resume_prompt(profile, persona)
        else:
            prompt = self._build_github_prompt(profile, persona)

        try:
            text = self.llm.generate(prompt)
            questions = self._parse_response(text)
        except Exception as exc:
            raise RuntimeError(
                f"Question generation failed: {exc}\n"
                "Check your NVIDIA_API_KEY and network connection."
            ) from exc
        return questions

    def generate_intro(self, profile: dict, persona: dict) -> str:
        """Stream a short persona-voiced intro greeting for the candidate."""
        source = profile.get("source", "github")

        if source == "resume":
            role = profile.get("detected_role", "professional")
            industry = profile.get("industry", "")
            skills = ", ".join(profile.get("skills", [])[:5])
            context = (
                f"who is a {role} in {industry} with skills in {skills}"
                if skills else f"who is a {role} in {industry}"
            )
        else:
            languages = ", ".join(list(profile.get("languages", {}).keys())[:5])
            context = f"(@{profile.get('username', '')}), who works primarily with {languages}"

        candidate_name = profile.get('name', 'the candidate')
        prompt = (
            f"{persona['description']}\n\n"
            f"You are about to interview {candidate_name}, {context}.\n\n"
            f"Write a short, warm welcome message — exactly 2 sentences, no more.\n\n"
            f"Sentence 1: Greet them by first name and mention ONE specific thing from their "
            f"background (a project, skill, or repo) that impressed you.\n"
            f"Sentence 2: Say you're looking forward to the conversation.\n\n"
            f"STRICT RULES:\n"
            f"- Maximum 2 sentences. Stop after the second sentence.\n"
            f"- NO questions of any kind. No question marks.\n"
            f"- NO phrases like 'we'll be looking for', 'we're eager to explore', "
            f"'can you walk us through', 'tell us about', or any variation.\n"
            f"- NO listing their skills/qualifications. Just pick ONE thing to compliment.\n"
            f"- NO time-based greetings like 'good morning', 'good afternoon', 'good evening'.\n"
            f"- NO corporate jargon like 'strategic vision', 'drive innovation', 'leverage'.\n"
            f"- Tone: casual and human, like a friendly senior colleague.\n"
            f"- Plain text only, no markdown.\n\n"
            f"Example tone (do NOT copy): \"Hey Alex, your work on the Mesh renderer "
            f"really caught my eye — the approach to deferred shading is clever. "
            f"Looking forward to chatting with you today.\""
        )
        try:
            chunks = []
            for chunk in self.llm.generate_stream(prompt):
                chunks.append(chunk)
            return "".join(chunks)
        except Exception:
            return f"Welcome, {profile.get('name', 'Candidate')}. I've reviewed your background and I'm looking forward to our conversation today."

    def _build_resume_prompt(self, profile: dict, persona: dict) -> str:
        """Build prompt for resume-based candidate profiles (any role/industry)."""
        # Work experience summary
        work_summary = ""
        for job in profile.get("work_experience", []):
            highlights = "\n".join(f"      - {h}" for h in job.get("highlights", []))
            work_summary += (
                f"\n    {job.get('title', 'N/A')} at {job.get('company', 'N/A')} ({job.get('duration', 'N/A')})\n"
                f"      Highlights:\n{highlights}\n"
            )

        # Projects summary
        projects_summary = ""
        for proj in profile.get("projects", []):
            tools = ", ".join(proj.get("technologies_or_tools", []))
            projects_summary += (
                f"\n    {proj.get('name', 'N/A')}: {proj.get('description', 'N/A')}\n"
                f"      Tools/Methods: {tools}\n"
            )

        # Skills
        skills_list = ", ".join(profile.get("skills", []))

        # Education
        edu_summary = ""
        for edu in profile.get("education", []):
            edu_summary += f"    {edu.get('degree', 'N/A')} from {edu.get('institution', 'N/A')} ({edu.get('year', 'N/A')})\n"

        # Additional context (user-provided achievement description)
        extra_context = ""
        if profile.get("achievement_description"):
            extra_context = f"\n  CANDIDATE'S SELF-DESCRIBED BEST ACHIEVEMENT:\n    {profile['achievement_description']}\n"

        # GitHub context if available
        github_context = ""
        if profile.get("github_profile"):
            gh = profile["github_profile"]
            repos = gh.get("top_repos", [])
            if repos:
                github_context = "\n  GITHUB PROJECTS:\n"
                for repo in repos[:3]:
                    github_context += f"    {repo['name']}: {repo.get('description', 'N/A')} ({repo.get('primary_language', 'N/A')})\n"

        role = profile.get("detected_role", "Professional")
        industry = profile.get("industry", "General")
        seniority = profile.get("seniority", "Mid Level")

        return f"""{persona['description']}

{persona.get('question_style', '')}

You are preparing questions for a candidate applying for a role in {industry}.

CANDIDATE PROFILE:
  Name: {profile.get('name', 'Unknown')}
  Current/Recent Role: {role}
  Industry: {industry}
  Seniority Level: {seniority}
  Years of Experience: {profile.get('years_of_experience', 'Unknown')}
  Skills: {skills_list}

  WORK EXPERIENCE:
{work_summary}

  PROJECTS & ACHIEVEMENTS:
{projects_summary}

  EDUCATION:
{edu_summary}

  CERTIFICATIONS: {', '.join(profile.get('certifications', []))}
{extra_context}{github_context}

Generate exactly 15 interview questions in this distribution:
- 4 EXPERIENCE: Questions about their actual jobs listed on the resume. Reference specific roles, companies, and achievements.
- 4 PROJECT: Questions about their projects or notable achievements. Ask for details, challenges, and outcomes.
- 3 SKILL: Questions testing their listed skills with practical scenarios relevant to {industry}.
- 2 SITUATIONAL: Hypothetical but realistic scenarios they might face in their role as a {role} in {industry}.
- 2 CURVEBALL: Unexpected questions that test creative thinking, adaptability, and thinking on their feet. These should still be relevant to their field.

RULES:
- Every question MUST reference something specific from their resume — actual jobs, skills, projects, or experiences
- Questions must feel like the interviewer actually read the resume — NEVER generic
- The language and terminology must be appropriate for {industry} and a {role}
- Vary difficulty: 4 easy, 6 medium, 5 hard
- Stay in character as {persona.get('short_name', persona['name'])}
- Adapt your vocabulary and references to {industry} — use domain-specific terminology

Return ONLY a valid JSON array where each element has:
  "id": sequential integer starting at 1,
  "category": one of "experience", "project", "skill", "situational", "curveball",
  "question": the full question text,
  "context": which job/skill/project this targets (brief),
  "difficulty": "easy", "medium", or "hard",
  "ideal_signals": list of 3-4 key points a strong answer would include

Return raw JSON only. No markdown fences, no commentary."""

    def _build_github_prompt(self, profile: dict, persona: dict) -> str:
        """Construct the LLM prompt for GitHub-based profiles."""
        repos_summary = ""
        for repo in profile.get("top_repos", []):
            langs = ", ".join(repo.get("languages", {}).keys())
            commits = "\n".join(f"    - {c['message']}" for c in repo.get("recent_commits", [])[:5])
            repos_summary += (
                f"\n  Repo: {repo['name']}\n"
                f"    Description: {repo.get('description', 'N/A')}\n"
                f"    Languages: {langs}\n"
                f"    Stars: {repo.get('stars', 0)} | Forks: {repo.get('forks', 0)}\n"
                f"    Topics: {', '.join(repo.get('topics', []))}\n"
                f"    Recent commits:\n{commits}\n"
                f"    README excerpt: {repo.get('readme_snippet', '')[:300]}\n"
            )

        languages = profile.get("languages", {})
        lang_summary = ", ".join(
            f"{lang} ({info['percentage']}%)" for lang, info in languages.items()
        )

        commit_info = profile.get("commit_patterns", {})

        return f"""{persona['description']}

{persona.get('question_style', '')}

You are preparing questions for a software engineering candidate.

CANDIDATE PROFILE:
  Name: {profile.get('name', 'Unknown')}
  GitHub: {profile.get('username')}
  Bio: {profile.get('bio', 'N/A')}
  Account age: {profile.get('account_age_years', '?')} years
  Public repos: {profile.get('public_repos', 0)}
  Followers: {profile.get('followers', 0)}
  Language breakdown: {lang_summary}
  Commit habits: {commit_info.get('active_days_per_week', '?')} active days/week, avg message length {commit_info.get('avg_message_length', '?')} chars

TOP REPOSITORIES:
{repos_summary}

Generate exactly 15 interview questions in this distribution:
- 4 EXPERIENCE: Questions about their work patterns, commit habits, and GitHub activity.
- 4 PROJECT: Deep questions about their actual repositories, code, and design choices.
- 3 SKILL: Questions testing their primary languages and frameworks with practical scenarios.
- 2 SITUATIONAL: Hypothetical scenarios a developer might face (debugging, architecture decisions, team conflicts).
- 2 CURVEBALL: Unexpected questions that test creative thinking and adaptability.

RULES:
- Every question MUST reference at least one of their actual repos, languages, or visible patterns
- No generic textbook questions — ground everything in THEIR work
- Vary difficulty: 4 easy, 6 medium, 5 hard
- Stay in character as {persona.get('short_name', persona['name'])}

Return ONLY a valid JSON array where each element has:
  "id": sequential integer starting at 1,
  "category": one of "experience", "project", "skill", "situational", "curveball",
  "question": the full question text,
  "context": which repo/skill this targets (brief),
  "difficulty": "easy", "medium", or "hard",
  "ideal_signals": list of 3-4 key points a strong answer would include

Return raw JSON only. No markdown fences, no commentary."""

    def _parse_response(self, text: str) -> list[dict]:
        """Parse and validate the JSON question set from the LLM response."""
        cleaned = text.strip()

        # Strip markdown code fences if present
        if cleaned.startswith("```"):
            first_newline = cleaned.find("\n")
            if first_newline != -1:
                cleaned = cleaned[first_newline + 1:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        # Find the JSON array boundaries — LLM sometimes appends extra text
        start = cleaned.find("[")
        if start == -1:
            raise ValueError(f"No JSON array found in LLM response:\n{text[:300]}")

        # Walk forward to find the matching closing bracket
        depth = 0
        end = -1
        in_string = False
        escape_next = False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = i
                    break

        if end == -1:
            raise ValueError(f"Unclosed JSON array in LLM response:\n{text[:300]}")

        try:
            questions = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in array: {exc}\n{cleaned[start:start+200]}") from exc

        if not isinstance(questions, list):
            raise ValueError(f"Expected a JSON array, got {type(questions).__name__}")

        if len(questions) < 5:
            raise ValueError(f"Expected ~15 questions, only got {len(questions)}")

        # Normalize fields
        for i, q in enumerate(questions):
            q["id"] = q.get("id", i + 1)
            q.setdefault("category", "experience")
            q.setdefault("question", "")
            q.setdefault("context", "")
            q.setdefault("difficulty", "medium")
            q.setdefault("ideal_signals", [])

        return questions
