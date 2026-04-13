"""Fetches and analyzes a candidate's GitHub profile for interview preparation."""

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from github import Auth, Github, GithubException

load_dotenv()


class GitHubFetcher:
    """Pulls a candidate's public GitHub data and distills it into an interview-ready profile."""

    def __init__(self):
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            raise EnvironmentError(
                "GITHUB_TOKEN not found in .env file.\n"
                "Create a token at https://github.com/settings/tokens (read-only public access is enough).\n"
                "Then add it to your .env file: GITHUB_TOKEN=ghp_your_token_here"
            )
        self.client = Github(auth=Auth.Token(token), per_page=30)

    def fetch_profile(self, username: str) -> dict:
        """Build a complete candidate profile from their GitHub presence."""
        try:
            user = self.client.get_user(username)
            # Force a fetch to validate the user exists
            _ = user.login
        except GithubException as exc:
            msg = exc.data.get("message", str(exc)) if hasattr(exc, "data") and exc.data else str(exc)
            raise ValueError(
                f"Could not find GitHub user '{username}': {msg}\n"
                f"Double-check the username and ensure your GITHUB_TOKEN is valid."
            ) from exc

        repos = self._rank_repos(user)
        top_repos = repos[:5]

        profile = {
            "username": user.login,
            "name": user.name or user.login,
            "bio": user.bio or "",
            "public_repos": user.public_repos,
            "followers": user.followers,
            "created_at": user.created_at.isoformat(),
            "account_age_years": round(
                (datetime.now(timezone.utc) - user.created_at.replace(tzinfo=timezone.utc)).days / 365, 1
            ),
            "top_repos": [self._analyze_repo(r) for r in top_repos],
            "languages": self._aggregate_languages(top_repos),
            "commit_patterns": self._analyze_commit_patterns(top_repos),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

        self._save_profile(username, profile)
        return profile

    def _rank_repos(self, user) -> list:
        """Rank non-fork repos by a weighted score of stars, forks, recency, and size."""
        repos = []
        try:
            for repo in user.get_repos(type="owner", sort="updated"):
                if repo.fork:
                    continue
                repos.append(repo)
                if len(repos) >= 20:
                    break
        except GithubException:
            pass

        if not repos:
            raise ValueError(
                f"User '{user.login}' has no non-fork public repositories to analyze."
            )

        def score(repo):
            stars = repo.stargazers_count
            forks = repo.forks_count
            pushed = repo.pushed_at or repo.created_at
            days_since_push = (datetime.now(timezone.utc) - pushed.replace(tzinfo=timezone.utc)).days
            recency = max(0, 365 - days_since_push) / 365
            return (stars * 3) + (forks * 2) + (recency * 10) + (repo.size / 1000)

        repos.sort(key=score, reverse=True)
        return repos

    def _analyze_repo(self, repo) -> dict:
        """Extract interview-relevant details from a single repository."""
        languages = {}
        try:
            languages = self._clean_languages(repo.get_languages())
        except GithubException:
            pass

        readme_snippet = self._get_readme_snippet(repo)
        recent_commits = self._get_recent_commits(repo, limit=10)

        topics = []
        try:
            topics = repo.get_topics()
        except GithubException:
            pass

        return {
            "name": repo.name,
            "description": repo.description or "",
            "stars": repo.stargazers_count,
            "forks": repo.forks_count,
            "primary_language": repo.language or "Unknown",
            "languages": languages,
            "topics": topics,
            "open_issues": repo.open_issues_count,
            "readme_snippet": readme_snippet,
            "recent_commits": recent_commits,
            "created_at": repo.created_at.isoformat(),
            "last_pushed": (repo.pushed_at or repo.created_at).isoformat(),
        }

    def _get_recent_commits(self, repo, limit: int = 10) -> list[dict]:
        """Safely iterate commits with a hard cap."""
        commits = []
        try:
            for i, commit in enumerate(repo.get_commits()):
                if i >= limit:
                    break
                commits.append({
                    "message": commit.commit.message.split("\n")[0][:120],
                    "date": commit.commit.author.date.isoformat(),
                })
        except GithubException:
            pass
        return commits

    def _get_readme_snippet(self, repo, max_chars: int = 500) -> str:
        """Grab the first chunk of the README for context."""
        try:
            readme = repo.get_readme()
            content = readme.decoded_content.decode("utf-8", errors="replace")
            return content[:max_chars].strip()
        except GithubException:
            return ""

    @staticmethod
    def _clean_languages(raw: dict) -> dict:
        """Filter out non-integer entries (e.g. 'url') from the languages dict."""
        return {k: v for k, v in raw.items() if isinstance(v, int)}

    def _aggregate_languages(self, repos: list) -> dict:
        """Merge language byte counts across repos into a ranked summary."""
        totals = Counter()
        for repo in repos:
            try:
                totals.update(self._clean_languages(repo.get_languages()))
            except GithubException:
                continue

        total_bytes = sum(totals.values()) or 1
        return {
            lang: {"bytes": count, "percentage": round(count / total_bytes * 100, 1)}
            for lang, count in totals.most_common(10)
        }

    def _analyze_commit_patterns(self, repos: list) -> dict:
        """Derive commit habits — frequency, consistency, message quality."""
        all_dates = []
        message_lengths = []

        for repo in repos[:5]:
            try:
                for i, commit in enumerate(repo.get_commits()):
                    if i >= 30:
                        break
                    date = commit.commit.author.date
                    all_dates.append(date)
                    message_lengths.append(len(commit.commit.message))
            except GithubException:
                continue

        if not all_dates:
            return {"total_sampled": 0, "avg_message_length": 0, "active_days_per_week": 0}

        weekdays = Counter(d.strftime("%A") for d in all_dates)
        unique_weeks = len({d.isocalendar()[:2] for d in all_dates}) or 1

        return {
            "total_sampled": len(all_dates),
            "avg_message_length": round(sum(message_lengths) / len(message_lengths)),
            "active_days_per_week": round(len(set(d.date() for d in all_dates)) / unique_weeks, 1),
            "busiest_day": weekdays.most_common(1)[0][0] if weekdays else "N/A",
            "commit_span_days": (max(all_dates) - min(all_dates)).days if len(all_dates) > 1 else 0,
        }

    def _save_profile(self, username: str, profile: dict) -> Path:
        """Persist the profile to data/ for later reference."""
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        filepath = data_dir / f"{username}.json"
        filepath.write_text(json.dumps(profile, indent=2, default=str), encoding="utf-8")
        return filepath
