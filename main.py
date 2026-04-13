"""Entry point for the Interviewer Agent — runs the full pipeline end to end."""

import sys
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from src.evaluation.cheat_detector import CheatDetector
from src.evaluation.scorer import AnswerScorer
from src.ingestion.github_fetcher import GitHubFetcher
from src.ingestion.resume_parser import ResumeParser
from src.interview.personas import get_persona, list_personas
from src.interview.question_generator import QuestionGenerator
from src.interview.session import InterviewSession
from src.report.generator import ReportGenerator

load_dotenv()
console = Console()


# -- Banner ----------------------------------------------------------------

BANNER = r"""
  ___       _                  _                        _                    _
 |_ _|_ __ | |_ ___ _ ____   _(_) _____      _____ _ __| |    __ _  __ _  __| |_
  | || '_ \| __/ _ \ '__\ \ / / |/ _ \ \ /\ / / _ \ '__| |   / _` |/ _` |/ _` __|
  | || | | | ||  __/ |   \ V /| |  __/\ V  V /  __/ |  | |  | (_| | (_| | (_| |_
 |___|_| |_|\__\___|_|    \_/ |_|\___| \_/\_/ \___|_|  |_|   \__,_|\__, |\__,_\__|
                                                                     |___/
"""


def main():
    """Run the complete interview pipeline: ingest -> generate -> interview -> evaluate -> report."""
    console.print(f"[bold cyan]{BANNER}[/bold cyan]")
    console.print(
        "  [dim]AI-powered interviews grounded in real experience[/dim]\n"
    )

    # -- Step 0: Get candidate profile -------------------------------------
    profile = _get_candidate_profile()

    # -- Step 1: Choose persona (adapted to detected industry) -------------
    industry = profile.get("industry", "")
    detected_role = profile.get("detected_role", "")
    persona = _select_persona(industry, detected_role)

    # -- Step 2: Generate questions ----------------------------------------
    console.print()
    with _spinner("Generating personalized questions...") as (progress, task):
        try:
            generator = QuestionGenerator()
            questions = generator.generate(profile, persona)
        except (EnvironmentError, RuntimeError) as exc:
            progress.stop()
            _error(str(exc))
            sys.exit(1)
        cat_counts = {}
        for q in questions:
            cat_counts[q.get("category", "?")] = cat_counts.get(q.get("category", "?"), 0) + 1
        cats = " | ".join(f"{c}: {n}" for c, n in cat_counts.items())
        progress.update(task, description=f"[green]{len(questions)} questions ready ({cats})[/green]")

    # Generate persona intro (streamed)
    console.print()
    with _spinner("Preparing interviewer...") as (progress, task):
        intro_text = generator.generate_intro(profile, persona)
        progress.update(task, description=f"[green]{persona.get('short_name', persona['name'])} is ready[/green]")

    # -- Step 3: Run interview ---------------------------------------------
    session = InterviewSession(
        questions=questions,
        candidate_name=profile.get("name", "Candidate"),
        persona=persona,
    )
    session.show_intro(intro_text)
    responses = session.run()

    # -- Step 4: Score answers ---------------------------------------------
    answered_count = sum(1 for r in responses if not r["skipped"])
    if answered_count == 0:
        console.print(Panel(
            "[yellow]No questions were answered. Skipping evaluation.[/yellow]",
            border_style="yellow",
        ))
        sys.exit(0)

    console.print()
    with _progress_bar("Evaluating answers", total=answered_count) as (progress, task):
        try:
            scorer = AnswerScorer(persona)
            scored = []
            for r in responses:
                result = scorer.score_all([r], profile)[0]
                scored.append(result)
                if not r["skipped"]:
                    progress.advance(task)
            score_summary = scorer.compute_summary(scored)
        except EnvironmentError as exc:
            progress.stop()
            _error(str(exc))
            sys.exit(1)

    _show_score_summary(score_summary)

    # -- Step 5: Integrity checks ------------------------------------------
    console.print()
    with _progress_bar("Running integrity checks", total=answered_count) as (progress, task):
        try:
            detector = CheatDetector()
            for r in scored:
                detector.check_all([r], profile)
                if not r["skipped"]:
                    progress.advance(task)
            integrity_summary = detector.summarize_flags(scored)
        except EnvironmentError as exc:
            progress.stop()
            _error(str(exc))
            sys.exit(1)

    if integrity_summary["flagged_count"] > 0:
        console.print(f"  [red bold]!! {integrity_summary['flagged_count']} flagged responses[/red bold]")
    elif integrity_summary["suspicious_count"] > 0:
        console.print(f"  [yellow]? {integrity_summary['suspicious_count']} suspicious responses[/yellow]")
    else:
        console.print(f"  [green]All responses clean[/green]")

    # -- Step 6: Generate report -------------------------------------------
    console.print()
    with _spinner("Generating evaluation report...") as (progress, task):
        try:
            reporter = ReportGenerator(persona)
            paths = reporter.generate(profile, scored, score_summary, integrity_summary)
        except (EnvironmentError, RuntimeError) as exc:
            progress.stop()
            _error(str(exc))
            sys.exit(1)
        progress.update(task, description="[green]Reports generated[/green]")

    # -- Done --------------------------------------------------------------
    console.print()
    results = Table.grid(padding=(0, 2))
    results.add_row(
        Text("  Markdown:", style="bold"),
        Text(paths["markdown"], style="cyan underline"),
    )
    results.add_row(
        Text("  PDF:", style="bold"),
        Text(paths["pdf"], style="cyan underline"),
    )

    console.print(Panel(
        results,
        border_style="green",
        title="[bold green]Evaluation Complete[/bold green]",
        padding=(1, 2),
    ))
    console.print()


# -- Profile ingestion ----------------------------------------------------

def _get_candidate_profile() -> dict:
    """Get candidate profile from either resume file or GitHub username."""
    console.print("  [bold]How would you like to provide candidate information?[/bold]\n")
    console.print("    [bold yellow]1[/bold yellow]  Upload a resume (PDF, DOCX, or TXT)")
    console.print("    [bold yellow]2[/bold yellow]  Enter a GitHub username")
    console.print()

    while True:
        choice = console.input("  [bold]Choose (1-2):[/bold] ").strip()
        if choice in ("1", "2"):
            break
        console.print("  [red]Invalid choice. Enter 1 or 2.[/red]")

    if choice == "1":
        return _get_resume_profile()
    else:
        return _get_github_profile()


def _get_resume_profile() -> dict:
    """Parse a resume file and return the candidate profile."""
    console.print()
    file_path = console.input("  [bold]Enter path to resume file:[/bold] ").strip().strip('"').strip("'")

    if not file_path:
        console.print("[red]  No file path provided. Exiting.[/red]")
        sys.exit(1)

    path = Path(file_path)
    if not path.exists():
        console.print(f"[red]  File not found: {file_path}[/red]")
        sys.exit(1)

    console.print()
    with _spinner(f"Parsing resume [bold]{path.name}[/bold]...") as (progress, task):
        try:
            parser = ResumeParser()
            profile = parser.parse_file(str(path))
        except (EnvironmentError, ValueError, RuntimeError) as exc:
            progress.stop()
            _error(str(exc))
            sys.exit(1)
        progress.update(task, description=f"[green]Resume parsed — {profile.get('name', 'Candidate')}[/green]")

    console.print()
    _show_resume_profile_card(profile)

    # Ask for additional context if no GitHub
    console.print()
    console.print("  [dim]Briefly describe your best project or achievement in 2-3 sentences[/dim]")
    console.print("  [dim](Press Enter to skip):[/dim]")
    achievement = console.input("  [cyan]>[/cyan] ").strip()
    if achievement:
        profile["achievement_description"] = achievement

    return profile


def _get_github_profile() -> dict:
    """Fetch GitHub profile (original flow)."""
    console.print()
    if len(sys.argv) > 1:
        username = sys.argv[1]
        console.print(f"  [bold]Candidate:[/bold] {username}\n")
    else:
        username = console.input("  [bold]Enter candidate's GitHub username:[/bold] ").strip()
        console.print()

    if not username:
        console.print("[red]  No username provided. Exiting.[/red]")
        sys.exit(1)

    with _spinner(f"Fetching GitHub profile for [bold]@{username}[/bold]...") as (progress, task):
        try:
            fetcher = GitHubFetcher()
            profile = fetcher.fetch_profile(username)
        except (EnvironmentError, ValueError) as exc:
            progress.stop()
            _error(str(exc))
            sys.exit(1)
        progress.update(task, description=f"[green]Profile loaded — {profile['name']}[/green]")

    console.print()
    _show_github_profile_card(profile)
    return profile


# -- UI helpers ------------------------------------------------------------

def _select_persona(industry: str = "", detected_role: str = "") -> dict:
    """Show interactive persona selection menu."""
    personas = list_personas(industry=industry, detected_role=detected_role)

    console.print()
    console.print("  [bold]Select interviewer persona:[/bold]\n")

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("#", style="bold yellow", width=3)
    table.add_column("Persona", style="bold")
    table.add_column("Style", style="dim")

    for i, p in enumerate(personas, 1):
        color = p.get("color", "white")
        icon = p.get("icon", "?")
        table.add_row(
            str(i),
            Text(f"{icon}  {p.get('short_name', p['name'])}", style=f"bold {color}"),
            p.get("tagline", ""),
        )

    console.print(table)
    console.print()

    while True:
        choice = console.input("  [bold]Choose (1-4):[/bold] ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(personas):
                selected = personas[idx]
                color = selected.get("color", "white")
                name = selected.get("short_name", selected["name"])
                console.print(f"\n  [{color} bold]{selected['icon']}  {name}[/{color} bold] selected\n")
                return selected
        except ValueError:
            pass
        console.print("  [red]Invalid choice. Enter a number 1-4.[/red]")


def _show_resume_profile_card(profile: dict):
    """Display a rich candidate profile card for resume-based profiles."""
    skills = profile.get("skills", [])[:8]
    skill_items = [f"[bold]{s}[/bold]" for s in skills]

    projects = profile.get("projects", [])
    project_items = []
    for p in projects[:3]:
        tools = ", ".join(p.get("technologies_or_tools", [])[:3])
        project_items.append(f"[cyan]{p.get('name', 'N/A')}[/cyan] [dim]{tools}[/dim]")

    left = Table.grid(padding=(0, 0))
    left.add_row(Text(f"  {profile.get('name', 'Candidate')}", style="bold white"))
    left.add_row(Text(f"  {profile.get('detected_role', '')}", style="dim"))
    if profile.get("bio"):
        left.add_row(Text(f"  {profile['bio']}", style="italic"))
    left.add_row(Text(""))
    left.add_row(Text("  Skills:", style="bold cyan"))
    for item in skill_items:
        left.add_row(Text(f"    "), Text.from_markup(item))

    right = Table.grid(padding=(0, 0))
    right.add_row(Text(f"  [dim]Industry:[/dim] {profile.get('industry', 'N/A')}"))
    right.add_row(Text(f"  [dim]Seniority:[/dim] {profile.get('seniority', 'N/A')}"))
    right.add_row(Text(f"  [dim]Experience:[/dim] {profile.get('years_of_experience', 'N/A')} years"))
    right.add_row(Text(""))
    right.add_row(Text("  Projects:", style="bold cyan"))
    for item in project_items:
        right.add_row(Text(f"    "), Text.from_markup(item))

    layout = Columns([left, right], padding=(0, 4))

    console.print(Panel(
        layout,
        border_style="blue",
        title="[bold]Candidate Profile[/bold]",
        padding=(1, 1),
    ))


def _show_github_profile_card(profile: dict):
    """Display a rich candidate profile card for GitHub-based profiles."""
    languages = profile.get("languages", {})
    lang_items = []
    for lang, info in list(languages.items())[:6]:
        pct = info["percentage"]
        bar_len = max(1, int(pct / 5))
        lang_items.append(f"[bold]{lang}[/bold] [green]{'#' * bar_len}[/green] {pct}%")

    repos = profile.get("top_repos", [])
    repo_items = []
    for r in repos:
        stars = f"[yellow]*[/yellow]{r['stars']}" if r["stars"] else ""
        repo_items.append(f"[cyan]{r['name']}[/cyan] {stars} [dim]{r.get('primary_language', '')}[/dim]")

    commit_info = profile.get("commit_patterns", {})

    left = Table.grid(padding=(0, 0))
    left.add_row(Text(f"  {profile['name']}", style="bold white"))
    left.add_row(Text(f"  @{profile['username']}", style="dim"))
    if profile.get("bio"):
        left.add_row(Text(f"  {profile['bio']}", style="italic"))
    left.add_row(Text(""))
    left.add_row(Text("  Languages:", style="bold cyan"))
    for item in lang_items:
        left.add_row(Text(f"    "), Text.from_markup(item))

    right = Table.grid(padding=(0, 0))
    right.add_row(Text("  Top Repos:", style="bold cyan"))
    for item in repo_items:
        right.add_row(Text(f"    "), Text.from_markup(item))
    right.add_row(Text(""))
    right.add_row(Text(f"  [dim]Followers:[/dim] {profile.get('followers', 0)}  [dim]Public repos:[/dim] {profile.get('public_repos', 0)}"))
    right.add_row(Text(f"  [dim]Account age:[/dim] {profile.get('account_age_years', '?')} years"))
    right.add_row(Text(f"  [dim]Commit pace:[/dim] {commit_info.get('active_days_per_week', '?')} days/week"))

    layout = Columns([left, right], padding=(0, 4))

    console.print(Panel(
        layout,
        border_style="blue",
        title="[bold]Candidate Profile[/bold]",
        padding=(1, 1),
    ))


def _show_score_summary(summary: dict):
    """Display the scoring results as a rich table."""
    table = Table(title="Score Summary", show_header=True, header_style="bold", border_style="cyan")
    table.add_column("Dimension", style="bold")
    table.add_column("Score", justify="center")
    table.add_column("Bar", justify="left")

    for dim, val in summary.get("by_dimension", {}).items():
        filled = int(round(val / 10 * 15))
        bar = f"[green]{'#' * filled}[/green][dim]{'.' * (15 - filled)}[/dim]"
        color = "green" if val >= 7 else "yellow" if val >= 5 else "red"
        table.add_row(dim.title(), f"[{color}]{val}/10[/{color}]", bar)

    overall = summary.get("overall", 0)
    color = "green bold" if overall >= 7 else "yellow bold" if overall >= 5 else "red bold"
    table.add_row("", "", "")
    table.add_row("[bold]OVERALL[/bold]", f"[{color}]{overall}/10[/{color}]", "")

    console.print()
    console.print(table)


def _spinner(description: str):
    """Context manager for a spinner progress indicator."""
    progress = Progress(
        SpinnerColumn("dots"),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        console=console,
    )
    task = progress.add_task(description, total=None)
    progress.start()

    class _Ctx:
        def __enter__(self):
            return progress, task
        def __exit__(self, *args):
            progress.stop()

    return _Ctx()


def _progress_bar(description: str, total: int):
    """Context manager for a progress bar."""
    progress = Progress(
        SpinnerColumn("dots"),
        TextColumn("{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("[bold]{task.completed}/{task.total}[/bold]"),
        TimeElapsedColumn(),
        console=console,
    )
    task = progress.add_task(description, total=total)
    progress.start()

    class _Ctx:
        def __enter__(self):
            return progress, task
        def __exit__(self, *args):
            progress.stop()

    return _Ctx()


def _error(message: str):
    """Display a formatted error panel."""
    console.print()
    console.print(Panel(
        f"[red]{message}[/red]",
        border_style="red",
        title="[bold red]Error[/bold red]",
        padding=(1, 2),
    ))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n  [dim]Interview cancelled.[/dim]")
        sys.exit(0)
    except Exception as exc:
        _error(f"Unexpected error: {exc}\n\n{traceback.format_exc()}")
        sys.exit(1)
