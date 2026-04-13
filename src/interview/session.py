"""Runs the live interview session in the terminal with a rich UI."""

import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, Future

from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from src.interview.adaptive import AdaptiveEngine
from src.interview.candidate_model import CandidateModel
from src.interview.followup import FollowUpGenerator

QUESTION_TIME_LIMIT = 100  # seconds per question


DIFFICULTY_STYLES = {
    "EASY": ("green", ">>"),
    "MEDIUM": ("yellow", ">>>"),
    "HARD": ("red", ">>>>"),
}

CATEGORY_ICONS = {
    "technical": "T",
    "behavioral": "B",
    "architecture": "A",
    "gotcha": "G",
    "experience": "E",
    "project": "P",
    "skill": "S",
    "situational": "?",
    "curveball": "!",
}


class InterviewSession:
    """Conducts a timed, interactive interview with adaptive difficulty."""

    def __init__(self, questions: list[dict], candidate_name: str, persona: dict):
        self.questions = questions
        self.candidate_name = candidate_name
        self.persona = persona
        self.console = Console()
        self.responses = []
        self.engine = AdaptiveEngine(questions)
        self.followup_gen = FollowUpGenerator(persona)
        self.candidate_model = CandidateModel()

    def run(self) -> list[dict]:
        """Execute the full interview loop with adaptive question selection."""
        self._show_welcome()

        total = len(self.questions)
        # First question — always easy/medium
        question = self.engine.pick_first()
        idx = 0

        while question is not None:
            response = self._ask_question(idx, question, total)
            self.responses.append(response)

            # Record in candidate model for cross-answer reasoning
            try:
                self.candidate_model.record_answer(
                    question=question,
                    answer=response["answer"],
                    scores=response.get("scores"),
                    skipped=response["skipped"],
                )
            except Exception:
                pass

            # Follow-up logic: probe deeper if the answer warrants it
            if not response["skipped"] and response["answer"]:
                try:
                    fu_context = self.candidate_model.get_context_for_followup()
                    fu = self.followup_gen.should_followup(
                        question=question,
                        answer=response["answer"],
                        time_seconds=response["time_seconds"],
                        cross_answer_context=fu_context,
                    )
                    if fu["action"] != "move_on" and fu["followup_question"]:
                        fu_response = self._ask_followup(idx, fu, question, total)
                        self.responses.append(fu_response)
                except Exception:
                    pass  # follow-up is optional — don't break the interview

            # Feed result to adaptive engine
            if response["skipped"]:
                self.engine.record_skip()
            else:
                self.engine.record_score(response.get("score_total_raw", 20))

            idx += 1
            remaining = total - idx
            if remaining > 0:
                self._show_progress_bar(idx, total)
                # Show adaptive status
                status = self.engine.get_status()
                perf = status["rolling_performance"]
                target = status["target_difficulty"]
                color = "green" if perf >= 7 else "yellow" if perf >= 4.5 else "red"
                self.console.print(
                    f"  [dim]Adapting: performance [/dim][{color} bold]{perf}/10[/{color} bold]"
                    f"[dim] → next difficulty:[/dim] [{color}]{target}[/{color}]"
                )
                question = self.engine.pick_next()
            else:
                question = None

        self._show_closing()
        return self.responses

    def show_intro(self, intro_text: str):
        """Display a streamed persona intro with typing effect."""
        color = self.persona.get("color", "cyan")
        name = self.persona["name"]

        self.console.print()
        self.console.print(f"  [{color} bold]{name}:[/{color} bold]", end=" ")
        for char in intro_text:
            print(char, end="", flush=True)
            time.sleep(0.015)
        print()
        self.console.print()

    # -- internals ---------------------------------------------------------

    def _show_welcome(self):
        """Display the interview banner with persona info."""
        color = self.persona.get("color", "cyan")
        icon = self.persona.get("icon", "?")

        # Build persona badge
        persona_info = Table.grid(padding=(0, 1))
        persona_info.add_row(
            Text(f" {icon} ", style=f"bold white on {color}"),
            Text(f" {self.persona['name']} ", style=f"bold {color}"),
            Text(f"  {self.persona.get('tagline', '')}", style="dim"),
        )

        # Build header
        header = Table.grid(padding=(0, 0))
        header.add_row(Text(""))
        header.add_row(Text("  INTERVIEWER AGENT", style="bold cyan"))
        header.add_row(Text(f"  Candidate: {self.candidate_name}", style="bold white"))
        header.add_row(Text(""))
        header.add_row(persona_info)
        header.add_row(Text(""))

        self.console.print()
        self.console.print(Panel(
            header,
            border_style=color,
            title=f"[bold]Interview Session[/bold]",
            subtitle=f"[dim]{len(self.questions)} questions[/dim]",
        ))

        # Instructions
        instructions = Table.grid(padding=(0, 2))
        instructions.add_row(
            Text("  ENTER", style="bold green"),
            Text("Submit answer (blank line)", style="dim"),
        )
        instructions.add_row(
            Text("  skip", style="bold yellow"),
            Text("Skip current question", style="dim"),
        )
        instructions.add_row(
            Text("  quit", style="bold red"),
            Text("End interview early", style="dim"),
        )
        self.console.print(instructions)
        self.console.print()

        self.console.input("[dim]  Press Enter to begin the interview...[/dim]")

    def _ask_question(self, idx: int, question: dict, total: int = 0) -> dict:
        """Present one question, collect the answer, track timing."""
        if total == 0:
            total = len(self.questions)
        category = question.get("category", "general").upper()
        difficulty = question.get("difficulty", "medium").upper()
        question_text = question.get("question", "")
        context = question.get("context", "")

        cat_icon = CATEGORY_ICONS.get(question.get("category", ""), "?")

        # Divider
        self.console.print()
        self.console.print(Rule(style="dim"))

        # Question number and metadata bar
        meta = Table.grid(padding=(0, 1))
        meta.add_row(
            Text(f" Q{idx + 1}/{total} ", style="bold black on yellow"),
            Text(f" {cat_icon} ", style=f"bold white on magenta"),
            Text(f" {category} ", style="bold magenta"),
        )
        self.console.print(meta)

        if context:
            self.console.print(f"  [dim italic]Targeting: {context}[/dim italic]")
        self.console.print()

        # Question body with typing effect
        color = self.persona.get("color", "cyan")
        self.console.print(Panel(
            question_text,
            border_style=color,
            padding=(1, 3),
            title=f"[{color}]{self.persona['name']}[/{color}]",
        ))
        self.console.print()

        # Collect answer with 2-minute countdown
        self.console.print("  [dim]Your answer (blank line to submit, 'skip' to pass, 'quit' to end):[/dim]")
        self.console.print(f"  [dim]You have [bold]{QUESTION_TIME_LIMIT}[/bold] seconds[/dim]\n")

        start_time = time.time()
        timed_out = False
        answer_lines: list[str] = []

        # Run input collection in a thread so we can enforce the time limit
        input_done = threading.Event()

        def _read_input():
            nonlocal answer_lines
            answer_lines = self._collect_input_timed(input_done)

        input_thread = threading.Thread(target=_read_input, daemon=True)
        input_thread.start()

        # Wait for either input completion or timeout
        remaining = QUESTION_TIME_LIMIT
        while remaining > 0 and not input_done.is_set():
            input_done.wait(timeout=1)
            remaining = QUESTION_TIME_LIMIT - int(time.time() - start_time)
            if not input_done.is_set() and remaining > 0:
                color = "red bold" if remaining <= 30 else "yellow" if remaining <= 60 else "dim"
                print(f"\r  [{color}]{remaining}s[/{color}] remaining  ", end="", flush=True)

        elapsed = round(time.time() - start_time, 1)

        if not input_done.is_set():
            timed_out = True
            print()  # newline after countdown
            self.console.print("\n  [red bold]⏱ Time is up.[/red bold]")

        answer_text = "\n".join(answer_lines).strip()

        # Handle quit
        if answer_text.lower() == "quit":
            self.console.print("[red bold]  Interview ended by candidate.[/red bold]")
            self._show_closing()
            sys.exit(0)

        skipped = answer_text.lower() == "skip" or answer_text == ""

        # Determine timing status
        if timed_out:
            timing_status = "timeout"
        elif elapsed >= QUESTION_TIME_LIMIT * 0.9:
            timing_status = "full"
        else:
            timing_status = "early"

        if skipped:
            self.console.print("  [yellow]>> Skipped[/yellow]")
            answer_text = ""
        else:
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
            self.console.print(f"  [green]>> Recorded[/green] [dim]({time_str})[/dim]")

        return {
            "question_id": question.get("id", idx + 1),
            "category": question.get("category", "general"),
            "question": question_text,
            "answer": answer_text,
            "skipped": skipped,
            "time_seconds": elapsed,
            "timing_status": timing_status,
            "context": context,
            "difficulty": question.get("difficulty", "medium"),
            "ideal_signals": question.get("ideal_signals", []),
            "score_total_raw": 20,  # placeholder — real score comes from scorer later
        }

    def _ask_followup(self, idx: int, followup: dict, original_question: dict, total: int) -> dict:
        """Present a follow-up question and collect the answer."""
        action = followup["action"]
        fu_text = followup["followup_question"]

        color = self.persona.get("color", "cyan")

        self.console.print()
        self.console.print("  [yellow bold]\\[Follow-up][/yellow bold]")
        self.console.print(Panel(
            fu_text,
            border_style=color,
            padding=(1, 3),
            title=f"[{color}]{self.persona['name']}[/{color}]",
        ))
        self.console.print()

        self.console.print("  [dim]Your answer (blank line to submit, 'skip' to pass):[/dim]")
        self.console.print(f"  [dim]You have [bold]{QUESTION_TIME_LIMIT}[/bold] seconds[/dim]\n")

        start_time = time.time()
        timed_out = False
        answer_lines: list[str] = []
        input_done = threading.Event()

        def _read_input():
            nonlocal answer_lines
            answer_lines = self._collect_input_timed(input_done)

        input_thread = threading.Thread(target=_read_input, daemon=True)
        input_thread.start()

        remaining = QUESTION_TIME_LIMIT
        while remaining > 0 and not input_done.is_set():
            input_done.wait(timeout=1)
            remaining = QUESTION_TIME_LIMIT - int(time.time() - start_time)

        elapsed = round(time.time() - start_time, 1)
        if not input_done.is_set():
            timed_out = True
            print()
            self.console.print("\n  [red bold]⏱ Time is up.[/red bold]")

        answer_text = "\n".join(answer_lines).strip()

        skipped = answer_text.lower() == "skip" or answer_text == ""
        timing_status = "timeout" if timed_out else ("full" if elapsed >= QUESTION_TIME_LIMIT * 0.9 else "early")

        if skipped:
            self.console.print("  [yellow]>> Skipped[/yellow]")
            answer_text = ""
        else:
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
            self.console.print(f"  [green]>> Recorded[/green] [dim]({time_str})[/dim]")

        return {
            "question_id": original_question.get("id", idx + 1),
            "category": original_question.get("category", "general"),
            "question": f"[Follow-up] {fu_text}",
            "answer": answer_text,
            "skipped": skipped,
            "time_seconds": elapsed,
            "timing_status": timing_status,
            "context": original_question.get("context", ""),
            "difficulty": original_question.get("difficulty", "medium"),
            "ideal_signals": [],
            "is_followup": True,
            "followup_type": action,
            "score_total_raw": 20,
        }

    def _collect_input(self) -> list[str]:
        """Read multiline input until blank line submission."""
        lines = []
        while True:
            try:
                line = self.console.input("  [cyan]>[/cyan] ")
            except (EOFError, KeyboardInterrupt):
                break
            if line.strip().lower() in ("skip", "quit"):
                return [line.strip()]
            if line == "" and lines:
                break
            lines.append(line)
        return lines

    def _collect_input_timed(self, done_event: threading.Event) -> list[str]:
        """Read multiline input until blank line, setting done_event when finished."""
        lines = []
        try:
            while not done_event.is_set():
                try:
                    line = input("  > ")
                except (EOFError, KeyboardInterrupt):
                    break
                if line.strip().lower() in ("skip", "quit"):
                    lines = [line.strip()]
                    break
                if line == "" and lines:
                    break
                lines.append(line)
        finally:
            done_event.set()
        return lines

    def _show_progress_bar(self, completed: int, total: int = 0):
        """Show a mini progress indicator between questions."""
        if total == 0:
            total = len(self.questions)
        filled = int(completed / total * 20)
        empty = 20 - filled
        bar = f"[green]{'#' * filled}[/green][dim]{'.' * empty}[/dim]"
        pct = int(completed / total * 100)
        self.console.print(f"\n  [{bar}] {pct}% — {total - completed} remaining\n")

    def _show_closing(self):
        """Show interview summary panel."""
        answered = sum(1 for r in self.responses if not r["skipped"])
        skipped = sum(1 for r in self.responses if r["skipped"])
        total_time = sum(r["time_seconds"] for r in self.responses)
        mins = int(total_time // 60)
        secs = int(total_time % 60)

        # Build summary table
        summary = Table(show_header=False, box=None, padding=(0, 2))
        summary.add_column(style="bold")
        summary.add_column()
        summary.add_row("[green]Answered[/green]", str(answered))
        summary.add_row("[yellow]Skipped[/yellow]", str(skipped))
        summary.add_row("[cyan]Total time[/cyan]", f"{mins}m {secs}s")
        summary.add_row("[magenta]Persona[/magenta]", self.persona["name"])

        self.console.print()
        self.console.print(Panel(
            summary,
            border_style="green",
            title="[bold]Interview Complete[/bold]",
            subtitle="[dim]Generating evaluation...[/dim]",
        ))
        self.console.print()
