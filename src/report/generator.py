"""Generates interview evaluation reports in Markdown and PDF formats.

All report files are encrypted at rest using the shared Fernet key.
"""

import json
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.llm_client import LLMClient, parse_json_object


class ReportGenerator:
    """Produces a comprehensive hiring report from scored interview data."""

    def __init__(self, persona: dict):
        self.llm = LLMClient()
        self.persona = persona
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)

    def generate(
        self,
        profile: dict,
        responses: list[dict],
        score_summary: dict,
        integrity_summary: dict,
        stream_callback=None,
    ) -> dict:
        """Generate both Markdown and PDF reports. Returns paths to both files."""
        narrative = self._generate_narrative(
            profile, responses, score_summary, integrity_summary, stream_callback
        )

        # Sanitize username to prevent path traversal — this is LLM-derived input
        raw_username = profile.get("username", "candidate")
        safe_username = re.sub(r"[^a-zA-Z0-9_\-]", "_", raw_username)[:80] or "candidate"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        base_name = f"{safe_username}_{timestamp}"

        md_path = self._write_markdown(
            base_name, profile, responses, score_summary, integrity_summary, narrative
        )
        pdf_path = self._write_pdf(
            base_name, profile, responses, score_summary, integrity_summary, narrative
        )

        return {"markdown": str(md_path), "pdf": str(pdf_path)}

    def _generate_narrative(self, profile, responses, score_summary, integrity_summary, stream_callback=None) -> dict:
        """Use the LLM (with streaming) to produce the subjective evaluation sections."""
        answered = [r for r in responses if not r["skipped"]]
        qa_summary = ""
        for r in answered[:10]:
            qa_summary += (
                f"\nQ: {r['question']}\n"
                f"A: {r['answer'][:300]}\n"
                f"Scores: {r.get('scores', {})} | Feedback: {r.get('feedback', '')}\n"
            )

        prompt = f"""You are writing a professional hiring evaluation report as a {self.persona['name']}.

{self.persona.get('scoring_bias', '')}

CANDIDATE: {profile.get('name', 'Candidate')} ({profile.get('detected_role', profile.get('username', 'N/A'))})
SKILLS/LANGUAGES: {', '.join(profile.get('skills', list(profile.get('languages', {}).keys())))}
OVERALL SCORE: {score_summary['overall']}/10
SCORES BY DIMENSION: {json.dumps(score_summary.get('by_dimension', {}))}
INTEGRITY: {integrity_summary['flagged_count']} flagged, {integrity_summary['suspicious_count']} suspicious out of {integrity_summary['total_questions']} questions

INTERVIEW Q&A SAMPLE:
{qa_summary}

Generate the following sections as a JSON object:
{{
  "executive_summary": "3-4 sentence overview of the candidate's performance",
  "strengths": ["strength 1", "strength 2", "strength 3"],
  "weaknesses": ["weakness 1", "weakness 2", "weakness 3"],
  "red_flags": ["any concerns worth noting — empty array if none"],
  "recommendation": "STRONG_HIRE | HIRE | LEAN_HIRE | LEAN_NO_HIRE | NO_HIRE | STRONG_NO_HIRE",
  "recommendation_rationale": "2-3 sentence justification",
  "followup_questions": ["q1", "q2", "q3", "q4", "q5"]
}}

Be honest, specific, and reference actual answers. Raw JSON only."""

        try:
            full_text = ""
            for chunk in self.llm.generate_stream(prompt):
                full_text += chunk
                if stream_callback:
                    stream_callback(chunk)
            return self._parse_narrative(full_text)
        except Exception:
            return self._default_narrative()

    def _parse_narrative(self, text: str) -> dict:
        """Parse the narrative JSON from the LLM response."""
        try:
            return parse_json_object(text)
        except (ValueError, json.JSONDecodeError):
            return self._default_narrative()

    def _default_narrative(self) -> dict:
        return {
            "executive_summary": "Evaluation narrative could not be generated.",
            "strengths": [],
            "weaknesses": [],
            "red_flags": [],
            "recommendation": "LEAN_NO_HIRE",
            "recommendation_rationale": "Automated evaluation failed — manual review required.",
            "followup_questions": [],
        }

    # ---- MARKDOWN REPORT -------------------------------------------------

    def _write_markdown(self, base_name, profile, responses, score_summary, integrity_summary, narrative) -> Path:
        path = self.output_dir / f"{base_name}.md.enc"
        lines = []

        # Header
        lines.append(f"# Interview Evaluation Report")
        lines.append("")
        lines.append(f"| Field | Value |")
        lines.append(f"|-------|-------|")
        lines.append(f"| **Candidate** | {profile.get('name', 'Candidate')} ({profile.get('detected_role', '@' + profile.get('username', 'N/A'))}) |")
        lines.append(f"| **Date** | {datetime.now(timezone.utc).strftime('%B %d, %Y')} |")
        lines.append(f"| **Interviewer** | {self.persona['name']} |")
        lines.append(f"| **Overall Score** | {score_summary['overall']}/10 |")
        lines.append(f"| **Recommendation** | **{narrative.get('recommendation', 'N/A')}** |")
        lines.append("")

        # Executive Summary
        lines.append("---")
        lines.append("## Executive Summary")
        lines.append(narrative.get("executive_summary", "N/A"))
        lines.append("")

        # Score Breakdown
        lines.append("## Score Breakdown")
        lines.append("")
        lines.append("### By Dimension")
        lines.append("| Dimension | Score | |")
        lines.append("|-----------|-------|-|")
        for dim, val in score_summary.get("by_dimension", {}).items():
            bar = self._text_bar(val)
            lines.append(f"| {dim.title()} | {val}/10 | `{bar}` |")
        lines.append("")

        lines.append("### By Category")
        lines.append("| Category | Avg Score |")
        lines.append("|----------|-----------|")
        for cat, val in score_summary.get("by_category", {}).items():
            lines.append(f"| {cat.title()} | {val}/40 |")
        lines.append("")

        lines.append(f"### Stats")
        lines.append(f"- Answered: {score_summary.get('answered_count', 0)}")
        lines.append(f"- Skipped: {score_summary.get('skipped_count', 0)}")
        lines.append("")

        # Strengths & Weaknesses
        lines.append("## Strengths")
        for s in narrative.get("strengths", []):
            lines.append(f"- {s}")
        lines.append("")

        lines.append("## Weaknesses")
        for w in narrative.get("weaknesses", []):
            lines.append(f"- {w}")
        lines.append("")

        # Integrity
        lines.append("## Integrity Check")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Clean | {integrity_summary['clean_count']} |")
        lines.append(f"| Suspicious | {integrity_summary['suspicious_count']} |")
        lines.append(f"| Flagged | {integrity_summary['flagged_count']} |")
        lines.append(f"| Integrity Score | {integrity_summary['integrity_score']}/10 |")
        if integrity_summary.get("all_flags"):
            lines.append("")
            lines.append("### Flags")
            for f in integrity_summary["all_flags"]:
                lines.append(f"- **Q{f['question_id']}**: {f['flag']}")
        lines.append("")

        # Red Flags
        if narrative.get("red_flags"):
            lines.append("## Red Flags")
            for rf in narrative["red_flags"]:
                lines.append(f"- {rf}")
            lines.append("")

        # Recommendation
        lines.append("## Recommendation")
        lines.append(f"**{narrative.get('recommendation', 'N/A')}**")
        lines.append("")
        lines.append(narrative.get("recommendation_rationale", ""))
        lines.append("")

        # Follow-ups
        lines.append("## Suggested Follow-Up Questions")
        for i, q in enumerate(narrative.get("followup_questions", []), 1):
            lines.append(f"{i}. {q}")
        lines.append("")

        # Detailed Q&A
        lines.append("---")
        lines.append("## Detailed Q&A Log")
        lines.append("")
        for r in responses:
            status = "SKIPPED" if r["skipped"] else f"Score: {r.get('score_total', 'N/A')}/40"
            lines.append(f"### Q{r['question_id']} [{r['category'].upper()}] — {status}")
            lines.append(f"> {r['question']}")
            lines.append("")
            if r["skipped"]:
                lines.append("*Skipped*")
            else:
                lines.append(f"**Answer:** {r['answer']}")
                lines.append("")
                lines.append(f"**Time:** {r['time_seconds']}s | **Feedback:** {r.get('feedback', 'N/A')}")
                scores = r.get("scores", {})
                lines.append(
                    f"**Scores:** Accuracy={scores.get('accuracy', '-')} | "
                    f"Depth={scores.get('depth', '-')} | "
                    f"Communication={scores.get('communication', '-')} | "
                    f"Ownership={scores.get('ownership', '-')}"
                )
                integrity = r.get("integrity", {})
                if integrity.get("flags"):
                    lines.append(f"**Integrity flags:** {', '.join(integrity['flags'])}")
            lines.append("")

        lines.append("---")
        lines.append(f"*Generated by Interviewer Agent | {self.persona['name']} persona*")

        from src.web.session_store import encrypt_bytes
        path.write_bytes(encrypt_bytes("\n".join(lines).encode("utf-8")))
        return path

    @staticmethod
    def _text_bar(value: float, max_val: float = 10, width: int = 20) -> str:
        filled = int(round(value / max_val * width))
        return "#" * filled + "." * (width - filled)

    # ---- PDF REPORT ------------------------------------------------------

    def _write_pdf(self, base_name, profile, responses, score_summary, integrity_summary, narrative) -> Path:
        path = self.output_dir / f"{base_name}.pdf.enc"
        # Build PDF into a temp file, then encrypt it
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        import os as _os
        _os.close(tmp_fd)
        doc = SimpleDocTemplate(
            tmp_path, pagesize=A4,
            topMargin=20 * mm, bottomMargin=20 * mm,
            leftMargin=18 * mm, rightMargin=18 * mm,
        )
        styles = getSampleStyleSheet()
        story = []

        title_style = ParagraphStyle("ReportTitle", parent=styles["Title"], fontSize=22, spaceAfter=6)
        h2 = ParagraphStyle("H2", parent=styles["Heading2"], spaceBefore=16, spaceAfter=6)
        h3 = ParagraphStyle("H3", parent=styles["Heading3"], spaceBefore=10, spaceAfter=4)
        body = styles["BodyText"]

        # Title block
        story.append(Paragraph("Interview Evaluation Report", title_style))
        story.append(Spacer(1, 4))

        meta_data = [
            ["Candidate", f"{profile.get('name', 'Candidate')} ({profile.get('detected_role', '@' + profile.get('username', 'N/A'))})"],
            ["Date", datetime.now(timezone.utc).strftime("%B %d, %Y")],
            ["Interviewer", self.persona["name"]],
            ["Overall Score", f"{score_summary['overall']}/10"],
            ["Recommendation", narrative.get("recommendation", "N/A")],
        ]
        meta_table = Table(meta_data, colWidths=[100, 350])
        meta_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("LINEBELOW", (0, -1), (-1, -1), 1, colors.grey),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 14))

        # Executive Summary
        story.append(Paragraph("Executive Summary", h2))
        story.append(Paragraph(self._escape_xml(narrative.get("executive_summary", "N/A")), body))
        story.append(Spacer(1, 8))

        # Score table
        story.append(Paragraph("Score Breakdown", h2))
        score_data = [["Dimension", "Score", "Bar"]]
        for dim, val in score_summary.get("by_dimension", {}).items():
            bar = self._text_bar(val, width=15)
            score_data.append([dim.title(), f"{val}/10", bar])
        score_table = Table(score_data, colWidths=[100, 60, 200])
        score_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (2, 1), (2, -1), "Courier"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#ecf0f1")]),
            ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ]))
        story.append(score_table)
        story.append(Spacer(1, 10))

        # Strengths & Weaknesses
        story.append(Paragraph("Strengths", h2))
        for s in narrative.get("strengths", []):
            story.append(Paragraph(f"\u2022  {self._escape_xml(s)}", body))
        story.append(Spacer(1, 6))

        story.append(Paragraph("Weaknesses", h2))
        for w in narrative.get("weaknesses", []):
            story.append(Paragraph(f"\u2022  {self._escape_xml(w)}", body))
        story.append(Spacer(1, 6))

        # Integrity
        story.append(Paragraph("Integrity Check", h2))
        integrity_data = [
            ["Clean", str(integrity_summary["clean_count"])],
            ["Suspicious", str(integrity_summary["suspicious_count"])],
            ["Flagged", str(integrity_summary["flagged_count"])],
            ["Integrity Score", f"{integrity_summary['integrity_score']}/10"],
        ]
        int_table = Table(integrity_data, colWidths=[120, 80])
        int_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ]))
        story.append(int_table)
        story.append(Spacer(1, 6))

        # Red flags
        if narrative.get("red_flags"):
            story.append(Paragraph("Red Flags", h2))
            for rf in narrative["red_flags"]:
                story.append(Paragraph(f"\u2022  {self._escape_xml(rf)}", body))
            story.append(Spacer(1, 6))

        # Recommendation
        story.append(Paragraph("Recommendation", h2))
        story.append(Paragraph(f"<b>{self._escape_xml(narrative.get('recommendation', 'N/A'))}</b>", body))
        story.append(Paragraph(self._escape_xml(narrative.get("recommendation_rationale", "")), body))
        story.append(Spacer(1, 6))

        # Follow-up questions
        story.append(Paragraph("Suggested Follow-Up Questions", h2))
        for i, q in enumerate(narrative.get("followup_questions", []), 1):
            story.append(Paragraph(f"{i}. {self._escape_xml(q)}", body))

        # Q&A Log
        story.append(Spacer(1, 14))
        story.append(Paragraph("Detailed Q&amp;A Log", h2))
        for r in responses:
            status = "SKIPPED" if r["skipped"] else f"Score: {r.get('score_total', 'N/A')}/40"
            story.append(Paragraph(
                f"<b>Q{r['question_id']} [{r['category'].upper()}] — {status}</b>", h3
            ))
            story.append(Paragraph(f"<i>{self._escape_xml(r['question'])}</i>", body))
            if not r["skipped"]:
                story.append(Paragraph(f"<b>Answer:</b> {self._escape_xml(r['answer'][:500])}", body))
                story.append(Paragraph(f"<b>Feedback:</b> {self._escape_xml(r.get('feedback', 'N/A'))}", body))
            story.append(Spacer(1, 4))

        try:
            doc.build(story)
            # Encrypt the PDF and write to the final .enc path
            from src.web.session_store import encrypt_bytes
            pdf_data = Path(tmp_path).read_bytes()
            path.write_bytes(encrypt_bytes(pdf_data))
        except Exception as exc:
            raise RuntimeError(f"PDF generation failed: {exc}") from exc
        finally:
            # Always delete the plaintext temp file
            Path(tmp_path).unlink(missing_ok=True)

        return path

    @staticmethod
    def _escape_xml(text: str) -> str:
        """Escape XML special characters for ReportLab paragraphs."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
