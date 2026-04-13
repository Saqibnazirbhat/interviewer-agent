"""SQLite store for completed interview results — powers the comparison dashboard.

Each row represents one completed interview with aggregated scores, enabling
filtering, sorting, and side-by-side candidate comparison.
"""

import json
import logging
import time
from pathlib import Path

import aiosqlite

logger = logging.getLogger("interviewer.results")

DB_PATH = Path("data") / "results.db"


class ResultsStore:
    """Async SQLite store for completed interview results."""

    def __init__(self, db_path: str | Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False

    async def _ensure_db(self):
        if self._initialized:
            return
        async with aiosqlite.connect(str(self.db_path)) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    session_id TEXT PRIMARY KEY,
                    candidate_name TEXT NOT NULL,
                    role TEXT DEFAULT '',
                    industry TEXT DEFAULT '',
                    persona_name TEXT DEFAULT '',
                    overall_score REAL DEFAULT 0,
                    accuracy REAL DEFAULT 0,
                    depth REAL DEFAULT 0,
                    communication REAL DEFAULT 0,
                    ownership REAL DEFAULT 0,
                    answered_count INTEGER DEFAULT 0,
                    skipped_count INTEGER DEFAULT 0,
                    total_questions INTEGER DEFAULT 0,
                    integrity_score REAL DEFAULT 10,
                    flagged_count INTEGER DEFAULT 0,
                    recommendation TEXT DEFAULT '',
                    by_category TEXT DEFAULT '{}',
                    skills_demonstrated TEXT DEFAULT '[]',
                    contradiction_count INTEGER DEFAULT 0,
                    completed_at REAL NOT NULL
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_results_completed
                ON results(completed_at DESC)
            """)
            await db.commit()
        self._initialized = True

    async def save(self, session_id: str, data: dict):
        """Save a completed interview result."""
        await self._ensure_db()
        now = time.time()
        async with aiosqlite.connect(str(self.db_path)) as db:
            await db.execute(
                """INSERT INTO results (
                    session_id, candidate_name, role, industry, persona_name,
                    overall_score, accuracy, depth, communication, ownership,
                    answered_count, skipped_count, total_questions,
                    integrity_score, flagged_count, recommendation,
                    by_category, skills_demonstrated, contradiction_count,
                    completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    overall_score=excluded.overall_score,
                    accuracy=excluded.accuracy,
                    depth=excluded.depth,
                    communication=excluded.communication,
                    ownership=excluded.ownership,
                    recommendation=excluded.recommendation,
                    completed_at=excluded.completed_at""",
                (
                    session_id,
                    data.get("candidate_name", "Unknown"),
                    data.get("role", ""),
                    data.get("industry", ""),
                    data.get("persona_name", ""),
                    data.get("overall_score", 0),
                    data.get("accuracy", 0),
                    data.get("depth", 0),
                    data.get("communication", 0),
                    data.get("ownership", 0),
                    data.get("answered_count", 0),
                    data.get("skipped_count", 0),
                    data.get("total_questions", 0),
                    data.get("integrity_score", 10),
                    data.get("flagged_count", 0),
                    data.get("recommendation", ""),
                    json.dumps(data.get("by_category", {})),
                    json.dumps(data.get("skills_demonstrated", [])),
                    data.get("contradiction_count", 0),
                    now,
                ),
            )
            await db.commit()

    async def get_all(self, limit: int = 100, offset: int = 0) -> list[dict]:
        """Retrieve all results, newest first."""
        await self._ensure_db()
        async with aiosqlite.connect(str(self.db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM results ORDER BY completed_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
            rows = await cursor.fetchall()
            results = []
            for row in rows:
                r = dict(row)
                r["by_category"] = json.loads(r.get("by_category", "{}"))
                r["skills_demonstrated"] = json.loads(r.get("skills_demonstrated", "[]"))
                results.append(r)
            return results

    async def get_by_id(self, session_id: str) -> dict | None:
        """Retrieve a single result by session ID."""
        await self._ensure_db()
        async with aiosqlite.connect(str(self.db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM results WHERE session_id = ?", (session_id,)
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            r = dict(row)
            r["by_category"] = json.loads(r.get("by_category", "{}"))
            r["skills_demonstrated"] = json.loads(r.get("skills_demonstrated", "[]"))
            return r

    async def count(self) -> int:
        """Count total results."""
        await self._ensure_db()
        async with aiosqlite.connect(str(self.db_path)) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM results")
            row = await cursor.fetchone()
            return row[0] if row else 0
