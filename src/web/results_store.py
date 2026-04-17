"""SQLite store for completed interview results — powers the comparison dashboard.

All evaluation data is encrypted at rest as a single blob per row.
Only session_id (primary key) and completed_at (for ordering) are stored
in plaintext — no scores, recommendations, or candidate details are
readable without the DATA_ENCRYPTION_KEY.
"""

import json
import logging
import time
from pathlib import Path

import aiosqlite

from src.web.session_store import _fernet

logger = logging.getLogger("interviewer.results")

DB_PATH = Path("data") / "results.db"


def _encrypt_blob(data: dict) -> str:
    """Serialize a dict to JSON and encrypt it."""
    plaintext = json.dumps(data, default=str).encode("utf-8")
    return _fernet.encrypt(plaintext).decode()


def _decrypt_blob(ciphertext: str) -> dict:
    """Decrypt and deserialize a stored blob back to a dict."""
    try:
        plaintext = _fernet.decrypt(ciphertext.encode()).decode()
        return json.loads(plaintext)
    except Exception:
        # Fallback: might be unencrypted legacy data (plain JSON)
        try:
            return json.loads(ciphertext)
        except Exception:
            return {}


class ResultsStore:
    """Async SQLite store for completed interview results."""

    _SCHEMA_VERSION = 2  # v2 = encrypted blob

    def __init__(self, db_path: str | Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False

    async def _ensure_db(self):
        if self._initialized:
            return
        async with aiosqlite.connect(str(self.db_path)) as db:
            # Check if the old wide-column schema exists and migrate
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='results'"
            )
            table_exists = await cursor.fetchone()

            if table_exists:
                # Check schema version by looking for the 'data' column
                col_cursor = await db.execute("PRAGMA table_info(results)")
                columns = {row[1] for row in await col_cursor.fetchall()}
                if "data" not in columns:
                    # Old schema — migrate rows to encrypted blob format
                    await self._migrate_v1_to_v2(db)
            else:
                await db.execute("""
                    CREATE TABLE results (
                        session_id TEXT PRIMARY KEY,
                        data TEXT NOT NULL,
                        completed_at REAL NOT NULL
                    )
                """)
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_results_completed
                    ON results(completed_at DESC)
                """)
                await db.commit()

        self._initialized = True

    async def _migrate_v1_to_v2(self, db):
        """Migrate from the old wide-column schema to encrypted blob."""
        logger.info("Migrating results.db from v1 (wide columns) to v2 (encrypted blob)...")
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM results ORDER BY completed_at DESC")
        old_rows = await cursor.fetchall()
        db.row_factory = None

        # Collect all data before dropping
        migrated = []
        for row in old_rows:
            r = dict(row)
            sid = r.pop("session_id")
            completed_at = r.pop("completed_at")
            # Parse JSON fields that were stored as strings
            for json_field in ("by_category", "skills_demonstrated"):
                if json_field in r and isinstance(r[json_field], str):
                    try:
                        r[json_field] = json.loads(r[json_field])
                    except (json.JSONDecodeError, TypeError):
                        pass
            # Decrypt any previously encrypted PII fields
            for field in ("candidate_name", "role", "industry"):
                val = r.get(field, "")
                if val:
                    try:
                        r[field] = _fernet.decrypt(val.encode()).decode()
                    except Exception:
                        pass  # already plaintext
            migrated.append((sid, r, completed_at))

        # Recreate table with new schema
        await db.execute("DROP TABLE results")
        await db.execute("""
            CREATE TABLE results (
                session_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                completed_at REAL NOT NULL
            )
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_results_completed
            ON results(completed_at DESC)
        """)

        # Re-insert with encrypted blobs
        for sid, data, completed_at in migrated:
            await db.execute(
                "INSERT INTO results (session_id, data, completed_at) VALUES (?, ?, ?)",
                (sid, _encrypt_blob(data), completed_at),
            )
        await db.commit()
        logger.info("Migrated %d results to encrypted blob format.", len(migrated))

    async def save(self, session_id: str, data: dict):
        """Save a completed interview result — fully encrypted."""
        await self._ensure_db()
        now = time.time()
        async with aiosqlite.connect(str(self.db_path)) as db:
            await db.execute(
                """INSERT INTO results (session_id, data, completed_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(session_id) DO UPDATE SET
                       data=excluded.data,
                       completed_at=excluded.completed_at""",
                (session_id, _encrypt_blob(data), now),
            )
            await db.commit()

    async def get_all(self, limit: int = 100, offset: int = 0) -> list[dict]:
        """Retrieve all results, newest first."""
        await self._ensure_db()
        async with aiosqlite.connect(str(self.db_path)) as db:
            cursor = await db.execute(
                "SELECT session_id, data, completed_at FROM results ORDER BY completed_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
            rows = await cursor.fetchall()
            results = []
            for session_id, enc_data, completed_at in rows:
                r = _decrypt_blob(enc_data)
                r["session_id"] = session_id
                r["completed_at"] = completed_at
                results.append(r)
            return results

    async def get_by_id(self, session_id: str) -> dict | None:
        """Retrieve a single result by session ID."""
        await self._ensure_db()
        async with aiosqlite.connect(str(self.db_path)) as db:
            cursor = await db.execute(
                "SELECT data, completed_at FROM results WHERE session_id = ?",
                (session_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            r = _decrypt_blob(row[0])
            r["session_id"] = session_id
            r["completed_at"] = row[1]
            return r

    async def count(self) -> int:
        """Count total results."""
        await self._ensure_db()
        async with aiosqlite.connect(str(self.db_path)) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM results")
            row = await cursor.fetchone()
            return row[0] if row else 0
