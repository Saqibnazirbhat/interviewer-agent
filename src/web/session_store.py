"""SQLite-backed session store for the Interviewer Agent.

Sessions survive server restarts. Expired sessions (>24h) are cleaned
up automatically on each read/write cycle.
"""

import json
import logging
import time
from pathlib import Path

import aiosqlite

logger = logging.getLogger("interviewer.sessions")

DB_PATH = Path("data") / "sessions.db"
TTL_SECONDS = 24 * 60 * 60  # 24 hours

# Fields that are stored as-is (not JSON-serialized)
_META_FIELDS = {"state"}

# Fields containing complex objects that can't survive JSON round-trip
# (e.g. AdaptiveEngine instances) — stored in a process-local cache
_LIVE_CACHE: dict[str, dict] = {}


def _serialize_session(session: dict) -> str:
    """Serialize a session dict to JSON, stripping non-serializable objects."""
    # Fields with live Python objects — stored in process-local cache, not JSON
    _LIVE_FIELDS = {"adaptive_engine", "followup_gen"}
    safe = {}
    for k, v in session.items():
        if k in _LIVE_FIELDS:
            continue  # handled by live cache
        try:
            json.dumps(v)
            safe[k] = v
        except (TypeError, ValueError):
            logger.debug("Skipping non-serializable field '%s' in session", k)
    return json.dumps(safe)


def _deserialize_session(raw: str, session_id: str) -> dict:
    """Deserialize a session JSON string and merge with live cache."""
    session = json.loads(raw)
    # Merge back any live objects (like AdaptiveEngine)
    if session_id in _LIVE_CACHE:
        session.update(_LIVE_CACHE[session_id])
    return session


class SessionStore:
    """Async SQLite session store with TTL expiry."""

    def __init__(self, db_path: str | Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False

    async def _ensure_db(self):
        """Create the sessions table if it doesn't exist."""
        if self._initialized:
            return
        async with aiosqlite.connect(str(self.db_path)) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_updated
                ON sessions(updated_at)
            """)
            await db.commit()
        self._initialized = True

    async def get(self, session_id: str) -> dict | None:
        """Retrieve a session by ID. Returns None if not found or expired."""
        await self._ensure_db()
        cutoff = time.time() - TTL_SECONDS
        async with aiosqlite.connect(str(self.db_path)) as db:
            cursor = await db.execute(
                "SELECT data FROM sessions WHERE session_id = ? AND updated_at > ?",
                (session_id, cutoff),
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return _deserialize_session(row[0], session_id)

    async def put(self, session_id: str, session: dict):
        """Create or update a session."""
        await self._ensure_db()
        now = time.time()
        data = _serialize_session(session)

        # Cache live objects (non-serializable Python instances)
        live = _LIVE_CACHE.get(session_id, {})
        for field in ("adaptive_engine", "followup_gen"):
            if field in session:
                live[field] = session[field]
        if live:
            _LIVE_CACHE[session_id] = live

        async with aiosqlite.connect(str(self.db_path)) as db:
            await db.execute(
                """INSERT INTO sessions (session_id, data, created_at, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(session_id) DO UPDATE SET data=excluded.data, updated_at=excluded.updated_at""",
                (session_id, data, now, now),
            )
            await db.commit()

    async def delete(self, session_id: str):
        """Remove a session."""
        await self._ensure_db()
        _LIVE_CACHE.pop(session_id, None)
        async with aiosqlite.connect(str(self.db_path)) as db:
            await db.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            await db.commit()

    async def cleanup_expired(self) -> int:
        """Remove all expired sessions. Returns count of deleted sessions."""
        await self._ensure_db()
        cutoff = time.time() - TTL_SECONDS

        # Clean live cache
        expired_ids = []
        async with aiosqlite.connect(str(self.db_path)) as db:
            cursor = await db.execute(
                "SELECT session_id FROM sessions WHERE updated_at <= ?", (cutoff,)
            )
            rows = await cursor.fetchall()
            expired_ids = [r[0] for r in rows]

            if expired_ids:
                await db.execute(
                    "DELETE FROM sessions WHERE updated_at <= ?", (cutoff,)
                )
                await db.commit()

        for sid in expired_ids:
            _LIVE_CACHE.pop(sid, None)

        if expired_ids:
            logger.info("Cleaned up %d expired sessions", len(expired_ids))
        return len(expired_ids)

    async def count(self) -> int:
        """Count active (non-expired) sessions."""
        await self._ensure_db()
        cutoff = time.time() - TTL_SECONDS
        async with aiosqlite.connect(str(self.db_path)) as db:
            cursor = await db.execute(
                "SELECT COUNT(*) FROM sessions WHERE updated_at > ?", (cutoff,)
            )
            row = await cursor.fetchone()
            return row[0] if row else 0
