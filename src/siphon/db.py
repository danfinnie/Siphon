"""SQLite database layer for Siphon – plain sqlite3, no ORM."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

SCHEMA = """\
CREATE TABLE IF NOT EXISTS feeds (
    name            TEXT PRIMARY KEY,
    url             TEXT NOT NULL,
    feed_type       TEXT NOT NULL DEFAULT 'youtube',
    last_checked_at TEXT,
    last_error      TEXT,
    image_url       TEXT,
    channel_id      TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS episodes (
    video_id        TEXT NOT NULL,
    feed_name       TEXT NOT NULL REFERENCES feeds(name) ON DELETE CASCADE,
    title           TEXT NOT NULL,
    description     TEXT,
    thumbnail_url   TEXT,
    channel_name    TEXT,
    duration        INTEGER,
    upload_date     TEXT,
    discovered_at   TEXT NOT NULL DEFAULT (datetime('now')),
    eligible_at     TEXT,
    status          TEXT NOT NULL DEFAULT 'pending',
    file_path       TEXT,
    file_size       INTEGER,
    mime_type       TEXT DEFAULT 'video/mp4',
    error           TEXT,
    retry_count     INTEGER NOT NULL DEFAULT 0,
    llm_trim_status TEXT,
    llm_segments_json TEXT,
    llm_cuts_applied INTEGER,
    llm_retry_count INTEGER NOT NULL DEFAULT 0,
    sb_cuts_applied INTEGER,
    filter_reason   TEXT,
    whisper_device  TEXT,
    whisper_duration_seconds REAL,
    claude_duration_seconds REAL,
    ffmpeg_duration_seconds REAL,
    whisper_word_count INTEGER,
    whisper_segment_count INTEGER,
    transcript_size_bytes INTEGER,
    whisper_model   TEXT,
    updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (video_id, feed_name)
);

CREATE INDEX IF NOT EXISTS idx_episodes_feed_status ON episodes(feed_name, status);
CREATE INDEX IF NOT EXISTS idx_episodes_eligible ON episodes(status, eligible_at);
"""

# Migrations applied after initial schema creation
MIGRATIONS = [
    # Add feed_type column if upgrading from older schema
    "ALTER TABLE feeds ADD COLUMN feed_type TEXT NOT NULL DEFAULT 'youtube'",
    # Add LLM columns if upgrading from older schema
    "ALTER TABLE episodes ADD COLUMN llm_trim_status TEXT",
    "ALTER TABLE episodes ADD COLUMN llm_segments_json TEXT",
    "ALTER TABLE episodes ADD COLUMN llm_cuts_applied INTEGER",
    # Add image_url column for podcast artwork
    "ALTER TABLE feeds ADD COLUMN image_url TEXT",
    # Add sb_cuts_applied column for SponsorBlock segment counts
    "ALTER TABLE episodes ADD COLUMN sb_cuts_applied INTEGER",
    # Add channel_id for YouTube RSS feed lookups
    "ALTER TABLE feeds ADD COLUMN channel_id TEXT",
    # Add llm_retry_count for LLM error retry tracking
    "ALTER TABLE episodes ADD COLUMN llm_retry_count INTEGER NOT NULL DEFAULT 0",
    # Add filter_reason for distinguishing filter types
    "ALTER TABLE episodes ADD COLUMN filter_reason TEXT",
    # Add whisper_device for CPU vs CUDA tracking
    "ALTER TABLE episodes ADD COLUMN whisper_device TEXT",
    # Pipeline split: per-stage timing and metrics
    "ALTER TABLE episodes ADD COLUMN whisper_duration_seconds REAL",
    "ALTER TABLE episodes ADD COLUMN claude_duration_seconds REAL",
    "ALTER TABLE episodes ADD COLUMN ffmpeg_duration_seconds REAL",
    "ALTER TABLE episodes ADD COLUMN whisper_word_count INTEGER",
    "ALTER TABLE episodes ADD COLUMN whisper_segment_count INTEGER",
    "ALTER TABLE episodes ADD COLUMN transcript_size_bytes INTEGER",
    "ALTER TABLE episodes ADD COLUMN whisper_model TEXT",
    # SponsorBlock seconds removed (total duration of all SB segments)
    "ALTER TABLE episodes ADD COLUMN sb_seconds_removed REAL",
    # Pocket Casts podcast UUID for auto-prune
    "ALTER TABLE feeds ADD COLUMN pc_uuid TEXT",
    "ALTER TABLE feeds ADD COLUMN pc_last_checked TEXT",
]


class Database:
    """Thin wrapper around a single sqlite3 connection."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.conn = self._connect()
        self.conn.executescript(SCHEMA)
        self._apply_migrations()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _apply_migrations(self) -> None:
        """Run ALTER TABLE statements, ignoring 'duplicate column' errors."""
        for sql in MIGRATIONS:
            try:
                self.conn.execute(sql)
                self.conn.commit()
            except sqlite3.OperationalError:
                pass  # column already exists

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------
    # Feed CRUD
    # ------------------------------------------------------------------

    def upsert_feed(
        self, name: str, url: str, feed_type: str = "youtube", image_url: str | None = None
    ) -> None:
        self.conn.execute(
            """INSERT INTO feeds (name, url, feed_type, image_url) VALUES (?, ?, ?, ?)
               ON CONFLICT(name) DO UPDATE SET url = excluded.url,
               feed_type = excluded.feed_type,
               image_url = COALESCE(excluded.image_url, feeds.image_url)""",
            (name, url, feed_type, image_url),
        )
        self.conn.commit()

    def update_feed_channel_id(self, name: str, channel_id: str) -> None:
        """Store the YouTube channel ID for RSS lookups."""
        self.conn.execute(
            "UPDATE feeds SET channel_id = ? WHERE name = ?",
            (channel_id, name),
        )
        self.conn.commit()

    def update_feed_image(self, name: str, image_url: str | None) -> None:
        """Update a feed's image_url."""
        self.conn.execute(
            "UPDATE feeds SET image_url = ? WHERE name = ?",
            (image_url, name),
        )
        self.conn.commit()

    def get_feed(self, name: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM feeds WHERE name = ?", (name,)
        ).fetchone()
        return dict(row) if row else None

    def get_all_feeds(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM feeds").fetchall()
        return [dict(r) for r in rows]

    def get_feeds_to_check(
        self, limit: int, feed_type: str | None = None
    ) -> list[dict]:
        if feed_type is not None:
            rows = self.conn.execute(
                "SELECT * FROM feeds WHERE feed_type = ? "
                "ORDER BY last_checked_at ASC NULLS FIRST LIMIT ?",
                (feed_type, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM feeds ORDER BY last_checked_at ASC NULLS FIRST LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def update_feed_checked(self, name: str, error: str | None = None) -> None:
        self.conn.execute(
            "UPDATE feeds SET last_checked_at = datetime('now'), last_error = ? WHERE name = ?",
            (error, name),
        )
        self.conn.commit()

    def delete_feed(self, name: str) -> None:
        """Delete a feed and all its episodes."""
        self.conn.execute("DELETE FROM feeds WHERE name = ?", (name,))
        self.conn.commit()

    # ------------------------------------------------------------------
    # Episode CRUD
    # ------------------------------------------------------------------

    def insert_episode(
        self,
        video_id: str,
        feed_name: str,
        title: str,
        description: str | None = None,
        thumbnail_url: str | None = None,
        channel_name: str | None = None,
        duration: int | None = None,
        upload_date: str | None = None,
        eligible_at: str | None = None,
        status: str = "pending",
        filter_reason: str | None = None,
    ) -> None:
        self.conn.execute(
            """INSERT OR IGNORE INTO episodes
               (video_id, feed_name, title, description, thumbnail_url,
                channel_name, duration, upload_date, eligible_at, status, filter_reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                video_id,
                feed_name,
                title,
                description,
                thumbnail_url,
                channel_name,
                duration,
                upload_date,
                eligible_at,
                status,
                filter_reason,
            ),
        )
        self.conn.commit()

    def get_episode(self, video_id: str, feed_name: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM episodes WHERE video_id = ? AND feed_name = ?",
            (video_id, feed_name),
        ).fetchone()
        return dict(row) if row else None

    def get_episodes_by_feed(
        self, feed_name: str, status: str | None = None
    ) -> list[dict]:
        if status is not None:
            rows = self.conn.execute(
                "SELECT * FROM episodes WHERE feed_name = ? AND status = ?",
                (feed_name, status),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM episodes WHERE feed_name = ?",
                (feed_name,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_processed_episodes(self, limit: int = 200, feed_name: str | None = None) -> list[dict]:
        """Episodes that finished ad detection, newest first (for the Episodes page)."""
        sql = "SELECT * FROM episodes WHERE status = 'done' AND llm_trim_status = 'done'"
        params: list = []
        if feed_name:
            sql += " AND feed_name = ?"
            params.append(feed_name)
        sql += " ORDER BY updated_at DESC, discovered_at DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def get_done_episodes_by_feed(self, feed_name: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM episodes WHERE feed_name = ? AND status = 'done' ORDER BY upload_date DESC",
            (feed_name,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_eligible_episodes(
        self, limit: int, feed_type: str | None = None
    ) -> list[dict]:
        if feed_type is not None:
            rows = self.conn.execute(
                """SELECT e.* FROM episodes e
                   JOIN feeds f ON e.feed_name = f.name
                   WHERE ((e.status = 'pending' AND e.eligible_at <= datetime('now'))
                          OR e.status = 'eligible')
                     AND f.feed_type = ?
                   LIMIT ?""",
                (feed_type, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """SELECT * FROM episodes
                   WHERE (status = 'pending' AND eligible_at <= datetime('now'))
                      OR status = 'eligible'
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def update_episode_status(
        self, video_id: str, feed_name: str, status: str, **kwargs: object
    ) -> None:
        sets = ["status = ?", "updated_at = datetime('now')"]
        params: list[object] = [status]

        for col, val in kwargs.items():
            sets.append(f"{col} = ?")
            params.append(val)

        params.extend([video_id, feed_name])
        sql = f"UPDATE episodes SET {', '.join(sets)} WHERE video_id = ? AND feed_name = ?"
        self.conn.execute(sql, params)
        self.conn.commit()

    def get_pending_whisper(self, limit: int = 1) -> list[dict]:
        """Get episodes with status='pending_whisper', ordered by discovered_at."""
        rows = self.conn.execute(
            """SELECT e.*, f.feed_type FROM episodes e
               JOIN feeds f ON e.feed_name = f.name
               WHERE e.status = 'pending_whisper'
               ORDER BY e.discovered_at ASC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_pending_claude(self, limit: int = 3) -> list[dict]:
        """Get episodes with status='pending_claude' and file_path is set."""
        rows = self.conn.execute(
            """SELECT e.*, f.feed_type FROM episodes e
               JOIN feeds f ON e.feed_name = f.name
               WHERE e.status = 'pending_claude'
                 AND e.file_path IS NOT NULL
               ORDER BY e.discovered_at ASC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_episodes_needing_llm(self, limit: int = 5, max_retries: int = 3) -> list[dict]:
        """Get done episodes that need LLM processing."""
        rows = self.conn.execute(
            """SELECT e.*, f.feed_type FROM episodes e
               JOIN feeds f ON e.feed_name = f.name
               WHERE e.status = 'done'
                 AND (e.llm_trim_status IS NULL
                      OR e.llm_trim_status = 'pending'
                      OR (e.llm_trim_status = 'error' AND e.llm_retry_count < ?))
                 AND e.file_path IS NOT NULL
               LIMIT ?""",
            (max_retries, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_disk_usage(self) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(file_size), 0) AS total FROM episodes WHERE status = 'done'"
        ).fetchone()
        return int(row["total"])

    def get_oldest_done_episodes(self, limit: int, feed_name: str | None = None) -> list[dict]:
        if feed_name:
            rows = self.conn.execute(
                "SELECT * FROM episodes WHERE status = 'done' AND feed_name = ? ORDER BY upload_date ASC LIMIT ?",
                (feed_name, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM episodes WHERE status = 'done' ORDER BY upload_date ASC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_feed_episode_count(self, feed_name: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS cnt FROM episodes WHERE status = 'done' AND feed_name = ?",
            (feed_name,),
        ).fetchone()
        return int(row["cnt"])

    def get_recent_download_count(
        self, hours: int = 1, feed_type: str | None = None
    ) -> int:
        """Count episodes that completed downloading within the last *hours*.

        Includes done, pending_whisper, and pending_claude statuses since
        episodes move through these pipeline stages after downloading.
        """
        statuses = "('done', 'pending_whisper', 'pending_claude')"
        if feed_type is not None:
            row = self.conn.execute(
                "SELECT COUNT(*) AS cnt FROM episodes e "
                "JOIN feeds f ON e.feed_name = f.name "
                f"WHERE e.status IN {statuses} AND e.updated_at >= datetime('now', ?) "
                "AND f.feed_type = ?",
                (f"-{hours} hours", feed_type),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*) AS cnt FROM episodes "
                f"WHERE status IN {statuses} AND updated_at >= datetime('now', ?)",
                (f"-{hours} hours",),
            ).fetchone()
        return int(row["cnt"])

    def promote_eligible_episodes(self) -> int:
        cursor = self.conn.execute(
            "UPDATE episodes SET status = 'eligible', updated_at = datetime('now') "
            "WHERE status = 'pending' AND eligible_at <= datetime('now')"
        )
        self.conn.commit()
        return cursor.rowcount

    def reset_stale_downloads(self, hours: int = 6) -> None:
        self.conn.execute(
            "UPDATE episodes SET status = 'eligible', updated_at = datetime('now') "
            f"WHERE status = 'downloading' AND updated_at < datetime('now', '-{hours} hours')"
        )
        self.conn.commit()

    def retry_failed_episodes(self, max_retries: int = 3) -> None:
        self.conn.execute(
            "UPDATE episodes SET status = 'eligible', updated_at = datetime('now') "
            "WHERE status = 'failed' AND retry_count < ?",
            (max_retries,),
        )
        self.conn.commit()

    def delete_episodes_by_feed(self, feed_name: str) -> int:
        """Delete all episodes for a feed. Returns count deleted."""
        cursor = self.conn.execute(
            "DELETE FROM episodes WHERE feed_name = ?", (feed_name,)
        )
        self.conn.commit()
        return cursor.rowcount

    def update_feed_date_cutoff_in_episodes(
        self, feed_name: str, date_cutoff: str
    ) -> None:
        """Mark episodes older than date_cutoff as 'pruned' for a feed."""
        self.conn.execute(
            "UPDATE episodes SET status = 'pruned', updated_at = datetime('now') "
            "WHERE feed_name = ? AND status = 'done' AND upload_date < ?",
            (feed_name, date_cutoff),
        )
        self.conn.commit()
