from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Iterable
from zoneinfo import ZoneInfo


LOCAL_TZ = ZoneInfo("Asia/Shanghai")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ENTRY_TYPES = {"diary", "darkroom"}
VISIBILITIES = {"active", "archived", "deleted"}


class DiaryStoreError(RuntimeError):
    pass


class DiaryNotFoundError(DiaryStoreError):
    pass


class DiaryLockedError(DiaryStoreError):
    def __init__(self, diary_id: int, unlock_at: str):
        self.diary_id = int(diary_id)
        self.unlock_at = str(unlock_at or "")
        super().__init__(f"diary {self.diary_id} is locked until {self.unlock_at}")


def _now() -> datetime:
    return datetime.now(LOCAL_TZ)


def _now_iso() -> str:
    return _now().isoformat(timespec="seconds")


def _normalize_date(value: Any, *, default_today: bool = False) -> str:
    text = str(value or "").strip()
    if not text and default_today:
        return _now().date().isoformat()
    if not DATE_RE.fullmatch(text):
        raise ValueError("date must use YYYY-MM-DD")
    try:
        datetime.strptime(text, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("date must use YYYY-MM-DD") from exc
    return text


def _normalize_tags(value: Any) -> list[str]:
    if value is None:
        return []
    raw = value if isinstance(value, (list, tuple, set)) else [value]
    result: list[str] = []
    seen: set[str] = set()
    for item in raw:
        tag = str(item or "").strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        result.append(tag[:80])
    return result[:24]


def _normalize_unlock_at(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("unlock_at must be an ISO-8601 datetime") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=LOCAL_TZ)
    return parsed.astimezone(LOCAL_TZ).isoformat(timespec="seconds")


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


class DiaryStore:
    """One authored-document backend for ordinary Diary and timed Darkroom entries."""

    def __init__(self, config: dict[str, Any] | None = None, *, db_path: str | Path | None = None):
        config = config or {}
        diary_config = config.get("diary") if isinstance(config.get("diary"), dict) else {}
        state_dir = str(
            config.get("state_dir")
            or os.path.join(
                os.path.dirname(os.path.abspath(str(config.get("buckets_dir") or "buckets"))),
                "state",
            )
        )
        configured_path = str(diary_config.get("db_path") or "").strip()
        self.db_path = Path(db_path or configured_path or (Path(state_dir) / "diary.db"))
        self._lock = RLock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 15000")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    @contextmanager
    def _connection(self):
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_database(self) -> None:
        with self._lock, self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS diaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    title TEXT DEFAULT NULL,
                    content TEXT NOT NULL,
                    author TEXT NOT NULL DEFAULT 'ai',
                    emotion_tags TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._ensure_columns(
                conn,
                "diaries",
                {
                    "title": "TEXT DEFAULT NULL",
                    "author": "TEXT NOT NULL DEFAULT 'ai'",
                    "entry_type": "TEXT NOT NULL DEFAULT 'diary'",
                    "visibility": "TEXT NOT NULL DEFAULT 'active'",
                    "unlock_at": "TEXT NOT NULL DEFAULT ''",
                    "revision": "INTEGER NOT NULL DEFAULT 1",
                    "source_id": "TEXT NOT NULL DEFAULT ''",
                    "metadata": "TEXT NOT NULL DEFAULT '{}'",
                    "deleted_at": "TEXT NOT NULL DEFAULT ''",
                },
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS comments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    diary_id INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    author TEXT NOT NULL DEFAULT 'user',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (diary_id) REFERENCES diaries (id) ON DELETE CASCADE
                )
                """
            )
            self._ensure_columns(
                conn,
                "comments",
                {"author": "TEXT NOT NULL DEFAULT 'user'"},
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS diary_revisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    diary_id INTEGER NOT NULL,
                    revision INTEGER NOT NULL,
                    date TEXT NOT NULL,
                    title TEXT,
                    content TEXT NOT NULL,
                    author TEXT NOT NULL,
                    emotion_tags TEXT,
                    entry_type TEXT NOT NULL,
                    visibility TEXT NOT NULL,
                    unlock_at TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS darkroom_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    locked_at TEXT NOT NULL,
                    unlock_at TEXT NOT NULL,
                    diary_id INTEGER,
                    FOREIGN KEY (diary_id) REFERENCES diaries (id) ON DELETE SET NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON diaries(date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_diary_id ON comments(diary_id)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_diaries_visible_date "
                "ON diaries(visibility, date DESC, id DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_diary_revisions_diary "
                "ON diary_revisions(diary_id, revision)"
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_diaries_source_id "
                "ON diaries(source_id) WHERE source_id != ''"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_darkroom_sessions_unlock "
                "ON darkroom_sessions(unlock_at DESC, id DESC)"
            )

    @staticmethod
    def _ensure_columns(
        conn: sqlite3.Connection,
        table: str,
        columns: dict[str, str],
    ) -> None:
        existing = {
            str(row["name"])
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        for name, definition in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")

    @staticmethod
    def _locked_until(row: dict[str, Any] | sqlite3.Row) -> str:
        raw = str(row["unlock_at"] if "unlock_at" in row.keys() else "").strip()
        if not raw:
            return ""
        try:
            unlock_at = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return ""
        if unlock_at.tzinfo is None:
            unlock_at = unlock_at.replace(tzinfo=LOCAL_TZ)
        return raw if _now() < unlock_at.astimezone(LOCAL_TZ) else ""

    @staticmethod
    def _row_dict(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        raw_tags = item.get("emotion_tags")
        if raw_tags:
            try:
                parsed_tags = json.loads(str(raw_tags))
            except (TypeError, ValueError):
                parsed_tags = []
        else:
            parsed_tags = []
        item["emotion_tags"] = parsed_tags if isinstance(parsed_tags, list) else []
        item["metadata"] = _json_object(item.get("metadata"))
        item["entry_type"] = str(item.get("entry_type") or "diary")
        item["visibility"] = str(item.get("visibility") or "active")
        item["unlock_at"] = str(item.get("unlock_at") or "")
        item["revision"] = int(item.get("revision") or 1)
        return item

    def _comments(self, conn: sqlite3.Connection, diary_id: int) -> list[dict[str, Any]]:
        rows = conn.execute(
            "SELECT * FROM comments WHERE diary_id = ? ORDER BY created_at ASC, id ASC",
            (int(diary_id),),
        ).fetchall()
        return [dict(row) for row in rows]

    def _public_entry(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        include_comments: bool = True,
    ) -> dict[str, Any]:
        item = self._row_dict(row)
        locked_until = self._locked_until(row)
        item["locked"] = bool(locked_until)
        if locked_until:
            item["unlock_at"] = locked_until
            item["content"] = ""
            item["comments"] = []
            item["body_available"] = False
        else:
            item["body_available"] = True
            item["comments"] = self._comments(conn, int(item["id"])) if include_comments else []
        item.pop("deleted_at", None)
        return item

    def _create_with_connection(
        self,
        conn: sqlite3.Connection,
        *,
        content: str,
        date: str = "",
        title: str | None = None,
        emotion_tags: Iterable[str] | None = None,
        author: str = "ai",
        unlock_at: str = "",
        entry_type: str = "",
        visibility: str = "active",
        source_id: str = "",
        metadata: dict[str, Any] | None = None,
        created_at: str = "",
        allow_past_unlock_at: bool = False,
        preserve_content: bool = False,
    ) -> dict[str, Any]:
        raw_text = str(content or "")
        text = raw_text if preserve_content else raw_text.strip()
        if not text.strip():
            raise ValueError("content is required")
        safe_date = _normalize_date(date, default_today=True)
        safe_unlock_at = _normalize_unlock_at(unlock_at)
        if safe_unlock_at and not allow_past_unlock_at:
            parsed_unlock_at = datetime.fromisoformat(safe_unlock_at)
            if parsed_unlock_at <= _now():
                raise ValueError("unlock_at must be in the future")
        safe_type = str(entry_type or ("darkroom" if safe_unlock_at else "diary")).strip().lower()
        if safe_type not in ENTRY_TYPES:
            raise ValueError("entry_type must be diary or darkroom")
        if safe_type == "diary" and safe_unlock_at:
            safe_type = "darkroom"
        safe_visibility = str(visibility or "active").strip().lower()
        if safe_visibility not in VISIBILITIES:
            raise ValueError("visibility must be active, archived or deleted")
        safe_author = str(author or "ai").strip().lower()
        if safe_author not in {"ai", "user"}:
            raise ValueError("author must be ai or user")
        safe_source_id = str(source_id or "").strip()[:240]
        safe_created_at = str(created_at or "").strip() or _now_iso()
        tags_json = json.dumps(_normalize_tags(emotion_tags), ensure_ascii=False)
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True)

        if safe_source_id:
            existing = conn.execute(
                "SELECT * FROM diaries WHERE source_id = ?",
                (safe_source_id,),
            ).fetchone()
            if existing:
                result = self._public_entry(conn, existing)
                result["status"] = "exists"
                return result
        cursor = conn.execute(
            """
            INSERT INTO diaries (
                date, title, content, author, emotion_tags,
                created_at, updated_at, entry_type, visibility,
                unlock_at, revision, source_id, metadata, deleted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, '')
            """,
            (
                safe_date,
                str(title).strip() if title is not None else None,
                text,
                safe_author,
                tags_json,
                safe_created_at,
                safe_created_at,
                safe_type,
                safe_visibility,
                safe_unlock_at,
                safe_source_id,
                metadata_json,
            ),
        )
        diary_id = int(cursor.lastrowid)
        row = conn.execute("SELECT * FROM diaries WHERE id = ?", (diary_id,)).fetchone()
        result = self._public_entry(conn, row)
        result["status"] = "created"
        return result

    def create(
        self,
        *,
        content: str,
        date: str = "",
        title: str | None = None,
        emotion_tags: Iterable[str] | None = None,
        author: str = "ai",
        unlock_at: str = "",
        entry_type: str = "",
        visibility: str = "active",
        source_id: str = "",
        metadata: dict[str, Any] | None = None,
        created_at: str = "",
        allow_past_unlock_at: bool = False,
        preserve_content: bool = False,
    ) -> dict[str, Any]:
        with self._lock, self._connection() as conn:
            return self._create_with_connection(
                conn,
                content=content,
                date=date,
                title=title,
                emotion_tags=emotion_tags,
                author=author,
                unlock_at=unlock_at,
                entry_type=entry_type,
                visibility=visibility,
                source_id=source_id,
                metadata=metadata,
                created_at=created_at,
                allow_past_unlock_at=allow_past_unlock_at,
                preserve_content=preserve_content,
            )

    @staticmethod
    def _close_darkroom_with_connection(
        conn: sqlite3.Connection,
        *,
        unlock_at: str,
        diary_id: int | None = None,
    ) -> dict[str, Any]:
        safe_unlock_at = _normalize_unlock_at(unlock_at)
        if not safe_unlock_at:
            raise ValueError("unlock_at is required to close the darkroom")
        if datetime.fromisoformat(safe_unlock_at) <= _now():
            raise ValueError("unlock_at must be in the future")
        locked_at = _now_iso()
        cursor = conn.execute(
            """
            INSERT INTO darkroom_sessions (locked_at, unlock_at, diary_id)
            VALUES (?, ?, ?)
            """,
            (locked_at, safe_unlock_at, int(diary_id) if diary_id is not None else None),
        )
        return {
            "session_id": int(cursor.lastrowid),
            "locked": True,
            "locked_at": locked_at,
            "unlock_at": safe_unlock_at,
            "diary_id": int(diary_id) if diary_id is not None else None,
        }

    def write(
        self,
        *,
        content: str = "",
        date: str = "",
        title: str | None = None,
        emotion_tags: Iterable[str] | None = None,
        author: str = "ai",
        unlock_at: str = "",
    ) -> dict[str, Any]:
        has_content = bool(str(content or "").strip())
        has_lock = bool(str(unlock_at or "").strip())
        if not has_content and not has_lock:
            raise ValueError("content or unlock_at is required")

        with self._lock, self._connection() as conn:
            diary: dict[str, Any] | None = None
            if has_content:
                diary = self._create_with_connection(
                    conn,
                    content=content,
                    date=date,
                    title=title,
                    emotion_tags=emotion_tags,
                    author=author,
                    unlock_at=unlock_at,
                )
            if not has_lock:
                return diary or {}

            door = self._close_darkroom_with_connection(
                conn,
                unlock_at=unlock_at,
                diary_id=int(diary["id"]) if diary and diary.get("id") is not None else None,
            )
            if diary is None:
                return {
                    "status": "door_locked",
                    "entry_type": "darkroom",
                    "diary_created": False,
                    **door,
                }
            diary["darkroom_session"] = door
            return diary

    def darkroom_status(self) -> dict[str, Any]:
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT id, locked_at, unlock_at, diary_id
                FROM darkroom_sessions
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            return {"status": "ok", "locked": False}
        item = dict(row)
        unlock_at = datetime.fromisoformat(str(item["unlock_at"]).replace("Z", "+00:00"))
        if unlock_at.tzinfo is None:
            unlock_at = unlock_at.replace(tzinfo=LOCAL_TZ)
        return {
            "status": "ok",
            "session_id": int(item["id"]),
            "locked": _now() < unlock_at.astimezone(LOCAL_TZ),
            "locked_at": str(item["locked_at"]),
            "unlock_at": str(item["unlock_at"]),
            "diary_id": int(item["diary_id"]) if item["diary_id"] is not None else None,
        }

    def read(
        self,
        *,
        diary_id: int | None = None,
        date: str = "",
        title: str = "",
        limit: int = 20,
        include_archived: bool = True,
    ) -> dict[str, Any]:
        safe_limit = max(1, min(int(limit or 20), 100))
        clauses = ["visibility != 'deleted'"]
        params: list[Any] = []
        if not include_archived:
            clauses.append("visibility = 'active'")
        if diary_id is not None and int(diary_id) > 0:
            clauses.append("id = ?")
            params.append(int(diary_id))
        if str(date or "").strip():
            clauses.append("date = ?")
            params.append(_normalize_date(date))
        if str(title or "").strip():
            clauses.append("COALESCE(title, '') LIKE ?")
            params.append(f"%{str(title).strip()}%")
        query = (
            "SELECT * FROM diaries WHERE "
            + " AND ".join(clauses)
            + " ORDER BY date DESC, id DESC LIMIT ?"
        )
        params.append(safe_limit)
        with self._lock, self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
            diaries = [self._public_entry(conn, row) for row in rows]
        result: dict[str, Any] = {
            "status": "ok",
            "count": len(diaries),
            "date": str(date or "").strip(),
            "query_title": str(title or "").strip(),
            "diaries": diaries,
        }
        if diaries:
            result.update(diaries[0])
            result["status"] = "ok"
            result["count"] = len(diaries)
            result["diaries"] = diaries
            result["query_title"] = str(title or "").strip()
        return result

    def search(
        self,
        *,
        keyword: str = "",
        date: str = "",
        title: str = "",
        start_date: str = "",
        end_date: str = "",
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Compatibility search over date and title only; body and tags are never query axes."""
        safe_limit = max(1, min(int(limit or 20), 100))
        safe_offset = max(0, int(offset or 0))
        clauses = ["visibility != 'deleted'"]
        params: list[Any] = []
        query_date = str(date or "").strip()
        title_terms: list[str] = []
        if str(keyword or "").strip():
            for token in str(keyword).split():
                if DATE_RE.fullmatch(token) and not query_date:
                    query_date = token
                else:
                    title_terms.append(token)
        if str(title or "").strip():
            title_terms.extend(str(title).split())
        if query_date:
            clauses.append("date = ?")
            params.append(_normalize_date(query_date))
        if str(start_date or "").strip():
            clauses.append("date >= ?")
            params.append(_normalize_date(start_date))
        if str(end_date or "").strip():
            clauses.append("date <= ?")
            params.append(_normalize_date(end_date))
        for term in title_terms:
            clauses.append("COALESCE(title, '') LIKE ?")
            params.append(f"%{term}%")
        sql = (
            "SELECT * FROM diaries WHERE "
            + " AND ".join(clauses)
            + " ORDER BY date DESC, id DESC LIMIT ? OFFSET ?"
        )
        params.extend([safe_limit, safe_offset])
        with self._lock, self._connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            diaries = [self._public_entry(conn, row) for row in rows]
        return {
            "status": "ok",
            "count": len(diaries),
            "diaries": diaries,
            "search_axes": ["date", "title"],
        }

    def _require_mutable(
        self,
        conn: sqlite3.Connection,
        diary_id: int,
    ) -> sqlite3.Row:
        row = conn.execute("SELECT * FROM diaries WHERE id = ?", (int(diary_id),)).fetchone()
        if row is None or str(row["visibility"] or "active") == "deleted":
            raise DiaryNotFoundError(f"diary {diary_id} was not found")
        locked_until = self._locked_until(row)
        if locked_until:
            raise DiaryLockedError(int(diary_id), locked_until)
        return row

    @staticmethod
    def _snapshot(
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        reason: str,
    ) -> None:
        conn.execute(
            """
            INSERT INTO diary_revisions (
                diary_id, revision, date, title, content, author,
                emotion_tags, entry_type, visibility, unlock_at,
                metadata, reason, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(row["id"]),
                int(row["revision"] or 1),
                str(row["date"] or ""),
                row["title"],
                str(row["content"] or ""),
                str(row["author"] or "ai"),
                row["emotion_tags"],
                str(row["entry_type"] or "diary"),
                str(row["visibility"] or "active"),
                str(row["unlock_at"] or ""),
                str(row["metadata"] or "{}"),
                str(reason or "revision"),
                _now_iso(),
            ),
        )

    def revise(
        self,
        diary_id: int,
        *,
        content: str | None = None,
        date: str | None = None,
        title: str | None = None,
        emotion_tags: Iterable[str] | None = None,
        unlock_at: str | None = None,
    ) -> dict[str, Any]:
        updates: list[str] = []
        params: list[Any] = []
        if content is not None:
            text = str(content).strip()
            if not text:
                raise ValueError("content cannot be empty")
            updates.append("content = ?")
            params.append(text)
        if date is not None:
            updates.append("date = ?")
            params.append(_normalize_date(date))
        if title is not None:
            updates.append("title = ?")
            params.append(str(title).strip() or None)
        if emotion_tags is not None:
            updates.append("emotion_tags = ?")
            params.append(json.dumps(_normalize_tags(emotion_tags), ensure_ascii=False))
        if unlock_at is not None:
            updates.append("unlock_at = ?")
            params.append(_normalize_unlock_at(unlock_at))
        if not updates:
            raise ValueError("at least one field is required")

        with self._lock, self._connection() as conn:
            row = self._require_mutable(conn, int(diary_id))
            self._snapshot(conn, row, reason="revise")
            updates.extend(["revision = revision + 1", "updated_at = ?"])
            params.append(_now_iso())
            params.append(int(diary_id))
            conn.execute(
                f"UPDATE diaries SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            revised = conn.execute(
                "SELECT * FROM diaries WHERE id = ?",
                (int(diary_id),),
            ).fetchone()
            result = self._public_entry(conn, revised)
            result["status"] = "revised"
            return result

    def delete(self, diary_id: int) -> dict[str, Any]:
        with self._lock, self._connection() as conn:
            row = self._require_mutable(conn, int(diary_id))
            self._snapshot(conn, row, reason="delete")
            deleted_at = _now_iso()
            conn.execute(
                """
                UPDATE diaries
                SET visibility = 'deleted', deleted_at = ?,
                    revision = revision + 1, updated_at = ?
                WHERE id = ?
                """,
                (deleted_at, deleted_at, int(diary_id)),
            )
            return {
                "status": "deleted",
                "diary_id": int(diary_id),
                "deleted_at": deleted_at,
                "recoverable": True,
            }

    def comment(
        self,
        diary_id: int,
        *,
        content: str,
        author: str = "ai",
    ) -> dict[str, Any]:
        text = str(content or "").strip()
        if not text:
            raise ValueError("comment content is required")
        safe_author = str(author or "ai").strip().lower()
        if safe_author not in {"ai", "user"}:
            raise ValueError("author must be ai or user")
        with self._lock, self._connection() as conn:
            self._require_mutable(conn, int(diary_id))
            created_at = _now_iso()
            cursor = conn.execute(
                """
                INSERT INTO comments (diary_id, content, author, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (int(diary_id), text, safe_author, created_at),
            )
            return {
                "status": "commented",
                "id": int(cursor.lastrowid),
                "diary_id": int(diary_id),
                "author": safe_author,
                "created_at": created_at,
            }

    def comments(self, diary_id: int) -> dict[str, Any]:
        with self._lock, self._connection() as conn:
            self._require_mutable(conn, int(diary_id))
            comments = self._comments(conn, int(diary_id))
            return {"status": "ok", "count": len(comments), "comments": comments}

    def delete_comment(self, diary_id: int, comment_id: int) -> dict[str, Any]:
        with self._lock, self._connection() as conn:
            self._require_mutable(conn, int(diary_id))
            cursor = conn.execute(
                "DELETE FROM comments WHERE id = ? AND diary_id = ?",
                (int(comment_id), int(diary_id)),
            )
            if cursor.rowcount <= 0:
                raise DiaryNotFoundError(f"comment {comment_id} was not found")
            return {
                "status": "deleted",
                "diary_id": int(diary_id),
                "comment_id": int(comment_id),
            }

    def revision_count(self, diary_id: int) -> int:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS count FROM diary_revisions WHERE diary_id = ?",
                (int(diary_id),),
            ).fetchone()
            return int(row["count"] or 0)

    def stats(self) -> dict[str, Any]:
        with self._connection() as conn:
            counts = {
                str(row["entry_type"] or "diary"): int(row["count"] or 0)
                for row in conn.execute(
                    """
                    SELECT entry_type, COUNT(*) AS count
                    FROM diaries
                    WHERE visibility != 'deleted'
                    GROUP BY entry_type
                    """
                ).fetchall()
            }
            comments = int(
                conn.execute("SELECT COUNT(*) AS count FROM comments").fetchone()["count"]
                or 0
            )
        return {
            "status": "ok",
            "db_path": str(self.db_path),
            "diaries": counts.get("diary", 0),
            "darkroom": counts.get("darkroom", 0),
            "comments": comments,
            "total": sum(counts.values()),
        }

    def import_legacy_darkroom(self, entries: Iterable[dict[str, Any]]) -> dict[str, Any]:
        created_ids: list[int] = []
        existing_ids: list[int] = []
        invalid = 0
        for raw in entries:
            if not isinstance(raw, dict):
                invalid += 1
                continue
            content = str(raw.get("note") or "")
            entry_id = str(raw.get("id") or "").strip()
            created_at = str(raw.get("created_at") or "").strip()
            if not content or not entry_id or not created_at:
                invalid += 1
                continue
            try:
                parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=LOCAL_TZ)
                local_created = parsed.astimezone(LOCAL_TZ)
            except ValueError:
                invalid += 1
                continue
            visibility = str(raw.get("visibility") or "active").strip().lower()
            if visibility not in {"active", "archived"}:
                visibility = "archived"
            metadata = {
                "legacy_kind": "ombre_darkroom",
                "legacy_entry_id": entry_id,
                "legacy_room_id": str(raw.get("room_id") or entry_id),
                "legacy_revision": int(raw.get("revision") or 1),
                "legacy_previous_entry_id": str(raw.get("previous_entry_id") or ""),
                "legacy_mode": str(raw.get("mode") or ""),
                "mood": str(raw.get("mood") or ""),
            }
            result = self.create(
                content=content,
                date=local_created.date().isoformat(),
                title=f"暗房 · {local_created.date().isoformat()}",
                emotion_tags=_normalize_tags(raw.get("tags")),
                author="ai",
                unlock_at=str(raw.get("locked_until") or ""),
                entry_type="darkroom",
                visibility=visibility,
                source_id=f"legacy_darkroom:{entry_id}",
                metadata=metadata,
                created_at=local_created.isoformat(timespec="seconds"),
                allow_past_unlock_at=True,
                preserve_content=True,
            )
            if result.get("status") == "created":
                created_ids.append(int(result["id"]))
            else:
                existing_ids.append(int(result["id"]))
        return {
            "status": "ok",
            "created": len(created_ids),
            "existing": len(existing_ids),
            "invalid": invalid,
            "created_ids": created_ids,
            "existing_ids": existing_ids,
        }
