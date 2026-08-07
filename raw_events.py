from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any


logger = logging.getLogger("ombre_brain.raw_events")

ALLOWED_RAW_ROLES = {"user", "assistant"}
RAW_EVENT_DEFAULT_SOURCE = "raw"

INJECTION_SECTION_RE = re.compile(
    r"(?im)^\s*(?:"
    r"Core Memory|Recalled Memory|Recent Context|Just Now Chat Context|"
    r"Related Memory|Dream Context|Additional private memory detail|"
    r"Long-term State Summary"
    r")\s*:?\s*$"
)
CLIENT_ATTACHMENT_RE = re.compile(r"<attachment\b[^>]*>[\s\S]*?</attachment>", re.IGNORECASE)
SELF_CLOSING_ATTACHMENT_RE = re.compile(r"<attachment\b[^>]*/>", re.IGNORECASE)
WORKSPACE_ATTACHMENT_RE = re.compile(
    r"<workspace_attachment>[\s\S]*?</workspace_attachment>",
    re.IGNORECASE,
)
CLIENT_CONTEXT_BLOCK_TITLES = {
    "当前时间",
    "当前电量",
    "当前天气",
    "当前位置",
    "当前屏幕应用",
    "应用使用时长",
    "最近通知",
    "相关记忆",
    "屏幕文本",
}


def strip_raw_client_context(text: str) -> str:
    cleaned = WORKSPACE_ATTACHMENT_RE.sub("", str(text or ""))
    cleaned = CLIENT_ATTACHMENT_RE.sub("", cleaned)
    cleaned = SELF_CLOSING_ATTACHMENT_RE.sub("", cleaned)
    cleaned = _strip_client_context_blocks(cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _strip_client_context_blocks(text: str) -> str:
    kept: list[str] = []
    skipping = False
    for line in str(text or "").splitlines():
        stripped = line.strip()
        title = ""
        if stripped.startswith("【") and "】" in stripped:
            title = stripped[1 : stripped.index("】")].strip()
        if title:
            skipping = title in CLIENT_CONTEXT_BLOCK_TITLES
            if skipping:
                continue
        if not skipping:
            kept.append(line)
    return "\n".join(kept)


def raw_event_text_looks_injected(text: str, raw: dict[str, Any] | None = None) -> bool:
    raw = raw or {}
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    flags = {
        str(raw.get("kind") or "").lower(),
        str(raw.get("source_type") or "").lower(),
        str(metadata.get("kind") or "").lower(),
        str(metadata.get("source_type") or "").lower(),
    }
    if flags & {"injection", "memory_injection", "tool", "tool_result", "system", "developer"}:
        return True
    stripped = str(text or "").strip()
    if stripped.startswith("Live private context for the current turn"):
        return True
    if INJECTION_SECTION_RE.search(stripped):
        return True
    return "[bucket_id:" in stripped and any(
        marker in stripped
        for marker in (
            "Recalled Memory",
            "Related Memory",
            "Recent Context",
            "Core Memory",
        )
    )


class RawEventStore:
    """Append-only-ish raw dialogue archive with optional FTS search."""

    def __init__(self, config: dict):
        config = config or {}
        raw_cfg = config.get("raw_events", {}) if isinstance(config.get("raw_events", {}), dict) else {}
        state_dir = config.get("state_dir") or os.path.join(
            os.path.dirname(os.path.abspath(config.get("buckets_dir", "buckets"))),
            "state",
        )
        self.db_path = str(raw_cfg.get("db_path") or os.path.join(state_dir, "raw_events.sqlite"))
        self.max_ingest_batch = max(1, min(5000, int(raw_cfg.get("max_ingest_batch", 1000))))
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.fts_enabled = False
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                source_event_id TEXT NOT NULL DEFAULT '',
                event_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                ingested_at TEXT NOT NULL,
                conversation_id TEXT NOT NULL DEFAULT '',
                session_id TEXT NOT NULL DEFAULT '',
                client TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                UNIQUE(source, event_hash)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_events_created ON raw_events(created_at DESC, id DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_events_source ON raw_events(source, created_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_events_role ON raw_events(role, created_at DESC)")
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_events_source_event_id
            ON raw_events(source, source_event_id)
            WHERE source_event_id != ''
            """
        )
        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS raw_events_fts
                USING fts5(text, source, conversation_id, session_id, content='raw_events', content_rowid='id')
                """
            )
            self.fts_enabled = True
        except sqlite3.OperationalError as exc:
            self.fts_enabled = False
            logger.warning("raw_events FTS5 disabled: %s", exc)
        conn.commit()
        conn.close()

    def ingest(self, events: list[dict[str, Any]], *, source: str = "") -> dict[str, Any]:
        safe_source = self._clean_source(source)
        now = self._now_iso()
        items = []
        inserted = 0
        duplicate = 0
        rejected = 0
        for raw in list(events or [])[: self.max_ingest_batch]:
            normalized, reason = self._normalize_event(raw, default_source=safe_source, ingested_at=now)
            if reason:
                rejected += 1
                items.append(
                    {
                        "status": "rejected",
                        "reason": reason,
                        "source_event_id": str((raw or {}).get("source_event_id") or (raw or {}).get("id") or ""),
                    }
                )
                continue
            status, row_id = self._insert_event(normalized)
            if status == "inserted":
                inserted += 1
            else:
                duplicate += 1
            items.append(
                {
                    "status": status,
                    "id": row_id,
                    "source": normalized["source"],
                    "source_event_id": normalized["source_event_id"],
                    "role": normalized["role"],
                }
            )
        return {
            "ok": True,
            "inserted": inserted,
            "duplicate": duplicate,
            "rejected": rejected,
            "items": items,
        }

    def search(
        self,
        query: str = "",
        *,
        limit: int = 10,
        source: str = "",
        role: str = "",
        conversation_id: str = "",
        session_id: str = "",
        since: str = "",
        until: str = "",
    ) -> dict[str, Any]:
        safe_limit = max(1, min(100, int(limit or 10)))
        cleaned_query = str(query or "").strip()
        filters, params = self._search_filters(
            source=source,
            role=role,
            conversation_id=conversation_id,
            session_id=session_id,
            since=since,
            until=until,
        )
        rows = self._search_fts(cleaned_query, filters, params, safe_limit) if cleaned_query else []
        if len(rows) < safe_limit:
            rows = self._merge_rows(
                rows,
                self._search_like(cleaned_query, filters, params, safe_limit) if cleaned_query else self._search_recent(filters, params, safe_limit),
                safe_limit,
            )
        return {
            "ok": True,
            "query": cleaned_query,
            "count": len(rows),
            "items": [self._row_to_event(row) for row in rows],
        }

    def list_events_between(
        self,
        *,
        start_at: datetime,
        end_at: datetime,
        limit: int = 40,
        source: str = "",
        conversation_id: str = "",
        session_id: str = "",
    ) -> list[dict[str, Any]]:
        try:
            raw_limit = int(limit)
        except (TypeError, ValueError):
            raw_limit = 40
        safe_limit = max(0, min(10000, raw_limit))
        filters, params = self._search_filters(
            source=source,
            conversation_id=conversation_id,
            session_id=session_id,
        )
        compare_tz = start_at.tzinfo or end_at.tzinfo

        def parse_local(value: Any) -> datetime | None:
            try:
                parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
            except ValueError:
                return None
            if compare_tz is None:
                return parsed.replace(tzinfo=None) if parsed.tzinfo is not None else parsed
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=compare_tz)
            return parsed.astimezone(compare_tz)

        start = start_at
        end = end_at
        if compare_tz is not None:
            if start.tzinfo is None:
                start = start.replace(tzinfo=compare_tz)
            else:
                start = start.astimezone(compare_tz)
            if end.tzinfo is None:
                end = end.replace(tzinfo=compare_tz)
            else:
                end = end.astimezone(compare_tz)
        elif start.tzinfo is not None:
            start = start.replace(tzinfo=None)
        elif end.tzinfo is not None:
            end = end.replace(tzinfo=None)

        # Keep the exact timezone-aware comparison below, but narrow the SQL
        # query first so older ranges are not lost behind a recent-row LIMIT.
        # One day of padding on each side covers ISO timestamps stored with a
        # different UTC offset while still using idx_raw_events_created.
        coarse_start = (start - timedelta(days=1)).date().isoformat()
        coarse_end = (end + timedelta(days=1)).date().isoformat()
        conn = self._connect()
        rows = conn.execute(
            f"""
            SELECT e.*
            FROM raw_events e
            WHERE e.created_at >= ? AND e.created_at < ? {filters}
            ORDER BY e.created_at DESC, e.id DESC
            """,
            [coarse_start, coarse_end, *params],
        ).fetchall()
        conn.close()

        selected: list[dict[str, Any]] = []
        for row in rows:
            created = parse_local(row["created_at"])
            if created is None or not (start <= created < end):
                continue
            selected.append(self._row_to_event(row))
            if safe_limit > 0 and len(selected) >= safe_limit:
                break
        return selected

    def _insert_event(self, event: dict[str, Any]) -> tuple[str, int | None]:
        conn = self._connect()
        try:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO raw_events
                (source, source_event_id, event_hash, role, text, created_at, ingested_at,
                 conversation_id, session_id, client, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event["source"],
                    event["source_event_id"],
                    event["event_hash"],
                    event["role"],
                    event["text"],
                    event["created_at"],
                    event["ingested_at"],
                    event["conversation_id"],
                    event["session_id"],
                    event["client"],
                    event["metadata_json"],
                ),
            )
            if cursor.rowcount:
                row_id = int(cursor.lastrowid or 0)
                if self.fts_enabled:
                    try:
                        conn.execute(
                            """
                            INSERT INTO raw_events_fts(rowid, text, source, conversation_id, session_id)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                row_id,
                                event["text"],
                                event["source"],
                                event["conversation_id"],
                                event["session_id"],
                            ),
                        )
                    except sqlite3.OperationalError as exc:
                        logger.warning("raw_events FTS insert failed: %s", exc)
                conn.commit()
                return "inserted", row_id
            conn.commit()
            row_id = self._find_existing_id(conn, event)
            return "duplicate", row_id
        finally:
            conn.close()

    def _find_existing_id(self, conn: sqlite3.Connection, event: dict[str, Any]) -> int | None:
        if event.get("source_event_id"):
            row = conn.execute(
                "SELECT id FROM raw_events WHERE source = ? AND source_event_id = ? LIMIT 1",
                (event["source"], event["source_event_id"]),
            ).fetchone()
            if row:
                return int(row["id"])
        row = conn.execute(
            "SELECT id FROM raw_events WHERE source = ? AND event_hash = ? LIMIT 1",
            (event["source"], event["event_hash"]),
        ).fetchone()
        return int(row["id"]) if row else None

    def _normalize_event(
        self,
        raw: dict[str, Any] | None,
        *,
        default_source: str,
        ingested_at: str,
    ) -> tuple[dict[str, Any] | None, str]:
        if not isinstance(raw, dict):
            return None, "invalid_event"
        role = str(raw.get("role") or "").strip().lower()
        if role not in ALLOWED_RAW_ROLES:
            return None, "invalid_role"
        text = strip_raw_client_context(self._coerce_text(raw.get("text", raw.get("content", ""))))
        if not text:
            return None, "empty_text"
        if self._looks_injected(text, raw):
            return None, "injected_context"

        source = self._clean_source(raw.get("source") or default_source)
        source_event_id = str(raw.get("source_event_id") or raw.get("event_id") or raw.get("id") or "").strip()
        conversation_id = str(raw.get("conversation_id") or raw.get("thread_id") or "").strip()
        session_id = str(raw.get("session_id") or "").strip()
        client = str(raw.get("client") or "").strip()
        created_at = self._clean_time(raw.get("created_at") or raw.get("timestamp") or raw.get("time") or ingested_at)
        metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        metadata_json = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        event_hash = self._event_hash(
            source=source,
            source_event_id=source_event_id,
            role=role,
            text=text,
            created_at=created_at,
            conversation_id=conversation_id,
            session_id=session_id,
        )
        return {
            "source": source,
            "source_event_id": source_event_id,
            "event_hash": event_hash,
            "role": role,
            "text": text,
            "created_at": created_at,
            "ingested_at": ingested_at,
            "conversation_id": conversation_id,
            "session_id": session_id,
            "client": client,
            "metadata_json": metadata_json,
        }, ""

    @staticmethod
    def _clean_source(value: Any) -> str:
        text = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value or RAW_EVENT_DEFAULT_SOURCE).strip())
        return text[:80] or RAW_EVENT_DEFAULT_SOURCE

    @staticmethod
    def _clean_time(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return RawEventStore._now_iso()
        return text[:80]

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _coerce_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    item_type = str(item.get("type") or "").lower()
                    if item_type in {"tool_result", "tool_use", "function_call", "function_result"}:
                        continue
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(part for part in parts if part)
        return str(value or "")

    @staticmethod
    def _looks_injected(text: str, raw: dict[str, Any]) -> bool:
        return raw_event_text_looks_injected(text, raw)

    @staticmethod
    def _event_hash(**parts: str) -> str:
        payload = json.dumps(parts, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> dict[str, Any]:
        metadata = {}
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except Exception:
            metadata = {}
        return {
            "id": row["id"],
            "source": row["source"],
            "source_event_id": row["source_event_id"],
            "role": row["role"],
            "text": row["text"],
            "created_at": row["created_at"],
            "ingested_at": row["ingested_at"],
            "conversation_id": row["conversation_id"],
            "session_id": row["session_id"],
            "client": row["client"],
            "metadata": metadata,
        }

    def _search_filters(
        self,
        *,
        source: str = "",
        role: str = "",
        conversation_id: str = "",
        session_id: str = "",
        since: str = "",
        until: str = "",
    ) -> tuple[str, list[Any]]:
        clauses = []
        params: list[Any] = []
        if source:
            clauses.append("e.source = ?")
            params.append(self._clean_source(source))
        role = str(role or "").strip().lower()
        if role in ALLOWED_RAW_ROLES:
            clauses.append("e.role = ?")
            params.append(role)
        if conversation_id:
            clauses.append("e.conversation_id = ?")
            params.append(str(conversation_id))
        if session_id:
            clauses.append("e.session_id = ?")
            params.append(str(session_id))
        if since:
            clauses.append("e.created_at >= ?")
            params.append(str(since))
        if until:
            clauses.append("e.created_at <= ?")
            params.append(str(until))
        return (" AND " + " AND ".join(clauses)) if clauses else "", params

    def _search_fts(self, query: str, filters: str, params: list[Any], limit: int) -> list[sqlite3.Row]:
        if not self.fts_enabled or not query:
            return []
        match = '"' + query.replace('"', '""') + '"'
        conn = self._connect()
        try:
            return conn.execute(
                f"""
                SELECT e.*
                FROM raw_events_fts f
                JOIN raw_events e ON e.id = f.rowid
                WHERE raw_events_fts MATCH ? {filters}
                ORDER BY bm25(raw_events_fts), e.created_at DESC, e.id DESC
                LIMIT ?
                """,
                [match, *params, limit],
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()

    def _search_like(self, query: str, filters: str, params: list[Any], limit: int) -> list[sqlite3.Row]:
        conn = self._connect()
        try:
            return conn.execute(
                f"""
                SELECT e.*
                FROM raw_events e
                WHERE e.text LIKE ? {filters}
                ORDER BY e.created_at DESC, e.id DESC
                LIMIT ?
                """,
                [f"%{query}%", *params, limit],
            ).fetchall()
        finally:
            conn.close()

    def _search_recent(self, filters: str, params: list[Any], limit: int) -> list[sqlite3.Row]:
        conn = self._connect()
        try:
            return conn.execute(
                f"""
                SELECT e.*
                FROM raw_events e
                WHERE 1 = 1 {filters}
                ORDER BY e.created_at DESC, e.id DESC
                LIMIT ?
                """,
                [*params, limit],
            ).fetchall()
        finally:
            conn.close()

    @staticmethod
    def _merge_rows(first: list[sqlite3.Row], second: list[sqlite3.Row], limit: int) -> list[sqlite3.Row]:
        rows = []
        seen = set()
        for row in [*(first or []), *(second or [])]:
            row_id = int(row["id"])
            if row_id in seen:
                continue
            seen.add(row_id)
            rows.append(row)
            if len(rows) >= limit:
                break
        return rows
