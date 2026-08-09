"""Canonical source-bound Fact and Event storage.

Facts and Events are extracted directly from trusted raw dialogue. They are
not derived from Scene prose and do not participate in recall, embeddings, or
the memory graph yet. Exact source refs remain the evidence boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import unicodedata
from contextlib import closing
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from scene_evidence import normalize_evidence_ref


SCHEMA_VERSION = "fact-event-v1"
ITEM_TYPES = frozenset({"fact", "event"})
ITEM_STATUSES = frozenset({"active", "archived", "superseded", "tombstoned"})
LOCAL_TZ = ZoneInfo("Asia/Shanghai")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:@/-]+$")


class FactEventStore:
    """SQLite owner for reviewed Fact/Event objects and their raw evidence."""

    def __init__(self, config: dict[str, Any] | None = None, *, create: bool = False):
        config = config or {}
        state_dir = str(
            config.get("state_dir")
            or os.path.join(
                os.path.dirname(os.path.abspath(config.get("buckets_dir", "buckets"))),
                "state",
            )
        )
        self.db_path = os.path.join(state_dir, "fact_events.sqlite")
        if create:
            self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS fact_events (
                    item_id TEXT PRIMARY KEY,
                    schema_version TEXT NOT NULL,
                    fingerprint TEXT NOT NULL UNIQUE,
                    origin_id TEXT UNIQUE,
                    item_type TEXT NOT NULL CHECK(item_type IN ('fact', 'event')),
                    title TEXT NOT NULL DEFAULT '',
                    body TEXT NOT NULL,
                    importance INTEGER NOT NULL DEFAULT 1 CHECK(importance BETWEEN 1 AND 5),
                    status TEXT NOT NULL DEFAULT 'active'
                        CHECK(status IN ('active', 'archived', 'superseded', 'tombstoned')),
                    local_date TEXT NOT NULL,
                    local_end_date TEXT NOT NULL,
                    local_start_time TEXT NOT NULL,
                    local_end_time TEXT NOT NULL,
                    source_started_at TEXT NOT NULL,
                    source_ended_at TEXT NOT NULL,
                    supersedes_item_id TEXT NOT NULL DEFAULT '',
                    covered_by_scene_id TEXT NOT NULL DEFAULT '',
                    covered_at TEXT NOT NULL DEFAULT '',
                    injection_count INTEGER NOT NULL DEFAULT 0,
                    last_injected_at TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS fact_event_sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id TEXT NOT NULL,
                    source_system TEXT NOT NULL,
                    session_id TEXT NOT NULL DEFAULT '',
                    thread_id TEXT NOT NULL DEFAULT '',
                    message_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    content TEXT NOT NULL DEFAULT '',
                    snapshot_ref TEXT NOT NULL DEFAULT '',
                    content_sha256 TEXT NOT NULL,
                    hash_algorithm TEXT NOT NULL,
                    evidence_kind TEXT NOT NULL,
                    binding_method TEXT NOT NULL,
                    UNIQUE(item_id, source_system, session_id, message_id),
                    FOREIGN KEY(item_id) REFERENCES fact_events(item_id) ON DELETE RESTRICT
                );

                CREATE INDEX IF NOT EXISTS idx_fact_events_date_type
                ON fact_events(local_date, item_type, status, local_start_time, item_id);

                CREATE INDEX IF NOT EXISTS idx_fact_events_surface
                ON fact_events(status, importance, injection_count, local_date);

                CREATE INDEX IF NOT EXISTS idx_fact_event_sources_message
                ON fact_event_sources(source_system, session_id, message_id);
                """
            )

    def write_many(self, raw_items: Any) -> dict[str, Any]:
        """Write a batch with per-item rejection and idempotent origin checks."""

        if not isinstance(raw_items, list) or not raw_items:
            raise ValueError("items must be a non-empty list")
        if len(raw_items) > 500:
            raise ValueError("at most 500 items may be written in one batch")

        prepared: list[tuple[int, dict[str, Any]]] = []
        results: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_items):
            try:
                prepared.append((index, _normalize_item(raw)))
            except ValueError as exc:
                results.append({"index": index, "status": "rejected", "error": str(exc)})

        self._init_db()
        now = _now()
        inserted = 0
        idempotent = 0
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                for index, item in prepared:
                    result = self._write_one(conn, item, now=now)
                    results.append({"index": index, **result})
                    if result["status"] == "inserted":
                        inserted += 1
                    elif result["status"] == "idempotent":
                        idempotent += 1
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        results.sort(key=lambda item: int(item["index"]))
        return {
            "ok": True,
            "inserted": inserted,
            "idempotent": idempotent,
            "rejected": sum(1 for item in results if item["status"] == "rejected"),
            "items": results,
        }

    def _write_one(
        self,
        conn: sqlite3.Connection,
        item: dict[str, Any],
        *,
        now: str,
    ) -> dict[str, Any]:
        origin_id = item["origin_id"] or None
        if origin_id:
            existing = conn.execute(
                "SELECT item_id, fingerprint FROM fact_events WHERE origin_id=?",
                (origin_id,),
            ).fetchone()
            if existing is not None:
                if str(existing["fingerprint"]) != item["fingerprint"]:
                    return {
                        "status": "rejected",
                        "error": "origin_id already exists with different content",
                        "origin_id": item["origin_id"],
                    }
                return {
                    "status": "idempotent",
                    "item_id": str(existing["item_id"]),
                    "origin_id": item["origin_id"],
                }

        existing = conn.execute(
            "SELECT item_id FROM fact_events WHERE fingerprint=?",
            (item["fingerprint"],),
        ).fetchone()
        if existing is not None:
            return {
                "status": "idempotent",
                "item_id": str(existing["item_id"]),
                "origin_id": item["origin_id"],
            }

        supersedes_id = item["supersedes_item_id"]
        if supersedes_id:
            previous = conn.execute(
                "SELECT item_type, status FROM fact_events WHERE item_id=?",
                (supersedes_id,),
            ).fetchone()
            if previous is None:
                return {"status": "rejected", "error": "supersedes item not found"}
            if str(previous["item_type"]) != item["item_type"]:
                return {"status": "rejected", "error": "supersedes item type mismatch"}

        conn.execute(
            """
            INSERT INTO fact_events (
                item_id, schema_version, fingerprint, origin_id, item_type, title,
                body, importance, status, local_date, local_end_date,
                local_start_time, local_end_time, source_started_at,
                source_ended_at, supersedes_item_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item["item_id"],
                SCHEMA_VERSION,
                item["fingerprint"],
                origin_id,
                item["item_type"],
                item["title"],
                item["body"],
                item["importance"],
                item["local_date"],
                item["local_end_date"],
                item["local_start_time"],
                item["local_end_time"],
                item["source_started_at"],
                item["source_ended_at"],
                supersedes_id,
                now,
                now,
            ),
        )
        for ref in item["source_refs"]:
            conn.execute(
                """
                INSERT INTO fact_event_sources (
                    item_id, source_system, session_id, thread_id, message_id,
                    role, created_at, content, snapshot_ref, content_sha256,
                    hash_algorithm, evidence_kind, binding_method
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item["item_id"],
                    ref["source_system"],
                    ref["session_id"],
                    ref["thread_id"],
                    ref["message_id"],
                    ref["role"],
                    ref["created_at"],
                    ref["content"],
                    ref["snapshot_ref"],
                    ref["content_sha256"],
                    ref["hash_algorithm"],
                    ref["evidence_kind"],
                    ref["binding_method"],
                ),
            )
        if supersedes_id:
            conn.execute(
                """
                UPDATE fact_events
                SET status='superseded', updated_at=?
                WHERE item_id=? AND status='active'
                """,
                (now, supersedes_id),
            )
        return {
            "status": "inserted",
            "item_id": item["item_id"],
            "origin_id": item["origin_id"],
        }

    def read(self, item_id: str, *, include_sources: bool = True) -> dict[str, Any] | None:
        safe_id = _required_identifier(item_id, "item_id", 128)
        if not os.path.exists(self.db_path):
            return None
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM fact_events WHERE item_id=?", (safe_id,)).fetchone()
            return self._row_payload(conn, row, include_sources=include_sources) if row else None

    def list(
        self,
        *,
        item_type: str = "",
        status: str = "active",
        date: str = "",
        limit: int = 100,
        offset: int = 0,
        include_sources: bool = False,
    ) -> dict[str, Any]:
        if not os.path.exists(self.db_path):
            return {"items": [], "count": 0, "limit": limit, "offset": offset}
        safe_type = str(item_type or "").strip().lower()
        if safe_type and safe_type not in ITEM_TYPES:
            raise ValueError("item_type must be fact or event")
        safe_status = str(status or "active").strip().lower()
        if safe_status != "all" and safe_status not in ITEM_STATUSES:
            raise ValueError("unsupported status")
        safe_date = str(date or "").strip()
        if safe_date and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", safe_date):
            raise ValueError("date must use YYYY-MM-DD")
        safe_limit = max(1, min(500, int(limit or 100)))
        safe_offset = max(0, int(offset or 0))

        clauses: list[str] = []
        params: list[Any] = []
        if safe_type:
            clauses.append("item_type=?")
            params.append(safe_type)
        if safe_status != "all":
            clauses.append("status=?")
            params.append(safe_status)
        if safe_date:
            clauses.append("local_date<=? AND local_end_date>=?")
            params.extend([safe_date, safe_date])
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with closing(self._connect()) as conn:
            count = int(conn.execute(f"SELECT COUNT(*) FROM fact_events{where}", params).fetchone()[0])
            rows = conn.execute(
                f"""
                SELECT * FROM fact_events{where}
                ORDER BY local_date DESC, local_start_time DESC, item_id DESC
                LIMIT ? OFFSET ?
                """,
                [*params, safe_limit, safe_offset],
            ).fetchall()
            return {
                "items": [
                    self._row_payload(conn, row, include_sources=include_sources)
                    for row in rows
                ],
                "count": count,
                "limit": safe_limit,
                "offset": safe_offset,
            }

    def mark_injected(self, item_ids: Any) -> dict[str, Any]:
        ids = _normalize_ids(item_ids, "item_ids", limit=500)
        if not ids or not os.path.exists(self.db_path):
            return {"updated": 0, "item_ids": []}
        placeholders = ",".join("?" for _ in ids)
        now = _now()
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                f"SELECT item_id FROM fact_events WHERE item_id IN ({placeholders}) AND status='active'",
                ids,
            ).fetchall()
            found = [str(row["item_id"]) for row in rows]
            if found:
                found_placeholders = ",".join("?" for _ in found)
                conn.execute(
                    f"""
                    UPDATE fact_events
                    SET injection_count=injection_count+1,
                        last_injected_at=?, updated_at=?
                    WHERE item_id IN ({found_placeholders})
                    """,
                    [now, now, *found],
                )
            conn.commit()
        return {"updated": len(found), "item_ids": found}

    def archive_events_covered_by_scene(
        self,
        scene_id: str,
        evidence_refs: Any,
    ) -> dict[str, Any]:
        safe_scene_id = _required_identifier(scene_id, "scene_id", 128)
        if not isinstance(evidence_refs, list) or not evidence_refs:
            return {"archived": 0, "item_ids": []}
        return self.archive_events_covered_by_scenes({safe_scene_id: evidence_refs})

    def archive_events_covered_by_scenes(
        self,
        scene_groups: Any,
    ) -> dict[str, Any]:
        """Archive active Events fully covered by any Scene's active evidence."""

        if not isinstance(scene_groups, dict) or not scene_groups or not os.path.exists(self.db_path):
            return {"archived": 0, "item_ids": []}
        normalized_groups: list[tuple[str, set[tuple[str, str, str]]]] = []
        for raw_scene_id, raw_refs in scene_groups.items():
            safe_scene_id = _required_identifier(raw_scene_id, "scene_id", 128)
            if not isinstance(raw_refs, list) or not raw_refs:
                continue
            normalized_groups.append(
                (
                    safe_scene_id,
                    {_source_key(normalize_evidence_ref(raw)) for raw in raw_refs},
                )
            )
        if not normalized_groups:
            return {"archived": 0, "item_ids": []}

        now = _now()
        archived: list[str] = []
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT item_id FROM fact_events WHERE item_type='event' AND status='active'"
            ).fetchall()
            for row in rows:
                item_id = str(row["item_id"])
                source_rows = conn.execute(
                    """
                    SELECT source_system, session_id, message_id
                    FROM fact_event_sources WHERE item_id=?
                    """,
                    (item_id,),
                ).fetchall()
                item_keys = {
                    (str(ref["source_system"]), str(ref["session_id"]), str(ref["message_id"]))
                    for ref in source_rows
                }
                covering = [
                    (len(scene_keys) - len(item_keys), scene_id)
                    for scene_id, scene_keys in normalized_groups
                    if item_keys and item_keys.issubset(scene_keys)
                ]
                if covering:
                    _, scene_id = min(covering)
                    conn.execute(
                        """
                        UPDATE fact_events
                        SET status='archived', covered_by_scene_id=?, covered_at=?, updated_at=?
                        WHERE item_id=? AND status='active'
                        """,
                        (scene_id, now, now, item_id),
                    )
                    archived.append(item_id)
            conn.commit()
        return {"archived": len(archived), "item_ids": archived}

    def stats(self) -> dict[str, Any]:
        if not os.path.exists(self.db_path):
            return {"total": 0, "facts": 0, "events": 0, "active": 0}
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) total,
                       SUM(item_type='fact') facts,
                       SUM(item_type='event') events,
                       SUM(status='active') active
                FROM fact_events
                """
            ).fetchone()
        return {key: int(row[key] or 0) for key in ("total", "facts", "events", "active")}

    @staticmethod
    def _row_payload(
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        include_sources: bool,
    ) -> dict[str, Any]:
        payload = {key: row[key] for key in row.keys()}
        payload["origin_id"] = str(payload.get("origin_id") or "")
        if include_sources:
            refs = conn.execute(
                "SELECT * FROM fact_event_sources WHERE item_id=? ORDER BY created_at, id",
                (str(row["item_id"]),),
            ).fetchall()
            payload["source_refs"] = [
                {key: ref[key] for key in ref.keys() if key not in {"id", "item_id"}}
                for ref in refs
            ]
        return payload


def _normalize_item(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("each item must be an object")
    item_type = str(raw.get("type") or raw.get("item_type") or "").strip().lower()
    if item_type not in ITEM_TYPES:
        raise ValueError("type must be fact or event")
    body = _required_text(raw.get("body"), "body", 1600 if item_type == "event" else 500)
    title = str(raw.get("title") or "").strip()
    if item_type == "event":
        title = _required_text(title, "event title", 160)
    elif title:
        raise ValueError("fact must not carry a title")
    try:
        importance = int(raw.get("importance", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("importance must be an integer from 1 to 5") from exc
    if not 1 <= importance <= 5:
        raise ValueError("importance must be an integer from 1 to 5")

    refs_raw = raw.get("source_refs")
    if not isinstance(refs_raw, list) or not refs_raw:
        raise ValueError("source_refs must be a non-empty list")
    if len(refs_raw) > 24:
        raise ValueError("at most 24 source refs may support one item")
    refs = [normalize_evidence_ref(item) for item in refs_raw]
    primary_count = sum(1 for ref in refs if ref["evidence_kind"] == "primary")
    if primary_count != 1:
        raise ValueError("exactly one source ref must be primary")
    keys = [_source_key(ref) for ref in refs]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate source refs are not allowed")
    refs.sort(key=lambda ref: (_parse_timestamp(ref["created_at"]), _source_key(ref)))
    bounds = _source_bounds(refs)

    origin_id = _optional_identifier(raw.get("origin_id"), "origin_id", 200)
    supersedes_id = _optional_identifier(
        raw.get("supersedes_item_id"), "supersedes_item_id", 128
    )
    normalized_body = _normalized_proposition(body)
    fingerprint_material = {
        "schema_version": SCHEMA_VERSION,
        "type": item_type,
        "title": _normalized_proposition(title),
        "body": normalized_body,
        "sources": [list(_source_key(ref)) for ref in refs],
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    item_id = f"{item_type}_{fingerprint[:24]}"
    return {
        "item_id": item_id,
        "fingerprint": fingerprint,
        "origin_id": origin_id,
        "item_type": item_type,
        "title": title,
        "body": body,
        "importance": importance,
        "supersedes_item_id": supersedes_id,
        "source_refs": refs,
        **bounds,
    }


def _source_bounds(refs: list[dict[str, Any]]) -> dict[str, str]:
    values = [(_parse_timestamp(ref["created_at"]), str(ref["created_at"])) for ref in refs]
    values.sort(key=lambda item: item[0])
    start_dt, start_raw = values[0]
    end_dt, end_raw = values[-1]
    local_start = start_dt.astimezone(LOCAL_TZ)
    local_end = end_dt.astimezone(LOCAL_TZ)
    return {
        "local_date": local_start.date().isoformat(),
        "local_end_date": local_end.date().isoformat(),
        "local_start_time": local_start.strftime("%H:%M"),
        "local_end_time": local_end.strftime("%H:%M"),
        "source_started_at": start_raw,
        "source_ended_at": end_raw,
    }


def _parse_timestamp(value: Any) -> datetime:
    text = _required_text(value, "source created_at", 128)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("source created_at must be an ISO timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _source_key(ref: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(ref.get("source_system") or ""),
        str(ref.get("session_id") or ""),
        str(ref.get("message_id") or ""),
    )


def _normalized_proposition(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.rstrip("。！？!?；;，,")


def _normalize_ids(raw: Any, field: str, *, limit: int) -> list[str]:
    if not isinstance(raw, list):
        raise ValueError(f"{field} must be a list")
    if len(raw) > limit:
        raise ValueError(f"{field} may contain at most {limit} values")
    result: list[str] = []
    for value in raw:
        item_id = _required_identifier(value, field, 128)
        if item_id not in result:
            result.append(item_id)
    return result


def _required_text(value: Any, field: str, limit: int) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} is required")
    if len(text) > limit:
        raise ValueError(f"{field} is too long")
    return text


def _required_identifier(value: Any, field: str, limit: int) -> str:
    text = _required_text(value, field, limit)
    if not _IDENTIFIER_RE.fullmatch(text):
        raise ValueError(f"{field} contains unsupported characters")
    return text


def _optional_identifier(value: Any, field: str, limit: int) -> str:
    text = str(value or "").strip()
    return _required_identifier(text, field, limit) if text else ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
