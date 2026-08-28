"""Canonical source-bound Fact and Event storage.

Facts and Events are extracted directly from trusted raw dialogue. They are
not derived from Scene prose and do not participate in the memory graph.
Rebuildable recall derivatives may embed their bodies, while this canonical
store and its exact source refs remain the evidence boundary.
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
FACT_RELATIONS = frozenset({"duplicate", "reinforces", "updates", "contradicts", "supersedes"})
FACT_RELATION_STATUSES = frozenset({"pending", "accepted", "rejected", "superseded"})
LOCAL_TZ = ZoneInfo("Asia/Shanghai")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:@/-]+$")


def _normalized_search_text(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).casefold()


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
        conn.create_function("normalized_search_text", 1, _normalized_search_text, deterministic=True)
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

                CREATE TABLE IF NOT EXISTS fact_relation_proposals (
                    proposal_id TEXT PRIMARY KEY,
                    new_fact_id TEXT NOT NULL,
                    candidate_fact_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
                    status TEXT NOT NULL DEFAULT 'pending',
                    review_reason TEXT NOT NULL DEFAULT '',
                    review_confidence REAL,
                    explicit_correction INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(new_fact_id, candidate_fact_id, relation),
                    FOREIGN KEY(new_fact_id) REFERENCES fact_events(item_id) ON DELETE RESTRICT,
                    FOREIGN KEY(candidate_fact_id) REFERENCES fact_events(item_id) ON DELETE RESTRICT
                );

                CREATE INDEX IF NOT EXISTS idx_fact_relation_proposals_status
                ON fact_relation_proposals(status, created_at, proposal_id);
                """
            )
            columns = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(fact_events)").fetchall()
            }
            if "fact_type" not in columns:
                conn.execute("ALTER TABLE fact_events ADD COLUMN fact_type TEXT NOT NULL DEFAULT ''")
            if "atomic_question" not in columns:
                conn.execute(
                    "ALTER TABLE fact_events ADD COLUMN atomic_question TEXT NOT NULL DEFAULT ''"
                )
            proposal_columns = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(fact_relation_proposals)").fetchall()
            }
            if "review_reason" not in proposal_columns:
                conn.execute(
                    "ALTER TABLE fact_relation_proposals ADD COLUMN review_reason TEXT NOT NULL DEFAULT ''"
                )
            if "review_confidence" not in proposal_columns:
                conn.execute("ALTER TABLE fact_relation_proposals ADD COLUMN review_confidence REAL")
            if "explicit_correction" not in proposal_columns:
                conn.execute(
                    "ALTER TABLE fact_relation_proposals ADD COLUMN explicit_correction INTEGER NOT NULL DEFAULT 0"
                )

    def write_many(self, raw_items: Any) -> dict[str, Any]:
        """Write an idempotent batch atomically; one rejection rolls back every new item."""

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
        if results:
            invalid = {int(item["index"]): str(item["error"]) for item in results}
            results = [
                {
                    "index": index,
                    "status": "rejected",
                    "error": invalid.get(index, "batch rolled back because another item is invalid"),
                }
                for index in range(len(raw_items))
            ]
            return {
                "ok": False,
                "inserted": 0,
                "idempotent": 0,
                "rejected": len(results),
                "items": results,
            }

        self._init_db()
        now = _now()
        inserted = 0
        idempotent = 0
        rejected_index: int | None = None
        rejected_error = ""
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                for index, item in prepared:
                    result = self._write_one(conn, item, now=now)
                    if result["status"] == "rejected":
                        rejected_index = index
                        rejected_error = str(result.get("error") or "rejected")
                        raise ValueError(rejected_error)
                    results.append({"index": index, **result})
                    if result["status"] == "inserted":
                        inserted += 1
                    elif result["status"] == "idempotent":
                        idempotent += 1
                conn.commit()
            except ValueError:
                conn.rollback()
                if rejected_index is None:
                    raise
            except Exception:
                conn.rollback()
                raise

        if rejected_index is not None:
            results = [
                {
                    "index": index,
                    "status": "rejected",
                    "error": (
                        rejected_error
                        if index == rejected_index
                        else "batch rolled back because another item was rejected"
                    ),
                }
                for index in range(len(raw_items))
            ]
            return {
                "ok": False,
                "inserted": 0,
                "idempotent": 0,
                "rejected": len(results),
                "items": results,
            }

        results.sort(key=lambda item: int(item["index"]))
        return {
            "ok": True,
            "inserted": inserted,
            "idempotent": idempotent,
            "rejected": sum(1 for item in results if item["status"] == "rejected"),
            "items": results,
        }

    def replace_many(self, raw_items: Any) -> dict[str, Any]:
        """Atomically write reviewed replacements and supersede every predecessor."""

        if not isinstance(raw_items, list) or not raw_items:
            raise ValueError("items must be a non-empty list")
        if len(raw_items) > 500:
            raise ValueError("at most 500 items may be written in one batch")

        prepared: list[tuple[int, dict[str, Any], list[str]]] = []
        predecessor_ids: set[str] = set()
        origin_ids: set[str] = set()
        fingerprints: set[str] = set()
        for index, raw in enumerate(raw_items):
            if not isinstance(raw, dict):
                raise ValueError("each item must be an object")
            raw_predecessors = raw.get("supersedes_item_ids")
            if not isinstance(raw_predecessors, list) or not raw_predecessors:
                raise ValueError("supersedes_item_ids must be a non-empty list")
            if len(raw_predecessors) > 100:
                raise ValueError("at most 100 predecessor items may be superseded")
            predecessors = [
                _required_identifier(value, "supersedes_item_ids", 128)
                for value in raw_predecessors
            ]
            if len(set(predecessors)) != len(predecessors):
                raise ValueError("duplicate supersedes_item_ids are not allowed")
            if predecessor_ids.intersection(predecessors):
                raise ValueError("a predecessor may appear in only one replacement")
            predecessor_ids.update(predecessors)

            explicit_primary = str(raw.get("supersedes_item_id") or "").strip()
            if explicit_primary and explicit_primary != predecessors[0]:
                raise ValueError("supersedes_item_id must equal the first supersedes_item_ids entry")
            normalized = _normalize_item(
                {**raw, "supersedes_item_id": predecessors[0]}
            )
            origin_id = str(normalized["origin_id"] or "")
            if origin_id and origin_id in origin_ids:
                raise ValueError("duplicate origin_id values are not allowed")
            if normalized["fingerprint"] in fingerprints:
                raise ValueError("duplicate replacement fingerprints are not allowed")
            if origin_id:
                origin_ids.add(origin_id)
            fingerprints.add(normalized["fingerprint"])
            prepared.append((index, normalized, predecessors))

        self._init_db()
        now = _now()
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                existing_items: list[sqlite3.Row | None] = []
                for _, item, _ in prepared:
                    existing = None
                    if item["origin_id"]:
                        existing = conn.execute(
                            "SELECT item_id, fingerprint FROM fact_events WHERE origin_id=?",
                            (item["origin_id"],),
                        ).fetchone()
                        if existing is not None and str(existing["fingerprint"]) != item["fingerprint"]:
                            raise ValueError("origin_id already exists with different content")
                    if existing is None:
                        existing = conn.execute(
                            "SELECT item_id, fingerprint FROM fact_events WHERE fingerprint=?",
                            (item["fingerprint"],),
                        ).fetchone()
                    existing_items.append(existing)

                existing_count = sum(item is not None for item in existing_items)
                if existing_count:
                    if existing_count != len(prepared):
                        raise ValueError("replacement batch is only partially present")
                    for _, _, predecessors in prepared:
                        placeholders = ",".join("?" for _ in predecessors)
                        rows = conn.execute(
                            f"SELECT item_id, status FROM fact_events WHERE item_id IN ({placeholders})",
                            predecessors,
                        ).fetchall()
                        statuses = {str(row["item_id"]): str(row["status"]) for row in rows}
                        if any(statuses.get(item_id) != "superseded" for item_id in predecessors):
                            raise ValueError("idempotent replacement has non-superseded predecessors")
                    conn.rollback()
                    return {
                        "ok": True,
                        "inserted": 0,
                        "idempotent": len(prepared),
                        "rejected": 0,
                        "items": [
                            {
                                "index": index,
                                "status": "idempotent",
                                "item_id": str(existing_items[position]["item_id"]),
                                "origin_id": item["origin_id"],
                                "superseded_item_ids": predecessors,
                            }
                            for position, (index, item, predecessors) in enumerate(prepared)
                        ],
                    }

                for _, item, predecessors in prepared:
                    placeholders = ",".join("?" for _ in predecessors)
                    rows = conn.execute(
                        f"SELECT item_id, item_type, status FROM fact_events WHERE item_id IN ({placeholders})",
                        predecessors,
                    ).fetchall()
                    previous = {str(row["item_id"]): row for row in rows}
                    if len(previous) != len(predecessors):
                        raise ValueError("supersedes item not found")
                    if any(str(previous[item_id]["item_type"]) != item["item_type"] for item_id in predecessors):
                        raise ValueError("supersedes item type mismatch")
                    if any(str(previous[item_id]["status"]) != "active" for item_id in predecessors):
                        raise ValueError("all superseded items must be active")

                results: list[dict[str, Any]] = []
                for index, item, predecessors in prepared:
                    written = self._write_one(conn, item, now=now)
                    if written.get("status") != "inserted":
                        raise ValueError(written.get("error") or "replacement was not inserted")
                    for predecessor_id in predecessors[1:]:
                        changed = conn.execute(
                            """
                            UPDATE fact_events
                            SET status='superseded', updated_at=?
                            WHERE item_id=? AND status='active'
                            """,
                            (now, predecessor_id),
                        )
                        if changed.rowcount != 1:
                            raise ValueError("predecessor status changed during replacement")
                    results.append(
                        {
                            "index": index,
                            **written,
                            "superseded_item_ids": predecessors,
                        }
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        return {
            "ok": True,
            "inserted": len(prepared),
            "idempotent": 0,
            "rejected": 0,
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
                source_ended_at, supersedes_item_id, created_at, updated_at,
                fact_type, atomic_question
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                item["fact_type"],
                item["atomic_question"],
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
        query: str = "",
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
        safe_query = unicodedata.normalize("NFKC", str(query or "")).strip()
        if len(safe_query) > 200:
            raise ValueError("query must be at most 200 characters")
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
        if safe_query:
            clauses.append(
                "(INSTR(normalized_search_text(title), normalized_search_text(?)) > 0 "
                "OR INSTR(normalized_search_text(body), normalized_search_text(?)) > 0)"
            )
            params.extend([safe_query, safe_query])
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with closing(self._connect()) as conn:
            count = int(conn.execute(f"SELECT COUNT(*) FROM fact_events{where}", params).fetchone()[0])
            rows = conn.execute(
                f"""
                SELECT fact_events.*,
                       (
                           SELECT COUNT(*)
                           FROM fact_event_sources
                           WHERE fact_event_sources.item_id = fact_events.item_id
                       ) AS source_count
                FROM fact_events{where}
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

    def revise(
        self,
        item_id: str,
        *,
        title: Any = None,
        body: Any = None,
        importance: Any = None,
    ) -> dict[str, Any]:
        """Revise prose by superseding the item; update importance as metadata."""

        current = self.read(item_id, include_sources=True)
        if current is None:
            raise ValueError("item not found")
        if str(current["status"]) not in {"active", "archived"}:
            raise ValueError("only active or archived items may be revised")

        item_type = str(current["item_type"])
        next_title = str(current["title"] if title is None else title).strip()
        next_body = str(current["body"] if body is None else body).strip()
        next_importance = current["importance"] if importance is None else importance
        candidate = _normalize_item(
            {
                "type": item_type,
                "title": next_title,
                "body": next_body,
                "importance": next_importance,
                "fact_type": current.get("fact_type", ""),
                "atomic_question": current.get("atomic_question", ""),
                "supersedes_item_id": str(current["item_id"]),
                "source_refs": current["source_refs"],
            }
        )

        if candidate["fingerprint"] == str(current["fingerprint"]):
            now = _now()
            with closing(self._connect()) as conn:
                conn.execute(
                    "UPDATE fact_events SET importance=?, updated_at=? WHERE item_id=?",
                    (candidate["importance"], now, str(current["item_id"])),
                )
                conn.commit()
            return {
                "ok": True,
                "status": "updated",
                "item": self.read(str(current["item_id"]), include_sources=True),
            }

        result = self.write_many(
            [
                {
                    "type": item_type,
                    "title": next_title,
                    "body": next_body,
                    "importance": candidate["importance"],
                    "fact_type": current.get("fact_type", ""),
                    "atomic_question": current.get("atomic_question", ""),
                    "supersedes_item_id": str(current["item_id"]),
                    "source_refs": current["source_refs"],
                }
            ]
        )
        written = result["items"][0]
        if written["status"] != "inserted":
            raise ValueError(written.get("error") or "revision did not create a new item")
        return {
            "ok": True,
            "status": "superseded",
            "previous_item_id": str(current["item_id"]),
            "item": self.read(str(written["item_id"]), include_sources=True),
        }

    def set_status(self, item_id: str, status: str) -> dict[str, Any]:
        """Archive or restore one current Fact/Event item."""

        safe_id = _required_identifier(item_id, "item_id", 128)
        target = str(status or "").strip().lower()
        if target not in {"active", "archived"}:
            raise ValueError("status must be active or archived")
        current = self.read(safe_id, include_sources=False)
        if current is None:
            raise ValueError("item not found")
        if str(current["status"]) not in {"active", "archived"}:
            raise ValueError("only active or archived items may change status")
        now = _now()
        with closing(self._connect()) as conn:
            conn.execute(
                "UPDATE fact_events SET status=?, updated_at=? WHERE item_id=?",
                (target, now, safe_id),
            )
            conn.commit()
        return {
            "ok": True,
            "status": target,
            "item": self.read(safe_id, include_sources=True),
        }

    def delete(self, item_id: str) -> dict[str, Any]:
        """Permanently delete one Fact/Event and its complete revision family."""

        safe_id = _required_identifier(item_id, "item_id", 128)
        self._init_db()
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                current = conn.execute(
                    "SELECT item_id, item_type FROM fact_events WHERE item_id=?",
                    (safe_id,),
                ).fetchone()
                if current is None:
                    raise ValueError("item not found")

                family = {safe_id}
                changed = True
                while changed:
                    changed = False
                    rows = conn.execute(
                        "SELECT item_id, supersedes_item_id FROM fact_events"
                    ).fetchall()
                    for row in rows:
                        candidate = str(row["item_id"])
                        previous = str(row["supersedes_item_id"] or "")
                        if candidate in family and previous and previous not in family:
                            family.add(previous)
                            changed = True
                        if previous in family and candidate not in family:
                            family.add(candidate)
                            changed = True

                item_ids = sorted(family)
                placeholders = ",".join("?" for _ in item_ids)
                conn.execute(
                    f"""
                    DELETE FROM fact_relation_proposals
                    WHERE new_fact_id IN ({placeholders})
                       OR candidate_fact_id IN ({placeholders})
                    """,
                    [*item_ids, *item_ids],
                )
                conn.execute(
                    f"DELETE FROM fact_event_sources WHERE item_id IN ({placeholders})",
                    item_ids,
                )
                conn.execute(
                    f"DELETE FROM fact_events WHERE item_id IN ({placeholders})",
                    item_ids,
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return {
            "ok": True,
            "deleted": len(item_ids),
            "item_ids": item_ids,
            "item_type": str(current["item_type"]),
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

    def relation_candidates(self, new_item_ids: Any, *, limit_per_fact: int = 30) -> dict[str, Any]:
        """Return bounded old-Fact pools; meal facts intentionally bypass relation work."""

        ids = _normalize_ids(new_item_ids, "new_item_ids", limit=500)
        if not ids or not os.path.exists(self.db_path):
            return {"groups": [], "skipped": []}
        safe_limit = max(1, min(100, int(limit_per_fact or 30)))
        self._init_db()
        groups: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        with closing(self._connect()) as conn:
            for item_id in ids:
                current = conn.execute(
                    "SELECT * FROM fact_events WHERE item_id=? AND item_type='fact'",
                    (item_id,),
                ).fetchone()
                if current is None:
                    skipped.append({"item_id": item_id, "reason": "not_a_fact"})
                    continue
                if str(current["fact_type"] or "") == "meal":
                    skipped.append({"item_id": item_id, "reason": "meal_fact"})
                    continue
                rows = conn.execute(
                    """
                    SELECT * FROM fact_events
                    WHERE item_type='fact' AND status='active' AND item_id<>?
                      AND fact_type<>'meal'
                    ORDER BY
                        CASE WHEN fact_type<>'' AND fact_type=? THEN 0 ELSE 1 END,
                        CASE WHEN atomic_question<>'' AND atomic_question=? THEN 0 ELSE 1 END,
                        importance DESC, local_date DESC, local_start_time DESC, item_id DESC
                    LIMIT ?
                    """,
                    (
                        item_id,
                        str(current["fact_type"] or ""),
                        str(current["atomic_question"] or ""),
                        min(300, safe_limit * 3),
                    ),
                ).fetchall()
                rows = [
                    row
                    for row in rows
                    if str(row["fact_type"] or "") != ""
                    or not _legacy_fact_looks_like_meal(str(row["body"] or ""))
                ][:safe_limit]
                groups.append(
                    {
                        "new_fact": self._row_payload(conn, current, include_sources=True),
                        "candidates": [
                            self._row_payload(conn, row, include_sources=True) for row in rows
                        ],
                    }
                )
        return {"groups": groups, "skipped": skipped}

    def propose_relations(self, raw_proposals: Any) -> dict[str, Any]:
        """Persist model suggestions for review without changing either Fact."""

        if not isinstance(raw_proposals, list):
            raise ValueError("proposals must be a list")
        if len(raw_proposals) > 500:
            raise ValueError("at most 500 proposals may be written in one batch")
        self._init_db()
        now = _now()
        results: list[dict[str, Any]] = []
        inserted = 0
        idempotent = 0
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                for index, raw in enumerate(raw_proposals):
                    try:
                        proposal = _normalize_relation_proposal(raw)
                        rows = conn.execute(
                            "SELECT item_id, item_type FROM fact_events WHERE item_id IN (?, ?)",
                            (proposal["new_fact_id"], proposal["candidate_fact_id"]),
                        ).fetchall()
                        if len(rows) != 2 or any(str(row["item_type"]) != "fact" for row in rows):
                            raise ValueError("both relation endpoints must be existing Facts")
                        existing = conn.execute(
                            """
                            SELECT proposal_id FROM fact_relation_proposals
                            WHERE new_fact_id=? AND candidate_fact_id=? AND relation=?
                            """,
                            (
                                proposal["new_fact_id"],
                                proposal["candidate_fact_id"],
                                proposal["relation"],
                            ),
                        ).fetchone()
                        if existing is not None:
                            idempotent += 1
                            results.append(
                                {
                                    "index": index,
                                    "status": "idempotent",
                                    "proposal_id": str(existing["proposal_id"]),
                                }
                            )
                            continue
                        conn.execute(
                            """
                            INSERT INTO fact_relation_proposals (
                                proposal_id, new_fact_id, candidate_fact_id, relation,
                                reason, confidence, status, review_reason,
                                review_confidence, explicit_correction, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                proposal["proposal_id"],
                                proposal["new_fact_id"],
                                proposal["candidate_fact_id"],
                                proposal["relation"],
                                proposal["reason"],
                                proposal["confidence"],
                                proposal["review_status"],
                                proposal["review_reason"],
                                proposal["review_confidence"],
                                int(proposal["explicit_correction"]),
                                now,
                                now,
                            ),
                        )
                        if (
                            proposal["review_status"] == "accepted"
                            and proposal["relation"] == "supersedes"
                            and proposal["explicit_correction"]
                            and proposal["review_confidence"] >= 0.9
                        ):
                            conn.execute(
                                """
                                UPDATE fact_events
                                SET status='superseded', updated_at=?
                                WHERE item_id=? AND status='active'
                                """,
                                (now, proposal["candidate_fact_id"]),
                            )
                            conn.execute(
                                """
                                UPDATE fact_events
                                SET supersedes_item_id=?, updated_at=?
                                WHERE item_id=? AND status='active'
                                """,
                                (
                                    proposal["candidate_fact_id"],
                                    now,
                                    proposal["new_fact_id"],
                                ),
                            )
                        inserted += 1
                        results.append(
                            {
                                "index": index,
                                "status": "inserted",
                                "proposal_id": proposal["proposal_id"],
                            }
                        )
                    except ValueError as exc:
                        results.append({"index": index, "status": "rejected", "error": str(exc)})
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return {
            "ok": True,
            "inserted": inserted,
            "idempotent": idempotent,
            "rejected": sum(item["status"] == "rejected" for item in results),
            "items": results,
        }

    def list_relation_proposals(self, *, status: str = "pending", limit: int = 100) -> dict[str, Any]:
        self._init_db()
        safe_status = str(status or "pending").strip().lower()
        if safe_status != "all" and safe_status not in FACT_RELATION_STATUSES:
            raise ValueError("unsupported proposal status")
        safe_limit = max(1, min(500, int(limit or 100)))
        where = "" if safe_status == "all" else " WHERE p.status=?"
        params: list[Any] = [] if safe_status == "all" else [safe_status]
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"""
                SELECT p.*, n.body AS new_fact_body, c.body AS candidate_fact_body
                FROM fact_relation_proposals p
                JOIN fact_events n ON n.item_id=p.new_fact_id
                JOIN fact_events c ON c.item_id=p.candidate_fact_id
                {where}
                ORDER BY p.created_at DESC, p.proposal_id DESC LIMIT ?
                """,
                [*params, safe_limit],
            ).fetchall()
        return {"items": [{key: row[key] for key in row.keys()} for row in rows]}

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
            payload["source_count"] = len(payload["source_refs"])
        else:
            payload["source_count"] = int(
                payload.get("source_count")
                if "source_count" in payload
                else conn.execute(
                    "SELECT COUNT(*) FROM fact_event_sources WHERE item_id=?",
                    (str(row["item_id"]),),
                ).fetchone()[0]
            )
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
    if len(refs_raw) > 100:
        raise ValueError("at most 100 source refs may support one item")
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
    fact_type = str(raw.get("fact_type") or "").strip().lower() if item_type == "fact" else ""
    atomic_question = (
        str(raw.get("atomic_question") or "").strip()[:240] if item_type == "fact" else ""
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
        "fact_type": fact_type,
        "atomic_question": atomic_question,
        "source_refs": refs,
        **bounds,
    }


def _normalize_relation_proposal(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("each proposal must be an object")
    new_fact_id = _required_identifier(raw.get("new_fact_id"), "new_fact_id", 128)
    candidate_fact_id = _required_identifier(
        raw.get("candidate_fact_id"), "candidate_fact_id", 128
    )
    if new_fact_id == candidate_fact_id:
        raise ValueError("a Fact cannot relate to itself")
    relation = str(raw.get("relation") or "").strip().lower()
    if relation not in FACT_RELATIONS:
        raise ValueError("unsupported Fact relation")
    reason = _required_text(raw.get("reason"), "reason", 800)
    try:
        confidence = float(raw.get("confidence"))
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be between 0 and 1") from exc
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    review_status = str(raw.get("review_status") or "").strip().lower()
    if review_status not in {"accepted", "rejected"}:
        raise ValueError("review_status must be accepted or rejected")
    review_reason = _required_text(raw.get("review_reason"), "review_reason", 800)
    try:
        review_confidence = float(raw.get("review_confidence"))
    except (TypeError, ValueError) as exc:
        raise ValueError("review_confidence must be between 0 and 1") from exc
    if not 0 <= review_confidence <= 1:
        raise ValueError("review_confidence must be between 0 and 1")
    explicit_correction = bool(raw.get("explicit_correction"))
    material = f"{new_fact_id}\n{candidate_fact_id}\n{relation}"
    proposal_id = "factrel_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
    return {
        "proposal_id": proposal_id,
        "new_fact_id": new_fact_id,
        "candidate_fact_id": candidate_fact_id,
        "relation": relation,
        "reason": reason,
        "confidence": confidence,
        "review_status": review_status,
        "review_reason": review_reason,
        "review_confidence": review_confidence,
        "explicit_correction": explicit_correction,
    }


def _legacy_fact_looks_like_meal(body: str) -> bool:
    text = unicodedata.normalize("NFKC", str(body or "")).strip().lower()
    meal_markers = ("早餐", "早饭", "午饭", "午餐", "晚饭", "晚餐", "夜宵", "外卖")
    eating_markers = ("吃了", "吃过", "没吃饭", "没有吃饭")
    return any(marker in text for marker in meal_markers) or any(
        marker in text for marker in eating_markers
    )


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
