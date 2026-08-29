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
MAX_SOURCE_REFS_PER_ITEM = 4000


class FactEventSettlementBlockedError(ValueError):
    """A settlement precondition drifted; callers must perform zero writes."""


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

                CREATE TABLE IF NOT EXISTS fact_event_replacement_edges (
                    predecessor_id TEXT PRIMARY KEY,
                    successor_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(predecessor_id) REFERENCES fact_events(item_id) ON DELETE RESTRICT,
                    FOREIGN KEY(successor_id) REFERENCES fact_events(item_id) ON DELETE RESTRICT
                );

                CREATE INDEX IF NOT EXISTS idx_fact_event_replacement_edges_successor
                ON fact_event_replacement_edges(successor_id, predecessor_id);

                CREATE TABLE IF NOT EXISTS fact_event_settlement_operations (
                    operation_id TEXT PRIMARY KEY,
                    request_sha256 TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

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
            conn.execute(
                """
                INSERT OR IGNORE INTO fact_event_replacement_edges (
                    predecessor_id, successor_id, created_at
                )
                SELECT child.supersedes_item_id, child.item_id, child.created_at
                FROM fact_events AS child
                JOIN fact_events AS predecessor
                  ON predecessor.item_id=child.supersedes_item_id
                WHERE child.supersedes_item_id<>''
                  AND (
                      SELECT COUNT(*)
                      FROM fact_events AS sibling
                      WHERE sibling.supersedes_item_id=child.supersedes_item_id
                  )=1
                """
            )
            conn.commit()

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

    def settle(self, operation_id: Any, raw_items: Any) -> dict[str, Any]:
        """Atomically create and replace a mixed batch with exact predecessor receipts."""

        safe_operation_id = _required_identifier(operation_id, "operation_id", 200)
        if not isinstance(raw_items, list) or not raw_items:
            raise ValueError("items must be a non-empty list")
        if len(raw_items) > 500:
            raise ValueError("at most 500 items may be settled in one batch")

        request_sha256 = self._settlement_request_sha256(raw_items)
        prepared: list[tuple[int, dict[str, Any], list[str], dict[str, dict[str, Any]]]] = []
        predecessor_ids: set[str] = set()
        origin_ids: set[str] = set()
        fingerprints: set[str] = set()
        for index, raw in enumerate(raw_items):
            if not isinstance(raw, dict):
                raise ValueError("each item must be an object")
            raw_predecessors = raw.get("supersedes_item_ids")
            predecessors = (
                _normalize_ids(raw_predecessors, "supersedes_item_ids", limit=100)
                if raw_predecessors is not None
                else []
            )
            if raw_predecessors is not None and not predecessors:
                raise ValueError("supersedes_item_ids must be non-empty when present")
            if predecessor_ids.intersection(predecessors):
                raise ValueError("a predecessor may appear in only one settlement item")
            predecessor_ids.update(predecessors)

            raw_expected = raw.get("expected_predecessors")
            if predecessors:
                expected = _normalize_expected_predecessors(raw_expected)
                if set(expected) != set(predecessors):
                    raise ValueError(
                        "expected_predecessors must exactly match supersedes_item_ids"
                    )
            elif raw_expected not in (None, []):
                raise ValueError("create items must not carry expected_predecessors")
            else:
                expected = {}

            normalized = _normalize_item(
                {
                    **raw,
                    "supersedes_item_id": predecessors[0] if predecessors else "",
                }
            )
            origin_id = str(normalized["origin_id"] or "")
            if origin_id and origin_id in origin_ids:
                raise ValueError("duplicate origin_id values are not allowed")
            if normalized["fingerprint"] in fingerprints:
                raise ValueError("duplicate settlement fingerprints are not allowed")
            if origin_id:
                origin_ids.add(origin_id)
            fingerprints.add(normalized["fingerprint"])
            prepared.append((index, normalized, predecessors, expected))

        self._init_db()
        now = _now()
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                previous_operation = conn.execute(
                    """
                    SELECT request_sha256, result_json
                    FROM fact_event_settlement_operations WHERE operation_id=?
                    """,
                    (safe_operation_id,),
                ).fetchone()
                if previous_operation is not None:
                    if str(previous_operation["request_sha256"]) != request_sha256:
                        raise FactEventSettlementBlockedError(
                            "operation_id already exists with a different request"
                        )
                    stored = json.loads(str(previous_operation["result_json"]))
                    stored = self._settlement_replay(stored)
                    conn.rollback()
                    return stored

                for _, item, predecessors, expected in prepared:
                    existing = None
                    if item["origin_id"]:
                        existing = conn.execute(
                            "SELECT item_id, fingerprint FROM fact_events WHERE origin_id=?",
                            (item["origin_id"],),
                        ).fetchone()
                    if existing is None:
                        existing = conn.execute(
                            "SELECT item_id, fingerprint FROM fact_events WHERE fingerprint=?",
                            (item["fingerprint"],),
                        ).fetchone()
                    if existing is not None:
                        raise FactEventSettlementBlockedError(
                            "settlement item already exists outside this operation"
                        )

                    replacement_source_keys = {
                        _source_key(ref) for ref in item["source_refs"]
                    }
                    for predecessor_id in predecessors:
                        family = self._replacement_family_payload(conn, predecessor_id)
                        if not family["ok"]:
                            raise FactEventSettlementBlockedError(
                                "replacement family is invalid for "
                                f"{predecessor_id}: " + ",".join(family["issues"])
                            )
                        if not family["is_exact_active_leaf"]:
                            raise FactEventSettlementBlockedError(
                                f"predecessor is not its exact active leaf: {predecessor_id}"
                            )
                        family_ids = list(family.get("family_ids") or [])
                        family_rows = conn.execute(
                            f"""
                            SELECT item_id, origin_id
                            FROM fact_events
                            WHERE item_id IN ({','.join('?' for _ in family_ids)})
                            """,
                            family_ids,
                        ).fetchall()
                        untrusted_family_ids = sorted(
                            str(row["item_id"])
                            for row in family_rows
                            if not str(row["origin_id"] or "").startswith("haven_bridge:")
                        )
                        if untrusted_family_ids:
                            raise FactEventSettlementBlockedError(
                                "manual_or_untrusted_provenance: "
                                + ",".join(untrusted_family_ids)
                            )
                        predecessor = conn.execute(
                            """
                            SELECT item_type, status, fingerprint
                            FROM fact_events WHERE item_id=?
                            """,
                            (predecessor_id,),
                        ).fetchone()
                        if predecessor is None:
                            raise FactEventSettlementBlockedError(
                                f"predecessor not found: {predecessor_id}"
                            )
                        if str(predecessor["item_type"]) != item["item_type"]:
                            raise FactEventSettlementBlockedError(
                                "predecessor item type mismatch"
                            )
                        receipt = expected[predecessor_id]
                        if str(predecessor["fingerprint"]) != receipt["fingerprint"]:
                            raise FactEventSettlementBlockedError(
                                f"predecessor fingerprint drifted: {predecessor_id}"
                            )
                        source_rows = conn.execute(
                            """
                            SELECT source_system, session_id, message_id
                            FROM fact_event_sources WHERE item_id=?
                            """,
                            (predecessor_id,),
                        ).fetchall()
                        actual_source_keys = {
                            (
                                str(row["source_system"]),
                                str(row["session_id"]),
                                str(row["message_id"]),
                            )
                            for row in source_rows
                        }
                        if actual_source_keys != receipt["source_keys"]:
                            raise FactEventSettlementBlockedError(
                                f"predecessor source keys drifted: {predecessor_id}"
                            )
                        if not actual_source_keys.issubset(replacement_source_keys):
                            raise FactEventSettlementBlockedError(
                                f"replacement drops predecessor sources: {predecessor_id}"
                            )

                results: list[dict[str, Any]] = []
                for index, item, predecessors, _ in prepared:
                    written = self._write_one(conn, item, now=now)
                    if written.get("status") != "inserted":
                        raise ValueError(written.get("error") or "settlement item was not inserted")
                    item_id = str(written["item_id"])
                    for predecessor_id in predecessors:
                        conn.execute(
                            """
                            INSERT INTO fact_event_replacement_edges (
                                predecessor_id, successor_id, created_at
                            ) VALUES (?, ?, ?)
                            """,
                            (predecessor_id, item_id, now),
                        )
                    for predecessor_id in predecessors[1:]:
                        changed = conn.execute(
                            """
                            UPDATE fact_events SET status='superseded', updated_at=?
                            WHERE item_id=? AND status='active'
                            """,
                            (now, predecessor_id),
                        )
                        if changed.rowcount != 1:
                            raise FactEventSettlementBlockedError(
                                "predecessor status changed during settlement"
                            )
                    results.append(
                        {
                            "index": index,
                            "status": "inserted",
                            "item_id": item_id,
                            "origin_id": item["origin_id"],
                            "fingerprint": item["fingerprint"],
                            "superseded_item_ids": predecessors,
                            "source_refs": item["source_refs"],
                        }
                    )

                result = {
                    "ok": True,
                    "operation_id": safe_operation_id,
                    "replayed": False,
                    "inserted": len(results),
                    "idempotent": 0,
                    "items": results,
                }
                conn.execute(
                    """
                    INSERT INTO fact_event_settlement_operations (
                        operation_id, request_sha256, result_json, created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        safe_operation_id,
                        request_sha256,
                        json.dumps(
                            result,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        now,
                    ),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return result

    def settlement_receipt(self, operation_id: Any, raw_items: Any) -> dict[str, Any] | None:
        """Return an exact committed operation replay without touching canonical rows."""

        safe_operation_id = _required_identifier(operation_id, "operation_id", 200)
        if not isinstance(raw_items, list) or not raw_items:
            raise ValueError("items must be a non-empty list")
        if not os.path.exists(self.db_path):
            return None
        request_sha256 = self._settlement_request_sha256(raw_items)
        self._init_db()
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT request_sha256, result_json
                FROM fact_event_settlement_operations WHERE operation_id=?
                """,
                (safe_operation_id,),
            ).fetchone()
        if row is None:
            return None
        if str(row["request_sha256"]) != request_sha256:
            raise FactEventSettlementBlockedError(
                "operation_id already exists with a different request"
            )
        result = json.loads(str(row["result_json"]))
        return self._settlement_replay(result)

    @staticmethod
    def _settlement_request_sha256(raw_items: list[Any]) -> str:
        return hashlib.sha256(
            json.dumps(
                raw_items,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _settlement_replay(stored: dict[str, Any]) -> dict[str, Any]:
        result = dict(stored)
        replay_items = [
            {**item, "status": "idempotent"}
            for item in result.get("items") or []
            if isinstance(item, dict)
        ]
        result["items"] = replay_items
        result["inserted"] = 0
        result["idempotent"] = len(replay_items)
        result["replayed"] = True
        return result

    def read_many(
        self,
        item_ids: Any,
        *,
        include_sources: bool = True,
        resolve_active_successors: bool = False,
    ) -> dict[str, Any]:
        """Read exact canonical IDs without search or successor substitution."""

        ids = _normalize_ids(item_ids, "item_ids", limit=500)
        if not ids or not os.path.exists(self.db_path):
            return {
                "ok": True,
                "items": [],
                "missing_item_ids": ids,
                "families": [],
                "resolutions": [],
                "resolved_items": [],
            }
        self._init_db()
        with closing(self._connect()) as conn:
            rows = {
                str(row["item_id"]): row
                for row in conn.execute(
                    f"SELECT * FROM fact_events WHERE item_id IN ({','.join('?' for _ in ids)})",
                    ids,
                ).fetchall()
            }
            found_ids = [item_id for item_id in ids if item_id in rows]
            families = [
                self._replacement_family_payload(conn, item_id)
                for item_id in found_ids
            ]
            resolved_ids = list(
                dict.fromkeys(
                    str(active_leaf_id or "")
                    for family in families
                    if resolve_active_successors
                    for active_leaf_id in family.get("family_active_leaves") or []
                    if str(active_leaf_id or "")
                )
            )
            resolved_rows = {
                str(row["item_id"]): row
                for row in (
                    conn.execute(
                        f"SELECT * FROM fact_events WHERE item_id IN ({','.join('?' for _ in resolved_ids)})",
                        resolved_ids,
                    ).fetchall()
                    if resolved_ids
                    else []
                )
            }
            return {
                "ok": True,
                "items": [
                    self._row_payload(conn, rows[item_id], include_sources=include_sources)
                    for item_id in found_ids
                ],
                "missing_item_ids": [item_id for item_id in ids if item_id not in rows],
                "families": families,
                "resolutions": [
                    {
                        "requested_item_id": str(family.get("item_id") or ""),
                        "resolved_item_id": (
                            str(family.get("active_leaf_id") or "")
                            if family.get("ok")
                            else ""
                        ),
                        "family_active_leaves": list(
                            family.get("family_active_leaves") or []
                        ),
                        "predecessor_event_ids": list(
                            family.get("predecessor_event_ids") or []
                        ),
                        "lineage": list(family.get("edges") or []),
                        "forked": bool(family.get("forked")),
                        "blocked": not bool(family.get("ok")),
                        "issues": list(family.get("issues") or []),
                    }
                    for family in families
                ],
                "resolved_items": [
                    self._row_payload(
                        conn,
                        resolved_rows[item_id],
                        include_sources=include_sources,
                    )
                    for item_id in resolved_ids
                    if item_id in resolved_rows
                ],
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
                    for predecessor_id in predecessors:
                        family = self._replacement_family_payload(conn, predecessor_id)
                        if not family["ok"]:
                            raise ValueError(
                                "replacement family is invalid: "
                                + ",".join(family["issues"])
                            )
                        if not family["is_exact_active_leaf"]:
                            raise ValueError(
                                f"supersedes item is not its exact active leaf: {predecessor_id}"
                            )

                results: list[dict[str, Any]] = []
                for index, item, predecessors in prepared:
                    written = self._write_one(conn, item, now=now)
                    if written.get("status") != "inserted":
                        raise ValueError(written.get("error") or "replacement was not inserted")
                    successor_id = str(written["item_id"])
                    for predecessor_id in predecessors:
                        conn.execute(
                            """
                            INSERT INTO fact_event_replacement_edges (
                                predecessor_id, successor_id, created_at
                            ) VALUES (?, ?, ?)
                            """,
                            (predecessor_id, successor_id, now),
                        )
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

    def replacement_family(self, item_id: str) -> dict[str, Any]:
        """Resolve one complete replacement family and its unique exact active leaf."""

        safe_id = _required_identifier(item_id, "item_id", 128)
        if not os.path.exists(self.db_path):
            return {
                "ok": False,
                "item_id": safe_id,
                "family_ids": [],
                "active_leaf_id": "",
                "is_exact_active_leaf": False,
                "family_active_leaves": [],
                "predecessor_event_ids": [],
                "forked": False,
                "issues": ["item_not_found"],
            }
        self._init_db()
        with closing(self._connect()) as conn:
            return self._replacement_family_payload(conn, safe_id)

    @staticmethod
    def _replacement_family_payload(
        conn: sqlite3.Connection,
        item_id: str,
    ) -> dict[str, Any]:
        rows = conn.execute(
            "SELECT item_id, status, supersedes_item_id FROM fact_events"
        ).fetchall()
        by_id = {str(row["item_id"]): row for row in rows}
        if item_id not in by_id:
            return {
                "ok": False,
                "item_id": item_id,
                "family_ids": [],
                "active_leaf_id": "",
                "is_exact_active_leaf": False,
                "family_active_leaves": [],
                "predecessor_event_ids": [],
                "forked": False,
                "issues": ["item_not_found"],
            }

        edge_rows = conn.execute(
            "SELECT predecessor_id, successor_id FROM fact_event_replacement_edges"
        ).fetchall()
        edge_pairs = {
            (str(row["predecessor_id"]), str(row["successor_id"]))
            for row in edge_rows
        }
        edge_pairs.update(
            (str(row["supersedes_item_id"]), str(row["item_id"]))
            for row in rows
            if str(row["supersedes_item_id"] or "")
        )

        issues: set[str] = set()
        outgoing: dict[str, set[str]] = {}
        neighbors: dict[str, set[str]] = {}
        for predecessor_id, successor_id in edge_pairs:
            if predecessor_id not in by_id or successor_id not in by_id:
                issues.add("missing_family_member")
                continue
            outgoing.setdefault(predecessor_id, set()).add(successor_id)
            neighbors.setdefault(predecessor_id, set()).add(successor_id)
            neighbors.setdefault(successor_id, set()).add(predecessor_id)
        if any(len(successors) > 1 for successors in outgoing.values()):
            issues.add("forked_successor_family")

        family = {item_id}
        pending = [item_id]
        while pending:
            current = pending.pop()
            for neighbor in neighbors.get(current, set()):
                if neighbor not in family:
                    family.add(neighbor)
                    pending.append(neighbor)

        visiting: set[str] = set()
        visited: set[str] = set()

        def _visit(node_id: str) -> None:
            if node_id in visiting:
                issues.add("cyclic_successor_family")
                return
            if node_id in visited:
                return
            visiting.add(node_id)
            for successor_id in outgoing.get(node_id, set()):
                if successor_id in family:
                    _visit(successor_id)
            visiting.remove(node_id)
            visited.add(node_id)

        for family_id in family:
            _visit(family_id)

        graph_leaves = sorted(
            family_id for family_id in family if not outgoing.get(family_id)
        )
        active_leaves = sorted(
            family_id
            for family_id in graph_leaves
            if str(by_id[family_id]["status"]) == "active"
        )
        if len(graph_leaves) != 1:
            issues.add("multiple_family_leaves")
        if len(active_leaves) == 0:
            issues.add("no_active_leaf")
        elif len(active_leaves) > 1:
            issues.add("multiple_active_leaves")

        active_leaf_id = active_leaves[0] if len(active_leaves) == 1 else ""
        return {
            "ok": not issues,
            "item_id": item_id,
            "family_ids": sorted(family),
            "active_leaf_id": active_leaf_id,
            "is_exact_active_leaf": active_leaf_id == item_id and not issues,
            "family_active_leaves": active_leaves,
            "predecessor_event_ids": sorted(
                family_id for family_id in family if outgoing.get(family_id)
            ),
            "forked": (
                "forked_successor_family" in issues
                or "multiple_family_leaves" in issues
                or "multiple_active_leaves" in issues
            ),
            "edges": [
                {"predecessor_id": predecessor_id, "successor_id": successor_id}
                for predecessor_id, successor_id in sorted(edge_pairs)
                if predecessor_id in family and successor_id in family
            ],
            "issues": sorted(issues),
        }

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
                    edge_rows = conn.execute(
                        """
                        SELECT predecessor_id, successor_id
                        FROM fact_event_replacement_edges
                        """
                    ).fetchall()
                    for row in edge_rows:
                        predecessor = str(row["predecessor_id"])
                        successor = str(row["successor_id"])
                        if successor in family and predecessor not in family:
                            family.add(predecessor)
                            changed = True
                        if predecessor in family and successor not in family:
                            family.add(successor)
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
                    f"""
                    DELETE FROM fact_event_replacement_edges
                    WHERE predecessor_id IN ({placeholders})
                       OR successor_id IN ({placeholders})
                    """,
                    [*item_ids, *item_ids],
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
    if len(refs_raw) > MAX_SOURCE_REFS_PER_ITEM:
        raise ValueError(
            f"at most {MAX_SOURCE_REFS_PER_ITEM} source refs may support one item"
        )
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


def _normalize_expected_predecessors(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("expected_predecessors must be a non-empty list")
    if len(raw) > 100:
        raise ValueError("expected_predecessors may contain at most 100 values")
    result: dict[str, dict[str, Any]] = {}
    for value in raw:
        if not isinstance(value, dict):
            raise ValueError("each expected predecessor must be an object")
        item_id = _required_identifier(value.get("item_id"), "expected predecessor item_id", 128)
        if item_id in result:
            raise ValueError("duplicate expected predecessor item_id")
        fingerprint = str(value.get("fingerprint") or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            raise ValueError("expected predecessor fingerprint must be SHA-256")
        raw_source_keys = value.get("source_keys")
        if not isinstance(raw_source_keys, list) or not raw_source_keys:
            raise ValueError("expected predecessor source_keys must be a non-empty list")
        if len(raw_source_keys) > MAX_SOURCE_REFS_PER_ITEM:
            raise ValueError(
                "expected predecessor source_keys may contain at most "
                f"{MAX_SOURCE_REFS_PER_ITEM} values"
            )
        source_keys: set[tuple[str, str, str]] = set()
        for raw_key in raw_source_keys:
            if not isinstance(raw_key, dict):
                raise ValueError("each expected predecessor source key must be an object")
            source_system = _required_text(
                raw_key.get("source_system"),
                "expected source_system",
                80,
            )
            session_id = str(raw_key.get("session_id") or "").strip()
            message_id = _required_text(
                raw_key.get("message_id"),
                "expected message_id",
                200,
            )
            key = (source_system, session_id, message_id)
            if key in source_keys:
                raise ValueError("duplicate expected predecessor source key")
            source_keys.add(key)
        result[item_id] = {
            "fingerprint": fingerprint,
            "source_keys": source_keys,
        }
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
