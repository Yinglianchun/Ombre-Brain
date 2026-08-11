"""Source evidence bindings for canonical Scenes.

Scene Markdown remains the authored source. This sidecar stores only the
relationship between a Scene and an exact source message (or a separately
addressable snapshot), so binding or reversibly unbinding evidence never
rewrites the Scene or queues an embedding refresh. Source snapshots stay
immutable; binding-state changes are recorded as append-only events.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any


EVIDENCE_KINDS = frozenset({"primary", "supporting", "adjacent_context"})
HASH_ALGORITHM = "sha256-utf8"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def content_sha256(content: str) -> str:
    """Hash the supplied string exactly as UTF-8; do not trim or normalize."""

    if not isinstance(content, str):
        raise ValueError("evidence content must be a string")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def normalize_evidence_ref(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("each evidence ref must be an object")

    source_system = _required_text(raw.get("source_system"), "source_system", 80)
    session_id = _optional_identifier(raw.get("session_id"), 160)
    thread_id = _optional_identifier(raw.get("thread_id"), 200)
    if not session_id and not thread_id:
        raise ValueError("evidence ref requires session_id or thread_id")
    message_id = _required_identifier(raw.get("message_id"), "message_id", 160)
    role = _required_text(raw.get("role"), "role", 40).lower()
    created_at = _required_text(raw.get("created_at"), "created_at", 128)
    binding_method = _required_text(raw.get("binding_method"), "binding_method", 80)
    evidence_kind = str(raw.get("evidence_kind") or "primary").strip().lower()
    if evidence_kind not in EVIDENCE_KINDS:
        raise ValueError(
            "evidence_kind must be one of: " + ", ".join(sorted(EVIDENCE_KINDS))
        )

    has_content = "content" in raw and raw.get("content") is not None
    content = raw.get("content") if has_content else ""
    if has_content and not isinstance(content, str):
        raise ValueError("evidence content must be a string")
    snapshot_ref = str(raw.get("snapshot_ref") or "").strip()[:500]
    supplied_hash = str(raw.get("content_sha256") or "").strip().lower()
    if has_content:
        computed_hash = content_sha256(content)
        if supplied_hash and supplied_hash != computed_hash:
            raise ValueError("content_sha256 does not match exact UTF-8 content")
        supplied_hash = computed_hash
    elif not snapshot_ref:
        raise ValueError("evidence ref requires exact content or snapshot_ref")
    elif not _SHA256_RE.fullmatch(supplied_hash):
        raise ValueError("snapshot-only evidence requires a valid content_sha256")

    return {
        "source_system": source_system,
        "session_id": session_id,
        "thread_id": thread_id,
        "message_id": message_id,
        "role": role,
        "created_at": created_at,
        "content": content,
        "snapshot_ref": snapshot_ref,
        "content_sha256": supplied_hash,
        "hash_algorithm": HASH_ALGORITHM,
        "evidence_kind": evidence_kind,
        "binding_method": binding_method,
    }


class SceneEvidenceStore:
    """SQLite sidecar for immutable Scene-to-source evidence bindings."""

    def __init__(self, config: dict[str, Any] | None = None, *, create: bool = False):
        config = config or {}
        state_dir = str(
            config.get("state_dir")
            or os.path.join(
                os.path.dirname(os.path.abspath(config.get("buckets_dir", "buckets"))),
                "state",
            )
        )
        self.db_path = os.path.join(state_dir, "scene_evidence.sqlite")
        if create:
            self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scene_evidence (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scene_id TEXT NOT NULL,
                    source_system TEXT NOT NULL,
                    session_id TEXT NOT NULL DEFAULT '',
                    thread_id TEXT NOT NULL DEFAULT '',
                    message_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    content TEXT NOT NULL DEFAULT '',
                    snapshot_ref TEXT NOT NULL DEFAULT '',
                    content_sha256 TEXT NOT NULL,
                    hash_algorithm TEXT NOT NULL DEFAULT 'sha256-utf8',
                    evidence_kind TEXT NOT NULL,
                    binding_method TEXT NOT NULL,
                    bound_by TEXT NOT NULL DEFAULT '',
                    bound_at TEXT NOT NULL,
                    UNIQUE(scene_id, source_system, session_id, message_id)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_scene_evidence_scene
                ON scene_evidence(scene_id, created_at, id)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scene_evidence_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    evidence_id INTEGER NOT NULL,
                    scene_id TEXT NOT NULL,
                    action TEXT NOT NULL CHECK(action IN ('bind', 'unbind')),
                    actor TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(evidence_id) REFERENCES scene_evidence(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_scene_evidence_events_latest
                ON scene_evidence_events(evidence_id, id DESC)
                """
            )
            conn.commit()
        finally:
            conn.close()

    def bind(
        self,
        scene_id: str,
        evidence_refs: list[dict[str, Any]],
        *,
        bound_by: str = "",
    ) -> dict[str, Any]:
        safe_scene_id = _required_identifier(scene_id, "scene_id", 128)
        if not isinstance(evidence_refs, list) or not evidence_refs:
            raise ValueError("evidence_refs must be a non-empty list")
        normalized = [normalize_evidence_ref(item) for item in evidence_refs]
        safe_bound_by = str(bound_by or "").strip()[:120]
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._init_db()
        conn = self._connect()
        inserted = 0
        reactivated = 0
        existing_count = 0
        rows: list[sqlite3.Row] = []
        try:
            conn.execute("BEGIN IMMEDIATE")
            for ref in normalized:
                existing = conn.execute(
                    """
                    SELECT * FROM scene_evidence
                    WHERE scene_id=? AND source_system=? AND session_id=? AND message_id=?
                    """,
                    (
                        safe_scene_id,
                        ref["source_system"],
                        ref["session_id"],
                        ref["message_id"],
                    ),
                ).fetchone()
                if existing is not None:
                    if not _same_source_snapshot(existing, ref):
                        raise ValueError(
                            "evidence key already exists with different source content"
                        )
                    if _latest_evidence_action(conn, int(existing["id"])) == "unbind":
                        _record_evidence_event(
                            conn,
                            evidence_id=int(existing["id"]),
                            scene_id=safe_scene_id,
                            action="bind",
                            actor=safe_bound_by,
                            created_at=now,
                        )
                        reactivated += 1
                    else:
                        existing_count += 1
                    continue
                conn.execute(
                    """
                    INSERT INTO scene_evidence (
                        scene_id, source_system, session_id, thread_id, message_id,
                        role, created_at, content, snapshot_ref, content_sha256,
                        hash_algorithm, evidence_kind, binding_method, bound_by, bound_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        safe_scene_id,
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
                        safe_bound_by,
                        now,
                    ),
                )
                inserted += 1
            conn.commit()
            rows = _active_scene_evidence_rows(conn, safe_scene_id)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return {
            "scene_id": safe_scene_id,
            "evidence_status": "bound" if rows else "unbound",
            "bound_count": inserted + reactivated,
            "inserted_count": inserted,
            "reactivated_count": reactivated,
            "existing_count": existing_count,
            "idempotent": inserted == 0 and reactivated == 0,
            "evidence_refs": [_row_to_ref(row) for row in rows],
        }

    def unbind(
        self,
        scene_id: str,
        evidence_ids: list[int | str],
        *,
        unbound_by: str = "",
    ) -> dict[str, Any]:
        """Deactivate exact bindings without deleting their source snapshots."""

        safe_scene_id = _required_identifier(scene_id, "scene_id", 128)
        if not isinstance(evidence_ids, list) or not evidence_ids:
            raise ValueError("evidence_ids must be a non-empty list")
        safe_ids: list[int] = []
        for raw_id in evidence_ids:
            try:
                evidence_id = int(raw_id)
            except (TypeError, ValueError) as exc:
                raise ValueError("each evidence id must be a positive integer") from exc
            if evidence_id <= 0:
                raise ValueError("each evidence id must be a positive integer")
            if evidence_id not in safe_ids:
                safe_ids.append(evidence_id)
        if len(safe_ids) > 50:
            raise ValueError("at most 50 evidence ids can be unbound at once")

        safe_actor = str(unbound_by or "").strip()[:120]
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._init_db()
        conn = self._connect()
        unbound_count = 0
        already_unbound_count = 0
        try:
            conn.execute("BEGIN IMMEDIATE")
            placeholders = ",".join("?" for _ in safe_ids)
            rows = conn.execute(
                f"SELECT id, scene_id FROM scene_evidence WHERE id IN ({placeholders})",
                safe_ids,
            ).fetchall()
            by_id = {int(row["id"]): row for row in rows}
            if len(by_id) != len(safe_ids):
                raise ValueError("one or more evidence ids do not exist")
            if any(str(by_id[evidence_id]["scene_id"]) != safe_scene_id for evidence_id in safe_ids):
                raise ValueError("one or more evidence ids belong to another Scene")

            for evidence_id in safe_ids:
                if _latest_evidence_action(conn, evidence_id) == "unbind":
                    already_unbound_count += 1
                    continue
                _record_evidence_event(
                    conn,
                    evidence_id=evidence_id,
                    scene_id=safe_scene_id,
                    action="unbind",
                    actor=safe_actor,
                    created_at=now,
                )
                unbound_count += 1
            conn.commit()
            active_rows = _active_scene_evidence_rows(conn, safe_scene_id)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return {
            "scene_id": safe_scene_id,
            "evidence_status": "bound" if active_rows else "unbound",
            "unbound_count": unbound_count,
            "already_unbound_count": already_unbound_count,
            "idempotent": unbound_count == 0,
            "evidence_refs": [_row_to_ref(row) for row in active_rows],
        }

    def unbind_all(self, scene_id: str, *, unbound_by: str = "") -> dict[str, Any]:
        """Deactivate every active binding when its Scene is deleted."""

        safe_scene_id = _required_identifier(scene_id, "scene_id", 128)
        if not os.path.exists(self.db_path):
            return {
                "scene_id": safe_scene_id,
                "evidence_status": "unbound",
                "unbound_count": 0,
                "idempotent": True,
                "evidence_refs": [],
            }
        self._init_db()
        conn = self._connect()
        safe_actor = str(unbound_by or "").strip()[:120]
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            conn.execute("BEGIN IMMEDIATE")
            rows = _active_scene_evidence_rows(conn, safe_scene_id)
            for row in rows:
                _record_evidence_event(
                    conn,
                    evidence_id=int(row["id"]),
                    scene_id=safe_scene_id,
                    action="unbind",
                    actor=safe_actor,
                    created_at=now,
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return {
            "scene_id": safe_scene_id,
            "evidence_status": "unbound",
            "unbound_count": len(rows),
            "idempotent": not rows,
            "evidence_refs": [],
        }

    def list_for_scene(self, scene_id: str) -> list[dict[str, Any]]:
        safe_scene_id = _required_identifier(scene_id, "scene_id", 128)
        if not os.path.exists(self.db_path):
            return []
        # Existing production sidecars predate the event table. Migrate before
        # the first read so a restart never depends on a bind/unbind happening
        # first.
        self._init_db()
        conn = self._connect()
        try:
            rows = _active_scene_evidence_rows(conn, safe_scene_id)
        finally:
            conn.close()
        return [_row_to_ref(row) for row in rows]

    def list_active_scene_groups(self) -> dict[str, list[dict[str, Any]]]:
        """Return active evidence grouped by Scene for exact coverage checks."""

        if not os.path.exists(self.db_path):
            return {}
        self._init_db()
        conn = self._connect()
        try:
            scene_rows = conn.execute(
                "SELECT DISTINCT scene_id FROM scene_evidence ORDER BY scene_id"
            ).fetchall()
            return {
                str(row["scene_id"]): [
                    _row_to_ref(ref)
                    for ref in _active_scene_evidence_rows(conn, str(row["scene_id"]))
                ]
                for row in scene_rows
                if str(row["scene_id"])
            }
        finally:
            conn.close()


def _latest_evidence_action(conn: sqlite3.Connection, evidence_id: int) -> str:
    row = conn.execute(
        "SELECT action FROM scene_evidence_events WHERE evidence_id=? ORDER BY id DESC LIMIT 1",
        (evidence_id,),
    ).fetchone()
    return str(row["action"] or "bind") if row is not None else "bind"


def _record_evidence_event(
    conn: sqlite3.Connection,
    *,
    evidence_id: int,
    scene_id: str,
    action: str,
    actor: str,
    created_at: str,
) -> None:
    conn.execute(
        """
        INSERT INTO scene_evidence_events (evidence_id, scene_id, action, actor, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (evidence_id, scene_id, action, actor, created_at),
    )


def _active_scene_evidence_rows(
    conn: sqlite3.Connection,
    scene_id: str,
) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT evidence.*
        FROM scene_evidence AS evidence
        WHERE evidence.scene_id=?
          AND COALESCE(
                (
                    SELECT event.action
                    FROM scene_evidence_events AS event
                    WHERE event.evidence_id=evidence.id
                    ORDER BY event.id DESC
                    LIMIT 1
                ),
                'bind'
              )='bind'
        ORDER BY evidence.created_at, evidence.id
        """,
        (scene_id,),
    ).fetchall()


def _same_source_snapshot(row: sqlite3.Row, ref: dict[str, Any]) -> bool:
    if str(row["content_sha256"] or "") != str(ref["content_sha256"] or ""):
        return False
    if str(row["content"] or "") != str(ref["content"] or ""):
        return False
    return str(row["snapshot_ref"] or "") == str(ref["snapshot_ref"] or "")


def _row_to_ref(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "scene_id": str(row["scene_id"] or ""),
        "source_system": str(row["source_system"] or ""),
        "session_id": str(row["session_id"] or ""),
        "thread_id": str(row["thread_id"] or ""),
        "message_id": str(row["message_id"] or ""),
        "role": str(row["role"] or ""),
        "created_at": str(row["created_at"] or ""),
        "content": str(row["content"] or ""),
        "snapshot_ref": str(row["snapshot_ref"] or ""),
        "content_sha256": str(row["content_sha256"] or ""),
        "hash_algorithm": str(row["hash_algorithm"] or HASH_ALGORITHM),
        "evidence_kind": str(row["evidence_kind"] or ""),
        "binding_method": str(row["binding_method"] or ""),
        "bound_by": str(row["bound_by"] or ""),
        "bound_at": str(row["bound_at"] or ""),
    }


def _required_text(value: Any, field: str, limit: int) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{field} is required")
    return result[:limit]


def _optional_identifier(value: Any, limit: int) -> str:
    if value is None:
        return ""
    return str(value).strip()[:limit]


def _required_identifier(value: Any, field: str, limit: int) -> str:
    result = _optional_identifier(value, limit)
    if not result:
        raise ValueError(f"{field} is required")
    return result
