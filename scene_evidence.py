"""Append-only source evidence bindings for canonical Scenes.

Scene Markdown remains the authored source. This sidecar stores only the
relationship between a Scene and an exact source message (or a separately
addressable snapshot), so binding evidence never rewrites the Scene or queues
an embedding refresh.
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
            rows = conn.execute(
                "SELECT * FROM scene_evidence WHERE scene_id=? ORDER BY created_at, id",
                (safe_scene_id,),
            ).fetchall()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return {
            "scene_id": safe_scene_id,
            "evidence_status": "bound" if rows else "unbound",
            "bound_count": inserted,
            "existing_count": existing_count,
            "idempotent": inserted == 0,
            "evidence_refs": [_row_to_ref(row) for row in rows],
        }

    def list_for_scene(self, scene_id: str) -> list[dict[str, Any]]:
        safe_scene_id = _required_identifier(scene_id, "scene_id", 128)
        if not os.path.exists(self.db_path):
            return []
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM scene_evidence WHERE scene_id=? ORDER BY created_at, id",
                (safe_scene_id,),
            ).fetchall()
        finally:
            conn.close()
        return [_row_to_ref(row) for row in rows]


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
