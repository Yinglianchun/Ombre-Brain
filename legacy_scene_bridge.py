from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any

from memory_evidence import (
    evidence_is_verbatim,
    legacy_moment_content_hash,
    normalize_evidence_text,
    scene_content_hash,
)


SCENE_KIND = "scene"
LEGACY_MOMENT_KIND = "legacy_moment"
BRIDGE_RELATION_TYPES = frozenset(
    {"same_event", "context_of", "updates", "contradicts", "evidenced_by"}
)


class LegacySceneBridgeStore:
    """Reviewed, evidence-bound links between canonical Scenes and legacy Moments."""

    def __init__(self, config: dict, *, create: bool = False):
        state_dir = str(
            config.get("state_dir")
            or os.path.join(
                os.path.dirname(os.path.abspath(config.get("buckets_dir", "buckets"))),
                "state",
            )
        )
        self.db_path = os.path.join(state_dir, "legacy_scene_bridges.sqlite")
        if create:
            os.makedirs(state_dir, exist_ok=True)
            self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS legacy_scene_bridges (
                    bridge_id TEXT PRIMARY KEY,
                    source_kind TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    target_kind TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    directionality TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    reason TEXT NOT NULL,
                    scene_evidence TEXT NOT NULL,
                    legacy_evidence TEXT NOT NULL,
                    scene_hash TEXT NOT NULL,
                    legacy_hash TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    reviewed_by TEXT NOT NULL,
                    reviewed_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_legacy_scene_bridges_active "
                "ON legacy_scene_bridges(active, relation_type, source_kind, source_id)"
            )
            conn.commit()
        finally:
            conn.close()

    def save_reviewed(
        self,
        *,
        scene: dict,
        legacy_moment: dict,
        relation_type: str,
        scene_evidence: str,
        legacy_evidence: str,
        reviewed_by: str,
        confidence: float = 1.0,
        reason: str = "",
        scene_is_source: bool = True,
    ) -> dict:
        scene_id = str((scene or {}).get("id") or "").strip()
        scene_meta = (
            scene.get("metadata", {})
            if isinstance((scene or {}).get("metadata"), dict)
            else {}
        )
        scene_content = str((scene or {}).get("content") or "")
        legacy_moment_id = str((legacy_moment or {}).get("moment_id") or "").strip()
        legacy_text = str((legacy_moment or {}).get("text") or "")
        relation = str(relation_type or "").strip()
        reviewer = str(reviewed_by or "").strip()
        if (
            not scene_id
            or str(scene_meta.get("memory_value_source") or "") != "authored_scene"
            or not scene_content.strip()
        ):
            raise ValueError("canonical authored Scene required")
        if not legacy_moment_id or not legacy_text.strip():
            raise ValueError("legacy Moment required")
        if str((legacy_moment or {}).get("node_kind") or "") == SCENE_KIND:
            raise ValueError("bridge target must be a legacy Moment")
        if relation not in BRIDGE_RELATION_TYPES:
            raise ValueError("unsupported bridge relation")
        if not reviewer:
            raise ValueError("reviewed_by is required")
        if not evidence_is_verbatim(scene_content, scene_evidence):
            raise ValueError("scene evidence is not verbatim")
        if not evidence_is_verbatim(legacy_text, legacy_evidence):
            raise ValueError("legacy evidence is not verbatim")

        if relation == "same_event" or scene_is_source:
            source_kind, source_id = SCENE_KIND, scene_id
            target_kind, target_id = LEGACY_MOMENT_KIND, legacy_moment_id
        else:
            source_kind, source_id = LEGACY_MOMENT_KIND, legacy_moment_id
            target_kind, target_id = SCENE_KIND, scene_id
        directionality = "symmetric" if relation == "same_event" else "directed"
        bridge_id = _bridge_id(source_kind, source_id, target_kind, target_id, relation)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        row = {
            "bridge_id": bridge_id,
            "source_kind": source_kind,
            "source_id": source_id,
            "target_kind": target_kind,
            "target_id": target_id,
            "relation_type": relation,
            "directionality": directionality,
            "confidence": _clamp(confidence),
            "reason": str(reason or "").strip()[:500],
            "scene_evidence": normalize_evidence_text(scene_evidence)[:240],
            "legacy_evidence": normalize_evidence_text(legacy_evidence)[:240],
            "scene_hash": scene_content_hash(scene),
            "legacy_hash": legacy_moment_content_hash(legacy_moment),
            "active": 1,
            "reviewed_by": reviewer,
            "reviewed_at": now,
            "updated_at": now,
        }
        self._init_db()
        conn = self._connect()
        try:
            if relation == "same_event":
                conflict = conn.execute(
                    """
                    SELECT bridge_id, source_id, target_id
                    FROM legacy_scene_bridges
                    WHERE active = 1 AND relation_type = 'same_event'
                      AND ((source_kind = ? AND source_id = ?)
                        OR (target_kind = ? AND target_id = ?))
                      AND bridge_id <> ?
                    LIMIT 1
                    """,
                    (
                        LEGACY_MOMENT_KIND,
                        legacy_moment_id,
                        LEGACY_MOMENT_KIND,
                        legacy_moment_id,
                        bridge_id,
                    ),
                ).fetchone()
                if conflict is not None:
                    raise ValueError("legacy Moment already belongs to another active same_event bridge")
            conn.execute(
                """
                INSERT INTO legacy_scene_bridges (
                    bridge_id, source_kind, source_id, target_kind, target_id,
                    relation_type, directionality, confidence, reason,
                    scene_evidence, legacy_evidence, scene_hash, legacy_hash,
                    active, reviewed_by, reviewed_at, updated_at
                ) VALUES (
                    :bridge_id, :source_kind, :source_id, :target_kind, :target_id,
                    :relation_type, :directionality, :confidence, :reason,
                    :scene_evidence, :legacy_evidence, :scene_hash, :legacy_hash,
                    :active, :reviewed_by, :reviewed_at, :updated_at
                )
                ON CONFLICT(bridge_id) DO UPDATE SET
                    confidence = excluded.confidence,
                    reason = excluded.reason,
                    scene_evidence = excluded.scene_evidence,
                    legacy_evidence = excluded.legacy_evidence,
                    scene_hash = excluded.scene_hash,
                    legacy_hash = excluded.legacy_hash,
                    active = 1,
                    reviewed_by = excluded.reviewed_by,
                    reviewed_at = excluded.reviewed_at,
                    updated_at = excluded.updated_at
                """,
                row,
            )
            conn.commit()
        finally:
            conn.close()
        return dict(row)

    def list_edges(self, *, include_inactive: bool = False) -> list[dict]:
        if not os.path.exists(self.db_path):
            return []
        conn = self._connect()
        try:
            table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                "AND name = 'legacy_scene_bridges'"
            ).fetchone()
            if table is None:
                return []
            where = "" if include_inactive else " WHERE active = 1"
            rows = conn.execute(
                "SELECT * FROM legacy_scene_bridges" + where + " ORDER BY reviewed_at, bridge_id"
            ).fetchall()
        finally:
            conn.close()
        return [dict(row) for row in rows]

    def recall_edges(
        self,
        scene_map: dict[str, dict],
        legacy_moment_map: dict[str, dict],
    ) -> list[dict]:
        valid: list[dict] = []
        for edge in self.list_edges():
            scene_id = _typed_endpoint_id(edge, SCENE_KIND)
            legacy_moment_id = _typed_endpoint_id(edge, LEGACY_MOMENT_KIND)
            scene = scene_map.get(scene_id)
            legacy_moment = legacy_moment_map.get(legacy_moment_id)
            if not isinstance(scene, dict) or not isinstance(legacy_moment, dict):
                continue
            if scene_content_hash(scene) != str(edge.get("scene_hash") or ""):
                continue
            if legacy_moment_content_hash(legacy_moment) != str(edge.get("legacy_hash") or ""):
                continue
            if not evidence_is_verbatim(str(scene.get("content") or ""), edge.get("scene_evidence")):
                continue
            if not evidence_is_verbatim(str(legacy_moment.get("text") or ""), edge.get("legacy_evidence")):
                continue
            payload = dict(edge)
            payload.update(
                {
                    "source": _graph_endpoint_id(edge["source_kind"], edge["source_id"]),
                    "target": _graph_endpoint_id(edge["target_kind"], edge["target_id"]),
                    "scene_id": scene_id,
                    "legacy_moment_id": legacy_moment_id,
                    "graph_scope": "legacy_scene_bridge",
                }
            )
            valid.append(payload)
        return valid

    def deactivate(self, bridge_id: str) -> bool:
        normalized = str(bridge_id or "").strip()
        if not normalized or not os.path.exists(self.db_path):
            return False
        conn = self._connect()
        try:
            cursor = conn.execute(
                "UPDATE legacy_scene_bridges SET active = 0, updated_at = ? WHERE bridge_id = ?",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"), normalized),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()


def _typed_endpoint_id(edge: dict, kind: str) -> str:
    if str(edge.get("source_kind") or "") == kind:
        return str(edge.get("source_id") or "")
    if str(edge.get("target_kind") or "") == kind:
        return str(edge.get("target_id") or "")
    return ""


def _graph_endpoint_id(kind: str, object_id: str) -> str:
    normalized = str(object_id or "").strip()
    return f"scene:{normalized}" if kind == SCENE_KIND else normalized


def _bridge_id(
    source_kind: str,
    source_id: str,
    target_kind: str,
    target_id: str,
    relation_type: str,
) -> str:
    payload = json.dumps(
        [source_kind, source_id, target_kind, target_id, relation_type],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "legacy_scene_bridge_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _clamp(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return max(0.0, min(1.0, round(number, 3)))
