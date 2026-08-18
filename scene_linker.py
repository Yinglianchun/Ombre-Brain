from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any

from openai import AsyncOpenAI

from self_anchor import is_self_anchor_bucket


logger = logging.getLogger("ombre_brain.scene_linker")

SCENE_LINKER_VERSION = "scene-linker-v1"
SCENE_EDGE_PROPOSAL_ID_RE = re.compile(r"^scene_edge_[0-9a-f]{24}$")
SCENE_EDGE_PROPOSAL_STATUSES = frozenset({"pending", "accepted", "rejected", "superseded"})
SCENE_EDGE_LIFECYCLE_STATUSES = frozenset(
    {"active", "needs_review", "cancelled", "archived", "replaced"}
)
SCENE_EDGE_PROPOSAL_ORIGINS = frozenset(
    {"automatic", "manual", "snapshot_revalidation", "manual_relink"}
)
SCENE_EDGE_PROPOSAL_ORIGIN_PRIORITY = {
    "automatic": 0,
    "snapshot_revalidation": 1,
    "manual": 2,
    "manual_relink": 2,
}
SCENE_EDGE_REVIEW_CONFIRMATIONS = {
    "accept": "ACCEPT_SCENE_EDGE",
    "reject": "REJECT_SCENE_EDGE",
}
SCENE_RELATION_TYPES = frozenset(
    {
        "continues",
        "echoes",
        "resolves",
        "contrasts_with",
        "evidenced_by",
    }
)
SYMMETRIC_SCENE_RELATIONS = frozenset({"echoes", "contrasts_with"})
DIRECTED_ORIENTATIONS = frozenset({"candidate_to_new", "new_to_candidate"})


def _scene_edge_id(source_scene_id: str, target_scene_id: str, relation_type: str) -> str:
    source = str(source_scene_id or "").strip()
    target = str(target_scene_id or "").strip()
    relation = str(relation_type or "").strip()
    if relation in SYMMETRIC_SCENE_RELATIONS and target < source:
        source, target = target, source
    identity = json.dumps(
        {
            "source": source,
            "target": target,
            "relation": relation,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "scene_rel_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def _scene_edge_snapshot_hashes(
    source_scene_id: str,
    target_scene_id: str,
    relation_type: str,
    anchor_scene_id: str,
    anchor_hash: str,
    candidate_hash: str,
) -> tuple[str, str]:
    source = str(source_scene_id or "").strip()
    target = str(target_scene_id or "").strip()
    relation = str(relation_type or "").strip()
    anchor = str(anchor_scene_id or "").strip()
    source_hash = str(anchor_hash if source == anchor else candidate_hash)
    target_hash = str(anchor_hash if target == anchor else candidate_hash)
    if relation in SYMMETRIC_SCENE_RELATIONS and target < source:
        source_hash, target_hash = target_hash, source_hash
    return source_hash, target_hash


def _ensure_scene_edge_schema(conn: sqlite3.Connection) -> None:
    """Keep reviewed Scene relations separate from the legacy JSONL graph."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scene_edges (
            edge_id TEXT PRIMARY KEY,
            source_scene_id TEXT NOT NULL,
            target_scene_id TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            directionality TEXT NOT NULL,
            confidence REAL NOT NULL,
            reason TEXT NOT NULL,
            source_evidence TEXT NOT NULL,
            target_evidence TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            target_hash TEXT NOT NULL,
            proposal_id TEXT NOT NULL UNIQUE,
            linker_version TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            accepted_at TEXT NOT NULL,
            accepted_by TEXT,
            deactivated_at TEXT,
            deactivated_by TEXT,
            deactivation_reason TEXT,
            lifecycle_status TEXT NOT NULL DEFAULT 'active',
            restored_at TEXT,
            restored_by TEXT,
            supersedes_edge_id TEXT,
            replaced_by_edge_id TEXT,
            updated_at TEXT NOT NULL
        )
        """
    )
    columns = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(scene_edges)").fetchall()
    }
    text_columns = {
        "deactivated_at",
        "deactivated_by",
        "deactivation_reason",
        "restored_at",
        "restored_by",
        "supersedes_edge_id",
        "replaced_by_edge_id",
    }
    for name in text_columns:
        if name not in columns:
            conn.execute(f"ALTER TABLE scene_edges ADD COLUMN {name} TEXT")
    if "lifecycle_status" not in columns:
        conn.execute(
            "ALTER TABLE scene_edges ADD COLUMN lifecycle_status TEXT NOT NULL DEFAULT 'active'"
        )
    conn.execute(
        """
        UPDATE scene_edges
           SET lifecycle_status = CASE
               WHEN deactivation_reason = 'scene_archived' THEN 'archived'
               WHEN deactivation_reason = 'scene_content_changed' THEN 'needs_review'
               WHEN deactivation_reason = 'relinked' THEN 'replaced'
               ELSE 'cancelled'
           END
         WHERE active = 0 AND lifecycle_status = 'active'
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_scene_edges_active_source "
        "ON scene_edges(active, source_scene_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_scene_edges_active_target "
        "ON scene_edges(active, target_scene_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_scene_edges_lifecycle "
        "ON scene_edges(lifecycle_status, updated_at DESC)"
    )


class SceneEdgeStore:
    """Reviewed Scene-only graph stored beside proposals, never in legacy JSONL."""

    def __init__(self, config: dict, *, create: bool = True):
        self.db_path = SceneEdgeProposalStore.path_for(config)
        if create or os.path.exists(self.db_path):
            if create:
                os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            conn = self._connect()
            conn.execute("PRAGMA journal_mode=WAL")
            _ensure_scene_edge_schema(conn)
            conn.commit()
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def list_edges(self, *, include_inactive: bool = False) -> list[dict]:
        if not os.path.exists(self.db_path):
            return []
        conn = self._connect()
        try:
            table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'scene_edges'"
            ).fetchone()
            if table is None:
                return []
            where = "" if include_inactive else " WHERE active = 1"
            rows = conn.execute(
                "SELECT * FROM scene_edges" + where + " ORDER BY accepted_at DESC, edge_id"
            ).fetchall()
        finally:
            conn.close()
        return [self._edge_payload(dict(row)) for row in rows]

    def recall_edges(self, scene_map: dict[str, dict]) -> list[dict]:
        """Return only formal edges still grounded in both current Scene bodies."""
        valid: list[dict] = []
        for edge in self.list_edges():
            source = scene_map.get(str(edge.get("source") or ""))
            target = scene_map.get(str(edge.get("target") or ""))
            if not _is_authored_scene(source) or not _is_authored_scene(target):
                continue
            if str(edge.get("source_hash") or "") != _scene_hash(source):
                continue
            if str(edge.get("target_hash") or "") != _scene_hash(target):
                continue
            source_ok, _ = _evidence_is_verbatim(
                str(source.get("content") or ""), edge.get("source_evidence")
            )
            target_ok, _ = _evidence_is_verbatim(
                str(target.get("content") or ""), edge.get("target_evidence")
            )
            if source_ok and target_ok:
                valid.append(edge)
        return valid

    def stats(self) -> dict:
        if not os.path.exists(self.db_path):
            return {"total": 0, "active": 0, "inactive": 0}
        conn = self._connect()
        try:
            table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'scene_edges'"
            ).fetchone()
            if table is None:
                return {"total": 0, "active": 0, "inactive": 0}
            rows = conn.execute(
                """
                SELECT active, lifecycle_status, COUNT(*) AS count
                  FROM scene_edges
                 GROUP BY active, lifecycle_status
                """
            ).fetchall()
        finally:
            conn.close()
        counts: dict[int, int] = {}
        lifecycle: dict[str, int] = {}
        for row in rows:
            counts[int(row["active"])] = counts.get(int(row["active"]), 0) + int(row["count"])
            state = str(row["lifecycle_status"] or ("active" if row["active"] else "cancelled"))
            lifecycle[state] = lifecycle.get(state, 0) + int(row["count"])
        return {
            "total": sum(counts.values()),
            "active": counts.get(1, 0),
            "inactive": counts.get(0, 0),
            "needs_review": lifecycle.get("needs_review", 0),
            "cancelled": lifecycle.get("cancelled", 0),
            "archived": lifecycle.get("archived", 0),
            "replaced": lifecycle.get("replaced", 0),
            "lifecycle": lifecycle,
        }

    def deactivate_for_scene(
        self,
        scene_id: str,
        *,
        reviewer: str = "system",
        reason: str = "scene_deleted",
        lifecycle_status: str = "cancelled",
    ) -> int:
        normalized = str(scene_id or "").strip()
        if not normalized or not os.path.exists(self.db_path):
            return 0
        normalized_status = str(lifecycle_status or "cancelled").strip().lower()
        if normalized_status not in SCENE_EDGE_LIFECYCLE_STATUSES - {"active"}:
            raise ValueError("invalid Scene edge lifecycle status")
        conn = self._connect()
        try:
            table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'scene_edges'"
            ).fetchone()
            if table is None:
                return 0
            cursor = conn.execute(
                """
                UPDATE scene_edges
                   SET active = 0,
                       lifecycle_status = ?,
                       deactivated_at = ?,
                       deactivated_by = ?,
                       deactivation_reason = ?,
                       updated_at = ?
                 WHERE active = 1 AND (source_scene_id = ? OR target_scene_id = ?)
                """,
                (
                    normalized_status,
                    _now_utc(),
                    str(reviewer or "system").strip() or "system",
                    str(reason or "scene_deleted").strip() or "scene_deleted",
                    _now_utc(),
                    normalized,
                    normalized,
                ),
            )
            conn.commit()
            return max(0, int(cursor.rowcount or 0))
        finally:
            conn.close()

    def deactivate_edge(
        self,
        edge_id: str,
        *,
        scene_id: str,
        reviewer: str = "",
        reason: str = "manual_remove",
        lifecycle_status: str = "cancelled",
    ) -> dict:
        """Soft-disable one exact reviewed edge while preserving its audit row."""
        normalized_edge = str(edge_id or "").strip()
        normalized_scene = str(scene_id or "").strip()
        normalized_status = str(lifecycle_status or "cancelled").strip().lower()
        if normalized_status not in SCENE_EDGE_LIFECYCLE_STATUSES - {"active"}:
            raise ValueError("invalid Scene edge lifecycle status")
        if not normalized_edge or not normalized_scene or not os.path.exists(self.db_path):
            return {"status": "not_found", "edge_id": normalized_edge}
        conn = self._connect()
        try:
            _ensure_scene_edge_schema(conn)
            row = conn.execute(
                "SELECT * FROM scene_edges WHERE edge_id = ?",
                (normalized_edge,),
            ).fetchone()
            if row is None:
                return {"status": "not_found", "edge_id": normalized_edge}
            edge = self._edge_payload(dict(row))
            if normalized_scene not in {edge["source"], edge["target"]}:
                return {
                    "status": "invalid",
                    "reason": "scene_not_edge_endpoint",
                    "edge": edge,
                }
            if not edge["active"]:
                return {"status": "unchanged", "edge": edge}
            now = _now_utc()
            cursor = conn.execute(
                """
                UPDATE scene_edges
                   SET active = 0,
                       lifecycle_status = ?,
                       deactivated_at = ?,
                       deactivated_by = ?,
                       deactivation_reason = ?,
                       updated_at = ?
                 WHERE edge_id = ? AND active = 1
                """,
                (
                    normalized_status,
                    now,
                    str(reviewer or "").strip() or None,
                    str(reason or "manual_remove").strip() or "manual_remove",
                    now,
                    normalized_edge,
                ),
            )
            conn.commit()
            updated = conn.execute(
                "SELECT * FROM scene_edges WHERE edge_id = ?",
                (normalized_edge,),
            ).fetchone()
            return {
                "status": "deactivated" if int(cursor.rowcount or 0) == 1 else "unchanged",
                "edge": self._edge_payload(dict(updated)),
            }
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def restore_edge(
        self,
        edge_id: str,
        scene_map: dict[str, dict],
        *,
        reviewer: str = "",
    ) -> dict:
        """Restore one inactive edge only when both current Scenes still support it."""
        normalized = str(edge_id or "").strip()
        if not normalized or not os.path.exists(self.db_path):
            return {"status": "not_found", "edge_id": normalized}
        conn = self._connect()
        try:
            _ensure_scene_edge_schema(conn)
            row = conn.execute(
                "SELECT * FROM scene_edges WHERE edge_id = ?",
                (normalized,),
            ).fetchone()
            if row is None:
                return {"status": "not_found", "edge_id": normalized}
            edge = self._edge_payload(dict(row))
            if edge["active"]:
                return {"status": "unchanged", "edge": edge}
            source = scene_map.get(edge["source"])
            target = scene_map.get(edge["target"])
            validation_error = self._current_edge_error(edge, source, target)
            if validation_error:
                return {
                    "status": "stale",
                    "reason": validation_error,
                    "edge": edge,
                }
            now = _now_utc()
            conn.execute(
                """
                UPDATE scene_edges
                   SET active = 1,
                       lifecycle_status = 'active',
                       restored_at = ?,
                       restored_by = ?,
                       deactivated_at = NULL,
                       deactivated_by = NULL,
                       deactivation_reason = NULL,
                       updated_at = ?
                 WHERE edge_id = ? AND active = 0
                """,
                (now, str(reviewer or "").strip() or None, now, normalized),
            )
            conn.commit()
            updated = conn.execute(
                "SELECT * FROM scene_edges WHERE edge_id = ?",
                (normalized,),
            ).fetchone()
            return {"status": "restored", "edge": self._edge_payload(dict(updated))}
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _current_edge_error(edge: dict, source: dict | None, target: dict | None) -> str:
        if not _is_authored_scene(source):
            return "source_scene_inactive"
        if not _is_authored_scene(target):
            return "target_scene_inactive"
        if str(edge.get("source_hash") or "") != _scene_hash(source):
            return "source_scene_changed"
        if str(edge.get("target_hash") or "") != _scene_hash(target):
            return "target_scene_changed"
        source_ok, _ = _evidence_is_verbatim(
            str((source or {}).get("content") or ""), edge.get("source_evidence")
        )
        if not source_ok:
            return "source_evidence_stale"
        target_ok, _ = _evidence_is_verbatim(
            str((target or {}).get("content") or ""), edge.get("target_evidence")
        )
        return "" if target_ok else "target_evidence_stale"

    @staticmethod
    def _edge_payload(row: dict) -> dict:
        return {
            "edge_id": str(row.get("edge_id") or ""),
            "source": str(row.get("source_scene_id") or ""),
            "target": str(row.get("target_scene_id") or ""),
            "relation_type": str(row.get("relation_type") or ""),
            "directionality": str(row.get("directionality") or ""),
            "confidence": _clamp(row.get("confidence")),
            "reason": str(row.get("reason") or ""),
            "source_evidence": str(row.get("source_evidence") or ""),
            "target_evidence": str(row.get("target_evidence") or ""),
            "source_hash": str(row.get("source_hash") or ""),
            "target_hash": str(row.get("target_hash") or ""),
            "proposal_id": str(row.get("proposal_id") or ""),
            "linker_version": str(row.get("linker_version") or ""),
            "active": bool(row.get("active")),
            "accepted_at": str(row.get("accepted_at") or ""),
            "accepted_by": str(row.get("accepted_by") or ""),
            "deactivated_at": str(row.get("deactivated_at") or ""),
            "deactivated_by": str(row.get("deactivated_by") or ""),
            "deactivation_reason": str(row.get("deactivation_reason") or ""),
            "lifecycle_status": str(
                row.get("lifecycle_status") or ("active" if row.get("active") else "cancelled")
            ),
            "restored_at": str(row.get("restored_at") or ""),
            "restored_by": str(row.get("restored_by") or ""),
            "supersedes_edge_id": str(row.get("supersedes_edge_id") or ""),
            "replaced_by_edge_id": str(row.get("replaced_by_edge_id") or ""),
            "updated_at": str(row.get("updated_at") or ""),
            "graph_scope": "scene",
        }

SCENE_LINKER_PROMPT = """\
你是 Scene Linker。你的任务不是摘要、打标签或判断情绪，而是判断一条新 Scene 与少量旧 Scene 之间，是否存在足以支持自然回忆扩散的经历关系。

只允许以下关系：
- continues：同一经历、约定、行动或问题后来继续展开。方向从较早/前置 Scene 指向较晚/后续 Scene。
- echoes：表面话题可以不同，但两次经历中有具体而独特的表达、动作或关系 pattern 彼此回响。它是对称关系。
- resolves：后一条 Scene 使前一条 Scene 中明确存在的悬念、承诺或未完成问题落地。方向从未解决 Scene 指向解决 Scene。
- contrasts_with：两次具体经历形成有意义的反差，且反差本身有助于理解变化。它是对称关系。
- evidenced_by：一条 Scene 中的判断、选择或承诺，被另一条更具体的经历直接证明。方向从判断/承诺 Scene 指向证据 Scene。

严格禁止：
- 仅仅因为人物相同、时间接近、都提到爱/难过/开心、关键词相似或主题宽泛而连边；
- emotional_echo、relates_to 或任何未列出的关系；
- 改写 Scene、补造事实、输出标签、情绪分数、importance 或新 Scene；
- 用概括代替证据。new_scene_evidence 和 candidate_scene_evidence 必须分别逐字摘自输入原文。

orientation 规则：
- echoes / contrasts_with 必须是 symmetric；
- 其他关系必须是 candidate_to_new 或 new_to_candidate。

没有可靠关系时返回 {"edges": []}。不要为了凑数量连边。
同一个 candidate_scene_id 最多返回一条边；如果多个关系都勉强成立，只选最具体、证据最强的一条。
只返回 JSON：
{
  "edges": [
    {
      "candidate_scene_id": "候选 ID",
      "relation_type": "continues|echoes|resolves|contrasts_with|evidenced_by",
      "orientation": "candidate_to_new|new_to_candidate|symmetric",
      "confidence": 0.0,
      "reason": "说明两段具体经历为什么构成这个关系",
      "new_scene_evidence": "逐字摘自新 Scene 的短句",
      "candidate_scene_evidence": "逐字摘自候选 Scene 的短句"
    }
  ]
}
"""


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _environment_value(name: str) -> str:
    """Read a process env value, with Windows user/machine scope fallback."""
    normalized = str(name or "").strip()
    if not normalized:
        return ""
    value = os.environ.get(normalized, "")
    if value or os.name != "nt":
        return str(value or "").strip()
    try:
        import winreg

        scopes = (
            (winreg.HKEY_CURRENT_USER, r"Environment"),
            (
                winreg.HKEY_LOCAL_MACHINE,
                r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
            ),
        )
        for hive, subkey in scopes:
            try:
                with winreg.OpenKey(hive, subkey) as key:
                    scoped, _ = winreg.QueryValueEx(key, normalized)
            except OSError:
                continue
            if str(scoped or "").strip():
                return str(scoped).strip()
    except Exception:
        return ""
    return ""


def _scene_hash(scene: dict) -> str:
    # Formal relationships are grounded in Scene prose. Title/cue edits may
    # change candidate routing, but they must not invalidate a verbatim edge.
    payload = {
        "id": str(scene.get("id") or ""),
        "content": str(scene.get("content") or ""),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _clamp(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return max(0.0, min(1.0, round(number, 3)))


def _is_authored_scene(scene: dict | None) -> bool:
    if not isinstance(scene, dict):
        return False
    meta = scene.get("metadata", {}) if isinstance(scene.get("metadata"), dict) else {}
    if str(meta.get("memory_value_source") or "") != "authored_scene":
        return False
    if is_self_anchor_bucket(scene):
        return False
    if str(meta.get("type") or "").lower() in {"feel", "archived"}:
        return False
    if any(bool(meta.get(key)) for key in ("archived", "resolved", "digested")):
        return False
    return bool(str(scene.get("id") or "").strip() and str(scene.get("content") or "").strip())


def _clip_scene_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    limit = max(120, int(limit or 120))
    if len(text) <= limit:
        return text
    head = max(80, int(limit * 0.62))
    tail = max(40, limit - head - 5)
    return text[:head].rstrip() + "\n…\n" + text[-tail:].lstrip()


def _evidence_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    return text.strip("`'\"“”‘’《》「」『』 ")


def _evidence_is_verbatim(content: str, evidence: Any) -> tuple[bool, str]:
    excerpt = _evidence_text(evidence)
    compact_excerpt = re.sub(r"\s+", "", excerpt)
    compact_content = re.sub(r"\s+", "", str(content or ""))
    if len(compact_excerpt) < 6:
        return False, excerpt
    return compact_excerpt in compact_content, excerpt[:180]


def _parse_json_object(raw: str) -> dict | None:
    cleaned = str(raw or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(cleaned)
    except (TypeError, ValueError, json.JSONDecodeError):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    return parsed if isinstance(parsed, dict) else None


class SceneEdgeProposalStore:
    """Reviewable Scene-edge suggestions. It never writes memory_edges.jsonl."""

    def __init__(self, config: dict):
        state_dir = self.state_dir(config)
        os.makedirs(state_dir, exist_ok=True)
        self.db_path = os.path.join(state_dir, "scene_edge_proposals.sqlite")
        self._init_db()

    @staticmethod
    def state_dir(config: dict) -> str:
        return str(
            config.get("state_dir")
            or os.path.join(
                os.path.dirname(os.path.abspath(config.get("buckets_dir", "buckets"))),
                "state",
            )
        )

    @classmethod
    def path_for(cls, config: dict) -> str:
        return os.path.join(cls.state_dir(config), "scene_edge_proposals.sqlite")

    @classmethod
    def exists(cls, config: dict) -> bool:
        return os.path.exists(cls.path_for(config))

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scene_edge_proposals (
                proposal_id TEXT PRIMARY KEY,
                anchor_scene_id TEXT NOT NULL,
                candidate_scene_id TEXT NOT NULL,
                source_scene_id TEXT NOT NULL,
                target_scene_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                directionality TEXT NOT NULL,
                confidence REAL NOT NULL,
                reason TEXT NOT NULL,
                source_evidence TEXT NOT NULL,
                target_evidence TEXT NOT NULL,
                anchor_hash TEXT NOT NULL,
                candidate_hash TEXT NOT NULL,
                model TEXT NOT NULL,
                linker_version TEXT NOT NULL,
                canonical_edge_id TEXT NOT NULL DEFAULT '',
                proposal_origin TEXT NOT NULL DEFAULT 'automatic',
                created_by TEXT,
                supersedes_edge_id TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                reviewed_at TEXT,
                reviewed_by TEXT
            )
            """
        )
        columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(scene_edge_proposals)").fetchall()
        }
        if "reviewed_at" not in columns:
            conn.execute("ALTER TABLE scene_edge_proposals ADD COLUMN reviewed_at TEXT")
        if "reviewed_by" not in columns:
            conn.execute("ALTER TABLE scene_edge_proposals ADD COLUMN reviewed_by TEXT")
        if "canonical_edge_id" not in columns:
            conn.execute(
                "ALTER TABLE scene_edge_proposals "
                "ADD COLUMN canonical_edge_id TEXT NOT NULL DEFAULT ''"
            )
        if "proposal_origin" not in columns:
            conn.execute(
                "ALTER TABLE scene_edge_proposals "
                "ADD COLUMN proposal_origin TEXT NOT NULL DEFAULT 'automatic'"
            )
        for name in ("created_by", "supersedes_edge_id"):
            if name not in columns:
                conn.execute(f"ALTER TABLE scene_edge_proposals ADD COLUMN {name} TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_scene_edge_proposals_pending "
            "ON scene_edge_proposals(status, updated_at DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_scene_edge_proposals_anchor "
            "ON scene_edge_proposals(anchor_scene_id, updated_at DESC)"
        )
        _ensure_scene_edge_schema(conn)
        self._backfill_canonical_edge_ids(conn)
        self._consolidate_pending_edges(conn)
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_scene_edge_proposals_one_pending_per_edge "
            "ON scene_edge_proposals(canonical_edge_id) "
            "WHERE status = 'pending' AND canonical_edge_id <> ''"
        )
        conn.commit()
        conn.close()

    @staticmethod
    def _backfill_canonical_edge_ids(conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            """
            SELECT proposal_id, source_scene_id, target_scene_id,
                   relation_type, canonical_edge_id
              FROM scene_edge_proposals
            """
        ).fetchall()
        updates = []
        for row in rows:
            canonical_edge_id = _scene_edge_id(
                str(row["source_scene_id"] or ""),
                str(row["target_scene_id"] or ""),
                str(row["relation_type"] or ""),
            )
            if str(row["canonical_edge_id"] or "") != canonical_edge_id:
                updates.append((canonical_edge_id, str(row["proposal_id"])))
        if updates:
            conn.executemany(
                "UPDATE scene_edge_proposals SET canonical_edge_id = ? WHERE proposal_id = ?",
                updates,
            )

    @staticmethod
    def _consolidate_pending_edges(conn: sqlite3.Connection) -> None:
        """Keep one review card per formal Scene edge without rewriting decisions."""
        now = _now_utc()
        active_edges = {
            str(row["edge_id"]): (
                str(row["source_hash"] or ""),
                str(row["target_hash"] or ""),
            )
            for row in conn.execute(
                "SELECT edge_id, source_hash, target_hash FROM scene_edges WHERE active = 1"
            ).fetchall()
        }
        rows = conn.execute(
            """
            SELECT proposal_id, canonical_edge_id,
                   anchor_scene_id, source_scene_id, target_scene_id, relation_type,
                   anchor_hash, candidate_hash,
                   confidence, proposal_origin, updated_at, created_at
              FROM scene_edge_proposals
             WHERE status = 'pending' AND canonical_edge_id <> ''
            """
        ).fetchall()
        grouped: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            grouped.setdefault(str(row["canonical_edge_id"]), []).append(row)

        superseded_ids: list[str] = []
        for canonical_edge_id, proposals in grouped.items():
            formal_hashes = active_edges.get(canonical_edge_id)
            if formal_hashes is not None:
                matching_formal = [
                    row
                    for row in proposals
                    if _scene_edge_snapshot_hashes(
                        str(row["source_scene_id"] or ""),
                        str(row["target_scene_id"] or ""),
                        str(row["relation_type"] or ""),
                        str(row["anchor_scene_id"] or ""),
                        str(row["anchor_hash"] or ""),
                        str(row["candidate_hash"] or ""),
                    )
                    == formal_hashes
                ]
                matching_ids = {
                    str(row["proposal_id"]) for row in matching_formal
                }
                superseded_ids.extend(matching_ids)
                proposals = [
                    row
                    for row in proposals
                    if str(row["proposal_id"]) not in matching_ids
                ]
            if len(proposals) <= 1:
                continue
            keep = max(
                proposals,
                key=lambda row: (
                    SCENE_EDGE_PROPOSAL_ORIGIN_PRIORITY.get(
                        str(row["proposal_origin"] or "automatic"), 0
                    ),
                    float(row["confidence"] or 0.0),
                    str(row["updated_at"] or ""),
                    str(row["created_at"] or ""),
                    str(row["proposal_id"] or ""),
                ),
            )
            superseded_ids.extend(
                str(row["proposal_id"])
                for row in proposals
                if str(row["proposal_id"]) != str(keep["proposal_id"])
            )
        if superseded_ids:
            conn.executemany(
                """
                UPDATE scene_edge_proposals
                   SET status = 'superseded', updated_at = ?
                 WHERE proposal_id = ? AND status = 'pending'
                """,
                [(now, proposal_id) for proposal_id in superseded_ids],
            )

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict | None:
        return dict(row) if row is not None else None

    def replace_for_anchor(
        self,
        anchor: dict,
        candidate_map: dict[str, dict],
        edges: list[dict],
        *,
        model: str,
        linker_version: str = SCENE_LINKER_VERSION,
    ) -> list[dict]:
        anchor_id = str(anchor.get("id") or "").strip()
        if not anchor_id:
            raise ValueError("Scene edge proposal requires an anchor Scene id")
        anchor_hash = _scene_hash(anchor)
        model_name = str(model or "").strip()
        version = str(linker_version or SCENE_LINKER_VERSION).strip()
        now = _now_utc()
        conn = self._connect()
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            UPDATE scene_edge_proposals
               SET status = 'superseded', updated_at = ?
             WHERE anchor_scene_id = ? AND linker_version = ? AND status = 'pending'
               AND proposal_origin = 'automatic'
            """,
            (now, anchor_id, version),
        )
        proposal_ids: list[str] = []
        for edge in edges:
            candidate_id = str(edge.get("candidate_scene_id") or "").strip()
            candidate = candidate_map.get(candidate_id)
            if not candidate:
                continue
            candidate_hash = _scene_hash(candidate)
            source_scene_id = str(edge.get("source_scene_id") or "").strip()
            target_scene_id = str(edge.get("target_scene_id") or "").strip()
            relation_type = str(edge.get("relation_type") or "").strip()
            canonical_edge_id = _scene_edge_id(
                source_scene_id,
                target_scene_id,
                relation_type,
            )
            source_hash, target_hash = _scene_edge_snapshot_hashes(
                source_scene_id,
                target_scene_id,
                relation_type,
                anchor_id,
                anchor_hash,
                candidate_hash,
            )
            formal_edge = conn.execute(
                """
                SELECT source_hash, target_hash
                  FROM scene_edges
                 WHERE edge_id = ? AND active = 1
                """,
                (canonical_edge_id,),
            ).fetchone()
            if formal_edge is not None and (
                str(formal_edge["source_hash"] or ""),
                str(formal_edge["target_hash"] or ""),
            ) == (source_hash, target_hash):
                conn.execute(
                    """
                    UPDATE scene_edge_proposals
                       SET status = 'superseded', updated_at = ?
                     WHERE canonical_edge_id = ? AND status = 'pending'
                    """,
                    (now, canonical_edge_id),
                )
                continue

            confidence = float(edge["confidence"])
            existing = conn.execute(
                """
                SELECT * FROM scene_edge_proposals
                 WHERE canonical_edge_id = ? AND status = 'pending'
                 ORDER BY confidence DESC, updated_at DESC
                 LIMIT 1
                """,
                (canonical_edge_id,),
            ).fetchone()
            if existing is not None:
                if str(existing["proposal_origin"] or "automatic") != "automatic":
                    continue
                proposal_id = str(existing["proposal_id"])
                existing_hashes = _scene_edge_snapshot_hashes(
                    str(existing["source_scene_id"] or ""),
                    str(existing["target_scene_id"] or ""),
                    str(existing["relation_type"] or ""),
                    str(existing["anchor_scene_id"] or ""),
                    str(existing["anchor_hash"] or ""),
                    str(existing["candidate_hash"] or ""),
                )
                snapshot_is_fresh = existing_hashes == (source_hash, target_hash)
                if not snapshot_is_fresh or confidence > float(existing["confidence"] or 0.0):
                    conn.execute(
                        """
                        UPDATE scene_edge_proposals
                           SET anchor_scene_id = ?, candidate_scene_id = ?,
                               source_scene_id = ?, target_scene_id = ?,
                               relation_type = ?, directionality = ?,
                               confidence = ?, reason = ?,
                               source_evidence = ?, target_evidence = ?,
                                anchor_hash = ?, candidate_hash = ?,
                                model = ?, linker_version = ?,
                                canonical_edge_id = ?,
                                proposal_origin = 'automatic', created_by = 'scene_linker',
                                supersedes_edge_id = NULL, updated_at = ?
                         WHERE proposal_id = ? AND status = 'pending'
                        """,
                        (
                            anchor_id,
                            candidate_id,
                            source_scene_id,
                            target_scene_id,
                            relation_type,
                            str(edge["directionality"]),
                            confidence,
                            str(edge["reason"]),
                            str(edge["source_evidence"]),
                            str(edge["target_evidence"]),
                            anchor_hash,
                            candidate_hash,
                            model_name,
                            version,
                            canonical_edge_id,
                            now,
                            proposal_id,
                        ),
                    )
                if proposal_id not in proposal_ids:
                    proposal_ids.append(proposal_id)
                continue

            identity = json.dumps(
                {
                    "anchor": anchor_id,
                    "candidate": candidate_id,
                    "source": source_scene_id,
                    "target": target_scene_id,
                    "relation": relation_type,
                    "directionality": edge.get("directionality"),
                    "anchor_hash": anchor_hash,
                    "candidate_hash": candidate_hash,
                    "model": model_name,
                    "version": version,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            proposal_id = "scene_edge_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
            if proposal_id not in proposal_ids:
                proposal_ids.append(proposal_id)
            conn.execute(
                """
                INSERT INTO scene_edge_proposals (
                    proposal_id, anchor_scene_id, candidate_scene_id,
                    source_scene_id, target_scene_id, relation_type, directionality,
                    confidence, reason, source_evidence, target_evidence,
                    anchor_hash, candidate_hash, model, linker_version,
                    canonical_edge_id, proposal_origin, created_by,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'automatic', 'scene_linker', 'pending', ?, ?)
                ON CONFLICT(proposal_id) DO UPDATE SET
                    anchor_scene_id = excluded.anchor_scene_id,
                    candidate_scene_id = excluded.candidate_scene_id,
                    source_scene_id = excluded.source_scene_id,
                    target_scene_id = excluded.target_scene_id,
                    relation_type = excluded.relation_type,
                    directionality = excluded.directionality,
                    confidence = excluded.confidence,
                    reason = excluded.reason,
                    source_evidence = excluded.source_evidence,
                    target_evidence = excluded.target_evidence,
                    anchor_hash = excluded.anchor_hash,
                    candidate_hash = excluded.candidate_hash,
                    model = excluded.model,
                    linker_version = excluded.linker_version,
                    canonical_edge_id = excluded.canonical_edge_id,
                    proposal_origin = excluded.proposal_origin,
                    created_by = excluded.created_by,
                    supersedes_edge_id = NULL,
                    status = CASE
                        WHEN scene_edge_proposals.status IN ('accepted', 'rejected')
                        THEN scene_edge_proposals.status
                        ELSE 'pending'
                    END,
                    updated_at = excluded.updated_at
                """,
                (
                    proposal_id,
                    anchor_id,
                    candidate_id,
                    source_scene_id,
                    target_scene_id,
                    relation_type,
                    edge["directionality"],
                    confidence,
                    str(edge["reason"]),
                    str(edge["source_evidence"]),
                    str(edge["target_evidence"]),
                    anchor_hash,
                    candidate_hash,
                    model_name,
                    version,
                    canonical_edge_id,
                    now,
                    now,
                ),
            )
        conn.commit()
        rows = []
        for proposal_id in proposal_ids:
            row = conn.execute(
                "SELECT * FROM scene_edge_proposals WHERE proposal_id = ?",
                (proposal_id,),
            ).fetchone()
            if row is not None:
                rows.append(dict(row))
        conn.close()
        return rows

    def create_review_proposal(
        self,
        anchor: dict,
        candidate: dict,
        edge: dict,
        *,
        model: str,
        proposal_origin: str,
        created_by: str,
        supersedes_edge_id: str = "",
        linker_version: str = SCENE_LINKER_VERSION,
        nonce: str = "",
    ) -> dict:
        """Create one evidence-validated proposal without invoking the model."""
        origin = str(proposal_origin or "").strip().lower()
        if origin not in SCENE_EDGE_PROPOSAL_ORIGINS:
            raise ValueError("invalid Scene edge proposal origin")
        anchor_id = str(anchor.get("id") or "").strip()
        candidate_id = str(candidate.get("id") or "").strip()
        if not anchor_id or not candidate_id or anchor_id == candidate_id:
            raise ValueError("Scene edge proposal requires two distinct Scenes")
        source_scene_id = str(edge.get("source_scene_id") or "").strip()
        target_scene_id = str(edge.get("target_scene_id") or "").strip()
        relation_type = str(edge.get("relation_type") or "").strip()
        canonical_edge_id = _scene_edge_id(source_scene_id, target_scene_id, relation_type)
        anchor_hash = _scene_hash(anchor)
        candidate_hash = _scene_hash(candidate)
        source_hash, target_hash = _scene_edge_snapshot_hashes(
            source_scene_id,
            target_scene_id,
            relation_type,
            anchor_id,
            anchor_hash,
            candidate_hash,
        )
        now = _now_utc()
        identity = json.dumps(
            {
                "anchor": anchor_id,
                "candidate": candidate_id,
                "source": source_scene_id,
                "target": target_scene_id,
                "relation": relation_type,
                "directionality": str(edge.get("directionality") or ""),
                "anchor_hash": anchor_hash,
                "candidate_hash": candidate_hash,
                "origin": origin,
                "supersedes_edge_id": str(supersedes_edge_id or ""),
                "nonce": str(nonce or ""),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        proposal_id = "scene_edge_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            _ensure_scene_edge_schema(conn)
            formal_edge = conn.execute(
                "SELECT source_hash, target_hash FROM scene_edges WHERE edge_id = ? AND active = 1",
                (canonical_edge_id,),
            ).fetchone()
            if formal_edge is not None and (
                str(formal_edge["source_hash"] or ""),
                str(formal_edge["target_hash"] or ""),
            ) == (source_hash, target_hash):
                conn.rollback()
                return {
                    "status": "already_active",
                    "canonical_edge_id": canonical_edge_id,
                }
            conn.execute(
                """
                UPDATE scene_edge_proposals
                   SET status = 'superseded', updated_at = ?
                 WHERE canonical_edge_id = ? AND status = 'pending'
                """,
                (now, canonical_edge_id),
            )
            conn.execute(
                """
                INSERT INTO scene_edge_proposals (
                    proposal_id, anchor_scene_id, candidate_scene_id,
                    source_scene_id, target_scene_id, relation_type, directionality,
                    confidence, reason, source_evidence, target_evidence,
                    anchor_hash, candidate_hash, model, linker_version,
                    canonical_edge_id, proposal_origin, created_by, supersedes_edge_id,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                ON CONFLICT(proposal_id) DO UPDATE SET
                    confidence = excluded.confidence,
                    reason = excluded.reason,
                    source_evidence = excluded.source_evidence,
                    target_evidence = excluded.target_evidence,
                    model = excluded.model,
                    linker_version = excluded.linker_version,
                    created_by = excluded.created_by,
                    supersedes_edge_id = excluded.supersedes_edge_id,
                    status = CASE
                        WHEN scene_edge_proposals.status IN ('accepted', 'rejected')
                        THEN scene_edge_proposals.status
                        ELSE 'pending'
                    END,
                    updated_at = excluded.updated_at
                """,
                (
                    proposal_id,
                    anchor_id,
                    candidate_id,
                    source_scene_id,
                    target_scene_id,
                    relation_type,
                    str(edge.get("directionality") or ""),
                    float(edge.get("confidence") or 0.0),
                    str(edge.get("reason") or ""),
                    str(edge.get("source_evidence") or ""),
                    str(edge.get("target_evidence") or ""),
                    anchor_hash,
                    candidate_hash,
                    str(model or "").strip(),
                    str(linker_version or SCENE_LINKER_VERSION).strip(),
                    canonical_edge_id,
                    origin,
                    str(created_by or "").strip() or None,
                    str(supersedes_edge_id or "").strip() or None,
                    now,
                    now,
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM scene_edge_proposals WHERE proposal_id = ?",
                (proposal_id,),
            ).fetchone()
            return {"status": str(row["status"]), "proposal": dict(row)}
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def refresh_after_scene_content_change(
        self,
        scene_id: str,
        scene_map: dict[str, dict],
        *,
        created_by: str = "system",
    ) -> dict:
        """Invalidate stale snapshots and create exact-evidence revalidation cards."""
        normalized = str(scene_id or "").strip()
        current_scene = scene_map.get(normalized)
        if not normalized or not isinstance(current_scene, dict):
            return {"status": "skipped", "reason": "scene_missing"}
        current_hash = _scene_hash(current_scene)
        now = _now_utc()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            _ensure_scene_edge_schema(conn)
            stale_cursor = conn.execute(
                """
                UPDATE scene_edge_proposals
                   SET status = 'superseded', updated_at = ?
                 WHERE status = 'pending'
                   AND ((anchor_scene_id = ? AND anchor_hash <> ?)
                     OR (candidate_scene_id = ? AND candidate_hash <> ?))
                """,
                (now, normalized, current_hash, normalized, current_hash),
            )
            rows = conn.execute(
                """
                SELECT * FROM scene_edges
                 WHERE active = 1 AND (source_scene_id = ? OR target_scene_id = ?)
                """,
                (normalized, normalized),
            ).fetchall()
            affected: list[dict] = []
            for row in rows:
                edge = SceneEdgeStore._edge_payload(dict(row))
                stored_hash = edge["source_hash"] if edge["source"] == normalized else edge["target_hash"]
                if stored_hash == current_hash:
                    continue
                conn.execute(
                    """
                    UPDATE scene_edges
                       SET active = 0,
                           lifecycle_status = 'needs_review',
                           deactivated_at = ?,
                           deactivated_by = ?,
                           deactivation_reason = 'scene_content_changed',
                           updated_at = ?
                     WHERE edge_id = ? AND active = 1
                    """,
                    (now, str(created_by or "system"), now, edge["edge_id"]),
                )
                affected.append(edge)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        revalidation_ids: list[str] = []
        relink_required = 0
        for edge in affected:
            source = scene_map.get(edge["source"])
            target = scene_map.get(edge["target"])
            source_ok, _ = _evidence_is_verbatim(
                str((source or {}).get("content") or ""), edge["source_evidence"]
            )
            target_ok, _ = _evidence_is_verbatim(
                str((target or {}).get("content") or ""), edge["target_evidence"]
            )
            if not (_is_authored_scene(source) and _is_authored_scene(target) and source_ok and target_ok):
                relink_required += 1
                continue
            candidate_id = edge["target"] if edge["source"] == normalized else edge["source"]
            result = self.create_review_proposal(
                current_scene,
                scene_map[candidate_id],
                {
                    "source_scene_id": edge["source"],
                    "target_scene_id": edge["target"],
                    "relation_type": edge["relation_type"],
                    "directionality": edge["directionality"],
                    "confidence": edge["confidence"],
                    "reason": edge["reason"],
                    "source_evidence": edge["source_evidence"],
                    "target_evidence": edge["target_evidence"],
                },
                model="deterministic_revalidation",
                proposal_origin="snapshot_revalidation",
                created_by=created_by,
                supersedes_edge_id=edge["edge_id"],
                linker_version=edge["linker_version"] or SCENE_LINKER_VERSION,
                nonce=now,
            )
            proposal = result.get("proposal") or {}
            if proposal.get("proposal_id"):
                revalidation_ids.append(str(proposal["proposal_id"]))
        return {
            "status": "updated",
            "scene_id": normalized,
            "stale_proposals_superseded": max(0, int(stale_cursor.rowcount or 0)),
            "formal_edges_needing_review": len(affected),
            "revalidation_proposal_ids": revalidation_ids,
            "normal_relink_required": relink_required > 0 or not affected,
        }

    def get(self, proposal_id: str) -> dict | None:
        normalized = str(proposal_id or "").strip()
        if not normalized:
            return None
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM scene_edge_proposals WHERE proposal_id = ?",
            (normalized,),
        ).fetchone()
        conn.close()
        return self._row(row)

    def list(
        self,
        *,
        status: str = "pending",
        anchor_scene_id: str = "",
        limit: int = 100,
    ) -> list[dict]:
        conn = self._connect()
        bounded = max(1, min(int(limit or 100), 1000))
        normalized_status = str(status or "pending").strip().lower()
        if normalized_status != "all" and normalized_status not in SCENE_EDGE_PROPOSAL_STATUSES:
            conn.close()
            raise ValueError("invalid Scene edge proposal status")
        clauses = []
        params: list[Any] = []
        if normalized_status != "all":
            clauses.append("status = ?")
            params.append(normalized_status)
        normalized_anchor = str(anchor_scene_id or "").strip()
        if normalized_anchor:
            clauses.append("anchor_scene_id = ?")
            params.append(normalized_anchor)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        params.append(bounded)
        rows = conn.execute(
            f"""
            SELECT * FROM scene_edge_proposals{where}
             ORDER BY updated_at DESC, confidence DESC LIMIT ?
            """,
            params,
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def list_pending(self, *, anchor_scene_id: str = "", limit: int = 100) -> list[dict]:
        return self.list(status="pending", anchor_scene_id=anchor_scene_id, limit=limit)

    def supersede_stale_pending(self, scene_map: dict[str, dict]) -> int:
        """Close every pending card whose endpoint snapshot is no longer current."""
        rows = self.list(status="pending", limit=1000)
        stale_ids: list[str] = []
        for row in rows:
            anchor = scene_map.get(str(row.get("anchor_scene_id") or ""))
            candidate = scene_map.get(str(row.get("candidate_scene_id") or ""))
            if (
                not _is_authored_scene(anchor)
                or not _is_authored_scene(candidate)
                or _scene_hash(anchor) != str(row.get("anchor_hash") or "")
                or _scene_hash(candidate) != str(row.get("candidate_hash") or "")
            ):
                stale_ids.append(str(row.get("proposal_id") or ""))
        if not stale_ids:
            return 0
        now = _now_utc()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.executemany(
                """
                UPDATE scene_edge_proposals
                   SET status = 'superseded', updated_at = ?
                 WHERE proposal_id = ? AND status = 'pending'
                """,
                [(now, proposal_id) for proposal_id in stale_ids],
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return len(stale_ids)

    def set_status(
        self,
        proposal_id: str,
        status: str,
        *,
        expected_status: str | None = None,
        reviewed_by: str = "",
    ) -> dict | None:
        normalized = str(status or "").strip().lower()
        if normalized not in SCENE_EDGE_PROPOSAL_STATUSES:
            raise ValueError("invalid Scene edge proposal status")
        expected = str(expected_status or "").strip().lower()
        if expected and expected not in SCENE_EDGE_PROPOSAL_STATUSES:
            raise ValueError("invalid expected Scene edge proposal status")
        proposal_id = str(proposal_id or "").strip()
        now = _now_utc()
        conn = self._connect()
        reviewed_at = now if normalized in {"accepted", "rejected"} else None
        reviewer = str(reviewed_by or "").strip()[:80] if reviewed_at else None
        if expected:
            conn.execute(
                """
                UPDATE scene_edge_proposals
                   SET status = ?, updated_at = ?, reviewed_at = ?, reviewed_by = ?
                 WHERE proposal_id = ? AND status = ?
                """,
                (normalized, now, reviewed_at, reviewer, proposal_id, expected),
            )
        else:
            conn.execute(
                """
                UPDATE scene_edge_proposals
                   SET status = ?, updated_at = ?, reviewed_at = ?, reviewed_by = ?
                 WHERE proposal_id = ?
                """,
                (normalized, now, reviewed_at, reviewer, proposal_id),
            )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM scene_edge_proposals WHERE proposal_id = ?",
            (proposal_id,),
        ).fetchone()
        conn.close()
        return self._row(row)

    def promote(self, proposal_id: str, *, reviewed_by: str = "") -> dict | None:
        """Atomically accept one proposal and publish it to the Scene-only graph."""
        normalized = str(proposal_id or "").strip()
        now = _now_utc()
        reviewer = str(reviewed_by or "").strip()[:80]
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM scene_edge_proposals WHERE proposal_id = ?",
                (normalized,),
            ).fetchone()
            if row is None or str(row["status"] or "") != "pending":
                conn.rollback()
                return None
            proposal = dict(row)
            source_id = str(proposal.get("source_scene_id") or "")
            target_id = str(proposal.get("target_scene_id") or "")
            anchor_id = str(proposal.get("anchor_scene_id") or "")
            source_hash = (
                str(proposal.get("anchor_hash") or "")
                if source_id == anchor_id
                else str(proposal.get("candidate_hash") or "")
            )
            target_hash = (
                str(proposal.get("anchor_hash") or "")
                if target_id == anchor_id
                else str(proposal.get("candidate_hash") or "")
            )
            relation_type = str(proposal.get("relation_type") or "")
            supersedes_edge_id = str(proposal.get("supersedes_edge_id") or "").strip()
            source_evidence = str(proposal.get("source_evidence") or "")
            target_evidence = str(proposal.get("target_evidence") or "")
            if relation_type in SYMMETRIC_SCENE_RELATIONS and target_id < source_id:
                source_id, target_id = target_id, source_id
                source_hash, target_hash = target_hash, source_hash
                source_evidence, target_evidence = target_evidence, source_evidence
            edge_id = _scene_edge_id(source_id, target_id, relation_type)
            _ensure_scene_edge_schema(conn)
            conn.execute(
                """
                INSERT INTO scene_edges (
                    edge_id, source_scene_id, target_scene_id, relation_type,
                    directionality, confidence, reason,
                    source_evidence, target_evidence, source_hash, target_hash,
                    proposal_id, linker_version, active,
                    accepted_at, accepted_by, lifecycle_status,
                    restored_at, restored_by, supersedes_edge_id, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, 'active', ?, ?, ?, ?)
                ON CONFLICT(edge_id) DO UPDATE SET
                    directionality = excluded.directionality,
                    confidence = excluded.confidence,
                    reason = excluded.reason,
                    source_evidence = excluded.source_evidence,
                    target_evidence = excluded.target_evidence,
                    source_hash = excluded.source_hash,
                    target_hash = excluded.target_hash,
                    proposal_id = excluded.proposal_id,
                    linker_version = excluded.linker_version,
                    active = 1,
                    accepted_at = excluded.accepted_at,
                    accepted_by = excluded.accepted_by,
                    lifecycle_status = 'active',
                    restored_at = excluded.restored_at,
                    restored_by = excluded.restored_by,
                    supersedes_edge_id = excluded.supersedes_edge_id,
                    deactivated_at = NULL,
                    deactivated_by = NULL,
                    deactivation_reason = NULL,
                    replaced_by_edge_id = NULL,
                    updated_at = excluded.updated_at
                """,
                (
                    edge_id,
                    source_id,
                    target_id,
                    relation_type,
                    str(proposal.get("directionality") or ""),
                    float(proposal.get("confidence") or 0.0),
                    str(proposal.get("reason") or ""),
                    source_evidence,
                    target_evidence,
                    source_hash,
                    target_hash,
                    normalized,
                    str(proposal.get("linker_version") or SCENE_LINKER_VERSION),
                    now,
                    reviewer or None,
                    now if supersedes_edge_id else None,
                    (reviewer or None) if supersedes_edge_id else None,
                    supersedes_edge_id or None,
                    now,
                ),
            )
            if supersedes_edge_id and supersedes_edge_id != edge_id:
                conn.execute(
                    """
                    UPDATE scene_edges
                       SET active = 0,
                           lifecycle_status = 'replaced',
                           deactivated_at = COALESCE(deactivated_at, ?),
                           deactivated_by = COALESCE(deactivated_by, ?),
                           deactivation_reason = 'relinked',
                           replaced_by_edge_id = ?,
                           updated_at = ?
                     WHERE edge_id = ?
                    """,
                    (now, reviewer or None, edge_id, now, supersedes_edge_id),
                )
            updated = conn.execute(
                """
                UPDATE scene_edge_proposals
                   SET status = 'accepted', updated_at = ?, reviewed_at = ?, reviewed_by = ?
                 WHERE proposal_id = ? AND status = 'pending'
                """,
                (now, now, reviewer or None, normalized),
            )
            if int(updated.rowcount or 0) != 1:
                conn.rollback()
                return None
            conn.execute(
                """
                UPDATE scene_edge_proposals
                   SET status = 'superseded', updated_at = ?
                 WHERE canonical_edge_id = ?
                   AND proposal_id <> ?
                   AND status = 'pending'
                """,
                (now, edge_id, normalized),
            )
            conn.commit()
            edge_row = conn.execute(
                "SELECT * FROM scene_edges WHERE edge_id = ?",
                (edge_id,),
            ).fetchone()
            proposal_row = conn.execute(
                "SELECT * FROM scene_edge_proposals WHERE proposal_id = ?",
                (normalized,),
            ).fetchone()
            return {
                "edge": SceneEdgeStore._edge_payload(dict(edge_row)),
                "proposal": dict(proposal_row),
            }
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def stats(self) -> dict:
        conn = self._connect()
        rows = conn.execute(
            "SELECT status, COUNT(*) AS count FROM scene_edge_proposals GROUP BY status"
        ).fetchall()
        conn.close()
        counts = {str(row["status"]): int(row["count"]) for row in rows}
        return {
            "total": sum(counts.values()),
            "pending": counts.get("pending", 0),
            "accepted": counts.get("accepted", 0),
            "rejected": counts.get("rejected", 0),
            "superseded": counts.get("superseded", 0),
        }


class SceneLinker:
    """Asynchronous, proposal-only linker for canonical authored Scenes."""

    def __init__(
        self,
        config: dict,
        *,
        proposal_store: SceneEdgeProposalStore | None = None,
        clients: dict[str, Any] | None = None,
    ):
        self.config = config
        cfg = config.get("scene_linker", {}) if isinstance(config.get("scene_linker"), dict) else {}
        self.enabled = bool(cfg.get("enabled", False))
        self.auto_enabled = bool(cfg.get("auto_enabled", False))
        self.semantic_candidates = max(0, min(int(cfg.get("semantic_candidates", 8)), 20))
        self.recent_candidates = max(0, min(int(cfg.get("recent_candidates", 4)), 20))
        self.max_candidates = max(1, min(int(cfg.get("max_candidates", 10)), 24))
        self.max_proposals = max(1, min(int(cfg.get("max_proposals", 3)), 6))
        self.min_confidence = max(0.0, min(float(cfg.get("min_confidence", 0.78)), 1.0))
        self.source_chars = max(500, min(int(cfg.get("source_chars", 2600)), 8000))
        self.candidate_chars = max(300, min(int(cfg.get("candidate_chars", 1200)), 4000))
        self.candidate_timeout_seconds = max(
            0.5,
            min(float(cfg.get("candidate_timeout_seconds", 5.0)), 30.0),
        )
        self.timeout_seconds = max(5.0, min(float(cfg.get("timeout_seconds", 60.0)), 300.0))
        self.max_concurrency = max(1, min(int(cfg.get("max_concurrency", 1)), 4))
        self._model_semaphore = asyncio.Semaphore(self.max_concurrency)
        self._review_lock = asyncio.Lock()
        self.linker_version = str(cfg.get("version") or SCENE_LINKER_VERSION).strip()
        self._proposal_store = proposal_store
        self.providers = self._load_providers(cfg, clients or {})

    @property
    def can_auto_link(self) -> bool:
        return self.enabled and self.auto_enabled and any(provider.get("client") for provider in self.providers)

    def status(self) -> dict:
        proposal_store = self.proposal_store(create=False)
        proposal_stats = (
            proposal_store.stats()
            if proposal_store is not None
            else {"total": 0, "pending": 0, "accepted": 0, "rejected": 0, "superseded": 0}
        )
        scene_edge_stats = SceneEdgeStore(self.config, create=False).stats()
        return {
            "enabled": self.enabled,
            "auto_enabled": self.auto_enabled,
            "configured_models": [provider["name"] for provider in self.providers],
            "ready_models": [provider["name"] for provider in self.providers if provider.get("client")],
            "model_pool": [
                {
                    "position": index + 1,
                    "name": provider["name"],
                    "model": provider.get("model") or "",
                    "ready": bool(provider.get("client")),
                }
                for index, provider in enumerate(self.providers)
            ],
            "max_concurrency": self.max_concurrency,
            "proposal_store": proposal_stats,
            "scene_edges": scene_edge_stats,
            "writes_canonical_memory": False,
            "writes_edge_store": False,
            "writes_scene_edges_on_accept": True,
        }

    def list_scene_edges(self, *, include_inactive: bool = False) -> list[dict]:
        return SceneEdgeStore(self.config, create=False).list_edges(
            include_inactive=include_inactive
        )

    def recall_scene_edges(self, scene_map: dict[str, dict]) -> list[dict]:
        return SceneEdgeStore(self.config, create=False).recall_edges(scene_map)

    def deactivate_scene_edges(
        self,
        scene_id: str,
        *,
        reviewer: str = "system",
        reason: str = "scene_deleted",
        lifecycle_status: str = "cancelled",
    ) -> int:
        return SceneEdgeStore(self.config, create=False).deactivate_for_scene(
            scene_id,
            reviewer=reviewer,
            reason=reason,
            lifecycle_status=lifecycle_status,
        )

    def deactivate_scene_edge(
        self,
        edge_id: str,
        *,
        scene_id: str,
        reviewer: str = "",
        reason: str = "manual_remove",
    ) -> dict:
        return SceneEdgeStore(self.config, create=False).deactivate_edge(
            edge_id,
            scene_id=scene_id,
            reviewer=reviewer,
            reason=reason,
        )

    async def handle_scene_content_changed(self, scene_id: str, bucket_mgr) -> dict:
        store = self.proposal_store(create=False)
        if store is None:
            return {
                "status": "skipped",
                "reason": "no_scene_edge_store",
                "normal_relink_required": True,
            }
        normalized = str(scene_id or "").strip()
        current = await bucket_mgr.get(normalized)
        if not isinstance(current, dict):
            return {
                "status": "skipped",
                "reason": "scene_missing",
                "normal_relink_required": False,
            }
        incident = [
            edge
            for edge in SceneEdgeStore(self.config, create=False).list_edges(
                include_inactive=False
            )
            if normalized in {edge["source"], edge["target"]}
        ]
        scene_map = {normalized: current}
        for edge in incident:
            other_id = edge["target"] if edge["source"] == normalized else edge["source"]
            if other_id not in scene_map:
                other = await bucket_mgr.get(other_id)
                if isinstance(other, dict):
                    scene_map[other_id] = other
        return store.refresh_after_scene_content_change(
            normalized,
            scene_map,
            created_by="scene_edit",
        )

    async def create_manual_proposal(
        self,
        *,
        source_scene_id: str,
        target_scene_id: str,
        relation_type: str,
        source_evidence: str,
        target_evidence: str,
        reason: str,
        bucket_mgr,
        created_by: str,
        confidence: float = 1.0,
        supersedes_edge_id: str = "",
    ) -> dict:
        source_id = str(source_scene_id or "").strip()
        target_id = str(target_scene_id or "").strip()
        source = await bucket_mgr.get(source_id)
        target = await bucket_mgr.get(target_id)
        if not _is_authored_scene(source) or not _is_authored_scene(target):
            return {"status": "invalid", "reason": "active_authored_scenes_required"}
        relation = str(relation_type or "").strip()
        raw = {
            "candidate_scene_id": target_id,
            "relation_type": relation,
            "orientation": "symmetric" if relation in SYMMETRIC_SCENE_RELATIONS else "new_to_candidate",
            "confidence": confidence,
            "reason": reason,
            "new_scene_evidence": source_evidence,
            "candidate_scene_evidence": target_evidence,
        }
        normalized, rejected = self._normalize_edges(source, {target_id: target}, [raw])
        if not normalized:
            return {
                "status": "invalid",
                "reason": str((rejected[0] if rejected else {}).get("reason") or "invalid_proposal"),
            }
        superseded = str(supersedes_edge_id or "").strip()
        old = None
        if superseded:
            old = next(
                (
                    edge
                    for edge in SceneEdgeStore(self.config, create=False).list_edges(
                        include_inactive=True
                    )
                    if edge["edge_id"] == superseded
                ),
                None,
            )
            if old is None:
                return {"status": "not_found", "reason": "superseded_edge_not_found"}
            if source_id not in {old["source"], old["target"]} and target_id not in {
                old["source"], old["target"]
            }:
                return {"status": "invalid", "reason": "relink_must_share_an_endpoint"}
        result = self._store().create_review_proposal(
            source,
            target,
            normalized[0],
            model="manual",
            proposal_origin="manual_relink" if superseded else "manual",
            created_by=created_by,
            supersedes_edge_id=superseded,
            linker_version=self.linker_version,
        )
        if old is not None and result.get("status") == "pending":
            SceneEdgeStore(self.config, create=False).deactivate_edge(
                superseded,
                scene_id=old["source"],
                reviewer=created_by,
                reason="manual_relink",
            )
        return {
            **result,
            "canonical_memory_changed": False,
            "memory_edges_changed": False,
            "scene_edges_changed": bool(superseded),
        }

    async def restore_scene_edge(
        self,
        edge_id: str,
        bucket_mgr,
        *,
        reviewed_by: str,
    ) -> dict:
        store = SceneEdgeStore(self.config, create=False)
        edge = next(
            (
                item
                for item in store.list_edges(include_inactive=True)
                if item["edge_id"] == str(edge_id or "").strip()
            ),
            None,
        )
        if edge is None:
            return {"status": "not_found", "edge_id": str(edge_id or "").strip()}
        scene_map = {
            edge["source"]: await bucket_mgr.get(edge["source"]),
            edge["target"]: await bucket_mgr.get(edge["target"]),
        }
        return store.restore_edge(edge["edge_id"], scene_map, reviewer=reviewed_by)

    async def restore_archived_scene_edges(self, scene_id: str, bucket_mgr) -> dict:
        normalized = str(scene_id or "").strip()
        store = SceneEdgeStore(self.config, create=False)
        candidates = [
            edge
            for edge in store.list_edges(include_inactive=True)
            if edge["lifecycle_status"] == "archived"
            and normalized in {edge["source"], edge["target"]}
        ]
        restored: list[str] = []
        stale: list[str] = []
        for edge in candidates:
            result = await self.restore_scene_edge(
                edge["edge_id"], bucket_mgr, reviewed_by="scene_restore"
            )
            if result.get("status") == "restored":
                restored.append(edge["edge_id"])
            else:
                stale.append(edge["edge_id"])
        return {"restored_edge_ids": restored, "stale_edge_ids": stale}

    async def reconcile_lifecycle(self, bucket_mgr) -> dict:
        """Run deterministic startup maintenance without invoking a linker model."""
        proposal_store = self.proposal_store(create=False)
        if proposal_store is None:
            return {"status": "skipped", "reason": "no_scene_edge_store"}
        buckets = await bucket_mgr.list_all(include_archive=True)
        scene_map = {
            str(bucket.get("id") or ""): bucket
            for bucket in buckets
            if isinstance(bucket, dict) and str(bucket.get("id") or "")
        }
        stale_pending = proposal_store.supersede_stale_pending(scene_map)
        edge_store = SceneEdgeStore(self.config, create=False)
        edges = edge_store.list_edges(include_inactive=True)
        stale_scene_ids: set[str] = set()
        archive_deactivated = 0
        missing_deactivated = 0
        for edge in edges:
            if not edge["active"]:
                continue
            source = scene_map.get(edge["source"])
            target = scene_map.get(edge["target"])
            if not _is_authored_scene(source) or not _is_authored_scene(target):
                missing = source is None or target is None
                endpoint = edge["source"] if not _is_authored_scene(source) else edge["target"]
                result = edge_store.deactivate_edge(
                    edge["edge_id"],
                    scene_id=endpoint,
                    reviewer="startup_maintenance",
                    reason="scene_missing" if missing else "scene_archived",
                    lifecycle_status="cancelled" if missing else "archived",
                )
                if result.get("status") == "deactivated":
                    if missing:
                        missing_deactivated += 1
                    else:
                        archive_deactivated += 1
                continue
            if edge["source_hash"] != _scene_hash(source):
                stale_scene_ids.add(edge["source"])
            if edge["target_hash"] != _scene_hash(target):
                stale_scene_ids.add(edge["target"])

        needs_review = 0
        revalidation_ids: list[str] = []
        relink_scene_ids: list[str] = []
        for scene_id in sorted(stale_scene_ids):
            result = proposal_store.refresh_after_scene_content_change(
                scene_id,
                scene_map,
                created_by="startup_maintenance",
            )
            needs_review += int(result.get("formal_edges_needing_review") or 0)
            revalidation_ids.extend(result.get("revalidation_proposal_ids") or [])
            if result.get("normal_relink_required"):
                relink_scene_ids.append(scene_id)

        pending_rows = proposal_store.list(status="pending", limit=1000)
        rejected_rows = proposal_store.list(status="rejected", limit=1000)
        repaired_revalidation_ids: list[str] = []
        for edge in edges:
            if edge["lifecycle_status"] != "needs_review":
                continue
            source = scene_map.get(edge["source"])
            target = scene_map.get(edge["target"])
            if not _is_authored_scene(source) or not _is_authored_scene(target):
                continue
            source_ok, _ = _evidence_is_verbatim(
                str(source.get("content") or ""), edge["source_evidence"]
            )
            target_ok, _ = _evidence_is_verbatim(
                str(target.get("content") or ""), edge["target_evidence"]
            )
            if not source_ok or not target_ok:
                continue
            current_hashes = (_scene_hash(source), _scene_hash(target))

            def proposal_matches_current(row: dict) -> bool:
                return _scene_edge_snapshot_hashes(
                    str(row.get("source_scene_id") or ""),
                    str(row.get("target_scene_id") or ""),
                    str(row.get("relation_type") or ""),
                    str(row.get("anchor_scene_id") or ""),
                    str(row.get("anchor_hash") or ""),
                    str(row.get("candidate_hash") or ""),
                ) == current_hashes

            protected_pending = any(
                str(row.get("canonical_edge_id") or "") == edge["edge_id"]
                and str(row.get("proposal_origin") or "automatic") != "automatic"
                and proposal_matches_current(row)
                for row in pending_rows
            )
            rejected_revalidation = any(
                str(row.get("supersedes_edge_id") or "") == edge["edge_id"]
                and str(row.get("proposal_origin") or "") == "snapshot_revalidation"
                and proposal_matches_current(row)
                for row in rejected_rows
            )
            if protected_pending or rejected_revalidation:
                continue
            result = proposal_store.create_review_proposal(
                source,
                target,
                {
                    "source_scene_id": edge["source"],
                    "target_scene_id": edge["target"],
                    "relation_type": edge["relation_type"],
                    "directionality": edge["directionality"],
                    "confidence": edge["confidence"],
                    "reason": edge["reason"],
                    "source_evidence": edge["source_evidence"],
                    "target_evidence": edge["target_evidence"],
                },
                model="deterministic_revalidation",
                proposal_origin="snapshot_revalidation",
                created_by="startup_maintenance",
                supersedes_edge_id=edge["edge_id"],
                linker_version=edge["linker_version"] or SCENE_LINKER_VERSION,
                nonce=f"startup_repair:{current_hashes[0]}:{current_hashes[1]}",
            )
            proposal = result.get("proposal") or {}
            if proposal.get("proposal_id"):
                proposal_id = str(proposal["proposal_id"])
                repaired_revalidation_ids.append(proposal_id)
                revalidation_ids.append(proposal_id)

        restored_archived: list[str] = []
        for edge in edges:
            if edge["lifecycle_status"] != "archived":
                continue
            result = edge_store.restore_edge(
                edge["edge_id"],
                scene_map,
                reviewer="startup_maintenance",
            )
            if result.get("status") == "restored":
                restored_archived.append(edge["edge_id"])
        return {
            "status": "updated",
            "stale_proposals_superseded": stale_pending,
            "formal_edges_needing_review": needs_review,
            "revalidation_proposal_ids": revalidation_ids,
            "repaired_revalidation_proposal_ids": repaired_revalidation_ids,
            "normal_relink_scene_ids": relink_scene_ids,
            "archive_deactivated": archive_deactivated,
            "missing_scene_deactivated": missing_deactivated,
            "restored_archived_edge_ids": restored_archived,
        }

    @staticmethod
    def _load_providers(cfg: dict, clients: dict[str, Any]) -> list[dict]:
        raw_models = cfg.get("models") if isinstance(cfg.get("models"), list) else []
        if not raw_models and any(cfg.get(key) for key in ("model", "base_url", "api_key")):
            raw_models = [cfg]
        providers = []
        for index, raw in enumerate(raw_models):
            if not isinstance(raw, dict) or not bool(raw.get("enabled", True)):
                continue
            name = str(raw.get("name") or raw.get("model") or f"model_{index + 1}").strip()
            model_env = str(raw.get("model_env") or "").strip()
            base_url_env = str(raw.get("base_url_env") or "").strip()
            api_key_env = str(raw.get("api_key_env") or "OMBRE_SCENE_LINKER_API_KEY").strip()
            model = str((_environment_value(model_env) if model_env else "") or raw.get("model") or "").strip()
            base_url = str(
                (_environment_value(base_url_env) if base_url_env else "")
                or raw.get("base_url")
                or ""
            ).strip().rstrip("/")
            api_key = str(_environment_value(api_key_env) or raw.get("api_key") or "").strip()
            client = clients.get(name)
            if client is None and model and base_url and api_key:
                client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=float(raw.get("timeout_seconds", 60.0)))
            token_parameter = str(raw.get("token_parameter") or "max_tokens").strip()
            if token_parameter not in {"max_tokens", "max_completion_tokens"}:
                token_parameter = "max_tokens"
            providers.append(
                {
                    "name": name,
                    "model": model,
                    "base_url": base_url,
                    "client": client,
                    "max_tokens": max(200, min(int(raw.get("max_tokens", 1100)), 4000)),
                    "token_parameter": token_parameter,
                    "temperature": raw.get("temperature"),
                }
            )
        return providers

    async def link_scene(self, scene_id: str, bucket_mgr, embedding_engine=None) -> dict:
        scene_id = str(scene_id or "").strip()
        if not self.enabled:
            return self._result("disabled", scene_id)
        anchor = await bucket_mgr.get(scene_id)
        if not _is_authored_scene(anchor):
            return self._result("skipped", scene_id, reason="not_active_authored_scene")
        candidates = await self._candidate_scenes(anchor, bucket_mgr, embedding_engine)
        if not candidates:
            return self._result("no_candidates", scene_id, candidate_count=0)
        candidate_map = {str(item.get("id") or ""): item for item in candidates}
        payload = {
            "new_scene": self._scene_payload(anchor, self.source_chars),
            "candidate_scenes": [
                self._scene_payload(candidate, self.candidate_chars) for candidate in candidates
            ],
            "max_edges": self.max_proposals,
        }
        attempts = []
        ready_provider_seen = False
        for provider in self.providers:
            client = provider.get("client")
            if client is None or not provider.get("model"):
                attempts.append({"model": provider["name"], "status": "unavailable"})
                continue
            ready_provider_seen = True
            try:
                async with self._model_semaphore:
                    parsed = await self._call_provider(provider, payload)
            except Exception as exc:
                attempts.append(
                    {
                        "model": provider["name"],
                        "status": "call_failed",
                        "error": str(exc)[:180],
                    }
                )
                continue
            if parsed is None or not isinstance(parsed.get("edges"), list):
                attempts.append({"model": provider["name"], "status": "invalid_json_contract"})
                continue
            raw_edges = parsed.get("edges") or []
            normalized, rejected = self._normalize_edges(anchor, candidate_map, raw_edges)
            if raw_edges and (not normalized or rejected):
                attempts.append(
                    {
                        "model": provider["name"],
                        "status": "evidence_contract_failed",
                        "rejected": len(rejected),
                    }
                )
                continue
            rows = self._store().replace_for_anchor(
                anchor,
                candidate_map,
                normalized,
                model=provider["name"],
                linker_version=self.linker_version,
            )
            attempts.append(
                {
                    "model": provider["name"],
                    "status": "accepted_response",
                    "proposals": len(rows),
                    "rejected": len(rejected),
                }
            )
            return self._result(
                "proposed" if rows else "no_edges",
                scene_id,
                model=provider["name"],
                candidate_count=len(candidates),
                proposal_count=len(rows),
                proposal_ids=[row["proposal_id"] for row in rows],
                rejected_count=len(rejected),
                attempts=attempts,
            )
        return self._result(
            "failed" if ready_provider_seen else "unavailable",
            scene_id,
            candidate_count=len(candidates),
            attempts=attempts,
        )

    def proposal_store(self, *, create: bool = False) -> SceneEdgeProposalStore | None:
        if self._proposal_store is not None:
            return self._proposal_store
        if not create and not SceneEdgeProposalStore.exists(self.config):
            return None
        self._proposal_store = SceneEdgeProposalStore(self.config)
        return self._proposal_store

    def _store(self) -> SceneEdgeProposalStore:
        store = self.proposal_store(create=True)
        if store is None:  # pragma: no cover - create=True always returns a store
            raise RuntimeError("Scene edge proposal store unavailable")
        return store

    @staticmethod
    def _proposal_snapshot_error(
        proposal: dict,
        anchor: dict | None,
        candidate: dict | None,
        *,
        min_confidence: float,
    ) -> str:
        if not isinstance(anchor, dict):
            return "anchor_scene_missing"
        if not isinstance(candidate, dict):
            return "candidate_scene_missing"
        if not _is_authored_scene(anchor):
            return "anchor_scene_inactive"
        if not _is_authored_scene(candidate):
            return "candidate_scene_inactive"
        if str(anchor.get("id") or "") != str(proposal.get("anchor_scene_id") or ""):
            return "anchor_scene_id_mismatch"
        if str(candidate.get("id") or "") != str(proposal.get("candidate_scene_id") or ""):
            return "candidate_scene_id_mismatch"
        if _scene_hash(anchor) != str(proposal.get("anchor_hash") or ""):
            return "anchor_scene_changed"
        if _scene_hash(candidate) != str(proposal.get("candidate_hash") or ""):
            return "candidate_scene_changed"

        anchor_id = str(anchor.get("id") or "")
        candidate_id = str(candidate.get("id") or "")
        source_id = str(proposal.get("source_scene_id") or "")
        target_id = str(proposal.get("target_scene_id") or "")
        if {source_id, target_id} != {anchor_id, candidate_id}:
            return "edge_scene_ids_mismatch"
        relation = str(proposal.get("relation_type") or "")
        directionality = str(proposal.get("directionality") or "")
        if relation not in SCENE_RELATION_TYPES:
            return "relation_not_allowed"
        if relation in SYMMETRIC_SCENE_RELATIONS and directionality != "symmetric":
            return "symmetric_directionality_required"
        if relation not in SYMMETRIC_SCENE_RELATIONS and directionality != "directed":
            return "directed_directionality_required"
        if _clamp(proposal.get("confidence")) < min_confidence:
            return "confidence_below_current_threshold"

        scene_map = {anchor_id: anchor, candidate_id: candidate}
        source = scene_map.get(source_id)
        target = scene_map.get(target_id)
        source_ok, _ = _evidence_is_verbatim(
            str((source or {}).get("content") or ""),
            proposal.get("source_evidence"),
        )
        if not source_ok:
            return "source_evidence_stale"
        target_ok, _ = _evidence_is_verbatim(
            str((target or {}).get("content") or ""),
            proposal.get("target_evidence"),
        )
        if not target_ok:
            return "target_evidence_stale"
        return ""

    @staticmethod
    def _proposal_scene_payload(scene: dict | None, *, include_context: bool) -> dict | None:
        if not isinstance(scene, dict):
            return None
        meta = scene.get("metadata", {}) if isinstance(scene.get("metadata"), dict) else {}
        payload = {
            "scene_id": str(scene.get("id") or ""),
            "name": str(meta.get("name") or scene.get("id") or ""),
            "date": str(meta.get("date") or meta.get("created") or ""),
            "active_authored_scene": _is_authored_scene(scene),
        }
        if include_context:
            payload["content"] = _clip_scene_text(scene.get("content"), 3200)
        return payload

    async def list_proposals(
        self,
        bucket_mgr,
        *,
        status: str = "pending",
        proposal_id: str = "",
        anchor_scene_id: str = "",
        limit: int = 20,
        include_context: bool = False,
    ) -> dict:
        store = self.proposal_store(create=False)
        if store is None:
            return {
                "status": "ok",
                "sidecar_exists": False,
                "count": 0,
                "proposals": [],
                "writes_canonical_memory": False,
                "writes_edge_store": False,
                "writes_scene_edges_on_accept": True,
            }
        normalized_id = str(proposal_id or "").strip()
        if normalized_id:
            row = store.get(normalized_id)
            rows = [row] if row else []
        else:
            rows = store.list(
                status=status,
                anchor_scene_id=anchor_scene_id,
                limit=limit,
            )
        payloads = []
        for row in rows:
            anchor = await bucket_mgr.get(str(row.get("anchor_scene_id") or ""))
            candidate = await bucket_mgr.get(str(row.get("candidate_scene_id") or ""))
            snapshot_error = self._proposal_snapshot_error(
                row,
                anchor,
                candidate,
                min_confidence=self.min_confidence,
            )
            payloads.append(
                {
                    **row,
                    "review_state": snapshot_error or "ready",
                    "anchor_scene": self._proposal_scene_payload(
                        anchor,
                        include_context=include_context,
                    ),
                    "candidate_scene": self._proposal_scene_payload(
                        candidate,
                        include_context=include_context,
                    ),
                }
            )
        return {
            "status": "ok",
            "sidecar_exists": True,
            "count": len(payloads),
            "proposals": payloads,
            "stats": store.stats(),
            "writes_canonical_memory": False,
            "writes_edge_store": False,
            "writes_scene_edges_on_accept": True,
        }

    async def review_proposal(
        self,
        proposal_id: str,
        decision: str,
        confirm: str,
        bucket_mgr,
        edge_store=None,
        *,
        reviewed_by: str = "mcp",
    ) -> dict:
        normalized_id = str(proposal_id or "").strip()
        normalized_decision = str(decision or "").strip().lower()
        if not SCENE_EDGE_PROPOSAL_ID_RE.fullmatch(normalized_id):
            return {"status": "error", "error": "invalid proposal_id"}
        if normalized_decision not in SCENE_EDGE_REVIEW_CONFIRMATIONS:
            return {"status": "error", "error": "decision must be accept or reject"}
        required = SCENE_EDGE_REVIEW_CONFIRMATIONS[normalized_decision]
        if str(confirm or "") != required:
            return {
                "status": "confirmation_required",
                "proposal_id": normalized_id,
                "decision": normalized_decision,
                "required": required,
                "memory_edges_changed": False,
                "scene_edges_changed": False,
            }

        store = self.proposal_store(create=False)
        if store is None:
            return {"status": "not_found", "proposal_id": normalized_id}
        async with self._review_lock:
            proposal = store.get(normalized_id)
            if not proposal:
                return {"status": "not_found", "proposal_id": normalized_id}
            if str(proposal.get("status") or "") != "pending":
                return {
                    "status": "conflict",
                    "error": "proposal is not pending",
                    "proposal": proposal,
                    "memory_edges_changed": False,
                    "scene_edges_changed": False,
                }
            if normalized_decision == "reject":
                updated = store.set_status(
                    normalized_id,
                    "rejected",
                    expected_status="pending",
                    reviewed_by=reviewed_by,
                )
                if not updated or str(updated.get("status") or "") != "rejected":
                    return {
                        "status": "conflict",
                        "error": "proposal is not pending",
                        "proposal": updated or proposal,
                        "memory_edges_changed": False,
                        "scene_edges_changed": False,
                    }
                return {
                    "status": "rejected",
                    "proposal": updated,
                    "canonical_memory_changed": False,
                    "memory_edges_changed": False,
                    "scene_edges_changed": False,
                }

            anchor = await bucket_mgr.get(str(proposal.get("anchor_scene_id") or ""))
            candidate = await bucket_mgr.get(str(proposal.get("candidate_scene_id") or ""))
            snapshot_error = self._proposal_snapshot_error(
                proposal,
                anchor,
                candidate,
                min_confidence=self.min_confidence,
            )
            if snapshot_error:
                return {
                    "status": "stale",
                    "error": snapshot_error,
                    "proposal": proposal,
                    "canonical_memory_changed": False,
                    "memory_edges_changed": False,
                    "scene_edges_changed": False,
                }

            _ = edge_store  # legacy call-site compatibility; Scene edges never use JSONL.
            promoted = store.promote(normalized_id, reviewed_by=reviewed_by)
            if not promoted:
                return {
                    "status": "conflict",
                    "error": "proposal is not pending",
                    "proposal": store.get(normalized_id) or proposal,
                    "memory_edges_changed": False,
                    "scene_edges_changed": False,
                }
            return {
                "status": "accepted",
                "proposal": promoted["proposal"],
                "edge": promoted["edge"],
                "canonical_memory_changed": False,
                "memory_edges_changed": False,
                "scene_edges_changed": True,
            }

    async def _candidate_scenes(self, anchor: dict, bucket_mgr, embedding_engine) -> list[dict]:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        anchor_id = str(anchor.get("id") or "")
        scene_map = {
            str(item.get("id") or ""): item
            for item in all_buckets
            if _is_authored_scene(item) and str(item.get("id") or "") != anchor_id
        }
        if not scene_map:
            return []
        selected: list[dict] = []
        selected_ids: set[str] = set()

        def add(candidate_id: str) -> None:
            candidate_id = str(candidate_id or "").strip()
            if candidate_id in selected_ids or candidate_id not in scene_map:
                return
            selected_ids.add(candidate_id)
            selected.append(scene_map[candidate_id])

        if (
            self.semantic_candidates > 0
            and embedding_engine is not None
            and bool(getattr(embedding_engine, "enabled", False))
        ):
            try:
                similar = await asyncio.wait_for(
                    embedding_engine.search_similar(
                        self._candidate_query(anchor),
                        top_k=max(self.semantic_candidates * 3, 12),
                    ),
                    timeout=self.candidate_timeout_seconds,
                )
            except Exception as exc:
                logger.debug("Scene linker semantic candidate lookup failed: %s", exc)
                similar = []
            semantic_added = 0
            for item in similar or []:
                candidate_id = item[0] if isinstance(item, (list, tuple)) and item else ""
                before = len(selected)
                add(candidate_id)
                if len(selected) > before:
                    semantic_added += 1
                if semantic_added >= self.semantic_candidates or len(selected) >= self.max_candidates:
                    break

        def recent_key(item: dict) -> tuple[str, str]:
            meta = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
            return (
                str(meta.get("created") or meta.get("updated_at") or meta.get("last_active") or ""),
                str(item.get("id") or ""),
            )

        recent_added = 0
        for item in sorted(scene_map.values(), key=recent_key, reverse=True):
            before = len(selected)
            add(str(item.get("id") or ""))
            if len(selected) > before:
                recent_added += 1
            if recent_added >= self.recent_candidates or len(selected) >= self.max_candidates:
                break
        return selected[: self.max_candidates]

    @staticmethod
    def _candidate_query(scene: dict) -> str:
        meta = scene.get("metadata", {}) if isinstance(scene.get("metadata"), dict) else {}
        return "\n".join(
            part
            for part in (
                str(meta.get("name") or "").strip(),
                " | ".join(str(cue) for cue in meta.get("scene_cues", []) or [] if str(cue).strip()),
                _clip_scene_text(scene.get("content"), 2600),
            )
            if part
        )

    @staticmethod
    def _scene_payload(scene: dict, content_limit: int) -> dict:
        meta = scene.get("metadata", {}) if isinstance(scene.get("metadata"), dict) else {}
        return {
            "scene_id": str(scene.get("id") or ""),
            "name": str(meta.get("name") or ""),
            "date": str(meta.get("date") or meta.get("created") or ""),
            "content": _clip_scene_text(scene.get("content"), content_limit),
            "scene_cues": [str(cue) for cue in meta.get("scene_cues", []) or []][:8],
        }

    async def _call_provider(self, provider: dict, payload: dict) -> dict | None:
        options: dict[str, Any] = {
            "model": provider["model"],
            "messages": [
                {"role": "system", "content": SCENE_LINKER_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            provider["token_parameter"]: provider["max_tokens"],
        }
        if provider.get("temperature") is not None:
            options["temperature"] = float(provider["temperature"])
        response = await asyncio.wait_for(
            provider["client"].chat.completions.create(**options),
            timeout=self.timeout_seconds,
        )
        try:
            raw = response.choices[0].message.content if response.choices else ""
        except (AttributeError, IndexError, TypeError):
            raw = ""
        return _parse_json_object(str(raw or ""))

    def _normalize_edges(
        self,
        anchor: dict,
        candidate_map: dict[str, dict],
        raw_edges: list,
    ) -> tuple[list[dict], list[dict]]:
        anchor_id = str(anchor.get("id") or "")
        anchor_content = str(anchor.get("content") or "")
        accepted: dict[str, dict] = {}
        rejected = []
        for item in raw_edges:
            if not isinstance(item, dict):
                rejected.append({"reason": "edge_not_object"})
                continue
            candidate_id = str(item.get("candidate_scene_id") or "").strip()
            candidate = candidate_map.get(candidate_id)
            relation = str(item.get("relation_type") or "").strip()
            orientation = str(item.get("orientation") or "").strip()
            if not candidate:
                rejected.append({"reason": "candidate_not_allowed", "candidate": candidate_id})
                continue
            if relation not in SCENE_RELATION_TYPES:
                rejected.append({"reason": "relation_not_allowed", "relation": relation})
                continue
            if relation in SYMMETRIC_SCENE_RELATIONS:
                if orientation != "symmetric":
                    rejected.append({"reason": "symmetric_orientation_required", "relation": relation})
                    continue
                source_scene_id, target_scene_id = anchor_id, candidate_id
                directionality = "symmetric"
            else:
                if orientation not in DIRECTED_ORIENTATIONS:
                    rejected.append({"reason": "directed_orientation_required", "relation": relation})
                    continue
                directionality = "directed"
                if orientation == "candidate_to_new":
                    source_scene_id, target_scene_id = candidate_id, anchor_id
                else:
                    source_scene_id, target_scene_id = anchor_id, candidate_id
            confidence = _clamp(item.get("confidence"))
            if confidence < self.min_confidence:
                rejected.append({"reason": "confidence_below_threshold", "candidate": candidate_id})
                continue
            reason = re.sub(r"\s+", " ", str(item.get("reason") or "").strip())[:360]
            if len(reason) < 12:
                rejected.append({"reason": "reason_too_thin", "candidate": candidate_id})
                continue
            anchor_ok, anchor_evidence = _evidence_is_verbatim(
                anchor_content,
                item.get("new_scene_evidence"),
            )
            candidate_ok, candidate_evidence = _evidence_is_verbatim(
                str(candidate.get("content") or ""),
                item.get("candidate_scene_evidence"),
            )
            if not anchor_ok or not candidate_ok:
                rejected.append({"reason": "evidence_not_verbatim", "candidate": candidate_id})
                continue
            if source_scene_id == anchor_id:
                source_evidence, target_evidence = anchor_evidence, candidate_evidence
            else:
                source_evidence, target_evidence = candidate_evidence, anchor_evidence
            if relation in SYMMETRIC_SCENE_RELATIONS and target_scene_id < source_scene_id:
                source_scene_id, target_scene_id = target_scene_id, source_scene_id
                source_evidence, target_evidence = target_evidence, source_evidence
            edge = {
                "anchor_scene_id": anchor_id,
                "candidate_scene_id": candidate_id,
                "source_scene_id": source_scene_id,
                "target_scene_id": target_scene_id,
                "relation_type": relation,
                "directionality": directionality,
                "confidence": confidence,
                "reason": reason,
                "source_evidence": source_evidence,
                "target_evidence": target_evidence,
            }
            # One candidate pair should create one review decision, not several
            # overlapping labels for the same two Scenes. Keep the strongest
            # fully evidenced proposal from this model response.
            key = candidate_id
            current = accepted.get(key)
            if current is None or confidence > float(current.get("confidence", 0.0)):
                accepted[key] = edge
        ordered = sorted(
            accepted.values(),
            key=lambda edge: (float(edge["confidence"]), edge["relation_type"]),
            reverse=True,
        )
        return ordered[: self.max_proposals], rejected

    @staticmethod
    def _result(status: str, scene_id: str, **extra: Any) -> dict:
        return {
            "status": status,
            "scene_id": scene_id,
            "canonical_memory_changed": False,
            "memory_edges_changed": False,
            "scene_edges_changed": False,
            **extra,
        }
