from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memory_evidence import (
    evidence_is_verbatim,
    legacy_moment_content_hash,
    scene_content_hash,
)
from memory_moments import parse_bucket_moments


LEGACY_REVIEW_DECISIONS = frozenset(
    {"same_event", "superseded", "still_current", "archive"}
)
LEGACY_REVIEW_STATUSES = frozenset(
    {"pending", "accepted", "rejected", "superseded", "stale"}
)
LEGACY_REVIEW_POST_LIFECYCLES = frozenset({"", "archive_after_bridge"})
ACCEPT_LEGACY_REVIEW_CONFIRM = "ACCEPT_LEGACY_REVIEW"
REJECT_LEGACY_REVIEW_CONFIRM = "REJECT_LEGACY_REVIEW"


class LegacyMemoryReviewStore:
    """Review-only legacy lifecycle and Scene-bridge proposals.

    Accepting a card records a human decision. It never archives a bucket,
    writes a bridge, changes a Scene, or updates recall state.
    """

    def __init__(self, config: dict, *, create: bool = True):
        state_dir = str(
            config.get("state_dir")
            or os.path.join(
                os.path.dirname(os.path.abspath(config.get("buckets_dir", "buckets"))),
                "state",
            )
        )
        self.db_path = os.path.join(state_dir, "legacy_memory_reviews.sqlite")
        if create:
            os.makedirs(state_dir, exist_ok=True)
            self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS legacy_memory_reviews (
                    proposal_id TEXT PRIMARY KEY,
                    legacy_bucket_id TEXT NOT NULL,
                    legacy_moment_id TEXT NOT NULL DEFAULT '',
                    decision TEXT NOT NULL,
                    scene_id TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 0,
                    reason TEXT NOT NULL,
                    legacy_evidence TEXT NOT NULL DEFAULT '',
                    scene_evidence TEXT NOT NULL DEFAULT '',
                    bucket_hash TEXT NOT NULL,
                    legacy_hash TEXT NOT NULL DEFAULT '',
                    scene_hash TEXT NOT NULL DEFAULT '',
                    successor_ref TEXT NOT NULL DEFAULT '',
                    post_review_lifecycle TEXT NOT NULL DEFAULT '',
                    proposed_by TEXT NOT NULL,
                    proposal_source TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    apply_state TEXT NOT NULL DEFAULT 'not_applied',
                    apply_error TEXT NOT NULL DEFAULT '',
                    apply_receipt_json TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    reviewed_at TEXT,
                    reviewed_by TEXT,
                    applied_at TEXT,
                    applied_by TEXT
                )
                """
            )
            columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(legacy_memory_reviews)"
                ).fetchall()
            }
            for name, ddl in (
                ("apply_error", "TEXT NOT NULL DEFAULT ''"),
                ("apply_receipt_json", "TEXT NOT NULL DEFAULT ''"),
                ("applied_at", "TEXT"),
                ("applied_by", "TEXT"),
            ):
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE legacy_memory_reviews ADD COLUMN {name} {ddl}"
                    )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_legacy_memory_reviews_status "
                "ON legacy_memory_reviews(status, updated_at DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_legacy_memory_reviews_bucket "
                "ON legacy_memory_reviews(legacy_bucket_id, updated_at DESC)"
            )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_legacy_memory_reviews_one_pending "
                "ON legacy_memory_reviews(legacy_bucket_id) WHERE status = 'pending'"
            )
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict | None:
        return dict(row) if row is not None else None

    def propose(
        self,
        *,
        legacy_bucket: dict,
        decision: str,
        reason: str,
        proposed_by: str,
        proposal_source: str = "manual",
        legacy_moment: dict | None = None,
        scene: dict | None = None,
        legacy_evidence: str = "",
        scene_evidence: str = "",
        confidence: float = 0.0,
        successor_ref: str = "",
        post_review_lifecycle: str = "",
    ) -> dict:
        normalized_decision = str(decision or "").strip().lower()
        if normalized_decision not in LEGACY_REVIEW_DECISIONS:
            raise ValueError("invalid legacy review decision")
        legacy_bucket_id = str((legacy_bucket or {}).get("id") or "").strip()
        if not legacy_bucket_id:
            raise ValueError("legacy bucket is required")
        if _is_canonical_scene(legacy_bucket):
            raise ValueError("review source must be a legacy bucket")
        normalized_reason = str(reason or "").strip()
        if not normalized_reason:
            raise ValueError("review reason is required")
        reviewer = str(proposed_by or "").strip()
        if not reviewer:
            raise ValueError("proposed_by is required")
        source = str(proposal_source or "manual").strip()[:80] or "manual"
        lifecycle = str(post_review_lifecycle or "").strip().lower()
        if lifecycle not in LEGACY_REVIEW_POST_LIFECYCLES:
            raise ValueError("invalid post-review lifecycle")

        legacy_moment_id = ""
        legacy_hash = ""
        scene_id = ""
        scene_hash = ""
        normalized_legacy_evidence = ""
        normalized_scene_evidence = ""
        bounded_confidence = _clamp(confidence)
        if normalized_decision == "same_event":
            legacy_moment_id = str((legacy_moment or {}).get("moment_id") or "").strip()
            scene_id = str((scene or {}).get("id") or "").strip()
            if not legacy_moment_id or not scene_id:
                raise ValueError("same_event requires a legacy Moment and Scene")
            if str((legacy_moment or {}).get("bucket_id") or "") != legacy_bucket_id:
                raise ValueError("legacy Moment does not belong to review bucket")
            if not _is_canonical_scene(scene):
                raise ValueError("same_event requires a canonical authored Scene")
            normalized_legacy_evidence = str(legacy_evidence or "").strip()
            normalized_scene_evidence = str(scene_evidence or "").strip()
            if not evidence_is_verbatim(
                str((legacy_moment or {}).get("text") or ""),
                normalized_legacy_evidence,
            ):
                raise ValueError("legacy evidence is not verbatim")
            if not evidence_is_verbatim(
                str((scene or {}).get("content") or ""),
                normalized_scene_evidence,
            ):
                raise ValueError("Scene evidence is not verbatim")
            legacy_hash = legacy_moment_content_hash(legacy_moment or {})
            scene_hash = scene_content_hash(scene or {})
            if bounded_confidence <= 0:
                raise ValueError("same_event confidence must be positive")
        elif any(
            value
            for value in (
                legacy_moment,
                scene,
                legacy_evidence,
                scene_evidence,
                lifecycle,
            )
        ):
            raise ValueError("only same_event accepts bridge fields")

        bucket_hash = _legacy_bucket_hash(legacy_bucket)
        identity = json.dumps(
            {
                "legacy_bucket_id": legacy_bucket_id,
                "legacy_moment_id": legacy_moment_id,
                "decision": normalized_decision,
                "scene_id": scene_id,
                "bucket_hash": bucket_hash,
                "legacy_hash": legacy_hash,
                "scene_hash": scene_hash,
                "successor_ref": str(successor_ref or "").strip(),
                "post_review_lifecycle": lifecycle,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        proposal_id = "legacy_review_" + hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()[:24]
        now = _now()
        row = {
            "proposal_id": proposal_id,
            "legacy_bucket_id": legacy_bucket_id,
            "legacy_moment_id": legacy_moment_id,
            "decision": normalized_decision,
            "scene_id": scene_id,
            "confidence": bounded_confidence,
            "reason": normalized_reason[:800],
            "legacy_evidence": normalized_legacy_evidence[:320],
            "scene_evidence": normalized_scene_evidence[:320],
            "bucket_hash": bucket_hash,
            "legacy_hash": legacy_hash,
            "scene_hash": scene_hash,
            "successor_ref": str(successor_ref or "").strip()[:300],
            "post_review_lifecycle": lifecycle,
            "proposed_by": reviewer[:80],
            "proposal_source": source,
            "created_at": now,
            "updated_at": now,
        }
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE legacy_memory_reviews
                   SET status = 'superseded', updated_at = ?
                 WHERE legacy_bucket_id = ? AND status = 'pending'
                   AND proposal_id <> ?
                """,
                (now, legacy_bucket_id, proposal_id),
            )
            connection.execute(
                """
                INSERT INTO legacy_memory_reviews (
                    proposal_id, legacy_bucket_id, legacy_moment_id,
                    decision, scene_id, confidence, reason,
                    legacy_evidence, scene_evidence,
                    bucket_hash, legacy_hash, scene_hash,
                    successor_ref, post_review_lifecycle,
                    proposed_by, proposal_source,
                    status, apply_state, created_at, updated_at
                ) VALUES (
                    :proposal_id, :legacy_bucket_id, :legacy_moment_id,
                    :decision, :scene_id, :confidence, :reason,
                    :legacy_evidence, :scene_evidence,
                    :bucket_hash, :legacy_hash, :scene_hash,
                    :successor_ref, :post_review_lifecycle,
                    :proposed_by, :proposal_source,
                    'pending', 'not_applied', :created_at, :updated_at
                )
                ON CONFLICT(proposal_id) DO UPDATE SET
                    confidence = excluded.confidence,
                    reason = excluded.reason,
                    legacy_evidence = excluded.legacy_evidence,
                    scene_evidence = excluded.scene_evidence,
                    successor_ref = excluded.successor_ref,
                    proposed_by = excluded.proposed_by,
                    proposal_source = excluded.proposal_source,
                    updated_at = excluded.updated_at
                """,
                row,
            )
            connection.commit()
            stored = connection.execute(
                "SELECT * FROM legacy_memory_reviews WHERE proposal_id = ?",
                (proposal_id,),
            ).fetchone()
            return dict(stored)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get(self, proposal_id: str) -> dict | None:
        normalized = str(proposal_id or "").strip()
        if not normalized or not os.path.exists(self.db_path):
            return None
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM legacy_memory_reviews WHERE proposal_id = ?",
                (normalized,),
            ).fetchone()
            return self._row(row)
        finally:
            connection.close()

    def list(
        self,
        *,
        status: str = "pending",
        proposal_id: str = "",
        limit: int = 100,
    ) -> list[dict]:
        if not os.path.exists(self.db_path):
            return []
        normalized_status = str(status or "pending").strip().lower()
        if normalized_status != "all" and normalized_status not in LEGACY_REVIEW_STATUSES:
            raise ValueError("invalid legacy review status")
        bounded = max(1, min(int(limit or 100), 1000))
        clauses: list[str] = []
        params: list[Any] = []
        if normalized_status != "all":
            clauses.append("status = ?")
            params.append(normalized_status)
        normalized_id = str(proposal_id or "").strip()
        if normalized_id:
            clauses.append("proposal_id = ?")
            params.append(normalized_id)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        params.append(bounded)
        connection = self._connect()
        try:
            rows = connection.execute(
                f"""
                SELECT * FROM legacy_memory_reviews{where}
                 ORDER BY updated_at DESC, proposal_id ASC LIMIT ?
                """,
                params,
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            connection.close()

    async def validate(self, proposal: dict, bucket_mgr: Any) -> dict:
        legacy_bucket_id = str((proposal or {}).get("legacy_bucket_id") or "")
        legacy_bucket = await bucket_mgr.get(legacy_bucket_id)
        if not legacy_bucket:
            return {"valid": False, "reason": "legacy_bucket_missing"}
        if _is_canonical_scene(legacy_bucket):
            return {"valid": False, "reason": "legacy_bucket_became_scene"}
        if _legacy_bucket_hash(legacy_bucket) != str(proposal.get("bucket_hash") or ""):
            return {"valid": False, "reason": "legacy_bucket_hash_changed"}
        if not _is_active_bucket(legacy_bucket):
            return {"valid": False, "reason": "legacy_bucket_not_active"}

        decision = str(proposal.get("decision") or "")
        context = {
            "legacy_bucket": _bucket_context(legacy_bucket),
            "scene": None,
            "legacy_moment": None,
        }
        if decision != "same_event":
            return {"valid": True, "reason": "", "context": context}

        moments = {
            str(moment.get("moment_id") or ""): moment
            for moment in parse_bucket_moments(legacy_bucket)
            if moment.get("moment_id")
        }
        legacy_moment_id = str(proposal.get("legacy_moment_id") or "")
        legacy_moment = moments.get(legacy_moment_id)
        if not legacy_moment:
            return {"valid": False, "reason": "legacy_moment_missing", "context": context}
        if legacy_moment_content_hash(legacy_moment) != str(proposal.get("legacy_hash") or ""):
            return {
                "valid": False,
                "reason": "legacy_moment_hash_changed",
                "context": context,
            }
        if not evidence_is_verbatim(
            str(legacy_moment.get("text") or ""),
            proposal.get("legacy_evidence"),
        ):
            return {
                "valid": False,
                "reason": "legacy_evidence_changed",
                "context": context,
            }

        scene = await bucket_mgr.get(str(proposal.get("scene_id") or ""))
        if not scene or not _is_canonical_scene(scene):
            return {"valid": False, "reason": "scene_missing", "context": context}
        if not _is_active_bucket(scene):
            return {"valid": False, "reason": "scene_not_active", "context": context}
        if scene_content_hash(scene) != str(proposal.get("scene_hash") or ""):
            return {"valid": False, "reason": "scene_hash_changed", "context": context}
        if not evidence_is_verbatim(
            str(scene.get("content") or ""),
            proposal.get("scene_evidence"),
        ):
            return {"valid": False, "reason": "scene_evidence_changed", "context": context}
        context.update(
            {
                "scene": _bucket_context(scene),
                "legacy_moment": {
                    "moment_id": legacy_moment_id,
                    "section": str(legacy_moment.get("section") or ""),
                    "preview": _preview(legacy_moment.get("text")),
                },
            }
        )
        return {"valid": True, "reason": "", "context": context}

    async def list_for_review(
        self,
        bucket_mgr: Any,
        *,
        status: str = "pending",
        proposal_id: str = "",
        limit: int = 100,
        include_context: bool = False,
    ) -> dict:
        proposals = self.list(status=status, proposal_id=proposal_id, limit=limit)
        payloads = []
        for proposal in proposals:
            item = dict(proposal)
            validation = await self.validate(item, bucket_mgr)
            item["review_state"] = (
                "ready" if validation.get("valid") else "stale"
            )
            item["validation_reason"] = str(validation.get("reason") or "")
            if include_context:
                item["context"] = validation.get("context")
            payloads.append(item)
        return {
            "status": "ok",
            "proposals": payloads,
            "count": len(payloads),
            "actions_applied": False,
        }

    async def review(
        self,
        proposal_id: str,
        decision: str,
        confirm: str,
        bucket_mgr: Any,
        *,
        reviewed_by: str,
    ) -> dict:
        normalized_id = str(proposal_id or "").strip()
        normalized_decision = str(decision or "").strip().lower()
        if normalized_decision not in {"accept", "reject"}:
            return {"status": "error", "error": "decision must be accept or reject"}
        expected_confirm = (
            ACCEPT_LEGACY_REVIEW_CONFIRM
            if normalized_decision == "accept"
            else REJECT_LEGACY_REVIEW_CONFIRM
        )
        if str(confirm or "").strip() != expected_confirm:
            return {
                "status": "confirmation_required",
                "required_confirm": expected_confirm,
            }
        reviewer = str(reviewed_by or "").strip()
        if not reviewer:
            return {"status": "error", "error": "reviewed_by is required"}
        proposal = self.get(normalized_id)
        if proposal is None:
            return {"status": "not_found", "proposal_id": normalized_id}
        if str(proposal.get("status") or "") != "pending":
            return {
                "status": "conflict",
                "proposal": proposal,
                "error": "proposal is not pending",
            }

        validation = await self.validate(proposal, bucket_mgr)
        if not validation.get("valid"):
            stale = self._set_review_status(
                normalized_id,
                "stale",
                reviewed_by=reviewer,
            )
            return {
                "status": "stale",
                "proposal": stale,
                "validation_reason": validation.get("reason"),
                "actions_applied": False,
            }
        target_status = "accepted" if normalized_decision == "accept" else "rejected"
        reviewed = self._set_review_status(
            normalized_id,
            target_status,
            reviewed_by=reviewer,
        )
        return {
            "status": target_status,
            "proposal": reviewed,
            "prepared_action": _prepared_action(reviewed or {}),
            "actions_applied": False,
        }

    def _set_review_status(
        self,
        proposal_id: str,
        status: str,
        *,
        reviewed_by: str,
    ) -> dict | None:
        if status not in {"accepted", "rejected", "stale"}:
            raise ValueError("invalid review status")
        now = _now()
        connection = self._connect()
        try:
            connection.execute(
                """
                UPDATE legacy_memory_reviews
                   SET status = ?, updated_at = ?, reviewed_at = ?, reviewed_by = ?
                 WHERE proposal_id = ? AND status = 'pending'
                """,
                (status, now, now, str(reviewed_by)[:80], proposal_id),
            )
            connection.commit()
            row = connection.execute(
                "SELECT * FROM legacy_memory_reviews WHERE proposal_id = ?",
                (proposal_id,),
            ).fetchone()
            return self._row(row)
        finally:
            connection.close()

    def stats(self) -> dict:
        if not os.path.exists(self.db_path):
            return {
                "total": 0,
                "pending": 0,
                "accepted": 0,
                "rejected": 0,
                "superseded": 0,
                "stale": 0,
            }
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM legacy_memory_reviews GROUP BY status"
            ).fetchall()
        finally:
            connection.close()
        counts = {str(row["status"]): int(row["count"]) for row in rows}
        return {
            "total": sum(counts.values()),
            **{status: counts.get(status, 0) for status in LEGACY_REVIEW_STATUSES},
        }

    def begin_apply(
        self,
        proposal_id: str,
        *,
        applied_by: str,
        receipt: dict,
    ) -> dict | None:
        """Reserve one accepted proposal for an explicit publisher."""
        reviewer = str(applied_by or "").strip()
        if not reviewer:
            raise ValueError("applied_by is required")
        now = _now()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE legacy_memory_reviews
                   SET apply_state = 'applying', apply_error = '',
                       apply_receipt_json = ?, applied_by = ?, updated_at = ?
                 WHERE proposal_id = ? AND status = 'accepted'
                   AND apply_state IN ('not_applied', 'failed')
                """,
                (
                    json.dumps(receipt or {}, ensure_ascii=False, sort_keys=True),
                    reviewer[:80],
                    now,
                    str(proposal_id or "").strip(),
                ),
            )
            connection.commit()
            if cursor.rowcount != 1:
                return self.get(proposal_id)
            row = connection.execute(
                "SELECT * FROM legacy_memory_reviews WHERE proposal_id = ?",
                (str(proposal_id or "").strip(),),
            ).fetchone()
            return self._row(row)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def finish_apply(self, proposal_id: str, *, receipt: dict) -> dict | None:
        """Record a completed publisher action without changing review content."""
        now = _now()
        connection = self._connect()
        try:
            connection.execute(
                """
                UPDATE legacy_memory_reviews
                   SET apply_state = 'applied', apply_error = '',
                       apply_receipt_json = ?, applied_at = ?, updated_at = ?
                 WHERE proposal_id = ? AND status = 'accepted'
                   AND apply_state = 'applying'
                """,
                (
                    json.dumps(receipt or {}, ensure_ascii=False, sort_keys=True),
                    now,
                    now,
                    str(proposal_id or "").strip(),
                ),
            )
            connection.commit()
            row = connection.execute(
                "SELECT * FROM legacy_memory_reviews WHERE proposal_id = ?",
                (str(proposal_id or "").strip(),),
            ).fetchone()
            return self._row(row)
        finally:
            connection.close()

    def fail_apply(self, proposal_id: str, *, error: str) -> dict | None:
        """Keep a failed archive retryable while retaining its backup receipt."""
        now = _now()
        connection = self._connect()
        try:
            connection.execute(
                """
                UPDATE legacy_memory_reviews
                   SET apply_state = 'failed', apply_error = ?, updated_at = ?
                 WHERE proposal_id = ? AND status = 'accepted'
                   AND apply_state = 'applying'
                """,
                (
                    str(error or "apply_failed")[:300],
                    now,
                    str(proposal_id or "").strip(),
                ),
            )
            connection.commit()
            row = connection.execute(
                "SELECT * FROM legacy_memory_reviews WHERE proposal_id = ?",
                (str(proposal_id or "").strip(),),
            ).fetchone()
            return self._row(row)
        finally:
            connection.close()


def _prepared_action(proposal: dict) -> dict:
    decision = str(proposal.get("decision") or "")
    if decision == "same_event":
        return {
            "kind": "reviewed_bridge",
            "scene_id": str(proposal.get("scene_id") or ""),
            "legacy_moment_id": str(proposal.get("legacy_moment_id") or ""),
            "relation_type": "same_event",
            "confidence": float(proposal.get("confidence") or 0.0),
            "scene_evidence": str(proposal.get("scene_evidence") or ""),
            "legacy_evidence": str(proposal.get("legacy_evidence") or ""),
            "post_review_lifecycle": str(
                proposal.get("post_review_lifecycle") or ""
            ),
            "publish_required": True,
        }
    return {
        "kind": "legacy_lifecycle",
        "legacy_bucket_id": str(proposal.get("legacy_bucket_id") or ""),
        "decision": decision,
        "successor_ref": str(proposal.get("successor_ref") or ""),
        "apply_required": decision in {"archive", "superseded"},
    }


def _legacy_bucket_hash(bucket: dict) -> str:
    metadata = bucket.get("metadata") if isinstance(bucket.get("metadata"), dict) else {}
    payload = {
        "id": str(bucket.get("id") or ""),
        "content": str(bucket.get("content") or ""),
        "name": str(metadata.get("name") or ""),
        "type": str(metadata.get("type") or ""),
        "source": str(metadata.get("source") or ""),
        "tags": sorted(str(tag) for tag in metadata.get("tags", []) or []),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _is_canonical_scene(bucket: dict | None) -> bool:
    if not isinstance(bucket, dict):
        return False
    metadata = bucket.get("metadata") if isinstance(bucket.get("metadata"), dict) else {}
    return str(metadata.get("memory_value_source") or "") == "authored_scene"


def _is_active_bucket(bucket: dict) -> bool:
    metadata = bucket.get("metadata") if isinstance(bucket.get("metadata"), dict) else {}
    if metadata.get("active") is False or bool(metadata.get("deprecated")):
        return False
    if str(metadata.get("type") or "").strip().lower() in {"archive", "archived"}:
        return False
    path = Path(str(bucket.get("path") or ""))
    return not any(part.lower() in {"archive", "archived"} for part in path.parts)


def _bucket_context(bucket: dict) -> dict:
    metadata = bucket.get("metadata") if isinstance(bucket.get("metadata"), dict) else {}
    return {
        "id": str(bucket.get("id") or ""),
        "name": str(metadata.get("name") or bucket.get("id") or ""),
        "type": str(metadata.get("type") or ""),
        "preview": _preview(bucket.get("content")),
    }


def _preview(value: Any, limit: int = 320) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _clamp(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return max(0.0, min(1.0, round(number, 3)))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
