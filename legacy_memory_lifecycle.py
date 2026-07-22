from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from legacy_memory_review import LegacyMemoryReviewStore


ARCHIVE_REVIEWED_LEGACY_CONFIRM = "ARCHIVE_REVIEWED_LEGACY"


class LegacyMemoryLifecyclePublisher:
    """Apply one accepted legacy archive decision with backup and audit.

    This publisher only moves the original bucket to the archive tree. It never
    deletes source text, writes a Scene bridge, or changes Gateway recall rules.
    """

    def __init__(self, config: dict, review_store: LegacyMemoryReviewStore):
        self.review_store = review_store
        self.buckets_root = Path(
            str(config.get("buckets_dir") or "buckets")
        ).resolve()
        state_dir = Path(
            str(config.get("state_dir") or self.buckets_root.parent / "state")
        ).resolve()
        self.backup_root = state_dir / "legacy_memory_review_backups"

    async def publish_archive(
        self,
        proposal_id: str,
        bucket_mgr: Any,
        *,
        confirm: str = "",
        applied_by: str,
        dry_run: bool = False,
    ) -> dict:
        normalized_id = str(proposal_id or "").strip()
        proposal = self.review_store.get(normalized_id)
        if proposal is None:
            return {"status": "not_found", "proposal_id": normalized_id}
        if str(proposal.get("status") or "") != "accepted":
            return {
                "status": "conflict",
                "error": "proposal must be accepted before apply",
                "proposal": proposal,
                "actions_applied": False,
            }
        if str(proposal.get("decision") or "") != "archive":
            return {
                "status": "conflict",
                "error": "publisher only applies archive decisions",
                "proposal": proposal,
                "actions_applied": False,
            }
        reviewer = str(applied_by or "").strip()
        if not reviewer:
            return {
                "status": "error",
                "error": "applied_by is required",
                "actions_applied": False,
            }

        apply_state = str(proposal.get("apply_state") or "not_applied")
        if apply_state == "applied":
            return {
                "status": "applied",
                "proposal": proposal,
                "receipt": _receipt(proposal),
                "actions_applied": False,
                "idempotent": True,
            }

        bucket = await bucket_mgr.get(str(proposal.get("legacy_bucket_id") or ""))
        if apply_state in {"applying", "failed"} and _is_archived_storage(bucket):
            receipt = _receipt(proposal)
            receipt.update(_archived_receipt(bucket))
            if apply_state == "failed":
                proposal = self.review_store.begin_apply(
                    normalized_id,
                    applied_by=reviewer,
                    receipt=receipt,
                ) or proposal
            completed = self.review_store.finish_apply(
                normalized_id,
                receipt=receipt,
            )
            return {
                "status": "applied",
                "proposal": completed,
                "receipt": receipt,
                "actions_applied": False,
                "idempotent": True,
                "reconciled": True,
            }

        validation = await self.review_store.validate(proposal, bucket_mgr)
        if not validation.get("valid"):
            return {
                "status": "stale",
                "validation_reason": validation.get("reason"),
                "proposal": proposal,
                "actions_applied": False,
            }
        prepared_action = {
            "kind": "legacy_archive",
            "legacy_bucket_id": str(proposal.get("legacy_bucket_id") or ""),
            "source_preserved": True,
            "physical_delete": False,
            "restore_via": "BucketManager.activate",
        }
        if dry_run:
            return {
                "status": "ready",
                "proposal": proposal,
                "prepared_action": prepared_action,
                "actions_applied": False,
            }
        if str(confirm or "").strip() != ARCHIVE_REVIEWED_LEGACY_CONFIRM:
            return {
                "status": "confirmation_required",
                "required_confirm": ARCHIVE_REVIEWED_LEGACY_CONFIRM,
                "prepared_action": prepared_action,
                "actions_applied": False,
            }

        try:
            source_path = self._source_path(bucket)
            backup_path, backup_sha256 = self._backup(source_path, normalized_id)
        except Exception as exc:
            return {
                "status": "error",
                "error": str(exc),
                "actions_applied": False,
            }
        receipt = {
            "kind": "legacy_archive",
            "legacy_bucket_id": prepared_action["legacy_bucket_id"],
            "original_path": str(source_path),
            "backup_path": str(backup_path),
            "backup_sha256": backup_sha256,
            "source_preserved": True,
            "physical_delete": False,
        }
        applying = self.review_store.begin_apply(
            normalized_id,
            applied_by=reviewer,
            receipt=receipt,
        )
        if not applying or str(applying.get("apply_state") or "") != "applying":
            return {
                "status": "conflict",
                "error": "proposal could not enter applying state",
                "proposal": applying or proposal,
                "actions_applied": False,
            }

        try:
            archived = bool(await bucket_mgr.archive(prepared_action["legacy_bucket_id"]))
            current = await bucket_mgr.get(prepared_action["legacy_bucket_id"])
            if not archived and not _is_archived_storage(current):
                raise RuntimeError("bucket archive failed")
            if not _is_archived_storage(current):
                raise RuntimeError("archive verification failed")
            receipt.update(_archived_receipt(current))
            completed = self.review_store.finish_apply(
                normalized_id,
                receipt=receipt,
            )
            if not completed or str(completed.get("apply_state") or "") != "applied":
                raise RuntimeError("archive audit finalization failed")
        except Exception as exc:
            failed = self.review_store.fail_apply(
                normalized_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            return {
                "status": "error",
                "error": str(exc),
                "proposal": failed,
                "receipt": receipt,
                "actions_applied": _is_archived_storage(
                    await bucket_mgr.get(prepared_action["legacy_bucket_id"])
                ),
            }
        return {
            "status": "applied",
            "proposal": completed,
            "receipt": receipt,
            "actions_applied": True,
            "idempotent": False,
        }

    def _source_path(self, bucket: dict | None) -> Path:
        source_path = Path(str((bucket or {}).get("path") or "")).resolve()
        if not source_path.is_file():
            raise ValueError("legacy bucket source file is missing")
        try:
            source_path.relative_to(self.buckets_root)
        except ValueError as exc:
            raise ValueError("legacy bucket path is outside buckets root") from exc
        return source_path

    def _backup(self, source_path: Path, proposal_id: str) -> tuple[Path, str]:
        source_sha256 = _sha256_file(source_path)
        target_dir = self.backup_root / proposal_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{source_sha256}.md"
        if not target_path.exists():
            shutil.copy2(source_path, target_path)
        if _sha256_file(target_path) != source_sha256:
            raise RuntimeError("legacy archive backup verification failed")
        return target_path, source_sha256


def _receipt(proposal: dict | None) -> dict:
    raw = str((proposal or {}).get("apply_receipt_json") or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _is_archived_storage(bucket: dict | None) -> bool:
    if not isinstance(bucket, dict):
        return False
    metadata = bucket.get("metadata") if isinstance(bucket.get("metadata"), dict) else {}
    archived_type = str(metadata.get("type") or "").strip().lower() in {
        "archive",
        "archived",
    }
    archived_path = any(
        part.lower() in {"archive", "archived"}
        for part in Path(str(bucket.get("path") or "")).parts
    )
    return archived_type and archived_path


def _archived_receipt(bucket: dict | None) -> dict:
    return {
        "archived_path": str((bucket or {}).get("path") or ""),
        "archived": _is_archived_storage(bucket),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
