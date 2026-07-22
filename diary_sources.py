import hashlib
import re
from datetime import datetime, timezone
from typing import Any


class DiarySourceImporter:
    """Persist immutable diary snapshots as source records, never as memories."""

    SOURCE_VERSION = "diary-source-v1"

    async def import_snapshot(
        self,
        diary: dict[str, Any],
        bucket_mgr,
        *,
        requested_date: str = "",
        requested_title: str = "",
    ) -> dict[str, Any]:
        if not isinstance(diary, dict):
            return {"status": "invalid", "reason": "diary_not_object"}

        content = str(diary.get("content") or "").strip()
        if not content:
            return {"status": "invalid", "reason": "empty_diary"}

        date = self._date_value(diary.get("date")) or self._date_value(requested_date)
        if not date:
            return {"status": "invalid", "reason": "missing_or_invalid_date"}
        title = str(diary.get("title") or requested_title or date).strip()
        diary_id = str(diary.get("id") or diary.get("diary_id") or "").strip()
        revision = str(
            diary.get("revision")
            or diary.get("version")
            or diary.get("updated_at")
            or diary.get("modified_at")
            or ""
        ).strip()
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        source_key = diary_id or f"{date}\n{title}"
        id_digest = hashlib.sha256(
            f"{source_key}\n{content_hash}".encode("utf-8")
        ).hexdigest()[:12]
        bucket_id = f"diary_source_{date.replace('-', '')}_{id_digest}"

        existing = await bucket_mgr.get(bucket_id)
        if existing:
            return self._result(
                "exists",
                bucket_id,
                date=date,
                title=title,
                diary_id=diary_id,
                revision=revision,
                content_hash=content_hash,
                supersedes=str((existing.get("metadata", {}) or {}).get("supersedes_source_record_id") or ""),
            )

        supersedes = await self._latest_previous_snapshot_id(
            bucket_mgr,
            diary_id=diary_id,
            date=date,
            title=title,
            exclude_hash=content_hash,
        )
        imported_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        source_created_at = str(diary.get("created_at") or diary.get("created") or "").strip()
        source_updated_at = str(diary.get("updated_at") or diary.get("modified_at") or "").strip()
        await bucket_mgr.create(
            bucket_id=bucket_id,
            content=content,
            tags=["source_record", "diary_source"],
            importance=1,
            domain=["general"],
            valence=0.5,
            arousal=0.0,
            bucket_type="diary_source",
            name=f"日记原文 · {date} · {title}"[:80],
            source="haven_diary",
            created=imported_at,
            last_active=imported_at,
            updated_at=imported_at,
            confidence=1.0,
            date=date,
            extra_metadata={
                "source_record_version": self.SOURCE_VERSION,
                "source_record_immutable": True,
                "diary_id": diary_id,
                "diary_title": title,
                "diary_date": date,
                "diary_revision": revision,
                "diary_created_at": source_created_at,
                "diary_updated_at": source_updated_at,
                "content_hash": content_hash,
                "hash_algorithm": "sha256",
                "imported_at": imported_at,
                "supersedes_source_record_id": supersedes,
            },
        )
        return self._result(
            "created",
            bucket_id,
            date=date,
            title=title,
            diary_id=diary_id,
            revision=revision,
            content_hash=content_hash,
            supersedes=supersedes,
        )

    async def _latest_previous_snapshot_id(
        self,
        bucket_mgr,
        *,
        diary_id: str,
        date: str,
        title: str,
        exclude_hash: str,
    ) -> str:
        try:
            buckets = await bucket_mgr.list_all(include_archive=True)
        except Exception:
            return ""
        matches = []
        for bucket in buckets or []:
            if not isinstance(bucket, dict):
                continue
            meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
            tags = {str(tag or "").strip().lower() for tag in meta.get("tags", []) or []}
            if meta.get("type") != "diary_source" and "diary_source" not in tags:
                continue
            if str(meta.get("content_hash") or "") == exclude_hash:
                continue
            same_source = (
                bool(diary_id)
                and str(meta.get("diary_id") or "").strip() == diary_id
            ) or (
                not diary_id
                and str(meta.get("diary_date") or meta.get("date") or "").strip() == date
                and str(meta.get("diary_title") or "").strip() == title
            )
            if same_source:
                matches.append(bucket)
        matches.sort(
            key=lambda bucket: str(
                (bucket.get("metadata", {}) or {}).get("imported_at")
                or (bucket.get("metadata", {}) or {}).get("updated_at")
                or ""
            ),
            reverse=True,
        )
        return str(matches[0].get("id") or "") if matches else ""

    @staticmethod
    def _date_value(value: Any) -> str:
        text = str(value or "").strip()
        return text if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text) else ""

    @staticmethod
    def _result(
        status: str,
        bucket_id: str,
        *,
        date: str,
        title: str,
        diary_id: str,
        revision: str,
        content_hash: str,
        supersedes: str,
    ) -> dict[str, Any]:
        return {
            "status": status,
            "id": bucket_id,
            "evidence_ref": {
                "source_record_id": bucket_id,
                "content_hash": content_hash,
                "hash_algorithm": "sha256",
            },
            "layer": "source_record",
            "source": "haven_diary",
            "date": date,
            "title": title,
            "diary_id": diary_id,
            "revision": revision,
            "content_hash": content_hash,
            "supersedes_source_record_id": supersedes,
            "ordinary_recall": False,
            "model_rewritten": False,
        }
