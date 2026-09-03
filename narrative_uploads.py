from __future__ import annotations

import hashlib
import json
import os
import re
import zipfile
from datetime import datetime, timezone
from html import unescape
from io import BytesIO
from pathlib import Path
from typing import Any


UPLOAD_ID_RE = re.compile(r"^upload_[0-9a-f]{32}$")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_EXTRACTED_CHARS = 80_000
_TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".csv", ".json", ".yaml", ".yml", ".log"}
_TEXT_MIME_TYPES = {
    "application/json",
    "application/x-yaml",
    "text/csv",
    "text/markdown",
    "text/plain",
    "text/x-markdown",
    "text/yaml",
}


def _safe_filename(value: Any) -> str:
    name = Path(str(value or "upload").replace("\\", "/")).name.strip()
    name = re.sub(r"[\x00-\x1f\x7f]", "", name)
    return name[:240] or "upload"


def _extract_docx(raw: bytes) -> str:
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        info = archive.getinfo("word/document.xml")
        if info.file_size > 2_000_000:
            raise ValueError("docx_text_too_large")
        document = archive.read(info).decode("utf-8")
    text = re.sub(r"</w:p\s*>", "\n", document)
    text = re.sub(r"<w:tab\s*/>", "\t", text)
    text = re.sub(r"<[^>]+>", "", text)
    return unescape(text).strip()


def extract_upload_text(raw: bytes, filename: str, content_type: str) -> tuple[str, str]:
    suffix = Path(filename).suffix.lower()
    mime = str(content_type or "").split(";", 1)[0].strip().lower()
    try:
        if suffix == ".docx" or mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = _extract_docx(raw)
        elif suffix in _TEXT_EXTENSIONS or mime in _TEXT_MIME_TYPES or mime.startswith("text/"):
            text = raw.decode("utf-8-sig")
        else:
            return "", "metadata_only"
    except (UnicodeDecodeError, KeyError, OSError, ValueError, zipfile.BadZipFile):
        return "", "unavailable"
    text = text.replace("\x00", "").strip()
    if not text:
        return "", "empty"
    if len(text) > MAX_EXTRACTED_CHARS:
        return text[:MAX_EXTRACTED_CHARS], "truncated"
    return text, "extracted"


class NarrativeUploadStore:
    """Content-addressed local files that may be bound to Narrative Rolls."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        repo_root = Path(__file__).resolve().parent
        state_dir = Path(
            str(
                config.get("state_dir")
                or Path(str(config.get("buckets_dir") or repo_root / "buckets")).resolve().parent / "state"
            )
        ).resolve()
        self.root = (state_dir / "narrative_rolls" / "uploads").resolve()
        self.index_path = self.root / "index.json"
        self.blob_dir = self.root / "blobs"

    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {}
        items = raw.get("items") if isinstance(raw, dict) else None
        if not isinstance(items, list):
            return {}
        return {
            str(item.get("upload_id") or ""): item
            for item in items
            if isinstance(item, dict) and UPLOAD_ID_RE.fullmatch(str(item.get("upload_id") or ""))
        }

    def create(self, raw: bytes, *, filename: str, content_type: str = "") -> dict[str, Any]:
        if not raw:
            return {"status": "invalid", "reason": "empty_upload", "writes_performed": []}
        if len(raw) > MAX_UPLOAD_BYTES:
            return {"status": "invalid", "reason": "upload_too_large", "max_bytes": MAX_UPLOAD_BYTES, "writes_performed": []}
        safe_name = _safe_filename(filename)
        mime = str(content_type or "application/octet-stream").split(";", 1)[0].strip().lower()[:160]
        digest = hashlib.sha256(raw).hexdigest()
        upload_id = f"upload_{digest[:32]}"
        items = self._load()
        existing = items.get(upload_id)
        if existing:
            blob = self.blob_dir / str(existing.get("blob_name") or "")
            if str(existing.get("sha256") or "") != digest or not blob.is_file() or hashlib.sha256(blob.read_bytes()).hexdigest() != digest:
                return {"status": "conflict", "reason": "upload_integrity_conflict", "upload_id": upload_id, "writes_performed": []}
            return {"status": "ok", "created": False, **self._public(existing), "writes_performed": []}

        extracted_text, extraction_status = extract_upload_text(raw, safe_name, mime)
        record = {
            "upload_id": upload_id,
            "filename": safe_name,
            "content_type": mime,
            "size": len(raw),
            "sha256": digest,
            "extraction_status": extraction_status,
            "extracted_text": extracted_text,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "blob_name": f"{digest}.bin",
        }
        self.root.mkdir(parents=True, exist_ok=True)
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        blob_path = self.blob_dir / record["blob_name"]
        blob_tmp = blob_path.with_suffix(".tmp")
        index_tmp = self.index_path.with_suffix(".tmp")
        try:
            blob_tmp.write_bytes(raw)
            os.replace(blob_tmp, blob_path)
            next_items = [*items.values(), record]
            index_tmp.write_text(
                json.dumps({"schema_version": "narrative-upload-v1", "items": next_items}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            os.replace(index_tmp, self.index_path)
        except OSError as exc:
            blob_tmp.unlink(missing_ok=True)
            index_tmp.unlink(missing_ok=True)
            if not self.index_path.exists():
                blob_path.unlink(missing_ok=True)
            return {"status": "error", "reason": "upload_write_failed", "error": str(exc), "writes_performed": []}
        return {"status": "ok", "created": True, **self._public(record), "writes_performed": [upload_id]}

    def read(self, upload_id: str, *, include_text: bool = True) -> dict[str, Any]:
        safe_id = str(upload_id or "").strip()
        if not UPLOAD_ID_RE.fullmatch(safe_id):
            return {"status": "invalid", "reason": "invalid_upload_id", "upload_id": safe_id}
        record = self._load().get(safe_id)
        if not record:
            return {"status": "not_found", "upload_id": safe_id}
        blob = self.blob_dir / str(record.get("blob_name") or "")
        try:
            raw = blob.read_bytes()
        except OSError:
            return {"status": "invalid", "reason": "upload_blob_missing", "upload_id": safe_id}
        if len(raw) != int(record.get("size") or -1) or hashlib.sha256(raw).hexdigest() != str(record.get("sha256") or ""):
            return {"status": "invalid", "reason": "upload_blob_mismatch", "upload_id": safe_id}
        result = {"status": "ok", **self._public(record)}
        if include_text:
            result["extracted_text"] = str(record.get("extracted_text") or "")
        return result

    @staticmethod
    def _public(record: dict[str, Any]) -> dict[str, Any]:
        return {
            key: record.get(key)
            for key in (
                "upload_id",
                "filename",
                "content_type",
                "size",
                "sha256",
                "extraction_status",
                "created_at",
            )
        }
