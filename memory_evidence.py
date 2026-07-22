from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def scene_content_hash(scene: dict) -> str:
    payload = {
        "id": str((scene or {}).get("id") or ""),
        "content": str((scene or {}).get("content") or ""),
    }
    return _stable_hash(payload)


def legacy_moment_content_hash(moment: dict) -> str:
    payload = {
        "moment_id": str((moment or {}).get("moment_id") or ""),
        "text": str((moment or {}).get("text") or ""),
    }
    return _stable_hash(payload)


def normalize_evidence_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).strip(
        "`'\"“”‘’《》「」『』 "
    )


def evidence_is_verbatim(content: str, evidence: Any) -> bool:
    excerpt = re.sub(r"\s+", "", normalize_evidence_text(evidence))
    body = re.sub(r"\s+", "", str(content or ""))
    return len(excerpt) >= 6 and excerpt in body


def _stable_hash(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
