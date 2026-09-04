from __future__ import annotations

import hashlib
from typing import Any

from narrative_materials import material_delta


MATERIAL_ID_KEYS = (
    "event_ids",
    "scene_ids",
    "diary_ids",
    "darkroom_ids",
    "upload_ids",
)


def conversation_source_messages(source_refs: list[dict[str, Any]]) -> dict[str, Any]:
    """Return complete, hash-verified bound conversation text or a material fallback signal."""
    if not source_refs:
        return {"status": "fallback", "messages": []}

    messages: list[dict[str, Any]] = []
    for ref in source_refs:
        content = str(ref.get("content") or "")
        if not content:
            return {"status": "fallback", "messages": []}
        expected_hash = str(ref.get("content_sha256") or "").strip().lower()
        actual_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if not expected_hash or actual_hash != expected_hash:
            return {
                "status": "invalid",
                "message_id": ref.get("message_id"),
                "messages": [],
            }
        messages.append({
            "source_system": str(ref.get("source_system") or ""),
            "session_id": ref.get("session_id"),
            "message_id": ref.get("message_id"),
            "role": str(ref.get("role") or ""),
            "created_at": str(ref.get("created_at") or ""),
            "content": content,
            "content_sha256": expected_hash,
        })
    return {"status": "ok", "messages": messages}


def writer_material_ids(mode: str, current: dict, proposed: dict) -> dict[str, Any]:
    """Select all bound sources for rewrite, or only additions for update."""
    delta = material_delta(current, proposed)
    if mode != "update":
        return {"status": "ok", "scope": "all_bound", "material_ids": proposed, "delta": delta}

    if any(delta["removed"].get(key) for key in MATERIAL_ID_KEYS):
        return {
            "status": "invalid",
            "reason": "update_requires_rewrite_after_material_removal",
            "message": "这次删减了旧材料，请使用重写，让正文按新的完整材料重新落定。",
            "delta": delta,
        }
    if not any(delta["added"].get(key) for key in MATERIAL_ID_KEYS):
        return {
            "status": "invalid",
            "reason": "no_new_materials",
            "message": "没有新增材料可用于更新。",
            "delta": delta,
        }
    return {
        "status": "ok",
        "scope": "newly_added",
        "material_ids": delta["added"],
        "delta": delta,
    }
