from __future__ import annotations

import hashlib
import json
import re
from typing import Any


MATERIAL_KEYS = ("event_ids", "scene_ids", "diary_ids", "darkroom_ids", "upload_ids")
_EVENT_ID_RE = re.compile(r"^event_[0-9a-f]{24}$")
_SCENE_ID_RE = re.compile(r"^scene_[A-Za-z0-9_.:-]{1,120}$")
_UPLOAD_ID_RE = re.compile(r"^upload_[0-9a-f]{32}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def material_ids_from_narrative(narrative: dict[str, Any]) -> dict[str, list[Any]]:
    return {
        "event_ids": list(narrative.get("linked_event_ids") or []),
        "scene_ids": list(narrative.get("linked_scene_ids") or []),
        "diary_ids": list(narrative.get("linked_diary_ids") or []),
        "darkroom_ids": list(narrative.get("linked_darkroom_ids") or []),
        "upload_ids": list(narrative.get("linked_upload_ids") or []),
    }


def normalize_material_ids(
    value: Any,
    *,
    fallback: dict[str, Any] | None = None,
) -> dict[str, list[Any]]:
    if value is None:
        value = fallback
    if isinstance(value, dict) and set(value) == set(MATERIAL_KEYS[:-1]):
        value = {
            **value,
            "upload_ids": list((fallback or {}).get("upload_ids") or []),
        }
    if not isinstance(value, dict) or set(value) != set(MATERIAL_KEYS):
        raise ValueError("material_ids_must_contain_exact_source_lists")

    normalized: dict[str, list[Any]] = {}
    for key in MATERIAL_KEYS:
        raw = value.get(key)
        if not isinstance(raw, list):
            raise ValueError(f"{key}_must_be_array")
        items: list[Any] = []
        for candidate in raw:
            if key in {"diary_ids", "darkroom_ids"}:
                if isinstance(candidate, bool):
                    raise ValueError(f"invalid_{key[:-1]}")
                try:
                    item: Any = int(candidate)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid_{key[:-1]}") from exc
                if item <= 0:
                    raise ValueError(f"invalid_{key[:-1]}")
            else:
                item = str(candidate or "").strip()
                pattern = (
                    _EVENT_ID_RE
                    if key == "event_ids"
                    else _UPLOAD_ID_RE
                    if key == "upload_ids"
                    else _SCENE_ID_RE
                )
                if not pattern.fullmatch(item):
                    raise ValueError(f"invalid_{key[:-1]}")
            if item not in items:
                items.append(item)
        if len(items) > (8 if key == "upload_ids" else 500):
            raise ValueError(f"too_many_{key}")
        normalized[key] = items
    if sum(len(items) for items in normalized.values()) < 2:
        raise ValueError("at_least_two_narrative_materials_required")
    return normalized


def material_delta(
    current: dict[str, list[Any]],
    proposed: dict[str, list[Any]],
) -> dict[str, dict[str, list[Any]]]:
    added: dict[str, list[Any]] = {}
    removed: dict[str, list[Any]] = {}
    for key in MATERIAL_KEYS:
        before = list(current.get(key) or [])
        after = list(proposed.get(key) or [])
        added[key] = [item for item in after if item not in before]
        removed[key] = [item for item in before if item not in after]
    return {"added": added, "removed": removed}


def material_snapshot_sha256(materials: dict[str, Any]) -> str:
    frozen = {
        key: materials.get(key) or []
        for key in ("events", "scenes", "diaries", "darkrooms", "uploads")
    }
    payload = json.dumps(frozen, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def narrative_preview_fingerprint(
    *,
    narrative_id: str,
    revision: int,
    document_sha256: str,
    body: str,
    material_snapshot_sha256_value: str,
) -> str:
    document_hash = str(document_sha256 or "").strip().lower()
    material_hash = str(material_snapshot_sha256_value or "").strip().lower()
    if not _SHA256_RE.fullmatch(document_hash) or not _SHA256_RE.fullmatch(material_hash):
        raise ValueError("invalid_preview_hash_input")
    body_hash = hashlib.sha256(str(body or "").strip().encode("utf-8")).hexdigest()
    payload = "\n".join(
        (
            "narrative-material-preview-v1",
            str(narrative_id or "").strip(),
            str(int(revision)),
            document_hash,
            body_hash,
            material_hash,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def render_material_snapshot(materials: dict[str, Any]) -> str:
    lines = ["## 绑定材料快照", "", "<!-- managed:narrative-material-membership-v1 -->"]
    for event in materials.get("events") or []:
        lines.append(
            f"- event {event['event_id']} fingerprint:{event['fingerprint']}"
        )
    for scene in materials.get("scenes") or []:
        lines.append(
            f"- scene {scene['scene_id']} content_sha256:{scene['content_sha256']}"
        )
    for diary in materials.get("diaries") or []:
        lines.append(
            "- diary "
            f"diary:{diary['diary_id']} revision:{diary['revision']} "
            f"content_sha256:{diary['content_sha256']} "
            f"comments_sha256:{diary['comments_sha256']}"
        )
    for darkroom in materials.get("darkrooms") or []:
        lines.append(
            "- darkroom "
            f"darkroom:{darkroom['darkroom_id']} revision:{darkroom['revision']} "
            f"content_sha256:{darkroom['content_sha256']} "
            f"comments_sha256:{darkroom['comments_sha256']}"
        )
    for upload in materials.get("uploads") or []:
        lines.append(
            "- upload "
            f"{upload['upload_id']} sha256:{upload['sha256']} "
            f"filename:{json.dumps(upload['filename'], ensure_ascii=False)}"
        )
    return "\n".join(lines) + "\n"
