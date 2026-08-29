from __future__ import annotations

import hashlib
import re
from typing import Any, Awaitable, Callable


_MEMORY_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


def _preserve_existing_legacy_scene_scope(
    *,
    narrative_id: str,
    expected_revision: int,
    exact_document: str,
    current_roll: dict[str, Any],
) -> set[str]:
    if current_roll.get("status") != "ok":
        return set()
    if str(current_roll.get("narrative_id") or "") != narrative_id:
        return set()
    try:
        expected = int(expected_revision)
        current_revision = int(current_roll.get("revision") or 0)
    except (TypeError, ValueError):
        return set()
    current_document = str(current_roll.get("full_document") or "")
    if expected <= 0 or expected != current_revision or not current_document:
        return set()
    if not exact_document.startswith(current_document):
        return set()
    return {
        str(value or "").strip()
        for value in current_roll.get("linked_scene_ids", []) or []
        if str(value or "").strip()
    }


async def verify_narrative_scene_sources(
    *,
    narrative_id: str,
    expected_revision: int,
    exact_document: str,
    scene_ids: list[str],
    current_roll: dict[str, Any],
    get_scene: Callable[[str], Awaitable[dict[str, Any] | None]],
    is_active_canonical_scene: Callable[[dict[str, Any]], bool],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Verify Scene bindings, preserving only append-only legacy revision bindings."""

    preservable_ids = _preserve_existing_legacy_scene_scope(
        narrative_id=narrative_id,
        expected_revision=expected_revision,
        exact_document=exact_document,
        current_roll=current_roll,
    )
    resolved: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        if not _MEMORY_ID_RE.fullmatch(scene_id):
            errors.append({"scene_id": scene_id, "reason": "invalid_scene_id"})
            continue
        scene = await get_scene(scene_id)
        preserve_reason = ""
        if not scene:
            preserve_reason = "scene_not_found"
        elif not is_active_canonical_scene(scene):
            preserve_reason = "not_active_canonical_scene"
        if preserve_reason:
            if scene_id in preservable_ids:
                metadata = (
                    scene.get("metadata", {})
                    if isinstance(scene, dict) and isinstance(scene.get("metadata"), dict)
                    else {}
                )
                resolved.append(
                    {
                        "source_type": "scene",
                        "scene_id": scene_id,
                        "title": str(metadata.get("name") or scene_id),
                        "date": str(metadata.get("date") or ""),
                        "verification": "preserved_existing_legacy",
                        "preserved_existing_legacy": True,
                        "preserved_reason": preserve_reason,
                    }
                )
            else:
                errors.append({"scene_id": scene_id, "reason": preserve_reason})
            continue

        metadata = scene.get("metadata", {}) if isinstance(scene.get("metadata"), dict) else {}
        content_hash = hashlib.sha256(str(scene.get("content") or "").encode("utf-8")).hexdigest()
        if content_hash not in exact_document:
            errors.append(
                {
                    "scene_id": scene_id,
                    "reason": "scene_content_hash_missing_from_document",
                    "content_sha256": content_hash,
                }
            )
            continue
        resolved.append(
            {
                "source_type": "scene",
                "scene_id": scene_id,
                "title": str(metadata.get("name") or scene_id),
                "date": str(metadata.get("date") or ""),
                "content_sha256": content_hash,
            }
        )
    return resolved, errors
