from __future__ import annotations

import hashlib
import json
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


def diary_comments_sha256(comments: Any) -> str:
    """Hash the stable, readable comment snapshot returned with a Diary."""

    normalized = [
        {
            "id": int(comment.get("id") or 0),
            "author": str(comment.get("author") or ""),
            "created_at": str(comment.get("created_at") or ""),
            "content": str(comment.get("content") or ""),
        }
        for comment in comments or []
        if isinstance(comment, dict)
    ]
    normalized.sort(
        key=lambda comment: (
            comment["id"],
            comment["author"],
            comment["created_at"],
            comment["content"],
        )
    )
    stable_json = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(stable_json.encode("utf-8")).hexdigest()


def diary_source_marker(
    *,
    diary_id: int,
    revision: int,
    content_sha256: str,
    comments_sha256: str,
) -> str:
    """Return the exact source marker required in a Narrative publication document."""

    return (
        f"diary:{int(diary_id)} revision:{int(revision)} "
        f"content_sha256:{content_sha256} comments_sha256:{comments_sha256}"
    )


def darkroom_source_marker(
    *,
    darkroom_id: int,
    revision: int,
    content_sha256: str,
    comments_sha256: str,
) -> str:
    """Return the exact source marker required for one unlocked Darkroom snapshot."""

    return (
        f"darkroom:{int(darkroom_id)} revision:{int(revision)} "
        f"content_sha256:{content_sha256} comments_sha256:{comments_sha256}"
    )


def verify_narrative_diary_sources(
    *,
    exact_document: str,
    diary_ids: list[int],
    get_diary: Callable[[int], dict[str, Any] | None],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Verify active Diary bindings without copying Diary content into Narrative state."""

    resolved: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for diary_id in diary_ids:
        diary = get_diary(diary_id)
        if not diary:
            errors.append({"diary_id": diary_id, "reason": "diary_not_found"})
            continue
        if str(diary.get("entry_type") or "") != "diary":
            errors.append({"diary_id": diary_id, "reason": "not_ordinary_diary"})
            continue
        if str(diary.get("visibility") or "") != "active":
            errors.append({"diary_id": diary_id, "reason": "not_active_diary"})
            continue
        try:
            revision = int(diary.get("revision") or 0)
        except (TypeError, ValueError):
            revision = 0
        if revision <= 0 or not bool(diary.get("body_available")):
            errors.append({"diary_id": diary_id, "reason": "diary_content_unavailable"})
            continue
        content_sha256 = hashlib.sha256(
            str(diary.get("content") or "").encode("utf-8")
        ).hexdigest()
        comments_sha256 = diary_comments_sha256(diary.get("comments"))
        marker = diary_source_marker(
            diary_id=diary_id,
            revision=revision,
            content_sha256=content_sha256,
            comments_sha256=comments_sha256,
        )
        if marker not in exact_document:
            errors.append(
                {
                    "diary_id": diary_id,
                    "reason": "diary_snapshot_marker_missing_from_document",
                    "revision": revision,
                    "content_sha256": content_sha256,
                    "comments_sha256": comments_sha256,
                }
            )
            continue
        resolved.append(
            {
                "source_type": "diary",
                "diary_id": diary_id,
                "title": str(diary.get("title") or f"Diary {diary_id}"),
                "date": str(diary.get("date") or ""),
                "revision": revision,
                "content_sha256": content_sha256,
                "comments_sha256": comments_sha256,
            }
        )
    return resolved, errors


def verify_narrative_darkroom_sources(
    *,
    exact_document: str,
    darkroom_ids: list[int],
    get_darkroom: Callable[[int], dict[str, Any] | None],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Verify unlocked Darkroom snapshots without exposing their readable content."""

    resolved: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for darkroom_id in darkroom_ids:
        darkroom = get_darkroom(darkroom_id)
        if (
            not darkroom
            or str(darkroom.get("entry_type") or "") != "darkroom"
            or str(darkroom.get("visibility") or "") != "active"
            or darkroom.get("locked") is not False
            or not bool(darkroom.get("body_available"))
        ):
            errors.append(
                {"darkroom_id": darkroom_id, "reason": "darkroom_source_unavailable"}
            )
            continue
        try:
            revision = int(darkroom.get("revision") or 0)
        except (TypeError, ValueError):
            revision = 0
        if revision <= 0:
            errors.append(
                {"darkroom_id": darkroom_id, "reason": "darkroom_source_unavailable"}
            )
            continue
        content_sha256 = hashlib.sha256(
            str(darkroom.get("content") or "").encode("utf-8")
        ).hexdigest()
        comments_sha256 = diary_comments_sha256(darkroom.get("comments"))
        marker = darkroom_source_marker(
            darkroom_id=darkroom_id,
            revision=revision,
            content_sha256=content_sha256,
            comments_sha256=comments_sha256,
        )
        if marker not in exact_document:
            errors.append(
                {
                    "darkroom_id": darkroom_id,
                    "reason": "darkroom_snapshot_marker_missing_from_document",
                }
            )
            continue
        resolved.append(
            {
                "source_type": "darkroom",
                "darkroom_id": darkroom_id,
                "revision": revision,
                "content_sha256": content_sha256,
                "comments_sha256": comments_sha256,
            }
        )
    return resolved, errors
