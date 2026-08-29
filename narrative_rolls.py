from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from identity import identity_names


_SCENE_ID_RE = re.compile(r"\bscene_mig2_[A-Za-z0-9]+\b")
_EVENT_ID_RE = re.compile(r"\bevent_[0-9a-f]{24}\b")
_DIARY_ID_RE = re.compile(r"\bdiary:(\d{1,9})\b")
_DARKROOM_ID_RE = re.compile(r"\bdarkroom:(\d{1,9})\b")
_NARRATIVE_ID_RE = re.compile(r"^narrative_[A-Za-z0-9_.:-]{1,96}$")
_ARC_KEY_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}:[^\s\x00-\x1f\x7f]{1,127}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_BODY_HEADING_RE = re.compile(r"(?m)^## 第一人称叙事\s*$")
_NEXT_HEADING_RE = re.compile(r"(?m)^##\s+")
_COMPACT_RE = re.compile(r"[^0-9a-z\u4e00-\u9fff]+", re.IGNORECASE)

_EXACT_EVIDENCE_MARKERS = (
    "原话",
    "原文",
    "逐字",
    "当时怎么说",
    "具体怎么说",
    "谁说",
    "谁喊",
    "哪天",
    "什么时候",
    "几月几日",
    "具体日期",
    "什么型号",
    "型号是什么",
    "具体型号",
    "哪首歌",
    "歌名",
    "哪一首",
)

NARRATIVE_COLLECTION_QUERY_ALIASES = frozenset(
    {"叙事卷", "narrative", "narrativeroll", "narrativerolls"}
)


def _compact(value: Any) -> str:
    return _COMPACT_RE.sub("", str(value or "").strip().lower())


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_body(document: str) -> str:
    match = _BODY_HEADING_RE.search(document)
    if match is None:
        return ""
    start = match.end()
    next_heading = _NEXT_HEADING_RE.search(document, start)
    end = next_heading.start() if next_heading is not None else len(document)
    return document[start:end].strip()


def _query_requests_exact_evidence(query: str) -> bool:
    compact_query = _compact(query)
    return any(_compact(marker) in compact_query for marker in _EXACT_EVIDENCE_MARKERS)


def normalize_arc_key(value: Any) -> str:
    key = str(value or "").strip()
    return key if _ARC_KEY_RE.fullmatch(key) else ""


class NarrativeRollStore:
    """Registry for sourced, manually authored Narrative Projection arcs.

    The Markdown document remains the exact author-supplied artifact.  The
    registry supplies routing metadata and optimistic revisions; it never turns
    an arc into Scene evidence or an ordinary bucket.  Publication writes only
    text supplied by the current author and never invokes a model.
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.identity = identity_names(config)
        roll_cfg = config.get("narrative_rolls", {})
        if not isinstance(roll_cfg, dict):
            roll_cfg = {}
        repo_root = Path(__file__).resolve().parent
        state_dir = Path(
            str(
                config.get("state_dir")
                or Path(
                    str(config.get("buckets_dir") or repo_root / "buckets")
                ).resolve().parent
                / "state"
            )
        ).resolve()
        configured_path = str(roll_cfg.get("registry_path") or "").strip()
        registry_path = (
            Path(configured_path)
            if configured_path
            else state_dir / "narrative_rolls" / "registry.json"
        )
        if configured_path and not registry_path.is_absolute():
            registry_path = state_dir / registry_path
        self.registry_path = registry_path.resolve()
        configured_documents_dir = str(roll_cfg.get("documents_dir") or "").strip()
        documents_dir = (
            Path(configured_documents_dir)
            if configured_documents_dir
            else self.registry_path.parent / "revisions"
        )
        if configured_documents_dir and not documents_dir.is_absolute():
            documents_dir = self.registry_path.parent / documents_dir
        self.documents_dir = documents_dir.resolve()
        self.shadow_admission_enabled = bool(roll_cfg.get("shadow_admission_enabled", True))
        self.live_injection_enabled = bool(roll_cfg.get("live_injection_enabled", True))
        self.ambiguity_margin = max(0, int(roll_cfg.get("ambiguity_margin", 10)))
        self._cache_stamp: tuple[int, int] | None = None
        self._cache_items: list[dict[str, Any]] = []

    def _registry_stamp(self) -> tuple[int, int] | None:
        try:
            stat = self.registry_path.stat()
        except OSError:
            return None
        return int(stat.st_mtime_ns), int(stat.st_size)

    def _source_path(self, value: Any) -> Path:
        path = Path(str(value or "").strip())
        if not path.is_absolute():
            path = self.registry_path.parent / path
        return path.resolve()

    def _load(self) -> list[dict[str, Any]]:
        stamp = self._registry_stamp()
        if stamp is None:
            self._cache_stamp = None
            self._cache_items = []
            return []
        if stamp == self._cache_stamp:
            return self._cache_items
        try:
            raw = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            self._cache_stamp = stamp
            self._cache_items = []
            return []
        entries = raw.get("rolls", []) if isinstance(raw, dict) else []
        items: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for entry in entries if isinstance(entries, list) else []:
            if not isinstance(entry, dict):
                continue
            narrative_id = str(entry.get("narrative_id") or "").strip()
            if not narrative_id or narrative_id in seen_ids:
                continue
            seen_ids.add(narrative_id)
            source_path = self._source_path(entry.get("source_file"))
            item = {
                **entry,
                "narrative_id": narrative_id,
                "source_file": str(entry.get("source_file") or ""),
                "source_path": str(source_path),
                "integrity_status": "missing_source",
                "body": "",
                "full_document": "",
                "linked_scene_ids": [],
                "linked_event_ids": [],
                "linked_diary_ids": [],
                "linked_darkroom_ids": [],
                "arc_key": normalize_arc_key(entry.get("arc_key")),
                "parent_narrative_id": (
                    str(entry.get("parent_narrative_id") or "").strip()
                    if _NARRATIVE_ID_RE.fullmatch(
                        str(entry.get("parent_narrative_id") or "").strip()
                    )
                    else ""
                ),
            }
            try:
                document = source_path.read_text(encoding="utf-8")
            except OSError:
                items.append(item)
                continue
            actual_hash = _sha256_text(document)
            expected_hash = str(entry.get("document_sha256") or "").strip().lower()
            body = _extract_body(document)
            publication_status = str(entry.get("publication_status") or "reviewed").strip().lower()
            excluded_ids = {
                str(value or "").strip()
                for value in entry.get("excluded_scene_ids", []) or []
                if str(value or "").strip()
            }
            linked_scene_ids = list(
                dict.fromkeys(
                    str(scene_id or "").strip()
                    for scene_id in [
                        *(entry.get("linked_scene_ids", []) or []),
                        *_SCENE_ID_RE.findall(document),
                    ]
                    if str(scene_id or "").strip()
                    if scene_id not in excluded_ids
                )
            )
            excluded_event_ids = {
                str(value or "").strip()
                for value in entry.get("excluded_event_ids", []) or []
                if str(value or "").strip()
            }
            linked_event_ids = list(
                dict.fromkeys(
                    str(event_id or "").strip()
                    for event_id in [
                        *(entry.get("linked_event_ids", []) or []),
                        *_EVENT_ID_RE.findall(document),
                    ]
                    if str(event_id or "").strip()
                    if event_id not in excluded_event_ids
                )
            )
            excluded_diary_ids = {
                int(value)
                for value in entry.get("excluded_diary_ids", []) or []
                if str(value or "").strip().isdigit() and int(value) > 0
            }
            linked_diary_ids = list(
                dict.fromkeys(
                    diary_id
                    for diary_id in [
                        *(
                            int(value)
                            for value in entry.get("linked_diary_ids", []) or []
                            if str(value or "").strip().isdigit() and int(value) > 0
                        ),
                        *(int(value) for value in _DIARY_ID_RE.findall(document)),
                    ]
                    if diary_id not in excluded_diary_ids
                )
            )
            linked_darkroom_ids = list(
                dict.fromkeys(
                    darkroom_id
                    for darkroom_id in [
                        *(
                            int(value)
                            for value in entry.get("linked_darkroom_ids", []) or []
                            if str(value or "").strip().isdigit() and int(value) > 0
                        ),
                        *(int(value) for value in _DARKROOM_ID_RE.findall(document)),
                    ]
                )
            )
            if expected_hash and actual_hash != expected_hash:
                integrity_status = "hash_mismatch"
            elif not body and publication_status != "collecting":
                integrity_status = "missing_first_person_body"
            else:
                integrity_status = "ok"
            item.update(
                {
                    "integrity_status": integrity_status,
                    "actual_document_sha256": actual_hash,
                    "body": body if integrity_status == "ok" else "",
                    "body_sha256": _sha256_text(body) if body else "",
                    "body_chars": len(body),
                    "full_document": document if integrity_status == "ok" else "",
                    "linked_scene_ids": linked_scene_ids,
                    "linked_scene_count": len(linked_scene_ids),
                    "linked_event_ids": linked_event_ids,
                    "linked_event_count": len(linked_event_ids),
                    "linked_diary_ids": linked_diary_ids,
                    "linked_diary_count": len(linked_diary_ids),
                    "linked_darkroom_ids": linked_darkroom_ids,
                    "linked_darkroom_count": len(linked_darkroom_ids),
                }
            )
            items.append(item)
        self._cache_stamp = stamp
        self._cache_items = items
        return items

    @staticmethod
    def _light(item: dict[str, Any]) -> dict[str, Any]:
        return {
            key: item.get(key)
            for key in (
                "narrative_id",
                "arc_key",
                "parent_narrative_id",
                "revision",
                "title",
                "scope",
                "publication_status",
                "lifecycle",
                "time_start",
                "time_end",
                "primary_entities",
                "supporting_entities",
                "intent_tags",
                "query_cues",
                "current_status_cue",
                "source_file",
                "document_sha256",
                "actual_document_sha256",
                "body_sha256",
                "body_chars",
                "linked_scene_count",
                "linked_event_count",
                "linked_diary_ids",
                "linked_diary_count",
                "linked_darkroom_ids",
                "linked_darkroom_count",
                "integrity_status",
            )
        }

    def list(self, query: str = "", limit: int = 20) -> dict[str, Any]:
        safe_limit = max(1, min(int(limit or 20), 100))
        compact_query = _compact(query)
        if compact_query in NARRATIVE_COLLECTION_QUERY_ALIASES:
            compact_query = ""
        rows = []
        for item in self._load():
            searchable = _compact(
                " ".join(
                    [
                        str(item.get("title") or ""),
                        *(str(value or "") for value in item.get("title_aliases", []) or []),
                        *(str(value or "") for value in item.get("primary_entities", []) or []),
                        *(str(value or "") for value in item.get("supporting_entities", []) or []),
                        *(str(value or "") for value in item.get("query_cues", []) or []),
                    ]
                )
            )
            if compact_query and compact_query not in searchable and searchable not in compact_query:
                continue
            rows.append(self._light(item))
        return {
            "status": "ok",
            "mode": "narrative_roll_index",
            "query": str(query or ""),
            "count": len(rows),
            "items": rows[:safe_limit],
            "live_injection_enabled": self.live_injection_enabled,
        }

    def revision_targets(self) -> list[dict[str, Any]]:
        """Return reviewed routing metadata for the derived revision inbox.

        This intentionally excludes Narrative bodies and entity labels.  New
        sources may be offered for review only through title/query cues already
        authored on the roll, never through generic people or Word Map terms.
        """

        return [
            {
                key: item.get(key)
                for key in (
                    "narrative_id",
                    "revision",
                    "title",
                    "title_aliases",
                    "query_cues",
                    "current_status_cue",
                    "document_sha256",
                    "integrity_status",
                )
            }
            for item in self._load()
            if str(item.get("lifecycle") or "active") == "active"
            and str(item.get("publication_status") or "reviewed") in {"reviewed", "published"}
        ]

    def resolve_read_query(self, query: str = "", limit: int = 20) -> dict[str, Any]:
        """Read one exact roll identity, otherwise return the bounded index."""

        compact_query = _compact(query)
        if not compact_query or compact_query in NARRATIVE_COLLECTION_QUERY_ALIASES:
            return self.list(query=query, limit=limit)

        exact_matches = []
        for item in self._load():
            identities = (
                item.get("narrative_id"),
                item.get("title"),
                *(item.get("title_aliases", []) or []),
            )
            if compact_query in {_compact(value) for value in identities if value}:
                exact_matches.append(item)

        if len(exact_matches) == 1:
            return self.read(str(exact_matches[0].get("narrative_id") or ""))
        return self.list(query=query, limit=limit)

    def read(self, narrative_id: str) -> dict[str, Any]:
        safe_id = str(narrative_id or "").strip()
        item = next(
            (row for row in self._load() if str(row.get("narrative_id") or "") == safe_id),
            None,
        )
        if item is None:
            return {"status": "not_found", "narrative_id": safe_id}
        if item.get("integrity_status") != "ok":
            return {
                "status": "invalid",
                "narrative_id": safe_id,
                "integrity_status": item.get("integrity_status"),
                "source_file": item.get("source_file"),
                "expected_document_sha256": item.get("document_sha256"),
                "actual_document_sha256": item.get("actual_document_sha256"),
            }
        return {
            "status": "ok",
            "mode": "narrative_roll_full_read",
            **self._light(item),
            "linked_scene_ids": list(item.get("linked_scene_ids") or []),
            "linked_event_ids": list(item.get("linked_event_ids") or []),
            "linked_diary_ids": list(item.get("linked_diary_ids") or []),
            "linked_darkroom_ids": list(item.get("linked_darkroom_ids") or []),
            "history": [
                {
                    **history_item,
                    "linked_scene_ids": list(history_item.get("linked_scene_ids") or []),
                    "linked_event_ids": list(history_item.get("linked_event_ids") or []),
                    "linked_diary_ids": list(history_item.get("linked_diary_ids") or []),
                    "linked_darkroom_ids": list(history_item.get("linked_darkroom_ids") or []),
                }
                for history_item in item.get("history", []) or []
                if isinstance(history_item, dict)
            ],
            "body": str(item.get("body") or ""),
            "full_document": str(item.get("full_document") or ""),
            "reading_boundary": (
                "This collecting Arc is a source index, not authored narrative or original evidence. "
                "Read its linked Scene/Event/raw, Diary, and Darkroom sources selectively."
                if str(item.get("publication_status") or "") == "collecting"
                else "Narrative Roll is a sourced first-person projection, not original evidence. "
                "For exact dates or wording, read its linked Scene/Event/raw, Diary, and Darkroom sources."
            ),
        }

    def read_by_arc_key(self, arc_key: str) -> dict[str, Any]:
        """Resolve one active, intact Arc from its persisted stable key."""

        safe_key = normalize_arc_key(arc_key)
        if not safe_key:
            return {"status": "invalid", "reason": "invalid_arc_key", "arc_key": str(arc_key or "").strip()}
        matches = [
            item
            for item in self._load()
            if str(item.get("arc_key") or "") == safe_key
            and str(item.get("lifecycle") or "active") == "active"
        ]
        if not matches:
            return {"status": "not_found", "arc_key": safe_key}
        if len(matches) != 1:
            return {"status": "invalid", "reason": "duplicate_arc_key", "arc_key": safe_key}
        return self.read(str(matches[0].get("narrative_id") or ""))

    @staticmethod
    def source_scene_ids(document: str, explicit_ids: list[str] | None = None) -> list[str]:
        return list(
            dict.fromkeys(
                str(value or "").strip()
                for value in [*(explicit_ids or []), *_SCENE_ID_RE.findall(str(document or ""))]
                if str(value or "").strip()
            )
        )

    @staticmethod
    def source_event_ids(document: str, explicit_ids: list[str] | None = None) -> list[str]:
        return list(
            dict.fromkeys(
                str(value or "").strip()
                for value in [*(explicit_ids or []), *_EVENT_ID_RE.findall(str(document or ""))]
                if str(value or "").strip()
            )
        )

    @staticmethod
    def source_diary_ids(document: str, explicit_ids: list[int] | None = None) -> list[int]:
        values = [*(explicit_ids or []), *_DIARY_ID_RE.findall(str(document or ""))]
        return list(
            dict.fromkeys(
                int(value)
                for value in values
                if str(value or "").strip().isdigit() and int(value) > 0
            )
        )

    @staticmethod
    def source_darkroom_ids(document: str, explicit_ids: list[int] | None = None) -> list[int]:
        values = [*(explicit_ids or []), *_DARKROOM_ID_RE.findall(str(document or ""))]
        return list(
            dict.fromkeys(
                int(value)
                for value in values
                if str(value or "").strip().isdigit() and int(value) > 0
            )
        )

    @staticmethod
    def _string_list(values: Any, *, limit: int = 40) -> list[str]:
        if isinstance(values, str):
            source = re.split(r"[\n,|]+", values)
        elif isinstance(values, (list, tuple, set)):
            source = values
        else:
            source = []
        return list(
            dict.fromkeys(
                str(value or "").strip()
                for value in source
                if str(value or "").strip()
            )
        )[:limit]

    def publish(
        self,
        *,
        narrative_id: str,
        document: str,
        expected_revision: int,
        title: str,
        arc_key: str = "",
        parent_narrative_id: str = "",
        source_scene_ids: list[str] | None = None,
        source_event_ids: list[str] | None = None,
        source_diary_ids: list[int] | None = None,
        source_darkroom_ids: list[int] | None = None,
        title_aliases: list[str] | None = None,
        primary_entities: list[str] | None = None,
        supporting_entities: list[str] | None = None,
        intent_tags: list[str] | None = None,
        query_cues: list[str] | None = None,
        time_start: str = "",
        time_end: str = "",
        current_status_cue: str = "",
        publication_status: str = "reviewed",
        lifecycle: str = "active",
    ) -> dict[str, Any]:
        """Publish one exact authored revision with optimistic concurrency."""

        safe_id = str(narrative_id or "").strip()
        exact_document = str(document or "")
        safe_title = str(title or "").strip()
        if not _NARRATIVE_ID_RE.fullmatch(safe_id):
            return {"status": "invalid", "reason": "invalid_narrative_id", "narrative_id": safe_id}
        if not safe_title:
            return {"status": "invalid", "reason": "title_required", "narrative_id": safe_id}
        if not exact_document.strip():
            return {"status": "invalid", "reason": "document_required", "narrative_id": safe_id}

        publication_status = str(publication_status or "reviewed").strip().lower()
        if publication_status not in {"collecting", "reviewed", "published"}:
            return {
                "status": "invalid",
                "reason": "publication_status_must_be_collecting_reviewed_or_published",
                "narrative_id": safe_id,
            }
        if publication_status != "collecting" and not _extract_body(exact_document):
            return {
                "status": "invalid",
                "reason": "missing_first_person_body",
                "narrative_id": safe_id,
            }

        linked_scene_ids = self.source_scene_ids(exact_document, source_scene_ids)
        linked_event_ids = self.source_event_ids(exact_document, source_event_ids)
        linked_diary_ids = self.source_diary_ids(exact_document, source_diary_ids)
        linked_darkroom_ids = self.source_darkroom_ids(exact_document, source_darkroom_ids)
        if (
            len(linked_scene_ids)
            + len(linked_event_ids)
            + len(linked_diary_ids)
            + len(linked_darkroom_ids)
            < 2
        ):
            return {
                "status": "invalid",
                "reason": (
                    "at_least_two_source_scenes_required"
                    if not linked_event_ids
                    else "at_least_two_sources_required"
                ),
                "narrative_id": safe_id,
            }

        requested_arc_key = str(arc_key or "").strip()
        if requested_arc_key and not normalize_arc_key(requested_arc_key):
            return {
                "status": "invalid",
                "reason": "invalid_arc_key",
                "narrative_id": safe_id,
                "arc_key": requested_arc_key,
            }
        requested_parent_id = str(parent_narrative_id or "").strip()
        if requested_parent_id and not _NARRATIVE_ID_RE.fullmatch(requested_parent_id):
            return {
                "status": "invalid",
                "reason": "invalid_parent_narrative_id",
                "narrative_id": safe_id,
                "parent_narrative_id": requested_parent_id,
            }
        if requested_parent_id == safe_id:
            return {
                "status": "invalid",
                "reason": "parent_narrative_cannot_be_self",
                "narrative_id": safe_id,
            }
        missing_in_document = [scene_id for scene_id in linked_scene_ids if scene_id not in exact_document]
        if missing_in_document:
            return {
                "status": "invalid",
                "reason": "source_scene_id_missing_from_document",
                "narrative_id": safe_id,
                "scene_ids": missing_in_document,
            }
        missing_events_in_document = [
            event_id for event_id in linked_event_ids if event_id not in exact_document
        ]
        if missing_events_in_document:
            return {
                "status": "invalid",
                "reason": "source_event_id_missing_from_document",
                "narrative_id": safe_id,
                "event_ids": missing_events_in_document,
            }
        missing_diaries_in_document = [
            diary_id for diary_id in linked_diary_ids if f"diary:{diary_id}" not in exact_document
        ]
        if missing_diaries_in_document:
            return {
                "status": "invalid",
                "reason": "source_diary_id_missing_from_document",
                "narrative_id": safe_id,
                "diary_ids": missing_diaries_in_document,
            }
        missing_darkrooms_in_document = [
            darkroom_id
            for darkroom_id in linked_darkroom_ids
            if f"darkroom:{darkroom_id}" not in exact_document
        ]
        if missing_darkrooms_in_document:
            return {
                "status": "invalid",
                "reason": "source_darkroom_id_missing_from_document",
                "narrative_id": safe_id,
                "darkroom_ids": missing_darkrooms_in_document,
            }

        lifecycle = str(lifecycle or "active").strip().lower()
        if lifecycle not in {"active", "closed", "retired"}:
            return {
                "status": "invalid",
                "reason": "lifecycle_must_be_active_closed_or_retired",
                "narrative_id": safe_id,
            }
        for label, value in (("time_start", time_start), ("time_end", time_end)):
            if str(value or "").strip() and not _DATE_RE.fullmatch(str(value).strip()):
                return {
                    "status": "invalid",
                    "reason": f"{label}_must_be_yyyy_mm_dd",
                    "narrative_id": safe_id,
                }

        try:
            expected = max(0, int(expected_revision))
        except (TypeError, ValueError):
            return {"status": "invalid", "reason": "invalid_expected_revision", "narrative_id": safe_id}

        try:
            raw = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raw = {"schema_version": "narrative-roll-registry-v1", "rolls": []}
        except (OSError, ValueError, TypeError) as exc:
            return {
                "status": "error",
                "reason": "registry_unreadable",
                "error": str(exc),
                "narrative_id": safe_id,
            }
        if not isinstance(raw, dict):
            raw = {"schema_version": "narrative-roll-registry-v1", "rolls": []}
        entries = raw.get("rolls") if isinstance(raw.get("rolls"), list) else []
        current_index = next(
            (
                index
                for index, entry in enumerate(entries)
                if isinstance(entry, dict) and str(entry.get("narrative_id") or "") == safe_id
            ),
            None,
        )
        current = entries[current_index] if current_index is not None else None
        current_revision = int((current or {}).get("revision") or 0)
        if current_revision != expected:
            return {
                "status": "conflict",
                "reason": "revision_mismatch",
                "narrative_id": safe_id,
                "expected_revision": expected,
                "current_revision": current_revision,
            }


        current_arc_key = normalize_arc_key((current or {}).get("arc_key"))
        safe_arc_key = normalize_arc_key(requested_arc_key) or current_arc_key
        if publication_status == "collecting" and not safe_arc_key:
            return {
                "status": "invalid",
                "reason": "arc_key_required_for_collecting",
                "narrative_id": safe_id,
            }
        if current_arc_key and requested_arc_key and safe_arc_key != current_arc_key:
            return {
                "status": "conflict",
                "reason": "arc_key_is_stable",
                "narrative_id": safe_id,
                "arc_key": current_arc_key,
            }
        duplicate = next(
            (
                entry
                for entry in entries
                if isinstance(entry, dict)
                and str(entry.get("narrative_id") or "") != safe_id
                and normalize_arc_key(entry.get("arc_key")) == safe_arc_key
            ),
            None,
        ) if safe_arc_key else None
        if duplicate is not None:
            return {
                "status": "conflict",
                "reason": "arc_key_already_exists",
                "narrative_id": safe_id,
                "arc_key": safe_arc_key,
            }

        current_parent_id = str((current or {}).get("parent_narrative_id") or "").strip()
        if current is not None and requested_parent_id and requested_parent_id != current_parent_id:
            return {
                "status": "conflict",
                "reason": "parent_narrative_id_is_stable",
                "narrative_id": safe_id,
                "parent_narrative_id": current_parent_id,
            }
        safe_parent_id = current_parent_id or requested_parent_id
        if safe_parent_id:
            parent_entry = next(
                (
                    entry
                    for entry in entries
                    if isinstance(entry, dict)
                    and str(entry.get("narrative_id") or "").strip() == safe_parent_id
                ),
                None,
            )
            if parent_entry is None:
                return {
                    "status": "invalid",
                    "reason": "parent_narrative_not_found",
                    "narrative_id": safe_id,
                    "parent_narrative_id": safe_parent_id,
                }
            if str(parent_entry.get("lifecycle") or "active").strip().lower() != "active":
                return {
                    "status": "invalid",
                    "reason": "parent_narrative_not_active",
                    "narrative_id": safe_id,
                    "parent_narrative_id": safe_parent_id,
                }

        revision = current_revision + 1
        document_hash = _sha256_text(exact_document)
        published_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        document_dir = self.documents_dir / safe_id
        document_path = document_dir / f"revision-{revision:04d}.md"
        try:
            source_file = os.path.relpath(document_path, self.registry_path.parent).replace("\\", "/")
        except ValueError:
            source_file = str(document_path)

        history = list((current or {}).get("history", []) or [])
        if current:
            history.append(
                {
                    key: current.get(key)
                    for key in (
                        "revision",
                        "publication_status",
                        "lifecycle",
                        "source_file",
                        "document_sha256",
                        "published_at",
                        "linked_scene_ids",
                        "linked_event_ids",
                        "linked_diary_ids",
                        "linked_darkroom_ids",
                        "arc_key",
                        "parent_narrative_id",
                    )
                }
            )

        entry = {
            "narrative_id": safe_id,
            "arc_key": safe_arc_key,
            "parent_narrative_id": safe_parent_id,
            "revision": revision,
            "scope": "arc",
            "title": safe_title,
            "title_aliases": self._string_list(title_aliases),
            "publication_status": publication_status,
            "lifecycle": lifecycle,
            "time_start": str(time_start or "").strip(),
            "time_end": str(time_end or "").strip(),
            "primary_entities": self._string_list(primary_entities),
            "supporting_entities": self._string_list(supporting_entities),
            "intent_tags": self._string_list(intent_tags),
            "query_cues": self._string_list(query_cues),
            "current_status_cue": str(current_status_cue or "").strip(),
            "source_file": source_file,
            "document_sha256": document_hash,
            "linked_scene_ids": linked_scene_ids,
            "linked_event_ids": linked_event_ids,
            "linked_diary_ids": linked_diary_ids,
            "linked_darkroom_ids": linked_darkroom_ids,
            "published_at": published_at,
            "published_by": f"{self.identity['ai_name']}_manual",
            "history": history,
        }
        if current_index is None:
            entries.append(entry)
        else:
            entries[current_index] = entry
        raw["schema_version"] = str(raw.get("schema_version") or "narrative-roll-registry-v1")
        raw["rolls"] = entries

        document_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        document_tmp = document_path.with_suffix(document_path.suffix + ".tmp")
        registry_tmp = self.registry_path.with_suffix(self.registry_path.suffix + ".tmp")
        try:
            document_tmp.write_text(exact_document, encoding="utf-8", newline="")
            registry_tmp.write_text(
                json.dumps(raw, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            os.replace(document_tmp, document_path)
            os.replace(registry_tmp, self.registry_path)
        except OSError as exc:
            for temp_path in (document_tmp, registry_tmp):
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
            return {
                "status": "error",
                "reason": "publication_write_failed",
                "error": str(exc),
                "narrative_id": safe_id,
            }

        self._cache_stamp = None
        self._cache_items = []
        result = self.read(safe_id)
        result.update(
            {
                "status": "created" if current is None else "updated",
                "expected_revision": expected,
                "source_scene_ids": linked_scene_ids,
                "source_event_ids": linked_event_ids,
                "source_diary_ids": linked_diary_ids,
                "source_darkroom_ids": linked_darkroom_ids,
                "canonical_scene_changed": False,
                "model_called": False,
            }
        )
        return result

    def shadow_match(self, query: str, direct_first_hop_scene_ids: list[str]) -> dict[str, Any]:
        direct_ids = list(
            dict.fromkeys(
                str(value or "").strip()
                for value in direct_first_hop_scene_ids or []
                if str(value or "").strip()
            )
        )
        base = {
            "enabled": self.shadow_admission_enabled,
            "mode": "live_gate" if self.live_injection_enabled else "shadow_only",
            "live_injection_enabled": self.live_injection_enabled,
            "visible_injection": False,
            "exact_evidence_route": _query_requests_exact_evidence(query),
            "direct_first_hop_scene_ids": direct_ids,
            "candidate_narrative_ids": [],
            "admitted_narrative_id": "",
            "status": "disabled" if not self.shadow_admission_enabled else "not_admitted",
            "reason": "shadow_disabled" if not self.shadow_admission_enabled else "no_strong_match",
            "candidates": [],
        }
        if not self.shadow_admission_enabled:
            return base
        compact_query = _compact(query)
        candidates: list[dict[str, Any]] = []
        for item in self._load():
            if item.get("integrity_status") != "ok":
                continue
            publication_status = str(item.get("publication_status") or "")
            allowed_publication_statuses = (
                {"reviewed", "published"}
                if self.live_injection_enabled
                else {"offline_reviewed", "reviewed", "published"}
            )
            if publication_status not in allowed_publication_statuses:
                continue
            if str(item.get("lifecycle") or "active") == "retired":
                continue
            title_aliases = [
                str(value or "").strip()
                for value in [
                    item.get("title"),
                    *(item.get("title_aliases", []) or []),
                ]
                if str(value or "").strip()
            ]
            primary_entities = [
                str(value or "").strip()
                for value in item.get("primary_entities", []) or []
                if str(value or "").strip()
            ]
            supporting_entities = [
                str(value or "").strip()
                for value in item.get("supporting_entities", []) or []
                if str(value or "").strip()
            ]
            query_cues = [
                str(value or "").strip()
                for value in item.get("query_cues", []) or []
                if str(value or "").strip()
            ]
            matched_titles = [value for value in title_aliases if _compact(value) in compact_query]
            matched_primary = [value for value in primary_entities if _compact(value) in compact_query]
            matched_supporting = [value for value in supporting_entities if _compact(value) in compact_query]
            matched_query_cues = [value for value in query_cues if _compact(value) in compact_query]
            matched_scenes = [
                scene_id
                for scene_id in item.get("linked_scene_ids", []) or []
                if scene_id in direct_ids
            ]
            exact_title = bool(matched_titles)
            two_independent_scenes = len(set(matched_scenes)) >= 2
            primary_with_cue = bool(matched_primary and matched_query_cues)
            supporting_with_cue = bool(matched_supporting and matched_query_cues)
            exact_primary = bool(matched_primary)
            admitted = (
                exact_title
                or two_independent_scenes
                or exact_primary
                or primary_with_cue
                or supporting_with_cue
            )
            if not any((matched_titles, matched_primary, matched_supporting, matched_scenes)):
                continue
            score = (
                (100 if exact_title else 0)
                + (80 + len(set(matched_scenes)) * 5 if two_independent_scenes else 0)
                + (50 if exact_primary else 0)
                + min(30, len(matched_query_cues) * 15)
                + (25 if supporting_with_cue else 0)
                + min(10, len(matched_supporting) * 3)
            )
            if exact_title:
                reason = "exact_title"
            elif two_independent_scenes:
                reason = "two_independent_first_hop_scenes"
            elif primary_with_cue:
                reason = "primary_entity_and_roll_query_cue"
            elif exact_primary:
                reason = "exact_primary_entity"
            elif supporting_with_cue:
                reason = "supporting_entity_and_roll_query_cue"
            else:
                reason = "supporting_entity_only"
            candidates.append(
                {
                    "narrative_id": item.get("narrative_id"),
                    "title": item.get("title"),
                    "score": score,
                    "admission_eligible": admitted,
                    "reason": reason,
                    "matched_titles": matched_titles,
                    "matched_primary_entities": matched_primary,
                    "matched_supporting_entities": matched_supporting,
                    "matched_query_cues": matched_query_cues,
                    "matched_scene_ids": matched_scenes,
                }
            )
        candidates.sort(key=lambda row: (-int(row.get("score") or 0), str(row.get("narrative_id") or "")))
        base["candidates"] = candidates[:10]
        eligible = [row for row in candidates if row.get("admission_eligible")]
        base["candidate_narrative_ids"] = [
            str(row.get("narrative_id") or "") for row in eligible
        ]
        if base["exact_evidence_route"] and candidates:
            base.update(
                {
                    "status": "source_detail_required",
                    "reason": "exact_date_wording_or_single_detail_prefers_scene_raw",
                }
            )
            return base
        if not eligible:
            if candidates:
                base["reason"] = "supporting_entity_is_not_an_admission_gate"
            return base
        top = eligible[0]
        second = eligible[1] if len(eligible) > 1 else None
        top_exact_title = str(top.get("reason") or "") == "exact_title"
        unambiguous = (
            second is None
            or top_exact_title
            or int(top.get("score") or 0) - int(second.get("score") or 0) >= self.ambiguity_margin
        )
        if not unambiguous:
            base.update(
                {
                    "status": "ambiguous_index_only",
                    "reason": "multiple_rolls_without_clear_winner",
                }
            )
            return base
        base.update(
            {
                "status": "shadow_admitted",
                "reason": str(top.get("reason") or "strong_match"),
                "admitted_narrative_id": str(top.get("narrative_id") or ""),
                "matched_scene_ids": list(top.get("matched_scene_ids") or []),
            }
        )
        return base

    def prepare_injection(
        self,
        query: str,
        direct_first_hop_scene_ids: list[str],
    ) -> tuple[str, str, dict[str, Any]]:
        """Return authored body, compact fallback index, and auditable gate debug.

        The first value is only the exact ``## 第一人称叙事`` body.  Source
        ledgers, linked ids, model notes, and the rest of the Markdown document
        stay behind the explicit ``read_narrative_roll`` surface.
        """

        debug = self.shadow_match(query, direct_first_hop_scene_ids)
        if not self.live_injection_enabled:
            return "", "", debug
        if str(debug.get("status") or "") != "shadow_admitted":
            return "", "", debug

        narrative_id = str(debug.get("admitted_narrative_id") or "").strip()
        item = next(
            (
                row
                for row in self._load()
                if str(row.get("narrative_id") or "").strip() == narrative_id
            ),
            None,
        )
        if item is None or str(item.get("integrity_status") or "") != "ok":
            debug.update(
                {
                    "status": "injection_blocked",
                    "reason": "admitted_roll_unavailable_or_invalid",
                }
            )
            return "", "", debug

        body = str(item.get("body") or "").strip()
        if not body:
            debug.update(
                {
                    "status": "injection_blocked",
                    "reason": "admitted_roll_body_empty",
                }
            )
            return "", "", debug

        title = str(item.get("title") or narrative_id).strip()
        time_start = str(item.get("time_start") or "").strip()
        time_end = str(item.get("time_end") or "").strip()
        time_range = " – ".join(value for value in (time_start, time_end) if value)
        current_status = str(item.get("current_status_cue") or item.get("lifecycle") or "active").strip()
        index_lines = [
            f"title: {title}",
            f"narrative_id: {narrative_id}",
        ]
        if time_range:
            index_lines.append(f"time_range: {time_range}")
        if current_status:
            index_lines.append(f"current_status: {current_status}")

        debug.update(
            {
                "status": "ready_for_injection",
                "visible_injection": False,
                "body_chars": len(body),
                "body_sha256": str(item.get("body_sha256") or ""),
                "injection_mode": "pending_budget",
            }
        )
        return body, "\n".join(index_lines), debug
