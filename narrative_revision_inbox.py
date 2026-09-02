from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_STATUS_VALUES = {"pending", "dismissed", "absorbed"}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _compact(value: Any) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", str(value or "").lower())


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


def _excerpt(value: Any, limit: int = 420) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


class NarrativeRevisionInbox:
    """Derived review queue for human-authored Narrative Roll revisions.

    Entries are routing hints, never memory evidence and never publication
    authority.  The source identifiers, exact hashes and excerpts let the
    current author inspect the canonical Scene/Window Shadow before writing a
    revision through ``publish_narrative``.
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        roll_cfg = config.get("narrative_rolls", {})
        if not isinstance(roll_cfg, dict):
            roll_cfg = {}
        repo_root = Path(__file__).resolve().parent
        state_dir = Path(
            str(
                config.get("state_dir")
                or Path(str(config.get("buckets_dir") or repo_root / "buckets")).resolve().parent
                / "state"
            )
        ).resolve()
        configured = str(roll_cfg.get("revision_inbox_path") or "").strip()
        path = Path(configured) if configured else state_dir / "narrative_rolls" / "revision_inbox.json"
        if configured and not path.is_absolute():
            path = state_dir / path
        self.path = path.resolve()
        self._lock = threading.RLock()

    def _load(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"schema_version": "narrative-revision-inbox-v1", "items": []}
        except (OSError, ValueError, TypeError):
            return {"schema_version": "narrative-revision-inbox-v1", "items": []}
        if not isinstance(raw, dict) or not isinstance(raw.get("items"), list):
            return {"schema_version": "narrative-revision-inbox-v1", "items": []}
        return raw

    def _save(self, raw: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(
            json.dumps(raw, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temp, self.path)

    @staticmethod
    def _target_anchors(target: dict[str, Any]) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        fields = (
            ("title", [target.get("title")]),
            ("title_alias", target.get("title_aliases") or []),
            ("query_cue", target.get("query_cues") or []),
            ("current_status_cue", [target.get("current_status_cue")]),
        )
        seen: set[str] = set()
        for kind, values in fields:
            for value in values:
                text = str(value or "").strip()
                compact = _compact(text)
                if not compact or compact in seen:
                    continue
                seen.add(compact)
                rows.append({"kind": kind, "text": text, "compact": compact})
        return rows

    @classmethod
    def _match(
        cls,
        target: dict[str, Any],
        *,
        source_cues: list[str],
        source_text: str,
        allow_source_text_match: bool,
    ) -> list[dict[str, str]]:
        matches: list[dict[str, str]] = []
        compact_text = _compact(source_text)
        compact_cues = [(cue, _compact(cue)) for cue in source_cues if _compact(cue)]
        for anchor in cls._target_anchors(target):
            anchor_compact = anchor["compact"]
            for cue, cue_compact in compact_cues:
                exact = cue_compact == anchor_compact
                contained = (
                    min(len(cue_compact), len(anchor_compact)) >= 4
                    and (cue_compact in anchor_compact or anchor_compact in cue_compact)
                )
                if exact or contained:
                    matches.append(
                        {
                            "reason": "authored_scene_cue",
                            "source_cue": cue,
                            "anchor_kind": anchor["kind"],
                            "anchor": anchor["text"],
                        }
                    )
                    break
            else:
                if allow_source_text_match and len(anchor_compact) >= 4 and anchor_compact in compact_text:
                    matches.append(
                        {
                            "reason": "reviewed_anchor_in_source",
                            "source_cue": "",
                            "anchor_kind": anchor["kind"],
                            "anchor": anchor["text"],
                        }
                    )
        unique: list[dict[str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for match in matches:
            key = (match["reason"], match["source_cue"], match["anchor"])
            if key not in seen:
                seen.add(key)
                unique.append(match)
        return unique[:12]

    @staticmethod
    def _proposal_id(narrative_id: str, source_type: str, source_id: str, source_sha256: str) -> str:
        digest = _sha256(f"{narrative_id}\n{source_type}\n{source_id}\n{source_sha256}")[:24]
        return f"nrev_{digest}"

    def _consider(
        self,
        source: dict[str, Any],
        narrative_targets: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        source_type = str(source.get("source_type") or "").strip()
        source_id = str(source.get("source_id") or "").strip()
        source_text = str(source.get("source_text") or "")
        source_hash = str(source.get("source_sha256") or "").strip().lower() or _sha256(source_text)
        if source_type not in {"scene", "window_shadow"} or not source_id or not source_text.strip():
            return []
        source_cues = _string_list(source.get("source_cues"), limit=16)
        now = _now_utc()
        created: list[dict[str, Any]] = []
        with self._lock:
            raw = self._load()
            items = raw["items"]
            known = {str(item.get("proposal_id") or "") for item in items if isinstance(item, dict)}
            for target in narrative_targets:
                narrative_id = str(target.get("narrative_id") or "").strip()
                if not narrative_id or str(target.get("integrity_status") or "") != "ok":
                    continue
                matches = self._match(
                    target,
                    source_cues=source_cues,
                    source_text=source_text,
                    allow_source_text_match=bool(source.get("allow_source_text_match", False)),
                )
                if not matches:
                    continue
                proposal_id = self._proposal_id(narrative_id, source_type, source_id, source_hash)
                if proposal_id in known:
                    continue
                row = {
                    "proposal_id": proposal_id,
                    "narrative_id": narrative_id,
                    "narrative_title": str(target.get("title") or narrative_id),
                    "baseline_revision": int(target.get("revision") or 0),
                    "baseline_document_sha256": str(target.get("document_sha256") or ""),
                    "source_type": source_type,
                    "source_id": source_id,
                    "source_date": str(source.get("source_date") or "").strip(),
                    "source_sha256": source_hash,
                    "source_title": str(source.get("source_title") or source_id).strip(),
                    "source_excerpt": _excerpt(source_text),
                    "source_scene_ids": _string_list(source.get("source_scene_ids"), limit=80),
                    "matched_anchors": matches,
                    "status": "pending",
                    "draft_delta": "",
                    "review_note": "",
                    "created_at": now,
                    "updated_at": now,
                    "reviewed_at": "",
                    "absorbed_revision": 0,
                    "derived_only": True,
                    "evidence_authority": False,
                }
                items.append(row)
                known.add(proposal_id)
                created.append(dict(row))
            if created:
                raw["schema_version"] = "narrative-revision-inbox-v1"
                self._save(raw)
        return created

    def consider_scene(
        self,
        scene: dict[str, Any],
        narrative_targets: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        metadata = scene.get("metadata", {}) if isinstance(scene.get("metadata"), dict) else {}
        scene_id = str(scene.get("id") or metadata.get("id") or "").strip()
        content = str(scene.get("content") or "")
        return self._consider(
            {
                "source_type": "scene",
                "source_id": scene_id,
                "source_date": metadata.get("date") or metadata.get("created_at") or "",
                "source_sha256": _sha256(content),
                "source_title": metadata.get("name") or scene_id,
                "source_text": content,
                "source_cues": metadata.get("scene_cues") or [],
                "source_scene_ids": [scene_id],
                "allow_source_text_match": False,
            },
            narrative_targets,
        )

    def consider_window_shadow(
        self,
        window: dict[str, Any],
        narrative_targets: list[dict[str, Any]],
        *,
        attached_scenes: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        scene_cues: list[str] = []
        for scene in attached_scenes or []:
            metadata = scene.get("metadata", {}) if isinstance(scene.get("metadata"), dict) else {}
            scene_cues.extend(_string_list(metadata.get("scene_cues"), limit=16))
        return self._consider(
            {
                "source_type": "window_shadow",
                "source_id": window.get("window_id"),
                "source_date": window.get("source_date") or window.get("created_at") or "",
                "source_sha256": window.get("source_hash") or _sha256(window.get("content")),
                "source_title": "Window Shadow",
                "source_text": window.get("content"),
                "source_cues": scene_cues,
                "source_scene_ids": window.get("scene_bucket_ids") or [],
                "allow_source_text_match": True,
            },
            narrative_targets,
        )

    def consider_stale_roll(
        self,
        narrative: dict[str, Any],
        *,
        latest_material: dict[str, Any],
        material_count: int,
    ) -> list[dict[str, Any]]:
        """Flag one existing roll whose bound material changed after publication."""

        narrative_id = str(narrative.get("narrative_id") or "").strip()
        baseline_revision = int(narrative.get("revision") or 0)
        baseline_hash = str(narrative.get("document_sha256") or "").strip().lower()
        latest_at = str(latest_material.get("updated_at") or "").strip()
        source_type = str(latest_material.get("source_type") or "material").strip()
        source_id = str(latest_material.get("source_id") or "").strip()
        if not narrative_id or baseline_revision < 1 or not latest_at or not source_id:
            return []

        proposal_id = self._proposal_id(
            narrative_id,
            "material_freshness",
            str(baseline_revision),
            baseline_hash,
        )
        now = _now_utc()
        with self._lock:
            raw = self._load()
            existing = next(
                (
                    item
                    for item in raw["items"]
                    if isinstance(item, dict) and str(item.get("proposal_id") or "") == proposal_id
                ),
                None,
            )
            if existing is not None:
                existing.update(
                    {
                        "latest_material_at": latest_at,
                        "latest_material_type": source_type,
                        "latest_material_id": source_id,
                        "material_count": int(material_count or 0),
                        "updated_at": now,
                    }
                )
                self._save(raw)
                return []

            row = {
                "proposal_id": proposal_id,
                "proposal_kind": "existing_roll_update",
                "narrative_id": narrative_id,
                "narrative_title": str(narrative.get("title") or narrative_id),
                "baseline_revision": baseline_revision,
                "baseline_document_sha256": baseline_hash,
                "narrative_published_at": str(narrative.get("published_at") or ""),
                "latest_material_at": latest_at,
                "latest_material_type": source_type,
                "latest_material_id": source_id,
                "material_count": int(material_count or 0),
                "source_type": "material_freshness",
                "source_id": source_id,
                "source_date": latest_at,
                "source_sha256": str(latest_material.get("source_sha256") or ""),
                "source_title": str(latest_material.get("title") or source_id),
                "source_excerpt": _excerpt(latest_material.get("excerpt") or ""),
                "source_scene_ids": [source_id] if source_type == "scene" else [],
                "source_event_ids": [source_id] if source_type == "event" else [],
                "matched_anchors": [
                    {
                        "reason": "material_newer_than_narrative",
                        "source_cue": "",
                        "anchor_kind": "published_at",
                        "anchor": str(narrative.get("published_at") or ""),
                    }
                ],
                "status": "pending",
                "draft_delta": "",
                "review_note": "",
                "created_at": now,
                "updated_at": now,
                "reviewed_at": "",
                "absorbed_revision": 0,
                "derived_only": True,
                "evidence_authority": False,
            }
            raw["items"].append(row)
            self._save(raw)
            return [dict(row)]

    def consider_new_roll_candidates(
        self,
        candidates: list[dict[str, Any]],
        *,
        model: str,
    ) -> list[dict[str, Any]]:
        """Store external-model groupings as review hints, never as Narrative Rolls."""

        now = _now_utc()
        created: list[dict[str, Any]] = []
        with self._lock:
            raw = self._load()
            known = {
                str(item.get("proposal_id") or "")
                for item in raw["items"]
                if isinstance(item, dict)
            }
            for candidate in candidates:
                event_ids = _string_list(candidate.get("source_event_ids"), limit=40)
                title = str(candidate.get("title") or "").strip()
                reason = str(candidate.get("reason") or "").strip()
                if len(event_ids) < 2 or not title or not reason:
                    continue
                joined_ids = "\n".join(sorted(event_ids))
                proposal_id = self._proposal_id("new_roll", "event_group", joined_ids, _sha256(title))
                if proposal_id in known:
                    continue
                candidate_id = f"candidate_{proposal_id[5:]}"
                row = {
                    "proposal_id": proposal_id,
                    "proposal_kind": "new_roll_candidate",
                    "narrative_id": candidate_id,
                    "narrative_title": title,
                    "baseline_revision": 0,
                    "baseline_document_sha256": "",
                    "source_type": "event_group",
                    "source_id": candidate_id,
                    "source_date": str(candidate.get("latest_date") or ""),
                    "source_sha256": _sha256(joined_ids),
                    "source_title": title,
                    "source_excerpt": _excerpt(reason),
                    "source_scene_ids": [],
                    "source_event_ids": event_ids,
                    "matched_anchors": [
                        {
                            "reason": "external_model_grouping",
                            "source_cue": "",
                            "anchor_kind": "model",
                            "anchor": str(model or "external_model"),
                        }
                    ],
                    "candidate_confidence": str(candidate.get("confidence") or "medium"),
                    "candidate_model": str(model or ""),
                    "status": "pending",
                    "draft_delta": "",
                    "review_note": "",
                    "created_at": now,
                    "updated_at": now,
                    "reviewed_at": "",
                    "absorbed_revision": 0,
                    "derived_only": True,
                    "evidence_authority": False,
                }
                raw["items"].append(row)
                known.add(proposal_id)
                created.append(dict(row))
            if created:
                self._save(raw)
        return created

    def scan_metadata(self) -> dict[str, Any]:
        with self._lock:
            raw = self._load()
        value = raw.get("scan") if isinstance(raw.get("scan"), dict) else {}
        return dict(value)

    def record_scan(self, result: dict[str, Any]) -> None:
        with self._lock:
            raw = self._load()
            raw["scan"] = {
                **(raw.get("scan") if isinstance(raw.get("scan"), dict) else {}),
                **dict(result),
                "last_scan_at": _now_utc(),
            }
            self._save(raw)

    def reconcile_stale_rolls(self, stale_narrative_ids: set[str]) -> list[str]:
        """Remove pending derived freshness hints that the current scan disproves."""

        active = {str(value or "").strip() for value in stale_narrative_ids if str(value or "").strip()}
        removed: list[str] = []
        with self._lock:
            raw = self._load()
            kept = []
            for item in raw["items"]:
                if (
                    isinstance(item, dict)
                    and str(item.get("proposal_kind") or "") == "existing_roll_update"
                    and str(item.get("status") or "") == "pending"
                    and str(item.get("narrative_id") or "") not in active
                ):
                    removed.append(str(item.get("proposal_id") or ""))
                    continue
                kept.append(item)
            if removed:
                raw["items"] = kept
                self._save(raw)
        return removed

    def list(
        self,
        *,
        status: str = "pending",
        narrative_id: str = "",
        limit: int = 50,
    ) -> dict[str, Any]:
        safe_status = str(status or "pending").strip().lower()
        if safe_status not in _STATUS_VALUES | {"all"}:
            return {"status": "invalid", "reason": "invalid_status", "allowed": sorted(_STATUS_VALUES)}
        safe_narrative_id = str(narrative_id or "").strip()
        safe_limit = max(1, min(int(limit or 50), 200))
        with self._lock:
            items = [dict(item) for item in self._load()["items"] if isinstance(item, dict)]
        if safe_status != "all":
            items = [item for item in items if str(item.get("status") or "") == safe_status]
        if safe_narrative_id:
            items = [item for item in items if str(item.get("narrative_id") or "") == safe_narrative_id]
            items.sort(key=lambda item: (str(item.get("source_date") or ""), str(item.get("created_at") or "")))
        else:
            items.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        bounded_items = items[:safe_limit]
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in bounded_items:
            grouped.setdefault(str(item.get("narrative_id") or ""), []).append(item)
        trajectories = []
        for group_id, group_items in grouped.items():
            group_items.sort(
                key=lambda item: (
                    str(item.get("source_date") or ""),
                    str(item.get("created_at") or ""),
                )
            )
            trajectories.append(
                {
                    "narrative_id": group_id,
                    "narrative_title": str(group_items[0].get("narrative_title") or group_id),
                    "count": len(group_items),
                    "items": group_items,
                }
            )
        trajectories.sort(
            key=lambda group: max(
                (str(item.get("source_date") or item.get("created_at") or "") for item in group["items"]),
                default="",
            ),
            reverse=True,
        )
        return {
            "status": "ok",
            "mode": "derived_review_queue",
            "count": len(items),
            "items": bounded_items,
            "trajectories": trajectories,
            "scan": self.scan_metadata(),
            "boundary": "Review hints are not evidence and never update Narrative Rolls automatically.",
        }

    def review(
        self,
        proposal_id: str,
        *,
        action: str,
        draft_delta: str = "",
        note: str = "",
    ) -> dict[str, Any]:
        safe_id = str(proposal_id or "").strip()
        safe_action = str(action or "").strip().lower()
        if safe_action not in {"save_draft", "dismiss", "reopen"}:
            return {
                "status": "invalid",
                "reason": "invalid_action",
                "allowed": ["save_draft", "dismiss", "reopen"],
            }
        with self._lock:
            raw = self._load()
            item = next(
                (row for row in raw["items"] if isinstance(row, dict) and row.get("proposal_id") == safe_id),
                None,
            )
            if item is None:
                return {"status": "not_found", "proposal_id": safe_id}
            if safe_action == "save_draft":
                item["draft_delta"] = str(draft_delta or "").strip()
                item["review_note"] = str(note or "").strip()
                item["status"] = "pending"
            elif safe_action == "dismiss":
                item["review_note"] = str(note or "").strip()
                item["status"] = "dismissed"
            else:
                item["status"] = "pending"
                if note:
                    item["review_note"] = str(note).strip()
            item["reviewed_at"] = _now_utc()
            item["updated_at"] = item["reviewed_at"]
            self._save(raw)
            return {"status": "updated", "item": dict(item)}

    def mark_absorbed(
        self,
        narrative_id: str,
        *,
        source_scene_ids: list[str],
        revision: int,
    ) -> list[str]:
        linked = set(_string_list(source_scene_ids, limit=500))
        changed: list[str] = []
        with self._lock:
            raw = self._load()
            now = _now_utc()
            for item in raw["items"]:
                if not isinstance(item, dict):
                    continue
                if str(item.get("narrative_id") or "") != str(narrative_id or "").strip():
                    continue
                if (
                    str(item.get("proposal_kind") or "") == "existing_roll_update"
                    and int(item.get("baseline_revision") or 0) < int(revision or 0)
                ):
                    pass
                else:
                    if not linked:
                        continue
                    proposal_sources = set(_string_list(item.get("source_scene_ids"), limit=500))
                    if not proposal_sources or not proposal_sources.issubset(linked):
                        continue
                if item.get("status") == "absorbed":
                    continue
                item["status"] = "absorbed"
                item["absorbed_revision"] = int(revision or 0)
                item["reviewed_at"] = now
                item["updated_at"] = now
                changed.append(str(item.get("proposal_id") or ""))
            if changed:
                self._save(raw)
        return changed
