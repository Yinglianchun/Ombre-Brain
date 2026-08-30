from __future__ import annotations

from typing import Any


ARC_RANK_REQUEST_FIELDS = frozenset({"arc_key", "query", "top_k"})


def normalize_arc_rank_request(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "invalid", "reason": "json_body_must_be_object"}
    unexpected = sorted(str(key) for key in set(payload) - ARC_RANK_REQUEST_FIELDS)
    if unexpected:
        return {
            "status": "invalid",
            "reason": "unexpected_request_fields",
            "fields": unexpected,
        }
    arc_key = str(payload.get("arc_key") or "").strip()
    query = str(payload.get("query") or "").strip()
    if not arc_key:
        return {"status": "invalid", "reason": "arc_key_required"}
    if not query:
        return {"status": "invalid", "reason": "query_required"}
    if len(query) > 4000:
        return {"status": "invalid", "reason": "query_too_long"}
    try:
        top_k = int(payload.get("top_k", 8))
    except (TypeError, ValueError):
        return {"status": "invalid", "reason": "invalid_top_k"}
    if top_k < 1 or top_k > 8:
        return {"status": "invalid", "reason": "top_k_must_be_between_1_and_8"}
    return {"status": "ok", "arc_key": arc_key, "query": query, "top_k": top_k}


class NarrativeArcMemberRanker:
    """Rank frozen direct Arc members without changing registry or indexes."""

    def __init__(
        self,
        store: Any,
        embedding_engine: Any,
        event_index: Any,
        passage_index: Any,
        automatic_event_links: Any = None,
    ) -> None:
        self.store = store
        self.embedding_engine = embedding_engine
        self.event_index = event_index
        self.passage_index = passage_index
        self.automatic_event_links = automatic_event_links

    async def rank(self, *, arc_key: str, query: str, top_k: int = 8) -> dict[str, Any]:
        request = normalize_arc_rank_request(
            {"arc_key": arc_key, "query": query, "top_k": top_k}
        )
        if request.get("status") != "ok":
            return request

        arc = self.store.read_by_arc_key(request["arc_key"])
        if arc.get("status") != "ok":
            return arc

        automatic_links = (
            self.automatic_event_links(request["arc_key"])
            if callable(self.automatic_event_links)
            else []
        )
        event_ids = list(
            dict.fromkeys(
                [
                    *(str(value) for value in arc.get("linked_event_ids", []) if value),
                    *(
                        str(item.get("event_id") or "")
                        for item in automatic_links or []
                        if isinstance(item, dict) and str(item.get("event_id") or "")
                    ),
                ]
            )
        )
        scene_ids = list(dict.fromkeys(str(value) for value in arc.get("linked_scene_ids", []) if value))
        if event_ids:
            event_availability = self.event_index.read_availability()
            if event_availability.get("status") != "ok":
                return {
                    "status": "unavailable",
                    "lane": "event",
                    "reason": event_availability.get("reason") or "index_unavailable",
                    "arc_key": request["arc_key"],
                }
        if scene_ids:
            scene_availability = self.passage_index.read_availability()
            if scene_availability.get("status") != "ok":
                return {
                    "status": "unavailable",
                    "lane": "scene",
                    "reason": scene_availability.get("reason") or "index_unavailable",
                    "arc_key": request["arc_key"],
                }
        try:
            query_embedding = await self.embedding_engine.embed_query(request["query"])
        except Exception as exc:
            return {
                "status": "unavailable",
                "reason": "query_embedding_failed",
                "error": str(exc),
                "arc_key": request["arc_key"],
            }
        if not query_embedding:
            return {
                "status": "unavailable",
                "reason": "query_embedding_empty",
                "arc_key": request["arc_key"],
            }

        event_result = {"status": "ok", "matches": [], "indexed_memory_ids": []}
        if event_ids:
            try:
                event_result = self.event_index.search_by_embedding(
                    query_embedding,
                    top_k=max(1, len(event_ids)),
                    memory_kinds=("event",),
                    min_importance=1,
                    allowed_memory_ids=event_ids,
                )
            except Exception:
                return {
                    "status": "unavailable",
                    "lane": "event",
                    "reason": "index_read_failed",
                    "arc_key": request["arc_key"],
                }
            if event_result.get("status") != "ok":
                return {
                    "status": "unavailable",
                    "lane": "event",
                    "reason": event_result.get("reason") or "index_unavailable",
                    "arc_key": request["arc_key"],
                }
        scene_owners = {("scene", scene_id) for scene_id in scene_ids}
        scene_result = {"status": "ok", "matches": [], "indexed_owner_ids": []}
        if scene_ids:
            try:
                scene_result = self.passage_index.search_by_embedding(
                    query_embedding,
                    top_k=max(1, len(scene_ids)),
                    owner_kinds=("scene",),
                    passages_per_owner=1,
                    allowed_owner_ids=scene_owners,
                )
            except Exception:
                return {
                    "status": "unavailable",
                    "lane": "scene",
                    "reason": "index_read_failed",
                    "arc_key": request["arc_key"],
                }
            if scene_result.get("status") != "ok":
                return {
                    "status": "unavailable",
                    "lane": "scene",
                    "reason": scene_result.get("reason") or "index_unavailable",
                    "arc_key": request["arc_key"],
                }

        ranked: list[dict[str, Any]] = []
        for item in event_result.get("matches", []) or []:
            memory_id = str(item.get("memory_id") or "")
            if memory_id in event_ids:
                ranked.append(
                    {
                        "memory_kind": "event",
                        "memory_id": memory_id,
                        "score": float(item.get("score") or 0.0),
                    }
                )
        for item in scene_result.get("matches", []) or []:
            memory_id = str(item.get("owner_id") or "")
            if memory_id in scene_ids:
                ranked.append(
                    {
                        "memory_kind": "scene",
                        "memory_id": memory_id,
                        "score": float(item.get("score") or 0.0),
                    }
                )
        ranked.sort(key=lambda item: (-item["score"], item["memory_kind"], item["memory_id"]))

        indexed_events = {
            str(value) for value in event_result.get("indexed_memory_ids", []) or []
        }
        indexed_scenes = {
            str(item.get("owner_id") or "")
            for item in scene_result.get("indexed_owner_ids", []) or []
            if str(item.get("owner_kind") or "") == "scene"
        }
        unindexed = [
            *(
                {"memory_kind": "event", "memory_id": event_id}
                for event_id in event_ids
                if event_id not in indexed_events
            ),
            *(
                {"memory_kind": "scene", "memory_id": scene_id}
                for scene_id in scene_ids
                if scene_id not in indexed_scenes
            ),
        ]
        unindexed.sort(key=lambda item: (item["memory_kind"], item["memory_id"]))
        return {
            "status": "ok",
            "mode": "bounded_arc_member_rank",
            "arc_key": request["arc_key"],
            "narrative_id": arc.get("narrative_id"),
            "publication_status": arc.get("publication_status"),
            "top_k": request["top_k"],
            "direct_member_count": len(event_ids) + len(scene_ids),
            "ranked_members": ranked[: request["top_k"]],
            "unindexed_members": unindexed,
            "membership_source": "narrative_roll_registry+automatic_event_links",
            "membership_changed": False,
            "index_write_attempted": False,
        }
