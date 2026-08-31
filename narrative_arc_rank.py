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
        passage_availability = self.passage_index.read_availability()
        scene_whole_search = getattr(
            self.embedding_engine,
            "search_scene_whole_by_embedding",
            None,
        )
        if (
            scene_ids
            and passage_availability.get("status") != "ok"
            and not callable(scene_whole_search)
        ):
            return {
                "status": "unavailable",
                "lane": "scene",
                "reason": passage_availability.get("reason") or "index_unavailable",
                "arc_key": request["arc_key"],
            }
        if event_ids and passage_availability.get("status") != "ok":
            event_availability = self.event_index.read_availability()
            if event_availability.get("status") != "ok":
                return {
                    "status": "unavailable",
                    "lane": "event",
                    "reason": passage_availability.get("reason")
                    or event_availability.get("reason")
                    or "index_unavailable",
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

        event_result = {
            "status": str(passage_availability.get("status") or "unavailable"),
            "reason": passage_availability.get("reason"),
            "matches": [],
            "indexed_owner_ids": [],
        }
        if event_ids and passage_availability.get("status") == "ok":
            try:
                event_result = self.passage_index.search_by_embedding(
                    query_embedding,
                    top_k=max(1, len(event_ids)),
                    owner_kinds=("event",),
                    passages_per_owner=1,
                    allowed_owner_ids={("event", event_id) for event_id in event_ids},
                )
            except Exception:
                return {
                    "status": "unavailable",
                    "lane": "event",
                    "reason": "index_read_failed",
                    "arc_key": request["arc_key"],
                }
        event_whole_result = (
            self.event_index.search_by_embedding(
                query_embedding,
                top_k=max(1, len(event_ids)),
                memory_kinds=("event",),
                min_importance_by_kind={"event": 1},
                allowed_memory_ids=set(event_ids),
            )
            if event_ids
            else {"status": "ok", "matches": []}
        )
        if (
            event_ids
            and event_result.get("status") != "ok"
            and event_whole_result.get("status") != "ok"
        ):
            return {
                "status": "unavailable",
                "lane": "event",
                "reason": event_whole_result.get("reason")
                or event_result.get("reason")
                or "index_unavailable",
                "arc_key": request["arc_key"],
            }
        scene_owners = {("scene", scene_id) for scene_id in scene_ids}
        scene_result = {
            "status": str(passage_availability.get("status") or "unavailable"),
            "reason": passage_availability.get("reason"),
            "matches": [],
            "indexed_owner_ids": [],
        }
        if scene_ids and passage_availability.get("status") == "ok":
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
        scene_whole_matches = (
            scene_whole_search(
                query_embedding,
                scene_ids=set(scene_ids),
                top_k=max(1, len(scene_ids)),
            )
            if scene_ids and callable(scene_whole_search)
            else []
        )
        if (
            scene_ids
            and scene_result.get("status") != "ok"
            and not callable(scene_whole_search)
        ):
            return {
                "status": "unavailable",
                "lane": "scene",
                "reason": scene_result.get("reason") or "index_unavailable",
                "arc_key": request["arc_key"],
            }

        ranked_event_map: dict[str, dict[str, Any]] = {}
        for item in event_result.get("matches", []) or []:
            memory_id = str(item.get("owner_id") or "")
            if memory_id in event_ids:
                passage = next(iter(item.get("passages") or []), {})
                score = float(item.get("score") or 0.0)
                ranked_event_map[memory_id] = {
                    "memory_kind": "event",
                    "memory_id": memory_id,
                    "score": score,
                    "score_components": {"passage_embedding": score},
                    "selected_embedding_route": "passage",
                    "passage_ordinal": passage.get("ordinal"),
                    "start_offset": passage.get("start_offset"),
                    "end_offset": passage.get("end_offset"),
                }
        for item in event_whole_result.get("matches", []) or []:
            memory_id = str(item.get("memory_id") or "")
            if memory_id not in event_ids:
                continue
            score = float(item.get("score") or 0.0)
            current = ranked_event_map.get(memory_id)
            if current is None:
                current = {
                    "memory_kind": "event",
                    "memory_id": memory_id,
                    "score": score,
                    "score_components": {},
                    "selected_embedding_route": "whole",
                    "passage_ordinal": None,
                    "start_offset": None,
                    "end_offset": None,
                }
                ranked_event_map[memory_id] = current
            current["score_components"]["whole_embedding"] = score
            if score > float(current["score"]):
                current.update(
                    score=score,
                    selected_embedding_route="whole",
                    passage_ordinal=None,
                    start_offset=None,
                    end_offset=None,
                )
        ranked_scene_map: dict[str, dict[str, Any]] = {}
        for item in scene_result.get("matches", []) or []:
            memory_id = str(item.get("owner_id") or "")
            if memory_id in scene_ids:
                passage = next(iter(item.get("passages") or []), {})
                score = float(item.get("score") or 0.0)
                ranked_scene_map[memory_id] = {
                    "memory_kind": "scene",
                    "memory_id": memory_id,
                    "score": score,
                    "score_components": {"passage_embedding": score},
                    "selected_embedding_route": "passage",
                    "passage_ordinal": passage.get("ordinal"),
                    "start_offset": passage.get("start_offset"),
                    "end_offset": passage.get("end_offset"),
                }
        for item in scene_whole_matches:
            memory_id = str(item.get("scene_id") or "")
            if memory_id not in scene_ids:
                continue
            score = float(item.get("score") or 0.0)
            current = ranked_scene_map.get(memory_id)
            if current is None:
                current = {
                    "memory_kind": "scene",
                    "memory_id": memory_id,
                    "score": score,
                    "score_components": {},
                    "selected_embedding_route": "whole",
                    "passage_ordinal": None,
                    "start_offset": None,
                    "end_offset": None,
                }
                ranked_scene_map[memory_id] = current
            current["score_components"]["whole_embedding"] = score
            if score > float(current["score"]):
                current.update(
                    score=score,
                    selected_embedding_route="whole",
                    passage_ordinal=None,
                    start_offset=None,
                    end_offset=None,
                )
        ranked_events = list(ranked_event_map.values())
        ranked_scenes = list(ranked_scene_map.values())
        ranked_events.sort(key=lambda item: (-item["score"], item["memory_id"]))
        ranked_scenes.sort(key=lambda item: (-item["score"], item["memory_id"]))
        for lane_rows in (ranked_events, ranked_scenes):
            for lane_rank, row in enumerate(lane_rows, start=1):
                row["lane_rank"] = lane_rank

        ranked: list[dict[str, Any]] = []
        for lane_rank in range(max(len(ranked_events), len(ranked_scenes))):
            for lane_name, lane_rows in (
                ("event", ranked_events),
                ("scene", ranked_scenes),
            ):
                if lane_rank < len(lane_rows) and len(ranked) < request["top_k"]:
                    ranked.append({**lane_rows[lane_rank], "candidate_lane": lane_name})

        indexed_events = {
            *ranked_event_map,
        }
        indexed_scenes = {
            *ranked_scene_map,
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
            "ranked_members": ranked,
            "lanes": {
                "event": {
                    "direct_member_count": len(event_ids),
                    "ranked_members": ranked_events,
                },
                "scene": {
                    "direct_member_count": len(scene_ids),
                    "ranked_members": ranked_scenes,
                },
            },
            "cross_lane_score_comparison": False,
            "within_owner_embedding_score": "max",
            "embedding_routes": ["whole", "passage_for_long_owner"],
            "importance_gate_applied": False,
            "unindexed_members": unindexed,
            "membership_source": "narrative_roll_registry+automatic_event_links",
            "membership_changed": False,
            "index_write_attempted": False,
        }
