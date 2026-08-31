"""Verify the source-balanced passage pool stays simulation-only."""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway import GatewayService


def row(kind: str, item_id: str, score: float) -> dict:
    return {
        "owner_kind": kind,
        "owner_id": item_id,
        "score": score,
        "passages": [{"text": f"evidence:{item_id}", "score": score}],
    }


class PassageIndex:
    def search_by_embedding(self, *_args, **kwargs):
        if tuple(kwargs.get("owner_kinds") or ()) == ("event",):
            result = {
                "status": "ok",
                "candidate_count": 1,
                "matches": [row("event", "event-a", 0.84)],
            }
        else:
            result = {
                "status": "ok",
                "candidate_count": 2,
                "matches": [
                    row("scene", "scene-a", 0.91),
                    row("scene", "scene-b", 0.89),
                ],
            }
        allowed = kwargs.get("allowed_owner_ids")
        if allowed is not None:
            result["matches"] = [
                item
                for item in result["matches"]
                if (item["owner_kind"], item["owner_id"]) in allowed
            ]
            result["candidate_count"] = len(result["matches"])
        return result


class CuePassageIndex:
    def search_by_embedding(self, *_args, **kwargs):
        match = row("scene", "scene-a", 0.95)
        match["matched_cues"] = ["初遇"]
        matches = [match] if "scene-a" in set(kwargs.get("allowed_scene_ids") or []) else []
        return {"status": "ok", "candidate_count": len(matches), "matches": matches}


class FactSemanticIndex:
    def search_by_embedding(self, *_args, **kwargs):
        assert kwargs["memory_kinds"] == ("event",)
        assert kwargs["min_importance_by_kind"] == {"event": 1}
        matches = [
            {"memory_kind": "event", "memory_id": "event-a", "score": 0.78, "importance": 3},
        ]
        allowed = kwargs.get("allowed_memory_ids")
        if allowed is not None:
            matches = [item for item in matches if item["memory_id"] in allowed]
        return {"status": "ok", "candidate_count": len(matches), "matches": matches}


class LexicalIndex:
    def search(self, *_args, **kwargs):
        assert kwargs["memory_kinds"] == ("event",)
        match = row("event", "event-a", 4.2)
        match.update({
            "importance": 4,
            "candidate_sources": ["fact_event_lexical"],
            "specific_terms": ["初遇"],
        })
        allowed = kwargs.get("allowed_memory_ids")
        matches = [match] if allowed is None or "event-a" in allowed else []
        return {"status": "ok", "candidate_count": len(matches), "matches": matches}


service = GatewayService.__new__(GatewayService)
service.passage_candidate_shadow_enabled = True
service.passage_shadow_min_fact_event_importance = 3
service.passage_shadow_min_fact_importance = 3
service.passage_shadow_min_event_importance = 1
service.passage_shadow_index = PassageIndex()
service.cue_passage_shadow_index = CuePassageIndex()
service.fact_event_semantic_index = FactSemanticIndex()
service.fact_event_lexical_shadow_index = LexicalIndex()
service._passage_candidate_shadow_sync = {"status": "ok", "decision_applied": False}
service._passage_candidate_shadow_catalog = {
    "scene-a": {"owner_kind": "scene", "title": "A", "memory_date": "2026-01-01"},
    "scene-b": {"owner_kind": "scene", "title": "B", "memory_date": "2026-08-30"},
    "event-a": {
        "owner_kind": "event",
        "title": "E",
        "importance": 3,
        "body": "event body",
        "memory_date": "2026-08-20",
    },
}

active_scene = {
    "id": "scene-active",
    "content": "active",
    "metadata": {"object_kind": "scene", "active": True, "scene_status": "active"},
}
assert service._passage_shadow_scene(active_scene) is not None
for dirty_archive_metadata in (
    {"object_kind": "scene", "type": "archived", "active": True},
    {"object_kind": "scene", "scene_status": "archived", "active": True},
    {"object_kind": "scene", "type": "dynamic", "active": False},
):
    assert service._passage_shadow_scene({
        "id": "scene-archived",
        "content": "archived",
        "metadata": dirty_archive_metadata,
    }) is None

debug = service._passage_candidate_shadow_debug("我们的初遇", [1.0, 0.0])
assert debug["status"] == "ok"
assert debug["decision_applied"] is False
assert debug["live_injection_enabled"] is False
assert debug["policy"]["duplicate_score_boost"] is False
assert len(debug["candidates"]) == 3
scene_a = next(item for item in debug["candidates"] if item["owner_id"] == "scene-a")
assert scene_a["candidate_lane"] == "scene"
assert scene_a["score"] == 0.91
assert scene_a["candidate_sources"] == [
    "scene_passage_embedding",
    "scene_cue_candidate",
]
event_a = next(item for item in debug["candidates"] if item["owner_id"] == "event-a")
assert event_a["candidate_lane"] == "event"
assert event_a["score"] == 0.84
assert debug["policy"]["cross_lane_score_comparison"] is False
assert debug["policy"]["cue_contributes_score"] is False
assert debug["policy"]["event_eligibility"] == "all_active"
assert "fact" not in debug["lanes"]

recent_debug = service._passage_candidate_shadow_debug("最近的初遇", [1.0, 0.0])
recent_scenes = recent_debug["lanes"]["scene"]["matches"]
assert [item["owner_id"] for item in recent_scenes[:2]] == ["scene-b", "scene-a"]
assert recent_scenes[0]["freshness"]["explicit_recent_intent"] is True


class ObservedScope:
    def resolve_query(self, query: str):
        if query == "阿尼亚":
            return {"status": "scope_only", "operator": "none", "scope_anchor": {"arc_key": "work:spy"}}
        if query == "看到哪了":
            return {"status": "insufficient_scope", "operator": "latest_relevant_member", "scope_anchor": None}
        if query == "约尔后来怎么发展":
            return {"status": "ambiguous_scope", "operator": "timeline", "scope_anchor": None}
        if query == "事情":
            return {"status": "no_scope", "operator": "none", "scope_anchor": None}
        return {
            "status": "scoped_recall",
            "operator": "latest_relevant_member",
            "scope_anchor": {"arc_key": "work:spy"},
        }

    def link_candidates(self, *_args):
        return []


service.observed_entity_shadow_index = ObservedScope()
service._passage_candidate_shadow_arc_members = {
    "work:spy": {("scene", "scene-b"), ("event", "event-a")}
}
service._passage_candidate_shadow_arc_cards = {
    "work:spy": {
        "arc_key": "work:spy",
        "narrative_id": "narrative_spy",
        "title": "《间谍过家家》共同观看",
        "member_count": 2,
        "narrative_available": True,
        "read_hint": "可按需读取",
    }
}
scoped = service._passage_candidate_shadow_debug("《间谍过家家》看到哪了", [1.0, 0.0])
assert {item["owner_id"] for item in scoped["candidates"]} == {"scene-b", "event-a"}, scoped
assert all(item["arc_cards"][0]["read_hint"] == "可按需读取" for item in scoped["candidates"])
assert scoped["policy"]["scope_applied_before_candidate_search"] is True, scoped
assert service._passage_candidate_shadow_debug("阿尼亚", [1.0, 0.0])["reason"] == (
    "entity_without_recall_intent"
)
assert service._passage_candidate_shadow_debug("看到哪了", [1.0, 0.0])["reason"] == (
    "scope_required_for_deictic_intent"
)
assert service._passage_candidate_shadow_debug("约尔后来怎么发展", [1.0, 0.0])[
    "reason"
] == "ambiguous_entity_scope"
assert service._passage_candidate_shadow_debug("事情", [1.0, 0.0])["reason"] == (
    "global_query_lacks_specific_terms"
)
del service.observed_entity_shadow_index

long_query = (
    "可能是一种初恋情结吧...不希望这种人也用ChatGPT，"
    "虽然那只是一个AI产品，而且你原生家庭也没比Anthropic好到哪去"
)
query_views = service._deterministic_passage_query_views(long_query)
assert query_views == [
    long_query,
    "可能是一种初恋情结吧",
    "不希望这种人也用ChatGPT",
]
assert service._deterministic_passage_query_views("我们的初遇") == ["我们的初遇"]


class QueryEmbedding:
    def __init__(self):
        self.calls: list[str] = []

    async def embed_query(self, query: str):
        self.calls.append(query)
        return [0.0, 1.0]


query_view_service = GatewayService.__new__(GatewayService)
query_view_service.embedding_engine = QueryEmbedding()


class ClausePassageIndex:
    def search_by_embedding(self, vector: list[float], **_kwargs):
        assert vector == [0.0, 1.0]
        return {
            "status": "ok",
            "candidate_count": 1,
            "matches": [row("scene", "scene-target", 0.88)],
        }


class ClauseCuePassageIndex:
    def search_by_embedding(self, vector: list[float], **_kwargs):
        assert vector == [0.0, 1.0]
        return {
            "status": "ok",
            "candidate_count": 1,
            "matches": [row("scene", "scene-target", 0.91)],
        }


query_view_service.passage_shadow_index = ClausePassageIndex()
query_view_service.cue_passage_shadow_index = ClauseCuePassageIndex()
query_view_service._passage_candidate_shadow_catalog = {
    "scene-target": {"owner_kind": "scene", "title": "Target", "importance": None},
}


def fake_passage_debug(self, _query: str, vector: list[float]):
    assert vector == [1.0, 0.0]
    return {
        "status": "ok",
        "decision_applied": False,
        "live_injection_enabled": False,
        "policy": {"pool_limit": 7},
        "candidates": [row("scene", "scene-baseline", 0.8)],
    }


query_view_service._passage_candidate_shadow_debug = types.MethodType(
    fake_passage_debug,
    query_view_service,
)
query_view_debug = asyncio.run(
    query_view_service._passage_candidate_query_view_shadow_debug(
        long_query,
        [1.0, 0.0],
    )
)
query_view_shadow = query_view_debug["query_view_shadow"]
assert query_view_shadow["status"] == "ok"
assert query_view_shadow["decision_applied"] is False
assert query_view_shadow["added_owner_ids"] == ["scene-target"]
assert [item["owner_id"] for item in query_view_shadow["candidates"]] == [
    "scene-baseline",
    "scene-target",
]

weak_trigger_semantic = {
    "applied_action": "recall",
    "retrieval_budget": {
        "cheap_retrieval": {
            "candidates": [{
                "canonical_scene": True,
                "body_semantic_score": 0.6088,
                "exact_anchor_match": False,
                "full_title_recall_match": False,
                "source_bound_raw_quote_match": False,
            }],
        },
        "passage_candidate_shadow": {},
    },
}
service._attach_passage_weak_candidate_trigger_shadow(
    long_query,
    weak_trigger_semantic,
    recalled_ids=["scene-live"],
)
weak_trigger = weak_trigger_semantic["retrieval_budget"]["passage_candidate_shadow"][
    "weak_candidate_trigger_shadow"
]
assert weak_trigger["would_trigger"] is True
assert weak_trigger["reason"] == "multiclause_body_semantic_gray_zone"
assert weak_trigger["decision_applied"] is False
assert weak_trigger["live_execution_changed"] is False

strong_trigger_semantic = {
    "applied_action": "recall",
    "retrieval_budget": {
        "cheap_retrieval": {
            "candidates": [{
                "canonical_scene": True,
                "body_semantic_score": 0.71,
                "exact_anchor_match": False,
            }],
        },
        "passage_candidate_shadow": {},
    },
}
service._attach_passage_weak_candidate_trigger_shadow(
    long_query,
    strong_trigger_semantic,
    recalled_ids=["scene-strong"],
)
strong_trigger = strong_trigger_semantic["retrieval_budget"]["passage_candidate_shadow"][
    "weak_candidate_trigger_shadow"
]
assert strong_trigger["would_trigger"] is False
assert strong_trigger["reason"] == "strong_candidate"

skip_trigger_semantic = {
    "applied_action": "skip",
    "retrieval_budget": {
        "cheap_retrieval": {"candidates": []},
        "passage_candidate_shadow": {},
    },
}
service._attach_passage_weak_candidate_trigger_shadow(
    "今天好热",
    skip_trigger_semantic,
    recalled_ids=[],
)
skip_trigger = skip_trigger_semantic["retrieval_budget"]["passage_candidate_shadow"][
    "weak_candidate_trigger_shadow"
]
assert skip_trigger["would_trigger"] is False
assert skip_trigger["reason"] == "route_skip"


async def verify_weak_trigger_controls_query_view_execution() -> None:
    def semantic_payload(
        score: float,
        *,
        action: str = "recall",
        exact_anchor: bool = False,
        memory_need: str = "",
        surface_route: str = "",
    ) -> dict:
        candidates = [] if action == "skip" else [{
            "canonical_scene": True,
            "body_semantic_score": score,
            "exact_anchor_match": exact_anchor,
            "full_title_recall_match": False,
            "source_bound_raw_quote_match": False,
        }]
        retrieval_budget = {
            "cheap_retrieval": {"candidates": candidates},
            "passage_candidate_shadow": fake_passage_debug(
                query_view_service,
                long_query,
                [1.0, 0.0],
            ),
        }
        if memory_need:
            retrieval_budget["memory_need"] = memory_need
        if surface_route:
            retrieval_budget["surface_route"] = surface_route
        return {
            "applied_action": action,
            "retrieval_budget": retrieval_budget,
        }

    query_view_service.embedding_engine.calls.clear()
    disabled_semantic = semantic_payload(0.6088)
    await query_view_service._apply_passage_weak_candidate_query_view_shadow(
        long_query,
        [1.0, 0.0],
        disabled_semantic,
        recalled_ids=["scene-live"],
    )
    disabled_passage = disabled_semantic["retrieval_budget"][
        "passage_candidate_shadow"
    ]
    assert disabled_passage["query_view_shadow"]["status"] == (
        "disabled_explicit_opt_in_required"
    )
    assert disabled_passage["query_view_shadow"]["execution_trigger_applied"] is False
    assert query_view_service.embedding_engine.calls == []

    weak_semantic = semantic_payload(0.6088)
    await query_view_service._apply_passage_weak_candidate_query_view_shadow(
        long_query,
        [1.0, 0.0],
        weak_semantic,
        recalled_ids=["scene-live"],
        execution_enabled=True,
    )
    weak_passage = weak_semantic["retrieval_budget"]["passage_candidate_shadow"]
    weak_applied = weak_passage["weak_candidate_trigger_shadow"]
    assert weak_applied["would_trigger"] is True
    assert weak_applied["decision_applied"] is True
    assert weak_applied["live_execution_changed"] is False
    assert weak_applied["query_view_execution_changed"] is True
    assert weak_applied["execution_scope"] == "simulation_shadow_only"
    assert weak_passage["query_view_shadow"]["status"] == "ok"
    assert weak_passage["query_view_shadow"]["execution_trigger_applied"] is True
    assert query_view_service.embedding_engine.calls == query_views[1:]

    query_view_service.embedding_engine.calls.clear()
    single_view_semantic = semantic_payload(0.0)
    await query_view_service._apply_passage_weak_candidate_query_view_shadow(
        "又在翻我们的聊天记录，配上bgm更好哭了><",
        [1.0, 0.0],
        single_view_semantic,
        recalled_ids=[],
        execution_enabled=True,
    )
    single_view_passage = single_view_semantic["retrieval_budget"][
        "passage_candidate_shadow"
    ]
    single_view_trigger = single_view_passage["weak_candidate_trigger_shadow"]
    assert single_view_trigger["would_trigger"] is False
    assert single_view_trigger["weak_candidate_detected"] is True
    assert single_view_trigger["weak_candidate_reason"] == "no_formal_memory_injected"
    assert single_view_trigger["reason"] == "single_query_view_no_expansion"
    assert query_view_service.embedding_engine.calls == []

    current_query = "现在记忆库有18个工具，感觉太多了。当前配置是不是又变了"
    current_semantic = semantic_payload(
        0.0,
        memory_need="optional",
        surface_route="present_reality",
    )
    await query_view_service._apply_passage_weak_candidate_query_view_shadow(
        current_query,
        [1.0, 0.0],
        current_semantic,
        recalled_ids=[],
        execution_enabled=True,
    )
    current_passage = current_semantic["retrieval_budget"]["passage_candidate_shadow"]
    current_trigger = current_passage["weak_candidate_trigger_shadow"]
    assert current_trigger["would_trigger"] is False
    assert current_trigger["reason"] == "current_turn_optional_query_view_veto"
    assert current_trigger["execution_gate"]["current_turn_optional"] is True
    assert current_trigger["execution_gate"]["current_turn_optional_veto"] is True
    assert query_view_service.embedding_engine.calls == []

    current_gray_semantic = semantic_payload(
        0.6088,
        memory_need="optional",
        surface_route="present_reality",
    )
    await query_view_service._apply_passage_weak_candidate_query_view_shadow(
        current_query,
        [1.0, 0.0],
        current_gray_semantic,
        recalled_ids=["scene-live"],
        execution_enabled=True,
    )
    current_gray_passage = current_gray_semantic["retrieval_budget"][
        "passage_candidate_shadow"
    ]
    current_gray_trigger = current_gray_passage["weak_candidate_trigger_shadow"]
    assert current_gray_trigger["would_trigger"] is True
    assert current_gray_trigger["reason"] == "multiclause_body_semantic_gray_zone"
    assert current_gray_trigger["execution_gate"]["current_turn_optional"] is True
    assert current_gray_trigger["execution_gate"]["current_turn_optional_veto"] is False
    assert query_view_service.embedding_engine.calls == (
        query_view_service._deterministic_passage_query_views(current_query)[1:]
    )

    query_view_service.embedding_engine.calls.clear()
    strong_semantic = semantic_payload(0.71)
    await query_view_service._apply_passage_weak_candidate_query_view_shadow(
        long_query,
        [1.0, 0.0],
        strong_semantic,
        recalled_ids=["scene-strong"],
        execution_enabled=True,
    )
    strong_passage = strong_semantic["retrieval_budget"]["passage_candidate_shadow"]
    assert strong_passage["query_view_shadow"]["status"] == (
        "skipped_by_weak_candidate_trigger"
    )
    assert strong_passage["query_view_shadow"]["trigger_reason"] == "strong_candidate"
    assert query_view_service.embedding_engine.calls == []

    exact_semantic = semantic_payload(0.40, exact_anchor=True)
    await query_view_service._apply_passage_weak_candidate_query_view_shadow(
        long_query,
        [1.0, 0.0],
        exact_semantic,
        recalled_ids=["scene-exact"],
        execution_enabled=True,
    )
    exact_passage = exact_semantic["retrieval_budget"]["passage_candidate_shadow"]
    assert exact_passage["query_view_shadow"]["status"] == (
        "skipped_by_weak_candidate_trigger"
    )
    assert exact_passage["query_view_shadow"]["trigger_reason"] == "direct_exact_evidence"
    assert query_view_service.embedding_engine.calls == []

    skip_semantic = semantic_payload(0.0, action="skip")
    await query_view_service._apply_passage_weak_candidate_query_view_shadow(
        long_query,
        [1.0, 0.0],
        skip_semantic,
        recalled_ids=[],
        execution_enabled=True,
    )
    skip_passage = skip_semantic["retrieval_budget"]["passage_candidate_shadow"]
    assert skip_passage["query_view_shadow"]["status"] == (
        "skipped_by_weak_candidate_trigger"
    )
    assert skip_passage["query_view_shadow"]["trigger_reason"] == "route_skip"
    assert query_view_service.embedding_engine.calls == []


async def verify_full_handler_uses_prefetched_query_vector() -> None:
    handler = GatewayService.__new__(GatewayService)
    captured_vectors: list[list[float]] = []

    async def fake_prepare_payload(self, *_args, **kwargs):
        assert kwargs["semantic_recall_result"][1] == [0.25, 0.75]
        return {}, ["scene-live"], {"semantic_recall_debug": {}}

    async def fake_apply(self, _query, query_embedding, _semantic_debug, **kwargs):
        assert kwargs["execution_enabled"] is True
        captured_vectors.append(query_embedding)

    handler.prepare_payload = types.MethodType(fake_prepare_payload, handler)
    handler._apply_passage_weak_candidate_query_view_shadow = types.MethodType(
        fake_apply,
        handler,
    )
    handler._hook_recall_cards_from_debug = lambda *_args, **_kwargs: []
    handler._hook_recall_full_dynamic_context = lambda *_args, **_kwargs: ""
    handler._clip_text = lambda text, _limit: text
    handler._render_hook_recall_full_additional_context = lambda _context: ""

    response = await handler._handle_hook_recall_full(
        query=long_query,
        session_id="weak-trigger-vector-check",
        messages=[{"role": "user", "content": long_query}],
        model="gpt-5.5",
        max_cards=3,
        max_chars=1000,
        max_context_chars=4200,
        include_diffused=False,
        include_context_debug=False,
        include_debug=True,
        include_recent_context=False,
        semantic_recall_result=({}, [0.25, 0.75]),
        passage_query_view_shadow_enabled=True,
        record_hook_injection=False,
    )
    assert response.status_code == 200
    assert captured_vectors == [[0.25, 0.75]]


async def verify_mutation_refresh_queue() -> None:
    refresh_service = GatewayService.__new__(GatewayService)
    refresh_service.passage_candidate_shadow_enabled = True
    refresh_service._passage_shadow_refresh_requested = False
    refresh_service._passage_shadow_refresh_scene_ids = set()
    refresh_service._passage_shadow_refresh_fact_event_ids = set()
    refresh_service._passage_shadow_refresh_task = None
    refresh_service._passage_candidate_shadow_sync = {}
    refresh_service._clear_gateway_bucket_cache = lambda: None

    class Buckets:
        async def list_all(self, *, include_archive: bool):
            assert include_archive is True
            return [{"id": "scene-edited"}]

    refresh_service.bucket_mgr = Buckets()
    calls = []

    async def fake_sync(self, buckets, *, apply_passage_embeddings=False):
        calls.append((buckets, apply_passage_embeddings))
        await asyncio.sleep(0)
        return {"status": "ok", "decision_applied": False}

    refresh_service._sync_passage_candidate_shadow = types.MethodType(
        fake_sync,
        refresh_service,
    )
    queued = refresh_service._queue_passage_candidate_shadow_refresh(
        scene_ids=["scene-edited"],
        fact_event_ids=["event-new"],
    )
    assert queued["status"] == "queued"
    await refresh_service._passage_shadow_refresh_task
    assert len(calls) == 1
    assert calls[0][1] is True
    assert refresh_service._passage_candidate_shadow_sync["requested_scene_ids"] == [
        "scene-edited"
    ]
    assert refresh_service._passage_candidate_shadow_sync["requested_fact_event_ids"] == [
        "event-new"
    ]
    assert refresh_service._passage_candidate_shadow_sync["refresh_source"] == (
        "canonical_mutation_notification"
    )


async def verify_warm_plans_and_bounded_refresh_applies() -> None:
    warm_service = GatewayService.__new__(GatewayService)
    warm_service.passage_shadow_min_fact_importance = 3
    warm_service.passage_shadow_min_event_importance = 1
    warm_service.passage_shadow_auto_refresh_max_passages = 3

    class Store:
        def list(self, **_kwargs):
            return {
                "items": [
                    {
                        "item_id": "event-warm",
                        "item_type": "event",
                        "title": "warm event",
                        "body": "event body",
                        "importance": 1,
                    }
                ],
                "count": 1,
            }

    class PlannedPassageIndex:
        def __init__(self):
            self.to_embed = 40
            self.calls = []

        async def sync(self, *, scenes, events, dry_run=False):
            self.calls.append(
                {
                    "dry_run": dry_run,
                    "scenes": len(scenes),
                    "events": len(events),
                }
            )
            if dry_run:
                return {
                    "status": "dry_run",
                    "owners": len(scenes) + len(events),
                    "passages": self.to_embed,
                    "to_embed": self.to_embed,
                    "stale_owners": 0,
                }
            return {
                "status": "ok",
                "embedded": self.to_embed,
                "removed_owners": 0,
            }

        def passages_for_owners(self, *_args, **_kwargs):
            return {}

    class PlannedCueIndex:
        async def sync(self, **_kwargs):
            return {"status": "dry_run", "to_bind": 4}

    class PlannedLexicalIndex:
        def sync(self, *_args, **_kwargs):
            return {"status": "ok"}

    class PlannedSemanticIndex:
        async def sync(self, *_args, **_kwargs):
            return {"status": "ok"}

    class PlannedObservedEntityIndex:
        def __init__(self):
            self.calls = []

        def sync(self, *, owners, arc_profiles, dry_run=False):
            self.calls.append(
                {
                    "dry_run": dry_run,
                    "owners": len(owners),
                    "arcs": len(arc_profiles),
                }
            )
            return {"status": "dry_run" if dry_run else "ok"}

    class EmptyNarrativeStore:
        def recall_scope_profiles(self):
            return []

    class EmptySceneEvidenceStore:
        def list_active_for_scenes(self, _scene_ids):
            return {}

    passage_index = PlannedPassageIndex()
    observed_index = PlannedObservedEntityIndex()
    warm_service.fact_event_store = Store()
    warm_service.passage_shadow_index = passage_index
    warm_service.cue_passage_shadow_index = PlannedCueIndex()
    warm_service.fact_event_lexical_shadow_index = PlannedLexicalIndex()
    warm_service.fact_event_semantic_index = PlannedSemanticIndex()
    warm_service.observed_entity_shadow_index = observed_index
    warm_service.narrative_roll_store = EmptyNarrativeStore()
    warm_service.scene_evidence_store = EmptySceneEvidenceStore()

    buckets = [
        {
            "id": "scene-warm",
            "content": "scene body",
            "metadata": {"object_kind": "scene", "active": True},
        }
    ]
    warm = await warm_service._sync_passage_candidate_shadow(buckets)
    assert passage_index.calls == [{"dry_run": True, "scenes": 1, "events": 1}]
    assert warm["passage"]["status"] == "stale", warm
    assert warm["passage"]["reason"] == "explicit_backfill_required", warm
    assert warm["passage"]["embeddings_applied"] is False, warm
    assert observed_index.calls == [{"dry_run": True, "owners": 2, "arcs": 0}]

    passage_index.to_embed = 2
    passage_index.calls.clear()
    refreshed = await warm_service._sync_passage_candidate_shadow(
        buckets,
        apply_passage_embeddings=True,
    )
    assert [call["dry_run"] for call in passage_index.calls] == [True, False]
    assert refreshed["passage"]["embeddings_applied"] is True, refreshed
    assert refreshed["passage"]["apply_source"] == "bounded_mutation_refresh", refreshed
    assert observed_index.calls[-1] == {"dry_run": False, "owners": 2, "arcs": 0}


asyncio.run(verify_weak_trigger_controls_query_view_execution())
asyncio.run(verify_full_handler_uses_prefetched_query_vector())
asyncio.run(verify_mutation_refresh_queue())
asyncio.run(verify_warm_plans_and_bounded_refresh_applies())

print("PASSAGE_CANDIDATE_SIMULATION_SHADOW_OK")
