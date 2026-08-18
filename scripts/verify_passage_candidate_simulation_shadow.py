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
    def search_by_embedding(self, *_args, **_kwargs):
        return {"status": "ok", "candidate_count": 2, "matches": [
            row("scene", "scene-a", 0.91),
            row("scene", "scene-b", 0.89),
        ]}


class CuePassageIndex:
    def search_by_embedding(self, *_args, **_kwargs):
        match = row("scene", "scene-a", 0.95)
        match["matched_cues"] = ["初遇"]
        return {"status": "ok", "candidate_count": 1, "matches": [match]}


class FactSemanticIndex:
    def search_by_embedding(self, *_args, **kwargs):
        assert kwargs["min_importance"] == 3
        return {"status": "ok", "candidate_count": 2, "matches": [
            {"memory_kind": "fact", "memory_id": "fact-a", "score": 0.82, "importance": 4},
            {"memory_kind": "event", "memory_id": "event-a", "score": 0.78, "importance": 3},
        ]}


class LexicalIndex:
    def search(self, *_args, **_kwargs):
        match = row("fact", "fact-a", 4.2)
        match.update({
            "importance": 4,
            "candidate_sources": ["fact_event_lexical"],
            "specific_terms": ["初遇"],
        })
        return {"status": "ok", "candidate_count": 1, "matches": [match]}


service = GatewayService.__new__(GatewayService)
service.passage_candidate_shadow_enabled = True
service.passage_shadow_min_fact_event_importance = 3
service.passage_shadow_index = PassageIndex()
service.cue_passage_shadow_index = CuePassageIndex()
service.fact_event_semantic_index = FactSemanticIndex()
service.fact_event_lexical_shadow_index = LexicalIndex()
service._passage_candidate_shadow_sync = {"status": "ok", "decision_applied": False}
service._passage_candidate_shadow_catalog = {
    "scene-a": {"owner_kind": "scene", "title": "A"},
    "scene-b": {"owner_kind": "scene", "title": "B"},
    "fact-a": {"owner_kind": "fact", "title": "F", "importance": 4, "body": "fact body"},
    "event-a": {"owner_kind": "event", "title": "E", "importance": 3, "body": "event body"},
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
assert len(debug["candidates"]) == 4
scene_a = next(item for item in debug["candidates"] if item["owner_id"] == "scene-a")
assert scene_a["candidate_lane"] == "cue_passage"
assert scene_a["candidate_sources"] == ["cue_passage_embedding", "passage_embedding"]
fact_a = next(item for item in debug["candidates"] if item["owner_id"] == "fact-a")
assert fact_a["candidate_sources"] == [
    "fact_event_body_embedding",
    "fact_event_lexical",
]

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
    ) -> dict:
        candidates = [] if action == "skip" else [{
            "canonical_scene": True,
            "body_semantic_score": score,
            "exact_anchor_match": exact_anchor,
            "full_title_recall_match": False,
            "source_bound_raw_quote_match": False,
        }]
        return {
            "applied_action": action,
            "retrieval_budget": {
                "cheap_retrieval": {"candidates": candidates},
                "passage_candidate_shadow": fake_passage_debug(
                    query_view_service,
                    long_query,
                    [1.0, 0.0],
                ),
            },
        }

    query_view_service.embedding_engine.calls.clear()
    weak_semantic = semantic_payload(0.6088)
    await query_view_service._apply_passage_weak_candidate_query_view_shadow(
        long_query,
        [1.0, 0.0],
        weak_semantic,
        recalled_ids=["scene-live"],
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
    strong_semantic = semantic_payload(0.71)
    await query_view_service._apply_passage_weak_candidate_query_view_shadow(
        long_query,
        [1.0, 0.0],
        strong_semantic,
        recalled_ids=["scene-strong"],
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
    )
    skip_passage = skip_semantic["retrieval_budget"]["passage_candidate_shadow"]
    assert skip_passage["query_view_shadow"]["status"] == (
        "skipped_by_weak_candidate_trigger"
    )
    assert skip_passage["query_view_shadow"]["trigger_reason"] == "route_skip"
    assert query_view_service.embedding_engine.calls == []


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

    async def fake_sync(self, buckets):
        calls.append(buckets)
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
    assert refresh_service._passage_candidate_shadow_sync["requested_scene_ids"] == [
        "scene-edited"
    ]
    assert refresh_service._passage_candidate_shadow_sync["requested_fact_event_ids"] == [
        "event-new"
    ]
    assert refresh_service._passage_candidate_shadow_sync["refresh_source"] == (
        "canonical_mutation_notification"
    )


asyncio.run(verify_weak_trigger_controls_query_view_execution())
asyncio.run(verify_mutation_refresh_queue())

print("PASSAGE_CANDIDATE_SIMULATION_SHADOW_OK")
