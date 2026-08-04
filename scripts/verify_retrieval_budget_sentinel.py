"""Sentinel safety checks: vector reuse, no admission, and fail-open veto order."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway import GatewayService
from recall_policy import RecallPolicy


def route_debug() -> dict:
    return {
        "route": "present_chitchat",
        "route_action": "skip",
        "confidence": 0.96,
        "margin": 0.22,
        "threshold": 0.72,
        "reason": "matched_skip_route",
    }


class Embedding:
    def __init__(self, rows=None, error: Exception | None = None):
        self.rows = rows or []
        self.error = error
        self.calls: list[tuple[list[float], int]] = []

    async def search_similar_by_embedding(self, vector, *, top_k):
        self.calls.append((list(vector), top_k))
        if self.error:
            raise self.error
        return list(self.rows)


def planner_service(rows=None, error=None) -> GatewayService:
    service = GatewayService.__new__(GatewayService)
    service.recall_policy = RecallPolicy()
    service.gateway_tz = timezone.utc
    service.first_card_min_score = 0.55
    service.retrieval_budget_sentinel_rescue_floor = 0.55
    service.embedding_query_timeout_seconds = 0
    service.embedding_engine = Embedding(rows, error)
    service._is_semantic_candidate_bucket = lambda _bucket: True
    service._bucket_title_anchor_terms = lambda _query, _bucket: []
    service._bucket_authored_cue_terms = lambda _query, _bucket: []

    async def buckets(**_kwargs):
        return [{"id": "scene-low"}, {"id": "scene-high"}]

    service._list_gateway_buckets = buckets
    return service


low_service = planner_service([("scene-low", 0.42), ("scene-high", 0.37)])
low_budget = low_service._build_retrieval_budget_debug("刚吃完饭", route_debug())
low = asyncio.run(low_service._retrieval_budget_sentinel_debug(
    "刚吃完饭", [0.1, 0.2], low_budget
))
assert low_service.embedding_engine.calls == [([0.1, 0.2], 2)]
assert low["called"] is True
assert low["floor_qualified_count"] == 0
assert low["expanded"] is False
assert low["reranked"] is False
assert low["injection_allowed"] is False
assert low["recorded"] is False
assert all(row["would_inject"] is False for row in low["candidates"])

high_service = planner_service([("scene-high", 0.63)])
high_budget = high_service._build_retrieval_budget_debug("刚吃完饭", route_debug())
high = asyncio.run(high_service._retrieval_budget_sentinel_debug(
    "刚吃完饭", [0.1, 0.2], high_budget
))
assert high["floor_qualified_count"] == 1
assert high["reason"] == "candidate_over_rescue_floor"

failure_service = planner_service(error=RuntimeError("offline"))
failure_budget = failure_service._build_retrieval_budget_debug("晚安", route_debug())
failure = asyncio.run(failure_service._retrieval_budget_sentinel_debug(
    "晚安", [0.1, 0.2], failure_budget
))
assert failure["called"] is False
assert failure["reason"] == "body_sentinel_failed:RuntimeError"


class Request:
    headers = {}

    async def json(self):
        return {"query": "刚吃完饭", "simulation": True}


class Router:
    async def route_with_vector(self, _query):
        return route_debug(), [0.1, 0.2]

    @staticmethod
    def should_apply_skip(_debug):
        return True


veto_service = planner_service([("scene-low", 0.42)])
veto_service.semantic_recall_router = Router()
veto_service.upstream_default_model = "test"
veto_service.upstream_models = []
veto_service._authorize = lambda _header: None

async def veto(_query, _skip, _vector, debug):
    debug["scene_evidence_veto"] = {"applied": True, "reason": "trusted_scene_evidence"}
    return False

async def fast(*_args, **_kwargs):
    return [], [], {"hook_recall_debug": {"candidate_count": 0}}

veto_service._apply_semantic_scene_evidence_veto = veto
veto_service._hook_recall_fast_cards = fast
veto_service._render_hook_recall_additional_context = lambda _cards: ""
response = asyncio.run(veto_service.handle_hook_recall(Request()))
data = json.loads(response.body.decode("utf-8"))
budget = data["debug"]["semantic_recall_debug"]["retrieval_budget"]
assert budget["skip_ready"] is True
assert budget["budget_skip_applied"] is False
assert budget["route_skip_deferred"] is True
assert budget["deferred_reason"] == "scene_evidence_veto"
assert data["debug"]["semantic_recall_debug"]["skip_applied"] is False

print("retrieval budget sentinel verification passed")
