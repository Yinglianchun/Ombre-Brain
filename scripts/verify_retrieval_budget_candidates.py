"""Candidate-budget checks independent of live bucket data."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway import GatewayService
from memory_recall.retrieval_budget import build_retrieval_budget


class Embedding:
    enabled = True

    def __init__(self):
        self.top_ks: list[int] = []

    async def search_similar_by_embedding(self, _vector, *, top_k):
        self.top_ks.append(top_k)
        return [("scene-high", 0.71), ("scene-low", 0.40)]


service = GatewayService.__new__(GatewayService)
service.embedding_engine = Embedding()
service.embedding_query_timeout_seconds = 0
service.semantic_candidate_top_k = 30
service.first_card_min_score = 0.55
service._is_canonical_scene_bucket = lambda _bucket: False

budget = build_retrieval_budget(
    "昨天好热",
    route="present_chitchat",
    route_action="skip",
    semantic_debug={
        "route": "present_chitchat",
        "route_action": "skip",
        "confidence": 0.70,
        "margin": 0.02,
        "threshold": 0.72,
    },
    date_hint={"reference": "昨天"},
)
assert budget["effective_budget"] == "shallow"
assert budget["channels"] == ["exact_anchor", "lexical", "body_semantic"]
debug = {"retrieval_budget": budget}

scores = asyncio.run(service._get_semantic_candidates(
    "昨天好热",
    {"scene-high", "scene-low"},
    query_embedding=[0.1, 0.2],
    semantic_recall_debug=debug,
))
assert service.embedding_engine.top_ks == [3]
assert scores == {"scene-high": 0.71, "scene-low": 0.4}

qualified, suppressed = service._apply_retrieval_budget_candidate_floor(
    [
        {"bucket": {"id": "scene-high"}, "score": 0.71, "semantic_score": 0.71},
        {"bucket": {"id": "scene-low"}, "score": 0.40, "semantic_score": 0.40},
        {"bucket": {"id": "scene-exact"}, "score": 0.20, "exact_anchor_match": True},
    ],
    budget,
    candidate_count=3,
)
assert [item["bucket"]["id"] for item in qualified] == ["scene-high", "scene-exact"]
assert [item["bucket"]["id"] for item in suppressed] == ["scene-low"]
assert suppressed[0]["admission_reason"] == "below_absolute_floor"
assert budget["cheap_retrieval"]["candidate_count"] == 3
assert budget["cheap_retrieval"]["floor_qualified_count"] == 2
assert budget["cheap_retrieval"]["stop_reason"] == "candidates_over_absolute_floor"
candidate_debug = {
    item["bucket_id"]: item
    for item in budget["cheap_retrieval"]["candidates"]
}
assert candidate_debug["scene-high"]["semantic_profile"] == "legacy_or_unknown"
assert candidate_debug["scene-high"]["candidate_sources"] == ["body_semantic"]
assert candidate_debug["scene-high"]["cue_semantic"] == {"status": "unavailable", "score": None}
assert candidate_debug["scene-high"]["reranker_shadow"]["status"] == "eligible_not_called"
assert candidate_debug["scene-low"]["final_admission_source"] == "below_absolute_floor"
assert budget["rerank"]["would_call"] is True
assert budget["rerank"]["called"] is False

empty_budget = build_retrieval_budget("普通问题")
empty_qualified, empty_suppressed = service._apply_retrieval_budget_candidate_floor(
    [{"bucket": {"id": "scene-low"}, "score": 0.20}],
    empty_budget,
    candidate_count=1,
)
assert empty_qualified == []
assert len(empty_suppressed) == 1
assert empty_budget["cheap_retrieval"]["stop_reason"] == "no_candidate_over_absolute_floor"
assert empty_budget["rerank"]["would_call"] is False

print("retrieval budget candidate verification passed")
