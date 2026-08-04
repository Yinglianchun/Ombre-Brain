"""Verify Recall Simulation ablation is bounded and candidate-only."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway import (
    GatewayService,
    normalize_recall_ablation_mode,
    recall_ablation_debug_payload,
)
from memory_recall.retrieval_budget import build_retrieval_budget


assert normalize_recall_ablation_mode(None) == "normal"
assert normalize_recall_ablation_mode("without_cues") == "without_cues"
assert normalize_recall_ablation_mode("without_embedding") == "without_embedding"
try:
    normalize_recall_ablation_mode("other")
except ValueError:
    pass
else:
    raise AssertionError("invalid ablation must fail closed")

without_cues = recall_ablation_debug_payload("without_cues", source="manual_simulation")
assert without_cues["authored_cues_enabled"] is False
assert without_cues["body_embedding_enabled"] is True
assert without_cues["route_embedding_enabled"] is True
assert without_cues["route_decision_unchanged"] is True
assert without_cues["evidence_veto_unchanged"] is True


class Request:
    headers = {}

    def __init__(self, body):
        self.body = body

    async def json(self):
        return dict(self.body)


service = GatewayService.__new__(GatewayService)
service._authorize = lambda _header: None
service.upstream_default_model = "test"
service.upstream_models = []

not_simulation = asyncio.run(service.handle_hook_recall(Request({
    "query": "测试",
    "recall_mode": "full",
    "recall_ablation": "without_cues",
})))
assert json.loads(not_simulation.body.decode("utf-8"))["error"] == (
    "recall_ablation_requires_simulation_debug_full"
)

not_full = asyncio.run(service.handle_hook_recall(Request({
    "query": "测试",
    "simulation": True,
    "recall_mode": "fast",
    "recall_ablation": "without_embedding",
})))
assert json.loads(not_full.body.decode("utf-8"))["error"] == (
    "recall_ablation_requires_simulation_debug_full"
)


class Embedding:
    enabled = True

    def __init__(self):
        self.called = False

    async def search_similar_by_embedding(self, _vector, *, top_k):
        self.called = True
        return [("scene", 0.9)]


candidate_service = GatewayService.__new__(GatewayService)
candidate_service.embedding_engine = Embedding()
candidate_service.embedding_query_timeout_seconds = 0
candidate_service.semantic_candidate_top_k = 30
budget = build_retrieval_budget("测试")
budget["recall_ablation"] = recall_ablation_debug_payload("without_embedding")
scores = asyncio.run(candidate_service._get_semantic_candidates(
    "测试",
    {"scene"},
    query_embedding=[0.1, 0.2],
    semantic_recall_debug={"retrieval_budget": budget},
))
assert scores == {}
assert candidate_service.embedding_engine.called is False

print("recall ablation contract verification passed")
