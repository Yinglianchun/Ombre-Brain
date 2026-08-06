"""Verify explicit-simulation reranker scoring remains debug-only."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway import GatewayService
from reranker_engine import RerankResult, RerankerEngine


class FakeEvidenceStore:
    def list_for_scene(self, scene_id: str) -> list[dict]:
        if scene_id == "scene-watermelon":
            return [{"content": "餐桌上，菜旁边摆着一个巨型西瓜。"}]
        return []


class FakeShadowReranker:
    enabled = False
    simulation_shadow_enabled = True
    shadow_ready = True
    model = "Qwen/Qwen3-Reranker-4B"
    candidate_limit = 30

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], int | None]] = []

    async def rerank_shadow(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        self.calls.append((query, list(documents), top_n))
        return [RerankResult(index=0, score=0.83), RerankResult(index=1, score=0.21)]


def service(engine: object) -> GatewayService:
    instance = GatewayService.__new__(GatewayService)
    instance.reranker_engine = engine
    instance.scene_evidence_store = FakeEvidenceStore()
    instance.recall_fusion_mode = "weighted"
    instance._is_canonical_scene_bucket = lambda bucket: str(
        (bucket.get("metadata") or {}).get("object_kind") or ""
    ) == "scene"
    instance._bucket_rerank_candidate_priority = lambda _query, item: (
        -float(item.get("semantic_score") or 0.0),
    )
    return instance


def candidate(scene_id: str, title: str, body: str, score: float) -> dict:
    return {
        "bucket": {
            "id": scene_id,
            "content": body,
            "metadata": {
                "object_kind": "scene",
                "name": title,
                "scene_cues": ["八月一日的西瓜", "巨型西瓜还没吃完"],
            },
        },
        "score": score,
        "combined_score": score,
        "semantic_score": score,
        "budget_floor": 0.55,
        "budget_reranker_entry_floor": 0.50,
        "budget_floor_qualified": score >= 0.55,
        "budget_gray_zone_qualified": 0.50 <= score < 0.55,
        "budget_reranker_eligible": True,
    }


engine = FakeShadowReranker()
instance = service(engine)
candidates = [
    candidate("scene-watermelon", "菜旁边的巨型西瓜", "我们切开了那个巨型西瓜。", 0.515),
    candidate("scene-unknown", "没有原文的旧 Scene", "旧 Scene 正文。", 0.56),
]
original_scores = [item["score"] for item in candidates]
budget = {
    "recall_ablation": {"mode": "normal"},
    "rerank": {"would_call": True, "called": False},
    "cheap_retrieval": {
        "candidates": [
            instance._retrieval_budget_candidate_debug_row(item) for item in candidates
        ]
    },
}
scored = asyncio.run(
    instance._apply_simulation_reranker_shadow(
        "那天餐桌上的水果",
        candidates,
        budget,
    )
)
assert scored is candidates
assert engine.enabled is False, "production reranker must remain disabled"
assert len(engine.calls) == 1
query, documents, top_n = engine.calls[0]
assert query == "那天餐桌上的水果"
assert top_n == 2
watermelon_document = next(
    document for document in documents if "title: 菜旁边的巨型西瓜" in document
)
unknown_document = next(
    document for document in documents if "title: 没有原文的旧 Scene" in document
)
assert "body: 我们切开了那个巨型西瓜。" in watermelon_document
assert "cues: 八月一日的西瓜 | 巨型西瓜还没吃完" in watermelon_document
assert "source_evidence_status: bound" in watermelon_document
assert "餐桌上，菜旁边摆着一个巨型西瓜。" in watermelon_document
assert "source_evidence_status: unknown" in unknown_document
assert [item["score"] for item in candidates] == original_scores
assert [item["combined_score"] for item in candidates] == original_scores
assert all("rerank_score" not in item for item in candidates)
assert [item["reranker_shadow_score"] for item in candidates] == [0.21, 0.83]
assert budget["rerank"]["called"] is True
assert budget["rerank"]["score_count"] == 2
assert budget["rerank"]["decision_applied"] is False
assert budget["rerank"]["telemetry_generated_at"]
rows = budget["cheap_retrieval"]["candidates"]
assert rows[0]["reranker_shadow"]["score"] == 0.21
assert rows[0]["reranker_shadow"]["evidence_status"] == "bound"
assert rows[0]["reranker_shadow"]["decision_applied"] is False
assert rows[0]["reranker_shadow"]["telemetry_generated_at"] == budget["rerank"]["telemetry_generated_at"]
assert rows[1]["reranker_shadow"]["score"] == 0.83
assert rows[1]["reranker_shadow"]["evidence_status"] == "unknown"

# Cue ablation removes cue text from the reranker document as well as candidate
# discovery; body and bound source evidence remain available.
without_cues_engine = FakeShadowReranker()
without_cues_service = service(without_cues_engine)
without_cues_candidate = candidate(
    "scene-watermelon",
    "菜旁边的巨型西瓜",
    "我们切开了那个巨型西瓜。",
    0.56,
)
without_cues_budget = {
    "recall_ablation": {"mode": "without_cues"},
    "rerank": {"would_call": True},
    "cheap_retrieval": {
        "candidates": [
            without_cues_service._retrieval_budget_candidate_debug_row(
                without_cues_candidate
            )
        ]
    },
}
asyncio.run(
    without_cues_service._apply_simulation_reranker_shadow(
        "那天餐桌上的水果",
        [without_cues_candidate],
        without_cues_budget,
    )
)
without_cues_document = without_cues_engine.calls[0][1][0]
assert "cues: \n" in without_cues_document
assert "八月一日的西瓜" not in without_cues_document
assert "body: 我们切开了那个巨型西瓜。" in without_cues_document

# A disabled shadow gate is a clean no-op even when production reranking has
# credentials.  Conversely, the config can make shadow ready while keeping the
# production enabled flag false.
disabled_engine = FakeShadowReranker()
disabled_engine.simulation_shadow_enabled = False
disabled_engine.shadow_ready = False
disabled_service = service(disabled_engine)
disabled_candidate = candidate("scene-watermelon", "西瓜", "正文", 0.56)
disabled_budget = {
    "rerank": {"would_call": True},
    "cheap_retrieval": {
        "candidates": [disabled_service._retrieval_budget_candidate_debug_row(disabled_candidate)]
    },
}
asyncio.run(
    disabled_service._apply_simulation_reranker_shadow(
        "西瓜",
        [disabled_candidate],
        disabled_budget,
    )
)
assert disabled_engine.calls == []
assert disabled_budget["rerank"]["reason"] == "simulation_shadow_disabled"
assert disabled_budget["rerank"]["called_false_reason"] == "simulation_shadow_disabled"
assert disabled_budget["cheap_retrieval"]["candidates"][0]["reranker_shadow"]["called"] is False
assert disabled_budget["cheap_retrieval"]["candidates"][0]["reranker_shadow"]["called_false_reason"] == "simulation_shadow_disabled"
assert "reranker_shadow_score" not in disabled_candidate

engine_config = {
    "embedding": {"base_url": "https://example.invalid", "api_key": "test-key"},
    "reranker": {"enabled": False, "simulation_shadow_enabled": True},
}
configured_engine = RerankerEngine(engine_config)
assert configured_engine.enabled is False
assert configured_engine.shadow_ready is True

print("retrieval budget reranker shadow verification passed")
