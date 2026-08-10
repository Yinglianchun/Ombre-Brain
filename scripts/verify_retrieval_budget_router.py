"""Behavioral simulation checks replayed from the preserved old worktree."""

from __future__ import annotations

import asyncio
import sys
from datetime import timezone
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway import GatewayService
from recall_policy import RecallPolicy
from reranker_engine import RerankResult


def route_debug() -> dict:
    return {
        "route": "present_chitchat",
        "route_action": "skip",
        "confidence": 0.96,
        "margin": 0.22,
        "threshold": 0.72,
    }


def low_confidence_skip_route_debug() -> dict:
    return {
        "enabled": True,
        "active": True,
        "called": True,
        "shadow_only": False,
        "route": "present_chitchat",
        "route_action": "skip",
        "confidence": 0.51,
        "margin": 0.01,
        "threshold": 0.72,
        "skip_applied": False,
        "reason": "below_threshold",
    }


class BucketManager:
    @staticmethod
    def filter_specific_lexical_terms(terms, _buckets, **_kwargs):
        return list(terms)

    @staticmethod
    def _calc_time_score(_meta):
        return 0.5


class StateStore:
    @staticmethod
    def get_recent_bucket_ids(_session_id, _rounds):
        return set()


class ShadowReranker:
    enabled = False
    simulation_shadow_enabled = True
    shadow_ready = True
    model = "Qwen/Qwen3-Reranker-4B"
    candidate_limit = 30

    def __init__(self, score: float) -> None:
        self.score = score
        self.calls = 0

    async def rerank_shadow(self, _query, documents, top_n=None):
        self.calls += 1
        assert documents
        assert top_n == len(documents)
        return [RerankResult(index=0, score=self.score)]


class EmptyEvidenceStore:
    @staticmethod
    def list_for_scene(_scene_id):
        return []


def service(semantic_scores: dict[str, float]) -> GatewayService:
    instance = GatewayService.__new__(GatewayService)
    instance.recall_policy = RecallPolicy()
    instance.config = {
        "recall_thresholds": {
            "vector_min_score": 0.50,
            "skip_route_vector_min_score": 0.60,
        }
    }
    instance.gateway_tz = timezone.utc
    instance.first_card_min_score = 0.55
    instance.retrieval_budget_sentinel_rescue_floor = 0.55
    instance.inject_max_cards = 2
    instance.dynamic_top_k = 8
    instance.semantic_candidate_top_k = 30
    instance.skip_recent_rounds = 4
    instance.recall_fusion_mode = "dynamic"
    instance.bucket_mgr = BucketManager()
    instance.state_store = StateStore()
    instance._add_timing_ms = lambda *_args, **_kwargs: None
    instance._query_has_relevance_facet = lambda _query: False
    instance._is_dynamic_candidate = lambda _bucket: True
    instance._is_identity_name_candidate_bucket = lambda _query, _bucket: False
    instance._is_relevance_suppressed = lambda _query, _bucket: False
    instance._is_relevance_candidate_bucket = lambda _query, _bucket: False
    instance._is_semantic_candidate_bucket = lambda _bucket: True
    instance._retrieval_alias_hits = lambda _query, _ids: []
    instance._normalized_recall_query = lambda query: query
    instance._get_keyword_candidates = lambda _query, _buckets: {}

    async def semantic(_query, eligible_ids, **_kwargs):
        return {key: score for key, score in semantic_scores.items() if key in eligible_ids}

    instance._get_semantic_candidates = semantic
    instance._query_looks_emotional_reason_lookup = lambda _query: False
    instance._get_exact_anchor_candidates = lambda *_args, **_kwargs: ({}, {})
    instance._bucket_authored_cue_terms = lambda _query, _bucket: []
    instance._recall_query_plan = lambda query: SimpleNamespace(
        activated_axis_multi=False,
        locatable_terms=(
            ["巨型水果"] if "巨型水果" in query else
            ["水果"] if "水果" in query else
            ["西瓜"] if "西瓜" in query else []
        ),
        specific_terms=(
            ["巨型水果"] if "巨型水果" in query else
            ["水果"] if "水果" in query else
            ["西瓜"] if "西瓜" in query else []
        ),
    )
    instance._planner_lexical_match_terms = lambda terms: list(terms or [])
    instance._query_anchor_terms_for_diversity = lambda _query: []
    instance._bucket_matched_query_terms = lambda _bucket, _terms: []
    instance._bucket_relevance_multiplier = lambda _query, _bucket: 1.0
    instance._dynamic_alpha_debug = lambda _scores: {"alpha": 0.35, "confidence": "test"}
    instance._is_canonical_scene_bucket = lambda _bucket: True
    instance._canonical_scene_semantic_route_guard = (
        GatewayService._canonical_scene_semantic_route_guard.__get__(instance, GatewayService)
    )
    instance._canonical_scene_semantic_threshold = lambda _bucket: 0.50
    instance._bucket_primary_candidate_rank = lambda _query, item: (-item["score"], item["bucket"]["id"])
    instance._bucket_final_candidate_rank = lambda _query, item, **_kwargs: (-item["score"], item["bucket"]["id"])
    def admit(_query, item):
        if item.get("exact_anchor_match") or item.get("authored_cue_match"):
            return True
        semantic_score = float(item.get("semantic_score") or 0.0)
        route_guard = item.get("semantic_route_guard") or {}
        if route_guard.get("deferred_to_reranker_shadow"):
            item["admission_reason"] = "candidate_only_requires_reranker"
            return False
        if 0.50 <= semantic_score < 0.55:
            item["admission_reason"] = "candidate_only_requires_reranker"
            return False
        if semantic_score >= float(item.get("canonical_scene_semantic_threshold") or 0.50):
            return True
        item["admission_reason"] = (
            "candidate_only_requires_reranker"
            if item.get("authored_cue_candidate_match")
            else "scene_without_authored_or_semantic_evidence"
            if item.get("cue_semantic_candidate_match")
            else "scene_below_semantic_route_threshold"
        )
        return False

    instance._admit_bucket_for_recall = admit
    async def forbidden_reranker(*_args, **_kwargs):
        raise AssertionError("simulation shadow must not call the production reranker")
    instance._rerank_scored_bucket_candidates = forbidden_reranker
    return instance


watermelon = {
    "id": "scene-watermelon",
    "name": "巨型西瓜",
    "metadata": {
        "name": "巨型西瓜",
        "importance": 5,
        "memory_value_source": "authored_scene",
        "object_kind": "scene",
        "scene_cues": ["巨型西瓜还没吃完"],
    },
}

mixed_service = service({"scene-watermelon": 0.72})
mixed_budget = mixed_service._build_retrieval_budget_debug(
    "刚吃完饭，昨天那个巨型西瓜还没吃完",
    route_debug(),
)
assert mixed_budget["effective_budget"] == "normal"
assert mixed_budget["anchor_override"] is True
mixed_debug = {"retrieval_budget": mixed_budget}
mixed_admitted, mixed_suppressed = asyncio.run(
    mixed_service._dynamic_bucket_candidate_items(
        "刚吃完饭，昨天那个巨型西瓜还没吃完",
        "simulation",
        [watermelon],
        semantic_recall_debug=mixed_debug,
        allow_semantic_session_dedupe=False,
    )
)
assert [item["bucket"]["id"] for item in mixed_admitted] == ["scene-watermelon"]
assert mixed_suppressed == []

address_service = service({"scene-watermelon": 0.72})
address_budget = address_service._build_retrieval_budget_debug("哥哥", route_debug())
address_admitted, address_suppressed = asyncio.run(
    address_service._dynamic_bucket_candidate_items(
        "哥哥",
        "simulation-address-only",
        [watermelon],
        semantic_recall_debug={"retrieval_budget": address_budget},
        allow_semantic_session_dedupe=False,
    )
)
assert address_admitted == []
assert address_suppressed[0]["admission_reason"] == "surface_only_query"
assert address_suppressed[0]["recall_policy_debug"]["surface_only_kind"] == "address_only"

topic_service = service({"scene-watermelon": 0.72})
topic_service.episode_verifier_shadow_enabled = True
topic_service.episode_verifier_model = "deepseek-v4-flash"
topic_service.episode_verifier_timeout_seconds = 2
topic_service.episode_verifier_max_candidates = 1
topic_service.episode_verifier_min_confidence = 0.70

async def topic_verdict(_payload):
    return (
        '{"decisions":[{"candidate_id":"scene-watermelon",'
        '"verdict":"same_topic_only","confidence":0.95,'
        '"current_evidence_span":"","grounded_cue":"",'
        '"reason":"current activity only"}]}',
        "",
    )

topic_service._episode_verifier_completion = topic_verdict
topic_budget = topic_service._build_retrieval_budget_debug(
    "我在看水果", route_debug()
)
topic_admitted, topic_suppressed = asyncio.run(
    topic_service._dynamic_bucket_candidate_items(
        "我在看水果",
        "simulation-same-topic-only",
        [watermelon],
        semantic_recall_debug={"retrieval_budget": topic_budget},
        allow_semantic_session_dedupe=False,
    )
)
assert topic_admitted == []
assert topic_suppressed[0]["admission_reason"] == "episode_verifier_rejected"
assert topic_suppressed[0]["episode_verifier_shadow"]["verdict"] == "same_topic_only"


class CueSemanticIndex:
    def __init__(self):
        self.calls = 0

    def search_by_vector(self, vector, *, current_cue_hashes, top_k):
        self.calls += 1
        assert vector == [0.1, 0.2]
        assert set(current_cue_hashes) == {"scene-watermelon"}
        assert top_k == 8
        return {
            "status": "available",
            "dataset_version": 1,
            "profile": {"model": "Qwen/Qwen3-Embedding-4B"},
            "stale_scene_count": 0,
            "matches": [
                {
                    "scene_id": "scene-watermelon",
                    "score": 0.68,
                    "matched_cues": ["巨型西瓜还没吃完"],
                }
            ],
        }


cue_semantic_service = service({"scene-watermelon": 0.20})
cue_semantic_service.cue_semantic_index = CueSemanticIndex()
cue_semantic_budget = cue_semantic_service._build_retrieval_budget_debug(
    "刚吃完饭，昨天那个巨型西瓜还没吃完",
    route_debug(),
)
assert cue_semantic_budget["effective_budget"] == "normal"
cue_semantic_admitted, cue_semantic_suppressed = asyncio.run(
    cue_semantic_service._dynamic_bucket_candidate_items(
        "刚吃完饭，昨天那个巨型西瓜还没吃完",
        "simulation",
        [watermelon],
        query_embedding=[0.1, 0.2],
        semantic_recall_debug={"retrieval_budget": cue_semantic_budget},
        allow_semantic_session_dedupe=False,
    )
)
assert cue_semantic_admitted == []
assert cue_semantic_suppressed[0]["admission_reason"] == "pending_reranker_shadow_candidate_only"
assert cue_semantic_suppressed[0]["cue_semantic_candidate_match"] is True
assert cue_semantic_suppressed[0]["cue_semantic_score"] == 0.68
assert cue_semantic_suppressed[0]["authored_cue_match"] is False
assert cue_semantic_budget["cue_semantic"]["status"] == "available"
assert cue_semantic_budget["cue_semantic"]["candidate_count"] == 1
assert cue_semantic_budget["rerank"]["would_call"] is True

date_service = service({"scene-watermelon": 0.32})
date_budget = date_service._build_retrieval_budget_debug("昨天好热", route_debug())
assert date_budget["effective_budget"] == "shallow"
date_admitted, date_suppressed = asyncio.run(
    date_service._dynamic_bucket_candidate_items(
        "昨天好热",
        "simulation",
        [watermelon],
        semantic_recall_debug={"retrieval_budget": date_budget},
        allow_semantic_session_dedupe=False,
    )
)
assert date_admitted == []
assert date_suppressed[0]["admission_reason"] == "below_absolute_floor"

no_target_service = service({})
no_target_budget = no_target_service._build_retrieval_budget_debug("昨天好热", route_debug())
no_target_admitted, _ = asyncio.run(
    no_target_service._dynamic_bucket_candidate_items(
        "昨天好热",
        "simulation",
        [watermelon],
        semantic_recall_debug={"retrieval_budget": no_target_budget},
        allow_semantic_session_dedupe=False,
    )
)
assert no_target_admitted == []
assert no_target_budget["cheap_retrieval"]["stop_reason"] == "no_cheap_candidates"

cue_service = service({"scene-watermelon": 0.20})
cue_service._bucket_authored_cue_terms = lambda query, bucket: (
    ["给予别人善意也是在善待自己"]
    if query == "给予别人善意也是在善待自己" and bucket.get("id") == "scene-watermelon"
    else []
)
cue_budget = cue_service._build_retrieval_budget_debug(
    "给予别人善意也是在善待自己",
    route_debug(),
)
assert cue_budget["effective_budget"] == "shallow"
cue_admitted, cue_suppressed = asyncio.run(
    cue_service._dynamic_bucket_candidate_items(
        "给予别人善意也是在善待自己",
        "simulation",
        [watermelon],
        semantic_recall_debug={"retrieval_budget": cue_budget},
        allow_semantic_session_dedupe=False,
    )
)
assert cue_admitted == []
assert len(cue_suppressed) == 1
assert cue_suppressed[0]["admission_reason"] == "candidate_only_requires_reranker"
assert cue_suppressed[0]["authored_cue_match"] is False
assert cue_suppressed[0]["authored_cue_candidate_match"] is True
cue_service._finalize_retrieval_budget_candidate_debug(
    cue_budget,
    cue_admitted,
    cue_suppressed,
)
cue_debug = cue_budget["cheap_retrieval"]["candidates"][0]
assert cue_debug["cue_lexical_match"] is True
assert cue_debug["cue_lexical_role"] == "candidate_only"
assert "matched_cues" not in cue_debug
assert cue_debug["final_admission_source"] == "candidate_only_requires_reranker"
assert cue_debug["reranker_shadow"]["status"] == "simulation_shadow_disabled"
assert cue_debug["reranker_shadow"]["called_false_reason"] == "simulation_shadow_disabled"

strong_body_cue_service = service({"scene-watermelon": 0.72})
strong_body_cue_service._bucket_authored_cue_terms = cue_service._bucket_authored_cue_terms
strong_body_cue_budget = strong_body_cue_service._build_retrieval_budget_debug(
    "给予别人善意也是在善待自己",
    route_debug(),
)
strong_body_admitted, strong_body_suppressed = asyncio.run(
    strong_body_cue_service._dynamic_bucket_candidate_items(
        "给予别人善意也是在善待自己",
        "simulation",
        [watermelon],
        semantic_recall_debug={"retrieval_budget": strong_body_cue_budget},
        allow_semantic_session_dedupe=False,
    )
)
assert [item["bucket"]["id"] for item in strong_body_admitted] == ["scene-watermelon"]
assert strong_body_suppressed == []
assert strong_body_admitted[0]["authored_cue_match"] is False
assert strong_body_admitted[0]["authored_cue_candidate_match"] is True

without_cues_budget = cue_service._build_retrieval_budget_debug(
    "给予别人善意也是在善待自己",
    route_debug(),
)
without_cues_budget["recall_ablation"] = {"mode": "without_cues"}
without_cues_admitted, _ = asyncio.run(
    cue_service._dynamic_bucket_candidate_items(
        "给予别人善意也是在善待自己",
        "simulation",
        [watermelon],
        semantic_recall_debug={"retrieval_budget": without_cues_budget},
        allow_semantic_session_dedupe=False,
    )
)
assert without_cues_admitted == []
assert without_cues_budget["cheap_retrieval"]["stop_reason"] == "no_candidate_over_absolute_floor"


class ForbiddenCueSemanticIndex:
    def search_by_vector(self, *_args, **_kwargs):
        raise AssertionError("ordinary Hook retrieval must not query the cue shadow index")


ordinary_service = service({"scene-watermelon": 0.72})
ordinary_service.cue_semantic_index = ForbiddenCueSemanticIndex()
ordinary_admitted, ordinary_suppressed = asyncio.run(
    ordinary_service._dynamic_bucket_candidate_items(
        "巨型西瓜",
        "ordinary-hook",
        [watermelon],
        query_embedding=[0.1, 0.2],
        semantic_recall_debug={},
        allow_semantic_session_dedupe=False,
        allow_rerank=False,
    )
)
assert [item["bucket"]["id"] for item in ordinary_admitted] == ["scene-watermelon"]
assert ordinary_suppressed == []
assert "cue_semantic_candidate_match" not in ordinary_admitted[0]

# Named regressions from the live shadow simulator.  The weaker paraphrase is
# eligible for reranking only because the query has a reference plus concrete
# entity facet; it must not become a directly admitted Scene.
fruit_query = "那天餐桌上的水果"
fruit_service = service({"scene-watermelon": 0.515})
fruit_shadow = ShadowReranker(0.82)
fruit_service.reranker_engine = fruit_shadow
fruit_service.scene_evidence_store = EmptyEvidenceStore()
fruit_budget = fruit_service._build_retrieval_budget_debug(
    fruit_query,
    low_confidence_skip_route_debug(),
)
assert fruit_budget["effective_budget"] == "normal"
assert fruit_budget["anchor_override"] is True
fruit_admitted, fruit_suppressed = asyncio.run(
    fruit_service._dynamic_bucket_candidate_items(
        fruit_query,
        "simulation-fruit",
        [watermelon],
        semantic_recall_debug={
            **low_confidence_skip_route_debug(),
            "retrieval_budget": fruit_budget,
        },
        allow_semantic_session_dedupe=False,
    )
)
assert fruit_admitted == [], "gray-zone candidates must never bypass reranking"
assert len(fruit_suppressed) == 1
assert [item["bucket"]["id"] for item in fruit_admitted] == [], "shadow high score must not promote a candidate"
assert fruit_suppressed[0]["admission_reason"] == "pending_reranker_shadow_gray_zone"
assert fruit_suppressed[0]["recall_policy_debug"]["reranker_required"] is True
assert fruit_suppressed[0]["budget_gray_zone_qualified"] is True
assert fruit_suppressed[0]["reranker_shadow_score"] == 0.82
assert fruit_budget["rerank"]["would_call"] is True
assert fruit_budget["rerank"]["called"] is True
assert fruit_budget["rerank"]["decision_applied"] is False
assert fruit_shadow.calls == 1

# At 0.555 the current absolute floor is already satisfied.  In simulation,
# the low-confidence skip-route guard is diagnostic only and must defer the
# final decision to the future reranker shadow.
giant_fruit_query = "那天餐桌上的巨型水果"
giant_fruit_service = service({"scene-watermelon": 0.555})
giant_fruit_shadow = ShadowReranker(0.74)
giant_fruit_service.reranker_engine = giant_fruit_shadow
giant_fruit_service.scene_evidence_store = EmptyEvidenceStore()
giant_fruit_budget = giant_fruit_service._build_retrieval_budget_debug(
    giant_fruit_query,
    low_confidence_skip_route_debug(),
)
assert giant_fruit_budget["effective_budget"] == "normal"
giant_fruit_admitted, giant_fruit_suppressed = asyncio.run(
    giant_fruit_service._dynamic_bucket_candidate_items(
        giant_fruit_query,
        "simulation-giant-fruit",
        [watermelon],
        semantic_recall_debug={
            **low_confidence_skip_route_debug(),
            "retrieval_budget": giant_fruit_budget,
        },
        allow_semantic_session_dedupe=False,
    )
)
assert giant_fruit_admitted == []
assert len(giant_fruit_suppressed) == 1
assert [item["bucket"]["id"] for item in giant_fruit_admitted] == [], "shadow high score must not inject a candidate"
assert giant_fruit_suppressed[0]["admission_reason"] == "pending_reranker_shadow_route_guard"
assert giant_fruit_suppressed[0]["recall_policy_debug"]["reranker_required"] is True
assert giant_fruit_suppressed[0]["recall_policy_debug"]["normal_admission_reason"] == (
    "scene_below_semantic_route_threshold"
)
assert giant_fruit_suppressed[0]["reranker_shadow_score"] == 0.74
assert giant_fruit_budget["rerank"]["would_call"] is True
assert giant_fruit_budget["rerank"]["called"] is True
assert giant_fruit_budget["rerank"]["decision_applied"] is False
assert giant_fruit_shadow.calls == 1

# The same 0.555 candidate on the ordinary, non-simulation path keeps the
# existing 0.60 low-confidence route guard and remains rejected.
ordinary_guard_service = service({"scene-watermelon": 0.555})
ordinary_shadow = ShadowReranker(0.99)
ordinary_guard_service.reranker_engine = ordinary_shadow
ordinary_guard_service.scene_evidence_store = EmptyEvidenceStore()
ordinary_guard_admitted, ordinary_guard_suppressed = asyncio.run(
    ordinary_guard_service._dynamic_bucket_candidate_items(
        giant_fruit_query,
        "ordinary-hook-route-guard",
        [watermelon],
        semantic_recall_debug=low_confidence_skip_route_debug(),
        allow_semantic_session_dedupe=False,
        allow_rerank=False,
    )
)
assert ordinary_guard_admitted == []
assert len(ordinary_guard_suppressed) == 1
assert ordinary_guard_suppressed[0]["admission_reason"] == "scene_below_semantic_route_threshold"
assert ordinary_guard_suppressed[0]["canonical_scene_semantic_threshold"] == 0.60
assert not ordinary_guard_suppressed[0]["semantic_route_guard"].get(
    "deferred_to_reranker_shadow"
)
assert ordinary_shadow.calls == 0, "ordinary recall must never call simulation shadow"

print("retrieval budget router verification passed")
