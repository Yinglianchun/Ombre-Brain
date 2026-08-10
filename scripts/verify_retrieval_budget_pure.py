"""Pure-function checks for retrieval-budget planning.

This suite deliberately imports no Gateway code.  It is the first transplant
gate for the byte-identical planner module.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_recall.retrieval_budget import (
    BUDGET_DEEP,
    BUDGET_NORMAL,
    BUDGET_SHALLOW,
    BUDGET_SKIP,
    apply_fact_event_probe,
    build_retrieval_budget,
    finalize_retrieval_budget,
    partition_candidates_by_absolute_floor,
)


def pure_route_debug() -> dict:
    return {
        "route": "present_chitchat",
        "route_action": "skip",
        "confidence": 0.96,
        "margin": 0.22,
        "threshold": 0.72,
    }


def build(query: str, **kwargs) -> dict:
    return build_retrieval_budget(
        query,
        route="present_chitchat",
        route_action="skip",
        semantic_debug=pure_route_debug(),
        **kwargs,
    )


pure_surface = build("刚吃完饭")
assert pure_surface["pure_chitchat_prior"] is True
assert pure_surface["pure_surface_chitchat"] is False
assert pure_surface["effective_budget"] == BUDGET_SHALLOW
assert pure_surface["initial_budget"] == BUDGET_SHALLOW
assert pure_surface["final_budget"] == BUDGET_SHALLOW
assert pure_surface["sentinel"]["called"] is False

pure_skip = finalize_retrieval_budget(
    pure_surface,
    {
        "called": True,
        "candidate_count": 2,
        "floor_qualified_count": 0,
        "candidates": [
            {"bucket_id": "near-chat-1", "rescue_score": 0.31, "would_inject": False},
            {"bucket_id": "near-chat-2", "rescue_score": 0.42, "would_inject": False},
        ],
    },
)
assert pure_skip["skip_ready"] is True
assert pure_skip["effective_budget"] == BUDGET_SKIP
assert pure_skip["final_budget"] == BUDGET_SKIP
assert pure_skip["channels"] == []
assert pure_skip["semantic_top_k"] == 0

rescued = finalize_retrieval_budget(
    build("换个说法也行"),
    {
        "called": True,
        "candidate_count": 1,
        "floor_qualified_count": 1,
        "candidates": [
            {"bucket_id": "scene-watermelon", "rescue_score": 0.63, "would_inject": False},
        ],
    },
)
assert rescued["skip_ready"] is False
assert rescued["effective_budget"] == BUDGET_NORMAL
assert rescued["sentinel"]["candidates"][0]["would_inject"] is False

unavailable = finalize_retrieval_budget(build("晚安"), {"called": False})
assert unavailable["skip_ready"] is False
assert unavailable["effective_budget"] == BUDGET_SHALLOW
assert unavailable["sentinel"]["reason"] == "sentinel_unavailable_fail_open"

mixed = build(
    "刚吃完饭，昨天那个巨型西瓜还没吃完",
    planner={"locatable_terms": ["西瓜"], "specific_terms": ["西瓜"]},
    date_hint={"reference": "昨天"},
)
mixed_kinds = {(facet["kind"], facet["value"]) for facet in mixed["query_facets"]}
assert ("entity", "西瓜") in mixed_kinds
assert ("reference_entity", "昨天那个 + 西瓜") in mixed_kinds
assert mixed["anchor_override"] is True
assert mixed["effective_budget"] == BUDGET_NORMAL
assert mixed["pure_chitchat_prior"] is False

date_only = build("昨天好热", date_hint={"reference": "昨天"})
assert date_only["date_only"] is True
assert date_only["anchor_override"] is False
assert date_only["effective_budget"] == BUDGET_SHALLOW

quoted_title = build('我还记得“巨型西瓜”吗')
assert quoted_title["anchor_override"] is True
assert quoted_title["effective_budget"] == BUDGET_NORMAL
assert quoted_title["final_budget"] == BUDGET_DEEP

fact_budget = apply_fact_event_probe(
    build("我是不是紫外线过敏"),
    {
        "status": "ok",
        "matches": [
            {
                "memory_id": "fact-uv",
                "memory_kind": "fact",
                "score": 0.76,
                "importance": 3,
                "covered_by_scene_id": "",
            }
        ],
    },
)
assert fact_budget["final_budget"] == BUDGET_SHALLOW
assert fact_budget["escalation_reason"] == "fact_candidate_over_rescue_floor"

event_budget = apply_fact_event_probe(
    build("明信片那件事"),
    {
        "status": "ok",
        "matches": [
            {
                "memory_id": "event-postcard",
                "memory_kind": "event",
                "score": 0.71,
                "importance": 3,
                "covered_by_scene_id": "",
            }
        ],
    },
)
assert event_budget["final_budget"] == BUDGET_DEEP
assert event_budget["escalation_reason"] == "event_candidate_over_rescue_floor"

low_importance_event = apply_fact_event_probe(
    build("我在刷小红书"),
    {
        "status": "ok",
        "matches": [
            {
                "memory_id": "event-xiaohongshu",
                "memory_kind": "event",
                "score": 0.65,
                "importance": 1,
                "covered_by_scene_id": "",
            }
        ],
    },
)
assert low_importance_event["final_budget"] == BUDGET_SHALLOW
assert low_importance_event["typed_qualified_count"] == 0

bare_address = apply_fact_event_probe(
    build("哥哥"),
    {
        "status": "ok",
        "matches": [
            {
                "memory_id": "event-address",
                "memory_kind": "event",
                "score": 0.90,
                "importance": 5,
                "covered_by_scene_id": "",
            }
        ],
    },
)
assert bare_address["surface_only_kind"] == "address_only"
assert bare_address["final_budget"] == BUDGET_SHALLOW

covered_budget = finalize_retrieval_budget(
    build("刚吃完饭"),
    {"called": True, "floor_qualified_count": 0, "candidates": []},
)
assert covered_budget["skip_ready"] is True
apply_fact_event_probe(
    covered_budget,
    {
        "status": "ok",
        "matches": [
            {
                "memory_id": "event-covered",
                "memory_kind": "event",
                "score": 0.70,
                "importance": 3,
                "covered_by_scene_id": "scene-covered",
            }
        ],
    },
)
assert covered_budget["skip_ready"] is False
assert covered_budget["final_budget"] == BUDGET_DEEP
assert covered_budget["escalation_reason"] == "typed_candidate_covered_by_scene"

qualified, suppressed = partition_candidates_by_absolute_floor(
    [
        {"bucket_id": "scene-watermelon", "score": 0.62},
        {"bucket_id": "scene-no-target", "score": 0.41},
        {"bucket_id": "scene-exact", "score": 0.20, "exact_anchor_match": True},
    ],
    absolute_floor=0.55,
)
assert [row["bucket_id"] for row in qualified] == ["scene-watermelon", "scene-exact"]
assert [row["bucket_id"] for row in suppressed] == ["scene-no-target"]
assert suppressed[0]["admission_reason"] == "below_absolute_floor"

print("retrieval budget pure verification passed")
