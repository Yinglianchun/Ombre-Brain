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
    BUDGET_NORMAL,
    BUDGET_SHALLOW,
    BUDGET_SKIP,
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
