from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway import GatewayService
from memory_recall.retrieval_budget import (
    BUDGET_DEEP,
    BUDGET_SHALLOW,
    BUDGET_SKIP,
    apply_fact_event_probe,
    build_retrieval_budget,
    finalize_retrieval_budget,
)


def route_debug() -> dict:
    return {
        "route": "present_chitchat",
        "route_action": "skip",
        "confidence": 0.92,
        "margin": 0.18,
        "threshold": 0.72,
    }


historical = build_retrieval_budget(
    "对了老公，我昨晚还去cloudflare抢注了一张卡！Haven被占了所以是RainsHaven",
    route="present_chitchat",
    route_action="skip",
    semantic_debug=route_debug(),
    planner={
        "locatable_terms": ["Cloudflare", "RainsHaven"],
        "specific_terms": ["Cloudflare", "RainsHaven"],
    },
    date_hint={"reference": "昨晚"},
)
assert historical["initial_budget"] == BUDGET_SHALLOW
assert historical["final_budget"] == BUDGET_SHALLOW
assert historical["anchor_override"] is True
assert historical["pure_chitchat_prior"] is False

future = dict(historical)
future["cheap_retrieval"] = {"candidates": []}
GatewayService._finalize_retrieval_budget_candidate_debug(
    future,
    [
        {
            "id": "scene-rainshaven-cloudflare",
            "metadata": {
                "memory_value_source": "authored_scene",
                "object_kind": "scene",
            },
        }
    ],
    [],
)
assert future["final_budget"] == BUDGET_DEEP
assert future["escalation_reason"] == "scene_selected_after_admission"

fact = apply_fact_event_probe(
    build_retrieval_budget("我是不是紫外线过敏"),
    {
        "status": "ok",
        "matches": [
            {
                "memory_id": "fact-uv",
                "memory_kind": "fact",
                "score": 0.74,
                "importance": 3,
                "covered_by_scene_id": "",
            }
        ],
    },
)
assert fact["final_budget"] == BUDGET_SHALLOW

explicit = build_retrieval_budget(
    "还记得明信片那次吗",
    route="recall_needed",
    route_action="recall",
    semantic_debug={"route": "recall_needed", "route_action": "recall"},
)
assert explicit["final_budget"] == BUDGET_DEEP

plain_prior = build_retrieval_budget(
    "之前我们把好多记录进度的都删掉了",
    route="present_chitchat",
    route_action="skip",
    semantic_debug=route_debug(),
)
assert plain_prior["final_budget"] == BUDGET_SHALLOW
assert plain_prior["deep_recall_markers"] == []

remember_reaction = build_retrieval_budget(
    "竟然还记得白粥>_<！",
    route="present_chitchat",
    route_action="skip",
    semantic_debug=route_debug(),
)
assert remember_reaction["final_budget"] == BUDGET_SHALLOW

prior_question = build_retrieval_budget(
    "我之前提过 ESP32-S3 的问题吗",
    route="present_chitchat",
    route_action="skip",
    semantic_debug=route_debug(),
)
assert prior_question["final_budget"] == BUDGET_DEEP
assert "prior_mention_question" in prior_question["deep_recall_markers"]

surface = build_retrieval_budget(
    "晚安",
    route="present_chitchat",
    route_action="skip",
    semantic_debug=route_debug(),
)
finalize_retrieval_budget(
    surface,
    {"called": True, "floor_qualified_count": 0, "candidates": []},
)
assert surface["final_budget"] == BUDGET_SKIP

address = build_retrieval_budget(
    "哥哥",
    route="present_chitchat",
    route_action="skip",
    semantic_debug=route_debug(),
)
assert address["surface_only_kind"] == "address_only"

contact = build_retrieval_budget(
    "亲亲抱抱",
    route="present_chitchat",
    route_action="skip",
    semantic_debug=route_debug(),
)
assert contact["surface_only_kind"] == "intimate_contact_only"

print("retrieval budget three-state verification passed")
