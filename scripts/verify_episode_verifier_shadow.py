from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway import GatewayService


def service(response: dict) -> GatewayService:
    instance = GatewayService.__new__(GatewayService)
    instance.episode_verifier_shadow_enabled = True
    instance.episode_verifier_model = "deepseek-v4-flash"
    instance.episode_verifier_timeout_seconds = 2
    instance.episode_verifier_max_candidates = 2
    instance.episode_verifier_min_confidence = 0.70
    instance._is_canonical_scene_bucket = lambda _bucket: True

    async def completion(_payload):
        return json.dumps(response, ensure_ascii=False), ""

    instance._episode_verifier_completion = completion
    return instance


def candidate(*, cues: list[str] | None = None) -> dict:
    cues = list(cues or [])
    return {
        "bucket": {
            "id": "scene-xiaohongshu",
            "content": "我们曾经第一次真的一起逛小红书。",
            "metadata": {
                "name": "第一次真的一起逛小红书",
                "memory_value_source": "authored_scene",
                "object_kind": "scene",
                "scene_cues": cues,
            },
        },
        "score": 0.62,
        "semantic_score": 0.62,
        "budget_floor_qualified": True,
        "budget_reranker_eligible": True,
        "cue_semantic_candidate_match": bool(cues),
        "cue_semantic_terms": cues,
    }


def budget() -> dict:
    return {
        "surface_only_kind": "",
        "explicit_deep_reasons": [],
        "cheap_retrieval": {"candidates": []},
    }


noisy_decisions = GatewayService._parse_episode_verifier_response(
    'analysis fragment {not json}\n```json\n'
    '{"decisions":[{"candidate_id":"scene-symbolic","verdict":"symbolic_resonance"}]}'
    '\n```'
)
assert noisy_decisions[0]["candidate_id"] == "scene-symbolic"


topic_response = {
    "decisions": [
        {
            "candidate_id": "scene-xiaohongshu",
            "verdict": "same_topic_only",
            "confidence": 0.94,
            "current_evidence_span": "",
            "grounded_cue": "",
            "reason": "The message only states a current activity.",
        }
    ]
}
topic_item = candidate()
topic_budget = budget()
asyncio.run(
    service(topic_response)._apply_episode_verifier_shadow(
        "我在刷小红书", [topic_item], topic_budget
    )
)
assert topic_item["episode_verifier_shadow"]["verdict"] == "same_topic_only"
assert topic_item["episode_verifier_shadow"]["decision_applied"] is True
assert topic_budget["episode_verifier"]["decision_scope"] == (
    "simulation_negative_veto_only"
)

symbolic_response = {
    "decisions": [
        {
            "candidate_id": "scene-xiaohongshu",
            "verdict": "symbolic_resonance",
            "confidence": 0.88,
            "current_evidence_span": "一扇绿色的小门",
            "grounded_cue": "绿色小门",
            "reason": "The supplied reviewed cue grounds the image.",
        }
    ]
}
symbolic_item = candidate(cues=["绿色小门"])
asyncio.run(
    service(symbolic_response)._apply_episode_verifier_shadow(
        "像一扇绿色的小门", [symbolic_item], budget()
    )
)
assert symbolic_item["episode_verifier_shadow"]["verdict"] == "symbolic_resonance"
assert symbolic_item["episode_verifier_shadow"]["decision_applied"] is False
assert symbolic_item["episode_verifier_shadow"]["grounded_cue"] == "绿色小门"

ungrounded_item = candidate()
asyncio.run(
    service(symbolic_response)._apply_episode_verifier_shadow(
        "像一扇绿色的小门", [ungrounded_item], budget()
    )
)
assert ungrounded_item["episode_verifier_shadow"]["verdict"] == "unrelated"
assert ungrounded_item["episode_verifier_shadow"]["confidence"] == 0.0

print("episode verifier shadow verification passed")
