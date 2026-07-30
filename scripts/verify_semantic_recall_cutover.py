from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway import GatewayService
from memory_recall.semantic_router import SemanticRecallRouter
from recall_policy import RecallPolicy


class EmbeddingStub:
    enabled = False
    model = "stub"
    query_instruction = ""
    max_chars = 0


def router(mode: str | None = None, *, shadow_enabled: bool = False) -> SemanticRecallRouter:
    semantic_config: dict[str, object] = {"shadow_enabled": shadow_enabled}
    if mode is not None:
        semantic_config["mode"] = mode
    return SemanticRecallRouter(
        {"gateway": {"semantic_recall_router": semantic_config}},
        EmbeddingStub(),
    )


def service_with(active_router: SemanticRecallRouter) -> GatewayService:
    service = GatewayService.__new__(GatewayService)
    service.semantic_recall_router = active_router
    return service


active = router("active")
assert active.enabled is True
assert active.active is True
assert active.debug_base("想你了")["shadow_only"] is False
assert active.should_apply_skip({"would_skip": True}) is True
assert active.should_apply_skip({"would_skip": False}) is False

shadow = router("shadow")
assert shadow.enabled is True
assert shadow.active is False
assert shadow.debug_base("想你了")["shadow_only"] is True
assert shadow.should_apply_skip({"would_skip": True}) is False

legacy_shadow = router(None, shadow_enabled=True)
assert legacy_shadow.mode == "shadow"

active_service = service_with(active)
assert active_service._recall_entry_skip_routes(
    {"would_skip": False},
    sentinel_route="tone_only",
    low_signal_auto_recall=True,
) == (False, False, True, False)
assert active_service._recall_entry_skip_routes(
    {"would_skip": True},
    sentinel_route="",
    low_signal_auto_recall=False,
) == (True, False, False, False)

shadow_service = service_with(shadow)
assert shadow_service._recall_entry_skip_routes(
    {"would_skip": True},
    sentinel_route="tone_only",
    low_signal_auto_recall=True,
) == (False, True, True, True)

assert active_service.semantic_recall_router.active is True
assert shadow_service.semantic_recall_router.active is False
assert GatewayService._legacy_auto_vague_skip_applies(
    True,
    semantic_entry_routed=True,
) is False
assert GatewayService._legacy_auto_vague_skip_applies(
    True,
    semantic_entry_routed=False,
) is True

policy = RecallPolicy()
vague_query = "你怎么看"
vague_plan = policy.plan_query(vague_query)
assert vague_plan.auto_too_vague is True

semantic_candidate = policy.assess(
    vague_query,
    {},
    query_plan=vague_plan,
    semantic_score=0.80,
    auto=True,
    semantic_entry_routed=True,
)
assert semantic_candidate.admit_direct is True
assert semantic_candidate.reason == "non_explicit_query"
assert semantic_candidate.debug["auto_too_vague"] is False
assert semantic_candidate.debug["legacy_auto_vague_would_suppress"] is True

legacy_candidate = policy.assess(
    vague_query,
    {},
    query_plan=vague_plan,
    semantic_score=0.80,
    auto=True,
    semantic_entry_routed=False,
)
assert legacy_candidate.admit_direct is False
assert legacy_candidate.reason == "auto_vague_query_without_topic"

print("semantic recall cutover verification passed")
