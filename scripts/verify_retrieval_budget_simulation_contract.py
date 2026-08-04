"""Verify budget debug is explicit-simulation-only at the Hook boundary."""

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


class Request:
    def __init__(self, body: dict):
        self._body = body
        self.headers = {}

    async def json(self) -> dict:
        return dict(self._body)


class Router:
    async def route_with_vector(self, query: str):
        return (
            {
                "route": "present_chitchat",
                "route_action": "recall",
                "confidence": 0.96,
                "margin": 0.22,
                "threshold": 0.72,
            },
            [0.1, 0.2],
        )

    @staticmethod
    def should_apply_skip(debug: dict) -> bool:
        return False


def service() -> tuple[GatewayService, list[str]]:
    instance = GatewayService.__new__(GatewayService)
    instance.recall_policy = RecallPolicy()
    instance.gateway_tz = timezone.utc
    instance.first_card_min_score = 0.55
    instance.retrieval_budget_sentinel_rescue_floor = 0.55
    instance.semantic_recall_router = Router()
    instance.upstream_default_model = "test"
    instance.upstream_models = []
    instance._authorize = lambda _header: None
    recorded: list[str] = []

    async def evidence_veto(query, semantic_skip, vector, debug):
        return semantic_skip

    async def fast_cards(*args, **kwargs):
        return (
            [{"id": "scene-test", "title": "test", "content": "not exported"}],
            ["scene-test"],
            {"hook_recall_debug": {"candidate_count": 1}},
        )

    instance._apply_semantic_scene_evidence_veto = evidence_veto
    instance._hook_recall_fast_cards = fast_cards
    instance._render_hook_recall_additional_context = lambda _cards: "rendered"
    instance._record_hook_recall_injection = lambda _session_id, ids: recorded.extend(ids)
    return instance, recorded


def payload(response) -> dict:
    return json.loads(response.body.decode("utf-8"))


default_service, default_recorded = service()
default = payload(asyncio.run(default_service.handle_hook_recall(Request({
    "query": "普通问题",
    "channel": "haven_bridge",
}))))
assert set(default) == {
    "ok", "query", "session_id", "cards", "notes", "additional_context", "recalled_ids", "debug"
}
assert set(default["debug"]) == {"query", "candidate_count", "snowflake_boosted"}
assert default_recorded == ["scene-test"]

false_service, false_recorded = service()
explicit_false = payload(asyncio.run(false_service.handle_hook_recall(Request({
    "query": "普通问题",
    "channel": "haven_bridge",
    "simulation": False,
}))))
assert explicit_false == default
assert false_recorded == ["scene-test"]

simulation_service, simulation_recorded = service()
simulation = payload(asyncio.run(simulation_service.handle_hook_recall(Request({
    "query": "普通问题",
    "channel": "haven_bridge",
    "simulation": True,
}))))
budget = simulation["debug"]["semantic_recall_debug"]["retrieval_budget"]
assert budget["mode"] == "simulation_shadow"
assert budget["sentinel"]["called"] is False
assert simulation_recorded == []

print("retrieval budget simulation contract verification passed")
