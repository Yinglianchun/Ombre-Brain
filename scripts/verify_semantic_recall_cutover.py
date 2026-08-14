from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway import GatewayService
from memory_recall.semantic_router import SemanticRecallRouter, load_route_source
from recall_policy import RecallPolicy


class EmbeddingStub:
    enabled = False
    model = "stub"
    query_instruction = ""
    max_chars = 0


class EnabledEmbeddingStub(EmbeddingStub):
    enabled = True

    async def embed_query(self, _text: str):
        return [1.0, 0.0]


class RequestStub:
    headers: dict[str, str] = {}

    def __init__(self, body: dict[str, object]):
        self.body = body

    async def json(self) -> dict[str, object]:
        return self.body


class HookRouterStub:
    active = True

    def __init__(self, *, skip: bool):
        self.skip = skip

    async def route_with_vector(self, query: str):
        return {
            "enabled": True,
            "called": True,
            "route": "present_chitchat" if self.skip else "present_reality",
            "route_action": "skip" if self.skip else "recall",
            "would_skip": self.skip,
            "recommended_action": "skip" if self.skip else "recall",
            "confidence": 0.94 if self.skip else 0.86,
            "margin": 0.18 if self.skip else 0.12,
            "threshold": 0.72,
        }, [0.1, 0.2]

    @staticmethod
    def should_apply_skip(debug):
        return bool(debug.get("would_skip"))


def router(mode: str | None = None, *, shadow_enabled: bool = False) -> SemanticRecallRouter:
    semantic_config: dict[str, object] = {"shadow_enabled": shadow_enabled}
    if mode is not None:
        semantic_config["mode"] = mode
    return SemanticRecallRouter(
        {"gateway": {"semantic_recall_router": semantic_config}},
        EmbeddingStub(),
    )


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

route_source = load_route_source(ROOT / "resources" / "semantic_recall_routes.json")
assert route_source["dataset_version"] == 9
route_examples = {
    route["name"]: {
        item["text"]: item["source"]
        for item in route["utterances"]
    }
    for route in route_source["routes"]
}
route_actions = {route["name"]: route["action"] for route in route_source["routes"]}
assert route_actions["present_reality"] == "skip"
assert route_actions["recall_needed"] == "recall"
assert next(
    route["threshold"]
    for route in route_source["routes"]
    if route["name"] == "present_reality"
) == 0.60
assert not next(route for route in route_source["routes"] if route["name"] == "simple_contact").get("enabled", True)
assert route_examples["simple_contact"] == {}
for query in ("爱你", "最爱哥哥了", "想和哥哥贴贴"):
    assert route_examples["present_chitchat"][query] == "historical_false_positive"
for query in (
    "回来看看你",
    "就是想来看看你",
    "在吗哥哥",
    "你在干嘛呀",
    "想你了",
    "抱抱我",
    "亲一下",
    "我只是来黏你一下",
):
    assert route_examples["present_chitchat"][query] == "seed"
assert route_examples["present_chitchat"]["晚安"] == "historical_false_positive"
for query in ("我在刷小红书", "亲亲抱抱", "老公"):
    assert query in route_examples["present_chitchat"]
for query in (
    "但router似乎不大准！也不算陪伴与贴近吧!",
    "记忆库ui还没做完，但试着接上真实hook观察了",
    "哈，还没开始删呢",
    "影分身说，现在记忆库有18个工具……感觉太多了TvT",
):
    assert route_examples["present_reality"][query] == "historical_false_positive"
for query in ("今天太阳很大", "我有点头疼", "刚看到一本书"):
    assert route_examples["present_reality"][query] == "seed"
    assert query not in route_examples["present_chitchat"]
assert (
    route_examples["recall_needed"]["还记得我们第一次说晚安那次吗"]
    == "hard_negative"
)
assert route_examples["recall_needed"]["哥哥，我们那时候原话是怎么说的呀"] == "seed"
assert "把那段原话找出来" not in route_examples["recall_needed"]

policy = RecallPolicy()
candidate = policy.assess(
    "你怎么看",
    {},
    semantic_score=0.80,
    auto=True,
)
assert candidate.admit_direct is True
assert candidate.reason == "non_explicit_query"
assert "auto_too_vague" not in candidate.debug
assert "legacy_auto_vague_would_suppress" not in candidate.debug


async def verify_hook_modes_use_the_same_semantic_entry() -> None:
    service = GatewayService.__new__(GatewayService)
    service._authorize = lambda authorization: None
    service.upstream_default_model = ""
    service.upstream_models = []
    service.semantic_recall_router = HookRouterStub(skip=True)
    service.semantic_scene_evidence_veto_mode = "off"

    async def unexpected_scene_rescue(*_args, **_kwargs):
        raise AssertionError("direct chitchat skip must precede Scene retrieval")

    service._apply_semantic_scene_evidence_veto = unexpected_scene_rescue

    recorded: list[tuple[str, list[str]]] = []

    class StateStoreStub:
        @staticmethod
        def record_success(session_id: str, recalled_ids: list[str]):
            recorded.append((session_id, list(recalled_ids)))

    service.state_store = StateStoreStub()
    service._record_hook_recall_injection = GatewayService._record_hook_recall_injection.__get__(service)

    skipped = await service.handle_hook_recall(
        RequestStub({"query": "老公亲亲抱抱", "include_debug": True})
    )
    skipped_body = json.loads(skipped.body)
    assert skipped_body["cards"] == []
    assert skipped_body["debug"]["hook_recall_debug"]["skip_reason"] == "semantic_recall_skip"
    assert skipped_body["debug"]["hook_recall_debug"]["mode"] == "fast_bucket"
    assert skipped_body["debug"]["semantic_recall_debug"]["direct_skip"] == {
        "applied": True,
        "reason": "high_confidence_pure_chitchat",
    }

    async def full_recall(**kwargs):
        raise AssertionError("full recall must not run after a semantic skip")

    service._handle_hook_recall_full = full_recall
    skipped_full = await service.handle_hook_recall(
        RequestStub(
            {
                "query": "老公亲亲抱抱",
                "recall_mode": "full",
                "include_debug": True,
            }
        )
    )
    skipped_full_body = json.loads(skipped_full.body)
    assert skipped_full_body["cards"] == []
    assert skipped_full_body["additional_context"] == ""
    assert skipped_full_body["recalled_ids"] == []
    assert skipped_full_body["debug"]["hook_recall_debug"] == {
        "mode": "full_gateway",
        "skip_reason": "semantic_recall_skip",
    }

    captured: dict[str, object] = {}

    async def fast_cards(query: str, session_id: str, **kwargs):
        captured.update(kwargs)
        return [], [], {"hook_recall_debug": {"candidate_count": 0}}

    async def no_scene_veto(*_args, **_kwargs):
        return False

    service._apply_semantic_scene_evidence_veto = no_scene_veto
    service._hook_recall_fast_cards = fast_cards
    service._render_hook_recall_additional_context = lambda cards: ""
    current_state = await service.handle_hook_recall(
        RequestStub({"query": "回来看看你", "include_debug": True})
    )
    current_state_body = json.loads(current_state.body)
    current_state_route = current_state_body["debug"]["semantic_recall_debug"]
    assert current_state_body["ok"] is True
    assert current_state_route["direct_skip"]["applied"] is False
    assert current_state_route["memory_need"] == "optional"
    assert current_state_route["live_probe_mode"] == "optional_shallow"

    service.semantic_recall_router = HookRouterStub(skip=False)
    recalled = await service.handle_hook_recall(
        RequestStub({"query": "我去修记忆库了", "include_debug": True})
    )
    recalled_body = json.loads(recalled.body)
    assert recalled_body["ok"] is True
    assert captured["query_embedding"] == [0.1, 0.2]
    assert captured["allow_semantic"] is True
    assert captured["allow_rerank"] is False
    assert recorded == []

    await service.handle_hook_recall(
        RequestStub(
            {
                "query": "我有点头疼",
                "include_debug": True,
                "allow_rerank": True,
            }
        )
    )
    assert captured["allow_rerank"] is True

    async def injected_cards(query: str, session_id: str, **kwargs):
        return [{"bucket_id": "scene-one", "text": "memory"}], ["scene-one"], {
            "hook_recall_debug": {"candidate_count": 1}
        }

    service._hook_recall_fast_cards = injected_cards
    injected = await service.handle_hook_recall(
        RequestStub(
            {
                "query": "还记得那件事吗",
                "session_id": "haven_bridge:42",
                "channel": "haven_bridge",
            }
        )
    )
    assert json.loads(injected.body)["recalled_ids"] == ["scene-one"]
    assert recorded == [("haven_bridge:42", ["scene-one"])]


asyncio.run(verify_hook_modes_use_the_same_semantic_entry())


async def verify_skip_route_must_clear_confidence_gates() -> None:
    service = SemanticRecallRouter(
        {"gateway": {"semantic_recall_router": {"mode": "active"}}},
        EnabledEmbeddingStub(),
    )
    service._load_index = lambda: ({"embedding": {"dimension": 2}}, "")
    service._score_routes = lambda _index, _vector: [
        {"name": "present_chitchat", "action": "skip", "score": 0.51, "threshold": 0.60, "top_examples": []},
        {"name": "recall_needed", "action": "recall", "score": 0.49, "threshold": 0.72, "top_examples": []},
    ]
    service._best_boundary_veto = lambda _index, _vector, _winner: None
    debug, _vector = await service.route_with_vector("宝宝，看到你的回复了")
    assert debug["route"] == "present_chitchat"
    assert debug["would_skip"] is False
    assert debug["recommended_action"] == "recall"
    assert debug["reason"] == "below_threshold"
    assert debug["threshold_met"] is False
    assert debug["margin_met"] is False

    service._score_routes = lambda _index, _vector: [
        {"name": "present_chitchat", "action": "skip", "score": 0.65, "threshold": 0.60, "top_examples": []},
        {"name": "recall_needed", "action": "recall", "score": 0.63, "threshold": 0.72, "top_examples": []},
    ]
    narrow, _vector = await service.route_with_vector("安静下来以后，有些归属也不会消失，对不对")
    assert narrow["would_skip"] is False
    assert narrow["recommended_action"] == "recall"
    assert narrow["reason"] == "insufficient_margin"
    assert narrow["threshold_met"] is True
    assert narrow["margin_met"] is False

    service._score_routes = lambda _index, _vector: [
        {"name": "present_chitchat", "action": "skip", "score": 0.65, "threshold": 0.60, "top_examples": []},
        {"name": "recall_needed", "action": "recall", "score": 0.30, "threshold": 0.72, "top_examples": []},
    ]
    skipped, _vector = await service.route_with_vector("今天心情还不错")
    assert skipped["would_skip"] is True
    assert skipped["recommended_action"] == "skip"
    assert skipped["reason"] == "matched_skip_route"
    assert skipped["threshold_met"] is True
    assert skipped["margin_met"] is True

    service._best_boundary_veto = lambda _index, _vector, _winner: {
        "route": "recall_needed",
        "action": "recall",
        "text": "边界正例",
        "score": 0.63,
        "passes_threshold": True,
        "beats_skip": True,
        "deficit": 0.0,
        "within_deficit": True,
    }
    guarded, _vector = await service.route_with_vector("哥哥是我的宇宙")
    assert guarded["would_skip"] is False
    assert guarded["reason"] == "boundary_veto"


asyncio.run(verify_skip_route_must_clear_confidence_gates())

print("semantic recall cutover verification passed")
