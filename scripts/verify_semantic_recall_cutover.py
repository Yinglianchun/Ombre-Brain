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
            "would_skip": self.skip,
            "recommended_action": "skip" if self.skip else "recall",
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
assert route_source["dataset_version"] == 7
route_examples = {
    route["name"]: {
        item["text"]: item["source"]
        for item in route["utterances"]
    }
    for route in route_source["routes"]
}
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
for query in (
    "但router似乎不大准！也不算陪伴与贴近吧!",
    "记忆库ui还没做完，但试着接上真实hook观察了",
    "哈，还没开始删呢",
    "影分身说，现在记忆库有18个工具……感觉太多了TvT",
):
    assert route_examples["present_chitchat"][query] == "historical_false_positive"
for query in (
    "嘿嘿，没什么事",
    "嗯嗯好",
    "这个判断好像不太准",
    "这个分类是不是弄错了",
    "还没开始弄呢",
    "这个页面还没做完，先接上真实数据看看",
):
    assert query not in route_examples["simple_contact"]
    assert route_examples["present_chitchat"][query] == "seed"
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


async def verify_fast_hook_uses_the_same_semantic_entry() -> None:
    service = GatewayService.__new__(GatewayService)
    service._authorize = lambda authorization: None
    service.upstream_default_model = ""
    service.upstream_models = []
    service.semantic_recall_router = HookRouterStub(skip=True)

    skipped = await service.handle_hook_recall(
        RequestStub({"query": "回来看看你", "include_debug": True})
    )
    skipped_body = json.loads(skipped.body)
    assert skipped_body["cards"] == []
    assert skipped_body["debug"]["hook_recall_debug"]["skip_reason"] == "semantic_recall_skip"

    captured: dict[str, object] = {}

    async def fast_cards(query: str, session_id: str, **kwargs):
        captured.update(kwargs)
        return [], [], {"hook_recall_debug": {"candidate_count": 0}}

    service.semantic_recall_router = HookRouterStub(skip=False)
    service._hook_recall_fast_cards = fast_cards
    service._render_hook_recall_additional_context = lambda cards: ""
    recalled = await service.handle_hook_recall(
        RequestStub({"query": "我去修记忆库了", "include_debug": True})
    )
    recalled_body = json.loads(recalled.body)
    assert recalled_body["ok"] is True
    assert captured["query_embedding"] == [0.1, 0.2]
    assert captured["allow_semantic"] is True


asyncio.run(verify_fast_hook_uses_the_same_semantic_entry())

print("semantic recall cutover verification passed")
