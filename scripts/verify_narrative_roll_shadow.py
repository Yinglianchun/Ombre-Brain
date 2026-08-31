from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway import GatewayService
from narrative_rolls import NarrativeRollStore


SCENE_A = "scene_mig2_alpha"
SCENE_B = "scene_mig2_beta"


def build_store(state_dir: str) -> NarrativeRollStore:
    store = NarrativeRollStore(
        {
            "state_dir": state_dir,
            "narrative_rolls": {
                "shadow_admission_enabled": True,
                "live_injection_enabled": True,
            },
        }
    )
    document = f"""# Narrative Roll

## 第一人称叙事

我们借别人的故事说自己。

## 来源

- {SCENE_A}
- {SCENE_B}
"""
    result = store.publish(
        narrative_id="narrative_stories_reflect_us",
        document=document,
        expected_revision=0,
        title="我们借别人的故事说自己",
        source_scene_ids=[SCENE_A, SCENE_B],
        primary_entities=["哥哥"],
        query_cues=["借故事"],
    )
    assert result["status"] == "created", result
    return store


def verify_shadow_gate() -> None:
    with TemporaryDirectory(prefix="ombre-narrative-shadow-") as temp_dir:
        service = GatewayService.__new__(GatewayService)
        service.narrative_roll_store = build_store(temp_dir)

        class ScopeResolver:
            @staticmethod
            def resolve_query(query: str) -> dict:
                if "我们借别人的故事说自己" not in query:
                    return {
                        "status": "no_scope",
                        "scope_anchor": None,
                        "retrieval_allowed": False,
                    }
                return {
                    "status": "scoped_recall" if "整体" in query else "scope_only",
                    "scope_anchor": {"arc_key": ""},
                    "retrieval_allowed": "整体" in query,
                }

        service.observed_entity_shadow_index = ScopeResolver()

        exact_title = service._narrative_roll_shadow_debug(
            "我们借别人的故事说自己",
            [],
            route_allowed=True,
        )
        assert exact_title["status"] == "not_admitted", exact_title
        assert exact_title["reason"] == "scope_intent_conjunction_not_satisfied", exact_title
        assert exact_title["would_admit_narrative_id"] == "narrative_stories_reflect_us"
        assert exact_title["admitted_narrative_id"] == ""
        assert exact_title["mode"] == "shadow_only"
        assert exact_title["gateway_live_injection_enabled"] is False
        assert exact_title["visible_injection"] is False
        assert exact_title["available_narrative_count"] == 1
        assert "body" not in exact_title

        scoped_title = service._narrative_roll_shadow_debug(
            "我们借别人的故事说自己整体讲讲",
            [],
            route_allowed=True,
        )
        assert scoped_title["status"] == "shadow_admitted", scoped_title
        assert scoped_title["admitted_narrative_id"] == "narrative_stories_reflect_us"

        two_scenes = service._narrative_roll_shadow_debug(
            "这条普通语句没有标题或实体",
            [SCENE_A, SCENE_B],
            route_allowed=True,
        )
        assert two_scenes["status"] == "shadow_admitted", two_scenes
        assert two_scenes["reason"] == "two_independent_first_hop_scenes", two_scenes

        blocked_chitchat = service._narrative_roll_shadow_debug(
            "想哥哥",
            [],
            route_allowed=False,
            route_block_reason="semantic_recall_skip",
        )
        assert blocked_chitchat["status"] == "route_blocked", blocked_chitchat
        assert blocked_chitchat["reason"] == "semantic_recall_skip"
        assert blocked_chitchat["would_admit_narrative_id"] == "narrative_stories_reflect_us"
        assert blocked_chitchat["admitted_narrative_id"] == ""
        assert blocked_chitchat["visible_injection"] is False


async def verify_fast_hook_debug_wiring() -> None:
    with TemporaryDirectory(prefix="ombre-narrative-hook-") as temp_dir:
        service = GatewayService.__new__(GatewayService)
        service.narrative_roll_store = build_store(temp_dir)
        scene = {
            "id": SCENE_A,
            "metadata": {"memory_value_source": "authored_scene"},
            "content": "一条已被普通召回选中的 Scene。",
        }

        service._query_planner_debug_base = lambda query: {"query": query}

        async def list_buckets(*, include_archive: bool = False):
            return [scene]

        async def select_buckets(*args, **kwargs):
            return [scene], [], {"query": args[0]}

        service._list_gateway_buckets = list_buckets
        service._dynamic_recall_search_query = lambda query: query
        service._select_dynamic_buckets = select_buckets
        service._with_explicit_source_record_buckets = lambda query, selected, all_buckets: selected
        service._hook_recall_card_from_bucket = lambda bucket, **kwargs: None
        service._format_selected_bucket_debug = lambda bucket, **kwargs: {"bucket_id": bucket["id"]}
        service._format_suppressed_bucket_debug = lambda item, **kwargs: item

        cards, recalled_ids, debug = await service._hook_recall_fast_cards(
            "我们借别人的故事说自己",
            "narrative-shadow-test",
            max_cards=2,
            max_chars=800,
            include_diffused=False,
            allow_semantic=True,
            query_embedding=None,
            semantic_recall_debug={},
            allow_semantic_session_dedupe=False,
            allow_rerank=False,
        )
        assert cards == []
        assert recalled_ids == []
        narrative = debug["narrative_recall_debug"]
        assert narrative["status"] == "shadow_admitted", narrative
        assert narrative["direct_first_hop_scene_ids"] == [SCENE_A]
        assert narrative["visible_injection"] is False
        assert "body" not in narrative


def main() -> int:
    verify_shadow_gate()
    asyncio.run(verify_fast_hook_debug_wiring())
    print("narrative roll shadow verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
