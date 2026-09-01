from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway import GatewayService
from gateway_state import GatewayStateStore


class _Reranker:
    def __init__(self) -> None:
        self.calls = 0

    async def rerank(self, _query: str, documents: list[str], top_n: int | None = None):
        _ = top_n
        self.calls += 1
        return [SimpleNamespace(index=index, score=0.92) for index in range(len(documents))]


class _NarrativeStore:
    @staticmethod
    def arc_card_by_key(arc_key: str) -> dict:
        return {
            "status": "ok",
            "item": {
                "arc_key": arc_key,
                "narrative_id": "narrative_spy",
                "title": "间谍过家家",
                "narrative_available": True,
            },
        }


def candidate(owner_kind: str, owner_id: str, *, with_arc: bool) -> dict:
    row = {
        "owner_kind": owner_kind,
        "owner_id": owner_id,
        "title": f"title {owner_id}",
        "memory_date": "2026-08-31",
        "score": 0.81,
        "passages": [{"text": f"body {owner_id}", "score": 0.81}],
    }
    if with_arc:
        row["arc_cards"] = [
            {
                "arc_key": "work:spy",
                "narrative_id": "narrative_spy",
                "title": "间谍过家家",
                "narrative_available": True,
                "read_hint": "可按需读取",
            }
        ]
    return row


async def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        service = GatewayService.__new__(GatewayService)
        service.typed_event_scene_live_enabled = True
        service.typed_event_scene_live_max_cards = 2
        service.state_store = GatewayStateStore(str(Path(directory) / "gateway-state.db"))
        service.reranker_engine = _Reranker()
        service.narrative_roll_store = _NarrativeStore()
        service._passage_candidate_shadow_catalog = {
            "event-spy": {
                "owner_kind": "event",
                "title": "看到第140话",
                "body": "我们一起看到第140话，约好明天继续。",
                "memory_date": "2026-08-31",
                "recallable": True,
            },
            "event-off": {
                "owner_kind": "event",
                "title": "关闭召回的 Event",
                "body": "这条 Event 仍在 shadow，但不能进入 live。",
                "memory_date": "2026-08-31",
                "recallable": False,
            },
            "event-unreviewed": {
                "owner_kind": "event",
                "title": "未审核的 Event",
                "body": "这条 Event 也只能留在 shadow。",
                "memory_date": "2026-08-31",
                "recallable": None,
            },
            "scene-free": {
                "owner_kind": "scene",
                "title": "没有 Arc 的 Scene",
                "body": "这是一段没有 Arc 的 Scene 内容。",
                "memory_date": "2026-08-30",
            },
        }

        def debug(query: str, _embedding: list[float]) -> dict:
            if query == "entity only":
                return {
                    "status": "not_retrieved",
                    "reason": "entity_without_recall_intent",
                    "entity_scope": {"status": "scope_only"},
                }
            if query == "gated":
                return {
                    "status": "ok",
                    "entity_scope": {
                        "status": "global_recall",
                        "operator": "none",
                        "intent": "none",
                    },
                    "candidates": [
                        candidate("event", "event-off", with_arc=False),
                        candidate("event", "event-unreviewed", with_arc=False),
                        candidate("scene", "scene-free", with_arc=False),
                    ],
                }
            scoped = query.startswith("spy")
            row = candidate(
                "event" if scoped else "scene",
                "event-spy" if scoped else "scene-free",
                with_arc=scoped,
            )
            if query.startswith("绒绒"):
                row["candidate_sources"] = [
                    "scene_whole_embedding",
                    "scene_cue_candidate",
                ]
            owner_entity_matches = (
                [
                    {
                        "owner_kind": "scene",
                        "owner_id": "scene-free",
                        "entity": "阿尼亚",
                        "source_kind": "observed_entity",
                    }
                ]
                if query.startswith(("阿尼亚说过什么", "阿尼亚是谁"))
                else []
            )
            return {
                "status": "ok",
                "entity_scope": (
                    {
                        "status": "scoped_recall",
                        "operator": "member_search",
                        "intent": "member_search",
                        "intent_view": "第一次一起看的时候",
                        "scope_anchor": {"arc_key": "work:spy"},
                    }
                    if scoped
                    else {"status": "global_recall", "operator": "none", "intent": "none"}
                ),
                "owner_entity_matches": owner_entity_matches,
                "candidates": [row],
            }

        service._passage_candidate_shadow_debug = debug

        async def menu(_arc_key: str) -> dict:
            return {
                "status": "ok",
                "arc_key": "work:spy",
                "title": "间谍过家家",
                "material_count": 12,
                "menu_truncated": True,
                "menu_fingerprint": "menu-spy-v1",
                "materials": [
                    {
                        "index": 0,
                        "kind": "narrative",
                        "id": "narrative_spy",
                        "title": "间谍过家家",
                        "date": "",
                    },
                    {
                        "index": 11,
                        "kind": "event",
                        "id": "event-spy",
                        "title": "看到第140话",
                        "date": "2026-08-31",
                    },
                ],
            }

        service._typed_arc_material_menu = menu

        first = await service._typed_event_scene_live_context(
            "spy detail",
            "session-a",
            [0.1],
        )
        assert first["status"] == "injected", first
        assert "我们一起看到第140话" in first["context"], first
        assert "Arc: 间谍过家家 (key=work:spy) [可按需读取]" in first["context"], first
        assert "[arc_materials key=work:spy]" in first["context"], first
        assert "read_arc_materials" in first["context"], first
        assert first["menus_included"] == ["work:spy"], first
        assert "叙事正文" not in first["context"], first
        service.state_store.record_arc_material_menu_injection(
            "session-a",
            "work:spy",
            menu_fingerprint=first["menu_fingerprints"]["work:spy"],
        )

        second = await service._typed_event_scene_live_context(
            "spy detail",
            "session-a",
            [0.1],
        )
        assert "我们一起看到第140话" in second["context"], second
        assert "[arc_materials key=work:spy]" not in second["context"], second
        assert second["menus_suppressed"] == ["work:spy"], second

        other_session = await service._typed_event_scene_live_context(
            "spy detail",
            "session-b",
            [0.1],
        )
        assert other_session["menus_included"] == ["work:spy"], other_session

        free = await service._typed_event_scene_live_context(
            "free detail",
            "session-a",
            [0.1],
        )
        assert "可能是你的相关记忆，若无关可忽略" in free["context"], free
        assert "[arc_materials" not in free["context"], free

        daily_semantic = {
            "route": "present_chitchat",
            "route_action": "skip",
            "confidence": 0.94,
            "margin": 0.40,
        }
        reranker_calls_before_daily = service.reranker_engine.calls
        daily = await service._typed_event_scene_live_context(
            "抱抱我",
            "session-a",
            [0.1],
            semantic_recall_debug=daily_semantic,
        )
        assert daily["status"] == "not_retrieved", daily
        assert daily["reason"] == "daily_surface_without_memory_intent", daily
        assert daily["surface_reranker_gate"]["applied"] is True, daily
        assert daily["candidate_count"] == 1, daily
        assert daily["context"] == "", daily
        assert service.reranker_engine.calls == reranker_calls_before_daily, daily

        daily_reality = await service._typed_event_scene_live_context(
            "今天好热啊",
            "session-a",
            [0.1],
            semantic_recall_debug={
                **daily_semantic,
                "route": "present_reality",
            },
        )
        assert daily_reality["status"] == "not_retrieved", daily_reality
        assert service.reranker_engine.calls == reranker_calls_before_daily, daily_reality

        noisy_single_entity = await service._typed_event_scene_live_context(
            "晚安老公",
            "session-a",
            [0.1],
            semantic_recall_debug={
                **daily_semantic,
                "retrieval_budget": {
                    "anchor_override": True,
                    "query_facets": [
                        {
                            "kind": "entity",
                            "value": "晚安",
                            "source": "query_planner.locatable_terms",
                        },
                        {
                            "kind": "entity",
                            "value": "老公",
                            "source": "query_planner.locatable_terms",
                        },
                    ],
                },
            },
        )
        assert noisy_single_entity["status"] == "not_retrieved", noisy_single_entity
        assert service.reranker_engine.calls == reranker_calls_before_daily, noisy_single_entity

        planner_only_detail = await service._typed_event_scene_live_context(
            "路人怎么评价这件事",
            "session-a",
            [0.1],
            semantic_recall_debug={
                **daily_semantic,
                "retrieval_budget": {
                    "anchor_override": True,
                    "query_facets": [
                        {"kind": "entity", "value": "路人"},
                        {"kind": "entity", "value": "评价"},
                    ],
                },
            },
        )
        assert planner_only_detail["status"] == "not_retrieved", planner_only_detail
        assert planner_only_detail["surface_reranker_gate"]["has_memory_backed_detail"] is False, planner_only_detail
        assert service.reranker_engine.calls == reranker_calls_before_daily, planner_only_detail

        named_detail = await service._typed_event_scene_live_context(
            "绒绒怎么评价澜这个名字",
            "session-a",
            [0.1],
            semantic_recall_debug={
                **daily_semantic,
                "retrieval_budget": {
                    "anchor_override": True,
                    "query_facets": [
                        {"kind": "entity", "value": "绒绒"},
                        {"kind": "entity", "value": "评价"},
                    ],
                },
            },
        )
        assert named_detail["status"] == "injected", named_detail
        assert named_detail["surface_reranker_gate"]["has_anchor"] is False, named_detail
        assert named_detail["surface_reranker_gate"]["has_memory_backed_detail"] is True, named_detail
        assert named_detail["surface_reranker_gate"]["matched_candidate_sources"] == [
            "scene_cue_candidate"
        ], named_detail
        assert service.reranker_engine.calls == reranker_calls_before_daily + 1, named_detail

        observed_detail = await service._typed_event_scene_live_context(
            "阿尼亚说过什么",
            "session-a",
            [0.1],
            semantic_recall_debug=daily_semantic,
        )
        assert observed_detail["status"] == "injected", observed_detail
        assert observed_detail["surface_reranker_gate"]["has_memory_backed_detail"] is True, observed_detail
        assert observed_detail["surface_reranker_gate"]["owner_entity_matches"], observed_detail
        assert service.reranker_engine.calls == reranker_calls_before_daily + 2, observed_detail

        observed_definition = await service._typed_event_scene_live_context(
            "阿尼亚是谁",
            "session-a",
            [0.1],
            semantic_recall_debug=daily_semantic,
        )
        assert observed_definition["status"] == "injected", observed_definition
        assert observed_definition["surface_reranker_gate"]["has_memory_backed_detail"] is True, observed_definition
        assert observed_definition["surface_reranker_gate"]["matched_detail_markers"] == ["是谁"], observed_definition
        assert observed_definition["surface_reranker_gate"]["owner_entity_matches"], observed_definition
        assert service.reranker_engine.calls == reranker_calls_before_daily + 3, observed_definition

        explicit_recall = await service._typed_event_scene_live_context(
            "还记得 free detail 吗",
            "session-a",
            [0.1],
            semantic_recall_debug=daily_semantic,
        )
        assert explicit_recall["status"] == "injected", explicit_recall
        assert explicit_recall["surface_reranker_gate"]["applied"] is False, explicit_recall
        assert explicit_recall["surface_reranker_gate"]["has_explicit_recall"] is True, explicit_recall
        assert service.reranker_engine.calls == reranker_calls_before_daily + 4, explicit_recall

        scoped_surface = await service._typed_event_scene_live_context(
            "spy detail",
            "session-c",
            [0.1],
            semantic_recall_debug=daily_semantic,
        )
        assert scoped_surface["status"] == "injected", scoped_surface
        assert scoped_surface["surface_reranker_gate"]["applied"] is False, scoped_surface
        assert scoped_surface["surface_reranker_gate"]["has_arc_scope"] is True, scoped_surface
        assert service.reranker_engine.calls == reranker_calls_before_daily + 5, scoped_surface

        gated = await service._typed_event_scene_live_context(
            "gated",
            "session-a",
            [0.1],
        )
        assert gated["status"] == "injected", gated
        assert gated["selected_refs"] == ["scene:scene-free"], gated
        assert "event-off" not in gated["context"], gated
        assert "event-unreviewed" not in gated["context"], gated
        assert gated["excluded_event_refs_by_recallable"] == [
            "event:event-off",
            "event:event-unreviewed",
        ], gated

        veto = await service._typed_event_scene_live_context(
            "entity only",
            "session-a",
            [0.1],
        )
        assert veto["context"] == "", veto
        assert veto["status"] == "not_retrieved", veto

    print("TYPED_LIVE_CONTEXT_OK")


if __name__ == "__main__":
    asyncio.run(main())
