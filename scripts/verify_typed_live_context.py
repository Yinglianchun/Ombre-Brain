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
    @staticmethod
    async def rerank(_query: str, documents: list[str], top_n: int | None = None):
        _ = top_n
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
            scoped = query.startswith("spy")
            row = candidate(
                "event" if scoped else "scene",
                "event-spy" if scoped else "scene-free",
                with_arc=scoped,
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
