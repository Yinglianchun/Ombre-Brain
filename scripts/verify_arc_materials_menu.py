from __future__ import annotations

import asyncio
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import server


ARC_KEY = "work:arc-material-test"


class _NarrativeStore:
    @staticmethod
    def find_arc_cards(query: str, limit: int = 5) -> dict:
        return {
            "status": "ok",
            "query": query,
            "count": 1,
            "items": [
                {
                    "arc_key": ARC_KEY,
                    "narrative_id": "narrative_arc_material_test",
                    "title": "Arc material test",
                    "narrative_available": True,
                    "read_hint": "可按需读取",
                }
            ][:limit],
        }

    @staticmethod
    def arc_material_profile_by_key(arc_key: str) -> dict:
        assert arc_key == ARC_KEY
        return {
            "status": "ok",
            "arc_key": ARC_KEY,
            "card": {
                "arc_key": ARC_KEY,
                "narrative_id": "narrative_arc_material_test",
                "title": "Arc material test",
                "narrative_available": True,
            },
            "linked_scene_ids": ["scene_material_test"],
            "linked_event_ids": [f"event_material_{index:02d}" for index in range(1, 12)],
            "linked_diary_ids": [42],
            "linked_darkroom_count": 1,
            "body_included": False,
        }

    @staticmethod
    def arc_card_by_key(_arc_key: str) -> dict:
        return {"status": "not_found"}


class _EntityIndex:
    @staticmethod
    def resolve_query(_query: str) -> dict:
        return {"status": "no_scope", "scope_anchor": {}}


class _FactStore:
    @staticmethod
    def arc_event_links(_arc_key: str) -> list[dict]:
        return []

    @staticmethod
    def read(event_id: str, *, include_sources: bool = False) -> dict:
        _ = include_sources
        index = int(event_id.rsplit("_", 1)[-1])
        return {
            "item_type": "event",
            "item_id": event_id,
            "title": f"Event {index}",
            "local_end_date": f"2026-01-{index:02d}",
        }


class _BucketManager:
    @staticmethod
    async def get(scene_id: str) -> dict:
        return {
            "id": scene_id,
            "content": "Scene body must not enter the menu.",
            "metadata": {"name": "Scene material", "date": "2026-01-12"},
        }


class _DiaryStore:
    @staticmethod
    def read(*, diary_id: int, limit: int = 1) -> dict:
        _ = limit
        return {
            "status": "ok",
            "diaries": [
                {
                    "id": diary_id,
                    "title": "Diary material",
                    "date": "2026-01-13",
                    "content": "Diary body must not enter the menu.",
                }
            ],
        }


async def main() -> None:
    tools = {tool.name: tool for tool in await server.mcp.list_tools()}
    assert "read_arc_materials" in tools
    schema = tools["read_arc_materials"].inputSchema
    assert set(schema["required"]) == {"arc_key", "picks"}, schema
    assert "context" not in schema["properties"], schema

    originals = {
        "narrative_roll_store": server.narrative_roll_store,
        "observed_entity_shadow_index": server.observed_entity_shadow_index,
        "fact_event_store": server.fact_event_store,
        "bucket_mgr": server.bucket_mgr,
        "diary_store": server.diary_store,
        "read_memory": server.read_memory,
        "read_diary": server.read_diary,
    }
    calls: list[tuple[str, str]] = []

    async def fake_read_memory(memory_type: str, memory_id: str, include_content: bool = True):
        calls.append((memory_type, memory_id))
        return {"status": "ok", "memory_type": memory_type, "memory_id": memory_id}

    async def fake_read_diary(diary_id: int = 0, **_kwargs):
        calls.append(("diary", str(diary_id)))
        return {"status": "ok", "diary_id": diary_id}

    try:
        server.narrative_roll_store = _NarrativeStore()
        server.observed_entity_shadow_index = _EntityIndex()
        server.fact_event_store = _FactStore()
        server.bucket_mgr = _BucketManager()
        server.diary_store = _DiaryStore()
        server._arc_material_menu_snapshots.clear()

        found = await server.find_arc("Arc material test")
        menu = found["items"][0]
        assert menu["material_count"] == 14, menu
        assert menu["menu_truncated"] is True, menu
        assert menu["body_included"] is False, menu
        assert "all_materials" not in menu, menu
        assert [item["index"] for item in menu["materials"]] == [
            0,
            1,
            2,
            3,
            4,
            9,
            10,
            11,
            12,
            13,
        ], menu
        assert all("content" not in item for item in menu["materials"]), menu

        server.read_memory = fake_read_memory
        server.read_diary = fake_read_diary
        server._arc_material_menu_snapshots.clear()
        read = await server.read_arc_materials(ARC_KEY, [0, 2, 13])
        assert read["status"] == "ok", read
        assert read["picks"] == [0, 2, 13], read
        assert calls == [
            ("narrative", "narrative_arc_material_test"),
            ("event", "event_material_02"),
            ("diary", "42"),
        ], calls

        hidden = await server.read_arc_materials(ARC_KEY, [8])
        assert hidden["reason"] == "material_index_not_in_menu", hidden
        too_many = await server.read_arc_materials(ARC_KEY, [0, 1, 2, 3, 4, 9])
        assert too_many["reason"] == "too_many_picks", too_many
        assert too_many["max_picks"] == 5, too_many
    finally:
        for name, value in originals.items():
            setattr(server, name, value)
        server._arc_material_menu_snapshots.clear()

    print("ARC_MATERIALS_MENU_OK")


if __name__ == "__main__":
    asyncio.run(main())
