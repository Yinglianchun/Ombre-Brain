"""Verify exact typed reads stay separate from associative recall and writes."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server


class _LatestShadowStore:
    @staticmethod
    def latest() -> dict:
        return {
            "window_id": "window_latest_facade",
            "content": "最新窗影本身就是交接。",
            "source_hash": "a" * 64,
        }


class _SceneBucketManager:
    async def get(self, scene_id: str) -> dict:
        return {
            "id": scene_id,
            "content": "这是一条从窗影抽出的具体 Scene。",
            "metadata": {
                "name": "窗影中的 Scene",
                "object_kind": "scene",
                "memory_value_source": "authored_scene",
                "active": True,
            },
        }


async def main() -> None:
    tools = {tool.name: tool for tool in await server.mcp.list_tools()}
    registered = {tool.name for tool in server.mcp._tool_manager.list_tools()}
    assert len(tools) == 16
    assert registered == set(tools)
    assert {
        "recall_memory",
        "read_memory",
        "write_scene",
        "edit_scene",
        "set_scene_status",
        "close_window",
        "revise_window_shadow",
        "publish_narrative",
    } <= set(tools)
    for retired_name in (
        "recall",
        "write_memory",
        "edit_memory",
        "read_portrait",
        "publish_portrait",
    ):
        assert retired_name not in tools

    read_schema = tools["read_memory"].inputSchema
    assert set(read_schema["required"]) == {"memory_type", "memory_id"}
    assert read_schema["properties"]["memory_type"]["enum"] == [
        "scene",
        "shadow",
        "narrative",
    ]
    assert "query" not in read_schema["properties"]
    assert "date" not in read_schema["properties"]
    recall_schema = tools["recall_memory"].inputSchema["properties"]
    assert {"query", "scene_id", "date", "include_related"} <= set(recall_schema)

    originals = {
        "_read_scene_memory": server._read_scene_memory,
        "_read_window_shadow_memory": server._read_window_shadow_memory,
        "_read_narrative_memory": server._read_narrative_memory,
        "window_shadow_store": server.window_shadow_store,
        "_recall_memory": server._recall_memory,
        "_recall_from_scene_id": server._recall_from_scene_id,
        "bucket_mgr": server.bucket_mgr,
        "_build_mcp_diffused_memory_block": server._build_mcp_diffused_memory_block,
    }
    calls: list[tuple[str, dict | str]] = []

    async def fake_scene(scene_id: str):
        calls.append(("scene", scene_id))
        return {"status": "ok", "scene_id": scene_id}

    async def fake_shadow(**kwargs):
        calls.append(("shadow", kwargs))
        return {"status": "ok", "window_id": kwargs["window_id"]}

    async def fake_narrative(narrative_id: str):
        calls.append(("narrative", narrative_id))
        return {"status": "ok", "narrative_id": narrative_id}

    async def fake_recall(**kwargs):
        calls.append(("query_recall", kwargs))
        return "query-recall"

    async def fake_scene_recall(scene_id: str, **kwargs):
        calls.append(("scene_recall", {"scene_id": scene_id, **kwargs}))
        return "scene-recall"

    async def fake_related_block(source_buckets, *_args, **_kwargs):
        calls.append(("related_block", source_buckets[0]["id"]))
        return "[bucket_id:scene_related] 一条有效关联 Scene"

    try:
        server._read_scene_memory = fake_scene
        server._read_window_shadow_memory = fake_shadow
        server._read_narrative_memory = fake_narrative
        server.window_shadow_store = _LatestShadowStore()
        server._recall_memory = fake_recall
        server._recall_from_scene_id = fake_scene_recall

        inline_scene_id = "window_parent_scene_1"
        scene = await server.read_memory(memory_type="scene", memory_id=inline_scene_id)
        shadow = await server.read_memory(
            memory_type="shadow",
            memory_id="window_parent",
        )
        latest = await server.read_memory(memory_type="shadow", memory_id="latest")
        narrative = await server.read_memory(
            memory_type="narrative",
            memory_id="narrative_thread",
        )
        query_recall = await server.recall_memory(
            query="雨天",
            include_related=True,
        )
        scene_recall = await server.recall_memory(
            scene_id=inline_scene_id,
            include_related=True,
        )
        mixed = await server.recall_memory(query="雨天", scene_id=inline_scene_id)

        server.bucket_mgr = _SceneBucketManager()
        server._build_mcp_diffused_memory_block = fake_related_block
        anchored = await originals["_recall_from_scene_id"](
            inline_scene_id,
            include_related=True,
            related_per_memory=1,
            edge_min_confidence=0.55,
            max_tokens=3000,
        )

        assert scene["scene_id"] == inline_scene_id
        assert shadow["window_id"] == "window_parent"
        assert latest["window"]["window_id"] == "window_latest_facade"
        assert narrative["narrative_id"] == "narrative_thread"
        assert query_recall == "query-recall"
        assert scene_recall == "scene-recall"
        assert "两种召回入口" in mixed
        assert "指定 Scene" in anchored
        assert "scene_related" in anchored
        assert [name for name, _ in calls] == [
            "scene",
            "shadow",
            "narrative",
            "query_recall",
            "scene_recall",
            "related_block",
        ]
    finally:
        for name, value in originals.items():
            setattr(server, name, value)

    print("typed exact read and associative recall facade verified")


if __name__ == "__main__":
    asyncio.run(main())
