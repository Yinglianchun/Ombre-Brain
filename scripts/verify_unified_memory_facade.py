"""Verify the compact daily memory facade and its runtime dispatch."""

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


async def main() -> None:
    tools = {tool.name: tool for tool in await server.mcp.list_tools()}
    assert len(tools) == 14
    assert {"read_memory", "write_memory", "edit_memory"} <= set(tools)
    for retired_name in (
        "recall",
        "write_scene",
        "edit_scene",
        "set_scene_status",
        "revise_window_shadow",
        "read_portrait",
    ):
        assert retired_name not in tools

    originals = {
        "_recall_memory": server._recall_memory,
        "read_portrait": server.read_portrait,
        "window_shadow_store": server.window_shadow_store,
        "write_scene": server.write_scene,
        "edit_scene": server.edit_scene,
        "set_scene_status": server.set_scene_status,
        "revise_window_shadow": server.revise_window_shadow,
    }
    calls: list[tuple[str, dict]] = []

    async def fake_recall(**kwargs):
        calls.append(("search", kwargs))
        return "scene-search"

    async def fake_portrait(**kwargs):
        calls.append(("portrait", kwargs))
        return {"status": "ok", **kwargs}

    async def fake_write(**kwargs):
        calls.append(("write", kwargs))
        return "scene-created"

    async def fake_edit_scene(**kwargs):
        calls.append(("edit_scene", kwargs))
        return {"status": "updated"}

    async def fake_status(**kwargs):
        calls.append(("status", kwargs))
        return {"status": "archived"}

    async def fake_shadow(**kwargs):
        calls.append(("shadow", kwargs))
        return {"status": "revised"}

    try:
        server._recall_memory = fake_recall
        server.read_portrait = fake_portrait
        server.window_shadow_store = _LatestShadowStore()
        server.write_scene = fake_write
        server.edit_scene = fake_edit_scene
        server.set_scene_status = fake_status
        server.revise_window_shadow = fake_shadow

        searched = await server.read_memory(query="雨天", date="2026-08-01")
        opened = await server.read_memory(memory_type="shadow")
        portrait = await server.read_memory(
            memory_type="portrait",
            scope="relationship",
            include_evidence_text=False,
        )
        written = await server.write_memory(content="一件具体经历。", cues="提到这件事")
        edited = await server.edit_memory(
            memory_id="scene_facade",
            expected_updated_at="revision-one",
            title="修订标题",
        )
        archived = await server.edit_memory(
            memory_id="scene_facade",
            expected_updated_at="revision-two",
            status="archived",
        )
        revised_shadow = await server.edit_memory(
            memory_id="window_latest_facade",
            content="完整修订窗影。",
            expected_source_hash="a" * 64,
            idempotency_key="facade-revision",
        )

        assert searched == "scene-search"
        assert opened["window"]["window_id"] == "window_latest_facade"
        assert opened["ordinary_recall"] is False
        assert portrait["scope"] == "relationship"
        assert written == "scene-created"
        assert edited["status"] == "updated"
        assert archived["status"] == "archived"
        assert revised_shadow["status"] == "revised"
        assert [name for name, _ in calls] == [
            "search",
            "portrait",
            "write",
            "edit_scene",
            "status",
            "shadow",
        ]
    finally:
        for name, value in originals.items():
            setattr(server, name, value)

    print("unified daily memory facade verified")


if __name__ == "__main__":
    asyncio.run(main())
