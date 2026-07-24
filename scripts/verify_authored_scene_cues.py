"""Verify that every new canonical Scene carries author-written recall cues."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server


class _NoopDecay:
    async def ensure_started(self) -> None:
        return None


async def main() -> None:
    tools = {tool.name: tool for tool in await server.mcp.list_tools()}
    write_schema = tools["write_scene"].inputSchema
    close_schema = tools["close_window"].inputSchema
    close_scene_schema = close_schema["$defs"]["CloseWindowSceneInput"]

    assert "cues" in write_schema["required"]
    assert set(close_scene_schema["required"]) == {"content", "cues"}
    assert server._authored_scene_cues("") == []
    assert server._authored_scene_cues("提到 authored cues") == ["提到 authored cues"]

    server.decay_engine = _NoopDecay()
    rejected = await server._write_scene_memory(
        '标题与引句“都不该被拿来补 cue”。',
        title="描述性标题",
        cues="",
    )
    assert "必须由当前作者亲自写至少一个 cues" in rejected

    shadow = "## 这一窗之后，什么留在了我身上\n我记住了这条作者契约。"
    records, error = server._window_shadow_scene_records(
        shadow,
        [
            {
                "content": "这段 Scene 原文与召回入口分开保存。",
                "title": "作者权",
                "cues": ["提到 Scene 的作者权", "问 cues 应该由谁写"],
            }
        ],
    )
    assert error == ""
    assert records[0]["cues"] == ["提到 Scene 的作者权", "问 cues 应该由谁写"]

    records, error = server._window_shadow_scene_records(
        (
            f"{shadow}\n\n"
            "## 想留下的记忆\n"
            "### scene | 提到 close_window | 问内联 Scene 的 cues\n"
            "内联 heading 只携带抽取边界和亲自写下的召回入口。"
        ),
        None,
    )
    assert error == ""
    assert records[0]["cues"] == ["提到 close_window", "问内联 Scene 的 cues"]

    records, error = server._window_shadow_scene_records(
        shadow,
        [{"content": "没有 authored cues 的 Scene。", "title": "不能顶替 cue"}],
    )
    assert records == []
    assert "缺少有效 cues" in error

    print("authored Scene cues verified")


if __name__ == "__main__":
    asyncio.run(main())
