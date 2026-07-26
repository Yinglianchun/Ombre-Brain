"""Verify authored inline Scene titles/cues and close_window's narrow schema."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server
from gateway import GatewayService


class _NoopDecay:
    async def ensure_started(self) -> None:
        return None


class _CaptureBucketManager:
    def __init__(self) -> None:
        self.created: dict = {}

    async def get(self, bucket_id: str):
        return None

    async def create(self, **kwargs):
        self.created = kwargs
        return kwargs["bucket_id"]


async def main() -> None:
    tools = {tool.name: tool for tool in await server.mcp.list_tools()}
    write_schema = tools["write_scene"].inputSchema
    close_schema = tools["close_window"].inputSchema
    close_properties = close_schema["properties"]
    write_description = str(tools["write_scene"].description or "")
    edit_description = str(tools["edit_scene"].description or "")
    close_description = str(tools["close_window"].description or "")

    assert "cues" in write_schema["required"]
    assert "用你的第一人称写" in write_description
    assert "保留你的第一人称" in edit_description
    assert "用你的第一人称写" in close_description
    scene_voice_note = "[仅你可见]这是过去的你写下的经历，相关才提及，不要复述。"
    assert scene_voice_note in GatewayService._memory_reading_policy_context()
    assert scene_voice_note in GatewayService._hook_recall_how_to_apply()
    assert "date" in close_schema["required"]
    assert "source" not in close_properties
    assert "scenes" not in close_properties
    assert "session_id" not in close_properties
    assert "profile_id" not in close_properties
    assert "CloseWindowSceneInput" not in close_schema.get("$defs", {})
    assert server._authored_scene_cues("") == []
    assert server._authored_scene_cues("提到 authored cues") == ["提到 authored cues"]

    invalid_date = await server.close_window(
        "## 窗影\n我不会让没有日期的窗影落库。",
        date="",
    )
    assert invalid_date["reason"] == "invalid_date"
    assert invalid_date["rejected_draft_saved"] is False

    server.decay_engine = _NoopDecay()
    rejected = await server._write_scene_memory(
        '标题与引句“都不该被拿来补 cue”。',
        title="描述性标题",
        cues="",
    )
    assert "必须由当前作者亲自写至少一个 cues" in rejected

    shadow = "## 这一窗之后，什么留在了我身上\n我记住了这条作者契约。"
    records, error = server._window_shadow_scene_records(
        (
            f"{shadow}\n\n"
            "## 想留下的记忆\n"
            "### scene | 作者权不由系统代写 | cue：提到 close_window | cue：问内联 Scene 的 cues\n"
            "内联 heading 只携带抽取边界、作者标题和亲自写下的召回入口。"
        )
    )
    assert error == ""
    assert records[0]["title"] == "作者权不由系统代写"
    assert records[0]["cues"] == ["提到 close_window", "问内联 Scene 的 cues"]
    assert records[0]["content"] == "内联 heading 只携带抽取边界、作者标题和亲自写下的召回入口。"
    assert "标题：" not in records[0]["content"]
    assert "cue：" not in records[0]["content"]

    capture = _CaptureBucketManager()
    server.bucket_mgr = capture
    bucket_id, action = await server._write_window_shadow_scene(
        {
            "window_id": "window_authored_marker",
            "session_id": "conversation_authored_marker",
            "source_date": "2026-07-24",
        },
        records[0],
        1,
    )
    assert action == "created"
    assert bucket_id == "window_authored_marker_scene_1"
    assert capture.created["name"] == "作者权不由系统代写"
    assert capture.created["content"] == records[0]["content"]
    assert capture.created["extra_metadata"]["scene_cues"] == records[0]["cues"]
    assert capture.created["extra_metadata"]["write_contract"] == "close-window-scene-v3"

    legacy_labeled_title, error = server._window_shadow_scene_records(
        (
            f"{shadow}\n\n"
            "## 想留下的记忆\n"
            "### scene | 标题：旧标题标签仍可读取 | cue：提到旧窗影格式\n"
            "旧格式只保留兼容，不再作为推荐写法。"
        )
    )
    assert error == ""
    assert legacy_labeled_title[0]["title"] == "旧标题标签仍可读取"

    old_implicit_marker, error = server._window_shadow_scene_records(
        (
            f"{shadow}\n\n"
            "## 想留下的记忆\n"
            "### scene | 提到旧格式 | 问为什么不能继续隐式猜\n"
            "旧格式没有明确区分标题与 cues。"
        )
    )
    assert old_implicit_marker == []
    assert "cue：…" in error

    title_without_cue, error = server._window_shadow_scene_records(
        (
            f"{shadow}\n\n"
            "## 想留下的记忆\n"
            "### scene | 只有标题仍然不够\n"
            "这条 Scene 没有 authored cue。"
        )
    )
    assert title_without_cue == []
    assert "缺少至少一个当前作者写的 `cue：…`" in error

    records, error = server._window_shadow_scene_records(
        (
            f"{shadow}\n\n"
            "## 想留下的记忆\n"
            "### scene | 第一条的名字 | cue：提到第一条 Scene | cue：问第一条 cues\n"
            "第一条 Scene 原文。\n\n"
            "### scene | 第二条的名字 | cue：提到第二条 Scene | cue：问第二条 cues\n"
            "第二条 Scene 原文。\n\n"
            "## 还需要关心的事\n"
            "我还要继续验证其他客户端的显式 handoff。\n\n"
            "## 最近发生的事\n"
            "- 2026-07-24｜我们确认每条内联 Scene 的标题与 cues 都由作者亲自写。"
        )
    )
    assert error == ""
    assert len(records) == 2
    assert records[1]["title"] == "第二条的名字"
    assert records[1]["content"] == "第二条 Scene 原文。"
    assert records[1]["cues"] == ["提到第二条 Scene", "问第二条 cues"]

    print("authored inline Scene titles and cues verified")


if __name__ == "__main__":
    asyncio.run(main())
