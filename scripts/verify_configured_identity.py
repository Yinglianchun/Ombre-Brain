"""Verify public tool copy stays concise and runtime identity is config-driven."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server
from identity import identity_names
from mcp_surface import DAILY_TOOL_NAMES
from narrative_rolls import NarrativeRollStore


async def main() -> None:
    tools = {tool.name: tool for tool in await server.mcp.list_tools()}
    assert set(tools) == set(DAILY_TOOL_NAMES)
    assert "find_arc" in tools
    assert "narrative_revision_inbox" in tools
    assert "review_narrative_revision" in tools

    forbidden_copy = (
        "Haven",
        "小雨",
        "优先读取",
        "兼容",
        "fallback",
        "回退",
        "Bridge 默认",
        "自动塞进",
    )
    retired_tool_copy = (
        "recall(",
        "write_memory",
        "edit_memory",
        "read_portrait",
    )
    for name, tool in tools.items():
        description = str(tool.description or "")
        assert description, name
        assert len(description) <= 420, (name, len(description))
        for fragment in forbidden_copy:
            assert fragment not in description, (name, fragment)
        for fragment in retired_tool_copy:
            assert fragment not in description, (name, fragment)

    root = Path(__file__).resolve().parents[1]
    for relative_path in (
        "CLAUDE_PROMPT.md",
        "docs/Tool Guide.md",
        "docs/memory-layer-contract.md",
    ):
        copy = (root / relative_path).read_text(encoding="utf-8")
        for fragment in ("Haven", "小雨"):
            assert fragment not in copy, (relative_path, fragment)

    readme = (root / "README.md").read_text(encoding="utf-8")
    for fragment in ("关于小雨", "以 Haven 身份", "当前 Haven"):
        assert fragment not in readme, ("README.md", fragment)

    gateway = (root / "gateway.py").read_text(encoding="utf-8")
    assert "Recent Haven Bridge" not in gateway

    read_schema = tools["read_memory"].inputSchema
    assert read_schema["properties"]["memory_type"]["enum"] == [
        "scene",
        "shadow",
        "narrative",
        "fact",
        "event",
    ]
    assert set(read_schema["required"]) == {"memory_type", "memory_id"}
    recall_schema = tools["recall_memory"].inputSchema
    assert {"query", "scene_id", "date", "include_related"} <= set(
        recall_schema["properties"]
    )
    for retired_name in (
        "recall",
        "write_memory",
        "edit_memory",
        "read_portrait",
        "publish_portrait",
    ):
        assert retired_name not in tools

    custom_config = {
        "identity": {
            "ai_name": "Ombre",
            "user_name": "Rain",
            "user_display_name": "雨天",
            "user_aliases": ["她"],
        }
    }
    names = identity_names(custom_config)
    assert names["ai_name"] == "Ombre"
    assert names["user_display_name"] == "雨天"

    with tempfile.TemporaryDirectory(prefix="ombre-configured-identity-") as temp_dir:
        store = NarrativeRollStore(
            {
                **custom_config,
                "state_dir": temp_dir,
            }
        )
        document = (
            "# 配置身份验证卷\n\n"
            "## 第一人称叙事\n\n"
            "我只验证发布者来自 identity.ai_name。\n\n"
            "## 来源账\n\n"
            "- scene_mig2_identity_a\n"
            "- scene_mig2_identity_b\n"
        )
        result = store.publish(
            narrative_id="narrative_configured_identity",
            document=document,
            expected_revision=0,
            title="配置身份验证卷",
            source_scene_ids=[
                "scene_mig2_identity_a",
                "scene_mig2_identity_b",
            ],
        )
        assert result["status"] == "created"
        registry = json.loads(store.registry_path.read_text(encoding="utf-8"))
        assert registry["rolls"][0]["published_by"] == "Ombre_manual"

    dashboard = (root / "dashboard.html").read_text(encoding="utf-8")
    assert "dashboardAiAuthor = 'Haven'" not in dashboard
    assert "dashboardCommentAuthor = 'Rain'" not in dashboard
    assert "当前 Haven" not in dashboard

    print("configured identity and concise public tool copy verified")


if __name__ == "__main__":
    asyncio.run(main())
