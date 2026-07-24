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
from narrative_rolls import NarrativeRollStore


async def main() -> None:
    tools = {tool.name: tool for tool in await server.mcp.list_tools()}
    assert len(tools) == 15

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
    for name, tool in tools.items():
        description = str(tool.description or "")
        assert description, name
        assert len(description) <= 420, (name, len(description))
        for fragment in forbidden_copy:
            assert fragment not in description, (name, fragment)

    recall_schema = tools["recall"].inputSchema
    assert recall_schema["properties"]["mode"]["enum"] == ["memory", "handoff"]

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

    dashboard = (Path(__file__).resolve().parents[1] / "dashboard.html").read_text(
        encoding="utf-8"
    )
    assert "dashboardAiAuthor = 'Haven'" not in dashboard
    assert "dashboardCommentAuthor = 'Rain'" not in dashboard
    assert "当前 Haven" not in dashboard

    print("configured identity and concise public tool copy verified")


if __name__ == "__main__":
    asyncio.run(main())
