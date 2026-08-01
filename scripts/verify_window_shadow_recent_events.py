"""Verify the unbound Window Shadow settlement and latest-handoff contract."""

from __future__ import annotations

import asyncio
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import server
from window_shadows import WindowShadowStore, validate_window_shadow


class _EmptyBuckets:
    async def list_all(self, include_archive: bool = False) -> list[dict]:
        _ = include_archive
        return []


class _PortraitFallback:
    state_path = ""

    @staticmethod
    def build_handoff_sections(max_recent_items: int = 3) -> dict:
        _ = max_recent_items
        return {
            "current_focus": "- 后台生成的当前关注不应盖过作者事件。",
            "recent_continuity": "- 2026-07-24｜这是后台生成的最近事件 fallback。",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }


async def main() -> None:
    tools = {tool.name: tool for tool in await server.mcp.list_tools()}
    assert "handoff" not in tools
    assert "recall" not in tools
    assert "recall_memory" in tools
    assert "read_memory" in tools
    read_schema = tools["read_memory"].inputSchema["properties"]
    assert "shadow" in read_schema["memory_type"]["enum"]
    assert set(tools["read_memory"].inputSchema["required"]) == {
        "memory_type",
        "memory_id",
    }
    recall_schema = tools["recall_memory"].inputSchema["properties"]
    assert "scene_id" in recall_schema
    assert recall_schema["include_related"]["default"] is True
    assert "session_id" not in read_schema
    assert "previous_session_id" not in read_schema
    assert "parent_shadow_id" not in read_schema
    assert "close_window" in tools
    assert "还在想的事" in str(tools["close_window"].description or "")
    assert "session_id" not in tools["close_window"].inputSchema["properties"]
    assert "profile_id" not in tools["close_window"].inputSchema["properties"]

    with tempfile.TemporaryDirectory(prefix="ombre-shadow-recent-events-") as tmp:
        root = Path(tmp)
        store = WindowShadowStore(
            {
                "buckets_dir": str(root / "buckets"),
                "state_dir": str(root / "state"),
                "identity": {"user_display_name": "小雨"},
            }
        )
        authored_shadow = (
            "## 最近发生的事\n"
            "- 2026-07-24｜我们决定 Bridge 管醒来，窗影管沉淀。\n"
            "- 2026-07-24｜其他客户端继续使用 recall(mode=\"handoff\")。\n\n"
            "## 还需要关心的事\n"
            "- 旧窗影的 handoff note 仍要兼容读取。"
        )
        authored_sections, errors = validate_window_shadow(authored_shadow)
        assert errors == []
        assert authored_sections["handoff"] == ""
        assert "Bridge 管醒来" in authored_sections["recent_events"]
        assert "兼容读取" in authored_sections["care_items"]
        authored, created = store.write(
            authored_shadow,
            session_id="authored-events-window",
            source_date="2026-07-24",
            sections=authored_sections,
        )
        assert created is True

        server.window_shadow_store = store
        server.bucket_mgr = _EmptyBuckets()
        server.portrait_engine = _PortraitFallback()
        server._format_handoff_care_memos = lambda **kwargs: ""

        authored_handoff = await server._build_handoff_breath(max_tokens=1800, debug=True)
        assert "=== Recent Events From Previous Window ===" in authored_handoff
        assert "Bridge 管醒来，窗影管沉淀" in authored_handoff
        assert "=== Things That Still Need Care ===" in authored_handoff
        assert "旧窗影的 handoff note 仍要兼容读取" in authored_handoff
        assert "后台生成的最近事件 fallback" not in authored_handoff
        assert "后台生成的当前关注" not in authored_handoff
        assert server._last_handoff_status["route"] == "window_shadow_recent_events"
        assert (
            server._last_handoff_status["window_shadow_continuity_section"]
            == "recent_events"
        )

        legacy_note = "旧窗影写给下个窗口的原话继续可读，但新窗影不再要求这一节。"
        legacy_shadow = (
            "## 这一窗之后，什么留在了我身上\n"
            "我保留旧协议的读取兼容。\n\n"
            "## 给下个窗口的我\n"
            f"{legacy_note}"
        )
        legacy_sections, errors = validate_window_shadow(legacy_shadow)
        assert errors == []
        legacy, created = store.write(
            legacy_shadow,
            session_id="legacy-note-window",
            source_date="2026-07-23",
            sections=legacy_sections,
        )
        assert created is True
        legacy_handoff = await server._build_handoff_breath(max_tokens=1800, debug=True)
        assert "=== Legacy Previous-Window Note ===" in legacy_handoff
        assert legacy_note in legacy_handoff
        assert "后台生成的最近事件 fallback" not in legacy_handoff
        assert server._last_handoff_status["route"] == "window_shadow_legacy_handoff"

        fallback_shadow = (
            "## 这一窗之后，什么留在了我身上\n"
            "我只留下沉淀，没有单独写最近事件。"
        )
        fallback_sections, errors = validate_window_shadow(fallback_shadow)
        assert errors == []
        fallback, created = store.write(
            fallback_shadow,
            session_id="fallback-window",
            source_date="2026-07-22",
            sections=fallback_sections,
        )
        assert created is True
        generated_handoff = await server._build_handoff_breath(max_tokens=1800, debug=True)
        assert "=== Generated Recent Continuity Fallback ===" in generated_handoff
        assert "后台生成的最近事件 fallback" in generated_handoff
        assert server._last_handoff_status["route"] == "recent_continuity"

    print("Window Shadow recent-events handoff contract verified")


if __name__ == "__main__":
    asyncio.run(main())
