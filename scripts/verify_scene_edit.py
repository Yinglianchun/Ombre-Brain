"""Verify safe in-place authored Scene edits and the public MCP schema."""

from __future__ import annotations

import asyncio
import copy
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server
from bucket_manager import BucketManager


class _EditableBucketManager:
    def __init__(self) -> None:
        self.bucket = {
            "id": "scene_edit_contract",
            "content": "我和小雨保留下来的原 Scene 正文。",
            "metadata": {
                "id": "scene_edit_contract",
                "name": "旧标题",
                "type": "dynamic",
                "object_kind": "scene",
                "memory_value_source": "authored_scene",
                "write_contract": "close-window-scene-v3",
                "scene_cues": ["提到旧标题"],
                "source": "close_window",
                "window_shadow_id": "window_edit_contract",
                "window_shadow_session_id": "session_edit_contract",
                "window_shadow_index": 1,
                "scene_source_hash": server.WindowShadowStore.source_hash(
                    "我和小雨保留下来的原 Scene 正文。"
                ),
                "created": "2026-07-24T10:00:00+08:00",
                "updated_at": datetime.fromisoformat("2026-07-24T10:00:00+08:00"),
                "last_active": "2026-07-24T10:00:00+08:00",
            },
        }
        self.update_count = 0

    async def get(self, bucket_id: str):
        if bucket_id != self.bucket["id"]:
            return None
        return copy.deepcopy(self.bucket)

    async def update(self, bucket_id: str, **kwargs) -> bool:
        if bucket_id != self.bucket["id"]:
            return False
        self.update_count += 1
        if "name" in kwargs:
            self.bucket["metadata"]["name"] = kwargs["name"]
        if "content" in kwargs:
            self.bucket["content"] = kwargs["content"]
        self.bucket["metadata"].update(kwargs.get("extra_metadata") or {})
        self.bucket["metadata"]["updated_at"] = f"2026-07-24T10:00:0{self.update_count}+08:00"
        self.bucket["metadata"]["last_active"] = self.bucket["metadata"]["updated_at"]
        return True


class _CaptureMoments:
    def __init__(self) -> None:
        self.ids: list[str] = []

    def upsert_bucket(self, bucket: dict):
        self.ids.append(bucket["id"])
        return []


async def main() -> None:
    tools = {tool.name: tool for tool in await server.mcp.list_tools()}
    assert "edit_scene" in tools
    schema = tools["edit_scene"].inputSchema
    assert set(schema["required"]) == {"scene_id", "expected_updated_at"}
    assert {"title", "content", "cues"} <= set(schema["properties"])

    manager = _EditableBucketManager()
    moments = _CaptureMoments()
    server.bucket_mgr = manager
    server.memory_moment_store = moments
    server._queue_embedding_refresh_if_changed = lambda *_args: True
    server._queue_scene_linking = lambda *_args: True
    server._refresh_entity_edges_for_bucket = lambda *_args: 1

    missing_version = await server._edit_scene_memory(
        "scene_edit_contract",
        expected_updated_at="",
        title="新标题",
    )
    assert missing_version["reason"] == "expected_updated_at_required"

    conflict = await server._edit_scene_memory(
        "scene_edit_contract",
        expected_updated_at="2026-07-24T09:59:59+08:00",
        title="新标题",
    )
    assert conflict["status"] == "conflict"
    assert manager.update_count == 0

    title_only = await server._edit_scene_memory(
        "scene_edit_contract",
        expected_updated_at="2026-07-24T10:00:00+08:00",
        title="小雨亲自改的标题",
    )
    assert title_only["status"] == "updated"
    assert title_only["changed_fields"] == ["title"]
    assert title_only["revision"] == 2
    assert title_only["scene_id"] == "scene_edit_contract"
    assert manager.bucket["content"] == "我和小雨保留下来的原 Scene 正文。"
    assert manager.bucket["metadata"]["scene_cues"] == ["提到旧标题"]
    assert manager.bucket["metadata"]["window_shadow_id"] == "window_edit_contract"
    first_history = manager.bucket["metadata"]["scene_revision_history"]
    assert first_history[0]["title"] == "旧标题"
    assert first_history[0]["content"] == "我和小雨保留下来的原 Scene 正文。"

    revised = await server._edit_scene_memory(
        "scene_edit_contract",
        expected_updated_at=title_only["updated_at"],
        content="我用第一人称重新校正了这条 Scene，但仍保留原版本证据。",
        cues=["提到第一人称修订", "问 Scene 原位编辑"],
    )
    assert revised["status"] == "updated"
    assert revised["changed_fields"] == ["content", "cues"]
    assert revised["revision"] == 3
    assert manager.bucket["metadata"]["scene_cues"] == [
        "提到第一人称修订",
        "问 Scene 原位编辑",
    ]
    assert manager.bucket["metadata"]["scene_cues_reviewed_at"]
    assert len(manager.bucket["metadata"]["scene_revision_history"]) == 2
    assert manager.bucket["metadata"]["scene_revision_history"][1]["title"] == "小雨亲自改的标题"
    assert moments.ids == []
    assert server._window_shadow_scene_source_valid(manager.bucket) is True

    stale_retry = await server._edit_scene_memory(
        "scene_edit_contract",
        expected_updated_at=title_only["updated_at"],
        title="不应覆盖新版本",
    )
    assert stale_retry["status"] == "conflict"
    assert manager.bucket["metadata"]["name"] == "小雨亲自改的标题"

    reviewed_unchanged_cues = copy.deepcopy(manager.bucket)
    reviewed_unchanged_cues["metadata"].pop("scene_cues_reviewed_at")
    manager.bucket = reviewed_unchanged_cues
    cue_review = await server._edit_scene_memory(
        "scene_edit_contract",
        expected_updated_at=revised["updated_at"],
        cues=["提到第一人称修订", "问 Scene 原位编辑"],
    )
    assert cue_review["status"] == "updated"
    assert cue_review["changed_fields"] == ["cues_review"]
    assert manager.bucket["metadata"]["scene_cues_reviewed_at"]

    blank_title = await server._edit_scene_memory(
        "scene_edit_contract",
        expected_updated_at=cue_review["updated_at"],
        title="",
    )
    assert blank_title["reason"] == "scene_title_required"

    with tempfile.TemporaryDirectory(prefix="ombre-scene-edit-") as temp_dir:
        real_manager = BucketManager({"buckets_dir": temp_dir})
        original = "真实文件往返也必须保留这条 revision 1 正文。"
        await real_manager.create(
            bucket_id="scene_file_roundtrip",
            name="文件往返旧标题",
            content=original,
            source="window_shadow",
            extra_metadata={
                "object_kind": "scene",
                "memory_value_source": "authored_scene",
                "write_contract": "close-window-scene-v3",
                "scene_cues": ["提到文件往返"],
                "window_shadow_id": "window_file_roundtrip",
                "window_shadow_session_id": "session_file_roundtrip",
                "window_shadow_index": 1,
                "scene_source_hash": server.WindowShadowStore.source_hash(original),
            },
        )
        server.bucket_mgr = real_manager
        loaded = await real_manager.get("scene_file_roundtrip")
        loaded_updated_at = loaded["metadata"]["updated_at"]
        version_token = (
            loaded_updated_at.isoformat()
            if isinstance(loaded_updated_at, datetime)
            else str(loaded_updated_at)
        )
        roundtrip = await server._edit_scene_memory(
            "scene_file_roundtrip",
            expected_updated_at=version_token,
            content="真实文件往返后的第一人称修订正文。",
        )
        assert roundtrip["status"] == "updated"
        reloaded = await real_manager.get("scene_file_roundtrip")
        assert reloaded["metadata"]["window_shadow_id"] == "window_file_roundtrip"
        assert reloaded["metadata"]["scene_revision_history"][0]["content"] == original
        assert server._window_shadow_scene_source_valid(reloaded) is True

    print("authored Scene in-place edit contract verified")


if __name__ == "__main__":
    asyncio.run(main())
