"""Verify authored Scene archive/restore semantics and the public MCP schema."""

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


class _IndexRecorder:
    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.upserted: list[str] = []

    def delete_bucket(self, bucket_id: str):
        self.deleted.append(bucket_id)
        return {"deleted": 1}

    def upsert_bucket(self, bucket: dict):
        self.upserted.append(bucket["id"])
        return [{"bucket_id": bucket["id"]}]


class _EmbeddingRecorder:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def delete_embedding(self, bucket_id: str) -> None:
        self.deleted.append(bucket_id)


class _EntityRecorder:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def delete_for_bucket(self, bucket_id: str):
        self.deleted.append(bucket_id)
        return 1


class _NodeRecorder:
    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.upserted: list[str] = []

    def delete(self, bucket_id: str) -> bool:
        self.deleted.append(bucket_id)
        return True

    def upsert_bucket(self, bucket: dict):
        self.upserted.append(bucket["id"])
        return {"bucket_id": bucket["id"]}


def _token(bucket: dict) -> str:
    value = bucket["metadata"]["updated_at"]
    return value.isoformat() if isinstance(value, datetime) else str(value)


async def main() -> None:
    tools = {tool.name: tool for tool in await server.mcp.list_tools()}
    assert len(tools) == 18
    assert "narrative_revision_inbox" in tools
    assert "review_narrative_revision" in tools
    assert "set_scene_status" in tools
    schema = tools["set_scene_status"].inputSchema
    assert set(schema["required"]) == {"scene_id", "status", "expected_updated_at"}
    assert schema["properties"]["status"]["enum"] == ["active", "archived"]

    with tempfile.TemporaryDirectory(prefix="ombre-scene-status-") as temp_dir:
        manager = BucketManager({"buckets_dir": temp_dir})
        original_content = "这条 Scene 的正文、来源和年轮在归档后都必须原样留下。"
        original_history = [
            {
                "revision": 1,
                "content": "第一版正文。",
                "saved_at": "2026-07-24T10:00:00+08:00",
            }
        ]
        original_comments = [
            {
                "id": "comment_scene_status",
                "content": "这条年轮也不能消失。",
                "created_at": "2026-07-24T11:00:00+08:00",
            }
        ]
        await manager.create(
            bucket_id="scene_status_contract",
            name="归档契约",
            content=original_content,
            source="close_window",
            extra_metadata={
                "object_kind": "scene",
                "memory_value_source": "authored_scene",
                "write_contract": "close-window-scene-v3",
                "scene_cues": ["提到归档契约"],
                "scene_revision": 2,
                "scene_revision_history": copy.deepcopy(original_history),
                "comments": copy.deepcopy(original_comments),
                "comment_count": 1,
                "window_shadow_id": "window_scene_status",
                "window_shadow_session_id": "session_scene_status",
                "scene_source_hash": server.WindowShadowStore.source_hash(original_content),
            },
        )
        await manager.create(
            bucket_id="legacy_status_contract",
            name="旧桶不能冒充 Scene",
            content="旧桶正文。",
            source="legacy",
        )
        await manager.create(
            bucket_id="immutable_status_contract",
            name="不可变来源",
            content="不可变来源正文。",
            source="import",
            extra_metadata={
                "object_kind": "scene",
                "memory_value_source": "authored_scene",
                "source_record_immutable": True,
            },
        )

        moments = _IndexRecorder()
        embeddings = _EmbeddingRecorder()
        entities = _EntityRecorder()
        nodes = _NodeRecorder()
        server.bucket_mgr = manager
        server.memory_moment_store = moments
        server.embedding_engine = embeddings
        server.entity_edge_store = entities
        server.memory_node_store = nodes
        server._queue_embedding_refresh = lambda *_args: True
        server._refresh_entity_edges_for_bucket = lambda bucket: 1
        server._queue_scene_linking = lambda *_args: True

        original = await manager.get("scene_status_contract")
        original_updated_at = _token(original)

        conflict = await server._set_scene_status_memory(
            "scene_status_contract",
            status="archived",
            expected_updated_at="2026-07-24T09:59:59+08:00",
        )
        assert conflict["status"] == "conflict"
        assert _scene_in(await manager.list_all(), "scene_status_contract")

        legacy = await manager.get("legacy_status_contract")
        rejected_legacy = await server._set_scene_status_memory(
            "legacy_status_contract",
            status="archived",
            expected_updated_at=_token(legacy),
        )
        assert rejected_legacy["reason"] == "not_authored_scene"

        immutable = await manager.get("immutable_status_contract")
        rejected_immutable = await server._set_scene_status_memory(
            "immutable_status_contract",
            status="archived",
            expected_updated_at=_token(immutable),
        )
        assert rejected_immutable["reason"] == "immutable_source_record"

        archived = await server._set_scene_status_memory(
            "scene_status_contract",
            status="archived",
            expected_updated_at=original_updated_at,
        )
        assert archived["status"] == "updated"
        assert archived["scene_status"] == "archived"
        assert archived["ordinary_recall"] is False
        assert archived["exact_read"] is True
        assert not _scene_in(await manager.list_all(), "scene_status_contract")
        assert _scene_in(
            await manager.list_all(include_archive=True),
            "scene_status_contract",
        )
        exact = await server._read_scene_memory("scene_status_contract")
        assert exact["content"] == original_content
        assert exact["metadata"]["type"] == "archived"
        assert exact["metadata"]["active"] is False
        assert exact["metadata"]["scene_status"] == "archived"
        assert exact["metadata"]["scene_revision_history"] == original_history
        assert exact["metadata"]["comments"] == original_comments
        assert exact["metadata"]["window_shadow_id"] == "window_scene_status"
        assert moments.deleted == ["scene_status_contract"]
        assert embeddings.deleted == ["scene_status_contract"]
        assert entities.deleted == ["scene_status_contract"]
        assert nodes.deleted == ["scene_status_contract"]

        unchanged = await server._set_scene_status_memory(
            "scene_status_contract",
            status="archived",
            expected_updated_at=archived["updated_at"],
        )
        assert unchanged["status"] == "unchanged"
        assert len(unchanged["scene"]["metadata"]["scene_status_history"]) == 1

        restored = await server._set_scene_status_memory(
            "scene_status_contract",
            status="active",
            expected_updated_at=archived["updated_at"],
        )
        assert restored["status"] == "updated"
        assert restored["scene_status"] == "active"
        assert restored["ordinary_recall"] is True
        assert _scene_in(await manager.list_all(), "scene_status_contract")
        reloaded = await manager.get("scene_status_contract")
        assert reloaded["content"] == original_content
        assert reloaded["metadata"]["type"] == "dynamic"
        assert reloaded["metadata"]["active"] is True
        assert reloaded["metadata"]["scene_revision_history"] == original_history
        assert reloaded["metadata"]["comments"] == original_comments
        assert reloaded["metadata"]["window_shadow_id"] == "window_scene_status"
        assert len(reloaded["metadata"]["scene_status_history"]) == 2
        assert moments.upserted == ["scene_status_contract"]
        assert nodes.upserted == ["scene_status_contract"]

    print("authored Scene archive/restore contract verified")


def _scene_in(buckets: list[dict], scene_id: str) -> bool:
    return any(str(bucket.get("id") or "") == scene_id for bucket in buckets)


if __name__ == "__main__":
    asyncio.run(main())
