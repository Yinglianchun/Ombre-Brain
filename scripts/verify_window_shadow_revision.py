"""Verify append-only Window Shadow revision and active-head handoff behavior."""

from __future__ import annotations

import asyncio
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import server
from window_shadows import WindowShadowStore, validate_window_shadow


ORIGINAL = """
# Window Shadow

## 这一窗之后，什么留在了我身上
我记住了第一版窗影。

## 还在想的事
- 我还要把修订边界验证清楚。

## 给下个窗口的我
醒来先读第一版。

## 想留下的记忆
### scene | 修订不改 Scene | cue：提到窗影修订
我和你确认，窗影修订不能偷偷改掉已经落库的 Scene。
""".strip()

REVISED = ORIGINAL.replace("第一版窗影", "修订后的窗影").replace(
    "醒来先读第一版。",
    "醒来先读修订后的版本。",
)


def verify_schema_migration() -> None:
    with tempfile.TemporaryDirectory(prefix="ombre-shadow-revision-migration-") as tmp:
        state_dir = Path(tmp) / "state"
        state_dir.mkdir(parents=True)
        db_path = state_dir / "window_shadows.sqlite"
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE window_shadows (
                window_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL DEFAULT '',
                profile_id TEXT NOT NULL DEFAULT '',
                parent_shadow_id TEXT NOT NULL DEFAULT '',
                idempotency_key TEXT NOT NULL DEFAULT '',
                source_date TEXT NOT NULL DEFAULT '',
                version TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                content TEXT NOT NULL,
                sections_json TEXT NOT NULL DEFAULT '{}',
                moment_bucket_ids_json TEXT NOT NULL DEFAULT '[]',
                continue_scene_id TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO window_shadows(
                window_id, version, source_hash, content, created_at, updated_at
            ) VALUES ('window_legacy', 'window-shadow-v4', 'legacy-hash',
                      '## 窗影\n我保留旧窗影。', '2026-07-01T00:00:00Z', '2026-07-01T00:00:00Z')
            """
        )
        conn.commit()
        conn.close()
        store = WindowShadowStore(
            {"state_dir": str(state_dir), "buckets_dir": str(Path(tmp) / "buckets")}
        )
        migrated = store.get("window_legacy")
        assert migrated["revision_root_id"] == "window_legacy"
        assert migrated["revision_number"] == 1
        assert migrated["supersedes_window_id"] == ""


async def main() -> None:
    verify_schema_migration()
    tools = {tool.name: tool for tool in await server.mcp.list_tools()}
    assert "revise_window_shadow" in tools
    schema = tools["revise_window_shadow"].inputSchema
    assert set(schema["required"]) == {
        "window_id",
        "shadow",
        "expected_source_hash",
        "idempotency_key",
    }

    with tempfile.TemporaryDirectory(prefix="ombre-shadow-revision-") as tmp:
        store = WindowShadowStore(
            {
                "state_dir": str(Path(tmp) / "state"),
                "buckets_dir": str(Path(tmp) / "buckets"),
                "identity": {"user_display_name": "用户"},
            }
        )
        original_sections, errors = validate_window_shadow(ORIGINAL)
        assert errors == []
        original, created = store.write(
            ORIGINAL,
            session_id="finished-window",
            source_date="2026-08-01",
            sections=original_sections,
        )
        assert created is True
        original = store.attach_scene_buckets(
            original["window_id"],
            ["scene_existing"],
            continue_scene_id="scene_existing",
        )
        server.window_shadow_store = store

        wrong_hash = await server.revise_window_shadow(
            original["window_id"],
            REVISED,
            "0" * 64,
            "revise:test-window",
        )
        assert wrong_hash["reason"] == "source_hash_mismatch"
        assert store.stats()["count"] == 1

        revised = await server.revise_window_shadow(
            original["window_id"],
            REVISED,
            original["source_hash"],
            "revise:test-window",
        )
        assert revised["status"] == "revised"
        assert revised["supersedes_window_id"] == original["window_id"]
        assert revised["revision_root_id"] == original["window_id"]
        assert revised["revision_number"] == 2
        assert revised["scene_bucket_ids"] == ["scene_existing"]
        assert revised["continue_scene_id"] == "scene_existing"
        assert store.get(original["window_id"])["content"] == ORIGINAL
        assert store.get(revised["window_id"])["content"] == REVISED
        assert store.latest()["window_id"] == revised["window_id"]
        assert store.handoff_projection(original["window_id"])["window_id"] == revised["window_id"]
        assert store.portrait_materials()[0]["window_id"] == revised["window_id"]

        replay = await server.revise_window_shadow(
            original["window_id"],
            REVISED,
            original["source_hash"],
            "revise:test-window",
        )
        assert replay["status"] == "existing"
        assert replay["window_id"] == revised["window_id"]
        assert store.stats()["count"] == 2

        changed_scene = REVISED.replace(
            "窗影修订不能偷偷改掉已经落库的 Scene",
            "这句话试图改写已经落库的 Scene",
        )
        scene_rewrite = await server.revise_window_shadow(
            revised["window_id"],
            changed_scene,
            revised["source_hash"],
            "revise:changed-scene",
        )
        assert scene_rewrite["reason"] == "scene_layer_changed"

        third_text = REVISED.replace("醒来先读修订后的版本。", "醒来先读第三版。")
        third = await server.revise_window_shadow(
            revised["window_id"],
            third_text,
            revised["source_hash"],
            "revise:third-version",
        )
        assert third["status"] == "revised"
        assert third["revision_root_id"] == original["window_id"]
        assert third["revision_number"] == 3
        assert store.latest()["window_id"] == third["window_id"]
        old_read = await server.read_memory(
            memory_id=original["window_id"],
            memory_type="shadow",
        )
        assert old_read["window"]["content"] == ORIGINAL
        assert old_read["revision_head_id"] == third["window_id"]
        assert old_read["is_revision_head"] is False

        old_again = await server.revise_window_shadow(
            original["window_id"],
            REVISED.replace("修订后的版本", "另一版"),
            original["source_hash"],
            "revise:old-again",
        )
        assert old_again["reason"] == "window_already_superseded"

        newer_text = "## 窗影\n我是一扇后来正常关闭的新窗。"
        newer_sections, errors = validate_window_shadow(newer_text)
        assert errors == []
        newer, created = store.write(
            newer_text,
            session_id="newer-window",
            source_date="2026-08-02",
            sections=newer_sections,
        )
        assert created is True
        not_latest = await server.revise_window_shadow(
            third["window_id"],
            third_text.replace("第三版", "第四版"),
            third["source_hash"],
            "revise:not-latest",
        )
        assert not_latest["reason"] == "revision_target_not_latest"
        assert store.latest()["window_id"] == newer["window_id"]

    print("append-only Window Shadow revision verified")


if __name__ == "__main__":
    asyncio.run(main())
