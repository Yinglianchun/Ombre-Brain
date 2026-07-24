"""Exercise rejected close_window draft recovery without touching live memory data."""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import server
from window_shadows import WindowShadowRejectedDraftStore, WindowShadowStore


class _NoopDecay:
    async def ensure_started(self) -> None:
        return None


class _GatewayState:
    def resolve_window_conversation(self, **kwargs) -> dict:
        raise AssertionError("close_window must not resolve a Gateway window")

    def close_window_session(self, **kwargs) -> dict:
        raise AssertionError("close_window must not close a Gateway window")


def _shadow(self_delta: str, *, inline_scene: bool = False) -> str:
    text = (
        "## 这一窗之后，什么留在了我身上\n"
        f"{self_delta}\n\n"
        "## 最近发生的事\n"
        "- 2026-07-24｜失败稿只允许沿同一份原稿修正具体报错。\n\n"
        "## 还需要关心的事\n"
        "- 成功前不能让失败稿进入 canonical Shadow 或 handoff。"
    )
    if inline_scene:
        text += (
            "\n\n## 想留下的记忆\n"
            "### scene | 标题：失败稿仍由作者局部修复 | cue：提到 close_window 失败稿 | cue：问为什么不能整篇重写\n"
            "第一次校验失败后，原稿仍应逐字留给同一次关窗重试。"
        )
    return text


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="ombre-rejected-shadow-draft-") as tmp:
        root = Path(tmp)
        test_config = {
            "buckets_dir": str(root / "buckets"),
            "state_dir": str(root / "state"),
            "identity": {"user_display_name": "小雨"},
        }
        canonical_store = WindowShadowStore(test_config)
        draft_store = WindowShadowRejectedDraftStore(test_config)
        assert canonical_store.db_path != draft_store.db_path

        server.window_shadow_store = canonical_store
        server.window_shadow_rejected_draft_store = draft_store
        server.gateway_state_store = _GatewayState()
        server.decay_engine = _NoopDecay()
        server._queue_embedding_refresh = lambda bucket_id: None
        server._queue_scene_linking = lambda bucket_id: None

        async def _fake_write_scene(window: dict, scene: dict, index: int):
            return f"scene-rejected-draft-{index}", "created"

        server._write_window_shadow_scene = _fake_write_scene

        tools = {tool.name: tool for tool in await server.mcp.list_tools()}
        close_schema = tools["close_window"].inputSchema["properties"]
        assert "rejected_draft_source_hash" in close_schema
        assert "read_rejected_draft" in close_schema
        assert "rejected_draft_section_patch" in close_schema
        assert "scenes" not in close_schema
        assert "session_id" not in close_schema
        assert "profile_id" not in close_schema

        request_key = "bridge:rejected-draft:text-fix"
        first_shadow = _shadow("失败不能授权下一次重写整篇窗影。")
        first = await server._close_window_commit(
            first_shadow,
            idempotency_key=request_key,
        )
        assert first["status"] == "invalid"
        assert first["reason"] == "invalid_window_shadow"
        assert first["rejected_draft_saved"] is True
        assert first["rejected_draft"]["shadow"] == first_shadow
        assert first["rejected_draft"]["canonical"] is False
        assert first["rejected_draft"]["ordinary_recall"] is False
        assert first["rejected_draft"]["handoff_visible"] is False
        assert first["rejected_draft"]["validation"]["fix_scope"] == [
            "shadow.self"
        ]
        assert canonical_store.stats()["count"] == 0
        assert draft_store.stats()["count"] == 1

        rewritten = await server._close_window_commit(
            _shadow("这是一篇重新生成的长窗影。" * 20),
            idempotency_key=request_key,
        )
        assert rewritten["status"] == "invalid"
        assert rewritten["reason"] == "rejected_draft_reuse_required"
        assert rewritten["rejected_draft"]["shadow"] == first_shadow
        assert draft_store.get(request_key)["shadow"] == first_shadow

        whole_rewrite = _shadow("我拿着旧稿 hash，只修 self 段。" * 20).replace(
            "失败稿只允许沿同一份原稿修正具体报错。",
            "我把 recent_events 也偷偷改掉了。",
        )
        rewritten_with_hash = await server._close_window_commit(
            whole_rewrite,
            idempotency_key=request_key,
            rejected_draft_source_hash=first["rejected_draft"]["source_hash"],
        )
        assert rewritten_with_hash["status"] == "invalid"
        assert (
            rewritten_with_hash["reason"]
            == "rejected_draft_outside_fix_scope_changed"
        )
        assert rewritten_with_hash["rejected_draft"]["shadow"] == first_shadow
        assert draft_store.get(request_key)["shadow"] == first_shadow

        recovered = await server._close_window_commit(
            "",
            idempotency_key=request_key,
            read_rejected_draft=True,
        )
        assert recovered["status"] == "rejected"
        assert recovered["reason"] == "rejected_draft_available"
        assert recovered["rejected_draft"]["shadow"] == first_shadow

        corrected_shadow = _shadow("我会沿着原稿继续，只修这一处。" * 20)
        corrected = await server._close_window_commit(
            corrected_shadow,
            idempotency_key=request_key,
            rejected_draft_source_hash=first["rejected_draft"]["source_hash"],
        )
        assert corrected["status"] == "created"
        assert corrected["rejected_draft_cleared"] is True
        assert draft_store.get(request_key) is None
        canonical = canonical_store.get(corrected["window_id"])
        assert canonical["content"] == corrected_shadow

        replay = await server._close_window_commit(
            "这份文字不应覆盖第一次成功写入。",
            idempotency_key=request_key,
        )
        assert replay["status"] == "existing"
        assert replay["idempotent_replay"] is True
        assert replay["window_id"] == corrected["window_id"]
        assert canonical_store.get(replay["window_id"])["content"] == corrected_shadow

        parameter_key = "bridge:rejected-draft:parameter-fix"
        inline_shadow = _shadow(
            "我会沿着原稿继续，只修请求参数。" * 20,
            inline_scene=True,
        )
        invalid_index = await server._close_window_commit(
            inline_shadow,
            idempotency_key=parameter_key,
            continue_scene_index=2,
        )
        assert invalid_index["status"] == "invalid"
        assert invalid_index["reason"] == "invalid_continue_scene_index"
        assert invalid_index["rejected_draft"]["shadow"] == inline_shadow
        assert invalid_index["rejected_draft"]["validation"]["fix_scope"] == [
            "request.continue_scene_index"
        ]

        changed_parameter_retry = await server._close_window_commit(
            inline_shadow + "\n这行不该被添加。",
            idempotency_key=parameter_key,
            continue_scene_index=1,
            rejected_draft_source_hash=invalid_index["rejected_draft"]["source_hash"],
        )
        assert changed_parameter_retry["status"] == "invalid"
        assert (
            changed_parameter_retry["reason"]
            == "rejected_draft_outside_fix_scope_changed"
        )
        assert changed_parameter_retry["rejected_draft"]["shadow"] == inline_shadow

        parameter_retry = await server._close_window_commit(
            inline_shadow,
            idempotency_key=parameter_key,
            continue_scene_index=1,
        )
        assert parameter_retry["status"] == "created"
        assert parameter_retry["scene_count"] == 1
        assert parameter_retry["rejected_draft_cleared"] is True
        assert draft_store.get(parameter_key) is None
        assert canonical_store.stats()["count"] == 2
        assert draft_store.stats()["count"] == 0

        section_patch_key = "bridge:rejected-draft:section-patch"
        no_cues_shadow = _shadow(
            "我会沿着原稿继续，只修失败的 Scene 段落。" * 15,
            inline_scene=True,
        ).replace(
            "### scene | 标题：失败稿仍由作者局部修复 | cue：提到 close_window 失败稿 | cue：问为什么不能整篇重写",
            "### scene | 标题：这条 Scene 还缺少 cue",
        )
        no_cues = await server._close_window_commit(
            no_cues_shadow,
            idempotency_key=section_patch_key,
        )
        assert no_cues["status"] == "invalid"
        assert no_cues["reason"] == "invalid_scene"
        assert no_cues["rejected_draft"]["validation"]["fix_scope"] == [
            "shadow.moments",
        ]

        disallowed_patch = await server._close_window_commit(
            "",
            idempotency_key=section_patch_key,
            rejected_draft_source_hash=no_cues["rejected_draft"]["source_hash"],
            rejected_draft_section_patch={"recent_events": "不应允许修改这一段。"},
        )
        assert disallowed_patch["status"] == "invalid"
        assert disallowed_patch["reason"] == "rejected_draft_section_patch_not_allowed"

        section_patched = await server._close_window_commit(
            "",
            idempotency_key=section_patch_key,
            rejected_draft_source_hash=no_cues["rejected_draft"]["source_hash"],
            rejected_draft_section_patch={
                "moments": (
                    "### scene | 标题：失败稿只修 Scene 段落 | cue：提到 close_window 失败稿 | cue：问局部修复如何保留原稿\n"
                    "第一次校验失败后，服务端只替换获准修改的 Scene 段落。"
                )
            },
        )
        assert section_patched["status"] == "created"
        assert section_patched["scene_count"] == 1
        patched_canonical = canonical_store.get(section_patched["window_id"])
        assert "服务端只替换获准修改的 Scene 段落。" in patched_canonical["content"]
        assert (
            "## 这一窗之后，什么留在了我身上\n"
            "我会沿着原稿继续，只修失败的 Scene 段落。"
            in patched_canonical["content"]
        )
        assert draft_store.get(section_patch_key) is None

    print("rejected Window Shadow draft contract verified")


if __name__ == "__main__":
    asyncio.run(main())
