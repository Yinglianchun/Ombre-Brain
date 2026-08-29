from __future__ import annotations

import ast
import hashlib
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diary_store import DiaryStore
from narrative_rolls import NarrativeRollStore
from narrative_source_verification import (
    darkroom_source_marker,
    diary_comments_sha256,
    verify_narrative_darkroom_sources,
    verify_narrative_diary_sources,
)


EVENT_ID = "event_111111111111111111111111"


def _marker(darkroom: dict) -> str:
    return darkroom_source_marker(
        darkroom_id=int(darkroom["id"]),
        revision=int(darkroom["revision"]),
        content_sha256=hashlib.sha256(
            str(darkroom["content"]).encode("utf-8")
        ).hexdigest(),
        comments_sha256=diary_comments_sha256(darkroom.get("comments")),
    )


def _document(*markers: str) -> str:
    source_lines = "\n".join(f"- `{value}`" for value in markers)
    return (
        "# Darkroom source test\n\n"
        "## 第一人称叙事\n\n"
        "我只把已经解锁、且在发布当下可核验的暗房原件作为来源。\n\n"
        "## 材料目录\n\n"
        f"- `{EVENT_ID}`\n{source_lines}\n"
    )


def _get_diary(store: DiaryStore, diary_id: int) -> dict | None:
    result = store.read(diary_id=diary_id, limit=1, include_archived=True)
    if result.get("count") != 1 or int(result.get("id") or 0) != diary_id:
        return None
    return result


def _assert_private_failure(errors: list[dict], *secrets: str) -> None:
    serialized = json.dumps(errors, ensure_ascii=False)
    for secret in secrets:
        assert secret not in serialized, (secret, errors)
    for forbidden_key in (
        "title",
        "content",
        "comments",
        "unlock_at",
        "content_sha256",
        "comments_sha256",
    ):
        assert all(forbidden_key not in error for error in errors), errors


def _verify_server_wiring() -> None:
    tree = ast.parse((ROOT / "server.py").read_text(encoding="utf-8"))
    publish = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "publish_narrative"
    )
    assert "source_darkroom_ids" in {argument.arg for argument in publish.args.args}
    calls = [node for node in ast.walk(publish) if isinstance(node, ast.Call)]
    assert any(
        isinstance(call.func, ast.Name)
        and call.func.id == "verify_narrative_darkroom_sources"
        for call in calls
    )
    store_publish = next(
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "publish"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "narrative_roll_store"
    )
    assert "source_darkroom_ids" in {keyword.arg for keyword in store_publish.keywords}


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="narrative-darkroom-source-") as temp_dir:
        state_dir = Path(temp_dir) / "state"
        diary_store = DiaryStore({"state_dir": str(state_dir)})
        narrative_store = NarrativeRollStore({"state_dir": str(state_dir)})

        darkroom_a = diary_store.create(
            content="暗房正文甲，不能进入 Narrative registry 或错误回执。",
            date="2026-08-17",
            title="暗房标题甲",
            entry_type="darkroom",
        )
        darkroom_b = diary_store.create(
            content="暗房正文乙，也只能用冻结哈希引用。",
            date="2026-08-18",
            title="暗房标题乙",
            entry_type="darkroom",
        )
        locked = diary_store.create(
            content="仍锁着的暗房正文。",
            date="2026-08-19",
            title="仍锁着的暗房",
            entry_type="darkroom",
            unlock_at="2099-01-01T00:00:00+08:00",
        )
        archived = diary_store.create(
            content="归档暗房正文。",
            date="2026-08-20",
            title="归档暗房",
            entry_type="darkroom",
            visibility="archived",
        )
        ordinary_diary = diary_store.create(
            content="普通日记不能借 Darkroom 轴发布。",
            date="2026-08-21",
            title="普通日记",
        )
        deleted = diary_store.create(
            content="删除后的暗房不可见。",
            date="2026-08-22",
            title="删除暗房",
            entry_type="darkroom",
        )
        diary_store.delete(int(deleted["id"]))

        resolved, errors = verify_narrative_darkroom_sources(
            exact_document=_document(_marker(darkroom_a)),
            darkroom_ids=[int(darkroom_a["id"])],
            get_darkroom=lambda diary_id: _get_diary(diary_store, diary_id),
        )
        assert errors == [], errors
        assert resolved == [
            {
                "source_type": "darkroom",
                "darkroom_id": darkroom_a["id"],
                "revision": darkroom_a["revision"],
                "content_sha256": hashlib.sha256(
                    str(darkroom_a["content"]).encode("utf-8")
                ).hexdigest(),
                "comments_sha256": diary_comments_sha256(darkroom_a.get("comments")),
            }
        ], resolved
        assert not ({"title", "content", "comments", "unlock_at"} & resolved[0].keys())

        _, ordinary_diary_errors = verify_narrative_diary_sources(
            exact_document=_document(_marker(darkroom_a)),
            diary_ids=[int(darkroom_a["id"])],
            get_diary=lambda diary_id: _get_diary(diary_store, diary_id),
        )
        assert ordinary_diary_errors[0]["reason"] == "not_ordinary_diary"

        unavailable_ids = [
            int(locked["id"]),
            int(archived["id"]),
            int(ordinary_diary["id"]),
            int(deleted["id"]),
            999999,
        ]
        _, unavailable_errors = verify_narrative_darkroom_sources(
            exact_document=_document("darkroom:999999"),
            darkroom_ids=unavailable_ids,
            get_darkroom=lambda diary_id: _get_diary(diary_store, diary_id),
        )
        assert [error["reason"] for error in unavailable_errors] == [
            "darkroom_source_unavailable"
        ] * len(unavailable_ids), unavailable_errors
        _assert_private_failure(
            unavailable_errors,
            "仍锁着的暗房正文。",
            "归档暗房正文。",
            "普通日记不能借 Darkroom 轴发布。",
            "删除后的暗房不可见。",
        )

        _, missing_lock_state_errors = verify_narrative_darkroom_sources(
            exact_document=_document(),
            darkroom_ids=[707],
            get_darkroom=lambda _darkroom_id: {
                "id": 707,
                "entry_type": "darkroom",
                "visibility": "active",
                "body_available": True,
                "revision": 1,
                "content": "没有明确 unlocked 状态也不可信。",
                "comments": [],
            },
        )
        assert missing_lock_state_errors == [
            {"darkroom_id": 707, "reason": "darkroom_source_unavailable"}
        ]
        _assert_private_failure(missing_lock_state_errors, "没有明确 unlocked 状态也不可信。")

        old_marker = _marker(darkroom_a)
        diary_store.revise(int(darkroom_a["id"]), content="暗房正文甲已经修订。")
        _, content_drift_errors = verify_narrative_darkroom_sources(
            exact_document=_document(old_marker),
            darkroom_ids=[int(darkroom_a["id"])],
            get_darkroom=lambda diary_id: _get_diary(diary_store, diary_id),
        )
        assert content_drift_errors == [
            {
                "darkroom_id": darkroom_a["id"],
                "reason": "darkroom_snapshot_marker_missing_from_document",
            }
        ]
        _assert_private_failure(content_drift_errors, "暗房正文甲已经修订。")

        current_darkroom_a = _get_diary(diary_store, int(darkroom_a["id"]))
        assert current_darkroom_a is not None
        marker_before_comment = _marker(current_darkroom_a)
        diary_store.comment(
            int(darkroom_a["id"]), content="只改变评论也必须使快照失效。", author="user"
        )
        _, comment_drift_errors = verify_narrative_darkroom_sources(
            exact_document=_document(marker_before_comment),
            darkroom_ids=[int(darkroom_a["id"])],
            get_darkroom=lambda diary_id: _get_diary(diary_store, diary_id),
        )
        assert comment_drift_errors[0]["reason"] == (
            "darkroom_snapshot_marker_missing_from_document"
        )
        _assert_private_failure(comment_drift_errors, "只改变评论也必须使快照失效。")

        current_darkroom_a = _get_diary(diary_store, int(darkroom_a["id"]))
        assert current_darkroom_a is not None
        document_v1 = _document(_marker(current_darkroom_a))
        created = narrative_store.publish(
            narrative_id="narrative_darkroom_source_test",
            document=document_v1,
            expected_revision=0,
            title="Darkroom source test",
            source_event_ids=[EVENT_ID],
            source_darkroom_ids=[int(current_darkroom_a["id"])],
        )
        assert created["status"] == "created", created
        assert created["linked_darkroom_ids"] == [darkroom_a["id"]], created
        assert created["linked_darkroom_count"] == 1, created

        document_v2 = _document(_marker(darkroom_b))
        updated = narrative_store.publish(
            narrative_id="narrative_darkroom_source_test",
            document=document_v2,
            expected_revision=1,
            title="Darkroom source test",
            source_event_ids=[EVENT_ID],
            source_darkroom_ids=[int(darkroom_b["id"])],
        )
        assert updated["status"] == "updated", updated
        assert updated["linked_darkroom_ids"] == [darkroom_b["id"]], updated
        assert updated["history"][0]["linked_darkroom_ids"] == [darkroom_a["id"]], updated

        registry_text = narrative_store.registry_path.read_text(encoding="utf-8")
        for secret in (
            str(current_darkroom_a["content"]),
            str(darkroom_b["content"]),
            "只改变评论也必须使快照失效。",
            "暗房标题甲",
            "暗房标题乙",
        ):
            assert secret not in registry_text, secret

        registry = json.loads(registry_text)
        entry = registry["rolls"][0]
        entry.pop("linked_darkroom_ids", None)
        for history_item in entry.get("history", []):
            history_item.pop("linked_darkroom_ids", None)
        narrative_store.registry_path.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        narrative_store._cache_stamp = None
        legacy = narrative_store.read("narrative_darkroom_source_test")
        assert legacy["status"] == "ok", legacy
        assert legacy["linked_darkroom_ids"] == [darkroom_b["id"]], legacy
        assert legacy["history"][0]["linked_darkroom_ids"] == [], legacy

    _verify_server_wiring()
    print(
        "PASS: Narrative Darkroom links are unlocked-only, snapshot-bound, "
        "legacy-safe, and privacy-preserving"
    )


if __name__ == "__main__":
    main()
