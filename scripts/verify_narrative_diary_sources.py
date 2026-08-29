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
    diary_comments_sha256,
    diary_source_marker,
    verify_narrative_diary_sources,
)


def _marker(diary: dict) -> str:
    return diary_source_marker(
        diary_id=int(diary["id"]),
        revision=int(diary["revision"]),
        content_sha256=hashlib.sha256(str(diary["content"]).encode("utf-8")).hexdigest(),
        comments_sha256=diary_comments_sha256(diary.get("comments")),
    )


def _document(*markers: str) -> str:
    return "# Diary Arc\n\n## 材料目录\n\n" + "\n".join(f"- `{value}`" for value in markers) + "\n"


def _get_diary(store: DiaryStore, diary_id: int) -> dict | None:
    result = store.read(diary_id=diary_id, limit=1, include_archived=True)
    if result.get("count") != 1 or int(result.get("id") or 0) != diary_id:
        return None
    return result


def _verify_server_wiring() -> None:
    tree = ast.parse((ROOT / "server.py").read_text(encoding="utf-8"))
    publish = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "publish_narrative"
    )
    assert "source_diary_ids" in {argument.arg for argument in publish.args.args}
    calls = [node for node in ast.walk(publish) if isinstance(node, ast.Call)]
    assert any(
        isinstance(call.func, ast.Name) and call.func.id == "verify_narrative_diary_sources"
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
    assert "source_diary_ids" in {keyword.arg for keyword in store_publish.keywords}


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="narrative-diary-source-") as temp_dir:
        state_dir = Path(temp_dir) / "state"
        diary_store = DiaryStore({"state_dir": str(state_dir)})
        narrative_store = NarrativeRollStore({"state_dir": str(state_dir)})

        diary_a = diary_store.create(
            content="第一篇只留在 Diary 数据库里的正文。",
            date="2026-08-17",
            title="第一篇",
        )
        diary_b = diary_store.create(
            content="第二篇也不应复制到 Narrative registry。",
            date="2026-08-18",
            title="第二篇",
        )
        diary_c = diary_store.create(
            content="第三篇用于替换来源。",
            date="2026-08-19",
            title="第三篇",
        )
        archived = diary_store.create(
            content="归档来源不能新挂入 Narrative。",
            date="2026-08-20",
            title="归档篇",
            visibility="archived",
        )
        darkroom = diary_store.create(
            content="Darkroom 不是普通 Diary 来源。",
            date="2026-08-21",
            title="暗房篇",
            entry_type="darkroom",
        )

        document_v1 = _document(_marker(diary_a), _marker(diary_b))
        resolved, errors = verify_narrative_diary_sources(
            exact_document=document_v1,
            diary_ids=[int(diary_a["id"]), int(diary_b["id"])],
            get_diary=lambda diary_id: _get_diary(diary_store, diary_id),
        )
        assert errors == [], errors
        assert [row["diary_id"] for row in resolved] == [diary_a["id"], diary_b["id"]]
        assert all("content" not in row for row in resolved), resolved
        assert all("comments" not in row for row in resolved), resolved

        diary_store.comment(int(diary_a["id"]), content="后来补上的一条评论。", author="user")
        diary_a_with_comment = _get_diary(diary_store, int(diary_a["id"]))
        assert diary_a_with_comment is not None
        assert diary_a_with_comment["revision"] == diary_a["revision"]
        assert diary_comments_sha256(diary_a_with_comment["comments"]) == diary_comments_sha256(
            list(reversed(diary_a_with_comment["comments"]))
        )
        _, added_comment_errors = verify_narrative_diary_sources(
            exact_document=document_v1,
            diary_ids=[int(diary_a["id"])],
            get_diary=lambda diary_id: _get_diary(diary_store, diary_id),
        )
        assert added_comment_errors[0]["reason"] == "diary_snapshot_marker_missing_from_document"

        current_diary_a_marker = _marker(diary_a_with_comment)
        modified_comment_diary = {
            **diary_a_with_comment,
            "comments": [
                {**comment, "content": "同一条评论被改写了。"}
                for comment in diary_a_with_comment["comments"]
            ],
        }
        _, modified_comment_errors = verify_narrative_diary_sources(
            exact_document=_document(current_diary_a_marker),
            diary_ids=[int(diary_a["id"])],
            get_diary=lambda _diary_id: modified_comment_diary,
        )
        assert modified_comment_errors[0]["reason"] == "diary_snapshot_marker_missing_from_document"
        document_v1 = _document(current_diary_a_marker, _marker(diary_b))

        stale_document = _document(
            diary_source_marker(
                diary_id=int(diary_a["id"]),
                revision=int(diary_a["revision"]) + 1,
                content_sha256="0" * 64,
                comments_sha256="0" * 64,
            )
        )
        _, stale_errors = verify_narrative_diary_sources(
            exact_document=stale_document,
            diary_ids=[int(diary_a["id"])],
            get_diary=lambda diary_id: _get_diary(diary_store, diary_id),
        )
        assert stale_errors[0]["reason"] == "diary_snapshot_marker_missing_from_document"

        _, inactive_errors = verify_narrative_diary_sources(
            exact_document=_document(_marker(archived), _marker(darkroom)),
            diary_ids=[int(archived["id"]), int(darkroom["id"]), 999999],
            get_diary=lambda diary_id: _get_diary(diary_store, diary_id),
        )
        assert [row["reason"] for row in inactive_errors] == [
            "not_active_diary",
            "not_ordinary_diary",
            "diary_not_found",
        ], inactive_errors

        created = narrative_store.publish(
            narrative_id="narrative_diary_source_test",
            document=document_v1,
            expected_revision=0,
            title="Diary source test",
            source_diary_ids=[int(diary_a["id"]), int(diary_b["id"])],
            publication_status="collecting",
            arc_key="project:diary-source-test",
        )
        assert created["status"] == "created", created
        assert created["linked_diary_ids"] == [diary_a["id"], diary_b["id"]], created
        assert created["linked_diary_count"] == 2, created
        listed = narrative_store.list(query="Diary source test")["items"][0]
        assert listed["linked_diary_ids"] == [diary_a["id"], diary_b["id"]], listed
        assert listed["linked_diary_count"] == 2, listed

        conflict = narrative_store.publish(
            narrative_id="narrative_diary_source_test",
            document=_document(_marker(diary_b), _marker(diary_c)),
            expected_revision=0,
            title="Diary source test",
            source_diary_ids=[int(diary_b["id"]), int(diary_c["id"])],
            publication_status="collecting",
        )
        assert conflict["reason"] == "revision_mismatch", conflict

        updated = narrative_store.publish(
            narrative_id="narrative_diary_source_test",
            document=_document(_marker(diary_b), _marker(diary_c)),
            expected_revision=1,
            title="Diary source test",
            source_diary_ids=[int(diary_b["id"]), int(diary_c["id"])],
            publication_status="collecting",
        )
        assert updated["status"] == "updated", updated
        assert updated["linked_diary_ids"] == [diary_b["id"], diary_c["id"]], updated
        assert updated["history"][0]["linked_diary_ids"] == [diary_a["id"], diary_b["id"]], updated

        registry_text = narrative_store.registry_path.read_text(encoding="utf-8")
        assert str(diary_a["content"]) not in registry_text
        assert str(diary_b["content"]) not in registry_text
        assert str(diary_c["content"]) not in registry_text

        registry = json.loads(registry_text)
        entry = registry["rolls"][0]
        entry.pop("linked_diary_ids", None)
        for history_item in entry.get("history", []):
            history_item.pop("linked_diary_ids", None)
        narrative_store.registry_path.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        narrative_store._cache_stamp = None
        legacy = narrative_store.read("narrative_diary_source_test")
        assert legacy["status"] == "ok", legacy
        assert legacy["linked_diary_ids"] == [diary_b["id"], diary_c["id"]], legacy
        assert legacy["history"][0]["linked_diary_ids"] == [], legacy

    _verify_server_wiring()
    print(
        "PASS: Narrative Diary links are active, snapshot-bound, legacy-safe, "
        "and body/comment-free"
    )


if __name__ == "__main__":
    main()
