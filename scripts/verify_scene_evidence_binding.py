from __future__ import annotations

import hashlib
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene_evidence import SceneEvidenceStore, content_sha256, normalize_evidence_ref


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="ombre-scene-evidence-") as temp_dir:
        store = SceneEvidenceStore({"state_dir": str(Path(temp_dir))})
        exact = "第一行\n第二行  \n"
        ref = {
            "source_system": "haven_bridge",
            "session_id": 17,
            "thread_id": "thread-a",
            "message_id": 42,
            "role": "user",
            "created_at": "2026-08-03T10:00:00+08:00",
            "content": exact,
            "content_sha256": hashlib.sha256(exact.encode("utf-8")).hexdigest(),
            "binding_method": "approved_candidate_auto",
            "evidence_kind": "primary",
        }
        assert content_sha256(exact) == ref["content_sha256"]
        first = store.bind("scene_test", [ref], bound_by="test")
        assert first["evidence_status"] == "bound"
        assert first["bound_count"] == 1
        assert first["idempotent"] is False

        second = store.bind("scene_test", [ref], bound_by="retry")
        assert second["idempotent"] is True
        assert second["existing_count"] == 1
        saved = store.list_for_scene("scene_test")
        assert len(saved) == 1
        assert saved[0]["content"] == exact
        assert saved[0]["content_sha256"] == ref["content_sha256"]

        # Simulate opening a production sidecar created before binding-state
        # events existed. A read must migrate it without requiring a write.
        legacy_conn = sqlite3.connect(store.db_path)
        try:
            legacy_conn.execute("DROP TABLE scene_evidence_events")
            legacy_conn.commit()
        finally:
            legacy_conn.close()
        assert len(store.list_for_scene("scene_test")) == 1

        unbound = store.unbind("scene_test", [saved[0]["id"]], unbound_by="serein_ui")
        assert unbound["unbound_count"] == 1
        assert unbound["evidence_status"] == "unbound"
        assert store.list_for_scene("scene_test") == []

        repeated_unbind = store.unbind("scene_test", [saved[0]["id"]], unbound_by="retry")
        assert repeated_unbind["idempotent"] is True
        assert repeated_unbind["already_unbound_count"] == 1

        reactivated = store.bind("scene_test", [ref], bound_by="serein_ui")
        assert reactivated["reactivated_count"] == 1
        assert reactivated["bound_count"] == 1
        assert len(store.list_for_scene("scene_test")) == 1

        conflicting_content = "漂移后的原文"
        conflicting = {
            **ref,
            "content": conflicting_content,
            "content_sha256": content_sha256(conflicting_content),
        }
        try:
            store.bind("scene_test", [conflicting])
        except ValueError as error:
            assert "different source content" in str(error)
        else:
            raise AssertionError("conflicting evidence key was accepted")

        try:
            normalize_evidence_ref(
                {**ref, "content": "换行不同\n", "content_sha256": ref["content_sha256"]}
            )
        except ValueError as error:
            assert "does not match" in str(error)
        else:
            raise AssertionError("hash mismatch was accepted")

        supporting = {
            **ref,
            "message_id": 43,
            "role": "assistant",
            "content": "相邻支持原文",
            "content_sha256": content_sha256("相邻支持原文"),
            "evidence_kind": "supporting",
        }
        adjacent = {
            **ref,
            "message_id": 44,
            "content": None,
            "content_sha256": content_sha256("无法本地展开的快照"),
            "snapshot_ref": "haven_bridge://session/17/message/44",
            "evidence_kind": "adjacent_context",
        }
        result = store.bind("scene_test", [supporting, adjacent])
        assert result["bound_count"] == 2
        assert {item["evidence_kind"] for item in store.list_for_scene("scene_test")} == {
            "primary",
            "supporting",
            "adjacent_context",
        }
        grouped = store.list_active_scene_groups()
        assert list(grouped) == ["scene_test"]
        assert len(grouped["scene_test"]) == 3

    print("scene evidence binding verification passed")


if __name__ == "__main__":
    main()
