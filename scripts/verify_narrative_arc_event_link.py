from __future__ import annotations

import ast
import hashlib
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fact_events import FactEventStore


def _ref(message_id: int, role: str, content: str, kind: str = "supporting") -> dict:
    return {
        "source_system": "haven_bridge",
        "session_id": "22",
        "thread_id": "thread-arc-link",
        "message_id": str(message_id),
        "role": role,
        "created_at": f"2026-08-30T01:{message_id % 60:02d}:00Z",
        "content": content,
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "evidence_kind": kind,
        "binding_method": "bridge_fact_event_sync_v1",
    }


def _expected(item: dict) -> dict:
    return {
        "item_id": item["item_id"],
        "fingerprint": item["fingerprint"],
        "source_keys": [
            {
                "source_system": str(ref["source_system"]),
                "session_id": str(ref.get("session_id") or ""),
                "message_id": str(ref["message_id"]),
            }
            for ref in item["source_refs"]
        ],
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="narrative-arc-event-link-") as temp_dir:
        store = FactEventStore({"state_dir": temp_dir}, create=True)
        first_source = _ref(9601, "user", "继续缝补记忆系统。", "primary")
        first_reply = _ref(9602, "assistant", "原文读取入口已经接好。")
        written = store.write_many(
            [
                {
                    "type": "event",
                    "title": "认清取原文的正门",
                    "body": "我们把绑定原文的入口接回了记忆系统。",
                    "origin_id": "haven_bridge:arc-link-old",
                    "source_refs": [first_source, first_reply],
                }
            ]
        )
        old = store.read(written["items"][0]["item_id"], include_sources=True)
        original_connect = store._connect
        request_connections = 0

        def counted_connect():
            nonlocal request_connections
            request_connections += 1
            return original_connect()

        store._connect = counted_connect
        store._init_db()
        assert request_connections == 0, "initialized stores must not rerun schema DDL"
        linked = store.link_arc_events(
            "project:serein-memory-system",
            [{"event_id": old["item_id"], "fingerprint": old["fingerprint"]}],
        )
        assert linked["status"] == "updated" and linked["body_unchanged"] is True, linked
        assert request_connections == 1, "Arc linking should open only its write transaction"
        assert store.arc_event_links("project:serein-memory-system")[0]["event_id"] == old["item_id"]
        assert request_connections == 2, "Arc reads should not rerun schema DDL"
        idempotent = store.link_arc_events(
            "project:serein-memory-system",
            [{"event_id": old["item_id"], "fingerprint": old["fingerprint"]}],
        )
        assert idempotent["status"] == "idempotent", idempotent

        continuation = _ref(9603, "user", "第二天继续验证旧入口。")
        landing = _ref(9604, "assistant", "同一条经历终于落定。")
        replacement_payload = {
            "type": "event",
            "title": "原文读取入口终于落定",
            "body": "旧 Event 与第二天的验证被收回同一条完整经历。",
            "origin_id": "haven_bridge:arc-link-successor",
            "source_refs": [first_source, first_reply, continuation, landing],
            "supersedes_item_ids": [old["item_id"]],
            "expected_predecessors": [_expected(old)],
        }
        settled = store.settle("op-arc-link-migration", [replacement_payload])
        successor_id = settled["items"][0]["item_id"]
        assert settled["items"][0]["migrated_arc_keys"] == [
            "project:serein-memory-system"
        ], settled
        links = store.arc_event_links("project:serein-memory-system")
        assert [item["event_id"] for item in links] == [successor_id], links
        assert links[0]["fingerprint"] == settled["items"][0]["fingerprint"]

    server_source = (ROOT / "server.py").read_text(encoding="utf-8")
    tree = ast.parse(server_source)
    append_route = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "api_append_narrative_arc_event_materials"
    )
    route_text = ast.get_source_segment(server_source, append_route) or ""
    assert "read_by_arc_key" in route_text
    assert "fact_event_store.link_arc_events" in route_text
    assert "event_fingerprint_mismatch" in route_text
    assert "narrative_roll_store.publish" not in route_text
    assert "body_unchanged" not in route_text or "link_arc_events" in route_text

    read_route = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_read_narrative_memory"
    )
    read_text = ast.get_source_segment(server_source, read_route) or ""
    assert "automatic_event_links" in read_text
    assert "arc_event_links" in read_text

    print("NARRATIVE_ARC_EVENT_LINK_OK")


if __name__ == "__main__":
    main()
