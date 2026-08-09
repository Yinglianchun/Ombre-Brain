"""Focused contract checks for canonical source-bound Fact/Event storage."""

from __future__ import annotations

import sqlite3
import sys
import tempfile
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fact_events import FactEventStore
from scene_evidence import content_sha256


def ref(message_id: int, role: str, content: str, created_at: str, kind: str) -> dict:
    return {
        "source_system": "haven_bridge",
        "session_id": "19",
        "thread_id": "thread-contract",
        "message_id": str(message_id),
        "role": role,
        "created_at": created_at,
        "content": content,
        "content_sha256": content_sha256(content),
        "evidence_kind": kind,
        "binding_method": "bridge_fact_event_sync_v1",
    }


def main() -> None:
    source_a = ref(
        8101,
        "user",
        "我紫外线过敏，太阳大的时候皮肤会不舒服。",
        "2026-08-08T08:15:00Z",
        "primary",
    )
    source_b = ref(
        8102,
        "assistant",
        "那出门时我会记得提醒你遮阳。",
        "2026-08-08T08:17:00Z",
        "supporting",
    )

    with tempfile.TemporaryDirectory(prefix="fact-events-") as temp_dir:
        store = FactEventStore({"state_dir": temp_dir}, create=True)
        fact = {
            "type": "fact",
            "body": "小雨紫外线过敏。",
            "importance": 4,
            "origin_id": "bridge-candidate-fact-1",
            "source_refs": [source_a],
        }
        event = {
            "type": "event",
            "title": "记住小雨需要遮阳",
            "body": "小雨说明自己紫外线过敏，Haven答应出门时提醒她遮阳。",
            "importance": 3,
            "origin_id": "bridge-candidate-event-1",
            "source_refs": [source_a, source_b],
        }

        first = store.write_many([fact, event])
        assert first["inserted"] == 2 and first["rejected"] == 0, first
        fact_id = first["items"][0]["item_id"]
        event_id = first["items"][1]["item_id"]
        assert fact_id.startswith("fact_") and event_id.startswith("event_")

        repeated = store.write_many([fact, event])
        assert repeated["idempotent"] == 2 and repeated["inserted"] == 0
        conflict = store.write_many([{**fact, "body": "同一来源候选被悄悄改写。"}])
        assert conflict["rejected"] == 1
        assert "origin_id" in conflict["items"][0]["error"]

        fact_row = store.read(fact_id)
        assert fact_row and fact_row["local_date"] == "2026-08-08"
        assert fact_row["local_start_time"] == "16:15"
        assert fact_row["source_started_at"] == source_a["created_at"]
        assert fact_row["importance"] == 4 and fact_row["injection_count"] == 0

        event_row = store.read(event_id)
        assert event_row and event_row["local_start_time"] == "16:15"
        assert event_row["local_end_time"] == "16:17"
        assert len(event_row["source_refs"]) == 2

        listed = store.list(item_type="fact", date="2026-08-08")
        assert listed["count"] == 1 and listed["items"][0]["item_id"] == fact_id
        assert "source_refs" not in listed["items"][0]

        injected = store.mark_injected([fact_id, event_id, "fact_missing"])
        assert injected["updated"] == 2
        assert store.read(fact_id)["injection_count"] == 1

        partial_scene = store.archive_events_covered_by_scene("scene_partial", [source_a])
        assert partial_scene["archived"] == 0
        covered = store.archive_events_covered_by_scene(
            "scene_covering_event",
            [source_a, source_b],
        )
        assert covered == {"archived": 1, "item_ids": [event_id]}
        assert store.read(event_id)["status"] == "archived"
        assert store.read(event_id)["covered_by_scene_id"] == "scene_covering_event"
        assert store.read(fact_id)["status"] == "active"

        late_event = {
            **event,
            "origin_id": "bridge-candidate-event-late",
            "body": "小雨提到紫外线过敏，Haven答应提醒她遮阳。",
        }
        late = store.write_many([late_event])
        late_event_id = late["items"][0]["item_id"]
        reconciled = store.archive_events_covered_by_scenes(
            {"scene_already_present": [source_a, source_b]}
        )
        assert reconciled == {"archived": 1, "item_ids": [late_event_id]}
        assert store.read(late_event_id)["covered_by_scene_id"] == "scene_already_present"

        invalid = store.write_many(
            [
                {"type": "event", "body": "没有标题。", "source_refs": [source_a]},
                {**fact, "origin_id": "bad-fact-title", "title": "事实不需要标题"},
            ]
        )
        assert invalid["rejected"] == 2 and invalid["inserted"] == 0

        stats = store.stats()
        assert stats == {"total": 3, "facts": 1, "events": 2, "active": 1}
        with closing(sqlite3.connect(store.db_path)) as conn:
            assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
            tables = {
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            assert "fact_event_embeddings" not in tables
            assert "fact_event_edges" not in tables

    print("FACT_EVENTS_OK")


if __name__ == "__main__":
    main()
