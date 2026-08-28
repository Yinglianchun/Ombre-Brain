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
            "fact_type": "health",
            "atomic_question": "小雨是否有紫外线过敏？",
            "importance": 4,
            "origin_id": "bridge-candidate-fact-1",
            "source_refs": [source_a],
        }
        event = {
            "type": "event",
            "title": "记住小雨，出门需要遮阳",
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
        atomic = store.write_many(
            [
                {
                    **event,
                    "title": "这一条必须随整批回滚",
                    "body": "只要同批另一项冲突，这条也不能写入。",
                    "origin_id": "bridge-candidate-event-atomic-rollback",
                },
                {**fact, "body": "同一 origin 的冲突内容。"},
            ]
        )
        assert atomic["inserted"] == 0 and atomic["rejected"] == 2, atomic
        assert store.list(item_type="event", query="必须随整批回滚")["count"] == 0

        fact_row = store.read(fact_id)
        assert fact_row and fact_row["local_date"] == "2026-08-08"
        assert fact_row["local_start_time"] == "16:15"
        assert fact_row["source_started_at"] == source_a["created_at"]
        assert fact_row["importance"] == 4 and fact_row["injection_count"] == 0
        assert fact_row["fact_type"] == "health"
        assert fact_row["atomic_question"] == "小雨是否有紫外线过敏？"

        event_row = store.read(event_id)
        assert event_row and event_row["local_start_time"] == "16:15"
        assert event_row["local_end_time"] == "16:17"
        assert len(event_row["source_refs"]) == 2

        listed = store.list(item_type="fact", date="2026-08-08")
        assert listed["count"] == 1 and listed["items"][0]["item_id"] == fact_id
        assert "source_refs" not in listed["items"][0]
        assert listed["items"][0]["source_count"] == 1
        assert store.list(item_type="event")["items"][0]["source_count"] == 2
        assert store.list(item_type="event", query="紫外线过敏")["items"][0]["item_id"] == event_id
        assert store.list(item_type="event", query="记住小雨")["items"][0]["item_id"] == event_id
        assert store.list(item_type="event", query="记住小雨,出门")["items"][0]["item_id"] == event_id
        assert store.list(item_type="event", query="haven")["items"][0]["item_id"] == event_id
        assert store.list(item_type="event", query="不存在的作品名")["count"] == 0

        injected = store.mark_injected([fact_id, event_id, "fact_missing"])
        assert injected["updated"] == 2
        assert store.read(fact_id)["injection_count"] == 1

        importance_revision = store.revise(fact_id, importance=5)
        assert importance_revision["status"] == "updated"
        assert importance_revision["item"]["item_id"] == fact_id
        assert importance_revision["item"]["importance"] == 5

        archived_fact = store.set_status(fact_id, "archived")
        assert archived_fact["item"]["status"] == "archived"
        assert store.list(item_type="fact", status="active")["count"] == 0
        restored_fact = store.set_status(fact_id, "active")
        assert restored_fact["item"]["status"] == "active"

        legacy_meal_source = ref(
            8105,
            "user",
            "昨天午饭吃了面条。",
            "2026-08-08T09:15:00Z",
            "primary",
        )
        legacy_meal = store.write_many(
            [
                {
                    "type": "fact",
                    "body": "小雨昨天午饭吃了面条。",
                    "origin_id": "bridge-legacy-meal",
                    "source_refs": [legacy_meal_source],
                }
            ]
        )
        legacy_meal_id = legacy_meal["items"][0]["item_id"]

        source_c = ref(
            8103,
            "user",
            "医生确认我不是紫外线过敏，只是晒伤。",
            "2026-08-09T08:15:00Z",
            "primary",
        )
        correction = store.write_many(
            [
                {
                    "type": "fact",
                    "body": "医生确认小雨不是紫外线过敏，只是晒伤。",
                    "fact_type": "health",
                    "atomic_question": "小雨是否有紫外线过敏？",
                    "importance": 4,
                    "origin_id": "bridge-candidate-fact-2",
                    "source_refs": [source_c],
                }
            ]
        )
        correction_id = correction["items"][0]["item_id"]
        relation_groups = store.relation_candidates([correction_id], limit_per_fact=10)
        assert relation_groups["groups"][0]["candidates"][0]["item_id"] == fact_id
        assert legacy_meal_id not in {
            item["item_id"] for item in relation_groups["groups"][0]["candidates"]
        }
        proposal = store.propose_relations(
            [
                {
                    "new_fact_id": correction_id,
                    "candidate_fact_id": fact_id,
                    "relation": "supersedes",
                    "reason": "医生结论与先前自述不能同时作为当前诊断成立。",
                    "confidence": 0.96,
                    "review_status": "accepted",
                    "review_reason": "新原文明确纠正旧判断，且证据来自小雨直接转述医生结论。",
                    "review_confidence": 0.97,
                    "explicit_correction": True,
                }
            ]
        )
        assert proposal["inserted"] == 1
        assert store.propose_relations(
            [
                {
                    "new_fact_id": correction_id,
                    "candidate_fact_id": fact_id,
                    "relation": "supersedes",
                    "reason": "重复提交不会新建。",
                    "confidence": 0.96,
                    "review_status": "accepted",
                    "review_reason": "重复提交。",
                    "review_confidence": 0.97,
                    "explicit_correction": True,
                }
            ]
        )["idempotent"] == 1
        pending = store.list_relation_proposals(status="accepted")
        assert pending["items"][0]["new_fact_id"] == correction_id
        assert store.read(fact_id)["status"] == "superseded"
        assert store.read(correction_id)["supersedes_item_id"] == fact_id

        meal_source = ref(
            8104,
            "user",
            "今天吃了牛肉面。",
            "2026-08-09T09:15:00Z",
            "primary",
        )
        meal = store.write_many(
            [
                {
                    "type": "fact",
                    "body": "小雨今天吃了牛肉面。",
                    "fact_type": "meal",
                    "atomic_question": "小雨今天吃了什么？",
                    "origin_id": "bridge-candidate-meal-1",
                    "source_refs": [meal_source],
                }
            ]
        )
        meal_id = meal["items"][0]["item_id"]
        meal_candidates = store.relation_candidates([meal_id])
        assert meal_candidates["groups"] == []
        assert meal_candidates["skipped"] == [{"item_id": meal_id, "reason": "meal_fact"}]

        event_revision = store.revise(
            event_id,
            title="记住出门时替小雨遮阳",
            body="小雨说明自己紫外线过敏，Haven答应在出门时提醒她遮阳。",
        )
        revised_event_id = event_revision["item"]["item_id"]
        assert event_revision["status"] == "superseded"
        assert revised_event_id != event_id
        assert store.read(event_id)["status"] == "superseded"
        assert len(event_revision["item"]["source_refs"]) == 2
        event_id = revised_event_id

        partial_scene = store.archive_events_covered_by_scene("scene_partial", [source_a])
        assert partial_scene["archived"] == 0
        covered = store.archive_events_covered_by_scene(
            "scene_covering_event",
            [source_a, source_b],
        )
        assert covered == {"archived": 1, "item_ids": [event_id]}
        assert store.read(event_id)["status"] == "archived"
        assert store.read(event_id)["covered_by_scene_id"] == "scene_covering_event"
        assert store.read(fact_id)["status"] == "superseded"

        deleted_correction = store.delete(correction_id)
        assert deleted_correction["deleted"] == 2
        assert store.list_relation_proposals(status="all")["items"] == []
        deleted_meal = store.delete(meal_id)
        assert deleted_meal["deleted"] == 1
        assert store.delete(legacy_meal_id)["deleted"] == 1
        assert store.read(fact_id) is None
        assert store.list(item_type="fact", status="active")["count"] == 0

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

        deleted_event = store.delete(event_id)
        assert deleted_event["deleted"] == 2
        assert set(deleted_event["item_ids"]) == {
            event_id,
            event_revision["previous_item_id"],
        }
        assert store.read(event_id) is None
        assert store.read(event_revision["previous_item_id"]) is None

        stats = store.stats()
        assert stats == {"total": 1, "facts": 0, "events": 1, "active": 0}
        with closing(sqlite3.connect(store.db_path)) as conn:
            assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
            tables = {
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            assert "fact_event_embeddings" not in tables
            assert "fact_event_edges" not in tables

    with tempfile.TemporaryDirectory(prefix="fact-event-replacements-") as temp_dir:
        store = FactEventStore({"state_dir": temp_dir}, create=True)
        old_one = {
            "type": "event",
            "title": "第一条旧事件",
            "body": "小雨说明自己紫外线过敏。",
            "importance": 1,
            "origin_id": "bridge-old-event-1",
            "source_refs": [source_a],
        }
        old_two = {
            "type": "event",
            "title": "第二条旧事件",
            "body": "我记住出门时要提醒小雨遮阳。",
            "importance": 1,
            "origin_id": "bridge-old-event-2",
            "source_refs": [{**source_b, "evidence_kind": "primary"}],
        }
        old_result = store.write_many([old_one, old_two])
        old_ids = [item["item_id"] for item in old_result["items"]]
        replacement = {
            "type": "event",
            "title": "两条旧事件合成一条",
            "body": "小雨说明自己紫外线过敏，我记住出门时要提醒她遮阳。",
            "importance": 3,
            "origin_id": "bridge-reviewed-event-replacement-1",
            "source_refs": [source_a, source_b],
            "supersedes_item_ids": old_ids,
        }
        replacement_result = store.replace_many([replacement])
        replacement_id = replacement_result["items"][0]["item_id"]
        assert replacement_result["inserted"] == 1
        assert all(store.read(item_id)["status"] == "superseded" for item_id in old_ids)
        assert store.read(replacement_id)["status"] == "active"
        assert store.read(replacement_id)["supersedes_item_id"] == old_ids[0]
        repeated_replacement = store.replace_many([replacement])
        assert repeated_replacement["inserted"] == 0
        assert repeated_replacement["idempotent"] == 1

    print("FACT_EVENTS_OK")


if __name__ == "__main__":
    main()
