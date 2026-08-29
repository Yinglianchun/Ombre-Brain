"""Focused verifier for mixed atomic Fact/Event settlement and exact lineage reads."""

from __future__ import annotations

import asyncio
import copy
import sqlite3
import sys
import tempfile
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fact_events import FactEventStore
from scene_evidence import content_sha256


def ref(message_id: int, role: str, content: str, kind: str = "supporting") -> dict:
    return {
        "source_system": "haven_bridge",
        "session_id": "22",
        "thread_id": "thread-settlement",
        "message_id": str(message_id),
        "role": role,
        "created_at": f"2026-08-29T02:{message_id % 60:02d}:00Z",
        "content": content,
        "content_sha256": content_sha256(content),
        "evidence_kind": kind,
        "binding_method": "bridge_fact_event_sync_v1",
    }


def source_keys(item: dict) -> list[dict[str, str]]:
    return [
        {
            "source_system": str(value["source_system"]),
            "session_id": str(value.get("session_id") or ""),
            "message_id": str(value["message_id"]),
        }
        for value in item["source_refs"]
    ]


def expected(item: dict) -> dict:
    return {
        "item_id": item["item_id"],
        "fingerprint": item["fingerprint"],
        "source_keys": source_keys(item),
    }


def verify_dynamic_read_blockers() -> None:
    import server

    class NarrativeFixture:
        def __init__(self, linked_event_ids: list[str]):
            self.linked_event_ids = linked_event_ids

        def _load(self) -> list[dict]:
            return [
                {
                    "narrative_id": "narrative_fixture",
                    "lifecycle": "active",
                    "linked_event_ids": self.linked_event_ids,
                }
            ] if self.linked_event_ids else []

    class SceneFixture:
        def __init__(self, groups: dict[str, list[dict]]):
            self.groups = groups

        def list_active_scene_groups(self) -> dict[str, list[dict]]:
            return self.groups

    async def active_scene(scene_id: str) -> tuple[str, str]:
        return scene_id, ""

    base = {
        "ok": True,
        "items": [],
        "missing_item_ids": [],
        "families": [
            {
                "item_id": "event_old",
                "family_ids": ["event_old", "event_leaf"],
            }
        ],
        "resolutions": [
            {
                "requested_item_id": "event_old",
                "resolved_item_id": "event_leaf",
                "family_active_leaves": ["event_leaf"],
                "predecessor_event_ids": ["event_old"],
                "lineage": [],
                "forked": False,
                "blocked": False,
                "issues": [],
            }
        ],
        "resolved_items": [
            {
                "item_id": "event_leaf",
                "origin_id": "haven_bridge:fixture-leaf",
                "source_refs": [
                    {
                        "source_system": "haven_bridge",
                        "session_id": "22",
                        "message_id": "9201",
                    }
                ],
            }
        ],
    }

    with (
        patch.object(server, "narrative_roll_store", NarrativeFixture(["event_old"])),
        patch.object(server, "scene_evidence_store", SceneFixture({})),
    ):
        narrative = asyncio.run(
            server._materialize_fact_event_read_blockers(copy.deepcopy(base))
        )
    assert narrative["resolved_items"][0]["narrative_ref"] is True
    assert narrative["resolved_items"][0]["scene_ref"] is False
    assert narrative["resolutions"][0]["blocked"] is True
    assert narrative["resolutions"][0]["blocking_reasons"] == [
        "active_narrative_reference"
    ]

    scene_ref = {
        "source_system": "haven_bridge",
        "session_id": "22",
        "message_id": "9201",
    }
    with (
        patch.object(server, "narrative_roll_store", NarrativeFixture([])),
        patch.object(
            server,
            "scene_evidence_store",
            SceneFixture({"scene_fixture": [scene_ref]}),
        ),
        patch.object(server, "_validate_scene_evidence_target", active_scene),
    ):
        scene = asyncio.run(
            server._materialize_fact_event_read_blockers(copy.deepcopy(base))
        )
    assert scene["resolved_items"][0]["narrative_ref"] is False
    assert scene["resolved_items"][0]["scene_ref"] is True
    assert scene["resolutions"][0]["blocked"] is True
    assert scene["resolutions"][0]["blocking_reasons"] == [
        "active_scene_dependency"
    ]


def main() -> None:
    source_a = ref(9201, "user", "回答上一段，也开启下一段。", "primary")
    source_b = ref(9202, "assistant", "接住第一段。")
    source_c = ref(9203, "user", "转入新的完整问题。", "primary")
    source_d = ref(9204, "assistant", "回答新的问题。")
    source_e = ref(9205, "user", "单独新建的事实。", "primary")

    with tempfile.TemporaryDirectory(prefix="fact-event-settlement-") as temp_dir:
        store = FactEventStore({"state_dir": temp_dir}, create=True)
        old_result = store.write_many(
            [
                {
                    "type": "event",
                    "title": "旧事件一",
                    "body": "第一段完整讨论。",
                    "origin_id": "haven_bridge:settlement-old-a",
                    "source_refs": [source_a, source_b],
                },
                {
                    "type": "event",
                    "title": "旧事件二",
                    "body": "第二段完整讨论。",
                    "origin_id": "haven_bridge:settlement-old-b",
                    "source_refs": [source_c, source_d],
                },
            ]
        )
        old_items = [
            store.read(value["item_id"], include_sources=True)
            for value in old_result["items"]
        ]
        assert all(old_items)

        create_payload = {
            "type": "fact",
            "body": "一条同批新建的事实。",
            "origin_id": "haven_bridge:settlement-create",
            "source_refs": [source_e],
        }
        replacement_payload = {
            "type": "event",
            "title": "两条旧事件按原文合并",
            "body": "第一段与第二段被审核为同一完整经历。",
            "origin_id": "haven_bridge:settlement-merged",
            "source_refs": [
                source_a,
                source_b,
                {**source_c, "evidence_kind": "supporting"},
                source_d,
            ],
            "supersedes_item_ids": [item["item_id"] for item in old_items],
            "expected_predecessors": [expected(item) for item in old_items],
        }
        request_items = [create_payload, replacement_payload]
        settled = store.settle("op-mixed-atomic-1", request_items)
        assert settled["ok"] and settled["inserted"] == 2 and settled["idempotent"] == 0
        assert not settled["replayed"]
        successor_id = settled["items"][1]["item_id"]
        assert settled["items"][1]["superseded_item_ids"] == [
            item["item_id"] for item in old_items
        ]

        for old_item in old_items:
            family = store.replacement_family(old_item["item_id"])
            assert family["ok"] and family["active_leaf_id"] == successor_id, family
            assert not family["is_exact_active_leaf"]
            assert family["family_active_leaves"] == [successor_id]
            assert set(family["predecessor_event_ids"]) == {
                item["item_id"] for item in old_items
            }
            assert not family["forked"]

        exact = store.read_many(
            [old_items[0]["item_id"], successor_id],
            include_sources=True,
            resolve_active_successors=True,
        )
        assert [item["item_id"] for item in exact["items"]] == [
            old_items[0]["item_id"],
            successor_id,
        ]
        assert [item["item_id"] for item in exact["resolved_items"]] == [successor_id]
        assert all(value["resolved_item_id"] == successor_id for value in exact["resolutions"])
        assert all(not value["forked"] for value in exact["resolutions"])

        replay = store.settle("op-mixed-atomic-1", request_items)
        assert replay["inserted"] == 0 and replay["idempotent"] == 2
        assert replay["replayed"]
        assert all(item["status"] == "idempotent" for item in replay["items"])
        assert [item["item_id"] for item in replay["items"]] == [
            item["item_id"] for item in settled["items"]
        ]
        try:
            store.settle("op-mixed-atomic-1", [create_payload])
        except ValueError as exc:
            assert "different request" in str(exc)
        else:
            raise AssertionError("operation_id request drift was accepted")

        before_count = store.stats()["total"]
        bad_create = {
            "type": "fact",
            "body": "这一条必须随整批回滚。",
            "origin_id": "haven_bridge:settlement-must-rollback",
            "source_refs": [ref(9206, "user", "整批回滚。", "primary")],
        }
        bad_replacement = {
            **replacement_payload,
            "origin_id": "haven_bridge:settlement-bad-drift",
            "body": "漂移的旧收据不能写。",
            "supersedes_item_ids": [successor_id],
            "expected_predecessors": [
                {
                    **expected(store.read(successor_id, include_sources=True)),
                    "fingerprint": "0" * 64,
                }
            ],
        }
        try:
            store.settle("op-mixed-atomic-rollback", [bad_create, bad_replacement])
        except ValueError:
            pass
        else:
            raise AssertionError("drifted predecessor receipt was accepted")
        assert store.stats()["total"] == before_count
        assert store.list(query="这一条必须随整批回滚", status="all")["count"] == 0

        manual_root_result = store.write_many(
            [
                {
                    "type": "event",
                    "title": "自动生成的旧叶子",
                    "body": "这一条后来被手工修订。",
                    "origin_id": "haven_bridge:manual-family-root",
                    "source_refs": [ref(9207, "user", "后来手工修订。", "primary")],
                }
            ]
        )
        manual_root = store.read(manual_root_result["items"][0]["item_id"], include_sources=True)
        manual_successor_result = store.write_many(
            [
                {
                    "type": "event",
                    "title": "手工修订后的当前叶子",
                    "body": "正文由用户手工修订，不能被夜间任务覆盖。",
                    "origin_id": "manual-user-revision",
                    "supersedes_item_id": manual_root["item_id"],
                    "source_refs": manual_root["source_refs"],
                }
            ]
        )
        manual_successor = store.read(
            manual_successor_result["items"][0]["item_id"],
            include_sources=True,
        )
        manual_family = store.replacement_family(manual_root["item_id"])
        assert manual_family["active_leaf_id"] == manual_successor["item_id"]
        assert not manual_family["is_exact_active_leaf"]
        manual_resolution = store.read_many(
            [manual_root["item_id"]],
            include_sources=True,
            resolve_active_successors=True,
        )
        assert manual_resolution["resolutions"][0]["resolved_item_id"] == manual_successor["item_id"]
        assert manual_resolution["resolved_items"][0]["item_id"] == manual_successor["item_id"]
        manual_payload = {
            "type": "event",
            "title": "夜间任务不得覆盖手工叶子",
            "body": "这一条必须被 provenance 门阻断。",
            "origin_id": "haven_bridge:must-not-replace-manual",
            "supersedes_item_ids": [manual_successor["item_id"]],
            "expected_predecessors": [expected(manual_successor)],
            "source_refs": manual_successor["source_refs"],
        }
        manual_count = store.stats()["total"]
        try:
            store.settle("op-manual-provenance-block", [manual_payload])
        except ValueError as exc:
            assert "manual_or_untrusted_provenance" in str(exc)
        else:
            raise AssertionError("manual successor provenance was replaced")
        assert store.stats()["total"] == manual_count

        with closing(sqlite3.connect(store.db_path)) as conn:
            assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
            edges = conn.execute(
                """
                SELECT predecessor_id, successor_id
                FROM fact_event_replacement_edges
                WHERE successor_id=? ORDER BY predecessor_id
                """,
                (successor_id,),
            ).fetchall()
            assert set(edges) == {
                (old_items[0]["item_id"], successor_id),
                (old_items[1]["item_id"], successor_id),
            }
            assert conn.execute(
                "SELECT COUNT(*) FROM fact_event_settlement_operations"
            ).fetchone()[0] == 1

    with tempfile.TemporaryDirectory(prefix="fact-event-fork-") as temp_dir:
        store = FactEventStore({"state_dir": temp_dir}, create=True)
        fork_source = ref(9210, "user", "旧祖先被错误地分叉。", "primary")
        root_result = store.write_many(
            [
                {
                    "type": "event",
                    "title": "分叉根",
                    "body": "旧祖先。",
                    "origin_id": "haven_bridge:fork-root",
                    "source_refs": [fork_source],
                }
            ]
        )
        root_id = root_result["items"][0]["item_id"]
        children = []
        for ordinal in (1, 2):
            child = store.write_many(
                [
                    {
                        "type": "event",
                        "title": f"分叉叶子 {ordinal}",
                        "body": f"错误分叉后的第 {ordinal} 条当前叶子。",
                        "origin_id": f"haven_bridge:fork-child-{ordinal}",
                        "supersedes_item_id": root_id,
                        "source_refs": [fork_source],
                    }
                ]
            )
            children.append(child["items"][0]["item_id"])
        fork_read = store.read_many(
            [root_id],
            include_sources=True,
            resolve_active_successors=True,
        )
        assert fork_read["resolutions"][0]["forked"]
        assert fork_read["resolutions"][0]["blocked"]
        assert set(fork_read["resolutions"][0]["family_active_leaves"]) == set(children)
        assert fork_read["resolved_items"] == []

    with tempfile.TemporaryDirectory(prefix="fact-event-wide-source-") as temp_dir:
        store = FactEventStore({"state_dir": temp_dir}, create=True)
        many_refs = [
            ref(
                9300 + ordinal,
                "user" if ordinal % 2 == 0 else "assistant",
                f"长讨论中的第 {ordinal + 1} 条原文。",
                "primary" if ordinal == 0 else "supporting",
            )
            for ordinal in range(101)
        ]
        wide_root_result = store.write_many(
            [
                {
                    "type": "event",
                    "title": "拥有一百零一条来源的旧事件",
                    "body": "完整长讨论不因一百条来源上限而被截断。",
                    "origin_id": "haven_bridge:wide-source-root",
                    "source_refs": many_refs,
                }
            ]
        )
        wide_root = store.read(
            wide_root_result["items"][0]["item_id"],
            include_sources=True,
        )
        wide_settlement = store.settle(
            "op-wide-source-preservation",
            [
                {
                    "type": "event",
                    "title": "一百零一条来源完整保留",
                    "body": "审核后的事件仍然完整拥有全部一百零一条原文。",
                    "origin_id": "haven_bridge:wide-source-successor",
                    "source_refs": many_refs,
                    "supersedes_item_ids": [wide_root["item_id"]],
                    "expected_predecessors": [expected(wide_root)],
                }
            ],
        )
        assert wide_settlement["inserted"] == 1
        wide_successor = store.read(
            wide_settlement["items"][0]["item_id"],
            include_sources=True,
        )
        assert wide_root["source_count"] == 101
        assert wide_successor["source_count"] == 101

    server_text = (ROOT / "server.py").read_text(encoding="utf-8")
    assert '@mcp.custom_route("/api/fact-events/settlement"' in server_text
    assert '@mcp.custom_route("/api/fact-events/read-many"' in server_text
    assert '"writes_performed": []' in server_text
    assert '"kind": "active_narrative"' in server_text
    assert '"kind": "active_scene"' in server_text
    verify_dynamic_read_blockers()
    print("FACT_EVENT_SETTLEMENT_OK")


if __name__ == "__main__":
    main()

