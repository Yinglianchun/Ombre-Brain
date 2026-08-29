from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_shadow_unassigned_arc_candidates_germany import build_unassigned_inventory
from scripts.shadow_arc_candidates_germany import Node


def node(node_id: str, text: str, source_id: int, *, routine: bool = False) -> Node:
    return Node(
        node_id=node_id,
        kind="event",
        title=text.split("\n", 1)[0],
        text=text,
        date="2026-08-29",
        fingerprint=node_id.removeprefix("event_").ljust(64, "0"),
        source_ids=(source_id,),
        session_ids=(22,),
        track_ids=(),
        track_texts=(),
        statuses=("unsettled",),
        routine_flags=(routine,),
    )


nodes = [
    node("event_111111111111111111111111", "《间谍过家家》阿尼亚与黄昏", 1),
    node("event_222222222222222222222222", "继续《间谍过家家》阿尼亚与黄昏", 2),
    node("event_333333333333333333333333", "《Stone Memory 的第一行》机制观察", 3),
    node("event_444444444444444444444444", "复查《Stone Memory 的第一行》", 4),
    node("event_555555555555555555555555", "《Stone Memory 的第一行》早安", 5, routine=True),
]
rolls = [
    {
        "narrative_id": "narrative_spy_family",
        "title": "间谍过家家",
        "arc_key": "work:间谍过家家",
        "linked_event_ids": [nodes[0].node_id, nodes[1].node_id],
        "linked_scene_ids": [],
        "title_aliases": ["间谍过家家"],
        "primary_entities": ["间谍过家家"],
        "supporting_entities": ["阿尼亚", "黄昏"],
        "query_cues": ["阿尼亚", "黄昏"],
    }
]

result = build_unassigned_inventory(nodes, rolls)
assert result["counts"] == {
    "active_nodes": 5,
    "assigned_unique_nodes": 2,
    "unassigned_nodes": 3,
    "eligible_unassigned_nodes": 2,
    "gated_unassigned_nodes": 1,
    "candidate_groups": 1,
}, result["counts"]
stone = result["requested_checks"]["stone_memory_candidates"]
assert len(stone) == 1 and stone[0]["support_count"] == 2, stone
spy = result["requested_checks"]["spy_family_coverage"]
assert spy["registry_member_ids"] == [nodes[0].node_id, nodes[1].node_id], spy
assert spy["unassigned_direct_label_support_ids"] == [], spy
assert result["requested_checks"]["spy_family_extension_candidates"] is None
assert result["gated_receipts"] == [
    {"node_id": nodes[4].node_id, "decision": "exclude", "reason": "routine_only"}
], result["gated_receipts"]

print("PASS: all-unassigned Arc discovery preserves registry coverage and gates")
