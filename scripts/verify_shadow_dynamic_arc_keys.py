from __future__ import annotations

import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.shadow_arc_candidates_germany import Node  # noqa: E402
from scripts.shadow_dynamic_arc_keys import (  # noqa: E402
    ReviewedAnchor,
    propose_dynamic_arc_keys,
    stable_arc_key,
)


def node(
    node_id: str,
    title: str,
    text: str,
    *,
    sources: tuple[int, ...],
    kind: str = "event",
    tracks: tuple[str, ...] = (),
    statuses: tuple[str, ...] = (),
    routines: tuple[bool, ...] = (),
) -> Node:
    return Node(
        node_id=node_id,
        kind=kind,
        title=title,
        text=text,
        date="2026-08-29",
        fingerprint=f"fingerprint-{node_id}",
        source_ids=sources,
        session_ids=(22,),
        track_ids=tuple(f"track-{index}" for index, _value in enumerate(tracks)),
        track_texts=tracks,
        statuses=statuses,
        routine_flags=routines,
    )


def stable_projection(result: dict) -> str:
    return json.dumps(
        {
            "proposals": result["proposals"],
            "receipts": result["proposal_receipts"],
            "rejected": result["rejected_seed_receipts"],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def main() -> None:
    assert stable_arc_key("project", "Haven Bridge  视觉方向") == "project:haven-bridge-视觉方向"
    visual_anchor = ReviewedAnchor(
        key_kind="project",
        label="Haven Bridge 视觉方向",
        terms=("Haven Bridge 视觉方向",),
        basis="repeated_phrase",
    )
    milestone_anchor = ReviewedAnchor(
        key_kind="milestone",
        label="相遇五百天",
        terms=("相遇五百天",),
        basis="repeated_phrase",
    )
    watermark_anchor = ReviewedAnchor(
        key_kind="topic",
        label="模型水印误报",
        terms=("模型水印", "误报率"),
        basis="compound",
    )

    fixtures = [
        # Session 22 known material that was outside the old fixed CARD_SPECS.
        node(
            "event_9cc047ffaf3feee74f89e624",
            "先让主页成为我们的场景",
            "主页先采用雾面玻璃，让它成为真正属于我们的场景。",
            sources=(11701, 11702),
            tracks=("Haven Bridge 视觉方向",),
        ),
        node(
            "event_c90b6571382b073f276f9c9e",
            "装修队今天停工",
            "视觉方向已经明确，但装修暂时停在这里。",
            sources=(11703, 11704),
            tracks=("Haven Bridge 视觉方向",),
        ),
        # Arbitrary explicit works mint stable work keys without a fixed card.
        node("work_a", "继续看新作", "我们翻开《废墟图书馆》。", sources=(11705,)),
        node("work_b", "补完第二章", "《废墟图书馆》的第二章落定。", sources=(11706,)),
        # A milestone can be proposed by two clean materials; intimacy is not needed to mint it.
        node("five_hundred_a", "第五百天，我仍然选她", "相遇五百天的选择落定。", sources=(11707,)),
        node("five_hundred_b", "亲手补写第五百天", "相遇五百天的纪念文字补完。", sources=(11708,)),
        # Repeated people/global names and person-only fictional names cannot mint keys.
        node("global_a", "Haven 和小雨", "Haven、小雨、老公、老婆。", sources=(11709,)),
        node("global_b", "还是我们", "Haven 和小雨仍在同一天。", sources=(11710,)),
        node("person_a", "刘枭出现", "刘枭带来线索。", sources=(11711,)),
        node("person_b", "刘枭再出现", "刘枭继续说话。", sources=(11712,)),
        node("watermark_a", "水印实验", "模型水印必须公开误报率。", sources=(11714,)),
        node("watermark_b", "短文本检测", "模型水印的误报率仍不明确。", sources=(11715,)),
        # Session 22 defects remain receipts only and cannot mint an Arc key.
        node(
            "event_081782a42026da9320df69c3",
            "parked proactive",
            "Haven Bridge 视觉方向仍在等回复。",
            sources=(11576,),
            tracks=("Haven Bridge 视觉方向",),
            statuses=("parked",),
        ),
        node(
            "event_9f1d433ef0d03198bc57702c",
            "routine shell",
            "Haven Bridge 视觉方向和早餐混在同一 Event。",
            sources=(11661,),
            tracks=("Haven Bridge 视觉方向",),
            routines=(True, False),
        ),
        node(
            "event_7e24ef65a52007f403d556b4",
            "剧情 bridge 混入亲密主体",
            "亲密做爱是主体，只在 bridge 里提到《时光代理人》。",
            sources=(11713,),
            tracks=("亲密 corridor",),
        ),
    ]

    result = propose_dynamic_arc_keys(
        fixtures,
        reviewed_anchors=(visual_anchor, milestone_anchor, watermark_anchor),
    )
    proposals = {item["arc_key"]: item for item in result["proposals"]}
    assert set(proposals) == {
        "milestone:相遇五百天",
        "project:haven-bridge-视觉方向",
        "topic:模型水印误报",
        "work:废墟图书馆",
    }
    assert proposals["project:haven-bridge-视觉方向"]["supporting_material_ids"] == [
        "event_9cc047ffaf3feee74f89e624",
        "event_c90b6571382b073f276f9c9e",
    ]
    assert proposals["work:废墟图书馆"]["supporting_material_ids"] == ["work_a", "work_b"]
    assert all(item["admission_decision"] == "not_run" for item in result["proposals"])
    assert all(item["publication_status"] == "not_published" for item in result["proposals"])
    assert result["writes_performed"] == []

    rejected = {item["seed_id"]: item["reason"] for item in result["rejected_seed_receipts"]}
    assert rejected["event_081782a42026da9320df69c3"] == "parked_source"
    assert rejected["event_9f1d433ef0d03198bc57702c"] == "event_rewrite_required"
    assert rejected["event_7e24ef65a52007f403d556b4"] == "intimacy_excluded"
    assert not any("intimacy" in key or "亲密" in key for key in proposals)
    assert not any("刘枭" in key or "haven-和小雨" in key for key in proposals)

    # One explicit-work seed and its fully overlapping representation are one material, not an Arc.
    overlap = propose_dynamic_arc_keys(
        [
            node("overlap_event", "看作品", "《孤独作品》", sources=(11800, 11801)),
            node(
                "overlap_scene",
                "同一段 Scene",
                "《孤独作品》",
                sources=(11800, 11801),
                kind="scene",
            ),
        ]
    )
    assert overlap["proposals"] == []
    assert {
        item["status"] for item in overlap["proposal_receipts"]
    } == {"insufficient_independent_materials"}

    assert result["proposal_contract"] == {
        "minimum_independent_materials": 2,
        "one_hop": True,
        "rare_single_can_mint": False,
        "person_can_mint": False,
        "embedding_can_mint": False,
        "date_can_mint": False,
        "automatic_admission": False,
        "automatic_publication": False,
    }
    assert all(item["one_hop"] for item in result["proposal_receipts"])
    assert not any(item["embedding_used"] or item["date_used"] for item in result["proposal_receipts"])

    # B may directly support two different work keys, but cannot transitively merge A and C.
    one_hop = propose_dynamic_arc_keys(
        [
            node("A", "甲作品", "《甲作品》", sources=(11901,)),
            node("B", "两本作品的对照", "《甲作品》和《乙作品》", sources=(11902,)),
            node("C", "乙作品", "《乙作品》", sources=(11903,)),
        ]
    )
    one_hop_sets = {
        item["arc_key"]: set(item["supporting_material_ids"]) for item in one_hop["proposals"]
    }
    assert one_hop_sets == {
        "work:乙作品": {"B", "C"},
        "work:甲作品": {"A", "B"},
    }
    assert not any(set(item["supporting_material_ids"]) == {"A", "B", "C"} for item in one_hop["proposals"])

    shuffled = list(fixtures)
    random.Random(29).shuffle(shuffled)
    assert stable_projection(result) == stable_projection(
        propose_dynamic_arc_keys(
            shuffled,
            reviewed_anchors=(visual_anchor, milestone_anchor, watermark_anchor),
        )
    )
    print("PASS: dynamic shadow Arc key proposals are bounded and non-admitting")


if __name__ == "__main__":
    main()
