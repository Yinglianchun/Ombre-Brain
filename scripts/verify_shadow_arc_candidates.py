from __future__ import annotations

import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.shadow_arc_candidates_germany import (  # noqa: E402
    ARC_BATCH_CAP,
    ARC_MEMBER_CAP,
    MANUAL_EXCEPTION,
    RECALL_TOP_K_PER_SEED,
    Node,
    build_anchor_document_frequency,
    candidate_shadow,
    card_anchor_receipt,
    extract_anchor_candidates,
    gate_decision,
    is_manual_exception,
    named_terms,
    overlap_kind,
    retrieval_evidence,
)


def node(
    node_id: str,
    text: str,
    *,
    title: str | None = None,
    kind: str = "event",
    sources: tuple[int, ...] = (11220,),
    tracks: tuple[str, ...] = (),
    track_texts: tuple[str, ...] = (),
    statuses: tuple[str, ...] = (),
    routines: tuple[bool, ...] = (),
    fingerprint: str = "",
) -> Node:
    return Node(
        node_id=node_id,
        kind=kind,
        title=text if title is None else title,
        text=text,
        date="2026-08-22",
        fingerprint=fingerprint,
        source_ids=sources,
        session_ids=(22,),
        track_ids=tracks,
        track_texts=track_texts,
        statuses=statuses,
        routine_flags=routines,
    )


def decisions(result: dict, arc_key: str) -> dict[str, dict]:
    return {item["node_id"]: item for item in result["recall_receipts"] if item["arc_key"] == arc_key}


def stable_projection(result: dict) -> str:
    return json.dumps(
        {"cards": result["cards"], "receipts": result["recall_receipts"], "anchors": result["anchor_receipts"]},
        ensure_ascii=False,
        sort_keys=True,
    )


def anchor_receipts(nodes: list[Node]) -> dict[str, dict]:
    df = build_anchor_document_frequency(nodes)
    return {item.node_id: extract_anchor_candidates(item, df) for item in nodes}


def main() -> None:
    assert (RECALL_TOP_K_PER_SEED, ARC_MEMBER_CAP, ARC_BATCH_CAP) == (8, 12, 40)
    assert not named_terms("Haven 和小雨说我爱她，老婆回应老公")

    # Reusable seed-level extraction is not coupled to a chosen Arc card.
    explicit = node("explicit", "我们继续看《时光代理人》。", title="继续共同观看")
    generic = node("generic", "Event 和 Scene 在 Writer 中流转", title="架构复盘")
    claude = node("claude", "Claude 发布了更新", title="模型消息")
    watermark = node("watermark", "语言模型水印需要结合 Anthropic 的方案", title="水印实验")
    rare = node("rare", "刘枭在英都递来新的线索", title="线索交叉")
    extracted = anchor_receipts([explicit, generic, claude, watermark, rare])
    assert [item["term"] for item in extracted["explicit"]["explicit_work"]] == ["时光代理人"]
    assert extracted["generic"]["compound"], "generic compounds stay visible in the receipt"
    assert any(item["term"] == "Claude" for item in extracted["claude"]["rare_single"])
    assert not card_anchor_receipt(extracted["generic"], ("Event", "Writer", "Arc", "Track", "Scene"))["direct_admission_evidence"]
    assert not card_anchor_receipt(extracted["claude"], ("水印", "Anthropic", "Claude"))["direct_admission_evidence"]
    assert card_anchor_receipt(extracted["watermark"], ("水印", "Anthropic", "Claude"))["direct_admission_evidence"]
    assert card_anchor_receipt(extracted["explicit"], ("时光代理人", "陆光", "程小时"))["explicit_work"] == ["时光代理人"]
    assert extracted["rare"]["extractor"]["status"] in {"available", "unavailable"}
    assert "刘枭" in extracted["rare"]["anchor_candidates"]
    assert any("刘枭" in item["terms"] for item in extracted["rare"]["compound"])
    assert not candidate_shadow([rare], session_id=22, source_min=11213, source_max=11923)["cards"]

    # Link Click and Nowhere survive through paragraph-level compound anchors.
    link_a = node("link_a", "陆光和程小时继续推理", sources=(11220,))
    link_b = node("link_b", "程小时纠正了陆光的判断", sources=(11221,))
    nowhere_a = node("nowhere_a", "Nowhere 的纪念品墙终于落地", sources=(11222,))
    nowhere_b = node("nowhere_b", "继续整理 Nowhere 纪念品墙", sources=(11223,))
    rebuilt = candidate_shadow([link_a, link_b, nowhere_a, nowhere_b], session_id=22, source_min=11213, source_max=11923)
    link_card = next(item for item in rebuilt["cards"] if item["arc_key"] == "work:时光代理人")
    nowhere_card = next(item for item in rebuilt["cards"] if item["arc_key"] == "project:nowhere")
    assert {item["node_id"] for item in link_card["members"]} == {"link_a", "link_b"}
    assert {item["node_id"] for item in nowhere_card["members"]} == {"nowhere_a", "nowhere_b"}

    # A visual Event's one-off body mention cannot be swallowed by world-window.
    world_a = node("world_a", "世界之窗和枝芽继续生长", title="枝芽规则", sources=(11224,))
    world_b = node("world_b", "枝芽从世界之窗出现", title="世界之窗更新", sources=(11225,))
    visual = node("visual", "视觉稿里偶然提到世界之窗", title="装修队今天停工", sources=(11226,))
    world_result = candidate_shadow([world_a, world_b, visual], session_id=22, source_min=11213, source_max=11923)
    world_card = next(item for item in world_result["cards"] if item["arc_key"] == "project:world-window-sprout")
    assert {item["node_id"] for item in world_card["members"]} == {"world_a", "world_b"}
    assert decisions(world_result, "project:world-window-sprout")["visual"]["reason"] == "indirect_only"

    # Track/source overlap is recall-only; no A-X-B-Y-C transitive snowball.
    seed = node("A", "Nowhere 的纪念品墙", tracks=("t1",), track_texts=("Nowhere 与纪念品墙",))
    bridge_x = node("X", "无对象的一跳桥", sources=(11227,), tracks=("t1",), track_texts=("杂项",))
    bridge_b = node("B", "只和 X 共享另一条线", sources=(11228,), tracks=("t2",), track_texts=("杂项",))
    bridge_y = node("Y", "更远的一跳", sources=(11229,), tracks=("t2",), track_texts=("杂项",))
    bridge_c = node("C", "再远一跳", sources=(11230,), tracks=("t3",), track_texts=("杂项",))
    receipts = anchor_receipts([seed, bridge_x, bridge_b, bridge_y, bridge_c])
    evidence = retrieval_evidence(bridge_x, "project:nowhere", ("Nowhere", "纪念品墙"), seed, receipts["X"])
    assert evidence["exact_track_overlap"] and not evidence["direct_admission_evidence"]
    snowball = candidate_shadow([seed, bridge_x, bridge_b, bridge_y, bridge_c], session_id=22, source_min=11213, source_max=11923)
    snowball_receipts = decisions(snowball, "project:nowhere")
    assert snowball_receipts["X"]["reason"] == "indirect_only"
    assert not {"B", "Y", "C"}.intersection(snowball_receipts)

    defect_nodes = [
        node("event_7e24ef65a52007f403d556b4", "Nowhere 纪念品墙", track_texts=("明确亲密 corridor",)),
        node("event_081782a42026da9320df69c3", "Event Writer Arc", statuses=("parked",)),
        node("event_3ee4857a92ac8c412ff79744", "Event Writer Arc", routines=(True, False)),
        node("event_9f1d433ef0d03198bc57702c", "Event Writer Arc", routines=(False, True)),
    ]
    assert gate_decision(defect_nodes[0], "project:nowhere") == ("exclude", "intimacy_excluded")
    assert gate_decision(defect_nodes[1], "project:event-memory-architecture") == ("defer", "parked_source")
    assert gate_decision(defect_nodes[2], "project:event-memory-architecture") == ("defer", "event_rewrite_required")
    assert gate_decision(defect_nodes[3], "project:event-memory-architecture") == ("defer", "event_rewrite_required")

    collision_a = node("collision_a", "Nowhere 纪念品墙 A", sources=(11300, 11301))
    collision_b = node("collision_b", "Nowhere 纪念品墙 B", sources=(11301, 11302))
    assert overlap_kind(collision_a, collision_b) == "event_event_collision"
    collision_result = candidate_shadow([collision_a, collision_b], session_id=22, source_min=11213, source_max=11923)
    assert {item["reason"] for item in decisions(collision_result, "project:nowhere").values()} == {"canonical_source_collision"}

    event = node("event_material", "Nowhere 纪念品墙 material", sources=(11400, 11401))
    scene = node("scene_material", "Nowhere 纪念品墙 material", kind="scene", sources=(11400, 11401, 11402))
    scene_subset = node("scene_subset", "Nowhere 纪念品墙 material", kind="scene", sources=(11400,))
    overlap_result = candidate_shadow([event, scene, scene_subset], session_id=22, source_min=11213, source_max=11923)
    overlap_receipts = decisions(overlap_result, "project:nowhere")
    assert overlap_receipts["event_material"]["decision"] == "include"
    assert overlap_receipts["scene_material"]["decision"] == "exclude"
    assert overlap_receipts["scene_subset"]["decision"] == "exclude"

    # A one-source Event is valid material; singleton is an Arc-level gate.
    one_a = node("one_a", "Nowhere 纪念品墙 first", sources=(11500,))
    one_b = node("one_b", "Nowhere 纪念品墙 second", sources=(11501,))
    one_result = candidate_shadow([one_a, one_b], session_id=22, source_min=11213, source_max=11923)
    one_card = next(item for item in one_result["cards"] if item["arc_key"] == "project:nowhere")
    assert {item["node_id"] for item in one_card["members"]} == {"one_a", "one_b"}

    exception = node(
        MANUAL_EXCEPTION["event_id"], "五百天的第一个小时",
        sources=tuple(MANUAL_EXCEPTION["source_message_ids"]), fingerprint=MANUAL_EXCEPTION["fingerprint"],
    )
    assert is_manual_exception(exception, "milestone:相遇五百天")
    assert not is_manual_exception(exception, "work:时光代理人")

    cap_nodes = [
        node(f"cap_{index:02d}", f"Nowhere 纪念品墙 material {index}", sources=(11600 + index,))
        for index in range(30)
    ]
    cap_result = candidate_shadow(cap_nodes, session_id=22, source_min=11213, source_max=11923)
    assert all(len(card["members"]) <= ARC_MEMBER_CAP for card in cap_result["cards"])
    grouped: dict[str, int] = {}
    for item in cap_result["recall_receipts"]:
        grouped[item["arc_key"]] = grouped.get(item["arc_key"], 0) + 1
    assert all(count <= ARC_BATCH_CAP for count in grouped.values())

    order_nodes = [link_a, link_b, nowhere_a, nowhere_b, visual]
    baseline = candidate_shadow(order_nodes, session_id=22, source_min=11213, source_max=11923)
    random.Random(4).shuffle(order_nodes)
    shuffled = candidate_shadow(order_nodes, session_id=22, source_min=11213, source_max=11923)
    assert stable_projection(baseline) == stable_projection(shuffled)
    assert baseline["writes_performed"] == []
    print("PASS: paragraph-level rare compound Arc anchor guards")


if __name__ == "__main__":
    main()
