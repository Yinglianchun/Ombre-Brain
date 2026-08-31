from __future__ import annotations

import json
import re
import sys
import tempfile
from contextlib import closing
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_recall.observed_entities import (
    ObservedEntityShadowIndex,
    extract_observed_entities,
)
from memory_recall import fact_event_lexical_shadow as lexical_module
from memory_recall.typed_candidate_shadow import (
    balanced_typed_pool,
    build_event_lane,
    build_scene_lane,
)


def candidate(kind: str, owner_id: str, score: float) -> dict:
    return {
        "owner_kind": kind,
        "owner_id": owner_id,
        "score": score,
        "passages": [{"ordinal": 0, "text": owner_id, "score": score}],
    }


def source(message_id: str, content: str) -> dict:
    return {
        "source_system": "haven_bridge",
        "session_id": "fixture",
        "message_id": message_id,
        "content": content,
    }


def verify_typed_lanes() -> None:
    scene_lane = build_scene_lane(
        [candidate("scene", "scene-a", 0.71), candidate("scene", "scene-b", 0.69)],
        [
            {
                **candidate("scene", "scene-a", 0.98),
                "matched_cues": ["第一次共同追番"],
            },
            {
                **candidate("scene", "scene-cue-only", 0.96),
                "matched_cues": ["隐晦的未来问法"],
            },
        ],
        [candidate("scene", "scene-a", 0.76)],
    )
    scene_a = next(row for row in scene_lane if row["owner_id"] == "scene-a")
    cue_only = next(row for row in scene_lane if row["owner_id"] == "scene-cue-only")
    assert scene_a["score"] == 0.76, scene_a
    assert scene_a["candidate_sources"] == [
        "scene_passage_embedding",
        "scene_whole_embedding",
        "scene_cue_candidate",
    ], scene_a
    assert scene_a["score_components"] == {
        "scene_passage_embedding": 0.71,
        "scene_whole_embedding": 0.76,
    }, scene_a
    assert cue_only["score"] is None, cue_only
    assert cue_only["passages"] == [], cue_only

    event_lane = build_event_lane(
        [candidate("event", "event-low-importance", 0.83)],
        [candidate("event", "event-low-importance", 0.74)],
        [{**candidate("event", "event-lexical-only", 8.7), "specific_terms": ["作品名"]}],
    )
    event = next(row for row in event_lane if row["owner_id"] == "event-low-importance")
    lexical_only = next(row for row in event_lane if row["owner_id"] == "event-lexical-only")
    assert event["score"] == 0.83, event
    assert event["score_components"] == {
        "event_passage_embedding": 0.83,
        "event_whole_embedding": 0.74,
    }, event
    assert lexical_only["score"] is None, lexical_only

    pool = balanced_typed_pool(
        [("scene", scene_lane, 2), ("event", event_lane, 2)],
        limit=4,
    )
    assert {row["candidate_lane"] for row in pool} == {"scene", "event"}, pool
    assert all("lane_rank" in row for row in pool), pool


def verify_lexical_kind_floor() -> None:
    original = lexical_module.lexical_terms
    lexical_module.lexical_terms = lambda text: re.findall(r"[a-z]{2,}", str(text).lower())
    try:
        with tempfile.TemporaryDirectory(prefix="typed-lexical-shadow-") as temp_dir:
            index = lexical_module.FactEventLexicalShadowIndex(
                {
                    "state_dir": temp_dir,
                    "buckets_dir": str(Path(temp_dir) / "buckets"),
                    "fact_event_lexical_shadow": {"max_df_ratio": 1.0},
                }
            )
            index.sync(
                [
                    {
                        "item_id": "event-low",
                        "item_type": "event",
                        "title": "anchor event",
                        "body": "anchor detail",
                        "importance": 1,
                        "status": "active",
                    },
                    {
                        "item_id": "fact-low",
                        "item_type": "fact",
                        "body": "anchor fact",
                        "importance": 1,
                        "status": "active",
                    },
                ],
                min_importance=1,
            )
            result = index.search(
                "anchor",
                min_importance_by_kind={"event": 1, "fact": 3},
            )
            assert [row["owner_id"] for row in result["matches"]] == ["event-low"], result
            scoped = index.search(
                "anchor",
                min_importance_by_kind={"event": 1, "fact": 1},
                allowed_memory_ids={"fact-low"},
            )
            assert [row["owner_id"] for row in scoped["matches"]] == ["fact-low"], scoped
    finally:
        lexical_module.lexical_terms = original


def verify_observed_entities_and_scope() -> None:
    fragment_rows = extract_observed_entities(
        [source("fragment", "阿尼亚抱着玩偶，后来阿尼亚又出现了。")],
        known_terms={},
        stop_keys=frozenset(),
    )
    fragment = next(
        (row for row in fragment_rows if row["entity_text"] == "尼亚"),
        None,
    )
    if fragment is not None:
        assert fragment["scope_eligible"] is False, fragment

    with tempfile.TemporaryDirectory(prefix="observed-entity-shadow-") as temp_dir:
        config = {
            "state_dir": str(Path(temp_dir) / "state"),
            "buckets_dir": str(Path(temp_dir) / "buckets"),
        }
        index = ObservedEntityShadowIndex(config)
        arcs = [
            {
                "arc_key": "work:spy-family",
                "title": "《间谍过家家》共同观看",
                "title_aliases": ["间谍过家家"],
                "primary_entities": ["阿尼亚"],
                "supporting_entities": ["约尔"],
                "members": [
                    {"owner_kind": "scene", "owner_id": "scene-spy"},
                    {"owner_kind": "event", "owner_id": "event-spy"},
                    {"owner_kind": "event", "owner_id": "event-loner"},
                ],
            },
            {
                "arc_key": "project:yor",
                "title": "约尔角色研究",
                "title_aliases": [],
                "primary_entities": [],
                "supporting_entities": ["约尔"],
                "members": [{"owner_kind": "scene", "owner_id": "scene-yor"}],
            },
        ]
        owners = [
            {
                "owner_kind": "scene",
                "owner_id": "scene-spy",
                "source_refs": [source("1", "“奇美拉”玩偶出现了，后来又提到“奇美拉”。“正确”又被说成“正确”。“在本地的codex好萌啊TT”这句话又写成“在本地的codex好萌啊TT”。")],
            },
            {
                "owner_kind": "event",
                "owner_id": "event-spy",
                "source_refs": [source("2", "阿尼亚抱着“奇美拉”，还说最喜欢奇美拉。“正确”再次被说成“正确”。“在本地的codex好萌啊TT”被重复成“在本地的codex好萌啊TT”。")],
            },
            {
                "owner_kind": "event",
                "owner_id": "event-loner",
                "source_refs": [source("3", "“邦德曼”登场，大家又说起邦德曼。")],
            },
            {
                "owner_kind": "scene",
                "owner_id": "scene-yor",
                "source_refs": [source("4", "约尔出场了，约尔开始行动。")],
            },
            {
                "owner_kind": "event",
                "owner_id": "event-new",
                "source_refs": [
                    source("5", "继续看《间谍过家家》，阿尼亚拿着“奇美拉”。"),
                    source("6", "阿尼亚又抱起“奇美拉”。"),
                ],
            },
        ]
        dry_run = index.sync(owners=owners, arc_profiles=arcs, dry_run=True)
        assert dry_run["canonical_writes"] is False, dry_run
        result = index.sync(owners=owners, arc_profiles=arcs)
        assert result["status"] == "ok", result
        assert result["owners_extracted"] == len(owners), result
        assert result["owners_unchanged"] == 0, result
        assert result["decision_applied"] is False, result
        assert result["canonical_writes"] is False, result
        assert result["source_text_queryable"] is False, result

        unrelated = {
            row["entity"] for row in index.owner_entities("event", "event-loner")
        }
        assert "间谍过家家" not in unrelated, unrelated
        assert "SPY×FAMILY" not in unrelated, unrelated

        no_op = index.sync(owners=owners, arc_profiles=arcs)
        assert no_op["owners_extracted"] == 0, no_op
        assert no_op["owners_unchanged"] == len(owners), no_op
        assert no_op["owners_deleted"] == 0, no_op
        assert all(
            receipt["inserted"] == receipt["updated"] == receipt["deleted"] == 0
            for receipt in no_op["reconciliation"].values()
        ), no_op

        observed = index.owner_entities("event", "event-new")
        by_entity = {row["entity"]: row for row in observed}
        assert by_entity["阿尼亚"]["occurrence_count"] == 2, observed
        assert by_entity["奇美拉"]["source_count"] == 2, observed
        assert by_entity["奇美拉"]["scope_eligible"] is True, observed
        assert by_entity["间谍过家家"]["confidence_basis"] == "explicit_work_title", observed

        links = index.link_candidates("event", "event-new")
        spy = next(row for row in links if row["arc_key"] == "work:spy-family")
        signal_kinds = {row["kind"] for row in spy["signals"]}
        assert "authored_title_alias" in signal_kinds, spy
        assert "repeated_observed_entity" in signal_kinds, spy
        assert spy["admission_eligible"] is False, spy

        scoped = index.resolve_query("阿尼亚后来怎么发展")
        assert scoped["status"] == "scoped_recall", scoped
        assert scoped["scope_anchor"]["arc_key"] == "work:spy-family", scoped
        assert scoped["operator"] == "timeline", scoped
        assert scoped["retrieval_allowed"] is True, scoped

        entity_only = index.resolve_query("阿尼亚")
        assert entity_only["status"] == "scope_only", entity_only
        assert entity_only["retrieval_allowed"] is False, entity_only

        no_entity = index.resolve_query("看到哪了")
        assert no_entity["status"] == "insufficient_scope", no_entity
        assert no_entity["operator"] == "latest_relevant_member", no_entity

        ambiguous = index.resolve_query("约尔后来怎么发展")
        assert ambiguous["status"] == "ambiguous_scope", ambiguous
        assert ambiguous["retrieval_allowed"] is False, ambiguous

        aggregated = index.resolve_query("奇美拉后来怎么发展")
        assert aggregated["status"] == "scoped_recall", aggregated
        assert aggregated["scope_anchor"]["source_kind"] == "observed", aggregated

        long_quote = index.resolve_query("在本地的codex好萌啊TT后来怎么发展")
        assert long_quote["status"] == "insufficient_scope", long_quote

        quoted_predicate = index.resolve_query("正确后来怎么发展")
        assert quoted_predicate["status"] == "insufficient_scope", quoted_predicate

        self_proof = index.resolve_query("邦德曼后来怎么发展")
        assert self_proof["status"] == "insufficient_scope", self_proof
        assert self_proof["retrieval_allowed"] is False, self_proof

        generic = index.resolve_query("哥哥后来怎么发展")
        assert generic["status"] == "insufficient_scope", generic
        assert generic["scope_anchor"] is None, generic

        gold = json.loads(
            (ROOT / "resources" / "typed_recall_scope_gold_v1.json").read_text(
                encoding="utf-8"
            )
        )
        passed = 0
        for case in gold["cases"]:
            actual = index.resolve_query(case["query"])
            actual_arc_key = str((actual.get("scope_anchor") or {}).get("arc_key") or "")
            receipt = {
                "status": actual.get("status"),
                "intent": actual.get("intent"),
                "operator": actual.get("operator"),
                "arc_key": actual_arc_key,
                "retrieval_allowed": actual.get("retrieval_allowed"),
            }
            expected = {
                "status": case["expected_status"],
                "intent": case["expected_intent"],
                "operator": case["expected_operator"],
                "arc_key": case["expected_arc_key"],
                "retrieval_allowed": case["expected_retrieval_allowed"],
            }
            assert receipt == expected, {"case": case["id"], "actual": receipt, "expected": expected}
            passed += 1
        assert passed == len(gold["cases"]), {"passed": passed, "total": len(gold["cases"])}

        changed_owners = json.loads(json.dumps(owners, ensure_ascii=False))
        changed_event = next(
            owner for owner in changed_owners if owner["owner_id"] == "event-new"
        )
        changed_event["source_refs"].append(source("7", "奇美拉又被阿尼亚抱了一次。"))
        changed = index.sync(owners=changed_owners, arc_profiles=arcs)
        assert changed["owners_extracted"] == 1, changed
        assert changed["owners_unchanged"] == len(owners) - 1, changed
        changed_entities = {
            row["entity"]: row for row in index.owner_entities("event", "event-new")
        }
        assert changed_entities["奇美拉"]["source_count"] == 3, changed_entities

        owners_without_loner = [
            owner for owner in changed_owners if owner["owner_id"] != "event-loner"
        ]
        removed = index.sync(owners=owners_without_loner, arc_profiles=arcs)
        assert removed["owners_extracted"] == 0, removed
        assert removed["owners_deleted"] == 1, removed
        assert index.owner_entities("event", "event-loner") == [], removed

        membership_changed_arcs = json.loads(json.dumps(arcs, ensure_ascii=False))
        membership_changed_arcs[0]["members"].append(
            {"owner_kind": "event", "owner_id": "event-new"}
        )
        membership_changed = index.sync(
            owners=owners_without_loner,
            arc_profiles=membership_changed_arcs,
        )
        assert membership_changed["owners_extracted"] == 0, membership_changed
        assert membership_changed["owners_unchanged"] == len(owners_without_loner), (
            membership_changed
        )
        assert membership_changed["reconciliation"]["arc_observed_entities"][
            "updated"
        ] > 0, membership_changed


def verify_legacy_owner_cleanup() -> None:
    with tempfile.TemporaryDirectory(prefix="observed-entity-legacy-") as temp_dir:
        index = ObservedEntityShadowIndex(
            {
                "state_dir": str(Path(temp_dir) / "state"),
                "buckets_dir": str(Path(temp_dir) / "buckets"),
            }
        )
        index._init_db()
        with closing(index._connect()) as conn:
            conn.execute(
                """
                INSERT INTO observed_entities(
                    owner_kind, owner_id, entity_key, entity_text,
                    occurrence_count, source_count, confidence_basis,
                    known_roles_json, known_arc_keys_json, supports_json, updated_at
                ) VALUES ('event', 'legacy-event', 'legacy', 'legacy', 2, 1,
                          'repeated_bound_source', '[]', '[]', '[]', 'old')
                """
            )
            conn.commit()
        migrated = index.sync(owners=[], arc_profiles=[])
        assert migrated["owners_deleted"] == 1, migrated
        assert index.owner_entities("event", "legacy-event") == [], migrated


if __name__ == "__main__":
    verify_typed_lanes()
    verify_lexical_kind_floor()
    verify_observed_entities_and_scope()
    verify_legacy_owner_cleanup()
    print("TYPED_EVENT_SCENE_ENTITY_SHADOW_OK GOLD=9/9")
