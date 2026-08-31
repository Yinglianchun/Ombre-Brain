from __future__ import annotations

import json
import tempfile
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from narrative_rolls import NarrativeRollStore


EVENT_A = "event_111111111111111111111111"
EVENT_B = "event_222222222222222222222222"
SCENE_A = "scene_mig2_alpha"
SCENE_B = "scene_mig2_beta"


def _document(*source_ids: str) -> str:
    source_lines = "\n".join(f"- {source_id}" for source_id in source_ids)
    return (
        "# Test Arc\n\n"
        "## 第一人称叙事\n\n"
        "我把几段已经成形的经历放回同一条长线；细节仍由来源本身负责。\n\n"
        "## 来源账\n\n"
        f"{source_lines}\n"
    )


def _collection_document(*source_ids: str) -> str:
    source_lines = "\n".join(f"- {source_id}" for source_id in source_ids)
    return f"# Material collection\n\n## 材料目录\n\n{source_lines}\n"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="narrative-event-arc-") as temp_dir:
        state_dir = Path(temp_dir) / "state"
        store = NarrativeRollStore({"state_dir": str(state_dir)})

        created = store.publish(
            narrative_id="narrative_event_only_test",
            document=_document(EVENT_A, EVENT_B),
            expected_revision=0,
            title="Event-only test",
            arc_key="work:event-only-test",
            title_aliases=["事件长线测试"],
            primary_entities=["测试角色"],
            source_event_ids=[EVENT_A, EVENT_B],
        )
        assert created["status"] == "created", created
        assert created["linked_scene_ids"] == [], created
        assert created["linked_event_ids"] == [EVENT_A, EVENT_B], created
        assert created["linked_event_count"] == 2, created

        full = store.read("narrative_event_only_test")
        assert full["status"] == "ok", full
        assert full["linked_event_ids"] == [EVENT_A, EVENT_B], full
        assert "Scene/Event/raw" in full["reading_boundary"], full

        index = store.list(query="Event-only test")
        assert index["count"] == 1, index
        assert index["items"][0]["linked_event_count"] == 2, index

        exact_card = store.find_arc_cards("事件长线测试")
        assert exact_card["count"] == 1, exact_card
        assert exact_card["items"][0]["arc_key"] == "work:event-only-test", exact_card
        assert exact_card["items"][0]["narrative_available"] is True, exact_card
        assert exact_card["items"][0]["read_hint"] == "可按需读取", exact_card
        assert "body" not in exact_card["items"][0], exact_card

        entity_card = store.find_arc_cards("测试角色")
        assert entity_card["items"][0]["match_reason"] == "exact_entity", entity_card

        fuzzy_card = store.find_arc_cards("事件长线测式")
        assert fuzzy_card["count"] == 1, fuzzy_card
        assert fuzzy_card["items"][0]["match_reason"] == "fuzzy_title", fuzzy_card
        by_key = store.arc_card_by_key("work:event-only-test")
        assert by_key["status"] == "ok", by_key
        assert by_key["body_included"] is False, by_key
        assert "body" not in by_key["item"], by_key

        second_parent = store.publish(
            narrative_id="narrative_second_parent_test",
            document=_document(EVENT_A, EVENT_B),
            expected_revision=0,
            title="Second parent",
            source_event_ids=[EVENT_A, EVENT_B],
        )
        assert second_parent["status"] == "created", second_parent
        closed_parent = store.publish(
            narrative_id="narrative_closed_parent_test",
            document=_document(EVENT_A, EVENT_B),
            expected_revision=0,
            title="Closed parent",
            source_event_ids=[EVENT_A, EVENT_B],
            lifecycle="closed",
        )
        assert closed_parent["status"] == "created", closed_parent

        collecting = store.publish(
            narrative_id="narrative_event_collection_test",
            document=_collection_document(EVENT_A, EVENT_B),
            expected_revision=0,
            title="Event material collection",
            source_event_ids=[EVENT_A, EVENT_B],
            publication_status="collecting",
            arc_key="work:test-event-collection",
            parent_narrative_id="narrative_event_only_test",
        )
        assert collecting["status"] == "created", collecting
        assert collecting["body"] == "", collecting
        assert collecting["linked_event_ids"] == [EVENT_A, EVENT_B], collecting
        assert collecting["arc_key"] == "work:test-event-collection", collecting
        assert collecting["parent_narrative_id"] == "narrative_event_only_test", collecting
        assert store.read_by_arc_key("work:test-event-collection")["narrative_id"] == collecting["narrative_id"]
        collecting_index = store.list(query="Event material collection")
        assert collecting_index["items"][0]["parent_narrative_id"] == "narrative_event_only_test", collecting_index
        assert "source index" in collecting["reading_boundary"], collecting
        shadow = store.shadow_match("Event material collection", [])
        assert shadow["status"] == "not_admitted", shadow

        inherited_key = store.publish(
            narrative_id="narrative_event_collection_test",
            document=_collection_document(EVENT_A, EVENT_B),
            expected_revision=1,
            title="Event material collection",
            source_event_ids=[EVENT_A, EVENT_B],
            publication_status="collecting",
        )
        assert inherited_key["status"] == "updated", inherited_key
        assert inherited_key["arc_key"] == "work:test-event-collection", inherited_key
        assert inherited_key["parent_narrative_id"] == "narrative_event_only_test", inherited_key

        changed_key = store.publish(
            narrative_id="narrative_event_collection_test",
            document=_collection_document(EVENT_A, EVENT_B),
            expected_revision=2,
            title="Event material collection",
            source_event_ids=[EVENT_A, EVENT_B],
            publication_status="collecting",
            arc_key="work:changed-key",
        )
        assert changed_key["reason"] == "arc_key_is_stable", changed_key

        changed_parent = store.publish(
            narrative_id="narrative_event_collection_test",
            document=_collection_document(EVENT_A, EVENT_B),
            expected_revision=2,
            title="Event material collection",
            source_event_ids=[EVENT_A, EVENT_B],
            publication_status="collecting",
            parent_narrative_id="narrative_second_parent_test",
        )
        assert changed_parent["reason"] == "parent_narrative_id_is_stable", changed_parent

        invalid_parent = store.publish(
            narrative_id="narrative_invalid_parent_test",
            document=_collection_document(EVENT_A, EVENT_B),
            expected_revision=0,
            title="Invalid parent",
            source_event_ids=[EVENT_A, EVENT_B],
            publication_status="collecting",
            arc_key="work:invalid-parent-test",
            parent_narrative_id="not a narrative id",
        )
        assert invalid_parent["reason"] == "invalid_parent_narrative_id", invalid_parent

        missing_parent = store.publish(
            narrative_id="narrative_missing_parent_test",
            document=_collection_document(EVENT_A, EVENT_B),
            expected_revision=0,
            title="Missing parent",
            source_event_ids=[EVENT_A, EVENT_B],
            publication_status="collecting",
            arc_key="work:missing-parent-test",
            parent_narrative_id="narrative_parent_does_not_exist",
        )
        assert missing_parent["reason"] == "parent_narrative_not_found", missing_parent

        inactive_parent = store.publish(
            narrative_id="narrative_inactive_parent_child_test",
            document=_collection_document(EVENT_A, EVENT_B),
            expected_revision=0,
            title="Inactive parent child",
            source_event_ids=[EVENT_A, EVENT_B],
            publication_status="collecting",
            arc_key="work:inactive-parent-test",
            parent_narrative_id="narrative_closed_parent_test",
        )
        assert inactive_parent["reason"] == "parent_narrative_not_active", inactive_parent

        self_parent = store.publish(
            narrative_id="narrative_self_parent_test",
            document=_collection_document(EVENT_A, EVENT_B),
            expected_revision=0,
            title="Self parent",
            source_event_ids=[EVENT_A, EVENT_B],
            publication_status="collecting",
            arc_key="work:self-parent-test",
            parent_narrative_id="narrative_self_parent_test",
        )
        assert self_parent["reason"] == "parent_narrative_cannot_be_self", self_parent

        duplicate_key = store.publish(
            narrative_id="narrative_duplicate_arc_key_test",
            document=_collection_document(EVENT_A, EVENT_B),
            expected_revision=0,
            title="Duplicate Arc key",
            source_event_ids=[EVENT_A, EVENT_B],
            publication_status="collecting",
            arc_key="work:test-event-collection",
        )
        assert duplicate_key["reason"] == "arc_key_already_exists", duplicate_key

        missing_key = store.publish(
            narrative_id="narrative_missing_arc_key_test",
            document=_collection_document(EVENT_A, EVENT_B),
            expected_revision=0,
            title="Missing Arc key",
            source_event_ids=[EVENT_A, EVENT_B],
            publication_status="collecting",
        )
        assert missing_key["reason"] == "arc_key_required_for_collecting", missing_key

        bodyless_reviewed = store.publish(
            narrative_id="narrative_bodyless_reviewed_test",
            document=_collection_document(EVENT_A, EVENT_B),
            expected_revision=0,
            title="Bodyless reviewed test",
            source_event_ids=[EVENT_A, EVENT_B],
            publication_status="reviewed",
        )
        assert bodyless_reviewed["reason"] == "missing_first_person_body", bodyless_reviewed

        one_source = store.publish(
            narrative_id="narrative_one_source_test",
            document=_document(EVENT_A),
            expected_revision=0,
            title="One source test",
            source_scene_ids=[],
            source_event_ids=[EVENT_A],
        )
        assert one_source["reason"] == "at_least_two_sources_required", one_source

        missing_source = store.publish(
            narrative_id="narrative_missing_source_test",
            document=_document(EVENT_A),
            expected_revision=0,
            title="Missing source test",
            source_scene_ids=[],
            source_event_ids=[EVENT_A, EVENT_B],
        )
        assert missing_source["reason"] == "source_event_id_missing_from_document", missing_source
        assert missing_source["event_ids"] == [EVENT_B], missing_source

        scene_legacy = store.publish(
            narrative_id="narrative_scene_legacy_test",
            document=_document(SCENE_A, SCENE_B),
            expected_revision=0,
            title="Scene legacy test",
            source_scene_ids=[SCENE_A, SCENE_B],
        )
        assert scene_legacy["status"] == "created", scene_legacy
        assert scene_legacy["linked_scene_ids"] == [SCENE_A, SCENE_B], scene_legacy
        assert scene_legacy["linked_event_ids"] == [], scene_legacy
        assert scene_legacy["arc_key"] == "", scene_legacy

        registry = json.loads(store.registry_path.read_text(encoding="utf-8"))
        legacy_entry = next(
            row
            for row in registry["rolls"]
            if row["narrative_id"] == "narrative_scene_legacy_test"
        )
        legacy_entry.pop("arc_key", None)
        legacy_entry.pop("parent_narrative_id", None)
        store.registry_path.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        store._cache_stamp = None
        legacy_without_key = store.read("narrative_scene_legacy_test")
        assert legacy_without_key["status"] == "ok", legacy_without_key
        assert legacy_without_key["arc_key"] == "", legacy_without_key
        assert legacy_without_key["parent_narrative_id"] == "", legacy_without_key

    print("PASS: Narrative Rolls support Event-only arcs and preserve Scene sources")


if __name__ == "__main__":
    main()
