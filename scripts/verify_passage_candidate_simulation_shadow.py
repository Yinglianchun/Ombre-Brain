"""Verify the source-balanced passage pool stays simulation-only."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway import GatewayService


def row(kind: str, item_id: str, score: float) -> dict:
    return {
        "owner_kind": kind,
        "owner_id": item_id,
        "score": score,
        "passages": [{"text": f"evidence:{item_id}", "score": score}],
    }


class PassageIndex:
    def search_by_embedding(self, *_args, **_kwargs):
        return {"status": "ok", "candidate_count": 2, "matches": [
            row("scene", "scene-a", 0.91),
            row("scene", "scene-b", 0.89),
        ]}


class CuePassageIndex:
    def search_by_embedding(self, *_args, **_kwargs):
        match = row("scene", "scene-a", 0.95)
        match["matched_cues"] = ["初遇"]
        return {"status": "ok", "candidate_count": 1, "matches": [match]}


class FactSemanticIndex:
    def search_by_embedding(self, *_args, **kwargs):
        assert kwargs["min_importance"] == 3
        return {"status": "ok", "candidate_count": 2, "matches": [
            {"memory_kind": "fact", "memory_id": "fact-a", "score": 0.82, "importance": 4},
            {"memory_kind": "event", "memory_id": "event-a", "score": 0.78, "importance": 3},
        ]}


class LexicalIndex:
    def search(self, *_args, **_kwargs):
        match = row("fact", "fact-a", 4.2)
        match.update({
            "importance": 4,
            "candidate_sources": ["fact_event_lexical"],
            "specific_terms": ["初遇"],
        })
        return {"status": "ok", "candidate_count": 1, "matches": [match]}


service = GatewayService.__new__(GatewayService)
service.passage_candidate_shadow_enabled = True
service.passage_shadow_min_fact_event_importance = 3
service.passage_shadow_index = PassageIndex()
service.cue_passage_shadow_index = CuePassageIndex()
service.fact_event_semantic_index = FactSemanticIndex()
service.fact_event_lexical_shadow_index = LexicalIndex()
service._passage_candidate_shadow_sync = {"status": "ok", "decision_applied": False}
service._passage_candidate_shadow_catalog = {
    "scene-a": {"owner_kind": "scene", "title": "A"},
    "scene-b": {"owner_kind": "scene", "title": "B"},
    "fact-a": {"owner_kind": "fact", "title": "F", "importance": 4, "body": "fact body"},
    "event-a": {"owner_kind": "event", "title": "E", "importance": 3, "body": "event body"},
}

active_scene = {
    "id": "scene-active",
    "content": "active",
    "metadata": {"object_kind": "scene", "active": True, "scene_status": "active"},
}
assert service._passage_shadow_scene(active_scene) is not None
for dirty_archive_metadata in (
    {"object_kind": "scene", "type": "archived", "active": True},
    {"object_kind": "scene", "scene_status": "archived", "active": True},
    {"object_kind": "scene", "type": "dynamic", "active": False},
):
    assert service._passage_shadow_scene({
        "id": "scene-archived",
        "content": "archived",
        "metadata": dirty_archive_metadata,
    }) is None

debug = service._passage_candidate_shadow_debug("我们的初遇", [1.0, 0.0])
assert debug["status"] == "ok"
assert debug["decision_applied"] is False
assert debug["live_injection_enabled"] is False
assert debug["policy"]["duplicate_score_boost"] is False
assert len(debug["candidates"]) == 4
scene_a = next(item for item in debug["candidates"] if item["owner_id"] == "scene-a")
assert scene_a["candidate_lane"] == "cue_passage"
assert scene_a["candidate_sources"] == ["cue_passage_embedding", "passage_embedding"]
fact_a = next(item for item in debug["candidates"] if item["owner_id"] == "fact-a")
assert fact_a["candidate_sources"] == [
    "fact_event_body_embedding",
    "fact_event_lexical",
]

print("PASSAGE_CANDIDATE_SIMULATION_SHADOW_OK")
