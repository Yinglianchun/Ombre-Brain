from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import sys
import tempfile
from contextlib import closing
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_recall.fact_event_semantic import FactEventSemanticIndex
from memory_recall.passage_shadow import PassageShadowIndex
from narrative_arc_rank import NarrativeArcMemberRanker, normalize_arc_rank_request
from narrative_rolls import NarrativeRollStore


EVENT_A = "event_111111111111111111111111"
EVENT_MISSING = "event_222222222222222222222222"
EVENT_OTHER_SAME_TRACK = "event_333333333333333333333333"
SCENE_A = "scene_mig2_arc_a"
SCENE_MISSING = "scene_mig2_arc_missing"
SCENE_OTHER_ARC = "scene_mig2_arc_b"


class FakeEmbeddingEngine:
    enabled = True
    model = "arc-rank-test-v1"
    document_instruction = ""

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [1.0, 0.0]


def _document(*source_ids: str) -> str:
    return "# Arc\n\n## 材料目录\n\n" + "\n".join(f"- {value}" for value in source_ids) + "\n"


def _snapshot(paths: list[Path]) -> dict[str, tuple[int, str]]:
    return {
        str(path): (
            path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in paths
    }


def _seed_event_index(index: FactEventSemanticIndex) -> None:
    index._init_db()
    with closing(sqlite3.connect(index.db_path)) as conn:
        for memory_id, vector in (
            (EVENT_A, [1.0, 0.0]),
            (EVENT_OTHER_SAME_TRACK, [1.0, 0.0]),
        ):
            conn.execute(
                """
                INSERT INTO fact_event_embeddings(
                    item_id, item_type, source_hash, embedding, model, dimension,
                    importance, local_date, local_start_time, covered_by_scene_id, updated_at
                ) VALUES (?, 'event', 'hash', ?, ?, 2, 5, '', '', '', 'now')
                """,
                (memory_id, json.dumps(vector), index.embedding_engine.model),
            )
        conn.commit()


def _seed_passage_index(index: PassageShadowIndex) -> None:
    index._init_db()
    with closing(sqlite3.connect(index.db_path)) as conn:
        for owner_kind, owner_id, vector in (
            ("event", EVENT_A, [0.8, 0.2]),
            ("event", EVENT_OTHER_SAME_TRACK, [1.0, 0.0]),
            ("scene", SCENE_A, [0.8, 0.2]),
            ("scene", SCENE_OTHER_ARC, [1.0, 0.0]),
        ):
            conn.execute(
                """
                INSERT INTO memory_passage_embeddings(
                    owner_kind, owner_id, ordinal, source_hash, content_hash,
                    start_offset, end_offset, text, embedding, model, dimension, updated_at
                ) VALUES (?, ?, 0, 'hash', 'content', 0, 4, 'test', ?, ?, 2, 'now')
                """,
                (owner_kind, owner_id, json.dumps(vector), index.embedding_engine.model),
            )
        conn.commit()


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="narrative-arc-rank-") as temp_dir:
        state_dir = Path(temp_dir) / "state"
        config = {
            "state_dir": str(state_dir),
            "buckets_dir": str(Path(temp_dir) / "buckets"),
            "gateway": {"fact_event_recall_shadow_enabled": True},
        }
        store = NarrativeRollStore(config)
        arc_b = store.publish(
            narrative_id="narrative_arc_rank_b",
            document=_document(EVENT_OTHER_SAME_TRACK, SCENE_OTHER_ARC),
            expected_revision=0,
            title="Arc B",
            arc_key="work:arc-b",
            source_event_ids=[EVENT_OTHER_SAME_TRACK],
            source_scene_ids=[SCENE_OTHER_ARC],
            publication_status="collecting",
        )
        assert arc_b["status"] == "created", arc_b
        arc_a = store.publish(
            narrative_id="narrative_arc_rank_a",
            document=_document(EVENT_A, EVENT_MISSING, SCENE_A, SCENE_MISSING),
            expected_revision=0,
            title="Arc A",
            arc_key="work:arc-a",
            parent_narrative_id="narrative_arc_rank_b",
            source_event_ids=[EVENT_A, EVENT_MISSING],
            source_scene_ids=[SCENE_A, SCENE_MISSING],
            publication_status="collecting",
        )
        assert arc_a["status"] == "created", arc_a
        assert arc_a["parent_narrative_id"] == "narrative_arc_rank_b", arc_a

        engine = FakeEmbeddingEngine()
        engine.search_scene_whole_by_embedding = lambda _vector, *, scene_ids, top_k: (
            [{"scene_id": SCENE_A, "score": 0.99}]
            if SCENE_A in scene_ids
            else []
        )[:top_k]
        event_index = FactEventSemanticIndex(config, engine)
        passage_index = PassageShadowIndex(config, engine)
        _seed_event_index(event_index)
        _seed_passage_index(passage_index)
        ranker = NarrativeArcMemberRanker(store, engine, event_index, passage_index)

        tracked_paths = [
            store.registry_path,
            *sorted(store.documents_dir.rglob("*.md")),
            Path(event_index.db_path),
            Path(passage_index.db_path),
        ]
        before = _snapshot(tracked_paths)

        unknown = await ranker.rank(arc_key="work:unknown", query="same topic", top_k=8)
        assert unknown["status"] == "not_found", unknown
        assert engine.queries == [], engine.queries

        result = await ranker.rank(arc_key="work:arc-a", query="same topic", top_k=8)
        assert result["status"] == "ok", result
        ranked_ids = [row["memory_id"] for row in result["ranked_members"]]
        assert ranked_ids == [EVENT_A, SCENE_A], result
        assert EVENT_OTHER_SAME_TRACK not in ranked_ids, result
        assert SCENE_OTHER_ARC not in ranked_ids, result
        assert result["unindexed_members"] == [
            {"memory_kind": "event", "memory_id": EVENT_MISSING},
            {"memory_kind": "scene", "memory_id": SCENE_MISSING},
        ], result
        assert all(
            row["selected_embedding_route"] == "whole"
            for row in result["ranked_members"]
        ), result
        assert result["within_owner_embedding_score"] == "max", result
        assert result["membership_changed"] is False, result
        assert result["index_write_attempted"] is False, result

        repeated = await ranker.rank(arc_key="work:arc-a", query="same topic", top_k=8)
        assert repeated["ranked_members"] == result["ranked_members"], repeated
        assert _snapshot(tracked_paths) == before

        forbidden_members = normalize_arc_rank_request(
            {"arc_key": "work:arc-a", "query": "same topic", "event_ids": [EVENT_OTHER_SAME_TRACK]}
        )
        assert forbidden_members == {
            "status": "invalid",
            "reason": "unexpected_request_fields",
            "fields": ["event_ids"],
        }, forbidden_members
        assert normalize_arc_rank_request(
            {"arc_key": "work:arc-a", "query": "same topic", "top_k": 9}
        )["reason"] == "top_k_must_be_between_1_and_8"

    with tempfile.TemporaryDirectory(prefix="narrative-arc-rank-missing-") as temp_dir:
        state_dir = Path(temp_dir) / "state"
        config = {
            "state_dir": str(state_dir),
            "buckets_dir": str(Path(temp_dir) / "buckets"),
            "gateway": {"fact_event_recall_shadow_enabled": True},
        }
        store = NarrativeRollStore(config)
        created = store.publish(
            narrative_id="narrative_arc_missing_scene_index",
            document=_document(SCENE_A, SCENE_MISSING),
            expected_revision=0,
            title="Missing Scene index",
            arc_key="work:missing-scene-index",
            source_scene_ids=[SCENE_A, SCENE_MISSING],
            publication_status="collecting",
        )
        assert created["status"] == "created", created
        engine = FakeEmbeddingEngine()
        event_index = FactEventSemanticIndex(config, engine)
        passage_index = PassageShadowIndex(config, engine)
        ranker = NarrativeArcMemberRanker(store, engine, event_index, passage_index)
        tracked_paths = [store.registry_path, *sorted(store.documents_dir.rglob("*.md"))]
        before = _snapshot(tracked_paths)
        assert not Path(passage_index.db_path).exists()
        unavailable = await ranker.rank(
            arc_key="work:missing-scene-index",
            query="same topic",
            top_k=8,
        )
        assert unavailable == {
            "status": "unavailable",
            "lane": "scene",
            "reason": "index_missing",
            "arc_key": "work:missing-scene-index",
        }, unavailable
        assert engine.queries == [], engine.queries
        assert not Path(passage_index.db_path).exists()
        assert _snapshot(tracked_paths) == before

    with tempfile.TemporaryDirectory(prefix="narrative-arc-rank-disabled-") as temp_dir:
        state_dir = Path(temp_dir) / "state"
        config = {
            "state_dir": str(state_dir),
            "buckets_dir": str(Path(temp_dir) / "buckets"),
            "gateway": {"fact_event_recall_shadow_enabled": False},
        }
        store = NarrativeRollStore(config)
        created = store.publish(
            narrative_id="narrative_arc_disabled_event_index",
            document=_document(EVENT_A, EVENT_MISSING),
            expected_revision=0,
            title="Disabled Event index",
            arc_key="work:disabled-event-index",
            source_event_ids=[EVENT_A, EVENT_MISSING],
            publication_status="collecting",
        )
        assert created["status"] == "created", created
        engine = FakeEmbeddingEngine()
        event_index = FactEventSemanticIndex(config, engine)
        passage_index = PassageShadowIndex(config, engine)
        ranker = NarrativeArcMemberRanker(store, engine, event_index, passage_index)
        tracked_paths = [store.registry_path, *sorted(store.documents_dir.rglob("*.md"))]
        before = _snapshot(tracked_paths)
        assert not Path(event_index.db_path).exists()
        unavailable = await ranker.rank(
            arc_key="work:disabled-event-index",
            query="same topic",
            top_k=8,
        )
        assert unavailable == {
            "status": "unavailable",
            "lane": "event",
            "reason": "index_missing",
            "arc_key": "work:disabled-event-index",
        }, unavailable
        assert engine.queries == [], engine.queries
        assert not Path(event_index.db_path).exists()
        assert _snapshot(tracked_paths) == before

    print("PASS: bounded Arc member rank is registry-scoped and mutation-free")


if __name__ == "__main__":
    asyncio.run(main())
