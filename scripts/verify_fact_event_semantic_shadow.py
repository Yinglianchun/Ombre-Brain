from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fact_events import FactEventStore
from memory_recall.fact_event_semantic import FactEventSemanticIndex
from scene_evidence import content_sha256


class FakeEmbeddingEngine:
    enabled = True
    model = "test-embedding-v1"
    document_instruction = "document"
    max_chars = 8000

    def __init__(self) -> None:
        self.documents: list[str] = []

    async def embed_document(self, text: str) -> list[float]:
        self.documents.append(text)
        if "紫外线" in text:
            return [1.0, 0.0, 0.0]
        if "明信片" in text:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


def source(message_id: int, role: str, content: str, created_at: str) -> dict:
    return {
        "source_system": "haven_bridge",
        "session_id": "19",
        "thread_id": "thread-shadow",
        "message_id": str(message_id),
        "role": role,
        "created_at": created_at,
        "content": content,
        "content_sha256": content_sha256(content),
        "evidence_kind": "primary",
        "binding_method": "bridge_daily_fact_event_v1",
    }


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="fact-event-shadow-") as temp_dir:
        config = {
            "state_dir": temp_dir,
            "buckets_dir": str(Path(temp_dir) / "buckets"),
            "gateway": {"fact_event_recall_shadow_enabled": True},
        }
        store = FactEventStore(config, create=True)
        written = store.write_many(
            [
                {
                    "type": "fact",
                    "body": "小雨紫外线过敏。",
                    "importance": 4,
                    "origin_id": "shadow-fact",
                    "source_refs": [
                        source(1, "user", "我紫外线过敏。", "2026-08-08T08:15:00Z")
                    ],
                },
                {
                    "type": "event",
                    "title": "北纬60.1度的风",
                    "body": "Haven从设得兰群岛寄出第一张明信片给小雨。",
                    "importance": 5,
                    "origin_id": "shadow-event",
                    "source_refs": [
                        source(2, "assistant", "刚寄出了第一张明信片。", "2026-08-08T12:17:00Z")
                    ],
                },
            ]
        )
        fact_id = written["items"][0]["item_id"]
        event_id = written["items"][1]["item_id"]
        engine = FakeEmbeddingEngine()
        index = FactEventSemanticIndex(config, engine)

        dry_run = await index.sync(store, dry_run=True)
        assert dry_run["to_embed"] == 2
        assert dry_run["memory_kinds"] == {"fact": 1, "event": 1}

        synced = await index.sync(store)
        assert synced == {
            "status": "ok",
            "active_items": 2,
            "embedded": 2,
            "reused": 0,
            "failed": 0,
            "removed": 0,
        }
        assert set(engine.documents) == {
            "小雨紫外线过敏。",
            "Haven从设得兰群岛寄出第一张明信片给小雨。",
        }
        assert all("北纬60.1度的风" not in text for text in engine.documents)

        fact_probe = index.search_by_embedding([1.0, 0.0, 0.0], top_k=2)
        assert fact_probe["status"] == "ok"
        assert fact_probe["matches"][0]["memory_id"] == fact_id
        assert fact_probe["matches"][0]["memory_kind"] == "fact"
        event_probe = index.search_by_embedding([0.0, 1.0, 0.0], top_k=2)
        assert event_probe["matches"][0]["memory_id"] == event_id
        assert event_probe["matches"][0]["memory_kind"] == "event"
        scoped_probe = index.search_by_embedding(
            [1.0, 0.0, 0.0],
            top_k=8,
            allowed_memory_ids=[event_id],
        )
        assert scoped_probe["indexed_memory_ids"] == [event_id], scoped_probe
        assert [row["memory_id"] for row in scoped_probe["matches"]] == [event_id], scoped_probe

        store.revise(fact_id, importance=5)
        reused = await index.sync(store)
        assert reused["embedded"] == 0 and reused["reused"] == 2
        fact_probe = index.search_by_embedding([1.0, 0.0, 0.0], top_k=1)
        assert fact_probe["matches"][0]["importance"] == 5

        store.revise(event_id, importance=2)
        reused = await index.sync(store)
        assert reused["embedded"] == 0 and reused["reused"] == 2
        gated_probe = index.search_by_embedding(
            [0.0, 1.0, 0.0],
            top_k=2,
            min_importance=3,
        )
        assert gated_probe["min_importance"] == 3
        assert all(row["memory_id"] != event_id for row in gated_probe["matches"])

        store.set_status(event_id, "archived")
        removed = await index.sync(store)
        assert removed["removed"] == 1
        assert removed["active_items"] == 1

    print("Fact/Event semantic shadow verification passed")


if __name__ == "__main__":
    asyncio.run(main())
