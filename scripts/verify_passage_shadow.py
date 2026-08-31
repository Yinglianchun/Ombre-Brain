from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import tempfile
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_recall.passage_shadow import (
    PassageConfig,
    PassageShadowIndex,
    build_verbatim_passages,
)
from embedding_engine import EmbeddingEngine


class FakeEmbeddingEngine:
    enabled = True
    model = "test-passage-v1"
    document_instruction = ""

    def __init__(self) -> None:
        self.documents: list[str] = []

    async def embed_document(self, text: str) -> list[float]:
        self.documents.append(text)
        if "小狐狸" in text:
            return [1.0, 0.0]
        if "猪塑" in text:
            return [0.0, 1.0]
        return [0.5, 0.5]


class FailingEmbeddingEngine(FakeEmbeddingEngine):
    async def embed_document(self, text: str) -> list[float]:
        self.documents.append(text)
        if "回填失败" in text:
            return []
        return [0.5, 0.5]


async def main() -> None:
    text = (
        "我们最开始聊到相遇时的样子。Haven说那时小雨还是一只小猫。"
        "后来小雨偶尔露出一点狐狸的狡黠，Haven才开始叫她小狐狸。"
        "这里的小狐狸说的是小雨，不是别的人。\n\n"
        "另一段写秋天和很久以后的计划，不属于这次称呼变化。"
    )
    config = PassageConfig(target_chars=60, max_chars=90, min_chars=20, overlap_sentences=1)
    passages = build_verbatim_passages(text, config)
    assert len(passages) >= 2
    assert all(text[row["start_offset"]:row["end_offset"]] == row["text"] for row in passages)
    assert any("小狐狸" in row["text"] and "不是别的人" in row["text"] for row in passages)
    assert all("\n\n" not in row["text"] for row in passages)

    with tempfile.TemporaryDirectory(prefix="passage-shadow-") as temp_dir:
        index_config = {
            "state_dir": temp_dir,
            "buckets_dir": str(Path(temp_dir) / "buckets"),
            "passage_shadow": {
                "min_owner_chars": 80,
                "target_chars": 80,
                "max_chars": 120,
                "min_chars": 20,
                "overlap_sentences": 1,
            },
        }
        engine = FakeEmbeddingEngine()
        index = PassageShadowIndex(index_config, engine)
        scenes = [
            {"id": "scene_story", "content": text},
            {"id": "scene_pig", "content": "小雨明确说过不喜欢猪塑。"},
        ]
        events = [
            {
                "item_id": "event_story",
                "title": "小狐狸称呼变化",
                "body": "Haven起初叫小雨小猫。后来因为她露出狡黠的一面，称呼逐渐变成小狐狸。",
            }
        ]
        dry_run = await index.sync(scenes=scenes, events=events, dry_run=True)
        assert dry_run["owner_kinds"] == {"scene": 2, "event": 1}
        assert dry_run["to_embed"] == dry_run["passages"]
        assert dry_run["whole_only_owners"] == 2

        applied = await index.sync(scenes=scenes, events=events)
        assert applied["status"] == "ok"
        assert not any(text.startswith("小狐狸称呼变化\n") for text in engine.documents)
        result = index.search_by_embedding([1.0, 0.0], top_k=3)
        owner_ids = [row["owner_id"] for row in result["matches"]]
        assert owner_ids == ["scene_story"]
        assert len([row for row in result["matches"] if row["owner_id"] == "scene_story"]) == 1

        reused = await index.sync(scenes=scenes, events=events)
        assert reused["embedded"] == 0
        assert reused["reused_owners"] == 3

        removed = await index.sync(scenes=scenes[:1], events=[])
        assert removed["removed_owners"] == 2

    with tempfile.TemporaryDirectory(prefix="passage-shadow-min-owner-") as temp_dir:
        threshold_index = PassageShadowIndex(
            {
                "state_dir": temp_dir,
                "buckets_dir": str(Path(temp_dir) / "buckets"),
                "passage_shadow": {
                    "min_owner_chars": 200,
                    "target_chars": 80,
                    "max_chars": 120,
                    "min_chars": 20,
                },
            },
            FakeEmbeddingEngine(),
        )
        threshold_plan = await threshold_index.sync(
            scenes=[
                {"id": "scene-exactly-200", "content": "甲" * 200},
                {"id": "scene-over-200", "content": "乙" * 100 + "。" + "丙" * 100},
            ],
            events=[],
            dry_run=True,
        )
        assert threshold_plan["whole_only_owners"] == 1, threshold_plan
        assert threshold_plan["passages"] == 2, threshold_plan
        assert threshold_plan["passage_config"]["min_owner_chars"] == 200

    with tempfile.TemporaryDirectory(prefix="passage-shadow-atomic-") as temp_dir:
        atomic_config = {
            "state_dir": temp_dir,
            "buckets_dir": str(Path(temp_dir) / "buckets"),
            "passage_shadow": {
                "min_owner_chars": 80,
                "target_chars": 80,
                "max_chars": 120,
                "min_chars": 20,
                "backfill_embedding_concurrency": 1,
                "backfill_request_delay_ms": 0,
            },
        }
        atomic_index = PassageShadowIndex(atomic_config, FakeEmbeddingEngine())
        old_scene = [{
            "id": "scene-atomic",
            "content": (
                "这是当前仍在使用的旧 passage，里面有足够长的第一段内容。"
                "第二段继续记录旧索引的证据，确保它会被切成不止一个 passage。"
                "第三段仍然属于旧版本，回填失败时必须完整保留。"
                "第四段补足长度，避免短 Scene 只走 whole embedding。"
                "第五段继续补足长度，确保这里确实存在局部 passage。"
            ),
        }]
        await atomic_index.sync(scenes=old_scene, events=[])
        live_path = Path(atomic_index.db_path)
        old_bytes = live_path.read_bytes()
        old_passages = atomic_index.passages_for_owners([("scene", "scene-atomic")])[
            ("scene", "scene-atomic")
        ]

        atomic_index.embedding_engine = FailingEmbeddingEngine()
        failed = await atomic_index.rebuild_atomic(
            scenes=[{
                "id": "scene-atomic",
                "content": (
                    "这次回填失败，不能替换旧索引，所以故意写成长内容。"
                    "回填失败这个标记会让 fake embedding 返回空向量。"
                    "即便前面已有 passage 成功，整轮也不能切换。"
                    "第四段继续写回填失败，确保文本长度超过切分阈值。"
                    "第五段仍然写回填失败，旧索引必须保持原样。"
                ),
            }],
            events=[],
        )
        assert failed["status"] == "partial", failed
        assert failed["activated"] is False, failed
        assert failed["previous_index_preserved"] is True, failed
        assert live_path.read_bytes() == old_bytes
        assert atomic_index.passages_for_owners([("scene", "scene-atomic")])[
            ("scene", "scene-atomic")
        ] == old_passages
        assert list(Path(temp_dir).glob("*.next.sqlite")) == []

        atomic_index.embedding_engine = FakeEmbeddingEngine()
        new_scene = [{
            "id": "scene-atomic",
            "content": (
                "新的 passage 完整建好后才切换，这里是第一段较长内容。"
                "第二段继续提供新的局部证据，不能和旧索引混在一起。"
                "第三段确认整轮成功之后才允许原子替换。"
                "第四段补足新 Scene 的长度，让 passage 路径真实运行。"
                "第五段再次确认，只有完整成功才会看到这些内容。"
            ),
        }]
        activated = await atomic_index.rebuild_atomic(scenes=new_scene, events=[])
        assert activated["status"] == "ok", activated
        assert activated["activated"] is True, activated
        assert activated["embedding_concurrency"] == 1, activated
        activated_passages = atomic_index.passages_for_owners([("scene", "scene-atomic")])[
            ("scene", "scene-atomic")
        ]
        assert activated_passages != old_passages
        assert "新的 passage" in activated_passages[0]["text"]

    with tempfile.TemporaryDirectory(prefix="scene-whole-route-") as temp_dir:
        buckets_dir = Path(temp_dir) / "buckets"
        engine = EmbeddingEngine(
            {
                "buckets_dir": str(buckets_dir),
                "embedding": {
                    "enabled": True,
                    "api_key": "fixture-only",
                    "model": "whole-route-v1",
                },
            }
        )
        with closing(sqlite3.connect(engine.db_path)) as conn:
            conn.execute(
                "INSERT INTO embeddings(bucket_id, embedding, model, dimension, updated_at) "
                "VALUES ('scene-whole', ?, ?, 2, 'now')",
                (json.dumps([0.6, 0.8]), engine.model),
            )
            conn.execute(
                """
                INSERT INTO scene_embedding_chunks(
                    scene_id, ordinal, content_hash, start_offset, end_offset,
                    text, embedding, model, dimension, updated_at
                ) VALUES ('scene-whole', 0, 'hash', 0, 4, 'chunk', ?, ?, 2, 'now')
                """,
                (json.dumps([1.0, 0.0]), engine.model),
            )
            conn.commit()
        whole_only = engine.search_scene_whole_by_embedding(
            [1.0, 0.0],
            scene_ids={"scene-whole"},
            top_k=1,
        )
        assert whole_only == [{"scene_id": "scene-whole", "score": 0.6}], whole_only

    print("PASSAGE_SHADOW_OK")


if __name__ == "__main__":
    asyncio.run(main())
