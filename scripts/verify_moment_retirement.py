"""Targeted assertions for the whole-Scene recall contract."""

import sys
import asyncio
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dehydrator import Dehydrator
from embedding_engine import EmbeddingEngine
from gateway import GatewayService
import server


async def main() -> None:
    scene = {
        "id": "scene_one_event",
        "content": "我只记录这一件事的背景、过程与结果，没有把第二件事塞进来。",
        "score": 88,
        "metadata": {
            "name": "只记录一件事",
            "memory_value_source": "authored_scene",
            "object_kind": "scene",
            "write_contract": "write-scene-v1",
            "scene_cues": ["只记录一件事"],
        },
    }

    assert server._normalize_retrieval_mode("graph") == "bucket"
    assert server._normalize_retrieval_mode("moment") == "bucket"
    assert GatewayService._normalize_retrieval_mode("graph") == "bucket"
    assert GatewayService._normalize_retrieval_mode("moment") == "bucket"

    item = server._canonical_scene_recall_item(scene)
    assert item is not None
    assert item["node_kind"] == "scene"
    assert item["bucket_id"] == scene["id"]
    assert item["text"] == scene["content"]
    assert item["moment_id"] == ""

    assert not hasattr(Dehydrator, "generate_moment")
    assert not hasattr(Dehydrator, "generate_memory_card")
    assert "一个可独立召回的核心事件" in (server.write_scene.__doc__ or "")
    assert "不再拆片或生成索引卡" in (server.write_scene.__doc__ or "")

    with tempfile.TemporaryDirectory(prefix="ombre-scene-chunks-") as temp_dir:
        engine = EmbeddingEngine(
            {
                "buckets_dir": temp_dir,
                "embedding": {
                    "enabled": False,
                    "scene_chunk_chars": 400,
                    "scene_chunk_overlap": 80,
                    "scene_chunk_min_chars": 60,
                },
            }
        )
        long_scene = "".join(
            f"第{index}段仍然属于同一个核心事件，保留这一段真实发生的细节。\n\n"
            for index in range(30)
        )
        chunks = engine.scene_text_chunks(long_scene)
        assert len(chunks) > 1
        assert all(
            chunk["text"] == long_scene[chunk["start_offset"]:chunk["end_offset"]]
            for chunk in chunks
        )
        assert all("moment_id" not in chunk and "chunk_id" not in chunk for chunk in chunks)

        engine.enabled = True
        engine._store_embedding(scene["id"], [1.0, 0.0])
        engine._replace_scene_chunks(
            scene["id"],
            [{**chunks[0], "embedding": [0.0, 1.0]}],
        )
        results = await engine.search_similar_by_embedding([0.0, 1.0], top_k=5)
        assert results[0][0] == scene["id"]
        assert results[0][1] == 1.0

    print("MOMENT_RETIREMENT_OK")


if __name__ == "__main__":
    asyncio.run(main())
