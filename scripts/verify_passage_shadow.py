from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_recall.passage_shadow import (
    PassageConfig,
    PassageShadowIndex,
    build_verbatim_passages,
)


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
                "body": "Haven起初叫小雨小猫。后来因为她露出狡黠的一面，称呼逐渐变成小狐狸。",
            }
        ]
        dry_run = await index.sync(scenes=scenes, events=events, dry_run=True)
        assert dry_run["owner_kinds"] == {"scene": 2, "event": 1}
        assert dry_run["to_embed"] == dry_run["passages"]

        applied = await index.sync(scenes=scenes, events=events)
        assert applied["status"] == "ok"
        result = index.search_by_embedding([1.0, 0.0], top_k=3)
        owner_ids = [row["owner_id"] for row in result["matches"]]
        assert owner_ids[0] in {"scene_story", "event_story"}
        assert "scene_pig" in owner_ids
        assert len([row for row in result["matches"] if row["owner_id"] == "scene_story"]) == 1

        reused = await index.sync(scenes=scenes, events=events)
        assert reused["embedded"] == 0
        assert reused["reused_owners"] == 3

        removed = await index.sync(scenes=scenes[:1], events=[])
        assert removed["removed_owners"] == 2

    print("PASSAGE_SHADOW_OK")


if __name__ == "__main__":
    asyncio.run(main())
