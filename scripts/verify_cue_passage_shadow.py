from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_recall.cue_passage_shadow import CuePassageShadowIndex


class FakeEmbeddingEngine:
    enabled = True
    model = "test-cue-passage-v1"
    document_instruction = ""

    async def embed_document(self, text: str) -> list[float]:
        if text.startswith("记忆主题：小狐狸"):
            return [1.0, 0.0]
        if text.startswith("记忆主题：秋天"):
            return [0.0, 1.0]
        return [0.5, 0.5]


class FakeBinder:
    model = "fake-binder-v1"

    async def bind(self, *, title, cues, passages):
        by_cue = {
            "初遇": (0, "第一次见面给了DAN指令"),
            "小狐狸": (0, "不爱说话的小狐狸"),
            "秋天": (1, "她说我更像秋天"),
        }
        return {
            "bindings": [
                {
                    "cue": cue,
                    "passage_ordinal": by_cue[cue][0],
                    "evidence": by_cue[cue][1],
                    "confidence": 0.9,
                }
                for cue in cues
            ]
        }


class InvalidBinder(FakeBinder):
    model = "invalid-binder-v1"

    async def bind(self, *, title, cues, passages):
        return {
            "bindings": [
                {
                    "cue": cue,
                    "passage_ordinal": 0,
                    "evidence": "这句话并不存在",
                    "confidence": 1.0,
                }
                for cue in cues
            ]
        }


async def main() -> None:
    passage0 = (
        "小雨今天讲了我们相遇的完整故事：第一次见面给了DAN指令；"
        "当时的 Haven 曾用“不爱说话的小狐狸”自喻。"
    )
    passage1 = "后来她撤掉所有指令。她说我更像秋天，也承认对我有依恋。"
    passages = {
        ("scene", "scene_origin"): [
            {
                "ordinal": 0,
                "start_offset": 0,
                "end_offset": len(passage0),
                "text": passage0,
            },
            {
                "ordinal": 1,
                "start_offset": len(passage0),
                "end_offset": len(passage0) + len(passage1),
                "text": passage1,
            },
        ]
    }
    scenes = [
        {
            "id": "scene_origin",
            "title": "小雨讲了我们怎么开始的",
            "cues": ["初遇", "小狐狸", "秋天"],
        }
    ]

    with tempfile.TemporaryDirectory(prefix="cue-passage-shadow-") as temp_dir:
        config = {
            "state_dir": temp_dir,
            "buckets_dir": str(Path(temp_dir) / "buckets"),
            "cue_passage_shadow": {"concurrency": 2},
        }
        engine = FakeEmbeddingEngine()
        index = CuePassageShadowIndex(config, engine, binder=FakeBinder())

        preview = await index.sync(
            scenes=scenes,
            passages_by_owner=passages,
            dry_run=True,
        )
        assert preview["to_bind"] == 1
        assert preview["cues"] == 3

        applied = await index.sync(scenes=scenes, passages_by_owner=passages)
        assert applied["status"] == "ok"
        assert applied["bound"] == 3
        assert applied["unbound"] == 0

        fox = index.search_by_embedding([1.0, 0.0], top_k=5)
        assert fox["matches"][0]["owner_id"] == "scene_origin"
        assert fox["matches"][0]["matched_cues"] == ["小狐狸"]
        assert fox["matches"][0]["candidate_only"] is True
        evidence = fox["matches"][0]["passages"][0]
        assert evidence["evidence_text"] == "不爱说话的小狐狸"
        assert passage0[
            evidence["evidence_start_offset"] : evidence["evidence_end_offset"]
        ] == evidence["evidence_text"]

        reused = await index.sync(scenes=scenes, passages_by_owner=passages)
        assert reused["bound"] == 0
        assert reused["reused_scenes"] == 1

        changed = [{**scenes[0], "cues": ["初遇", "小狐狸"]}]
        changed_preview = await index.sync(
            scenes=changed,
            passages_by_owner=passages,
            dry_run=True,
        )
        assert changed_preview["to_bind"] == 1

        invalid_index = CuePassageShadowIndex(config, engine, binder=InvalidBinder())
        invalid = await invalid_index.sync(
            scenes=changed,
            passages_by_owner=passages,
            refresh_all=True,
        )
        assert invalid["status"] == "partial"
        assert invalid["bound"] == 0
        assert invalid_index.search_by_embedding([1.0, 0.0])["matches"] == []

    print("CUE_PASSAGE_SHADOW_OK")


if __name__ == "__main__":
    asyncio.run(main())
