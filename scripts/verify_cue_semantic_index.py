from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_recall.cue_semantic import (
    CUE_INDEX_REBUILD_CONFIRMATION,
    CueSemanticIndex,
    build_cue_source,
    cue_index_selection_summary,
    scene_cue_hash,
    scene_cues_are_reviewed,
)


class FakeEmbeddingEngine:
    enabled = True
    model = "Qwen/Qwen3-Embedding-4B"
    query_instruction = "retrieve memory"
    document_instruction = ""
    max_chars = 6000

    async def embed_document(self, text: str) -> list[float]:
        if text == "explode":
            raise RuntimeError("synthetic_embedding_failure")
        vectors = {
            "4月8日命名日": [1.0, 0.0, 0.0],
            "名字诞生": [0.9, 0.1, 0.0],
            "巨型西瓜还没吃完": [0.0, 1.0, 0.0],
            "旧迁移标签": [0.0, 0.0, 1.0],
        }
        return vectors.get(text, [0.2, 0.2, 0.2])


def scene(
    scene_id: str,
    cues: list[str],
    *,
    source: str = "manual",
    reviewed: bool = False,
    bucket_type: str = "dynamic",
) -> dict:
    metadata = {
        "name": scene_id,
        "memory_value_source": "authored_scene",
        "object_kind": "scene",
        "type": bucket_type,
        "source": source,
        "scene_cues": cues,
    }
    if source == "scene_migration":
        metadata["migration_source_bucket_id"] = f"legacy-{scene_id}"
        metadata["migration_source_tags"] = list(cues)
    if reviewed:
        metadata["scene_cues_reviewed_at"] = "2026-08-05T08:00:00Z"
    return {"id": scene_id, "content": f"body {scene_id}", "metadata": metadata}


assert scene_cues_are_reviewed({"scene_cues": ["未经审核"]}) is False
assert scene_cues_are_reviewed(
    {
        "scene_cues": ["新 cue"],
        "last_edit_source": "edit_scene",
        "scene_revision_history": [{"cues": ["旧 cue"]}],
    }
) is True


async def verify() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        engine = FakeEmbeddingEngine()
        index = CueSemanticIndex(
            {
                "buckets_dir": str(root),
                "gateway": {
                    "cue_semantic_shadow": {
                        "publish_dir": str(root / "cue-index"),
                    }
                },
            },
            engine,
        )
        buckets = [
            scene("scene-native", ["4月8日命名日", "名字诞生"]),
            scene(
                "scene-reviewed-migration",
                ["巨型西瓜还没吃完"],
                source="scene_migration",
                reviewed=True,
            ),
            scene(
                "scene-unreviewed-migration",
                ["旧迁移标签"],
                source="scene_migration",
            ),
            scene(
                "scene-archived",
                ["旧迁移标签"],
                source="scene_migration",
                reviewed=True,
                bucket_type="archived",
            ),
        ]
        source = build_cue_source(buckets, dataset_version=1)
        assert [item["scene_id"] for item in source["scenes"]] == [
            "scene-native",
            "scene-reviewed-migration",
        ]
        assert cue_index_selection_summary(buckets) == {
            "total_buckets": 4,
            "eligible_scenes": 2,
            "excluded_scenes": 2,
            "reasons": {
                "eligible": 2,
                "inactive": 1,
                "unreviewed_migration": 1,
            },
        }

        built = await index.rebuild(
            buckets=buckets,
            expected_dataset_version=0,
            confirmation=CUE_INDEX_REBUILD_CONFIRMATION,
        )
        assert built["status"] == "available"
        assert built["dataset_version"] == 1
        assert built["scene_count"] == 2
        assert built["cue_count"] == 3

        hashes = {
            item["scene_id"]: item["cue_sha256"]
            for item in source["scenes"]
        }
        result = index.search_by_vector(
            [1.0, 0.0, 0.0],
            current_cue_hashes=hashes,
            top_k=3,
        )
        assert result["status"] == "available"
        assert result["matches"][0]["scene_id"] == "scene-native"
        assert result["matches"][0]["matched_cues"] == ["4月8日命名日"]

        stale_hashes = dict(hashes)
        stale_hashes["scene-native"] = scene_cue_hash(["后来改过的 cue"])
        stale = index.search_by_vector(
            [1.0, 0.0, 0.0],
            current_cue_hashes=stale_hashes,
            top_k=3,
        )
        assert stale["stale_scene_count"] == 1
        assert all(row["scene_id"] != "scene-native" for row in stale["matches"])

        try:
            await index.rebuild(
                buckets=buckets,
                expected_dataset_version=0,
                confirmation=CUE_INDEX_REBUILD_CONFIRMATION,
            )
        except ValueError as exc:
            assert str(exc) == "cue_semantic_rebuild_version_conflict:1"
        else:
            raise AssertionError("stale cue-index rebuild should fail")

        manifest_before = index.active_manifest_path.read_text(encoding="utf-8")
        failing_buckets = [scene("scene-failing", ["explode"])]
        try:
            await index.rebuild(
                buckets=failing_buckets,
                expected_dataset_version=1,
                confirmation=CUE_INDEX_REBUILD_CONFIRMATION,
            )
        except RuntimeError as exc:
            assert str(exc) == "synthetic_embedding_failure"
        else:
            raise AssertionError("failed cue embedding should abort activation")
        assert index.active_manifest_path.read_text(encoding="utf-8") == manifest_before
        assert index.status()["dataset_version"] == 1

        engine.document_instruction = "changed profile"
        mismatch = index.search_by_vector(
            [1.0, 0.0, 0.0],
            current_cue_hashes=hashes,
            top_k=3,
        )
        assert mismatch == {
            "status": "unavailable",
            "reason": "cue_index_embedding_profile_mismatch",
            "matches": [],
        }
        assert index.current_dataset_version() == 1
        try:
            await index.rebuild(
                buckets=buckets,
                expected_dataset_version=0,
                confirmation=CUE_INDEX_REBUILD_CONFIRMATION,
            )
        except ValueError as exc:
            assert str(exc) == "cue_semantic_rebuild_version_conflict:1"
        else:
            raise AssertionError("profile drift must not reset the publish CAS version")

        manifest = json.loads(index.active_manifest_path.read_text(encoding="utf-8"))
        assert manifest["mode"] == "simulation_shadow"


asyncio.run(verify())
print("cue semantic index verification passed")
