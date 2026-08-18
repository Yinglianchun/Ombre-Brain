"""Verify reviewed Scene one-hop expansion stays candidate-only and bounded."""

import asyncio
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway import GatewayService


def scene(scene_id: str) -> dict:
    return {
        "id": scene_id,
        "content": f"content:{scene_id}",
        "metadata": {
            "object_kind": "scene",
            "memory_value_source": "authored_scene",
            "active": True,
        },
    }


class EdgeStore:
    @staticmethod
    def recall_edges(_scene_map):
        return [
            {
                "edge_id": "edge-target",
                "source": "seed",
                "target": "target",
                "relation_type": "echoes",
                "confidence": 0.91,
            },
            {
                "edge_id": "edge-low",
                "source": "seed",
                "target": "low",
                "relation_type": "echoes",
                "confidence": 0.99,
            },
            {
                "edge_id": "edge-wrong-relation",
                "source": "seed",
                "target": "noise",
                "relation_type": "evidenced_by",
                "confidence": 0.99,
            },
        ]


async def main() -> None:
    service = GatewayService.__new__(GatewayService)
    service.scene_edge_store = EdgeStore()
    service._is_canonical_scene_bucket = lambda _bucket: True
    service._canonical_scene_domain_policy = lambda _bucket: ("general", "normal")

    calls = 0

    async def list_buckets(*, include_archive: bool):
        nonlocal calls
        calls += 1
        assert include_archive is False
        return [scene(value) for value in ("seed", "target", "low", "noise")]

    service._list_gateway_buckets = list_buckets
    semantic = {
        "retrieval_budget": {
            "cheap_retrieval": {
                "candidates": [
                    {
                        "bucket_id": "target",
                        "canonical_scene": True,
                        "body_semantic_score": 0.51,
                    },
                    {
                        "bucket_id": "low",
                        "canonical_scene": True,
                        "body_semantic_score": 0.49,
                    },
                    {
                        "bucket_id": "noise",
                        "canonical_scene": True,
                        "body_semantic_score": 0.90,
                    },
                ]
            }
        }
    }
    await service._attach_reviewed_scene_onehop_shadow(
        semantic,
        recalled_ids=["seed"],
    )
    shadow = semantic["retrieval_budget"]["reviewed_scene_onehop_shadow"]
    assert shadow["status"] == "expanded"
    assert shadow["decision_applied"] is False
    assert shadow["affects_recall"] is False
    assert shadow["live_injection_enabled"] is False
    assert shadow["candidate_count"] == 1
    assert shadow["candidates"] == [
        {
            "candidate_only": True,
            "scene_id": "target",
            "seed_scene_id": "seed",
            "edge_id": "edge-target",
            "relation_type": "echoes",
            "body_semantic_score": 0.51,
            "edge_confidence": 0.91,
        }
    ]
    assert calls == 1

    no_seed = {"retrieval_budget": {"cheap_retrieval": {"candidates": []}}}
    await service._attach_reviewed_scene_onehop_shadow(no_seed, recalled_ids=[])
    no_seed_shadow = no_seed["retrieval_budget"]["reviewed_scene_onehop_shadow"]
    assert no_seed_shadow["status"] == "no_seed"
    assert no_seed_shadow["candidates"] == []
    assert calls == 1

    print("REVIEWED_SCENE_ONEHOP_SHADOW_OK")


if __name__ == "__main__":
    asyncio.run(main())
