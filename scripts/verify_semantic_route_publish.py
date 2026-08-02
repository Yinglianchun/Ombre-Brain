from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_recall.semantic_router import (
    ROUTE_PUBLISH_CONFIRMATION,
    SemanticRecallRouter,
    build_route_index,
)
from gateway import GatewayService


class FakeEmbeddingEngine:
    enabled = True
    model = "Qwen/Qwen3-Embedding-4B"
    query_instruction = "semantic route verification"
    max_chars = 6000

    async def embed_query(self, text: str) -> list[float]:
        if text == "explode":
            raise RuntimeError("synthetic_embedding_failure")
        checksum = sum(ord(character) for character in text)
        return [float((checksum % 97) + 1), float((checksum % 53) + 1)]


class RequestStub:
    headers: dict[str, str] = {}

    def __init__(self, body: dict | None = None):
        self.body = body or {}

    async def json(self) -> dict:
        return self.body


def source_payload() -> dict:
    return {
        "schema_version": 1,
        "dataset_version": 7,
        "routes": [
            {
                "name": "present_chitchat",
                "label": "此刻闲聊",
                "action": "skip",
                "utterances": [
                    {"text": "今天好热", "role": "typical", "origin": "import"},
                ],
            },
            {
                "name": "recall_needed",
                "label": "明确回望",
                "action": "recall",
                "utterances": [
                    {"text": "昨天我们聊了什么", "role": "typical", "origin": "import"},
                    {"text": "回来继续昨天的话题", "role": "boundary", "origin": "import"},
                ],
            },
        ],
    }


async def verify() -> None:
    with TemporaryDirectory() as directory:
        state_dir = Path(directory)
        source_path = state_dir / "seed.json"
        index_path = state_dir / "seed-index.json"
        source_path.write_text(
            json.dumps(source_payload(), ensure_ascii=False),
            encoding="utf-8",
        )
        engine = FakeEmbeddingEngine()
        await build_route_index(
            source_path=source_path,
            output_path=index_path,
            embedding_engine=engine,
        )
        router = SemanticRecallRouter(
            {
                "buckets_dir": str(state_dir),
                "gateway": {
                    "semantic_recall_router": {
                        "mode": "active",
                        "routes_path": str(source_path),
                        "index_path": str(index_path),
                    }
                },
            },
            engine,
        )

        initial = router.dataset_payload()
        assert initial["dataset_version"] == 7
        assert initial["source_kind"] == "seed"
        assert initial["index_ready"] is True

        routes = initial["routes"]
        routes[0]["utterances"].append(
            {
                "text": "我们先继续改这个页面吧",
                "role": "typical",
                "origin": "online_false_positive",
                "status": "draft",
            }
        )
        published = await router.publish_dataset(
            routes=routes,
            expected_dataset_version=7,
            confirmation=ROUTE_PUBLISH_CONFIRMATION,
        )
        assert published["dataset_version"] == 8
        assert published["source_kind"] == "published"
        assert published["index_ready"] is True
        assert published["published"]["boundary_example_count"] == 1

        manifest = json.loads(router.active_manifest_path.read_text(encoding="utf-8"))
        generation_dir = router.publish_dir / manifest["generation"]
        source = json.loads((generation_dir / "source.json").read_text(encoding="utf-8"))
        index = json.loads((generation_dir / "index.json").read_text(encoding="utf-8"))
        assert source["dataset_version"] == 8
        assert source["routes"][1]["utterances"][1]["role"] == "boundary"
        indexed_texts = {
            item["text"]
            for route in index["routes"]
            for item in route["utterances"]
        }
        assert "回来继续昨天的话题" not in indexed_texts
        assert "我们先继续改这个页面吧" in indexed_texts
        assert json.loads(source_path.read_text(encoding="utf-8"))["dataset_version"] == 7

        try:
            await router.publish_dataset(
                routes=routes,
                expected_dataset_version=7,
                confirmation=ROUTE_PUBLISH_CONFIRMATION,
            )
        except ValueError as exc:
            assert str(exc) == "route_publish_version_conflict:8"
        else:
            raise AssertionError("stale publisher should be rejected")

        failing_routes = json.loads(json.dumps(published["routes"], ensure_ascii=False))
        failing_routes[0]["utterances"].append(
            {
                "text": "explode",
                "role": "typical",
                "origin": "manual",
                "status": "draft",
            }
        )
        active_before_failure = router.active_manifest_path.read_text(encoding="utf-8")
        try:
            await router.publish_dataset(
                routes=failing_routes,
                expected_dataset_version=8,
                confirmation=ROUTE_PUBLISH_CONFIRMATION,
            )
        except RuntimeError as exc:
            assert str(exc) == "synthetic_embedding_failure"
        else:
            raise AssertionError("embedding failure should abort publication")
        assert router.active_manifest_path.read_text(encoding="utf-8") == active_before_failure
        assert router.dataset_payload()["dataset_version"] == 8

        service = GatewayService.__new__(GatewayService)
        service._authorize = lambda _authorization: None
        service.semantic_recall_router = router
        read_response = await service.handle_semantic_recall_routes(RequestStub())
        assert read_response.status_code == 200
        assert json.loads(read_response.body)["dataset_version"] == 8
        conflict_response = await service.handle_semantic_recall_publish(
            RequestStub(
                {
                    "expected_dataset_version": 7,
                    "confirm": ROUTE_PUBLISH_CONFIRMATION,
                    "routes": routes,
                }
            )
        )
        assert conflict_response.status_code == 409
        assert json.loads(conflict_response.body)["error"] == "route_publish_version_conflict:8"


asyncio.run(verify())
print("semantic route atomic publish verification passed")
