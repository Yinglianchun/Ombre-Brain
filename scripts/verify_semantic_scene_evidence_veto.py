from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embedding_engine import EmbeddingEngine
from gateway import GatewayService
from starlette.responses import JSONResponse


def scene(
    scene_id: str,
    *,
    title: str = "Scene",
    cues: list[str] | None = None,
    reviewed: bool = False,
    tags: list[str] | None = None,
    policy: str = "normal",
) -> dict:
    return {
        "id": scene_id,
        "content": f"body for {scene_id}",
        "metadata": {
            "name": title,
            "object_kind": "scene",
            "memory_value_source": "authored_scene",
            "scene_cues": list(cues or []),
            "scene_cues_reviewed_at": "2026-08-03T00:00:00Z" if reviewed else "",
            "tags": list(tags or []),
            "test_policy": policy,
        },
    }


class ProbeEmbeddingStub:
    enabled = True

    def __init__(self, rows: list[dict] | None = None):
        self.rows = list(rows or [])
        self.calls = 0

    async def search_scene_evidence_by_embedding(self, query_embedding, *, scene_ids, top_k):
        self.calls += 1
        assert query_embedding == [0.1, 0.2]
        return [row for row in self.rows if row["scene_id"] in scene_ids][:top_k]


class HookRouterStub:
    active = True

    async def route_with_vector(self, query: str):
        return {"enabled": True, "called": True, "would_skip": True}, [0.1, 0.2]

    @staticmethod
    def should_apply_skip(debug):
        return bool(debug.get("would_skip"))


class RequestStub:
    headers: dict[str, str] = {}

    def __init__(self, body: dict):
        self.body = body

    async def json(self):
        return self.body


def service_for(buckets: list[dict], rows: list[dict] | None = None, *, mode: str = "active") -> GatewayService:
    service = GatewayService.__new__(GatewayService)
    service.semantic_scene_evidence_veto_mode = mode
    service.semantic_scene_evidence_veto_enabled = mode != "off"
    service.semantic_scene_evidence_veto_min_score = 0.48
    service.semantic_scene_evidence_veto_min_margin = 0.03
    service.semantic_scene_evidence_veto_debug_limit = 3
    service.embedding_query_timeout_seconds = 0
    service.embedding_engine = ProbeEmbeddingStub(rows)

    async def list_buckets(*, include_archive: bool = False):
        assert include_archive is False
        return buckets

    service._list_gateway_buckets = list_buckets
    service._is_canonical_scene_bucket = lambda bucket: (
        isinstance(bucket, dict)
        and str((bucket.get("metadata") or {}).get("object_kind") or "") == "scene"
    )
    service._is_semantic_candidate_bucket = lambda bucket: (
        str((bucket.get("metadata") or {}).get("type") or "") != "archived"
    )
    service._bucket_scene_cues_are_reviewed = lambda meta: bool(meta.get("scene_cues_reviewed_at"))
    service._bucket_authored_cue_terms = lambda query, bucket: [
        cue
        for cue in (bucket.get("metadata") or {}).get("scene_cues", [])
        if cue and cue in query
    ]
    service._bucket_title_anchor_terms = lambda query, bucket: (
        [(bucket.get("metadata") or {}).get("name")]
        if (bucket.get("metadata") or {}).get("name") in query
        else []
    )

    def policy_rejection(bucket, *, direct_evidence=None, explicit_id=False):
        policy = str((bucket.get("metadata") or {}).get("test_policy") or "normal")
        if policy == "excluded":
            return {"reason": "domain_excluded"}
        if policy == "explicit_only" and not direct_evidence:
            return {"reason": "domain_explicit_only"}
        return None

    service._canonical_scene_domain_policy_rejection = policy_rejection
    return service


async def run_probe(service: GatewayService, query: str = "query") -> tuple[bool, dict]:
    debug: dict = {}
    skip = await service._apply_semantic_scene_evidence_veto(
        query,
        True,
        [0.1, 0.2],
        debug,
    )
    return skip, debug["scene_evidence_veto"]


async def verify_probe_contracts() -> None:
    cue_scene = scene("cue", cues=["shared vow"], reviewed=True)
    skip, debug = await run_probe(service_for([cue_scene]), "shared vow")
    assert skip is False
    assert debug["applied"] is True
    assert debug["candidate_field"] == "authored_cue"

    legacy_tag_scene = scene("legacy", tags=["shared vow"])
    skip, debug = await run_probe(
        service_for(
            [legacy_tag_scene],
            [{"scene_id": "legacy", "score": 0.30, "field": "scene_body"}],
        ),
        "shared vow",
    )
    assert skip is True
    assert debug["applied"] is False

    low = service_for(
        [scene("low"), scene("lower")],
        [
            {"scene_id": "low", "score": 0.47, "field": "scene_body"},
            {"scene_id": "lower", "score": 0.20, "field": "scene_body"},
        ],
    )
    skip, debug = await run_probe(low)
    assert skip is True
    assert debug["body_top_score"] == 0.47
    assert debug["candidates"][0]["reason"] == "below_absolute_body_threshold"

    narrow_margin = service_for(
        [scene("first"), scene("second")],
        [
            {"scene_id": "first", "score": 0.55, "field": "scene_body_chunk"},
            {"scene_id": "second", "score": 0.53, "field": "scene_body"},
        ],
    )
    skip, debug = await run_probe(narrow_margin)
    assert skip is True
    assert debug["body_margin"] == 0.02
    assert debug["candidates"][0]["reason"] == "below_body_margin"

    strong_body = service_for(
        [scene("first"), scene("second")],
        [
            {"scene_id": "first", "score": 0.55, "field": "scene_body_chunk"},
            {"scene_id": "second", "score": 0.50, "field": "scene_body"},
        ],
    )
    skip, debug = await run_probe(strong_body)
    assert skip is False
    assert debug["candidate_id"] == "first"
    assert debug["candidate_field"] == "scene_body_chunk"

    excluded = service_for(
        [scene("excluded", policy="excluded")],
        [{"scene_id": "excluded", "score": 0.99, "field": "scene_body"}],
    )
    skip, debug = await run_probe(excluded)
    assert skip is True
    assert debug["candidates"][0]["reason"] == "domain_excluded"

    explicit_only = scene(
        "explicit",
        cues=["private name"],
        reviewed=True,
        policy="explicit_only",
    )
    body_only = service_for(
        [explicit_only],
        [{"scene_id": "explicit", "score": 0.99, "field": "scene_body"}],
    )
    skip, debug = await run_probe(body_only, "unrelated paraphrase")
    assert skip is True
    assert debug["candidates"][0]["reason"] == "domain_explicit_only"
    skip, debug = await run_probe(service_for([explicit_only]), "private name")
    assert skip is False
    assert debug["candidate_field"] == "authored_cue"

    shadow = service_for([cue_scene], mode="shadow")
    skip, debug = await run_probe(shadow, "shared vow")
    assert skip is True
    assert debug["would_apply"] is True
    assert debug["applied"] is False
    assert debug["reason"] == "shadow_would_veto"


async def verify_veto_never_injects() -> None:
    target = scene("cue", cues=["shared vow"], reviewed=True)
    service = service_for([target])
    service._authorize = lambda authorization: None
    service.upstream_default_model = ""
    service.upstream_models = []
    service.semantic_recall_router = HookRouterStub()
    called = {"normal_retrieval": False}

    async def fast_cards(query: str, session_id: str, **kwargs):
        called["normal_retrieval"] = True
        assert kwargs["query_embedding"] == [0.1, 0.2]
        return [], [], {"hook_recall_debug": {"candidate_count": 0}}

    service._hook_recall_fast_cards = fast_cards
    service._render_hook_recall_additional_context = lambda cards: ""
    response = await service.handle_hook_recall(
        RequestStub({"query": "shared vow", "include_debug": True})
    )
    body = json.loads(response.body)
    assert called["normal_retrieval"] is True
    assert body["cards"] == []
    assert body["recalled_ids"] == []
    assert body["debug"]["semantic_recall_debug"]["scene_evidence_veto"]["applied"] is True


async def verify_full_hook_reuses_route_vector() -> None:
    target = scene("cue", cues=["shared vow"], reviewed=True)
    service = service_for([target])
    service._authorize = lambda authorization: None
    service.upstream_default_model = ""
    service.upstream_models = []
    router = HookRouterStub()
    router.calls = 0
    original_route = router.route_with_vector

    async def counted_route(query: str):
        router.calls += 1
        return await original_route(query)

    router.route_with_vector = counted_route
    service.semantic_recall_router = router

    async def full_recall(**kwargs):
        debug, vector = kwargs["semantic_recall_result"]
        assert vector == [0.1, 0.2]
        assert debug["scene_evidence_veto"]["applied"] is True
        return JSONResponse({"ok": True, "cards": [], "recalled_ids": []})

    service._handle_hook_recall_full = full_recall
    response = await service.handle_hook_recall(
        RequestStub({"query": "shared vow", "recall_mode": "full", "include_debug": True})
    )
    assert json.loads(response.body)["ok"] is True
    assert router.calls == 1
    assert service.embedding_engine.calls == 1


async def verify_embedding_provenance() -> None:
    with TemporaryDirectory(prefix="ombre-scene-veto-") as temp_dir:
        engine = EmbeddingEngine(
            {
                "buckets_dir": temp_dir,
                "embedding": {
                    "enabled": True,
                    "api_key": "test-only",
                    "base_url": "http://127.0.0.1:9/v1",
                    "model": "test-model",
                },
            }
        )
        conn = sqlite3.connect(engine.db_path)
        conn.executemany(
            "INSERT INTO embeddings(bucket_id, embedding, model, dimension, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                ("scene-a", json.dumps([0.8, 0.2]), "test-model", 2, "now"),
                ("legacy-tag-hit", json.dumps([1.0, 0.0]), "test-model", 2, "now"),
            ],
        )
        conn.execute(
            "INSERT INTO scene_embedding_chunks"
            "(scene_id, ordinal, content_hash, start_offset, end_offset, text, embedding, model, dimension, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "scene-a",
                0,
                "hash",
                10,
                20,
                "verbatim body span",
                json.dumps([1.0, 0.0]),
                "test-model",
                2,
                "now",
            ),
        )
        conn.commit()
        conn.close()

        rows = await engine.search_scene_evidence_by_embedding(
            [1.0, 0.0],
            scene_ids={"scene-a"},
            top_k=3,
        )
        assert [row["scene_id"] for row in rows] == ["scene-a"]
        assert rows[0]["field"] == "scene_body_chunk"
        assert rows[0]["chunk_ordinal"] == 0


async def main() -> None:
    await verify_probe_contracts()
    await verify_veto_never_injects()
    await verify_full_hook_reuses_route_vector()
    await verify_embedding_provenance()


asyncio.run(main())
print("semantic Scene evidence veto verification passed")
