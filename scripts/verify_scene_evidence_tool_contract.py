from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import server  # noqa: E402


class FakeEvidenceStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict], str]] = []

    def bind(self, scene_id: str, refs: list[dict], *, bound_by: str = "") -> dict:
        self.calls.append((scene_id, refs, bound_by))
        return {
            "scene_id": scene_id,
            "evidence_status": "bound",
            "bound_count": len(refs),
            "existing_count": 0,
            "idempotent": False,
            "evidence_refs": refs,
        }

    def list_for_scene(self, scene_id: str) -> list[dict]:
        return self.calls[-1][1] if self.calls and self.calls[-1][0] == scene_id else []

    def unbind(self, scene_id: str, evidence_ids: list[int | str], *, unbound_by: str = "") -> dict:
        _ = evidence_ids, unbound_by
        return {
            "scene_id": scene_id,
            "evidence_status": "unbound",
            "unbound_count": 1,
            "already_unbound_count": 0,
            "idempotent": False,
            "evidence_refs": [],
        }


async def main() -> None:
    scene_id = "scene_contract_exact_id"
    scene = {
        "id": scene_id,
        "content": "正文",
        "metadata": {
            "object_kind": "scene",
            "memory_value_source": "authored_scene",
            "active": True,
        },
    }
    ref = {
        "source_system": "haven_bridge",
        "session_id": "7",
        "thread_id": "thread-7",
        "message_id": "42",
        "role": "user",
        "created_at": "2026-08-04T03:00:00Z",
        "content": "原文  \n",
        "binding_method": "approved_candidate_auto",
        "evidence_kind": "primary",
    }
    fake_store = FakeEvidenceStore()
    with (
        patch.object(server, "generate_scene_id", return_value=scene_id),
        patch.object(
            server,
            "_write_scene_memory",
            AsyncMock(return_value="新建→合同验证 relationship"),
        ) as write_memory,
        patch.object(server.bucket_mgr, "get", AsyncMock(return_value=scene)),
        patch.object(server, "scene_evidence_store", fake_store),
    ):
        result = await server.write_scene(
            "正文",
            ["合同验证"],
            title="合同验证",
            evidence_refs=[ref],
        )
    assert "[scene_id:scene_contract_exact_id]" in result
    assert "[evidence_status:bound]" in result
    assert write_memory.await_args.kwargs["scene_id"] == scene_id
    assert fake_store.calls[0][0] == scene_id
    assert fake_store.calls[0][1][0]["content"] == "原文  \n"

    with (
        patch.object(server.bucket_mgr, "get", AsyncMock(return_value=scene)),
        patch.object(server, "scene_evidence_store", fake_store),
    ):
        rebound = await server.bind_scene_evidence(scene_id, [ref], bound_by="test")
        unbound = await server.unbind_scene_evidence(scene_id, [1], unbound_by="test")
        read_result = await server.read_scene_evidence(scene_id)
    assert rebound["status"] == "bound"
    assert unbound["status"] == "unbound"
    assert read_result["status"] == "ok"
    assert read_result["evidence_status"] == "bound"

    print(json.dumps({"status": "ok", "scene_id": scene_id}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
