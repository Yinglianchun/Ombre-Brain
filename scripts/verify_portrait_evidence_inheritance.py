"""Verify one-read/one-publish Portrait evidence inheritance."""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import server
from portrait_engine import DailyPortraitMaintainer


class _SceneBuckets:
    def __init__(self, scene: dict):
        self.scene = scene

    async def get(self, bucket_id: str) -> dict | None:
        return self.scene if bucket_id == self.scene["id"] else None


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="ombre-portrait-evidence-") as tmp:
        root = Path(tmp)
        engine = DailyPortraitMaintainer(
            {
                "state_dir": str(root / "state"),
                "portrait": {
                    "enabled": False,
                    "state_path": str(root / "state" / "portrait_state.json"),
                },
            }
        )
        scene = {
            "id": "scene_portrait_source",
            "content": "这条亲历 Scene 是画像候选的原文证据。",
            "metadata": {
                "name": "画像候选的来源",
                "object_kind": "scene",
                "date": "2026-07-25",
            },
        }
        state = engine.load_state()
        state["stable_candidates"].append(
            {
                "scope": "user",
                "text": "候选观察",
                "evidence": [{"bucket_id": scene["id"]}],
                "status": "candidate",
                "created_at": "2026-07-25T12:00:00+08:00",
            }
        )
        engine.save_state(state)

        server.portrait_engine = engine
        server.bucket_mgr = _SceneBuckets(scene)

        reviewed = await server.read_portrait(scope="user")
        scope_state = reviewed["scopes"]["user"]
        assert "publication_boundary" not in reviewed
        assert scope_state["revision"] == 0
        assert (
            scope_state["candidate_materials"][-1]["resolved_evidence"][0]["content"]
            == scene["content"]
        )

        published = await server.publish_portrait(
            scope="user",
            text="我确认这条观察可以进入用户画像。",
            expected_revision=scope_state["revision"],
        )
        assert published["status"] == "updated"
        assert published["revision"] == 1
        assert published["resolved_evidence"][0]["bucket_id"] == scene["id"]
        assert published["evidence_inherited_from"] == "2026-07-25T12:00:00+08:00"

    print("Portrait candidate evidence inheritance verified")


if __name__ == "__main__":
    asyncio.run(main())
