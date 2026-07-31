#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from types import MethodType


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from portrait_engine import DailyPortraitMaintainer
from reflection_engine import ReflectionEngine


class _BucketManager:
    def __init__(self) -> None:
        self.write_attempted = False

    async def get(self, _bucket_id: str):
        return None

    async def create(self, **_kwargs):
        self.write_attempted = True
        raise AssertionError("reflection fallback attempted to write a bucket")

    async def update(self, *_args, **_kwargs):
        self.write_attempted = True
        raise AssertionError("reflection fallback attempted to update a bucket")


def _reflection_materials() -> dict:
    return {
        "buckets": [{"id": "memory-1", "name": "旧记忆标题"}],
        "daily_impressions": [],
        "persona_events": [],
        "conversation_turns": [],
        "commitments": [{"id": "commitment-1", "name": "旧承诺标题"}],
        "diary": None,
    }


async def _verify_reflection_skips() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        engine = ReflectionEngine(
            {
                "buckets_dir": str(Path(temp_dir) / "buckets"),
                "memory_objects": {"legacy_writes_enabled": True},
                "reflection": {
                    "enabled": True,
                    "daily_enabled": True,
                    "daily_min_memory_items": 0,
                },
            }
        )
        manager = _BucketManager()

        async def materials(self, *_args, **_kwargs):
            return _reflection_materials()

        async def invalid_output(self, *_args, **_kwargs):
            return {"title": "2026-07-31 日印象", "content": ""}

        engine._reflection_materials = MethodType(materials, engine)
        engine._reflect_model_client = MethodType(
            lambda self: (object(), "test-model", False),
            engine,
        )
        engine._api_reflect = MethodType(invalid_output, engine)
        result = await engine.reflect(
            "daily",
            manager,
            force=True,
            now=datetime.fromisoformat("2026-07-31T23:00:00+08:00"),
        )
        assert result["status"] == "skipped", result
        assert result["reason"] == "invalid_model_output", result
        assert manager.write_attempted is False

        async def generator_error(self, *_args, **_kwargs):
            raise RuntimeError("simulated model failure")

        engine._api_reflect = MethodType(generator_error, engine)
        result = await engine.reflect(
            "daily",
            manager,
            force=True,
            now=datetime.fromisoformat("2026-07-31T23:00:00+08:00"),
        )
        assert result["status"] == "skipped", result
        assert result["reason"] == "generator_error", result
        assert manager.write_attempted is False
        assert not hasattr(engine, "_fallback_reflection")


async def _verify_portrait_skips() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        state_path = Path(temp_dir) / "portrait_state.json"
        engine = DailyPortraitMaintainer(
            {
                "portrait": {
                    "enabled": True,
                    "daily_enabled": True,
                    "auto_initial_enabled": True,
                    "state_path": str(state_path),
                }
            }
        )

        async def reconcile(self, _bucket_mgr):
            return {"status": "ok"}

        async def materials(self, *_args, **_kwargs):
            return {
                "date": "2026-07-31",
                "initial": True,
                "buckets": [{"bucket_id": "memory-1", "name": "旧记忆标题"}],
                "daily_bucket_count": 1,
                "evidence_scope_limits": {},
                "existing_bucket_ids": ["memory-1"],
                "persona_stable_evidence": {},
                "persona_events": [],
                "window_shadows": [],
                "previous_portrait": self._portrait_snapshot(self._empty_state()),
            }

        async def generator_error(self, *_args, **_kwargs):
            raise RuntimeError("simulated model failure")

        engine.reconcile_evidence = MethodType(reconcile, engine)
        engine._daily_materials = MethodType(materials, engine)
        engine._api_patch = MethodType(generator_error, engine)
        engine.client = object()
        result = await engine.maintain_daily(
            _BucketManager(),
            force=True,
            now=datetime.fromisoformat("2026-07-31T23:00:00+08:00"),
        )
        assert result["status"] == "skipped", result
        assert result["reason"] == "generator_error", result
        assert not state_path.exists(), "failed portrait generation wrote state"

        engine.client = None
        result = await engine.maintain_daily(
            _BucketManager(),
            force=True,
            now=datetime.fromisoformat("2026-07-31T23:00:00+08:00"),
        )
        assert result["status"] == "skipped", result
        assert result["reason"] == "generator_unavailable", result
        assert not state_path.exists(), "unavailable portrait generator wrote state"
        assert not hasattr(engine, "_fallback_patch")


async def main() -> None:
    await _verify_reflection_skips()
    await _verify_portrait_skips()
    print("generation failure skip checks passed")


if __name__ == "__main__":
    asyncio.run(main())
