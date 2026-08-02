"""Verify grounded daily summaries and relationship observations."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reflection_engine import ReflectionEngine


class _FakeCompletions:
    def __init__(self, *, content: str = "", model: str = "", error: Exception | None = None):
        self.content = content
        self.model = model
        self.error = error

    async def create(self, **_kwargs):
        if self.error:
            raise self.error
        message = SimpleNamespace(content=self.content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice], model=self.model)


class _FakeClient:
    def __init__(self, **kwargs):
        self.chat = SimpleNamespace(completions=_FakeCompletions(**kwargs))


class _TurnStore:
    def list_conversation_turns_between(self, **_kwargs):
        return [
            {
                "id": 17,
                "session_id": "session-grounding",
                "round_id": 3,
                "created_at": "2026-07-17T12:00:00+08:00",
                "user_text": "我还是想看没 summary 的思考链。",
                "assistant_text": "我解释了当前限制。",
            }
        ]


def _reflection_engine(root: Path) -> ReflectionEngine:
    engine = ReflectionEngine(
        {
            "state_dir": str(root / "state"),
            "reflection": {
                "enabled": True,
                "daily_activity_summary_enabled": True,
                "model": "primary-model",
            },
        }
    )
    engine.enabled = True
    engine.daily_activity_summary_enabled = True
    return engine


async def _verify_failed_generation_is_skipped_and_audited(root: Path) -> None:
    engine = _reflection_engine(root)
    engine.client = _FakeClient(content="not json", model="actual-primary")
    engine.model = "requested-primary"
    engine.dehydration_client = _FakeClient(error=RuntimeError("retry unavailable"))
    engine.daily_activity_summary_dehydration_client = engine.dehydration_client
    engine.dehydration_model = "requested-retry"
    engine.dehydration_base_url = "https://retry.invalid/v1"
    engine.dehydration_api_key = "configured"

    result = await engine.run_daily_activity_summary(
        conversation_turn_store=_TurnStore(),
        daily_impressions=[
            {
                "id": "reflection_daily_2026-07-17",
                "content": "合作顺畅，信任感增强。",
            }
        ],
        key="2026-07-17",
    )
    assert result["status"] == "skipped"
    assert result["reason"] == "activity_summary_model_unusable"
    assert "activity_summary" not in result
    diagnostic = result["diagnostics"]
    assert diagnostic["fallback_used"] is False
    assert diagnostic["retry_used"] is True
    assert diagnostic["attempts"][0]["requested_model"] == "requested-primary"
    assert diagnostic["attempts"][0]["response_model"] == "actual-primary"
    assert diagnostic["attempts"][0]["raw_response"] == "not json"
    assert diagnostic["attempts"][0]["failure_reason"] == "invalid_json_or_non_object"
    assert diagnostic["attempts"][1]["error_type"] == "RuntimeError"

    rows = [
        json.loads(line)
        for line in Path(engine.daily_activity_summary_diagnostics_path)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert rows[-1] == diagnostic
    assert "围绕" not in json.dumps(result, ensure_ascii=False)


async def _verify_source_ids_are_filtered(root: Path) -> None:
    engine = _reflection_engine(root)
    engine.client = _FakeClient(
        content=json.dumps(
            {
                "summary": "继续排查思考链展示限制，并保留用户仍在追问这一事实。",
                "confidence": 0.82,
                "source_turn_ids": [17, 999],
                "source_event_ids": [9999],
            },
            ensure_ascii=False,
        ),
        model="actual-primary",
    )
    engine.model = "requested-primary"
    result = await engine.run_daily_activity_summary(
        conversation_turn_store=_TurnStore(),
        key="2026-07-17",
    )
    assert result["status"] == "ready"
    item = result["activity_summary"]
    assert item["source_turn_ids"] == [17]
    assert item["source_event_ids"] == []


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="ombre-summary-grounding-") as tmp:
        root = Path(tmp)
        await _verify_failed_generation_is_skipped_and_audited(root)
        await _verify_source_ids_are_filtered(root)

    print("Summary grounding verified")


if __name__ == "__main__":
    asyncio.run(main())
