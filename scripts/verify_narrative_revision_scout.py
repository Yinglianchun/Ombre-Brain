from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from narrative_revision_scout import (
    build_new_roll_candidate_prompt,
    normalize_new_roll_candidates,
    propose_new_roll_candidates,
)


EVENTS = [
    {"event_id": "event_a", "date": "2026-08-01", "title": "开始", "summary": "第一步。"},
    {"event_id": "event_b", "date": "2026-08-02", "title": "继续", "summary": "同一件事继续。"},
    {"event_id": "event_c", "date": "2026-08-03", "title": "旁支", "summary": "另一件事。"},
]


class FakeCompletions:
    def __init__(self):
        self.options = None

    async def create(self, **options):
        self.options = options
        content = json.dumps({
            "candidates": [{
                "title": "持续发生的一件事",
                "reason": "A 与 B 是同一条推进线。",
                "source_event_ids": ["event_a", "event_b", "event_foreign"],
                "confidence": "high",
                "latest_date": "2026-08-02",
            }]
        }, ensure_ascii=False)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


async def main() -> int:
    messages = build_new_roll_candidate_prompt(EVENTS)
    assert len(messages) == 2
    assert "不是 Writer" in messages[0]["content"]

    normalized = normalize_new_roll_candidates(
        {"candidates": [{
            "title": "持续发生的一件事",
            "reason": "A 与 B 是同一条推进线。",
            "source_event_ids": ["event_a", "event_b", "event_foreign"],
            "confidence": "medium",
        }]},
        {"event_a", "event_b", "event_c"},
    )
    assert normalized[0]["source_event_ids"] == ["event_a", "event_b"]

    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    proposed = await propose_new_roll_candidates(
        client=client,
        model="external-test-model",
        events=EVENTS,
        completion_options={"temperature": 0.0},
    )
    assert proposed[0]["title"] == "持续发生的一件事"
    assert completions.options["model"] == "external-test-model"
    assert completions.options["temperature"] == 0.0
    print("narrative revision scout verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
