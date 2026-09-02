from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from narrative_revision_scout import (  # noqa: E402
    build_keyword_corridors,
    build_new_roll_candidate_prompt,
    normalize_new_roll_candidates,
    propose_new_roll_candidates,
)


MATERIALS = [
    {
        "source_type": "event",
        "source_id": "event_new",
        "date": "2026-09-02",
        "updated_at": "2026-09-02T10:00:00+00:00",
        "title": "英都篇继续讨论刘枭",
        "summary": "《时光代理人》英都篇的能力线索。",
        "source_excerpt": "我们继续核对刘枭与能力标记。",
        "search_text": "《时光代理人》 英都 刘枭 能力标记",
        "bound_narrative_ids": [],
        "is_unbound": True,
    },
    {
        "source_type": "event",
        "source_id": "event_old",
        "date": "2026-08-28",
        "updated_at": "2026-08-28T10:00:00+00:00",
        "title": "时光代理人剧情推演",
        "summary": "英都与刘枭的旧讨论。",
        "source_excerpt": "",
        "search_text": "时光代理人 英都 刘枭",
        "bound_narrative_ids": ["narrative_link_click"],
        "is_unbound": False,
    },
    {
        "source_type": "scene",
        "source_id": "scene_old",
        "date": "2026-08-22",
        "updated_at": "2026-08-22T10:00:00+00:00",
        "title": "共同观看时光代理人",
        "summary": "一起看英都篇。",
        "source_excerpt": "画面中的能力标记与人物位置。",
        "search_text": "时光代理人 英都 能力标记",
        "bound_narrative_ids": [],
        "is_unbound": True,
    },
    {
        "source_type": "event",
        "source_id": "event_noise",
        "date": "2026-09-01",
        "updated_at": "2026-09-01T10:00:00+00:00",
        "title": "过期 token",
        "summary": "一次凭据维护。",
        "source_excerpt": "",
        "search_text": "过期 token 配置",
        "bound_narrative_ids": [],
        "is_unbound": True,
    },
]


class FakeCompletions:
    def __init__(self, payload):
        self.payload = payload
        self.options = []

    async def create(self, **options):
        self.options.append(options)
        content = json.dumps(self.payload, ensure_ascii=False)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


async def main() -> int:
    rules = "只做候选侦察，不写正文，不发布。"
    corridors = build_keyword_corridors(MATERIALS, ["event:event_new"], max_candidates_per_seed=8)
    assert len(corridors) == 1
    assert corridors[0]["seed_key"] == "event:event_new"
    assert [item["source_id"] for item in corridors[0]["candidates"]] == ["scene_old"]
    assert "event_old" not in {item["source_id"] for item in corridors[0]["candidates"]}
    assert "event_noise" not in {item["source_id"] for item in corridors[0]["candidates"]}
    assert corridors[0]["inventory_size"] == 4
    assert any(item["term"] == "时光代理人" for item in corridors[0]["keywords"])
    assert build_keyword_corridors(MATERIALS, ["event:event_old"]) == []

    prompt = build_new_roll_candidate_prompt(corridors, role_rules=rules)
    assert len(prompt) == 2
    assert prompt[0]["content"] == rules
    assert "source_excerpt" in prompt[1]["content"]
    assert "换一种表达" not in prompt[1]["content"]

    payload = {
        "candidates": [
            {
                "seed_source_type": "event",
                "seed_source_id": "event_new",
                "title": "《时光代理人》共同观看与推演",
                "reason": "新旧材料共同推进同一作品的持续讨论。",
                "materials": [
                    {"source_type": "event", "source_id": "event_new"},
                    {"source_type": "event", "source_id": "event_old"},
                    {"source_type": "scene", "source_id": "scene_old"},
                    {"source_type": "event", "source_id": "foreign"},
                ],
                "confidence": "high",
                "latest_date": "2026-09-02",
            }
        ]
    }
    normalized = normalize_new_roll_candidates(payload, corridors)
    assert normalized[0]["source_event_ids"] == ["event_new"]
    assert normalized[0]["source_scene_ids"] == ["scene_old"]
    assert normalized[0]["seed_source_id"] == "event_new"

    completions = FakeCompletions(payload)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    proposed = await propose_new_roll_candidates(
        client=client,
        model="gpt-5.6-terra",
        corridors=corridors,
        role_rules=rules,
        completion_options={"max_tokens": 800, "temperature": 0.0},
    )
    assert proposed[0]["source_scene_ids"] == ["scene_old"]
    assert len(completions.options) == 1
    assert completions.options[0]["model"] == "gpt-5.6-terra"
    assert "tools" not in completions.options[0]

    server_text = (ROOT / "server.py").read_text(encoding="utf-8")
    assert "propose_seed_search_queries" not in server_text
    assert "build_query_corridors" not in server_text
    assert server_text.count("candidates = await propose_new_roll_candidates(") == 1
    assert "host_literal_keywords_then_active_exact_search_then_terra_review" in server_text
    assert "reconcile_bound_new_roll_materials" in server_text
    print("NARRATIVE_REVISION_SCOUT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
