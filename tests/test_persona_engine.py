import json
import sqlite3
from copy import deepcopy
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from persona_engine import FALLBACK_GUIDANCE, PersonaStateEngine


class FakePersonaClient:
    def __init__(self, content: str):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )
        self.content = content

    async def _create(self, **kwargs):
        message = SimpleNamespace(content=self.content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


def _persona_config(test_config: dict, **persona_overrides) -> dict:
    cfg = deepcopy(test_config)
    cfg["dehydration"]["api_key"] = ""
    cfg["persona"] = {
        **cfg["persona"],
        "api_key": "",
        **persona_overrides,
    }
    return cfg


def _event_payload(**overrides) -> str:
    data = {
        "event_type": "affection",
        "perceived_intent": "user expresses warmth",
        "affect_delta": {"valence": 0.05, "arousal": 0.02},
        "relationship_delta": {"affinity": 0.02, "dominance": 0.0, "defensiveness": -0.01, "trust": 0.02},
        "personality_delta": {
            "openness": 0.002,
            "conscientiousness": 0.0,
            "extraversion": 0.001,
            "agreeableness": 0.003,
            "neuroticism": -0.001,
        },
        "mood_label": "warm_touched",
        "reply_guidance": "Respond warmly and gently.",
        "confidence": 0.9,
    }
    data.update(overrides)
    return json.dumps(data, ensure_ascii=False)


def _event_count(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM persona_events").fetchone()[0]
    conn.close()
    return count


def test_persona_initializes_default_global_and_session_state(test_config):
    engine = PersonaStateEngine(_persona_config(test_config))
    state = engine.get_current_state("session-a")

    assert state["profile_id"] == "haven_xiaoyu"
    assert state["personality"]["agreeableness"] == pytest.approx(0.66)
    assert state["relationship"]["affinity"] == pytest.approx(0.86)
    assert state["affect"]["mood_label"] == "warm_neutral"
    assert "Current Inner State" in engine.format_state_block(state)


@pytest.mark.asyncio
async def test_persona_llm_update_clips_deltas_and_records_event(test_config):
    cfg = _persona_config(test_config)
    engine = PersonaStateEngine(cfg)
    engine.client = FakePersonaClient(
        _event_payload(
            affect_delta={"valence": 10, "arousal": 10},
            relationship_delta={"affinity": 10, "dominance": 10, "defensiveness": 10, "trust": 10},
            personality_delta={
                "openness": 10,
                "conscientiousness": 10,
                "extraversion": 10,
                "agreeableness": 10,
                "neuroticism": 10,
            },
        )
    )

    state = await engine.update_from_user_message("session-a", "爱你爱你")

    assert state["personality"]["openness"] == pytest.approx(0.57)
    assert state["relationship"]["affinity"] == pytest.approx(0.89)
    assert state["relationship"]["defensiveness"] == pytest.approx(0.15)
    assert state["affect"]["valence"] == pytest.approx(0.74)
    assert state["affect"]["arousal"] == pytest.approx(0.52)
    assert state["reply_guidance"] == "Respond warmly and gently."
    assert _event_count(engine.db_path) == 1


def test_persona_session_mood_half_life_decay(test_config):
    cfg = _persona_config(test_config, session_mood_half_life_minutes=90)
    engine = PersonaStateEngine(cfg)
    engine.get_current_state("session-decay")

    old_time = (datetime.now() - timedelta(minutes=90)).isoformat(timespec="seconds")
    conn = sqlite3.connect(engine.db_path)
    conn.execute(
        """
        UPDATE persona_session_state
        SET valence = ?, arousal = ?, session_defensiveness = ?, updated_at = ?
        WHERE profile_id = ? AND session_id = ?
        """,
        (1.0, 1.0, 1.0, old_time, engine.profile_id, "session-decay"),
    )
    conn.commit()
    conn.close()

    state = engine.get_current_state("session-decay")

    assert state["affect"]["valence"] == pytest.approx(0.78, abs=0.01)
    assert state["affect"]["arousal"] == pytest.approx(0.67, abs=0.01)
    assert state["relationship"]["defensiveness"] == pytest.approx(0.56, abs=0.01)


@pytest.mark.asyncio
async def test_persona_malformed_json_keeps_state_and_records_raw_response(test_config):
    engine = PersonaStateEngine(_persona_config(test_config))
    before = engine.get_current_state("session-bad-json")
    engine.client = FakePersonaClient("```json\nnot-json\n```")

    after = await engine.update_from_user_message("session-bad-json", "今天怪怪的")

    assert after["personality"] == before["personality"]
    assert after["relationship"] == before["relationship"]
    assert after["reply_guidance"] == FALLBACK_GUIDANCE

    conn = sqlite3.connect(engine.db_path)
    row = conn.execute("SELECT raw_response, error FROM persona_events").fetchone()
    conn.close()
    assert "not-json" in row[0]
    assert "malformed JSON" in row[1]


@pytest.mark.asyncio
async def test_persona_missing_key_uses_existing_state_fallback(test_config):
    engine = PersonaStateEngine(_persona_config(test_config))

    state = await engine.update_from_user_message("session-no-key", "哥哥你在吗")

    assert state["reply_guidance"] == FALLBACK_GUIDANCE
    assert state["affect"]["mood_label"] == "warm_neutral"
    assert _event_count(engine.db_path) == 1
