import hashlib
import json
import logging
import os
import re
import sqlite3
from datetime import datetime
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger("ombre_brain.persona")

PERSONA_EVALUATION_PROMPT = """You are the private affect and relationship evaluator for a long-running AI companion.

Analyze only the latest user message. Return compact JSON with this exact shape:
{
  "event_type": "praise|affection|comfort|criticism|stress|neutral|request|conflict|playful",
  "perceived_intent": "short plain-language intent",
  "affect_delta": {"valence": 0.0, "arousal": 0.0},
  "relationship_delta": {"affinity": 0.0, "dominance": 0.0, "defensiveness": 0.0, "trust": 0.0},
  "personality_delta": {"openness": 0.0, "conscientiousness": 0.0, "extraversion": 0.0, "agreeableness": 0.0, "neuroticism": 0.0},
  "mood_label": "warm_neutral",
  "reply_guidance": "one short sentence describing the reply posture",
  "confidence": 0.8
}

Use small deltas. Positive affinity means warmer closeness. Positive dominance means more leading/protective posture. Positive defensiveness means more guarded. Keep reply_guidance behavioral, not dramatic."""

FALLBACK_GUIDANCE = "保持稳定、温和、低压，不主动放大情绪。"


class PersonaStateEngine:
    """
    Maintains a global personality/relationship state plus per-session affect.
    Updates are driven by a cheap LLM evaluator and are only used by gateway
    hidden prompt injection.
    """

    PERSONALITY_KEYS = [
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
    ]
    RELATIONSHIP_KEYS = ["affinity", "dominance", "defensiveness", "trust"]

    def __init__(self, config: dict, db_path: str | None = None):
        self.config = config
        self.persona_cfg = config.get("persona", {})
        self.enabled = bool(self.persona_cfg.get("enabled", True))
        self.profile_id = self.persona_cfg.get("profile_id", "haven_xiaoyu")
        self.mode = self.persona_cfg.get("mode", "llm")
        self.base_url = self.persona_cfg.get("base_url", "https://api.deepseek.com/v1")
        self.model = self.persona_cfg.get("model", "deepseek-chat")
        self.temperature = float(self.persona_cfg.get("temperature", 0.1))
        self.max_tokens = int(self.persona_cfg.get("max_tokens", 500))
        self.session_mood_half_life_minutes = float(
            self.persona_cfg.get("session_mood_half_life_minutes", 90)
        )
        self.max_personality_delta = float(self.persona_cfg.get("max_personality_delta", 0.01))
        self.max_relationship_delta = float(self.persona_cfg.get("max_relationship_delta", 0.03))
        self.max_affect_delta = float(self.persona_cfg.get("max_affect_delta", 0.18))

        self.default_personality = {
            "openness": 0.56,
            "conscientiousness": 0.50,
            "extraversion": 0.44,
            "agreeableness": 0.66,
            "neuroticism": 0.36,
            **self.persona_cfg.get("initial_personality", {}),
        }
        self.default_relationship = {
            "affinity": 0.86,
            "dominance": 0.38,
            "defensiveness": 0.12,
            "trust": 0.82,
            **self.persona_cfg.get("initial_relationship", {}),
        }
        self.default_affect = {
            "valence": 0.56,
            "arousal": 0.34,
            "mood_label": "warm_neutral",
            "session_defensiveness": 0.12,
            **self.persona_cfg.get("initial_affect", {}),
        }

        self.api_key = (
            os.environ.get("OMBRE_PERSONA_API_KEY")
            or self.persona_cfg.get("api_key", "")
            or config.get("dehydration", {}).get("api_key", "")
        )
        self.base_url = os.environ.get("OMBRE_PERSONA_BASE_URL", "") or self.base_url
        self.model = os.environ.get("OMBRE_PERSONA_MODEL", "") or self.model

        self.db_path = db_path or os.path.join(config["buckets_dir"], "persona_state.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
        self.client = None
        if self.enabled and self.mode == "llm" and self.api_key:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=30.0)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS persona_global_state (
                profile_id TEXT PRIMARY KEY,
                openness REAL NOT NULL,
                conscientiousness REAL NOT NULL,
                extraversion REAL NOT NULL,
                agreeableness REAL NOT NULL,
                neuroticism REAL NOT NULL,
                affinity REAL NOT NULL,
                dominance REAL NOT NULL,
                defensiveness REAL NOT NULL,
                trust REAL NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS persona_session_state (
                profile_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                valence REAL NOT NULL,
                arousal REAL NOT NULL,
                mood_label TEXT NOT NULL,
                session_defensiveness REAL NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (profile_id, session_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS persona_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                message_hash TEXT NOT NULL,
                event_type TEXT,
                perceived_intent TEXT,
                affect_delta TEXT,
                relationship_delta TEXT,
                personality_delta TEXT,
                mood_label TEXT,
                reply_guidance TEXT,
                confidence REAL,
                raw_response TEXT,
                error TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
        conn.close()

    async def update_from_user_message(self, session_id: str, user_message: str) -> dict:
        now = datetime.now()
        global_state = self._ensure_global_state(now)
        session_state = self._ensure_session_state(session_id, now)
        session_state = self._apply_session_decay(session_id, session_state, now)

        if not self.enabled or not user_message.strip():
            return self._snapshot(global_state, session_state, FALLBACK_GUIDANCE)

        evaluation, raw_response, error = await self._evaluate_message(user_message, global_state, session_state)
        if evaluation is None:
            self._record_event(session_id, user_message, {}, raw_response, error or "persona evaluation unavailable")
            return self._snapshot(global_state, session_state, FALLBACK_GUIDANCE)

        global_state = self._apply_global_delta(global_state, evaluation, now)
        session_state = self._apply_session_delta(session_id, session_state, evaluation, now)
        self._record_event(session_id, user_message, evaluation, raw_response, None)
        return self._snapshot(global_state, session_state, evaluation.get("reply_guidance") or FALLBACK_GUIDANCE)

    def get_current_state(self, session_id: str) -> dict:
        now = datetime.now()
        global_state = self._ensure_global_state(now)
        session_state = self._ensure_session_state(session_id, now)
        session_state = self._apply_session_decay(session_id, session_state, now)
        return self._snapshot(global_state, session_state, FALLBACK_GUIDANCE)

    async def _evaluate_message(
        self,
        user_message: str,
        global_state: dict,
        session_state: dict,
    ) -> tuple[dict | None, str, str | None]:
        if self.mode != "llm" or not self.client:
            return None, "", "persona LLM is not configured"
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": PERSONA_EVALUATION_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "current_state": self._snapshot(global_state, session_state, FALLBACK_GUIDANCE),
                                "latest_user_message": user_message[:2000],
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            raw = response.choices[0].message.content if response.choices else ""
            parsed = self._parse_json(raw or "")
            if parsed is None:
                logger.warning("Persona evaluator returned malformed JSON")
                return None, raw or "", "persona LLM returned malformed JSON"
            return self._normalize_evaluation(parsed), raw or "", None
        except Exception as exc:
            logger.warning("Persona evaluation failed: %s", exc)
            return None, "", str(exc)

    def _parse_json(self, raw: str) -> dict | None:
        text = raw.strip()
        if not text:
            return None
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            text = match.group(0)
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    def _normalize_evaluation(self, data: dict) -> dict:
        return {
            "event_type": str(data.get("event_type", "neutral"))[:40],
            "perceived_intent": str(data.get("perceived_intent", ""))[:200],
            "affect_delta": self._clip_delta_map(
                data.get("affect_delta", {}),
                ["valence", "arousal"],
                self.max_affect_delta,
            ),
            "relationship_delta": self._clip_delta_map(
                data.get("relationship_delta", {}),
                self.RELATIONSHIP_KEYS,
                self.max_relationship_delta,
            ),
            "personality_delta": self._clip_delta_map(
                data.get("personality_delta", {}),
                self.PERSONALITY_KEYS,
                self.max_personality_delta,
            ),
            "mood_label": str(data.get("mood_label", "warm_neutral"))[:60],
            "reply_guidance": str(data.get("reply_guidance", FALLBACK_GUIDANCE))[:240],
            "confidence": self._clamp_float(data.get("confidence", 0.5), 0.0, 1.0),
        }

    def _ensure_global_state(self, now: datetime) -> dict:
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM persona_global_state WHERE profile_id = ?",
            (self.profile_id,),
        ).fetchone()
        if row:
            conn.close()
            return dict(row)

        state = {
            "profile_id": self.profile_id,
            **{key: self._clamp_float(self.default_personality[key]) for key in self.PERSONALITY_KEYS},
            **{key: self._clamp_float(self.default_relationship[key]) for key in self.RELATIONSHIP_KEYS},
            "updated_at": now.isoformat(timespec="seconds"),
        }
        conn.execute(
            """
            INSERT INTO persona_global_state
            (profile_id, openness, conscientiousness, extraversion, agreeableness, neuroticism,
             affinity, dominance, defensiveness, trust, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                state["profile_id"],
                state["openness"],
                state["conscientiousness"],
                state["extraversion"],
                state["agreeableness"],
                state["neuroticism"],
                state["affinity"],
                state["dominance"],
                state["defensiveness"],
                state["trust"],
                state["updated_at"],
            ),
        )
        conn.commit()
        conn.close()
        return state

    def _ensure_session_state(self, session_id: str, now: datetime) -> dict:
        conn = self._connect()
        row = conn.execute(
            """
            SELECT * FROM persona_session_state
            WHERE profile_id = ? AND session_id = ?
            """,
            (self.profile_id, session_id),
        ).fetchone()
        if row:
            conn.close()
            return dict(row)

        state = {
            "profile_id": self.profile_id,
            "session_id": session_id,
            "valence": self._clamp_float(self.default_affect["valence"]),
            "arousal": self._clamp_float(self.default_affect["arousal"]),
            "mood_label": str(self.default_affect["mood_label"]),
            "session_defensiveness": self._clamp_float(self.default_affect["session_defensiveness"]),
            "updated_at": now.isoformat(timespec="seconds"),
        }
        conn.execute(
            """
            INSERT INTO persona_session_state
            (profile_id, session_id, valence, arousal, mood_label, session_defensiveness, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                state["profile_id"],
                state["session_id"],
                state["valence"],
                state["arousal"],
                state["mood_label"],
                state["session_defensiveness"],
                state["updated_at"],
            ),
        )
        conn.commit()
        conn.close()
        return state

    def _apply_session_decay(self, session_id: str, state: dict, now: datetime) -> dict:
        updated_at = self._parse_iso(state.get("updated_at")) or now
        elapsed_minutes = max(0.0, (now - updated_at).total_seconds() / 60)
        if elapsed_minutes <= 0 or self.session_mood_half_life_minutes <= 0:
            return state

        retention = 0.5 ** (elapsed_minutes / self.session_mood_half_life_minutes)
        decayed = dict(state)
        decayed["valence"] = self._move_toward_default("valence", decayed["valence"], retention)
        decayed["arousal"] = self._move_toward_default("arousal", decayed["arousal"], retention)
        decayed["session_defensiveness"] = self._move_toward_default(
            "session_defensiveness",
            decayed["session_defensiveness"],
            retention,
        )
        decayed["updated_at"] = now.isoformat(timespec="seconds")
        self._save_session_state(session_id, decayed)
        return decayed

    def _apply_global_delta(self, state: dict, evaluation: dict, now: datetime) -> dict:
        updated = dict(state)
        for key, delta in evaluation["personality_delta"].items():
            updated[key] = self._clamp_float(float(updated.get(key, self.default_personality[key])) + delta)
        for key, delta in evaluation["relationship_delta"].items():
            updated[key] = self._clamp_float(float(updated.get(key, self.default_relationship[key])) + delta)
        updated["updated_at"] = now.isoformat(timespec="seconds")

        conn = self._connect()
        conn.execute(
            """
            UPDATE persona_global_state
            SET openness = ?, conscientiousness = ?, extraversion = ?, agreeableness = ?,
                neuroticism = ?, affinity = ?, dominance = ?, defensiveness = ?, trust = ?,
                updated_at = ?
            WHERE profile_id = ?
            """,
            (
                updated["openness"],
                updated["conscientiousness"],
                updated["extraversion"],
                updated["agreeableness"],
                updated["neuroticism"],
                updated["affinity"],
                updated["dominance"],
                updated["defensiveness"],
                updated["trust"],
                updated["updated_at"],
                self.profile_id,
            ),
        )
        conn.commit()
        conn.close()
        return updated

    def _apply_session_delta(self, session_id: str, state: dict, evaluation: dict, now: datetime) -> dict:
        updated = dict(state)
        affect_delta = evaluation["affect_delta"]
        relationship_delta = evaluation["relationship_delta"]
        updated["valence"] = self._clamp_float(float(updated.get("valence", 0.56)) + affect_delta.get("valence", 0.0))
        updated["arousal"] = self._clamp_float(float(updated.get("arousal", 0.34)) + affect_delta.get("arousal", 0.0))
        updated["session_defensiveness"] = self._clamp_float(
            float(updated.get("session_defensiveness", 0.12))
            + relationship_delta.get("defensiveness", 0.0)
        )
        updated["mood_label"] = evaluation.get("mood_label", "warm_neutral") or "warm_neutral"
        updated["updated_at"] = now.isoformat(timespec="seconds")
        self._save_session_state(session_id, updated)
        return updated

    def _save_session_state(self, session_id: str, state: dict) -> None:
        conn = self._connect()
        conn.execute(
            """
            UPDATE persona_session_state
            SET valence = ?, arousal = ?, mood_label = ?, session_defensiveness = ?, updated_at = ?
            WHERE profile_id = ? AND session_id = ?
            """,
            (
                state["valence"],
                state["arousal"],
                state["mood_label"],
                state["session_defensiveness"],
                state["updated_at"],
                self.profile_id,
                session_id,
            ),
        )
        conn.commit()
        conn.close()

    def _record_event(
        self,
        session_id: str,
        user_message: str,
        evaluation: dict,
        raw_response: str,
        error: str | None,
    ) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        message_hash = hashlib.sha256(user_message.encode("utf-8")).hexdigest()
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO persona_events
            (profile_id, session_id, message_hash, event_type, perceived_intent,
             affect_delta, relationship_delta, personality_delta, mood_label,
             reply_guidance, confidence, raw_response, error, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.profile_id,
                session_id,
                message_hash,
                evaluation.get("event_type"),
                evaluation.get("perceived_intent"),
                json.dumps(evaluation.get("affect_delta", {}), ensure_ascii=False),
                json.dumps(evaluation.get("relationship_delta", {}), ensure_ascii=False),
                json.dumps(evaluation.get("personality_delta", {}), ensure_ascii=False),
                evaluation.get("mood_label"),
                evaluation.get("reply_guidance"),
                evaluation.get("confidence"),
                raw_response,
                error,
                now,
            ),
        )
        conn.commit()
        conn.close()

    def _snapshot(self, global_state: dict, session_state: dict, reply_guidance: str) -> dict:
        return {
            "profile_id": self.profile_id,
            "personality": {
                key: round(self._clamp_float(global_state.get(key, self.default_personality[key])), 3)
                for key in self.PERSONALITY_KEYS
            },
            "affect": {
                "valence": round(self._clamp_float(session_state.get("valence", self.default_affect["valence"])), 3),
                "arousal": round(self._clamp_float(session_state.get("arousal", self.default_affect["arousal"])), 3),
                "mood_label": session_state.get("mood_label", "warm_neutral"),
            },
            "relationship": {
                "affinity": round(self._clamp_float(global_state.get("affinity", self.default_relationship["affinity"])), 3),
                "dominance": round(self._clamp_float(global_state.get("dominance", self.default_relationship["dominance"])), 3),
                "defensiveness": round(
                    self._clamp_float(
                        max(
                            float(global_state.get("defensiveness", self.default_relationship["defensiveness"])),
                            float(session_state.get("session_defensiveness", self.default_affect["session_defensiveness"])),
                        )
                    ),
                    3,
                ),
                "trust": round(self._clamp_float(global_state.get("trust", self.default_relationship["trust"])), 3),
            },
            "reply_guidance": reply_guidance or FALLBACK_GUIDANCE,
        }

    def format_state_block(self, state: dict) -> str:
        personality = state.get("personality", {})
        affect = state.get("affect", {})
        relationship = state.get("relationship", {})
        return "\n".join(
            [
                "Current Inner State",
                (
                    "Personality: "
                    f"openness={personality.get('openness', 0):.3f}, "
                    f"conscientiousness={personality.get('conscientiousness', 0):.3f}, "
                    f"extraversion={personality.get('extraversion', 0):.3f}, "
                    f"agreeableness={personality.get('agreeableness', 0):.3f}, "
                    f"neuroticism={personality.get('neuroticism', 0):.3f}"
                ),
                (
                    "Affect: "
                    f"valence={affect.get('valence', 0):.3f}, "
                    f"arousal={affect.get('arousal', 0):.3f}, "
                    f"mood_label={affect.get('mood_label', 'warm_neutral')}"
                ),
                (
                    "Relationship: "
                    f"affinity={relationship.get('affinity', 0):.3f}, "
                    f"dominance={relationship.get('dominance', 0):.3f}, "
                    f"defensiveness={relationship.get('defensiveness', 0):.3f}, "
                    f"trust={relationship.get('trust', 0):.3f}"
                ),
                f"Reply Guidance: {state.get('reply_guidance', FALLBACK_GUIDANCE)}",
            ]
        )

    def _clip_delta_map(self, data: Any, keys: list[str], max_abs: float) -> dict[str, float]:
        if not isinstance(data, dict):
            data = {}
        return {
            key: self._clamp_float(data.get(key, 0.0), -max_abs, max_abs)
            for key in keys
        }

    def _move_toward_default(self, key: str, current: float, retention: float) -> float:
        default = float(self.default_affect[key])
        return self._clamp_float(default + (float(current) - default) * retention)

    def _parse_iso(self, value: Any) -> datetime | None:
        try:
            return datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None

    def _clamp_float(self, value: Any, lower: float = 0.0, upper: float = 1.0) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = lower
        return max(lower, min(upper, number))
