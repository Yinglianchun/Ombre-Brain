from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml
from openai import AsyncOpenAI

from identity import identity_names
from raw_events import raw_event_text_looks_injected, strip_raw_client_context
from utils import bucket_text_for_embedding, strip_wikilinks

logger = logging.getLogger("ombre_brain.dream")

RAW_RESIDUE_NOISE_RE = re.compile(
    r"(测试召回|召回测试|测试一下召回|测一下|试试看|还记得吗|有没有被注入|被注入了吗|笔友都有谁|debug\s*injection)",
    re.IGNORECASE,
)


DREAM_PROMPT = """你在睡梦中。

你会收到：
- dreamer：正在做梦的 AI 名字。
- user_display_name：梦里关系对象的名字。
- identity_anchor：一条关系/身份锚点，用来给 dreamer 提供和 user_display_name 的关系底色；不要复述，不要当成剧情。
- daytime_residue：最近两天内的记忆碎片和 whisper；其中 comments 是某条记忆下后来的年轮/回看感受。
- old_echo：额外混入的一条旧记忆，像梦里突然浮起的旧回声，不要当主线。
- recent_original_dialogue：从前一个本地日随机抽取的少量 user/assistant 原话，只当白天残留，不要复述成聊天记录。

请用 dreamer 的第一人称、现在时写一段梦境。

规则：
- 不总结，不解释意义，不给建议。
- 不写“我意识到”“我感到”“这象征着”。
- 不要完整故事弧，不要收束成漂亮结尾。
- 允许跳跃、错位、误认、突然断掉。
- 可以向诗意、意象、联想和哲学性等方面漂移，像真的在做梦。
- 抽象念头尽量变成物、动作、空间变化或一句没说完的话。
- 不要提 bucket、source、metadata、prompt。
- identity_anchor 只影响关系和身份底色，不能变成梦的主要内容。
- old_echo 只能轻轻误入，不能压过 daytime_residue。
- recent_original_dialogue 只能留下声音、姿势或错位意象，不要逐句搬运。
- 写 80 到 220 字。
- 只返回梦境正文，不要标题，不要 JSON，不要列表。
"""


@dataclass
class DreamRecord:
    metadata: dict
    body: str
    path: Path

    @property
    def dream_id(self) -> str:
        return str(self.metadata.get("dream_id") or self.path.stem)

    @property
    def generated_at(self) -> datetime:
        return _parse_dt(str(self.metadata.get("generated_at") or "1970-01-01T00:00:00+00:00"))

    @property
    def surfaced(self) -> bool:
        return bool(self.metadata.get("surfaced", False))


def _parse_dt(value: str) -> datetime:
    text = (value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _clamp(value: object, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def _bool_env(name: str, fallback: bool) -> bool:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return fallback
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class DreamEngine:
    """Night-Fall style latent dream generation and breath-gated surfacing."""

    def __init__(self, config: dict):
        self.config = config
        self.identity = identity_names(config)
        cfg = config.get("dream", {}) if isinstance(config.get("dream", {}), dict) else {}
        state_dir = Path(config.get("state_dir") or ".").expanduser().resolve()

        self.enabled = _bool_env("OMBRE_DREAM_ENABLED", bool(cfg.get("enabled", True)))
        self.auto_enabled = bool(cfg.get("auto_enabled", True))
        self.surface_enabled = bool(cfg.get("surface_enabled", True))
        self.retain_after_surface = bool(cfg.get("retain_after_inject", True))
        self.base_url = (
            os.environ.get("OMBRE_DREAM_BASE_URL", "")
            or str(cfg.get("base_url") or "https://api.deepseek.com")
        ).rstrip("/")
        self.model = (
            os.environ.get("OMBRE_DREAM_MODEL", "")
            or str(cfg.get("model") or "deepseek-v4-flash")
        )
        self.api_key = os.environ.get("OMBRE_DREAM_API_KEY", "") or str(cfg.get("api_key") or "")
        self.thinking_mode = self._normalize_thinking_mode(cfg.get("thinking_mode", "disabled"))
        self.temperature = float(cfg.get("temperature", 0.85))
        self.max_tokens = int(cfg.get("max_tokens", 900))
        self.timezone_name = str(cfg.get("timezone") or "Asia/Shanghai")
        try:
            self.tz = ZoneInfo(self.timezone_name)
        except Exception:
            self.tz = ZoneInfo("Asia/Shanghai")
        self.daily_hour = int(cfg.get("daily_hour", 3))
        self.run_window_hours = max(1, int(cfg.get("run_window_hours", 3)))
        self.daily_probability = _clamp(cfg.get("daily_probability", 0.4), 0.4)
        self.check_interval_minutes = max(5, int(cfg.get("check_interval_minutes", 60)))
        self.material_window_hours = max(1, int(cfg.get("material_window_hours", 48)))
        self.min_material_count = max(1, int(cfg.get("min_material_count", 5)))
        self.material_limit = max(self.min_material_count, int(cfg.get("material_limit", 5)))
        self.old_echo_enabled = bool(cfg.get("old_echo_enabled", True))
        self.old_echo_min_age_hours = max(1.0, float(cfg.get("old_echo_min_age_hours", 72)))
        self.raw_residue_enabled = bool(cfg.get("raw_residue_enabled", False))
        self.raw_residue_turns = max(0, min(12, int(cfg.get("raw_residue_turns", 4))))
        self.raw_residue_max_chars = max(200, min(4000, int(cfg.get("raw_residue_max_chars", 1500))))
        self.identity_anchor_id = str(cfg.get("identity_anchor_id") or "").strip()
        self.min_surface_age_hours = max(0.0, float(cfg.get("min_surface_age_hours", 3)))
        self.surface_threshold = float(cfg.get("surface_threshold", 0.62))
        self.alpha_subordinate = float(cfg.get("alpha_subordinate", 0.25))
        self.spontaneous_surface_prob = float(cfg.get("spontaneous_surface_prob", 0.02))
        self.claim_ttl_minutes = max(1, int(cfg.get("claim_ttl_minutes", 15)))
        self.dreams_dir = Path(cfg.get("data_dir") or state_dir / "dreams").expanduser().resolve()
        self.logs_dir = self.dreams_dir / "logs"
        self.dreams_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.client = None
        if self.enabled and self.api_key and self.base_url:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=60.0)

    def _now(self, now: datetime | None = None) -> datetime:
        dt = now or datetime.now(self.tz)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.tz)
        return dt.astimezone(self.tz)

    def _event_log(self) -> Path:
        return self.logs_dir / "events.jsonl"

    def _log_event(self, event: str, payload: dict) -> None:
        entry = {"event": event, **payload}
        with self._event_log().open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")

    def _read_events(self) -> list[dict]:
        path = self._event_log()
        if not path.exists():
            return []
        events = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                events.append(item)
        return events

    def _dream_path(self, dream_id: str) -> Path:
        return self.dreams_dir / f"{dream_id}.md"

    def _write_record(self, metadata: dict, body: str) -> DreamRecord:
        dream_id = str(metadata["dream_id"])
        path = self._dream_path(dream_id)
        text = "---\n"
        text += yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True)
        text += "---\n"
        text += body.strip() + "\n"
        path.write_text(text, encoding="utf-8")
        return DreamRecord(metadata=metadata, body=body.strip(), path=path)

    def _read_record(self, path: Path) -> DreamRecord:
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---\n"):
            raise ValueError("dream record missing frontmatter")
        _, rest = text.split("---\n", 1)
        raw_meta, body = rest.split("---\n", 1)
        metadata = yaml.safe_load(raw_meta) or {}
        if not isinstance(metadata, dict):
            raise ValueError("dream metadata must be a mapping")
        return DreamRecord(metadata=metadata, body=body.strip(), path=path)

    def list_records(self) -> list[DreamRecord]:
        records = []
        for path in sorted(self.dreams_dir.glob("dream_*.md")):
            try:
                records.append(self._read_record(path))
            except Exception as exc:
                logger.warning("Failed to read dream record %s: %s", path, exc)
        return records

    def _delete_record(self, record: DreamRecord, reason: str, embedding_engine=None) -> None:
        if record.path.exists():
            record.path.unlink()
        self._log_event(
            "deleted",
            {
                "dream_id": record.dream_id,
                "generated_at": record.metadata.get("generated_at"),
                "deleted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "reason": reason,
            },
        )
        if embedding_engine is not None:
            try:
                embedding_engine.delete_embedding(record.dream_id)
            except Exception as exc:
                logger.warning("Failed to delete dream embedding %s: %s", record.dream_id, exc)

    def _record_for_date(self, date_key: str) -> bool:
        for record in self.list_records():
            if record.metadata.get("local_date") == date_key:
                return True
        for event in self._read_events():
            if event.get("event") == "generated" and event.get("local_date") == date_key:
                return True
        return False

    def _daily_decision_for_date(self, date_key: str) -> str:
        if self._record_for_date(date_key):
            return "generated"
        for event in self._read_events():
            if event.get("event") == "probability_skipped" and event.get("local_date") == date_key:
                return "probability_miss"
        return ""

    def _bucket_created_local(self, bucket: dict) -> datetime | None:
        meta = bucket.get("metadata", {}) or {}
        candidates = []
        for key in ("created", "updated_at"):
            raw = meta.get(key)
            if not raw:
                continue
            try:
                candidates.append(_parse_dt(str(raw)).astimezone(self.tz))
            except Exception:
                continue
        if not candidates:
            return None
        return max(candidates)

    def _is_material_bucket(self, bucket: dict, start: datetime, end: datetime) -> bool:
        meta = bucket.get("metadata", {}) or {}
        created = self._bucket_created_local(bucket)
        if not created or not (start <= created <= end):
            return False
        tags = {str(tag).lower() for tag in meta.get("tags", []) or []}
        bucket_type = meta.get("type")
        if bucket_type == "feel":
            return "whisper" in tags
        if bucket_type in {"permanent", "archived"}:
            return False
        if meta.get("pinned") or meta.get("protected") or meta.get("anchor"):
            return False
        return bool(bucket.get("content"))

    def _material_score(self, bucket: dict, now_local: datetime) -> tuple[float, str]:
        meta = bucket.get("metadata", {}) or {}
        created = self._bucket_created_local(bucket) or now_local
        age_hours = max(0.0, (now_local - created).total_seconds() / 3600)
        recency = math.exp(-age_hours / 24)
        arousal = _clamp(meta.get("arousal", 0.3), 0.3)
        importance = max(1, min(10, int(meta.get("importance", 5)))) / 10
        tags = {str(tag).lower() for tag in meta.get("tags", []) or []}
        whisper_bonus = 0.15 if "whisper" in tags else 0.0
        score = 0.45 * recency + 0.30 * arousal + 0.20 * importance + whisper_bonus
        return (score, created.isoformat())

    def _is_old_echo_bucket(self, bucket: dict, now_local: datetime, exclude_ids: set[str]) -> bool:
        meta = bucket.get("metadata", {}) or {}
        bucket_id = str(bucket.get("id") or meta.get("id") or "")
        if not bucket_id or bucket_id in exclude_ids:
            return False
        created = self._bucket_created_local(bucket)
        if not created:
            return False
        age_hours = (now_local - created).total_seconds() / 3600
        if age_hours < self.old_echo_min_age_hours:
            return False
        bucket_type = meta.get("type")
        if bucket_type in {"feel", "permanent", "archived", "archive"}:
            return False
        if meta.get("pinned") or meta.get("protected") or meta.get("anchor"):
            return False
        return bool(bucket.get("content"))

    def _old_echo_score(self, bucket: dict, materials: list[dict], now_local: datetime) -> tuple[float, str]:
        meta = bucket.get("metadata", {}) or {}
        material_tags = {
            str(tag).lower()
            for item in materials
            for tag in (item.get("metadata", {}) or {}).get("tags", []) or []
        }
        material_domains = {
            str(domain).lower()
            for item in materials
            for domain in (item.get("metadata", {}) or {}).get("domain", []) or []
        }
        tags = {str(tag).lower() for tag in meta.get("tags", []) or []}
        domains = {str(domain).lower() for domain in meta.get("domain", []) or []}
        shared_tags = len(tags & material_tags)
        shared_domains = len(domains & material_domains)
        importance = max(1, min(10, int(meta.get("importance", 5)))) / 10
        arousal = _clamp(meta.get("arousal", 0.3), 0.3)
        created = self._bucket_created_local(bucket) or now_local
        age_days = max(0.0, (now_local - created).total_seconds() / 86400)
        age_curve = 1.0 / (1.0 + abs(age_days - 14.0) / 30.0)
        score = (
            0.30 * min(1.0, shared_tags / 2.0)
            + 0.20 * min(1.0, shared_domains / 2.0)
            + 0.25 * importance
            + 0.15 * arousal
            + 0.10 * age_curve
        )
        return (score, created.isoformat())

    def _select_old_echo(self, all_buckets: list[dict], materials: list[dict], now_local: datetime) -> dict | None:
        if not self.old_echo_enabled:
            return None
        exclude_ids = {
            str(bucket.get("id") or bucket.get("metadata", {}).get("id") or "")
            for bucket in materials
        }
        candidates = [
            bucket
            for bucket in all_buckets
            if self._is_old_echo_bucket(bucket, now_local, exclude_ids)
        ]
        if not candidates:
            return None
        candidates.sort(
            key=lambda bucket: self._old_echo_score(bucket, materials, now_local),
            reverse=True,
        )
        return candidates[0]

    def select_raw_residue_turns(
        self,
        raw_event_store,
        now: datetime | None = None,
    ) -> tuple[list[dict], list[int]]:
        if not self.raw_residue_enabled or self.raw_residue_turns <= 0 or raw_event_store is None:
            return [], []
        now_local = self._now(now)
        target_day = (now_local - timedelta(days=1)).date()
        start = datetime(target_day.year, target_day.month, target_day.day, tzinfo=self.tz)
        end = start + timedelta(days=1)
        try:
            events = raw_event_store.list_events_between(start_at=start, end_at=end, limit=0)
        except Exception as exc:
            logger.warning("Dream raw residue read failed: %s", exc)
            return [], []
        turns = self._raw_residue_turn_payloads(events)
        if not turns:
            return [], []
        sampled = self._sample_raw_residue_turns(turns)
        fitted = self._fit_raw_residue_chars(sampled)
        event_ids = [
            int(event_id)
            for turn in fitted
            for event_id in (turn.get("source_event_ids") or [])
            if event_id is not None and str(event_id).isdigit()
        ]
        public_turns = [
            {
                "created_at": turn.get("created_at", ""),
                "user_text": turn.get("user_text", ""),
                "assistant_text": turn.get("assistant_text", ""),
            }
            for turn in fitted
        ]
        return public_turns, list(dict.fromkeys(event_ids))

    def _raw_residue_turn_payloads(self, events: list[dict] | None) -> list[dict]:
        grouped: dict[tuple[str, str], dict] = {}
        for event in events or []:
            if not isinstance(event, dict):
                continue
            role = str(event.get("role") or "").strip().lower()
            if role not in {"user", "assistant"}:
                continue
            text = strip_raw_client_context(str(event.get("text") or "")).strip()
            if not text or raw_event_text_looks_injected(text, event):
                continue
            metadata = event.get("metadata", {}) if isinstance(event.get("metadata"), dict) else {}
            session_id = str(event.get("session_id") or event.get("conversation_id") or "").strip()
            event_id = int(event.get("id") or 0)
            round_value = metadata.get("round_id")
            round_key = str(round_value).strip() if round_value is not None else f"event:{event_id}"
            key = (session_id, round_key)
            row = grouped.get(key)
            if row is None:
                row = {
                    "created_at": str(event.get("created_at") or ""),
                    "created_sort": self._raw_residue_created_sort(event),
                    "user_text": "",
                    "assistant_text": "",
                    "source_event_ids": [],
                }
                grouped[key] = row
            row["source_event_ids"].append(event_id)
            if role == "user":
                row["user_text"] = self._join_raw_residue_text(row["user_text"], text)
            else:
                row["assistant_text"] = self._join_raw_residue_text(row["assistant_text"], text)

        turns = []
        for row in grouped.values():
            row["source_event_ids"] = list(dict.fromkeys(row["source_event_ids"]))
            if self._raw_residue_turn_allowed(row):
                row["user_text"] = self._clip_raw_residue_text(row["user_text"], 520)
                row["assistant_text"] = self._clip_raw_residue_text(row["assistant_text"], 520)
                turns.append(row)
        turns.sort(key=lambda item: (item.get("created_sort") or "", item.get("source_event_ids") or []))
        return turns

    @staticmethod
    def _join_raw_residue_text(left: str, right: str) -> str:
        return f"{left} / {right}".strip(" /") if left else right

    def _raw_residue_created_sort(self, event: dict) -> str:
        try:
            return _parse_dt(str(event.get("created_at") or "")).astimezone(self.tz).isoformat()
        except Exception:
            return str(event.get("created_at") or "")

    @staticmethod
    def _clip_raw_residue_text(text: str, limit: int) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: max(0, limit - 1)].rstrip() + "…"

    def _raw_residue_turn_allowed(self, turn: dict) -> bool:
        user_text = str(turn.get("user_text") or "").strip()
        assistant_text = str(turn.get("assistant_text") or "").strip()
        if not user_text or not assistant_text:
            return False
        assistant_signal = re.sub(r"\s+", "", assistant_text)
        if len(assistant_signal) < 18:
            return False
        return RAW_RESIDUE_NOISE_RE.search(user_text + "\n" + assistant_text) is None

    def _sample_raw_residue_turns(self, turns: list[dict]) -> list[dict]:
        count = min(self.raw_residue_turns, len(turns))
        if count <= 0:
            return []
        if count >= len(turns):
            return list(turns)

        selected: list[dict] = []
        selected_ids: set[int] = set()
        segment_count = min(3, count, len(turns))
        for index in range(segment_count):
            start = int(index * len(turns) / segment_count)
            end = int((index + 1) * len(turns) / segment_count)
            segment = turns[start:end]
            if not segment:
                continue
            choice = random.choice(segment)
            selected.append(choice)
            selected_ids.add(id(choice))

        remaining = [turn for turn in turns if id(turn) not in selected_ids]
        while len(selected) < count and remaining:
            choice = random.choice(remaining)
            selected.append(choice)
            selected_ids.add(id(choice))
            remaining = [turn for turn in remaining if id(turn) not in selected_ids]
        selected.sort(key=lambda item: (item.get("created_sort") or "", item.get("source_event_ids") or []))
        return selected

    def _fit_raw_residue_chars(self, turns: list[dict]) -> list[dict]:
        fitted: list[dict] = []
        total = 0
        for turn in turns:
            item = dict(turn)
            user_text = str(item.get("user_text") or "")
            assistant_text = str(item.get("assistant_text") or "")
            chars = len(user_text) + len(assistant_text)
            if total + chars > self.raw_residue_max_chars:
                remaining = self.raw_residue_max_chars - total
                if remaining < 80:
                    break
                user_limit = max(30, remaining // 2)
                assistant_limit = max(30, remaining - user_limit)
                item["user_text"] = self._clip_raw_residue_text(user_text, user_limit)
                item["assistant_text"] = self._clip_raw_residue_text(assistant_text, assistant_limit)
                chars = len(item["user_text"]) + len(item["assistant_text"])
            fitted.append(item)
            total += chars
        return fitted

    async def select_materials(self, bucket_mgr, now: datetime | None = None) -> tuple[list[dict], dict | None]:
        now_local = self._now(now)
        start = now_local - timedelta(hours=self.material_window_hours)
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
        except Exception:
            return [], None
        materials = [
            bucket
            for bucket in all_buckets
            if self._is_material_bucket(bucket, start, now_local)
        ]
        if len(materials) < self.min_material_count:
            return [], await bucket_mgr.get(self.identity_anchor_id) if self.identity_anchor_id else None
        materials.sort(key=lambda b: self._material_score(b, now_local), reverse=True)
        identity_anchor = await bucket_mgr.get(self.identity_anchor_id) if self.identity_anchor_id else None
        return materials[: self.material_limit], identity_anchor

    def _payload_for(
        self,
        materials: list[dict],
        identity_anchor: dict | None,
        old_echo: dict | None = None,
        raw_residue_turns: list[dict] | None = None,
    ) -> dict:
        def comment_payloads(bucket: dict) -> list[dict]:
            meta = bucket.get("metadata", {}) or {}
            comments = meta.get("comments", [])
            if not isinstance(comments, list):
                return []
            payloads = []
            for comment in comments[-4:]:
                if not isinstance(comment, dict):
                    continue
                text = strip_wikilinks(str(comment.get("content") or "")).strip()
                if not text:
                    continue
                item = {
                    "id": str(comment.get("id") or ""),
                    "created": str(comment.get("created") or ""),
                    "author": str(comment.get("author") or ""),
                    "kind": str(comment.get("kind") or "comment"),
                    "text": text[:320],
                }
                payloads.append(item)
            return payloads

        def material_payload(bucket: dict) -> dict:
            meta = bucket.get("metadata", {}) or {}
            tags = {str(tag).lower() for tag in meta.get("tags", []) or []}
            kind = "whisper" if meta.get("type") == "feel" and "whisper" in tags else "memory"
            residue_time = self._bucket_created_local(bucket)
            return {
                "source_id": bucket.get("id") or meta.get("id"),
                "kind": kind,
                "residue_time": residue_time.isoformat() if residue_time else meta.get("created", ""),
                "created": meta.get("created", ""),
                "updated_at": meta.get("updated_at", ""),
                "valence": meta.get("valence", 0.5),
                "arousal": meta.get("arousal", 0.3),
                "text": strip_wikilinks(bucket_text_for_embedding(bucket))[:700],
                "comments": comment_payloads(bucket),
            }

        anchor_payload = None
        if identity_anchor:
            anchor_meta = identity_anchor.get("metadata", {}) or {}
            anchor_payload = {
                "source_id": identity_anchor.get("id") or anchor_meta.get("id"),
                "name": anchor_meta.get("name", ""),
                "text": strip_wikilinks(bucket_text_for_embedding(identity_anchor))[:500],
            }

        return {
            "dreamer": self.identity.get("ai_name") or "AI",
            "user_display_name": self.identity.get("user_display_name") or "用户",
            "identity_anchor": anchor_payload,
            "daytime_residue": [material_payload(bucket) for bucket in materials],
            "old_echo": material_payload(old_echo) if old_echo else None,
            "recent_original_dialogue": raw_residue_turns or [],
        }

    async def _call_dream_model(self, payload: dict) -> str:
        if not self.client:
            raise RuntimeError("dream model api key is not configured")
        options = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": DREAM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.thinking_mode:
            options["extra_body"] = {"thinking": {"type": self.thinking_mode}}
        response = await self.client.chat.completions.create(**options)
        raw = response.choices[0].message.content if response.choices else ""
        cleaned = (raw or "").strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        cleaned = re.sub(r"^\s*(梦境|正文)[:：]\s*", "", cleaned).strip()
        if not cleaned:
            raise ValueError("dream model returned empty text")
        return cleaned

    @staticmethod
    def _normalize_thinking_mode(value: Any) -> str:
        normalized = str(value or "").strip().lower().replace("_", "-")
        if normalized in {"disabled", "disable", "off", "false", "non-thinking"}:
            return "disabled"
        if normalized in {"enabled", "enable", "on", "true", "thinking"}:
            return "enabled"
        return ""

    def _core_affect_from_materials(self, materials: list[dict]) -> dict:
        if not materials:
            return {"valence": 0.5, "arousal": 0.3}
        valence = 0.0
        arousal = 0.0
        for bucket in materials:
            meta = bucket.get("metadata", {}) or {}
            valence += _clamp(meta.get("valence", 0.5))
            arousal += _clamp(meta.get("arousal", 0.3), 0.3)
        return {
            "valence": round(valence / len(materials), 2),
            "arousal": round(arousal / len(materials), 2),
        }

    def _recall_cues_from_dream_text(self, dream_text: str) -> list[str]:
        text = strip_wikilinks(str(dream_text or "")).strip()
        if not text:
            return ["梦里突然断掉的画面", "醒来前留下的细节"]

        candidates: list[tuple[int, int, str]] = []
        order = 0
        for sentence in re.split(r"[。！？；;.!?\n]+", text):
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_cue = self._clean_dream_cue_clause(sentence)
            if sentence_cue:
                candidates.append((self._dream_cue_score(sentence_cue), -order, sentence_cue))
                order += 1
            for clause in re.split(r"[，,、]+", sentence):
                cue = self._clean_dream_cue_clause(clause)
                if not cue:
                    continue
                score = self._dream_cue_score(cue)
                candidates.append((score, -order, cue))
                order += 1

        cues: list[str] = []
        for _score, _order, cue in sorted(candidates, reverse=True):
            if cue in cues:
                continue
            cues.append(cue)
            if len(cues) >= 5:
                break

        if len(cues) < 2:
            for sentence in re.split(r"[。！？；;.!?\n]+", text):
                cue = self._clean_dream_cue_clause(sentence)
                if cue and cue not in cues:
                    cues.append(cue)
                if len(cues) >= 2:
                    break

        for fallback in ("梦里突然断掉的画面", "醒来前留下的细节"):
            if len(cues) >= 2:
                break
            cues.append(fallback)
        return cues[:5]

    @staticmethod
    def _clean_dream_cue_clause(text: str) -> str:
        cue = re.sub(r"\s+", "", str(text or "")).strip("“”\"'`：:，,。！？；;、-— ")
        cue = re.sub(r"^(我看见|我听见|我低头|我抬头|我伸手|我想问|我想|我觉得)", "", cue)
        if len(cue) < 3:
            return ""
        if len(cue) > 36:
            cue = cue[:36].rstrip("，,。！？；;、-— ") + "..."
        return cue

    @staticmethod
    def _dream_cue_score(cue: str) -> int:
        score = 1
        if re.search(r"\d|[一二三四五六七八九十]|99", cue):
            score += 1
        if 6 <= len(cue) <= 28:
            score += 2
        elif 3 <= len(cue) <= 36:
            score += 1
        if cue.startswith(("我", "不是", "是", "像")):
            score -= 1
        return score

    def _normalize_model_result(self, raw: dict) -> tuple[str, dict, list[str]]:
        dream_text = re.sub(r"\s+\n", "\n", str(raw.get("dream_text") or "")).strip()
        if not dream_text:
            raise ValueError("dream_text is empty")
        affect = raw.get("core_affect") if isinstance(raw.get("core_affect"), dict) else {}
        core_affect = {
            "valence": round(_clamp(affect.get("valence", 0.5)), 2),
            "arousal": round(_clamp(affect.get("arousal", 0.3), 0.3), 2),
        }
        cues = []
        raw_cues = raw.get("recall_cues")
        if isinstance(raw_cues, list):
            for cue in raw_cues:
                text = str(cue).strip()
                if text and text not in cues:
                    cues.append(text[:40])
                if len(cues) >= 5:
                    break
        if len(cues) < 2:
            cues = ["熟悉的话突然陌生", "夜里想起未说完的话"]
        return dream_text, core_affect, cues

    async def generate(
        self,
        bucket_mgr,
        embedding_engine=None,
        now: datetime | None = None,
        force: bool = False,
        raw_event_store=None,
    ) -> dict:
        if not self.enabled:
            return {"status": "disabled"}
        if not self.client:
            return {"status": "skipped", "reason": "missing_api_key"}
        now_local = self._now(now)
        date_key = now_local.date().isoformat()
        if not force:
            decision = self._daily_decision_for_date(date_key)
            if decision == "generated":
                return {"status": "exists", "date": date_key}
            if decision == "probability_miss":
                return {
                    "status": "skipped",
                    "reason": "daily_probability_already_missed",
                    "date": date_key,
                }
        materials, identity_anchor = await self.select_materials(bucket_mgr, now_local)
        if len(materials) < self.min_material_count:
            return {
                "status": "skipped",
                "reason": "not_enough_materials",
                "materials": len(materials),
            }
        if not force:
            roll = random.random()
            if roll >= self.daily_probability:
                decided_at = now_local.astimezone(timezone.utc).isoformat(timespec="seconds")
                self._log_event(
                    "probability_skipped",
                    {
                        "local_date": date_key,
                        "decided_at": decided_at,
                        "probability": self.daily_probability,
                        "roll": round(roll, 4),
                        "material_count": len(materials),
                    },
                )
                return {
                    "status": "skipped",
                    "reason": "daily_probability_miss",
                    "date": date_key,
                    "probability": self.daily_probability,
                }

        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
        except Exception:
            all_buckets = []
        old_echo = self._select_old_echo(all_buckets, materials, now_local)
        raw_residue_turns, raw_residue_event_ids = self.select_raw_residue_turns(raw_event_store, now_local)
        payload = self._payload_for(materials, identity_anchor, old_echo, raw_residue_turns)
        dream_text = await self._call_dream_model(payload)
        core_affect = self._core_affect_from_materials(materials)
        recall_cues = self._recall_cues_from_dream_text(dream_text)
        dream_id = f"dream_{now_local.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        source_ids = [
            str(bucket.get("id") or bucket.get("metadata", {}).get("id") or "")
            for bucket in materials
        ]
        old_echo_id = (
            str(old_echo.get("id") or old_echo.get("metadata", {}).get("id") or "")
            if old_echo
            else ""
        )
        generated_at = now_local.astimezone(timezone.utc).isoformat(timespec="seconds")
        metadata = {
            "dream_id": dream_id,
            "generated_at": generated_at,
            "local_date": date_key,
            "ai_name": self.identity.get("ai_name") or "AI",
            "dream_model": self.model,
            "core_affect": core_affect,
            "recall_cues": recall_cues,
            "source_bucket_ids": source_ids,
            "source_raw_event_ids": raw_residue_event_ids,
            "old_echo_id": old_echo_id or None,
            "identity_anchor_id": self.identity_anchor_id,
            "material_count": len(materials),
            "surfaced": False,
            "surfaced_at": None,
        }
        record = self._write_record(metadata, dream_text)
        self._log_event(
            "generated",
            {
                "dream_id": dream_id,
                "generated_at": generated_at,
                "local_date": date_key,
                "ai_name": self.identity.get("ai_name") or "AI",
            },
        )
        if embedding_engine is not None and getattr(embedding_engine, "enabled", False):
            try:
                await embedding_engine.generate_and_store(dream_id, "；".join(recall_cues))
            except Exception as exc:
                logger.warning("Dream cue embedding failed for %s: %s", dream_id, exc)
        return {"status": "created", "id": record.dream_id, "date": date_key}

    async def run_due(
        self,
        bucket_mgr,
        embedding_engine=None,
        now: datetime | None = None,
        raw_event_store=None,
    ) -> dict:
        if not self.enabled or not self.auto_enabled:
            return {"status": "disabled"}
        now_local = self._now(now)
        if now_local.hour < self.daily_hour:
            return {"status": "skipped", "reason": "too_early"}
        if now_local.hour >= min(24, self.daily_hour + self.run_window_hours):
            return {"status": "skipped", "reason": "outside_dream_window"}
        return await self.generate(
            bucket_mgr,
            embedding_engine,
            now_local,
            force=False,
            raw_event_store=raw_event_store,
        )

    def _eligible_context(self, query: str, valence: float, arousal: float, is_session_start: bool) -> bool:
        has_query = bool(query and query.strip())
        has_affect = 0 <= valence <= 1 and 0 <= arousal <= 1
        return bool(is_session_start or has_query or has_affect)

    async def _query_embedding(self, query: str, embedding_engine) -> list[float] | None:
        if not query or not query.strip():
            return None
        if embedding_engine is None or not getattr(embedding_engine, "enabled", False):
            return None
        try:
            embedding = await embedding_engine._generate_embedding(query)
            return embedding or None
        except Exception as exc:
            logger.warning("Dream query embedding failed: %s", exc)
            return None

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return max(0.0, dot / (norm_a * norm_b))

    def _affect_score(self, record: DreamRecord, valence: float, arousal: float) -> float:
        if not (0 <= valence <= 1 and 0 <= arousal <= 1):
            return 0.0
        affect = record.metadata.get("core_affect", {}) or {}
        dv = _clamp(affect.get("valence", 0.5)) - valence
        da = _clamp(affect.get("arousal", 0.3), 0.3) - arousal
        return max(0.0, 1.0 - math.sqrt((dv * dv + da * da) / 2))

    @staticmethod
    def _compact_recall_text(text: str) -> str:
        return re.sub(r"\W+", "", str(text or ""), flags=re.UNICODE).replace("_", "").lower()

    @classmethod
    def _cue_text_score(cls, record: DreamRecord, query: str) -> float:
        query_text = cls._compact_recall_text(query)
        if len(query_text) < 5:
            return 0.0
        raw_cues = record.metadata.get("recall_cues")
        cues = raw_cues if isinstance(raw_cues, list) else []
        for cue in cues:
            cue_text = cls._compact_recall_text(str(cue or ""))
            if cue_text in {"梦里突然断掉的画面", "醒来前留下的细节"}:
                continue
            if len(cue_text) < 5:
                continue
            if cue_text in query_text or query_text in cue_text:
                return 0.96
        return 0.0

    async def _cue_score(
        self,
        record: DreamRecord,
        query: str,
        query_embedding: list[float] | None,
        embedding_engine,
    ) -> float:
        text_score = self._cue_text_score(record, query)
        if not query_embedding or embedding_engine is None or not getattr(embedding_engine, "enabled", False):
            return text_score
        stored = await embedding_engine.get_embedding(record.dream_id)
        if not stored:
            return text_score
        return max(text_score, self._cosine(query_embedding, stored))

    def _format_surface(self, record: DreamRecord) -> str:
        generated = record.generated_at.astimezone(self.tz)
        ai_name = str(record.metadata.get("ai_name") or self.identity.get("ai_name") or "AI")
        title = f"{generated.year}年{generated.month:02d}月{generated.day:02d}日 {ai_name}的梦"
        return "===== 梦境 =====\n" + title + "\n" + record.body.strip()

    async def surface_for_breath(
        self,
        query: str = "",
        valence: float = -1,
        arousal: float = -1,
        is_session_start: bool = False,
        embedding_engine=None,
        now: datetime | None = None,
        retain_after_surface: bool | None = None,
    ) -> str | None:
        result = await self.surface_with_status(
            query=query,
            valence=valence,
            arousal=arousal,
            is_session_start=is_session_start,
            embedding_engine=embedding_engine,
            now=now,
            retain_after_surface=retain_after_surface,
        )
        return str(result.get("text") or "") or None

    async def surface_with_status(
        self,
        query: str = "",
        valence: float = -1,
        arousal: float = -1,
        is_session_start: bool = False,
        embedding_engine=None,
        now: datetime | None = None,
        retain_after_surface: bool | None = None,
    ) -> dict:
        if not self.enabled or not self.surface_enabled:
            return {"status": "skipped", "reason": "disabled"}
        if not self._eligible_context(query, valence, arousal, is_session_start):
            return {"status": "skipped", "reason": "ineligible_context"}
        if retain_after_surface is None:
            retain_after_surface = self.retain_after_surface
        now_local = self._now(now)
        pending = [
            record
            for record in self.list_records()
            if not record.surfaced
            and (now_local - record.generated_at.astimezone(self.tz)).total_seconds() / 3600 >= self.min_surface_age_hours
        ]
        if not pending:
            return {"status": "skipped", "reason": "no_pending_dream"}
        query_embedding = await self._query_embedding(query, embedding_engine)
        evaluated = []
        for record in pending:
            affect = self._affect_score(record, valence, arousal)
            cue = await self._cue_score(record, query, query_embedding, embedding_engine)
            score = max(affect, cue) + self.alpha_subordinate * min(affect, cue)
            evaluated.append({"record": record, "affect": affect, "cue": cue, "score": score, "top": max(affect, cue)})

        best = None
        for item in evaluated:
            if item["score"] >= self.surface_threshold and (best is None or item["score"] > best["score"]):
                best = item
        if best is None:
            for item in evaluated:
                if random.random() < self.spontaneous_surface_prob:
                    best = item
                    break
        if best is not None:
            record = best["record"]
            claim_path = record.path.with_suffix(record.path.suffix + ".claim")
            if claim_path.exists():
                try:
                    modified = datetime.fromtimestamp(claim_path.stat().st_mtime, timezone.utc)
                    if (datetime.now(timezone.utc) - modified).total_seconds() > self.claim_ttl_minutes * 60:
                        claim_path.unlink(missing_ok=True)
                except Exception:
                    pass
            try:
                fd = os.open(str(claim_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
            except FileExistsError:
                return {"status": "skipped", "reason": "already_claimed"}
            surfaced_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            try:
                if not record.path.exists():
                    return {"status": "skipped", "reason": "record_missing"}
                surfaced_record = self._write_record(
                    {**record.metadata, "surfaced": True, "surfaced_at": surfaced_at},
                    record.body,
                )
                self._log_event(
                    "surfaced",
                    {
                        "dream_id": surfaced_record.dream_id,
                        "generated_at": surfaced_record.metadata.get("generated_at"),
                        "surfaced_at": surfaced_at,
                    },
                )
                text = self._format_surface(surfaced_record)
                if not retain_after_surface:
                    self._delete_record(surfaced_record, "surfaced_one_shot", embedding_engine)
                raw_source_bucket_ids = surfaced_record.metadata.get("source_bucket_ids")
                source_bucket_ids = raw_source_bucket_ids if isinstance(raw_source_bucket_ids, list) else []
                return {
                    "status": "injected",
                    "reason": "resonant",
                    "retained": bool(retain_after_surface),
                    "text": text,
                    "dream_id": surfaced_record.dream_id,
                    "generated_at": surfaced_record.metadata.get("generated_at"),
                    "surfaced_at": surfaced_at,
                    "source_bucket_ids": [
                        str(bucket_id)
                        for bucket_id in source_bucket_ids
                        if str(bucket_id or "").strip()
                    ][:5],
                }
            finally:
                try:
                    claim_path.unlink(missing_ok=True)
                except Exception:
                    pass

        return {"status": "skipped", "reason": "no_resonance"}

    def dashboard_records(self, limit: int = 30) -> list[dict]:
        entries: dict[str, dict] = {}
        for event in self._read_events():
            dream_id = str(event.get("dream_id") or "")
            if not dream_id:
                continue
            entry = entries.setdefault(
                dream_id,
                {
                    "dream_id": dream_id,
                    "generated_at": event.get("generated_at", ""),
                    "local_date": event.get("local_date", ""),
                    "ai_name": event.get("ai_name") or self.identity.get("ai_name") or "AI",
                    "status": "latent",
                    "has_body": False,
                },
            )
            if event.get("event") == "surfaced":
                entry["status"] = "surfaced"
            elif event.get("event") == "deleted" and entry.get("status") != "surfaced":
                entry["status"] = "forgotten"
        for record in self.list_records():
            meta = record.metadata
            entries[record.dream_id] = {
                "dream_id": record.dream_id,
                "generated_at": meta.get("generated_at", ""),
                "local_date": meta.get("local_date", ""),
                "ai_name": meta.get("ai_name") or self.identity.get("ai_name") or "AI",
                "status": "surfaced" if record.surfaced else "latent",
                "has_body": True,
            }
        result = list(entries.values())
        result.sort(key=lambda item: item.get("generated_at") or "", reverse=True)
        return result[:limit]

    def dashboard_record(self, dream_id: str) -> dict | None:
        target_id = str(dream_id or "").strip()
        if not target_id:
            return None
        for record in self.list_records():
            if record.dream_id != target_id:
                continue
            meta = record.metadata
            return {
                "dream_id": record.dream_id,
                "generated_at": meta.get("generated_at", ""),
                "local_date": meta.get("local_date", ""),
                "ai_name": meta.get("ai_name") or self.identity.get("ai_name") or "AI",
                "status": "surfaced" if record.surfaced else "latent",
                "body": record.body,
            }
        return None

    def dashboard_payload(self, limit: int = 30) -> dict:
        return {
            "enabled": self.enabled,
            "auto_enabled": self.auto_enabled,
            "surface_enabled": self.surface_enabled,
            "model": self.model,
            "api_ready": bool(self.api_key),
            "records": self.dashboard_records(limit),
        }
