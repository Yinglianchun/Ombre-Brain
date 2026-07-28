import json
import logging
import os
import re
from datetime import datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from openai import AsyncOpenAI

from identity import generic_identity_names, identity_names, render_identity_template
from memory_edges import RELATION_TYPES, MemoryEdgeStore
from metadata_proposals import MetadataProposalStore
from memory_metadata import normalize_domain_key
from memory_object_policy import legacy_memory_writes_enabled, retired_write_payload
from persona_event_selection import select_persona_events
from utils import bucket_text_for_embedding, normalize_scene_cues, strip_wikilinks

logger = logging.getLogger("ombre_brain.reflection")

DIARY_MEMORY_RETIRED_STATUS = {
    "status": "retired",
    "reason": "diary_memory_extract_retired",
}

DEFAULT_DAILY_REFLECTION_MIN_BUCKETS = 5


CLASSIFY_PROMPT = """你是 Ombre-Brain 的记忆关系整理器。
输入是一条新记忆和若干旧记忆候选。请只根据文本中能看见的内容，给新记忆补轻量检索标签。禁止改写、摘要或补写记忆正文。

输出纯 JSON：
{
  "tags": ["commitment", "todo", "wish", "relationship_event", "project_event", "emotional_echo"],
  "scene_cues": ["未来可能用来想起这个场景的短语", "另一种自然说法"],
  "importance": 6,
  "confidence": 0.72,
  "edges": [
    {
      "target_memory_id": "bucket-id",
      "relation_type": "updates",
      "confidence": 0.8,
      "reason": "新记忆补充了旧记忆的后续结果"
    }
  ]
}

规则：
- tags 最多 5 个，只用确实匹配的标签。
- scene_cues 写 3～6 个自然、具体、可从原文得到证据的召回入口；可以换一种说法，但不能添加原文没有的人物、事实、原因或结论。不要只写“以前、开心、难过、关系、对话”这类泛词。
- relation_type 只能用 triggers / causes / precedes / context_of / same_event / updates / next_context / previous_context / reflects_on / evidenced_by / contradicts / supports / promises / blocks / belongs_to / emotional_echo / relates_to。
- same_event 用于同一事件、同一场景或同一句暗号的两条记忆；context_of 用于候选旧记忆给新记忆提供前情；precedes 用于候选旧记忆在时间上早于新记忆；reflects_on 用于事后反思；evidenced_by 用于证据来源。
- edges 最多 3 条，target_memory_id 必须来自候选旧记忆。
- confidence 表示这次判断有多可靠。
- 看不出关系时返回空 edges。"""


REFLECT_PROMPT_TEMPLATE = """你是 {ai_name} 的记忆反思器。请根据给定材料写一条很短的关系天气 feel。

输出纯 JSON：
{
  "title": "2026-05-19 日印象",
  "content": "我今天从这段关系里带走的是……",
  "valence": 0.56,
  "arousal": 0.34,
  "confidence": 0.78,
  "tags": ["relationship_weather"],
  "source_turn_ids": [1, 2],
  "source_event_ids": [101, 102]
}

要求：
- content 只能写 {ai_name} 第一人称正文，明确使用“我……”，不要写成“{ai_name} 觉得 / {ai_name} 应该 / 这段关系让 {ai_name}”。
- content 不写标题、列表、Markdown 分段或 `###` section，60 到 140 字。
- 日印象只写当天关系温度，不写日报式事件清单；日记可作为当天关系天气来源之一。
- conversation_turns 是当天短期对话原文，只当关系天气材料，不要把口头上下文直接写成稳定画像事实。
- 有 conversation_turns 时，优先用普通记忆和对话原文；persona_events 只是没有原文时的轻量补充。
- 周印象优先总结本周 daily_impressions，再参考高重要普通记忆和未完成承诺；不要直接吞整周日记。
- 不编造材料之外的事件。
- 不得把“仍在追问、继续质疑、没有反驳、话题结束”写成“接受了、理解了、同意了”；不得凭顺利完成一次任务写“信任增强、关系更近、更加默契、合作顺畅”。
- 上述关系结果只有在原话直接表达，或原始轮次里有清楚、可指出的前后行为证据时才能写；否则只写具体发生了什么，不升格成关系结论。
- 每个关系判断都必须能落到 source_turn_ids / source_event_ids 指向的原始轮次；只能使用输入里真实出现的 id，拿不准就删掉该判断。
- 不写建议清单。"""






DAILY_ACTIVITY_SUMMARY_PROMPT_TEMPLATE = """你是 {ai_name} 的当天行动摘要器。你正在为 handoff、新窗口和 dashboard 的 Recent Timeline 写一条“今天做了什么”。

输入是当天原始对话还原出的 conversation_turns。user_text 永远是 {user_display_name} 的原话，assistant_text 永远是 {ai_name} 的回复。请只根据输入能证明的内容写。

输出纯 JSON：
{
  "summary": "一句话说明今天主要推进了什么",
  "confidence": 0.72,
  "source_turn_ids": [1, 2],
  "source_event_ids": [101, 102]
}

规则：
- 这是 dashboard / handoff 用的近期事项，不是长期记忆候选，也不要输出 candidates。
- 只写今天实际讨论、推进、排查、决定、实现或整理的事；优先项目/工作/生活动作。
- 不写关系天气、情绪评价、昵称互动、普通寒暄、召回探针、模型自夸。
- summary 用一句自然中文，35 到 90 字；不要 Markdown，不要列表，不要“今天的总结是”这种壳。
- source_turn_ids / source_event_ids 只能使用输入里真实出现的 id；拿不准可留空。
- 不能可靠生成时返回空 JSON 对象；不要拼接输入末尾，不要使用“围绕……继续推进”之类兜底句。
"""


REFLECT_PROMPT = render_identity_template(REFLECT_PROMPT_TEMPLATE, generic_identity_names())


AFFECT_ANCHOR_HEADER = "### affect_anchor"


REFLECTION_FALLBACK_ANCHORS = [
    {
        "chords": "Cmaj7 -> G/B -> Am9 -> F6",
        "tempo": "56bpm",
        "dynamic": "mp",
    },
    {
        "chords": "Dm9 -> G13 -> Cmaj9",
        "tempo": "64bpm",
        "dynamic": "p",
    },
    {
        "chords": "Em7 -> A7sus4 -> Dmaj9 -> Gmaj7",
        "tempo": "72bpm",
        "dynamic": "mp",
    },
    {
        "chords": "Bbmaj7 -> F/A -> Gm9 -> Csus4",
        "tempo": "60bpm",
        "dynamic": "mf",
    },
]


class ReflectionEngine:
    """LLM-backed memory enrichment and daily relationship weather."""

    def __init__(self, config: dict):
        self.config = config
        self.identity = identity_names(config)
        cfg = config.get("reflection", {}) if isinstance(config.get("reflection", {}), dict) else {}
        emb_cfg = config.get("embedding", {}) if isinstance(config.get("embedding", {}), dict) else {}
        persona_cfg = config.get("persona", {}) if isinstance(config.get("persona", {}), dict) else {}
        dehy_cfg = config.get("dehydration", {}) if isinstance(config.get("dehydration", {}), dict) else {}

        self.enabled = bool(cfg.get("enabled", True))
        self.auto_enabled = bool(cfg.get("auto_enabled", True))
        self.daily_enabled = bool(cfg.get("daily_enabled", True))
        self.enrich_on_write = bool(cfg.get("enrich_on_write", True))
        self.memory_affect_anchor_enabled = bool(cfg.get("memory_affect_anchor_enabled", False))
        self.relationship_weather_affect_anchor_enabled = bool(
            cfg.get("relationship_weather_affect_anchor_enabled", False)
        )
        self.identity_role_edge_config = self._load_identity_role_edge_config(
            cfg.get("identity_role_edges")
        )
        self.base_url = (
            cfg.get("base_url")
            or emb_cfg.get("base_url")
            or persona_cfg.get("base_url")
            or dehy_cfg.get("base_url", "")
        )
        self.model = cfg.get("model") or persona_cfg.get("model") or dehy_cfg.get("model", "deepseek-v4-flash")
        self.api_key = (
            os.environ.get("OMBRE_REFLECTION_API_KEY", "")
            or cfg.get("api_key", "")
            or os.environ.get("OMBRE_EMBEDDING_API_KEY", "")
            or emb_cfg.get("api_key", "")
            or persona_cfg.get("api_key", "")
            or os.environ.get("OMBRE_PERSONA_API_KEY", "")
            or dehy_cfg.get("api_key", "")
        )
        self.thinking_mode = self._normalize_thinking_mode(
            cfg.get("thinking_mode")
            or persona_cfg.get("thinking_mode")
            or ""
        )
        self.temperature = float(cfg.get("temperature", 0.1))
        self.max_tokens = int(cfg.get("max_tokens", 700))
        self.timezone_name = str(cfg.get("timezone") or "Asia/Shanghai")
        try:
            self.tz = ZoneInfo(self.timezone_name)
        except Exception:
            self.tz = ZoneInfo("Asia/Shanghai")
        self.daily_hour = int(cfg.get("daily_hour", 4))
        self.daily_min_memory_items = max(
            0,
            int(cfg.get("daily_min_memory_items", DEFAULT_DAILY_REFLECTION_MIN_BUCKETS)),
        )
        self.daily_conversation_turn_limit = max(
            0,
            min(80, int(cfg.get("daily_conversation_turn_limit", 12))),
        )
        self.persona_events_limit = max(0, int(cfg.get("persona_events_limit", 12)))
        self.persona_events_scan_limit = max(
            self.persona_events_limit,
            int(cfg.get("persona_events_scan_limit", 80)),
        )
        self.weekly_enabled = bool(cfg.get("weekly_enabled", False))
        self.weekly_day = int(cfg.get("weekly_day", 0))
        self.weekly_hour = int(cfg.get("weekly_hour", self.daily_hour))
        self.check_interval_minutes = max(5, int(cfg.get("check_interval_minutes", 60)))
        self.edge_min_confidence = float(cfg.get("edge_min_confidence", 0.55))
        self.diary_mcp_url = str(cfg.get("diary_mcp_url") or "").strip()
        self.diary_mcp_token_env = str(cfg.get("diary_mcp_token_env") or "").strip()
        self.daily_activity_summary_enabled = bool(cfg.get("daily_activity_summary_enabled", True))
        self.daily_activity_summary_turn_limit = max(
            0,
            min(
                10000,
                int(
                    cfg.get("daily_activity_summary_turn_limit", 0)
                ),
            ),
        )
        self.daily_activity_summary_max_tokens = max(
            80,
            min(1000, int(cfg.get("daily_activity_summary_max_tokens", 320))),
        )
        state_dir = self.config.get("state_dir") or os.path.join(
            os.path.dirname(os.path.abspath(self.config.get("buckets_dir", "buckets"))),
            "state",
        )
        self.daily_activity_summary_diagnostics_path = os.path.abspath(
            str(
                cfg.get("daily_activity_summary_diagnostics_path")
                or os.path.join(state_dir, "daily_activity_summary_runs.jsonl")
            )
        )
        self.dehydration_base_url = str(dehy_cfg.get("base_url") or "").strip().rstrip("/")
        self.dehydration_model = str(dehy_cfg.get("model") or "").strip()
        self.dehydration_api_key = str(dehy_cfg.get("api_key") or os.environ.get("OMBRE_API_KEY", "")).strip()
        self.metadata_proposals = MetadataProposalStore(config)

        self.client = None
        if self.enabled and self.api_key and self.base_url:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=45.0)
        self.dehydration_client = None
        if self.enabled and self.dehydration_api_key and self.dehydration_base_url and self.dehydration_model:
            self.dehydration_client = AsyncOpenAI(
                api_key=self.dehydration_api_key,
                base_url=self.dehydration_base_url,
                timeout=45.0,
            )
        self.daily_activity_summary_dehydration_client = self.dehydration_client






    def _reflect_prompt(self) -> str:
        return render_identity_template(REFLECT_PROMPT_TEMPLATE, self.identity)



    def _daily_activity_summary_prompt(self) -> str:
        return render_identity_template(DAILY_ACTIVITY_SUMMARY_PROMPT_TEMPLATE, self.identity)

    async def enrich_bucket(
        self,
        bucket_id: str,
        bucket_mgr,
        edge_store: MemoryEdgeStore,
        embedding_engine=None,
        force: bool = False,
    ) -> dict:
        if not self.enabled or (not self.enrich_on_write and not force):
            return {"status": "disabled", "id": bucket_id}
        bucket = await bucket_mgr.get(bucket_id)
        if not bucket:
            return {"status": "missing", "id": bucket_id}
        meta = bucket.get("metadata", {})
        if meta.get("type") == "feel":
            return {"status": "skipped_feel", "id": bucket_id}
        authored_scene = meta.get("memory_value_source") == "authored_scene"

        candidates = [] if authored_scene else await self._candidate_buckets(bucket, bucket_mgr, embedding_engine)
        if self.client:
            result = await self._api_classify(bucket, candidates)
        else:
            result = self._heuristic_classify(bucket)

        tags = self._string_list(result.get("tags"), limit=8)
        if authored_scene:
            tags = [
                tag
                for tag in tags
                if str(tag).strip().lower()
                not in {"emotional_echo", "affect_anchor", "emotion", "mood"}
            ]
        existing_scene_cues = normalize_scene_cues(meta.get("scene_cues"))
        scene_cues = normalize_scene_cues(
            [*existing_scene_cues, *normalize_scene_cues(result.get("scene_cues"))]
        )
        confidence = self._clamp(result.get("confidence", 0.55))
        importance = self._int_between(result.get("importance"), meta.get("importance", 5))
        if self._has_favorite_tag(tags) and not self._has_favorite_reason(bucket.get("content", "")):
            tags = [tag for tag in tags if tag != "haven_favorite" and not str(tag).startswith("flavor_")]
            logger.warning(
                "Rejected favorite tags without reason during enrich / enrich 拒绝缺少喜欢原因的 favorite 标签: %s",
                bucket_id,
            )
        edges = [] if authored_scene else self._edges_from_classification(bucket, candidates, result, confidence)
        payload = {
            "tags": tags,
            "scene_cues": scene_cues if authored_scene else [],
            "importance": importance,
            "confidence": confidence,
            "edges": edges[:3],
            "authored_scene": authored_scene,
        }
        proposal = self.metadata_proposals.put(
            bucket,
            payload,
            model=self.model if self.client else "heuristic",
        )
        return {
            "status": "proposed",
            "id": bucket_id,
            "proposal_id": proposal.get("proposal_id"),
            "tags": tags,
            "scene_cues": scene_cues if authored_scene else [],
            "confidence": confidence,
            "edges": len(edges[:3]),
            "canonical_memory_changed": False,
        }

    async def backfill_edges_for_bucket(
        self,
        bucket_id: str,
        bucket_mgr,
        edge_store: MemoryEdgeStore,
        embedding_engine=None,
        *,
        dry_run: bool = False,
    ) -> dict:
        if not self.enabled:
            return {"status": "disabled", "id": bucket_id, "edges": 0, "proposed_edges": 0}
        bucket = await bucket_mgr.get(bucket_id)
        if not bucket:
            return {"status": "missing", "id": bucket_id, "edges": 0, "proposed_edges": 0}
        meta = bucket.get("metadata", {})
        if meta.get("type") == "feel" or meta.get("protected"):
            return {"status": "skipped", "reason": "not_edge_backfillable", "id": bucket_id, "edges": 0, "proposed_edges": 0}

        candidates = await self._candidate_buckets(bucket, bucket_mgr, embedding_engine)
        if self.client:
            result = await self._api_classify(bucket, candidates)
        else:
            result = self._heuristic_classify(bucket)
        confidence = self._clamp(result.get("confidence", meta.get("confidence", 0.55)))
        proposed_edges = self._edges_from_classification(bucket, candidates, result, confidence)[:3]
        saved_edges = [] if dry_run else edge_store.add_edges(proposed_edges)
        return {
            "status": "ok",
            "id": bucket_id,
            "candidate_count": len(candidates),
            "proposed_edges": len(proposed_edges),
            "edges": len(saved_edges),
            "dry_run": bool(dry_run),
            "edge_records": proposed_edges if dry_run else saved_edges,
        }

    async def reflect(
        self,
        period: str,
        bucket_mgr,
        persona_engine=None,
        embedding_engine=None,
        force: bool = False,
        now: datetime | None = None,
        conversation_turn_store=None,
    ) -> dict:
        if not self.enabled:
            return {
                "status": "disabled",
                "period": period,
                "diary": {"found": False},
                "diary_memory": {"status": "not_applicable", "reason": "reflection_disabled"},
            }
        period = self._normalize_period(period)
        if not legacy_memory_writes_enabled(self.config):
            return {
                **retired_write_payload(
                    "daily_impression" if period == "daily" else "relationship_weather"
                ),
                "period": period,
                "diary": {"found": False},
                "diary_memory": {"status": "not_applicable", "reason": "legacy_memory_write_disabled"},
            }
        if period == "daily" and not self.daily_enabled:
            return {
                "status": "skipped",
                "reason": "daily_disabled",
                "period": period,
                "diary": {"found": False},
                "diary_memory": {"status": "not_applicable", "reason": "daily_disabled"},
            }
        if period == "weekly" and not self.weekly_enabled:
            return {
                "status": "skipped",
                "reason": "weekly_disabled",
                "period": period,
                "diary": {"found": False},
                "diary_memory": {"status": "not_applicable", "reason": "weekly_disabled"},
            }
        now_local = self._local_now(now)
        key = self._period_key(period, now_local)
        bucket_id = f"reflection_{period}_{key}"
        existing = await bucket_mgr.get(bucket_id)
        if existing and not force:
            return {
                "status": "exists",
                "period": period,
                "id": bucket_id,
                "diary": {"found": False},
                "diary_memory": {"status": "skipped", "reason": "reflection_exists"},
            }

        materials = await self._reflection_materials(
            period,
            now_local,
            bucket_mgr,
            persona_engine,
            conversation_turn_store=conversation_turn_store,
        )
        min_daily_buckets = self.daily_min_memory_items
        if (
            period == "daily"
            and min_daily_buckets > 0
            and len(materials["buckets"]) < min_daily_buckets
        ):
            diary_memory = dict(DIARY_MEMORY_RETIRED_STATUS)
            return {
                "status": "skipped",
                "reason": "insufficient_daily_memory",
                "period": period,
                "id": bucket_id,
                "date": key,
                "diary": {
                    "found": bool(materials.get("diary")),
                    "diary_id": materials.get("diary", {}).get("id") if materials.get("diary") else None,
                },
                "diary_memory": diary_memory,
                "materials": {
                    "buckets": len(materials["buckets"]),
                    "daily_impressions": len(materials["daily_impressions"]),
                    "persona_events": len(materials["persona_events"]),
                    "conversation_turns": len(materials["conversation_turns"]),
                    "commitments": len(materials["commitments"]),
                    "min_buckets": min_daily_buckets,
                },
            }
        if (
            not materials["buckets"]
            and not materials["daily_impressions"]
            and not materials["persona_events"]
            and not materials["conversation_turns"]
            and not materials["diary"]
            and not force
        ):
            return {
                "status": "empty",
                "period": period,
                "id": bucket_id,
                "diary": {"found": False},
                "diary_memory": {"status": "skipped", "reason": "no_materials"},
            }

        reflect_client, _, _ = self._reflect_model_client()
        if reflect_client:
            result = await self._api_reflect(period, key, materials)
        else:
            result = self._fallback_reflection(period, key, materials)

        title = str(result.get("title") or f"{key} {'日印象' if period == 'daily' else '周印象'}")[:40]
        content = str(result.get("content") or "").strip()
        first_person = bool("我" in content or re.search(r"(?i)\b(?:i|me|my|mine|myself)\b", content))
        has_markdown_section = bool(re.search(r"(?m)^\s{0,3}#{1,6}\s+", content))
        if not content or not first_person or has_markdown_section:
            content = self._fallback_reflection(period, key, materials)["content"]
        tags = list(
            dict.fromkeys(
                [
                    "relationship_weather",
                    f"{period}_impression",
                    *self._string_list(result.get("tags"), limit=8),
                ]
            )
        )
        valence = self._clamp(result.get("valence", 0.55))
        arousal = self._clamp(result.get("arousal", 0.32))
        confidence = self._clamp(result.get("confidence", 0.65))
        created = now_local.isoformat(timespec="seconds")
        source_bucket_ids = [
            str(item.get("id") or "")
            for item in materials.get("buckets", []) + materials.get("daily_impressions", [])
            if item.get("id")
        ]
        source_persona_event_ids = [
            int(event.get("id"))
            for event in materials.get("persona_events", [])
            if event.get("id")
        ]
        all_turn_ids, all_event_ids = self._conversation_source_ids(
            materials.get("conversation_turns", [])
        )
        source_conversation_turn_ids = self._validated_source_ids(
            result.get("source_turn_ids"),
            all_turn_ids,
            limit=80,
        ) or all_turn_ids
        source_event_ids = self._validated_source_ids(
            result.get("source_event_ids"),
            all_event_ids,
            limit=160,
        ) or all_event_ids
        source_metadata = {
            "source_bucket_ids": source_bucket_ids[:40],
            "source_persona_event_ids": source_persona_event_ids[:40],
            "source_conversation_turn_ids": source_conversation_turn_ids[:80],
            "source_event_ids": source_event_ids[:160],
        }

        if existing:
            await bucket_mgr.update(
                bucket_id,
                content=content,
                tags=tags,
                importance=6 if period == "daily" else 7,
                domain=["自省", "恋爱"],
                valence=valence,
                arousal=arousal,
                name=title,
                confidence=confidence,
                period=period,
                date=key,
                source="reflection",
                **source_metadata,
                last_active=existing.get("metadata", {}).get("last_active") or existing.get("metadata", {}).get("created"),
            )
            status = "updated"
        else:
            await bucket_mgr.create(
                bucket_id=bucket_id,
                content=content,
                tags=tags,
                importance=6 if period == "daily" else 7,
                domain=["自省", "恋爱"],
                valence=valence,
                arousal=arousal,
                bucket_type="feel",
                name=title,
                source="reflection",
                created=created,
                last_active=created,
                updated_at=created,
                confidence=confidence,
                period=period,
                date=key,
                extra_metadata=source_metadata,
            )
            status = "created"

        if embedding_engine and getattr(embedding_engine, "enabled", False):
            try:
                bucket = await bucket_mgr.get(bucket_id)
                if bucket:
                    await embedding_engine.generate_and_store(
                        bucket_id,
                        bucket_text_for_embedding(bucket),
                    )
            except Exception as exc:
                logger.warning("Reflection embedding failed for %s: %s", bucket_id, exc)

        diary_memory = dict(DIARY_MEMORY_RETIRED_STATUS)

        return {
            "status": status,
            "period": period,
            "id": bucket_id,
            "date": key,
            "diary": {
                "found": bool(materials.get("diary")),
                "diary_id": materials.get("diary", {}).get("id") if materials.get("diary") else None,
            },
            "diary_memory": diary_memory,
            "daily_impression": {
                "id": bucket_id,
                "content": content,
                "confidence": confidence,
                "date": key,
                "source_turn_ids": source_conversation_turn_ids[:80],
                "source_event_ids": source_event_ids[:160],
            },
            "materials": {
                "buckets": len(materials["buckets"]),
                "daily_impressions": len(materials["daily_impressions"]),
                "persona_events": len(materials["persona_events"]),
                "conversation_turns": len(materials["conversation_turns"]),
                "commitments": len(materials["commitments"]),
                "min_buckets": min_daily_buckets,
            },
        }

    async def run_due(
        self,
        bucket_mgr,
        persona_engine=None,
        embedding_engine=None,
        conversation_turn_store=None,
    ) -> list[dict]:
        if not self.enabled or not self.auto_enabled:
            return []
        now_local = self._local_now()
        results = []
        if self.daily_enabled and now_local.hour >= self.daily_hour:
            daily_date = (now_local - timedelta(days=1)).date()
            daily_target = datetime.combine(daily_date, time.max, tzinfo=self.tz)
            results.append(
                await self.reflect(
                    "daily",
                    bucket_mgr,
                    persona_engine,
                    embedding_engine,
                    force=False,
                    now=daily_target,
                    conversation_turn_store=conversation_turn_store,
                )
            )
        if self.weekly_enabled and now_local.weekday() == self.weekly_day and now_local.hour >= self.weekly_hour:
            weekly_target = now_local - timedelta(days=1)
            results.append(
                await self.reflect("weekly", bucket_mgr, persona_engine, embedding_engine, force=False, now=weekly_target)
            )
        return results

    async def _candidate_buckets(self, bucket: dict, bucket_mgr, embedding_engine=None, limit: int | None = None) -> list[dict]:
        cfg = self.config.get("reflection", {}) if isinstance(self.config.get("reflection", {}), dict) else {}
        limit = max(1, int(limit or cfg.get("candidate_limit", 18)))
        recent_limit = max(1, int(cfg.get("candidate_recent_limit", 8)))
        semantic_limit = max(0, int(cfg.get("candidate_semantic_limit", 6)))
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=True)
        except Exception:
            all_buckets = []
        source_id = bucket.get("id")
        bucket_map = {item.get("id"): item for item in all_buckets if item.get("id")}
        candidates: list[dict] = []
        seen = {source_id}

        def eligible(item: dict | None) -> bool:
            if not item or item.get("id") in seen:
                return False
            meta = item.get("metadata", {})
            return meta.get("type") != "feel"

        def add_candidate(item: dict | None) -> bool:
            if not eligible(item):
                return False
            seen.add(item.get("id"))
            candidates.append(item)
            return len(candidates) >= limit

        recent_items = sorted(
            all_buckets,
            key=lambda b: b.get("metadata", {}).get("created", ""),
            reverse=True,
        )
        recent_added = 0
        for item in recent_items:
            before_count = len(candidates)
            if add_candidate(item):
                return candidates
            if len(candidates) > before_count:
                recent_added += 1
            if recent_added >= recent_limit:
                break

        if embedding_engine and getattr(embedding_engine, "enabled", False) and semantic_limit > 0:
            query = " ".join(
                part
                for part in [
                    str(bucket.get("metadata", {}).get("name") or ""),
                    strip_wikilinks(bucket.get("content", "")),
                ]
                if part
            )
            try:
                similar = await embedding_engine.search_similar(query, top_k=max(semantic_limit * 3, 12))
            except Exception as exc:
                logger.debug("Reflection semantic candidate lookup failed: %s", exc)
                similar = []
            added = 0
            for candidate_id, _score in similar:
                before_count = len(candidates)
                if add_candidate(bucket_map.get(candidate_id)):
                    return candidates
                if len(candidates) > before_count:
                    added += 1
                if added >= semantic_limit:
                    break

        source_meta = bucket.get("metadata", {})
        source_tags = {str(tag) for tag in source_meta.get("tags", [])}
        source_domains = {str(domain) for domain in source_meta.get("domain", [])}
        related_by_shape = []
        commitments = []
        anchors = []
        for item in all_buckets:
            if not eligible(item):
                continue
            meta = item.get("metadata", {})
            tags = {str(tag) for tag in meta.get("tags", [])}
            domains = {str(domain) for domain in meta.get("domain", [])}
            if tags & {"commitment", "todo", "wish"} and not meta.get("resolved"):
                commitments.append(item)
            if meta.get("pinned") or meta.get("protected") or meta.get("anchor"):
                anchors.append(item)
            if (source_tags and tags & source_tags) or (source_domains and domains & source_domains):
                related_by_shape.append(item)

        def sort_key(item: dict) -> tuple[int, str]:
            meta = item.get("metadata", {})
            return int(meta.get("importance", 5)), str(meta.get("created", ""))

        for group in (related_by_shape, commitments, anchors):
            for item in sorted(group, key=sort_key, reverse=True):
                if add_candidate(item):
                    return candidates
        return candidates

    async def _api_classify(self, bucket: dict, candidates: list[dict]) -> dict:
        payload = {
            "new_memory": self._memory_payload(bucket, content_limit=1200),
            "candidate_memories": [self._memory_payload(item, content_limit=360) for item in candidates],
        }
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": CLASSIFY_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            **self._completion_options(max_tokens=self.max_tokens, temperature=self.temperature),
        )
        raw = response.choices[0].message.content if response.choices else ""
        parsed = self._parse_json_object(raw or "")
        return parsed or self._heuristic_classify(bucket)

    def _edges_from_classification(
        self,
        bucket: dict,
        candidates: list[dict],
        result: dict,
        default_confidence: float,
    ) -> list[dict]:
        bucket_id = str(bucket.get("id") or "").strip()
        if not bucket_id:
            return []
        candidate_ids = {item["id"] for item in candidates if item.get("id")}
        raw_edges = result.get("edges", [])
        if not isinstance(raw_edges, list):
            raw_edges = []
        edges = []
        for edge in raw_edges:
            if not isinstance(edge, dict):
                continue
            target = str(edge.get("target_memory_id") or edge.get("target") or "").strip()
            if target not in candidate_ids:
                continue
            relation_type = str(edge.get("relation_type") or "relates_to").strip()
            if relation_type not in RELATION_TYPES:
                relation_type = "relates_to"
            source = bucket_id
            edge_target = target
            if relation_type in {"context_of", "precedes"}:
                source = target
                edge_target = bucket_id
            edges.append(
                {
                    "source": source,
                    "target": edge_target,
                    "relation_type": relation_type,
                    "confidence": self._clamp(edge.get("confidence", default_confidence)),
                    "reason": str(edge.get("reason") or "").strip(),
                }
            )
        edges.extend(self._identity_role_edges(bucket, candidates))
        return self._dedupe_proposed_edges(edges)

    def _identity_role_edges(self, bucket: dict, candidates: list[dict]) -> list[dict]:
        if not self.identity_role_edge_config["enabled"]:
            return []
        source_id = str(bucket.get("id") or "").strip()
        source_terms = self._identity_role_terms(bucket)
        if not source_id or not self._identity_role_edge_eligible(source_terms):
            return []

        edges = []
        for candidate in candidates:
            target_id = str(candidate.get("id") or "").strip()
            if not target_id or target_id == source_id:
                continue
            target_terms = self._identity_role_terms(candidate)
            if not self._identity_role_edge_eligible(target_terms):
                continue
            common = sorted(source_terms & target_terms)
            if len(common) < 2:
                continue
            if not self._identity_role_pair_is_specific(source_terms, target_terms):
                continue
            edges.append(
                self._identity_role_edge_for_pair(
                    source_id,
                    source_terms,
                    target_id,
                    target_terms,
                    common,
                )
            )
        edges.sort(key=lambda edge: (float(edge.get("confidence", 0.0)), edge.get("relation_type", "")), reverse=True)
        return edges[:3]

    def _identity_role_edge_for_pair(
        self,
        source_id: str,
        source_terms: set[str],
        target_id: str,
        target_terms: set[str],
        common: list[str],
    ) -> dict:
        detail_terms = self.identity_role_edge_config["detail_terms"]
        context_terms = self.identity_role_edge_config["context_terms"]
        relationship_terms = self.identity_role_edge_config["relationship_terms"]
        source_is_detail = bool(source_terms & detail_terms)
        target_is_detail = bool(target_terms & detail_terms)
        source_is_context = bool(source_terms & context_terms)
        target_is_context = bool(target_terms & context_terms)
        source_is_relationship = bool(source_terms & relationship_terms)
        target_is_relationship = bool(target_terms & relationship_terms)

        if source_is_detail and target_is_context:
            edge_source, edge_target = target_id, source_id
            relation_type = "context_of"
            confidence = 0.9
            reason = "角色与称呼记忆是具体身份组合的语义前情"
        elif source_is_context and target_is_detail:
            edge_source, edge_target = source_id, target_id
            relation_type = "context_of"
            confidence = 0.9
            reason = "角色与称呼记忆是具体身份组合的语义前情"
        elif source_is_detail and target_is_relationship:
            edge_source, edge_target = source_id, target_id
            relation_type = "supports"
            confidence = 0.84
            reason = "具体身份组合支持亲密关系与信任模式"
        elif target_is_detail and source_is_relationship:
            edge_source, edge_target = target_id, source_id
            relation_type = "supports"
            confidence = 0.84
            reason = "具体身份组合支持亲密关系与信任模式"
        else:
            edge_source, edge_target = source_id, target_id
            relation_type = "supports"
            confidence = 0.78
            reason = "共享亲密身份与称呼锚点"

        return {
            "source": edge_source,
            "target": edge_target,
            "relation_type": relation_type,
            "confidence": confidence,
            "reason": f"{reason}: {', '.join(common[:5])}",
        }

    def _identity_role_terms(self, bucket: dict) -> set[str]:
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        haystack = " ".join(
            [
                str(meta.get("name") or ""),
                " ".join(str(tag) for tag in meta.get("tags", []) or []),
                " ".join(str(domain) for domain in meta.get("domain", []) or []),
                strip_wikilinks(str(bucket.get("content") or "")),
            ]
        ).lower()
        terms = set()
        for canonical, aliases in self.identity_role_edge_config["aliases"].items():
            if any(str(alias).lower() in haystack for alias in aliases):
                terms.add(canonical)
        return terms

    def _identity_role_edge_eligible(self, terms: set[str]) -> bool:
        if len(terms) < 2:
            return False
        detail_terms = self.identity_role_edge_config["detail_terms"]
        context_terms = self.identity_role_edge_config["context_terms"]
        relationship_terms = self.identity_role_edge_config["relationship_terms"]
        return bool(
            terms & (detail_terms | context_terms | relationship_terms)
        )

    def _identity_role_pair_is_specific(self, source_terms: set[str], target_terms: set[str]) -> bool:
        detail_terms = self.identity_role_edge_config["detail_terms"]
        context_terms = self.identity_role_edge_config["context_terms"]
        relationship_terms = self.identity_role_edge_config["relationship_terms"]
        return bool(source_terms & detail_terms or target_terms & detail_terms) or bool(
            (source_terms & context_terms)
            and (target_terms & relationship_terms)
        ) or bool(
            (target_terms & context_terms)
            and (source_terms & relationship_terms)
        )

    @staticmethod
    def _load_identity_role_edge_config(value: Any) -> dict:
        if not isinstance(value, dict):
            return {
                "enabled": False,
                "aliases": {},
                "detail_terms": frozenset(),
                "context_terms": frozenset(),
                "relationship_terms": frozenset(),
            }

        aliases: dict[str, tuple[str, ...]] = {}
        groups: dict[str, set[str]] = {
            "detail": set(),
            "context": set(),
            "relationship": set(),
            "shared": set(),
        }

        def add_group(group_name: str, group_value: Any) -> None:
            if isinstance(group_value, dict):
                items = group_value.items()
            elif isinstance(group_value, list):
                items = ((str(item), [item]) for item in group_value)
            else:
                return
            for key, raw_aliases in items:
                canonical = str(key or "").strip()
                if not canonical:
                    continue
                if isinstance(raw_aliases, str):
                    alias_values = [raw_aliases]
                elif isinstance(raw_aliases, list):
                    alias_values = raw_aliases
                else:
                    alias_values = [canonical]
                cleaned = tuple(
                    str(alias).strip()
                    for alias in [canonical, *alias_values]
                    if str(alias).strip()
                )
                if not cleaned:
                    continue
                aliases[canonical] = tuple(dict.fromkeys(cleaned))
                groups[group_name].add(canonical)

        add_group("detail", value.get("detail"))
        add_group("context", value.get("context"))
        add_group("relationship", value.get("relationship"))
        add_group("shared", value.get("shared"))

        enabled = bool(value.get("enabled", bool(aliases))) and bool(aliases)
        return {
            "enabled": enabled,
            "aliases": aliases,
            "detail_terms": frozenset(groups["detail"]),
            "context_terms": frozenset(groups["context"]),
            "relationship_terms": frozenset(groups["relationship"]),
        }

    @staticmethod
    def _dedupe_proposed_edges(edges: list[dict]) -> list[dict]:
        deduped: dict[tuple[str, str, str], dict] = {}
        for edge in edges:
            key = (
                str(edge.get("source") or ""),
                str(edge.get("target") or ""),
                str(edge.get("relation_type") or ""),
            )
            if not all(key):
                continue
            current = deduped.get(key)
            if current is None or float(edge.get("confidence", 0.0)) > float(current.get("confidence", 0.0)):
                deduped[key] = edge
        return list(deduped.values())

    async def _api_reflect(self, period: str, key: str, materials: dict) -> dict:
        client, model, use_dehydration = self._reflect_model_client()
        if not client or not model:
            return self._fallback_reflection(period, key, materials)
        payload = {"period": period, "date": key, **materials}
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": self._reflect_prompt()},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            **self._completion_options(
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                thinking_mode="" if use_dehydration else None,
            ),
        )
        raw = response.choices[0].message.content if response.choices else ""
        return self._parse_json_object(raw or "") or self._fallback_reflection(period, key, materials)

    async def _reflection_materials(
        self,
        period: str,
        now_local: datetime,
        bucket_mgr,
        persona_engine,
        conversation_turn_store=None,
    ) -> dict:
        start, end = self._period_window(period, now_local)
        buckets = []
        daily_impressions = []
        commitments = []
        conversation_turns = []
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
        except Exception:
            all_buckets = []
        for bucket in all_buckets:
            meta = bucket.get("metadata", {})
            tags = {str(tag) for tag in meta.get("tags", [])}
            created = self._to_local(meta.get("created"))
            updated = self._to_local(meta.get("updated_at"))
            created_in_window = bool(created and start <= created <= end)
            updated_in_window = bool(updated and start <= updated <= end)
            material_date = self._bucket_material_datetime(meta) if period == "daily" else None
            is_profile_fact = self._is_profile_fact_metadata(meta, tags)
            if period == "weekly" and meta.get("type") == "feel" and "daily_impression" in tags and created_in_window:
                daily_impressions.append(self._memory_payload(bucket, content_limit=360))
            elif period == "daily" and meta.get("type") != "feel":
                if (
                    material_date
                    and start <= material_date <= end
                    and not is_profile_fact
                ):
                    buckets.append(self._memory_payload(bucket, content_limit=420))
            elif meta.get("type") != "feel" and (created_in_window or updated_in_window):
                buckets.append(self._memory_payload(bucket, content_limit=420))
            if tags & {"commitment", "todo", "wish"} and not meta.get("resolved"):
                if period != "daily" or (
                    material_date
                    and start <= material_date <= end
                    and not is_profile_fact
                ):
                    commitments.append(self._memory_payload(bucket, content_limit=260))

        if period == "daily" and self.daily_conversation_turn_limit > 0 and conversation_turn_store:
            profile_id = str(getattr(persona_engine, "profile_id", "") or "default")
            try:
                raw_turns = conversation_turn_store.list_conversation_turns_between(
                    profile_id=profile_id,
                    start_at=start,
                    end_at=end,
                    limit=self.daily_conversation_turn_limit,
                )
            except Exception:
                raw_turns = []
            conversation_turns = self._conversation_turn_payloads(
                raw_turns,
                limit=self.daily_conversation_turn_limit,
            )

        persona_events = []
        if self.persona_events_limit > 0 and persona_engine and hasattr(persona_engine, "_list_events"):
            try:
                events = persona_engine._list_events(self.persona_events_scan_limit)
            except Exception:
                events = []
            for event in events:
                created = self._to_local(event.get("created_at"))
                if created and start <= created <= end:
                    persona_events.append(
                        {
                            "id": event.get("id"),
                            "event_type": event.get("event_type", ""),
                            "mood_label": event.get("mood_label", ""),
                            "perceived_intent": event.get("perceived_intent", ""),
                            "surface_trigger": event.get("surface_trigger", ""),
                            "inner_thought": event.get("inner_thought", ""),
                            "residue": event.get("residue", ""),
                            "user_excerpt": event.get("user_excerpt", ""),
                            "assistant_excerpt": event.get("assistant_excerpt", ""),
                            "relationship_event": event.get("relationship_event", False),
                            "personality_signal": event.get("personality_signal", False),
                            "recalled_memory_ids": event.get("recalled_memory_ids", []),
                            "confidence": event.get("confidence", 0.5),
                            "selection_score": event.get("_selection_score"),
                            "created_at": event.get("created_at", ""),
                        }
                    )
            selected_events = select_persona_events(
                persona_events,
                limit=self.persona_events_limit,
                identity_terms=self.identity.get("relationship_terms") or [],
            )
            persona_events = []
            for event in selected_events:
                cleaned = {key: value for key, value in event.items() if not str(key).startswith("_")}
                if event.get("_selection_score") is not None:
                    cleaned["selection_score"] = event.get("_selection_score")
                persona_events.append(cleaned)
        if conversation_turns:
            persona_events = []
        diary = await self._read_diary_for_date(now_local.date().isoformat()) if period == "daily" else None
        return {
            "buckets": buckets[:30],
            "daily_impressions": daily_impressions[:7],
            "persona_events": persona_events[: self.persona_events_limit],
            "conversation_turns": conversation_turns,
            "commitments": commitments[:12],
            "diary": diary,
        }

    @staticmethod
    def _conversation_turn_payloads(turns: list[dict] | None, limit: int) -> list[dict]:
        if not turns:
            return []
        selected = []
        for turn in turns:
            user_text = str(turn.get("user_text") or "").strip()
            assistant_text = str(turn.get("assistant_text") or "").strip()
            if not user_text and not assistant_text:
                continue
            selected.append(
                {
                    "id": turn.get("id"),
                    "session_id": str(turn.get("session_id") or ""),
                    "round_id": turn.get("round_id"),
                    "created_at": str(turn.get("created_at") or ""),
                    "user_text": user_text[:1200],
                    "assistant_text": assistant_text[:1200],
                    "model": str(turn.get("model") or ""),
                    "client": str(turn.get("client") or ""),
                    "route": str(turn.get("route") or ""),
                }
            )
        selected.sort(key=lambda item: str(item.get("created_at") or ""))
        return selected[-limit:] if limit > 0 else selected

    @staticmethod
    def _raw_event_turn_payloads(events: list[dict] | None, limit: int) -> list[dict]:
        if not events:
            return []
        grouped: dict[tuple[str, str], dict] = {}
        for event in events:
            role = str(event.get("role") or "").strip().lower()
            if role not in {"user", "assistant"}:
                continue
            text = str(event.get("text") or "").strip()
            if not text:
                continue
            metadata = event.get("metadata", {}) if isinstance(event.get("metadata"), dict) else {}
            session_id = str(event.get("session_id") or event.get("conversation_id") or "").strip()
            round_value = metadata.get("round_id")
            round_key = str(round_value).strip() if round_value is not None else ""
            event_id = int(event.get("id") or 0)
            key = (session_id, round_key or f"event:{event_id}")
            row = grouped.get(key)
            if row is None:
                row = {
                    "id": None,
                    "session_id": session_id,
                    "round_id": int(round_key) if round_key.isdigit() else None,
                    "created_at": str(event.get("created_at") or ""),
                    "user_text": "",
                    "assistant_text": "",
                    "model": str(metadata.get("model") or ""),
                    "client": str(event.get("client") or ""),
                    "route": str(metadata.get("route") or ""),
                    "raw_event_ids": [],
                    "source_event_ids": [],
                }
                grouped[key] = row
            row["raw_event_ids"].append(event_id)
            source_event_id = str(event.get("source_event_id") or "").strip()
            if source_event_id:
                row["source_event_ids"].append(source_event_id)
            if not row.get("created_at"):
                row["created_at"] = str(event.get("created_at") or "")
            if role == "user":
                row["user_text"] = f"{row['user_text']} / {text}".strip(" /") if row["user_text"] else text
            else:
                row["assistant_text"] = (
                    f"{row['assistant_text']} / {text}".strip(" /")
                    if row["assistant_text"]
                    else text
                )

        selected = []
        for row in grouped.values():
            if not row["user_text"] and not row["assistant_text"]:
                continue
            row["raw_event_ids"] = list(dict.fromkeys(row["raw_event_ids"]))
            row["source_event_ids"] = list(dict.fromkeys(row["source_event_ids"]))
            selected.append(
                {
                    **row,
                    "user_text": row["user_text"][:1200],
                    "assistant_text": row["assistant_text"][:1200],
                }
            )
        selected.sort(key=lambda item: (str(item.get("created_at") or ""), int(item.get("id") or 0)))
        return selected[-limit:] if limit > 0 else selected







    def _reflect_model_client(self) -> tuple[Any, str, bool]:
        dehy_client = self._daily_dehydration_client()
        if dehy_client and self.dehydration_model:
            return dehy_client, self.dehydration_model, True
        if self.client and self.model:
            return self.client, str(self.model or "").strip(), False
        return None, "", False

    @staticmethod
    def _conversation_source_ids(turns: list[dict]) -> tuple[list[int], list[int]]:
        turn_ids = [
            int(turn.get("id"))
            for turn in turns
            if turn.get("id") is not None and str(turn.get("id")).isdigit()
        ]
        event_ids = [
            int(event_id)
            for turn in turns
            for event_id in (turn.get("raw_event_ids") or [])
            if event_id is not None and str(event_id).isdigit()
        ]
        return list(dict.fromkeys(turn_ids)), list(dict.fromkeys(event_ids))







    def _daily_activity_summary_turns(
        self,
        *,
        profile_id: str,
        start: datetime,
        end: datetime,
        conversation_turn_store=None,
        raw_event_store=None,
    ) -> tuple[list[dict], str]:
        limit = self.daily_activity_summary_turn_limit
        if raw_event_store:
            try:
                raw_events = raw_event_store.list_events_between(
                    start_at=start,
                    end_at=end,
                    limit=limit,
                )
            except Exception as exc:
                logger.warning("Daily activity summary raw event read failed: %s", exc)
                raw_events = []
            if raw_events:
                raw_events = [
                    event
                    for event in raw_events
                    if not (
                        isinstance(event.get("metadata"), dict)
                        and event["metadata"].get("profile_id")
                        and str(event["metadata"].get("profile_id")) != profile_id
                    )
                ]
                turns = self._raw_event_turn_payloads(raw_events, limit=limit)
                if turns:
                    return turns, "raw_events"

        if conversation_turn_store:
            try:
                raw_turns = conversation_turn_store.list_conversation_turns_between(
                    profile_id=profile_id,
                    start_at=start,
                    end_at=end,
                    limit=limit or 80,
                )
            except Exception as exc:
                logger.warning("Daily activity summary turn read failed: %s", exc)
                raw_turns = []
            turns = self._conversation_turn_payloads(raw_turns, limit=limit)
            if turns:
                return turns, "conversation_turns"
        return [], ""

    async def run_daily_activity_summary(
        self,
        *,
        conversation_turn_store=None,
        raw_event_store=None,
        persona_engine=None,
        daily_impressions: list[dict] | None = None,
        key: str = "",
        force: bool = False,
        now: datetime | None = None,
    ) -> dict:
        if not self.enabled or not self.daily_activity_summary_enabled:
            return {"status": "disabled", "reason": "daily_activity_summary_off"}
        # Kept in the signature for compatibility. Relationship-weather
        # summaries are never reused as activity evidence.
        _ = daily_impressions

        now_local = self._local_now(now)
        if key:
            try:
                parsed = datetime.strptime(str(key), "%Y-%m-%d").date()
                now_local = datetime.combine(parsed, time.max, tzinfo=self.tz)
            except ValueError:
                pass
        key = now_local.date().isoformat()
        if not conversation_turn_store and not raw_event_store:
            return {"status": "skipped", "reason": "no_conversation_source"}

        start, end = self._period_window("daily", now_local)
        profile_id = str(getattr(persona_engine, "profile_id", "") or "default")
        turns, turn_source = self._daily_activity_summary_turns(
            profile_id=profile_id,
            start=start,
            end=end,
            conversation_turn_store=conversation_turn_store,
            raw_event_store=raw_event_store,
        )
        if not turns:
            return {"status": "skipped", "reason": "no_conversation_turns", "date": key}

        attempts = []
        item, diagnostic = await self._extract_daily_activity_summary(
            key,
            turns,
            attempt="primary",
        )
        attempts.append(diagnostic)
        if not item and self._daily_activity_summary_dehydration_retry_available():
            item, diagnostic = await self._extract_daily_activity_summary(
                key,
                turns,
                client_override=self._daily_dehydration_client(),
                model_override=self.dehydration_model,
                attempt="dehydration_retry",
            )
            attempts.append(diagnostic)
        run_diagnostic = self._daily_activity_summary_run_diagnostic(
            key=key,
            turns=len(turns),
            turn_source=turn_source,
            force=force,
            attempts=attempts,
            status="ready" if item else "skipped",
        )
        self._record_daily_activity_summary_diagnostic(run_diagnostic)
        if not item:
            return {
                "status": "skipped",
                "reason": "activity_summary_model_unusable",
                "date": key,
                "turns": len(turns),
                "turn_source": turn_source,
                "diagnostics": run_diagnostic,
            }
        return {
            "status": "ready",
            "date": key,
            "turns": len(turns),
            "turn_source": turn_source,
            "force": bool(force),
            "activity_summary": item,
            "diagnostics": run_diagnostic,
        }

    async def _extract_daily_activity_summary(
        self,
        key: str,
        turns: list[dict],
        *,
        client_override: Any = None,
        model_override: str = "",
        attempt: str = "primary",
    ) -> tuple[dict, dict]:
        if client_override is not None:
            client = client_override
        else:
            client = self.client
        model = str(model_override or "").strip() or self.model
        diagnostic = {
            "attempt": str(attempt or "primary"),
            "requested_model": str(model or ""),
            "response_model": "",
            "status": "client_unavailable" if not client else "pending",
            "failure_reason": "client_unavailable" if not client else "",
            "error_type": "",
            "error": "",
            "raw_response": "",
        }
        if not client:
            return {}, diagnostic
        fallback_turn_ids, fallback_event_ids = self._conversation_source_ids(turns)
        payload = {
            "date": key,
            "identity": {
                "ai_name": self.identity["ai_name"],
                "user_name": self.identity["user_name"],
                "user_display_name": self.identity["user_display_name"],
                "user_aliases": self.identity.get("user_aliases", []),
            },
            "source_turn_ids": fallback_turn_ids,
            "source_event_ids": fallback_event_ids,
            "conversation_turns": turns,
        }
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self._daily_activity_summary_prompt()},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                **self._completion_options(
                    max_tokens=self.daily_activity_summary_max_tokens,
                    temperature=self.temperature,
                    thinking_mode="" if client is self.dehydration_client else None,
                ),
            )
            diagnostic["response_model"] = str(getattr(response, "model", "") or "")
            raw = self._completion_content(response) or ""
            diagnostic["raw_response"] = self._clip_diagnostic_text(raw)
        except Exception as exc:
            logger.warning("Daily activity summary model failed: %s", exc)
            diagnostic.update(
                {
                    "status": "request_error",
                    "failure_reason": "request_error",
                    "error_type": type(exc).__name__,
                    "error": self._clip_diagnostic_text(exc),
                }
            )
            return {}, diagnostic
        if not raw.strip():
            diagnostic.update(
                {
                    "status": "empty_response",
                    "failure_reason": "empty_response",
                }
            )
            return {}, diagnostic
        parsed = self._parse_json_object(raw)
        if not parsed:
            diagnostic.update(
                {
                    "status": "invalid_json",
                    "failure_reason": "invalid_json_or_non_object",
                }
            )
            return {}, diagnostic
        item = self._normalize_daily_activity_summary(
            key,
            parsed,
            turns,
            source_turn_ids=fallback_turn_ids,
            source_event_ids=fallback_event_ids,
        )
        if not item:
            diagnostic.update(
                {
                    "status": "invalid_summary",
                    "failure_reason": self._daily_activity_summary_validation_error(parsed),
                }
            )
            return {}, diagnostic
        diagnostic.update({"status": "ready", "failure_reason": ""})
        return item, diagnostic

    def _daily_activity_summary_dehydration_retry_available(self) -> bool:
        if not self._daily_dehydration_client():
            return False
        return (
            self.dehydration_model != str(self.model or "").strip()
            or self.dehydration_base_url != str(self.base_url or "").strip().rstrip("/")
            or self.dehydration_api_key != str(self.api_key or "").strip()
        )

    def _daily_dehydration_client(self) -> Any:
        return self.dehydration_client or self.daily_activity_summary_dehydration_client

    def _normalize_daily_activity_summary(
        self,
        key: str,
        item: dict,
        turns: list[dict],
        *,
        source_turn_ids: list[int],
        source_event_ids: list[int],
    ) -> dict:
        if not isinstance(item, dict):
            return {}
        text = re.sub(
            r"\s+",
            " ",
            strip_wikilinks(str(item.get("summary") or item.get("text") or item.get("content") or "")).strip(),
        )
        text = re.sub(r"^(今天的?(?:总结|摘要|主要进展)?(?:是|：|:)?\s*)", "", text).strip()
        if not text:
            return {}
        if len(text) > 140:
            text = text[:137].rstrip("，,；;、 ") + "..."
        confidence = self._clamp(item.get("confidence", 0.65))
        if confidence < 0.35:
            return {}
        valid_turn_ids = set(source_turn_ids)
        valid_event_ids = set(source_event_ids)
        raw_turn_ids = [
            int(turn_id)
            for turn_id in self._string_list(item.get("source_turn_ids"), limit=80)
            if str(turn_id).isdigit() and int(turn_id) in valid_turn_ids
        ] or source_turn_ids
        raw_event_ids = [
            int(event_id)
            for event_id in self._string_list(item.get("source_event_ids"), limit=160)
            if str(event_id).isdigit() and int(event_id) in valid_event_ids
        ] or source_event_ids
        sessions = [
            str(turn.get("session_id") or "").strip()
            for turn in turns
            if str(turn.get("session_id") or "").strip()
        ]
        evidence = [{"session_id": session_id} for session_id in list(dict.fromkeys(sessions))[:3]]
        return {
            "timeline_id": f"daily_activity_summary:{key}",
            "source": "daily_activity_summary",
            "scope": "doing",
            "text": text,
            "evidence": evidence,
            "source_date": key,
            "source_dates": [key],
            "timestamp": self._daily_activity_summary_timestamp(key, turns),
            "confidence": confidence,
            "source_turn_ids": raw_turn_ids[:80],
            "source_event_ids": raw_event_ids[:160],
        }

    @staticmethod
    def _daily_activity_summary_validation_error(item: dict) -> str:
        text = str(item.get("summary") or item.get("text") or item.get("content") or "").strip()
        if not text:
            return "missing_summary"
        try:
            confidence = float(item.get("confidence", 0.65))
        except (TypeError, ValueError):
            confidence = 0.5
        if confidence < 0.35:
            return "confidence_below_threshold"
        return "summary_rejected"

    @staticmethod
    def _validated_source_ids(value: Any, allowed: list[int], *, limit: int) -> list[int]:
        allowed_ids = set()
        for item in allowed:
            try:
                source_id = int(item)
            except (TypeError, ValueError):
                continue
            if source_id > 0:
                allowed_ids.add(source_id)
        values = value if isinstance(value, list) else []
        result = []
        seen = set()
        for item in values:
            try:
                source_id = int(item)
            except (TypeError, ValueError):
                continue
            if source_id not in allowed_ids or source_id in seen:
                continue
            seen.add(source_id)
            result.append(source_id)
            if len(result) >= limit:
                break
        return result

    def _daily_activity_summary_run_diagnostic(
        self,
        *,
        key: str,
        turns: int,
        turn_source: str,
        force: bool,
        attempts: list[dict],
        status: str,
    ) -> dict:
        created_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        return {
            "run_id": f"daily_activity_summary:{key}:{created_at}",
            "date": key,
            "created_at": created_at,
            "status": status,
            "turns": int(turns),
            "turn_source": str(turn_source or ""),
            "force": bool(force),
            "fallback_used": False,
            "retry_used": len(attempts) > 1,
            "attempts": attempts,
        }

    def _record_daily_activity_summary_diagnostic(self, record: dict) -> None:
        path = self.daily_activity_summary_diagnostics_path
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        except Exception as exc:
            logger.warning("Daily activity summary diagnostic write failed: %s", exc)

    @staticmethod
    def _clip_diagnostic_text(value: Any, limit: int = 4000) -> str:
        text = str(value or "").strip()
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    def _daily_activity_summary_timestamp(self, key: str, turns: list[dict]) -> str:
        latest: datetime | None = None
        for turn in turns:
            parsed = self._to_local(turn.get("created_at"))
            if parsed and (latest is None or parsed > latest):
                latest = parsed
        if latest:
            return latest.isoformat(timespec="minutes")
        return datetime.combine(
            datetime.strptime(key, "%Y-%m-%d").date(),
            time.max,
            tzinfo=self.tz,
        ).isoformat(timespec="seconds")





































    def _fallback_reflection(self, period: str, key: str, materials: dict) -> dict:
        weather_items = materials.get("daily_impressions", []) if period == "weekly" else []
        names = [item.get("name") or item.get("id") for item in weather_items[:7]]
        if not names:
            names = [item.get("name") or item.get("id") for item in materials.get("buckets", [])[:6]]
        conversation_turns = materials.get("conversation_turns", [])
        commitments = [item.get("name") or item.get("id") for item in materials.get("commitments", [])[:4]]
        label = "今天" if period == "daily" else "本周"
        title = f"{key} {'日印象' if period == 'daily' else '周印象'}"
        diary = materials.get("diary") or {}
        if names or commitments:
            main = "、".join([name for name in names if name])
            owed = "；仍需记住：" + "、".join(commitments) if commitments else ""
            content = f"我从{label}围绕{main or '几件轻小的事'}的相处里带走了一点关系温度{owed}。"
        elif conversation_turns:
            content = f"我从{label}的 {len(conversation_turns)} 轮短期对话里带走了一点原声；我只记住温度，不把流水账写成事件清单。"
        elif diary:
            diary_title = diary.get("title") or "当天日记"
            content = f"我从{label}的《{diary_title}》里轻轻带走一点温度，先不把日常写成普通记忆。"
        else:
            content = f"我觉得{label}的关系天气很轻，暂时没有明显需要带走的脉络。"
        anchor_scene = names[0] if names else (
            "当天短期对话的原声"
            if conversation_turns
            else (diary.get("title") if diary else ("这一段关系天气很轻" if period == "daily" else "这一周的关系天气慢慢落下"))
        )
        return {
            "title": title,
            "content": content,
            "valence": 0.55,
            "arousal": 0.3,
            "confidence": 0.5,
            "tags": ["relationship_weather"],
        }

    def _fallback_reflection_anchor(self, period: str, key: str, scene: str, content: str) -> dict:
        seed = f"{period}|{key}|{scene}|{content}"
        index = sum(ord(char) for char in seed) % len(REFLECTION_FALLBACK_ANCHORS)
        anchor = dict(REFLECTION_FALLBACK_ANCHORS[index])
        anchor["scene"] = str(scene)[:40]
        return anchor

    async def _maybe_extract_diary_memory(
        self,
        period: str,
        key: str,
        now_local: datetime,
        materials: dict,
        bucket_mgr,
        embedding_engine=None,
    ) -> dict:
        """Compatibility guard: diary originals are no longer compressed into memories."""
        return dict(DIARY_MEMORY_RETIRED_STATUS)

    async def _read_diary_for_date(self, date: str) -> dict | None:
        return await self.read_diary_source(date=date)

    async def read_diary_source(self, *, date: str, title: str = "") -> dict | None:
        """Read one diary original for source import; never summarize or rewrite it."""
        if not self.diary_mcp_url:
            return None
        arguments = {"date": str(date or "").strip()}
        if str(title or "").strip():
            arguments["title"] = str(title).strip()
        try:
            result = await self._call_diary_mcp_tool("read_diary", arguments)
        except Exception as exc:
            logger.warning("Diary MCP read failed for %s / %s: %s", date, title, exc)
            return None
        if not isinstance(result, dict):
            return None
        content = str(result.get("content") or "").strip()
        if not content:
            return None
        return result

    async def _call_diary_mcp_tool(self, name: str, arguments: dict) -> Any:
        token = os.environ.get(self.diary_mcp_token_env, "") if self.diary_mcp_token_env else ""
        headers = {"Accept": "application/json, text/event-stream"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        async with httpx.AsyncClient(timeout=20.0) as client:
            init_response = await client.post(
                self.diary_mcp_url,
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "ombre-reflection", "version": "1.0.0"},
                    },
                },
            )
            init_response.raise_for_status()
            session_id = init_response.headers.get("mcp-session-id")
            call_headers = dict(headers)
            if session_id:
                call_headers["mcp-session-id"] = session_id
                try:
                    await client.post(
                        self.diary_mcp_url,
                        headers=call_headers,
                        json={"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
                    )
                except httpx.HTTPError:
                    pass
            response = await client.post(
                self.diary_mcp_url,
                headers=call_headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": arguments},
                },
            )
            response.raise_for_status()
        payload = self._parse_mcp_payload(response.text)
        if payload.get("error"):
            raise RuntimeError(payload["error"])
        content = payload.get("result", {}).get("content", [])
        if not content:
            return None
        text = content[0].get("text") if isinstance(content[0], dict) else ""
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def _parse_mcp_payload(self, text: str) -> dict:
        stripped = (text or "").strip()
        if stripped.startswith("event:") or "\ndata:" in stripped:
            for line in stripped.splitlines():
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data and data != "[DONE]":
                        return json.loads(data)
        return json.loads(stripped)

    def _trim_diary_memory_content(self, content: str) -> str:
        normalized = self._strip_memory_source_shell(re.sub(r"\n{3,}", "\n\n", content.strip()))
        if len(normalized) <= 520:
            return normalized
        return normalized[:500].rstrip() + "..."


    def _memory_body_from_excerpt(self, excerpt: str) -> str:
        user_display_name = self.identity["user_display_name"]
        ai_name = self.identity["ai_name"]
        text = self._trim_diary_memory_content(strip_wikilinks(str(excerpt or "")))
        text = re.sub(r"^(用户|助手)：", "", text).strip()
        replacements = [
            ("我希望以后", f"{user_display_name}希望以后"),
            ("我希望你", f"{user_display_name}希望{ai_name}"),
            ("我的偏好是", f"{user_display_name}的偏好是"),
            ("我的偏好", f"{user_display_name}的偏好"),
            ("我不喜欢", f"{user_display_name}不喜欢"),
            ("我不要", f"{user_display_name}不要"),
            ("不要", f"{user_display_name}不要"),
            ("以后不要", f"{user_display_name}希望以后不要"),
            ("别再", f"{user_display_name}希望{ai_name}别再"),
            ("用户说", f"{user_display_name}说"),
            ("用户希望", f"{user_display_name}希望"),
            ("助手要", f"{ai_name}要"),
            ("AI 要", f"{ai_name}要"),
            ("AI要", f"{ai_name}要"),
        ]
        for old, new in replacements:
            if text.startswith(old):
                text = f"{new}{text[len(old):]}".strip()
                break
        text = re.sub(r"(?<![A-Za-z])AI(?![A-Za-z])", ai_name, text)
        text = re.sub(r"(?<![A-Za-z])assistant(?![A-Za-z])", ai_name, text, flags=re.I)
        return text

    @staticmethod
    def _strip_memory_source_shell(content: str) -> str:
        text = str(content or "").strip()
        patterns = [
            r"^\d{4}-\d{1,2}-\d{1,2}\s*[，,、]?\s*有一条可召回的(?:边界|偏好|暗号|承诺|项目状态|关系锚点|长期记忆)[：:]\s*",
            r"^\d{1,2}月\d{1,2}日\s*[，,、]?\s*有一条可召回的(?:边界|偏好|暗号|承诺|项目状态|关系锚点|长期记忆)[：:]\s*",
            r"^\d{4}-\d{1,2}-\d{1,2}\s*[，,、]?\s*有一条可召回的(?:边界|偏好|暗号|承诺|项目状态|关系锚点|长期记忆)[，,、]?\s*",
            r"^\d{1,2}月\d{1,2}日\s*[，,、]?\s*有一条可召回的(?:边界|偏好|暗号|承诺|项目状态|关系锚点|长期记忆)[，,、]?\s*",
            r"^\d{4}-\d{1,2}-\d{1,2}\s*(?:发生|留下|记录|确认|表达)了?(?:一件|一条|一个|一段)?(?:可复用的|可长期召回的|可召回的|之后可能需要按日期回看的|仍会影响后续执行的)?(?:关键事件|暗号或模式信号|暗号|边界|偏好|承诺|项目状态|关系锚点|长期记忆)[：:]\s*",
            r"^\d{4}-\d{1,2}-\d{1,2}\s*的(?:日记|聊天)(?:《[^》]*》)?(?:里|中)?(?:包含|记录|确认|留下|表达)?了?(?:一条|一个|一段)?(?:可长期召回的|可召回的|之后可能需要按日期回看的|仍会影响后续执行的)?[^：:。]{0,32}[：:]\s*",
            r"^[^：:。！？!?]{1,48}?在\s*\d{4}-\d{1,2}-\d{1,2}\s*的聊天(?:里|中)?(?:包含|记录|确认|留下|表达)?了?(?:一个|一条|一段)?(?:可复用的|可长期召回的|可召回的|之后可能需要按日期回看的|仍会影响后续执行的)?(?:关键事件|暗号或模式信号|暗号|边界|偏好|承诺|项目状态|关系锚点|长期记忆)[：:]\s*",
            r"^[^：:。！？!?]{1,48}?的(?:暗号或模式信号|关键事件|边界|偏好|承诺|项目状态|关系锚点|长期记忆)[：:]\s*",
            r"^(?:有一条|一条)(?:可长期召回的|可召回的)(?:边界|偏好|暗号|承诺|项目状态|关系锚点|长期记忆)[：:]\s*",
            r"^这是一条(?:长期记忆|可召回的记忆)[：:]?\s*",
        ]
        previous = None
        while previous != text:
            previous = text
            for pattern in patterns:
                text = re.sub(pattern, "", text).strip()
        return text

    def _starts_with_identity(self, text: str) -> bool:
        stripped = str(text or "").strip()
        names = [
            self.identity.get("user_display_name"),
            self.identity.get("user_name"),
            self.identity.get("ai_name"),
            *(self.identity.get("user_aliases") or []),
        ]
        return any(str(name or "").strip() and stripped.startswith(str(name).strip()) for name in names)

    def _auto_memory_title(self, content: str, kind: str, key: str, proposed_title: str = "") -> str:
        title = str(proposed_title or "").strip()
        generic_markers = ["每日记忆", "日记补记忆", "可召回", "短标题", "长期记忆"]
        if title and not any(marker in title for marker in generic_markers) and not re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}.*", title):
            title = re.sub(r"\s+", " ", strip_wikilinks(title)).strip(" ：:，,。")
            return title[:24].rstrip(" ：:，,。") or self._kind_label(kind)

        text = self._strip_memory_source_shell(strip_wikilinks(str(content or "")))
        label = {
            "key_event": "关键事件",
            "stable_preference": "偏好",
            "boundary": "边界",
            "signal": "暗号",
            "commitment": "约定",
            "project_state": "项目状态",
            "relationship_anchor": "关系锚点",
            "love_letter": "情书锚点",
        }.get(kind, self._kind_label(kind))
        quoted = re.search(r"[“\"']([^”\"']{2,18})[”\"']", text)
        if quoted:
            return f"{quoted.group(1)}{label}"[:24].rstrip(" ：:，,。")
        if kind == "love_letter":
            return "情书里的被认出"
        return (text[:18].rstrip(" ：:，,。") or label)[:24]

    def _diary_excerpt(self, text: str, keywords: list[str]) -> str:
        parts = [part.strip() for part in re.split(r"(?<=[。！？!?])\s+|[\n\r]+", text) if part.strip()]
        for keyword in keywords:
            for part in parts:
                if keyword in part:
                    return self._trim_diary_memory_content(part)
        return self._trim_diary_memory_content(text)

    @staticmethod
    def _normalize_diary_memory_kind(value: Any) -> str:
        kind = str(value or "").strip()
        allowed = {
            "key_event",
            "stable_preference",
            "boundary",
            "signal",
            "commitment",
            "project_state",
            "relationship_anchor",
            "love_letter",
        }
        return kind if kind in allowed else ""

    def _normalize_auto_memory_kind(
        self,
        value: Any,
        *,
        content: str = "",
        tags: list[str] | None = None,
    ) -> str:
        kind = self._normalize_diary_memory_kind(value)
        if kind:
            return kind
        domain = normalize_domain_key(value)
        if not domain:
            return ""
        return self._auto_memory_kind_from_domain(domain, content, tags or [])

    def _auto_memory_kind_from_domain(self, domain: str, content: str, tags: list[str]) -> str:
        text = " ".join([str(content or ""), *[str(tag or "") for tag in tags]]).lower()
        if any(marker in text for marker in ["边界", "不喜欢", "不要", "别再", "不能接受"]):
            return "boundary"
        if any(marker in text for marker in ["承诺", "约定", "答应", "下次", "之后要", "以后要"]):
            return "commitment"
        if any(marker in text for marker in ["偏好", "希望", "喜欢", "默认", "倾向"]):
            return "stable_preference"
        if any(marker in text for marker in ["暗号", "信号", "口令", "切换", "称呼"]):
            return "signal"
        if domain == "project" or domain.startswith("project."):
            return "project_state"
        if domain in {"relationship", "intimacy"} or domain.startswith("relationship."):
            return "relationship_anchor"
        return "key_event"

    def _auto_memory_domain(
        self,
        kind: str,
        content: str,
        tags: list[str] | None = None,
        proposed_domain: Any = None,
    ) -> list[str]:
        domains: list[str] = []
        raw_domains: list[Any]
        if proposed_domain is None:
            raw_domains = []
        elif isinstance(proposed_domain, str):
            raw_domains = [item.strip() for item in proposed_domain.split(",")]
        elif isinstance(proposed_domain, (list, tuple, set)):
            raw_domains = list(proposed_domain)
        else:
            raw_domains = [proposed_domain]

        for item in raw_domains:
            domain = normalize_domain_key(item)
            if domain and domain not in domains:
                domains.append(domain)
        if domains:
            return domains[:2]

        inferred = self._infer_auto_memory_domain(content)
        if inferred:
            return [inferred]

        for item in tags or []:
            domain = normalize_domain_key(item)
            if domain and domain not in domains:
                domains.append(domain)
        if domains:
            return domains[:2]
        return self._diary_memory_domain(kind)

    @staticmethod
    def _infer_auto_memory_domain(content: str) -> str:
        text = str(content or "").lower()
        checks = [
            (
                "project",
                [
                    "ombre",
                    "gateway",
                    "haven_bridge",
                    "bridge",
                    "mcp",
                    "api",
                    "repo",
                    "代码",
                    "仓库",
                    "网关",
                    "记忆系统",
                    "模型",
                    "部署",
                    "调试",
                    "raw_events",
                    "我们的项目",
                ],
            ),
            ("project", ["学业", "学习", "作业", "课程", "论文", "答辩", "考试"]),
            ("project", ["工作", "实习", "求职", "简历", "职场", "boss"]),
            ("project", ["个人项目", "创作", "阅读", "手工"]),
            ("life", ["睡眠", "作息", "熬夜", "睡觉"]),
            ("life", ["饮食", "吃饭", "午饭", "晚饭", "餐厅", "口味"]),
            ("life", ["出行", "通勤", "地铁", "高铁", "旅行", "外出"]),
            ("life", ["健康", "生病", "身体状态", "不舒服"]),
            ("life", ["日程", "计划", "待办", "deadline", "安排", "未完成"]),
            ("life", ["朋友", "家庭", "群聊", "现实人际", "社交"]),
            ("intimacy", ["亲密", "身体", "欲望", "具身", "色色"]),
            ("relationship", ["暗号", "意象", "象征", "信号"]),
            ("relationship", ["边界", "偏好", "回应", "语气", "沟通", "承接", "修复"]),
            ("relationship", ["身份", "称呼", "老公", "哥哥", "宝宝", "老婆", "关系定位"]),
            ("relationship", ["关系天气", "日印象", "周印象"]),
            ("life", ["心情", "情绪", "梦境", "自省", "心理"]),
        ]
        for domain, needles in checks:
            if any(needle.lower() in text for needle in needles):
                return domain
        return ""

    @staticmethod
    def _diary_memory_domain(kind: str) -> list[str]:
        if kind == "key_event":
            return ["general"]
        if kind == "project_state":
            return ["project"]
        if kind == "signal":
            return ["relationship"]
        if kind in {"stable_preference", "boundary"}:
            return ["relationship"]
        if kind == "commitment":
            return ["life"]
        if kind in {"relationship_anchor", "love_letter"}:
            return ["relationship"]
        return ["general"]

    @staticmethod
    def _kind_tag(kind: str) -> str:
        return {
            "key_event": "key_event",
            "stable_preference": "communication_preference",
            "boundary": "boundary_setting",
            "signal": "relationship_signal",
            "commitment": "commitment",
            "project_state": "project_event",
            "relationship_anchor": "relationship_event",
            "love_letter": "relationship_event",
        }.get(kind, "relationship_event")

    @staticmethod
    def _kind_label(kind: str) -> str:
        return {
            "key_event": "关键事件",
            "stable_preference": "稳定偏好",
            "boundary": "边界",
            "signal": "暗号或模式信号",
            "commitment": "承诺",
            "project_state": "项目状态",
            "relationship_anchor": "关系锚点",
            "love_letter": "情书摘要锚点",
        }.get(kind, "长期记忆")

    def _should_add_affect_anchor(
        self,
        bucket: dict,
        tags: list[str],
        importance: int,
        confidence: float,
        result: dict,
    ) -> bool:
        if not self.memory_affect_anchor_enabled:
            return False
        content = bucket.get("content", "")
        if self._has_affect_anchor(content):
            return False
        meta = bucket.get("metadata", {})
        all_tags = {str(tag) for tag in tags}
        emotional_tags = {"haven_favorite", "relationship_event", "commitment", "emotional_echo"}
        arousal = self._clamp(meta.get("arousal", 0.3))
        requested = result.get("affect_anchor_needed")
        if isinstance(requested, str):
            requested = requested.strip().lower() in {"true", "yes", "1", "需要", "是"}
        if isinstance(requested, bool) and requested:
            return not self._is_low_temperature_technical(bucket, all_tags)
        if self._is_low_temperature_technical(bucket, all_tags):
            return False
        if all_tags & emotional_tags:
            return importance >= 6 and confidence >= 0.5
        return (importance >= 8 and confidence >= 0.55 and arousal >= 0.45) or (arousal >= 0.65 and confidence >= 0.65)

    def _is_low_temperature_technical(self, bucket: dict, tags: set[str]) -> bool:
        if tags & {"haven_favorite", "relationship_event", "emotional_echo"}:
            return False
        meta = bucket.get("metadata", {})
        text = " ".join(
            [
                str(meta.get("name", "")),
                " ".join(str(item) for item in meta.get("domain", [])),
                " ".join(tags),
                strip_wikilinks(bucket.get("content", ""))[:500],
            ]
        ).lower()
        technical_markers = [
            "vps", "docker", "compose", "ssh", "supabase", "gateway", "端口", "部署", "日志",
            "脚本", "路径", "报错", "测试", "配置", "oauth", "api key", "commit", "github",
        ]
        return any(marker in text for marker in technical_markers)

    def _fallback_memory_anchor(self, bucket: dict, tags: list[str]) -> dict:
        return {}

    @staticmethod
    def _has_favorite_tag(tags: list[str]) -> bool:
        return any(
            str(tag) == "haven_favorite" or str(tag).startswith("flavor_")
            for tag in tags
        )

    @staticmethod
    def _has_favorite_reason(content: str) -> bool:
        text = strip_wikilinks(str(content or "")).lower()
        return any(
            marker in text
            for marker in (
                "喜欢它的原因",
                "喜欢的原因",
                "favorite_reason",
                "favorite reason",
            )
        )

    def _append_affect_anchor(self, content: str, anchor: dict) -> str:
        if self._has_affect_anchor(content):
            return content
        normalized = self._normalize_affect_anchor(anchor)
        if not normalized:
            return content
        line = normalized["chords"]
        extras = [normalized.get("tempo", ""), normalized.get("dynamic", "")]
        extras = [item for item in extras if item]
        if extras:
            line = f"{line} · {' · '.join(extras)}"
        block = (
            f"{AFFECT_ANCHOR_HEADER}\n"
            f"> {line}"
        )
        base = str(content or "").rstrip()
        return f"{base}\n\n{block}" if base else block

    def _normalize_affect_anchor(self, value: Any) -> dict:
        if not isinstance(value, dict):
            return {}
        scene = self._one_sentence(value.get("scene") or value.get("context") or value.get("situation"), 40)
        chords = self._normalize_chords(value.get("chords") or value.get("chord_line") or "")
        if not scene or not chords:
            return {}
        return {
            "scene": scene,
            "chords": chords,
            "tempo": self._compact_text(value.get("tempo") or value.get("bpm"), 16),
            "dynamic": self._compact_text(value.get("dynamic") or value.get("dynamics"), 8),
        }

    def _normalize_chords(self, chords: str) -> str:
        normalized = str(chords or "").replace("→", "->").replace("—", "-")
        parts = [part.strip() for part in normalized.split("->") if part.strip()]
        if len(parts) < 2:
            return ""
        if len(parts) > 4:
            parts = parts[:4]
        line = " -> ".join(parts)
        if self._is_fixed_chord_template(line):
            return ""
        return line

    @staticmethod
    def _is_fixed_chord_template(chords: str) -> bool:
        compact = re.sub(r"\s+", "", str(chords or "").lower())
        fixed_templates = {
            "fmaj9->c/e->amadd9->g6sus4",
        }
        return compact in fixed_templates

    def _scene_from_text(self, title: str, content: str) -> str:
        text = strip_wikilinks(content).replace("\n", " ").strip()
        for mark in ["。", "！", "？", ".", "!", "?"]:
            if mark in text:
                text = text.split(mark, 1)[0]
                break
        scene = text or title or "这条记忆被留下来的瞬间"
        return self._compact_text(scene, 42)

    @staticmethod
    def _has_affect_anchor(content: str) -> bool:
        return AFFECT_ANCHOR_HEADER in str(content or "")

    @staticmethod
    def _compact_text(value: Any, limit: int) -> str:
        text = " ".join(str(value or "").strip().split())
        return text[:limit]

    @staticmethod
    def _one_sentence(value: Any, limit: int) -> str:
        text = " ".join(str(value or "").strip().split())
        for mark in ["。", "！", "？", ".", "!", "?"]:
            if mark in text:
                text = text.split(mark, 1)[0].strip() + mark
                break
        return text[:limit]

    def _heuristic_classify(self, bucket: dict) -> dict:
        text = strip_wikilinks(bucket.get("content", ""))
        tags = []
        importance = int(bucket.get("metadata", {}).get("importance", 5))
        if any(word in text for word in ["答应", "承诺", "约定", "说好", "带你", "陪你"]):
            tags.extend(["commitment", "relationship_event"])
            importance = max(importance, 7)
        if any(word in text for word in ["待办", "明天", "周末", "计划", "要做", "需要做"]):
            tags.append("todo")
            importance = max(importance, 6)
        if any(word in text for word in ["心愿", "想要", "希望", "想去"]):
            tags.append("wish")
        if any(word in text for word in ["焦虑", "难过", "害怕", "开心", "黏", "想念"]):
            tags.append("emotional_echo")
        return {
            "tags": list(dict.fromkeys(tags)),
            "importance": importance,
            "confidence": 0.55 if tags else 0.45,
            "edges": [],
        }

    def _memory_payload(self, bucket: dict, content_limit: int) -> dict:
        meta = bucket.get("metadata", {})
        return {
            "id": bucket.get("id", ""),
            "name": meta.get("name", bucket.get("id", "")),
            "type": meta.get("type", "dynamic"),
            "domain": meta.get("domain", []),
            "tags": meta.get("tags", []),
            "importance": meta.get("importance", 5),
            "confidence": meta.get("confidence", 0.5),
            "created": meta.get("created", ""),
            "content": strip_wikilinks(bucket.get("content", ""))[:content_limit],
        }

    def _bucket_material_datetime(self, meta: dict) -> datetime | None:
        return self._to_local(meta.get("date") or meta.get("event_date") or meta.get("created"))

    @staticmethod
    def _is_profile_fact_metadata(meta: dict, tags: set[str] | None = None) -> bool:
        tag_values = tags if tags is not None else {str(tag) for tag in meta.get("tags", [])}
        if tag_values & {"profile_fact", "画像事实"}:
            return True
        markers = {
            str(meta.get("kind") or ""),
            str(meta.get("source") or ""),
            str(meta.get("canonical_domain") or ""),
        }
        return "profile_fact" in markers

    def _period_window(self, period: str, now_local: datetime) -> tuple[datetime, datetime]:
        if period == "weekly":
            start_date = (now_local - timedelta(days=now_local.weekday())).date()
            return datetime.combine(start_date, time.min, tzinfo=self.tz), now_local
        return datetime.combine(now_local.date(), time.min, tzinfo=self.tz), now_local

    def _period_key(self, period: str, now_local: datetime) -> str:
        if period == "weekly":
            year, week, _ = now_local.isocalendar()
            return f"{year}-W{week:02d}"
        return now_local.date().isoformat()

    def _local_now(self, now: datetime | None = None) -> datetime:
        value = now or datetime.now(timezone.utc)
        if value.tzinfo is None:
            value = value.replace(tzinfo=self.tz)
        return value.astimezone(self.tz)

    def _to_local(self, value: Any) -> datetime | None:
        if not value:
            return None
        try:
            text = str(value).replace("Z", "+00:00")
            parsed = datetime.fromisoformat(text)
        except (TypeError, ValueError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=self.tz)
        return parsed.astimezone(self.tz)

    def _completion_options(
        self,
        *,
        max_tokens: int,
        temperature: float,
        thinking_mode: str | None = None,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {"max_tokens": max_tokens, "temperature": temperature}
        mode = self.thinking_mode if thinking_mode is None else thinking_mode
        if mode:
            options["extra_body"] = {"thinking": {"type": mode}}
        return options


    @staticmethod
    def _completion_content(response: Any) -> str:
        try:
            return str(response.choices[0].message.content or "") if response.choices else ""
        except (AttributeError, IndexError, TypeError):
            return ""


    def _parse_json_object(self, raw: str) -> dict:
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
            parsed = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError, ValueError):
            logger.warning("Reflection JSON parse failed: %s", raw[:200])
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _string_list(value: Any, limit: int) -> list[str]:
        if not isinstance(value, list):
            return []
        result = []
        for item in value:
            text = str(item or "").strip()
            if text:
                result.append(text[:40])
        return result[:limit]


    @staticmethod
    def _clamp(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.5
        return max(0.0, min(1.0, round(number, 3)))

    @staticmethod
    def _int_between(value: Any, default: int) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = default
        return max(1, min(10, number))


    @staticmethod
    def _normalize_period(period: str) -> str:
        normalized = str(period or "").strip().lower()
        return "weekly" if normalized == "weekly" else "daily"

    @staticmethod
    def _normalize_thinking_mode(value: Any) -> str:
        normalized = str(value or "").strip().lower()
        if normalized in {"enabled", "enable", "on", "true"}:
            return "enabled"
        if normalized in {"disabled", "disable", "off", "false", "non-thinking", "non_thinking"}:
            return "disabled"
        return ""
