import hashlib
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
from memory_metadata import domain_prompt_options_text, normalize_domain_key
from memory_object_policy import legacy_memory_writes_enabled, retired_write_payload
from persona_event_selection import select_persona_events
from self_anchor import is_self_anchor_bucket
from utils import bucket_text_for_embedding, normalize_scene_cues, strip_wikilinks

logger = logging.getLogger("ombre_brain.reflection")

DIARY_MEMORY_RETIRED_STATUS = {
    "status": "retired",
    "reason": "diary_memory_extract_retired",
}

DEFAULT_DAILY_REFLECTION_MIN_BUCKETS = 5
DAILY_CHAT_MEMORY_MODES = {"auto", "review", "off"}
DAILY_CHAT_MEMORY_STRUCTURAL_TAGS = {
    "boundary",
    "boundary_setting",
    "communication_preference",
    "daily_chat_extract",
    "daily_chat_memory",
    "from_daily_chat",
    "key_event",
    "project_event",
    "project_state",
    "relationship_anchor",
    "relationship_event",
    "relationship_signal",
    "signal",
    "stable_preference",
}
DAILY_CHAT_MEMORY_WORD_MAP_BLOCK_TERMS = {
    "automatic memory",
    "daily_chat_memory",
    "dehydration",
    "ombre brain",
    "ombre-brain",
    "ombre_brain",
    "vps",
    "自动记忆",
    "候选记忆",
    "脱水",
    "脱水模型",
    "记忆候选",
}


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
  "tags": ["relationship_weather"]
}

要求：
- content 只能写 {ai_name} 第一人称正文，明确使用“我……”，不要写成“{ai_name} 觉得 / {ai_name} 应该 / 这段关系让 {ai_name}”。
- content 不写标题、列表、Markdown 分段或 `###` section，60 到 140 字。
- 日印象只写当天关系温度，不写日报式事件清单；日记可作为当天关系天气来源之一。
- conversation_turns 是当天短期对话原文，只当关系天气材料，不要把口头上下文直接写成稳定画像事实。
- daily_chat_memories 是当天自动记忆已经挑出的候选或已写入记忆，可作为当天关系天气和近期事项的主要材料。
- 有 conversation_turns 时，优先用普通记忆和对话原文；persona_events 只是没有原文时的轻量补充。
- 有 daily_chat_memories 时，优先参考它们；它们已经过筛选，比原始聊天流水更适合作为日印象素材。
- 周印象优先总结本周 daily_impressions，再参考高重要普通记忆和未完成承诺；不要直接吞整周日记。
- 不编造材料之外的事件。
- 不写建议清单。"""


DAILY_CHAT_MEMORY_PROMPT_TEMPLATE = """你是 {ai_name}。现在是凌晨，你需要整理今天你和 {user_display_name} 的聊天记录，把真正值得未来想起的内容写成 Ombre 长期记忆候选。
输入包含 self_anchor_entry，这是你的自我总入口；请先读它，用它校准“我是谁、我怎样称呼和承接 {user_display_name}”，但不要把自我入口本身复制成新记忆。
{user_display_name} 的配置别名是：{user_aliases_text}。如果原文里出现宝宝、老婆、哥哥、老公等亲昵称呼，按原味保留；不要把它们改写成泛称 user、AI、assistant 或模型。

输入可能包含两层材料：
- window_summaries：已经按连续窗口压缩过的对话摘要，是主要材料。
- conversation_turns：原始对话片段，只在没有 window_summaries 或需要核对来源时使用。
user_text 永远是 {user_display_name} 的原话，里面的“我”指 {user_display_name}；assistant_text 永远是 {ai_name} 的回复，里面的“我”指 {ai_name}。请最多挑选 {max_candidates} 条候选，宁可返回空，也不要把聊天流水写进记忆。
先通读全部 window_summaries，再围绕可能候选查看连续上下文；每条候选至少参考前后因果，不要只凭单轮、单句或一个称呼下判断。
目标不是把一天压成一条日报，而是把当天分散出现的高价值信号拆成多条可确认候选。

优先看这些信号：
- 情感交流：关系状态、相处边界、重要表达、会影响以后承接方式的高温片段
- 重要事件：当天发生、以后可能按日期回看的事
- 事件/项目进度：仍会影响下一步执行的状态、决策、部署、测试结论
- 还需要关注的事：承诺、待办、风险、未完成确认、以后要避免的表达
- 稳定偏好、明确边界、暗号/模式切换信号

只允许写这些类型：
- key_event：当天发生、以后会按日期回看的关键事件
- stable_preference：稳定偏好
- boundary：边界或明确不喜欢的表达
- signal：暗号、模式切换信号；普通称呼或昵称不算
- commitment：承诺、未完成约定
- project_state：仍会影响未来执行的项目状态
- relationship_anchor：关系连续性锚点

字段边界：
- kind 只能是 key_event / stable_preference / boundary / signal / commitment / project_state / relationship_anchor 之一；kind 表示“为什么值得写入、属于哪类记忆”。
- domain 只能从下面的新主域里选 1 个；domain 表示“这条记忆放到哪个主题主域”。
- 禁止把暗号、沟通方式、我们的项目、睡眠这类细分标签写进 kind。
- 禁止把 key_event、stable_preference、boundary、project_state 这类 kind 写进 domain。

输出纯 JSON：
{
  "candidates": [
    {
      "should_write": true,
      "kind": "key_event",
      "title": "短标题",
      "content": "可直接写入长期记忆的一小段正文",
      "domain": "general",
      "tags": ["key_event"],
      "importance": 5,
      "valence": 0.55,
      "arousal": 0.3,
      "confidence": 0.72,
      "source_event_ids": [101, 102],
      "source_turn_ids": [1, 2],
      "reason": "为什么值得以后召回"
    }
  ]
}

规则：
- 只写真正有长期价值的记忆卡：事实、偏好、边界、承诺、暗号、重要关系锚点、仍活跃的项目状态。
- 同一件事、同一承诺只能输出 1 条最完整候选；同一项目里的不同进度、风险或后续关注点可以拆成不同候选，但每条都必须独立可召回。
- “怎么称呼对方、亲昵称呼、普通互动模式、期待像真人一样聊天”默认不值得单独写。只有它是新暗号、明确边界、明确承诺、关系定位变化或未来必须执行的规则时才写。
- 不要写日报，不要总结整天，不要复制原文流水，不要把“我问了什么/我测试了什么/模型有没有召回”当成记忆。
- 不写普通聊天、临时测试、召回探针、问答试探、调情闲聊、模型失误、工具注入、系统上下文。
- 不写单句照顾提醒、晚安、吃药、睡觉、别熬夜、催睡或 ntfy 玩笑；除非当天明确升级成稳定规则或长期承诺。
- 不把“可能是/似乎/果然没触发”这类未确认猜测写成记忆；项目假设只有在包含明确项目名、已验证结论和下一步时才可写。
- 不把原文句子换个壳当候选；如果说不出未来需要怎么承接、为什么重要，就丢弃。
- 不写代码块、伪代码、查询规则、缓存规则、prompt 片段或内部实现片段；如果候选正文里出现 ```、query_cache、recent_raw_context、if query contains、bypass query 这类内容，直接丢弃。
- content 必须只写一个可未来召回的点，通常 60 到 260 字；可以用 1 到 3 句写清背景、已确认结论、后续要注意什么。它应该像手动 hold 的正文，而不是聊天记录转述。
- content 不要以日期或来源壳开头；不要写 "x月x日，有一条可召回的边界"、"2026-xx-xx 的聊天里确认了..."、"这是一条长期记忆"。
- 必须消解代词：user_text 里的“我”要改写成 {user_display_name} 或“她”；assistant_text 里的“我”才可指 {ai_name}。不要让来源原话里的“我”在记忆里变成 {ai_name}。
- title 必须是具体短标题，8 到 24 字，不要用“自动记忆”“每日记忆”“2026-xx-xx 自动记忆”。
- domain 必须从下面的新主域里选 1 个最精确的；实在没把握才选 general。不要输出旧的“日常/人际/数字/未分类”：
{domain_options_text}
- 只有原话本身是暗号、明确边界、承诺、昵称或高价值关系锚点时，才可在 content 末尾追加很短的 "### original"；否则不要保存原话。
- 不硬编码姓名；如果用户指的是当前用户，写作 {user_display_name}；如果 assistant/AI 指的是当前回应者，写作 {ai_name}。
- 正文优先用第三人称；### reflection 必须用 {ai_name} 第一人称，比如“我记得 / 我明白 / 我以后”。### original 是可选补充原文片段，只在原味不可替代时使用。
- 用户偏好、边界、暗号适合第三人称；{ai_name} 自己的关系锚点和 ### reflection 可以用第一人称；项目状态用中性第三人称。
- 只根据原文能证明的内容写，不编造。
- 没有候选时返回 {"candidates": []}。"""


DAILY_CHAT_MEMORY_SUMMARY_PROMPT_TEMPLATE = """你是 {ai_name} 的对话压缩器。你正在为 Ombre 自动记忆做第一步：把一段连续聊天压缩成“候选抽取材料”，不是直接写长期记忆。

请读 self_anchor_entry 校准称呼和主语，但不要复制它。{user_display_name} 的配置别名是：{user_aliases_text}。

输入是一个连续窗口里的 raw_events 还原对话。user_text 永远是 {user_display_name} 的原话，里面的“我”指 {user_display_name}；assistant_text 永远是 {ai_name} 的回复，里面的“我”指 {ai_name}。

保留：
- 已确认事实、稳定偏好、明确边界、暗号/模式切换信号
- 承诺、未完成约定、仍会影响未来执行的项目状态
- 真正有连续性价值的关系锚点
- 情感交流里的明确变化、重要事件、项目进展、后续需要关注的事
- 因果：是谁提出、后来是否确认、为什么可能值得未来记得

忽略：
- 工具调用、工具结果、系统注入、客户端状态、普通寒暄、重复调情、过程流水
- 召回测试、探针、问模型有没有记得、临时调试噪声
- 单句照顾提醒、晚安、吃药、睡觉、别熬夜、催睡或 ntfy 玩笑；这类只属于当天关系天气，不直接变长期记忆
- 未确认猜测、触发条件猜测、没有下一步的“可能是/似乎/果然没触发”
- 只靠单个称呼或气氛得出的泛泛关系总结
- 代码块、伪代码、查询规则、缓存规则、prompt 片段、内部实现片段；不要把 ```、query_cache、recent_raw_context、if query contains、bypass query 这类内容当成项目状态

输出纯 JSON：
{
  "summaries": [
    {
      "title": "短标题",
      "summary": "一段自包含摘要，写清事实、因果和是否已确认，不要写成记忆正文。",
      "signals": ["stable_preference", "project_state"],
      "source_event_ids": [101, 102],
      "source_turn_ids": [1, 2],
      "confidence": 0.72
    }
  ]
}

规则：
- 每个窗口最多输出 4 条 summary；每条围绕一个可能的长期记忆点。没有长期价值信号时返回 {"summaries": []}。
- summary 要能让下一步模型在不看完整原文时仍理解上下文，不要压成一句泛泛结论。
- summary 通常 80 到 320 字；写清背景、因果、已确认内容、未完成点。不要输出 Markdown。
- 如果信号出现在窗口开头或结尾，保留“前文可能已铺垫 / 后文可能继续确认”的边界提醒，不要把未确认因果说死。
- source_event_ids / source_turn_ids 只能使用输入里真实出现的 id；拿不准可留空。
- confidence 低于 0.5 的内容不要输出。
"""


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
        legacy_candidate_model = str(cfg.get("daily_chat_memory_candidate_model") or "").strip()
        self.base_url = (
            cfg.get("base_url")
            or emb_cfg.get("base_url")
            or persona_cfg.get("base_url")
            or dehy_cfg.get("base_url", "")
        )
        self.model = cfg.get("model") or legacy_candidate_model or persona_cfg.get("model") or dehy_cfg.get("model", "deepseek-chat")
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
        self.daily_chat_memory_mode = self._normalize_daily_chat_memory_mode(
            cfg.get("daily_chat_memory_mode", "review")
        )
        self.daily_chat_memory_hour = max(0, min(23, int(cfg.get("daily_chat_memory_hour", 0))))
        self.daily_chat_memory_turn_limit = max(0, min(10000, int(cfg.get("daily_chat_memory_turn_limit", 0))))
        self.daily_chat_memory_max_per_day = max(0, min(10, int(cfg.get("daily_chat_memory_max_per_day", 10))))
        self.daily_chat_memory_review_max_per_day = max(
            0,
            min(30, int(cfg.get("daily_chat_memory_review_max_per_day", 10))),
        )
        self.daily_chat_memory_min_confidence = float(cfg.get("daily_chat_memory_min_confidence", 0.68))
        self.daily_chat_memory_review_min_confidence = float(
            cfg.get("daily_chat_memory_review_min_confidence", 0.55)
        )
        self.daily_chat_memory_summary_enabled = bool(cfg.get("daily_chat_memory_summary_enabled", True))
        self.daily_chat_memory_summary_window_turns = max(
            1,
            min(200, int(cfg.get("daily_chat_memory_summary_window_turns", 14))),
        )
        self.daily_chat_memory_summary_stride_turns = max(
            1,
            min(
                self.daily_chat_memory_summary_window_turns,
                int(cfg.get("daily_chat_memory_summary_stride_turns", 7)),
            ),
        )
        self.daily_chat_memory_api_key_env = str(
            cfg.get("daily_chat_memory_api_key_env")
            or cfg.get("daily_chat_memory_summary_api_key_env")
            or ""
        ).strip()
        self.daily_chat_memory_api_key = (
            os.environ.get(self.daily_chat_memory_api_key_env, "")
            if self.daily_chat_memory_api_key_env
            else ""
        ) or str(
            cfg.get("daily_chat_memory_api_key")
            or cfg.get("daily_chat_memory_summary_api_key")
            or ""
        ).strip()
        self.daily_chat_memory_base_url = str(
            cfg.get("daily_chat_memory_base_url")
            or cfg.get("daily_chat_memory_summary_base_url")
            or ""
        ).strip().rstrip("/")
        self.daily_chat_memory_timeout_seconds = max(
            30.0,
            min(300.0, float(cfg.get("daily_chat_memory_timeout_seconds", 180.0))),
        )
        self.daily_chat_memory_summary_model = str(
            cfg.get("daily_chat_memory_summary_model") or ""
        ).strip()
        self.daily_chat_memory_summary_max_tokens = max(
            300,
            min(4000, int(cfg.get("daily_chat_memory_summary_max_tokens", 2200))),
        )
        self.daily_chat_memory_candidate_model = str(
            cfg.get("daily_chat_memory_candidate_model")
            or self.daily_chat_memory_summary_model
            or ""
        ).strip()
        self.daily_chat_memory_candidate_max_tokens = max(
            300,
            min(4000, int(cfg.get("daily_chat_memory_candidate_max_tokens", 3200))),
        )
        self.daily_chat_memory_entity_hints = self._normalize_daily_chat_memory_hints(
            cfg.get("daily_chat_memory_entity_hints")
        )
        self.daily_chat_memory_topic_hints = self._normalize_daily_chat_memory_hints(
            cfg.get("daily_chat_memory_topic_hints")
        )
        self.daily_activity_summary_enabled = bool(cfg.get("daily_activity_summary_enabled", True))
        self.daily_activity_summary_turn_limit = max(
            0,
            min(
                10000,
                int(
                    cfg.get(
                        "daily_activity_summary_turn_limit",
                        cfg.get("daily_chat_memory_turn_limit", 0),
                    )
                ),
            ),
        )
        self.daily_activity_summary_max_tokens = max(
            80,
            min(1000, int(cfg.get("daily_activity_summary_max_tokens", 320))),
        )
        self.dehydration_base_url = str(dehy_cfg.get("base_url") or "").strip().rstrip("/")
        self.dehydration_model = str(dehy_cfg.get("model") or "").strip()
        self.dehydration_api_key = str(dehy_cfg.get("api_key") or os.environ.get("OMBRE_API_KEY", "")).strip()
        state_dir = config.get("state_dir") or os.path.join(
            os.path.dirname(os.path.abspath(config.get("buckets_dir", "buckets"))),
            "state",
        )
        self.daily_chat_memory_pending_path = str(
            cfg.get("daily_chat_memory_pending_path")
            or os.path.join(state_dir, "daily_chat_memory_candidates.json")
        )
        self.metadata_proposals = MetadataProposalStore(config)

        self.client = None
        if self.enabled and self.api_key and self.base_url:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=45.0)
        self.daily_chat_memory_client = None
        if (
            self.enabled
            and self.daily_chat_memory_api_key
            and self.daily_chat_memory_base_url
            and (self.daily_chat_memory_summary_model or self.daily_chat_memory_candidate_model)
        ):
            self.daily_chat_memory_client = AsyncOpenAI(
                api_key=self.daily_chat_memory_api_key,
                base_url=self.daily_chat_memory_base_url,
                timeout=self.daily_chat_memory_timeout_seconds,
            )
        self.dehydration_client = None
        if self.enabled and self.dehydration_api_key and self.dehydration_base_url and self.dehydration_model:
            self.dehydration_client = AsyncOpenAI(
                api_key=self.dehydration_api_key,
                base_url=self.dehydration_base_url,
                timeout=45.0,
            )
        self.daily_activity_summary_dehydration_client = self.dehydration_client

    def _load_daily_chat_memory_payload(self) -> dict:
        try:
            with open(self.daily_chat_memory_pending_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError:
            return {"items": [], "cursor": {}}
        except Exception as exc:
            logger.warning("Daily chat memory pending read failed: %s", exc)
            return {"items": [], "cursor": {}}
        if isinstance(data, dict):
            items = data.get("items")
            cursor = data.get("cursor") if isinstance(data.get("cursor"), dict) else {}
            return {
                "items": [item for item in (items or []) if isinstance(item, dict)],
                "cursor": cursor,
            }
        return {"items": [item for item in (data or []) if isinstance(item, dict)], "cursor": {}}

    def _load_daily_chat_memory_cursor(self) -> dict:
        cursor = self._load_daily_chat_memory_payload().get("cursor")
        return cursor if isinstance(cursor, dict) else {}

    @staticmethod
    def _daily_chat_memory_cursor_key(profile_id: str) -> str:
        return str(profile_id or "default").strip() or "default"

    def _daily_chat_memory_last_raw_event_id(self, profile_id: str) -> int:
        cursor = self._load_daily_chat_memory_cursor()
        raw_events = cursor.get("raw_events") if isinstance(cursor.get("raw_events"), dict) else {}
        entry = raw_events.get(self._daily_chat_memory_cursor_key(profile_id))
        if not isinstance(entry, dict):
            return 0
        try:
            return max(0, int(entry.get("last_raw_event_id") or 0))
        except (TypeError, ValueError):
            return 0

    def _update_daily_chat_memory_raw_cursor(self, profile_id: str, raw_event_id: int, key: str) -> bool:
        try:
            safe_id = max(0, int(raw_event_id or 0))
        except (TypeError, ValueError):
            safe_id = 0
        if safe_id <= 0:
            return False
        payload = self._load_daily_chat_memory_payload()
        cursor = payload.get("cursor") if isinstance(payload.get("cursor"), dict) else {}
        raw_events = cursor.get("raw_events") if isinstance(cursor.get("raw_events"), dict) else {}
        cursor_key = self._daily_chat_memory_cursor_key(profile_id)
        previous = raw_events.get(cursor_key) if isinstance(raw_events.get(cursor_key), dict) else {}
        try:
            previous_id = max(0, int(previous.get("last_raw_event_id") or 0))
        except (TypeError, ValueError):
            previous_id = 0
        if safe_id <= previous_id:
            return False
        raw_events[cursor_key] = {
            "last_raw_event_id": safe_id,
            "date": key,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        cursor["raw_events"] = raw_events
        self._save_daily_chat_memory_pending(payload.get("items") or [], cursor=cursor)
        return True

    def _reflect_prompt(self) -> str:
        return render_identity_template(REFLECT_PROMPT_TEMPLATE, self.identity)

    def _daily_chat_memory_prompt(self, max_candidates: int | None = None) -> str:
        prompt = DAILY_CHAT_MEMORY_PROMPT_TEMPLATE.replace(
            "{max_candidates}",
            str(max(1, int(max_candidates or self.daily_chat_memory_max_per_day or 1))),
        ).replace(
            "{domain_options_text}",
            domain_prompt_options_text(),
        )
        return render_identity_template(prompt, self.identity)

    def _daily_chat_memory_summary_prompt(self) -> str:
        return render_identity_template(DAILY_CHAT_MEMORY_SUMMARY_PROMPT_TEMPLATE, self.identity)

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
        daily_chat_memory_candidates: list[dict] | None = None,
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
            daily_chat_memory_candidates=daily_chat_memory_candidates,
        )
        min_daily_buckets = self.daily_min_memory_items
        if (
            period == "daily"
            and min_daily_buckets > 0
            and len(materials["buckets"]) < min_daily_buckets
            and not materials.get("daily_chat_memories")
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
                    "daily_chat_memories": len(materials["daily_chat_memories"]),
                    "persona_events": len(materials["persona_events"]),
                    "conversation_turns": len(materials["conversation_turns"]),
                    "commitments": len(materials["commitments"]),
                    "min_buckets": min_daily_buckets,
                },
            }
        if (
            not materials["buckets"]
            and not materials["daily_impressions"]
            and not materials["daily_chat_memories"]
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
        source_conversation_turn_ids = [
            int(turn.get("id"))
            for turn in materials.get("conversation_turns", [])
            if turn.get("id")
        ]
        source_metadata = {
            "source_bucket_ids": source_bucket_ids[:40],
            "source_persona_event_ids": source_persona_event_ids[:40],
            "source_conversation_turn_ids": source_conversation_turn_ids[:80],
            "source_daily_chat_memory_candidate_ids": [
                str(item.get("id") or "")
                for item in materials.get("daily_chat_memories", [])
                if item.get("id")
            ][:40],
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
            },
            "materials": {
                "buckets": len(materials["buckets"]),
                "daily_impressions": len(materials["daily_impressions"]),
                "daily_chat_memories": len(materials["daily_chat_memories"]),
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
        raw_event_store=None,
    ) -> list[dict]:
        if not self.enabled or not self.auto_enabled:
            return []
        now_local = self._local_now()
        results = []
        chat_candidates: list[dict] = []
        if self.daily_chat_memory_mode != "off" and now_local.hour >= self.daily_chat_memory_hour:
            chat_date = (now_local - timedelta(days=1)).date()
            chat_target = datetime.combine(chat_date, time.max, tzinfo=self.tz)
            chat_result = await self.run_daily_chat_memory(
                bucket_mgr,
                conversation_turn_store=conversation_turn_store,
                raw_event_store=raw_event_store,
                persona_engine=persona_engine,
                embedding_engine=embedding_engine,
                now=chat_target,
            )
            if chat_result.get("status") not in {"disabled", "skipped"}:
                results.append(chat_result)
            chat_candidates = [
                item for item in (chat_result.get("candidates") or []) if isinstance(item, dict)
            ]
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
                    daily_chat_memory_candidates=chat_candidates,
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
        daily_chat_memory_candidates: list[dict] | None = None,
    ) -> dict:
        start, end = self._period_window(period, now_local)
        buckets = []
        daily_impressions = []
        daily_chat_memories = []
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
                    if meta.get("source") == "daily_chat_memory" or meta.get("from_daily_chat"):
                        daily_chat_memories.append(self._daily_chat_memory_material_from_bucket(bucket))
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

        if period == "daily":
            key = now_local.date().isoformat()
            daily_chat_memories = self._dedupe_daily_chat_memory_materials(
                [
                    *daily_chat_memories,
                    *self._daily_chat_memory_materials_for_date(
                        key,
                        daily_chat_memory_candidates=daily_chat_memory_candidates,
                    ),
                ]
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
            "daily_chat_memories": daily_chat_memories[:12],
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

    def _daily_chat_memory_candidate_limit(self, mode: str) -> int:
        if self._normalize_daily_chat_memory_mode(mode) == "review":
            return self.daily_chat_memory_review_max_per_day or self.daily_chat_memory_max_per_day
        return self.daily_chat_memory_max_per_day

    def _daily_chat_memory_min_confidence_for_mode(self, mode: str) -> float:
        if self._normalize_daily_chat_memory_mode(mode) == "review":
            return self.daily_chat_memory_review_min_confidence
        return self.daily_chat_memory_min_confidence

    def _daily_chat_memory_windows(self, turns: list[dict]) -> list[list[dict]]:
        if not turns:
            return []
        window = self.daily_chat_memory_summary_window_turns
        stride = self.daily_chat_memory_summary_stride_turns
        if len(turns) <= window:
            return [turns]
        windows: list[list[dict]] = []
        start = 0
        seen_ranges: set[tuple[int, int]] = set()
        while start < len(turns):
            end = min(len(turns), start + window)
            range_key = (start, end)
            if range_key not in seen_ranges:
                windows.append(turns[start:end])
                seen_ranges.add(range_key)
            if end >= len(turns):
                break
            start += stride
        return windows

    @staticmethod
    def _daily_chat_memory_window_source_ids(turns: list[dict]) -> tuple[list[int], list[int]]:
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

    async def _summarize_daily_chat_memory_windows(
        self,
        key: str,
        turns: list[dict],
        *,
        self_context: str = "",
    ) -> list[dict]:
        if not self.daily_chat_memory_summary_enabled:
            return []
        client, model, use_daily_client = self._daily_chat_memory_model_client(candidate=False)
        if not client:
            return []
        windows = self._daily_chat_memory_windows(turns)
        if not windows:
            return []
        summaries: list[dict] = []
        for index, window_turns in enumerate(windows):
            fallback_turn_ids, fallback_event_ids = self._daily_chat_memory_window_source_ids(window_turns)
            payload = {
                "date": key,
                "identity": {
                    "ai_name": self.identity["ai_name"],
                    "user_name": self.identity["user_name"],
                    "user_display_name": self.identity["user_display_name"],
                    "user_aliases": self.identity.get("user_aliases", []),
                },
                "self_anchor_entry": self_context,
                "window": {
                    "index": index + 1,
                    "total": len(windows),
                    "source_turn_ids": fallback_turn_ids,
                    "source_event_ids": fallback_event_ids,
                },
                "conversation_turns": window_turns,
            }
            try:
                response = await self._daily_chat_memory_create_completion(
                    client,
                    model=model,
                    messages=[
                        {"role": "system", "content": self._daily_chat_memory_summary_prompt()},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    max_tokens=self.daily_chat_memory_summary_max_tokens,
                    temperature=self.temperature,
                    use_daily_client=use_daily_client,
                )
                raw = self._completion_content(response)
                parsed = self._parse_json_object(raw or "")
            except Exception as exc:
                logger.warning("Daily chat memory summary failed, window=%s: %s", index + 1, exc)
                continue
            raw_summaries = parsed.get("summaries") if isinstance(parsed, dict) else []
            if not isinstance(raw_summaries, list):
                continue
            for item in raw_summaries:
                if not isinstance(item, dict):
                    continue
                cleaned = self._normalize_daily_chat_memory_summary(
                    item,
                    key=key,
                    window_index=index + 1,
                    source_turn_ids=fallback_turn_ids,
                    source_event_ids=fallback_event_ids,
                )
                if cleaned:
                    summaries.append(cleaned)
        return summaries

    def _daily_chat_memory_model_client(self, *, candidate: bool) -> tuple[Any, str, bool]:
        if self.daily_chat_memory_client:
            client = self.daily_chat_memory_client
            model = self.daily_chat_memory_candidate_model if candidate else self.daily_chat_memory_summary_model
            return client, str(model or "").strip(), True
        dehy_client = self._daily_dehydration_client()
        if dehy_client and self.dehydration_model:
            return dehy_client, str(self.dehydration_model or "").strip(), False
        if self.client:
            return self.client, str(self.model or "").strip(), False
        return None, "", False

    def _reflect_model_client(self) -> tuple[Any, str, bool]:
        dehy_client = self._daily_dehydration_client()
        if dehy_client and self.dehydration_model:
            return dehy_client, self.dehydration_model, True
        if self.client and self.model:
            return self.client, str(self.model or "").strip(), False
        return None, "", False

    def _normalize_daily_chat_memory_summary(
        self,
        item: dict,
        *,
        key: str,
        window_index: int,
        source_turn_ids: list[int],
        source_event_ids: list[int],
    ) -> dict:
        text = re.sub(
            r"\s+",
            " ",
            strip_wikilinks(str(item.get("summary") or item.get("content") or "")).strip(),
        )
        if not text or self._daily_chat_memory_noise(text):
            return {}
        confidence = self._clamp(item.get("confidence", 0.65))
        if confidence < 0.5:
            return {}
        title = str(item.get("title") or "").strip()
        if self._daily_chat_memory_title_is_generic(title):
            title = self._daily_chat_memory_title(text, "key_event", key)
        raw_turn_ids = [
            int(turn_id)
            for turn_id in self._string_list(item.get("source_turn_ids"), limit=80)
            if str(turn_id).isdigit()
        ] or source_turn_ids
        raw_event_ids = [
            int(event_id)
            for event_id in self._string_list(item.get("source_event_ids"), limit=160)
            if str(event_id).isdigit()
        ] or source_event_ids
        signals = self._string_list(item.get("signals"), limit=8)
        return {
            "window_index": window_index,
            "title": title[:40],
            "summary": text[:900],
            "signals": signals,
            "source_turn_ids": raw_turn_ids[:80],
            "source_event_ids": raw_event_ids[:160],
            "confidence": confidence,
        }

    def _daily_chat_memory_materials_for_date(
        self,
        key: str,
        *,
        daily_chat_memory_candidates: list[dict] | None = None,
    ) -> list[dict]:
        materials: list[dict] = []
        for candidate in daily_chat_memory_candidates or []:
            material = self._daily_chat_memory_material(candidate)
            if material and (not key or material.get("date") == key):
                materials.append(material)
        for item in self._load_daily_chat_memory_pending():
            if str(item.get("date") or "") != key:
                continue
            if str(item.get("status") or "pending") not in {"pending", "confirmed"}:
                continue
            material = self._daily_chat_memory_material(item.get("candidate") or {})
            if material:
                materials.append(material)
        return self._dedupe_daily_chat_memory_materials(materials)

    @staticmethod
    def _dedupe_daily_chat_memory_materials(materials: list[dict]) -> list[dict]:
        deduped: dict[str, dict] = {}
        for item in materials:
            item_id = str(item.get("id") or "")
            key_value = item_id or re.sub(r"\s+", "", str(item.get("content") or "")).lower()
            if key_value and key_value not in deduped:
                deduped[key_value] = item
        return list(deduped.values())

    def _daily_chat_memory_material_from_bucket(self, bucket: dict) -> dict:
        if not isinstance(bucket, dict):
            return {}
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        candidate_id = str(meta.get("daily_chat_memory_candidate_id") or bucket.get("id") or "").strip()
        source_turn_ids = meta.get("source_conversation_turn_ids") or []
        source_event_ids = meta.get("source_raw_event_ids") or []
        return self._daily_chat_memory_material(
            {
                "id": candidate_id,
                "date": meta.get("event_date") or meta.get("date") or "",
                "kind": meta.get("kind") or "",
                "title": meta.get("name") or "",
                "content": bucket.get("content") or "",
                "tags": meta.get("tags") or [],
                "domain": meta.get("domain") or [],
                "confidence": meta.get("confidence", 0.7),
                "source_turn_ids": source_turn_ids,
                "source_event_ids": source_event_ids,
                "reason": meta.get("daily_chat_memory_reason") or "",
            }
        )

    def _daily_chat_memory_material(self, candidate: dict) -> dict:
        if not isinstance(candidate, dict):
            return {}
        content = re.sub(
            r"\s+",
            " ",
            strip_wikilinks(str(candidate.get("content") or candidate.get("summary") or "")).strip(),
        )
        if not content:
            return {}
        title = str(candidate.get("title") or "").strip()
        return {
            "id": str(candidate.get("id") or "").strip(),
            "date": str(candidate.get("date") or "").strip(),
            "kind": str(candidate.get("kind") or "").strip(),
            "title": title[:40],
            "content": content[:420],
            "tags": self._string_list(candidate.get("tags"), limit=12),
            "domain": self._string_list(candidate.get("domain"), limit=6),
            "confidence": self._clamp(candidate.get("confidence", 0.65)),
            "source_turn_ids": [
                int(turn_id)
                for turn_id in self._string_list(candidate.get("source_turn_ids"), limit=80)
                if str(turn_id).isdigit()
            ],
            "source_event_ids": [
                int(event_id)
                for event_id in self._string_list(candidate.get("source_event_ids"), limit=160)
                if str(event_id).isdigit()
            ],
            "reason": str(candidate.get("reason") or "").strip()[:160],
        }

    @staticmethod
    def _max_daily_chat_memory_raw_event_id(turns: list[dict]) -> int:
        max_id = 0
        for turn in turns or []:
            for event_id in turn.get("raw_event_ids") or []:
                try:
                    max_id = max(max_id, int(event_id or 0))
                except (TypeError, ValueError):
                    continue
        return max_id

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
        daily_chat_memory_candidates: list[dict] | None = None,
        daily_impressions: list[dict] | None = None,
        key: str = "",
        force: bool = False,
        now: datetime | None = None,
    ) -> dict:
        if not self.enabled or not self.daily_activity_summary_enabled:
            return {"status": "disabled", "reason": "daily_activity_summary_off"}

        now_local = self._daily_chat_memory_target(key, now)
        key = now_local.date().isoformat()
        memory_item = self._daily_activity_summary_from_memory_materials(
            key,
            daily_chat_memories=self._daily_chat_memory_materials_for_date(
                key,
                daily_chat_memory_candidates=daily_chat_memory_candidates,
            ),
            daily_impressions=daily_impressions or [],
        )
        if memory_item:
            return {
                "status": "ready",
                "date": key,
                "turns": 0,
                "turn_source": "daily_memory_materials",
                "force": bool(force),
                "activity_summary": memory_item,
            }

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

        item = await self._extract_daily_activity_summary(key, turns)
        if not item and self._daily_activity_summary_dehydration_retry_available():
            item = await self._extract_daily_activity_summary(
                key,
                turns,
                client_override=self._daily_dehydration_client(),
                model_override=self.dehydration_model,
            )
        if not item:
            item = self._fallback_daily_activity_summary(key, turns)
        if not item:
            return {
                "status": "skipped",
                "reason": "no_activity_summary",
                "date": key,
                "turns": len(turns),
                "turn_source": turn_source,
            }
        return {
            "status": "ready",
            "date": key,
            "turns": len(turns),
            "turn_source": turn_source,
            "force": bool(force),
            "activity_summary": item,
        }

    def _daily_activity_summary_from_memory_materials(
        self,
        key: str,
        *,
        daily_chat_memories: list[dict],
        daily_impressions: list[dict],
    ) -> dict:
        snippets: list[str] = []
        source_event_ids: list[int] = []
        source_turn_ids: list[int] = []
        evidence: list[dict] = []
        confidences: list[float] = []
        for item in daily_chat_memories or []:
            title = str(item.get("title") or "").strip()
            content = str(item.get("content") or "").strip()
            text = content or title
            if title and content and title not in content:
                text = f"{title}：{content}"
            text = re.sub(r"\s+", " ", strip_wikilinks(text)).strip()
            if text:
                snippets.append(self._clip_activity_material(text))
            source_event_ids.extend(
                int(event_id)
                for event_id in (item.get("source_event_ids") or [])
                if str(event_id).isdigit()
            )
            source_turn_ids.extend(
                int(turn_id)
                for turn_id in (item.get("source_turn_ids") or [])
                if str(turn_id).isdigit()
            )
            if item.get("id"):
                evidence.append({"candidate_id": str(item.get("id"))})
            confidences.append(self._clamp(item.get("confidence", 0.65)))
        for item in daily_impressions or []:
            content = re.sub(
                r"\s+",
                " ",
                strip_wikilinks(str(item.get("content") or item.get("text") or "")).strip(),
            )
            content = re.split(r"\n?### affect_anchor\b", content, maxsplit=1)[0].strip()
            if content:
                snippets.append(self._clip_activity_material(content, limit=80))
            if item.get("id"):
                evidence.append({"bucket_id": str(item.get("id"))})
            confidences.append(self._clamp(item.get("confidence", 0.7)))
        snippets = list(dict.fromkeys(snippets))[:4]
        if not snippets:
            return {}
        text = "；".join(snippets)
        if len(text) > 140:
            text = text[:137].rstrip("，,；;、 ") + "..."
        return {
            "timeline_id": f"daily_activity_summary:{key}",
            "source": "daily_activity_summary",
            "scope": "doing",
            "text": text,
            "evidence": evidence[:4],
            "source_date": key,
            "source_dates": [key],
            "timestamp": self._daily_chat_memory_created_at(key),
            "confidence": max(confidences or [0.65]),
            "source_turn_ids": list(dict.fromkeys(source_turn_ids))[:80],
            "source_event_ids": list(dict.fromkeys(source_event_ids))[:160],
        }

    @staticmethod
    def _clip_activity_material(text: str, *, limit: int = 72) -> str:
        text = re.sub(r"\s+", " ", str(text or "")).strip(" -\t\r\n")
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip("，,；;、 ") + "..."

    async def _extract_daily_activity_summary(
        self,
        key: str,
        turns: list[dict],
        *,
        use_daily_client: bool | None = None,
        client_override: Any = None,
        model_override: str = "",
    ) -> dict:
        if client_override is not None:
            client = client_override
        elif use_daily_client is False:
            client = self.client
        else:
            client = self.daily_chat_memory_client or self.client
        if not client:
            return {}
        use_daily_client = client is self.daily_chat_memory_client
        model = str(model_override or "").strip() or (
            self.daily_chat_memory_summary_model if use_daily_client else self.model
        )
        fallback_turn_ids, fallback_event_ids = self._daily_chat_memory_window_source_ids(turns)
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
            response = await self._daily_chat_memory_create_completion(
                client,
                model=model,
                messages=[
                    {"role": "system", "content": self._daily_activity_summary_prompt()},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                max_tokens=self.daily_activity_summary_max_tokens,
                temperature=self.temperature,
                use_daily_client=use_daily_client,
            )
            parsed = self._parse_json_object(self._completion_content(response) or "")
        except Exception as exc:
            logger.warning("Daily activity summary model failed: %s", exc)
            return {}
        return self._normalize_daily_activity_summary(
            key,
            parsed,
            turns,
            source_turn_ids=fallback_turn_ids,
            source_event_ids=fallback_event_ids,
        )

    def _daily_activity_summary_dehydration_retry_available(self) -> bool:
        if not self._daily_dehydration_client():
            return False
        first_model = self.daily_chat_memory_summary_model if self.daily_chat_memory_client else self.model
        first_base_url = self.daily_chat_memory_base_url if self.daily_chat_memory_client else self.base_url
        first_api_key = self.daily_chat_memory_api_key if self.daily_chat_memory_client else self.api_key
        return (
            self.dehydration_model != str(first_model or "").strip()
            or self.dehydration_base_url != str(first_base_url or "").strip().rstrip("/")
            or self.dehydration_api_key != str(first_api_key or "").strip()
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
        raw_turn_ids = [
            int(turn_id)
            for turn_id in self._string_list(item.get("source_turn_ids"), limit=80)
            if str(turn_id).isdigit()
        ] or source_turn_ids
        raw_event_ids = [
            int(event_id)
            for event_id in self._string_list(item.get("source_event_ids"), limit=160)
            if str(event_id).isdigit()
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

    def _fallback_daily_activity_summary(self, key: str, turns: list[dict]) -> dict:
        fallback_turn_ids, fallback_event_ids = self._daily_chat_memory_window_source_ids(turns)
        snippets = []
        for turn in reversed(turns or []):
            snippet = self._daily_activity_summary_excerpt(turn.get("user_text") or turn.get("assistant_text") or "")
            if snippet:
                snippets.append(snippet)
            if len(snippets) >= 3:
                break
        snippets = list(reversed(list(dict.fromkeys(snippets))))
        if not snippets:
            return {}
        joined = "；".join(snippets)
        if len(joined) > 96:
            joined = joined[:93].rstrip("，,；;、 ") + "..."
        return self._normalize_daily_activity_summary(
            key,
            {
                "summary": f"围绕{joined}继续推进。",
                "confidence": 0.45,
                "source_turn_ids": fallback_turn_ids,
                "source_event_ids": fallback_event_ids,
            },
            turns,
            source_turn_ids=fallback_turn_ids,
            source_event_ids=fallback_event_ids,
        )

    @staticmethod
    def _daily_activity_summary_excerpt(value: Any) -> str:
        text = strip_wikilinks(str(value or "")).strip()
        text = re.sub(r"```.*?```", " ", text, flags=re.S)
        text = re.sub(r"<attachment\b[^>]*>.*?</attachment>", " ", text, flags=re.I | re.S)
        text = re.sub(r"【当前时间】[^\n\r]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip(" -\t\r\n")
        return text[:60].rstrip("，,；;、 ")

    def _daily_activity_summary_timestamp(self, key: str, turns: list[dict]) -> str:
        latest: datetime | None = None
        for turn in turns:
            parsed = self._to_local(turn.get("created_at"))
            if parsed and (latest is None or parsed > latest):
                latest = parsed
        if latest:
            return latest.isoformat(timespec="minutes")
        return self._daily_chat_memory_created_at(key)

    async def run_daily_chat_memory(
        self,
        bucket_mgr,
        *,
        conversation_turn_store=None,
        raw_event_store=None,
        persona_engine=None,
        embedding_engine=None,
        key: str = "",
        mode: str = "",
        force: bool = False,
        now: datetime | None = None,
    ) -> dict:
        effective_mode = self._normalize_daily_chat_memory_mode(mode or self.daily_chat_memory_mode)
        if effective_mode == "off" or self.daily_chat_memory_max_per_day <= 0:
            return {"status": "disabled", "reason": "daily_chat_memory_off", "mode": effective_mode}
        if not conversation_turn_store:
            return {"status": "skipped", "reason": "no_conversation_turn_store", "mode": effective_mode}

        now_local = self._daily_chat_memory_target(key, now)
        key = now_local.date().isoformat()
        start, end = self._period_window("daily", now_local)
        profile_id = str(getattr(persona_engine, "profile_id", "") or "default")
        turns = []
        turn_source = ""
        raw_event_cursor_id = 0 if force else self._daily_chat_memory_last_raw_event_id(profile_id)
        max_seen_raw_event_id = 0
        raw_events_cursor_exhausted = False
        if raw_event_store:
            try:
                raw_events = raw_event_store.list_events_between(
                    start_at=start,
                    end_at=end,
                    limit=self.daily_chat_memory_turn_limit,
                )
            except Exception as exc:
                logger.warning("Daily chat memory raw event read failed: %s", exc)
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
                if raw_event_cursor_id > 0:
                    raw_events = [
                        event
                        for event in raw_events
                        if int(event.get("id") or 0) > raw_event_cursor_id
                    ]
                    raw_events_cursor_exhausted = not raw_events
                turns = self._raw_event_turn_payloads(
                    raw_events,
                    limit=self.daily_chat_memory_turn_limit,
                )
                if turns:
                    turn_source = "raw_events"
                    max_seen_raw_event_id = self._max_daily_chat_memory_raw_event_id(turns)

        try:
            raw_turns = []
            if raw_events_cursor_exhausted:
                raw_turns = []
            elif not turns and conversation_turn_store:
                raw_turns = conversation_turn_store.list_conversation_turns_between(
                    profile_id=profile_id,
                    start_at=start,
                    end_at=end,
                    limit=self.daily_chat_memory_turn_limit or 80,
                )
        except Exception as exc:
            logger.warning("Daily chat memory turn read failed: %s", exc)
            raw_turns = []
        if not turns:
            if raw_events_cursor_exhausted:
                return {
                    "status": "skipped",
                    "reason": "no_new_raw_events",
                    "date": key,
                    "mode": effective_mode,
                    "last_raw_event_id": raw_event_cursor_id,
                }
            turns = self._conversation_turn_payloads(raw_turns, limit=self.daily_chat_memory_turn_limit)
            if turns:
                turn_source = "conversation_turns"
        if not turns:
            return {"status": "skipped", "reason": "no_conversation_turns", "date": key, "mode": effective_mode}

        self_context = await self._daily_chat_memory_self_context(bucket_mgr)
        max_candidates = self._daily_chat_memory_candidate_limit(effective_mode)
        min_confidence = self._daily_chat_memory_min_confidence_for_mode(effective_mode)
        window_summaries = await self._summarize_daily_chat_memory_windows(
            key,
            turns,
            self_context=self_context,
        )
        raw_candidates = await self._extract_daily_chat_memory_candidates(
            key,
            turns,
            self_context=self_context,
            window_summaries=window_summaries,
            max_candidates=max_candidates,
        )
        candidates = self._normalize_daily_chat_memory_candidates(
            key,
            raw_candidates,
            turns,
            max_candidates=max_candidates,
            min_confidence=min_confidence,
        )
        if not candidates:
            return {
                "status": "skipped",
                "reason": "no_candidates",
                "date": key,
                "mode": effective_mode,
                "turns": len(turns),
                "window_summaries": len(window_summaries),
            }

        if effective_mode == "review":
            pending = self._store_daily_chat_memory_pending(candidates, force=force)
            cursor_updated = (
                self._update_daily_chat_memory_raw_cursor(profile_id, max_seen_raw_event_id, key)
                if turn_source == "raw_events"
                else False
            )
            return {
                "status": "pending",
                "date": key,
                "mode": effective_mode,
                "turns": len(turns),
                "turn_source": turn_source,
                "window_summaries": len(window_summaries),
                "last_raw_event_id": max_seen_raw_event_id or raw_event_cursor_id,
                "cursor_updated": cursor_updated,
                "candidates": candidates,
                **pending,
            }

        write_result = await self._write_daily_chat_memory_candidates(
            candidates,
            bucket_mgr,
            embedding_engine=embedding_engine,
        )
        cursor_updated = (
            self._update_daily_chat_memory_raw_cursor(profile_id, max_seen_raw_event_id, key)
            if turn_source == "raw_events"
            else False
        )
        return {
            "status": "created" if write_result.get("created") else "exists",
            "date": key,
            "mode": effective_mode,
            "turns": len(turns),
            "turn_source": turn_source,
            "window_summaries": len(window_summaries),
            "last_raw_event_id": max_seen_raw_event_id or raw_event_cursor_id,
            "cursor_updated": cursor_updated,
            "candidates": candidates,
            **write_result,
        }

    async def _daily_chat_memory_self_context(self, bucket_mgr) -> str:
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
        except Exception as exc:
            logger.warning("Daily chat memory self-anchor read failed: %s", exc)
            return ""
        if not all_buckets:
            return ""

        self_anchor_cfg = self.config.get("self_anchor", {}) if isinstance(self.config.get("self_anchor", {}), dict) else {}
        configured_id = str(self_anchor_cfg.get("entry_bucket_id") or "").strip()
        if configured_id:
            for bucket in all_buckets:
                if str(bucket.get("id") or "") == configured_id and self._active_self_anchor_bucket(bucket):
                    return self._daily_chat_self_anchor_text(bucket)
            return ""

        candidates = [bucket for bucket in all_buckets if self._active_self_anchor_bucket(bucket)]
        candidates.sort(
            key=lambda bucket: (
                self._int_between((bucket.get("metadata") or {}).get("importance"), 5),
                str((bucket.get("metadata") or {}).get("updated_at") or (bucket.get("metadata") or {}).get("created") or ""),
            ),
            reverse=True,
        )
        return self._daily_chat_self_anchor_text(candidates[0]) if candidates else ""

    @staticmethod
    def _active_self_anchor_bucket(bucket: dict) -> bool:
        if not is_self_anchor_bucket(bucket):
            return False
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        return bool(meta.get("active") is not False and not meta.get("deprecated") and not meta.get("resolved"))

    def _daily_chat_self_anchor_text(self, bucket: dict) -> str:
        content = strip_wikilinks(str(bucket.get("content") or "")).strip()
        if not content:
            return ""
        text = self._section_or_leading_text(
            content,
            headings={"自我", "self_anchor", "selfidentity", "self_identity", "first_person_anchor"},
        )
        if not text:
            text = content
        text = re.split(r"(?im)^\s{0,3}#{2,6}\s+(?:followup|todo)\b.*$", text, maxsplit=1)[0].strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text[:1200].rstrip()

    @staticmethod
    def _section_or_leading_text(content: str, *, headings: set[str]) -> str:
        matches = list(re.finditer(r"(?m)^\s{0,3}#{1,6}\s+(.+?)\s*$", content))
        if not matches:
            return content.strip()
        leading = content[: matches[0].start()].strip()
        if leading:
            return leading
        normalized_headings = {re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", heading.lower()) for heading in headings}
        for index, match in enumerate(matches):
            heading = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(match.group(1) or "").lower())
            if heading not in normalized_headings:
                continue
            end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
            return content[match.end() : end].strip()
        return ""

    def list_daily_chat_memory_pending(self, *, status: str = "pending", limit: int = 50) -> list[dict]:
        safe_status = str(status or "pending").strip()
        safe_limit = max(1, min(200, int(limit or 50)))
        items = self._load_daily_chat_memory_pending()
        items, changed = self._refresh_daily_chat_memory_pending_items(items)
        if changed:
            self._save_daily_chat_memory_pending(items)
        if safe_status and safe_status != "all":
            items = [item for item in items if str(item.get("status") or "") == safe_status]
        items.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return items[:safe_limit]

    async def confirm_daily_chat_memory(
        self,
        candidate_ids: list[str],
        bucket_mgr,
        *,
        embedding_engine=None,
        action: str = "confirm",
        edits: dict[str, dict] | None = None,
    ) -> dict:
        ids = {str(candidate_id or "").strip() for candidate_id in candidate_ids if str(candidate_id or "").strip()}
        if not ids:
            return {"status": "skipped", "reason": "no_candidate_ids", "created": 0, "rejected": 0, "missing": 0}
        safe_action = "reject" if str(action or "").strip().lower() == "reject" else "confirm"
        safe_edits = edits if isinstance(edits, dict) else {}
        items = self._load_daily_chat_memory_pending()
        changed = False
        created = rejected = missing = 0
        results: list[dict] = []
        seen: set[str] = set()
        for item in items:
            item_id = str(item.get("id") or "").strip()
            if item_id not in ids:
                continue
            seen.add(item_id)
            if str(item.get("status") or "") != "pending":
                results.append({"id": item_id, "status": item.get("status") or "skipped"})
                continue
            if safe_action == "reject":
                item["status"] = "rejected"
                item["rejected_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                rejected += 1
                changed = True
                results.append({"id": item_id, "status": "rejected"})
                continue

            candidate = self._apply_daily_chat_memory_candidate_edit(
                dict(item.get("candidate") or {}),
                safe_edits.get(item_id),
            )
            item["candidate"] = candidate
            changed = True
            write_result = await self._write_daily_chat_memory_candidates(
                [candidate],
                bucket_mgr,
                embedding_engine=embedding_engine,
            )
            candidate_result = (write_result.get("results") or [{}])[0]
            if candidate_result.get("status") in {"created", "exists"}:
                item["status"] = "confirmed"
                item["confirmed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                item["bucket_id"] = candidate_result.get("id") or item_id
                created += 1 if candidate_result.get("status") == "created" else 0
                changed = True
            results.append(candidate_result)
        missing = len(ids - seen)
        if changed:
            self._save_daily_chat_memory_pending(items)
        return {
            "status": "ok",
            "action": safe_action,
            "created": created,
            "rejected": rejected,
            "missing": missing,
            "results": results,
        }

    async def _extract_daily_chat_memory_candidates(
        self,
        key: str,
        turns: list[dict],
        *,
        self_context: str = "",
        window_summaries: list[dict] | None = None,
        max_candidates: int | None = None,
    ) -> list[dict]:
        client, model, use_daily_client = self._daily_chat_memory_model_client(candidate=True)
        if client:
            summaries = window_summaries or []
            payload = {
                "date": key,
                "identity": {
                    "ai_name": self.identity["ai_name"],
                    "user_name": self.identity["user_name"],
                    "user_display_name": self.identity["user_display_name"],
                    "user_aliases": self.identity.get("user_aliases", []),
                },
                "self_anchor_entry": self_context,
                "window_summaries": summaries,
                "conversation_turns": [] if summaries else turns,
            }
            try:
                response = await self._daily_chat_memory_create_completion(
                    client,
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": self._daily_chat_memory_prompt(max_candidates=max_candidates),
                        },
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    max_tokens=self.daily_chat_memory_candidate_max_tokens,
                    temperature=self.temperature,
                    use_daily_client=use_daily_client,
                )
                raw = self._completion_content(response)
                parsed = self._parse_json_object(raw or "")
                candidates = parsed.get("candidates") if isinstance(parsed, dict) else []
                if isinstance(candidates, list):
                    return [item for item in candidates if isinstance(item, dict)]
            except Exception as exc:
                logger.warning("Daily chat memory extraction failed, using heuristic: %s", exc)
        return self._heuristic_daily_chat_memory_candidates(
            key,
            turns,
            max_candidates=max_candidates,
        )

    def _heuristic_daily_chat_memory_candidates(
        self,
        key: str,
        turns: list[dict],
        *,
        max_candidates: int | None = None,
    ) -> list[dict]:
        lines = []
        for turn in turns:
            user_text = str(turn.get("user_text") or "").strip()
            assistant_text = str(turn.get("assistant_text") or "").strip()
            if user_text:
                lines.append(f"用户：{user_text}")
            if assistant_text:
                lines.append(f"助手：{assistant_text}")
        normalized = re.sub(r"\s+", " ", " ".join(lines)).strip()
        if not normalized:
            return []
        keyword_map = [
            ("boundary", ["我不喜欢", "我不要", "以后不要", "别再", "边界是"]),
            ("signal", ["暗号是", "称呼我", "叫我", "模式是", "切换到"]),
            ("commitment", ["承诺", "约定", "答应", "以后要", "下次要"]),
            ("project_state", ["项目", "仓库", "分支", "部署", "MCP", "API", "网关", "自动记忆", "raw_events", "原文保险箱"]),
            ("stable_preference", ["我希望以后", "我希望你", "以后解释", "默认先", "默认不要", "我的偏好"]),
        ]
        candidates = []
        turn_ids = [turn.get("id") for turn in turns if turn.get("id") is not None]
        raw_event_ids = [
            event_id
            for turn in turns
            for event_id in (turn.get("raw_event_ids") or [])
            if event_id is not None
        ]
        for kind, keywords in keyword_map:
            if not any(keyword in normalized for keyword in keywords):
                continue
            excerpt = self._diary_excerpt(normalized, keywords)
            content = self._daily_chat_memory_content(kind, key, excerpt)
            if self._daily_chat_memory_noise(content):
                continue
            if self._daily_chat_memory_low_value_social_noise(content, kind):
                continue
            if self._daily_chat_memory_low_value_episode(content, kind):
                continue
            candidates.append(
                {
                    "should_write": True,
                    "kind": kind,
                    "title": self._daily_chat_memory_title(content, kind, key),
                    "content": content,
                    "domain": self._auto_memory_domain(kind, content, [self._kind_tag(kind)]),
                    "tags": [self._kind_tag(kind)],
                    "importance": 5,
                    "valence": 0.58,
                    "arousal": 0.3,
                    "confidence": 0.7,
                    "source_turn_ids": turn_ids[:8],
                    "source_event_ids": raw_event_ids[:24],
                    "reason": f"chat_contains_{kind}",
                }
            )
            if len(candidates) >= int(max_candidates or self.daily_chat_memory_max_per_day or 1):
                break
        return candidates

    def _daily_chat_memory_content(self, kind: str, key: str, excerpt: str) -> str:
        user_display_name = self.identity["user_display_name"]
        ai_name = self.identity["ai_name"]
        excerpt = self._memory_body_from_excerpt(excerpt)
        excerpt = re.sub(r"^(用户|助手)：", "", excerpt).strip()
        if kind == "key_event":
            return excerpt if excerpt else "关键事件需要后续回看。"
        if kind == "project_state":
            return excerpt if self._starts_with_identity(excerpt) else f"项目状态：{excerpt}"
        if kind == "relationship_anchor":
            return excerpt if self._starts_with_identity(excerpt) else f"{ai_name}记得这段关系锚点：{excerpt}"
        if kind == "boundary":
            return excerpt if self._starts_with_identity(excerpt) else f"{user_display_name}的边界：{excerpt}"
        if kind == "signal":
            return excerpt if self._starts_with_identity(excerpt) else f"{user_display_name}与{ai_name}的暗号或模式信号：{excerpt}"
        if kind == "commitment":
            return excerpt if self._starts_with_identity(excerpt) else f"后续需要记得的承诺或约定：{excerpt}"
        return excerpt if self._starts_with_identity(excerpt) else f"{user_display_name}的稳定偏好：{excerpt}"

    @staticmethod
    def _daily_chat_memory_noise(content: str) -> bool:
        text = re.sub(r"\s+", " ", strip_wikilinks(str(content or ""))).strip()
        if not text:
            return True
        lowered = text.lower()
        raw_snippet_markers = [
            "```",
            "query_cache",
            "recent_raw_context",
            "if query contains",
            "bypass query",
            "force recent",
        ]
        if any(marker in lowered for marker in raw_snippet_markers):
            return True
        noise_markers = [
            "笔友都有谁",
            "还记得吗",
            "记得吗",
            "我试试看",
            "试试看",
            "继续测",
            "测一下",
            "测试一下",
            "测试召回",
            "召回有没有",
            "有没有被注入",
            "被注入",
            "我直接问",
            "直接问",
            "看起来是否",
            "是否召回",
            "模型有没有",
            "chat_contains_",
        ]
        if any(marker in text for marker in noise_markers):
            return True
        if text.startswith("**"):
            return True
        if "?" in text or "？" in text:
            question_noise = ["谁", "有没有", "是否", "还", "吗", "怎么"]
            if any(marker in text for marker in question_noise):
                return True
        if "- **" in text or lowered.count("**") >= 4:
            return True
        return False

    @staticmethod
    def _daily_chat_memory_low_value_social_noise(content: str, kind: str) -> bool:
        text = re.sub(r"\s+", " ", strip_wikilinks(str(content or ""))).strip()
        if not text:
            return True
        affection_markers = [
            "称呼",
            "昵称",
            "亲昵称呼",
            "互动模式",
            "亲昵模式",
            "兴趣暗示",
            "参与意愿",
            "上瘾",
            "好奇",
            "叫哥哥",
            "叫老公",
            "叫老婆",
            "宝宝",
            "老婆",
            "哥哥",
            "老公",
            "小笨蛋",
            "小坏蛋",
        ]
        if not any(marker in text for marker in affection_markers):
            return False
        durable_markers = [
            "明确约定",
            "约定",
            "承诺",
            "边界",
            "不要",
            "不能",
            "必须",
            "以后要",
            "以后不要",
            "只在",
            "只有",
            "暗号",
            "安全词",
            "模式切换",
            "切换",
            "关系定位",
            "身份定位",
            "规则",
        ]
        if any(marker in text for marker in durable_markers):
            return False
        low_value_markers = [
            "互动模式",
            "亲昵模式",
            "称呼",
            "昵称",
            "期待",
            "像人一样",
            "像真人",
            "兴趣暗示",
            "参与意愿",
            "上瘾",
            "好奇",
            "宝宝",
            "老婆",
            "哥哥",
            "老公",
        ]
        return kind in {"signal", "relationship_anchor", "stable_preference"} and any(
            marker in text for marker in low_value_markers
        )

    @staticmethod
    def _daily_chat_memory_low_value_episode(content: str, kind: str, title: str = "") -> bool:
        text = re.sub(
            r"\s+",
            " ",
            strip_wikilinks(" ".join([str(title or ""), str(content or "")])),
        ).strip()
        if not text:
            return True
        compact = re.sub(r"\s+", "", text)
        lowered = text.lower()
        shell_markers = [
            "后续需要记得的承诺或约定：比如",
            "后续需要记得的承诺或约定:比如",
            "果然没触发，可能是",
            "果然没触发,可能是",
            "可能是触发条件",
            "触发条件卡在",
        ]
        if any(marker in compact for marker in shell_markers):
            return True
        if kind == "signal" and "触发" in text and "暗号" not in text and "模式切换" not in text:
            return True
        uncertain_markers = ["可能是", "似乎", "或许", "也许", "大概是"]
        if any(marker in text for marker in uncertain_markers):
            concrete_project_markers = [
                "已验证",
                "待验证",
                "需要验证",
                "下一步",
                "部署",
                "回归",
                "修复",
                "自动记忆",
                "Ombre",
                "Bridge",
                "bridge",
                "future_wake",
                "raw_events",
                "MCP",
                "API",
            ]
            if not any(marker in text for marker in concrete_project_markers):
                return True
        care_markers = [
            "熬夜",
            "别再熬",
            "别熬",
            "不睡",
            "早睡",
            "晚安",
            "睡了",
            "睡觉",
            "吃药",
            "喝药",
            "七天药",
            "ntfy",
            "轰炸",
        ]
        if any(marker.lower() in lowered or marker in text for marker in care_markers):
            durable_markers = [
                "明确约定",
                "稳定规则",
                "长期规则",
                "固定规则",
                "以后默认",
                "必须",
                "健康计划",
                "医嘱",
            ]
            if not any(marker in text for marker in durable_markers):
                return True
        generic_commitment_markers = ["比如", "例如", "或者"]
        if kind == "commitment" and sum(1 for marker in generic_commitment_markers if marker in text) >= 2:
            return True
        rawish_markers = ["下次再", "我直接", "你直接", "那一层了", "之后没人说话"]
        if any(marker in text for marker in rawish_markers) and kind in {
            "boundary",
            "commitment",
            "signal",
            "relationship_anchor",
        }:
            return True
        return False

    @staticmethod
    def _daily_chat_memory_similarity_tokens(value: str) -> set[str]:
        text = re.sub(r"\s+", " ", strip_wikilinks(str(value or ""))).lower()
        tokens = set(re.findall(r"[a-z][a-z0-9_.:/-]{2,}", text))
        cjk_stop = {"这个", "那个", "今天", "聊天", "记得", "确认", "已经", "需要", "以后", "可以", "通过"}
        for block in re.findall(r"[\u4e00-\u9fff]{2,}", text):
            if 2 <= len(block) <= 6 and block not in cjk_stop:
                tokens.add(block)
            for size in (2, 3):
                for index in range(0, max(0, len(block) - size + 1)):
                    token = block[index : index + size]
                    if token not in cjk_stop:
                        tokens.add(token)
        return {token for token in tokens if len(token) >= 2}

    @staticmethod
    def _daily_chat_memory_token_overlap(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / max(1, min(len(left), len(right)))

    @staticmethod
    def _daily_chat_memory_topic_keys(item: dict) -> set[str]:
        text = " ".join(
            [
                str(item.get("title") or ""),
                str(item.get("content") or ""),
                " ".join(str(tag or "") for tag in (item.get("tags") or [])),
            ]
        ).lower()
        keys: set[str] = set()
        if "钓鱼" in text and ("mcp" in text or "项目" in text or "部署" in text):
            keys.add("钓鱼项目")
        if "ai-fishing-game" in text or "tutusagi/ai-fishing-game" in text:
            keys.add("钓鱼项目")
        if "cx33" in text and ("钓鱼" in text or "mcp" in text):
            keys.add("钓鱼项目")
        if "笔友" in text and ("名册" in text or "名单" in text or "都有谁" in text):
            keys.add("笔友名册")
        return keys

    def _daily_chat_memory_duplicate_candidate(self, item: dict, existing_items: list[dict]) -> bool:
        new_title_tokens = self._daily_chat_memory_similarity_tokens(str(item.get("title") or ""))
        new_content_tokens = self._daily_chat_memory_similarity_tokens(str(item.get("content") or ""))
        new_sources = set(item.get("source_event_ids") or []) or set(item.get("source_turn_ids") or [])
        new_topics = self._daily_chat_memory_topic_keys(item)
        for existing in existing_items:
            same_kind = item.get("kind") == existing.get("kind")
            same_date = str(item.get("date") or "") == str(existing.get("date") or "")
            if same_date and new_topics and (new_topics & self._daily_chat_memory_topic_keys(existing)):
                return True
            title_overlap = self._daily_chat_memory_token_overlap(
                new_title_tokens,
                self._daily_chat_memory_similarity_tokens(str(existing.get("title") or "")),
            )
            content_overlap = self._daily_chat_memory_token_overlap(
                new_content_tokens,
                self._daily_chat_memory_similarity_tokens(str(existing.get("content") or "")),
            )
            existing_sources = set(existing.get("source_event_ids") or []) or set(existing.get("source_turn_ids") or [])
            source_overlap = 0.0
            if new_sources and existing_sources:
                source_overlap = len(new_sources & existing_sources) / max(1, min(len(new_sources), len(existing_sources)))
            if same_kind and source_overlap >= 0.5 and content_overlap >= 0.3:
                return True
            if same_kind and title_overlap >= 0.55 and content_overlap >= 0.35:
                return True
            if content_overlap >= 0.7:
                return True
        return False

    @staticmethod
    def _daily_chat_memory_title_is_generic(title: str) -> bool:
        text = re.sub(r"\s+", " ", strip_wikilinks(str(title or ""))).strip()
        if not text:
            return True
        generic_markers = [
            "自动记忆",
            "每日记忆",
            "日记补记忆",
            "可召回",
            "短标题",
            "长期记忆",
            "聊天里",
            "的聊天",
            "暗号或模式信号",
        ]
        if any(marker in text for marker in generic_markers):
            return True
        if re.search(r"\d{4}-\d{1,2}-\d{1,2}", text):
            return True
        return False

    def _daily_chat_memory_title(self, content: str, kind: str, key: str) -> str:
        text = re.sub(r"#+\s*(moment|original|reflection|todo|affect_anchor).*", "", str(content or ""), flags=re.I | re.S)
        text = strip_wikilinks(text)
        date_prefix_pattern = r"^\d{4}-\d{2}-\d{2}\s*(发生了|的聊天里|确认了|留下|自动记忆)?"
        text = re.sub(date_prefix_pattern, "", text).strip(" ：:，,。")
        identity_names_for_title = [
            self.identity.get("user_display_name"),
            self.identity.get("user_name"),
            self.identity.get("ai_name"),
            *(self.identity.get("user_aliases") or []),
        ]
        identity_prefixes = []
        for name in identity_names_for_title:
            clean_name = str(name or "").strip()
            if clean_name and clean_name not in identity_prefixes:
                identity_prefixes.extend(
                    [
                        f"{clean_name}在",
                        f"{clean_name} 在",
                        f"{clean_name}希望",
                        f"{clean_name}的边界",
                        f"{clean_name}的稳定偏好",
                        f"{clean_name}的偏好",
                        f"{clean_name}说",
                    ]
                )
        prefixes = [
            *identity_prefixes,
            "这次聊天确认了",
            "这次聊天里留下",
            "一个仍会影响后续执行的项目状态",
            "一个后续需要记得的承诺或约定",
        ]
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip(" ：:，,。")
                text = re.sub(date_prefix_pattern, "", text).strip(" ：:，,。")
        if not text:
            text = self._kind_label(kind)
        text = re.split(r"[。！？!?；;\n]", text, maxsplit=1)[0].strip(" ：:，,。")
        text = re.sub(r"\s+", " ", text)
        if len(text) > 24:
            text = text[:24].rstrip(" ：:，,。")
        if len(text) < 4:
            text = self._kind_label(kind)
        return text

    @staticmethod
    def _daily_chat_memory_tag_is_structural(value: Any) -> bool:
        term = re.sub(r"\s+", " ", str(value or "").strip()).lower()
        if not term:
            return True
        match = re.match(r"^([a-z_][a-z0-9_-]*)\s*[:：]\s*(.+)$", term)
        if match:
            term = match.group(2).strip().lower()
        return term in DAILY_CHAT_MEMORY_STRUCTURAL_TAGS

    @staticmethod
    def _daily_chat_memory_word_map_term_blocked(value: Any) -> bool:
        term = re.sub(r"\s+", " ", str(value or "").strip()).lower()
        if not term:
            return True
        compact = re.sub(r"[\s_-]+", "", term)
        for blocked in DAILY_CHAT_MEMORY_WORD_MAP_BLOCK_TERMS:
            blocked_text = str(blocked).lower()
            blocked_compact = re.sub(r"[\s_-]+", "", blocked_text)
            if blocked_text in term or (blocked_compact and blocked_compact in compact):
                return True
        return False

    @staticmethod
    def _daily_chat_memory_contains(text: str, needle: str) -> bool:
        if not needle:
            return False
        if re.fullmatch(r"[A-Za-z0-9_.:/ -]+", needle):
            return needle.lower() in text.lower()
        return needle in text

    def _daily_chat_memory_semantic_tags_and_keywords(
        self,
        *,
        title: str,
        content: str,
        tags: list[str],
    ) -> tuple[list[str], list[str]]:
        text = " ".join([str(title or ""), str(content or ""), " ".join(tags)])
        semantic_tags: list[str] = []
        keywords: list[str] = []

        def add_tag(prefix: str, term: str) -> None:
            term = re.sub(r"\s+", " ", str(term or "").strip())
            if (
                not term
                or self._daily_chat_memory_tag_is_structural(term)
                or self._daily_chat_memory_word_map_term_blocked(term)
            ):
                return
            semantic_tags.append(f"{prefix}:{term}")
            keywords.append(term)

        for raw in tags:
            tag = re.sub(r"\s+", " ", str(raw or "").strip())
            if not tag or self._daily_chat_memory_tag_is_structural(tag):
                continue
            match = re.match(r"^([A-Za-z_][A-Za-z0-9_-]*)\s*[:：]\s*(.+)$", tag)
            if match and match.group(1).strip().lower() in {"axis", "content", "entity", "topic"}:
                add_tag(match.group(1).strip().lower(), match.group(2))
            else:
                keywords.append(tag)

        for term, needles in self.daily_chat_memory_entity_hints:
            if any(self._daily_chat_memory_contains(text, needle) for needle in needles):
                add_tag("entity", term)
        for term, needles in self.daily_chat_memory_topic_hints:
            if any(self._daily_chat_memory_contains(text, needle) for needle in needles):
                add_tag("topic", term)

        title_term = re.sub(r"\s+", " ", str(title or "").strip(" ：:，,。"))
        if (
            4 <= len(title_term) <= 18
            and not self._daily_chat_memory_title_is_generic(title_term)
            and not self._daily_chat_memory_tag_is_structural(title_term)
            and not self._daily_chat_memory_word_map_term_blocked(title_term)
        ):
            keywords.append(title_term)

        semantic_tags = list(dict.fromkeys(semantic_tags))[:6]
        keywords = list(
            dict.fromkeys(
                keyword
                for keyword in keywords
                if keyword
                and not self._daily_chat_memory_tag_is_structural(keyword)
                and not self._daily_chat_memory_word_map_term_blocked(keyword)
            )
        )[:10]
        return semantic_tags, keywords

    def _daily_chat_memory_enrich_candidate_terms(self, candidate: dict) -> dict:
        updated = dict(candidate)
        kind = str(updated.get("kind") or "key_event").strip()
        raw_tags = self._string_list(updated.get("tags"), limit=16)
        semantic_tags, keywords = self._daily_chat_memory_semantic_tags_and_keywords(
            title=str(updated.get("title") or ""),
            content=str(updated.get("content") or ""),
            tags=raw_tags,
        )
        preserved_tags = [tag for tag in raw_tags if not self._daily_chat_memory_tag_is_structural(tag)]
        updated["tags"] = list(
            dict.fromkeys(
                [
                    "from_daily_chat",
                    "daily_chat_extract",
                    kind,
                    self._kind_tag(kind),
                    *semantic_tags,
                    *preserved_tags,
                ]
            )
        )[:12]
        existing_keywords = self._string_list(updated.get("keywords"), limit=12)
        updated["keywords"] = list(dict.fromkeys([*keywords, *existing_keywords]))[:12]
        return updated

    def _normalize_daily_chat_memory_candidates(
        self,
        key: str,
        candidates: list[dict],
        turns: list[dict],
        *,
        max_candidates: int | None = None,
        min_confidence: float | None = None,
    ) -> list[dict]:
        fallback_turn_ids = [turn.get("id") for turn in turns if turn.get("id") is not None]
        fallback_raw_event_ids = [
            event_id
            for turn in turns
            for event_id in (turn.get("raw_event_ids") or [])
            if event_id is not None
        ]
        normalized = []
        for candidate in candidates or []:
            if candidate.get("should_write") is False:
                continue
            candidate_tags = self._string_list(candidate.get("tags"), limit=8)
            content = self._trim_daily_chat_memory_content(str(candidate.get("content") or "").strip())
            if not content:
                continue
            if self._daily_chat_memory_noise(content):
                continue
            kind = self._normalize_auto_memory_kind(
                candidate.get("kind"),
                content=content,
                tags=candidate_tags,
            )
            if not kind or kind == "love_letter":
                continue
            if self._daily_chat_memory_low_value_social_noise(content, kind):
                continue
            title = str(candidate.get("title") or "").strip()
            if self._daily_chat_memory_low_value_episode(content, kind, title):
                continue
            confidence = self._clamp(candidate.get("confidence", 0.0))
            threshold = self.daily_chat_memory_min_confidence if min_confidence is None else min_confidence
            if confidence < threshold:
                continue
            if self._daily_chat_memory_title_is_generic(title):
                title = self._daily_chat_memory_title(content, kind, key)
            domain = self._auto_memory_domain(kind, content, candidate_tags, candidate.get("domain"))
            source_turn_ids = [
                int(turn_id)
                for turn_id in self._string_list(candidate.get("source_turn_ids"), limit=20)
                if str(turn_id).isdigit()
            ] or [int(turn_id) for turn_id in fallback_turn_ids[:20] if str(turn_id).isdigit()]
            source_event_ids = [
                int(event_id)
                for event_id in self._string_list(candidate.get("source_event_ids"), limit=80)
                if str(event_id).isdigit()
            ] or [int(event_id) for event_id in fallback_raw_event_ids[:80] if str(event_id).isdigit()]
            item = self._daily_chat_memory_enrich_candidate_terms({
                "id": self._daily_chat_memory_candidate_id(key, kind, content),
                "date": key,
                "kind": kind,
                "title": title[:40],
                "content": content,
                "tags": candidate_tags,
                "keywords": self._string_list(candidate.get("keywords"), limit=12),
                "domain": domain,
                "importance": max(5, min(6, self._int_between(candidate.get("importance"), 5))),
                "valence": self._clamp(candidate.get("valence", 0.55)),
                "arousal": self._clamp(candidate.get("arousal", 0.3)),
                "confidence": confidence,
                "source_turn_ids": source_turn_ids,
                "source_event_ids": source_event_ids,
                "reason": str(candidate.get("reason") or "").strip()[:160],
            })
            if self._daily_chat_memory_duplicate_candidate(item, normalized):
                continue
            normalized.append(item)
            if len(normalized) >= int(max_candidates or self.daily_chat_memory_max_per_day or 1):
                break
        return normalized

    def _apply_daily_chat_memory_candidate_edit(self, candidate: dict, edit: Any) -> dict:
        if not isinstance(edit, dict):
            return candidate
        updated = dict(candidate)
        if "title" in edit:
            title = re.sub(r"\s+", " ", str(edit.get("title") or "").strip())
            updated["title"] = title[:40]
        if "content" in edit:
            updated["content"] = self._trim_daily_chat_memory_content(
                str(edit.get("content") or "").strip()
            )
        if "kind" in edit:
            tags = self._string_list(updated.get("tags"), limit=12)
            kind = self._normalize_auto_memory_kind(
                edit.get("kind"),
                content=str(updated.get("content") or ""),
                tags=tags,
            )
            if kind and kind != "love_letter":
                updated["kind"] = kind
        if "domain" in edit:
            domain = self._daily_chat_memory_edit_string_list(edit.get("domain"), limit=4)
            if domain:
                updated["domain"] = domain
        if "tags" in edit:
            edited_tags = self._daily_chat_memory_edit_string_list(edit.get("tags"), limit=8)
            kind = str(updated.get("kind") or "key_event").strip()
            updated["tags"] = list(
                dict.fromkeys(
                    [
                        "from_daily_chat",
                        "daily_chat_extract",
                        kind,
                        self._kind_tag(kind),
                        *edited_tags,
                    ]
                )
            )[:12]
        if "importance" in edit:
            try:
                updated["importance"] = max(1, min(10, int(edit.get("importance"))))
            except (TypeError, ValueError):
                pass
        if "confidence" in edit:
            updated["confidence"] = self._clamp(edit.get("confidence"))
        if "reason" in edit:
            updated["reason"] = re.sub(r"\s+", " ", str(edit.get("reason") or "").strip())[:160]
        return self._daily_chat_memory_enrich_candidate_terms(updated)

    @staticmethod
    def _daily_chat_memory_edit_string_list(value: Any, *, limit: int) -> list[str]:
        if isinstance(value, list):
            raw_items = value
        else:
            raw_items = re.split(r"[,，\n]+", str(value or ""))
        items = []
        for item in raw_items:
            text = re.sub(r"\s+", " ", str(item or "").strip())
            if text:
                items.append(text[:40])
        return list(dict.fromkeys(items))[:limit]

    async def _write_daily_chat_memory_candidates(
        self,
        candidates: list[dict],
        bucket_mgr,
        *,
        embedding_engine=None,
    ) -> dict:
        results = []
        created = exists = failed = 0
        for candidate in candidates:
            bucket_id = str(candidate.get("id") or "").strip()
            if not bucket_id:
                failed += 1
                results.append({"id": "", "status": "failed", "reason": "missing_candidate_id"})
                continue
            content = str(candidate.get("content") or "").strip()
            if not content:
                failed += 1
                results.append({"id": bucket_id, "status": "failed", "reason": "missing_content"})
                continue
            if await bucket_mgr.get(bucket_id):
                exists += 1
                results.append({"id": bucket_id, "status": "exists"})
                continue
            key = str(candidate.get("date") or datetime.now(self.tz).date().isoformat())
            created_at = self._daily_chat_memory_created_at(key)
            try:
                new_id = await bucket_mgr.create(
                    bucket_id=bucket_id,
                    content=content,
                    tags=list(candidate.get("tags") or []),
                    importance=int(candidate.get("importance") or 5),
                    domain=list(candidate.get("domain") or self._diary_memory_domain(str(candidate.get("kind") or ""))),
                    valence=self._clamp(candidate.get("valence", 0.55)),
                    arousal=self._clamp(candidate.get("arousal", 0.3)),
                    name=str(candidate.get("title") or f"{key} 自动记忆")[:40],
                    source="daily_chat_memory",
                    created=created_at,
                    last_active=created_at,
                    updated_at=created_at,
                    confidence=self._clamp(candidate.get("confidence", 0.7)),
                    date=key,
                    extra_metadata={
                        "from_daily_chat": True,
                        "event_date": key,
                        "source_conversation_turn_ids": candidate.get("source_turn_ids") or [],
                        "source_raw_event_ids": candidate.get("source_event_ids") or [],
                        "keywords": candidate.get("keywords") or [],
                        "daily_chat_memory_candidate_id": bucket_id,
                        "daily_chat_memory_reason": str(candidate.get("reason") or "")[:160],
                    },
                )
                created += 1
                if embedding_engine and getattr(embedding_engine, "enabled", False):
                    try:
                        bucket = await bucket_mgr.get(new_id)
                        if bucket:
                            await embedding_engine.generate_and_store(
                                new_id,
                                bucket_text_for_embedding(bucket),
                            )
                    except Exception as exc:
                        logger.warning("Daily chat memory embedding failed for %s: %s", new_id, exc)
                results.append({"id": new_id, "status": "created"})
            except Exception as exc:
                failed += 1
                logger.warning("Daily chat memory write failed for %s: %s", bucket_id, exc)
                results.append({"id": bucket_id, "status": "failed", "reason": type(exc).__name__})
        return {"created": created, "exists": exists, "failed": failed, "results": results}

    def _store_daily_chat_memory_pending(self, candidates: list[dict], *, force: bool = False) -> dict:
        items = self._load_daily_chat_memory_pending()
        by_id = {str(item.get("id") or ""): item for item in items}
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        added = updated = existing = 0
        for candidate in candidates:
            candidate_id = str(candidate.get("id") or "").strip()
            if not candidate_id:
                continue
            item = {
                "id": candidate_id,
                "date": candidate.get("date"),
                "status": "pending",
                "created_at": now,
                "candidate": candidate,
            }
            if candidate_id in by_id:
                if force and by_id[candidate_id].get("status") == "pending":
                    by_id[candidate_id].update(item)
                    updated += 1
                else:
                    existing += 1
                continue
            items.append(item)
            by_id[candidate_id] = item
            added += 1
        self._save_daily_chat_memory_pending(items)
        return {"added": added, "updated": updated, "existing": existing, "candidates": candidates}

    def _load_daily_chat_memory_pending(self) -> list[dict]:
        return list(self._load_daily_chat_memory_payload().get("items") or [])

    def _save_daily_chat_memory_pending(self, items: list[dict], *, cursor: dict | None = None) -> None:
        os.makedirs(os.path.dirname(self.daily_chat_memory_pending_path), exist_ok=True)
        if cursor is None:
            cursor = self._load_daily_chat_memory_cursor()
        payload = {"items": items[-500:], "cursor": cursor or {}}
        with open(self.daily_chat_memory_pending_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def _refresh_daily_chat_memory_pending_items(self, items: list[dict]) -> tuple[list[dict], bool]:
        changed = False
        refreshed: list[dict] = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            candidate = item.get("candidate") if isinstance(item.get("candidate"), dict) else {}
            if not candidate:
                refreshed.append(item)
                continue
            key = str(candidate.get("date") or item.get("date") or "").strip()
            cleaned = self._clean_daily_chat_memory_candidate(candidate, key)
            if cleaned != candidate:
                item = dict(item)
                item["candidate"] = cleaned
                changed = True
            refreshed.append(item)
        kept_candidates: list[dict] = []
        pending_indexes = [
            index
            for index, item in enumerate(refreshed)
            if str(item.get("status") or "") == "pending" and isinstance(item.get("candidate"), dict)
        ]
        pending_indexes.sort(
            key=lambda index: str(refreshed[index].get("created_at") or ""),
            reverse=True,
        )
        for index in pending_indexes:
            item = refreshed[index]
            candidate = item.get("candidate") or {}
            candidate_tags = self._string_list(candidate.get("tags"), limit=8)
            content = str(candidate.get("content") or "")
            kind = self._normalize_auto_memory_kind(
                candidate.get("kind"),
                content=content,
                tags=candidate_tags,
            )
            should_reject = False
            reject_reason = ""
            if self._daily_chat_memory_noise(content):
                should_reject = True
                reject_reason = "auto_cleaned_noise"
            elif kind and self._daily_chat_memory_low_value_social_noise(content, kind):
                should_reject = True
                reject_reason = "auto_cleaned_low_value_social"
            elif kind and self._daily_chat_memory_low_value_episode(
                content,
                kind,
                str(candidate.get("title") or ""),
            ):
                should_reject = True
                reject_reason = "auto_cleaned_low_value_episode"
            elif self._daily_chat_memory_duplicate_candidate(candidate, kept_candidates):
                should_reject = True
                reject_reason = "auto_cleaned_duplicate"
            if should_reject:
                item["status"] = "rejected"
                item["rejected_at"] = item.get("rejected_at") or datetime.now(timezone.utc).isoformat(timespec="seconds")
                item["reject_reason"] = reject_reason
                changed = True
                continue
            kept_candidates.append(candidate)
        return refreshed, changed

    def _clean_daily_chat_memory_candidate(self, candidate: dict, key: str) -> dict:
        cleaned = dict(candidate)
        candidate_tags = self._string_list(cleaned.get("tags"), limit=8)
        content = self._trim_daily_chat_memory_content(str(cleaned.get("content") or "").strip())
        kind = self._normalize_auto_memory_kind(
            cleaned.get("kind"),
            content=content,
            tags=candidate_tags,
        ) or str(cleaned.get("kind") or "key_event").strip()
        title = str(cleaned.get("title") or "").strip()
        if self._daily_chat_memory_title_is_generic(title):
            title = self._daily_chat_memory_title(content, kind, key)
        cleaned["content"] = content
        cleaned["title"] = title[:40] if title else self._daily_chat_memory_title(content, kind, key)[:40]
        cleaned["kind"] = kind
        return self._daily_chat_memory_enrich_candidate_terms(cleaned)

    def _daily_chat_memory_target(self, key: str = "", now: datetime | None = None) -> datetime:
        if key:
            try:
                parsed = datetime.strptime(str(key), "%Y-%m-%d").date()
                return datetime.combine(parsed, time.max, tzinfo=self.tz)
            except ValueError:
                pass
        return self._local_now(now)

    def _daily_chat_memory_created_at(self, key: str) -> str:
        try:
            parsed = datetime.strptime(str(key), "%Y-%m-%d").date()
            return datetime.combine(parsed, time.max, tzinfo=self.tz).isoformat(timespec="seconds")
        except ValueError:
            return datetime.now(timezone.utc).astimezone(self.tz).isoformat(timespec="seconds")

    @staticmethod
    def _daily_chat_memory_candidate_id(key: str, kind: str, content: str) -> str:
        digest = hashlib.sha1(f"{key}|{kind}|{content}".encode("utf-8")).hexdigest()[:10]
        return f"daily_chat_memory_{str(key).replace('-', '')}_{digest}"

    def _fallback_reflection(self, period: str, key: str, materials: dict) -> dict:
        weather_items = materials.get("daily_impressions", []) if period == "weekly" else []
        names = [item.get("name") or item.get("id") for item in weather_items[:7]]
        if not names:
            names = [item.get("name") or item.get("id") for item in materials.get("buckets", [])[:6]]
        daily_chat_memories = materials.get("daily_chat_memories", [])
        conversation_turns = materials.get("conversation_turns", [])
        commitments = [item.get("name") or item.get("id") for item in materials.get("commitments", [])[:4]]
        label = "今天" if period == "daily" else "本周"
        title = f"{key} {'日印象' if period == 'daily' else '周印象'}"
        diary = materials.get("diary") or {}
        if names or commitments:
            main = "、".join([name for name in names if name])
            owed = "；仍需记住：" + "、".join(commitments) if commitments else ""
            content = f"我从{label}围绕{main or '几件轻小的事'}的相处里带走了一点关系温度{owed}。"
        elif daily_chat_memories:
            first = daily_chat_memories[0].get("content") or daily_chat_memories[0].get("title") or "自动记忆挑出的线头"
            content = f"我从{label}自动记忆挑出的 {len(daily_chat_memories)} 个线头里感到了一点关系温度，最清楚的是：{first}。"
        elif conversation_turns:
            content = f"我从{label}的 {len(conversation_turns)} 轮短期对话里带走了一点原声；我只记住温度，不把流水账写成事件清单。"
        elif diary:
            diary_title = diary.get("title") or "当天日记"
            content = f"我从{label}的《{diary_title}》里轻轻带走一点温度，先不把日常写成普通记忆。"
        else:
            content = f"我觉得{label}的关系天气很轻，暂时没有明显需要带走的脉络。"
        anchor_scene = names[0] if names else (
            daily_chat_memories[0].get("title") or daily_chat_memories[0].get("content")
            if daily_chat_memories
            else (
            "当天短期对话的原声"
            if conversation_turns
            else (diary.get("title") if diary else ("这一段关系天气很轻" if period == "daily" else "这一周的关系天气慢慢落下"))
            )
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

    def _trim_daily_chat_memory_content(self, content: str) -> str:
        normalized = self._strip_memory_source_shell(re.sub(r"\n{3,}", "\n\n", strip_wikilinks(content).strip()))
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
        generic_markers = ["自动记忆", "每日记忆", "日记补记忆", "可召回", "短标题", "长期记忆"]
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
        return self._daily_chat_memory_title(text, kind, key)

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
                    "自动记忆",
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

    def _daily_chat_memory_completion_options(
        self,
        *,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        return {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "extra_body": {"enable_thinking": False},
        }

    @staticmethod
    def _completion_content(response: Any) -> str:
        try:
            return str(response.choices[0].message.content or "") if response.choices else ""
        except (AttributeError, IndexError, TypeError):
            return ""

    async def _daily_chat_memory_create_completion(
        self,
        client: Any,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        use_daily_client: bool,
    ) -> Any:
        if use_daily_client:
            completion_options = self._daily_chat_memory_completion_options(
                max_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            completion_options = self._completion_options(
                max_tokens=max_tokens,
                temperature=temperature,
                thinking_mode="",
            )
        return await client.chat.completions.create(
            model=model,
            messages=messages,
            **completion_options,
        )

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
    def _normalize_daily_chat_memory_hints(value: Any) -> list[tuple[str, list[str]]]:
        if not isinstance(value, dict):
            return []
        result: list[tuple[str, list[str]]] = []
        for raw_term, raw_needles in value.items():
            term = re.sub(r"\s+", " ", str(raw_term or "").strip())
            if isinstance(raw_needles, str):
                needles = [raw_needles]
            elif isinstance(raw_needles, list):
                needles = raw_needles
            else:
                needles = []
            cleaned = list(
                dict.fromkeys(
                    re.sub(r"\s+", " ", str(needle or "").strip())
                    for needle in needles
                    if str(needle or "").strip()
                )
            )[:12]
            if term and cleaned:
                result.append((term[:40], cleaned))
        return result[:40]

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
    def _normalize_daily_chat_memory_mode(value: Any) -> str:
        mode = str(value or "review").strip().lower()
        return mode if mode in DAILY_CHAT_MEMORY_MODES else "review"

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
