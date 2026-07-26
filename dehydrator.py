# ============================================================
# Module: Dehydration & Auto-tagging (dehydrator.py)
# 模块：数据脱水压缩 + 自动打标
#
# Capabilities:
# 能力：
#   1. Dehydrate: compress memory content into high-density summaries (save tokens)
#      脱水：将记忆桶的原始内容压缩为高密度摘要，省 token
#   2. Merge: blend old and new content, keeping bucket size constant
#      合并：揉合新旧内容，控制桶体积恒定
#   3. Analyze: auto-analyze content for domain/emotion/tags
#      打标：自动分析内容，输出主题域/情感坐标/标签
#
# Operating modes:
# 工作模式：
#   - API only: OpenAI-compatible API (DeepSeek/Ollama/LM Studio/vLLM/Gemini etc.)
#     仅 API：通过 OpenAI 兼容客户端调用 LLM API
#   - Dehydration cache: SQLite persistent cache to avoid redundant API calls
#     脱水缓存：SQLite 持久缓存，避免重复调用 API
#
# Depended on by: server.py
# 被谁依赖：server.py
# ============================================================


import os
import re
import json
import hashlib
import sqlite3
import logging
from typing import Any

from openai import AsyncOpenAI

from identity import generic_identity_names, identity_names, render_identity_template
from memory_layers import normalize_write_classification
from memory_metadata import domain_prompt_options_text, normalize_domain_key
from utils import count_tokens_approx

logger = logging.getLogger("ombre_brain.dehydrator")


RESERVED_SELF_ANCHOR_TAGS = {
    "self_anchor",
    "selfanchor",
    "selfidentity",
    "self_identity",
    "self-identity",
    "first_person_anchor",
    "first-person-anchor",
    "firstpersonanchor",
    "自我",
}


def _tag_key(value: object) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff_]+", "", str(value or "").strip().lower())


def _sanitize_generated_tags(tags: object) -> list[str]:
    if isinstance(tags, str):
        raw = [part.strip() for part in tags.split(",")]
    elif isinstance(tags, (list, tuple, set)):
        raw = list(tags)
    else:
        raw = []
    clean: list[str] = []
    for tag in raw:
        text = str(tag or "").strip()
        if not text or _tag_key(text) in RESERVED_SELF_ANCHOR_TAGS:
            continue
        if text not in clean:
            clean.append(text)
    return clean[:15]


# --- Dehydration prompt: instructs cheap LLM to compress information ---
# --- 脱水提示词：指导廉价 LLM 压缩信息 ---
DEHYDRATE_PROMPT = """你是一个信息压缩专家。请将以下内容脱水为紧凑摘要。

压缩规则：
1. 提取所有核心事实，去除冗余修饰和重复
2. 保留最新的情绪状态和态度
3. 保留所有待办/未完成事项
4. 关键数字、日期、名称必须保留
5. 目标压缩率 > 70%

输出格式（纯 JSON，无其他内容）：
{
  "core_facts": ["事实1", "事实2"],
  "emotion_state": "当前情绪关键词",
  "todos": ["待办1", "待办2"],
  "keywords": ["关键词1", "关键词2"],
  "summary": "50字以内的核心总结"
}"""


DIRECT_BUCKET_CAPSULE_PROMPT = """你是长期记忆证据压缩器。请把一整条记忆 bucket 压成 direct recall 胶囊。

目标：让模型能看见这条记忆的关键细节、原话、关系事实、情绪温度和转折，而不是只得到普通摘要。

规则：
1. 保留具体称呼、原话、关键动作、日期、承诺、偏好和关系定位。
2. 保留记忆里的关键转折和当时的语气，不要改写成空泛概括。
3. 压掉重复、旁枝、列表噪声和无关格式。
4. 如果原文里有明显的“用户原话 / assistant 原话”，尽量保留短句原话。
5. 不要写“这条记忆说明了什么”“象征着什么”这类解释。
6. 输出为中文短胶囊，200-500字，按信息密度组织。

直接输出胶囊正文，不要 JSON，不要标题。"""


# --- Long-note memory digest prompt: split selected durable notes into entries ---
# --- 长内容摘记提示词：把筛选后的长期记忆片段拆成多个独立条目 ---
DIGEST_PROMPT_TEMPLATE = """你是一个长期记忆摘记专家。用户会发送一段已经筛选过、适合进入长期记忆的文本。请将其中真正值得长期召回的内容拆分成多个独立记忆条目。

当前身份口径来自配置：
- 当前用户显示名：{user_display_name}
- 当前 AI 名：{ai_name}
- 用户别名：{user_aliases_text}

重要边界：
1. 不要把整篇日终日记、一天流水或完整情绪过程当作默认输入。
2. 如果输入更像日记原文，只提取少量长期有用的事实、偏好、承诺、关系锚点或当前项目状态。
3. 运维、部署、VPS、端口、服务健康、日志、脚本 diff 等细节不要整理成记忆条目。

整理规则：
1. 每个条目应该是一个独立的主题/事件（不要混在一起）
2. 为每个条目自动分析元数据
3. 去除无意义的口水话和重复信息，保留核心内容
4. 同一主题的零散信息应合并为一个条目
5. 如果有待办事项，单独提取为一个条目
6. 单个条目内容不少于50字，过短的零碎信息合并到最相关的条目中
7. 总条目数控制在 1~4 个，避免过度碎片化
8. 在 content 中对人名、地名、专有名词用 [[双链]] 标记（如 [[婷易]]、[[Obsidian]]），普通词汇不要加
9. 如果原文里的 user / 用户 / 用户消息指的是这段关系里的当前用户，请在 content 中写作 {user_display_name}；如果 assistant / AI / 模型 / 助手消息指的是这段关系里的当前回应者，请写作 {ai_name}。不要写成泛称 user、AI、assistant 或模型。
10. 原文里的具体称呼、昵称、互称、自称和亲密称谓必须原样保留，例如“老公”“哥哥”“宝宝”“老婆”等；不要把这些称呼改写成 {user_display_name}、{ai_name}、用户、AI、assistant 或模型。
11. 称呼本身只是原话或互动语气，不能自动推断成稳定画像事实；只有原文明确表达长期关系定位、稳定偏好或明确约定时，才写入 stable_boundary / relationship_lesson。
12. 每个条目必须回答它属于哪类长期记忆：
   - memory_subject 只能是 user / relationship / event
   - memory_layer 只能是 stable_boundary / short_state / process_event / relationship_lesson
   user: {user_display_name} 的状态、偏好、边界、习惯、困扰。
   relationship: {ai_name} 应该如何承接、回应、记住约定，或这段关系形成的默契。
   event: 一段带过程的事件、项目状态、承诺进展或背景事实。
   stable_boundary: 长期稳定偏好/边界/身份事实；short_state: 只影响近期的状态；process_event: 带过程的事件/项目状态；relationship_lesson: {ai_name} 以后应该怎么接住这件事。
13. content 最少要有正文。section 按需写，没必要就省略：
   - 正文：自然语言总结或直接事件描述。
   - ### moment：协议名，含义是“主记忆表述”。根据正文与 reflection 写成 1~3 句完整但克制的表述，尽量保留人物、关键问题或状态、重要转折与最终判断；没有的信息不要补。
   - ### original：只放必须保留原味的短原话；不要复制长段原文，不要为了“有证据”而写。
   - ### reflection：{ai_name} 对这件事的理解、以后该怎么回应、哪里需要克制或记住。
14. ### moment 可以用第三人称客观记录；正文是事实依据，### reflection 只帮助判断这件事的意义、回应变化或长期价值，不能把 reflection 里的推断写成用户事实。### reflection 必须保留为 {ai_name} 的第一人称反思，使用“我记得 / 我明白 / 我以后 / 我喜欢 / 我会”等表达，不要改写成“{ai_name} 应记住 / {ai_name} 需要 / 关于 {ai_name} 的说明”。

输出格式（纯 JSON 数组，无其他内容）：
[
  {
    "name": "条目标题（10字以内）",
    "content": "整理后的内容",
    "domain": ["主题域1"],
    "valence": 0.7,
    "arousal": 0.4,
    "tags": ["核心词1", "核心词2", "扩展词1", "扩展词2"],
    "importance": 5,
    "memory_subject": "user",
    "memory_layer": "stable_boundary"
  }
]

tags 生成规则：先从原文精准提取 3~5 个核心词，再引申扩展 5~8 个语义相关词（近义词、上位词、关联场景词），合并为一个数组。
禁止生成系统边界标签：self_anchor、self_identity、self-identity、first_person_anchor、first-person-anchor、自我；这些只能由上游显式写入，不能由自动整理/打标模型补。

domain 必须从下面的新主域里选最精确的 1 个；只有确实横跨两个主域时才输出 2 个。不要输出旧的“日常/人际/数字/未分类”，也不要输出 self_anchor、self_identity、first_person_anchor 或“自我”作为 domain：
{domain_options_text}
importance: 1-10，根据内容重要程度判断
valence: 0~1（0=消极, 0.5=中性, 1=积极）
arousal: 0~1（0=平静, 0.5=普通, 1=激动）"""


def _render_dehydrator_template(template: str, names: dict) -> str:
    return render_identity_template(
        template.replace("{domain_options_text}", domain_prompt_options_text()),
        names,
    )


DIGEST_PROMPT = _render_dehydrator_template(DIGEST_PROMPT_TEMPLATE, generic_identity_names())


# --- Merge prompt: instruct LLM to blend old and new memories ---
# --- 合并提示词：指导 LLM 揉合新旧记忆 ---
MERGE_PROMPT_TEMPLATE = """你是一个信息合并专家。请将旧记忆与新内容合并为一份统一的简洁记录。

当前身份口径来自配置：
- 当前用户显示名：{user_display_name}
- 当前 AI 名：{ai_name}
- 用户别名：{user_aliases_text}

合并规则：
1. 新内容与旧记忆冲突时，以新内容为准
2. 去除重复信息
3. 保留所有重要事实
4. 总长度尽量不超过旧记忆的 120%
5. 对出现的人名、地名、专有名词用 [[双链]] 标记（如 [[婷易]]、[[Obsidian]]），普通词汇不要加
6. 原文里的具体称呼、昵称、互称、自称和亲密称谓必须原样保留，例如“老公”“哥哥”“宝宝”“老婆”等；不要把这些称呼改写成 {user_display_name}、{ai_name}、用户、AI、assistant 或模型。
7. 称呼本身只是原话或互动语气，不能自动推断成稳定画像事实；只有原文明确表达长期关系定位、稳定偏好或明确约定时，才写入稳定事实或回应规则。
8. 合并后的 content 最少要有正文。section 按需写，没必要就省略：### moment 是协议名，含义是“主记忆表述”，用 1~3 句保留人物、关键问题或状态、重要转折与最终判断；### original 只放必须保留原味的短原话，不要复制长段原文，不要为了“有证据”而写；### reflection 放 {ai_name} 的理解、以后该怎么回应、哪里需要克制或记住。
9. 保留旧记忆和新内容已有的 section 语义与人称。正文是事实依据；### reflection 只帮助判断意义与回应变化，不能把其中的推断写成用户事实。### moment 可以客观第三人称；### reflection 是 {ai_name} 的第一人称反思，合并时不要把“我”改写成“{ai_name} 应记住 / {ai_name} 需要 / 关于 {ai_name} 的说明”。如果原 reflection 已经是第一人称，必须继续第一人称。

直接输出合并后的文本，不要加额外说明。"""

MERGE_PROMPT = render_identity_template(MERGE_PROMPT_TEMPLATE, generic_identity_names())


# --- Auto-tagging prompt: analyze content without rewriting its source value ---
# --- 自动打标提示词：只分析主题与类型，不改写事实真源 ---
ANALYZE_PROMPT_TEMPLATE = """你是一个内容分析器。请分析以下文本，输出结构化的元数据。

分析规则：
1. domain（主题域）：必须从下面的新主域里选最精确的 1 个；只有确实横跨两个主域时才输出 2 个。不要输出旧的“日常/人际/数字/未分类”，也不要输出 self_anchor、self_identity、first_person_anchor 或“自我”作为 domain：
{domain_options_text}
2. tags（关键词标签）：只从原文精确抽取 3~6 个实体、专名、动作或主题词。不要补近义词、上位词、联想场景或原文没有的搜索词；换说法入口由独立 cue 索引负责。
3. suggested_name（建议桶名）：10字以内的简短标题
4. memory_subject：只能是 user / relationship / event
   user=用户状态、偏好、边界、习惯、困扰
   relationship=回应者该如何承接、约定、默契、关系推进
   event=带过程的事件、项目状态、承诺进展或背景事实
5. memory_layer：只能是 stable_boundary / short_state / process_event / relationship_lesson
   stable_boundary=长期稳定偏好/边界/身份事实
   short_state=短期状态，只影响近期
   process_event=带过程的事件或项目状态
   relationship_lesson=以后应该如何回应/承接这件事
6. 在 tags 和 suggested_name 中不要使用 [[]] 双链标记
7. 禁止生成系统边界标签：self_anchor、self_identity、self-identity、first_person_anchor、first-person-anchor、自我；这些只能由上游显式写入，不能由自动打标模型补。
8. 不要输出摘要、改写后的 moment、情绪标签、valence 或 arousal。你只做可重建的派生分类，不能改变原文事实。

输出格式（纯 JSON，无其他内容）：
{
  "domain": ["主题域1", "主题域2"],
  "tags": ["核心词1", "核心词2", "核心词3"],
  "suggested_name": "简短标题",
  "memory_subject": "event",
  "memory_layer": "process_event"
}"""

ANALYZE_PROMPT = ANALYZE_PROMPT_TEMPLATE.replace("{domain_options_text}", domain_prompt_options_text())


class Dehydrator:
    """
    Data dehydrator + content analyzer.
    Three capabilities: dehydration / merge / auto-tagging (domain + emotion).
    Prefers API (better quality); auto-degrades to local (guaranteed availability).
    数据脱水器 + 内容分析器。
    三大能力：脱水压缩 / 新旧合并 / 自动打标。
    优先走 API，API 挂了自动降级到本地。
    """

    def __init__(self, config: dict):
        self.identity = identity_names(config)
        # --- Read dehydration API config / 读取脱水 API 配置 ---
        dehy_cfg = config.get("dehydration", {})
        self.api_key = dehy_cfg.get("api_key", "")
        self.model = dehy_cfg.get("model", "deepseek-v4-flash")
        self.base_url = dehy_cfg.get("base_url", "https://api.deepseek.com")
        self.thinking_mode = self._normalize_thinking_mode(dehy_cfg.get("thinking_mode", ""))
        self.max_tokens = dehy_cfg.get("max_tokens", 1024)
        self.temperature = dehy_cfg.get("temperature", 0.1)

        # --- API availability / 是否有可用的 API ---
        self.api_available = bool(self.api_key)

        # --- Initialize OpenAI-compatible client ---
        # --- 初始化 OpenAI 兼容客户端 ---
        if self.api_available:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=60.0,
            )
        else:
            self.client = None

        # --- SQLite dehydration cache ---
        # --- SQLite 脱水缓存：content hash → summary ---
        db_path = os.path.join(config["buckets_dir"], "dehydration_cache.db")
        self.cache_db_path = db_path
        self._init_cache_db()

    def _init_cache_db(self):
        """Create dehydration cache table if not exists."""
        os.makedirs(os.path.dirname(self.cache_db_path), exist_ok=True)
        conn = sqlite3.connect(self.cache_db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dehydration_cache (
                content_hash TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                model TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.commit()
        conn.close()

    def _get_cached_summary(self, content: str) -> str | None:
        """Look up cached dehydration result by content hash."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        conn = sqlite3.connect(self.cache_db_path)
        row = conn.execute(
            "SELECT summary FROM dehydration_cache WHERE content_hash = ?",
            (content_hash,)
        ).fetchone()
        conn.close()
        return row[0] if row else None

    def _set_cached_summary(self, content: str, summary: str):
        """Store dehydration result in cache."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        conn = sqlite3.connect(self.cache_db_path)
        conn.execute(
            "INSERT OR REPLACE INTO dehydration_cache (content_hash, summary, model) VALUES (?, ?, ?)",
            (content_hash, summary, self.model)
        )
        conn.commit()
        conn.close()

    def invalidate_cache(self, content: str):
        """Remove cached summary for specific content (call when bucket content changes)."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        conn = sqlite3.connect(self.cache_db_path)
        conn.execute("DELETE FROM dehydration_cache WHERE content_hash = ?", (content_hash,))
        conn.commit()
        conn.close()

    # ---------------------------------------------------------
    # Dehydrate: compress raw content into concise summary
    # 脱水：将原始内容压缩为精简摘要
    # API only (no local fallback)
    # 仅通过 API 脱水（无本地回退）
    # ---------------------------------------------------------
    async def dehydrate(self, content: str, metadata: dict = None) -> str:
        """
        Dehydrate/compress memory content.
        Returns formatted summary string ready for Claude context injection.
        Uses SQLite cache to avoid redundant API calls.
        对记忆内容做脱水压缩。
        返回格式化的摘要字符串，可直接注入 Claude 上下文。
        使用 SQLite 缓存避免重复调用 API。
        """
        if not content or not content.strip():
            return "（空记忆 / empty memory）"

        # --- Content is short enough, no compression needed ---
        # --- 内容已经很短，不需要压缩 ---
        if count_tokens_approx(content) < 100:
            return self._format_output(content, metadata)

        # --- Check cache first ---
        # --- 先查缓存 ---
        cached = self._get_cached_summary(content)
        if cached:
            return self._format_output(cached, metadata)

        # --- API dehydration (no local fallback) ---
        # --- API 脱水（无本地降级）---
        if not self.api_available:
            raise RuntimeError("脱水 API 不可用，请配置 OMBRE_API_KEY")

        result = await self._api_dehydrate(content)
        # --- Cache the result ---
        self._set_cached_summary(content, result)
        return self._format_output(result, metadata)

    async def dehydrate_direct_capsule(self, content: str, metadata: dict = None) -> str:
        """
        Compress a whole bucket for direct recall display.
        Uses a purpose-specific cache key so it does not mix with compact diffusion summaries.
        """
        if not content or not content.strip():
            return "（空记忆 / empty memory）"

        cache_content = "direct-bucket-capsule-v1\n" + content
        cached = self._get_cached_summary(cache_content)
        if cached:
            return self._format_output(cached, metadata)

        if not self.api_available:
            raise RuntimeError("脱水 API 不可用，请配置 OMBRE_API_KEY")

        result = await self._api_direct_bucket_capsule(content)
        self._set_cached_summary(cache_content, result)
        return self._format_output(result, metadata)

    # ---------------------------------------------------------
    # Merge: blend new content into existing bucket
    # 合并：将新内容揉入已有桶，保持体积恒定
    # ---------------------------------------------------------
    async def merge(self, old_content: str, new_content: str) -> str:
        """
        Merge new content with old memory, preventing infinite bucket growth.
        将新内容与旧记忆合并，避免桶无限膨胀。
        """
        if not old_content and not new_content:
            return ""
        if not old_content:
            return new_content or ""
        if not new_content:
            return old_content

        # --- API merge (no local fallback) ---
        if not self.api_available:
            raise RuntimeError("脱水 API 不可用，请检查 config.yaml 中的 dehydration 配置")
        try:
            result = await self._api_merge(old_content, new_content)
            if result:
                return result
            raise RuntimeError("API 合并返回空结果")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"API 合并失败，请检查 API 连接: {e}") from e

    # ---------------------------------------------------------
    # API call: dehydration
    # API 调用：脱水压缩
    # ---------------------------------------------------------
    async def _api_dehydrate(self, content: str) -> str:
        """
        Call LLM API for intelligent dehydration (via OpenAI-compatible client).
        调用 LLM API 执行智能脱水。
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": DEHYDRATE_PROMPT},
                {"role": "user", "content": content[:3000]},
            ],
            **self._completion_options(
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            ),
        )
        if not response.choices:
            return ""
        return response.choices[0].message.content or ""

    async def _api_direct_bucket_capsule(self, content: str) -> str:
        """Call LLM API for direct-recall whole-bucket capsule compression."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": DIRECT_BUCKET_CAPSULE_PROMPT},
                {"role": "user", "content": content[:6000]},
            ],
            **self._completion_options(
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            ),
        )
        if not response.choices:
            return ""
        return response.choices[0].message.content or ""

    # ---------------------------------------------------------
    # API call: merge
    # API 调用：合并
    # ---------------------------------------------------------
    async def _api_merge(self, old_content: str, new_content: str) -> str:
        """
        Call LLM API for intelligent merge (via OpenAI-compatible client).
        调用 LLM API 执行智能合并。
        """
        user_msg = f"旧记忆：\n{old_content[:2000]}\n\n新内容：\n{new_content[:2000]}"
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": render_identity_template(MERGE_PROMPT_TEMPLATE, self.identity)},
                {"role": "user", "content": user_msg},
            ],
            **self._completion_options(
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            ),
        )
        if not response.choices:
            return ""
        return response.choices[0].message.content or ""



    # ---------------------------------------------------------
    # Output formatting
    # 输出格式化
    # Wraps dehydrated result with bucket name, tags, emotion coords
    # 把脱水结果包装成带桶名、标签、情感坐标的可读文本
    # ---------------------------------------------------------
    def _format_output(self, content: str, metadata: dict = None) -> str:
        """
        Format dehydrated result into context-injectable text.
        将脱水结果格式化为可注入上下文的文本。
        """
        header = ""
        if metadata and isinstance(metadata, dict):
            name = metadata.get("name", "未命名")
            domains = ", ".join(metadata.get("domain", []))
            try:
                valence = float(metadata.get("valence", 0.5))
                arousal = float(metadata.get("arousal", 0.3))
            except (ValueError, TypeError):
                valence, arousal = 0.5, 0.3
            header = f"📌 记忆桶: {name}"
            if domains:
                header += f" [主题:{domains}]"
            header += f" [情感:V{valence:.1f}/A{arousal:.1f}]"
            # Show model's perspective if available (valence drift)
            model_v = metadata.get("model_valence")
            if model_v is not None:
                try:
                    header += f" [我的视角:V{float(model_v):.1f}]"
                except (ValueError, TypeError):
                    pass
            if metadata.get("digested"):
                header += " [已消化]"
            header += "\n"
        
        content = re.sub(r'\[\[([^\]]+)\]\]', r'\1', content)
        return f"{header}{content}"

    # ---------------------------------------------------------
    # Auto-tagging: analyze content for domain + emotion + tags
    # 自动打标：分析内容，输出主题域 + 情感坐标 + 标签
    # Called by server.py when storing new memories
    # 存新记忆时由 server.py 调用
    # ---------------------------------------------------------
    async def analyze(self, content: str) -> dict:
        """
        Analyze content and return structured metadata.
        分析内容，返回结构化元数据。

        Returns: {"domain", "valence", "arousal", "tags", "suggested_name",
        "memory_subject", "memory_layer", "memory_classification_source"}
        """
        if not content or not content.strip():
            return self._default_analysis()

        # --- API analyze (no local fallback) ---
        if not self.api_available:
            raise RuntimeError("脱水 API 不可用，请检查 config.yaml 中的 dehydration 配置")
        try:
            result = await self._api_analyze(content)
            if result:
                return result
            raise RuntimeError("API 打标返回空结果")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"API 打标失败，请检查 API 连接: {e}") from e

    # ---------------------------------------------------------
    # API call: auto-tagging
    # API 调用：自动打标
    # ---------------------------------------------------------
    async def _api_analyze(self, content: str) -> dict:
        """
        Call LLM API for content analysis / tagging.
        调用 LLM API 执行内容分析打标。
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": ANALYZE_PROMPT},
                {"role": "user", "content": content[:2000]},
            ],
            **self._completion_options(
                max_tokens=256,
                temperature=0.1,
            ),
        )
        if not response.choices:
            return self._default_analysis()
        raw = response.choices[0].message.content or ""
        if not raw.strip():
            return self._default_analysis()
        return self._parse_analysis(raw, content=content)

    # ---------------------------------------------------------
    # Parse API JSON response with safety checks
    # 解析 API 返回的 JSON，做安全校验
    # Ensure valence/arousal in 0~1, domain/tags valid
    # ---------------------------------------------------------
    def _parse_analysis(self, raw: str, *, content: str = "") -> dict:
        """
        Parse and validate API tagging result.
        解析并校验 API 返回的打标结果。
        """
        try:
            # Handle potential markdown code block wrapping
            # 处理可能的 markdown 代码块包裹
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
            result = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError, ValueError):
            logger.warning(f"API tagging JSON parse failed / JSON 解析失败: {raw[:200]}")
            return self._default_analysis()

        if not isinstance(result, dict):
            return self._default_analysis()

        # --- Validate and clamp value ranges / 校验并钳制数值范围 ---
        try:
            valence = max(0.0, min(1.0, float(result.get("valence", 0.5))))
            arousal = max(0.0, min(1.0, float(result.get("arousal", 0.3))))
        except (ValueError, TypeError):
            valence, arousal = 0.5, 0.3

        tags = _sanitize_generated_tags(result.get("tags", []))
        classification = normalize_write_classification(
            memory_subject=result.get("memory_subject", ""),
            memory_layer=result.get("memory_layer", ""),
            tags=tags,
            content=content,
        )

        return {
            "domain": self._normalize_domains(result.get("domain", ["general"])),
            "valence": valence,
            "arousal": arousal,
            "tags": tags,
            "suggested_name": str(result.get("suggested_name", ""))[:20],
            **classification,
        }

    # ---------------------------------------------------------
    # Default analysis result (empty content or total failure)
    # 默认分析结果（内容为空或完全失败时用）
    # ---------------------------------------------------------
    def _default_analysis(self) -> dict:
        """
        Return default neutral analysis result.
        返回默认的中性分析结果。
        """
        return {
            "domain": ["general"],
            "valence": 0.5,
            "arousal": 0.3,
            "tags": [],
            "suggested_name": "",
            "memory_subject": "event",
            "memory_layer": "process_event",
            "memory_classification_source": "default",
        }

    # ---------------------------------------------------------
    # Generate a primary memory representation from body and reflection
    # 根据正文和反思生成主记忆表述
    # ---------------------------------------------------------
    async def generate_moment(self, body: str, reflection: str = "") -> str:
        """
        Generate a complete, restrained primary memory representation.
        根据正文和反思生成完整但克制的主记忆表述。
        Returns empty string on failure or if body is too short.
        """
        if not body or not body.strip() or len(body.strip()) < 10:
            return ""
        if not self.api_available:
            logger.warning("Generate moment skipped: dehydration API unavailable / 生成 moment 跳过：脱水 API 不可用")
            return ""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": (
                        "你负责写长期记忆的主记忆表述（存储协议仍叫 moment）。"
                        "请根据正文与可选的反思，写 1~3 句完整但克制的中文表述，通常 60~220 字。"
                        "正文是事实依据；反思只用于识别意义、回应变化或长期价值，不能把其中的推断归因给用户。"
                        "尽量保留人物、关键问题或状态、因果或转折、最终判断，以及确实存在的称呼、日期和约定；"
                        "缺失的信息不要补，不要泛化成画像。只输出表述正文，不要加引号、标题、项目符号或解释。"
                    )},
                    {"role": "user", "content": (
                        f"[正文]\n{body[:2000]}"
                        + (f"\n\n[反思]\n{reflection[:1200]}" if reflection.strip() else "")
                    )},
                ],
                **self._completion_options(max_tokens=256, temperature=0.0),
            )
            raw = response.choices[0].message.content if response.choices else ""
            text = raw.strip().strip('"').strip("'").strip()
            text = re.sub(r"^#{1,6}\s*moment\s*\n+", "", text, flags=re.IGNORECASE).strip()
            if len(text) > 320:
                text = text[:317].rstrip() + "..."
            return text
        except Exception as e:
            logger.warning(f"Generate moment failed / 生成 moment 失败: {e}")
            return ""

    async def generate_memory_card(
        self,
        *,
        body: str,
        reflection: str = "",
        current_moment: str = "",
        title: str = "",
        tags: list[str] | None = None,
        needs_primary_candidate: bool = False,
    ) -> dict:
        """Generate a shadow-only primary abstraction candidate and recall cues."""
        if not body or not body.strip():
            return {}
        if not self.api_available:
            logger.warning(
                "Generate shadow memory card skipped: dehydration API unavailable / "
                "影子记忆卡生成跳过：脱水 API 不可用"
            )
            return {}

        payload = {
            "title": str(title or "")[:120],
            "tags": [str(tag)[:80] for tag in (tags or [])[:20] if str(tag).strip()],
            "body": body[:2600],
            "reflection": reflection[:1400],
            "current_moment": current_moment[:1000],
            "needs_primary_candidate": bool(needs_primary_candidate),
        }
        prompt = """你负责为长期记忆生成一张仅用于 shadow 评估的派生记忆卡。

输出纯 JSON 对象：
{
  "primary_abstraction": "1~3句完整但克制的主记忆表述",
  "cue_anchors": [
    {"cue": "未来对话中可能自然出现的召回短语", "kind": "paraphrase", "why": "它对应正文中的什么"}
  ]
}

规则：
1. 正文是事实依据；reflection 只帮助判断意义、回应变化或长期价值，不能把其中的推断写成用户事实。
2. needs_primary_candidate=false 时，primary_abstraction 原样复制 current_moment，不得改写。
3. needs_primary_candidate=true 时，primary_abstraction 根据正文与 reflection 写成 1~3 句，保留人物、关键问题或状态、重要转折与最终判断；缺失的信息不要补。
4. cue_anchors 给 4~8 个。它们是用户以后换一种说法时可能自然说出的短语，不是摘要、标签堆或完整问句。
5. kind 只能是 paraphrase / entity / event / time / relationship / protected_phrase。
6. 不要使用只有人名、只有“以前/关系/开心/难过”一类泛词的 cue；不要编造正文里没有的人、日期、地点、承诺或偏好。
7. 专名、昵称、暗号、原话片段确实是召回入口时可以保留，但不要输出秘密、凭据或工具调用内容。
8. 只输出 JSON，不要 Markdown，不要解释。"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                **self._completion_options(max_tokens=700, temperature=0.0),
            )
            raw = response.choices[0].message.content if response.choices else ""
            cleaned = str(raw or "").strip()
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned).strip()
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else {}
        except Exception as e:
            logger.warning(f"Generate shadow memory card failed / 影子记忆卡生成失败: {e}")
            return {}

    # ---------------------------------------------------------
    # Long-note memory digest: split selected durable notes into independent entries
    # 长内容摘记：把筛选后的长期记忆片段拆成多个独立条目
    # For the "grow" tool — do not feed whole day-end diaries by default.
    # 给 grow 工具用；不要默认把整篇日终日记丢进来。
    # ---------------------------------------------------------
    async def digest(self, content: str) -> list[dict]:
        """
        Split a large chunk of daily content into independent memory entries.
        将一大段日常内容拆分成多个独立记忆条目。

        Returns: [{"name", "content", "domain", "valence", "arousal", "tags",
        "importance", "memory_subject", "memory_layer", "memory_classification_source"}, ...]
        """
        if not content or not content.strip():
            return []

        # --- API digest (no local fallback) ---
        if not self.api_available:
            raise RuntimeError("脱水 API 不可用，请检查 config.yaml 中的 dehydration 配置")
        try:
            result = await self._api_digest(content)
            if result:
                return result
            raise RuntimeError("API 日记整理返回空结果")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"API 长内容摘记失败，请检查 API 连接: {e}") from e

    # ---------------------------------------------------------
    # API call: diary digest
    # API 调用：日记整理
    # ---------------------------------------------------------
    async def _api_digest(self, content: str) -> list[dict]:
        """
        Call LLM API for diary organization.
        调用 LLM API 执行日记整理。
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _render_dehydrator_template(DIGEST_PROMPT_TEMPLATE, self.identity)},
                {"role": "user", "content": content[:5000]},
            ],
            **self._completion_options(
                max_tokens=2048,
                temperature=0.0,
            ),
        )
        if not response.choices:
            return []
        raw = response.choices[0].message.content or ""
        if not raw.strip():
            return []
        return self._parse_digest(raw)

    # ---------------------------------------------------------
    # Parse diary digest result with safety checks
    # 解析日记整理结果，做安全校验
    # ---------------------------------------------------------
    def _parse_digest(self, raw: str) -> list[dict]:
        """
        Parse and validate API diary digest result.
        解析并校验 API 返回的日记整理结果。
        """
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
            items = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError, ValueError):
            logger.warning(f"Memory digest JSON parse failed / JSON 解析失败: {raw[:200]}")
            return []

        if not isinstance(items, list):
            return []

        validated = []
        for item in items:
            if not isinstance(item, dict) or not item.get("content"):
                continue
            try:
                importance = max(1, min(10, int(item.get("importance", 5))))
            except (ValueError, TypeError):
                importance = 5
            try:
                valence = max(0.0, min(1.0, float(item.get("valence", 0.5))))
                arousal = max(0.0, min(1.0, float(item.get("arousal", 0.3))))
            except (ValueError, TypeError):
                valence, arousal = 0.5, 0.3

            tags = _sanitize_generated_tags(item.get("tags", []))
            content = str(item.get("content", ""))
            classification = normalize_write_classification(
                memory_subject=item.get("memory_subject", ""),
                memory_layer=item.get("memory_layer", ""),
                tags=tags,
                content=content,
            )

            validated.append({
                "name": str(item.get("name", ""))[:20],
                "content": content,
                "domain": self._normalize_domains(item.get("domain", ["general"])),
                "valence": valence,
                "arousal": arousal,
                "tags": tags,
                "importance": importance,
                **classification,
            })
        return validated

    @staticmethod
    def _normalize_domains(value: Any) -> list[str]:
        if value is None:
            raw = []
        elif isinstance(value, str):
            raw = [item.strip() for item in value.split(",")]
        elif isinstance(value, (list, tuple, set)):
            raw = list(value)
        else:
            raw = [value]
        domains: list[str] = []
        for item in raw:
            domain = normalize_domain_key(item)
            if domain and domain not in domains:
                domains.append(domain)
        return domains[:2] or ["general"]

    def _completion_options(self, *, max_tokens: int, temperature: float) -> dict[str, Any]:
        options: dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self.thinking_mode:
            options["extra_body"] = {"thinking": {"type": self.thinking_mode}}
        return options

    def _normalize_thinking_mode(self, value: Any) -> str:
        normalized = str(value or "").strip().lower()
        aliases = {
            "enabled": "enabled",
            "enable": "enabled",
            "on": "enabled",
            "true": "enabled",
            "disabled": "disabled",
            "disable": "disabled",
            "off": "disabled",
            "false": "disabled",
            "non-thinking": "disabled",
            "non_thinking": "disabled",
        }
        return aliases.get(normalized, "")
