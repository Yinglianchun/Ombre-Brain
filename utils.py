# ============================================================
# Module: Common Utilities (utils.py)
# 模块：通用工具函数
#
# Provides config loading, logging init, path safety, ID generation, etc.
# 提供配置加载、日志初始化、路径安全校验、ID 生成等基础能力
#
# Depended on by: server.py, bucket_manager.py, dehydrator.py, decay_engine.py
# 被谁依赖：server.py, bucket_manager.py, dehydrator.py, decay_engine.py
# ============================================================

import os
import re
import uuid
import yaml
import logging
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


LOCAL_TZ = ZoneInfo("Asia/Shanghai")


def _date_hint(year: int, month: int, day: int, label: str, tz=LOCAL_TZ) -> dict[str, str] | None:
    try:
        target = datetime(year, month, day, tzinfo=tz).date()
    except ValueError:
        return None
    return {"date": target.isoformat(), "label": label}


def _reference_now(now: datetime | None = None, tz=LOCAL_TZ) -> datetime:
    if now is None:
        return datetime.now(tz)
    if now.tzinfo is None:
        return now.replace(tzinfo=tz)
    return now.astimezone(tz)


def parse_human_date_reference(text: str, *, now: datetime | None = None, tz=LOCAL_TZ) -> dict[str, str] | None:
    """Parse common human date references into YYYY-MM-DD."""
    value = str(text or "").strip()
    if not value:
        return None
    base = _reference_now(now, tz)

    explicit = re.search(
        r"(?<!\d)(20\d{2})\s*(?:[-/.]|年)\s*(\d{1,2})\s*(?:[-/.]|月)\s*(\d{1,2})\s*(?:日|号)?(?!\d)",
        value,
    )
    if explicit:
        year, month, day = (int(part) for part in explicit.groups())
        return _date_hint(year, month, day, explicit.group(0), tz)

    short_year = re.search(
        r"(?<!\d)(\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*(?:日|号)?",
        value,
    )
    if short_year:
        year, month, day = (int(part) for part in short_year.groups())
        return _date_hint(2000 + year, month, day, short_year.group(0), tz)

    month_day = re.search(r"(?<![\d年/-])(\d{1,2})\s*月\s*(\d{1,2})\s*(?:日|号)?", value)
    if month_day:
        month, day = (int(part) for part in month_day.groups())
        return _date_hint(base.year, month, day, month_day.group(0), tz)

    relative_days = [
        ("大前天", -3),
        ("前天", -2),
        ("昨晚", -1),
        ("昨天", -1),
        ("昨日", -1),
        ("今晚", 0),
        ("今天", 0),
    ]
    for label, offset in relative_days:
        if label in value:
            return {"date": (base + timedelta(days=offset)).date().isoformat(), "label": label}
    return None


def strip_human_date_references(text: str) -> str:
    """Remove human date references from a query before topic extraction."""
    value = str(text or "")
    patterns = [
        r"(?<!\d)20\d{2}\s*(?:[-/.]|年)\s*\d{1,2}\s*(?:[-/.]|月)\s*\d{1,2}\s*(?:日|号)?(?!\d)",
        r"(?<!\d)\d{2}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*(?:日|号)?",
        r"(?<![\d年/-])\d{1,2}\s*月\s*\d{1,2}\s*(?:日|号)?",
    ]
    for pattern in patterns:
        value = re.sub(pattern, " ", value)
    for label in ("大前天", "前天", "昨晚", "昨天", "昨日", "今晚", "今天"):
        value = value.replace(label, " ")
    return value


def local_date_key(value, *, tz=LOCAL_TZ) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        explicit_hint = parse_human_date_reference(text, tz=tz)
        if explicit_hint and not any(label in text for label in ("大前天", "前天", "昨晚", "昨天", "昨日", "今晚", "今天")):
            return explicit_hint["date"]
        match = re.match(r"^\d{4}-\d{2}-\d{2}", text)
        return match.group(0) if match else ""
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(tz)
    return parsed.date().isoformat()


def load_config(config_path: str = None) -> dict:
    """
    Load configuration file.
    加载配置文件。

    Priority: environment variables > config.yaml > built-in defaults.
    优先级：环境变量 > config.yaml > 内置默认值。
    """
    # --- Built-in defaults (fallback so it runs even without config.yaml) ---
    # --- 内置默认配置（兜底，保证即使没有 config.yaml 也能跑）---
    defaults = {
        "transport": "stdio",
        "log_level": "INFO",
        "buckets_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "buckets"),
        "state_dir": "",
        "merge_threshold": 90,
        "import": {
            "chunk_target_tokens": 3500,
            "extract_max_input_chars": 0,
            "max_items_per_chunk": 5,
            "max_tags": 6,
            "max_tag_chars": 12,
        },
        "write_path": {
            "semantic_search_timeout_seconds": 3,
        },
        "identity": {
            "ai_name": "AI",
            "user_name": "User",
            "user_display_name": "用户",
            "user_aliases": ["对方"],
        },
        "dehydration": {
            "model": "deepseek-chat",
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "",
            "thinking_mode": "",
            "max_tokens": 1024,
            "temperature": 0.1,
        },
        "embedding": {
            "enabled": True,
            "model": "Qwen/Qwen3-Embedding-4B",
            "base_url": "https://api.siliconflow.cn/v1",
            "api_key": "",
            "max_chars": 6000,
            "query_instruction": "Given a memory search query, retrieve relevant long-term memory passages.",
            "document_instruction": "",
        },
        "reranker": {
            "enabled": True,
            "model": "Qwen/Qwen3-Reranker-4B",
            "base_url": "",
            "api_key": "",
            "candidate_limit": 20,
            "score_weight": 0.65,
            "timeout_seconds": 12,
        },
        "recall_diagnostics": {
            "enabled": False,
            "path": "",
            "max_candidates": 20,
            "max_text_chars": 220,
        },
        "recall_thresholds": {
            "vector_min_score": 0.50,
            "facet_vector_min_score": 0.45,
            "vague_vector_min_score": 0.40,
            "explicit_vector_min_score": 0.55,
            "vague_top_k": 50,
        },
        "word_map": {
            "enabled": False,
            "daily_rebuild_enabled": True,
            "daily_rebuild_hour": 4,
            "daily_rebuild_minute": 30,
            "daily_rebuild_include_archive": False,
            "daily_rebuild_check_interval_minutes": 15,
            "max_terms_per_bucket": 16,
            "edge_top_k": 10,
            "min_term_len": 2,
            "stopwords": [],
            "private_terms": [],
            "stopword_prefixes": [],
        },
        "raw_events": {
            "db_path": "",
            "max_ingest_batch": 1000,
        },
        "identity_semantics": {
            "enabled": False,
            "private_config_path": "",
            "min_confidence": 0.78,
            "evidence_tags": ["profile_fact", "ai_favorite", "favorite_memory"],
        },
        "moment_annotations": {
            "enabled": True,
            "max_summary_chars": 160,
            "max_evidence_spans": 3,
            "max_evidence_chars": 120,
        },
        "decay": {
            "lambda": 0.05,
            "threshold": 0.3,
            "check_interval_hours": 24,
            "emotion_weights": {
                "base": 1.0,
                "arousal_boost": 0.0,
            },
        },
        "matching": {
            "fuzzy_threshold": 50,
            "max_results": 5,
        },
        "anchor": {
            "max_count": 24,
            "min_age_hours": 24,
        },
        "node_facets": {
            "enabled": True,
            "store": "sqlite",
            "salience_min": 0.2,
            "salience_max": 1.3,
        },
        "memory_relevance": {
            "aliases": {
                "relationship_identity": [
                    "human-ai relationship",
                    "ai relationship",
                    "人机恋",
                    "人机关系",
                ],
                "intimacy": ["intimacy"],
                "embodiment": ["embodiment", "physical body", "具身", "身体", "形体"],
                "hardware_protocol": ["hardware", "protocol", "ble", "硬件", "协议"],
                "communication_action": ["email", "mail", "message", "发邮件", "邮件", "发消息"],
                "old_or_resolved": ["legacy", "deprecated", "resolved", "旧版", "废弃", "已解决"],
            },
            "blocked_facets": [],
            "section_hints": {},
        },
        "gateway": {
            "host": "0.0.0.0",
            "port": 8010,
            "default_session_id": "xiaoyu-main",
            "upstream_base_url": "",
            "upstream_default_model": "",
            "upstream_models": [],
            "upstreams": [],
            "head_recent_hours": 72,
            "dynamic_top_k": 10,
            "inject_max_cards": 2,
            "skip_recent_rounds": 5,
            "cooldown_hours": 6,
            "cooldown_floor": 0.3,
            "semantic_session_dedupe_enabled": True,
            "semantic_session_dedupe_threshold": 0.90,
            "semantic_session_dedupe_lexical_threshold": 0.82,
            "memory_sentinel_enabled": True,
            "domain_sentinel_enabled": True,
            "domain_sentinel_model": "Qwen/Qwen3.5-4B",
            "domain_sentinel_max_tokens": 260,
            "domain_sentinel_enable_thinking": False,
            "inject_total_budget": 1200,
            "core_memory_budget": 0,
            "recent_context_budget": 300,
            "recalled_memory_budget": 400,
            "direct_render_mode": "auto",
            "bucket_list_cache_ttl_seconds": 300,
            "portrait_memory_enabled": False,
            "portrait_memory_budget": 360,
            "portrait_memory_max_sources": 8,
            "portrait_memory_include_anchors": False,
            "relationship_weather_budget": 220,
            "favorite_memory_budget": 0,
            "favorite_memory_max_cards": 1,
            "related_memory_budget": 220,
            "operit_context_rewrite_enabled": False,
            "active_reminders_enabled": True,
            "active_reminder_inject_limit": 2,
            "core_memory_interval_rounds": 0,
            "current_inner_state_interval_rounds": 15,
            "relationship_weather_interval_rounds": 0,
            "favorite_memory_interval_rounds": 0,
            "semantic_weight": 0.45,
            "keyword_weight": 0.35,
            "importance_weight": 0.1,
            "freshness_weight": 0.1,
            "first_card_min_score": 0.55,
            "second_card_min_score": 0.50,
            "second_card_relative_score": 0.85,
            "high_confidence_semantic_score": 0.72,
            "high_confidence_keyword_score": 0.65,
            "high_confidence_cooldown_floor": 0.8,
        },
        "self_anchor": {
            "entry_bucket_id": "",
        },
        "persona": {
            "enabled": True,
            "profile_id": "haven_xiaoyu",
            "mode": "llm",
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat",
            "api_key": "",
            "thinking_mode": "",
            "temperature": 0.1,
            "max_tokens": 500,
            "global_decay_hours": 168,
            "session_mood_half_life_minutes": 90,
            "max_personality_delta": 0.01,
            "max_relationship_delta": 0.03,
            "max_affect_delta": 0.18,
            "event_batch_size": 2,
            "event_affect_total_threshold": 0.45,
            "event_affect_single_threshold": 0.14,
            "event_similarity_threshold": 0.82,
            "event_force_after_minutes": 30,
            "initial_personality": {
                "openness": 0.56,
                "conscientiousness": 0.50,
                "extraversion": 0.44,
                "agreeableness": 0.66,
                "neuroticism": 0.36,
            },
            "initial_relationship": {
                "affinity": 0.86,
                "dominance": 0.38,
                "defensiveness": 0.12,
                "trust": 0.82,
            },
            "initial_affect": {
                "valence": 0.56,
                "arousal": 0.34,
                "tenderness": 0.62,
                "possessiveness": 0.24,
                "longing": 0.34,
                "security": 0.68,
                "protective_drive": 0.52,
                "mood_label": "warm_neutral",
                "session_defensiveness": 0.12,
                "residue": "",
            },
        },
        "reflection": {
            "enabled": True,
            "auto_enabled": True,
            "daily_enabled": True,
            "enrich_on_write": True,
            "memory_affect_anchor_enabled": True,
            "relationship_weather_affect_anchor_enabled": True,
            "enrich_backfill_enabled": True,
            "enrich_backfill_limit": 5,
            "edge_backfill_limit": 5,
            "base_url": "",
            "model": "",
            "api_key": "",
            "thinking_mode": "",
            "temperature": 0.1,
            "max_tokens": 700,
            "timezone": "Asia/Shanghai",
            "daily_hour": 4,
            "daily_min_memory_items": 5,
            "daily_conversation_turn_limit": 12,
            "weekly_day": 0,
            "weekly_hour": 4,
            "check_interval_minutes": 60,
            "candidate_limit": 18,
            "candidate_recent_limit": 8,
            "candidate_semantic_limit": 6,
            "edge_min_confidence": 0.55,
            "diary_mcp_url": "",
            "diary_mcp_token_env": "",
            "diary_memory_extract_enabled": True,
            "diary_memory_extract_max_per_day": 1,
            "diary_memory_extract_min_confidence": 0.68,
            "daily_chat_memory_mode": "review",
            "daily_chat_memory_hour": 0,
            "daily_chat_memory_turn_limit": 0,
            "daily_chat_memory_max_per_day": 10,
            "daily_chat_memory_min_confidence": 0.68,
            "daily_chat_memory_review_max_per_day": 10,
            "daily_chat_memory_review_min_confidence": 0.55,
            "daily_chat_memory_summary_enabled": True,
            "daily_chat_memory_summary_window_turns": 14,
            "daily_chat_memory_summary_stride_turns": 7,
            "daily_chat_memory_api_key_env": "",
            "daily_chat_memory_base_url": "",
            "daily_chat_memory_timeout_seconds": 180,
            "daily_chat_memory_summary_model": "",
            "daily_chat_memory_summary_max_tokens": 2200,
            "daily_chat_memory_candidate_model": "",
            "daily_chat_memory_candidate_max_tokens": 3200,
            "daily_activity_summary_enabled": True,
            "daily_activity_summary_turn_limit": 0,
            "daily_activity_summary_max_tokens": 320,
        },
        "portrait": {
            "enabled": True,
            "auto_enabled": True,
            "auto_initial_enabled": False,
            "daily_enabled": True,
            "timezone": "Asia/Shanghai",
            "daily_hour": 4,
            "check_interval_minutes": 60,
            "state_path": "",
            "base_url": "",
            "model": "",
            "api_key": "",
            "thinking_mode": "",
            "temperature": 0.1,
            "max_tokens": 1800,
            "material_limit": 18,
            "first_run_material_limit": 160,
            "persona_events_limit": 24,
            "recent_buffer_max": 24,
            "staging_pool_max": 24,
            "candidate_max": 40,
            "stable_history_max": 20,
            "current_focus_days": 7,
        },
        "dream": {
            "enabled": True,
            "auto_enabled": True,
            "surface_enabled": True,
            "inject_enabled": False,
            "retain_after_inject": True,
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-v4-flash",
            "api_key": "",
            "thinking_mode": "disabled",
            "temperature": 0.85,
            "max_tokens": 900,
            "timezone": "Asia/Shanghai",
            "daily_hour": 3,
            "run_window_hours": 3,
            "daily_probability": 0.4,
            "check_interval_minutes": 60,
            "min_material_count": 5,
            "material_window_hours": 48,
            "material_limit": 5,
            "old_echo_enabled": True,
            "old_echo_min_age_hours": 72,
            "raw_residue_enabled": False,
            "raw_residue_turns": 4,
            "raw_residue_max_chars": 1500,
            "identity_anchor_id": "c0b8ddb7423e",
            "min_surface_age_hours": 3,
            "surface_threshold": 0.62,
            "attempt_threshold": 0.45,
            "alpha_subordinate": 0.25,
            "spontaneous_surface_prob": 0.02,
            "max_surface_attempts": 4,
            "claim_ttl_minutes": 15,
        },
    }

    # --- Load user config from YAML file ---
    # --- 从 YAML 文件加载用户自定义配置 ---
    if config_path is None:
        config_path = os.environ.get(
            "OMBRE_CONFIG_PATH",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"),
        )

    config = defaults.copy()
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                file_config = yaml.safe_load(f) or {}
            if isinstance(file_config, dict):
                config = _deep_merge(defaults, file_config)
            else:
                logging.warning(
                    f"Config file is not a valid YAML dict, using defaults / "
                    f"配置文件不是有效的 YAML 字典，使用默认配置: {config_path}"
                )
        except yaml.YAMLError as e:
            logging.warning(
                f"Failed to parse config file, using defaults / "
                f"配置文件解析失败，使用默认配置: {e}"
            )

    env_buckets_dir_early = os.environ.get("OMBRE_BUCKETS_DIR", "")
    if env_buckets_dir_early:
        config["buckets_dir"] = env_buckets_dir_early
    env_state_dir_early = os.environ.get("OMBRE_STATE_DIR", "")
    if env_state_dir_early:
        config["state_dir"] = env_state_dir_early

    runtime_config_path = os.environ.get("OMBRE_RUNTIME_CONFIG_PATH", "")
    if not runtime_config_path:
        runtime_state_dir = config.get("state_dir") or os.path.join(
            os.path.dirname(os.path.abspath(config["buckets_dir"])),
            "state",
        )
        runtime_config_path = os.path.join(runtime_state_dir, "config.runtime.yaml")
    config["_runtime_config_path"] = runtime_config_path
    if os.path.exists(runtime_config_path):
        try:
            with open(runtime_config_path, "r", encoding="utf-8") as f:
                runtime_config = yaml.safe_load(f) or {}
            if isinstance(runtime_config, dict):
                config = _deep_merge(config, runtime_config)
                config["_runtime_config_path"] = runtime_config_path
        except yaml.YAMLError as e:
            logging.warning(
                f"Failed to parse runtime config, ignoring / "
                f"运行时配置解析失败，已忽略: {e}"
            )

    # --- Environment variable overrides (highest priority) ---
    # --- 环境变量覆盖敏感/运行时配置（优先级最高）---
    env_api_key = os.environ.get("OMBRE_API_KEY", "")
    if env_api_key:
        config.setdefault("dehydration", {})["api_key"] = env_api_key

    env_base_url = os.environ.get("OMBRE_BASE_URL", "")
    if env_base_url:
        config.setdefault("dehydration", {})["base_url"] = env_base_url

    env_dehydration_base_url = os.environ.get("OMBRE_DEHYDRATION_BASE_URL", "")
    if env_dehydration_base_url:
        config.setdefault("dehydration", {})["base_url"] = env_dehydration_base_url

    env_dehydration_model = os.environ.get("OMBRE_DEHYDRATION_MODEL", "") or os.environ.get("OMBRE_MODEL", "")
    if env_dehydration_model:
        config.setdefault("dehydration", {})["model"] = env_dehydration_model

    env_embedding_api_key = os.environ.get("OMBRE_EMBEDDING_API_KEY", "")
    if env_embedding_api_key:
        config.setdefault("embedding", {})["api_key"] = env_embedding_api_key

    env_embedding_base_url = os.environ.get("OMBRE_EMBEDDING_BASE_URL", "")
    if env_embedding_base_url:
        config.setdefault("embedding", {})["base_url"] = env_embedding_base_url

    env_embedding_model = os.environ.get("OMBRE_EMBEDDING_MODEL", "")
    if env_embedding_model:
        config.setdefault("embedding", {})["model"] = env_embedding_model

    env_embedding_enabled = os.environ.get("OMBRE_EMBEDDING_ENABLED", "")
    if env_embedding_enabled:
        config.setdefault("embedding", {})["enabled"] = env_embedding_enabled.lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    env_embedding_max_chars = os.environ.get("OMBRE_EMBEDDING_MAX_CHARS", "")
    if env_embedding_max_chars:
        try:
            config.setdefault("embedding", {})["max_chars"] = int(env_embedding_max_chars)
        except ValueError:
            logging.warning(
                f"Invalid OMBRE_EMBEDDING_MAX_CHARS / 无效的 OMBRE_EMBEDDING_MAX_CHARS: {env_embedding_max_chars}"
            )

    env_embedding_query_instruction = os.environ.get("OMBRE_EMBEDDING_QUERY_INSTRUCTION", "")
    if env_embedding_query_instruction:
        config.setdefault("embedding", {})["query_instruction"] = env_embedding_query_instruction

    env_reranker_api_key = os.environ.get("OMBRE_RERANKER_API_KEY", "")
    if env_reranker_api_key:
        config.setdefault("reranker", {})["api_key"] = env_reranker_api_key

    env_reranker_base_url = os.environ.get("OMBRE_RERANKER_BASE_URL", "")
    if env_reranker_base_url:
        config.setdefault("reranker", {})["base_url"] = env_reranker_base_url

    env_reranker_model = os.environ.get("OMBRE_RERANKER_MODEL", "")
    if env_reranker_model:
        config.setdefault("reranker", {})["model"] = env_reranker_model

    env_reranker_enabled = os.environ.get("OMBRE_RERANKER_ENABLED", "")
    if env_reranker_enabled:
        config.setdefault("reranker", {})["enabled"] = env_reranker_enabled.lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    env_recall_diagnostics_enabled = os.environ.get("OMBRE_RECALL_DIAGNOSTICS_ENABLED", "")
    if env_recall_diagnostics_enabled:
        config.setdefault("recall_diagnostics", {})["enabled"] = env_recall_diagnostics_enabled.lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    env_recall_diagnostics_path = os.environ.get("OMBRE_RECALL_DIAGNOSTICS_PATH", "")
    if env_recall_diagnostics_path:
        config.setdefault("recall_diagnostics", {})["path"] = env_recall_diagnostics_path

    env_recall_diagnostics_max_candidates = os.environ.get("OMBRE_RECALL_DIAGNOSTICS_MAX_CANDIDATES", "")
    if env_recall_diagnostics_max_candidates:
        try:
            config.setdefault("recall_diagnostics", {})["max_candidates"] = int(env_recall_diagnostics_max_candidates)
        except ValueError:
            logging.warning(
                "Invalid OMBRE_RECALL_DIAGNOSTICS_MAX_CANDIDATES / "
                f"无效的 OMBRE_RECALL_DIAGNOSTICS_MAX_CANDIDATES: {env_recall_diagnostics_max_candidates}"
            )

    env_transport = os.environ.get("OMBRE_TRANSPORT", "")
    if env_transport:
        config["transport"] = env_transport

    env_buckets_dir = os.environ.get("OMBRE_BUCKETS_DIR", "")
    if env_buckets_dir:
        config["buckets_dir"] = env_buckets_dir

    env_state_dir = os.environ.get("OMBRE_STATE_DIR", "")
    if env_state_dir:
        config["state_dir"] = env_state_dir

    env_gateway_host = os.environ.get("OMBRE_GATEWAY_HOST", "")
    if env_gateway_host:
        config.setdefault("gateway", {})["host"] = env_gateway_host

    env_gateway_port = os.environ.get("OMBRE_GATEWAY_PORT", "")
    if env_gateway_port:
        try:
            config.setdefault("gateway", {})["port"] = int(env_gateway_port)
        except ValueError:
            logging.warning(
                f"Invalid OMBRE_GATEWAY_PORT / 无效的 OMBRE_GATEWAY_PORT: {env_gateway_port}"
            )

    env_gateway_base_url = os.environ.get("OMBRE_GATEWAY_UPSTREAM_BASE_URL", "")
    if env_gateway_base_url:
        config.setdefault("gateway", {})["upstream_base_url"] = env_gateway_base_url

    env_gateway_model = os.environ.get("OMBRE_GATEWAY_UPSTREAM_MODEL", "")
    if env_gateway_model:
        config.setdefault("gateway", {})["upstream_default_model"] = env_gateway_model

    env_gateway_models = os.environ.get("OMBRE_GATEWAY_UPSTREAM_MODELS", "")
    if env_gateway_models:
        config.setdefault("gateway", {})["upstream_models"] = [
            item.strip()
            for item in env_gateway_models.split(",")
            if item.strip()
        ]

    env_domain_sentinel_model = os.environ.get("OMBRE_DOMAIN_SENTINEL_MODEL", "")
    if env_domain_sentinel_model:
        config.setdefault("gateway", {})["domain_sentinel_model"] = env_domain_sentinel_model

    env_persona_api_key = os.environ.get("OMBRE_PERSONA_API_KEY", "")
    if env_persona_api_key:
        config.setdefault("persona", {})["api_key"] = env_persona_api_key

    env_persona_base_url = os.environ.get("OMBRE_PERSONA_BASE_URL", "")
    if env_persona_base_url:
        config.setdefault("persona", {})["base_url"] = env_persona_base_url

    env_persona_model = os.environ.get("OMBRE_PERSONA_MODEL", "")
    if env_persona_model:
        config.setdefault("persona", {})["model"] = env_persona_model

    env_reflection_api_key = os.environ.get("OMBRE_REFLECTION_API_KEY", "")
    if env_reflection_api_key:
        config.setdefault("reflection", {})["api_key"] = env_reflection_api_key

    env_reflection_base_url = os.environ.get("OMBRE_REFLECTION_BASE_URL", "")
    if env_reflection_base_url:
        config.setdefault("reflection", {})["base_url"] = env_reflection_base_url

    env_reflection_model = os.environ.get("OMBRE_REFLECTION_MODEL", "")
    if env_reflection_model:
        config.setdefault("reflection", {})["model"] = env_reflection_model

    env_reflection_candidate_model = os.environ.get("OMBRE_REFLECTION_CANDIDATE_MODEL", "")
    if env_reflection_candidate_model:
        config.setdefault("reflection", {})["daily_chat_memory_candidate_model"] = env_reflection_candidate_model

    env_diary_mcp_url = os.environ.get("OMBRE_DIARY_MCP_URL", "")
    if env_diary_mcp_url:
        config.setdefault("reflection", {})["diary_mcp_url"] = env_diary_mcp_url

    env_diary_mcp_token_env = os.environ.get("OMBRE_DIARY_MCP_TOKEN_ENV", "")
    if env_diary_mcp_token_env:
        config.setdefault("reflection", {})["diary_mcp_token_env"] = env_diary_mcp_token_env

    env_dream_api_key = os.environ.get("OMBRE_DREAM_API_KEY", "")
    if env_dream_api_key:
        config.setdefault("dream", {})["api_key"] = env_dream_api_key

    env_dream_base_url = os.environ.get("OMBRE_DREAM_BASE_URL", "")
    if env_dream_base_url:
        config.setdefault("dream", {})["base_url"] = env_dream_base_url

    env_dream_model = os.environ.get("OMBRE_DREAM_MODEL", "")
    if env_dream_model:
        config.setdefault("dream", {})["model"] = env_dream_model

    env_dream_enabled = os.environ.get("OMBRE_DREAM_ENABLED", "")
    if env_dream_enabled:
        config.setdefault("dream", {})["enabled"] = env_dream_enabled.lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    # --- Ensure bucket storage directories exist ---
    # --- 确保记忆桶存储目录存在 ---
    buckets_dir = config["buckets_dir"]
    if not config.get("state_dir"):
        config["state_dir"] = os.path.join(os.path.dirname(os.path.abspath(buckets_dir)), "state")
    os.makedirs(config["state_dir"], exist_ok=True)
    for subdir in ["permanent", "dynamic", "archive", "feel"]:
        os.makedirs(os.path.join(buckets_dir, subdir), exist_ok=True)

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Deep-merge two dicts; override values take precedence.
    深度合并两个字典，override 的值覆盖 base。
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def setup_logging(level: str = "INFO") -> None:
    """
    Initialize logging system.
    初始化日志系统。

    Note: In MCP stdio mode, stdout is occupied by the protocol;
    logs must go to stderr.
    注意：MCP stdio 模式下 stdout 被协议占用，日志只能走 stderr。
    """
    log_level = getattr(logging, level.upper(), None)
    if not isinstance(log_level, int):
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],  # StreamHandler defaults to stderr
    )


def generate_bucket_id() -> str:
    """
    Generate a unique bucket ID (12-char short UUID for readability).
    生成唯一的记忆桶 ID（12 位短 UUID，方便人类阅读）。
    """
    return uuid.uuid4().hex[:12]


def strip_wikilinks(text: str) -> str:
    """
    Remove Obsidian wikilink brackets: [[word]] → word
    去除 Obsidian 双链括号
    """
    return re.sub(r"\[\[([^\]]+)\]\]", r"\1", text) if text else text


_AFFECT_ANCHOR_RE = re.compile(r"(?ims)^###\s*affect_anchor\s*$.*?(?=^###\s+|\Z)")
_FOLLOWUP_SECTION_RE = re.compile(
    r"(?ims)^#{2,6}\s*(?:followup|followups|follow-up|followup_log|followups_log|followup-log|todo|to-do|todo_log|todo-log|next|后续|后续待办|后续记录|待办|待办事项|待办记录)\s*$.*?(?=^#{2,6}\s+|\Z)"
)
_DISPLAY_TEMPERATURE_SECTION_RE = re.compile(
    r"(?ims)^###\s*(?:affect_anchor|affect anchor|喜欢它的原因|favorite_reason|favorite reason)\s*$.*?(?=^###\s+|\Z)"
)
_TEMPERATURE_MEANING_LINE_RE = re.compile(r"(?m)^\s*含义[:：].*(?:\n|$)")
_CHORD_TOKEN_RE = re.compile(
    r"\b[A-G](?:#|b)?(?:maj|min|m|dim|aug)?\d*(?:sus\d*|add\d*|b\d+|#\d+)*(?:/[A-G](?:#|b)?)?\b"
)
_TEMPERATURE_MUSIC_TOKEN_RE = re.compile(r"\b(?:\d{2,3}\s*bpm|ppp|pp|mp|mf|ff|fff|p|f|add\s*\d+|sus\s*\d+)\b", re.I)


def _looks_like_temperature_chord_line(line: str) -> bool:
    text = str(line or "").strip()
    if not text:
        return False
    if text.startswith(">"):
        text = text[1:].strip()
    if not text or re.search(r"[\u4e00-\u9fff]", text):
        return False
    if not any(marker in text for marker in ("->", "→", "|", "·")) and "bpm" not in text.lower():
        return False
    if not _CHORD_TOKEN_RE.search(text):
        return False
    remainder = _CHORD_TOKEN_RE.sub("", text)
    remainder = _TEMPERATURE_MUSIC_TOKEN_RE.sub("", remainder)
    remainder = re.sub(r"[-→>·|/(),.:;_\s]+", "", remainder)
    return not remainder


def _strip_inline_temperature_chord_segments(line: str) -> str:
    match = re.search(r"\s>\s*(.+)$", str(line or ""))
    if match and _looks_like_temperature_chord_line(">" + match.group(1)):
        return str(line)[: match.start()].rstrip()
    return line


def strip_affect_anchor(text: str) -> str:
    """Remove the display-only affect anchor block from searchable text."""
    if not text:
        return text
    return _AFFECT_ANCHOR_RE.sub("", str(text)).strip()


def strip_followup_sections(text: str) -> str:
    """Remove followup/todo blocks from ordinary recall text."""
    if not text:
        return text
    return _FOLLOWUP_SECTION_RE.sub("", str(text)).strip()


def bucket_content_for_recall(bucket: dict) -> str:
    """Build bucket body text for ordinary recall/search, excluding task-only blocks."""
    if not isinstance(bucket, dict):
        return ""
    text = strip_wikilinks(str(bucket.get("content") or ""))
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    if str(meta.get("memory_value_source") or "") == "authored_scene":
        text = _strip_legacy_scene_wrapper(text)
    text = strip_affect_anchor(text)
    return strip_followup_sections(text).strip()


def _strip_legacy_scene_wrapper(text: str) -> str:
    """Read old authored Scenes as prose without rewriting their stored file."""
    raw = str(text or "").strip()
    return re.sub(
        r"\A\s{0,3}#{2,6}\s+(?:scene|场景)(?:\s*[:：|｜-]\s*[^\n]*)?\s*(?:\r?\n)+",
        "",
        raw,
        count=1,
        flags=re.IGNORECASE,
    ).strip()


def strip_display_temperature_sections(text: str) -> str:
    """Remove display-only temperature sections from direct bucket rendering."""
    if not text:
        return text
    return _DISPLAY_TEMPERATURE_SECTION_RE.sub("", str(text)).strip()


def strip_temperature_meaning_lines(text: str) -> str:
    """Remove template-like affect-anchor meaning and chord lines from rendered context."""
    if not text:
        return text
    cleaned = _TEMPERATURE_MEANING_LINE_RE.sub("", str(text))
    lines = []
    for line in cleaned.splitlines():
        line = _strip_inline_temperature_chord_segments(line)
        if _looks_like_temperature_chord_line(line):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def normalize_scene_cues(value: object, *, limit: int = 8, max_chars: int = 80) -> list[str]:
    """Normalize sidecar recall entrances without turning them into memory prose."""
    if isinstance(value, str):
        raw_values = re.split(r"[\r\n|｜]+", value)
    elif isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    else:
        raw_values = []

    generic = {
        "以前",
        "过去",
        "关系",
        "记忆",
        "事情",
        "开心",
        "难过",
        "情绪",
        "something",
        "memory",
    }
    cues: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        cue = re.sub(r"\s+", " ", str(raw or "")).strip(" \t\r\n-—*•、，,。.!！?？:：;；\"'“”‘’")
        if not cue:
            continue
        cue = cue[: max(1, int(max_chars))].rstrip()
        key = re.sub(r"[\s\W_]+", "", cue.lower())
        if len(key) < 2 or key in generic or key in seen:
            continue
        seen.add(key)
        cues.append(cue)
        if len(cues) >= max(1, int(limit)):
            break
    return cues


def bucket_text_for_embedding(bucket: dict) -> str:
    """
    Build the text sent to the embedding model for a bucket.
    Legacy buckets may include title/cues; canonical Scene vectors are body-only.
    """
    if not isinstance(bucket, dict):
        return ""

    meta = bucket.get("metadata", {})
    if not isinstance(meta, dict):
        meta = {}

    # A Scene vector represents only the verbatim experience. Title and cues
    # remain sparse lookup fields and must not pull the semantic vector away
    # from scene.content.  Do not run stored Scene prose through recall display
    # cleanup: even removing Obsidian brackets would stop this from being the
    # exact authored text.  Only old heading-wrapped Scenes keep their read
    # compatibility cleanup.
    if str(meta.get("memory_value_source") or "") == "authored_scene":
        raw_body = str(bucket.get("content") or "")
        if re.match(
            r"\A\s{0,3}#{2,6}\s+(?:scene|场景)(?:\s*[:：|｜-]\s*[^\n]*)?\s*(?:\r?\n)+",
            raw_body,
            flags=re.IGNORECASE,
        ):
            return _strip_legacy_scene_wrapper(raw_body)
        return raw_body

    title = strip_wikilinks(str(meta.get("name") or "")).strip()
    scene_cues = normalize_scene_cues(meta.get("scene_cues"))
    body = bucket_content_for_recall(bucket)

    parts = []
    if title:
        parts.append(f"Title: {title}")
        if body:
            parts.append(f"Content: {body}")
    elif body:
        parts.append(body)
    if scene_cues:
        parts.append("Recall cues: " + " | ".join(scene_cues))

    return "\n".join(parts).strip()


def active_scene_migration_map(buckets: list[dict]) -> dict[str, list[dict]]:
    """Map preserved legacy sources to their active migrated canonical Scenes."""
    output: dict[str, list[dict]] = {}
    for bucket in buckets or []:
        if not isinstance(bucket, dict):
            continue
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        source_id = str(meta.get("migration_source_bucket_id") or "").strip()
        if not source_id or str(meta.get("memory_value_source") or "") != "authored_scene":
            continue
        if (
            meta.get("type") in {"feel", "archived"}
            or meta.get("resolved")
            or meta.get("digested")
            or meta.get("deprecated")
            or meta.get("active") is False
        ):
            continue
        output.setdefault(source_id, []).append(bucket)
    for scenes in output.values():
        scenes.sort(key=lambda item: str(item.get("id") or ""))
    return output


def migrated_scene_for_legacy_source(
    migration_map: dict[str, list[dict]],
    source_bucket_id: str,
    source_moment_id: str = "",
) -> dict | None:
    """Resolve one legacy evidence ref to its reversible migrated Scene."""
    scenes = list((migration_map or {}).get(str(source_bucket_id or "").strip()) or [])
    if not scenes:
        return None
    moment_id = str(source_moment_id or "").strip()
    if moment_id:
        matches = []
        for scene in scenes:
            meta = scene.get("metadata", {}) if isinstance(scene.get("metadata"), dict) else {}
            if str(meta.get("migration_source_moment_id") or "").strip() == moment_id:
                matches.append(scene)
        if len(matches) == 1:
            return matches[0]
        if matches:
            return matches[0]
    return scenes[0] if len(scenes) == 1 else None


def suppress_migrated_legacy_sources(buckets: list[dict]) -> list[dict]:
    """Keep source files for rollback while removing duplicate ordinary candidates."""
    migration_map = active_scene_migration_map(buckets)
    if not migration_map:
        return list(buckets or [])
    source_ids = set(migration_map)
    return [
        bucket
        for bucket in buckets or []
        if str((bucket or {}).get("id") or "") not in source_ids
    ]


def sanitize_name(name: str) -> str:
    """
    Sanitize bucket name, keeping only safe characters.
    Prevents path traversal attacks (e.g. ../../etc/passwd).
    清洗桶名称，只保留安全字符。防止路径遍历攻击。
    """
    if not isinstance(name, str):
        return "unnamed"
    cleaned = re.sub(r"[^\w\s\u4e00-\u9fff-]", "", name, flags=re.UNICODE)
    cleaned = cleaned.strip()[:80]
    return cleaned if cleaned else "unnamed"


def safe_path(base_dir: str, filename: str) -> Path:
    """
    Construct a safe file path, ensuring it stays within base_dir.
    Prevents directory traversal.
    构造安全的文件路径，确保最终路径始终在 base_dir 内部。
    """
    base = Path(base_dir).resolve()
    target = (base / filename).resolve()
    if not str(target).startswith(str(base)):
        raise ValueError(
            f"Path safety check failed / 路径安全检查失败: "
            f"{target} is not inside / 不在 {base} 内"
        )
    return target


def count_tokens_approx(text: str) -> int:
    """
    Rough token count estimate.
    粗略估算 token 数。

    Chinese ≈ 1 char = 1.5 tokens, English ≈ 1 word = 1.3 tokens.
    Used to decide whether dehydration is needed; precision not required.
    中文 ≈ 1字=1.5token，英文 ≈ 1词=1.3token。
    用于判断是否需要脱水压缩，不追求精确。
    """
    if not text:
        return 0
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    english_words = len(re.findall(r"[a-zA-Z]+", text))
    return int(chinese_chars * 1.5 + english_words * 1.3 + len(text) * 0.05)


def now_iso() -> str:
    """
    Return current time as ISO format string.
    返回当前时间的 ISO 格式字符串。
    """
    return datetime.now(LOCAL_TZ).isoformat(timespec="seconds")
