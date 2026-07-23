# ============================================================
# Module: MCP Server Entry Point (server.py)
# 模块：MCP 服务器主入口
#
# Starts the Ombre Brain MCP service and registers memory
# operation tools for Claude to call.
# 启动 Ombre Brain MCP 服务，注册记忆操作工具供 Claude 调用。
#
# Core responsibilities:
# 核心职责：
#   - Initialize config, bucket manager, dehydrator, decay engine
#     初始化配置、记忆桶管理器、脱水器、衰减引擎
#   - Expose MCP tools:
#     暴露 MCP 工具：
#       breath — Surface unresolved memories or search by keyword
#                浮现未解决记忆 或 按关键词检索
#       resurface — Surface dormant memories without touching them
#                   只读浮现久未触碰的旧记忆
#       comment_bucket — Add a ring comment to a memory
#                        给记忆追加年轮
#       delete_bucket_comment — Delete one Haven-authored ring comment
#                               删除一条 Haven 自己写的年轮
#       hold   — Store a single memory
#                存储单条记忆
#       grow   — Store one complete first-person window shadow and copy explicit moments
#                保存整窗第一人称窗影，并拆出独立场景记忆
#       trace  — Modify metadata / resolved / delete
#                修改元数据 / resolved 标记 / 删除
#       pulse  — System status + bucket listing
#                系统状态 + 所有桶列表
#       reflect — Daily relationship weather
#                 日关系天气
#       introspection — Read recent memories for waking self-reflection
#                       读取最近记忆供清醒自省
#
# Startup:
# 启动方式：
#   Local:  python server.py
#   Remote: OMBRE_TRANSPORT=streamable-http python server.py
#   Docker: docker-compose up
# ============================================================

import os
import sys
import random
import logging
import asyncio
import hashlib
import hmac
import json as _json_lib
import re
import secrets
import time
from contextvars import ContextVar
from typing import Literal
from base64 import b64decode
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlencode, urlparse
from zoneinfo import ZoneInfo
import httpx


# --- Ensure same-directory modules can be imported ---
# --- 确保同目录下的模块能被正确导入 ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import Context

from bucket_manager import BucketManager
from dehydrator import Dehydrator
from decay_engine import DecayEngine
from darkroom import DarkroomStore
from diary_sources import DiarySourceImporter
from dream_engine import DreamEngine
from embedding_engine import EmbeddingEngine
from favorite_tags import has_favorite_memory_tag, has_favorite_policy_tag
from gateway_state import GatewayStateStore
from identity import identity_names
from identity_semantics import IdentitySemanticStore
from import_memory import ImportEngine
from legacy_memory_review import LegacyMemoryReviewStore
from legacy_memory_lifecycle import LegacyMemoryLifecyclePublisher
from memory_diffusion import (
    diffuse_memory,
    diffusion_options_from_config,
    format_diffusion_path,
    format_diffusion_trace,
    path_has_caution,
    path_has_old_version,
    seed_scores_for_buckets,
    should_suppress_context_candidate,
)
from memory_edges import (
    MemoryEdgeStore,
    legacy_edges_for_recall,
    legacy_moment_edges_for_recall,
)
from entity_edges import EntityEdgeStore, extract_entity_edges_from_bucket
from memory_cards import ShadowMemoryCardStore
from memory_moments import MemoryMomentStore, parse_bucket_moments
from memory_relevance import (
    active_facets,
    emotional_recall_plan,
    facets_for_text,
    memory_relevance_options_from_config,
    query_has_explicit_entity_marker,
    recall_search_query,
    recall_rank,
    relevance_decision,
    relevance_multiplier,
)
from memory_layers import (
    CONTEXT_ONLY_SECTIONS,
    LAYER_SOURCE_RECORD,
    bucket_layer_debug,
    bucket_runtime_gate_debug,
    can_bucket_be_related_target,
    can_moment_be_direct_seed,
    can_moment_be_recall_context,
    can_moment_be_related_target,
    infer_bucket_layer,
    moment_layer_debug,
    moment_runtime_gate_debug,
    normalize_write_classification,
)
from memory_metadata import domain_options, normalize_domain_key, normalize_memory_metadata
from memory_object_policy import (
    legacy_memory_writes_enabled,
    memory_object_write_status,
    retired_write_message,
    retired_write_payload,
)
from mcp_surface import OmbreFastMCP
from recall_policy import RecallPolicy, diffusion_seed_topic_term_has_specific_residue
from window_shadows import (
    HANDOFF_NOTE_MAX_CHARS,
    HANDOFF_NOTE_MIN_CHARS,
    WindowShadowStore,
    extract_window_shadow_moments,
    extract_window_shadow_scenes,
    handoff_note_char_count,
    is_bare_window_continue_query,
    validate_window_shadow,
)
from memory_nodes import MemoryNodeStore
from narrative_rolls import NarrativeRollStore
from persona_engine import PersonaStateEngine
from persona_event_selection import select_persona_events
from portrait_engine import DailyPortraitMaintainer
from raw_events import RawEventStore
from reflection_engine import ReflectionEngine
from recall_diagnostics import RecallDiagnosticsLogger
from reminder_store import ReminderStore
from reranker_engine import RerankerEngine
from scene_linker import SceneLinker
from self_anchor import SELF_ANCHOR_TAG, is_self_anchor_bucket, is_self_anchor_metadata
from scripts.migrate_affect_anchor_sections import plan_bucket_migration
from source_refs import source_ref_window
from word_map import WordMapStore, reflection_identity_terms
from utils import (
    active_scene_migration_map,
    bucket_content_for_recall,
    bucket_text_for_embedding,
    count_tokens_approx,
    generate_scene_id,
    LOCAL_TZ,
    local_date_key,
    load_config,
    migrated_scene_for_legacy_source,
    normalize_scene_cues,
    now_iso,
    parse_human_date_reference,
    setup_logging,
    strip_human_date_references,
    strip_display_temperature_sections,
    strip_followup_sections,
    strip_temperature_meaning_lines,
    strip_wikilinks,
    suppress_migrated_legacy_sources,
)


_SCENE_WRITE_SOURCE: ContextVar[str] = ContextVar(
    "scene_write_source",
    default="hold_scene",
)

# --- Load config & init logging / 加载配置 & 初始化日志 ---
config = load_config()
setup_logging(config.get("log_level", "INFO"))
logger = logging.getLogger("ombre_brain")

MEMORY_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
MEMORY_CARD_SHADOW_VERSION = "shadow-card-v1"
HANDOFF_RECENT_CONTINUITY_MAX_AGE_HOURS = 72

# Read-only runtime observability for the Dashboard. It never stores injected
# prose; only the route and source ids of the latest handoff request.
_last_handoff_status: dict = {}


def _coerce_memory_id(value) -> str:
    if value is None:
        return ""
    return str(value).strip()

# --- Initialize core components / 初始化核心组件 ---
bucket_mgr = BucketManager(config)                  # Bucket manager / 记忆桶管理器
dehydrator = Dehydrator(config)                      # Dehydrator / 脱水器
decay_engine = DecayEngine(config, bucket_mgr)       # Decay engine / 衰减引擎
embedding_engine = EmbeddingEngine(config)            # Embedding engine / 向量化引擎
reranker_engine = RerankerEngine(config)              # Reranker / 召回重排序
recall_diagnostics = RecallDiagnosticsLogger(config)  # Recall diagnostics / 召回诊断
import_engine = ImportEngine(config, bucket_mgr, dehydrator, embedding_engine)  # Import engine / 导入引擎
persona_engine = PersonaStateEngine(config)           # Persona state engine / 人格状态引擎
memory_edge_store = MemoryEdgeStore(config)            # Explicit memory relationship edges / 显式记忆关系边
entity_edge_store = EntityEdgeStore(config)            # Person/object hint edges / 人物对象轻边
memory_node_store = MemoryNodeStore(config)            # Computable memory node index / 可计算记忆节点
memory_moment_store = MemoryMomentStore(config)        # Structured bucket body/comment moment index / 记忆片段索引
shadow_memory_card_store = ShadowMemoryCardStore(config) # Non-authoritative primary/cue/evidence previews / 影子记忆卡
window_shadow_store = WindowShadowStore(config)         # Append-only full-window self narratives / 整窗第一人称窗影
reflection_engine = ReflectionEngine(config)           # Daily/weekly reflection worker / 关系天气
diary_source_importer = DiarySourceImporter()          # Lossless diary evidence snapshots / 无损日记证据快照
metadata_enricher = reflection_engine                  # Proposal-only metadata worker / 只提案、不改正文或权重
scene_linker = SceneLinker(config)                     # Async Scene-edge proposals / 异步 Scene 边提案
portrait_engine = DailyPortraitMaintainer(config, window_shadow_store=window_shadow_store) # Daily portrait state / 每日画像状态
dream_engine = DreamEngine(config)                     # Night dream worker / 夜梦
identity_semantic_store = IdentitySemanticStore(config) # Private relationship alias index / 私有关系语义索引
word_map_store = WordMapStore(config)                   # Derived generic word co-occurrence index / 派生通用词图
darkroom_store = DarkroomStore(config)                  # Private reflection room / 不回显正文的暗房
gateway_state_store = GatewayStateStore(os.path.join(config["buckets_dir"], "gateway_state.db"))
raw_event_store = RawEventStore(config)                  # Raw dialogue archive / 原文保险箱
reminder_store = ReminderStore(config)                    # Standalone care memos / 独立照顾备忘
narrative_roll_store = NarrativeRollStore(config)          # Reviewed sourced Narrative Projection arcs / 审阅后发布的叙事卷
legacy_memory_review_store = LegacyMemoryReviewStore(config) # Admin-only legacy lifecycle / bridge review cards
legacy_memory_lifecycle_publisher = LegacyMemoryLifecyclePublisher(
    config,
    legacy_memory_review_store,
) # Explicit admin-only archive publisher; never part of daily MCP

# --- Create MCP server instance / 创建 MCP 服务器实例 ---
# host="0.0.0.0" so Docker container's SSE is externally reachable
# stdio mode ignores host (no network)
mcp = OmbreFastMCP(
    "Ombre Brain",
    config=config,
    host="0.0.0.0",
    port=8000,
)


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _split_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _dashboard_env_path() -> str:
    return os.environ.get(
        "OMBRE_ENV_PATH",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
    )


def _quote_env_value(value: str) -> str:
    text = str(value or "")
    if not text or re.search(r"\s|#|=|['\"]", text):
        return _json_lib.dumps(text, ensure_ascii=False)
    return text


def _write_dashboard_env_values(updates: dict[str, str]) -> list[str]:
    updates = {
        str(key): str(value)
        for key, value in (updates or {}).items()
        if str(key or "").strip() and str(value or "")
    }
    if not updates:
        return []

    env_path = os.path.abspath(_dashboard_env_path())
    os.makedirs(os.path.dirname(env_path), exist_ok=True)
    lines = []
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

    remaining = dict(updates)
    output = []
    for line in lines:
        match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
        key = match.group(1) if match else ""
        if key in remaining:
            output.append(f"{key}={_quote_env_value(remaining.pop(key))}")
        else:
            output.append(line)
    for key, value in remaining.items():
        output.append(f"{key}={_quote_env_value(value)}")

    with open(env_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(output).rstrip() + "\n")

    for key, value in updates.items():
        os.environ[key] = value
    return [f"env.{key}" for key in updates]


def _dashboard_split_names(value) -> list[str]:
    if isinstance(value, str):
        candidates = re.split(r"[\n,]+", value)
    elif isinstance(value, list):
        candidates = value
    else:
        candidates = []
    names = []
    for candidate in candidates:
        name = str(candidate or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def _dashboard_api_key_values(value) -> list[str]:
    if isinstance(value, list):
        candidates = value
    elif isinstance(value, str):
        candidates = value.splitlines()
    else:
        candidates = []
    return [str(candidate or "").strip() for candidate in candidates if str(candidate or "").strip()]


def _dashboard_sanitize_env_names(value) -> list[str]:
    env_names = []
    for env_name in _dashboard_split_names(value):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_name):
            raise ValueError(f'invalid api key env name "{env_name}"')
        env_names.append(env_name)
    return env_names


def _dashboard_sanitize_upstream_models(raw_models) -> list:
    if isinstance(raw_models, str):
        raw_items = [item.strip() for item in raw_models.split(",")]
    elif isinstance(raw_models, list):
        raw_items = raw_models
    else:
        raw_items = []
    models = []
    seen = set()
    for raw_model in raw_items:
        if isinstance(raw_model, dict):
            public_model = str(
                raw_model.get("id")
                or raw_model.get("alias")
                or raw_model.get("name")
                or raw_model.get("model")
                or raw_model.get("upstream_model")
                or ""
            ).strip()
            upstream_model = str(
                raw_model.get("upstream_model")
                or raw_model.get("provider_model")
                or raw_model.get("target_model")
                or raw_model.get("model")
                or public_model
                or ""
            ).strip()
        else:
            public_model = str(raw_model or "").strip()
            upstream_model = public_model
        if not public_model or public_model in seen:
            continue
        seen.add(public_model)
        if upstream_model and upstream_model != public_model:
            models.append({"id": public_model, "upstream_model": upstream_model})
        else:
            models.append(public_model)
    return models


def _dashboard_normalize_upstream_protocol(value) -> str:
    protocol = str(value or "openai").strip().lower()
    if protocol in {"anthropic", "claude"}:
        return "anthropic"
    return "openai"


def _dashboard_sanitize_gateway_upstreams(raw_upstreams, existing_upstreams=None) -> list[dict]:
    if not isinstance(raw_upstreams, list):
        raise ValueError("gateway.upstreams must be a list")
    existing_by_name = {
        str(item.get("name") or "").strip(): item
        for item in (existing_upstreams or [])
        if isinstance(item, dict)
    }
    upstreams = []
    seen_names = set()
    for index, raw in enumerate(raw_upstreams, start=1):
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or f"upstream-{index}").strip() or f"upstream-{index}"
        if name in seen_names:
            raise ValueError(f'duplicate gateway upstream name "{name}"')
        seen_names.add(name)
        sanitized = {
            "name": name,
            "protocol": _dashboard_normalize_upstream_protocol(
                raw.get("protocol") or raw.get("api_format") or raw.get("type")
            ),
            "base_url": str(raw.get("base_url") or "").strip().rstrip("/"),
        }
        env_names = _dashboard_sanitize_env_names(raw.get("api_key_envs", raw.get("api_key_env", [])))
        if env_names:
            sanitized["api_key_envs"] = env_names
        for key in (
            "default_model",
            "prompt_cache",
            "prompt_cache_retention",
            "anthropic_version",
            "anthropic_beta",
        ):
            value = str(raw.get(key) or "").strip()
            if value:
                sanitized[key] = value
        models = _dashboard_sanitize_upstream_models(raw.get("models", []))
        if models:
            sanitized["models"] = models

        existing = existing_by_name.get(name, {})
        for secret_key in ("api_key", "api_keys"):
            if isinstance(existing, dict) and secret_key in existing:
                sanitized[secret_key] = existing[secret_key]
        upstreams.append(sanitized)
    return upstreams


def _dashboard_gateway_upstream_env_updates(raw_upstreams) -> dict[str, str]:
    updates = {}
    for raw in raw_upstreams or []:
        if not isinstance(raw, dict):
            continue
        env_names = _dashboard_sanitize_env_names(raw.get("api_key_envs", raw.get("api_key_env", [])))
        key_values = _dashboard_api_key_values(raw.get("api_key_values", []))
        if key_values and len(key_values) > len(env_names):
            name = str(raw.get("name") or "upstream").strip()
            raise ValueError(f'gateway upstream "{name}" has api key values without matching env names')
        for index, key_value in enumerate(key_values):
            updates[env_names[index]] = key_value
    return updates


def _dashboard_gateway_upstreams_have_key_values(raw_upstreams) -> bool:
    return any(
        isinstance(raw, dict) and bool(_dashboard_api_key_values(raw.get("api_key_values", [])))
        for raw in raw_upstreams or []
    )


def _dashboard_gateway_hot_upstreams(config_upstreams: list[dict], raw_upstreams, env_updates: dict[str, str]) -> list[dict]:
    key_values_by_name = {}
    for raw in raw_upstreams or []:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip()
        if not name:
            continue
        env_names = _dashboard_sanitize_env_names(raw.get("api_key_envs", raw.get("api_key_env", [])))
        values = _dashboard_api_key_values(raw.get("api_key_values", []))
        hot_keys = []
        for index, value in enumerate(values):
            if index < len(env_names):
                env_name = env_names[index]
                if env_updates.get(env_name):
                    hot_keys.append({"api_key": value, "label": f"env:{env_name}"})
        if hot_keys:
            key_values_by_name[name] = hot_keys

    hot_upstreams = []
    for upstream in config_upstreams:
        hot = dict(upstream)
        name = str(hot.get("name") or "").strip()
        if name in key_values_by_name:
            hot["api_keys"] = key_values_by_name[name]
        hot_upstreams.append(hot)
    return hot_upstreams


def _dashboard_gateway_upstreams_payload(gateway_cfg: dict) -> list[dict]:
    raw_upstreams = gateway_cfg.get("upstreams", [])
    if not isinstance(raw_upstreams, list):
        return []
    payload = []
    for raw in raw_upstreams:
        if not isinstance(raw, dict):
            continue
        env_names = _dashboard_split_names(raw.get("api_key_envs", raw.get("api_key_env", [])))
        direct_key_count = 1 if raw.get("api_key") else 0
        raw_api_keys = raw.get("api_keys", [])
        if isinstance(raw_api_keys, str):
            direct_key_count += len([item for item in raw_api_keys.split(",") if item.strip()])
        elif isinstance(raw_api_keys, list):
            direct_key_count += len([item for item in raw_api_keys if item])
        env_key_count = len([env_name for env_name in env_names if os.environ.get(env_name, "")])
        payload.append(
            {
                "name": str(raw.get("name") or "").strip(),
                "protocol": _dashboard_normalize_upstream_protocol(raw.get("protocol")),
                "base_url": str(raw.get("base_url") or "").strip(),
                "api_key_envs": env_names,
                "has_direct_api_key": direct_key_count > 0,
                "key_count": direct_key_count + env_key_count,
                "ready": bool(str(raw.get("base_url") or "").strip() and (direct_key_count or env_key_count)),
                "default_model": str(raw.get("default_model") or "").strip(),
                "prompt_cache": str(raw.get("prompt_cache") or "").strip(),
                "prompt_cache_retention": str(raw.get("prompt_cache_retention") or "").strip(),
                "anthropic_version": str(raw.get("anthropic_version") or "").strip(),
                "anthropic_beta": str(raw.get("anthropic_beta") or "").strip(),
                "models": _dashboard_sanitize_upstream_models(raw.get("models", [])),
            }
        )
    return payload


async def _hot_update_gateway_config(gateway_payload: dict) -> str | None:
    if not gateway_payload:
        return None
    admin_url = os.environ.get("OMBRE_GATEWAY_ADMIN_URL", "").strip()
    token = os.environ.get("OMBRE_GATEWAY_TOKEN", "").strip()
    if not admin_url or not token:
        return None
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.post(
                admin_url,
                headers={"Authorization": f"Bearer {token}"},
                json=gateway_payload,
            )
        if response.status_code >= 400:
            return f"gateway_hot_reload_failed:{response.status_code}"
        return "gateway_hot_reloaded"
    except Exception as exc:
        logger.warning("Gateway hot config update failed: %s", exc)
        return f"gateway_hot_reload_failed:{type(exc).__name__}"


def _gateway_debug_injections_url() -> str:
    admin_url = os.environ.get("OMBRE_GATEWAY_ADMIN_URL", "").strip()
    if not admin_url:
        return ""
    parsed = urlparse(admin_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/api/config"):
        path = path[: -len("/api/config")]
    return parsed._replace(path=f"{path}/api/debug/injections", query="", fragment="").geturl()


async def _fetch_gateway_injection_debug(
    *,
    session_id: str = "",
    limit: int = 10,
    include_context: bool = False,
) -> dict:
    debug_url = _gateway_debug_injections_url()
    token = os.environ.get("OMBRE_GATEWAY_TOKEN", "").strip()
    if not debug_url or not token:
        return {"status": "error", "error": "gateway_debug_not_configured", "items": []}

    params = {
        "limit": max(1, min(100, int(limit))),
        "include_context": "1" if include_context else "0",
    }
    if session_id:
        params["session_id"] = session_id
    try:
        timeout_seconds = float(os.environ.get("OMBRE_GATEWAY_DEBUG_TIMEOUT_SECONDS", "30"))
    except (TypeError, ValueError):
        timeout_seconds = 30.0
    timeout_seconds = max(5.0, min(90.0, timeout_seconds))

    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.get(
                debug_url,
                headers={"Authorization": f"Bearer {token}"},
                params=params,
            )
        if response.status_code >= 400:
            return {
                "status": "error",
                "error": "gateway_debug_failed",
                "status_code": response.status_code,
                "items": [],
            }
        payload = response.json()
    except Exception as exc:
        logger.warning("Gateway injection debug fetch failed: %s", exc)
        return {
            "status": "error",
            "error": f"gateway_debug_failed:{type(exc).__name__}",
            "items": [],
        }
    if not isinstance(payload, dict):
        return {"status": "error", "error": "gateway_debug_invalid_payload", "items": []}
    return {"status": "ok", "items": payload.get("items", []) if isinstance(payload.get("items"), list) else []}


DEFAULT_CHATGPT_OAUTH_REDIRECT_PREFIX = "https://chatgpt.com/connector/oauth/"
DEFAULT_CLAUDE_OAUTH_REDIRECT_URI = "https://claude.ai/api/mcp/auth_callback"


class ChatGptOAuthProvider:
    def __init__(
        self,
        client_id: str = "",
        client_secret: str = "",
        access_token: str = "",
        refresh_token: str = "",
        public_base_url: str = "",
        redirect_prefix: str = DEFAULT_CHATGPT_OAUTH_REDIRECT_PREFIX,
        redirect_uris: list[str] | None = None,
        token_ttl_seconds: int = 30 * 24 * 60 * 60,
    ) -> None:
        self.client_id = client_id.strip()
        self.client_secret = client_secret.strip()
        self.access_token = access_token.strip()
        self.refresh_token = refresh_token.strip()
        self.public_base_url = public_base_url.strip().rstrip("/")
        self.redirect_prefix = redirect_prefix.strip()
        raw_redirect_uris = (
            [DEFAULT_CLAUDE_OAUTH_REDIRECT_URI]
            if redirect_uris is None
            else redirect_uris
        )
        self.redirect_uris = tuple(
            uri.strip()
            for uri in raw_redirect_uris
            if uri.strip()
        )
        self.token_ttl_seconds = token_ttl_seconds
        self._codes: dict[str, tuple[str, float]] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.client_id and self.access_token)

    @property
    def token_auth_methods(self) -> list[str]:
        if self.client_secret:
            return ["client_secret_post", "client_secret_basic"]
        return ["none"]

    def external_base(self, request=None) -> str:
        if self.public_base_url:
            return self.public_base_url
        if request is not None:
            return str(request.base_url).rstrip("/")
        return ""

    def valid_client_id(self, client_id: str | None) -> bool:
        return bool(client_id) and hmac.compare_digest(client_id, self.client_id)

    def valid_client_secret(self, client_secret: str | None) -> bool:
        if not self.client_secret:
            return True
        return bool(client_secret) and hmac.compare_digest(client_secret, self.client_secret)

    def valid_redirect_uri(self, redirect_uri: str | None) -> bool:
        return bool(redirect_uri) and (
            bool(self.redirect_prefix and redirect_uri.startswith(self.redirect_prefix))
            or redirect_uri in self.redirect_uris
        )

    def create_authorization_code(self, redirect_uri: str) -> str:
        code = secrets.token_urlsafe(32)
        self._codes[code] = (redirect_uri, time.time() + 300)
        return code

    def consume_authorization_code(self, code: str | None, redirect_uri: str | None) -> bool:
        if not code:
            return False
        entry = self._codes.pop(code, None)
        if not entry:
            return False
        stored_redirect_uri, expires_at = entry
        if time.time() > expires_at:
            return False
        if redirect_uri and redirect_uri != stored_redirect_uri:
            return False
        return True

    def valid_access_token(self, token: str | None) -> bool:
        return bool(token) and hmac.compare_digest(token, self.access_token)

    def valid_refresh_token(self, token: str | None) -> bool:
        return bool(token) and hmac.compare_digest(token, self.refresh_token)


OMBRE_CHATGPT_OAUTH = ChatGptOAuthProvider(
    client_id=os.environ.get("OMBRE_CHATGPT_OAUTH_CLIENT_ID", ""),
    client_secret=os.environ.get("OMBRE_CHATGPT_OAUTH_CLIENT_SECRET", ""),
    access_token=os.environ.get("OMBRE_CHATGPT_OAUTH_ACCESS_TOKEN", ""),
    refresh_token=os.environ.get("OMBRE_CHATGPT_OAUTH_REFRESH_TOKEN", ""),
    public_base_url=os.environ.get("OMBRE_CHATGPT_OAUTH_PUBLIC_BASE_URL", ""),
    redirect_prefix=os.environ.get(
        "OMBRE_CHATGPT_OAUTH_REDIRECT_PREFIX",
        DEFAULT_CHATGPT_OAUTH_REDIRECT_PREFIX,
    ),
    redirect_uris=_split_csv(
        os.environ.get(
            "OMBRE_CHATGPT_OAUTH_REDIRECT_URIS",
            DEFAULT_CLAUDE_OAUTH_REDIRECT_URI,
        )
    ),
    token_ttl_seconds=_int_env("OMBRE_CHATGPT_OAUTH_TOKEN_TTL_SECONDS", 30 * 24 * 60 * 60),
)


def _default_oauth_protected_hosts() -> set[str]:
    raw = os.environ.get("OMBRE_CHATGPT_OAUTH_PROTECTED_HOSTS")
    hosts = set(_split_csv(raw)) if raw is not None else set()
    if raw is None and OMBRE_CHATGPT_OAUTH.public_base_url:
        host = urlparse(OMBRE_CHATGPT_OAUTH.public_base_url).hostname
        if host:
            hosts.add(host)
    return {host.lower() for host in hosts}


OMBRE_CHATGPT_OAUTH_PROTECTED_HOSTS = _default_oauth_protected_hosts()


def _oauth_public_path(path: str) -> bool:
    normalized = path.rstrip("/") or "/"
    return normalized in {
        "/oauth/authorize",
        "/oauth/token",
        "/.well-known/oauth-authorization-server",
        "/.well-known/oauth-protected-resource",
        "/.well-known/openid-configuration",
        "/mcp/oauth/authorize",
        "/mcp/oauth/token",
        "/mcp/.well-known/oauth-authorization-server",
        "/mcp/.well-known/oauth-protected-resource",
        "/mcp/.well-known/openid-configuration",
    }


def _mcp_path(path: str) -> bool:
    return path == "/mcp" or path.startswith("/mcp/")


def _bearer_token(headers: dict[str, str]) -> str | None:
    auth = headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return None
    return auth.split(" ", 1)[1].strip()


def _basic_client_credentials(headers: dict[str, str]) -> tuple[str | None, str | None]:
    auth = headers.get("authorization", "")
    if not auth.lower().startswith("basic "):
        return None, None
    try:
        decoded = b64decode(auth.split(" ", 1)[1]).decode("utf-8")
        client_id, client_secret = decoded.split(":", 1)
        return client_id, client_secret
    except Exception:
        return None, None


async def _oauth_form(request) -> dict[str, str]:
    body = await request.body()
    parsed = parse_qs(body.decode("utf-8"), keep_blank_values=True)
    return {key: values[-1] if values else "" for key, values in parsed.items()}


def _oauth_error(message: str, status_code: int = 400):
    from starlette.responses import JSONResponse
    return JSONResponse({"error": message}, status_code=status_code)


def _oauth_success_payload() -> dict:
    return {
        "access_token": OMBRE_CHATGPT_OAUTH.access_token,
        "token_type": "Bearer",
        "expires_in": OMBRE_CHATGPT_OAUTH.token_ttl_seconds,
        "refresh_token": OMBRE_CHATGPT_OAUTH.refresh_token,
        "scope": "",
    }


class OmbreChatGptOAuthMiddleware:
    def __init__(self, app, provider: ChatGptOAuthProvider, protected_hosts: set[str]) -> None:
        self.app = app
        self.provider = provider
        self.protected_hosts = {host.lower() for host in protected_hosts}

    async def __call__(self, scope, receive, send):
        if (
            scope.get("type") != "http"
            or not self.provider.enabled
            or scope.get("method") == "OPTIONS"
        ):
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if _oauth_public_path(path) or not _mcp_path(path) or not self._is_protected_host(scope):
            await self.app(scope, receive, send)
            return

        headers = {
            key.decode("latin1").lower(): value.decode("latin1")
            for key, value in scope.get("headers", [])
        }
        if self.provider.valid_access_token(_bearer_token(headers)):
            await self.app(scope, receive, send)
            return

        from starlette.responses import JSONResponse
        response = JSONResponse(
            {"error": "invalid_token"},
            status_code=401,
            headers={"WWW-Authenticate": 'Bearer realm="Ombre Brain"'},
        )
        await response(scope, receive, send)

    def _is_protected_host(self, scope) -> bool:
        if not self.protected_hosts:
            return False
        host = ""
        for key, value in scope.get("headers", []):
            if key.lower() == b"host":
                host = value.decode("latin1").split(":", 1)[0].lower()
                break
        return host in self.protected_hosts


def _current_time_iso() -> str:
    return now_iso()


_dashboard_sessions: dict[str, float] = {}


def _dashboard_auth_file() -> str:
    state_dir = config.get("state_dir") or os.path.join(
        os.path.dirname(os.path.abspath(config.get("buckets_dir", "buckets"))),
        "state",
    )
    return os.path.join(state_dir, ".dashboard_auth.json")


def _load_dashboard_password_hash() -> str | None:
    try:
        path = _dashboard_auth_file()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = _json_lib.load(f)
            return data.get("password_hash")
    except Exception:
        logger.warning("Failed to load dashboard auth file", exc_info=True)
    return None


def _save_dashboard_password_hash(password: str) -> None:
    salt = secrets.token_hex(16)
    digest = hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()
    path = _dashboard_auth_file()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        _json_lib.dump({"password_hash": f"{salt}:{digest}"}, f)


def _verify_dashboard_hash(password: str, stored: str) -> bool:
    if ":" not in stored:
        return False
    salt, digest = stored.split(":", 1)
    current = hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()
    return hmac.compare_digest(digest, current)


def _dashboard_setup_needed() -> bool:
    if os.environ.get("OMBRE_DASHBOARD_PASSWORD", ""):
        return False
    return _load_dashboard_password_hash() is None


@mcp.custom_route("/oauth/authorize", methods=["GET"])
@mcp.custom_route("/mcp/oauth/authorize", methods=["GET"])
async def chatgpt_oauth_authorize(request):
    from starlette.responses import RedirectResponse

    if not OMBRE_CHATGPT_OAUTH.enabled:
        return _oauth_error("oauth_not_configured", 404)

    params = request.query_params
    client_id = params.get("client_id")
    redirect_uri = params.get("redirect_uri")
    response_type = params.get("response_type")
    state = params.get("state")

    if response_type != "code":
        return _oauth_error("unsupported_response_type")
    if not OMBRE_CHATGPT_OAUTH.valid_client_id(client_id):
        return _oauth_error("invalid_client", 401)
    if not OMBRE_CHATGPT_OAUTH.valid_redirect_uri(redirect_uri):
        return _oauth_error("invalid_redirect_uri")

    code = OMBRE_CHATGPT_OAUTH.create_authorization_code(redirect_uri)
    query = {"code": code}
    if state:
        query["state"] = state
    separator = "&" if "?" in redirect_uri else "?"
    return RedirectResponse(url=f"{redirect_uri}{separator}{urlencode(query)}", status_code=302)


@mcp.custom_route("/oauth/token", methods=["POST"])
@mcp.custom_route("/mcp/oauth/token", methods=["POST"])
async def chatgpt_oauth_token(request):
    if not OMBRE_CHATGPT_OAUTH.enabled:
        return _oauth_error("oauth_not_configured", 404)

    form = await _oauth_form(request)
    basic_client_id, basic_client_secret = _basic_client_credentials(request.headers)
    client_id = basic_client_id or form.get("client_id")
    client_secret = basic_client_secret or form.get("client_secret")

    if not OMBRE_CHATGPT_OAUTH.valid_client_id(client_id):
        return _oauth_error("invalid_client", 401)
    if not OMBRE_CHATGPT_OAUTH.valid_client_secret(client_secret):
        return _oauth_error("invalid_client", 401)

    grant_type = form.get("grant_type")
    if grant_type == "authorization_code":
        if not OMBRE_CHATGPT_OAUTH.consume_authorization_code(form.get("code"), form.get("redirect_uri")):
            return _oauth_error("invalid_grant")
    elif grant_type == "refresh_token":
        if not OMBRE_CHATGPT_OAUTH.valid_refresh_token(form.get("refresh_token")):
            return _oauth_error("invalid_grant")
    else:
        return _oauth_error("unsupported_grant_type")

    from starlette.responses import JSONResponse
    return JSONResponse(_oauth_success_payload())


def _oauth_server_metadata(request) -> dict:
    base = OMBRE_CHATGPT_OAUTH.external_base(request)
    return {
        "issuer": base,
        "authorization_endpoint": f"{base}/oauth/authorize",
        "token_endpoint": f"{base}/oauth/token",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "token_endpoint_auth_methods_supported": OMBRE_CHATGPT_OAUTH.token_auth_methods,
    }


@mcp.custom_route("/.well-known/oauth-authorization-server", methods=["GET"])
@mcp.custom_route("/.well-known/openid-configuration", methods=["GET"])
@mcp.custom_route("/mcp/.well-known/oauth-authorization-server", methods=["GET"])
@mcp.custom_route("/mcp/.well-known/openid-configuration", methods=["GET"])
async def chatgpt_oauth_metadata(request):
    from starlette.responses import JSONResponse

    if not OMBRE_CHATGPT_OAUTH.enabled:
        return _oauth_error("oauth_not_configured", 404)
    return JSONResponse(_oauth_server_metadata(request))


@mcp.custom_route("/.well-known/oauth-protected-resource", methods=["GET"])
@mcp.custom_route("/mcp/.well-known/oauth-protected-resource", methods=["GET"])
async def chatgpt_oauth_resource_metadata(request):
    from starlette.responses import JSONResponse

    if not OMBRE_CHATGPT_OAUTH.enabled:
        return _oauth_error("oauth_not_configured", 404)
    base = OMBRE_CHATGPT_OAUTH.external_base(request)
    return JSONResponse(
        {
            "resource": f"{base}/mcp",
            "authorization_servers": [base],
            "bearer_methods_supported": ["header"],
        }
    )


def _verify_dashboard_password(password: str) -> bool:
    env_password = os.environ.get("OMBRE_DASHBOARD_PASSWORD", "")
    if env_password:
        return hmac.compare_digest(password, env_password)
    stored = _load_dashboard_password_hash()
    return bool(stored and _verify_dashboard_hash(password, stored))


def _create_dashboard_session() -> str:
    token = secrets.token_urlsafe(32)
    _dashboard_sessions[token] = time.time() + 86400 * 7
    return token


def _dashboard_authenticated(request) -> bool:
    token = request.cookies.get("ombre_session")
    if not token:
        return False
    expiry = _dashboard_sessions.get(token)
    if expiry is None or time.time() > expiry:
        _dashboard_sessions.pop(token, None)
        return False
    return True


def _require_dashboard_auth(request):
    from starlette.responses import JSONResponse
    if _dashboard_authenticated(request):
        return None
    return JSONResponse(
        {"error": "unauthorized", "setup_needed": _dashboard_setup_needed()},
        status_code=401,
    )


def _require_raw_api_auth(request):
    if _dashboard_authenticated(request) or _authorized_memory_write(request):
        return None
    return _require_dashboard_auth(request)


def _dashboard_login_response():
    from starlette.responses import JSONResponse
    token = _create_dashboard_session()
    response = JSONResponse({"ok": True})
    response.set_cookie(
        "ombre_session",
        token,
        httponly=True,
        samesite="lax",
        max_age=86400 * 7,
    )
    return response


def _memory_write_token() -> str:
    return (
        os.environ.get("OMBRE_MEMORY_WRITE_TOKEN")
        or os.environ.get("OMBRE_GATEWAY_TOKEN")
        or str(config.get("gateway", {}).get("token") or "")
    )


def _authorized_memory_write(request) -> bool:
    token = _memory_write_token()
    if not token:
        return False

    candidates = []
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        candidates.append(auth.split(" ", 1)[1].strip())
    for header_name in ("x-ombre-token", "x-api-key"):
        value = request.headers.get(header_name)
        if value:
            candidates.append(value.strip())
    return any(hmac.compare_digest(candidate, token) for candidate in candidates)


def _string_list(value, default: list[str]) -> list[str]:
    if value is None:
        return default
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    elif isinstance(value, list):
        items = [str(item).strip() for item in value]
    else:
        items = [str(value).strip()]
    return [item for item in items if item] or default


def _float_between(value, default: float, low: float = 0.0, high: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _int_between(value, default: int, low: int = 1, high: int = 10) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _date_key(value) -> str:
    match = re.search(r"\d{4}-\d{2}-\d{2}", str(value or ""))
    return match.group(0) if match else ""


def _filter_by_created_date(
    buckets: list[dict],
    *,
    created_date: str = "",
    created_from: str = "",
    created_to: str = "",
) -> tuple[list[dict], str]:
    exact = _date_key(created_date)
    start = exact or _date_key(created_from)
    end = exact or _date_key(created_to)
    if not start and not end:
        return buckets, ""

    filtered = []
    for bucket in buckets:
        day = _date_key((bucket.get("metadata") or {}).get("created", ""))
        if not day:
            continue
        if start and day < start:
            continue
        if end and day > end:
            continue
        filtered.append(bucket)

    if start and end and start == end:
        return filtered, f", created_date={start}"
    return filtered, f", created_from={start or '*'}, created_to={end or '*'}"


def _bucket_matches_breath_date(bucket: dict, date_key: str) -> bool:
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    if meta.get("date"):
        return local_date_key(meta.get("date")) == date_key
    for key in ("created", "updated_at", "last_active"):
        if local_date_key(meta.get(key)) == date_key:
            return True
    return False


def _breath_date_bucket_sort_key(bucket: dict) -> tuple[str, int]:
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    date_value = str(meta.get("date") or "")
    if not date_value:
        date_value = max(str(meta.get(key) or "") for key in ("updated_at", "last_active", "created"))
    try:
        importance = int(meta.get("importance", 5))
    except (TypeError, ValueError):
        importance = 5
    return date_value, importance


def _breath_query_requests_date_read(query: str) -> bool:
    text = str(query or "").strip()
    if not text or not parse_human_date_reference(text):
        return False
    recall_markers = (
        "聊",
        "说",
        "提",
        "讲",
        "讨论",
        "查",
        "找",
        "搜索",
        "记得",
        "记忆",
        "做了什么",
        "发生",
        "什么事",
        "什么",
    )
    return any(marker in text for marker in recall_markers)


def _strip_breath_date_query_shell(query: str) -> str:
    text = strip_human_date_references(query)
    shell_terms = {
        "我们", "咱们", "哥哥", "宝宝", "老婆", "我", "你",
        "还记得", "记不记得", "记得", "想起", "想起来", "回忆", "记忆",
        "在聊什么", "聊了什么", "聊什么", "聊过什么", "说了什么", "说什么",
        "提到什么", "讲了什么", "讨论什么", "做了什么", "发生了什么",
        "在聊", "聊", "说", "提到", "提", "讲", "讨论", "发生", "做",
        "查一下", "查", "搜索", "找一下", "找", "那次", "这次",
        "事情", "事", "什么", "为什么", "怎么回事", "怎么说",
        "有", "没有", "有没有", "是", "吗", "么", "嘛", "呢", "啊", "呀", "啦", "吧",
        "的", "了", "一下", "再", "一次",
    }
    identity = _identity()
    shell_terms.update(
        str(term)
        for term in [
            identity.get("ai_name"),
            identity.get("user_name"),
            identity.get("user_display_name"),
            *(identity.get("user_aliases") or []),
        ]
        if str(term or "").strip()
    )
    for term in sorted(shell_terms, key=lambda item: len(str(item)), reverse=True):
        if str(term).strip():
            text = text.replace(str(term), " ")
    return re.sub(r"[\s，。！？、,.!?:：;；~～（）()\[\]【】「」『』“”\"'`]+", " ", text).strip()


def _breath_date_topic_terms(query: str) -> list[str]:
    topic_query = _strip_breath_date_query_shell(query)
    if not topic_query:
        return []
    terms = list(_recall_policy().specific_query_terms(topic_query))
    terms.extend(re.findall(r"[A-Za-z]+[A-Za-z0-9_.:-]*|[\u4e00-\u9fff]{2,}", topic_query))
    seen = set()
    result = []
    for term in terms:
        cleaned = str(term or "").strip()
        key = _compact_lookup_key(cleaned)
        if not key or key in seen:
            continue
        if re.fullmatch(r"[a-z0-9_.:-]+", key) and len(key) < 3 and not re.search(r"\d", key):
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]+", key) and len(key) < 2:
            continue
        seen.add(key)
        result.append(cleaned)
    return result[:8]


def _breath_date_text_has_topic_terms(text: str, topic_terms: list[str]) -> bool:
    if not topic_terms:
        return True
    haystack = _compact_lookup_key(text)
    return any(_compact_lookup_key(term) in haystack for term in topic_terms if _compact_lookup_key(term))


def _breath_date_bucket_text(bucket: dict) -> str:
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    return " ".join(
        [
            str(meta.get("name") or bucket.get("id") or ""),
            str(meta.get("annotation_summary") or meta.get("summary") or ""),
            " ".join(str(tag) for tag in meta.get("tags", []) or []),
            " ".join(str(item) for item in meta.get("domain", []) or []),
            _rendered_bucket_content(bucket),
        ]
    )


async def _read_breath_date(
    *,
    date_key: str,
    label: str,
    query: str,
    max_tokens: int,
    max_results: int,
    domain_filter: list[str] | None = None,
) -> str:
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=True)
    except Exception as e:
        logger.error(f"Breath date listing failed / 日期记忆列桶失败: {e}")
        return "日期记忆暂时无法访问。"

    domain_set = {item.lower() for item in domain_filter or []}
    topic_terms = _breath_date_topic_terms(query)
    candidates = []
    for bucket in all_buckets:
        if not isinstance(bucket, dict) or is_self_anchor_bucket(bucket):
            continue
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        if meta.get("type") == "feel":
            continue
        if domain_set and not ({str(item).lower() for item in meta.get("domain", []) or []} & domain_set):
            continue
        if not _bucket_matches_breath_date(bucket, date_key):
            continue
        if topic_terms and not _breath_date_text_has_topic_terms(_breath_date_bucket_text(bucket), topic_terms):
            continue
        candidates.append(bucket)

    candidates.sort(key=_breath_date_bucket_sort_key, reverse=True)
    if not candidates:
        if topic_terms:
            return f"{date_key} 没有找到匹配主题的普通记忆。"
        return f"{date_key} 没有找到普通记忆。"

    results = []
    used = 0
    for bucket in candidates[:max_results]:
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        bucket_id = str(bucket.get("id") or "")
        title = str(meta.get("name") or bucket_id)
        date_part = " ".join(_bucket_date_meta_parts(bucket))
        text = (
            str(meta.get("annotation_summary") or meta.get("summary") or "").strip()
            or _clip_text(_rendered_bucket_content(bucket), 560)
        )
        entry = f"- [bucket_id:{bucket_id}] {date_part} {title}\n{text}".strip()
        tokens = count_tokens_approx(entry)
        if used + tokens > max_tokens and results:
            break
        results.append(entry)
        used += tokens
        if used >= max_tokens:
            break

    header = f"=== 日期记忆 {date_key}"
    if label and label != date_key:
        header += f" ({label})"
    header += " ==="
    if topic_terms:
        header += "\n主题过滤: " + ", ".join(topic_terms)
    return header + "\n" + "\n---\n".join(results)


def _bool_value(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _identity() -> dict:
    return identity_names(config)


def _ai_author_name() -> str:
    return _identity()["ai_name"]


def _dashboard_author_name() -> str:
    return _identity()["user_name"]


def _anchor_config() -> tuple[int, float]:
    anchor_cfg = config.get("anchor", {}) if isinstance(config.get("anchor", {}), dict) else {}
    max_count = _int_between(anchor_cfg.get("max_count"), 12, 1, 200)
    try:
        min_age_hours = float(anchor_cfg.get("min_age_hours", 24))
    except (TypeError, ValueError):
        min_age_hours = 24.0
    return max_count, max(0.0, min_age_hours)


def _bucket_age_hours(bucket: dict) -> float | None:
    created = bucket.get("metadata", {}).get("created", "")
    if not created:
        return None
    try:
        parsed = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds() / 3600)


async def _can_mark_anchor(bucket_id: str, bucket: dict) -> tuple[bool, str]:
    max_count, min_age_hours = _anchor_config()
    age_hours = _bucket_age_hours(bucket)
    if age_hours is not None and age_hours < min_age_hours:
        return (
            False,
            f"这条记忆还太新，anchor 至少等待 {min_age_hours:g} 小时后再标记。",
        )
    all_buckets = await bucket_mgr.list_all(include_archive=True)
    anchor_count = sum(
        1
        for b in all_buckets
        if b["id"] != bucket_id and b.get("metadata", {}).get("anchor")
    )
    if anchor_count >= max_count:
        return False, f"anchor 名额已满（{max_count} 条）。请先取消一条旧 anchor。"
    return True, ""


def _select_anchor_buckets(all_buckets: list[dict], limit: int = 2) -> list[dict]:
    limit = _int_between(limit, 2, 0, 12)
    if limit <= 0:
        return []
    anchors = [
        b for b in all_buckets
        if b.get("metadata", {}).get("anchor")
        and not is_self_anchor_bucket(b)
        and not b.get("metadata", {}).get("pinned")
        and not b.get("metadata", {}).get("protected")
        and b.get("metadata", {}).get("type") not in {"permanent", "feel"}
    ]
    anchors.sort(
        key=lambda b: (
            int(b.get("metadata", {}).get("importance", 5)),
            decay_engine.calculate_score(b.get("metadata", {})),
            b.get("metadata", {}).get("updated_at") or b.get("metadata", {}).get("created", ""),
        ),
        reverse=True,
    )
    return anchors[:limit]


def _select_self_anchor_buckets(all_buckets: list[dict], limit: int = 1) -> list[dict]:
    limit = _int_between(limit, 1, 0, 50)
    if limit <= 0:
        return []
    anchors = []
    for bucket in all_buckets:
        if not is_self_anchor_bucket(bucket):
            continue
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        if meta.get("active") is False or meta.get("deprecated") or meta.get("resolved"):
            continue
        anchors.append(bucket)
    anchors.sort(
        key=lambda b: (
            int((b.get("metadata") or {}).get("importance", 5)),
            decay_engine.calculate_score(b.get("metadata", {})),
            (b.get("metadata") or {}).get("updated_at") or (b.get("metadata") or {}).get("created", ""),
        ),
        reverse=True,
    )
    return anchors[:limit]


def _self_anchor_entry_bucket_id() -> str:
    cfg = config.get("self_anchor", {}) if isinstance(config.get("self_anchor", {}), dict) else {}
    return str(cfg.get("entry_bucket_id") or "").strip()


def _select_self_anchor_entry_bucket(all_buckets: list[dict]) -> dict | None:
    entry_id = _self_anchor_entry_bucket_id()
    if entry_id:
        for bucket in all_buckets:
            if str(bucket.get("id") or "") == entry_id:
                meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
                if meta.get("active") is not False and not meta.get("deprecated") and not meta.get("resolved"):
                    return bucket
        return None
    selected = _select_self_anchor_buckets(all_buckets, limit=1)
    return selected[0] if selected else None


def _is_self_anchor_recall_excluded_bucket(bucket: dict | None) -> bool:
    if not isinstance(bucket, dict):
        return False
    if is_self_anchor_bucket(bucket):
        return True
    entry_id = _self_anchor_entry_bucket_id()
    return bool(entry_id and str(bucket.get("id") or "") == entry_id)


def _is_self_anchor_recall_excluded_moment(moment: dict | None) -> bool:
    if not isinstance(moment, dict):
        return False
    if is_self_anchor_metadata(moment.get("metadata", {})):
        return True
    entry_id = _self_anchor_entry_bucket_id()
    return bool(entry_id and str(moment.get("bucket_id") or "") == entry_id)


def _prune_self_anchor_moment_index(all_buckets: list[dict]) -> None:
    for bucket in all_buckets or []:
        if not _is_self_anchor_recall_excluded_bucket(bucket):
            continue
        bucket_id = str(bucket.get("id") or "").strip()
        if bucket_id:
            memory_moment_store.delete_bucket(bucket_id)


def _self_anchor_body_text(bucket: dict, *, include_reflection: bool = False, max_chars: int = 260) -> str:
    sections = _profile_fact_sections(bucket.get("content", ""))
    body_text = _leading_body_text(bucket.get("content", ""))
    body_parts = []
    if body_text:
        body_parts.append(body_text)
    else:
        for key in (
            SELF_ANCHOR_TAG,
            "self_anchor",
            "selfidentity",
            "self_identity",
            "first_person_anchor",
            "fact",
        ):
            text = str(sections.get(_profile_key(key, ""), "") or "").strip()
            if text:
                body_parts.append(text)
                break
    if include_reflection:
        reflection = str(sections.get(_profile_key("reflection", ""), "") or "").strip()
        if reflection:
            body_parts.append(f"### reflection\n{reflection}")
    text = "\n\n".join(part for part in body_parts if part).strip()
    if not text:
        text = _handoff_clean_summary_text(bucket.get("content", ""), include_detail_sections=False)
    return _clip_text(_handoff_clean_summary_text(text, include_detail_sections=True), max_chars)


def _self_anchor_text(bucket: dict) -> str:
    return _self_anchor_body_text(bucket, include_reflection=False, max_chars=260)


def _format_handoff_self_anchor(all_buckets: list[dict], limit: int = 1) -> str:
    bucket = _select_self_anchor_entry_bucket(all_buckets)
    if not bucket:
        return ""
    return _self_anchor_text(bucket)


def _is_self_anchor_tag_read_request(query: str) -> bool:
    aliases = {
        SELF_ANCHOR_TAG,
        "self_anchor",
        "first_person_anchor",
        "first-person-anchor",
    }
    lowered = str(query or "").strip().lower().strip(" \t\r\n`[]()")
    tag_value = ""
    if lowered.startswith("tag:"):
        tag_value = lowered[4:].strip()
    elif lowered.startswith("标签:"):
        tag_value = lowered[3:].strip()
    elif lowered.startswith("#"):
        tag_value = lowered[1:].strip()
    if tag_value and tag_value in aliases:
        return True
    return False


def _is_self_anchor_domain(domain_key: str) -> bool:
    return domain_key in {SELF_ANCHOR_TAG, "self_anchor", "self_identity", "selfidentity", "self-identity"}


def _self_anchor_bucket_search_text(bucket: dict) -> str:
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    return "\n".join(
        str(part or "")
        for part in (
            bucket.get("id", ""),
            meta.get("name", ""),
            " ".join(str(tag) for tag in meta.get("tags", []) or []),
            strip_wikilinks(bucket.get("content", "")),
        )
    ).lower()


async def _read_self_anchor_tag_breath(max_tokens: int = 1000, limit: int = 3) -> str:
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
    except Exception as e:
        logger.error("Self-anchor read failed / 自我读取失败: %s", e)
        return "自我 anchor 暂时无法访问。"
    anchors = _select_self_anchor_buckets(all_buckets, limit=limit)
    if not anchors:
        return "还没有自我 anchor。"
    remaining = max(0, int(max_tokens or 0))
    rows = []
    for bucket in anchors:
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        text = strip_wikilinks(str(bucket.get("content") or "")).strip()
        if not text:
            continue
        row = f"[bucket_id:{bucket.get('id', '')}] {str(meta.get('name') or SELF_ANCHOR_TAG)}\n{text}"
        row_tokens = count_tokens_approx(row)
        if rows and row_tokens > remaining:
            break
        rows.append(row)
        remaining -= min(row_tokens, remaining)
        if remaining <= 0:
            break
    return "=== 自我 ===\n" + ("\n---\n".join(rows) if rows else "还没有可读的自我 anchor。")


async def _read_self_anchor_domain_breath(query: str = "", max_tokens: int = 1000, limit: int = 3) -> str:
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
    except Exception as e:
        logger.error("Self-anchor domain read failed / 自我入口读取失败: %s", e)
        return "自我入口暂时无法访问。"
    query_text = str(query or "").strip()
    if not query_text:
        bucket = _select_self_anchor_entry_bucket(all_buckets)
        if not bucket:
            return "还没有自我入口。"
        text = _self_anchor_text(bucket)
        return "=== 自我入口 ===\n" + (text or "自我入口暂时没有正文。")

    anchors = _select_self_anchor_buckets(all_buckets, limit=max(limit, 3))
    needle = query_text.lower()
    rows = []
    remaining = max(0, int(max_tokens or 0))
    for bucket in anchors:
        if needle not in _self_anchor_bucket_search_text(bucket):
            continue
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        text = strip_wikilinks(str(bucket.get("content") or "")).strip()
        if not text:
            continue
        row = f"[bucket_id:{bucket.get('id', '')}] {str(meta.get('name') or SELF_ANCHOR_TAG)}\n{text}"
        row_tokens = count_tokens_approx(row)
        if rows and row_tokens > remaining:
            break
        rows.append(row)
        remaining -= min(row_tokens, remaining)
        if len(rows) >= limit or remaining <= 0:
            break
    return "=== 自我分段 ===\n" + ("\n---\n".join(rows) if rows else "没有找到相关自我分段。")


def _normalize_breath_mode(value: object) -> str:
    mode = str(value or "").strip().lower()
    return mode if mode in {"", "handoff"} else ""


def _trim_handoff_text_to_token_budget(text: str, token_budget: int) -> str:
    clean = str(text or "").strip()
    if token_budget <= 0 or not clean:
        return ""
    if count_tokens_approx(clean) <= token_budget:
        return clean
    kept: list[str] = []
    for line in (line.strip() for line in clean.splitlines() if line.strip()):
        remaining = token_budget - count_tokens_approx("\n".join(kept))
        if remaining <= 0:
            break
        if count_tokens_approx(line) <= remaining:
            kept.append(line)
            continue
        sentences = [part.strip() for part in re.split(r"(?<=[。！？!?；;])", line) if part.strip()]
        sentence_rows: list[str] = []
        for sentence in sentences:
            candidate = "".join([*sentence_rows, sentence])
            if count_tokens_approx(candidate) > remaining:
                break
            sentence_rows.append(sentence)
        if sentence_rows:
            kept.append("".join(sentence_rows))
            break
        low, high = 0, len(line)
        while low < high:
            middle = (low + high + 1) // 2
            candidate = line[:middle].rstrip("，,；;：: ") + "…"
            if count_tokens_approx(candidate) <= remaining:
                low = middle
            else:
                high = middle - 1
        if low > 0:
            kept.append(line[:low].rstrip("，,；;：: ") + "…")
        break
    return "\n".join(kept)


def _format_budgeted_handoff_sections(
    intro: str,
    sections: list[tuple[str, str, int, bool] | tuple[str, str, int, bool, bool]],
    max_tokens: int,
) -> str:
    """Render handoff sections while preserving explicitly authored rows byte-for-byte."""
    budget = max(0, int(max_tokens or 0))
    if budget <= 0:
        return ""
    normalized = []
    for row in sections:
        if not str(row[1] or "").strip():
            continue
        title, content, cap, line_mode = row[:4]
        preserve_exact = bool(row[4]) if len(row) > 4 else False
        normalized.append((title, str(content).strip(), int(cap), bool(line_mode), preserve_exact))

    exact_cost = count_tokens_approx(intro)
    for title, content, _, _, preserve_exact in normalized:
        if preserve_exact:
            exact_cost += count_tokens_approx(f"\n\n=== {title} ===\n{content}")
    remaining = max(0, budget - exact_cost)
    rendered: list[tuple[str, str]] = []
    for title, content, cap, line_mode, preserve_exact in normalized:
        if preserve_exact:
            rendered.append((title, content))
            continue
        header_cost = count_tokens_approx(f"\n\n=== {title} ===\n")
        allocated = min(max(0, cap), max(0, remaining - header_cost))
        if allocated <= 0:
            continue
        trimmed = (
            _trim_lines_to_token_budget(content, allocated)
            if line_mode
            else _trim_handoff_text_to_token_budget(content, allocated)
        )
        if trimmed:
            rendered.append((title, trimmed))
            remaining -= header_cost + count_tokens_approx(trimmed)
    result = intro
    for title, content in rendered:
        result += f"\n\n=== {title} ===\n{content}"
    return result


def _latest_window_shadow_handoff_projection(
    *,
    session_id: str = "",
    parent_shadow_id: str = "",
    allow_latest_fallback: bool = True,
    self_max_tokens: int = 360,
    relationship_max_tokens: int = 360,
) -> dict[str, str]:
    """Carry authored Shadow layers without injecting its recalled Scene layer."""
    try:
        exact_parent = str(parent_shadow_id or "").strip()
        if exact_parent:
            projection = window_shadow_store.handoff_projection(exact_parent)
        elif allow_latest_fallback:
            projection = window_shadow_store.latest_handoff_projection(
                exclude_session_id=session_id
            )
        else:
            projection = None
    except Exception as exc:
        logger.warning("Latest window shadow projection failed / 最近窗影投影失败: %s", exc)
        return {}
    if not projection:
        return {}
    return {
        **projection,
        # This is the current Haven's authored note to the next window. It is
        # never summarized or truncated here.
        "handoff_note": str(projection.get("handoff_note") or "").strip(),
        "flowing_self": _trim_handoff_text_to_token_budget(
            projection.get("flowing_self", ""), self_max_tokens
        ),
        "recent_relationship": _trim_handoff_text_to_token_budget(
            projection.get("recent_relationship", ""), relationship_max_tokens
        ),
    }


def _emergency_raw_handoff_projection(
    *,
    session_id: str = "",
    previous_session_id: str = "",
    event_limit: int = 6,
    scan_limit: int = 80,
    max_tokens: int = 520,
) -> dict[str, str | bool]:
    """Restore a failed close from stored originals, without asking a model to summarize."""
    current_session = str(session_id or "").strip()
    requested_previous = str(previous_session_id or "").strip()
    try:
        if requested_previous:
            target_session = requested_previous
            inferred = False
        else:
            recent = raw_event_store.search(query="", limit=max(1, min(100, scan_limit)))
            target_session = next(
                (
                    str(item.get("session_id") or "").strip()
                    for item in recent.get("items", []) or []
                    if isinstance(item, dict)
                    and str(item.get("session_id") or "").strip()
                    and str(item.get("session_id") or "").strip() != current_session
                    and str(item.get("role") or "").strip().lower() in {"user", "assistant"}
                ),
                "",
            )
            inferred = True
        if not target_session or target_session == current_session:
            return {}
        payload = raw_event_store.search(
            query="",
            session_id=target_session,
            limit=max(1, min(20, event_limit)),
        )
    except Exception as exc:
        logger.warning("Emergency raw handoff failed / 原文救生交接失败: %s", exc)
        return {}

    rows = [
        item
        for item in payload.get("items", []) or []
        if isinstance(item, dict)
        and str(item.get("role") or "").strip().lower() in {"user", "assistant"}
        and str(item.get("text") or "").strip()
    ]
    rows.reverse()
    lines = []
    for item in rows:
        role = str(item.get("role") or "").strip().lower()
        label = "user" if role == "user" else "assistant"
        text = re.sub(r"\s+", " ", str(item.get("text") or "")).strip()
        text = _trim_handoff_text_to_token_budget(text, 110)
        if text:
            lines.append(f"- {label}: {text}")
    rendered = _trim_lines_to_token_budget("\n".join(lines), max_tokens)
    if not rendered:
        return {}
    sources = list(
        dict.fromkeys(
            str(item.get("source") or "").strip()
            for item in rows
            if str(item.get("source") or "").strip()
        )
    )
    return {
        "text": rendered,
        "session_id": target_session,
        "source": ",".join(sources),
        "inferred": inferred,
    }


def _parse_handoff_state_time(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _recent_continuity_fallback_status(
    portrait_sections: dict,
    *,
    now: datetime | None = None,
    max_age_hours: int = HANDOFF_RECENT_CONTINUITY_MAX_AGE_HOURS,
) -> dict:
    """Return a fresh nightly continuity fallback without rewriting its text."""
    sections = portrait_sections if isinstance(portrait_sections, dict) else {}
    text = str(sections.get("recent_continuity") or "").strip()
    updated_at = str(sections.get("updated_at") or "").strip()
    last_run_date = str(sections.get("last_run_date") or "").strip()
    timestamp = None
    timestamp_source = ""
    if last_run_date:
        try:
            # A date-only portrait run covers that local day. Treat the next
            # local midnight as its conservative completion timestamp.
            timestamp = (
                datetime.fromisoformat(last_run_date)
                .replace(tzinfo=LOCAL_TZ)
                .astimezone(timezone.utc)
                + timedelta(days=1)
            )
            timestamp_source = "last_run_date"
        except ValueError:
            timestamp = None
    if timestamp is None:
        timestamp = _parse_handoff_state_time(updated_at)
        timestamp_source = "updated_at" if timestamp else ""
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc)
    age_hours = (
        max(0.0, (current - timestamp).total_seconds() / 3600.0)
        if timestamp is not None
        else None
    )
    available = bool(text)
    fresh = bool(
        available
        and timestamp is not None
        and age_hours is not None
        and age_hours <= max(1, int(max_age_hours or 0))
    )
    if not available:
        reason = "empty"
    elif timestamp is None:
        reason = "missing_timestamp"
    elif not fresh:
        reason = "stale"
    else:
        reason = "fresh"
    return {
        "text": text if fresh else "",
        "available": available,
        "fresh": fresh,
        "reason": reason,
        "updated_at": updated_at,
        "last_run_date": last_run_date,
        "timestamp_source": timestamp_source,
        "age_hours": round(age_hours, 2) if age_hours is not None else None,
        "max_age_hours": max(1, int(max_age_hours or 0)),
    }


def _window_shadow_scene_index(
    shadow_projection: dict,
    all_buckets: list[dict],
    *,
    limit: int = 3,
) -> dict:
    """Project a tiny verified Scene index for one exact parent Shadow."""
    window_id = str((shadow_projection or {}).get("window_id") or "").strip()
    linked_ids = list(
        dict.fromkeys(
            str(value or "").strip()
            for value in (shadow_projection or {}).get("scene_bucket_ids", []) or []
            if str(value or "").strip()
        )
    )
    continue_scene_id = str(
        (shadow_projection or {}).get("continue_scene_id") or ""
    ).strip()
    if not window_id or not linked_ids:
        return {
            "text": "",
            "items": [],
            "continue_scene_id": "",
            "linked_count": len(linked_ids),
            "omitted_count": len(linked_ids),
        }

    ordered_ids = [
        *([continue_scene_id] if continue_scene_id in linked_ids else []),
        *(scene_id for scene_id in linked_ids if scene_id != continue_scene_id),
    ]
    bucket_map = {
        str(bucket.get("id") or "").strip(): bucket
        for bucket in all_buckets or []
        if isinstance(bucket, dict) and str(bucket.get("id") or "").strip()
    }
    refs = []
    for scene_id in ordered_ids:
        bucket = bucket_map.get(scene_id)
        if not bucket or not _is_canonical_scene_bucket(bucket):
            continue
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        if meta.get("active") is False or meta.get("deprecated") or meta.get("resolved"):
            continue
        if str(meta.get("source") or "").strip() != "window_shadow":
            continue
        if str(meta.get("window_shadow_id") or "").strip() != window_id:
            continue
        expected_hash = str(meta.get("scene_source_hash") or "").strip()
        if not expected_hash or expected_hash != WindowShadowStore.source_hash(
            str(bucket.get("content") or "").strip()
        ):
            continue
        title = re.sub(r"\s+", " ", str(meta.get("name") or scene_id)).strip()[:48]
        cues = normalize_scene_cues(meta.get("scene_cues"), limit=1, max_chars=80)
        refs.append(
            {
                "scene_id": scene_id,
                "title": title or scene_id,
                "date": _bucket_handoff_date(bucket),
                "cue": cues[0] if cues else "",
                "role": "open_thread" if scene_id == continue_scene_id else "context",
            }
        )
        if len(refs) >= max(1, min(int(limit or 3), 3)):
            break

    lines = []
    for ref in refs:
        fields = [
            f"[{ref['role']}]",
            str(ref.get("date") or ""),
            str(ref.get("title") or ""),
        ]
        if ref.get("cue"):
            fields.append(f"cue={ref['cue']}")
        fields.append(f"scene_id={ref['scene_id']}")
        lines.append(" · ".join(value for value in fields if value))
    return {
        "text": "\n".join(f"- {line}" for line in lines),
        "items": refs,
        "continue_scene_id": continue_scene_id if any(
            ref["scene_id"] == continue_scene_id for ref in refs
        ) else "",
        "linked_count": len(linked_ids),
        "omitted_count": max(0, len(linked_ids) - len(refs)),
    }


def _window_shadow_continue_scene(
    query: str,
    scene_index: dict,
    all_buckets: list[dict],
) -> dict:
    """Read exactly one verified parent-linked Scene for a bare first-turn continuation."""
    if not is_bare_window_continue_query(query):
        return {"text": "", "scene_id": "", "injected": False, "reason": "query_not_bare_continue"}
    continue_scene_id = str((scene_index or {}).get("continue_scene_id") or "").strip()
    if not continue_scene_id:
        return {"text": "", "scene_id": "", "injected": False, "reason": "no_continue_scene"}
    ref = next(
        (
            item
            for item in (scene_index or {}).get("items", []) or []
            if str(item.get("scene_id") or "").strip() == continue_scene_id
        ),
        None,
    )
    bucket = next(
        (
            item
            for item in all_buckets or []
            if isinstance(item, dict) and str(item.get("id") or "").strip() == continue_scene_id
        ),
        None,
    )
    content = str((bucket or {}).get("content") or "").strip()
    if not ref or not content:
        return {
            "text": "",
            "scene_id": continue_scene_id,
            "injected": False,
            "reason": "continue_scene_unavailable",
        }
    metadata_lines = [
        f"scene_id: {continue_scene_id}",
        f"title: {str(ref.get('title') or '').strip()}",
    ]
    if str(ref.get("date") or "").strip():
        metadata_lines.append(f"date: {str(ref.get('date') or '').strip()}")
    return {
        "text": "\n".join(metadata_lines) + "\n\n" + content,
        "scene_id": continue_scene_id,
        "injected": True,
        "reason": "bare_first_turn_continue",
    }


async def _build_handoff_breath(
    max_tokens: int = 1200,
    session_id: str = "",
    previous_session_id: str = "",
    parent_shadow_id: str = "",
    query: str = "",
    debug: bool = False,
) -> str:
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
    except Exception as e:
        logger.warning("Handoff breath bucket list failed / handoff 列桶失败: %s", e)
        all_buckets = []

    try:
        portrait_sections = portrait_engine.build_handoff_sections(max_recent_items=3)
    except Exception as e:
        logger.warning("Handoff portrait state failed / handoff portrait 状态失败: %s", e)
        portrait_sections = {}

    safe_session_id = str(session_id or "").strip()
    profile_cfg = config.get("persona", {}) if isinstance(config.get("persona", {}), dict) else {}
    profile_id = str(profile_cfg.get("profile_id") or "default").strip() or "default"
    window_session = (
        gateway_state_store.get_window_session(safe_session_id, profile_id=profile_id)
        if safe_session_id
        else None
    )
    declared_window = bool(window_session)
    resolved_parent_shadow_id = str(parent_shadow_id or "").strip()
    if not resolved_parent_shadow_id and window_session:
        resolved_parent_shadow_id = str(window_session.get("parent_shadow_id") or "").strip()
    resolved_previous_session_id = str(previous_session_id or "").strip()
    if not resolved_previous_session_id and window_session:
        resolved_previous_session_id = str(
            window_session.get("previous_conversation_id") or ""
        ).strip()

    current_focus = str(portrait_sections.get("current_focus") or "").strip()
    recent_continuity_status = _recent_continuity_fallback_status(portrait_sections)
    self_anchor = _format_handoff_self_anchor(all_buckets, limit=1)
    anchors = _format_handoff_anchors(all_buckets, limit=2)
    care_memos = _format_handoff_care_memos(session_id=session_id, limit=3)
    self_core = _trim_handoff_text_to_token_budget(self_anchor, 110)
    shadow_projection = _latest_window_shadow_handoff_projection(
        session_id=safe_session_id,
        parent_shadow_id=resolved_parent_shadow_id,
        allow_latest_fallback=not declared_window and not resolved_parent_shadow_id,
    )
    projected_handoff_note = str(shadow_projection.get("handoff_note") or "").strip()
    handoff_note_chars = handoff_note_char_count(projected_handoff_note)
    handoff_note_valid = bool(
        projected_handoff_note
        and HANDOFF_NOTE_MIN_CHARS <= handoff_note_chars <= HANDOFF_NOTE_MAX_CHARS
    )
    handoff_note = projected_handoff_note if handoff_note_valid else ""
    scene_index = (
        _window_shadow_scene_index(shadow_projection, all_buckets, limit=3)
        if handoff_note
        else {"text": "", "items": [], "continue_scene_id": "", "linked_count": 0, "omitted_count": 0}
    )
    continued_scene = (
        _window_shadow_continue_scene(query, scene_index, all_buckets)
        if handoff_note
        else {"text": "", "scene_id": "", "injected": False, "reason": "no_window_shadow"}
    )
    recent_continuity = (
        ""
        if handoff_note
        else str(recent_continuity_status.get("text") or "").strip()
    )
    emergency_raw = (
        {}
        if handoff_note or recent_continuity
        else _emergency_raw_handoff_projection(
            session_id=session_id,
            previous_session_id=resolved_previous_session_id,
        )
    )
    handoff_route = (
        "window_shadow"
        if handoff_note
        else "recent_continuity"
        if recent_continuity
        else "raw_events"
        if emergency_raw
        else "none"
    )
    _last_handoff_status.clear()
    _last_handoff_status.update(
        {
            "route": handoff_route,
            "observed_at": now_iso(),
            "session_id": safe_session_id,
            "previous_session_id": resolved_previous_session_id,
            "parent_shadow_id": resolved_parent_shadow_id,
            "window_shadow_id": str(shadow_projection.get("window_id") or ""),
            "window_shadow_handoff_note_valid": handoff_note_valid,
            "window_shadow_handoff_note_chars": handoff_note_chars,
            "linked_scene_ids": [
                str(item.get("scene_id") or "")
                for item in scene_index.get("items", [])
                if str(item.get("scene_id") or "")
            ],
            "continue_scene_id": str(scene_index.get("continue_scene_id") or ""),
            "linked_scene_omitted_count": int(scene_index.get("omitted_count") or 0),
            "continued_scene_injected": bool(continued_scene.get("injected")),
            "continued_scene_id": str(continued_scene.get("scene_id") or ""),
            "continued_scene_reason": str(continued_scene.get("reason") or ""),
            "recent_continuity_fresh": bool(recent_continuity_status.get("fresh")),
            "recent_continuity_reason": str(recent_continuity_status.get("reason") or ""),
            "recent_continuity_age_hours": recent_continuity_status.get("age_hours"),
            "recent_continuity_updated_at": str(
                recent_continuity_status.get("updated_at") or ""
            ),
            "emergency_raw_session_id": str(emergency_raw.get("session_id") or ""),
        }
    )

    sections = [
        (SELF_ANCHOR_TAG, self_core, 100, False),
        # A valid handoff_note is short by contract. Reserve it before every
        # optional section and preserve the current Haven's exact wording.
        ("Previous Window Shadow", handoff_note, 650, False, True),
        # Metadata-only down-drill entrances from the same verified parent Shadow.
        # Scene prose is never copied into the startup package.
        ("Linked Scene Index", str(scene_index.get("text") or ""), 240, True, True),
        # A bare first-turn `继续吧` reads exactly the authored primary Scene.
        # It is deliberately preserved whole and never diffused or summarized.
        ("Continued Scene", str(continued_scene.get("text") or ""), 1200, False, True),
        ("Current Focus", current_focus, 90, True),
        ("Recent Continuity", recent_continuity, 520, True),
        ("Emergency Recent Events", str(emergency_raw.get("text") or ""), 520, True),
        ("照顾备忘", care_memos, 180, True),
        ("Optional Anchors", anchors, 90, True),
    ]
    intro = "\n".join([
        "=== Handoff Context ===",
        "Use this compact private block to restore identity and life context in a new window. "
        "Do not treat it as a broad memory dump; use breath(query=...) for concrete events.",
    ])
    if debug:
        sections.append(
            (
                "Handoff Debug",
                f"portrait_state_path: {portrait_sections.get('state_path', getattr(portrait_engine, 'state_path', ''))}\n"
                f"portrait_updated_at: {portrait_sections.get('updated_at', '')}\n"
                f"portrait_last_run_date: {portrait_sections.get('last_run_date', '')}\n"
                f"handoff_route: {handoff_route}\n"
                f"window_declared: {declared_window}\n"
                f"window_conversation_id: {safe_session_id}\n"
                f"window_previous_conversation_id: {resolved_previous_session_id}\n"
                f"parent_shadow_id: {resolved_parent_shadow_id}\n"
                f"window_shadow_id: {shadow_projection.get('window_id', '')}\n"
                f"window_shadow_handoff_note_valid: {handoff_note_valid}\n"
                f"window_shadow_handoff_note_chars: {handoff_note_chars}\n"
                f"linked_scene_ids: {','.join(str(item.get('scene_id') or '') for item in scene_index.get('items', []))}\n"
                f"continue_scene_id: {scene_index.get('continue_scene_id', '')}\n"
                f"linked_scene_omitted_count: {scene_index.get('omitted_count', 0)}\n"
                f"continued_scene_injected: {continued_scene.get('injected', False)}\n"
                f"continued_scene_id: {continued_scene.get('scene_id', '')}\n"
                f"continued_scene_reason: {continued_scene.get('reason', '')}\n"
                f"recent_continuity_fresh: {recent_continuity_status.get('fresh', False)}\n"
                f"recent_continuity_reason: {recent_continuity_status.get('reason', '')}\n"
                f"recent_continuity_age_hours: {recent_continuity_status.get('age_hours', '')}\n"
                f"emergency_raw_session_id: {emergency_raw.get('session_id', '')}\n"
                f"emergency_raw_source: {emergency_raw.get('source', '')}\n"
                f"emergency_raw_inferred: {emergency_raw.get('inferred', '')}",
                100,
                True,
            )
        )
    return _format_budgeted_handoff_sections(intro, sections, max_tokens)


def _format_handoff_darkroom_door() -> str:
    try:
        status = darkroom_store.status()
    except Exception as exc:
        logger.warning("Handoff darkroom status failed / handoff 暗房状态失败: %s", exc)
        return ""
    count = int(status.get("count") or 0)
    last_entered = str(status.get("last_entered_at") or "").strip()
    lines = [
        str(status.get("door") or "暗房存在。门口只显示状态，不显示未显影正文。"),
        "darkroom_enter opens a new room by default; use new_room=false only when explicitly continuing the current active room. Unreleased draft text stays private until darkroom_view allows it.",
    ]
    if count:
        detail = f"entries={count}"
        if last_entered:
            detail += f", last_entered={last_entered}"
        lines.append(detail)
    else:
        lines.append("entries=0")
    return "\n".join(lines)


def _format_handoff_profile_facts(all_buckets: list[dict], limit: int = 6) -> str:
    facts = [bucket for bucket in all_buckets if _is_profile_fact_bucket(bucket)]
    facts.sort(
        key=lambda bucket: str(
            bucket.get("metadata", {}).get("updated_at")
            or bucket.get("metadata", {}).get("last_active")
            or bucket.get("metadata", {}).get("created")
            or ""
        ),
        reverse=True,
    )
    lines = []
    for bucket in facts[: max(0, limit)]:
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        sections = _profile_fact_sections(bucket.get("content", ""))
        fact = sections.get("fact") or strip_wikilinks(bucket.get("content", "")).strip()
        fact = _clip_text(fact, 220)
        if not fact:
            continue
        evidence = (
            meta.get("evidence_bucket_id")
            or meta.get("evidence_moment_id")
            or meta.get("source_bucket_id")
            or meta.get("source_moment_id")
        )
        bits = [f"bucket_id:{bucket.get('id', '')}"]
        if evidence:
            bits.append(f"evidence:{evidence}")
        lines.append(f"- [{' '.join(bits)}] {fact}")
    return "\n".join(lines)


def _format_handoff_relationship_weather(all_buckets: list[dict]) -> str:
    weather = []
    for bucket in all_buckets:
        if is_self_anchor_bucket(bucket):
            continue
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        tags = {str(tag) for tag in meta.get("tags", []) or []}
        if not ({"relationship_weather", "daily_impression", "weekly_impression"} & tags):
            continue
        weather.append(bucket)
    weather.sort(
        key=lambda bucket: str(
            bucket.get("metadata", {}).get("updated_at")
            or bucket.get("metadata", {}).get("created")
            or ""
        ),
        reverse=True,
    )
    lines = []
    recent_dates = _handoff_recent_date_keys()
    for index, bucket in enumerate(weather[:3]):
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        date_key = _bucket_handoff_date(bucket)
        bucket_id = str(bucket.get("id") or "")
        is_recent = bool(date_key and date_key in recent_dates)
        if is_recent or index == 0:
            text = _clip_text(_handoff_clean_summary_text(bucket.get("content", "")), 220)
            if text:
                prefix = f"{date_key}: " if date_key else ""
                lines.append(f"- [relationship_weather bucket_id:{bucket_id}] {prefix}{text}")
        else:
            text = _handoff_short_summary(bucket.get("content", ""), max_chars=72)
            if text:
                date_part = f"{date_key}: " if date_key else ""
                query = f"{date_key} 关系天气" if date_key else str(meta.get("name") or "关系天气")
                hint = _handoff_query_hint(query)
                suffix = f"；{hint}" if hint else ""
                lines.append(f"- [relationship_weather bucket_id:{bucket_id}] {date_part}{text}{suffix}")
    return "\n".join(lines)


def _format_handoff_recent_continuity(all_buckets: list[dict], limit: int = 3) -> str:
    candidates = []
    cutoff_hours = 24 * 4
    for bucket in all_buckets:
        if is_self_anchor_bucket(bucket):
            continue
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        if meta.get("pinned") or meta.get("protected") or meta.get("anchor"):
            continue
        if meta.get("active") is False or meta.get("deprecated"):
            continue
        if meta.get("type") == "feel":
            tags = {str(tag) for tag in meta.get("tags", []) or []}
            if not ({"relationship_weather", "daily_impression"} & tags):
                continue
        age = _bucket_age_hours(bucket)
        if age is None or age > cutoff_hours:
            continue
        candidates.append(bucket)
    candidates.sort(
        key=lambda bucket: str(
            bucket.get("metadata", {}).get("updated_at")
            or bucket.get("metadata", {}).get("created")
            or ""
        ),
        reverse=True,
    )
    lines = []
    for bucket in candidates[: max(0, limit)]:
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        name = str(meta.get("name") or bucket.get("id") or "").strip()
        created = str(meta.get("created") or "")[:10]
        text = _clip_text(bucket_content_for_recall(bucket), 160)
        if not text:
            continue
        lines.append(f"- [{created}] [bucket_id:{bucket.get('id', '')}] {name}: {text}")
    return "\n".join(lines)


def _is_pending_followup_domain(domain_key: str) -> bool:
    return domain_key in {"todo", "todos", "followup", "followups", "pending", "unfinished", "待办", "未完成"}


def _breath_query_requests_pending_followups(query: str) -> bool:
    text = str(query or "").strip().lower()
    if not text:
        return False
    compact = re.sub(r"\s+", "", text)
    if any(term in compact for term in ("待办", "没做完", "没做", "未完成", "还没做")):
        return True
    return any(term in text for term in ("todo", "pending", "unfinished", "followup", "follow-up"))


def _format_handoff_personal_recent_continuity(all_buckets: list[dict], limit: int = 3) -> str:
    rows = []
    recent_dates = _handoff_recent_date_keys()
    for bucket in all_buckets:
        if is_self_anchor_bucket(bucket):
            continue
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        tags = {str(tag) for tag in meta.get("tags", []) or []}
        if not ({"relationship_weather", "daily_impression"} & tags):
            continue
        date_key = _bucket_handoff_date(bucket)
        if date_key and date_key not in recent_dates:
            continue
        text = _handoff_clean_summary_text(bucket.get("content", ""))
        if not text:
            continue
        rows.append((date_key, str(meta.get("updated_at") or meta.get("created") or ""), text))
    rows.sort(key=lambda item: (item[0] or "", item[1]), reverse=True)
    lines = []
    for date_key, _updated, text in rows[: max(0, limit)]:
        trace = _handoff_persona_trace_for_date(date_key)
        weather = _handoff_recent_weather_phrase(text, max_chars=120 if trace else 180)
        date_label = date_key or "recent"
        if trace and weather:
            lines.append(f"- {date_label}: {trace}。关系天气：{weather}")
        elif trace:
            lines.append(f"- {date_label}: {trace}")
        elif weather:
            lines.append(f"- {date_label}: 关系天气：{weather}")
    return "\n".join(lines)


def _handoff_persona_trace_for_date(date_key: str, *, limit: int = 2) -> str:
    if not date_key or not hasattr(persona_engine, "_list_events"):
        return ""
    try:
        events = persona_engine._list_events(max(80, limit * 8))
    except Exception as exc:
        logger.warning("Handoff persona trace lookup failed / handoff persona trace 读取失败: %s", exc)
        return ""
    matched = []
    for event in events:
        if not isinstance(event, dict):
            continue
        created = _handoff_parse_local_datetime(event.get("created_at"))
        if created and created.date().isoformat() == date_key:
            matched.append(event)
    selected = select_persona_events(
        matched,
        limit=limit,
        identity_terms=_identity().get("relationship_terms") or [],
    )
    if not selected:
        return ""
    phrases = []
    for event in selected:
        phrase = _handoff_persona_event_phrase(event)
        if phrase:
            phrases.append(phrase)
    if not phrases:
        return ""
    return _clip_text("；".join(phrases), 180)


def _handoff_persona_event_phrase(event: dict) -> str:
    identity = _identity()
    user_name = identity.get("user_display_name") or identity.get("user_name") or "用户"
    ai_name = identity.get("ai_name") or "AI"
    user_excerpt = _handoff_clean_excerpt(event.get("user_excerpt"))
    assistant_excerpt = _handoff_clean_excerpt(event.get("assistant_excerpt"))
    parts = []
    if user_excerpt:
        parts.append(f"{user_name}说“{user_excerpt}”")
    if assistant_excerpt:
        parts.append(f"{ai_name}回“{assistant_excerpt}”")
    if not parts:
        return ""
    return _clip_text("；".join(parts), 150)


def _handoff_clean_excerpt(value: object, *, max_chars: int = 72) -> str:
    text = strip_wikilinks(str(value or "")).strip()
    if not text:
        return ""
    text = re.sub(r"\s*<attachment\b[^>]*>.*?</attachment>\s*", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"【当前时间】[^\n\r]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return _clip_text(text, max_chars)


def _handoff_recent_weather_phrase(text: str, *, max_chars: int = 150) -> str:
    clean = _handoff_clean_summary_text(text)
    clean = re.sub(r"^今天(?:的)?关系天气[：:]\s*", "", clean)
    clean = re.sub(r"^今天[：:]\s*", "", clean)
    return _clip_text(clean, max_chars)


def _handoff_parse_local_datetime(value: object) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(_handoff_timezone())


def _merge_handoff_recent_continuity(*blocks: str, max_lines: int = 5) -> str:
    lines = []
    seen = set()
    for block in blocks:
        for line in str(block or "").splitlines():
            line = line.strip()
            if not line or line in seen:
                continue
            seen.add(line)
            lines.append(line)
            if len(lines) >= max_lines:
                return "\n".join(lines)
    return "\n".join(lines)


def _handoff_context_line_key(line: str) -> str:
    text = str(line or "").strip()
    text = re.sub(
        r"^-\s+\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2})?(?:\s*/\s*[^:]+)?\s*:\s*",
        "",
        text,
    )
    text = re.sub(r"\s*\((?:bucket_id|moment_id|session_id):[^)]*\)\s*$", "", text)
    return re.sub(r"[\s，。；：、！？,.!?;:]+", "", text).lower()


def _remove_handoff_current_focus_overlap(recent: str, current_focus: str) -> str:
    focus_keys = [
        _handoff_context_line_key(line)
        for line in str(current_focus or "").splitlines()
        if len(_handoff_context_line_key(line)) >= 8
    ]
    if not focus_keys:
        return str(recent or "")
    kept = []
    for line in str(recent or "").splitlines():
        key = _handoff_context_line_key(line)
        if key and any(key == focus or key in focus or focus in key for focus in focus_keys):
            continue
        kept.append(line)
    return "\n".join(kept)


def _handoff_recent_continuity_is_natural(block: str) -> bool:
    lines = [line.strip() for line in str(block or "").splitlines() if line.strip()]
    if not lines:
        return False
    return any(
        re.match(r"^-\s+\d{4}-\d{2}-\d{2}:", line)
        and " / " not in line
        and not any(label in line for label in ("trace:", "personal:", "trigger:", "residue:"))
        for line in lines
    )


def _format_handoff_anchors(all_buckets: list[dict], limit: int = 2) -> str:
    lines = []
    for bucket in _select_anchor_buckets(all_buckets, limit=limit):
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        title = str(meta.get("name") or bucket.get("id") or "").strip()
        text = _handoff_short_summary(bucket.get("content", ""), max_chars=72)
        query = title or str(bucket.get("id") or "")
        hint = _handoff_query_hint(query)
        suffix = f"；{hint}" if hint else ""
        lines.append(f"- [bucket_id:{bucket.get('id', '')}] {title}: {text}{suffix}")
    return "\n".join(lines)


def _format_handoff_care_memos(session_id: str = "", limit: int = 3) -> str:
    safe_session = str(session_id or "").strip()
    try:
        next_round = gateway_state_store.get_current_round(safe_session) + 1
    except Exception:
        next_round = 0
    try:
        items = reminder_store.due(
            session_id=safe_session,
            channels=["gateway", "bridge"],
            round_id=next_round,
            now=datetime.now(LOCAL_TZ),
            limit=max(0, int(limit or 3)),
        )
    except Exception as exc:
        logger.warning("Handoff care memo lookup failed / handoff 照顾备忘读取失败: %s", exc)
        return ""
    rows: list[str] = []
    for item in items:
        date_hint = str(item.get("next_due_at") or item.get("start_at") or item.get("created_at") or "").strip()
        date_hint = date_hint[:10] if date_hint else "未定日期"
        title = _clip_text(item.get("title") or "照顾备忘", 40)
        content = _clip_text(item.get("content") or "", 90)
        if not content:
            continue
        rows.append(f"- {date_hint} {title}: {content}")
        if len(rows) >= max(0, int(limit or 3)):
            break
    return "\n".join(rows)


def _has_favorite_tag(tags: list | set | tuple | None) -> bool:
    return has_favorite_policy_tag(tags, ai_name=_ai_author_name())


_BASE_FAVORITE_REFLECTION_HEADINGS = {
    "reflection",
    "assistantreflection",
    "favoritereason",
    "喜欢它的原因",
    "喜欢的原因",
}
_LEGACY_FAVORITE_REFLECTION_HEADINGS = {
    "havenreflection",
    "haven喜欢它的原因",
    "haven喜欢的原因",
}


def _normalize_section_heading(value: str) -> str:
    text = strip_wikilinks(str(value or "")).strip().lower()
    text = re.sub(r"[#>`*_~]+", "", text).strip()
    text = re.sub(r"[:：].*$", "", text).strip()
    return re.sub(r"[\s_\-·/|（）()【】\[\]]+", "", text)


def _favorite_reflection_headings() -> set[str]:
    headings = set(_BASE_FAVORITE_REFLECTION_HEADINGS) | set(_LEGACY_FAVORITE_REFLECTION_HEADINGS)
    ai_heading = _normalize_section_heading(_ai_author_name())
    if ai_heading:
        headings.update(
            {
                f"{ai_heading}reflection",
                f"{ai_heading}喜欢它的原因",
                f"{ai_heading}喜欢的原因",
            }
        )
    return headings


def _has_favorite_reflection(content: str) -> bool:
    text = strip_wikilinks(str(content or ""))
    headings = _favorite_reflection_headings()
    for match in re.finditer(r"(?m)^\s{0,3}#{2,6}\s+(.+?)\s*$", text):
        if _normalize_section_heading(match.group(1)) in headings:
            return True
    return False


def _has_favorite_reason(content: str) -> bool:
    return _has_favorite_reflection(content)


def _favorite_reason_error() -> str:
    return "标记 favorite memory 需要在正文写明「### reflection」。旧的「喜欢它的原因」仍兼容。"


_NEW_WRITE_FORBIDDEN_HEADINGS = {
    "followup",
    "followups",
    "followuplog",
    "followupslog",
    "todo",
    "todolog",
    "next",
    "后续",
    "后续待办",
    "后续记录",
    "待办",
    "待办事项",
    "待办记录",
    "affectanchor",
}
_NEW_WRITE_REFLECTION_HEADINGS = {
    "reflection",
    "assistantreflection",
    "havenreflection",
}
_HOLD_SCENE_ALLOWED_HEADINGS = {
    "scene",
}


def _uses_first_person_voice(text: str) -> bool:
    value = str(text or "").strip()
    return bool("我" in value or re.search(r"(?i)\b(?:i|me|my|mine|myself)\b", value))


def _memory_write_contract_error(content: str, *, feel_only: bool = False) -> str:
    """Reject new memory shapes that the model should rewrite instead of persisting."""
    text = strip_wikilinks(str(content or "")).strip()
    matches = list(re.finditer(r"(?m)^\s{0,3}#{1,6}\s+(.+?)\s*$", text))
    if feel_only:
        if matches:
            return "feel 只写第一人称正文，不写标题、Markdown 分段或 ### section。"
        if not _uses_first_person_voice(text):
            return "feel 必须改成第一人称正文，用“我……”表达模型自己的感受。"
        return ""

    for index, match in enumerate(matches):
        heading = _normalize_section_heading(match.group(1))
        if heading in _NEW_WRITE_FORBIDDEN_HEADINGS:
            if heading == "affectanchor":
                return "新记忆不接受 ### affect_anchor；它不是模型可写的 content section。"
            return (
                "新记忆不接受 ### followup / todo。需要长期保留的回应变化请改写进第一人称 "
                "### reflection；需要到时提醒的事项请用 reminder_create。"
            )
        if heading not in _NEW_WRITE_REFLECTION_HEADINGS:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        reflection = text[match.end():end].strip()
        if not _uses_first_person_voice(reflection):
            return "### reflection 必须用模型第一人称写，用“我记得 / 我明白 / 我以后 / 我会”等表达。"
    return ""


def _hold_scene_contract_error(content: str) -> str:
    """Require one plain authored Scene body with no legacy section wrappers."""
    text = strip_wikilinks(str(content or "")).strip()
    if not text:
        return "Scene 正文为空，未保存。"
    heading = re.search(r"(?m)^\s{0,3}#{1,6}\s+\S.*$", text)
    if heading:
        return (
            "Scene 已经是记忆对象类型，正文不要再写 `## Scene`、`### scene`、`### moment` "
            "或其他 Markdown section；只传那段原文经历。"
        )
    return ""


def _authored_scene_title(content: str, explicit_title: str = "") -> str:
    title = str(explicit_title or "").strip()
    return title[:48] if title else ""


def _authored_scene_cues(content: str, *, title: str = "", explicit: object = None) -> list[str]:
    """Build a small cue sidecar without summarizing or changing the Scene text."""
    values: list[str] = []
    values.extend(normalize_scene_cues(explicit))
    if title and not re.fullmatch(r"(?:窗影)?(?:时刻|场景)\d+", title):
        values.append(title)

    text = strip_wikilinks(str(content or ""))
    for quoted in re.findall(r"[“\"]([^“”\"\n]{2,80})[”\"]", text):
        values.append(quoted)
    return normalize_scene_cues(values)


def _normalize_memory_sections_for_write(content: str) -> str:
    """Keep new tool-written buckets in the current moment/reflection/anchor shape."""
    raw = str(content or "").strip()
    if not raw:
        return ""
    migration = plan_bucket_migration(
        {"id": "write_preview", "content": raw, "metadata": {"name": "write_preview"}},
        body_only_moment="skip",
    )
    if migration:
        return migration.new_content.strip()
    return raw


def _has_memory_section(content: str, section: str) -> bool:
    target = _normalize_section_heading(section)
    text = strip_wikilinks(str(content or ""))
    for match in re.finditer(r"(?m)^\s{0,3}#{2,6}\s+(.+?)\s*$", text):
        if _normalize_section_heading(match.group(1)) == target:
            return True
    return False


def _leading_body_text(content: str) -> str:
    raw = str(content or "").strip()
    if not raw:
        return ""
    match = re.search(r"(?m)^\s{0,3}#{2,6}\s+\S.*$", raw)
    return (raw[: match.start()] if match else raw).strip()


def _fallback_moment_from_body(body_text: str) -> str:
    text = re.sub(r"\s+", " ", str(body_text or "").strip())
    if not text:
        return ""
    sentences = [part.strip() for part in re.findall(r"[^。！？!?]+[。！？!?]?", text) if part.strip()]
    selected: list[str] = []
    for sentence in sentences:
        candidate = "".join(selected + [sentence])
        if selected and len(candidate) > 240:
            break
        selected.append(sentence)
        if len(selected) >= 3 or len(candidate) >= 120:
            break
    summary = "".join(selected) or text
    return _clip_text(summary, 240).strip()


def _insert_moment_after_leading_body(content: str, moment: str) -> str:
    raw = str(content or "").strip()
    text = str(moment or "").strip()
    if not raw or not text:
        return raw
    moment_block = f"### moment\n{text}"
    match = re.search(r"(?m)^\s{0,3}#{2,6}\s+\S.*$", raw)
    if not match:
        return f"{raw}\n\n{moment_block}"
    body = raw[: match.start()].strip()
    rest = raw[match.start():].lstrip()
    if body:
        return f"{body}\n\n{moment_block}\n\n{rest}"
    return f"{moment_block}\n\n{rest}"


def _section_text_for_auto_moment(content: str) -> str:
    sections = _profile_fact_sections(content)
    for key in (
        SELF_ANCHOR_TAG,
        "self_anchor",
        "first_person_anchor",
        "anchor",
        "fact",
    ):
        text = str(sections.get(_profile_key(key, ""), "") or "").strip()
        if text:
            return text
    return ""


def _reflection_text_for_auto_moment(content: str) -> str:
    raw = strip_wikilinks(str(content or "")).strip()
    if not raw:
        return ""
    matches = list(re.finditer(r"(?m)^\s{0,3}#{2,6}\s+(.+?)\s*$", raw))
    for index, match in enumerate(matches):
        heading = _normalize_section_heading(match.group(1))
        if heading not in {"reflection", "assistantreflection", "havenreflection"}:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        return raw[match.end():end].strip()
    return ""


async def _auto_generate_moment_if_missing(content: str, *, section_fallback: bool = False) -> str:
    raw = str(content or "").strip()
    if not raw or _has_memory_section(raw, "moment"):
        return raw
    body_text = _leading_body_text(raw)
    if (not body_text or len(body_text) < 10) and section_fallback:
        body_text = _section_text_for_auto_moment(raw)
    if not body_text or len(body_text) < 10:
        return raw

    reflection_text = _reflection_text_for_auto_moment(raw)
    generated_moment = ""
    generator = getattr(dehydrator, "generate_moment", None)
    if callable(generator):
        try:
            generated_moment = await generator(body_text, reflection_text)
        except Exception as e:
            logger.warning("Auto moment generation failed / 自动 moment 生成失败: %s", e)

    generated_moment = str(generated_moment or "").strip() or _fallback_moment_from_body(body_text)
    return _insert_moment_after_leading_body(raw, generated_moment) if generated_moment else raw


def _is_self_anchor_write_content(
    self_anchor: object = False,
    domain: list | tuple | set | str | None = None,
) -> bool:
    return is_self_anchor_metadata({"self_anchor": self_anchor, "domain": domain or []})


async def _auto_generate_write_moment_if_needed(
    content: str,
    tags: list | tuple | set | None = None,
    *,
    self_anchor: object = False,
    domain: list | tuple | set | str | None = None,
) -> str:
    _ = tags
    if _is_self_anchor_write_content(self_anchor, domain):
        return str(content or "").strip()
    return await _auto_generate_moment_if_missing(content)


_SHADOW_CUE_KINDS = {
    "paraphrase",
    "entity",
    "event",
    "time",
    "relationship",
    "protected_phrase",
}
_GENERIC_SHADOW_CUE_KEYS = {
    "以前",
    "之前",
    "当时",
    "后来",
    "关系",
    "记忆",
    "事情",
    "开心",
    "难过",
}


def _generic_shadow_cue_keys() -> set[str]:
    keys = set(_GENERIC_SHADOW_CUE_KEYS)
    keys.update(
        re.sub(r"[\s\W_]+", "", str(term or "").lower())
        for term in _identity().get("relationship_terms") or []
        if str(term or "").strip()
    )
    return {key for key in keys if key}


def _shadow_memory_card_source_hash(bucket: dict) -> str:
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    payload = {
        "content": str(bucket.get("content") or ""),
        "name": str(meta.get("name") or bucket.get("name") or ""),
        "tags": sorted(str(tag) for tag in meta.get("tags", []) or []),
        "domain": sorted(str(domain) for domain in meta.get("domain", []) or []),
        "date": str(meta.get("date") or ""),
    }
    raw = _json_lib.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _shadow_memory_card_parts(bucket: dict) -> dict:
    moments = parse_bucket_moments(bucket)

    def first_text(*sections: str) -> str:
        for section in sections:
            for moment in moments:
                if str(moment.get("section") or "") == section:
                    text = str(moment.get("text") or "").strip()
                    if text:
                        return text
        return ""

    current_moment = first_text("moment")
    body = first_text("body", "fact", "evidence_context", "context")
    reflection = first_text("reflection", "feeling")
    if not body:
        body = _leading_body_text(str(bucket.get("content") or ""))
    if not body:
        body = current_moment

    evidence_refs = []
    for moment in moments:
        section = str(moment.get("section") or "")
        if section not in {
            "body",
            "fact",
            "moment",
            "original",
            "evidence_context",
            "context",
            "reflection",
        }:
            continue
        item = {
            "moment_id": str(moment.get("moment_id") or ""),
            "section": section,
            "source": str(moment.get("source") or ""),
            "source_id": str(moment.get("source_id") or ""),
        }
        meta = moment.get("metadata", {}) if isinstance(moment.get("metadata"), dict) else {}
        source_ref = meta.get("source_ref")
        if isinstance(source_ref, dict):
            item["source_ref"] = dict(source_ref)
        evidence_refs.append(item)
        if len(evidence_refs) >= 12:
            break
    return {
        "moments": moments,
        "body": body,
        "reflection": reflection,
        "current_moment": current_moment,
        "evidence_refs": evidence_refs,
    }


def _shadow_memory_card_needs_primary_candidate(parts: dict) -> bool:
    current = " ".join(str(parts.get("current_moment") or "").split())
    body = " ".join(str(parts.get("body") or "").split())
    reflection = " ".join(str(parts.get("reflection") or "").split())
    if not current:
        return True
    if len(current) >= 60:
        return False
    return len(body) >= len(current) + 30 or bool(reflection)


def _normalize_shadow_cue_anchors(value: object, limit: int = 8) -> list[dict]:
    if not isinstance(value, list):
        return []
    anchors: list[dict] = []
    seen = set()
    generic_keys = _generic_shadow_cue_keys()
    for raw in value:
        if isinstance(raw, str):
            cue = raw
            kind = "paraphrase"
            why = ""
        elif isinstance(raw, dict):
            cue = raw.get("cue")
            kind = str(raw.get("kind") or "paraphrase").strip().lower()
            why = raw.get("why")
        else:
            continue
        cue = re.sub(r"\s+", " ", str(cue or "")).strip(" \t\r\n\"'“”‘’")
        cue = _clip_text(cue, 80)
        cue_key = re.sub(r"[\s\W_]+", "", cue.lower())
        if len(cue_key) < 2 or cue_key in generic_keys or cue_key in seen:
            continue
        seen.add(cue_key)
        anchors.append(
            {
                "cue": cue,
                "kind": kind if kind in _SHADOW_CUE_KINDS else "paraphrase",
                "why": _clip_text(re.sub(r"\s+", " ", str(why or "")).strip(), 120),
            }
        )
        if len(anchors) >= max(1, int(limit)):
            break
    return anchors


def _shadow_memory_card_eligible(bucket: dict | None) -> bool:
    if not isinstance(bucket, dict) or is_self_anchor_bucket(bucket):
        return False
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    if meta.get("type") == "feel" or meta.get("protected"):
        return False
    return bool(str(bucket.get("content") or "").strip())


async def _build_shadow_memory_card(bucket: dict) -> dict:
    bucket_id = str(bucket.get("id") or "").strip()
    parts = _shadow_memory_card_parts(bucket)
    body = str(parts.get("body") or "").strip()
    current_moment = str(parts.get("current_moment") or "").strip()
    reflection = str(parts.get("reflection") or "").strip()
    needs_candidate = _shadow_memory_card_needs_primary_candidate(parts)
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    generated = {}
    generator = getattr(dehydrator, "generate_memory_card", None)
    if body and callable(generator):
        try:
            generated = await generator(
                body=body,
                reflection=reflection,
                current_moment=current_moment,
                title=str(meta.get("name") or bucket.get("name") or ""),
                tags=list(meta.get("tags", []) or []),
                needs_primary_candidate=needs_candidate,
            )
        except Exception as e:
            logger.warning("Shadow memory card generation failed / 影子记忆卡生成失败: %s: %s", bucket_id, e)

    if not isinstance(generated, dict):
        generated = {}
    generated_primary = re.sub(
        r"^#{1,6}\s*moment\s*\n+",
        "",
        str(generated.get("primary_abstraction") or "").strip(),
        flags=re.IGNORECASE,
    ).strip()
    if needs_candidate:
        primary = generated_primary or _fallback_moment_from_body(body)
        primary_source = "generated_candidate" if generated_primary else "body_fallback_candidate"
    else:
        primary = current_moment
        primary_source = "existing_moment"

    cue_anchors = _normalize_shadow_cue_anchors(generated.get("cue_anchors"), limit=8)
    if not primary:
        status = "skipped_no_primary"
    elif not cue_anchors:
        status = "partial_no_cues"
    else:
        status = "ready"
    return {
        "bucket_id": bucket_id,
        "source_hash": _shadow_memory_card_source_hash(bucket),
        "generation_version": MEMORY_CARD_SHADOW_VERSION,
        "primary_abstraction": _clip_text(primary, 360),
        "existing_moment": _clip_text(current_moment, 360),
        "primary_source": primary_source,
        "candidate_only": bool(needs_candidate),
        "cue_anchors": cue_anchors,
        "evidence_refs": parts.get("evidence_refs", []),
        "status": status,
        "model": str(getattr(dehydrator, "model", "") or ""),
    }


def _bucket_read_payload(bucket: dict) -> dict:
    meta = bucket.get("metadata", {})
    metadata_view = normalize_memory_metadata(bucket)
    fields = [
        "id",
        "name",
        "type",
        "domain",
        "tags",
        "facets",
        "importance",
        "valence",
        "arousal",
        "model_valence",
        "pinned",
        "protected",
        "resolved",
        "digested",
        "anchor",
        "source",
        "memory_value_source",
        "object_kind",
        "write_contract",
        "scene_cues",
        "confidence",
        "period",
        "date",
        "event_date",
        "created",
        "updated_at",
        "last_active",
        "activation_count",
        "comment_count",
        "comments",
        "profile_kind",
        "subject",
        "predicate",
        "object",
        "evidence",
        "source_bucket_ids",
        "source_persona_event_ids",
        "source_conversation_turn_ids",
        "source_raw_event_ids",
        "active",
        "deprecated",
    ]
    object_type = _bucket_object_kind(bucket)
    return {
        "id": bucket["id"],
        "object_type": object_type,
        "metadata": {key: meta.get(key) for key in fields if key in meta},
        "metadata_view": metadata_view,
        **metadata_view,
        "content": strip_wikilinks(bucket.get("content", "")),
        "score": decay_engine.calculate_score(meta),
    }


def _bucket_summary_payload(bucket: dict) -> dict:
    meta = bucket.get("metadata", {}) if isinstance(bucket, dict) else {}
    metadata_view = normalize_memory_metadata(bucket)
    return {
        "id": bucket.get("id", ""),
        "name": meta.get("name", bucket.get("id", "")),
        "type": meta.get("type", "dynamic"),
        "object_kind": _bucket_object_kind(bucket),
        "domain": meta.get("domain", []),
        "tags": meta.get("tags", []),
        "facets": meta.get("facets", []),
        "metadata_view": metadata_view,
        **metadata_view,
        "importance": meta.get("importance", 5),
        "valence": meta.get("valence", 0.5),
        "arousal": meta.get("arousal", 0.5),
        "confidence": meta.get("confidence", 0.5),
        "pinned": meta.get("pinned", False),
        "protected": meta.get("protected", False),
        "anchor": meta.get("anchor", False),
        "resolved": meta.get("resolved", False),
        "digested": meta.get("digested", False),
        "created": meta.get("created", ""),
        "updated_at": meta.get("updated_at", ""),
        "last_active": meta.get("last_active", ""),
        "content_preview": strip_wikilinks(bucket.get("content", ""))[:200],
    }


def _metadata_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _bucket_light_payload(bucket: dict) -> dict:
    meta = bucket.get("metadata", {}) if isinstance(bucket, dict) else {}
    metadata_view = normalize_memory_metadata(bucket)
    return {
        "id": bucket.get("id", ""),
        "bucket_id": bucket.get("id", ""),
        "name": meta.get("name", bucket.get("id", "")),
        "type": meta.get("type", "dynamic"),
        "object_kind": _bucket_object_kind(bucket),
        "domain": meta.get("domain", []),
        "tags": meta.get("tags", []),
        "facets": meta.get("facets", []),
        "source": meta.get("source", ""),
        "importance": meta.get("importance", 5),
        "confidence": meta.get("confidence", 0.5),
        "created": _metadata_text(meta.get("created")),
        "updated_at": _metadata_text(meta.get("updated_at")),
        "last_active": _metadata_text(meta.get("last_active")),
        "resolved": bool(meta.get("resolved", False)),
        "digested": bool(meta.get("digested", False)),
        "pinned": bool(meta.get("pinned", False)),
        "protected": bool(meta.get("protected", False)),
        "anchor": bool(meta.get("anchor", False)),
        "self_anchor": is_self_anchor_bucket(bucket),
        "metadata_view": metadata_view,
        **metadata_view,
    }


def _bucket_light_sort_key(item: dict) -> str:
    return str(item.get("created") or "")


def _bucket_dashboard_sort_key(item: dict) -> tuple[str, str]:
    return (
        str(item.get("created") or ""),
        str(item.get("id") or ""),
    )


def _identity_seed_alias_terms() -> set[str]:
    terms = set()
    try:
        terms |= {
            str(alias).strip()
            for node in identity_semantic_store.load_private_nodes()
            for alias in node.seed_aliases
            if str(alias).strip()
        }
    except Exception as e:
        logger.warning("Failed to load private identity seed aliases: %s", e)
    try:
        terms |= {str(item).strip() for item in reflection_identity_terms(config) if str(item).strip()}
    except Exception as e:
        logger.warning("Failed to load reflection identity role aliases: %s", e)
    return terms


def _refresh_word_map_private_terms(store: WordMapStore | None = None) -> list[str]:
    terms = _identity_seed_alias_terms()
    target = store or word_map_store
    if terms:
        target.private_terms |= terms
    return sorted(terms)


async def _rebuild_word_map_index(
    store: WordMapStore,
    manager: BucketManager,
    *,
    include_archive: bool = False,
) -> dict:
    private_terms = _refresh_word_map_private_terms(store)
    buckets = await manager.list_all(include_archive=include_archive)
    buckets = [bucket for bucket in buckets if not is_self_anchor_bucket(bucket)]
    stats = store.rebuild(buckets)
    return {
        "status": "rebuilt",
        "bucket_count": len(buckets),
        "include_archive": include_archive,
        "stats": stats,
        "private_terms_excluded": private_terms,
    }


def _word_map_payload(nodes_limit: int = 50, edges_limit: int = 50) -> dict:
    return {
        "enabled": bool(getattr(word_map_store, "enabled", False)),
        "stats": word_map_store.stats(),
        "nodes": word_map_store.list_nodes(_int_between(nodes_limit, 50, 1, 500)),
        "edges": word_map_store.list_edges(_int_between(edges_limit, 50, 1, 500)),
        "private_terms_excluded": _refresh_word_map_private_terms(),
    }


def _word_map_daily_rebuild_settings(config_arg: dict | None = None) -> dict[str, int | bool]:
    cfg_source = config_arg if isinstance(config_arg, dict) else config
    word_map_cfg = cfg_source.get("word_map", {}) if isinstance(cfg_source.get("word_map", {}), dict) else {}
    return {
        "enabled": _bool_value(word_map_cfg.get("enabled"), False)
        and _bool_value(word_map_cfg.get("daily_rebuild_enabled"), True),
        "hour": _int_between(word_map_cfg.get("daily_rebuild_hour"), 4, 0, 23),
        "minute": _int_between(word_map_cfg.get("daily_rebuild_minute"), 30, 0, 59),
        "include_archive": _bool_value(word_map_cfg.get("daily_rebuild_include_archive"), False),
        "check_interval_seconds": _int_between(
            word_map_cfg.get("daily_rebuild_check_interval_minutes"),
            15,
            1,
            1440,
        )
        * 60,
    }


def _word_map_daily_target(now: datetime, settings: dict[str, int | bool]) -> datetime:
    return now.replace(
        hour=int(settings.get("hour") or 0),
        minute=int(settings.get("minute") or 0),
        second=0,
        microsecond=0,
    )


def _word_map_should_run_daily_rebuild(
    now: datetime,
    last_run_date: str,
    settings: dict[str, int | bool],
) -> bool:
    if not settings.get("enabled"):
        return False
    date_key = now.date().isoformat()
    if last_run_date == date_key:
        return False
    return now >= _word_map_daily_target(now, settings)


def _identity_semantics_payload(alias_limit: int = 100) -> dict:
    aliases = identity_semantic_store.list_aliases()
    return {
        "enabled": bool(getattr(identity_semantic_store, "enabled", False)),
        "private_configured": bool(getattr(identity_semantic_store, "private_config_path", "")),
        "stats": identity_semantic_store.stats(),
        "aliases": aliases[: _int_between(alias_limit, 100, 1, 1000)],
    }


def _is_profile_fact_bucket(bucket: dict) -> bool:
    if is_self_anchor_bucket(bucket):
        return False
    meta = bucket.get("metadata", {}) if isinstance(bucket, dict) else {}
    tags = {str(tag).strip() for tag in meta.get("tags", []) or [] if str(tag).strip()}
    return "profile_fact" in tags or bool(meta.get("profile_kind"))


def _is_canonical_scene_bucket(bucket: dict | None) -> bool:
    if not isinstance(bucket, dict) or _is_profile_fact_bucket(bucket):
        return False
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    if meta.get("type") == "feel" or _is_daily_impression_feel_bucket(bucket):
        return False
    if str(meta.get("object_kind") or "").strip().lower() == "scene":
        return True
    if str(meta.get("memory_value_source") or "") == "authored_scene":
        return True
    return any(
        moment.get("section") == "scene"
        for moment in parse_bucket_moments(bucket, _recall_relevance_options())
    )


def _bucket_object_kind(bucket: dict | None) -> str:
    if not isinstance(bucket, dict):
        return "legacy_bucket"
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    explicit = str(meta.get("object_kind") or "").strip().lower()
    if explicit:
        return explicit
    if _is_canonical_scene_bucket(bucket):
        return "scene"
    bucket_type = str(meta.get("type") or "").strip().lower()
    if bucket_type == "feel":
        return "feel"
    if bucket_type == "diary_source":
        return "diary_source"
    return "legacy_bucket"


def _bucket_edges_for_recall(bucket_map: dict[str, dict]) -> list[dict]:
    """Combine reviewed Scene edges with one-hop discounted legacy compatibility."""
    scene_ids = {
        str(bucket_id)
        for bucket_id, bucket in (bucket_map or {}).items()
        if _is_canonical_scene_bucket(bucket)
    }
    scene_edges = scene_linker.recall_scene_edges(
        {scene_id: bucket_map[scene_id] for scene_id in scene_ids}
    )
    legacy_edges = legacy_edges_for_recall(
        memory_edge_store.list_edges(),
        scene_ids=scene_ids,
    )
    return [*scene_edges, *legacy_edges]


def _is_fact_evidence_recall_bucket(
    bucket: dict | None,
    *,
    evidence_moment_id: str = "",
) -> bool:
    """Accept canonical Scenes plus explicit legacy evidence during migration.

    New ProfileFacts are still required to bind to a canonical Scene. This is
    only a read-compatibility bridge for old Facts whose recorded evidence
    points to a content body/original/moment. Comments and reflections never
    become Fact evidence through this path.
    """
    if not isinstance(bucket, dict) or _is_profile_fact_bucket(bucket) or is_self_anchor_bucket(bucket):
        return False
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    tags = {str(tag).strip().lower() for tag in meta.get("tags", []) or [] if str(tag).strip()}
    if (
        meta.get("type") == "feel"
        or infer_bucket_layer(bucket) == LAYER_SOURCE_RECORD
        or {"daily_impression", "weekly_impression", "relationship_weather"} & tags
    ):
        return False
    allowed_sections = {"scene", "body", "original", "moment"}
    moments = [
        moment
        for moment in parse_bucket_moments(bucket, _recall_relevance_options())
        if moment.get("source") == "content" and moment.get("section") in allowed_sections
    ]
    requested_moment = str(evidence_moment_id or "").strip()
    if requested_moment:
        return any(str(moment.get("moment_id") or "") == requested_moment for moment in moments)
    return bool(moments)


def _profile_fact_sections(content: str) -> dict[str, str]:
    text = strip_wikilinks(str(content or "")).strip()
    if not text:
        return {}
    matches = list(re.finditer(r"(?m)^###\s+([^\n]+)\n?", text))
    if not matches:
        return {"fact": text}
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        heading = _profile_key(match.group(1), "")
        if not heading:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[heading] = text[start:end].strip()
    if "fact" not in sections and text[: matches[0].start()].strip():
        sections["fact"] = text[: matches[0].start()].strip()
    return sections


def _profile_kind_from_tags(tags: list | tuple | set | None) -> str:
    for tag in tags or []:
        text = str(tag or "").strip()
        if not text.startswith("profile_"):
            continue
        if text in {"profile_fact", "profile_predicate"} or text.startswith("profile_predicate_"):
            continue
        return text.removeprefix("profile_")
    return ""


def _profile_fact_state(meta: dict) -> str:
    if meta.get("deprecated") or meta.get("active") is False:
        return "deprecated"
    if meta.get("resolved") or meta.get("digested"):
        return "inactive"
    return "active"


def _profile_fact_evidence(bucket: dict) -> list[dict]:
    meta = bucket.get("metadata", {}) if isinstance(bucket, dict) else {}
    rows: list[dict] = []

    raw_evidence = meta.get("evidence")
    if isinstance(raw_evidence, dict):
        raw_evidence = [raw_evidence]
    if isinstance(raw_evidence, list):
        for item in raw_evidence:
            if not isinstance(item, dict):
                continue
            bucket_id = str(item.get("bucket_id") or item.get("id") or "").strip()
            moment_id = str(item.get("moment_id") or "").strip()
            if bucket_id:
                rows.append({"bucket_id": bucket_id, "moment_id": moment_id})

    for bucket_key, moment_key in (
        ("evidence_bucket_id", "evidence_moment_id"),
        ("source_bucket_id", "source_moment_id"),
    ):
        bucket_id = str(meta.get(bucket_key) or "").strip()
        moment_id = str(meta.get(moment_key) or "").strip()
        if bucket_id:
            rows.append({"bucket_id": bucket_id, "moment_id": moment_id})

    try:
        for edge in memory_edge_store.list_edges():
            if (
                str(edge.get("source") or "") == str(bucket.get("id") or "")
                and str(edge.get("relation_type") or "") == "evidenced_by"
            ):
                rows.append({"bucket_id": str(edge.get("target") or ""), "moment_id": ""})
    except Exception:
        pass

    bucket_ids_with_moment = {
        str(item.get("bucket_id") or "").strip()
        for item in rows
        if str(item.get("bucket_id") or "").strip() and str(item.get("moment_id") or "").strip()
    }
    seen: set[tuple[str, str]] = set()
    result: list[dict] = []
    for item in rows:
        bucket_id = str(item.get("bucket_id") or "").strip()
        moment_id = str(item.get("moment_id") or "").strip()
        if not bucket_id:
            continue
        if not moment_id and bucket_id in bucket_ids_with_moment:
            continue
        key = (bucket_id, moment_id)
        if key in seen:
            continue
        seen.add(key)
        result.append({"bucket_id": bucket_id, "moment_id": moment_id})
    return result


async def _profile_fact_payload(bucket: dict) -> dict:
    meta = bucket.get("metadata", {}) if isinstance(bucket, dict) else {}
    sections = _profile_fact_sections(bucket.get("content", ""))
    kind = str(meta.get("profile_kind") or _profile_kind_from_tags(meta.get("tags", [])) or "").strip()
    evidence_rows = []
    for item in _profile_fact_evidence(bucket):
        bucket_id = item["bucket_id"]
        evidence_bucket = await bucket_mgr.get(bucket_id) if MEMORY_ID_RE.fullmatch(bucket_id) else None
        evidence_meta = evidence_bucket.get("metadata", {}) if evidence_bucket else {}
        evidence_rows.append({
            "bucket_id": bucket_id,
            "moment_id": item.get("moment_id", ""),
            "name": evidence_meta.get("name", bucket_id),
            "exists": bool(evidence_bucket),
        })

    state = _profile_fact_state(meta)
    return {
        "id": bucket.get("id", ""),
        "name": meta.get("name", bucket.get("id", "")),
        "fact": sections.get("fact", strip_wikilinks(bucket.get("content", "")).strip()),
        "sections": sections,
        "kind": kind,
        "subject": meta.get("subject", ""),
        "predicate": meta.get("predicate", ""),
        "object": meta.get("object", ""),
        "evidence": evidence_rows,
        "confidence": meta.get("confidence"),
        "source": meta.get("source", "profile_fact"),
        "active": state == "active",
        "deprecated": state == "deprecated",
        "state": state,
        "tags": meta.get("tags", []),
        "created": meta.get("created", ""),
        "updated_at": meta.get("updated_at", ""),
        "last_active": meta.get("last_active", ""),
        "content_preview": strip_wikilinks(bucket.get("content", ""))[:200],
    }


def _profile_fact_tags(current_tags: list | tuple | set | None, kind: str, predicate: str) -> list[str]:
    tags = ["profile_fact", f"profile_{kind}"]
    if predicate:
        tags.append(f"profile_predicate_{predicate}")
    for tag in current_tags or []:
        text = str(tag or "").strip()
        if not text:
            continue
        if text == "profile_fact" or text.startswith("profile_"):
            continue
        tags.append(text)
    return list(dict.fromkeys(tags))


PROFILE_FACT_PROPOSAL_PROMPT_TEMPLATE = """你是一个证据化用户画像候选生成器。请只根据给定证据桶提出可能值得长期保存的画像事实。

身份：
- 当前用户：{user_display_name}
- 当前 AI：{ai_name}

边界：
1. 只能提出能被证据直接支持的事实，不要补常识，不要推测。
2. 不要提出 root prompt、pinned、protected、Core Memory 更新。
3. 不要把短期情绪当长期画像，除非证据明确显示稳定偏好、边界、习惯、关系锚点或重要日期。
4. 如果证据不足，返回 []。
5. 只输出 JSON 数组，不要 markdown，不要解释。

每个候选必须包含：
{{
  "fact": "一句可读中文事实",
  "profile_kind": "preference|boundary|habit|identity|relationship_anchor|life_fact|work_state|other",
  "subject": "user|ai|relationship",
  "predicate": "snake_case_or_short_key",
  "object": "事实对象，允许中文",
  "evidence_bucket_id": "必须等于给定 bucket id",
  "evidence_moment_id": "可为空",
  "confidence": 0.0,
  "reason": "为什么这条证据足够支撑"
}}

最多返回 3 条。"""


def _strip_json_wrapper(raw: str) -> str:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return text


async def _call_profile_fact_proposal_model(
    *,
    bucket: dict,
    evidence_moment_id: str = "",
    max_proposals: int = 3,
) -> str:
    if not getattr(dehydrator, "api_available", False) or not getattr(dehydrator, "client", None):
        raise RuntimeError("dehydration API is not configured")
    meta = bucket.get("metadata", {}) if isinstance(bucket, dict) else {}
    identity = _identity()
    prompt = PROFILE_FACT_PROPOSAL_PROMPT_TEMPLATE.format(
        user_display_name=identity.get("user_display_name") or identity.get("user_name") or "用户",
        ai_name=identity.get("ai_name") or "AI",
    )
    content = strip_wikilinks(bucket.get("content", ""))
    if evidence_moment_id:
        try:
            moments = memory_moment_store.upsert_bucket(bucket)
            selected = next(
                (moment for moment in moments if str(moment.get("moment_id") or "") == evidence_moment_id),
                None,
            )
            if selected:
                content = selected.get("source_window") or selected.get("text") or content
        except Exception as e:
            logger.warning("Profile fact proposal moment lookup failed: %s", e)
    evidence_payload = {
        "bucket_id": bucket.get("id", ""),
        "bucket_name": meta.get("name", bucket.get("id", "")),
        "bucket_tags": meta.get("tags", []),
        "bucket_domain": meta.get("domain", []),
        "evidence_moment_id": evidence_moment_id,
        "content": content[:5000],
        "max_proposals": max(1, min(3, int(max_proposals))),
    }
    response = await dehydrator.client.chat.completions.create(
        model=dehydrator.model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": _json_lib.dumps(evidence_payload, ensure_ascii=False)},
        ],
        **dehydrator._completion_options(max_tokens=900, temperature=0.0),
    )
    if not response.choices:
        return "[]"
    return response.choices[0].message.content or "[]"


def _normalize_profile_fact_proposal(
    item: dict,
    *,
    evidence_bucket_id: str,
    evidence_moment_id: str = "",
    existing_keys: set[str] | None = None,
) -> tuple[dict | None, str]:
    if not isinstance(item, dict):
        return None, "proposal is not an object"
    fact = str(item.get("fact") or "").strip()
    if not fact:
        return None, "missing fact"
    candidate_evidence = str(item.get("evidence_bucket_id") or "").strip()
    if candidate_evidence != evidence_bucket_id:
        return None, "evidence_bucket_id mismatch"
    candidate_moment = str(item.get("evidence_moment_id") or evidence_moment_id or "").strip()
    if candidate_moment and not MEMORY_ID_RE.fullmatch(candidate_moment):
        return None, "invalid evidence_moment_id"
    key = _normalize_profile_fact_key(fact)
    if existing_keys and key in existing_keys:
        return None, "duplicate profile fact"
    proposal = {
        "fact": fact,
        "profile_kind": _profile_key(item.get("profile_kind"), "other"),
        "subject": _profile_key(item.get("subject"), "user"),
        "predicate": _profile_key(item.get("predicate"), "related_to"),
        "object": str(item.get("object") or "").strip()[:160],
        "evidence_bucket_id": evidence_bucket_id,
        "evidence_moment_id": candidate_moment,
        "confidence": _float_between(item.get("confidence"), 0.7, 0.0, 1.0),
        "reason": _clip_text(str(item.get("reason") or "").strip(), 240),
    }
    return proposal, ""


def _parse_profile_fact_proposals(
    raw: str,
    *,
    evidence_bucket_id: str,
    evidence_moment_id: str = "",
    existing_keys: set[str] | None = None,
    max_proposals: int = 3,
) -> tuple[list[dict], list[dict]]:
    rejected: list[dict] = []
    try:
        parsed = _json_lib.loads(_strip_json_wrapper(raw))
    except (TypeError, ValueError, _json_lib.JSONDecodeError):
        return [], [{"reason": "invalid json", "raw": _clip_text(str(raw or ""), 240)}]
    if isinstance(parsed, dict) and isinstance(parsed.get("proposals"), list):
        parsed = parsed["proposals"]
    if not isinstance(parsed, list):
        return [], [{"reason": "json root is not a list"}]

    proposals: list[dict] = []
    seen: set[str] = set()
    for item in parsed:
        proposal, reason = _normalize_profile_fact_proposal(
            item,
            evidence_bucket_id=evidence_bucket_id,
            evidence_moment_id=evidence_moment_id,
            existing_keys=existing_keys,
        )
        if not proposal:
            rejected.append({"reason": reason, "proposal": item if isinstance(item, dict) else str(item)})
            continue
        key = _normalize_profile_fact_key(proposal["fact"])
        if key in seen:
            rejected.append({"reason": "duplicate in response", "proposal": proposal})
            continue
        seen.add(key)
        proposals.append(proposal)
        if len(proposals) >= max(1, min(3, int(max_proposals))):
            break
    return proposals, rejected


ANCHOR_PROPOSAL_PROMPT_TEMPLATE = """你是一个长期锚点候选生成器。请判断给定记忆桶是否值得被人工标为 anchor。

身份：
- 当前用户：{user_display_name}
- 当前 AI：{ai_name}

边界：
1. 只能判断这个既有 bucket 是否适合作为长期锚点，不要提出新记忆，不要改写正文。
2. 不要建议 pinned、protected、Core Memory 或 profile_fact 更新。
3. anchor 应该是未来长期会反复帮助理解用户、关系、承诺、重要经历或长期项目的记忆。
4. 不要把今天很强烈但未被时间验证的短期情绪当 anchor。
5. 如果不适合，返回 []。
6. 只输出 JSON 数组，不要 markdown，不要解释。

候选格式：
{{
  "bucket_id": "必须等于给定 bucket id",
  "anchor_kind": "relationship|identity|commitment|life_event|project|preference|other",
  "reason": "为什么它适合成为长期锚点",
  "future_use": "以后什么场景需要它",
  "confidence": 0.0
}}

最多返回 1 条。"""


def _anchor_proposal_static_rejection(bucket: dict) -> str:
    meta = bucket.get("metadata", {}) if isinstance(bucket, dict) else {}
    if meta.get("anchor"):
        return "already anchor"
    if meta.get("pinned") or meta.get("protected"):
        return "pinned/protected buckets are not anchor proposal targets"
    if _is_profile_fact_bucket(bucket):
        return "profile_fact buckets are not anchor proposal targets"
    if str(meta.get("type") or "").strip() == "feel":
        return "feel buckets are not anchor proposal targets"
    return ""


async def _call_anchor_proposal_model(
    *,
    bucket: dict,
) -> str:
    if not getattr(dehydrator, "api_available", False) or not getattr(dehydrator, "client", None):
        raise RuntimeError("dehydration API is not configured")
    meta = bucket.get("metadata", {}) if isinstance(bucket, dict) else {}
    identity = _identity()
    prompt = ANCHOR_PROPOSAL_PROMPT_TEMPLATE.format(
        user_display_name=identity.get("user_display_name") or identity.get("user_name") or "用户",
        ai_name=identity.get("ai_name") or "AI",
    )
    evidence_payload = {
        "bucket_id": bucket.get("id", ""),
        "bucket_name": meta.get("name", bucket.get("id", "")),
        "bucket_type": meta.get("type", ""),
        "bucket_tags": meta.get("tags", []),
        "bucket_domain": meta.get("domain", []),
        "importance": meta.get("importance"),
        "created": meta.get("created", ""),
        "updated_at": meta.get("updated_at", ""),
        "last_active": meta.get("last_active", ""),
        "content": strip_wikilinks(bucket.get("content", ""))[:5000],
    }
    response = await dehydrator.client.chat.completions.create(
        model=dehydrator.model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": _json_lib.dumps(evidence_payload, ensure_ascii=False)},
        ],
        **dehydrator._completion_options(max_tokens=500, temperature=0.0),
    )
    if not response.choices:
        return "[]"
    return response.choices[0].message.content or "[]"


def _normalize_anchor_proposal(
    item: dict,
    *,
    bucket_id: str,
) -> tuple[dict | None, str]:
    if not isinstance(item, dict):
        return None, "proposal is not an object"
    candidate_bucket_id = str(item.get("bucket_id") or "").strip()
    if candidate_bucket_id != bucket_id:
        return None, "bucket_id mismatch"
    reason = _clip_text(str(item.get("reason") or "").strip(), 260)
    future_use = _clip_text(str(item.get("future_use") or "").strip(), 220)
    if not reason:
        return None, "missing reason"
    proposal = {
        "bucket_id": bucket_id,
        "anchor_kind": _profile_key(item.get("anchor_kind"), "other"),
        "reason": reason,
        "future_use": future_use,
        "confidence": _float_between(item.get("confidence"), 0.7, 0.0, 1.0),
    }
    return proposal, ""


def _parse_anchor_proposals(
    raw: str,
    *,
    bucket_id: str,
) -> tuple[list[dict], list[dict]]:
    try:
        parsed = _json_lib.loads(_strip_json_wrapper(raw))
    except (TypeError, ValueError, _json_lib.JSONDecodeError):
        return [], [{"reason": "invalid json", "raw": _clip_text(str(raw or ""), 240)}]
    if isinstance(parsed, dict) and isinstance(parsed.get("proposals"), list):
        parsed = parsed["proposals"]
    if not isinstance(parsed, list):
        return [], [{"reason": "json root is not a list"}]

    proposals: list[dict] = []
    rejected: list[dict] = []
    for item in parsed:
        proposal, reason = _normalize_anchor_proposal(item, bucket_id=bucket_id)
        if not proposal:
            rejected.append({"reason": reason, "proposal": item if isinstance(item, dict) else str(item)})
            continue
        if proposals:
            rejected.append({"reason": "too many proposals", "proposal": proposal})
            continue
        proposals.append(proposal)
    return proposals, rejected


def _queue_memory_enrichment(bucket_id: str, *, force: bool = False) -> None:
    if not bucket_id:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.create_task(_enrich_memory_async(bucket_id, force=force))


async def _enrich_memory_async(bucket_id: str, *, force: bool = False) -> None:
    try:
        bucket = await bucket_mgr.get(bucket_id)
        if is_self_anchor_bucket(bucket):
            logger.debug("Skip self-anchor enrichment / 跳过自我入口关系补全: %s", bucket_id)
            return
        result = await metadata_enricher.enrich_bucket(
            bucket_id,
            bucket_mgr,
            memory_edge_store,
            embedding_engine=embedding_engine,
            force=force,
        )
        logger.debug("Metadata proposal complete / 记忆元数据提案完成: %s", result)
    except Exception as e:
        logger.warning("Memory enrichment failed / 记忆关系补全失败: %s: %s", bucket_id, e)


def _queue_scene_linking(bucket_id: str) -> bool:
    """Schedule proposal-only linking without delaying the authored Scene write."""
    bucket_id = str(bucket_id or "").strip()
    if not bucket_id or not scene_linker.can_auto_link:
        return False
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False
    loop.create_task(_link_scene_async(bucket_id))
    return True


async def _link_scene_async(bucket_id: str) -> None:
    try:
        result = await scene_linker.link_scene(
            bucket_id,
            bucket_mgr,
            embedding_engine=embedding_engine,
        )
        logger.debug("Scene linker proposal complete / Scene 边提案完成: %s", result)
    except Exception as exc:
        logger.warning("Scene linker failed / Scene 边提案失败: %s: %s", bucket_id, exc)


async def _refresh_entity_edges_for_bucket_id(bucket_id: str) -> int:
    bucket_id = str(bucket_id or "").strip()
    if not bucket_id:
        return 0
    bucket = await bucket_mgr.get(bucket_id)
    return _refresh_entity_edges_for_bucket(bucket)


def _refresh_entity_edges_for_bucket(bucket: dict | None) -> int:
    if not bucket or is_self_anchor_bucket(bucket):
        return 0
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    if meta.get("type") == "feel" or meta.get("protected"):
        return 0
    bucket_id = str(bucket.get("id") or "")
    edges = extract_entity_edges_from_bucket(bucket, _identity())
    saved = entity_edge_store.replace_bucket_edges(bucket_id, edges)
    return len(saved)


def _queue_embedding_refresh(bucket_id: str) -> bool:
    if not bucket_id or not getattr(embedding_engine, "enabled", False):
        return False
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False
    loop.create_task(_refresh_bucket_embedding_async(bucket_id))
    return True


def _bucket_embedding_text_changed(before: dict | None, after: dict | None) -> bool:
    return bucket_text_for_embedding(before or {}) != bucket_text_for_embedding(after or {})


def _queue_embedding_refresh_if_changed(
    bucket_id: str,
    before: dict | None,
    after: dict | None,
) -> bool:
    if not _bucket_embedding_text_changed(before, after):
        return False
    return _queue_embedding_refresh(bucket_id)


async def _refresh_bucket_embedding_async(bucket_id: str) -> None:
    try:
        ok = await _refresh_bucket_embedding(bucket_id)
        if not ok:
            logger.debug("Embedding refresh skipped or failed / 向量刷新跳过或失败: %s", bucket_id)
    except Exception as e:
        logger.warning("Embedding refresh failed / 向量刷新失败: %s: %s", bucket_id, e)


# =============================================================
# /health endpoint: lightweight keepalive
# 轻量保活接口
# For Cloudflare Tunnel or reverse proxy to ping, preventing idle timeout
# 供 Cloudflare Tunnel 或反代定期 ping，防止空闲超时断连
# =============================================================
@mcp.custom_route("/", methods=["GET"])
async def root_redirect(request):
    from starlette.responses import RedirectResponse
    return RedirectResponse(url="/dashboard")


@mcp.custom_route("/auth/status", methods=["GET"])
async def auth_status(request):
    from starlette.responses import JSONResponse
    return JSONResponse(
        {
            "authenticated": _dashboard_authenticated(request),
            "setup_needed": _dashboard_setup_needed(),
            "identity": {
                "ai_name": _ai_author_name(),
                "user_name": _dashboard_author_name(),
            },
        }
    )


@mcp.custom_route("/auth/setup", methods=["POST"])
async def auth_setup(request):
    from starlette.responses import JSONResponse
    if not _dashboard_setup_needed():
        return JSONResponse({"error": "already configured"}, status_code=400)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json"}, status_code=400)
    password = str(body.get("password") or "").strip()
    if len(password) < 6:
        return JSONResponse({"error": "password must be at least 6 characters"}, status_code=400)
    _save_dashboard_password_hash(password)
    return _dashboard_login_response()


@mcp.custom_route("/auth/login", methods=["POST"])
async def auth_login(request):
    from starlette.responses import JSONResponse
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json"}, status_code=400)
    password = str(body.get("password") or "")
    if _verify_dashboard_password(password):
        return _dashboard_login_response()
    return JSONResponse({"error": "password rejected"}, status_code=401)


@mcp.custom_route("/auth/logout", methods=["POST"])
async def auth_logout(request):
    from starlette.responses import JSONResponse
    token = request.cookies.get("ombre_session")
    if token:
        _dashboard_sessions.pop(token, None)
    response = JSONResponse({"ok": True})
    response.delete_cookie("ombre_session")
    return response


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    from starlette.responses import JSONResponse
    try:
        stats = await bucket_mgr.get_stats()
        return JSONResponse({
            "status": "ok",
            "buckets": stats["permanent_count"] + stats["dynamic_count"],
            "decay_engine": "running" if decay_engine.is_running else "stopped",
            "memory_edges": len(memory_edge_store.list_edges()),
            "scene_edges": len(scene_linker.list_scene_edges()),
            "entity_edges": len(entity_edge_store.list_edges()),
            "reflection": {
                "enabled": reflection_engine.enabled,
                "auto_enabled": reflection_engine.auto_enabled,
                "model": reflection_engine.model,
                "api_ready": bool(reflection_engine.api_key),
            },
            "scene_linker": scene_linker.status(),
            "portrait": {
                "enabled": portrait_engine.enabled,
                "auto_enabled": portrait_engine.auto_enabled,
                "auto_initial_enabled": getattr(portrait_engine, "auto_initial_enabled", False),
                "model": portrait_engine.model,
                "api_ready": bool(portrait_engine.api_key),
                "state_path": portrait_engine.state_path,
            },
            "mcp_surface": mcp.surface_status(),
            "memory_object_writes": memory_object_write_status(config),
        })
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


# =============================================================
# /breath-hook endpoint: Dedicated hook for SessionStart
# 会话启动专用挂载点
# =============================================================
@mcp.custom_route("/breath-hook", methods=["GET"])
async def breath_hook(request):
    from starlette.responses import PlainTextResponse
    try:
        requested_mode = str(request.query_params.get("mode") or "").strip().lower()
        if requested_mode in {"", "handoff"}:
            max_tokens = _int_between(request.query_params.get("max_tokens"), 1200, 0, 1600)
            session_id = str(request.query_params.get("session_id") or "").strip()
            previous_session_id = str(
                request.query_params.get("previous_session_id") or ""
            ).strip()
            parent_shadow_id = str(
                request.query_params.get("parent_shadow_id") or ""
            ).strip()
            query = str(request.query_params.get("query") or "").strip()
            return PlainTextResponse(
                await _recall_memory(
                    mode="handoff",
                    query=query,
                    max_tokens=max_tokens,
                    session_id=session_id,
                    previous_session_id=previous_session_id,
                    parent_shadow_id=parent_shadow_id,
                    include_core=False,
                    include_related=False,
                )
            )
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        # pinned
        pinned = [
            b for b in all_buckets
            if not is_self_anchor_bucket(b)
            and (b["metadata"].get("pinned") or b["metadata"].get("protected"))
        ]
        # top 2 unresolved by score
        unresolved = [b for b in all_buckets
                      if not is_self_anchor_bucket(b)
                      and not b["metadata"].get("resolved", False)
                      and b["metadata"].get("type") not in ("permanent", "feel")
                      and not b["metadata"].get("anchor", False)
                      and not b["metadata"].get("pinned")
                      and not b["metadata"].get("protected")]
        scored = sorted(unresolved, key=lambda b: decay_engine.calculate_score(b["metadata"]), reverse=True)
        anchors = _select_anchor_buckets(all_buckets, limit=2)

        parts = []
        token_budget = 10000
        for b in pinned:
            summary = await dehydrator.dehydrate(_bucket_text_for_embedding(b), {k: v for k, v in b["metadata"].items() if k != "tags"})
            parts.append(f"📌 [核心准则] {summary}")
            token_budget -= count_tokens_approx(summary)

        for b in anchors:
            if token_budget <= 0:
                break
            summary = await dehydrator.dehydrate(_bucket_text_for_embedding(b), {k: v for k, v in b["metadata"].items() if k != "tags"})
            entry = f"⚓ [长期锚点] [bucket_id:{b['id']}] {summary}"
            entry_tokens = count_tokens_approx(entry)
            if entry_tokens > token_budget:
                break
            parts.append(entry)
            token_budget -= entry_tokens

        # Diversity: top-1 fixed + shuffle rest from top-20
        candidates = list(scored)
        if len(candidates) > 1:
            top1 = [candidates[0]]
            pool = candidates[1:min(20, len(candidates))]
            random.shuffle(pool)
            candidates = top1 + pool + candidates[min(20, len(candidates)):]
        # Hard cap: max 20 surfacing buckets in hook
        candidates = candidates[:20]

        for b in candidates:
            if token_budget <= 0:
                break
            summary = await dehydrator.dehydrate(_bucket_text_for_embedding(b), {k: v for k, v in b["metadata"].items() if k != "tags"})
            summary_tokens = count_tokens_approx(summary)
            if summary_tokens > token_budget:
                break
            parts.append(summary)
            token_budget -= summary_tokens

        if not parts:
            return PlainTextResponse("")
        return PlainTextResponse("[Ombre Brain - 记忆浮现]\n" + "\n---\n".join(parts))
    except Exception as e:
        logger.warning(f"Breath hook failed: {e}")
        return PlainTextResponse("")


# =============================================================
# /introspection-hook endpoint: Dedicated hook for waking self-reflection
# 清醒自省专用挂载点。/dream-hook 暂时保留兼容旧接入。
# =============================================================
@mcp.custom_route("/introspection-hook", methods=["GET"])
@mcp.custom_route("/dream-hook", methods=["GET"])
async def dream_hook(request):
    from starlette.responses import PlainTextResponse
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        candidates = [
            b for b in all_buckets
            if b["metadata"].get("type") not in ("permanent", "feel")
            and not b["metadata"].get("pinned", False)
            and not b["metadata"].get("protected", False)
        ]
        candidates.sort(key=lambda b: b["metadata"].get("created", ""), reverse=True)
        recent = candidates[:10]

        if not recent:
            return PlainTextResponse("")

        parts = []
        for b in recent:
            meta = b["metadata"]
            resolved_tag = "[已解决]" if meta.get("resolved", False) else "[未解决]"
            parts.append(
                f"{meta.get('name', b['id'])} {resolved_tag} "
                f"V{meta.get('valence', 0.5):.1f}/A{meta.get('arousal', 0.3):.1f}\n"
                f"{_bucket_text_for_embedding(b)[:200]}"
            )

        return PlainTextResponse("[Ombre Brain - Introspection]\n" + "\n---\n".join(parts))
    except Exception as e:
        logger.warning(f"Introspection hook failed: {e}")
        return PlainTextResponse("")


# =============================================================
# Internal helper: merge-or-create
# 内部辅助：检查是否可合并，可以则合并，否则新建
# Shared by hold and grow to avoid duplicate logic
# hold 和 grow 共用，避免重复逻辑
# =============================================================
def _bucket_days_since_last_active(meta: dict) -> float:
    parsed = bucket_mgr._parse_iso_datetime(meta.get("last_active") or meta.get("created"))
    if parsed is None:
        return 9999.0
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return max(0.0, (now - parsed).total_seconds() / 86400)


def _format_readonly_related_memory(bucket: dict) -> str:
    meta = bucket.get("metadata", {})
    labels = []
    if meta.get("type") == "archived":
        labels.append("归档")
    if meta.get("resolved"):
        labels.append("已解决")
    if meta.get("digested"):
        labels.append("已消化")
    state = f" ({', '.join(labels)})" if labels else ""
    title = str(meta.get("name") or "").strip() or _bucket_context_snippet(bucket, max_chars=36) or bucket["id"]
    return (
        "\n旧记忆提示(只读): "
        f"可能相关「{title}」[bucket_id:{bucket['id']}]{state}"
    )


def _bucket_text_for_embedding(bucket: dict) -> str:
    meta = bucket.get("metadata", {})
    comments = meta.get("comments", [])
    comment_text = ""
    if isinstance(comments, list):
        comment_text = "\n".join(
            strip_wikilinks(str(comment.get("content", "")))
            for comment in comments
            if isinstance(comment, dict)
        )
    return f"{bucket_content_for_recall(bucket)}\n{comment_text}".strip()


def _bucket_context_snippet(bucket: dict, max_chars: int = 180) -> str:
    text = " ".join(bucket_content_for_recall(bucket).split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _compact_diffused_summary(bucket: dict, dehydrated: str, max_chars: int = 180) -> str:
    raw = str(dehydrated or "").strip()
    extracted = _summary_from_jsonish_text(raw)
    if extracted:
        return _clip_text(extracted, max_chars)

    meta = bucket.get("metadata", {}) or {}
    title = str(meta.get("name") or bucket.get("id") or "记忆").strip()
    return _clip_text(title, max_chars)


def _bucket_diffusion_path_summary(path, bucket_map: dict[str, dict], max_chars: int = 180) -> str:
    if not path:
        return ""
    return _clip_text(format_diffusion_path(path, bucket_map), max_chars)


def _breath_related_diffusion_options(top_k: int):
    options = diffusion_options_from_config(config)
    return replace(
        options,
        max_hops=1,
        top_k=max(0, int(top_k or 0)),
    )


def _summary_from_jsonish_text(text: str) -> str:
    if not text:
        return ""
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        try:
            data = _json_lib.loads(candidate)
        except Exception:
            continue
        if isinstance(data, dict):
            for key in ("summary", "memory_summary", "gist"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return " ".join(strip_wikilinks(value).split())
            core_facts = data.get("core_facts")
            if isinstance(core_facts, list) and core_facts:
                facts = [
                    " ".join(strip_wikilinks(str(item)).split())
                    for item in core_facts[:2]
                    if str(item).strip()
                ]
                if facts:
                    return "；".join(facts)
    return ""


def _clip_text(text: str, max_chars: int) -> str:
    compact = " ".join(strip_wikilinks(str(text or "")).split())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rstrip() + "..."


def _handoff_timezone():
    for section in ("portrait", "reflection"):
        value = config.get(section, {}) if isinstance(config.get(section, {}), dict) else {}
        name = str(value.get("timezone") or "").strip()
        if name:
            try:
                return ZoneInfo(name)
            except Exception:
                pass
    return ZoneInfo("Asia/Shanghai")


def _handoff_today_key() -> str:
    return datetime.now(_handoff_timezone()).date().isoformat()


def _handoff_recent_date_keys() -> set[str]:
    today = datetime.fromisoformat(_handoff_today_key()).date()
    return {today.isoformat(), (today - timedelta(days=1)).isoformat()}


def _bucket_handoff_date(bucket: dict) -> str:
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    explicit = str(meta.get("date") or "").strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", explicit):
        return explicit
    for key in ("created", "updated_at", "last_active"):
        raw = str(meta.get(key) or "").strip()
        if not raw:
            continue
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=_handoff_timezone())
        return parsed.astimezone(_handoff_timezone()).date().isoformat()
    return ""


def _handoff_clean_summary_text(content: str, *, include_detail_sections: bool = False) -> str:
    text = strip_display_temperature_sections(strip_temperature_meaning_lines(str(content or "")))
    text = strip_followup_sections(text)
    if not include_detail_sections:
        text = re.split(
            r"\n\s*###\s+(?:moment|affect_anchor|reflection|assistant_reflection)\b",
            text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
    text = re.sub(r"^#+\s*\S+\s*", "", text.strip())
    return " ".join(strip_wikilinks(text).split())


def _handoff_short_summary(content: str, *, max_chars: int = 72) -> str:
    text = _handoff_clean_summary_text(content)
    if not text:
        text = _handoff_clean_summary_text(content, include_detail_sections=True)
    match = re.search(r"[。！？!?；;]", text)
    if match and match.end() >= 18:
        text = text[:match.end()]
    return _clip_text(text, max_chars)


def _handoff_query_hint(query: str) -> str:
    query = str(query or "").strip()
    if not query:
        return ""
    escaped = query.replace("\\", "\\\\").replace('"', '\\"')
    return f"细节用 breath(query=\"{escaped}\") 查。"


async def _refresh_bucket_embedding(bucket_id: str) -> bool:
    if not getattr(embedding_engine, "enabled", False):
        return False
    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return False
    return await embedding_engine.generate_and_store(bucket_id, bucket_text_for_embedding(bucket))


def _delete_bucket_indexes(bucket_id: str) -> tuple[dict, list[str]]:
    cleanup: dict = {}
    errors: list[str] = []

    try:
        embedding_engine.delete_embedding(bucket_id)
        cleanup["embedding"] = True
    except Exception as e:
        logger.warning("Failed to delete embedding for bucket / 删除桶向量失败: %s: %s", bucket_id, e)
        errors.append("embedding")

    try:
        cleanup["moments"] = memory_moment_store.delete_bucket(bucket_id)
    except Exception as e:
        logger.warning("Failed to delete moment index for bucket / 删除桶 moment 索引失败: %s: %s", bucket_id, e)
        errors.append("moments")

    try:
        cleanup["shadow_memory_card"] = shadow_memory_card_store.delete(bucket_id)
    except Exception as e:
        logger.warning("Failed to delete shadow memory card / 删除影子记忆卡失败: %s: %s", bucket_id, e)
        errors.append("shadow_memory_card")

    try:
        cleanup["edges"] = memory_edge_store.delete_for_bucket(bucket_id)
    except Exception as e:
        logger.warning("Failed to delete memory edges for bucket / 删除桶关系边失败: %s: %s", bucket_id, e)
        errors.append("edges")

    try:
        cleanup["scene_edges"] = scene_linker.deactivate_scene_edges(bucket_id)
    except Exception as e:
        logger.warning("Failed to deactivate Scene edges / 停用 Scene 边失败: %s: %s", bucket_id, e)
        errors.append("scene_edges")

    try:
        cleanup["entity_edges"] = entity_edge_store.delete_for_bucket(bucket_id)
    except Exception as e:
        logger.warning("Failed to delete entity edges for bucket / 删除桶人物边失败: %s: %s", bucket_id, e)
        errors.append("entity_edges")

    try:
        cleanup["node"] = memory_node_store.delete(bucket_id)
    except Exception as e:
        logger.warning("Failed to delete memory node for bucket / 删除桶 node 索引失败: %s: %s", bucket_id, e)
        errors.append("node")

    return cleanup, errors


async def _delete_bucket_and_indexes(bucket_id: str) -> dict:
    bucket_id = str(bucket_id or "").strip()
    if not bucket_id or not MEMORY_ID_RE.fullmatch(bucket_id):
        return {"id": bucket_id, "status": "invalid", "reason": "invalid_bucket_id"}

    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return {"id": bucket_id, "status": "not_found", "reason": "not_found"}

    success = await bucket_mgr.delete(bucket_id)
    if not success:
        return {"id": bucket_id, "status": "failed", "reason": "delete_failed"}

    cleanup, errors = _delete_bucket_indexes(bucket_id)
    return {
        "id": bucket_id,
        "status": "deleted",
        "cleanup": cleanup,
        "cleanup_errors": errors,
    }


def _write_semantic_search_timeout_seconds() -> float:
    write_cfg = config.get("write_path", {}) if isinstance(config.get("write_path", {}), dict) else {}
    try:
        return max(0.0, float(write_cfg.get("semantic_search_timeout_seconds", 3)))
    except (TypeError, ValueError):
        return 3.0


async def _find_readonly_related_bucket(
    content: str,
    *,
    exclude_ids: set[str] | None = None,
) -> dict | None:
    exclude_ids = exclude_ids or set()
    candidates: dict[str, dict] = {}

    try:
        for bucket in await bucket_mgr.search(content, limit=8, include_archive=True):
            candidates[bucket["id"]] = {**bucket, "_related_score": float(bucket.get("score", 0.0))}
    except Exception as e:
        logger.warning(f"Related old memory keyword search failed / 相关旧记忆关键词搜索失败: {e}")

    if getattr(embedding_engine, "enabled", False):
        try:
            semantic_lookup = embedding_engine.search_similar(content, top_k=8)
            timeout_seconds = _write_semantic_search_timeout_seconds()
            if timeout_seconds > 0:
                similar = await asyncio.wait_for(semantic_lookup, timeout=timeout_seconds)
            else:
                similar = await semantic_lookup
            for bucket_id, similarity in similar:
                if bucket_id in candidates:
                    candidates[bucket_id]["_related_score"] = max(
                        candidates[bucket_id].get("_related_score", 0.0),
                        float(similarity) * 100.0,
                    )
                    continue
                bucket = await bucket_mgr.get(bucket_id)
                if bucket:
                    candidates[bucket_id] = {**bucket, "_related_score": float(similarity) * 100.0}
        except asyncio.TimeoutError:
            logger.warning(
                "Related old memory semantic search timed out after %.1fs / 写入时相关旧记忆语义搜索超时",
                _write_semantic_search_timeout_seconds(),
            )
        except Exception as e:
            logger.warning(f"Related old memory semantic search failed / 相关旧记忆语义搜索失败: {e}")

    ranked = []
    for bucket in candidates.values():
        meta = bucket.get("metadata", {})
        if bucket.get("id") in exclude_ids:
            continue
        if meta.get("type") == "feel":
            continue
        ranked.append(bucket)

    ranked.sort(
        key=lambda item: (
            item.get("_related_score", 0.0),
            _bucket_days_since_last_active(item.get("metadata", {})),
        ),
        reverse=True,
    )
    return ranked[0] if ranked else None


def _bucket_needs_memory_enrichment(bucket: dict) -> bool:
    meta = bucket.get("metadata", {}) if isinstance(bucket, dict) else {}
    if is_self_anchor_bucket(bucket):
        return False
    if meta.get("type") == "feel" or meta.get("protected"):
        return False
    if meta.get("memory_value_source") == "authored_scene":
        return False
    try:
        confidence = float(meta.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return confidence <= 0.0


def _bucket_allows_memory_edge_backfill(bucket: dict) -> bool:
    meta = bucket.get("metadata", {}) if isinstance(bucket, dict) else {}
    return bool(bucket and not is_self_anchor_bucket(bucket) and meta.get("type") != "feel" and not meta.get("protected"))


async def _backfill_memory_enrichment(
    limit: int | None = None,
    *,
    bucket_mgr_arg=None,
    reflection_engine_arg=None,
    edge_store_arg=None,
    embedding_engine_arg=None,
) -> dict:
    mgr = bucket_mgr_arg or bucket_mgr
    engine = reflection_engine_arg or metadata_enricher
    edge_store = edge_store_arg or memory_edge_store
    emb_engine = embedding_engine_arg or embedding_engine
    reflection_cfg = config.get("reflection", {}) if isinstance(config.get("reflection", {}), dict) else {}
    default_limit = _int_between(reflection_cfg.get("enrich_backfill_limit"), 5, 0, 50)
    limit = _int_between(limit, default_limit, 0, 50)
    if limit <= 0:
        return {"processed": 0, "ids": [], "errors": []}

    try:
        all_buckets = await mgr.list_all(include_archive=False)
    except Exception as e:
        logger.warning("Memory enrichment backfill list failed / enrich 补跑列桶失败: %s", e)
        return {"processed": 0, "ids": [], "errors": [str(e)]}

    candidates = [bucket for bucket in all_buckets if _bucket_needs_memory_enrichment(bucket)]
    candidates.sort(
        key=lambda item: item.get("metadata", {}).get("updated_at") or item.get("metadata", {}).get("created", ""),
        reverse=True,
    )

    processed: list[str] = []
    errors: list[str] = []
    for bucket in candidates[:limit]:
        bucket_id = bucket.get("id")
        if not bucket_id:
            continue
        try:
            await engine.enrich_bucket(
                bucket_id,
                mgr,
                edge_store,
                embedding_engine=emb_engine,
                force=True,
            )
            processed.append(bucket_id)
        except Exception as e:
            logger.warning("Memory enrichment backfill failed / enrich 补跑失败: %s: %s", bucket_id, e)
            errors.append(f"{bucket_id}: {e}")
    return {"processed": len(processed), "ids": processed, "errors": errors}


async def enrich_backfill(limit: int = 10) -> dict:
    """后台生成待审核的 tags/cues/importance/edge sidecar 提案；不会修改记忆或召回权重。"""
    return await _backfill_memory_enrichment(limit=limit)


async def _shadow_memory_card_candidates(
    *,
    limit: int,
    bucket_id: str = "",
    query: str = "",
    include_archive: bool = False,
    bucket_mgr_arg=None,
) -> list[dict]:
    mgr = bucket_mgr_arg or bucket_mgr
    bucket_id = str(bucket_id or "").strip()
    query = str(query or "").strip()
    if bucket_id:
        bucket = await mgr.get(bucket_id)
        return [bucket] if _shadow_memory_card_eligible(bucket) else []
    if query:
        try:
            buckets = await mgr.search(
                query,
                limit=max(limit, 20),
                include_archive=include_archive,
            )
        except TypeError:
            buckets = await mgr.search(query, limit=max(limit, 20))
    else:
        buckets = await mgr.list_all(include_archive=include_archive)
        buckets.sort(
            key=lambda item: str(
                item.get("metadata", {}).get("updated_at")
                or item.get("metadata", {}).get("created")
                or ""
            ),
            reverse=True,
        )
    return [bucket for bucket in buckets if _shadow_memory_card_eligible(bucket)][:limit]


async def memory_card_shadow_backfill(
    limit: int = 10,
    bucket_id: str = "",
    query: str = "",
    dry_run: bool = True,
    force: bool = False,
    include_archive: bool = False,
) -> dict:
    """生成 primary_abstraction/cue_anchors/evidence_refs 影子卡；默认只预览，不改 bucket，也不参与召回。"""
    limit = _int_between(limit, 10, 1, 25)
    bucket_id = str(bucket_id or "").strip()
    if bucket_id and not MEMORY_ID_RE.fullmatch(bucket_id):
        return {"status": "invalid", "reason": "invalid_bucket_id", "bucket_id": bucket_id}
    candidates = await _shadow_memory_card_candidates(
        limit=limit,
        bucket_id=bucket_id,
        query=query,
        include_archive=bool(include_archive),
    )
    results = []
    written = 0
    cached = 0
    errors = []
    for bucket in candidates:
        current_id = str(bucket.get("id") or "").strip()
        source_hash = _shadow_memory_card_source_hash(bucket)
        existing = shadow_memory_card_store.get(current_id)
        if (
            existing
            and not force
            and existing.get("source_hash") == source_hash
            and existing.get("generation_version") == MEMORY_CARD_SHADOW_VERSION
        ):
            item = dict(existing)
            item["action"] = "cached"
            results.append(item)
            cached += 1
            continue
        try:
            card = await _build_shadow_memory_card(bucket)
            card["action"] = "preview" if dry_run else "written"
            if not dry_run:
                stored = shadow_memory_card_store.upsert(card)
                stored["action"] = "written"
                card = stored
                written += 1
            results.append(card)
        except Exception as e:
            logger.warning("Shadow memory card backfill failed / 影子记忆卡补建失败: %s: %s", current_id, e)
            errors.append(f"{current_id}: {e}")
    return {
        "status": "ok",
        "mode": "shadow_only",
        "generation_version": MEMORY_CARD_SHADOW_VERSION,
        "dry_run": bool(dry_run),
        "processed": len(results),
        "written": written,
        "cached": cached,
        "errors": errors,
        "cards": results,
        "injection_enabled": False,
        "bucket_mutation": False,
    }


async def memory_card_shadow_read(bucket_id: str = "", limit: int = 20) -> dict:
    """读取已保存的影子记忆卡；这些卡不参与候选、gate、扩散或注入。"""
    bucket_id = str(bucket_id or "").strip()
    if bucket_id:
        if not MEMORY_ID_RE.fullmatch(bucket_id):
            return {"status": "invalid", "reason": "invalid_bucket_id", "bucket_id": bucket_id}
        card = shadow_memory_card_store.get(bucket_id)
        return {
            "status": "ok" if card else "not_found",
            "mode": "shadow_only",
            "card": card,
            "injection_enabled": False,
        }
    return {
        "status": "ok",
        "mode": "shadow_only",
        "stats": shadow_memory_card_store.stats(),
        "cards": shadow_memory_card_store.list(limit=_int_between(limit, 20, 1, 200)),
        "injection_enabled": False,
    }


async def _search_edge_backfill_buckets(mgr, query: str, limit: int) -> list[dict]:
    try:
        return await mgr.search(query, limit=max(limit, 20), include_archive=False)
    except TypeError:
        return await mgr.search(query, limit=max(limit, 20))


async def _edge_backfill_candidates(
    mgr,
    *,
    limit: int,
    bucket_id: str = "",
    query: str = "",
) -> tuple[list[dict], list[str]]:
    warnings: list[str] = []
    bucket_id = str(bucket_id or "").strip()
    query = str(query or "").strip()
    if bucket_id:
        bucket = await mgr.get(bucket_id)
        if not bucket:
            return [], [f"missing_bucket: {bucket_id}"]
        return ([bucket] if _bucket_allows_memory_edge_backfill(bucket) else []), []

    if query:
        try:
            buckets = await _search_edge_backfill_buckets(mgr, query, limit)
        except Exception as e:
            return [], [f"search_failed: {e}"]
    else:
        try:
            buckets = await mgr.list_all(include_archive=False)
        except Exception as e:
            return [], [f"list_failed: {e}"]
        buckets.sort(
            key=lambda item: item.get("metadata", {}).get("updated_at") or item.get("metadata", {}).get("created", ""),
            reverse=True,
        )

    selected = []
    seen = set()
    for bucket in buckets:
        current_id = str(bucket.get("id") or "")
        if not current_id or current_id in seen:
            continue
        if not _bucket_allows_memory_edge_backfill(bucket):
            continue
        selected.append(bucket)
        seen.add(current_id)
        if len(selected) >= limit:
            break
    return selected, warnings


async def _backfill_memory_edges(
    limit: int | None = None,
    *,
    bucket_id: str = "",
    query: str = "",
    dry_run: bool = False,
    bucket_mgr_arg=None,
    reflection_engine_arg=None,
    edge_store_arg=None,
    embedding_engine_arg=None,
) -> dict:
    mgr = bucket_mgr_arg or bucket_mgr
    engine = reflection_engine_arg or reflection_engine
    edge_store = edge_store_arg or memory_edge_store
    emb_engine = embedding_engine_arg or embedding_engine
    reflection_cfg = config.get("reflection", {}) if isinstance(config.get("reflection", {}), dict) else {}
    default_limit = _int_between(reflection_cfg.get("edge_backfill_limit"), 5, 0, 50)
    limit = 1 if str(bucket_id or "").strip() else _int_between(limit, default_limit, 0, 50)
    if limit <= 0:
        return {"processed": 0, "ids": [], "edges": 0, "proposed_edges": 0, "errors": [], "dry_run": bool(dry_run)}

    candidates, warnings = await _edge_backfill_candidates(
        mgr,
        limit=limit,
        bucket_id=bucket_id,
        query=query,
    )
    processed: list[str] = []
    results: list[dict] = []
    errors: list[str] = list(warnings)
    edge_count = 0
    proposed_count = 0
    for bucket in candidates:
        current_id = bucket.get("id")
        if not current_id:
            continue
        try:
            result = await engine.backfill_edges_for_bucket(
                current_id,
                mgr,
                edge_store,
                embedding_engine=emb_engine,
                dry_run=dry_run,
            )
            result = dict(result or {})
            processed.append(current_id)
            edge_count += int(result.get("edges", 0) or 0)
            proposed_count += int(result.get("proposed_edges", 0) or 0)
            results.append(result)
        except Exception as e:
            logger.warning("Memory edge backfill failed / 关系边补跑失败: %s: %s", current_id, e)
            errors.append(f"{current_id}: {e}")
    return {
        "processed": len(processed),
        "ids": processed,
        "edges": edge_count,
        "proposed_edges": proposed_count,
        "results": results,
        "errors": errors,
        "dry_run": bool(dry_run),
    }


async def _entity_edge_backfill_candidates(
    mgr,
    *,
    limit: int,
    bucket_id: str = "",
    query: str = "",
    include_archive: bool = False,
) -> tuple[list[dict], list[str]]:
    warnings: list[str] = []
    bucket_id = str(bucket_id or "").strip()
    query = str(query or "").strip()
    if bucket_id:
        bucket = await mgr.get(bucket_id)
        if not bucket:
            return [], [f"missing_bucket: {bucket_id}"]
        return ([bucket] if _bucket_allows_memory_edge_backfill(bucket) else []), []

    if query:
        try:
            try:
                buckets = await mgr.search(query, limit=max(limit, 20), include_archive=include_archive)
            except TypeError:
                buckets = await mgr.search(query, limit=max(limit, 20))
        except Exception as e:
            return [], [f"search_failed: {e}"]
    else:
        try:
            buckets = await mgr.list_all(include_archive=include_archive)
        except Exception as e:
            return [], [f"list_failed: {e}"]
        buckets.sort(
            key=lambda item: item.get("metadata", {}).get("updated_at") or item.get("metadata", {}).get("created", ""),
            reverse=True,
        )

    selected: list[dict] = []
    seen: set[str] = set()
    for bucket in buckets:
        current_id = str(bucket.get("id") or "")
        if not current_id or current_id in seen:
            continue
        if not _bucket_allows_memory_edge_backfill(bucket):
            continue
        selected.append(bucket)
        seen.add(current_id)
        if len(selected) >= limit:
            break
    return selected, warnings


async def _backfill_entity_edges(
    limit: int | None = None,
    *,
    bucket_id: str = "",
    query: str = "",
    dry_run: bool = True,
    include_archive: bool = False,
    bucket_mgr_arg=None,
    entity_edge_store_arg=None,
) -> dict:
    mgr = bucket_mgr_arg or bucket_mgr
    store = entity_edge_store_arg or entity_edge_store
    reflection_cfg = config.get("reflection", {}) if isinstance(config.get("reflection", {}), dict) else {}
    default_limit = _int_between(reflection_cfg.get("entity_edge_backfill_limit"), 25, 0, 500)
    limit = 1 if str(bucket_id or "").strip() else _int_between(limit, default_limit, 0, 500)
    if limit <= 0:
        return {
            "processed": 0,
            "ids": [],
            "edges": 0,
            "proposed_edges": 0,
            "results": [],
            "errors": [],
            "dry_run": bool(dry_run),
            "include_archive": bool(include_archive),
        }

    candidates, warnings = await _entity_edge_backfill_candidates(
        mgr,
        limit=limit,
        bucket_id=bucket_id,
        query=query,
        include_archive=include_archive,
    )
    processed: list[str] = []
    results: list[dict] = []
    errors: list[str] = list(warnings)
    edge_count = 0
    proposed_count = 0
    identity = _identity()
    for bucket in candidates:
        current_id = str(bucket.get("id") or "")
        if not current_id:
            continue
        try:
            proposed = extract_entity_edges_from_bucket(bucket, identity)
            saved = [] if dry_run else store.replace_bucket_edges(current_id, proposed)
            processed.append(current_id)
            edge_count += len(saved)
            proposed_count += len(proposed)
            meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
            results.append(
                {
                    "id": current_id,
                    "name": meta.get("name") or bucket.get("name") or current_id,
                    "edges": len(saved),
                    "proposed_edges": len(proposed),
                    "dry_run": bool(dry_run),
                    "edge_previews": [
                        {
                            "subject": edge.get("subject"),
                            "relation": edge.get("relation"),
                            "object_text": edge.get("object_text"),
                            "confidence": edge.get("confidence"),
                        }
                        for edge in proposed[:5]
                    ],
                }
            )
        except Exception as e:
            logger.warning("Entity edge backfill failed / 人物边补跑失败: %s: %s", current_id, e)
            errors.append(f"{current_id}: {e}")
    return {
        "processed": len(processed),
        "ids": processed,
        "edges": edge_count,
        "proposed_edges": proposed_count,
        "results": results,
        "errors": errors,
        "dry_run": bool(dry_run),
        "include_archive": bool(include_archive),
    }


async def edge_backfill(
    limit: int = 10,
    bucket_id: str = "",
    query: str = "",
    dry_run: bool = False,
) -> dict:
    """只补 memory_edges 关系边，不改 bucket 正文、tags、importance、confidence。可用 bucket_id 或 query 定向。"""
    return await _backfill_memory_edges(
        limit=limit,
        bucket_id=bucket_id,
        query=query,
        dry_run=dry_run,
    )


async def entity_edge_backfill(
    limit: int = 25,
    bucket_id: str = "",
    query: str = "",
    dry_run: bool = True,
    include_archive: bool = False,
) -> dict:
    """只补 entity_edges.jsonl，不改 bucket 正文、memory_edges、tags、importance。默认 dry-run。"""
    return await _backfill_entity_edges(
        limit=limit,
        bucket_id=bucket_id,
        query=query,
        dry_run=dry_run,
        include_archive=include_archive,
    )


async def _ensure_decay_engine_started_for_transport(transport_name: str) -> None:
    if transport_name not in ("sse", "streamable-http"):
        return
    try:
        await decay_engine.ensure_started()
    except Exception as e:
        logger.warning("Decay engine startup failed / 衰减引擎启动失败: %s", e)


async def _merge_or_create(
    content: str,
    tags: list,
    importance: int,
    domain: list,
    valence: float,
    arousal: float,
    name: str = "",
    *,
    allow_merge: bool = True,
    memory_subject: str = "",
    memory_layer: str = "",
    memory_classification_source: str = "",
    date: str = "",
    source: str = "",
    bucket_id: str = "",
    extra_metadata: dict | None = None,
    normalize_content: bool = True,
) -> tuple[str, str, bool, dict | None]:
    """
    Check if a similar bucket exists for merging; merge if so, create if not.
    Returns (bucket_id, display_name, is_merged).
    检查是否有相似桶可合并，有则合并，无则新建。
    返回 (桶ID, 显示名称, 是否合并)。
    """
    if normalize_content:
        content = _normalize_memory_sections_for_write(content)
    try:
        existing = await bucket_mgr.search(
            content,
            limit=1,
            domain_filter=domain or None,
            include_archive=False,
        )
    except Exception as e:
        logger.warning(f"Search for merge failed, creating new / 合并搜索失败，新建: {e}")
        existing = []

    related_bucket = await _find_readonly_related_bucket(content)

    if allow_merge and existing and existing[0].get("score", 0) > config.get("merge_threshold", 90):
        bucket = existing[0]
        # --- Never merge into pinned/protected buckets ---
        # --- 不合并到钉选/保护桶 ---
        if not (
            bucket["metadata"].get("pinned")
            or bucket["metadata"].get("protected")
            or bucket["metadata"].get("type") == "feel"
            or _is_profile_fact_bucket(bucket)
        ):
            try:
                merged = await dehydrator.merge(bucket["content"], content)
                merged = _normalize_memory_sections_for_write(merged)
                old_v = bucket["metadata"].get("valence", 0.5)
                old_a = bucket["metadata"].get("arousal", 0.3)
                merged_valence = round((old_v + valence) / 2, 2)
                merged_arousal = round((old_a + arousal) / 2, 2)
                await bucket_mgr.update(
                    bucket["id"],
                    content=merged,
                    tags=list(set(bucket["metadata"].get("tags", []) + tags)),
                    importance=max(bucket["metadata"].get("importance", 5), importance),
                    domain=list(set(bucket["metadata"].get("domain", []) + domain)),
                    valence=merged_valence,
                    arousal=merged_arousal,
                )
                _queue_embedding_refresh(bucket["id"])
                return bucket["id"], bucket["metadata"].get("name", bucket["id"]), True, related_bucket
            except Exception as e:
                logger.warning(f"Merge failed, creating new / 合并失败，新建: {e}")

    bucket_id = await bucket_mgr.create(
        content=content,
        tags=tags,
        importance=importance,
        domain=domain,
        valence=valence,
        arousal=arousal,
        name=name or None,
        bucket_id=bucket_id or None,
        date=date or None,
        source=source or None,
        extra_metadata={
            **_memory_classification_metadata(
                memory_subject,
                memory_layer,
                memory_classification_source,
            ),
            **(extra_metadata or {}),
        },
    )
    _queue_embedding_refresh(bucket_id)
    return bucket_id, name or bucket_id, False, related_bucket


def _memory_classification_metadata(
    memory_subject: str,
    memory_layer: str,
    memory_classification_source: str = "",
) -> dict:
    if not memory_subject or not memory_layer:
        return {}
    payload = {
        "memory_subject": str(memory_subject),
        "memory_layer": str(memory_layer),
    }
    if memory_classification_source:
        payload["memory_classification_source"] = str(memory_classification_source)
    return payload


async def _build_mcp_diffused_memory_block(
    source_buckets: list[dict],
    all_buckets: list[dict] | None,
    token_budget: int,
    limit_per_source: int,
    min_confidence: float,
    query_text: str = "",
    exclude_bucket_ids: set[str] | None = None,
) -> str:
    source_buckets = [bucket for bucket in source_buckets if not is_self_anchor_bucket(bucket)]
    if token_budget <= 0 or not source_buckets:
        return ""

    limit_per_source = _int_between(limit_per_source, 1, 0, 5)
    min_confidence = _float_between(min_confidence, 0.55, 0.0, 1.0)
    if limit_per_source <= 0:
        return ""

    source_ids = [bucket["id"] for bucket in source_buckets if bucket.get("id")]
    source_set = set(source_ids)
    exclude_set = source_set | set(exclude_bucket_ids or set())
    if not source_ids:
        return ""

    if all_buckets is None:
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
        except Exception as e:
            logger.warning(f"Failed to list buckets for diffused memory / 联想浮现列桶失败: {e}")
            all_buckets = []

    bucket_map = {
        bucket["id"]: bucket
        for bucket in all_buckets
        if bucket.get("id") and not is_self_anchor_bucket(bucket)
    }
    node_salience = None
    node_resonance = None
    if _node_facets_enabled(config):
        try:
            memory_node_store.bulk_upsert(list(bucket_map.values()))
            query_facets = memory_node_store.facets_for_text(query_text)
            node_salience = _node_salience_lookup
            node_resonance = _node_resonance_lookup(query_facets)
        except Exception as e:
            logger.warning(f"Failed to refresh memory nodes / 记忆节点刷新失败: {e}")

    edges = [
        edge
        for edge in _bucket_edges_for_recall(bucket_map)
        if float(edge.get("confidence", 0.0)) >= min_confidence
    ]
    diffusion_options = _breath_related_diffusion_options(len(source_ids) * limit_per_source)
    hits = diffuse_memory(
        seed_scores_for_buckets(source_buckets),
        edges,
        bucket_map,
        options=diffusion_options,
        exclude_ids=exclude_set,
        node_salience=node_salience,
        node_resonance=node_resonance,
        query_text=query_text,
    )

    parts = []
    seen_targets = set()
    remaining = token_budget
    query_plan = _recall_query_plan(query_text)
    allow_archive_targets = query_plan.allow_archive_targets
    for hit in hits:
        target_id = hit.bucket_id
        if not target_id or target_id in seen_targets:
            continue

        target = bucket_map.get(target_id)
        if not target:
            continue
        meta = target.get("metadata", {})
        if not can_bucket_be_related_target(target, explicit_lookup=allow_archive_targets):
            continue
        if (
            query_plan.enforce_topic_evidence
            and not _bucket_has_query_topic_evidence(query_text, target)
        ):
            continue

        try:
            clean_meta = {k: v for k, v in meta.items() if k != "tags"}
            raw_summary = await dehydrator.dehydrate(
                _bucket_text_for_embedding(target),
                clean_meta,
            )
            summary = _compact_diffused_summary(target, raw_summary)
            context = _bucket_temperature_context(target)
            path_summary = _bucket_diffusion_path_summary(hit.best_path, bucket_map)
            caution = (
                "路径含冲突/阻断，仅作边界背景。"
                if path_has_caution(hit.best_path)
                else "背景联想，不代表当前事实。"
            )
            if (
                diffusion_options.chain_walk_enabled
                and len(getattr(hit.best_path, "steps", ()) or ()) >= 2
            ):
                block = _format_bucket_chain_bundle(
                    target_id,
                    target,
                    summary,
                    context,
                    hit.best_path,
                    bucket_map,
                    caution,
                )
            else:
                path_part = f"路径: {path_summary}；" if path_summary else ""
                context_part = f"；语境: {context}" if context else ""
                block = f"- [bucket_id:{target_id}] {path_part}摘要: {summary}{context_part}（{caution}）"
            block_tokens = count_tokens_approx(block)
            if block_tokens > remaining:
                break
            parts.append(block)
            seen_targets.add(target_id)
            remaining -= block_tokens
            if remaining <= 0:
                break
        except Exception as e:
            logger.warning(f"Failed to build diffused memory block / 联想浮现构建失败: {e}")
            continue

    return "\n---\n".join(parts)


def _format_bucket_chain_bundle(
    target_id: str,
    target: dict,
    summary: str,
    temperature_context: str,
    path,
    bucket_map: dict[str, dict],
    note: str,
) -> str:
    nodes = tuple(str(node_id) for node_id in (getattr(path, "nodes", ()) or ()))
    seed_id = nodes[0] if nodes else ""
    seed_label = _inspect_bucket_label(bucket_map.get(seed_id), seed_id)
    target_label = _inspect_bucket_label(target, target_id)
    chain = _bucket_diffusion_path_summary(path, bucket_map)
    temperature_part = f"；temperature: {temperature_context}" if temperature_context else ""
    return (
        f"- Chain Bundle: seed {seed_label}；chain: {chain}；"
        f"target: {target_label}: {summary}{temperature_part}（{note}）"
    )


def _bucket_temperature_context(bucket: dict, max_items: int = 2, max_chars: int = 90) -> str:
    try:
        moments = parse_bucket_moments(bucket, _recall_relevance_options())
    except Exception:
        return ""
    contexts = [
        moment
        for moment in moments
        if moment.get("section") in MOMENT_TEMPERATURE_SECTIONS and _moment_text(moment, max_chars)
    ][:max_items]
    return " / ".join(f"[{_moment_label(moment)}] {_moment_text(moment, max_chars)}" for moment in contexts)


MOMENT_SECTION_LABELS = {
    "scene": "scene",
    "body": "body",
    "moment": "moment",
    "fact": "fact",
    "original": "original",
    "evidence_context": "evidence_context",
    "context": "context",
    "reflection": "reflection",
    "feeling": "feeling",
    "followup": "followup",
    "affect_anchor": "affect_anchor",
    "favorite_reason": "favorite_reason",
    "comment": "年轮",
}

TASK_ONLY_MOMENT_SECTIONS = {"followup", "followup_log"}
MOMENT_TEMPERATURE_SECTIONS = CONTEXT_ONLY_SECTIONS - TASK_ONLY_MOMENT_SECTIONS
PROFILE_CONTEXT_SECTIONS = ("evidence_context", "context", "reflection", "feeling", "comment")


def _moment_text(moment: dict, max_chars: int = 500) -> str:
    text = strip_temperature_meaning_lines(str(moment.get("text") or ""))
    return _clip_text(" ".join(text.split()), max_chars)


def _moment_label(moment: dict) -> str:
    section = str(moment.get("section") or "moment")
    return MOMENT_SECTION_LABELS.get(section, section)


def _moment_bucket_title(moment: dict) -> str:
    meta = moment.get("metadata", {}) if isinstance(moment.get("metadata"), dict) else {}
    return str(meta.get("bucket_name") or moment.get("bucket_id") or "").strip()


def _moments_by_bucket(moments: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for moment in moments:
        bucket_id = str(moment.get("bucket_id") or "")
        if bucket_id:
            grouped.setdefault(bucket_id, []).append(moment)
    for items in grouped.values():
        items.sort(key=lambda item: int(item.get("ordinal") or 0))
    return grouped


def _is_breath_recall_seed_bucket(bucket: dict | None) -> bool:
    if not isinstance(bucket, dict):
        return False
    if is_self_anchor_bucket(bucket) or _is_profile_fact_bucket(bucket):
        return False
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    if meta.get("type") != "feel":
        return True
    tags = {str(tag).lower() for tag in meta.get("tags", []) or []}
    return not ({"relationship_weather", "daily_impression", "weekly_impression"} & tags)


def _breath_recall_seed_buckets(buckets: list[dict]) -> list[dict]:
    return [bucket for bucket in buckets if _is_breath_recall_seed_bucket(bucket)]


async def _ordinary_seed_buckets_with_fact_routes(buckets: list[dict]) -> list[dict]:
    """Resolve matched ProfileFact indexes to their evidence Scenes.

    Facts remain searchable routing records, but never become visible candidates.
    """
    all_active = await bucket_mgr.list_all(include_archive=False)
    migration_map = active_scene_migration_map(all_active)
    migrated_source_ids = set(migration_map)
    ordinary = [
        bucket
        for bucket in _breath_recall_seed_buckets(buckets)
        if str(bucket.get("id") or "") not in migrated_source_ids
    ]
    by_id = {
        str(bucket.get("id") or ""): bucket
        for bucket in ordinary
        if str(bucket.get("id") or "")
    }
    for fact_bucket in buckets:
        if not _is_profile_fact_bucket(fact_bucket):
            continue
        meta = fact_bucket.get("metadata", {}) if isinstance(fact_bucket.get("metadata"), dict) else {}
        if _profile_fact_state(meta) != "active":
            continue
        try:
            fact_score = float(fact_bucket.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            fact_score = 0.0
        fact_sections = _profile_fact_sections(fact_bucket.get("content", ""))
        for evidence in _profile_fact_evidence(fact_bucket):
            source_evidence_id = str(evidence.get("bucket_id") or "").strip()
            if not source_evidence_id or not MEMORY_ID_RE.fullmatch(source_evidence_id):
                continue
            evidence_moment_id = str(evidence.get("moment_id") or "")
            migrated_scene = migrated_scene_for_legacy_source(
                migration_map,
                source_evidence_id,
                evidence_moment_id,
            )
            evidence_bucket = migrated_scene or await bucket_mgr.get(source_evidence_id)
            evidence_id = str((evidence_bucket or {}).get("id") or source_evidence_id)
            if not _is_fact_evidence_recall_bucket(
                evidence_bucket,
                # The moment id identifies the exact span in the preserved
                # source bucket; migrated Scenes have their own parsed moment
                # ids and are validated as canonical Scene objects instead.
                evidence_moment_id="" if migrated_scene is not None else evidence_moment_id,
            ):
                continue
            routed = dict(evidence_bucket)
            routed["score"] = max(float(routed.get("score", 0.0) or 0.0), fact_score * 0.98)
            route_payload = {
                "fact_id": str(fact_bucket.get("id") or ""),
                "fact": str(fact_sections.get("fact") or ""),
                "evidence_moment_id": evidence_moment_id,
                "legacy_evidence": not _is_canonical_scene_bucket(evidence_bucket),
            }
            if evidence_id != source_evidence_id:
                route_payload["source_evidence_bucket_id"] = source_evidence_id
            routed["profile_fact_route"] = route_payload
            existing = by_id.get(evidence_id)
            if not existing or float(existing.get("score", 0.0) or 0.0) < routed["score"]:
                by_id[evidence_id] = routed
    return sorted(
        by_id.values(),
        key=lambda bucket: float(bucket.get("score", 0.0) or 0.0),
        reverse=True,
    )


def _is_daily_impression_feel_bucket(bucket: dict | None) -> bool:
    if not isinstance(bucket, dict):
        return False
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    if meta.get("type") != "feel":
        return False
    tags = {str(tag).lower() for tag in meta.get("tags", []) or []}
    return (
        "daily_impression" in tags
        or str(meta.get("period") or "").lower() == "daily"
        or str(bucket.get("id") or "").startswith("reflection_daily_")
    )


def _moment_from_feel_bucket(moment: dict | None) -> bool:
    if not isinstance(moment, dict):
        return False
    meta = moment.get("metadata", {}) if isinstance(moment.get("metadata"), dict) else {}
    if meta.get("bucket_type") != "feel":
        return False
    tags = {str(tag).lower() for tag in meta.get("bucket_tags", []) or []}
    return bool({"relationship_weather", "daily_impression", "weekly_impression"} & tags)


def _recallable_moments(moments: list[dict]) -> list[dict]:
    return [
        moment for moment in moments
        if can_moment_be_recall_context(moment)
        and not _is_self_anchor_recall_excluded_moment(moment)
        and not _moment_from_feel_bucket(moment)
    ]


def _direct_recallable_moments(moments: list[dict], *, explicit_lookup: bool = False) -> list[dict]:
    return [
        moment for moment in _recallable_moments(moments)
        if can_moment_be_direct_seed(moment, explicit_lookup=explicit_lookup)
    ]


def _related_recallable_moments(moments: list[dict], *, explicit_lookup: bool = False) -> list[dict]:
    return [
        moment for moment in _recallable_moments(moments)
        if can_moment_be_related_target(moment, explicit_lookup=explicit_lookup)
    ]


def _representative_moment(moments: list[dict]) -> dict | None:
    for section in (
        "scene",
        "original",
        "body",
        "moment",
        "fact",
        "evidence_context",
        "context",
        "reflection",
        "feeling",
        "followup",
        "comment",
    ):
        for moment in moments:
            if moment.get("section") == section:
                return moment
    return moments[0] if moments else None


def _direct_representative_moment(moments: list[dict], *, explicit_lookup: bool = False) -> dict | None:
    return _representative_moment(_direct_recallable_moments(moments, explicit_lookup=explicit_lookup))


def _related_representative_moment(moments: list[dict], *, explicit_lookup: bool = False) -> dict | None:
    return _representative_moment(_related_recallable_moments(moments, explicit_lookup=explicit_lookup))


def _bucket_edges_as_moment_edges(bucket_edges: list[dict], grouped: dict[str, list[dict]]) -> list[dict]:
    edges = []
    for edge in bucket_edges or []:
        source_bucket = str(edge.get("source") or edge.get("source_memory_id") or "").strip()
        target_bucket = str(edge.get("target") or edge.get("target_memory_id") or "").strip()
        if not source_bucket or not target_bucket:
            continue
        target = _related_representative_moment(grouped.get(target_bucket, []), explicit_lookup=True)
        if not target:
            continue
        relation_type = str(edge.get("relation_type") or edge.get("type") or "relates_to")
        try:
            confidence = float(edge.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        for source in grouped.get(source_bucket, []):
            if not can_moment_be_direct_seed(source):
                continue
            edges.append(
                {
                    "source": source["moment_id"],
                    "target": target["moment_id"],
                    "bucket_id": source_bucket,
                    "relation_type": relation_type,
                    "confidence": max(0.0, min(1.0, confidence)),
                    "reason": edge.get("reason") or "bucket edge bridge",
                    "graph_scope": str(edge.get("graph_scope") or ""),
                }
            )
    return edges


def _moment_diffusion_map(moments: list[dict]) -> dict[str, dict]:
    mapped = {}
    for moment in moments:
        moment_id = moment.get("moment_id")
        if not moment_id:
            continue
        item = dict(moment)
        meta = dict(item.get("metadata", {}) or {})
        meta["importance"] = meta.get("bucket_importance", 5)
        meta["type"] = meta.get("bucket_type", "")
        meta["anchor"] = meta.get("bucket_anchor", False)
        meta["pinned"] = meta.get("bucket_pinned", False)
        meta["protected"] = meta.get("bucket_protected", False)
        meta["name"] = meta.get("bucket_name", "")
        meta["resolved"] = meta.get("bucket_resolved", False)
        meta["digested"] = meta.get("bucket_digested", False)
        item["metadata"] = meta
        mapped[str(moment_id)] = item
    return mapped


def _seed_scores_for_moments(moments: list[dict]) -> dict[str, float]:
    scores = {}
    for moment in moments:
        moment_id = str(moment.get("moment_id") or "")
        if not moment_id:
            continue
        try:
            score = float(moment.get("score", 0.65))
        except (TypeError, ValueError):
            score = 0.65
        scores[moment_id] = max(0.15, min(1.0, score))
    return scores


def _context_moments_for_seed(seed: dict, grouped: dict[str, list[dict]]) -> list[dict]:
    bucket_id = str(seed.get("bucket_id") or "")
    seed_id = seed.get("moment_id")
    bucket_moments = grouped.get(bucket_id, [])
    contexts = []

    def add_context(moment: dict) -> None:
        if moment.get("moment_id") == seed_id:
            return
        if any(existing.get("moment_id") == moment.get("moment_id") for existing in contexts):
            return
        contexts.append(moment)

    if _is_profile_fact_moment(seed):
        for section in PROFILE_CONTEXT_SECTIONS:
            for moment in bucket_moments:
                if moment.get("section") == section:
                    add_context(moment)
                    break
        return contexts[:4]

    seed_ordinal = int(seed.get("ordinal") or 0)
    for moment in bucket_moments:
        section = moment.get("section")
        ordinal = int(moment.get("ordinal") or 0)
        if abs(ordinal - seed_ordinal) == 1 and section not in MOMENT_TEMPERATURE_SECTIONS:
            add_context(moment)
    for section in ("affect_anchor", "favorite_reason", "comment"):
        for moment in bucket_moments:
            if moment.get("moment_id") != seed_id and moment.get("section") == section:
                add_context(moment)
                break
    return contexts[:4]


def _is_profile_fact_moment(moment: dict) -> bool:
    meta = moment.get("metadata", {}) if isinstance(moment.get("metadata"), dict) else {}
    tags = {str(tag) for tag in meta.get("tags", []) or []}
    tags.update(str(tag) for tag in meta.get("bucket_tags", []) or [])
    return "profile_fact" in tags or bool(meta.get("profile_kind"))


def _format_direct_moment(seed: dict, grouped: dict[str, list[dict]], token_budget: int) -> str:
    title = _moment_bucket_title(seed)
    head = f"[bucket_id:{seed['bucket_id']}] [moment_id:{seed['moment_id']}] {_moment_label(seed)}"
    if title and title != seed["bucket_id"]:
        head += f" {title}"
    parts = [head, _moment_text(seed, 520)]
    context_lines = [
        f"- [{_moment_label(moment)}] [moment_id:{moment['moment_id']}] {_moment_text(moment, 260)}"
        for moment in _context_moments_for_seed(seed, grouped)
    ]
    if context_lines:
        parts.append("语境:\n" + "\n".join(context_lines))
    block = "\n".join(parts)
    if count_tokens_approx(block) <= token_budget:
        return block
    compact = f"{head}\n{_moment_text(seed, 260)}"
    return compact if count_tokens_approx(compact) <= token_budget else ""


def _matching_year_ring(bucket: dict, query_text: str) -> dict | None:
    """Select one attached annotation; never turn it into a standalone seed."""
    query = str(query_text or "").lower()
    explicit_year_ring_lookup = any(
        marker in query
        for marker in ("年轮", "后来", "之后怎么看", "现在怎么看", "重新看", "再看", "later")
    )
    if not explicit_year_ring_lookup and not _is_canonical_scene_bucket(bucket):
        return None
    comments = [
        moment
        for moment in parse_bucket_moments(bucket, _recall_relevance_options())
        if moment.get("section") == "comment"
    ]
    if not comments:
        return None
    terms = _breath_lexical_match_terms(query_text)
    scored: list[tuple[int, str, dict]] = []
    for comment in comments:
        text = str(comment.get("text") or "").lower()
        score = sum(1 for term in terms if str(term or "").strip().lower() in text)
        created = str(comment.get("created_at") or "")
        if score:
            scored.append((score, created, comment))
    if scored:
        scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
        return scored[0][2]
    if explicit_year_ring_lookup:
        return max(comments, key=lambda item: str(item.get("created_at") or ""))
    return None


def _format_attached_year_ring(bucket: dict, query_text: str) -> str:
    ring = _matching_year_ring(bucket, query_text)
    if not ring:
        return ""
    created = str(ring.get("created_at") or "").strip()
    date_label = f" created:{created}" if created else ""
    return (
        f"[attached_annotation{date_label}] "
        f"{_moment_text(ring, 260)}"
    ).strip()


async def _format_direct_bucket(
    bucket: dict,
    moment: dict,
    grouped: dict[str, list[dict]],
    token_budget: int,
    *,
    query_text: str = "",
    direct_render_mode: str = "auto",
) -> str:
    original = _rendered_bucket_content(bucket)
    header = _direct_bucket_header(bucket, moment)
    attached_year_ring = _format_attached_year_ring(bucket, query_text)
    if _is_source_record_synthetic_moment(moment):
        return await _format_source_record_direct_bucket(
            bucket,
            moment,
            header,
            original,
            token_budget,
        )
    original_block = f"{header} bucket_original\n{original}" if original else f"{header} bucket_original"
    if attached_year_ring:
        original_block += "\n\n" + attached_year_ring
    if count_tokens_approx(original_block) <= token_budget:
        return original_block

    wants_capsule = direct_render_mode == "full" or (
        direct_render_mode == "auto"
        and (_bucket_is_high_value(bucket) or _query_requests_direct_detail(query_text))
    )
    if wants_capsule:
        try:
            capsule = await dehydrator.dehydrate_direct_capsule(
                original,
                _bucket_metadata_for_dehydration(bucket),
            )
            block = f"{header} bucket_capsule\n{capsule}\nmatched_moment: {_moment_text(moment, 220)}"
            if attached_year_ring:
                block += "\n" + attached_year_ring
            if count_tokens_approx(block) <= token_budget:
                return block
            compact = f"{header} bucket_capsule\n{_clip_text(capsule, 260)}"
            if attached_year_ring:
                compact += "\n" + _clip_text(attached_year_ring, 180)
            if count_tokens_approx(compact) <= token_budget:
                return compact
            return _trim_text_to_token_budget(compact, token_budget)
        except Exception as e:
            logger.warning(f"Direct bucket capsule failed / 直接命中整桶脱水失败: {e}")

    return _format_direct_bucket_window(
        bucket,
        moment,
        grouped,
        token_budget,
        attached_year_ring=attached_year_ring,
    )


async def _format_source_record_direct_bucket(
    bucket: dict,
    moment: dict,
    header: str,
    original: str,
    token_budget: int,
) -> str:
    meta = moment.get("metadata", {}) if isinstance(moment.get("metadata"), dict) else {}
    matched_label = "matched_fragment" if meta.get("source_record_fragment_seed") else "matched_source_record"
    matched = _moment_text(moment, 260)
    try:
        capsule = await dehydrator.dehydrate_direct_capsule(
            original or matched,
            _bucket_metadata_for_dehydration(bucket),
        )
    except Exception as e:
        logger.warning(f"Source record capsule failed / source_record 脱水失败: {e}")
        capsule = _source_record_capsule_seed_text(bucket) or matched
    block = f"{header} bucket_capsule\n{capsule}\n{matched_label}: {matched}"
    if count_tokens_approx(block) <= token_budget:
        return block
    compact = f"{header} bucket_capsule\n{_clip_text(capsule, 260)}\n{matched_label}: {matched}"
    if count_tokens_approx(compact) <= token_budget:
        return compact
    return _trim_text_to_token_budget(compact, token_budget)


def _direct_bucket_render_debug(
    bucket: dict | None,
    moment: dict | None,
    token_budget: int,
    *,
    query_text: str = "",
    direct_render_mode: str = "auto",
) -> dict:
    bucket = bucket or {}
    moment = moment or {}
    mode = _normalize_direct_render_mode(direct_render_mode)
    original = _rendered_bucket_content(bucket)
    header = _direct_bucket_header(bucket, moment)
    original_block = f"{header} bucket_original\n{original}" if original else f"{header} bucket_original"
    original_tokens = count_tokens_approx(original_block)
    budget = max(0, int(token_budget or 0))
    if _is_source_record_synthetic_moment(moment):
        meta = moment.get("metadata", {}) if isinstance(moment.get("metadata"), dict) else {}
        return {
            "mode": mode,
            "shape": "bucket_capsule",
            "reason": str(meta.get("source_record_direct_reason") or "source_record_direct"),
            "token_budget": budget,
            "original_tokens": original_tokens,
            "original_fits": False,
            "high_value": False,
            "detail_query": False,
            "wants_capsule": True,
        }
    high_value = _bucket_is_high_value(bucket)
    detail_query = _query_requests_direct_detail(query_text)
    original_fits = original_tokens <= budget
    wants_capsule = mode == "full" or (mode == "auto" and (high_value or detail_query))
    if original_fits:
        shape = "bucket_original"
        reason = "original_fits_budget"
    elif wants_capsule:
        shape = "bucket_capsule"
        if mode == "full":
            reason = "mode_full"
        elif detail_query:
            reason = "auto_detail_query"
        else:
            reason = "auto_high_value"
    else:
        shape = "bucket_window"
        reason = "long_bucket_window"
    return {
        "mode": mode,
        "shape": shape,
        "reason": reason,
        "token_budget": budget,
        "original_tokens": original_tokens,
        "original_fits": original_fits,
        "high_value": high_value,
        "detail_query": detail_query,
        "wants_capsule": wants_capsule,
    }


def _format_direct_bucket_window(
    bucket: dict,
    moment: dict,
    grouped: dict[str, list[dict]],
    token_budget: int,
    *,
    attached_year_ring: str = "",
) -> str:
    header = _direct_bucket_header(bucket, moment)
    original = _rendered_bucket_content(bucket)
    parts = [
        f"{header} bucket_window",
        f"matched_moment: {_moment_text(moment, 320)}",
    ]
    window = _original_window_around_moment(original, moment)
    if window:
        parts.append("original_window:\n" + window)
    context_lines = [
        f"- [{_moment_label(context)}] [moment_id:{context['moment_id']}] {_moment_text(context, 120)}"
        for context in _context_moments_for_seed(moment, grouped)
        if context.get("section") in MOMENT_TEMPERATURE_SECTIONS
    ][:2]
    if context_lines:
        parts.append("语境:\n" + "\n".join(context_lines))
    if attached_year_ring:
        parts.append(attached_year_ring)
    block = "\n".join(parts)
    if count_tokens_approx(block) <= token_budget:
        return block
    compact_parts = [
        f"{header} bucket_window",
        f"matched_moment: {_moment_text(moment, 120)}",
    ]
    if window:
        compact_parts.append("original_window:\n" + _clip_text(window, 220))
    compact = "\n".join(compact_parts)
    if count_tokens_approx(compact) <= token_budget:
        return compact
    return _trim_text_to_token_budget(compact, token_budget)


def _trim_text_to_token_budget(text: str, token_budget: int) -> str:
    if token_budget <= 0:
        return ""
    trimmed = str(text or "").strip()
    while trimmed and count_tokens_approx(trimmed) > token_budget:
        cut = max(1, int(len(trimmed) * 0.85))
        trimmed = trimmed[:cut].rstrip()
    return trimmed


def _trim_lines_to_token_budget(text: str, token_budget: int) -> str:
    if token_budget <= 0:
        return ""
    trimmed = str(text or "").strip()
    if not trimmed or count_tokens_approx(trimmed) <= token_budget:
        return trimmed
    kept: list[str] = []
    for line in trimmed.splitlines():
        line = line.rstrip()
        if not line.strip():
            continue
        candidate = "\n".join([*kept, line])
        if count_tokens_approx(candidate) > token_budget:
            break
        kept.append(line)
    return "\n".join(kept)


def _normalize_direct_render_mode(value: object) -> str:
    mode = str(value or "auto").strip().lower()
    return mode if mode in {"auto", "compact", "full"} else "auto"


def _normalize_retrieval_mode(value: object) -> str:
    mode = str(value or "graph").strip().lower()
    return mode if mode in {"graph", "bucket"} else "graph"


def _normalize_recall_fusion_mode(value: object) -> str:
    mode = str(value or "dynamic").strip().lower()
    return mode if mode in {"dynamic", "legacy"} else "dynamic"


def _query_resurface_enabled() -> bool:
    recall_cfg = config.get("recall", {}) if isinstance(config.get("recall", {}), dict) else {}
    return _bool_value(recall_cfg.get("query_resurface_enabled"), False)


def _word_map_hint_available() -> bool:
    gateway_cfg = config.get("gateway", {}) if isinstance(config.get("gateway", {}), dict) else {}
    return (
        _bool_value(gateway_cfg.get("word_map_hint_enabled"), False)
        and word_map_store is not None
        and bool(getattr(word_map_store, "enabled", False))
    )


def _word_map_hint_settings() -> dict[str, float | int]:
    gateway_cfg = config.get("gateway", {}) if isinstance(config.get("gateway", {}), dict) else {}
    return {
        "moment_boost": _float_between(gateway_cfg.get("word_map_hint_moment_boost"), 0.25, 0.0, 1.0),
        "neighbor_limit": _int_between(gateway_cfg.get("word_map_hint_neighbor_limit"), 6, 0, 40),
        "bucket_limit": _int_between(gateway_cfg.get("word_map_hint_bucket_limit"), 12, 1, 100),
    }


def _get_breath_word_map_hints(
    query: str,
    buckets: list[dict],
) -> tuple[dict[str, float], dict[str, dict]]:
    if not _word_map_hint_available():
        return {}, {}
    terms = _specific_query_terms(query)
    if not terms:
        return {}, {}
    eligible_ids = {
        str(bucket.get("id") or "")
        for bucket in buckets
        if _is_breath_recall_seed_bucket(bucket) and bucket.get("id")
    }
    if not eligible_ids:
        return {}, {}
    settings = _word_map_hint_settings()
    try:
        payload = word_map_store.hint_buckets_for_terms(
            terms,
            neighbor_limit=int(settings["neighbor_limit"]),
            bucket_limit=int(settings["bucket_limit"]),
        )
    except Exception as exc:
        logger.warning("Breath word map hint lookup failed / breath 词图提示查询失败: %s", exc)
        return {}, {}

    raw_scores = payload.get("bucket_scores", {}) if isinstance(payload, dict) else {}
    raw_evidence = payload.get("evidence", {}) if isinstance(payload, dict) else {}
    scores: dict[str, float] = {}
    debug: dict[str, dict] = {}
    for bucket_id, score in raw_scores.items():
        bucket_id = str(bucket_id or "")
        if bucket_id not in eligible_ids:
            continue
        scores[bucket_id] = max(0.0, min(1.0, float(score or 0.0)))
        evidence = raw_evidence.get(bucket_id, {}) if isinstance(raw_evidence, dict) else {}
        debug[bucket_id] = evidence if isinstance(evidence, dict) else {}
    return scores, debug


def _append_breath_word_map_matches(
    *,
    query: str,
    matches: list[dict],
    all_buckets: list[dict],
    seed_diagnostics: dict[str, dict],
) -> tuple[dict[str, float], dict[str, dict]]:
    word_map_scores, word_map_debug = _get_breath_word_map_hints(query, all_buckets)
    if not word_map_scores:
        return {}, {}

    bucket_map = {
        str(bucket.get("id") or ""): bucket
        for bucket in all_buckets
        if isinstance(bucket, dict) and bucket.get("id")
    }
    matched_ids = {str(bucket.get("id") or "") for bucket in matches if bucket.get("id")}
    for bucket_id, hint_score in word_map_scores.items():
        bucket = bucket_map.get(bucket_id)
        if not _is_breath_recall_seed_bucket(bucket):
            continue
        hint_debug = word_map_debug.get(bucket_id) or {}
        bucket["word_map_hint"] = True
        bucket["word_map_score"] = round(float(hint_score), 4)
        bucket["word_map_terms"] = list(hint_debug.get("direct_terms") or [])
        bucket["word_map_neighbor_terms"] = list(hint_debug.get("neighbor_terms") or [])
        seed = seed_diagnostics.setdefault(
            bucket_id,
            {
                "bucket_id": bucket_id,
                "bucket_name": (bucket.get("metadata") or {}).get("name") or bucket_id,
                "sources": [],
            },
        )
        if "word_map" not in seed["sources"]:
            seed["sources"].append("word_map")
        seed["word_map_score"] = round(float(hint_score), 4)
        seed["word_map_terms"] = bucket["word_map_terms"]
        seed["word_map_neighbor_terms"] = bucket["word_map_neighbor_terms"]
        if bucket_id not in matched_ids:
            bucket["score"] = max(0.15, min(1.0, float(hint_score)))
            matches.append(bucket)
            matched_ids.add(bucket_id)
    matches.sort(key=lambda bucket: float(bucket.get("score", 0.0) or 0.0), reverse=True)
    return word_map_scores, word_map_debug


def _breath_word_map_only_without_topic(query: str, moment: dict, seed_diagnostics: dict[str, dict]) -> bool:
    bucket_id = str(moment.get("bucket_id") or "")
    seed = seed_diagnostics.get(bucket_id, {})
    sources = set(seed.get("sources") or [])
    if "word_map" not in sources or len(sources - {"word_map"}) > 0:
        return False
    if _moment_has_query_topic_evidence(query, moment):
        return False
    return not _recall_policy().has_strong_score(rerank_score=moment.get("rerank_score"))


def _bucket_relevance_node(bucket: dict, score: float = 0.0) -> dict:
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    return {
        "id": bucket.get("id"),
        "text": bucket_content_for_recall(bucket),
        "score": score,
        "metadata": {
            "bucket_name": meta.get("name") or bucket.get("id"),
            "bucket_tags": meta.get("tags") or [],
            "bucket_domain": meta.get("domain") or [],
            "annotation_summary": meta.get("annotation_summary") or meta.get("summary") or "",
            "evidence_spans": meta.get("evidence_spans") or [],
        },
    }


def _is_source_record_bucket(bucket: dict | None) -> bool:
    return isinstance(bucket, dict) and infer_bucket_layer(bucket) == LAYER_SOURCE_RECORD


def _source_record_synthetic_moment_for_bucket(
    bucket: dict,
    query: str,
    *,
    selected_reason: str = "",
) -> dict | None:
    if not _is_source_record_bucket(bucket) or is_self_anchor_bucket(bucket):
        return None
    bucket_id = str(bucket.get("id") or "")
    if not bucket_id:
        return None
    fragment = _source_record_fragment_for_query(query, bucket)
    explicit_reason = _source_record_explicit_bucket_match_reason(query, bucket)
    if not fragment and not explicit_reason:
        return None
    fragment_seed = bool(fragment)
    reason = "source_record_fragment_direct" if fragment_seed else "source_record_explicit_bucket_capsule"
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    text = fragment or _source_record_capsule_seed_text(bucket)
    return {
        "moment_id": _source_record_synthetic_moment_id(bucket_id, reason, text),
        "bucket_id": bucket_id,
        "section": "source_fragment" if fragment_seed else "source_capsule",
        "text": text,
        "ordinal": 0,
        "source": "source_record_synthetic",
        "source_id": reason,
        "score": max(1.0, _score_to_unit(bucket.get("score", 0.0))),
        "admission_reason": reason,
        "metadata": {
            "bucket_name": meta.get("name") or bucket_id,
            "bucket_type": meta.get("type") or "source",
            "bucket_tags": list(meta.get("tags") or []),
            "bucket_domain": list(meta.get("domain") or []),
            "bucket_importance": meta.get("importance"),
            "bucket_created": meta.get("created"),
            "bucket_updated_at": meta.get("updated_at") or meta.get("last_active"),
            "source_record_direct": True,
            "source_record_direct_reason": reason,
            "source_record_match_reason": explicit_reason or selected_reason or "selected_bucket",
            "source_record_fragment_seed": fragment_seed,
            "source_record_capsule_only": not fragment_seed,
        },
        "_source_record_synthetic": True,
    }


def _source_record_synthetic_moments_for_matches(matches: list[dict], query: str) -> list[dict]:
    output = []
    seen = set()
    for bucket in matches or []:
        bucket_id = str((bucket or {}).get("id") or "")
        if not bucket_id or bucket_id in seen:
            continue
        moment = _source_record_synthetic_moment_for_bucket(
            bucket,
            query,
            selected_reason="seed_bucket",
        )
        if moment:
            output.append(moment)
            seen.add(bucket_id)
    return output


def _prepend_source_record_synthetic_moments(
    moments: list[dict],
    synthetics: list[dict],
) -> list[dict]:
    if not synthetics:
        return moments
    source_ids = {str(moment.get("bucket_id") or "") for moment in synthetics}
    return list(synthetics) + [
        moment for moment in moments
        if str(moment.get("bucket_id") or "") not in source_ids
    ]


def _source_record_fragment_for_query(query: str, bucket: dict, *, max_chars: int = 360) -> str:
    terms = _recall_policy().specific_query_terms(query)
    if not terms:
        return ""
    original = _rendered_bucket_content(bucket)
    if not original:
        return ""
    lowered = original.lower()
    matches = []
    for term in terms:
        needle = str(term or "").strip().lower()
        if len(needle) < 2:
            continue
        index = lowered.find(needle)
        if index >= 0:
            matches.append((index, needle))
    if not matches:
        return ""
    index, needle = sorted(matches, key=lambda item: (item[0], -len(item[1])))[0]
    half = max_chars // 2
    start = max(0, index - half)
    end = min(len(original), index + len(needle) + half)
    fragment = original[start:end].strip()
    if start > 0:
        fragment = "..." + fragment
    if end < len(original):
        fragment += "..."
    return fragment


def _source_record_explicit_bucket_match_reason(query: str, bucket: dict) -> str:
    if not query or not isinstance(bucket, dict):
        return ""
    bucket_id = str(bucket.get("id") or "")
    query_text = str(query or "")
    if bucket_id and bucket_id in query_text:
        return "explicit_bucket_id"
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    title = str(meta.get("name") or bucket_id or "").strip()
    title_key = _compact_lookup_key(title)
    query_key = _compact_lookup_key(query)
    if title_key and (query_key == title_key or title_key in query_key):
        return "explicit_bucket_title"
    for term in _recall_policy().specific_query_terms(query):
        term_key = _compact_lookup_key(term)
        if not term_key or len(term_key) < 2:
            continue
        if term_key == title_key or (len(term_key) >= 3 and term_key in title_key):
            return "explicit_bucket_title"
    return ""


def _compact_lookup_key(value: object) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(value or "").strip().lower())


def _source_record_synthetic_moment_id(bucket_id: str, reason: str, text: str) -> str:
    digest = hashlib.sha1(f"{bucket_id}\n{reason}\n{text}".encode("utf-8")).hexdigest()[:12]
    return f"{bucket_id}:source-record:{digest}"


def _source_record_capsule_seed_text(bucket: dict) -> str:
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    title = str(meta.get("name") or bucket.get("id") or "").strip()
    summary = str(meta.get("annotation_summary") or meta.get("summary") or "").strip()
    return _clip_text(" | ".join(part for part in (title, summary) if part), 260) or title


def _is_source_record_synthetic_moment(moment: dict | None) -> bool:
    meta = moment.get("metadata", {}) if isinstance(moment, dict) and isinstance(moment.get("metadata"), dict) else {}
    return bool(meta.get("source_record_direct") or (moment or {}).get("_source_record_synthetic"))


def _breath_moment_runtime_gate_payload(
    moment: dict,
    *,
    explicit_lookup: bool = False,
) -> dict:
    gate = moment_runtime_gate_debug(moment, explicit_lookup=explicit_lookup)
    if _is_source_record_synthetic_moment(moment):
        meta = moment.get("metadata", {}) if isinstance(moment.get("metadata"), dict) else {}
        reason = str(meta.get("source_record_direct_reason") or "source_record_direct")
        gate["source_record_direct_override"] = True
        gate["topic_evidence"] = {
            "required": bool(meta.get("source_record_fragment_seed")),
            "present": True if meta.get("source_record_fragment_seed") else None,
        }
        gate["direct_injection"] = {
            "allowed": True,
            "reason": reason,
        }
        gate["would_inject_direct"] = True
    return gate


def _direct_moments_for_bucket(bucket: dict, query: str = "") -> list[dict]:
    explicit_lookup = _query_explicitly_requests_archive_memory(query)
    return [
        moment for moment in parse_bucket_moments(bucket, _recall_relevance_options())
        if can_moment_be_recall_context(moment)
        and can_moment_be_direct_seed(moment, explicit_lookup=explicit_lookup)
    ]


def _query_requests_direct_detail(query: str) -> bool:
    text = str(query or "").strip().lower()
    if not text:
        return False
    phrases = (
        "细节",
        "原文",
        "完整",
        "整条",
        "整桶",
        "全部",
        "当时怎么说",
        "当时说了什么",
        "具体怎么说",
        "怎么写的",
        "旧记录",
    )
    return any(phrase in text for phrase in phrases)


def _bucket_is_high_value(bucket: dict) -> bool:
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    if meta.get("pinned") or meta.get("protected") or meta.get("anchor"):
        return True
    try:
        if int(meta.get("importance", 5)) >= 9:
            return True
    except (TypeError, ValueError):
        pass
    return has_favorite_memory_tag(meta.get("tags", []) or [], ai_name=_ai_author_name())


def _bucket_metadata_for_dehydration(bucket: dict) -> dict:
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    return {key: value for key, value in meta.items() if key not in {"tags", "comments"}}


def _date_yyyy_mm_dd(value) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.match(r"^\d{4}-\d{2}-\d{2}", text)
    return match.group(0) if match else text[:10]


def _bucket_date_meta_parts(bucket: dict | None = None, moment: dict | None = None) -> list[str]:
    bucket = bucket or {}
    moment = moment or {}
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    moment_meta = moment.get("metadata", {}) if isinstance(moment.get("metadata"), dict) else {}
    event_date = _date_yyyy_mm_dd(
        meta.get("date")
        or moment_meta.get("bucket_date")
        or moment_meta.get("date")
    )
    if event_date:
        return [f"[date:{event_date}]"]
    created = _date_yyyy_mm_dd(
        meta.get("created")
        or moment_meta.get("bucket_created")
        or moment.get("created_at")
    )
    return [f"[created:{created}]"] if created else []


def _direct_bucket_header(bucket: dict, moment: dict) -> str:
    bucket_id = str(bucket.get("id") or moment.get("bucket_id") or "")
    title = _moment_bucket_title(moment) or str((bucket.get("metadata", {}) or {}).get("name") or bucket_id)
    section = str(moment.get("section") or "body")
    date_part = " ".join(_bucket_date_meta_parts(bucket, moment))
    return (
        f"[bucket_id:{bucket_id}] [moment_id:{moment.get('moment_id') or ''}] "
        f"{date_part} {section} {title}"
    ).strip()


def _rendered_bucket_content(bucket: dict) -> str:
    text = strip_wikilinks(str(bucket.get("content") or ""))
    text = strip_display_temperature_sections(text)
    text = strip_followup_sections(text)
    return strip_temperature_meaning_lines(text).strip()


def _original_window_around_moment(original: str, moment: dict, max_chars: int = 760) -> str:
    text = str(original or "").strip()
    source_window = source_ref_window(
        moment,
        allowed_root=str(config.get("buckets_dir") or ""),
        max_chars=max_chars,
    )
    if not text:
        return source_window
    needle = strip_temperature_meaning_lines(strip_wikilinks(str(moment.get("text") or ""))).strip()
    compact_needle = " ".join(needle.split())
    compact_text = " ".join(text.split())
    if not compact_needle:
        return source_window or _clip_text(compact_text, max_chars)
    index = compact_text.find(compact_needle)
    if index < 0:
        index = compact_text.find(compact_needle[:80])
    if index < 0:
        return source_window or _clip_text(compact_text, max_chars)
    half = max_chars // 2
    start = max(0, index - half)
    end = min(len(compact_text), index + len(compact_needle) + half)
    window = compact_text[start:end].strip()
    if start > 0:
        window = "..." + window
    if end < len(compact_text):
        window += "..."
    return window


def _format_related_moment(
    moment: dict,
    caution: bool = False,
    path=None,
    moment_map: dict[str, dict] | None = None,
    chain_bundle: bool = False,
) -> str:
    moment_map = moment_map or {}
    if caution:
        note = "路径含冲突/阻断，仅作边界背景。"
    elif path is not None and path_has_old_version(path):
        note = "旧路径/旧版本背景，不代表当前事实。"
    else:
        note = "背景联想，不代表当前事实。"
    if chain_bundle and path is not None and len(getattr(path, "steps", ()) or ()) >= 2:
        return _format_related_chain_bundle(moment, note, path, moment_map)
    summary = _diffused_moment_summary(moment, path=path, moment_map=moment_map)
    context = _diffused_temperature_context(moment, path=path, moment_map=moment_map)
    path_part = ""
    if path is not None:
        path_summary = _moment_path_summary(path, moment_map)
        if path_summary:
            path_part = f"路径: {path_summary}；"
    context_part = f"；语境: {context}" if context else ""
    return (
        f"- [bucket_id:{moment['bucket_id']}] [moment_id:{moment['moment_id']}] "
        f"{path_part}摘要: {summary}{context_part}（{note}）"
    )


def _format_related_chain_bundle(
    moment: dict,
    note: str,
    path,
    moment_map: dict[str, dict],
) -> str:
    nodes = tuple(str(node_id) for node_id in (getattr(path, "nodes", ()) or ()))
    seed_id = nodes[0] if nodes else ""
    seed_label = _moment_node_label(moment_map.get(seed_id), seed_id)
    chain = _moment_path_summary(path, moment_map)
    target = _diffused_moment_summary(moment, path=None, moment_map=moment_map)
    temperature = _diffused_temperature_context(moment, path=path, moment_map=moment_map)
    temperature_part = f"；temperature: {temperature}" if temperature else ""
    return (
        f"- Chain Bundle: seed {seed_label}；chain: {chain}；"
        f"target: {target}{temperature_part}（{note}）"
    )


def _format_secondary_direct_moment(moment: dict) -> str:
    summary = _diffused_moment_summary(moment)
    return (
        f"- [bucket_id:{moment['bucket_id']}] [moment_id:{moment['moment_id']}] "
        f"摘要: {summary}（相关命中，来自同一查询语义。）"
    )


def _diffused_moment_summary(
    moment: dict,
    *,
    path=None,
    moment_map: dict[str, dict] | None = None,
    max_chars: int = 180,
) -> str:
    label = _moment_label(moment)
    title = _moment_bucket_title(moment) or str(moment.get("bucket_id") or "记忆")
    status = _moment_status_label(moment)
    parts = [f"{title} / {label}"]
    if status:
        parts.append(status)
    path_summary = _moment_path_summary(path, moment_map or {}) if path is not None else ""
    if path_summary:
        parts.append(f"路径 {path_summary}")
    return _clip_text("；".join(parts), max_chars)


def _diffused_temperature_context(
    moment: dict,
    *,
    path=None,
    moment_map: dict[str, dict] | None = None,
    max_items: int = 2,
    max_chars: int = 90,
) -> str:
    moment_map = moment_map or {}
    bucket_id = str(moment.get("bucket_id") or "")
    if not bucket_id:
        return ""
    contexts: list[dict] = []
    seen: set[str] = set()

    def add_context(candidate: dict | None) -> None:
        if len(contexts) >= max_items or not isinstance(candidate, dict):
            return
        if str(candidate.get("bucket_id") or "") != bucket_id:
            return
        if candidate.get("section") not in MOMENT_TEMPERATURE_SECTIONS:
            return
        moment_id = str(candidate.get("moment_id") or "")
        if not moment_id or moment_id == str(moment.get("moment_id") or "") or moment_id in seen:
            return
        text = _moment_text(candidate, max_chars)
        if not text:
            return
        seen.add(moment_id)
        contexts.append(candidate)

    for node_id in getattr(path, "nodes", ()) or ():
        add_context(moment_map.get(str(node_id)))
    for candidate in sorted(
        moment_map.values(),
        key=lambda item: int(item.get("ordinal") or 0) if isinstance(item, dict) else 0,
    ):
        add_context(candidate)
        if len(contexts) >= max_items:
            break

    return " / ".join(f"[{_moment_label(item)}] {_moment_text(item, max_chars)}" for item in contexts)


def _moment_status_label(moment: dict) -> str:
    meta = moment.get("metadata", {}) if isinstance(moment.get("metadata"), dict) else {}
    if meta.get("resolved") or meta.get("bucket_resolved"):
        return "已解决"
    if meta.get("digested") or meta.get("bucket_digested"):
        return "已消化"
    if str(meta.get("type") or meta.get("bucket_type") or "").lower() == "archived":
        return "归档"
    return ""


def _moment_path_summary(path, moment_map: dict[str, dict], max_chars: int = 140) -> str:
    if path is None:
        return ""
    nodes = tuple(str(node_id) for node_id in (getattr(path, "nodes", ()) or ()))
    if not nodes:
        return ""
    labels = [_moment_node_label(moment_map.get(nodes[0]), nodes[0])]
    for step in getattr(path, "steps", ()) or ():
        target_id = str(getattr(step, "target", "") or "")
        arrow = "<-" if getattr(step, "direction", "") == "incoming" else "->"
        labels.append(f"{arrow} {_moment_node_label(moment_map.get(target_id), target_id)}")
    return _clip_text(" ".join(labels), max_chars)


def _moment_node_label(moment: dict | None, fallback_id: str) -> str:
    if isinstance(moment, dict):
        return _clip_text(_moment_bucket_title(moment) or str(moment.get("bucket_id") or fallback_id), 48)
    return _clip_text(fallback_id, 48)


def _recall_relevance_options():
    return memory_relevance_options_from_config(config)


def _apply_recall_relevance_gate(query: str, candidates: list[dict]) -> list[dict]:
    options = _recall_relevance_options()
    filtered = []
    adjusted = False
    for moment in candidates:
        multiplier = relevance_multiplier(query, moment, options)
        if multiplier <= 0:
            adjusted = True
            continue
        item = dict(moment)
        try:
            score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        new_score = round(score * multiplier, 4)
        if new_score != score:
            adjusted = True
        item["score"] = new_score
        filtered.append(item)
    if adjusted:
        filtered.sort(key=lambda item: _recall_rank(query, item))
    return filtered


def _recall_admission_thresholds() -> tuple[float, float]:
    threshold_cfg = config.get("recall_thresholds", {}) or {}
    return (
        _float_between(threshold_cfg.get("explicit_admission_semantic_score"), 0.72, 0.0, 1.0),
        _float_between(threshold_cfg.get("explicit_admission_rerank_score"), 0.65, 0.0, 1.0),
    )


_RECALL_POLICY_INSTANCE: RecallPolicy | None = None


def _recall_policy() -> RecallPolicy:
    global _RECALL_POLICY_INSTANCE
    if _RECALL_POLICY_INSTANCE is not None:
        return _RECALL_POLICY_INSTANCE
    semantic_threshold, rerank_threshold = _recall_admission_thresholds()
    _RECALL_POLICY_INSTANCE = RecallPolicy(
        _recall_relevance_options(),
        semantic_threshold=semantic_threshold,
        rerank_threshold=rerank_threshold,
        ai_reaction_names=[identity_names(config).get("ai_name")],
        relationship_names=identity_names(config).get("relationship_terms") or [],
    )
    return _RECALL_POLICY_INSTANCE


def _recall_query_plan(query: str, *, context_mode: str = ""):
    return _recall_policy().plan_query(query, context_mode=context_mode)


def _breath_moment_admission_decision(
    query: str,
    moment: dict,
    seed_diagnostics: dict[str, dict],
    *,
    auto: bool = False,
):
    seed = seed_diagnostics.get(str(moment.get("bucket_id") or ""), {})
    return _recall_policy().assess(
        query,
        moment,
        semantic_score=seed.get("embedding_score"),
        rerank_score=moment.get("rerank_score"),
        high_confidence_edge="lexical" in (seed.get("sources") or []),
        context_only=moment.get("section") in MOMENT_TEMPERATURE_SECTIONS,
        auto=auto,
    )


def _breath_moment_has_reliable_seed_signal(
    query: str,
    moment: dict,
    seed_diagnostics: dict[str, dict],
) -> bool:
    if not isinstance(moment, dict) or not can_moment_be_direct_seed(moment):
        return False
    if should_suppress_context_candidate(query, moment, _recall_relevance_options()):
        return False
    meta = moment.get("metadata", {}) if isinstance(moment.get("metadata"), dict) else {}
    if _is_source_record_synthetic_moment(moment):
        return bool(meta.get("source_record_fragment_seed"))
    bucket_id = str(moment.get("bucket_id") or "")
    seed = seed_diagnostics.get(bucket_id, {})
    sources = set(seed.get("sources") or [])
    if "lexical" in sources:
        return True
    if _recall_policy().has_strong_score(
        semantic_score=seed.get("embedding_score"),
        rerank_score=moment.get("rerank_score"),
    ):
        return True
    return _breath_moment_has_reliable_topic_evidence_for_seed(query, moment)


def _breath_moment_has_reliable_topic_evidence_for_seed(query: str, moment: dict) -> bool:
    terms = [
        str(term).strip()
        for term in _specific_query_terms(query)
        if diffusion_seed_topic_term_has_specific_residue(term)
    ]
    if not terms:
        return False
    meta = moment.get("metadata", {}) if isinstance(moment.get("metadata"), dict) else {}
    fields = " ".join(
        [
            str(moment.get("text") or ""),
            str(meta.get("annotation_summary") or ""),
            str(meta.get("bucket_name") or ""),
            " ".join(str(tag) for tag in meta.get("bucket_tags", []) or []),
            " ".join(str(item) for item in meta.get("bucket_domain", []) or []),
        ]
    ).lower()
    return any(term.lower() in fields for term in terms)


def _promoted_breath_related_seed_moments(
    query: str,
    candidates: list[dict],
    displayed_bucket_ids: set[str],
    seed_diagnostics: dict[str, dict],
    *,
    limit: int,
) -> list[dict]:
    if limit <= 0:
        return []
    promoted = []
    seen = set(displayed_bucket_ids)
    for moment in candidates or []:
        bucket_id = str(moment.get("bucket_id") or "")
        if not bucket_id or bucket_id in seen:
            continue
        if not _breath_moment_has_reliable_seed_signal(query, moment, seed_diagnostics):
            continue
        item = dict(moment)
        item["promoted_related_seed"] = True
        promoted.append(item)
        seen.add(bucket_id)
    promoted.sort(
        key=lambda moment: (
            _recall_rank(query, moment)[0],
            -(_safe_float(moment.get("rerank_score")) or 0.0),
            -(_safe_float(moment.get("combined_score", moment.get("score"))) or 0.0),
        )
    )
    return promoted[:limit]


async def _rerank_breath_moment_candidates(query: str, candidates: list[dict]) -> list[dict]:
    if not candidates or not getattr(reranker_engine, "enabled", False):
        return candidates
    candidate_limit = min(
        len(candidates),
        max(1, int(getattr(reranker_engine, "candidate_limit", 20) or 20)),
    )
    head = candidates[:candidate_limit]
    tail = candidates[candidate_limit:]
    documents = [_moment_rerank_document(moment) for moment in head]
    try:
        results = await reranker_engine.rerank(query, documents, top_n=len(head))
    except Exception as e:
        logger.warning(f"Breath reranker failed; keeping original candidates / breath 重排失败，保留原候选: {e}")
        return candidates
    if not results:
        return candidates

    by_index = {result.index: result.score for result in results}
    weight = max(0.0, min(1.0, float(getattr(reranker_engine, "score_weight", 0.65))))
    reranked = []
    for index, moment in enumerate(head):
        item = dict(moment)
        rerank_score = by_index.get(index)
        try:
            base_score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            base_score = 0.0
        if rerank_score is None:
            item["rerank_score"] = None
            item["combined_score"] = base_score
        else:
            item["rerank_score"] = round(rerank_score, 4)
            item["combined_score"] = round(base_score * (1.0 - weight) + rerank_score * weight, 4)
            item["score"] = item["combined_score"]
        reranked.append(item)
    reranked.sort(
        key=lambda item: (
            _recall_rank(query, item)[0],
            item.get("rerank_score") is None,
            -(_safe_float(item.get("combined_score", item.get("score"))) or 0.0),
            -(_safe_float(item.get("score")) or 0.0),
        ),
    )
    return reranked + tail


def _moment_rerank_document(moment: dict) -> str:
    meta = moment.get("metadata", {}) if isinstance(moment.get("metadata"), dict) else {}
    fields = [
        f"title: {meta.get('bucket_name') or moment.get('bucket_id') or ''}",
        f"section: {moment.get('section') or ''}",
        f"domain: {' '.join(str(item) for item in meta.get('bucket_domain', []) or [])}",
        f"tags: {' '.join(str(item) for item in meta.get('bucket_tags', []) or [])}",
        f"summary: {meta.get('annotation_summary') or meta.get('summary') or ''}",
        f"facets: {_format_annotation_facets(meta)}",
        f"evidence: {_format_evidence_spans(meta)}",
        f"text: {moment.get('text') or ''}",
    ]
    return "\n".join(fields)[:4000]


def _format_annotation_facets(meta: dict) -> str:
    facets = meta.get("annotation_facets")
    if not isinstance(facets, dict):
        return ""
    parts = []
    for facet, score in sorted(facets.items(), key=lambda item: str(item[0])):
        try:
            parts.append(f"{facet}:{float(score):.2f}")
        except (TypeError, ValueError):
            continue
    return " ".join(parts)


def _format_evidence_spans(meta: dict, max_items: int = 3) -> str:
    spans = meta.get("evidence_spans")
    if not isinstance(spans, list):
        return ""
    parts = []
    for item in spans[:max_items]:
        if isinstance(item, dict):
            facet = str(item.get("facet") or "").strip()
            text = str(item.get("text") or item.get("span") or "").strip()
            if text:
                parts.append(f"{facet}: {text}" if facet else text)
        elif str(item).strip():
            parts.append(str(item).strip())
    return " | ".join(parts)


def _upsert_breath_seed_diagnostic(
    seed_diagnostics: dict[str, dict],
    bucket: dict,
    source: str,
    *,
    bucket_search_score: float | None = None,
    embedding_score: float | None = None,
) -> None:
    bucket_id = str(bucket.get("id") or "").strip()
    if not bucket_id:
        return
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    item = seed_diagnostics.setdefault(
        bucket_id,
        {
            "bucket_id": bucket_id,
            "bucket_name": meta.get("name") or bucket_id,
            "sources": [],
        },
    )
    if source and source not in item["sources"]:
        item["sources"].append(source)
    if bucket_search_score is not None:
        safe_score = _safe_float(bucket_search_score)
        if safe_score is not None:
            item["bucket_search_score"] = safe_score
            item["keyword_score"] = round(_score_to_unit(safe_score), 4)
    if embedding_score is not None:
        safe_embedding_score = _safe_float(embedding_score)
        if safe_embedding_score is not None:
            item["embedding_score"] = safe_embedding_score


def _score_to_unit(score: float) -> float:
    try:
        number = float(score)
    except (TypeError, ValueError):
        return 0.0
    if number > 1:
        number = number / 100.0
    return max(0.0, min(1.0, number))


def _write_breath_recall_diagnostics(
    *,
    query: str,
    recall_thresholds: dict,
    seed_diagnostics: dict[str, dict],
    pre_gate_candidates: list[dict],
    gated_candidates: list[dict],
    reranked_candidates: list[dict],
    returned_moments: list[dict],
    suppressed_candidates: list[dict],
    displayed_moment_ids: list[str],
    secondary_moment_ids: list[str],
    related_source_bucket_ids: list[str],
    related_included: bool,
    drift_included: bool,
    dream_included: bool,
    response_sections: list[str],
) -> None:
    if not getattr(recall_diagnostics, "enabled", False):
        return

    gated_by_id = _moment_index(gated_candidates)
    reranked_by_id = _moment_index(reranked_candidates)
    suppressed_by_id = _moment_index(suppressed_candidates)
    gated_rank = _moment_rank(gated_candidates)
    reranked_rank = _moment_rank(reranked_candidates)
    returned_ids = [str(moment.get("moment_id") or "") for moment in returned_moments if moment.get("moment_id")]
    returned_set = set(returned_ids)
    displayed_set = set(displayed_moment_ids)
    secondary_set = set(secondary_moment_ids)
    options = _recall_relevance_options()
    max_candidates = max(1, int(getattr(recall_diagnostics, "max_candidates", 20) or 20))

    candidates = []
    for index, moment in enumerate(pre_gate_candidates[:max_candidates]):
        moment_id = str(moment.get("moment_id") or "")
        bucket_id = str(moment.get("bucket_id") or "")
        decision = relevance_decision(query, moment, options)
        gated = gated_by_id.get(moment_id)
        reranked = reranked_by_id.get(moment_id)
        final = reranked or gated
        seed = seed_diagnostics.get(bucket_id, {})
        admission = _breath_moment_admission_decision(query, final or moment, seed_diagnostics)
        candidate = {
            "pre_rank": index,
            "gate_rank": gated_rank.get(moment_id),
            "final_rank": reranked_rank.get(moment_id),
            "bucket_id": bucket_id,
            "bucket_name": _moment_bucket_title(moment),
            "moment_id": moment_id,
            "section": moment.get("section"),
            "sources": seed.get("sources", []),
            "bucket_search_score": seed.get("bucket_search_score"),
            "keyword_score": seed.get("keyword_score"),
            "embedding_score": seed.get("embedding_score"),
            "word_map_score": seed.get("word_map_score"),
            "word_map_terms": seed.get("word_map_terms", []),
            "word_map_neighbor_terms": seed.get("word_map_neighbor_terms", []),
            "score_before_gate": _safe_float(moment.get("score")),
            "score_after_gate": _safe_float(gated.get("score")) if gated else None,
            "rerank_score": _safe_float(final.get("rerank_score")) if final else None,
            "combined_score": _safe_float(final.get("combined_score")) if final else None,
            "intent_rank": _recall_rank(query, final or moment)[0],
            "gate": "filtered" if decision.multiplier <= 0 else "kept",
            "gate_multiplier": round(float(decision.multiplier), 4),
            "gate_reasons": list(decision.reasons),
            "admission": "suppressed" if moment_id in suppressed_by_id else "admitted" if admission.admit else "suppressed",
            "admission_reason": admission.reason,
            "selected_returned": moment_id in returned_set,
            "selected_direct": moment_id in displayed_set,
            "selected_secondary": moment_id in secondary_set,
            "annotation_summary": (moment.get("metadata") or {}).get("annotation_summary"),
            "annotation_facets": (moment.get("metadata") or {}).get("annotation_facets", {}),
            "evidence_spans": (moment.get("metadata") or {}).get("evidence_spans", []),
            "text_preview": _diagnostic_text_preview(moment),
        }
        candidates.append(candidate)

    recall_diagnostics.write(
        {
            "source": "breath",
            "mode": "search",
            "query": str(query or ""),
            "recall_thresholds": recall_thresholds,
            "seed_buckets": list(seed_diagnostics.values())[:max_candidates],
            "candidates": candidates,
            "suppressed_candidates": [
                {
                    "bucket_id": str(moment.get("bucket_id") or ""),
                    "bucket_name": _moment_bucket_title(moment),
                    "moment_id": str(moment.get("moment_id") or ""),
                    "section": moment.get("section"),
                    "admission_reason": str(moment.get("_admission_reason") or "suppressed"),
                    "score": _safe_float(moment.get("score")),
                    "rerank_score": _safe_float(moment.get("rerank_score")),
                    "word_map_score": _safe_float(moment.get("word_map_score")),
                    "word_map_terms": list(moment.get("word_map_terms") or []),
                    "word_map_neighbor_terms": list(moment.get("word_map_neighbor_terms") or []),
                    "text_preview": _diagnostic_text_preview(moment),
                }
                for moment in suppressed_candidates[:max_candidates]
            ],
            "final": {
                "returned_moment_ids": returned_ids,
                "direct_moment_ids": displayed_moment_ids,
                "secondary_moment_ids": secondary_moment_ids,
                "related_source_bucket_ids": related_source_bucket_ids,
                "related_included": related_included,
                "drift_included": drift_included,
                "dream_included": dream_included,
                "response_sections": response_sections,
            },
        }
    )


def _format_suppressed_recall_candidate(moment: dict, seed_diagnostics: dict[str, dict]) -> str:
    bucket_id = str(moment.get("bucket_id") or "")
    seed = seed_diagnostics.get(bucket_id, {})
    parts = [
        f"- [bucket_id:{bucket_id}] [moment_id:{moment.get('moment_id') or ''}]",
        f"reason={moment.get('_admission_reason') or 'suppressed'}",
    ]
    if seed.get("embedding_score") is not None:
        parts.append(f"semantic={seed.get('embedding_score')}")
    if seed.get("word_map_score") is not None:
        parts.append(f"word_map={seed.get('word_map_score')}")
    if moment.get("rerank_score") is not None:
        parts.append(f"rerank={moment.get('rerank_score')}")
    preview = _diagnostic_text_preview(moment)
    if preview:
        parts.append(f"preview={preview}")
    return " ".join(parts)


def _moment_index(moments: list[dict]) -> dict[str, dict]:
    return {
        str(moment.get("moment_id")): moment
        for moment in moments
        if moment.get("moment_id")
    }


def _moment_rank(moments: list[dict]) -> dict[str, int]:
    return {
        str(moment.get("moment_id")): index
        for index, moment in enumerate(moments)
        if moment.get("moment_id")
    }


def _safe_float(value) -> float | None:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _diagnostic_text_preview(moment: dict) -> str:
    max_chars = max(0, int(getattr(recall_diagnostics, "max_text_chars", 220) or 0))
    if max_chars <= 0:
        return ""
    return _moment_text(moment, max_chars)


def _breath_recall_thresholds(query: str, max_results: int) -> dict:
    threshold_cfg = config.get("recall_thresholds", {}) or {}
    options = _recall_relevance_options()
    query_active = active_facets(facets_for_text(query, options))
    has_explicit = _query_has_explicit_entity_marker(query)
    is_vague = _query_is_vague_recall(query)

    base_vector_min = _float_between(threshold_cfg.get("vector_min_score"), 0.50, 0.0, 1.0)
    profile = "default"
    vector_min = base_vector_min
    top_k = max(max_results, 20)

    if has_explicit:
        profile = "explicit"
        vector_min = _float_between(threshold_cfg.get("explicit_vector_min_score"), 0.55, 0.0, 1.0)
    elif is_vague:
        profile = "vague"
        vector_min = _float_between(threshold_cfg.get("vague_vector_min_score"), 0.40, 0.0, 1.0)
        top_k = max(top_k, _int_between(threshold_cfg.get("vague_top_k"), 50, 20, 100))
    elif query_active:
        profile = "facet"
        vector_min = _float_between(threshold_cfg.get("facet_vector_min_score"), 0.45, 0.0, 1.0)

    return {
        "profile": profile,
        "vector_min_score": vector_min,
        "semantic_top_k": top_k,
        "query_facets": sorted(query_active),
        "has_explicit_entity": has_explicit,
        "is_vague": is_vague,
    }


BREATH_LEXICAL_DROP_PREFIXES = ("今天", "昨天", "刚才", "刚刚", "这次", "现在", "今晚", "昨晚")
BREATH_LEXICAL_GENERIC_TERMS = {
    "哭", "哭了", "哭吗", "哭呢", "今天哭", "昨天哭", "刚才哭", "刚刚哭",
    "今天", "昨天", "刚才", "刚刚", "这次", "现在", "最近", "记忆", "回忆",
    "原因", "为什么", "知道", "记得", "想起", "想起来", "什么", "事情",
}


def _breath_lexical_match_terms(query: str, all_buckets: list[dict] | None = None) -> list[str]:
    text = str(query or "").strip()
    if not text:
        return []
    terms: list[str] = []
    emotion_plan = emotional_recall_plan(text, memory_relevance_options_from_config(config))
    emotion_terms = list(emotion_plan.strong_terms)
    terms.extend(emotion_terms)
    for term in _recall_policy().specific_query_terms(text):
        cleaned = str(term or "").strip()
        if not cleaned:
            continue
        collapsed = re.sub(r"\s+", "", cleaned)
        for prefix in BREATH_LEXICAL_DROP_PREFIXES:
            if collapsed.startswith(prefix) and len(collapsed) > len(prefix):
                collapsed = collapsed[len(prefix):]
                break
        key = collapsed.lower()
        if not key or key in BREATH_LEXICAL_GENERIC_TERMS:
            continue
        if re.fullmatch(r"[a-z0-9_.:-]+", key) and len(key) < 3 and not re.search(r"\d", key):
            continue
        terms.append(collapsed)
    output = []
    seen = set()
    for term in terms:
        key = str(term).lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(str(term))
    if all_buckets is not None and hasattr(bucket_mgr, "filter_specific_lexical_terms"):
        preserve_terms = {term for term in output if term in emotion_terms}
        output = bucket_mgr.filter_specific_lexical_terms(
            output,
            all_buckets,
            preserve_terms=preserve_terms,
        )
    return output[:5]


def _bucket_matches_breath_lexical_terms(bucket: dict, terms: list[str]) -> bool:
    if not terms:
        return False
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    comments = meta.get("comments", []) if isinstance(meta.get("comments", []), list) else []
    year_ring_text = " ".join(
        str(comment.get("content") or "")
        for comment in comments
        if isinstance(comment, dict)
    )
    haystack = " ".join(
        [
            str(meta.get("name") or bucket.get("id") or ""),
            " ".join(str(tag) for tag in meta.get("tags", []) or []),
            " ".join(str(item) for item in meta.get("domain", []) or []),
            bucket_content_for_recall(bucket),
            year_ring_text,
        ]
    ).lower()
    return any(str(term or "").strip().lower() in haystack for term in terms)


def _breath_bucket_score(
    bucket: dict,
    *,
    topic: float,
    emotion: float,
    time_s: float,
    importance: float,
) -> float:
    _ = emotion  # legacy diagnostic input; P1 excludes it from ranking
    w_topic = float(getattr(bucket_mgr, "w_topic", 4.0) or 4.0)
    w_time = float(getattr(bucket_mgr, "w_time", 1.5) or 1.5)
    w_importance = float(getattr(bucket_mgr, "w_importance", 1.0) or 1.0)
    raw_total = (
        topic * w_topic
        + time_s * w_time
        + importance * w_importance
    )
    weight_sum = w_topic + w_time + w_importance
    normalized = (raw_total / weight_sum) * 100 if weight_sum > 0 else 0
    if (bucket.get("metadata") or {}).get("resolved", False):
        normalized *= 0.3
    return normalized


def _append_breath_lexical_matches(
    *,
    query: str,
    matches: list[dict],
    all_buckets: list[dict],
    seed_diagnostics: dict[str, dict],
    q_valence: float | None = None,
    q_arousal: float | None = None,
) -> list[str]:
    terms = _breath_lexical_match_terms(query, all_buckets=all_buckets)
    if not terms:
        return []
    matched_ids = {str(bucket.get("id") or "") for bucket in matches if bucket.get("id")}
    for bucket in all_buckets:
        bucket_id = str(bucket.get("id") or "")
        if not bucket_id:
            continue
        if not _is_breath_recall_seed_bucket(bucket):
            continue
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        if not _bucket_matches_breath_lexical_terms(bucket, terms):
            continue
        topic = (
            bucket_mgr._calc_topic_score(query, bucket)
            if hasattr(bucket_mgr, "_calc_topic_score")
            else 1.0
        )
        emotion = 0.0
        time_s = (
            bucket_mgr._calc_time_score(meta)
            if hasattr(bucket_mgr, "_calc_time_score")
            else 0.5
        )
        importance = max(1, min(10, int(meta.get("importance", 5)))) / 10.0
        score = max(
            _breath_bucket_score(
                bucket,
                topic=topic,
                emotion=emotion,
                time_s=time_s,
                importance=importance,
            ),
            float(getattr(bucket_mgr, "fuzzy_threshold", 50) or 50) + 5,
        )
        bucket["score"] = round(score, 2)
        bucket["lexical_match"] = True
        bucket["lexical_terms"] = terms
        _upsert_breath_seed_diagnostic(
            seed_diagnostics,
            bucket,
            "lexical",
            bucket_search_score=bucket.get("score"),
        )
        seed_diagnostics[bucket_id]["lexical_terms"] = terms
        if bucket_id not in matched_ids:
            matches.append(bucket)
            matched_ids.add(bucket_id)
    matches.sort(key=lambda bucket: float(bucket.get("score", 0.0) or 0.0), reverse=True)
    return terms


def _query_has_explicit_entity_marker(query: str) -> bool:
    return query_has_explicit_entity_marker(query)


def _query_is_vague_recall(query: str) -> bool:
    text = str(query or "").strip().lower()
    if not text:
        return False
    vague_markers = (
        "最近",
        "有趣",
        "想起来",
        "回忆",
        "记忆",
        "什么事",
        "有什么",
        "随便",
        "random",
        "recent",
        "interesting",
        "anything",
        "memory",
        "memories",
    )
    if any(marker in text for marker in vague_markers):
        return True
    terms = re.findall(r"[A-Za-z0-9_./:-]+|[\u4e00-\u9fff]{1,4}", text)
    return len(terms) <= 2 and not _query_has_explicit_entity_marker(text)


def _query_wants_body_chain(query: str) -> bool:
    return _recall_query_plan(query).wants_body_chain


def _query_requires_direct_topic_evidence(query: str) -> bool:
    return _recall_query_plan(query).requires_topic_evidence


def _recall_rank(query: str, moment: dict) -> tuple[int, float]:
    return recall_rank(query, moment, _recall_relevance_options())


async def _build_recall_debug_payload(
    query: str,
    *,
    max_candidates: int = 20,
    max_results: int = 3,
    max_tokens: int = 800,
    direct_render_mode: str = "auto",
    domain: str = "",
    valence: float | None = None,
    arousal: float | None = None,
) -> dict:
    query = str(query or "").strip()
    if not query:
        return {"status": "error", "error": "query_required"}

    max_candidates = _int_between(max_candidates, 20, 1, 100)
    max_results = _int_between(max_results, 3, 1, 20)
    max_tokens = _int_between(max_tokens, 800, 1, 20000)
    direct_render_mode = _normalize_direct_render_mode(direct_render_mode)
    domain_filter = [d.strip() for d in str(domain or "").split(",") if d.strip()] or None
    q_valence = valence if isinstance(valence, (int, float)) and 0 <= valence <= 1 else None
    q_arousal = arousal if isinstance(arousal, (int, float)) and 0 <= arousal <= 1 else None
    search_query = recall_search_query(query, _recall_relevance_options())
    warnings: list[str] = []

    try:
        raw_matches = await bucket_mgr.search(
            search_query,
            limit=max(max_candidates, max_results, 20),
            domain_filter=domain_filter,
            query_valence=q_valence,
            query_arousal=q_arousal,
        )
    except Exception as e:
        return {"status": "error", "error": "search_failed", "message": str(e)}
    matches = await _ordinary_seed_buckets_with_fact_routes(raw_matches)

    seed_diagnostics: dict[str, dict] = {}
    for bucket in matches:
        _upsert_breath_seed_diagnostic(
            seed_diagnostics,
            bucket,
            "profile_fact_route" if bucket.get("profile_fact_route") else "keyword",
            bucket_search_score=bucket.get("score"),
        )

    recall_thresholds = _breath_recall_thresholds(query, max_results)
    matched_ids = {bucket["id"] for bucket in matches if bucket.get("id")}
    try:
        vector_results = await embedding_engine.search_similar(
            search_query,
            top_k=int(recall_thresholds["semantic_top_k"]),
        )
        for bucket_id, sim_score in vector_results:
            if bucket_id in seed_diagnostics:
                seed_diagnostics[bucket_id]["embedding_score"] = round(float(sim_score), 4)
            if bucket_id not in matched_ids and sim_score >= recall_thresholds["vector_min_score"]:
                bucket = await bucket_mgr.get(bucket_id)
                if bucket:
                    bucket["score"] = round(sim_score * 100, 2)
                    bucket["vector_match"] = True
                    routed_buckets = await _ordinary_seed_buckets_with_fact_routes([bucket])
                else:
                    routed_buckets = []
                for routed in routed_buckets:
                    routed_id = str(routed.get("id") or "")
                    if not routed_id or routed_id in matched_ids:
                        continue
                    _upsert_breath_seed_diagnostic(
                        seed_diagnostics,
                        routed,
                        "profile_fact_route" if routed.get("profile_fact_route") else "vector",
                        embedding_score=sim_score,
                    )
                    matches.append(routed)
                    matched_ids.add(routed_id)
    except Exception as e:
        warnings.append(f"vector_search_failed: {e}")

    explicit_lookup = _query_explicitly_requests_archive_memory(query)
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=explicit_lookup)
    except Exception as e:
        warnings.append(f"list_buckets_failed: {e}")
        all_buckets = matches

    lexical_terms = _append_breath_lexical_matches(
        query=query,
        matches=matches,
        all_buckets=all_buckets,
        seed_diagnostics=seed_diagnostics,
        q_valence=q_valence,
        q_arousal=q_arousal,
    )
    if lexical_terms:
        recall_thresholds["lexical_terms"] = lexical_terms

    active_moments, _, _ = await _refresh_moment_graph(all_buckets)
    bucket_boosts = seed_scores_for_buckets(matches)
    searched_candidates = memory_moment_store.search_moment_items(
        search_query,
        active_moments,
        limit=max(max_candidates, max_results, 20),
        bucket_boosts=bucket_boosts,
        exclude_sections=TASK_ONLY_MOMENT_SECTIONS,
    )
    direct_candidates = _direct_recallable_moments(searched_candidates, explicit_lookup=explicit_lookup)
    source_record_moments = _source_record_synthetic_moments_for_matches(matches, query)
    direct_candidates = _prepend_source_record_synthetic_moments(
        direct_candidates,
        source_record_moments,
    )
    pre_gate_candidates = direct_candidates[:max_candidates]
    source_record_by_id = {
        str(moment.get("moment_id") or ""): moment
        for moment in source_record_moments
        if moment.get("moment_id")
    }
    gated_non_source = _apply_recall_relevance_gate(
        query,
        [
            moment for moment in direct_candidates
            if str(moment.get("moment_id") or "") not in source_record_by_id
        ],
    )
    gated_candidates = _prepend_source_record_synthetic_moments(
        gated_non_source,
        source_record_moments,
    )
    reranked_candidates = await _rerank_breath_moment_candidates(query, gated_candidates)
    reranked_candidates = _prepend_source_record_synthetic_moments(
        reranked_candidates,
        source_record_moments,
    )

    admitted_moments = []
    suppressed_moments = []
    for moment in reranked_candidates:
        if _is_source_record_synthetic_moment(moment):
            item = dict(moment)
            meta = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
            item["_admission_reason"] = str(meta.get("source_record_direct_reason") or "source_record_direct")
            item["_admission_debug"] = {
                "source_record_direct_override": True,
                "fragment_seed": bool(meta.get("source_record_fragment_seed")),
            }
            admitted_moments.append(item)
            continue
        admission = _breath_moment_admission_decision(
            query,
            moment,
            seed_diagnostics,
        )
        item = dict(moment)
        item["_admission_reason"] = admission.reason
        item["_admission_debug"] = getattr(admission, "debug", {})
        if admission.admit:
            admitted_moments.append(item)
        else:
            suppressed_moments.append(item)

    gated_by_id = _moment_index(gated_candidates)
    reranked_by_id = _moment_index(reranked_candidates)
    suppressed_by_id = _moment_index(suppressed_moments)
    gated_rank = _moment_rank(gated_candidates)
    reranked_rank = _moment_rank(reranked_candidates)
    returned_ids = [
        str(moment.get("moment_id") or "")
        for moment in admitted_moments[:max_results]
        if moment.get("moment_id")
    ]
    returned_set = set(returned_ids)
    returned_moments = admitted_moments[:max_results]
    displayed_moment_ids: list[str] = []
    displayed_bucket_ids: set[str] = set()
    for moment in returned_moments:
        bucket_id = str(moment.get("bucket_id") or "")
        moment_id = str(moment.get("moment_id") or "")
        if not bucket_id or bucket_id in displayed_bucket_ids:
            continue
        displayed_bucket_ids.add(bucket_id)
        if moment_id:
            displayed_moment_ids.append(moment_id)
        break
    query_plan = _recall_query_plan(query)
    secondary_moments = _secondary_direct_moments(
        query,
        returned_moments,
        displayed_bucket_ids,
        query_plan.secondary_direct_limit(1),
        query_plan=query_plan,
        seed_diagnostics=seed_diagnostics,
    )
    secondary_ids = {
        str(moment.get("moment_id") or "")
        for moment in secondary_moments
        if moment.get("moment_id")
    }
    displayed_set = set(displayed_moment_ids)
    bucket_map = {
        str(bucket.get("id") or ""): bucket
        for bucket in all_buckets
        if bucket.get("id") and not is_self_anchor_bucket(bucket)
    }
    options = _recall_relevance_options()

    candidates = []
    for index, moment in enumerate(pre_gate_candidates):
        moment_id = str(moment.get("moment_id") or "")
        bucket_id = str(moment.get("bucket_id") or "")
        decision = relevance_decision(query, moment, options)
        gated = gated_by_id.get(moment_id)
        final = reranked_by_id.get(moment_id) or gated
        seed = seed_diagnostics.get(bucket_id, {})
        bucket = bucket_map.get(bucket_id)
        admission = _breath_moment_admission_decision(query, final or moment, seed_diagnostics)
        candidates.append(
            {
                "pre_rank": index,
                "gate_rank": gated_rank.get(moment_id),
                "final_rank": reranked_rank.get(moment_id),
                "bucket_id": bucket_id,
                "bucket_name": _moment_bucket_title(moment),
                "moment_id": moment_id,
                "section": moment.get("section"),
                "sources": seed.get("sources", []),
                "bucket_search_score": seed.get("bucket_search_score"),
                "keyword_score": seed.get("keyword_score"),
                "embedding_score": seed.get("embedding_score"),
                "score_before_gate": _safe_float(moment.get("score")),
                "score_after_gate": _safe_float(gated.get("score")) if gated else None,
                "rerank_score": _safe_float(final.get("rerank_score")) if final else None,
                "combined_score": _safe_float(final.get("combined_score")) if final else None,
                "intent_rank": _recall_rank(query, final or moment)[0],
                "gate": "filtered" if decision.multiplier <= 0 else "kept",
                "gate_multiplier": round(float(decision.multiplier), 4),
                "gate_reasons": list(decision.reasons),
                "admission": (
                    "suppressed"
                    if moment_id in suppressed_by_id
                    else "admitted"
                    if admission.admit
                    else "suppressed"
                ),
                "admission_reason": admission.reason,
                "admission_debug": getattr(admission, "debug", {}),
                "selected_returned": moment_id in returned_set,
                "selected_direct": moment_id in displayed_set,
                "selected_secondary": moment_id in secondary_ids,
                "direct_render": _direct_bucket_render_debug(
                    bucket,
                    final or moment,
                    max_tokens,
                    query_text=query,
                    direct_render_mode=direct_render_mode,
                ) if bucket else {},
                "layer_debug": moment_layer_debug(final or moment, explicit_lookup=explicit_lookup),
                "runtime_gate": _breath_moment_runtime_gate_payload(
                    final or moment,
                    explicit_lookup=explicit_lookup,
                ),
                "annotation_summary": (moment.get("metadata") or {}).get("annotation_summary"),
                "annotation_facets": (moment.get("metadata") or {}).get("annotation_facets", {}),
                "evidence_spans": (moment.get("metadata") or {}).get("evidence_spans", []),
                "text_preview": _diagnostic_text_preview(moment),
            }
        )

    return {
        "status": "ok",
        "query": query,
        "search_query": search_query,
        "recall_thresholds": {
            **recall_thresholds,
            "max_tokens": max_tokens,
            "direct_render_mode": direct_render_mode,
        },
        "seed_buckets": list(seed_diagnostics.values())[:max_candidates],
        "candidate_count": len(pre_gate_candidates),
        "admitted_count": len(admitted_moments),
        "suppressed_count": len(suppressed_moments),
        "returned_moment_ids": returned_ids,
        "candidates": candidates,
        "warnings": warnings,
    }


async def _build_breath_debug_rerank_payload(
    query: str,
    *,
    max_candidates: int = 12,
    max_results: int = 3,
    max_tokens: int = 400,
    direct_render_mode: str = "compact",
    valence: float | None = None,
    arousal: float | None = None,
) -> dict:
    enabled = bool(getattr(reranker_engine, "enabled", False))
    base = {
        "requested": True,
        "enabled": enabled,
        "applied": False,
        "candidate_limit": int(getattr(reranker_engine, "candidate_limit", 20) or 20),
        "score_weight": _safe_float(getattr(reranker_engine, "score_weight", 0.65)),
    }
    if not str(query or "").strip():
        return {
            **base,
            "status": "ok",
            "skip_reason": "query_required",
            "candidate_count": 0,
            "admitted_count": 0,
            "suppressed_count": 0,
            "returned_moment_ids": [],
            "candidates": [],
            "warnings": [],
        }
    payload = await _build_recall_debug_payload(
        query,
        max_candidates=max_candidates,
        max_results=max_results,
        max_tokens=max_tokens,
        direct_render_mode=direct_render_mode,
        valence=valence,
        arousal=arousal,
    )
    payload.update(base)
    candidates = payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
    payload["applied"] = any(candidate.get("rerank_score") is not None for candidate in candidates)
    if payload.get("status") != "ok":
        payload["skip_reason"] = payload.get("error") or "recall_debug_failed"
    elif not enabled:
        payload["skip_reason"] = "reranker_disabled"
    elif not candidates:
        payload["skip_reason"] = "no_candidates"
    elif not payload["applied"]:
        payload["skip_reason"] = "no_rerank_scores"
    return payload


def _secondary_direct_limit(query: str, related_per_memory: int) -> int:
    return _recall_query_plan(query).secondary_direct_limit(related_per_memory)


def _secondary_direct_moments(
    query: str,
    candidates: list[dict],
    displayed_bucket_ids: set[str],
    limit: int,
    *,
    query_plan=None,
    seed_diagnostics: dict[str, dict] | None = None,
) -> list[dict]:
    if limit <= 0:
        return []
    query_plan = query_plan or _recall_query_plan(query)
    seed_diagnostics = seed_diagnostics or {}
    hidden = []
    seen_buckets = set(displayed_bucket_ids)
    for moment in candidates:
        bucket_id = str(moment.get("bucket_id") or "")
        if not bucket_id or bucket_id in seen_buckets:
            continue
        if moment.get("section") in MOMENT_TEMPERATURE_SECTIONS:
            continue
        if should_suppress_context_candidate(query, moment, _recall_relevance_options()):
            continue
        has_topic_evidence = _moment_has_query_topic_evidence(query, moment)
        if query_plan.enforce_topic_evidence and not has_topic_evidence:
            continue
        if query_plan.secondary_direct_requires_topic_evidence and not has_topic_evidence:
            seed = seed_diagnostics.get(bucket_id, {})
            if seed.get("embedding_score") is None:
                continue
        hidden.append(moment)
        seen_buckets.add(bucket_id)
    if query_plan.wants_body_chain:
        hidden.sort(key=lambda moment: _recall_rank(query, moment))
    return hidden[:limit]


def _moment_has_query_topic_evidence(query: str, moment: dict) -> bool:
    return _recall_policy().moment_has_topic_evidence(query, moment)


def _bucket_has_query_topic_evidence(query: str, bucket: dict) -> bool:
    return _recall_policy().bucket_has_topic_evidence(query, bucket)


def _specific_query_terms(query: str) -> list[str]:
    return _recall_policy().specific_query_terms(query)


def _query_explicitly_requests_archive_memory(query: str) -> bool:
    return _recall_query_plan(query).explicit_old_memory


def _representative_moments_by_bucket(
    moments: list[dict],
    *,
    explicit_lookup: bool = False,
) -> dict[str, dict]:
    grouped = _moments_by_bucket(moments)
    representatives = {}
    for bucket_id, bucket_moments in grouped.items():
        representative = _related_representative_moment(bucket_moments, explicit_lookup=explicit_lookup)
        if representative:
            representatives[bucket_id] = representative
    return representatives


async def _refresh_moment_graph(all_buckets: list[dict] | None = None) -> tuple[list[dict], dict[str, list[dict]], list[dict]]:
    if all_buckets is None:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
    _prune_self_anchor_moment_index(all_buckets)
    recallable_buckets = [bucket for bucket in all_buckets if not _is_self_anchor_recall_excluded_bucket(bucket)]
    recallable_bucket_ids = {
        str(bucket.get("id") or "")
        for bucket in recallable_buckets
        if str(bucket.get("id") or "")
    }
    memory_moment_store.bulk_upsert(recallable_buckets)
    moments = [
        moment
        for moment in _recallable_moments(memory_moment_store.list_all())
        if str(moment.get("bucket_id") or "") in recallable_bucket_ids
    ]
    grouped = _moments_by_bucket(moments)
    moment_ids = {
        str(moment.get("moment_id") or "")
        for moment in moments
        if str(moment.get("moment_id") or "")
    }
    edges = [
        edge
        for edge in legacy_moment_edges_for_recall(memory_moment_store.list_edges())
        if str(edge.get("source") or "") in moment_ids
        and str(edge.get("target") or "") in moment_ids
    ]
    bucket_map = {
        str(bucket.get("id") or ""): bucket
        for bucket in recallable_buckets
        if str(bucket.get("id") or "")
    }
    edges.extend(_bucket_edges_as_moment_edges(_bucket_edges_for_recall(bucket_map), grouped))
    return moments, grouped, edges


async def _build_mcp_moment_diffused_memory_block(
    seed_moments: list[dict],
    moments: list[dict],
    edges: list[dict],
    token_budget: int,
    limit_per_source: int,
    min_confidence: float,
    exclude_bucket_ids: set[str] | None = None,
    query_text: str = "",
) -> str:
    if token_budget <= 0 or not seed_moments:
        return ""
    limit_per_source = _int_between(limit_per_source, 1, 0, 5)
    if limit_per_source <= 0:
        return ""
    min_confidence = _float_between(min_confidence, 0.55, 0.0, 1.0)
    edges = [
        edge for edge in edges
        if float(edge.get("confidence", 0.0)) >= min_confidence
    ]
    allow_archive_targets = _recall_query_plan(query_text).allow_archive_targets
    moment_map = _moment_diffusion_map(moments)
    representatives = _representative_moments_by_bucket(
        moments,
        explicit_lookup=allow_archive_targets,
    )
    exclude_bucket_ids = set(exclude_bucket_ids or set())
    diffusion_options = _breath_related_diffusion_options(len(seed_moments) * limit_per_source)
    hits = diffuse_memory(
        _seed_scores_for_moments(seed_moments),
        edges,
        moment_map,
        options=diffusion_options,
        exclude_ids={moment["moment_id"] for moment in seed_moments if moment.get("moment_id")},
        query_text=query_text,
    )

    parts = []
    seen = set()
    remaining = token_budget
    for hit in hits:
        moment = moment_map.get(hit.bucket_id)
        if not moment or hit.bucket_id in seen:
            continue
        bucket_id = str(moment.get("bucket_id") or "")
        if bucket_id in exclude_bucket_ids:
            continue
        if not can_moment_be_related_target(moment, explicit_lookup=allow_archive_targets):
            replacement = representatives.get(bucket_id)
            if not replacement:
                continue
            moment = replacement
            if moment.get("moment_id") in seen:
                continue
            if not can_moment_be_related_target(moment, explicit_lookup=allow_archive_targets):
                continue
        block = _format_related_moment(
            moment,
            path_has_caution(hit.best_path),
            path=hit.best_path,
            moment_map=moment_map,
            chain_bundle=diffusion_options.chain_walk_enabled,
        )
        block_tokens = count_tokens_approx(block)
        if block_tokens > remaining:
            break
        parts.append(block)
        seen.add(hit.bucket_id)
        remaining -= block_tokens
        if remaining <= 0:
            break
    return "\n---\n".join(parts)


async def _collect_diffusion_seed_buckets(query: str, max_seeds: int) -> tuple[list[dict], list[str]]:
    warnings = []
    matches: list[dict] = []
    matched_ids: set[str] = set()
    search_limit = max(max_seeds, 20)
    options = _recall_relevance_options()

    def add_candidate(bucket: dict, source: str, score: float | None = None) -> None:
        bucket_id = bucket.get("id")
        if not bucket_id or bucket_id in matched_ids:
            return
        decision = relevance_decision(query, bucket, options)
        if decision.suppress:
            return
        candidate = dict(bucket)
        if score is not None:
            candidate["score"] = score
        meta = candidate.get("metadata", {}) if isinstance(candidate.get("metadata"), dict) else {}
        base_score = _safe_float(candidate.get("score"))
        if base_score is None:
            base_score = _safe_float(meta.get("score"))
        if base_score is None:
            base_score = _safe_float(meta.get("importance"))
        if base_score is None:
            base_score = 0.0
        candidate["score"] = round(base_score * float(decision.multiplier), 4)
        candidate["_inspect_source"] = source
        candidate["_inspect_relevance_reasons"] = list(decision.reasons)
        candidate["_inspect_relevance_multiplier"] = round(float(decision.multiplier), 4)
        matches.append(candidate)
        matched_ids.add(bucket_id)

    try:
        for bucket in await bucket_mgr.search(query, limit=search_limit):
            if not bucket or bucket.get("metadata", {}).get("type") == "feel":
                continue
            add_candidate(bucket, "keyword")
    except Exception as e:
        logger.warning(f"Inspect diffusion keyword search failed / 扩散诊断关键词检索失败: {e}")
        warnings.append(f"keyword_search_failed: {e}")

    try:
        vector_results = await embedding_engine.search_similar(query, top_k=search_limit)
        for bucket_id, sim_score in vector_results:
            if bucket_id in matched_ids:
                continue
            if sim_score <= 0.5:
                continue
            bucket = await bucket_mgr.get(bucket_id)
            if not bucket or bucket.get("metadata", {}).get("type") == "feel":
                continue
            add_candidate(bucket, "vector", round(sim_score * 100, 2))
            if matches and str(matches[-1].get("id") or "") == str(bucket_id):
                matches[-1]["vector_match"] = True
    except Exception as e:
        logger.warning(f"Inspect diffusion vector search failed / 扩散诊断向量检索失败: {e}")
        warnings.append(f"vector_search_failed: {e}")

    matches.sort(
        key=lambda bucket: (
            recall_rank(query, bucket, options)[0],
            -(_safe_float(bucket.get("score")) or 0.0),
        )
    )
    return matches[:max_seeds], warnings


def _inspect_bucket_label(bucket: dict | None, bucket_id: str) -> str:
    if not bucket:
        return bucket_id
    meta = bucket.get("metadata", {}) or {}
    return str(meta.get("name") or bucket.get("name") or bucket_id)


def _inspect_bucket_layer_payload(bucket: dict | None, *, explicit_lookup: bool = False) -> dict:
    return bucket_layer_debug(bucket, explicit_lookup=explicit_lookup)


def _inspect_bucket_runtime_gate_payload(
    bucket: dict | None,
    *,
    explicit_lookup: bool = False,
    query: str = "",
) -> dict:
    gate = bucket_runtime_gate_debug(bucket, explicit_lookup=explicit_lookup)
    query_plan = _recall_query_plan(query)
    topic_required = bool(query_plan.enforce_topic_evidence)
    has_topic_evidence = (
        _bucket_has_query_topic_evidence(query, bucket)
        if topic_required and isinstance(bucket, dict)
        else False
    )
    related_allowed = bool(gate["related_target"]["allowed"])
    related_reason = str(gate["related_target"]["reason"])
    if related_allowed and topic_required and not has_topic_evidence:
        related_allowed = False
        related_reason = "query_topic_evidence_missing"
    gate["topic_evidence"] = {
        "required": topic_required,
        "present": has_topic_evidence if topic_required else None,
    }
    gate["related_injection"] = {
        "allowed": related_allowed,
        "reason": related_reason,
    }
    gate["would_inject_related"] = related_allowed
    return gate


def _inspect_path_payload(path, bucket_map: dict[str, dict]) -> dict:
    return {
        "score": round(float(path.score), 4),
        "trace": format_diffusion_trace(path, bucket_map, use_labels=True),
        "nodes": list(path.nodes),
        "steps": [
            {
                "source": step.source,
                "target": step.target,
                "direction": step.direction,
                "relation_type": step.relation_type,
                "confidence": step.confidence,
                "reason": step.reason,
            }
            for step in path.steps
        ],
    }


async def inspect_diffusion(
    query: str,
    max_seeds: int = 3,
    max_hits: int = 5,
    edge_min_confidence: float = 0.55,
) -> dict:
    """只读诊断 query 如何沿 memory_edges 点亮联想记忆；不 touch bucket，不创建记忆。"""
    query = str(query or "").strip()
    if not query:
        return {"status": "error", "error": "query_required"}

    max_seeds = _int_between(max_seeds, 3, 1, 20)
    max_hits = _int_between(max_hits, 5, 0, 20)
    edge_min_confidence = _float_between(edge_min_confidence, 0.55, 0.0, 1.0)
    node_facets_enabled = _node_facets_enabled(config)
    query_facets = {}
    if node_facets_enabled:
        try:
            query_facets = memory_node_store.facets_for_text(query)
        except Exception as e:
            logger.warning(f"Inspect diffusion query facets failed / 扩散诊断 query facets 失败: {e}")

    seed_buckets, warnings = await _collect_diffusion_seed_buckets(query, max_seeds)
    if not seed_buckets:
        return {
            "status": "ok",
            "query": query,
            "node_facets_enabled": node_facets_enabled,
            "query_facets": query_facets,
            "seeds": [],
            "hits": [],
            "warnings": warnings,
        }

    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
    except Exception as e:
        logger.warning(f"Inspect diffusion list buckets failed / 扩散诊断列桶失败: {e}")
        all_buckets = []
        warnings.append(f"list_buckets_failed: {e}")

    bucket_map = {
        bucket["id"]: bucket
        for bucket in all_buckets
        if bucket.get("id") and not is_self_anchor_bucket(bucket)
    }
    for seed in seed_buckets:
        if is_self_anchor_bucket(seed):
            continue
        if seed.get("id"):
            bucket_map.setdefault(seed["id"], seed)

    node_salience = None
    node_resonance = None
    if node_facets_enabled:
        try:
            memory_node_store.bulk_upsert(list(bucket_map.values()))
            node_salience = _node_salience_lookup
            node_resonance = _node_resonance_lookup(query_facets)
        except Exception as e:
            logger.warning(f"Inspect diffusion node refresh failed / 扩散诊断节点刷新失败: {e}")
            warnings.append(f"node_refresh_failed: {e}")
            node_salience = None
            node_resonance = None

    edges = [
        edge
        for edge in _bucket_edges_for_recall(bucket_map)
        if float(edge.get("confidence", 0.0)) >= edge_min_confidence
    ]
    options = replace(diffusion_options_from_config(config), top_k=max_hits)
    seed_scores = seed_scores_for_buckets(seed_buckets)
    hits = diffuse_memory(
        seed_scores,
        edges,
        bucket_map,
        options=options,
        exclude_ids={bucket["id"] for bucket in seed_buckets if bucket.get("id")},
        node_salience=node_salience,
        node_resonance=node_resonance,
        query_text=query,
    )

    def node_values(bucket_id: str, bucket: dict | None) -> dict:
        if not bucket or not node_facets_enabled:
            return {"salience": None, "resonance": None, "facets": {}}
        try:
            node = memory_node_store.get(bucket_id) or memory_node_store.upsert_bucket(bucket)
            salience = memory_node_store.node_salience(bucket_id, bucket)
            resonance = (
                node_resonance(bucket_id, bucket)
                if node_resonance
                else 1.0
            )
            return {
                "salience": round(float(salience), 4),
                "resonance": round(float(resonance), 4),
                "facets": node.get("facets", {}),
            }
        except Exception as e:
            return {"salience": None, "resonance": None, "facets": {}, "error": str(e)}

    seed_payload = []
    explicit_lookup = _query_explicitly_requests_archive_memory(query)
    for bucket in seed_buckets:
        bucket_id = bucket.get("id", "")
        values = node_values(bucket_id, bucket)
        seed_payload.append(
            {
                "bucket_id": bucket_id,
                "name": _inspect_bucket_label(bucket, bucket_id),
                "source": bucket.get("_inspect_source", "keyword"),
                "seed_score": round(float(seed_scores.get(bucket_id, 0.0)), 4),
                "layer_debug": _inspect_bucket_layer_payload(bucket, explicit_lookup=explicit_lookup),
                "runtime_gate": _inspect_bucket_runtime_gate_payload(
                    bucket,
                    explicit_lookup=explicit_lookup,
                    query=query,
                ),
                **values,
            }
        )

    hit_payload = []
    for hit in hits:
        bucket = bucket_map.get(hit.bucket_id)
        values = node_values(hit.bucket_id, bucket)
        hit_payload.append(
            {
                "bucket_id": hit.bucket_id,
                "name": _inspect_bucket_label(bucket, hit.bucket_id),
                "score": hit.activation,
                "layer_debug": _inspect_bucket_layer_payload(bucket, explicit_lookup=explicit_lookup),
                "runtime_gate": _inspect_bucket_runtime_gate_payload(
                    bucket,
                    explicit_lookup=explicit_lookup,
                    query=query,
                ),
                **values,
                "path": format_diffusion_trace(hit.best_path, bucket_map, use_labels=True),
                "path_ids": list(hit.best_path.nodes),
                "caution": path_has_caution(hit.best_path),
                "paths": [_inspect_path_payload(path, bucket_map) for path in hit.paths],
            }
        )

    return {
        "status": "ok",
        "query": query,
        "node_facets_enabled": node_facets_enabled,
        "options": {
            "max_hops": options.max_hops,
            "top_k": options.top_k,
            "min_activation": options.min_activation,
            "edge_min_confidence": edge_min_confidence,
            "include_incoming": options.include_incoming,
        },
        "query_facets": query_facets,
        "seeds": seed_payload,
        "hits": hit_payload,
        "warnings": warnings,
    }


def _inspect_moment_payload(
    moment: dict,
    *,
    include_text: bool,
    include_source_window: bool = False,
) -> dict:
    text = str(moment.get("text") or "")
    payload = {
        "moment_id": moment.get("moment_id"),
        "bucket_id": moment.get("bucket_id"),
        "section": moment.get("section"),
        "ordinal": moment.get("ordinal"),
        "source": moment.get("source"),
        "source_id": moment.get("source_id"),
        "text_hash": moment.get("text_hash"),
        "text_length": len(text),
        "layer_debug": moment_layer_debug(moment),
        "runtime_gate": moment_runtime_gate_debug(moment),
        "metadata": moment.get("metadata", {}),
        "created_at": moment.get("created_at"),
        "updated_at": moment.get("updated_at"),
    }
    if include_text:
        payload["text"] = text
    else:
        payload["text_preview"] = _clip_text(" ".join(text.split()), 240)
    if include_source_window:
        source_window = source_ref_window(
            moment,
            allowed_root=str(config.get("buckets_dir") or ""),
            max_chars=760,
            context_lines=0,
        )
        if source_window:
            payload["source_window"] = source_window
    return payload


async def inspect_moments(bucket_id: str = "", limit: int = 20) -> dict:
    """只读诊断 bucket 如何被拆成 moment；写入/刷新 SQLite 索引，不 touch bucket。"""
    bucket_id = str(bucket_id or "").strip()
    limit = _int_between(limit, 20, 1, 200)

    if bucket_id:
        if not MEMORY_ID_RE.fullmatch(bucket_id):
            return {"status": "error", "error": "invalid bucket_id"}
        bucket = await bucket_mgr.get(bucket_id)
        if not bucket:
            return {"status": "error", "error": "not_found", "bucket_id": bucket_id}
        moments = memory_moment_store.upsert_bucket(bucket)
        edges = memory_moment_store.list_edges(bucket_id)
        meta = bucket.get("metadata", {}) or {}
        return {
            "status": "ok",
            "mode": "bucket",
            "bucket_id": bucket_id,
            "name": str(meta.get("name") or bucket.get("name") or bucket_id),
            "bucket_layer_debug": _inspect_bucket_layer_payload(bucket),
            "count": len(moments),
            "edge_count": len(edges),
            "db_path": memory_moment_store.db_path,
            "moments": [
                _inspect_moment_payload(
                    moment,
                    include_text=True,
                    include_source_window=True,
                )
                for moment in moments[:limit]
            ],
            "edges": edges[:limit],
        }

    buckets = await bucket_mgr.list_all(include_archive=False)
    indexed = memory_moment_store.bulk_upsert(buckets)
    stats = memory_moment_store.stats()
    sample = memory_moment_store.sample(limit)
    return {
        "status": "ok",
        "mode": "bulk",
        "indexed_buckets": indexed["buckets"],
        "indexed_moments": indexed["moments"],
        "total_buckets": stats["buckets"],
        "total_moments": stats["moments"],
        "total_edges": stats.get("edges", 0),
        "db_path": memory_moment_store.db_path,
        "sample": [
            _inspect_moment_payload(moment, include_text=False)
            for moment in sample
        ],
    }


def _node_facets_enabled(cfg: dict | None) -> bool:
    node_cfg = (cfg or {}).get("node_facets", {}) or {}
    if isinstance(node_cfg, dict):
        return _bool_value(node_cfg.get("enabled", True), True)
    return True


def _node_salience_lookup(bucket_id: str, bucket: dict) -> float:
    return memory_node_store.node_salience(bucket_id, bucket)


def _node_resonance_lookup(query_facets: dict):
    if not _has_active_facets(query_facets):
        return None

    def lookup(bucket_id: str, bucket: dict) -> float:
        return memory_node_store.node_resonance(bucket_id, query_facets, bucket)

    return lookup


def _has_active_facets(facets: dict | None) -> bool:
    for value in (facets or {}).values():
        if isinstance(value, dict):
            if any(float(item or 0) > 0 for item in value.values()):
                return True
        else:
            try:
                if float(value) > 0:
                    return True
            except (TypeError, ValueError):
                continue
    return False


# =============================================================
# Tool 0.8: reminders — standalone care memos
# 工具 0.8：reminders — 独立照顾备忘
# =============================================================
def _reminder_public_payload(item: dict | None) -> dict:
    if not item:
        return {}
    keys = [
        "id",
        "title",
        "content",
        "status",
        "source",
        "channel",
        "session_id",
        "start_at",
        "end_at",
        "next_due_at",
        "repeat_rule",
        "interval_rounds",
        "cooldown_minutes",
        "daily_limit",
        "daily_reminder_date",
        "daily_reminder_count",
        "max_injections",
        "last_reminded_at",
        "last_reminded_round",
        "reminder_count",
        "created_at",
        "updated_at",
        "resolved_at",
    ]
    return {key: item.get(key) for key in keys}


async def reminder_create(
    title: str,
    content: str,
    next_due_at: str = "",
    start_at: str = "",
    end_at: str = "",
    repeat_rule: str = "every_n_rounds",
    interval_rounds: int = 6,
    cooldown_minutes: int = 0,
    daily_limit: int = -1,
    max_injections: int = 0,
    channel: str = "global",
    session_id: str = "",
) -> dict:
    """创建独立照顾备忘；不写记忆桶，不触发 embedding。可设 start_at/end_at 和 daily_limit 控制每天出现次数；morning_evening 未指定时默认每天 2 次。"""
    try:
        item = reminder_store.create(
            title=title,
            content=content,
            next_due_at=next_due_at,
            start_at=start_at,
            end_at=end_at,
            repeat_rule=repeat_rule,
            interval_rounds=interval_rounds,
            cooldown_minutes=cooldown_minutes,
            daily_limit=daily_limit if daily_limit >= 0 else None,
            max_injections=max_injections,
            channel=channel,
            session_id=session_id,
            source="mcp",
        )
    except ValueError as exc:
        return {"error": str(exc)}
    return {"status": "created", "reminder": _reminder_public_payload(item)}


async def reminder_list(status: str = "active", limit: int = 20) -> dict:
    """列出独立照顾备忘；status 可用 active/done/archived/all。"""
    try:
        items = reminder_store.list(status=status, limit=_int_between(limit, 20, 1, 100))
    except ValueError as exc:
        return {"error": str(exc), "reminders": []}
    return {"count": len(items), "reminders": [_reminder_public_payload(item) for item in items]}


async def reminder_update(
    reminder_id: str,
    status: str = "",
    snooze_minutes: int = 0,
    next_due_at: str = "",
    title: str = "",
    content: str = "",
    daily_limit: int = -1,
    max_injections: int = -1,
) -> dict:
    """更新独立照顾备忘；完成用 status="done"，稍后用 snooze_minutes。"""
    reminder_id = _coerce_memory_id(reminder_id)
    if not reminder_id:
        return {"error": "missing reminder_id"}
    try:
        if snooze_minutes:
            item = reminder_store.snooze(reminder_id, minutes=_int_between(snooze_minutes, 60, 1, 525600))
        else:
            item = reminder_store.update(
                reminder_id,
                status=status or None,
                next_due_at=next_due_at if next_due_at != "" else None,
                title=title if title != "" else None,
                content=content if content != "" else None,
                daily_limit=daily_limit if daily_limit >= 0 else None,
                max_injections=max_injections if max_injections >= 0 else None,
            )
    except ValueError as exc:
        return {"error": str(exc)}
    if not item:
        return {"error": "not found", "id": reminder_id}
    return {"status": "updated", "reminder": _reminder_public_payload(item)}


# =============================================================
# Tool 1: breath — Breathe
# 工具 1：breath — 呼吸
#
# No args: surface highest-weight unresolved memories (active push)
# 无参数：浮现权重最高的未解决记忆
# With args: search by keyword/vector semantics. Emotion coordinates are legacy-only.
# 有参数：按关键词/向量语义检索；情绪坐标只保留旧接口兼容。
# =============================================================
async def _recall_memory(
    query: str = "",
    max_tokens: int = 10000,
    domain: str = "",
    date: str = "",
    valence: float = -1,
    arousal: float = -1,
    max_results: int = 20,
    include_related: bool = True,
    related_per_memory: int = 1,
    edge_min_confidence: float = 0.55,
    include_core: bool = True,
    core_limit: int = 3,
    is_session_start: bool = False,
    debug: bool = False,
    surface: str = "manual",
    direct_render_mode: str = "auto",
    retrieval_mode: str = "graph",
    mode: str = "",
    session_id: str = "",
    previous_session_id: str = "",
    parent_shadow_id: str = "",
) -> str:
    """只读检索记忆。查主题用 query；valence/arousal 仅兼容旧调用且不参与普通排序。新窗口轻交接用 mode="handoff"：session_id 若来自 Gateway 的开窗登记，会精确读取它的 parent Shadow；也可显式传 parent_shadow_id。连续性只走一个槽位：优先逐字注入上一窗手写 handoff_note；缺影时读取 72 小时内的 Recent Continuity；两者都不可用时，previous_session_id 可指定从哪一窗的 raw_events 原文尾部生成确定性救生包。直接父窗影若带 continue_scene_id，且新窗口第一句 query 只是“继续吧 / 接着来 / 然后呢”等裸续接词，会额外逐字读取这一条 Scene 全文；带具体对象的“继续做……”不触发，也不进行图扩散。整个 fallback 不调用模型总结。date 或 query 里的日期可查当天普通记忆；domain="feel"/"whisper" 读私密通道，domain="daily_impression" 才读日印象。日期支持 2026-06-15、2026.06.15、2026年6月15日、25年6月15日、6月15日。"""
    await decay_engine.ensure_started()
    max_results = _int_between(max_results, 20, 1, 50)
    max_tokens = _int_between(max_tokens, 10000, 0, 20000)
    include_related = _bool_value(include_related, True)
    related_per_memory = _int_between(related_per_memory, 1, 0, 5)
    edge_min_confidence = _float_between(edge_min_confidence, 0.55, 0.0, 1.0)
    include_core = _bool_value(include_core, True)
    core_limit = _int_between(core_limit, 3, 0, 20)
    is_session_start = _bool_value(is_session_start, False)
    debug = _bool_value(debug, False)
    surface_key = str(surface or "manual").strip().lower()
    auto_surface = surface_key in {"auto", "automatic", "bridge", "gateway"}
    direct_render_mode = _normalize_direct_render_mode(direct_render_mode)
    retrieval_mode = _normalize_retrieval_mode(retrieval_mode)
    mode_key = _normalize_breath_mode(mode)
    domain_key = domain.strip().lower()
    raw_date = str(date or "").strip()
    date_hint = parse_human_date_reference(raw_date or query)
    if raw_date and not date_hint:
        return '日期格式没看懂。可以用 date="2026-06-15"、date="2026.06.15"、date="2026年6月15日"、date="25年6月15日" 或 date="6月15日"。'
    date_key = ""
    date_label = ""
    if date_hint and (raw_date or _breath_query_requests_date_read(query)):
        date_key = date_hint["date"]
        date_label = date_hint.get("label", date_key)

    if not mode_key and is_session_start and not str(query or "").strip() and not domain_key:
        mode_key = "handoff"

    if mode_key == "handoff":
        return await _build_handoff_breath(
            max_tokens=min(max_tokens or 1200, 1600),
            session_id=session_id,
            previous_session_id=previous_session_id,
            parent_shadow_id=parent_shadow_id,
            query=query,
            debug=debug,
        )

    if _is_self_anchor_domain(domain_key):
        return await _read_self_anchor_domain_breath(
            query=query,
            max_tokens=max_tokens,
            limit=max_results,
        )

    if _is_pending_followup_domain(domain_key) or (not domain_key and _breath_query_requests_pending_followups(query)):
        return "旧 followup/todo 派生待办已停用；请使用 reminder_list 或 /api/reminders 查看独立照顾备忘。"

    # --- Feel/whisper retrieval: independent read-only channels ---
    # --- Feel/whisper 检索：独立只读入口 ---
    if domain_key in {"feel", "whisper", "daily_impression"}:
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
            feels = [b for b in all_buckets if b["metadata"].get("type") == "feel"]
            if domain_key == "daily_impression":
                feels = [b for b in feels if _is_daily_impression_feel_bucket(b)]
            else:
                feels = [b for b in feels if not _is_daily_impression_feel_bucket(b)]
            if domain_key == "whisper":
                feels = [
                    b for b in feels
                    if "whisper" in {str(tag).lower() for tag in b["metadata"].get("tags", []) or []}
                ]
            if date_key:
                feels = [b for b in feels if _bucket_matches_breath_date(b, date_key)]
            feels.sort(key=lambda b: b["metadata"].get("created", ""), reverse=True)
            if not feels:
                if date_key:
                    return f"{date_key} 没有找到 {domain_key}。"
                if domain_key == "whisper":
                    return "没有留下过 whisper。"
                if domain_key == "daily_impression":
                    return "没有留下过 daily_impression。"
                return "没有留下过 feel。"
            results = []
            for f in feels:
                meta = f["metadata"]
                created = meta.get("date") or meta.get("created", "")
                entry = f"[{created}] [bucket_id:{f['id']}]\n{strip_wikilinks(f['content'])}"
                results.append(entry)
                if count_tokens_approx("\n---\n".join(results)) > max_tokens:
                    break
            title = "whisper" if domain_key == "whisper" else ("daily_impression" if domain_key == "daily_impression" else "feel")
            return f"=== 你留下的 {title} ===\n" + "\n---\n".join(results)
        except Exception as e:
            logger.error(f"Feel retrieval failed: {e}")
            if domain_key == "whisper":
                return "读取 whisper 失败。"
            if domain_key == "daily_impression":
                return "读取 daily_impression 失败。"
            return "读取 feel 失败。"

    if _is_self_anchor_tag_read_request(query):
        return await _read_self_anchor_tag_breath(max_tokens=max_tokens, limit=max_results)

    if date_key:
        domain_filter = [d.strip() for d in domain.split(",") if d.strip()] or None
        return await _read_breath_date(
            date_key=date_key,
            label=date_label,
            query=query,
            max_tokens=max_tokens,
            max_results=max_results,
            domain_filter=domain_filter,
        )

    # --- No args or empty query: surfacing mode (weight pool active push) ---
    # --- 无参数或空query：浮现模式（权重池主动推送）---
    if not query or not query.strip():
        if auto_surface:
            return "没有找到可靠命中。"
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
        except Exception as e:
            logger.error(f"Failed to list buckets for surfacing / 浮现列桶失败: {e}")
            return "记忆系统暂时无法访问。"

        # --- Core buckets: protected first, pinned limited by core_limit ---
        # --- 核心桶：protected 优先，pinned 按 core_limit 限流 ---
        core_candidates = [
            b for b in all_buckets
            if not is_self_anchor_bucket(b)
            and not _is_profile_fact_bucket(b)
            and (b["metadata"].get("pinned") or b["metadata"].get("protected"))
        ]
        protected = [
            b for b in core_candidates
            if b["metadata"].get("protected")
        ]
        pinned = [
            b for b in core_candidates
            if b["metadata"].get("pinned") and not b["metadata"].get("protected")
        ]
        protected.sort(
            key=lambda b: decay_engine.calculate_score(b["metadata"]),
            reverse=True,
        )
        pinned.sort(
            key=lambda b: (
                int(b["metadata"].get("importance", 5)),
                decay_engine.calculate_score(b["metadata"]),
                b["metadata"].get("updated_at") or b["metadata"].get("created", ""),
            ),
            reverse=True,
        )
        selected_core = (protected + pinned)[:core_limit] if include_core else []
        selected_anchors = _select_anchor_buckets(all_buckets, limit=min(2, max_results))

        # --- Unresolved buckets: surface top N by weight ---
        # --- 未解决桶：按权重浮现前 N 条 ---
        unresolved = [
            b for b in all_buckets
            if not is_self_anchor_bucket(b)
            and not b["metadata"].get("resolved", False)
            and b["metadata"].get("type") not in ("permanent", "feel")
            and not b["metadata"].get("anchor", False)
            and not b["metadata"].get("pinned", False)
            and not b["metadata"].get("protected", False)
        ]

        logger.info(
            f"Breath surfacing: {len(all_buckets)} total, "
            f"{len(core_candidates)} core, {len(selected_anchors)} anchors, {len(unresolved)} unresolved"
        )

        scored = sorted(
            unresolved,
            key=lambda b: decay_engine.calculate_score(b["metadata"]),
            reverse=True,
        )

        if scored:
            top_scores = [(b["metadata"].get("name", b["id"]), decay_engine.calculate_score(b["metadata"])) for b in scored[:5]]
            logger.info(f"Top unresolved scores: {top_scores}")

        # --- Token-budgeted surfacing with diversity + hard cap ---
        # --- 按 token 预算浮现，带多样性 + 硬上限 ---
        # Top-1 always surfaces; rest sampled from top-20 for diversity
        token_budget = max_tokens
        core_results = []
        core_token_budget = min(token_budget, max(0, int(max_tokens * 0.25)))
        for b in selected_core:
            if core_token_budget <= 0 or token_budget <= 0:
                break
            try:
                clean_meta = {k: v for k, v in b["metadata"].items() if k != "tags"}
                summary = await dehydrator.dehydrate(_bucket_text_for_embedding(b), clean_meta)
                entry = f"📌 [核心准则] [bucket_id:{b['id']}] {summary}"
                entry_tokens = count_tokens_approx(entry)
                if entry_tokens > core_token_budget or entry_tokens > token_budget:
                    break
                core_results.append(entry)
                core_token_budget -= entry_tokens
                token_budget -= entry_tokens
            except Exception as e:
                logger.warning(f"Failed to dehydrate core bucket / 核心桶脱水失败: {e}")
                continue

        anchor_results = []
        anchor_buckets = []
        anchor_token_budget = min(token_budget, max(0, int(max_tokens * 0.18)))
        for b in selected_anchors:
            if anchor_token_budget <= 0 or token_budget <= 0:
                break
            try:
                clean_meta = {k: v for k, v in b["metadata"].items() if k != "tags"}
                summary = await dehydrator.dehydrate(_bucket_text_for_embedding(b), clean_meta)
                entry = f"⚓ [长期锚点] [bucket_id:{b['id']}] {summary}"
                entry_tokens = count_tokens_approx(entry)
                if entry_tokens > anchor_token_budget or entry_tokens > token_budget:
                    break
                anchor_results.append(entry)
                anchor_buckets.append(b)
                anchor_token_budget -= entry_tokens
                token_budget -= entry_tokens
            except Exception as e:
                logger.warning(f"Failed to dehydrate anchor bucket / anchor 桶脱水失败: {e}")
                continue

        candidates = list(scored)
        if len(candidates) > 1:
            # Ensure highest-score bucket is first, shuffle rest from top-20
            top1 = [candidates[0]]
            pool = candidates[1:min(20, len(candidates))]
            random.shuffle(pool)
            candidates = top1 + pool + candidates[min(20, len(candidates)):]
        # Hard cap: never surface more than max_results buckets
        candidates = candidates[:max_results]

        dynamic_results = []
        surfaced_buckets = []
        for b in candidates:
            if token_budget <= 0:
                break
            try:
                clean_meta = {k: v for k, v in b["metadata"].items() if k != "tags"}
                summary = await dehydrator.dehydrate(_bucket_text_for_embedding(b), clean_meta)
                score = decay_engine.calculate_score(b["metadata"])
                entry = f"[权重:{score:.2f}] [bucket_id:{b['id']}] {summary}"
                entry_tokens = count_tokens_approx(entry)
                if entry_tokens > token_budget:
                    break
                # NOTE: no touch() here — surfacing should NOT reset decay timer
                dynamic_results.append(entry)
                surfaced_buckets.append(b)
                token_budget -= entry_tokens
            except Exception as e:
                logger.warning(f"Failed to dehydrate surfaced bucket / 浮现脱水失败: {e}")
                continue

        related_block = ""
        related_sources = anchor_buckets + surfaced_buckets
        if include_related and related_sources:
            related_header_tokens = count_tokens_approx("=== 联想浮现 ===\n")
            related_block = await _build_mcp_diffused_memory_block(
                related_sources,
                all_buckets,
                max(0, token_budget - related_header_tokens),
                related_per_memory,
                edge_min_confidence,
                "",
            )

        parts = []
        if core_results:
            parts.append("=== 核心准则 ===\n" + "\n---\n".join(core_results))
        if anchor_results:
            parts.append("=== 长期锚点 ===\n" + "\n---\n".join(anchor_results))
        if dynamic_results:
            parts.append("=== 浮现记忆 ===\n" + "\n---\n".join(dynamic_results))
        if related_block:
            parts.append("=== 联想浮现 ===\n" + related_block)

        dream_block = await dream_engine.surface_for_breath(
            query="",
            valence=valence,
            arousal=arousal,
            is_session_start=is_session_start,
            embedding_engine=embedding_engine,
        )
        if dream_block:
            parts.append(dream_block)

        if not parts:
            return "权重池平静，没有需要处理的记忆。"
        return "\n\n".join(parts)

    # --- With args: search mode (keyword + vector dual channel) ---
    # --- 有参数：检索模式（关键词 + 向量双通道）---
    domain_filter = [d.strip() for d in domain.split(",") if d.strip()] or None
    q_valence = valence if 0 <= valence <= 1 else None
    q_arousal = arousal if 0 <= arousal <= 1 else None
    if auto_surface and _recall_policy().is_auto_query_too_vague(query):
        return "没有找到可靠命中。"
    search_query = recall_search_query(query, _recall_relevance_options())

    try:
        raw_matches = await bucket_mgr.search(
            search_query,
            limit=max(max_results, 20),
            domain_filter=domain_filter,
            query_valence=q_valence,
            query_arousal=q_arousal,
        )
    except Exception as e:
        logger.error(f"Search failed / 检索失败: {e}")
        return "检索过程出错，请稍后重试。"
    matches = await _ordinary_seed_buckets_with_fact_routes(raw_matches)

    seed_diagnostics: dict[str, dict] = {}
    for bucket in matches:
        _upsert_breath_seed_diagnostic(
            seed_diagnostics,
            bucket,
            "profile_fact_route" if bucket.get("profile_fact_route") else "keyword",
            bucket_search_score=bucket.get("score"),
        )

    # --- Vector similarity channel: find semantically related buckets ---
    # --- 向量相似度通道：找到语义相关的桶 ---
    recall_thresholds = _breath_recall_thresholds(query, max_results)
    matched_ids = {b["id"] for b in matches}
    try:
        vector_results = await embedding_engine.search_similar(
            search_query,
            top_k=int(recall_thresholds["semantic_top_k"]),
        )
        for bucket_id, sim_score in vector_results:
            if bucket_id in seed_diagnostics:
                seed_diagnostics[bucket_id]["embedding_score"] = round(float(sim_score), 4)
            if bucket_id not in matched_ids and sim_score >= recall_thresholds["vector_min_score"]:
                bucket = await bucket_mgr.get(bucket_id)
                if bucket:
                    bucket["score"] = round(sim_score * 100, 2)
                    bucket["vector_match"] = True
                    routed_buckets = await _ordinary_seed_buckets_with_fact_routes([bucket])
                else:
                    routed_buckets = []
                for routed in routed_buckets:
                    routed_id = str(routed.get("id") or "")
                    if not routed_id or routed_id in matched_ids:
                        continue
                    _upsert_breath_seed_diagnostic(
                        seed_diagnostics,
                        routed,
                        "profile_fact_route" if routed.get("profile_fact_route") else "vector",
                        embedding_score=sim_score,
                    )
                    matches.append(routed)
                    matched_ids.add(routed_id)
    except Exception as e:
        logger.warning(f"Vector search failed, using keyword only / 向量搜索失败: {e}")

    explicit_lookup = _query_explicitly_requests_archive_memory(query)
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=explicit_lookup)
    except Exception as e:
        logger.warning(f"Failed to list buckets for moment recall / moment 召回列桶失败: {e}")
        all_buckets = matches

    lexical_terms = _append_breath_lexical_matches(
        query=query,
        matches=matches,
        all_buckets=all_buckets,
        seed_diagnostics=seed_diagnostics,
        q_valence=q_valence,
        q_arousal=q_arousal,
    )
    word_map_scores, _word_map_debug = _append_breath_word_map_matches(
        query=query,
        matches=matches,
        all_buckets=all_buckets,
        seed_diagnostics=seed_diagnostics,
    )
    word_map_hint_bucket_ids = set(word_map_scores)

    if retrieval_mode == "bucket":
        direct_results = []
        token_used = 0
        displayed_moment_ids: list[str] = []
        returned_moments: list[dict] = []
        suppressed_buckets = []
        seen_bucket_ids: set[str] = set()
        for bucket in matches:
            if len(direct_results) >= max_results or token_used >= max_tokens:
                break
            bucket_id = str(bucket.get("id") or "")
            if not bucket_id or bucket_id in seen_bucket_ids:
                continue
            seed = seed_diagnostics.get(bucket_id, {})
            word_map_only = "word_map" in (seed.get("sources") or []) and len(set(seed.get("sources") or []) - {"word_map"}) == 0
            decision = _recall_policy().assess(
                query,
                _bucket_relevance_node(bucket, bucket.get("score", 0.0)),
                has_topic_evidence=_bucket_has_query_topic_evidence(query, bucket),
                semantic_score=seed.get("embedding_score"),
                high_confidence_edge="lexical" in (seed.get("sources") or []),
                auto=auto_surface,
            )
            if word_map_only and not _bucket_has_query_topic_evidence(query, bucket):
                reason = "word_map_topic_evidence_missing"
                debug_payload = {
                    "word_map_hint": True,
                    "word_map_score": seed.get("word_map_score"),
                    "has_topic_evidence": False,
                    "auto": bool(auto_surface),
                }
            else:
                reason = decision.reason
                debug_payload = decision.debug
            if (word_map_only and not _bucket_has_query_topic_evidence(query, bucket)) or not decision.admit_direct:
                suppressed_buckets.append(
                    {
                        "bucket_id": bucket_id,
                        "bucket_name": (bucket.get("metadata") or {}).get("name") or bucket_id,
                        "admission_reason": reason,
                        "recall_policy_debug": debug_payload,
                    }
                )
                continue
            bucket_moments = _direct_moments_for_bucket(bucket, query)
            moment = _representative_moment(bucket_moments)
            if not moment:
                moment = _source_record_synthetic_moment_for_bucket(
                    bucket,
                    query,
                    selected_reason="seed_bucket",
                )
            if not moment:
                continue
            grouped = {bucket_id: bucket_moments or [moment]}
            entry = await _format_direct_bucket(
                bucket,
                moment,
                grouped,
                max_tokens - token_used,
                query_text=query,
                direct_render_mode=direct_render_mode,
            )
            if not entry:
                break
            entry_tokens = count_tokens_approx(entry)
            if token_used + entry_tokens > max_tokens:
                break
            await bucket_mgr.touch(bucket_id)
            direct_results.append(entry)
            returned_moments.append(moment)
            displayed_moment_ids.append(str(moment.get("moment_id") or ""))
            seen_bucket_ids.add(bucket_id)
            token_used += entry_tokens

        dream_block = "" if auto_surface else await dream_engine.surface_for_breath(
            query=query,
            valence=valence,
            arousal=arousal,
            is_session_start=is_session_start,
            embedding_engine=embedding_engine,
        )
        response_parts = []
        response_sections = []
        if direct_results:
            response_parts.append("=== 直接命中记忆 ===\n" + "\n---\n".join(direct_results))
            response_sections.append("direct")
        if dream_block:
            response_sections.append("dream")
        _write_breath_recall_diagnostics(
            query=query,
            recall_thresholds={
                **recall_thresholds,
                "retrieval_mode": "bucket",
                "lexical_terms": lexical_terms,
                "word_map_hint_enabled": _word_map_hint_available(),
                "word_map_hint_bucket_ids": sorted(word_map_hint_bucket_ids),
            },
            seed_diagnostics=seed_diagnostics,
            pre_gate_candidates=returned_moments,
            gated_candidates=returned_moments,
            reranked_candidates=returned_moments,
            returned_moments=returned_moments,
            suppressed_candidates=[],
            displayed_moment_ids=displayed_moment_ids,
            secondary_moment_ids=[],
            related_source_bucket_ids=[],
            related_included=False,
            drift_included=False,
            dream_included=bool(dream_block),
            response_sections=response_sections,
        )
        if debug and suppressed_buckets:
            response_parts.append(
                "=== suppressed_bucket_candidates ===\n"
                + "\n".join(
                    f"- [bucket_id:{item['bucket_id']}] reason={item['admission_reason']}"
                    for item in suppressed_buckets[:10]
                )
            )
        if not response_parts:
            return dream_block or "没有找到可靠命中。"
        response_text = "\n\n".join(response_parts)
        if dream_block:
            response_text += "\n\n" + dream_block
        return response_text

    bucket_map = {
        bucket["id"]: bucket
        for bucket in all_buckets
        if bucket.get("id")
        and not is_self_anchor_bucket(bucket)
        and not _is_profile_fact_bucket(bucket)
        and (bucket.get("metadata") or {}).get("type") != "feel"
    }
    active_moments, grouped_moments, _ = await _refresh_moment_graph(all_buckets)
    bucket_boosts = seed_scores_for_buckets(matches)
    if word_map_hint_bucket_ids:
        moment_boost = float(_word_map_hint_settings()["moment_boost"])
        for bucket_id, hint_score in word_map_scores.items():
            seed = seed_diagnostics.get(bucket_id, {})
            sources = set(seed.get("sources") or [])
            if sources and len(sources - {"word_map"}) > 0:
                continue
            bucket_boosts[bucket_id] = max(0.0, min(1.0, float(hint_score) * moment_boost))
    moment_candidates = memory_moment_store.search_moment_items(
        search_query,
        active_moments,
        limit=max(max_results, 20),
        bucket_boosts=bucket_boosts,
        exclude_sections=TASK_ONLY_MOMENT_SECTIONS,
    )
    moment_candidates = _direct_recallable_moments(moment_candidates, explicit_lookup=explicit_lookup)
    source_record_moments = _source_record_synthetic_moments_for_matches(matches, query)
    moment_candidates = _prepend_source_record_synthetic_moments(
        moment_candidates,
        source_record_moments,
    )
    pre_gate_moment_candidates = list(moment_candidates)
    source_record_by_id = {
        str(moment.get("moment_id") or ""): moment
        for moment in source_record_moments
        if moment.get("moment_id")
    }
    gated_non_source = _apply_recall_relevance_gate(
        query,
        [
            moment for moment in moment_candidates
            if str(moment.get("moment_id") or "") not in source_record_by_id
        ],
    )
    gated_moment_candidates = _prepend_source_record_synthetic_moments(
        gated_non_source,
        source_record_moments,
    )
    moment_candidates = gated_moment_candidates
    moment_candidates = await _rerank_breath_moment_candidates(query, moment_candidates)
    moment_candidates = _prepend_source_record_synthetic_moments(
        moment_candidates,
        source_record_moments,
    )
    reranked_moment_candidates = list(moment_candidates)
    admitted_moments = []
    suppressed_moments = []
    for moment in moment_candidates:
        if _is_source_record_synthetic_moment(moment):
            item = dict(moment)
            meta = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
            item["_admission_reason"] = str(meta.get("source_record_direct_reason") or "source_record_direct")
            item["_admission_debug"] = {
                "source_record_direct_override": True,
                "fragment_seed": bool(meta.get("source_record_fragment_seed")),
            }
            admitted_moments.append(item)
            continue
        if str(moment.get("bucket_id") or "") in word_map_hint_bucket_ids:
            seed = seed_diagnostics.get(str(moment.get("bucket_id") or ""), {})
            moment["word_map_hint"] = True
            moment["word_map_score"] = seed.get("word_map_score")
            moment["word_map_terms"] = list(seed.get("word_map_terms") or [])
            moment["word_map_neighbor_terms"] = list(seed.get("word_map_neighbor_terms") or [])
        admission = _breath_moment_admission_decision(
            query,
            moment,
            seed_diagnostics,
            auto=auto_surface,
        )
        item = dict(moment)
        if _breath_word_map_only_without_topic(query, item, seed_diagnostics):
            item["_admission_reason"] = "word_map_topic_evidence_missing"
            item["_admission_debug"] = {
                "word_map_hint": True,
                "word_map_score": item.get("word_map_score"),
                "has_topic_evidence": False,
                "rerank_score": item.get("rerank_score"),
                "auto": bool(auto_surface),
            }
            suppressed_moments.append(item)
            continue
        item["_admission_reason"] = admission.reason
        if admission.admit:
            admitted_moments.append(item)
        else:
            suppressed_moments.append(item)
    moment_candidates = admitted_moments

    direct_results = []
    token_used = 0
    returned_moments = moment_candidates[:max_results]
    direct_display_limit = 1 if include_related else max_results
    displayed_bucket_ids: set[str] = set()
    displayed_moment_ids: list[str] = []
    for moment in returned_moments:
        if len(direct_results) >= direct_display_limit:
            break
        if token_used >= max_tokens:
            break
        bucket_id = str(moment.get("bucket_id") or "")
        if bucket_id in displayed_bucket_ids:
            continue
        try:
            bucket = bucket_map.get(bucket_id)
            if not bucket:
                continue
            entry = await _format_direct_bucket(
                bucket,
                moment,
                grouped_moments,
                max_tokens - token_used,
                query_text=query,
                direct_render_mode=direct_render_mode,
            )
            if not entry:
                break
            entry_tokens = count_tokens_approx(entry)
            if token_used + entry_tokens > max_tokens:
                break
            await bucket_mgr.touch(bucket_id)
            displayed_bucket_ids.add(bucket_id)
            displayed_moment_ids.append(str(moment.get("moment_id") or ""))
            direct_results.append(entry)
            token_used += entry_tokens
        except Exception as e:
            logger.warning(f"Failed to render direct moment / 直接命中片段渲染失败: {e}")
            continue

    related_entry = ""
    secondary_moment_ids: list[str] = []
    related_source_bucket_ids: list[str] = []
    if include_related and returned_moments:
        query_plan = _recall_query_plan(query)
        related_header = "=== 联想浮现 ===\n"
        related_budget = max_tokens - token_used - count_tokens_approx(related_header)
        related_parts = []
        secondary_moments = _secondary_direct_moments(
            query,
            returned_moments,
            displayed_bucket_ids,
            query_plan.secondary_direct_limit(related_per_memory),
            query_plan=query_plan,
            seed_diagnostics=seed_diagnostics,
        )
        for moment in secondary_moments:
            if related_budget <= 0:
                break
            block = _format_secondary_direct_moment(moment)
            block_tokens = count_tokens_approx(block)
            if block_tokens > related_budget:
                break
            related_parts.append(block)
            secondary_moment_ids.append(str(moment.get("moment_id") or ""))
            related_budget -= block_tokens

        related_source_buckets = []
        seen_source_bucket_ids = set()
        promoted_seed_moments = _promoted_breath_related_seed_moments(
            query,
            moment_candidates,
            displayed_bucket_ids,
            seed_diagnostics,
            limit=max(1, related_per_memory),
        )
        related_seed_bucket_ids = displayed_bucket_ids | {
            str(moment.get("bucket_id") or "")
            for moment in promoted_seed_moments
            if moment.get("bucket_id")
        }
        for moment in list(returned_moments) + promoted_seed_moments:
            bucket_id = str(moment.get("bucket_id") or "")
            if bucket_id not in related_seed_bucket_ids:
                continue
            bucket = bucket_map.get(bucket_id)
            if not bucket or bucket_id in seen_source_bucket_ids:
                continue
            if _is_source_record_bucket(bucket):
                continue
            related_source_buckets.append(bucket)
            related_source_bucket_ids.append(bucket_id)
            seen_source_bucket_ids.add(bucket_id)

        related_block = await _build_mcp_diffused_memory_block(
            related_source_buckets,
            all_buckets,
            max(0, related_budget),
            related_per_memory,
            edge_min_confidence,
            query_text=query,
            exclude_bucket_ids={str(moment.get("bucket_id") or "") for moment in secondary_moments},
        )
        if related_block:
            related_parts.append(related_block)
        if related_parts:
            related_entry = related_header + "\n---\n".join(related_parts)
            token_used += count_tokens_approx(related_entry)

    drift_entry = ""
    # --- Resurface: when search returns < 3, 40% chance to float dormant memories ---
    # --- 久未触碰浮现：检索结果不足 3 条时，40% 概率漂起旧桶 ---
    if (
        not related_entry
        and len(returned_moments) < 3
        and not recall_thresholds.get("has_explicit_entity")
        and _query_resurface_enabled()
        and not auto_surface
        and max_tokens > token_used
        and random.random() < 0.4
    ):
        try:
            matched_ids = {str(moment.get("bucket_id")) for moment in returned_moments}
            drifted = await _select_resurface_buckets(
                max_results=random.randint(1, 3),
                exclude_ids=matched_ids,
                include_archive=True,
            )
            if drifted:
                drift_results = []
                drift_remaining = (
                    max_tokens
                    - token_used
                    - count_tokens_approx("--- 久未碰过 ---\n")
                )
                for b in drifted:
                    if drift_remaining <= 0:
                        break
                    clean_meta = {k: v for k, v in b["metadata"].items() if k != "tags"}
                    summary = await dehydrator.dehydrate(_bucket_text_for_embedding(b), clean_meta)
                    dormant_days = _bucket_days_since_last_active(b["metadata"])
                    entry = f"[surface_type: resurface, dormant_days={dormant_days:.0f}]\n{summary}"
                    entry_tokens = count_tokens_approx(entry)
                    if entry_tokens > drift_remaining:
                        break
                    drift_results.append(entry)
                    drift_remaining -= entry_tokens
                if drift_results:
                    drift_entry = "--- 久未碰过 ---\n" + "\n---\n".join(drift_results)
                    if token_used + count_tokens_approx(drift_entry) <= max_tokens:
                        token_used += count_tokens_approx(drift_entry)
                    else:
                        drift_entry = ""
        except Exception as e:
            logger.warning(f"Resurface failed / 久未触碰浮现失败: {e}")

    dream_block = "" if auto_surface else await dream_engine.surface_for_breath(
        query=query,
        valence=valence,
        arousal=arousal,
        is_session_start=is_session_start,
        embedding_engine=embedding_engine,
    )

    response_parts = []
    response_sections = []
    if direct_results:
        response_parts.append("=== 直接命中记忆 ===\n" + "\n---\n".join(direct_results))
        response_sections.append("direct")
    if related_entry:
        response_parts.append(related_entry)
        response_sections.append("related")
    if drift_entry:
        response_parts.append(drift_entry)
        response_sections.append("drift")
    if dream_block:
        response_sections.append("dream")

    _write_breath_recall_diagnostics(
        query=query,
        recall_thresholds={
            **recall_thresholds,
            "lexical_terms": lexical_terms,
            "word_map_hint_enabled": _word_map_hint_available(),
            "word_map_hint_bucket_ids": sorted(word_map_hint_bucket_ids),
        },
        seed_diagnostics=seed_diagnostics,
        pre_gate_candidates=pre_gate_moment_candidates,
        gated_candidates=gated_moment_candidates,
        reranked_candidates=reranked_moment_candidates,
        returned_moments=returned_moments,
        suppressed_candidates=suppressed_moments,
        displayed_moment_ids=displayed_moment_ids,
        secondary_moment_ids=secondary_moment_ids,
        related_source_bucket_ids=related_source_bucket_ids,
        related_included=bool(related_entry),
        drift_included=bool(drift_entry),
        dream_included=bool(dream_block),
        response_sections=response_sections,
    )

    if debug and suppressed_moments:
        response_parts.append(
            "=== suppressed_candidates ===\n"
            + "\n".join(_format_suppressed_recall_candidate(moment, seed_diagnostics) for moment in suppressed_moments[:10])
        )

    if not response_parts:
        if recall_thresholds.get("has_explicit_entity") and suppressed_moments:
            return dream_block or "没有找到可靠命中。"
        return dream_block or "未找到相关记忆。"

    response_text = "\n\n".join(response_parts)
    if dream_block:
        response_text += "\n\n" + dream_block
    return response_text


async def _select_resurface_buckets(
    max_results: int = 1,
    *,
    exclude_ids: set[str] | None = None,
    include_archive: bool = True,
) -> list[dict]:
    exclude_ids = exclude_ids or set()
    max_results = max(1, min(5, int(max_results or 1)))
    all_buckets = await bucket_mgr.list_all(include_archive=include_archive)
    candidates = []
    for bucket in all_buckets:
        meta = bucket.get("metadata", {})
        if bucket.get("id") in exclude_ids:
            continue
        if is_self_anchor_bucket(bucket):
            continue
        if meta.get("type") in {"feel", "permanent"}:
            continue
        if meta.get("pinned") or meta.get("protected"):
            continue
        if meta.get("anchor"):
            continue
        dormant_days = _bucket_days_since_last_active(meta)
        importance = max(1, min(10, int(meta.get("importance", 5))))
        archived_bonus = 1.15 if meta.get("type") == "archived" else 1.0
        resurface_score = (dormant_days + 1.0) * (0.6 + importance / 10.0) * archived_bonus
        candidates.append((resurface_score, bucket))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [bucket for _, bucket in candidates[:max_results]]


# =============================================================
# Tool 1.4: resurface — dormant memory resurfacing
# 工具 1.4：resurface — 久未触碰记忆浮现
# =============================================================
async def resurface(max_results: int = 1, include_archive: bool = True, max_tokens: int = 800) -> str:
    """只读浮现久未触碰的旧记忆。越久没碰过越靠前；默认包含归档桶；不 touch,不刷新 last_active,不增加 activation_count。"""
    try:
        buckets = await _select_resurface_buckets(
            max_results=max_results,
            include_archive=include_archive,
        )
    except Exception as e:
        logger.error(f"Resurface listing failed / 久未触碰浮现列桶失败: {e}")
        return "旧记忆暂时无法浮现。"

    if not buckets:
        return "没有可浮现的旧记忆。"

    parts = []
    remaining = max(100, max_tokens)
    for bucket in buckets:
        meta = bucket.get("metadata", {})
        dormant_days = _bucket_days_since_last_active(meta)
        state = []
        if meta.get("type") == "archived":
            state.append("归档")
        if meta.get("resolved"):
            state.append("已解决")
        if meta.get("digested"):
            state.append("已消化")
        state_text = f" ({', '.join(state)})" if state else ""
        entry = (
            f"[bucket_id:{bucket['id']}] {meta.get('name', bucket['id'])}{state_text} "
            f"久未触碰 {dormant_days:.0f} 天\n"
            f"{_bucket_text_for_embedding(bucket).strip()[:420]}"
        )
        tokens = count_tokens_approx(entry)
        if tokens > remaining and parts:
            break
        parts.append(entry)
        remaining -= tokens
        if remaining <= 0:
            break

    return "=== 久未触碰的旧记忆 ===\n" + "\n---\n".join(parts)


# Python compatibility name for internal callers and legacy tests. It is not an
# MCP registration; chat models can only call the daily façade below.
breath = _recall_memory


# =============================================================
# Tool 1.5: read_bucket — exact archive-cabinet read
# 工具 1.5：read_bucket — 按 ID 精确读桶
# =============================================================
async def _read_scene_memory(bucket_id: str) -> dict:
    """按 bucket_id 精确读取完整记忆桶；trace/comment 前先读。只读，不刷新活跃度。"""
    bucket_id = _coerce_memory_id(bucket_id)
    if not bucket_id or not MEMORY_ID_RE.fullmatch(bucket_id):
        return {"error": "invalid bucket_id"}
    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return {"error": "not found", "id": bucket_id}
    return _bucket_read_payload(bucket)


read_bucket = _read_scene_memory


# =============================================================
# Tool 1.55: list_buckets_light — lightweight bucket index
# 工具 1.55：list_buckets_light — 轻量桶索引
# =============================================================
async def list_buckets_light(
    include_archive: bool = False,
    limit: int = 500,
    offset: int = 0,
) -> dict:
    """只读列出桶的轻量元数据；不返回正文，给同步脚本和外部索引用。"""
    safe_limit = max(1, min(int(limit or 500), 2000))
    safe_offset = max(0, int(offset or 0))
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=include_archive)
        items = [_bucket_light_payload(bucket) for bucket in all_buckets]
        items.sort(key=_bucket_light_sort_key, reverse=True)
        return {
            "buckets": items[safe_offset : safe_offset + safe_limit],
            "count": len(items),
            "include_archive": bool(include_archive),
            "limit": safe_limit,
            "offset": safe_offset,
        }
    except Exception as e:
        return {"error": str(e), "buckets": []}


# =============================================================
# Tool 1.56: Narrative Roll read-only index and full read
# 工具 1.56：叙事卷只读索引与全文读取
# =============================================================
async def narrative_rolls(query: str = "", limit: int = 20) -> dict:
    """只读搜索叙事卷索引；返回标题、时间范围、状态、实体与 narrative_id，不返回正文。按卷名或实体找卷；精确日期/原话仍应读 Scene 或 raw。"""
    return narrative_roll_store.list(query=query, limit=limit)


async def _read_narrative_memory(narrative_id: str) -> dict:
    """按 narrative_id 只读完整叙事卷：正文、当前状态、来源账、准确性边界、revision 与 linked_scene_ids 一起返回。Narrative Roll 是有来源的第一人称派生叙事，不是原始证据；精确日期和逐字原话继续下钻 Scene/raw。"""
    narrative_id = _coerce_memory_id(narrative_id)
    if not narrative_id or not MEMORY_ID_RE.fullmatch(narrative_id):
        return {"status": "invalid", "error": "invalid narrative_id"}
    return narrative_roll_store.read(narrative_id)


read_narrative_roll = _read_narrative_memory


@mcp.custom_route("/api/narrative-rolls", methods=["GET"])
async def api_narrative_rolls(request):
    """Read the lightweight Narrative Roll index or one exact full roll."""
    from starlette.responses import JSONResponse

    err = _require_dashboard_auth(request)
    if err:
        return err
    narrative_id = str(request.query_params.get("narrative_id") or "").strip()
    if narrative_id:
        result = await _read_narrative_memory(narrative_id)
    else:
        result = await narrative_rolls(
            query=str(request.query_params.get("query") or ""),
            limit=_int_between(request.query_params.get("limit"), 20, 1, 100),
        )
    status_code = (
        404
        if result.get("status") == "not_found"
        else 400
        if result.get("status") == "invalid"
        else 200
    )
    return JSONResponse(result, status_code=status_code)


# =============================================================
# Tool 1.57: Scene-edge proposal review
# 工具 1.57：只读查看并明确审核 Scene 边提案
# =============================================================
async def scene_edge_proposals(
    status: str = "pending",
    proposal_id: str = "",
    anchor_scene_id: str = "",
    limit: int = 20,
    include_context: bool = False,
) -> dict:
    """只读查看 Scene linker 提案；默认列 pending，不写正文或正式边。审核单条前传 proposal_id 与 include_context=true，检查两端 Scene、逐字证据和 review_state。"""
    normalized_id = _coerce_memory_id(proposal_id)
    normalized_anchor = _coerce_memory_id(anchor_scene_id)
    if normalized_id and not MEMORY_ID_RE.fullmatch(normalized_id):
        return {"status": "error", "error": "invalid proposal_id", "proposals": []}
    if normalized_anchor and not MEMORY_ID_RE.fullmatch(normalized_anchor):
        return {"status": "error", "error": "invalid anchor_scene_id", "proposals": []}
    try:
        return await scene_linker.list_proposals(
            bucket_mgr,
            status=status,
            proposal_id=normalized_id,
            anchor_scene_id=normalized_anchor,
            limit=max(1, min(int(limit or 20), 100)),
            include_context=bool(include_context),
        )
    except (TypeError, ValueError) as exc:
        return {"status": "error", "error": str(exc), "proposals": []}


async def review_scene_edge_proposal(
    proposal_id: str,
    decision: str,
    confirm: str = "",
) -> dict:
    """明确接受或拒绝一条 pending Scene 边提案。先用 scene_edge_proposals 读精确 proposal_id；只有用户明确同意后才可接受并传 confirm="ACCEPT_SCENE_EDGE"，拒绝传 confirm="REJECT_SCENE_EDGE"。接受前会重验两端 Scene hash、active 状态与逐字证据；拒绝不写正式边。"""
    return await scene_linker.review_proposal(
        _coerce_memory_id(proposal_id),
        decision,
        confirm,
        bucket_mgr,
        memory_edge_store,
        reviewed_by=_ai_author_name(),
    )


@mcp.custom_route("/api/scene-edge-proposals", methods=["GET"])
async def api_scene_edge_proposals(request):
    """Read Scene-edge proposals for dashboard inspection."""
    from starlette.responses import JSONResponse

    err = _require_dashboard_auth(request)
    if err:
        return err
    result = await scene_edge_proposals(
        status=str(request.query_params.get("status") or "pending"),
        proposal_id=str(request.query_params.get("proposal_id") or ""),
        anchor_scene_id=str(request.query_params.get("anchor_scene_id") or ""),
        limit=_int_between(request.query_params.get("limit"), 20, 1, 100),
        include_context=_bool_value(request.query_params.get("include_context"), False),
    )
    status_code = 400 if result.get("status") == "error" else 200
    return JSONResponse(result, status_code=status_code)


@mcp.custom_route("/api/scene-edge-proposals/review", methods=["POST"])
async def api_scene_edge_proposal_review(request):
    """Explicitly accept or reject one pending Scene-edge proposal."""
    from starlette.responses import JSONResponse

    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "json body must be an object"}, status_code=400)
    result = await scene_linker.review_proposal(
        _coerce_memory_id(body.get("proposal_id")),
        str(body.get("decision") or ""),
        str(body.get("confirm") or ""),
        bucket_mgr,
        memory_edge_store,
        reviewed_by=_dashboard_author_name(),
    )
    result_status = str(result.get("status") or "")
    if result_status == "not_found":
        status_code = 404
    elif result_status in {"conflict", "stale"}:
        status_code = 409
    elif result_status in {"error", "confirmation_required"}:
        status_code = 400
    else:
        status_code = 200
    return JSONResponse(result, status_code=status_code)


# =============================================================
# Admin: legacy memory lifecycle / Scene-bridge review
# 管理面：旧记忆生命周期 / Scene bridge 审阅
# =============================================================
@mcp.custom_route("/api/legacy-memory-reviews", methods=["GET"])
async def api_legacy_memory_reviews(request):
    """Read review-only legacy lifecycle and bridge proposals."""
    from starlette.responses import JSONResponse

    err = _require_dashboard_auth(request)
    if err:
        return err
    proposal_id = _coerce_memory_id(request.query_params.get("proposal_id"))
    if proposal_id and not MEMORY_ID_RE.fullmatch(proposal_id):
        return JSONResponse({"status": "error", "error": "invalid proposal_id"}, status_code=400)
    try:
        result = await legacy_memory_review_store.list_for_review(
            bucket_mgr,
            status=str(request.query_params.get("status") or "pending"),
            proposal_id=proposal_id,
            limit=_int_between(request.query_params.get("limit"), 20, 1, 100),
            include_context=_bool_value(request.query_params.get("include_context"), False),
        )
    except (TypeError, ValueError) as exc:
        return JSONResponse({"status": "error", "error": str(exc)}, status_code=400)
    return JSONResponse(result)


@mcp.custom_route("/api/legacy-memory-reviews/review", methods=["POST"])
async def api_legacy_memory_review(request):
    """Record an explicit review decision without applying it to memory state."""
    from starlette.responses import JSONResponse

    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"status": "error", "error": "invalid json body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse(
            {"status": "error", "error": "json body must be an object"},
            status_code=400,
        )
    proposal_id = _coerce_memory_id(body.get("proposal_id"))
    if not proposal_id or not MEMORY_ID_RE.fullmatch(proposal_id):
        return JSONResponse({"status": "error", "error": "invalid proposal_id"}, status_code=400)
    result = await legacy_memory_review_store.review(
        proposal_id,
        str(body.get("decision") or ""),
        str(body.get("confirm") or ""),
        bucket_mgr,
        reviewed_by=_dashboard_author_name(),
    )
    result_status = str(result.get("status") or "")
    if result_status == "not_found":
        status_code = 404
    elif result_status in {"conflict", "stale"}:
        status_code = 409
    elif result_status in {"error", "confirmation_required"}:
        status_code = 400
    else:
        status_code = 200
    return JSONResponse(result, status_code=status_code)


@mcp.custom_route("/api/legacy-memory-reviews/apply-archive", methods=["POST"])
async def api_legacy_memory_review_apply_archive(request):
    """Apply one accepted archive review; preserve source and a verified backup."""
    from starlette.responses import JSONResponse

    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"status": "error", "error": "invalid json body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse(
            {"status": "error", "error": "json body must be an object"},
            status_code=400,
        )
    proposal_id = _coerce_memory_id(body.get("proposal_id"))
    if not proposal_id or not MEMORY_ID_RE.fullmatch(proposal_id):
        return JSONResponse({"status": "error", "error": "invalid proposal_id"}, status_code=400)
    result = await legacy_memory_lifecycle_publisher.publish_archive(
        proposal_id,
        bucket_mgr,
        confirm=str(body.get("confirm") or ""),
        applied_by=_dashboard_author_name(),
        dry_run=_bool_value(body.get("dry_run"), False),
    )
    result_status = str(result.get("status") or "")
    if result_status == "not_found":
        status_code = 404
    elif result_status in {"conflict", "stale"}:
        status_code = 409
    elif result_status in {"error", "confirmation_required"}:
        status_code = 400
    else:
        status_code = 200
    return JSONResponse(result, status_code=status_code)


# =============================================================
# Tool 1.6: comment_bucket — add a ring/comment to a memory
# 工具 1.6：comment_bucket — 给记忆追加年轮
# =============================================================
async def _append_annotation(
    bucket_id: str,
    content: str,
    kind: str = "comment",
    valence: float = -1,
    arousal: float = -1,
    *,
    source: str = "annotate",
) -> dict:
    """给已有 bucket 追加年轮/补充感受；会 touch，不改正文。kind=feel 时 content 只能写“我……”第一人称正文，不写标题或任何 Markdown 分段。"""
    bucket_id = _coerce_memory_id(bucket_id)
    if not bucket_id or not MEMORY_ID_RE.fullmatch(bucket_id):
        return {"error": "invalid bucket_id"}
    if not content or not content.strip():
        return {"error": "empty content"}
    if str(kind or "").strip().lower() == "feel":
        contract_error = _memory_write_contract_error(content, feel_only=True)
        if contract_error:
            return {"error": "invalid feel content", "reason": contract_error}
    if not await bucket_mgr.get(bucket_id):
        return {"error": "not found", "id": bucket_id}

    entry = await bucket_mgr.add_comment(
        bucket_id,
        content,
        author=_ai_author_name(),
        kind=kind or "comment",
        valence=valence if 0 <= valence <= 1 else None,
        arousal=arousal if 0 <= arousal <= 1 else None,
        source=source,
        touch=True,
    )
    if not entry:
        return {"error": "write failed", "id": bucket_id}
    bucket = await bucket_mgr.get(bucket_id)
    embedding_queued = _queue_embedding_refresh(bucket_id)
    return {
        "status": "commented",
        "id": bucket_id,
        "comment": entry,
        "embedding_refreshed": False,
        "embedding_queued": embedding_queued,
        "metadata": _bucket_read_payload(bucket)["metadata"] if bucket else {},
    }


async def comment_bucket(
    bucket_id: str,
    content: str,
    kind: str = "comment",
    valence: float = -1,
    arousal: float = -1,
) -> dict:
    """Internal compatibility wrapper for old HTTP/Python callers."""
    return await _append_annotation(
        bucket_id,
        content,
        kind=kind,
        valence=valence,
        arousal=arousal,
        source="comment_bucket",
    )


# =============================================================
# Tool 1.7: delete_bucket_comment — delete one AI-authored ring
# 工具 1.7：delete_bucket_comment — 删除一条自己写的年轮
# =============================================================
async def delete_bucket_comment(bucket_id: str, comment_id: str) -> dict:
    """删除自己通过 comment_bucket 写入的一条年轮；不会删除 bucket，也不会删除小雨/dashboard 写的年轮。"""
    bucket_id = _coerce_memory_id(bucket_id)
    comment_id = _coerce_memory_id(comment_id)
    if not bucket_id or not MEMORY_ID_RE.fullmatch(bucket_id):
        return {"error": "invalid bucket_id"}
    if not comment_id or not MEMORY_ID_RE.fullmatch(comment_id):
        return {"error": "invalid comment_id"}
    if not await bucket_mgr.get(bucket_id):
        return {"error": "not found", "id": bucket_id}

    result = await bucket_mgr.delete_comment(
        bucket_id,
        comment_id,
        allowed_author=_ai_author_name(),
        allowed_source="comment_bucket",
    )
    if result.get("status") == "not_found":
        return {"error": "comment not found", "id": bucket_id, "comment_id": comment_id}
    if result.get("status") == "forbidden":
        return {
            "error": "forbidden",
            "reason": "only AI-authored comment_bucket year rings can be deleted",
            "id": bucket_id,
            "comment_id": comment_id,
        }
    if result.get("status") != "deleted":
        return {"error": "delete failed", "id": bucket_id, "comment_id": comment_id}

    embedding_queued = _queue_embedding_refresh(bucket_id)
    bucket = await bucket_mgr.get(bucket_id)
    return {
        "status": "deleted",
        "id": bucket_id,
        "comment_id": comment_id,
        "embedding_refreshed": False,
        "embedding_queued": embedding_queued,
        "metadata": _bucket_read_payload(bucket)["metadata"] if bucket else {},
    }


@mcp.custom_route("/api/bucket/{bucket_id}/comments", methods=["POST"])
async def api_bucket_comment(request):
    """Add a dashboard-authenticated user comment to a bucket."""
    from starlette.responses import JSONResponse

    err = _require_dashboard_auth(request)
    if err:
        return err

    bucket_id = request.path_params["bucket_id"]
    if not bucket_id or not MEMORY_ID_RE.fullmatch(bucket_id):
        return JSONResponse({"error": "invalid bucket_id"}, status_code=400)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "json body must be an object"}, status_code=400)

    content = str(body.get("content") or "").strip()
    if not content:
        return JSONResponse({"error": "empty content"}, status_code=400)
    if not await bucket_mgr.get(bucket_id):
        return JSONResponse({"error": "not found", "id": bucket_id}, status_code=404)

    valence = _float_between(body.get("valence"), -1.0)
    arousal = _float_between(body.get("arousal"), -1.0)
    entry = await bucket_mgr.add_comment(
        bucket_id,
        content,
        author=_dashboard_author_name(),
        kind=str(body.get("kind") or "comment"),
        valence=valence if 0 <= valence <= 1 else None,
        arousal=arousal if 0 <= arousal <= 1 else None,
        source="dashboard",
        touch=True,
    )
    if not entry:
        return JSONResponse({"error": "write failed", "id": bucket_id}, status_code=500)

    embedding_queued = _queue_embedding_refresh(bucket_id)

    bucket = await bucket_mgr.get(bucket_id)
    return JSONResponse({
        "status": "commented",
        "id": bucket_id,
        "comment": entry,
        "embedding_refreshed": False,
        "embedding_queued": embedding_queued,
        "metadata": _bucket_read_payload(bucket)["metadata"] if bucket else {},
    })


@mcp.custom_route("/api/bucket/{bucket_id}/comments/{comment_id}", methods=["DELETE"])
async def api_bucket_comment_delete(request):
    """Delete a dashboard-authenticated user comment from a bucket."""
    from starlette.responses import JSONResponse

    err = _require_dashboard_auth(request)
    if err:
        return err

    bucket_id = request.path_params["bucket_id"]
    comment_id = request.path_params["comment_id"]
    if not bucket_id or not MEMORY_ID_RE.fullmatch(bucket_id):
        return JSONResponse({"error": "invalid bucket_id"}, status_code=400)
    if not comment_id or not MEMORY_ID_RE.fullmatch(comment_id):
        return JSONResponse({"error": "invalid comment_id"}, status_code=400)
    if not await bucket_mgr.get(bucket_id):
        return JSONResponse({"error": "not found", "id": bucket_id}, status_code=404)

    result = await bucket_mgr.delete_comment(
        bucket_id,
        comment_id,
        allowed_author=_dashboard_author_name(),
        allowed_source="dashboard",
    )
    if result.get("status") == "not_found":
        return JSONResponse({"error": "comment not found"}, status_code=404)
    if result.get("status") == "forbidden":
        return JSONResponse({"error": "only dashboard user comments can be deleted"}, status_code=403)
    if result.get("status") != "deleted":
        return JSONResponse({"error": "delete failed"}, status_code=500)

    embedding_queued = _queue_embedding_refresh(bucket_id)

    bucket = await bucket_mgr.get(bucket_id)
    return JSONResponse({
        "status": "deleted",
        "id": bucket_id,
        "comment_id": comment_id,
        "embedding_refreshed": False,
        "embedding_queued": embedding_queued,
        "metadata": _bucket_read_payload(bucket)["metadata"] if bucket else {},
    })


# =============================================================
# Tool 2: hold — Hold on to this
# 工具 2：hold — 握住，留下来
# =============================================================
async def hold(
    content: str,
    tags: str = "",
    importance: int = 5,
    pinned: bool = False,
    feel: bool = False,
    whisper: bool = False,
    source_bucket: str = "",
    valence: float = -1,
    arousal: float = -1,
    title: str = "",
    date: str = "",
    domain: str = "",
    cues: str = "",
) -> str:
    """原样保存一件由当前 AI 写好的长期 Scene，不调用模型脱水、改写、命名或同步打标。普通 content 直接写一段完整原文经历，不要添加 `## Scene`、`### scene`、`### moment` 或 sibling section。Scene 对象本身就是类型和普通召回单位。多个场景分别调用 hold。cues 可用 `|` 或换行传 0～8 个未来可能自然出现的召回入口；它们只写入稀疏 sidecar 索引，不进入正文或 Scene 原文向量，也不会彼此扩散。若部署启用 Scene linker，写入返回后可异步生成待审关系边提案；提案必须引用两端 content 原句，不改正文、不自动写正式边。有来源的新理解用 comment_bucket/annotate 挂回来源。feel、whisper、daily impression 与 ProfileFact 默认拒绝新增；旧数据仍可读取。date/title/domain、valence/arousal 仅为旧客户端兼容。"""
    await decay_engine.ensure_started()

    # --- Input validation / 输入校验 ---
    if not content or not content.strip():
        return "内容为空，无法存储。"

    if (feel or whisper) and not legacy_memory_writes_enabled(config):
        return retired_write_message("whisper" if whisper else "feel")

    if not feel and not whisper:
        scene_error = _hold_scene_contract_error(content)
        if scene_error:
            return f"写入被拒绝：{scene_error}"
    else:
        contract_error = _memory_write_contract_error(content, feel_only=True)
        if contract_error:
            return f"写入被拒绝：{contract_error}"

    importance = max(1, min(10, importance))
    extra_tags = [t.strip() for t in tags.split(",") if t.strip()]
    retired_tags = {
        str(tag).strip().lower()
        for tag in extra_tags
        if str(tag).strip().lower()
        in {"feel", "whisper", "daily_impression", "weekly_impression", "relationship_weather", "profile_fact"}
    }
    if retired_tags and not legacy_memory_writes_enabled(config):
        return retired_write_message(sorted(retired_tags)[0])
    requested_domain = [d.strip() for d in str(domain or "").split(",") if d.strip()]
    event_date = str(date or "").strip()
    requested_valence = valence if 0 <= valence <= 1 else None
    requested_arousal = arousal if 0 <= arousal <= 1 else None

    async def create_whisper_bucket() -> str:
        whisper_valence = requested_valence if requested_valence is not None else 0.5
        whisper_arousal = requested_arousal if requested_arousal is not None else 0.3
        whisper_tags = list(dict.fromkeys(extra_tags + ["whisper"]))
        bucket_id = await bucket_mgr.create(
            content=content,
            tags=whisper_tags,
            importance=5,
            domain=requested_domain,
            valence=whisper_valence,
            arousal=whisper_arousal,
            name=None,
            bucket_type="feel",
            date=event_date or None,
        )
        _queue_embedding_refresh(bucket_id)
        return f"🫧whisper→{bucket_id}"

    if whisper:
        if source_bucket and source_bucket.strip():
            return "whisper 不需要 source_bucket；有源记忆的感受请用 comment_bucket。"
        return await create_whisper_bucket()

    # --- Feel mode: attach to source bucket as a ring comment when possible ---
        # --- Feel 模式：有源记忆时挂成年轮 ---
    if feel:
        # Feel valence/arousal = model's own perspective
        feel_valence = requested_valence if requested_valence is not None else 0.5
        feel_arousal = requested_arousal if requested_arousal is not None else 0.3
        source_id = (source_bucket or "").strip()
        if source_id:
            if not MEMORY_ID_RE.fullmatch(source_id):
                return "source_bucket 无效。"
            source = await bucket_mgr.get(source_id)
            if not source:
                return f"源记忆不存在: {source_id}"
            entry = await bucket_mgr.add_comment(
                source_id,
                content,
                author=_ai_author_name(),
                kind="feel",
                valence=feel_valence,
                arousal=feel_arousal,
                source="hold(feel=True)",
                touch=True,
            )
            if not entry:
                return "年轮写入失败。"
            _queue_embedding_refresh(source_id)
            return f"年轮→{source_id}#{entry['id']}"

        # No source bucket: keep a standalone feel for compatibility.
        # 没有源记忆时保留独立 whisper，兼容旧用法。
        return await create_whisper_bucket()

    scene_title = _authored_scene_title(content, title)
    scene_cues = _authored_scene_cues(content, title=scene_title, explicit=cues)
    scene_write_source = _SCENE_WRITE_SOURCE.get()
    if scene_write_source not in {"hold_scene", "write_scene"}:
        scene_write_source = "hold_scene"
    scene_write_contract = (
        "write-scene-v1"
        if scene_write_source == "write_scene"
        else "hold-scene-v2"
    )
    domain = requested_domain or ["未分类"]
    # Ordinary source records keep legacy coordinates only for schema/client compatibility.
    # They no longer influence recall ranking or decay, and the tagger no longer infers them.
    valence = requested_valence if requested_valence is not None else 0.5
    arousal = requested_arousal if requested_arousal is not None else 0.3
    suggested_name = scene_title

    all_tags = list(dict.fromkeys(extra_tags))
    classification = normalize_write_classification(
        memory_subject="",
        memory_layer="",
        tags=all_tags,
        content=content,
    )
    # --- Pinned buckets bypass merge and are created directly in permanent dir ---
    # --- 钉选桶跳过合并，直接新建到 permanent 目录 ---
    if pinned:
        related_bucket = await _find_readonly_related_bucket(content)
        bucket_id = await bucket_mgr.create(
            content=content,
            tags=all_tags,
            importance=10,
            domain=domain,
            valence=valence,
            arousal=arousal,
            name=suggested_name or None,
            bucket_id=generate_scene_id(),
            bucket_type="permanent",
            pinned=True,
            date=event_date or None,
            source=scene_write_source,
            extra_metadata={
                **_memory_classification_metadata(
                    classification["memory_subject"],
                    classification["memory_layer"],
                    classification["memory_classification_source"],
                ),
                "memory_value_source": "authored_scene",
                "object_kind": "scene",
                "write_contract": scene_write_contract,
                "scene_cues": scene_cues or None,
            },
        )
        _queue_embedding_refresh(bucket_id)
        _queue_scene_linking(bucket_id)
        related_note = _format_readonly_related_memory(related_bucket) if related_bucket else ""
        return f"📌钉选→{bucket_id} {','.join(domain)}{related_note}"

    # --- Step 2: merge or create / 合并或新建 ---
    bucket_id, result_name, is_merged, related_bucket = await _merge_or_create(
        content=content,
        tags=all_tags,
        importance=importance,
        domain=domain,
        valence=valence,
        arousal=arousal,
        name=suggested_name,
        allow_merge=False,
        memory_subject=classification["memory_subject"],
        memory_layer=classification["memory_layer"],
        memory_classification_source=classification["memory_classification_source"],
        date=event_date,
        source=scene_write_source,
        bucket_id=generate_scene_id(),
        extra_metadata={
            "memory_value_source": "authored_scene",
            "object_kind": "scene",
            "write_contract": scene_write_contract,
            "scene_cues": scene_cues or None,
        },
        normalize_content=False,
    )
    action = "合并→" if is_merged else "新建→"
    _queue_scene_linking(bucket_id)
    related_note = _format_readonly_related_memory(related_bucket) if related_bucket else ""
    return f"{action}{result_name} {','.join(domain)}{related_note}"


# =============================================================
# Tool 2.5: darkroom — Private unfinished reflection
# 工具 2.5：darkroom — 暗房，存放未显影的内在反思
# =============================================================
async def darkroom_enter(
    note: str,
    mode: str = "continue",
    mood: str = "",
    tags: str = "",
    source: str = "mcp",
    visibility: str = "active",
    lock_for: str = "",
    new_room: bool = True,
) -> dict:
    """写入一段未显影的私密反思；默认第一人称，不用第三人称自述；默认新开房间，new_room=false 才续写当前 active 房间；写错要撤回已有房间时传 new_room=false + visibility="retracted"；不回显 note 正文。"""
    try:
        return darkroom_store.enter(
            note,
            mood=mood,
            tags=tags,
            source=source,
            mode=mode,
            visibility=visibility,
            lock_for=lock_for,
            new_room=new_room,
        )
    except ValueError as exc:
        return {"status": "error", "error": str(exc)}


async def darkroom_rooms(limit: int = 20, visibility: str = "active") -> dict:
    """只读列出暗房门牌，不返回正文；默认列 active 房间，可传 visibility="all" 看全部门牌，用 room_id 再调用 darkroom_view。"""
    try:
        return darkroom_store.rooms(limit=limit, visibility=visibility)
    except ValueError as exc:
        return {"status": "error", "error": str(exc)}


async def darkroom_delete(room_id: str, confirm: str = "") -> dict:
    """从暗房主存储删除一整间房及全部 revisions；必须传精确 room_id 和 confirm="DELETE"，并保留本地私密备份。"""
    try:
        return darkroom_store.delete_room(room_id, confirm=confirm)
    except ValueError as exc:
        return {"status": "error", "error": str(exc), "room_id": str(room_id or "")}
    except KeyError:
        return {"status": "not_found", "error": "room not found", "room_id": str(room_id or "")}


async def darkroom_view(entry_id: str = "latest") -> dict:
    """只读查看一条已解锁的暗房内容；未到锁门时间不返回正文。"""
    try:
        return darkroom_store.view(entry_id=entry_id)
    except KeyError:
        return {"status": "error", "error": "entry not found"}


async def darkroom_status() -> dict:
    """查看暗房门口状态。不返回任何暗房正文。"""
    return darkroom_store.status()


async def darkroom_release(entry_id: str = "latest", reason: str = "") -> dict:
    """把一条暗房内容显影并带出来。这个工具会公开返回正文,只在明确想让内容可见时调用。"""
    try:
        return darkroom_store.release(entry_id=entry_id, reason=reason)
    except KeyError:
        return {"status": "error", "error": "entry not found"}


# =============================================================
# Tool 3: close_window — Atomically preserve one Shadow + authored Scenes
# 工具 3：close_window — 原子保存一整窗窗影与明确 Scene
# =============================================================
async def _write_window_shadow_scene(
    window: dict,
    scene: dict,
    index: int,
) -> tuple[str, str]:
    """Copy one authored scene into an ordinary recall bucket without rewriting it."""
    window_id = str(window.get("window_id") or "").strip()
    bucket_id = f"{window_id}_scene_{index}"
    existing = await bucket_mgr.get(bucket_id)
    if existing:
        existing_meta = existing.get("metadata", {}) if isinstance(existing.get("metadata"), dict) else {}
        if (
            str(existing_meta.get("window_shadow_id") or "") != window_id
            or str(existing_meta.get("scene_source_hash") or "")
            != WindowShadowStore.source_hash(str(scene.get("content") or "").strip())
        ):
            raise ValueError(f"scene bucket id collision: {bucket_id}")
        return bucket_id, "existing"
    scene_content = str(scene.get("content") or "").strip()
    contract_error = _hold_scene_contract_error(scene_content)
    if contract_error:
        raise ValueError(contract_error)
    tags = ["window_shadow_scene"]
    classification = normalize_write_classification(
        memory_subject="",
        memory_layer="",
        tags=tags,
        content=scene_content,
    )
    name = str(scene.get("title") or f"窗影场景{index}").strip()
    scene_cues = _authored_scene_cues(scene_content, title=name)
    await bucket_mgr.create(
        content=scene_content,
        tags=tags,
        importance=6,
        domain=["未分类"],
        valence=0.5,
        arousal=0.3,
        name=name,
        bucket_id=bucket_id,
        source="window_shadow",
        date=str(window.get("source_date") or "") or None,
        extra_metadata={
            "window_shadow_id": window_id,
            "window_shadow_session_id": str(window.get("session_id") or ""),
            "window_shadow_index": index,
            "scene_source_hash": WindowShadowStore.source_hash(scene_content),
            "memory_value_source": "authored_scene",
            "object_kind": "scene",
            "write_contract": "close-window-scene-v2",
            "scene_cues": scene_cues or None,
            **_memory_classification_metadata(
                classification["memory_subject"],
                classification["memory_layer"],
                classification["memory_classification_source"],
            ),
        },
    )
    return bucket_id, "created"


def _window_shadow_scene_records(
    shadow: str,
    explicit_scenes: list[str] | None,
    *,
    markdown_import: bool = False,
) -> tuple[list[dict[str, str]], str]:
    """Validate canonical Scene inputs before any Shadow or bucket is written."""
    if markdown_import:
        if any(str(value or "").strip() for value in (explicit_scenes or [])):
            return [], "markdown_import 只无损导入 Shadow，不接受 scenes；需要普通召回的场景请人工确认后另用 write_scene。"
        return [], ""
    inline_scenes = extract_window_shadow_scenes(shadow)
    provided = [str(value or "").strip() for value in (explicit_scenes or []) if str(value or "").strip()]
    if inline_scenes and provided:
        return [], "Scene 请只选一种写法：放进窗影的 `## 想留下的记忆`，或通过 scenes 参数传入，不能两边重复。"
    if inline_scenes:
        records = inline_scenes
    else:
        records = [
            {
                "title": _authored_scene_title(content) or f"窗影场景{index}",
                "content": content,
                "source_text": content,
            }
            for index, content in enumerate(provided, start=1)
        ]
    for index, record in enumerate(records, start=1):
        error = _hold_scene_contract_error(str(record.get("content") or ""))
        if error:
            return [], f"Scene {index} 格式不合格：{error}"
    return records, ""


async def _close_window_commit(
    shadow: str,
    *,
    scenes: list[str] | None = None,
    continue_scene_index: int = 0,
    session_id: str = "",
    profile_id: str = "",
    date: str = "",
    source: str = "",
    require_handoff_note: bool = True,
) -> dict:
    """Compensating transaction for one append-only Shadow and zero or more Scenes."""
    await decay_engine.ensure_started()
    text = str(shadow or "")
    if not text.strip():
        return {"status": "invalid", "reason": "empty_shadow", "error": "窗影内容为空，未保存。"}
    resolved_source = str(source or "").strip()
    markdown_import = resolved_source.lower() in {"markdown", "markdown_import", "md_import"}
    sections, validation_errors = validate_window_shadow(
        text,
        require_handoff_note=require_handoff_note and not markdown_import,
    )
    if validation_errors:
        return {
            "status": "invalid",
            "reason": "invalid_window_shadow",
            "error": "窗影格式不完整，未保存：" + ", ".join(validation_errors),
        }
    scene_records, scene_error = _window_shadow_scene_records(
        text,
        scenes,
        markdown_import=markdown_import,
    )
    if scene_error:
        return {"status": "invalid", "reason": "invalid_scene", "error": scene_error}
    try:
        requested_continue_scene_index = int(continue_scene_index or 0)
    except (TypeError, ValueError):
        return {
            "status": "invalid",
            "reason": "invalid_continue_scene_index",
            "error": "continue_scene_index 必须是 scenes 中从 1 开始的序号，或留空/传 0。",
        }
    if requested_continue_scene_index < 0 or requested_continue_scene_index > len(scene_records):
        return {
            "status": "invalid",
            "reason": "invalid_continue_scene_index",
            "error": (
                "continue_scene_index 超出本次 Scene 范围："
                f"收到 {requested_continue_scene_index}，本次共有 {len(scene_records)} 条 Scene。"
            ),
        }
    resolved_continue_scene_index = (
        requested_continue_scene_index
        if requested_continue_scene_index
        else 1
        if len(scene_records) == 1
        else 0
    )

    source_date = local_date_key(date) if str(date or "").strip() else _handoff_today_key()
    source_date = source_date or _handoff_today_key()
    profile_cfg = config.get("persona", {}) if isinstance(config.get("persona", {}), dict) else {}
    safe_profile_id = str(profile_id or profile_cfg.get("profile_id") or "default").strip() or "default"
    gateway_cfg = config.get("gateway", {}) if isinstance(config.get("gateway", {}), dict) else {}
    transport_session_id = str(
        session_id or gateway_cfg.get("default_session_id") or "main"
    ).strip() or "main"
    window_session = gateway_state_store.resolve_window_conversation(
        profile_id=safe_profile_id,
        transport_session_id=transport_session_id,
        rotate_closed=False,
    )
    if str(window_session.get("status") or "") in {"invalid", "conflict"}:
        return {
            "status": "error",
            "reason": "window_resolution_failed",
            "error": str(window_session.get("reason") or "window_resolution_failed"),
            "transport_session_id": transport_session_id,
        }
    conversation_id = str(window_session.get("conversation_id") or "").strip()
    if not conversation_id:
        return {
            "status": "error",
            "reason": "window_resolution_failed",
            "error": "window_resolution_missing_conversation_id",
            "transport_session_id": transport_session_id,
        }
    planned = window_shadow_store.plan(text, session_id=conversation_id)
    parent_shadow_id = str(
        (window_session or {}).get("parent_shadow_id") or ""
    ).strip()
    planned_window = {
        **planned,
        "session_id": conversation_id,
        "profile_id": safe_profile_id,
        "parent_shadow_id": parent_shadow_id,
        "source_date": source_date,
    }
    existing_window = window_shadow_store.get(planned["window_id"])
    created_scene_ids: list[str] = []
    scene_bucket_ids: list[str] = []
    window_created = False
    try:
        # Scenes are validated first and written before the Shadow becomes visible.
        for index, scene in enumerate(scene_records, start=1):
            bucket_id, action = await _write_window_shadow_scene(planned_window, scene, index)
            scene_bucket_ids.append(bucket_id)
            if action == "created":
                created_scene_ids.append(bucket_id)
        continue_scene_id = (
            scene_bucket_ids[resolved_continue_scene_index - 1]
            if resolved_continue_scene_index
            else ""
        )

        if existing_window:
            window = existing_window
        else:
            window, window_created = window_shadow_store.write(
                text,
                session_id=conversation_id,
                profile_id=safe_profile_id,
                parent_shadow_id=parent_shadow_id,
                source_date=source_date,
                sections=sections,
            )
        window = window_shadow_store.attach_scene_buckets(
            str(window.get("window_id") or planned["window_id"]),
            scene_bucket_ids,
            continue_scene_id=continue_scene_id,
        ) or window
        lifecycle = gateway_state_store.close_window_session(
            profile_id=safe_profile_id,
            conversation_id=conversation_id,
            shadow_id=str(window.get("window_id") or planned["window_id"]),
        )
        if lifecycle.get("status") in {"invalid", "conflict"}:
            raise RuntimeError(
                f"window_lifecycle_{lifecycle.get('reason') or lifecycle.get('status')}"
            )
    except Exception as exc:
        if window_created:
            window_shadow_store.delete(planned["window_id"])
        rollback_errors = []
        for bucket_id in reversed(created_scene_ids):
            try:
                result = await _delete_bucket_and_indexes(bucket_id)
                if result.get("status") != "deleted":
                    rollback_errors.append(f"{bucket_id}:{result.get('status')}")
            except Exception as rollback_exc:
                rollback_errors.append(f"{bucket_id}:{rollback_exc}")
        logger.warning("close_window rolled back after failure / 关窗写入失败并回滚: %s", exc)
        return {
            "status": "error",
            "reason": "atomic_write_failed",
            "error": str(exc),
            "rollback_errors": rollback_errors,
            "window_id": planned["window_id"],
        }

    for bucket_id in created_scene_ids:
        _queue_embedding_refresh(bucket_id)
        _queue_scene_linking(bucket_id)
    return {
        "status": "created" if window_created else "existing",
        "window_id": str(window.get("window_id") or planned["window_id"]),
        "conversation_id": conversation_id,
        "transport_session_id": transport_session_id,
        "profile_id": safe_profile_id,
        "parent_shadow_id": parent_shadow_id,
        "source_hash": str(window.get("source_hash") or planned["source_hash"]),
        "handoff_note_chars": handoff_note_char_count(sections.get("handoff", "")),
        "handoff_ready": bool(str(sections.get("handoff") or "").strip()),
        "scene_bucket_ids": scene_bucket_ids,
        "scene_count": len(scene_bucket_ids),
        "created_scene_count": len(created_scene_ids),
        "continue_scene_index": resolved_continue_scene_index,
        "continue_scene_id": continue_scene_id,
        "ordinary_recall": False,
        "markdown_import": markdown_import,
    }


@mcp.tool()
async def close_window(
    shadow: str,
    scenes: list[str] | None = None,
    session_id: str = "",
    profile_id: str = "",
    date: str = "",
    source: str = "",
    continue_scene_index: int = 0,
    context: Context | None = None,
) -> dict:
    """窗口结束时只调用这一次：原子保存一篇完整第一人称 Window Shadow，并同时保存其中明确写出的 0~N 个独立 Scene，不要在关窗后再次调用工具抽取 Scene。新窗影必须含 `## 给下个窗口的我`，正文为 250–400 个非空白字符：写这一窗真正改变了什么、未完线头和我想怎样继续；下个窗口会逐字注入这段，不做二次摘要。Shadow 全文仍原样保存且不进普通召回。session_id 可以继续传客户端固定身份（默认 main），Gateway 会把它解析为当前内部窗口；下次正常聊天自动换窗并认领这篇 Shadow。profile_id 是长期身份，通常沿用配置默认值。scenes 数组中的每个元素直接是一段原文经历，不带 `## Scene` / `### scene` / `### moment`；若 Scene 写在 Shadow 的 `## 想留下的记忆` 内，`### scene` 只作为抽取标记。多条 Scene 中若有下一窗说“继续吧”时应优先下钻的未完主线，传从 1 开始的 continue_scene_index；只有一条 Scene 时会自动认领，ID 由本次事务生成并写回窗影，无需调用者预知。后来才产生的新理解用 annotate 挂回来源。已经写好的 Markdown 用 source="markdown_import" 只转成窗影，不强制补 handoff_note，也不补造 Scene。"""
    _ = context
    resolved_source = str(source or "").strip().lower()
    if resolved_source in {"operit", "ob-auto-grow", "auto", "workflow", "worker"}:
        return {
            "status": "invalid",
            "reason": "automatic_shadow_rejected",
            "error": "close_window 只保存当前 AI 亲自写好的窗影，不接受旧自动摘要来源。",
        }
    return await _close_window_commit(
        shadow,
        scenes=scenes,
        continue_scene_index=continue_scene_index,
        session_id=session_id,
        profile_id=profile_id,
        date=date,
        source=source,
    )


async def grow(
    content: str,
    auto: bool = False,
    source: str = "",
    title: str = "",
    session_id: str = "",
    date: str = "",
    context: Context | None = None,
) -> str:
    """`close_window` 的兼容别名。原样保存整篇第一人称窗影，只拆显式 `### scene`；旧 `### moment` 仍留在原窗影中供回看，但不会自动升格为普通召回 Scene。新调用请改用 close_window。"""
    _ = (title, context)
    resolved_source = str(source or "").strip().lower()
    if _bool_value(auto, False) or resolved_source in {
        "operit",
        "ob-auto-grow",
        "auto",
        "workflow",
        "worker",
    }:
        return "grow 已退为 close_window 的兼容别名，不接受旧 auto-grow/自动摘要来源。"
    result = await _close_window_commit(
        content,
        session_id=session_id,
        date=date,
        source=source,
        require_handoff_note=False,
    )
    if result.get("status") in {"invalid", "error"}:
        return str(result.get("error") or result.get("reason") or "窗影保存失败。")
    action_label = "新窗影" if result.get("status") == "created" else "已有窗影"
    lines = [
        f"{action_label}→{result.get('window_id')}",
        "整篇原文已保存；普通召回=关闭；"
        f"拆出 Scene={result.get('scene_count', 0)}（新建{result.get('created_scene_count', 0)}）",
        "grow 仅为兼容别名；后续请使用 close_window。",
    ]
    if result.get("markdown_import"):
        lines.append("Markdown 导入只保存为窗影与 handoff 投影，不补造普通 Scene。")
    elif not result.get("scene_count"):
        lines.append("没有显式 `### scene`，因此没有创建普通记忆；旧 `### moment` 只保留在窗影原文中。")
    return "\n".join(lines)


async def _read_window_shadow_memory(
    window_id: str = "",
    limit: int = 10,
    include_content: bool = True,
) -> dict:
    """回看整篇窗影或列出最近窗影；完整窗影不参与普通 recall，handoff 只投影流动自我与最近关系原文。"""
    window_id = str(window_id or "").strip()
    if window_id:
        if not MEMORY_ID_RE.fullmatch(window_id):
            return {"status": "invalid", "reason": "invalid_window_id", "window_id": window_id}
        item = window_shadow_store.get(window_id)
        return {
            "status": "ok" if item else "not_found",
            "mode": "window_shadow",
            "window": item,
            "ordinary_recall": False,
        }
    return {
        "status": "ok",
        "mode": "window_shadow",
        "stats": window_shadow_store.stats(),
        "windows": window_shadow_store.list(
            limit=_int_between(limit, 10, 1, 100),
            include_content=bool(include_content),
        ),
        "ordinary_recall": False,
    }


window_shadow_read = _read_window_shadow_memory


# =============================================================
# Tool 3.5: profile_fact — manually solidify a user/profile fact
# 工具 3.5：profile_fact — 手动固化画像事实
# =============================================================
async def profile_fact(
    fact: str,
    evidence_bucket_id: str,
    profile_kind: str = "preference",
    subject: str = "user",
    predicate: str = "",
    object_value: str = "",
    evidence_moment_id: str = "",
    confidence: float = 0.9,
) -> str:
    """手动写入一条证据化 ProfileFact 索引。必须先有 evidence Scene；Fact 不进入普通候选、不参与扩散，也不会在 handoff 中单独倾倒。显式事实查询命中它时，系统沿 evidence 返回真正的 Scene。后来产生的新理解请写到证据 Scene 的带时间年轮，不写进 Fact。"""
    if not legacy_memory_writes_enabled(config):
        return retired_write_message("ProfileFact")
    fact = str(fact or "").strip()
    evidence_bucket_id = str(evidence_bucket_id or "").strip()
    if not fact:
        return "fact 为空，无法写入画像事实。"
    if not evidence_bucket_id or not MEMORY_ID_RE.fullmatch(evidence_bucket_id):
        return "请提供有效的 evidence_bucket_id。"

    evidence_bucket = await bucket_mgr.get(evidence_bucket_id)
    if not evidence_bucket:
        return f"证据记忆桶不存在: {evidence_bucket_id}"
    if not _is_canonical_scene_bucket(evidence_bucket):
        return (
            "evidence_bucket_id 必须指向一条 canonical Scene。"
            "旧 body / moment 仍可召回，但要先人工确认并提升为 Scene，才能支撑新的 ProfileFact。"
        )

    evidence_moment_id = str(evidence_moment_id or "").strip()
    if evidence_moment_id and not MEMORY_ID_RE.fullmatch(evidence_moment_id):
        return "evidence_moment_id 无效。"
    if evidence_moment_id:
        evidence_moment = memory_moment_store.get(evidence_moment_id)
        if (
            not evidence_moment
            or str(evidence_moment.get("bucket_id") or "") != evidence_bucket_id
            or evidence_moment.get("section") != "scene"
        ):
            return "evidence_moment_id 必须指向 evidence bucket 里的 Scene，年轮和 reflection 不能作为 Fact 证据。"
    if not evidence_moment_id:
        try:
            evidence_moments = memory_moment_store.upsert_bucket(evidence_bucket)
            representative = _representative_moment(evidence_moments)
            evidence_moment_id = str((representative or {}).get("moment_id") or "")
        except Exception as e:
            logger.warning("Profile fact evidence moment indexing failed: %s", e)
            evidence_moment_id = ""

    kind = _profile_key(profile_kind, "preference")
    subject_key = _profile_key(subject, "user")
    predicate_key = _profile_key(predicate, "")
    object_text = str(object_value or "").strip()
    confidence = _float_between(confidence, 0.9, 0.0, 1.0)
    body = _profile_fact_body(fact=fact)
    tags = ["profile_fact", f"profile_{kind}"]
    if predicate_key:
        tags.append(f"profile_predicate_{predicate_key}")
    evidence = {"bucket_id": evidence_bucket_id}
    if evidence_moment_id:
        evidence["moment_id"] = evidence_moment_id

    bucket_id = await bucket_mgr.create(
        content=body,
        tags=list(dict.fromkeys(tags)),
        importance=8,
        domain=list(dict.fromkeys(["profile", kind])),
        valence=0.5,
        arousal=0.3,
        name=_profile_fact_name(fact),
        bucket_type="permanent",
        confidence=confidence,
        source="profile_fact",
        extra_metadata={
            "profile_kind": kind,
            "subject": subject_key,
            "predicate": predicate_key,
            "object": object_text,
            "evidence": [evidence],
        },
    )
    _queue_embedding_refresh(bucket_id)
    try:
        created_bucket = await bucket_mgr.get(bucket_id)
        if created_bucket:
            memory_moment_store.upsert_bucket(created_bucket)
    except Exception as e:
        logger.warning("Profile fact moment indexing failed: %s", e)

    moment_note = f" moment={evidence_moment_id}" if evidence_moment_id else ""
    return f"profile_fact→{bucket_id} evidence→{evidence_bucket_id}{moment_note} index_only"


def _profile_fact_body(
    *,
    fact: str,
    evidence_context: str = "",
    reflection: str = "",
    followup: str = "",
) -> str:
    sections = [("fact", fact)]
    if str(evidence_context or "").strip():
        sections.append(("evidence_context", str(evidence_context).strip()))
    if str(reflection or "").strip():
        sections.append(("reflection", str(reflection).strip()))
    if str(followup or "").strip():
        sections.append(("followup", str(followup).strip()))
    return "\n\n".join(f"### {heading}\n{text.strip()}" for heading, text in sections)


def _profile_key(value: str, default: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return default
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^0-9a-zA-Z_\-\u4e00-\u9fff]+", "", text)
    return text or default


def _profile_fact_name(fact: str) -> str:
    return "画像事实：" + _clip_text(fact, 48)


# =============================================================
# Tool 4: trace — Trace, redraw the outline of a memory
# 工具 4：trace — 描摹，重新勾勒记忆的轮廓
# Also handles deletion (delete=True)
# 同时承接删除功能
# =============================================================
async def trace(
    bucket_id: str,
    name: str = "",
    domain: str = "",
    valence: float = -1,
    arousal: float = -1,
    importance: int = -1,
    tags: str = "",
    resolved: int = -1,
    pinned: int = -1,
    anchor: int = -1,
    digested: int = -1,
    content: str = "",
    date: str = "",
    delete: bool = False,
) -> str:
    """修改已有记忆，不创建新桶。tags/domain/content 是替换；date 可改事件日期；改前先 read_bucket。resolved/digested 让旧事沉底。只改元数据/date 不重建 embedding，改 content/name 才重建。"""

    bucket_id = _coerce_memory_id(bucket_id)
    if not bucket_id:
        return "请提供有效的 bucket_id。"

    # --- Delete mode / 删除模式 ---
    if delete:
        result = await _delete_bucket_and_indexes(bucket_id)
        return (
            f"已遗忘记忆桶: {bucket_id}"
            if result.get("status") == "deleted"
            else f"未找到记忆桶: {bucket_id}"
        )

    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return f"未找到记忆桶: {bucket_id}"

    # --- Collect only fields actually passed / 只收集用户实际传入的字段 ---
    updates = {}
    if name:
        updates["name"] = name
    if domain:
        updates["domain"] = [d.strip() for d in domain.split(",") if d.strip()]
    if 0 <= valence <= 1:
        updates["valence"] = valence
    if 0 <= arousal <= 1:
        updates["arousal"] = arousal
    if 1 <= importance <= 10:
        updates["importance"] = importance
    if tags:
        updates["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
    if resolved in (0, 1):
        updates["resolved"] = bool(resolved)
    if pinned in (0, 1):
        updates["pinned"] = bool(pinned)
        if pinned == 1:
            updates["importance"] = 10  # pinned → lock importance
    if anchor in (0, 1):
        if anchor == 1:
            ok, message = await _can_mark_anchor(bucket_id, bucket)
            if not ok:
                return message
        updates["anchor"] = bool(anchor)
    if digested in (0, 1):
        updates["digested"] = bool(digested)
    if content:
        updates["content"] = content
    event_date = str(date or "").strip()
    if event_date:
        updates["date"] = event_date

    if not updates:
        return "没有任何字段需要修改。"

    effective_tags = updates.get("tags", bucket.get("metadata", {}).get("tags", []))
    effective_content = updates.get("content", bucket.get("content", ""))
    if _has_favorite_tag(effective_tags) and not _has_favorite_reason(effective_content):
        return _favorite_reason_error()

    before_bucket = bucket
    success = await bucket_mgr.update(bucket_id, **updates)
    if not success:
        return f"修改失败: {bucket_id}"

    # Re-generate embedding if content or title changed.
    if "content" in updates or "name" in updates:
        after_bucket = await bucket_mgr.get(bucket_id)
        _queue_embedding_refresh_if_changed(bucket_id, before_bucket, after_bucket)

    changed = ", ".join(f"{k}={v}" for k, v in updates.items() if k != "content")
    if "content" in updates:
        changed += (", content=已替换" if changed else "content=已替换")
    # Explicit hint about resolved state change semantics
    # 特别提示 resolved 状态变化的语义
    if "resolved" in updates:
        if updates["resolved"]:
            changed += " → 已沉底，只在关键词触发时重新浮现"
        else:
            changed += " → 已重新激活，将参与浮现排序"
    if "digested" in updates:
        if updates["digested"]:
            changed += " → 已隐藏，保留但不再浮现"
        else:
            changed += " → 已取消隐藏，重新参与浮现"
    if "anchor" in updates:
        changed += " → 已标为 anchor" if updates["anchor"] else " → 已取消 anchor"
    return f"已修改记忆桶 {bucket_id}: {changed}"


# =============================================================
# Tool 5: pulse — Heartbeat, system status + memory listing
# 工具 5：pulse — 脉搏，系统状态 + 记忆列表
# =============================================================
async def pulse(include_archive: bool = False) -> str:
    """只读查看系统状态和记忆桶摘要；用于盘点和找 read_bucket/trace 候选。"""
    try:
        stats = await bucket_mgr.get_stats()
    except Exception as e:
        return f"获取系统状态失败: {e}"

    active_count = stats["permanent_count"] + stats["dynamic_count"] + stats["feel_count"]
    total_count = active_count + stats["archive_count"]
    visible_count = total_count if include_archive else active_count
    status = (
        f"=== Ombre Brain 记忆系统 ===\n"
        f"固化记忆桶: {stats['permanent_count']} 个\n"
        f"动态记忆桶: {stats['dynamic_count']} 个\n"
        f"情绪/印象桶: {stats['feel_count']} 个\n"
        f"归档记忆桶: {stats['archive_count']} 个\n"
        f"当前显示桶: {visible_count} 个\n"
        f"全量记忆桶: {total_count} 个\n"
        f"总存储大小: {stats['total_size_kb']:.1f} KB\n"
        f"衰减引擎: {'运行中' if decay_engine.is_running else '已停止'}\n"
    )

    # --- List all bucket summaries / 列出所有桶摘要 ---
    try:
        buckets = await bucket_mgr.list_all(include_archive=include_archive)
    except Exception as e:
        return status + f"\n列出记忆桶失败: {e}"

    if not buckets:
        return status + "\n记忆库为空。"

    lines = []
    for b in buckets:
        meta = b.get("metadata", {})
        if meta.get("pinned") or meta.get("protected"):
            icon = "📌"
        elif meta.get("anchor"):
            icon = "⚓"
        elif meta.get("type") == "permanent":
            icon = "📦"
        elif meta.get("type") == "feel":
            icon = "🫧"
        elif meta.get("type") == "archived":
            icon = "🗄️"
        elif meta.get("resolved", False):
            icon = "✅"
        else:
            icon = "💭"
        try:
            score = decay_engine.calculate_score(meta)
        except Exception:
            score = 0.0
        domains = ",".join(meta.get("domain", []))
        val = meta.get("valence", 0.5)
        aro = meta.get("arousal", 0.3)
        resolved_tag = " [已解决]" if meta.get("resolved", False) else ""
        lines.append(
            f"{icon} [{meta.get('name', b['id'])}]{resolved_tag} "
            f"bucket_id:{b['id']} "
            f"主题:{domains} "
            f"情感:V{val:.1f}/A{aro:.1f} "
            f"重要:{meta.get('importance', '?')} "
            f"权重:{score:.2f} "
            f"标签:{','.join(meta.get('tags', []))}"
        )

    return status + "\n=== 记忆列表 ===\n" + "\n".join(lines)


# =============================================================
# Tool 6: introspection — waking self-reflection over recent memories
# 工具 6：introspection — 清醒自省，消化最近的记忆
#
# Reads recent surface-level buckets (≤10), returns them for
# Claude to introspect under prompt guidance.
# 读取最近新增的表层桶（≤10个），返回给 Claude 在提示词引导下自主思考。
# Claude then decides: resolve some, write feels, or do nothing.
# =============================================================
async def introspection(
    limit: int = 10,
    offset: int = 0,
    created_date: str = "",
    created_from: str = "",
    created_to: str = "",
) -> str:
    """读取最近普通记忆供自省；可按日期翻页。放下用 trace，产生新感受用 comment_bucket。feel content 只写第一人称感受。"""
    await decay_engine.ensure_started()
    limit = _int_between(limit, 10, 1, 30)
    offset = _int_between(offset, 0, 0, 10000)
    date_args = {
        "created_date": created_date,
        "created_from": created_from,
        "created_to": created_to,
    }
    if any(str(value or "").strip() and not _date_key(value) for value in date_args.values()):
        return '创建日期格式请用 YYYY-MM-DD, 例如 introspection(created_date="2026-05-24")。'

    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
    except Exception as e:
        logger.error(f"Introspection failed to list buckets: {e}")
        return "记忆系统暂时无法访问。"

    # --- Filter: recent surface-level dynamic buckets (not permanent/pinned/feel) ---
    candidates = [
        b for b in all_buckets
        if b["metadata"].get("type") not in ("permanent", "feel")
        and not b["metadata"].get("pinned", False)
        and not b["metadata"].get("protected", False)
    ]

    candidates, date_filter_label = _filter_by_created_date(
        candidates,
        created_date=created_date,
        created_from=created_from,
        created_to=created_to,
    )

    # --- Sort by creation time desc, take requested page ---
    candidates.sort(key=lambda b: b["metadata"].get("created", ""), reverse=True)
    recent = candidates[offset : offset + limit]

    if not recent:
        if date_filter_label:
            return "这个创建日期范围内没有需要消化的新记忆。"
        return "没有需要消化的新记忆。"

    parts = []
    for b in recent:
        meta = b["metadata"]
        resolved_tag = " [已解决]" if meta.get("resolved", False) else " [未解决]"
        domains = ",".join(meta.get("domain", []))
        val = meta.get("valence", 0.5)
        aro = meta.get("arousal", 0.3)
        created = meta.get("created", "")
        parts.append(
            f"[{meta.get('name', b['id'])}]{resolved_tag} "
            f"主题:{domains} V{val:.1f}/A{aro:.1f} "
            f"创建:{created}\n"
            f"ID: {b['id']}\n"
            f"{_bucket_text_for_embedding(b)[:500]}"
        )

    header = (
        "=== Introspection ===\n"
        f"以下是你最近的普通记忆（offset={offset}, limit={limit}{date_filter_label}）。用第一人称想：\n"
        "- 这些东西里有什么在你这里留下了重量？\n"
        "- 有什么还没想清楚？\n"
        "- 有什么可以放下了？\n"
        "想完之后：值得放下的用 trace(bucket_id, resolved=1)；\n"
        "有沉淀的用 comment_bucket(bucket_id=\"bucket_id\", content=\"...\", kind=\"feel\", valence=你的感受) 写成年轮；content 只写第一人称感受，不补事件，不写分段标题。\n"
        "valence 是你对这段记忆的感受，不是事件本身的情绪。\n"
        "没有沉淀就不写，不强迫产出。\n"
    )

    # --- Connection hint: find most similar pair via embeddings ---
    connection_hint = ""
    if embedding_engine and embedding_engine.enabled and len(recent) >= 2:
        try:
            best_pair = None
            best_sim = 0.0
            ids = [b["id"] for b in recent]
            names = {b["id"]: b["metadata"].get("name", b["id"]) for b in recent}
            embeddings = {}
            for bid in ids:
                emb = await embedding_engine.get_embedding(bid)
                if emb is not None:
                    embeddings[bid] = emb
            for i, id_a in enumerate(ids):
                for id_b in ids[i+1:]:
                    if id_a in embeddings and id_b in embeddings:
                        sim = embedding_engine._cosine_similarity(embeddings[id_a], embeddings[id_b])
                        if sim > best_sim:
                            best_sim = sim
                            best_pair = (id_a, id_b)
            if best_pair and best_sim > 0.5:
                connection_hint = (
                    f"\n💭 [{names[best_pair[0]]}] 和 [{names[best_pair[1]]}] "
                    f"似乎有关联 (相似度:{best_sim:.2f})——不替你下结论，你自己想。\n"
                )
        except Exception as e:
            logger.warning(f"Introspection connection hint failed: {e}")

    # --- Feel crystallization hint: detect repeated feel themes ---
    crystal_hint = ""
    if embedding_engine and embedding_engine.enabled:
        try:
            feels = [b for b in all_buckets if b["metadata"].get("type") == "feel"]
            if len(feels) >= 3:
                feel_embeddings = {}
                for f in feels:
                    emb = await embedding_engine.get_embedding(f["id"])
                    if emb is not None:
                        feel_embeddings[f["id"]] = emb
                # Find clusters: feels with similarity > 0.7 to at least 2 others
                for fid, femb in feel_embeddings.items():
                    similar_feels = []
                    for oid, oemb in feel_embeddings.items():
                        if oid != fid:
                            sim = embedding_engine._cosine_similarity(femb, oemb)
                            if sim > 0.7:
                                similar_feels.append(oid)
                    if len(similar_feels) >= 2:
                        feel_bucket = next((f for f in feels if f["id"] == fid), None)
                        if feel_bucket and not feel_bucket["metadata"].get("pinned"):
                            content_preview = strip_wikilinks(feel_bucket["content"][:80])
                            crystal_hint = (
                                f"\n🔮 你已经写过 {len(similar_feels)+1} 条相似的 feel "
                                f"（围绕「{content_preview}…」）。"
                                f"如果这已经是确信而不只是感受了，"
                                f"你可以用 hold(content=\"...\", pinned=True) 升级它。"
                                f"不急，你自己决定。\n"
                            )
                            break
        except Exception as e:
            logger.warning(f"Introspection crystallization hint failed: {e}")

    profile_hint = (
        _profile_fact_candidate_hint(recent, all_buckets)
        if legacy_memory_writes_enabled(config)
        else ""
    )

    return header + "\n---\n".join(parts) + connection_hint + crystal_hint + profile_hint


PROFILE_FACT_CANDIDATE_PATTERN_SPECS = (
    ("preference", "likes", "喜欢", r"\s*(?:很|最|一直|特别|偏)?喜欢\s*([^。；;，,\n]{1,32})"),
    ("preference", "dislikes", "不喜欢", r"\s*(?:很|最|一直|特别)?不喜欢\s*([^。；;，,\n]{1,32})"),
    ("preference", "dislikes", "讨厌", r"\s*(?:很|最|一直|特别)?讨厌\s*([^。；;，,\n]{1,32})"),
    ("preference", "dislikes", "厌恶", r"\s*(?:很|最|一直|特别)?厌恶\s*([^。；;，,\n]{1,32})"),
    ("preference", "fears", "害怕", r"\s*(?:很|最|一直|特别)?害怕\s*([^。；;，,\n]{1,32})"),
    ("preference", "prefers", "偏好", r"\s*偏好\s*([^。；;，,\n]{1,32})"),
    ("boundary", "boundary", "雷点", r"的?雷点是\s*([^。；;，,\n]{1,32})"),
    ("habit", "habit", "习惯", r"(?:有个)?习惯是\s*([^。；;，,\n]{1,32})"),
)


BASE_NOISY_PROFILE_OBJECT_KEYS = {
    "哥哥",
    "老公",
    "宝宝",
    "宝贝",
    "老婆",
    "亲爱的",
    "你",
    "你啦",
    "你呀",
}


def _profile_fact_candidate_hint(recent: list[dict], all_buckets: list[dict]) -> str:
    existing = _existing_profile_fact_keys(all_buckets)
    candidates = []
    seen = set()
    patterns = _profile_fact_candidate_patterns()
    for bucket in recent:
        if len(candidates) >= 3:
            break
        meta = bucket.get("metadata", {}) or {}
        if "profile_fact" in {str(tag) for tag in meta.get("tags", []) or []}:
            continue
        text = strip_wikilinks(_bucket_text_for_embedding(bucket))
        for kind, predicate, verb, pattern in patterns:
            match = pattern.search(text)
            if not match:
                continue
            obj = _clean_profile_object(match.group(1))
            if not obj:
                continue
            if _is_noisy_profile_object(predicate, obj):
                continue
            fact = _render_profile_fact_candidate(predicate, verb, obj)
            key = _normalize_profile_fact_key(fact)
            if key in existing or key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "fact": fact,
                    "bucket_id": bucket.get("id", ""),
                    "profile_kind": kind,
                    "predicate": predicate,
                    "object_value": obj,
                    "evidence_context": _clip_text(text, 180),
                }
            )
            break
    if not candidates:
        return ""

    lines = [
        "\n=== 可能值得固化的画像事实 ===",
        "只作为候选，不会自动写入；确认后再调用 profile_fact(...)。",
    ]
    for item in candidates:
        args = [
            f"fact={_literal_arg(item['fact'])}",
            f"evidence_bucket_id={_literal_arg(item['bucket_id'])}",
            f"profile_kind={_literal_arg(item['profile_kind'])}",
            f"predicate={_literal_arg(item['predicate'])}",
            f"object_value={_literal_arg(item['object_value'])}",
            f"evidence_context={_literal_arg(item['evidence_context'])}",
        ]
        lines.append(
            f"- {item['fact']}\n"
            f"  证据桶: {item['bucket_id']}\n"
            f"  建议: profile_fact({', '.join(args)})"
        )
    return "\n" + "\n".join(lines) + "\n"


def _profile_fact_candidate_patterns() -> tuple[tuple[str, str, str, re.Pattern], ...]:
    subject = _profile_fact_subject_pattern()
    return tuple(
        (kind, predicate, verb, re.compile(subject + tail))
        for kind, predicate, verb, tail in PROFILE_FACT_CANDIDATE_PATTERN_SPECS
    )


def _profile_fact_subject_pattern() -> str:
    identity = _identity()
    values = [
        identity.get("user_display_name"),
        identity.get("user_name"),
        *(identity.get("user_aliases") or []),
        "用户",
        "她",
    ]
    terms = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in terms:
            terms.append(text)
    escaped = [re.escape(term) for term in sorted(terms, key=len, reverse=True)]
    return "(?:" + "|".join(escaped or ["用户", "她"]) + ")"


def _existing_profile_fact_keys(buckets: list[dict]) -> set[str]:
    keys = set()
    for bucket in buckets or []:
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        tags = {str(tag) for tag in meta.get("tags", []) or []}
        if "profile_fact" not in tags and not meta.get("profile_kind"):
            continue
        content = str(bucket.get("content") or "")
        for line in content.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                keys.add(_normalize_profile_fact_key(stripped))
                break
    return keys


def _render_profile_fact_candidate(predicate: str, verb: str, obj: str) -> str:
    user_name = _identity().get("user_display_name") or _identity().get("user_name") or "用户"
    if predicate == "boundary":
        return f"{user_name}的雷点是{obj}。"
    if predicate == "habit":
        return f"{user_name}的习惯是{obj}。"
    return f"{user_name}{verb}{obj}。"


def _clean_profile_object(value: str) -> str:
    text = strip_wikilinks(str(value or "")).strip()
    text = re.sub(r"^[“\"'「『（(]+|[”\"'」』）)]+$", "", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"(这件事|这个设定|这类东西|的时候)$", "", text)
    return text[:32].strip("。；;，,、 ")


def _is_noisy_profile_object(predicate: str, obj: str) -> bool:
    key = _normalize_profile_fact_key(obj)
    if not key or len(key) <= 1:
        return True
    return predicate in {"likes", "dislikes", "prefers"} and key in _noisy_profile_object_keys()


def _noisy_profile_object_keys() -> set[str]:
    keys = set(BASE_NOISY_PROFILE_OBJECT_KEYS)
    ai_name = str(_identity().get("ai_name") or "").strip()
    if ai_name:
        keys.add(ai_name)
        keys.add(f"小{ai_name}")
    return {_normalize_profile_fact_key(key) for key in keys if str(key or "").strip()}


def _normalize_profile_fact_key(value: str) -> str:
    return re.sub(r"[\s。；;，,、：:\"'“”‘’「」『』]+", "", str(value or "").lower())


def _literal_arg(value: str) -> str:
    return _json_lib.dumps(str(value or ""), ensure_ascii=False)


async def dream() -> str:
    """兼容旧客户端。旧 dream() 已改名为 introspection(); 夜梦由后台小模型自动生成。"""
    result = await introspection()
    return "dream() 已改名为 introspection()。夜梦由后台小模型自动生成，不需要主动调用工具。\n\n" + result


# =============================================================
# Tool 5.8: import_diary_source — lossless diary evidence snapshot
# 工具 5.8：import_diary_source — 无损日记证据快照
# =============================================================
async def import_diary_source(date: str, title: str = "") -> dict:
    """从 Haven-Diary 读取一篇日记，并逐字保存为不可变 source_record。只在需要给 Scene、Annotation 或 Narrative Roll 补逐字证据时调用；不会调用模型、不会脱水或改写、不会生成普通记忆，也不参与普通召回。相同正文重复导入会幂等返回原记录；日记正文发生修订时会新建 snapshot，保留旧版并用 supersedes_source_record_id 串联。date 必须为 YYYY-MM-DD；同日多篇时传精确 title。"""
    safe_date = str(date or "").strip()
    safe_title = str(title or "").strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", safe_date):
        return {
            "status": "invalid",
            "reason": "date_must_be_yyyy_mm_dd",
            "date": safe_date,
        }
    diary = await reflection_engine.read_diary_source(
        date=safe_date,
        title=safe_title,
    )
    if not diary:
        return {
            "status": "not_found",
            "reason": "diary_not_found_or_unavailable",
            "date": safe_date,
            "title": safe_title,
        }
    return await diary_source_importer.import_snapshot(
        diary,
        bucket_mgr,
        requested_date=safe_date,
        requested_title=safe_title,
    )


# =============================================================
# Tool 6: reflect — daily relationship weather
# 工具 6：reflect — 生成日印象
# =============================================================
async def reflect(period: str = "daily", force: bool = False) -> dict:
    """旧日/周印象兼容入口。默认拒绝新增并保留旧数据只读；仅显式启用 OMBRE_LEGACY_MEMORY_WRITES 时恢复旧流程。"""
    await decay_engine.ensure_started()
    return await reflection_engine.reflect(
        period=period,
        bucket_mgr=bucket_mgr,
        persona_engine=persona_engine,
        embedding_engine=embedding_engine,
        force=force,
        conversation_turn_store=gateway_state_store,
    )


async def portrait_maintain(force: bool = False, scope: str = "") -> dict:
    """整理每日 portrait state 与 stable candidates。模型不能发布 Stable；Stable 只接受当前 Haven 审阅证据后的本人定稿。不会写 profile_fact、anchor、pinned、protected 或 Core Memory。"""
    await decay_engine.ensure_started()
    force_scopes = [str(scope or "").strip()] if str(scope or "").strip() else []
    return await portrait_engine.maintain_daily(
        bucket_mgr,
        persona_engine,
        force=force,
        force_scopes=force_scopes,
    )


def _portrait_evidence_input_rows(value: object) -> list[dict]:
    if isinstance(value, (str, dict)):
        value = [value]
    if not isinstance(value, list):
        return []
    rows = []
    for item in value:
        if isinstance(item, str):
            clean = item.strip()
            if clean:
                rows.append({"bucket_id": clean})
        elif isinstance(item, dict):
            rows.append(dict(item))
    return rows


def _portrait_window_evidence(value: str) -> dict | None:
    key = str(value or "").strip()
    if not key:
        return None
    direct = window_shadow_store.get(key)
    if direct:
        return direct
    return next(
        (
            row
            for row in window_shadow_store.list(limit=200, include_content=True)
            if str(row.get("session_id") or "").strip() == key
        ),
        None,
    )


async def _resolve_portrait_publication_evidence(
    scope: str,
    evidence: object,
) -> tuple[list[dict], list[str], list[dict], list[dict]]:
    """Validate refs and let User Portrait follow Shadow/Roll refs to Scenes."""
    normalized: list[dict] = []
    source_dates: list[str] = []
    resolved: list[dict] = []
    errors: list[dict] = []
    for raw in _portrait_evidence_input_rows(evidence):
        bucket_id = str(raw.get("bucket_id") or raw.get("id") or "").strip()
        narrative_id = str(raw.get("narrative_id") or "").strip()
        window_ref = str(raw.get("window_id") or raw.get("session_id") or "").strip()
        if bucket_id:
            bucket = await bucket_mgr.get(bucket_id)
            if not bucket:
                errors.append({"ref": bucket_id, "reason": "bucket_not_found"})
                continue
            if not _is_canonical_scene_bucket(bucket):
                errors.append({"ref": bucket_id, "reason": "portrait_evidence_requires_scene"})
                continue
            meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
            row = {"bucket_id": bucket_id}
            moment_id = str(raw.get("moment_id") or "").strip()
            if moment_id:
                row["moment_id"] = moment_id
            normalized.append(row)
            date_key = _bucket_handoff_date(bucket)
            if date_key:
                source_dates.append(date_key)
            resolved.append(
                {
                    "type": "scene",
                    "bucket_id": bucket_id,
                    "title": str(meta.get("name") or bucket_id),
                    "date": date_key,
                    "content": str(bucket.get("content") or ""),
                }
            )
            continue

        if narrative_id:
            roll = narrative_roll_store.read(narrative_id)
            if str(roll.get("status") or "") != "ok":
                errors.append(
                    {
                        "ref": narrative_id,
                        "reason": "narrative_unavailable",
                        "integrity_status": str(roll.get("integrity_status") or ""),
                    }
                )
                continue
            normalized.append({"narrative_id": narrative_id})
            source_dates.extend(
                str(roll.get(key) or "").strip()
                for key in ("time_end", "time_start")
                if str(roll.get(key) or "").strip()
            )
            resolved.append(
                {
                    "type": "narrative_roll",
                    "narrative_id": narrative_id,
                    "title": str(roll.get("title") or narrative_id),
                    "time_start": str(roll.get("time_start") or ""),
                    "time_end": str(roll.get("time_end") or ""),
                    "linked_scene_ids": list(roll.get("linked_scene_ids") or []),
                    "body": str(roll.get("body") or ""),
                }
            )
            continue

        if window_ref:
            window = _portrait_window_evidence(window_ref)
            if not window:
                errors.append({"ref": window_ref, "reason": "window_shadow_not_found"})
                continue
            session_id = str(window.get("session_id") or window.get("window_id") or "").strip()
            source_date = str(window.get("source_date") or "").strip()
            if scope == "user":
                linked_scene_ids = list(window.get("scene_bucket_ids") or [])
                linked_count = 0
                for scene_id in linked_scene_ids:
                    scene = await bucket_mgr.get(str(scene_id or "").strip())
                    if not scene or not _is_canonical_scene_bucket(scene):
                        continue
                    normalized.append({"bucket_id": str(scene_id)})
                    scene_date = _bucket_handoff_date(scene)
                    if scene_date:
                        source_dates.append(scene_date)
                    scene_meta = (
                        scene.get("metadata", {})
                        if isinstance(scene.get("metadata"), dict)
                        else {}
                    )
                    resolved.append(
                        {
                            "type": "scene",
                            "bucket_id": str(scene_id),
                            "title": str(scene_meta.get("name") or scene_id),
                            "date": scene_date,
                            "content": str(scene.get("content") or ""),
                            "via_window_shadow": str(window.get("window_id") or ""),
                        }
                    )
                    linked_count += 1
                if linked_count == 0:
                    errors.append(
                        {
                            "ref": window_ref,
                            "reason": "user_window_shadow_requires_linked_scene",
                        }
                    )
                    continue
            else:
                normalized.append({"session_id": session_id, "role": "window_shadow"})
                if source_date:
                    source_dates.append(source_date)
                resolved.append(
                    {
                        "type": "window_shadow",
                        "window_id": str(window.get("window_id") or ""),
                        "session_id": session_id,
                        "date": source_date,
                        "content": str(window.get("content") or ""),
                        "linked_scene_ids": list(window.get("scene_bucket_ids") or []),
                    }
                )
            continue
        errors.append({"ref": raw, "reason": "unsupported_evidence_ref"})

    normalized = portrait_engine._normalize_evidence(normalized)
    source_dates = sorted(
        {str(value or "").strip() for value in source_dates if str(value or "").strip()},
        reverse=True,
    )[:8]
    return normalized, source_dates, resolved[:8], errors


async def _resolved_portrait_evidence(rows: object, *, include_text: bool) -> list[dict]:
    output = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        bucket_id = str(row.get("bucket_id") or "").strip()
        narrative_id = str(row.get("narrative_id") or "").strip()
        session_id = str(row.get("session_id") or "").strip()
        if bucket_id:
            bucket = await bucket_mgr.get(bucket_id)
            meta = bucket.get("metadata", {}) if isinstance((bucket or {}).get("metadata"), dict) else {}
            item = {
                "type": "scene",
                "bucket_id": bucket_id,
                "available": bool(bucket),
                "title": str(meta.get("name") or bucket_id),
                "date": _bucket_handoff_date(bucket) if bucket else "",
            }
            if include_text and bucket:
                item["content"] = str(bucket.get("content") or "")
            output.append(item)
        elif narrative_id:
            roll = narrative_roll_store.read(narrative_id)
            item = {
                "type": "narrative_roll",
                "narrative_id": narrative_id,
                "available": str(roll.get("status") or "") == "ok",
                "title": str(roll.get("title") or narrative_id),
                "time_start": str(roll.get("time_start") or ""),
                "time_end": str(roll.get("time_end") or ""),
                "linked_scene_ids": list(roll.get("linked_scene_ids") or []),
            }
            if include_text and item["available"]:
                item["body"] = str(roll.get("body") or "")
            output.append(item)
        elif session_id:
            window = _portrait_window_evidence(session_id)
            item = {
                "type": "window_shadow",
                "session_id": session_id,
                "available": bool(window),
                "window_id": str((window or {}).get("window_id") or ""),
                "date": str((window or {}).get("source_date") or ""),
                "linked_scene_ids": list((window or {}).get("scene_bucket_ids") or []),
            }
            if include_text and window:
                item["content"] = str(window.get("content") or "")
            output.append(item)
    return output


@mcp.tool()
async def read_portrait(scope: str = "", include_evidence_text: bool = True) -> dict:
    """读取 Haven 已审阅或待审的 User / Relationship Portrait。返回完整 stable、revision、source、证据、候选与历史；source=model 只代表候选材料，不代表已发布。scope 可为 user、relationship 或留空读取两者。明确审阅画像时使用，不要把 Current Self 画像重新造一份。"""
    result = portrait_engine.read_reviewed_portrait(scope=scope)
    if str(result.get("status") or "") != "ok":
        return result
    for item in (result.get("scopes") or {}).values():
        if not isinstance(item, dict):
            continue
        item["resolved_evidence"] = await _resolved_portrait_evidence(
            item.get("evidence", []),
            include_text=bool(include_evidence_text),
        )
    return result


async def _publish_portrait_memory(
    scope: str,
    text: str,
    expected_revision: int,
    evidence: list[dict] | None = None,
    locked: bool = True,
) -> dict:
    """由当前 Haven 亲手发布一版 User 或 Relationship Portrait。必须先 read_portrait，传 expected_revision，并给出 Scene、Narrative Roll 或 Window Shadow evidence。Relationship 可直接引用窗影；User 引用窗影时工具会沿 linked_scene_ids 回到 Scene，避免把一次观察冒充小雨事实。模型候选不会自动发布。"""
    safe_scope = str(scope or "").strip().lower()
    normalized, source_dates, resolved, errors = await _resolve_portrait_publication_evidence(
        safe_scope,
        evidence or [],
    )
    if errors:
        return {
            "status": "invalid",
            "reason": "invalid_evidence",
            "scope": safe_scope,
            "errors": errors,
            "resolved_evidence": resolved,
        }
    result = portrait_engine.publish_reviewed_portrait(
        scope=safe_scope,
        text=text,
        expected_revision=expected_revision,
        evidence=normalized,
        source_dates=source_dates,
        locked=locked,
    )
    result["resolved_evidence"] = resolved
    return result


publish_portrait_patch = _publish_portrait_memory


async def _write_scene_memory(
    content: str,
    *,
    title: str = "",
    date: str = "",
    domain: str = "",
    cues: str = "",
) -> str:
    """Store one authored Scene without entering the retired hold policy."""
    await decay_engine.ensure_started()
    if not content or not content.strip():
        return "内容为空，无法存储。"
    scene_error = _hold_scene_contract_error(content)
    if scene_error:
        return f"写入被拒绝：{scene_error}"

    scene_title = _authored_scene_title(content, title)
    scene_cues = _authored_scene_cues(content, title=scene_title, explicit=cues)
    requested_domain = [
        item.strip() for item in str(domain or "").split(",") if item.strip()
    ] or ["未分类"]
    classification = normalize_write_classification(
        memory_subject="",
        memory_layer="",
        tags=[],
        content=content,
    )
    bucket_id, result_name, is_merged, related_bucket = await _merge_or_create(
        content=content,
        tags=[],
        importance=5,
        domain=requested_domain,
        valence=0.5,
        arousal=0.3,
        name=scene_title,
        allow_merge=False,
        memory_subject=classification["memory_subject"],
        memory_layer=classification["memory_layer"],
        memory_classification_source=classification["memory_classification_source"],
        date=str(date or "").strip(),
        source="write_scene",
        bucket_id=generate_scene_id(),
        extra_metadata={
            "memory_value_source": "authored_scene",
            "object_kind": "scene",
            "write_contract": "write-scene-v1",
            "scene_cues": scene_cues or None,
        },
        normalize_content=False,
    )
    _queue_scene_linking(bucket_id)
    action = "合并→" if is_merged else "新建→"
    related_note = (
        _format_readonly_related_memory(related_bucket) if related_bucket else ""
    )
    return f"{action}{result_name} {','.join(requested_domain)}{related_note}"


# =============================================================
# Daily MCP façade — the eight actions advertised to chat models
# 日常 MCP façade —— 普通聊天模型只看见这八个动作
# =============================================================
@mcp.tool()
async def recall(
    query: str = "",
    mode: str = "memory",
    date: str = "",
    max_results: int = 2,
    max_tokens: int = 3000,
    session_id: str = "",
    previous_session_id: str = "",
    parent_shadow_id: str = "",
) -> str:
    """普通召回与相邻窗口交接的统一入口。mode="memory" 按 query/date 找 Scene 与受限叙事投影；mode="handoff" 读取 self anchor、直接父 Window Shadow 和未完线头。不会把 Portrait 自动塞进启动包；明确画像问题仍由现有显式门槛处理。"""
    safe_mode = str(mode or "memory").strip().lower()
    if safe_mode not in {"memory", "handoff"}:
        return "mode 只能是 memory 或 handoff。"
    return await _recall_memory(
        query=query,
        date=date,
        max_results=_int_between(max_results, 2, 1, 20),
        max_tokens=_int_between(max_tokens, 3000, 0, 20000),
        mode="handoff" if safe_mode == "handoff" else "",
        session_id=session_id,
        previous_session_id=previous_session_id,
        parent_shadow_id=parent_shadow_id,
    )


@mcp.tool()
async def read_memory(
    memory_id: str = "",
    memory_type: Literal["", "scene", "shadow", "narrative"] = "",
    query: str = "",
    limit: int = 20,
    include_content: bool = True,
) -> dict:
    """统一精确读取 Scene、Window Shadow 或 Narrative Roll。有 memory_id 时可由 scene id、window_*、narrative_* 推断类型；只有 query 时必须显式传 memory_type，绝不因同名猜测类型。memory_type="narrative" 且 query="叙事卷" 列目录，精确卷名或别名一次返回完整卷，模糊或冲突查询只返回候选。语义找 Scene 请用 recall。"""
    safe_id = _coerce_memory_id(memory_id)
    safe_type = str(memory_type or "").strip().lower()
    if safe_type in {"scene", "memory", "bucket"}:
        safe_type = "scene"
    elif safe_type in {"shadow", "window", "window_shadow"}:
        safe_type = "shadow"
    elif safe_type in {"narrative", "roll", "narrative_roll", "arc"}:
        safe_type = "narrative"
    elif safe_type:
        return {"status": "invalid", "reason": "unknown_memory_type", "memory_type": safe_type}
    elif safe_id.startswith("window_"):
        safe_type = "shadow"
    elif safe_id.startswith("narrative_"):
        safe_type = "narrative"
    elif not safe_id and str(query or "").strip():
        return {
            "status": "needs_memory_type",
            "reason": "query_only_read_must_choose_memory_type",
            "query": str(query or "").strip(),
            "allowed_memory_types": ["scene", "shadow", "narrative"],
        }
    else:
        safe_type = "scene"

    if safe_type == "narrative":
        if safe_id:
            return await _read_narrative_memory(safe_id)
        return narrative_roll_store.resolve_read_query(query=query, limit=limit)
    if safe_type == "shadow":
        return await _read_window_shadow_memory(
            window_id=safe_id,
            limit=limit,
            include_content=include_content,
        )
    if not safe_id:
        return {
            "status": "invalid",
            "reason": "scene_id_required_use_recall_for_search",
            "memory_type": "scene",
        }
    return await _read_scene_memory(safe_id)


@mcp.tool()
async def write_scene(
    content: str,
    title: str = "",
    date: str = "",
    domain: str = "",
    cues: str = "",
) -> str:
    """原样写一条具体 Scene。正文就是完整经历，不加 Markdown 类型标题；工具不脱水、不改写、不合并，也不能借此写 feel、whisper、日印象或 ProfileFact。"""
    return await _write_scene_memory(
        content,
        title=title,
        date=date,
        domain=domain,
        cues=cues,
    )


@mcp.tool()
async def annotate(
    source_id: str,
    content: str,
    kind: str = "comment",
) -> dict:
    """给一条已有来源追加带时间 Annotation。用于后来形成的新理解、修正或有来源的感受；Annotation 始终挂回来源，不独立扩散。"""
    return await _append_annotation(
        source_id,
        content,
        kind=kind,
        source="annotate",
    )


@mcp.tool()
async def publish_narrative(
    narrative_id: str,
    document: str,
    expected_revision: int,
    title: str,
    source_scene_ids: list[str],
    title_aliases: list[str] | None = None,
    primary_entities: list[str] | None = None,
    supporting_entities: list[str] | None = None,
    intent_tags: list[str] | None = None,
    query_cues: list[str] | None = None,
    time_start: str = "",
    time_end: str = "",
    current_status_cue: str = "",
    publication_status: str = "reviewed",
    lifecycle: str = "active",
) -> dict:
    """由当前 Haven 亲手发布、修订或封卷一份 Narrative Roll。document 必须是完整第一人称有来源 Markdown，至少列出两条 canonical Scene 及其逐字正文 hash；query_cues 是这一卷自己携带的审阅后路由提示，不存在全局主题词表；expected_revision=0 创建，之后必须带当前 revision。工具只保存传入原文并版本化，不调用模型、不改 Scene。"""
    exact_document = str(document or "")
    linked_ids = narrative_roll_store.source_scene_ids(exact_document, source_scene_ids)
    resolved_sources: list[dict] = []
    errors: list[dict] = []
    for scene_id in linked_ids:
        if not MEMORY_ID_RE.fullmatch(scene_id):
            errors.append({"scene_id": scene_id, "reason": "invalid_scene_id"})
            continue
        scene = await bucket_mgr.get(scene_id)
        if not scene:
            errors.append({"scene_id": scene_id, "reason": "scene_not_found"})
            continue
        metadata = scene.get("metadata", {}) if isinstance(scene.get("metadata"), dict) else {}
        if (
            not _is_canonical_scene_bucket(scene)
            or metadata.get("active") is False
            or metadata.get("deprecated")
            or metadata.get("resolved")
            or metadata.get("digested")
        ):
            errors.append({"scene_id": scene_id, "reason": "not_active_canonical_scene"})
            continue
        content_hash = hashlib.sha256(str(scene.get("content") or "").encode("utf-8")).hexdigest()
        if content_hash not in exact_document:
            errors.append(
                {
                    "scene_id": scene_id,
                    "reason": "scene_content_hash_missing_from_document",
                    "content_sha256": content_hash,
                }
            )
            continue
        resolved_sources.append(
            {
                "scene_id": scene_id,
                "title": str(metadata.get("name") or scene_id),
                "date": str(metadata.get("date") or ""),
                "content_sha256": content_hash,
            }
        )
    if errors:
        return {
            "status": "invalid",
            "reason": "source_verification_failed",
            "narrative_id": str(narrative_id or "").strip(),
            "errors": errors,
            "resolved_sources": resolved_sources,
        }
    result = narrative_roll_store.publish(
        narrative_id=narrative_id,
        document=exact_document,
        expected_revision=expected_revision,
        title=title,
        source_scene_ids=linked_ids,
        title_aliases=title_aliases,
        primary_entities=primary_entities,
        supporting_entities=supporting_entities,
        intent_tags=intent_tags,
        query_cues=query_cues,
        time_start=time_start,
        time_end=time_end,
        current_status_cue=current_status_cue,
        publication_status=publication_status,
        lifecycle=lifecycle,
    )
    result["resolved_sources"] = resolved_sources
    return result


@mcp.tool()
async def publish_portrait(
    scope: str,
    text: str,
    expected_revision: int,
    evidence: list[dict] | None = None,
    locked: bool = True,
) -> dict:
    """由当前 Haven 在 read_portrait 审阅后，带 optimistic revision 与可验证 evidence 发布 User 或 Relationship Portrait。模型候选不会自动发布。"""
    return await _publish_portrait_memory(
        scope=scope,
        text=text,
        expected_revision=expected_revision,
        evidence=evidence,
        locked=locked,
    )


async def _self_anchor_entry_payload() -> dict:
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
    except Exception as e:
        logger.warning("Portrait self-anchor entry list failed / 画像自我入口列桶失败: %s", e)
        return {}
    bucket = _select_self_anchor_entry_bucket(all_buckets)
    if not bucket:
        return {}
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    text = _self_anchor_body_text(bucket, include_reflection=True, max_chars=1200)
    return {
        "bucket_id": bucket.get("id", ""),
        "name": meta.get("name") or SELF_ANCHOR_TAG,
        "text": text,
        "configured": bool(_self_anchor_entry_bucket_id()),
        "updated_at": meta.get("updated_at") or meta.get("last_active") or meta.get("created", ""),
    }


async def _portrait_state_payload() -> dict:
    evidence_health = {}
    if hasattr(portrait_engine, "reconcile_evidence"):
        evidence_health = await portrait_engine.reconcile_evidence(bucket_mgr)
    state = portrait_engine.load_state()
    handoff_sections = {}
    if hasattr(portrait_engine, "build_handoff_sections"):
        try:
            handoff_sections = portrait_engine.build_handoff_sections(max_recent_items=3)
        except Exception as exc:
            logger.warning("Portrait handoff preview failed: %s", exc)
    recent_continuity_status = _recent_continuity_fallback_status(handoff_sections)
    try:
        window_shadow_stats = window_shadow_store.stats()
    except Exception as exc:
        logger.warning("Window shadow dashboard stats failed: %s", exc)
        window_shadow_stats = {}
    return {
        "state_path": getattr(portrait_engine, "state_path", ""),
        "enabled": bool(getattr(portrait_engine, "enabled", True)),
        "auto_enabled": bool(getattr(portrait_engine, "auto_enabled", True)),
        "auto_initial_enabled": bool(getattr(portrait_engine, "auto_initial_enabled", False)),
        "daily_enabled": bool(getattr(portrait_engine, "daily_enabled", True)),
        "generator_ready": bool(getattr(portrait_engine, "client", None)),
        "generator_model": str(getattr(portrait_engine, "model", "") or ""),
        "generator_source": str(getattr(portrait_engine, "model_source", "dehydration") or "dehydration"),
        "updated_at": state.get("updated_at", ""),
        "last_run_date": state.get("last_run_date", ""),
        "portrait": state.get("portrait", {}),
        "recent_activities": state.get("recent_activities", []),
        "recent_timeline": state.get("recent_timeline", []),
        "current_focus_items": (
            portrait_engine.current_focus_items(max_items=8)
            if hasattr(portrait_engine, "current_focus_items")
            else state.get("recent_activities", [])
        ),
        "current_focus": str(handoff_sections.get("current_focus") or ""),
        "recent_continuity": str(handoff_sections.get("recent_continuity") or ""),
        "recent_continuity_status": recent_continuity_status,
        "handoff_policy": {
            "order": ["window_shadow", "recent_continuity", "raw_events"],
            "recent_continuity_max_age_hours": HANDOFF_RECENT_CONTINUITY_MAX_AGE_HOURS,
            "window_shadow_section": "handoff_note",
        },
        "last_handoff": dict(_last_handoff_status),
        "window_shadow_stats": window_shadow_stats,
        "stable_candidates": state.get("stable_candidates", []),
        "profile_fact_candidates": state.get("profile_fact_candidates", []),
        "generation_status": (
            {
                scope: portrait_engine.scope_generation_status(scope)
                for scope in ("user", "persona", "relationship")
            }
            if hasattr(portrait_engine, "scope_generation_status")
            else {}
        ),
        "evidence_health": evidence_health,
        "self_anchor_entry": await _self_anchor_entry_payload(),
    }


async def portrait_state() -> dict:
    """读取当前 portrait state，供检查 handoff 画像来源。"""
    return await _portrait_state_payload()


# =============================================================
# Dashboard API endpoints (for lightweight Web UI)
# 仪表板 API（轻量 Web UI 用）
# =============================================================
@mcp.custom_route("/api/memories", methods=["POST"])
async def api_create_memory(request):
    """Create or update one memory bucket from a trusted C-side client."""
    from starlette.responses import JSONResponse

    if not _memory_write_token():
        return JSONResponse({"error": "memory write token is not configured"}, status_code=503)
    if not _authorized_memory_write(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "json body must be an object"}, status_code=400)

    title = str(body.get("title") or body.get("name") or "").strip()
    content = str(body.get("content") or "").strip()
    if not title:
        return JSONResponse({"error": "missing title"}, status_code=400)
    if not content:
        return JSONResponse({"error": "missing content"}, status_code=400)
    content = _normalize_memory_sections_for_write(content)

    requested_id = body.get("id")
    bucket_id = str(requested_id).strip() if requested_id else None
    if bucket_id and not MEMORY_ID_RE.fullmatch(bucket_id):
        return JSONResponse({"error": "invalid id"}, status_code=400)

    bucket_type = str(body.get("type") or "dynamic").strip()
    if bucket_type not in {"dynamic", "permanent", "feel"}:
        return JSONResponse({"error": "invalid type"}, status_code=400)

    now = _current_time_iso()
    domain = _string_list(body.get("domain"), ["general"])
    tags = _string_list(body.get("tags"), [])
    if _has_favorite_tag(tags) and not _has_favorite_reason(content):
        return JSONResponse({"error": _favorite_reason_error()}, status_code=400)
    importance = _int_between(body.get("importance"), 5)
    valence = _float_between(body.get("valence"), 0.5)
    arousal = _float_between(body.get("arousal"), 0.5)
    confidence = _float_between(body.get("confidence"), 0.5)
    pinned = _bool_value(body.get("pinned"), False)
    protected = _bool_value(body.get("protected"), False)
    anchor = _bool_value(body.get("anchor"), False)
    self_anchor = _bool_value(body.get("self_anchor"), False)
    resolved = _bool_value(body.get("resolved"), False)
    digested = _bool_value(body.get("digested"), False)
    event_date = str(body.get("date") or body.get("event_date") or "").strip()

    existing = await bucket_mgr.get(bucket_id) if bucket_id else None
    existing_meta = (
        existing.get("metadata", {})
        if isinstance(existing, dict) and isinstance(existing.get("metadata"), dict)
        else {}
    )
    retired_kind = ""
    normalized_tags = {str(tag).strip().lower() for tag in tags}
    if bucket_type == "feel" or existing_meta.get("type") == "feel":
        retired_kind = "feel"
    elif "profile_fact" in normalized_tags or (
        existing and _is_profile_fact_bucket(existing)
    ):
        retired_kind = "profile_fact"
    elif normalized_tags & {"whisper", "daily_impression", "weekly_impression", "relationship_weather"}:
        retired_kind = sorted(
            normalized_tags & {"whisper", "daily_impression", "weekly_impression", "relationship_weather"}
        )[0]
    if retired_kind and not legacy_memory_writes_enabled(config):
        return JSONResponse(retired_write_payload(retired_kind), status_code=410)
    if existing:
        before_bucket = existing
        update_kwargs = {
            "content": content,
            "tags": tags,
            "importance": importance,
            "domain": domain,
            "valence": valence,
            "arousal": arousal,
            "name": title,
            "resolved": resolved,
            "pinned": pinned,
            "anchor": anchor,
            "digested": digested,
            "confidence": confidence,
            "source": "chatgpt",
            "last_active": str(body.get("last_active") or now),
            "updated_at": str(body.get("updated_at") or now),
        }
        if "self_anchor" in body:
            update_kwargs["extra_metadata"] = {"self_anchor": self_anchor}
        if event_date:
            update_kwargs["date"] = event_date
        ok = await bucket_mgr.update(
            bucket_id,
            **update_kwargs,
        )
        if not ok:
            return JSONResponse({"error": "update failed"}, status_code=500)
        status = "updated"
        updated_bucket = await bucket_mgr.get(bucket_id)
        if embedding_engine.enabled:
            embedding_status = (
                "queued" if _queue_embedding_refresh_if_changed(bucket_id, before_bucket, updated_bucket) else "skipped"
            )
        else:
            embedding_status = "disabled"
    else:
        bucket_id = await bucket_mgr.create(
            content=content,
            tags=tags,
            importance=importance,
            domain=domain,
            valence=valence,
            arousal=arousal,
            bucket_type=bucket_type,
            name=title,
            pinned=pinned,
            protected=protected,
            anchor=anchor,
            resolved=resolved,
            digested=digested,
            confidence=confidence,
            bucket_id=bucket_id,
            source="chatgpt",
            created=str(body.get("created") or now),
            last_active=str(body.get("last_active") or now),
            updated_at=str(body.get("updated_at") or now),
            date=event_date or None,
            extra_metadata={"self_anchor": True} if self_anchor else None,
        )
        status = "created"
        if embedding_engine.enabled:
            embedding_status = "queued" if _queue_embedding_refresh(bucket_id) else "failed"
        else:
            embedding_status = "disabled"

    if bucket_type != "feel" and not is_self_anchor_metadata({"self_anchor": self_anchor, "domain": domain}):
        _queue_memory_enrichment(bucket_id)

    return JSONResponse({
        "status": status,
        "id": bucket_id,
        "source": "chatgpt",
        "embedding": embedding_status,
    })


@mcp.custom_route("/api/buckets", methods=["GET"])
async def api_buckets(request):
    """List all buckets with metadata (no content for efficiency)."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=True)
        result = []
        for b in all_buckets:
            meta = b.get("metadata", {})
            metadata_view = normalize_memory_metadata(b)
            result.append({
                "id": b["id"],
                "name": meta.get("name", b["id"]),
                "type": meta.get("type", "dynamic"),
                "object_kind": _bucket_object_kind(b),
                "domain": meta.get("domain", []),
                "tags": meta.get("tags", []),
                "facets": meta.get("facets", []),
                "metadata_view": metadata_view,
                **metadata_view,
                "source": meta.get("source", ""),
                "valence": meta.get("valence", 0.5),
                "arousal": meta.get("arousal", 0.3),
                "model_valence": meta.get("model_valence"),
                "importance": meta.get("importance", 5),
                "confidence": meta.get("confidence", 0.5),
                "resolved": meta.get("resolved", False),
                "pinned": meta.get("pinned", False),
                "protected": meta.get("protected", False),
                "anchor": meta.get("anchor", False),
                "digested": meta.get("digested", False),
                "self_anchor": is_self_anchor_bucket(b),
                "profile_kind": meta.get("profile_kind", ""),
                "memory_subject": meta.get("memory_subject", ""),
                "memory_layer": meta.get("memory_layer", ""),
                "period": meta.get("period"),
                "date": meta.get("date"),
                "event_date": meta.get("event_date"),
                "created": meta.get("created", ""),
                "last_active": meta.get("last_active", ""),
                "activation_count": meta.get("activation_count", 0),
                "comment_count": meta.get("comment_count", 0),
                "score": decay_engine.calculate_score(meta),
                "content_preview": strip_wikilinks(b.get("content", ""))[:200],
            })
        result.sort(key=_bucket_dashboard_sort_key, reverse=True)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/buckets/light", methods=["GET"])
async def api_buckets_light(request):
    """List lightweight bucket metadata without content previews."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        params = request.query_params
        include_archive = str(params.get("include_archive") or "").lower() in {"1", "true", "yes", "on"}
        limit = max(1, min(int(params.get("limit") or 500), 2000))
        offset = max(0, int(params.get("offset") or 0))
        all_buckets = await bucket_mgr.list_all(include_archive=include_archive)
        items = [_bucket_light_payload(bucket) for bucket in all_buckets]
        items.sort(key=_bucket_light_sort_key, reverse=True)
        return JSONResponse({
            "buckets": items[offset : offset + limit],
            "count": len(items),
            "include_archive": include_archive,
            "limit": limit,
            "offset": offset,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/domain-taxonomy", methods=["GET"])
async def api_domain_taxonomy(request):
    """Return canonical domain keys and display labels for dashboard controls."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    return JSONResponse({"domains": domain_options()})


@mcp.custom_route("/api/portrait-state", methods=["GET"])
async def api_portrait_state(request):
    """Read maintained portrait state for dashboard inspection."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        return JSONResponse(await _portrait_state_payload())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/portrait-maintain", methods=["POST"])
async def api_portrait_maintain(request):
    """Collect portrait state and candidates without publishing Stable heads."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    try:
        await decay_engine.ensure_started()
        scope = str(body.get("scope") or "").strip()
        if scope and scope not in {"user", "persona", "relationship"}:
            return JSONResponse({"error": "invalid scope"}, status_code=400)
        force = _bool_value(body.get("force"), False)
        maintain_kwargs = {"force": force}
        if scope:
            maintain_kwargs["force_scopes"] = [scope]
        elif force:
            maintain_kwargs["force_scopes"] = ["user", "persona", "relationship"]
        result = await portrait_engine.maintain_daily(
            bucket_mgr,
            persona_engine,
            **maintain_kwargs,
        )
        status_code = 409 if result.get("status") == "blocked" else 200
        return JSONResponse(result, status_code=status_code)
    except Exception as e:
        logger.warning("Portrait maintain API failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/portrait-state/items", methods=["DELETE"])
async def api_portrait_state_item_delete(request):
    """Delete one portrait state row or clear one maintained portrait paragraph."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "json body must be an object"}, status_code=400)
    if body.get("confirm") != "DELETE":
        return JSONResponse({"error": "confirmation required"}, status_code=400)

    raw_index = body.get("index")
    try:
        index = int(raw_index) if raw_index is not None and str(raw_index) != "" else None
    except (TypeError, ValueError):
        return JSONResponse({"error": "index must be an integer"}, status_code=400)
    result = portrait_engine.delete_state_item(
        area=str(body.get("area") or ""),
        scope=str(body.get("scope") or ""),
        layer=str(body.get("layer") or ""),
        index=index,
        text=str(body.get("text") or ""),
    )
    status = str(result.get("status") or "")
    if status == "deleted":
        return JSONResponse(result)
    if status == "not_found":
        return JSONResponse(result, status_code=404)
    if status == "conflict":
        return JSONResponse(result, status_code=409)
    return JSONResponse(result, status_code=400)


@mcp.custom_route("/api/portrait-state/items", methods=["POST"])
async def api_portrait_state_item_add(request):
    """Add one manual Current Focus item."""
    from starlette.responses import JSONResponse

    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "json body must be an object"}, status_code=400)
    if str(body.get("area") or "") != "recent_activities":
        return JSONResponse({"error": "only recent_activities can be added manually"}, status_code=400)
    result = portrait_engine.add_recent_activity(
        str(body.get("text") or ""),
        source_date=str(body.get("source_date") or ""),
    )
    return _portrait_mutation_response(result)


@mcp.custom_route("/api/portrait-state/items", methods=["PUT"])
async def api_portrait_state_item_edit(request):
    """Edit one Current Focus or portrait generation-evidence row."""
    from starlette.responses import JSONResponse

    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "json body must be an object"}, status_code=400)
    raw_index = body.get("index")
    try:
        index = int(raw_index) if raw_index is not None and str(raw_index) != "" else None
    except (TypeError, ValueError):
        return JSONResponse({"error": "index must be an integer"}, status_code=400)
    result = portrait_engine.edit_state_item(
        area=str(body.get("area") or ""),
        scope=str(body.get("scope") or ""),
        layer=str(body.get("layer") or ""),
        index=index,
        text=str(body.get("text") or ""),
        expected_text=str(body.get("expected_text") or ""),
    )
    return _portrait_mutation_response(result)


def _portrait_mutation_response(result: dict):
    from starlette.responses import JSONResponse

    status = str(result.get("status") or "")
    if status in {"updated", "unchanged"}:
        return JSONResponse(result)
    if status == "conflict":
        return JSONResponse(result, status_code=409)
    if status == "not_found":
        return JSONResponse(result, status_code=404)
    return JSONResponse(result, status_code=400)


def _portrait_expected_revision(body: dict) -> int | None:
    raw = body.get("expected_revision")
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


@mcp.custom_route("/api/portrait-state/stable", methods=["PUT"])
async def api_portrait_stable_edit(request):
    """Manually replace one stable portrait paragraph with optimistic locking."""
    from starlette.responses import JSONResponse

    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "json body must be an object"}, status_code=400)
    expected_revision = _portrait_expected_revision(body)
    if expected_revision is None:
        return JSONResponse({"error": "expected_revision is required"}, status_code=400)
    locked = _bool_value(body.get("locked"), False) if "locked" in body else None
    result = portrait_engine.edit_stable(
        scope=str(body.get("scope") or ""),
        text=str(body.get("text") or ""),
        expected_revision=expected_revision,
        locked=locked,
    )
    return _portrait_mutation_response(result)


@mcp.custom_route("/api/portrait-state/stable/lock", methods=["POST"])
async def api_portrait_stable_lock(request):
    """Lock or unlock one stable portrait paragraph."""
    from starlette.responses import JSONResponse

    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "json body must be an object"}, status_code=400)
    expected_revision = _portrait_expected_revision(body)
    if expected_revision is None or "locked" not in body:
        return JSONResponse({"error": "expected_revision and locked are required"}, status_code=400)
    result = portrait_engine.set_stable_lock(
        scope=str(body.get("scope") or ""),
        locked=_bool_value(body.get("locked"), False),
        expected_revision=expected_revision,
    )
    return _portrait_mutation_response(result)


@mcp.custom_route("/api/portrait-state/stable/rollback", methods=["POST"])
async def api_portrait_stable_rollback(request):
    """Restore one stable portrait history revision without discarding newer history."""
    from starlette.responses import JSONResponse

    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "json body must be an object"}, status_code=400)
    expected_revision = _portrait_expected_revision(body)
    try:
        target_revision = int(body.get("target_revision"))
    except (TypeError, ValueError):
        target_revision = None
    if expected_revision is None or target_revision is None:
        return JSONResponse(
            {"error": "expected_revision and target_revision are required"},
            status_code=400,
        )
    result = portrait_engine.rollback_stable(
        scope=str(body.get("scope") or ""),
        target_revision=target_revision,
        expected_revision=expected_revision,
    )
    return _portrait_mutation_response(result)


@mcp.custom_route("/api/portrait-state/reset", methods=["POST"])
async def api_portrait_state_reset(request):
    """Reset maintained portrait state so the next manual generation is an initial run."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "json body must be an object"}, status_code=400)
    if body.get("confirm") != "RESET":
        return JSONResponse({"error": "confirmation required"}, status_code=400)
    try:
        return JSONResponse(portrait_engine.reset_state())
    except Exception as e:
        logger.warning("Portrait state reset API failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/profile-facts", methods=["GET"])
async def api_profile_facts(request):
    """List evidence-bound profile facts for dashboard review."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=True)
        facts = [
            await _profile_fact_payload(bucket)
            for bucket in all_buckets
            if _is_profile_fact_bucket(bucket)
        ]
        facts.sort(
            key=lambda item: (
                item.get("state") == "active",
                str(item.get("updated_at") or item.get("last_active") or item.get("created") or ""),
            ),
            reverse=True,
        )
        return JSONResponse({"count": len(facts), "facts": facts})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/profile-facts/{bucket_id}", methods=["PATCH"])
async def api_profile_fact_update(request):
    """Confirm, edit, or deprecate a profile fact bucket."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err

    bucket_id = request.path_params["bucket_id"]
    if not bucket_id or not MEMORY_ID_RE.fullmatch(bucket_id):
        return JSONResponse({"error": "invalid bucket_id"}, status_code=400)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "json body must be an object"}, status_code=400)

    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return JSONResponse({"error": "not found"}, status_code=404)
    if not _is_profile_fact_bucket(bucket):
        return JSONResponse({"error": "not a profile_fact bucket"}, status_code=400)

    meta = bucket.get("metadata", {})
    action = str(body.get("action") or "").strip().lower()
    if action not in {"confirm", "deprecate", "edit"}:
        return JSONResponse({"error": "action must be confirm, deprecate, or edit"}, status_code=400)
    if action in {"confirm", "edit"} and not legacy_memory_writes_enabled(config):
        return JSONResponse(retired_write_payload("profile_fact"), status_code=410)

    updates: dict = {
        "last_active": meta.get("last_active") or meta.get("created"),
    }
    if action == "confirm":
        updates.update({"active": True, "deprecated": False, "resolved": False, "digested": False})
    elif action == "deprecate":
        updates.update({"active": False, "deprecated": True, "resolved": True, "digested": True})
    else:
        sections = _profile_fact_sections(bucket.get("content", ""))
        fact = str(body.get("fact", sections.get("fact", "")) or "").strip()
        if not fact:
            return JSONResponse({"error": "fact is required"}, status_code=400)
        kind = _profile_key(body.get("profile_kind", meta.get("profile_kind") or "preference"), "preference")
        subject = _profile_key(body.get("subject", meta.get("subject") or "user"), "user")
        predicate = _profile_key(body.get("predicate", meta.get("predicate") or ""), "")
        object_text = str(body.get("object", meta.get("object") or "") or "").strip()
        confidence = _float_between(body.get("confidence", meta.get("confidence")), 0.9, 0.0, 1.0)
        domain = list(dict.fromkeys(["profile", kind] + [
            str(item).strip()
            for item in meta.get("domain", []) or []
            if str(item).strip() and str(item).strip() not in {"profile", kind}
        ]))
        updates.update({
            "content": _profile_fact_body(fact=fact),
            "name": _profile_fact_name(fact),
            "tags": _profile_fact_tags(meta.get("tags", []), kind, predicate),
            "domain": domain,
            "profile_kind": kind,
            "subject": subject,
            "predicate": predicate,
            "object": object_text,
            "confidence": confidence,
            "source": meta.get("source") or "profile_fact",
        })

    before_bucket = bucket
    ok = await bucket_mgr.update(bucket_id, **updates)
    if not ok:
        return JSONResponse({"error": "update failed"}, status_code=500)

    updated_bucket = await bucket_mgr.get(bucket_id)
    if action == "edit":
        _queue_embedding_refresh_if_changed(bucket_id, before_bucket, updated_bucket)
        try:
            if updated_bucket:
                memory_moment_store.upsert_bucket(updated_bucket)
        except Exception as e:
            logger.warning("Profile fact moment reindex failed: %s", e)
    return JSONResponse({
        "status": action,
        "id": bucket_id,
        "fact": await _profile_fact_payload(updated_bucket),
    })


@mcp.custom_route("/api/profile-facts/{bucket_id}", methods=["DELETE"])
async def api_profile_fact_delete(request):
    """Hard-delete one profile fact bucket and clean its indexes."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err

    bucket_id = request.path_params["bucket_id"]
    if not bucket_id or not MEMORY_ID_RE.fullmatch(bucket_id):
        return JSONResponse({"error": "invalid bucket_id"}, status_code=400)
    try:
        body = await request.json()
    except Exception:
        body = {}
    if body is None:
        body = {}
    if not isinstance(body, dict):
        return JSONResponse({"error": "json body must be an object"}, status_code=400)
    if body.get("confirm") != "DELETE":
        return JSONResponse({"error": "confirmation required"}, status_code=400)

    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return JSONResponse({"error": "not found"}, status_code=404)
    if not _is_profile_fact_bucket(bucket):
        return JSONResponse({"error": "not a profile_fact bucket"}, status_code=400)
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    if meta.get("protected") or meta.get("pinned"):
        return JSONResponse({"error": "protected profile_fact cannot be deleted"}, status_code=403)

    result = await _delete_bucket_and_indexes(bucket_id)
    if result.get("status") != "deleted":
        return JSONResponse(result, status_code=500)
    return JSONResponse(result)


@mcp.custom_route("/api/profile-fact-proposals", methods=["POST"])
async def api_profile_fact_proposals(request):
    """Generate evidence-bound profile fact proposals from one bucket."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    if not legacy_memory_writes_enabled(config):
        return JSONResponse(retired_write_payload("profile_fact"), status_code=410)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "json body must be an object"}, status_code=400)

    bucket_id = str(body.get("bucket_id") or body.get("evidence_bucket_id") or "").strip()
    if not bucket_id or not MEMORY_ID_RE.fullmatch(bucket_id):
        return JSONResponse({"error": "invalid bucket_id"}, status_code=400)
    evidence_moment_id = str(body.get("evidence_moment_id") or body.get("moment_id") or "").strip()
    if evidence_moment_id and not MEMORY_ID_RE.fullmatch(evidence_moment_id):
        return JSONResponse({"error": "invalid evidence_moment_id"}, status_code=400)
    max_proposals = _int_between(body.get("max_proposals"), 3, 1, 3)

    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return JSONResponse({"error": "not found"}, status_code=404)
    if _is_profile_fact_bucket(bucket):
        return JSONResponse({"error": "profile_fact bucket cannot be evidence for proposal"}, status_code=400)
    if not _is_canonical_scene_bucket(bucket):
        return JSONResponse(
            {"error": "profile_fact proposals require a canonical Scene"},
            status_code=400,
        )
    if evidence_moment_id:
        evidence_moment = memory_moment_store.get(evidence_moment_id)
        if (
            not evidence_moment
            or str(evidence_moment.get("bucket_id") or "") != bucket_id
            or evidence_moment.get("section") != "scene"
        ):
            return JSONResponse(
                {"error": "evidence_moment_id must point to the canonical Scene"},
                status_code=400,
            )

    try:
        all_buckets = await bucket_mgr.list_all(include_archive=True)
        raw = await _call_profile_fact_proposal_model(
            bucket=bucket,
            evidence_moment_id=evidence_moment_id,
            max_proposals=max_proposals,
        )
        proposals, rejected = _parse_profile_fact_proposals(
            raw,
            evidence_bucket_id=bucket_id,
            evidence_moment_id=evidence_moment_id,
            existing_keys=_existing_profile_fact_keys(all_buckets),
            max_proposals=max_proposals,
        )
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=503)
    except Exception as e:
        logger.warning("Profile fact proposal failed: %s", e, exc_info=True)
        return JSONResponse({"error": f"proposal failed: {type(e).__name__}"}, status_code=502)

    meta = bucket.get("metadata", {})
    return JSONResponse({
        "status": "ok",
        "evidence": {
            "bucket_id": bucket_id,
            "moment_id": evidence_moment_id,
            "name": meta.get("name", bucket_id),
        },
        "proposals": proposals,
        "rejected": rejected,
        "model": getattr(dehydrator, "model", ""),
    })


@mcp.custom_route("/api/profile-fact-proposals/confirm", methods=["POST"])
async def api_profile_fact_proposal_confirm(request):
    """Confirm one proposal by writing through the existing profile_fact path."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    if not legacy_memory_writes_enabled(config):
        return JSONResponse(retired_write_payload("profile_fact"), status_code=410)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "json body must be an object"}, status_code=400)

    evidence_bucket_id = str(body.get("evidence_bucket_id") or "").strip()
    if not evidence_bucket_id or not MEMORY_ID_RE.fullmatch(evidence_bucket_id):
        return JSONResponse({"error": "invalid evidence_bucket_id"}, status_code=400)
    evidence_moment_id = str(body.get("evidence_moment_id") or "").strip()
    if evidence_moment_id and not MEMORY_ID_RE.fullmatch(evidence_moment_id):
        return JSONResponse({"error": "invalid evidence_moment_id"}, status_code=400)

    bucket = await bucket_mgr.get(evidence_bucket_id)
    if not bucket:
        return JSONResponse({"error": "evidence bucket not found"}, status_code=404)
    if not _is_canonical_scene_bucket(bucket):
        return JSONResponse(
            {"error": "evidence bucket must be a canonical Scene"},
            status_code=400,
        )

    proposal, reason = _normalize_profile_fact_proposal(
        body,
        evidence_bucket_id=evidence_bucket_id,
        evidence_moment_id=evidence_moment_id,
        existing_keys=_existing_profile_fact_keys(await bucket_mgr.list_all(include_archive=True)),
    )
    if not proposal:
        return JSONResponse({"error": reason or "invalid proposal"}, status_code=400)

    result = await profile_fact(
        fact=proposal["fact"],
        evidence_bucket_id=proposal["evidence_bucket_id"],
        profile_kind=proposal["profile_kind"],
        subject=proposal["subject"],
        predicate=proposal["predicate"],
        object_value=proposal["object"],
        evidence_moment_id=proposal["evidence_moment_id"],
        confidence=proposal["confidence"],
    )
    if not result.startswith("profile_fact→"):
        return JSONResponse({"error": result}, status_code=400)
    profile_id = result.split("profile_fact→", 1)[1].split(" ", 1)[0]
    created = await bucket_mgr.get(profile_id)
    return JSONResponse({
        "status": "created",
        "id": profile_id,
        "result": result,
        "fact": await _profile_fact_payload(created),
    })


@mcp.custom_route("/api/anchor-proposals", methods=["POST"])
async def api_anchor_proposals(request):
    """Generate one manual-confirm anchor proposal for an existing bucket."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "json body must be an object"}, status_code=400)

    bucket_id = str(body.get("bucket_id") or "").strip()
    if not bucket_id or not MEMORY_ID_RE.fullmatch(bucket_id):
        return JSONResponse({"error": "invalid bucket_id"}, status_code=400)

    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return JSONResponse({"error": "not found"}, status_code=404)

    meta = bucket.get("metadata", {})
    rejected: list[dict] = []
    reason = _anchor_proposal_static_rejection(bucket)
    if reason:
        rejected.append({"reason": reason, "bucket_id": bucket_id})
        return JSONResponse({
            "status": "ok",
            "bucket": _bucket_summary_payload(bucket),
            "proposals": [],
            "rejected": rejected,
            "model": getattr(dehydrator, "model", ""),
        })

    ok, gate_message = await _can_mark_anchor(bucket_id, bucket)
    if not ok:
        rejected.append({"reason": gate_message, "bucket_id": bucket_id})
        return JSONResponse({
            "status": "ok",
            "bucket": _bucket_summary_payload(bucket),
            "proposals": [],
            "rejected": rejected,
            "model": getattr(dehydrator, "model", ""),
        })

    try:
        raw = await _call_anchor_proposal_model(bucket=bucket)
        proposals, rejected = _parse_anchor_proposals(raw, bucket_id=bucket_id)
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=503)
    except Exception as e:
        logger.warning("Anchor proposal failed: %s", e, exc_info=True)
        return JSONResponse({"error": f"proposal failed: {type(e).__name__}"}, status_code=502)

    return JSONResponse({
        "status": "ok",
        "bucket": {
            **_bucket_summary_payload(bucket),
            "name": meta.get("name", bucket_id),
        },
        "proposals": proposals,
        "rejected": rejected,
        "model": getattr(dehydrator, "model", ""),
    })


@mcp.custom_route("/api/anchor-proposals/confirm", methods=["POST"])
async def api_anchor_proposal_confirm(request):
    """Confirm one anchor proposal by applying the existing trace(anchor=1) path."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "json body must be an object"}, status_code=400)

    bucket_id = str(body.get("bucket_id") or "").strip()
    if not bucket_id or not MEMORY_ID_RE.fullmatch(bucket_id):
        return JSONResponse({"error": "invalid bucket_id"}, status_code=400)

    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return JSONResponse({"error": "not found"}, status_code=404)
    if bucket.get("metadata", {}).get("anchor"):
        return JSONResponse({
            "status": "already_anchor",
            "id": bucket_id,
            "bucket": _bucket_summary_payload(bucket),
        })

    static_reason = _anchor_proposal_static_rejection(bucket)
    if static_reason:
        return JSONResponse({"error": static_reason}, status_code=400)

    proposal, reason = _normalize_anchor_proposal(body, bucket_id=bucket_id)
    if not proposal:
        return JSONResponse({"error": reason or "invalid proposal"}, status_code=400)

    result = await trace(bucket_id=bucket_id, anchor=1)
    if not result.startswith("已修改记忆桶"):
        return JSONResponse({"error": result}, status_code=400)

    updated = await bucket_mgr.get(bucket_id)
    return JSONResponse({
        "status": "anchored",
        "id": bucket_id,
        "result": result,
        "proposal": proposal,
        "bucket": _bucket_summary_payload(updated or bucket),
    })


@mcp.custom_route("/api/word-map", methods=["GET"])
async def api_word_map(request):
    """Return generic Word Map Lite diagnostics for dashboard review."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        nodes_limit = _int_between(request.query_params.get("nodes"), 50, 1, 500)
        edges_limit = _int_between(request.query_params.get("edges"), 50, 1, 500)
        return JSONResponse(_word_map_payload(nodes_limit, edges_limit))
    except Exception as e:
        logger.warning("Word Map diagnostics failed: %s", e, exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/word-map/rebuild", methods=["POST"])
async def api_word_map_rebuild(request):
    """Rebuild the generic Word Map Lite index from current buckets."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        body = await request.json()
    except Exception:
        body = {}
    if body is None:
        body = {}
    if not isinstance(body, dict):
        return JSONResponse({"error": "json body must be an object"}, status_code=400)
    try:
        include_archive = _bool_value(body.get("include_archive"), False)
        nodes_limit = _int_between(body.get("nodes"), 50, 1, 500)
        edges_limit = _int_between(body.get("edges"), 50, 1, 500)
        result = await _rebuild_word_map_index(
            word_map_store,
            bucket_mgr,
            include_archive=include_archive,
        )
        payload = _word_map_payload(nodes_limit, edges_limit)
        payload.update(result)
        return JSONResponse(payload)
    except Exception as e:
        logger.warning("Word Map rebuild failed: %s", e, exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/word-map/cards", methods=["GET"])
async def api_word_map_cards(request):
    """Return bucket evidence rows for one Word Map term."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    term = str(request.query_params.get("term", "") or "").strip()
    if not term:
        return JSONResponse({"error": "missing term parameter"}, status_code=400)
    limit = _int_between(request.query_params.get("limit"), 20, 1, 200)
    try:
        return JSONResponse({
            "term": term,
            "cards": word_map_store.cards_for_term(term, limit),
        })
    except Exception as e:
        logger.warning("Word Map cards lookup failed: %s", e, exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/identity-semantics", methods=["GET"])
async def api_identity_semantics(request):
    """Return private identity alias diagnostics for dashboard review."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        limit = _int_between(request.query_params.get("limit"), 100, 1, 1000)
        return JSONResponse(_identity_semantics_payload(limit))
    except Exception as e:
        logger.warning("Identity semantic diagnostics failed: %s", e, exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/identity-semantics/rebuild", methods=["POST"])
async def api_identity_semantics_rebuild(request):
    """Rebuild private identity alias evidence from current buckets."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        body = await request.json()
    except Exception:
        body = {}
    if body is None:
        body = {}
    if not isinstance(body, dict):
        return JSONResponse({"error": "json body must be an object"}, status_code=400)
    try:
        include_archive = _bool_value(body.get("include_archive"), False)
        limit = _int_between(body.get("limit"), 100, 1, 1000)
        buckets = await bucket_mgr.list_all(include_archive=include_archive)
        stats = identity_semantic_store.rebuild_alias_index(buckets)
        payload = _identity_semantics_payload(limit)
        payload.update({
            "status": "rebuilt",
            "bucket_count": len(buckets),
            "include_archive": include_archive,
            "stats": stats,
        })
        return JSONResponse(payload)
    except Exception as e:
        logger.warning("Identity semantic rebuild failed: %s", e, exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/buckets/delete", methods=["POST"])
async def api_buckets_delete(request):
    """Bulk-delete dashboard buckets and clean their indexes."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "json body must be an object"}, status_code=400)
    if body.get("confirm") != "DELETE":
        return JSONResponse({"error": "confirmation required"}, status_code=400)

    raw_ids = body.get("bucket_ids", [])
    if not isinstance(raw_ids, list) or not raw_ids:
        return JSONResponse({"error": "bucket_ids must be a non-empty list"}, status_code=400)
    if len(raw_ids) > 200:
        return JSONResponse({"error": "too many bucket_ids"}, status_code=400)

    seen: set[str] = set()
    bucket_ids: list[str] = []
    for raw_id in raw_ids:
        bucket_id = str(raw_id or "").strip()
        if bucket_id in seen:
            continue
        seen.add(bucket_id)
        bucket_ids.append(bucket_id)

    summary = {"deleted": 0, "skipped": 0, "not_found": 0, "invalid": 0, "failed": 0}
    results = []
    for bucket_id in bucket_ids:
        if not bucket_id or not MEMORY_ID_RE.fullmatch(bucket_id):
            summary["invalid"] += 1
            results.append({"id": bucket_id, "status": "invalid", "reason": "invalid_bucket_id"})
            continue

        bucket = await bucket_mgr.get(bucket_id)
        if not bucket:
            summary["not_found"] += 1
            results.append({"id": bucket_id, "status": "not_found", "reason": "not_found"})
            continue

        result = await _delete_bucket_and_indexes(bucket_id)
        status = str(result.get("status") or "failed")
        if status in summary:
            summary[status] += 1
        else:
            summary["failed"] += 1
        results.append(result)

    return JSONResponse({**summary, "results": results})


def _unique_clean_list(value, *, limit: int = 40) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = re.split(r"[,，、\n]+", value)
    elif isinstance(value, (list, tuple, set)):
        raw = value
    else:
        raw = [value]
    items: list[str] = []
    seen: set[str] = set()
    for item in raw:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        items.append(text[:64])
        if len(items) >= limit:
            break
    return items


def _merge_metadata_list(current, *, add: list[str] | None = None, remove: list[str] | None = None) -> tuple[list[str], bool]:
    values = _unique_clean_list(current, limit=120)
    original = list(values)
    remove_set = {str(item) for item in (remove or [])}
    if remove_set:
        values = [item for item in values if item not in remove_set]
    for item in add or []:
        if item not in values:
            values.append(item)
    return values, values != original


@mcp.custom_route("/api/buckets/bulk-update", methods=["POST"])
async def api_buckets_bulk_update(request):
    """Bulk-edit dashboard bucket metadata and archive state."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "json body must be an object"}, status_code=400)

    raw_ids = body.get("bucket_ids", [])
    if not isinstance(raw_ids, list) or not raw_ids:
        return JSONResponse({"error": "bucket_ids must be a non-empty list"}, status_code=400)
    if len(raw_ids) > 300:
        return JSONResponse({"error": "too many bucket_ids"}, status_code=400)

    bucket_ids: list[str] = []
    seen: set[str] = set()
    for raw_id in raw_ids:
        bucket_id = str(raw_id or "").strip()
        if bucket_id in seen:
            continue
        seen.add(bucket_id)
        bucket_ids.append(bucket_id)

    domain_key = ""
    if "domain" in body:
        domain_key = normalize_domain_key(body.get("domain"))
        if not domain_key:
            return JSONResponse({"error": "invalid domain"}, status_code=400)

    tags_add = _unique_clean_list(body.get("tags_add"))
    tags_remove = _unique_clean_list(body.get("tags_remove"))
    facets_add = _unique_clean_list(body.get("facets_add"))
    facets_remove = _unique_clean_list(body.get("facets_remove"))
    status = str(body.get("status") or "").strip().lower()
    if status and status not in {"archived", "active"}:
        return JSONResponse({"error": "status must be archived or active"}, status_code=400)

    if not any([domain_key, tags_add, tags_remove, facets_add, facets_remove, status]):
        return JSONResponse({"error": "no bulk operation requested"}, status_code=400)

    summary = {"matched": 0, "changed": 0, "unchanged": 0, "not_found": 0, "invalid": 0, "failed": 0}
    changed_ids: list[str] = []
    results: list[dict] = []

    for bucket_id in bucket_ids:
        if not bucket_id or not MEMORY_ID_RE.fullmatch(bucket_id):
            summary["invalid"] += 1
            results.append({"id": bucket_id, "status": "invalid", "reason": "invalid_bucket_id"})
            continue

        bucket = await bucket_mgr.get(bucket_id)
        if not bucket:
            summary["not_found"] += 1
            results.append({"id": bucket_id, "status": "not_found", "reason": "not_found"})
            continue
        summary["matched"] += 1
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}

        changed = False
        update_kwargs: dict = {}

        if domain_key and meta.get("domain") != [domain_key]:
            update_kwargs["domain"] = [domain_key]
            changed = True

        tags, tags_changed = _merge_metadata_list(meta.get("tags", []), add=tags_add, remove=tags_remove)
        if tags_changed:
            update_kwargs["tags"] = tags
            changed = True

        facets, facets_changed = _merge_metadata_list(meta.get("facets", []), add=facets_add, remove=facets_remove)
        if facets_changed:
            update_kwargs["facets"] = facets
            changed = True

        if update_kwargs:
            ok = await bucket_mgr.update(
                bucket_id,
                **update_kwargs,
                last_active=meta.get("last_active") or meta.get("created"),
            )
            if not ok:
                summary["failed"] += 1
                results.append({"id": bucket_id, "status": "failed", "reason": "update_failed"})
                continue

        if status == "archived":
            current = await bucket_mgr.get(bucket_id)
            current_type = ((current or {}).get("metadata") or {}).get("type")
            if current_type != "archived":
                ok = await bucket_mgr.archive(bucket_id)
                if not ok:
                    summary["failed"] += 1
                    results.append({"id": bucket_id, "status": "failed", "reason": "archive_failed"})
                    continue
                changed = True
        elif status == "active":
            current = await bucket_mgr.get(bucket_id)
            current_meta = (current or {}).get("metadata") or {}
            if current_meta.get("type") == "archived":
                ok = await bucket_mgr.activate(bucket_id)
                if not ok:
                    summary["failed"] += 1
                    results.append({"id": bucket_id, "status": "failed", "reason": "activate_failed"})
                    continue
                changed = True
            elif current_meta.get("active") is False or current_meta.get("resolved") or current_meta.get("deprecated"):
                ok = await bucket_mgr.update(
                    bucket_id,
                    active=True,
                    deprecated=False,
                    resolved=False,
                    last_active=current_meta.get("last_active") or current_meta.get("created"),
                )
                if not ok:
                    summary["failed"] += 1
                    results.append({"id": bucket_id, "status": "failed", "reason": "activate_failed"})
                    continue
                changed = True

        if changed:
            summary["changed"] += 1
            changed_ids.append(bucket_id)
            results.append({"id": bucket_id, "status": "changed"})
        else:
            summary["unchanged"] += 1
            results.append({"id": bucket_id, "status": "unchanged"})

    return JSONResponse({
        **summary,
        "changed_ids": changed_ids,
        "changed_count": len(changed_ids),
        "results": results,
    })


@mcp.custom_route("/api/bucket/{bucket_id}", methods=["GET"])
async def api_bucket_detail(request):
    """Get full bucket content by ID."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    bucket_id = request.path_params["bucket_id"]
    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(_bucket_read_payload(bucket))


@mcp.custom_route("/api/moments", methods=["GET"])
async def api_moments(request):
    """Return dashboard diagnostics for indexed memory moments."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err

    bucket_id = str(request.query_params.get("bucket_id", "") or "").strip()
    limit = _int_between(request.query_params.get("limit"), 20, 1, 200)
    payload = await inspect_moments(bucket_id=bucket_id, limit=limit)
    if payload.get("status") == "error":
        status_code = 404 if payload.get("error") == "not_found" else 400
        return JSONResponse(payload, status_code=status_code)
    return JSONResponse(payload)


@mcp.custom_route("/api/todos", methods=["GET"])
async def api_todos(request):
    """Deprecated: derived followup/todo items are disabled."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    return JSONResponse(
        {
            "count": 0,
            "todos": [],
            "disabled": True,
            "message": "Derived followup/todo items are disabled. Use /api/reminders instead.",
        }
    )


@mcp.custom_route("/api/todos/{todo_id}", methods=["PATCH"])
async def api_todo_update(request):
    """Deprecated: derived followup/todo items are disabled."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    return JSONResponse(
        {"error": "derived followup/todo items are disabled; use /api/reminders instead"},
        status_code=410,
    )


@mcp.custom_route("/api/todos/{todo_id}/writeback", methods=["POST"])
async def api_todo_writeback(request):
    """Deprecated: todo writeback is disabled."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    return JSONResponse(
        {"error": "todo writeback is disabled; use /api/reminders instead"},
        status_code=410,
    )


@mcp.custom_route("/api/reminders", methods=["GET"])
async def api_reminders(request):
    """List standalone care memos."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        status = str(request.query_params.get("status", "active") or "active").strip().lower()
        limit = _int_between(request.query_params.get("limit"), 50, 1, 200)
        items = reminder_store.list(status=status, limit=limit)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    return JSONResponse({"count": len(items), "reminders": [_reminder_public_payload(item) for item in items]})


@mcp.custom_route("/api/reminders", methods=["POST"])
async def api_reminder_create(request):
    """Create a standalone care memo without touching memory buckets."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "json body must be an object"}, status_code=400)
    try:
        item = reminder_store.create(
            title=str(body.get("title") or ""),
            content=str(body.get("content") or body.get("text") or ""),
            next_due_at=str(body.get("next_due_at") or ""),
            start_at=str(body.get("start_at") or ""),
            end_at=str(body.get("end_at") or ""),
            repeat_rule=str(body.get("repeat_rule") or "every_n_rounds"),
            interval_rounds=_int_between(body.get("interval_rounds"), 6, 0, 100000),
            cooldown_minutes=_int_between(body.get("cooldown_minutes"), 0, 0, 525600),
            daily_limit=_int_between(body.get("daily_limit"), 1, 0, 100)
            if "daily_limit" in body
            else None,
            max_injections=_int_between(body.get("max_injections"), 0, 0, 100000),
            channel=str(body.get("channel") or "global"),
            session_id=str(body.get("session_id") or ""),
            source=str(body.get("source") or "dashboard"),
        )
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    return JSONResponse({"status": "created", "reminder": _reminder_public_payload(item)})


@mcp.custom_route("/api/reminders/{reminder_id}", methods=["PATCH"])
async def api_reminder_update(request):
    """Update one standalone reminder. Snooze keeps it active and moves next_due_at."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    reminder_id = str(request.path_params.get("reminder_id") or "").strip()
    if not reminder_id:
        return JSONResponse({"error": "missing reminder_id"}, status_code=400)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "json body must be an object"}, status_code=400)
    try:
        if body.get("mark_reminded"):
            item = reminder_store.mark_reminded(
                reminder_id,
                round_id=_int_between(body.get("round_id"), 0, 0, 100000000),
            )
        elif body.get("snooze_minutes"):
            item = reminder_store.snooze(
                reminder_id,
                minutes=_int_between(body.get("snooze_minutes"), 60, 1, 525600),
            )
        else:
            content_value = None
            if "content" in body:
                content_value = body.get("content")
            elif "text" in body:
                content_value = body.get("text")
            item = reminder_store.update(
                reminder_id,
                title=body.get("title") if "title" in body else None,
                content=content_value,
                status=body.get("status") if "status" in body else None,
                channel=body.get("channel") if "channel" in body else None,
                session_id=body.get("session_id") if "session_id" in body else None,
                start_at=body.get("start_at") if "start_at" in body else None,
                end_at=body.get("end_at") if "end_at" in body else None,
                next_due_at=body.get("next_due_at") if "next_due_at" in body else None,
                repeat_rule=body.get("repeat_rule") if "repeat_rule" in body else None,
                interval_rounds=_int_between(body.get("interval_rounds"), 0, 0, 100000)
                if "interval_rounds" in body
                else None,
                cooldown_minutes=_int_between(body.get("cooldown_minutes"), 0, 0, 525600)
                if "cooldown_minutes" in body
                else None,
                daily_limit=_int_between(body.get("daily_limit"), 1, 0, 100)
                if "daily_limit" in body
                else None,
                max_injections=_int_between(body.get("max_injections"), 0, 0, 100000)
                if "max_injections" in body
                else None,
            )
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    if not item:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse({"status": "updated", "reminder": _reminder_public_payload(item)})


@mcp.custom_route("/api/bucket/{bucket_id}", methods=["PATCH"])
async def api_bucket_update(request):
    """Update dashboard-editable bucket body fields."""
    from starlette.responses import JSONResponse

    err = _require_dashboard_auth(request)
    if err:
        return err

    bucket_id = request.path_params["bucket_id"]
    if not bucket_id or not MEMORY_ID_RE.fullmatch(bucket_id):
        return JSONResponse({"error": "invalid bucket_id"}, status_code=400)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "json body must be an object"}, status_code=400)

    content = str(body.get("content") or "").strip() if "content" in body else None
    name = str(body.get("name") or "").strip() if "name" in body else None
    event_date = str(body.get("date") or "").strip() if "date" in body else None

    if content is None and name is None and event_date is None:
        return JSONResponse({"error": "missing content, name, or date"}, status_code=400)
    if event_date:
        normalized_date = local_date_key(event_date)
        if not normalized_date:
            return JSONResponse({"error": "invalid date"}, status_code=400)
        event_date = normalized_date

    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return JSONResponse({"error": "not found"}, status_code=404)

    meta = bucket.get("metadata", {})
    if content is not None:
        if not content:
            return JSONResponse({"error": "empty content"}, status_code=400)
        if _has_favorite_tag(meta.get("tags", [])) and not _has_favorite_reason(content):
            return JSONResponse({"error": _favorite_reason_error()}, status_code=400)

    update_kwargs = {}
    if content is not None:
        update_kwargs["content"] = content
    if name is not None:
        update_kwargs["name"] = name or None
    if event_date is not None:
        update_kwargs["date"] = event_date
    update_kwargs["last_active"] = meta.get("last_active") or meta.get("created")

    before_bucket = bucket
    ok = await bucket_mgr.update(bucket_id, **update_kwargs)
    if not ok:
        return JSONResponse({"error": "update failed"}, status_code=500)

    bucket = await bucket_mgr.get(bucket_id)
    embedding_queued = (
        _queue_embedding_refresh_if_changed(bucket_id, before_bucket, bucket)
        if (content is not None or name is not None)
        else False
    )
    return JSONResponse({
        "status": "updated",
        "id": bucket_id,
        "embedding_refreshed": False,
        "embedding_queued": embedding_queued,
        **_bucket_read_payload(bucket),
    })


@mcp.custom_route("/api/darkroom/status", methods=["GET"])
async def api_darkroom_status(request):
    """Return public darkroom door status. Never returns private notes."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    return JSONResponse(darkroom_store.status())


@mcp.custom_route("/api/search", methods=["GET"])
async def api_search(request):
    """Search buckets by query."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    query = request.query_params.get("q", "")
    if not query:
        return JSONResponse({"error": "missing q parameter"}, status_code=400)
    try:
        matches = await bucket_mgr.search(query, limit=10)
        result = []
        for b in matches:
            meta = b.get("metadata", {})
            result.append({
                "id": b["id"],
                "name": meta.get("name", b["id"]),
                "type": meta.get("type", "dynamic"),
                "score": b.get("score", 0),
                "domain": meta.get("domain", []),
                "valence": meta.get("valence", 0.5),
                "arousal": meta.get("arousal", 0.3),
                "resolved": meta.get("resolved", False),
                "pinned": meta.get("pinned", False),
                "anchor": meta.get("anchor", False),
                "digested": meta.get("digested", False),
                "last_active": meta.get("last_active", ""),
                "created": meta.get("created", ""),
                "content_preview": strip_wikilinks(b.get("content", ""))[:200],
            })
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


def _raw_ingest_events_from_body(
    body: dict,
    *,
    default_session_id: str = "",
    default_conversation_id: str = "",
) -> list[dict]:
    if not isinstance(body, dict):
        return []
    if isinstance(body.get("events"), list):
        events = [item for item in body.get("events", []) if isinstance(item, dict)]
    elif isinstance(body.get("event"), dict):
        events = [body["event"]]
    elif any(key in body for key in ("role", "text", "content")):
        events = [body]
    else:
        events = []

    fallback_session_id = str(default_session_id or "").strip()
    fallback_conversation_id = str(default_conversation_id or "").strip() or fallback_session_id
    common = {
        "source": body.get("source"),
        "conversation_id": body.get("conversation_id") or fallback_conversation_id,
        "session_id": body.get("session_id") or fallback_session_id,
        "client": body.get("client"),
    }
    for event in events:
        for key, value in common.items():
            if value is not None and key not in event:
                event[key] = value
    return events


@mcp.custom_route("/api/ingest-raw", methods=["POST"])
async def api_ingest_raw(request):
    """Ingest user/assistant raw dialogue events. Does not accept tools, system prompts, or memory injections."""
    from starlette.responses import JSONResponse
    err = _require_raw_api_auth(request)
    if err:
        return err
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        return JSONResponse({"error": "request body must be an object"}, status_code=400)

    header_session_id = str(request.headers.get("X-Ombre-Session-Id") or "").strip()
    events = _raw_ingest_events_from_body(
        body,
        default_session_id=header_session_id,
        default_conversation_id=header_session_id,
    )
    if not events:
        return JSONResponse({"error": "missing events"}, status_code=400)

    try:
        result = raw_event_store.ingest(events, source=str(body.get("source") or "raw"))
        return JSONResponse(result)
    except Exception as exc:
        logger.warning("raw ingest failed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


@mcp.custom_route("/api/search-raw", methods=["GET", "POST"])
async def api_search_raw(request):
    """Search raw dialogue events as a fallback archive. Returns only stored user/assistant originals."""
    from starlette.responses import JSONResponse
    err = _require_raw_api_auth(request)
    if err:
        return err

    params = dict(getattr(request, "query_params", {}) or {})
    body = {}
    try:
        parsed = await request.json()
        if isinstance(parsed, dict):
            body = parsed
    except Exception:
        body = {}

    def value(name: str, default: str = ""):
        return body.get(name, params.get(name, default))

    query = str(value("q", value("query", "")) or "")
    try:
        result = raw_event_store.search(
            query=query,
            limit=_int_between(value("limit", 10), 10, 1, 100),
            source=str(value("source", "") or ""),
            role=str(value("role", "") or ""),
            conversation_id=str(value("conversation_id", "") or ""),
            session_id=str(value("session_id", "") or ""),
            since=str(value("since", "") or ""),
            until=str(value("until", "") or ""),
        )
        return JSONResponse(result)
    except Exception as exc:
        logger.warning("raw search failed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


@mcp.custom_route("/api/network", methods=["GET"])
async def api_network(request):
    """Get embedding similarity network for visualization."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        nodes = []
        edges = []
        embeddings = {}

        for b in all_buckets:
            meta = b.get("metadata", {})
            bid = b["id"]
            nodes.append({
                "id": bid,
                "name": meta.get("name", bid),
                "type": meta.get("type", "dynamic"),
                "domain": meta.get("domain", []),
                "valence": meta.get("valence", 0.5),
                "arousal": meta.get("arousal", 0.3),
                "score": decay_engine.calculate_score(meta),
                "importance": meta.get("importance", 5),
                "confidence": meta.get("confidence", 0.5),
                "resolved": meta.get("resolved", False),
                "pinned": meta.get("pinned", False),
                "anchor": meta.get("anchor", False),
                "digested": meta.get("digested", False),
            })
            if embedding_engine and embedding_engine.enabled:
                emb = await embedding_engine.get_embedding(bid)
                if emb is not None:
                    embeddings[bid] = emb

        # Build soft edges from embeddings (higher threshold to avoid hairball graphs)
        ids = list(embeddings.keys())
        for i, id_a in enumerate(ids):
            for id_b in ids[i+1:]:
                sim = embedding_engine._cosine_similarity(embeddings[id_a], embeddings[id_b])
                if sim > 0.72:
                    edges.append({
                        "source": id_a,
                        "target": id_b,
                        "similarity": round(sim, 3),
                        "kind": "similarity",
                    })

        node_ids = {node["id"] for node in nodes}
        for edge in memory_edge_store.list_edges():
            if edge["source"] in node_ids and edge["target"] in node_ids:
                edges.append({
                    "source": edge["source"],
                    "target": edge["target"],
                    "similarity": edge["confidence"],
                    "kind": "memory_edge",
                    "relation_type": edge["relation_type"],
                    "confidence": edge["confidence"],
                    "reason": edge["reason"],
                })

        scene_map = {
            str(bucket.get("id") or ""): bucket
            for bucket in all_buckets
            if _is_canonical_scene_bucket(bucket)
        }
        for edge in scene_linker.recall_scene_edges(scene_map):
            if edge["source"] in node_ids and edge["target"] in node_ids:
                edges.append({
                    "source": edge["source"],
                    "target": edge["target"],
                    "similarity": edge["confidence"],
                    "kind": "scene_edge",
                    "relation_type": edge["relation_type"],
                    "confidence": edge["confidence"],
                    "reason": edge["reason"],
                    "proposal_id": edge["proposal_id"],
                })

        return JSONResponse({"nodes": nodes, "edges": edges})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/edges", methods=["GET"])
async def api_edges(request):
    """List explicit memory edges."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    return JSONResponse({"edges": memory_edge_store.list_edges()})


@mcp.custom_route("/api/scene-edges", methods=["GET"])
async def api_scene_edges(request):
    """List reviewed Scene-only edges; legacy JSONL edges are excluded."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    return JSONResponse({"edges": scene_linker.list_scene_edges()})


@mcp.custom_route("/api/breath-debug", methods=["GET"])
async def api_breath_debug(request):
    """Debug endpoint: simulate breath scoring and return per-bucket breakdown."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    query = request.query_params.get("q", "")
    q_valence = request.query_params.get("valence")
    q_arousal = request.query_params.get("arousal")
    q_valence = float(q_valence) if q_valence else None
    q_arousal = float(q_arousal) if q_arousal else None
    rerank_requested = _bool_value(request.query_params.get("rerank"), False)

    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        results = []
        w = {
            "topic": bucket_mgr.w_topic,
            "emotion": bucket_mgr.w_emotion,
            "time": bucket_mgr.w_time,
            "importance": bucket_mgr.w_importance,
        }
        w_sum = sum(w.values())
        lexical_terms = _breath_lexical_match_terms(query, all_buckets=all_buckets)
        topic_scores = (
            bucket_mgr.calc_topic_scores(query, all_buckets)
            if query and hasattr(bucket_mgr, "calc_topic_scores")
            else {}
        )

        for bucket in all_buckets:
            meta = bucket.get("metadata", {})
            bid = bucket["id"]
            try:
                topic = topic_scores.get(str(bid), 0.0) if query else 0.0
                emotion = bucket_mgr._calc_emotion_score(q_valence, q_arousal, meta)
                time_s = bucket_mgr._calc_time_score(meta)
                imp = max(1, min(10, int(meta.get("importance", 5)))) / 10.0

                raw_total = (
                    topic * w["topic"]
                    + emotion * w["emotion"]
                    + time_s * w["time"]
                    + imp * w["importance"]
                )
                normalized = (raw_total / w_sum) * 100 if w_sum > 0 else 0
                resolved = meta.get("resolved", False)
                if resolved:
                    normalized *= 0.3
                lexical_match = (
                    meta.get("type") != "feel"
                    and _bucket_matches_breath_lexical_terms(bucket, lexical_terms)
                )
                if lexical_match:
                    normalized = max(normalized, bucket_mgr.fuzzy_threshold + 5)

                results.append({
                    "id": bid,
                    "name": meta.get("name", bid),
                    "domain": meta.get("domain", []),
                    "type": meta.get("type", "dynamic"),
                    "resolved": resolved,
                    "pinned": meta.get("pinned", False),
                    "anchor": meta.get("anchor", False),
                    "layer_debug": _inspect_bucket_layer_payload(bucket),
                    "runtime_gate": _inspect_bucket_runtime_gate_payload(
                        bucket,
                        query=query,
                    ),
                    "scores": {
                        "topic": round(topic, 4),
                        "emotion": round(emotion, 4),
                        "time": round(time_s, 4),
                        "importance": round(imp, 4),
                        "lexical": 1.0 if lexical_match else 0.0,
                    },
                    "lexical_terms": lexical_terms if lexical_match else [],
                    "weights": w,
                    "raw_total": round(raw_total, 4),
                    "normalized": round(normalized, 2),
                    "passed_threshold": normalized >= bucket_mgr.fuzzy_threshold,
                })
            except Exception:
                continue

        results.sort(key=lambda x: x["normalized"], reverse=True)
        passed = [r for r in results if r["passed_threshold"]]
        payload = {
            "query": query,
            "valence": q_valence,
            "arousal": q_arousal,
            "weights": w,
            "lexical_terms": lexical_terms,
            "threshold": bucket_mgr.fuzzy_threshold,
            "total_candidates": len(results),
            "passed_count": len(passed),
            "results": results[:50],  # top 50 for debug
        }
        if rerank_requested:
            try:
                payload["rerank"] = await _build_breath_debug_rerank_payload(
                    query,
                    max_candidates=_int_between(
                        request.query_params.get("rerank_max_candidates"),
                        12,
                        1,
                        100,
                    ),
                    max_results=_int_between(
                        request.query_params.get("rerank_max_results"),
                        3,
                        1,
                        20,
                    ),
                    max_tokens=_int_between(
                        request.query_params.get("rerank_max_tokens"),
                        400,
                        1,
                        20000,
                    ),
                    direct_render_mode=request.query_params.get("direct_render_mode", "compact"),
                    valence=q_valence,
                    arousal=q_arousal,
                )
            except Exception as e:
                payload["rerank"] = {
                    "requested": True,
                    "enabled": bool(getattr(reranker_engine, "enabled", False)),
                    "applied": False,
                    "status": "error",
                    "skip_reason": "exception",
                    "error": str(e),
                    "candidates": [],
                    "warnings": [],
                }
        return JSONResponse(payload)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/diffusion-debug", methods=["GET"])
async def api_diffusion_debug(request):
    """Debug endpoint: inspect bucket-level diffusion paths for a query."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err

    query = request.query_params.get("q", "")
    max_seeds = _int_between(request.query_params.get("max_seeds"), 3, 1, 20)
    max_hits = _int_between(request.query_params.get("max_hits"), 5, 0, 20)
    edge_min_confidence = _float_between(
        request.query_params.get("edge_min_confidence"),
        0.55,
        0.0,
        1.0,
    )
    payload = await inspect_diffusion(
        query=query,
        max_seeds=max_seeds,
        max_hits=max_hits,
        edge_min_confidence=edge_min_confidence,
    )
    if payload.get("status") == "error":
        return JSONResponse(payload, status_code=400)
    return JSONResponse(payload)


@mcp.custom_route("/api/recall-debug", methods=["GET"])
async def api_recall_debug(request):
    """Debug endpoint: inspect query-to-moment recall candidates."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err

    q_valence = request.query_params.get("valence")
    q_arousal = request.query_params.get("arousal")
    q_valence = float(q_valence) if q_valence else None
    q_arousal = float(q_arousal) if q_arousal else None
    payload = await _build_recall_debug_payload(
        request.query_params.get("q", ""),
        max_candidates=_int_between(request.query_params.get("max_candidates"), 20, 1, 100),
        max_results=_int_between(request.query_params.get("max_results"), 3, 1, 20),
        max_tokens=_int_between(request.query_params.get("max_tokens"), 800, 1, 20000),
        direct_render_mode=request.query_params.get("direct_render_mode", "auto"),
        domain=request.query_params.get("domain", ""),
        valence=q_valence,
        arousal=q_arousal,
    )
    if payload.get("status") == "error":
        return JSONResponse(payload, status_code=400)
    return JSONResponse(payload)


@mcp.custom_route("/api/gateway-injections", methods=["GET"])
async def api_gateway_injections(request):
    """Dashboard-authenticated proxy for recent Gateway injection debug records."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err

    session_id = str(request.query_params.get("session_id", "") or "").strip()
    limit = _int_between(request.query_params.get("limit"), 10, 1, 100)
    include_context = str(request.query_params.get("include_context", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
    }
    payload = await _fetch_gateway_injection_debug(
        session_id=session_id,
        limit=limit,
        include_context=include_context,
    )
    status_code = 200 if payload.get("status") == "ok" else 502
    return JSONResponse(payload, status_code=status_code)


@mcp.custom_route("/api/reflection/run", methods=["POST"])
async def api_reflection_run(request):
    """Run daily reflection from dashboard or trusted local callers; weekly obeys reflection.weekly_enabled."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    try:
        result = await reflection_engine.reflect(
            period=str(body.get("period") or "daily"),
            bucket_mgr=bucket_mgr,
            persona_engine=persona_engine,
            embedding_engine=embedding_engine,
            force=_bool_value(body.get("force"), False),
            conversation_turn_store=gateway_state_store,
        )
        return JSONResponse(result)
    except Exception as e:
        logger.warning("Reflection API failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/daily-chat-memory/run", methods=["POST"])
async def api_daily_chat_memory_run(request):
    """Run daily Gateway chat memory extraction; review mode writes pending candidates only."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    try:
        result = await reflection_engine.run_daily_chat_memory(
            bucket_mgr,
            conversation_turn_store=gateway_state_store,
            raw_event_store=raw_event_store,
            persona_engine=persona_engine,
            embedding_engine=embedding_engine,
            key=str(body.get("date") or ""),
            mode=str(body.get("mode") or ""),
            force=_bool_value(body.get("force"), False),
        )
        try:
            activity_date = str(body.get("date") or result.get("date") or "")
            daily_impression = await _daily_impression_material_for_date(activity_date)
            activity_result = await reflection_engine.run_daily_activity_summary(
                conversation_turn_store=gateway_state_store,
                raw_event_store=raw_event_store,
                persona_engine=persona_engine,
                daily_chat_memory_candidates=[
                    item for item in (result.get("candidates") or []) if isinstance(item, dict)
                ],
                daily_impressions=[daily_impression] if daily_impression else [],
                key=str(body.get("date") or ""),
                force=_bool_value(body.get("force"), False),
            )
            result["daily_activity_summary"] = _store_daily_activity_summary_result(activity_result)
        except Exception as activity_exc:
            logger.warning("Daily activity summary side-run failed: %s", activity_exc)
            result["daily_activity_summary"] = {"status": "error", "error": str(activity_exc)}
        return JSONResponse(result)
    except Exception as e:
        logger.warning("Daily chat memory API failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


def _store_daily_activity_summary_result(result: dict, portrait_engine_arg=None) -> dict:
    if not isinstance(result, dict):
        return {"status": "invalid", "reason": "result_not_object"}
    if result.get("status") != "ready":
        return result
    item = result.get("activity_summary") if isinstance(result.get("activity_summary"), dict) else {}
    if not item:
        return {**result, "status": "skipped", "reason": "empty_activity_summary"}
    engine = portrait_engine_arg or portrait_engine
    date_key = str(result.get("date") or item.get("source_date") or "").strip()
    try:
        stored = engine.upsert_recent_timeline_item(item, date_key)
    except Exception as exc:
        logger.warning("Daily activity summary portrait upsert failed: %s", exc)
        return {**result, "status": "error", "error": str(exc)}
    return {**result, "status": "stored", "portrait": stored}


def _daily_activity_materials_from_reflection_results(results: list[dict]) -> tuple[list[dict], list[dict]]:
    candidates: list[dict] = []
    daily_impressions: list[dict] = []
    for result in results or []:
        if not isinstance(result, dict):
            continue
        candidates.extend(item for item in (result.get("candidates") or []) if isinstance(item, dict))
        daily_impression = result.get("daily_impression")
        if isinstance(daily_impression, dict):
            daily_impressions.append(daily_impression)
    return candidates, daily_impressions


async def _daily_impression_material_for_date(date_key: str, bucket_mgr_arg=None) -> dict:
    safe_date = _date_key(date_key)
    if not safe_date:
        return {}
    manager = bucket_mgr_arg or bucket_mgr
    try:
        bucket = await manager.get(f"reflection_daily_{safe_date}")
    except Exception:
        bucket = None
    if not bucket:
        return {}
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    return {
        "id": bucket.get("id") or f"reflection_daily_{safe_date}",
        "content": bucket.get("content") or "",
        "confidence": meta.get("confidence", 0.7),
        "date": safe_date,
    }


@mcp.custom_route("/api/daily-activity-summary/run", methods=["POST"])
async def api_daily_activity_summary_run(request):
    """Summarize what happened that day into portrait recent_timeline."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    try:
        activity_date = str(body.get("date") or "")
        daily_impression = await _daily_impression_material_for_date(activity_date)
        result = await reflection_engine.run_daily_activity_summary(
            conversation_turn_store=gateway_state_store,
            raw_event_store=raw_event_store,
            persona_engine=persona_engine,
            daily_impressions=[daily_impression] if daily_impression else [],
            key=str(body.get("date") or ""),
            force=_bool_value(body.get("force"), False),
        )
        return JSONResponse(_store_daily_activity_summary_result(result))
    except Exception as e:
        logger.warning("Daily activity summary API failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/daily-chat-memory/pending", methods=["GET"])
async def api_daily_chat_memory_pending(request):
    """List pending daily chat memory candidates."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    params = request.query_params
    items = reflection_engine.list_daily_chat_memory_pending(
        status=str(params.get("status") or "pending"),
        limit=_int_between(params.get("limit"), 50, 1, 200),
    )
    return JSONResponse({"status": "ok", "items": items})


@mcp.custom_route("/api/daily-chat-memory/confirm", methods=["POST"])
async def api_daily_chat_memory_confirm(request):
    """Confirm or reject pending daily chat memory candidates."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "json body must be an object"}, status_code=400)
    action = str(body.get("action") or "confirm").strip().lower()
    required = "REJECT" if action == "reject" else "WRITE"
    if body.get("confirm") != required:
        return JSONResponse({"error": f"confirmation required: {required}"}, status_code=400)
    ids = body.get("candidate_ids", [])
    if not isinstance(ids, list) or not ids:
        return JSONResponse({"error": "candidate_ids must be a non-empty list"}, status_code=400)
    result = await reflection_engine.confirm_daily_chat_memory(
        [str(item or "") for item in ids],
        bucket_mgr,
        embedding_engine=embedding_engine,
        action=action,
        edits=body.get("edits") if isinstance(body.get("edits"), dict) else None,
    )
    return JSONResponse(result)


@mcp.custom_route("/dashboard", methods=["GET"])
async def dashboard(request):
    """Serve the dashboard HTML page."""
    from starlette.responses import HTMLResponse
    import os
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    try:
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>dashboard.html not found</h1>", status_code=404)


@mcp.custom_route("/dashboard-assets/{path:path}", methods=["GET"])
async def dashboard_assets(request):
    """Serve small dashboard modules without turning dashboard.html into a bundle."""
    from starlette.responses import FileResponse, PlainTextResponse
    asset_path = str(request.path_params.get("path") or "").strip().replace("\\", "/")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "dashboard_assets"))
    target = os.path.abspath(os.path.join(base_dir, asset_path))
    if not target.startswith(base_dir + os.sep) or not os.path.isfile(target):
        return PlainTextResponse("dashboard asset not found", status_code=404)
    return FileResponse(target)


@mcp.custom_route("/api/persona", methods=["GET"])
async def api_persona_get(request):
    """Return Persona State Engine data for the local dashboard."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err

    def _bounded_int(value, default, lower, upper):
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = default
        return max(lower, min(upper, number))

    try:
        session_id = (request.query_params.get("session_id") or "").strip() or None
        events_limit = _bounded_int(request.query_params.get("events_limit"), 20, 1, 100)
        sessions_limit = _bounded_int(request.query_params.get("sessions_limit"), 20, 1, 100)
        return JSONResponse(
            persona_engine.get_dashboard_payload(
                session_id=session_id,
                events_limit=events_limit,
                sessions_limit=sessions_limit,
            )
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/dreams", methods=["GET"])
async def api_dreams(request):
    """Return dream dashboard metadata only. Dream bodies are never exposed here."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        limit = int(request.query_params.get("limit", "30"))
    except Exception:
        limit = 30
    try:
        return JSONResponse(dream_engine.dashboard_payload(limit=max(1, min(100, limit))))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/dreams/{dream_id}", methods=["GET"])
async def api_dream_detail(request):
    """Return one retained dream body for an authenticated dashboard reader."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    record = dream_engine.dashboard_record(request.path_params.get("dream_id", ""))
    if not record:
        return JSONResponse({"error": "dream body unavailable"}, status_code=404)
    return JSONResponse(record)


@mcp.custom_route("/api/config", methods=["GET"])
async def api_config_get(request):
    """Get current runtime config (safe fields only, API key masked)."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err

    def _mask_key(api_key: str) -> str:
        return f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else ("***" if api_key else "")

    dehy = config.get("dehydration", {})
    emb = config.get("embedding", {})
    rerank = config.get("reranker", {}) if isinstance(config.get("reranker", {}), dict) else {}
    gateway_cfg = config.get("gateway", {}) if isinstance(config.get("gateway", {}), dict) else {}
    recall_cfg = config.get("recall", {}) if isinstance(config.get("recall", {}), dict) else {}
    diffusion_options = diffusion_options_from_config(config)
    persona_cfg = config.get("persona", {}) if isinstance(config.get("persona", {}), dict) else {}
    dream_cfg = config.get("dream", {}) if isinstance(config.get("dream", {}), dict) else {}
    reflection_cfg = config.get("reflection", {}) if isinstance(config.get("reflection", {}), dict) else {}
    portrait_cfg = config.get("portrait", {}) if isinstance(config.get("portrait", {}), dict) else {}
    self_anchor_cfg = config.get("self_anchor", {}) if isinstance(config.get("self_anchor", {}), dict) else {}
    domain_sentinel_configured_model = str(gateway_cfg.get("domain_sentinel_model") or "").strip()
    domain_sentinel_effective_model = domain_sentinel_configured_model or str(dehy.get("model") or "").strip()
    domain_sentinel_base_url = str(
        gateway_cfg.get("domain_sentinel_base_url") or dehy.get("base_url") or emb.get("base_url") or ""
    ).strip()
    domain_sentinel_api_key = str(
        os.environ.get("OMBRE_DOMAIN_SENTINEL_API_KEY", "")
        or gateway_cfg.get("domain_sentinel_api_key", "")
        or dehy.get("api_key", "")
        or os.environ.get("OMBRE_API_KEY", "")
        or os.environ.get("OMBRE_EMBEDDING_API_KEY", "")
        or emb.get("api_key", "")
        or ""
    )
    reflection_base_url = str(
        reflection_cfg.get("base_url") or emb.get("base_url") or persona_cfg.get("base_url") or dehy.get("base_url") or ""
    ).strip()
    reflection_api_key = str(
        os.environ.get("OMBRE_REFLECTION_API_KEY", "")
        or reflection_cfg.get("api_key", "")
        or os.environ.get("OMBRE_EMBEDDING_API_KEY", "")
        or emb.get("api_key", "")
        or persona_cfg.get("api_key", "")
        or os.environ.get("OMBRE_PERSONA_API_KEY", "")
        or dehy.get("api_key", "")
        or ""
    )
    return JSONResponse({
        "dehydration": {
            "model": dehy.get("model", ""),
            "base_url": dehy.get("base_url", ""),
            "api_key_masked": _mask_key(dehy.get("api_key", "")),
            "max_tokens": dehy.get("max_tokens", 1024),
            "temperature": dehy.get("temperature", 0.1),
        },
        "embedding": {
            "enabled": emb.get("enabled", False),
            "model": emb.get("model", ""),
            "base_url": emb.get("base_url", ""),
            "api_key_masked": _mask_key(emb.get("api_key", "")),
            "effective_base_url": embedding_engine.base_url,
            "has_own_api_key": bool(emb.get("api_key", "")),
        },
        "reranker": {
            "enabled": bool(getattr(reranker_engine, "enabled", False)),
            "model": getattr(reranker_engine, "model", rerank.get("model", "Qwen/Qwen3-Reranker-4B")),
            "base_url": str(rerank.get("base_url") or getattr(reranker_engine, "base_url", "") or ""),
            "api_key_masked": _mask_key(rerank.get("api_key", "")),
            "effective_base_url": getattr(reranker_engine, "base_url", ""),
            "api_ready": bool(getattr(reranker_engine, "api_key", "") or rerank.get("api_key", "")),
            "has_own_api_key": bool(rerank.get("api_key", "")),
            "timeout_seconds": getattr(reranker_engine, "timeout", rerank.get("timeout_seconds", 12)),
            "candidate_limit": getattr(reranker_engine, "candidate_limit", rerank.get("candidate_limit", 20)),
            "score_weight": getattr(reranker_engine, "score_weight", rerank.get("score_weight", 0.65)),
        },
        "gateway": {
            "cooldown_hours": gateway_cfg.get("cooldown_hours", 6),
            "skip_recent_rounds": gateway_cfg.get("skip_recent_rounds", 5),
            "recent_context_cooldown_hours": gateway_cfg.get("recent_context_cooldown_hours", 6),
            "recent_context_reentry_idle_hours": gateway_cfg.get("recent_context_reentry_idle_hours", 24),
            "recent_context_budget": gateway_cfg.get("recent_context_budget", 300),
            "date_persona_trace_enabled": _bool_value(gateway_cfg.get("date_persona_trace_enabled"), True),
            "date_persona_trace_budget": gateway_cfg.get("date_persona_trace_budget", 220),
            "date_persona_trace_max_events": gateway_cfg.get("date_persona_trace_max_events", 5),
            "date_persona_trace_include_daily": _bool_value(gateway_cfg.get("date_persona_trace_include_daily"), True),
            "recalled_memory_budget": gateway_cfg.get("recalled_memory_budget", 400),
            "related_memory_budget": gateway_cfg.get("related_memory_budget", 220),
            "memory_sentinel_enabled": _bool_value(gateway_cfg.get("memory_sentinel_enabled"), True),
            "domain_sentinel_enabled": _bool_value(gateway_cfg.get("domain_sentinel_enabled"), True),
            "domain_sentinel_model": domain_sentinel_configured_model,
            "domain_sentinel_effective_model": domain_sentinel_effective_model,
            "domain_sentinel_base_url": str(gateway_cfg.get("domain_sentinel_base_url") or ""),
            "domain_sentinel_effective_base_url": domain_sentinel_base_url,
            "domain_sentinel_api_key_masked": _mask_key(domain_sentinel_api_key),
            "domain_sentinel_api_ready": bool(domain_sentinel_base_url and domain_sentinel_api_key),
            "domain_sentinel_enable_thinking": False,
            "domain_sentinel_max_tokens": gateway_cfg.get("domain_sentinel_max_tokens", 260),
            "current_inner_state_interval_rounds": gateway_cfg.get("current_inner_state_interval_rounds", 0),
            "direct_render_mode": _normalize_direct_render_mode(gateway_cfg.get("direct_render_mode", "auto")),
            "retrieval_mode": _normalize_retrieval_mode(gateway_cfg.get("retrieval_mode", "graph")),
            "operit_context_rewrite_enabled": _bool_value(
                gateway_cfg.get("operit_context_rewrite_enabled"),
                False,
            ),
            "recall_fusion_mode": _normalize_recall_fusion_mode(gateway_cfg.get("recall_fusion_mode", "dynamic")),
            "word_map_hint_enabled": _bool_value(gateway_cfg.get("word_map_hint_enabled"), False),
            "portrait_memory_enabled": _bool_value(gateway_cfg.get("portrait_memory_enabled"), False),
            "portrait_memory_budget": gateway_cfg.get("portrait_memory_budget", 360),
            "portrait_memory_max_sources": gateway_cfg.get("portrait_memory_max_sources", 8),
            "portrait_memory_include_anchors": _bool_value(
                gateway_cfg.get("portrait_memory_include_anchors"),
                False,
            ),
            "query_planner_enabled": _bool_value(gateway_cfg.get("query_planner_enabled"), True),
            "query_planner_model": gateway_cfg.get("query_planner_model", ""),
            "query_planner_min_chars": gateway_cfg.get("query_planner_min_chars", 16),
            "query_planner_max_queries": gateway_cfg.get("query_planner_max_queries", 3),
            "query_planner_max_tokens": gateway_cfg.get("query_planner_max_tokens", 360),
            "memory_detail_recall_enabled": _bool_value(gateway_cfg.get("memory_detail_recall_enabled"), False),
            "memory_detail_recall_max_ids": gateway_cfg.get("memory_detail_recall_max_ids", 3),
            "memory_detail_recall_budget": gateway_cfg.get("memory_detail_recall_budget", 1200),
            "upstreams": _dashboard_gateway_upstreams_payload(gateway_cfg),
        },
        "recall": {
            "query_resurface_enabled": _bool_value(recall_cfg.get("query_resurface_enabled"), False),
        },
        "self_anchor": {
            "entry_bucket_id": str(self_anchor_cfg.get("entry_bucket_id") or ""),
        },
        "memory_diffusion": {
            "enabled": diffusion_options.enabled,
            "max_hops": diffusion_options.max_hops,
            "top_k": diffusion_options.top_k,
            "min_activation": diffusion_options.min_activation,
            "max_paths_per_hit": diffusion_options.max_paths_per_hit,
            "chain_walk_enabled": diffusion_options.chain_walk_enabled,
            "chain_max_hops": diffusion_options.chain_max_hops,
            "chain_min_strength": diffusion_options.chain_min_strength,
            "chain_min_confidence": diffusion_options.chain_min_confidence,
            "chain_min_relation_priority": diffusion_options.chain_min_relation_priority,
            "chain_max_frontier": diffusion_options.chain_max_frontier,
        },
        "persona": {
            "enabled": bool(getattr(persona_engine, "enabled", persona_cfg.get("enabled", True))),
            "model": getattr(persona_engine, "model", persona_cfg.get("model", "")),
            "base_url": getattr(persona_engine, "base_url", persona_cfg.get("base_url", "")),
            "event_recording_enabled": _bool_value(
                getattr(
                    persona_engine,
                    "event_recording_enabled",
                    persona_cfg.get("event_recording_enabled"),
                ),
                True,
            ),
            "api_key_masked": _mask_key(getattr(persona_engine, "api_key", "") or persona_cfg.get("api_key", "")),
            "api_ready": bool(getattr(persona_engine, "api_key", "") or persona_cfg.get("api_key", "")),
        },
        "dream": {
            "enabled": dream_engine.enabled,
            "auto_enabled": dream_engine.auto_enabled,
            "surface_enabled": dream_engine.surface_enabled,
            "inject_enabled": _bool_value(dream_cfg.get("inject_enabled"), False),
            "retain_after_inject": _bool_value(dream_cfg.get("retain_after_inject"), True),
            "model": dream_engine.model,
            "base_url": dream_engine.base_url,
            "api_key_masked": _mask_key(dream_engine.api_key),
            "api_ready": bool(dream_engine.api_key),
            "temperature": dream_cfg.get("temperature", 0.85),
            "max_tokens": dream_cfg.get("max_tokens", 900),
            "daily_hour": dream_cfg.get("daily_hour", 3),
            "run_window_hours": dream_cfg.get("run_window_hours", 3),
            "daily_probability": dream_cfg.get("daily_probability", 0.4),
            "min_material_count": dream_cfg.get("min_material_count", 5),
            "material_window_hours": dream_cfg.get("material_window_hours", 48),
            "raw_residue_enabled": _bool_value(dream_cfg.get("raw_residue_enabled"), False),
            "raw_residue_turns": int(dream_cfg.get("raw_residue_turns", 4)),
            "raw_residue_max_chars": int(dream_cfg.get("raw_residue_max_chars", 1500)),
            "identity_anchor_id": dream_cfg.get("identity_anchor_id", ""),
        },
        "reflection": {
            "enabled": bool(
                reflection_cfg.get(
                    "enabled",
                    getattr(reflection_engine, "enabled", True),
                )
            ),
            "auto_enabled": bool(
                reflection_cfg.get(
                    "auto_enabled",
                    getattr(reflection_engine, "auto_enabled", True),
                )
            ),
            "daily_enabled": bool(
                reflection_cfg.get(
                    "daily_enabled",
                    getattr(reflection_engine, "daily_enabled", True),
                )
            ),
            "memory_affect_anchor_enabled": bool(
                reflection_cfg.get(
                    "memory_affect_anchor_enabled",
                    getattr(reflection_engine, "memory_affect_anchor_enabled", True),
                )
            ),
            "relationship_weather_affect_anchor_enabled": bool(
                reflection_cfg.get(
                    "relationship_weather_affect_anchor_enabled",
                    getattr(reflection_engine, "relationship_weather_affect_anchor_enabled", True),
                )
            ),
            "daily_min_memory_items": int(
                reflection_cfg.get(
                    "daily_min_memory_items",
                    getattr(reflection_engine, "daily_min_memory_items", 5),
                )
            ),
            "daily_conversation_turn_limit": int(
                reflection_cfg.get(
                    "daily_conversation_turn_limit",
                    getattr(reflection_engine, "daily_conversation_turn_limit", 12),
                )
            ),
            "daily_activity_summary_enabled": bool(
                reflection_cfg.get(
                    "daily_activity_summary_enabled",
                    getattr(reflection_engine, "daily_activity_summary_enabled", True),
                )
            ),
            "daily_activity_summary_turn_limit": int(
                reflection_cfg.get(
                    "daily_activity_summary_turn_limit",
                    getattr(reflection_engine, "daily_activity_summary_turn_limit", 0),
                )
            ),
            "daily_activity_summary_max_tokens": int(
                reflection_cfg.get(
                    "daily_activity_summary_max_tokens",
                    getattr(reflection_engine, "daily_activity_summary_max_tokens", 320),
                )
            ),
            "daily_chat_memory_mode": str(
                reflection_cfg.get(
                    "daily_chat_memory_mode",
                    getattr(reflection_engine, "daily_chat_memory_mode", "review"),
                )
                or "review"
            ),
            "daily_chat_memory_hour": int(
                reflection_cfg.get(
                    "daily_chat_memory_hour",
                    getattr(reflection_engine, "daily_chat_memory_hour", 0),
                )
            ),
            "daily_chat_memory_turn_limit": int(
                reflection_cfg.get(
                    "daily_chat_memory_turn_limit",
                    getattr(reflection_engine, "daily_chat_memory_turn_limit", 0),
                )
            ),
            "daily_chat_memory_max_per_day": int(
                reflection_cfg.get(
                    "daily_chat_memory_max_per_day",
                    getattr(reflection_engine, "daily_chat_memory_max_per_day", 3),
                )
            ),
            "model": getattr(reflection_engine, "model", reflection_cfg.get("model", "")),
            "thinking_mode": str(reflection_cfg.get("thinking_mode") or getattr(reflection_engine, "thinking_mode", "") or ""),
            "base_url": str(reflection_cfg.get("base_url") or ""),
            "effective_base_url": getattr(reflection_engine, "base_url", reflection_base_url),
            "api_key_masked": _mask_key(getattr(reflection_engine, "api_key", "") or reflection_api_key),
            "api_ready": bool(getattr(reflection_engine, "api_key", "") or reflection_api_key),
        },
        "portrait": {
            "enabled": bool(portrait_cfg.get("enabled", getattr(portrait_engine, "enabled", True))),
            "auto_enabled": bool(portrait_cfg.get("auto_enabled", getattr(portrait_engine, "auto_enabled", True))),
            "auto_initial_enabled": bool(
                portrait_cfg.get(
                    "auto_initial_enabled",
                    getattr(portrait_engine, "auto_initial_enabled", False),
                )
            ),
            "daily_enabled": bool(portrait_cfg.get("daily_enabled", getattr(portrait_engine, "daily_enabled", True))),
            "model": getattr(portrait_engine, "model", portrait_cfg.get("model", "")),
            "base_url": getattr(portrait_engine, "base_url", portrait_cfg.get("base_url", "")),
            "api_key_masked": _mask_key(getattr(portrait_engine, "api_key", "") or portrait_cfg.get("api_key", "")),
            "api_ready": bool(getattr(portrait_engine, "api_key", "") or portrait_cfg.get("api_key", "")),
            "state_path": getattr(portrait_engine, "state_path", ""),
            "daily_hour": portrait_cfg.get("daily_hour", getattr(portrait_engine, "daily_hour", 4)),
            "check_interval_minutes": portrait_cfg.get(
                "check_interval_minutes",
                getattr(portrait_engine, "check_interval_minutes", 60),
            ),
            "material_limit": portrait_cfg.get("material_limit", getattr(portrait_engine, "material_limit", 18)),
            "first_run_material_limit": portrait_cfg.get(
                "first_run_material_limit",
                getattr(portrait_engine, "first_run_material_limit", 160),
            ),
            "persona_events_limit": portrait_cfg.get(
                "persona_events_limit",
                getattr(portrait_engine, "persona_events_limit", 24),
            ),
            "user_rewrite_evidence_delta": portrait_cfg.get(
                "user_rewrite_evidence_delta",
                getattr(portrait_engine, "user_rewrite_evidence_delta", 10),
            ),
            "manual_suppress_days": portrait_cfg.get(
                "manual_suppress_days",
                getattr(portrait_engine, "manual_suppress_days", 14),
            ),
        },
        "merge_threshold": config.get("merge_threshold", 90),
        "transport": config.get("transport", "stdio"),
        "buckets_dir": config.get("buckets_dir", ""),
    })


@mcp.custom_route("/api/config", methods=["POST"])
async def api_config_update(request):
    """Hot-update runtime config. Optionally persist to config.yaml."""
    from starlette.responses import JSONResponse
    import yaml
    global dream_engine, persona_engine, portrait_engine, reflection_engine, metadata_enricher, reranker_engine
    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    updated = []
    env_updates: dict[str, str] = {}

    def _memory_diffusion_dashboard_config(payload) -> dict:
        if not isinstance(payload, dict):
            return {}
        sanitized = {}
        if "enabled" in payload:
            sanitized["enabled"] = _bool_value(payload.get("enabled"), True)
        if "max_hops" in payload:
            sanitized["max_hops"] = _int_between(payload.get("max_hops"), 2, 1, 8)
        if "top_k" in payload:
            sanitized["top_k"] = _int_between(payload.get("top_k"), 4, 0, 20)
        if "min_activation" in payload:
            sanitized["min_activation"] = _float_between(payload.get("min_activation"), 0.18, 0.0, 10.0)
        if "max_paths_per_hit" in payload:
            sanitized["max_paths_per_hit"] = _int_between(payload.get("max_paths_per_hit"), 3, 1, 10)
        if "chain_walk_enabled" in payload:
            sanitized["chain_walk_enabled"] = _bool_value(payload.get("chain_walk_enabled"), False)
        if "chain_max_hops" in payload:
            sanitized["chain_max_hops"] = _int_between(payload.get("chain_max_hops"), 6, 1, 12)
        if "chain_min_strength" in payload:
            sanitized["chain_min_strength"] = _float_between(payload.get("chain_min_strength"), 0.2, 0.0, 10.0)
        if "chain_min_confidence" in payload:
            sanitized["chain_min_confidence"] = _float_between(payload.get("chain_min_confidence"), 0.72, 0.0, 1.0)
        if "chain_min_relation_priority" in payload:
            sanitized["chain_min_relation_priority"] = _int_between(
                payload.get("chain_min_relation_priority"),
                60,
                0,
                100,
            )
        if "chain_max_frontier" in payload:
            sanitized["chain_max_frontier"] = _int_between(payload.get("chain_max_frontier"), 24, 1, 200)
        return sanitized

    gateway_hot_update_payload = {}

    # --- Dehydration config ---
    if "dehydration" in body:
        d = body["dehydration"]
        dehy = config.setdefault("dehydration", {})
        for key in ("model", "base_url", "max_tokens", "temperature", "thinking_mode"):
            if key in d:
                dehy[key] = d[key]
                updated.append(f"dehydration.{key}")
        if "api_key" in d and d["api_key"]:
            dehy["api_key"] = d["api_key"]
            env_updates["OMBRE_API_KEY"] = str(d["api_key"])
            updated.append("dehydration.api_key")
        # Hot-reload dehydrator
        dehydrator.model = dehy.get("model", "deepseek-chat")
        dehydrator.base_url = dehy.get("base_url", "")
        dehydrator.api_key = dehy.get("api_key", "")
        normalize_thinking = getattr(dehydrator, "_normalize_thinking_mode", None)
        if callable(normalize_thinking):
            dehydrator.thinking_mode = normalize_thinking(dehy.get("thinking_mode", ""))
        else:
            dehydrator.thinking_mode = str(dehy.get("thinking_mode") or "").strip()
        dehydrator.max_tokens = dehy.get("max_tokens", 1024)
        dehydrator.temperature = dehy.get("temperature", 0.1)
        dehydrator.api_available = bool(dehydrator.api_key)
        if hasattr(dehydrator, "client") and dehydrator.api_key:
            from openai import AsyncOpenAI
            dehydrator.client = AsyncOpenAI(
                api_key=dehydrator.api_key,
                base_url=dehydrator.base_url,
            )
        gateway_hot_update_payload["dehydration"] = {
            "model": dehydrator.model,
            "base_url": dehydrator.base_url,
            "api_key": dehydrator.api_key,
            "thinking_mode": dehy.get("thinking_mode", ""),
            "max_tokens": dehy.get("max_tokens", 1024),
            "temperature": dehy.get("temperature", 0.1),
        }
        if getattr(portrait_engine, "model_source", "dehydration") == "dehydration":
            portrait_engine = DailyPortraitMaintainer(config)

    # --- Embedding config ---
    if "embedding" in body:
        e = body["embedding"]
        emb = config.setdefault("embedding", {})
        if "enabled" in e:
            emb["enabled"] = bool(e["enabled"])
            updated.append("embedding.enabled")
        if "model" in e:
            emb["model"] = e["model"]
            updated.append("embedding.model")
        if "base_url" in e:
            emb["base_url"] = e["base_url"]
            updated.append("embedding.base_url")
        if "api_key" in e and e["api_key"]:
            emb["api_key"] = e["api_key"]
            env_updates["OMBRE_EMBEDDING_API_KEY"] = str(e["api_key"])
            updated.append("embedding.api_key")

        # Hot-reload embedding client; falls back to dehydration key/base_url when unset.
        embedding_engine.api_key = emb.get("api_key") or config.get("dehydration", {}).get("api_key", "")
        embedding_engine.base_url = (
            emb.get("base_url")
            or config.get("dehydration", {}).get("base_url", "")
            or "https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        embedding_engine.model = emb.get("model", "gemini-embedding-001")
        embedding_engine.enabled = bool(embedding_engine.api_key) and emb.get("enabled", True)
        if embedding_engine.enabled:
            from openai import AsyncOpenAI
            embedding_engine.client = AsyncOpenAI(
                api_key=embedding_engine.api_key,
                base_url=embedding_engine.base_url,
                timeout=30.0,
            )
        else:
            embedding_engine.client = None

    # --- Merge threshold ---
    if "merge_threshold" in body:
        config["merge_threshold"] = int(body["merge_threshold"])
        updated.append("merge_threshold")

    # --- Reranker config ---
    if "reranker" in body:
        r = body["reranker"]
        if not isinstance(r, dict):
            return JSONResponse({"error": "invalid reranker config"}, status_code=400)
        reranker_cfg = config.setdefault("reranker", {})
        reranker_gateway_payload = {}
        if "enabled" in r:
            reranker_cfg["enabled"] = _bool_value(r["enabled"], True)
            reranker_gateway_payload["enabled"] = reranker_cfg["enabled"]
            os.environ["OMBRE_RERANKER_ENABLED"] = "true" if reranker_cfg["enabled"] else "false"
            updated.append("reranker.enabled")
        for key in ("model", "base_url"):
            if key in r:
                reranker_cfg[key] = str(r[key] or "").strip()
                reranker_gateway_payload[key] = reranker_cfg[key]
                updated.append(f"reranker.{key}")
        if "timeout_seconds" in r:
            reranker_cfg["timeout_seconds"] = _float_between(r["timeout_seconds"], 12, 1, 120)
            reranker_gateway_payload["timeout_seconds"] = reranker_cfg["timeout_seconds"]
            updated.append("reranker.timeout_seconds")
        if "candidate_limit" in r:
            reranker_cfg["candidate_limit"] = _int_between(r["candidate_limit"], 20, 1, 100)
            reranker_gateway_payload["candidate_limit"] = reranker_cfg["candidate_limit"]
            updated.append("reranker.candidate_limit")
        if "score_weight" in r:
            reranker_cfg["score_weight"] = _float_between(r["score_weight"], 0.65, 0.0, 1.0)
            reranker_gateway_payload["score_weight"] = reranker_cfg["score_weight"]
            updated.append("reranker.score_weight")
        if "api_key" in r and r["api_key"]:
            reranker_cfg["api_key"] = str(r["api_key"])
            os.environ["OMBRE_RERANKER_API_KEY"] = reranker_cfg["api_key"]
            env_updates["OMBRE_RERANKER_API_KEY"] = reranker_cfg["api_key"]
            reranker_gateway_payload["api_key"] = reranker_cfg["api_key"]
            updated.append("reranker.api_key")
        if "base_url" in r:
            os.environ["OMBRE_RERANKER_BASE_URL"] = reranker_cfg.get("base_url", "")
        if "model" in r:
            os.environ["OMBRE_RERANKER_MODEL"] = reranker_cfg.get("model", "")
        reranker_engine = RerankerEngine(config)
        if reranker_gateway_payload:
            gateway_hot_update_payload["reranker"] = reranker_gateway_payload

    # --- Gateway memory surfacing config ---
    if "gateway" in body:
        g = body["gateway"]
        gateway_cfg = config.setdefault("gateway", {})
        gateway_hot_update_body = {}
        domain_sentinel_touched = any(
            key in g
            for key in (
                "domain_sentinel_enabled",
                "domain_sentinel_model",
                "domain_sentinel_base_url",
                "domain_sentinel_api_key",
                "domain_sentinel_enable_thinking",
            )
        )
        if "upstreams" in g:
            try:
                sanitized_upstreams = _dashboard_sanitize_gateway_upstreams(
                    g["upstreams"],
                    gateway_cfg.get("upstreams", []),
                )
                if body.get("persist_env", False):
                    gateway_env_updates = _dashboard_gateway_upstream_env_updates(g["upstreams"])
                    env_updates.update(gateway_env_updates)
                elif _dashboard_gateway_upstreams_have_key_values(g["upstreams"]):
                    return JSONResponse(
                        {"error": "gateway upstream api_key_values require persist_env=true"},
                        status_code=400,
                    )
            except ValueError as e:
                return JSONResponse({"error": str(e)}, status_code=400)
            gateway_cfg["upstreams"] = sanitized_upstreams
            gateway_hot_update_body["upstreams"] = _dashboard_gateway_hot_upstreams(
                sanitized_upstreams,
                g["upstreams"],
                env_updates,
            )
            updated.append("gateway.upstreams")
        if "cooldown_hours" in g:
            gateway_cfg["cooldown_hours"] = max(0.0, float(g["cooldown_hours"]))
            gateway_hot_update_body["cooldown_hours"] = gateway_cfg["cooldown_hours"]
            updated.append("gateway.cooldown_hours")
        if "skip_recent_rounds" in g:
            gateway_cfg["skip_recent_rounds"] = max(0, int(g["skip_recent_rounds"]))
            gateway_hot_update_body["skip_recent_rounds"] = gateway_cfg["skip_recent_rounds"]
            updated.append("gateway.skip_recent_rounds")
        if "recent_context_cooldown_hours" in g:
            gateway_cfg["recent_context_cooldown_hours"] = max(0.0, float(g["recent_context_cooldown_hours"]))
            gateway_hot_update_body["recent_context_cooldown_hours"] = gateway_cfg["recent_context_cooldown_hours"]
            updated.append("gateway.recent_context_cooldown_hours")
        if "recent_context_reentry_idle_hours" in g:
            gateway_cfg["recent_context_reentry_idle_hours"] = max(
                0.0,
                float(g["recent_context_reentry_idle_hours"]),
            )
            gateway_hot_update_body["recent_context_reentry_idle_hours"] = gateway_cfg[
                "recent_context_reentry_idle_hours"
            ]
            updated.append("gateway.recent_context_reentry_idle_hours")
        if "recent_context_budget" in g:
            gateway_cfg["recent_context_budget"] = max(0, int(g["recent_context_budget"]))
            gateway_hot_update_body["recent_context_budget"] = gateway_cfg["recent_context_budget"]
            updated.append("gateway.recent_context_budget")
        if "date_persona_trace_enabled" in g:
            gateway_cfg["date_persona_trace_enabled"] = _bool_value(g["date_persona_trace_enabled"], True)
            gateway_hot_update_body["date_persona_trace_enabled"] = gateway_cfg["date_persona_trace_enabled"]
            updated.append("gateway.date_persona_trace_enabled")
        if "date_persona_trace_budget" in g:
            gateway_cfg["date_persona_trace_budget"] = max(0, int(g["date_persona_trace_budget"]))
            gateway_hot_update_body["date_persona_trace_budget"] = gateway_cfg["date_persona_trace_budget"]
            updated.append("gateway.date_persona_trace_budget")
        if "date_persona_trace_max_events" in g:
            gateway_cfg["date_persona_trace_max_events"] = max(0, min(8, int(g["date_persona_trace_max_events"])))
            gateway_hot_update_body["date_persona_trace_max_events"] = gateway_cfg["date_persona_trace_max_events"]
            updated.append("gateway.date_persona_trace_max_events")
        if "date_persona_trace_include_daily" in g:
            gateway_cfg["date_persona_trace_include_daily"] = _bool_value(
                g["date_persona_trace_include_daily"],
                True,
            )
            gateway_hot_update_body["date_persona_trace_include_daily"] = gateway_cfg[
                "date_persona_trace_include_daily"
            ]
            updated.append("gateway.date_persona_trace_include_daily")
        if "recalled_memory_budget" in g:
            gateway_cfg["recalled_memory_budget"] = max(0, int(g["recalled_memory_budget"]))
            gateway_hot_update_body["recalled_memory_budget"] = gateway_cfg["recalled_memory_budget"]
            updated.append("gateway.recalled_memory_budget")
        if "related_memory_budget" in g:
            gateway_cfg["related_memory_budget"] = max(0, int(g["related_memory_budget"]))
            gateway_hot_update_body["related_memory_budget"] = gateway_cfg["related_memory_budget"]
            updated.append("gateway.related_memory_budget")
        if "memory_sentinel_enabled" in g:
            gateway_cfg["memory_sentinel_enabled"] = _bool_value(g["memory_sentinel_enabled"], True)
            gateway_hot_update_body["memory_sentinel_enabled"] = gateway_cfg["memory_sentinel_enabled"]
            updated.append("gateway.memory_sentinel_enabled")
        if "domain_sentinel_enabled" in g:
            gateway_cfg["domain_sentinel_enabled"] = _bool_value(g["domain_sentinel_enabled"], True)
            gateway_hot_update_body["domain_sentinel_enabled"] = gateway_cfg["domain_sentinel_enabled"]
            updated.append("gateway.domain_sentinel_enabled")
        if "domain_sentinel_model" in g:
            gateway_cfg["domain_sentinel_model"] = str(g["domain_sentinel_model"] or "").strip()
            gateway_hot_update_body["domain_sentinel_model"] = gateway_cfg["domain_sentinel_model"]
            updated.append("gateway.domain_sentinel_model")
        if "domain_sentinel_base_url" in g:
            gateway_cfg["domain_sentinel_base_url"] = str(g["domain_sentinel_base_url"] or "").strip()
            gateway_hot_update_body["domain_sentinel_base_url"] = gateway_cfg["domain_sentinel_base_url"]
            updated.append("gateway.domain_sentinel_base_url")
        if "domain_sentinel_api_key" in g and g["domain_sentinel_api_key"]:
            gateway_cfg["domain_sentinel_api_key"] = str(g["domain_sentinel_api_key"])
            os.environ["OMBRE_DOMAIN_SENTINEL_API_KEY"] = gateway_cfg["domain_sentinel_api_key"]
            env_updates["OMBRE_DOMAIN_SENTINEL_API_KEY"] = gateway_cfg["domain_sentinel_api_key"]
            gateway_hot_update_body["domain_sentinel_api_key"] = gateway_cfg["domain_sentinel_api_key"]
            updated.append("gateway.domain_sentinel_api_key")
        if domain_sentinel_touched:
            gateway_cfg["domain_sentinel_enable_thinking"] = False
            gateway_hot_update_body["domain_sentinel_enable_thinking"] = False
        if "domain_sentinel_max_tokens" in g:
            gateway_cfg["domain_sentinel_max_tokens"] = _int_between(
                g["domain_sentinel_max_tokens"],
                260,
                64,
                800,
            )
            gateway_hot_update_body["domain_sentinel_max_tokens"] = gateway_cfg["domain_sentinel_max_tokens"]
            updated.append("gateway.domain_sentinel_max_tokens")
        if "current_inner_state_interval_rounds" in g:
            gateway_cfg["current_inner_state_interval_rounds"] = max(
                0,
                int(g["current_inner_state_interval_rounds"]),
            )
            gateway_hot_update_body["current_inner_state_interval_rounds"] = gateway_cfg[
                "current_inner_state_interval_rounds"
            ]
            updated.append("gateway.current_inner_state_interval_rounds")
        if "direct_render_mode" in g:
            gateway_cfg["direct_render_mode"] = _normalize_direct_render_mode(g["direct_render_mode"])
            gateway_hot_update_body["direct_render_mode"] = gateway_cfg["direct_render_mode"]
            updated.append("gateway.direct_render_mode")
        if "retrieval_mode" in g:
            gateway_cfg["retrieval_mode"] = _normalize_retrieval_mode(g["retrieval_mode"])
            gateway_hot_update_body["retrieval_mode"] = gateway_cfg["retrieval_mode"]
            updated.append("gateway.retrieval_mode")
        if "operit_context_rewrite_enabled" in g:
            gateway_cfg["operit_context_rewrite_enabled"] = _bool_value(
                g["operit_context_rewrite_enabled"],
                False,
            )
            gateway_hot_update_body["operit_context_rewrite_enabled"] = gateway_cfg[
                "operit_context_rewrite_enabled"
            ]
            updated.append("gateway.operit_context_rewrite_enabled")
        if "recall_fusion_mode" in g:
            gateway_cfg["recall_fusion_mode"] = _normalize_recall_fusion_mode(g["recall_fusion_mode"])
            gateway_hot_update_body["recall_fusion_mode"] = gateway_cfg["recall_fusion_mode"]
            updated.append("gateway.recall_fusion_mode")
        if "word_map_hint_enabled" in g:
            gateway_cfg["word_map_hint_enabled"] = _bool_value(g["word_map_hint_enabled"], False)
            gateway_hot_update_body["word_map_hint_enabled"] = gateway_cfg["word_map_hint_enabled"]
            updated.append("gateway.word_map_hint_enabled")
        if "portrait_memory_enabled" in g:
            gateway_cfg["portrait_memory_enabled"] = _bool_value(g["portrait_memory_enabled"], False)
            gateway_hot_update_body["portrait_memory_enabled"] = gateway_cfg["portrait_memory_enabled"]
            updated.append("gateway.portrait_memory_enabled")
        if "portrait_memory_budget" in g:
            gateway_cfg["portrait_memory_budget"] = _int_between(g["portrait_memory_budget"], 360, 120, 2000)
            gateway_hot_update_body["portrait_memory_budget"] = gateway_cfg["portrait_memory_budget"]
            updated.append("gateway.portrait_memory_budget")
        if "portrait_memory_max_sources" in g:
            gateway_cfg["portrait_memory_max_sources"] = _int_between(g["portrait_memory_max_sources"], 8, 1, 20)
            gateway_hot_update_body["portrait_memory_max_sources"] = gateway_cfg["portrait_memory_max_sources"]
            updated.append("gateway.portrait_memory_max_sources")
        if "portrait_memory_include_anchors" in g:
            gateway_cfg["portrait_memory_include_anchors"] = _bool_value(
                g["portrait_memory_include_anchors"],
                False,
            )
            gateway_hot_update_body["portrait_memory_include_anchors"] = gateway_cfg[
                "portrait_memory_include_anchors"
            ]
            updated.append("gateway.portrait_memory_include_anchors")
        if "query_planner_enabled" in g:
            gateway_cfg["query_planner_enabled"] = _bool_value(g["query_planner_enabled"], True)
            gateway_hot_update_body["query_planner_enabled"] = gateway_cfg["query_planner_enabled"]
            updated.append("gateway.query_planner_enabled")
        if "query_planner_model" in g:
            gateway_cfg["query_planner_model"] = str(g["query_planner_model"] or "").strip()
            gateway_hot_update_body["query_planner_model"] = gateway_cfg["query_planner_model"]
            updated.append("gateway.query_planner_model")
        if "query_planner_min_chars" in g:
            gateway_cfg["query_planner_min_chars"] = _int_between(g["query_planner_min_chars"], 16, 0, 1000)
            gateway_hot_update_body["query_planner_min_chars"] = gateway_cfg["query_planner_min_chars"]
            updated.append("gateway.query_planner_min_chars")
        if "query_planner_max_queries" in g:
            gateway_cfg["query_planner_max_queries"] = _int_between(g["query_planner_max_queries"], 3, 1, 3)
            gateway_hot_update_body["query_planner_max_queries"] = gateway_cfg["query_planner_max_queries"]
            updated.append("gateway.query_planner_max_queries")
        if "query_planner_max_tokens" in g:
            gateway_cfg["query_planner_max_tokens"] = _int_between(g["query_planner_max_tokens"], 360, 128, 2000)
            gateway_hot_update_body["query_planner_max_tokens"] = gateway_cfg["query_planner_max_tokens"]
            updated.append("gateway.query_planner_max_tokens")
        if "memory_detail_recall_enabled" in g:
            gateway_cfg["memory_detail_recall_enabled"] = _bool_value(g["memory_detail_recall_enabled"], False)
            gateway_hot_update_body["memory_detail_recall_enabled"] = gateway_cfg["memory_detail_recall_enabled"]
            updated.append("gateway.memory_detail_recall_enabled")
        if "memory_detail_recall_max_ids" in g:
            gateway_cfg["memory_detail_recall_max_ids"] = _int_between(g["memory_detail_recall_max_ids"], 3, 1, 3)
            gateway_hot_update_body["memory_detail_recall_max_ids"] = gateway_cfg["memory_detail_recall_max_ids"]
            updated.append("gateway.memory_detail_recall_max_ids")
        if "memory_detail_recall_budget" in g:
            gateway_cfg["memory_detail_recall_budget"] = _int_between(g["memory_detail_recall_budget"], 1200, 200, 4000)
            gateway_hot_update_body["memory_detail_recall_budget"] = gateway_cfg["memory_detail_recall_budget"]
            updated.append("gateway.memory_detail_recall_budget")
        if gateway_hot_update_body:
            gateway_hot_update_payload["gateway"] = gateway_hot_update_body

    if "self_anchor" in body and isinstance(body["self_anchor"], dict):
        self_anchor_cfg = config.setdefault("self_anchor", {})
        self_anchor_cfg["entry_bucket_id"] = str(body["self_anchor"].get("entry_bucket_id") or "").strip()
        updated.append("self_anchor.entry_bucket_id")

    # --- Recall behavior config ---
    if "recall" in body:
        r = body["recall"]
        recall_cfg = config.setdefault("recall", {})
        if "query_resurface_enabled" in r:
            recall_cfg["query_resurface_enabled"] = _bool_value(r["query_resurface_enabled"], False)
            updated.append("recall.query_resurface_enabled")

    # --- Memory diffusion config ---
    if "memory_diffusion" in body:
        diffusion_payload = _memory_diffusion_dashboard_config(body["memory_diffusion"])
        diffusion_cfg = config.setdefault("memory_diffusion", {})
        for key, value in diffusion_payload.items():
            diffusion_cfg[key] = value
            updated.append(f"memory_diffusion.{key}")
        if diffusion_payload:
            gateway_hot_update_payload["memory_diffusion"] = diffusion_payload

    # --- Persona state config ---
    if "persona" in body:
        p = body["persona"]
        persona_cfg = config.setdefault("persona", {})
        persona_gateway_payload = {}
        if "enabled" in p:
            persona_cfg["enabled"] = bool(p["enabled"])
            persona_gateway_payload["enabled"] = persona_cfg["enabled"]
            updated.append("persona.enabled")
        if "event_recording_enabled" in p:
            persona_cfg["event_recording_enabled"] = bool(p["event_recording_enabled"])
            persona_gateway_payload["event_recording_enabled"] = persona_cfg["event_recording_enabled"]
            updated.append("persona.event_recording_enabled")
        for key in ("model", "base_url"):
            if key in p:
                persona_cfg[key] = str(p[key] or "").strip()
                persona_gateway_payload[key] = persona_cfg[key]
                updated.append(f"persona.{key}")
        if "api_key" in p and p["api_key"]:
            persona_cfg["api_key"] = str(p["api_key"])
            os.environ["OMBRE_PERSONA_API_KEY"] = persona_cfg["api_key"]
            env_updates["OMBRE_PERSONA_API_KEY"] = persona_cfg["api_key"]
            persona_gateway_payload["api_key"] = persona_cfg["api_key"]
            updated.append("persona.api_key")
        if "base_url" in persona_gateway_payload and persona_gateway_payload["base_url"]:
            os.environ["OMBRE_PERSONA_BASE_URL"] = persona_gateway_payload["base_url"]
        if "model" in persona_gateway_payload and persona_gateway_payload["model"]:
            os.environ["OMBRE_PERSONA_MODEL"] = persona_gateway_payload["model"]
        if persona_gateway_payload:
            persona_engine = PersonaStateEngine(config)
            gateway_hot_update_payload["persona"] = persona_gateway_payload

    # --- Reflection config ---
    if "reflection" in body:
        r = body["reflection"]
        reflection_cfg = config.setdefault("reflection", {})
        for key in (
            "enabled",
            "auto_enabled",
            "daily_enabled",
            "memory_affect_anchor_enabled",
            "relationship_weather_affect_anchor_enabled",
        ):
            if key in r:
                reflection_cfg[key] = bool(r[key])
                setattr(reflection_engine, key, reflection_cfg[key])
                updated.append(f"reflection.{key}")
        for key in ("model", "base_url"):
            if key in r:
                reflection_cfg[key] = str(r[key] or "").strip()
                updated.append(f"reflection.{key}")
        if "daily_min_memory_items" in r:
            reflection_cfg["daily_min_memory_items"] = _int_between(
                r.get("daily_min_memory_items"),
                5,
                0,
                100,
            )
            updated.append("reflection.daily_min_memory_items")
        if "daily_conversation_turn_limit" in r:
            reflection_cfg["daily_conversation_turn_limit"] = _int_between(
                r.get("daily_conversation_turn_limit"),
                12,
                0,
                80,
            )
            updated.append("reflection.daily_conversation_turn_limit")
        if "daily_activity_summary_enabled" in r:
            reflection_cfg["daily_activity_summary_enabled"] = bool(r.get("daily_activity_summary_enabled"))
            updated.append("reflection.daily_activity_summary_enabled")
        if "daily_activity_summary_turn_limit" in r:
            reflection_cfg["daily_activity_summary_turn_limit"] = _int_between(
                r.get("daily_activity_summary_turn_limit"),
                0,
                0,
                10000,
            )
            updated.append("reflection.daily_activity_summary_turn_limit")
        if "daily_activity_summary_max_tokens" in r:
            reflection_cfg["daily_activity_summary_max_tokens"] = _int_between(
                r.get("daily_activity_summary_max_tokens"),
                320,
                80,
                1000,
            )
            updated.append("reflection.daily_activity_summary_max_tokens")
        if "daily_chat_memory_mode" in r:
            mode = str(r.get("daily_chat_memory_mode") or "review").strip().lower()
            if mode not in {"auto", "review", "off"}:
                mode = "review"
            reflection_cfg["daily_chat_memory_mode"] = mode
            updated.append("reflection.daily_chat_memory_mode")
        if "daily_chat_memory_hour" in r:
            reflection_cfg["daily_chat_memory_hour"] = _int_between(
                r.get("daily_chat_memory_hour"),
                0,
                0,
                23,
            )
            updated.append("reflection.daily_chat_memory_hour")
        if "daily_chat_memory_turn_limit" in r:
            reflection_cfg["daily_chat_memory_turn_limit"] = _int_between(
                r.get("daily_chat_memory_turn_limit"),
                0,
                0,
                10000,
            )
            updated.append("reflection.daily_chat_memory_turn_limit")
        if "daily_chat_memory_max_per_day" in r:
            reflection_cfg["daily_chat_memory_max_per_day"] = _int_between(
                r.get("daily_chat_memory_max_per_day"),
                3,
                0,
                10,
            )
            updated.append("reflection.daily_chat_memory_max_per_day")
        if "thinking_mode" in r:
            thinking_mode = str(r.get("thinking_mode") or "").strip().lower()
            if thinking_mode not in {"", "enabled", "disabled"}:
                thinking_mode = ""
            reflection_cfg["thinking_mode"] = thinking_mode
            updated.append("reflection.thinking_mode")
        if "api_key" in r and r["api_key"]:
            reflection_cfg["api_key"] = str(r["api_key"])
            os.environ["OMBRE_REFLECTION_API_KEY"] = reflection_cfg["api_key"]
            env_updates["OMBRE_REFLECTION_API_KEY"] = reflection_cfg["api_key"]
            updated.append("reflection.api_key")
        if "base_url" in r and reflection_cfg.get("base_url"):
            os.environ["OMBRE_REFLECTION_BASE_URL"] = reflection_cfg["base_url"]
        if "model" in r and reflection_cfg.get("model"):
            os.environ["OMBRE_REFLECTION_MODEL"] = reflection_cfg["model"]
        reflection_engine = ReflectionEngine(config)
        metadata_enricher = reflection_engine

    # --- Portrait maintainer config ---
    if "portrait" in body:
        p = body["portrait"]
        portrait_cfg = config.setdefault("portrait", {})
        for key in (
            "enabled",
            "auto_enabled",
            "auto_initial_enabled",
            "daily_enabled",
        ):
            if key in p:
                portrait_cfg[key] = bool(p[key])
                updated.append(f"portrait.{key}")
        for key in (
            "model",
            "base_url",
            "state_path",
            "thinking_mode",
        ):
            if key in p:
                portrait_cfg[key] = str(p[key] or "").strip()
                updated.append(f"portrait.{key}")
        for key in (
            "temperature",
            "max_tokens",
            "daily_hour",
            "check_interval_minutes",
            "material_limit",
            "first_run_material_limit",
            "persona_events_limit",
            "recent_buffer_max",
            "staging_pool_max",
            "candidate_max",
            "user_rewrite_evidence_delta",
            "manual_suppress_days",
        ):
            if key in p:
                portrait_cfg[key] = p[key]
                updated.append(f"portrait.{key}")
        if "api_key" in p and p["api_key"]:
            portrait_cfg["api_key"] = str(p["api_key"])
            os.environ["OMBRE_PORTRAIT_API_KEY"] = portrait_cfg["api_key"]
            env_updates["OMBRE_PORTRAIT_API_KEY"] = portrait_cfg["api_key"]
            updated.append("portrait.api_key")
        if "base_url" in p and portrait_cfg.get("base_url"):
            os.environ["OMBRE_PORTRAIT_BASE_URL"] = portrait_cfg["base_url"]
        if "model" in p and portrait_cfg.get("model"):
            os.environ["OMBRE_PORTRAIT_MODEL"] = portrait_cfg["model"]
        portrait_engine = DailyPortraitMaintainer(config)

    # --- Dream config ---
    if "dream" in body:
        d = body["dream"]
        dream_cfg = config.setdefault("dream", {})
        for key in (
            "enabled",
            "auto_enabled",
            "surface_enabled",
            "inject_enabled",
            "retain_after_inject",
            "model",
            "base_url",
            "temperature",
            "max_tokens",
            "daily_hour",
            "run_window_hours",
            "daily_probability",
            "min_material_count",
            "material_window_hours",
            "raw_residue_enabled",
            "raw_residue_turns",
            "raw_residue_max_chars",
            "identity_anchor_id",
        ):
            if key in d:
                dream_cfg[key] = d[key]
                updated.append(f"dream.{key}")
        dream_gateway_payload = {}
        for key in ("enabled", "surface_enabled", "inject_enabled", "retain_after_inject"):
            if key in d:
                dream_gateway_payload[key] = dream_cfg[key]
        if dream_gateway_payload:
            gateway_hot_update_payload["dream"] = dream_gateway_payload
        if "api_key" in d and d["api_key"]:
            dream_cfg["api_key"] = str(d["api_key"])
            os.environ["OMBRE_DREAM_API_KEY"] = dream_cfg["api_key"]
            env_updates["OMBRE_DREAM_API_KEY"] = dream_cfg["api_key"]
            updated.append("dream.api_key")
        if "base_url" in d and dream_cfg.get("base_url"):
            os.environ["OMBRE_DREAM_BASE_URL"] = str(dream_cfg["base_url"])
        if "model" in d and dream_cfg.get("model"):
            os.environ["OMBRE_DREAM_MODEL"] = str(dream_cfg["model"])
        dream_engine = DreamEngine(config)

    hot_update_status = await _hot_update_gateway_config(gateway_hot_update_payload)
    if hot_update_status:
        updated.append(hot_update_status)

    # --- Persist to config.yaml if requested ---
    if body.get("persist", False):
        config_path = os.environ.get(
            "OMBRE_CONFIG_PATH",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"),
        )
        runtime_config_path = config.get("_runtime_config_path") or os.environ.get("OMBRE_RUNTIME_CONFIG_PATH", "")
        if not runtime_config_path:
            runtime_config_path = os.path.join(config.get("state_dir") or os.path.dirname(config_path), "config.runtime.yaml")

        def _apply_dashboard_config(save_config: dict) -> dict:
            save_config = save_config or {}
            if "dehydration" in body:
                sc_dehy = save_config.setdefault("dehydration", {})
                for key in ("model", "base_url", "max_tokens", "temperature"):
                    if key in body["dehydration"]:
                        sc_dehy[key] = body["dehydration"][key]
                # Never persist api_key to yaml (use env var)

            if "embedding" in body:
                sc_emb = save_config.setdefault("embedding", {})
                for key in ("enabled", "model", "base_url"):
                    if key in body["embedding"]:
                        sc_emb[key] = body["embedding"][key]
                # Never persist api_key to yaml (use env var)

            if "reranker" in body:
                sc_reranker = save_config.setdefault("reranker", {})
                if "enabled" in body["reranker"]:
                    sc_reranker["enabled"] = _bool_value(body["reranker"]["enabled"], True)
                for key in ("model", "base_url"):
                    if key in body["reranker"]:
                        sc_reranker[key] = str(body["reranker"][key] or "").strip()
                if "timeout_seconds" in body["reranker"]:
                    sc_reranker["timeout_seconds"] = _float_between(
                        body["reranker"]["timeout_seconds"],
                        12,
                        1,
                        120,
                    )
                if "candidate_limit" in body["reranker"]:
                    sc_reranker["candidate_limit"] = _int_between(
                        body["reranker"]["candidate_limit"],
                        20,
                        1,
                        100,
                    )
                if "score_weight" in body["reranker"]:
                    sc_reranker["score_weight"] = _float_between(
                        body["reranker"]["score_weight"],
                        0.65,
                        0.0,
                        1.0,
                    )
                # Never persist api_key to yaml (use env var)

            if "merge_threshold" in body:
                save_config["merge_threshold"] = int(body["merge_threshold"])

            if "gateway" in body:
                sc_gateway = save_config.setdefault("gateway", {})
                domain_sentinel_touched = any(
                    key in body["gateway"]
                    for key in (
                        "domain_sentinel_enabled",
                        "domain_sentinel_model",
                        "domain_sentinel_base_url",
                        "domain_sentinel_enable_thinking",
                    )
                )
                if "upstreams" in body["gateway"]:
                    sc_gateway["upstreams"] = _dashboard_sanitize_gateway_upstreams(
                        body["gateway"]["upstreams"],
                        sc_gateway.get("upstreams", []),
                    )
                if "cooldown_hours" in body["gateway"]:
                    sc_gateway["cooldown_hours"] = max(0.0, float(body["gateway"]["cooldown_hours"]))
                if "skip_recent_rounds" in body["gateway"]:
                    sc_gateway["skip_recent_rounds"] = max(0, int(body["gateway"]["skip_recent_rounds"]))
                if "recent_context_cooldown_hours" in body["gateway"]:
                    sc_gateway["recent_context_cooldown_hours"] = max(
                        0.0,
                        float(body["gateway"]["recent_context_cooldown_hours"]),
                    )
                if "recent_context_reentry_idle_hours" in body["gateway"]:
                    sc_gateway["recent_context_reentry_idle_hours"] = max(
                        0.0,
                        float(body["gateway"]["recent_context_reentry_idle_hours"]),
                    )
                if "recent_context_budget" in body["gateway"]:
                    sc_gateway["recent_context_budget"] = max(0, int(body["gateway"]["recent_context_budget"]))
                if "date_persona_trace_enabled" in body["gateway"]:
                    sc_gateway["date_persona_trace_enabled"] = _bool_value(
                        body["gateway"]["date_persona_trace_enabled"],
                        True,
                    )
                if "date_persona_trace_budget" in body["gateway"]:
                    sc_gateway["date_persona_trace_budget"] = max(
                        0,
                        int(body["gateway"]["date_persona_trace_budget"]),
                    )
                if "date_persona_trace_max_events" in body["gateway"]:
                    sc_gateway["date_persona_trace_max_events"] = max(
                        0,
                        min(8, int(body["gateway"]["date_persona_trace_max_events"])),
                    )
                if "date_persona_trace_include_daily" in body["gateway"]:
                    sc_gateway["date_persona_trace_include_daily"] = _bool_value(
                        body["gateway"]["date_persona_trace_include_daily"],
                        True,
                    )
                if "recalled_memory_budget" in body["gateway"]:
                    sc_gateway["recalled_memory_budget"] = max(0, int(body["gateway"]["recalled_memory_budget"]))
                if "related_memory_budget" in body["gateway"]:
                    sc_gateway["related_memory_budget"] = max(0, int(body["gateway"]["related_memory_budget"]))
                if "memory_sentinel_enabled" in body["gateway"]:
                    sc_gateway["memory_sentinel_enabled"] = _bool_value(
                        body["gateway"]["memory_sentinel_enabled"],
                        True,
                    )
                if "domain_sentinel_enabled" in body["gateway"]:
                    sc_gateway["domain_sentinel_enabled"] = _bool_value(
                        body["gateway"]["domain_sentinel_enabled"],
                        True,
                    )
                if "domain_sentinel_model" in body["gateway"]:
                    sc_gateway["domain_sentinel_model"] = str(
                        body["gateway"]["domain_sentinel_model"] or ""
                    ).strip()
                if "domain_sentinel_base_url" in body["gateway"]:
                    sc_gateway["domain_sentinel_base_url"] = str(
                        body["gateway"]["domain_sentinel_base_url"] or ""
                    ).strip()
                if domain_sentinel_touched:
                    sc_gateway["domain_sentinel_enable_thinking"] = False
                if "domain_sentinel_max_tokens" in body["gateway"]:
                    sc_gateway["domain_sentinel_max_tokens"] = _int_between(
                        body["gateway"]["domain_sentinel_max_tokens"],
                        260,
                        64,
                        800,
                    )
                if "current_inner_state_interval_rounds" in body["gateway"]:
                    sc_gateway["current_inner_state_interval_rounds"] = max(
                        0,
                        int(body["gateway"]["current_inner_state_interval_rounds"]),
                    )
                if "direct_render_mode" in body["gateway"]:
                    sc_gateway["direct_render_mode"] = _normalize_direct_render_mode(body["gateway"]["direct_render_mode"])
                if "retrieval_mode" in body["gateway"]:
                    sc_gateway["retrieval_mode"] = _normalize_retrieval_mode(body["gateway"]["retrieval_mode"])
                if "operit_context_rewrite_enabled" in body["gateway"]:
                    sc_gateway["operit_context_rewrite_enabled"] = _bool_value(
                        body["gateway"]["operit_context_rewrite_enabled"],
                        False,
                    )
                if "recall_fusion_mode" in body["gateway"]:
                    sc_gateway["recall_fusion_mode"] = _normalize_recall_fusion_mode(
                        body["gateway"]["recall_fusion_mode"]
                    )
                if "portrait_memory_enabled" in body["gateway"]:
                    sc_gateway["portrait_memory_enabled"] = _bool_value(
                        body["gateway"]["portrait_memory_enabled"],
                        False,
                    )
                if "portrait_memory_budget" in body["gateway"]:
                    sc_gateway["portrait_memory_budget"] = _int_between(
                        body["gateway"]["portrait_memory_budget"],
                        360,
                        120,
                        2000,
                    )
                if "portrait_memory_max_sources" in body["gateway"]:
                    sc_gateway["portrait_memory_max_sources"] = _int_between(
                        body["gateway"]["portrait_memory_max_sources"],
                        8,
                        1,
                        20,
                    )
                if "portrait_memory_include_anchors" in body["gateway"]:
                    sc_gateway["portrait_memory_include_anchors"] = _bool_value(
                        body["gateway"]["portrait_memory_include_anchors"],
                        False,
                    )
                if "query_planner_enabled" in body["gateway"]:
                    sc_gateway["query_planner_enabled"] = _bool_value(
                        body["gateway"]["query_planner_enabled"],
                        True,
                    )
                if "query_planner_model" in body["gateway"]:
                    sc_gateway["query_planner_model"] = str(body["gateway"]["query_planner_model"] or "").strip()
                if "query_planner_min_chars" in body["gateway"]:
                    sc_gateway["query_planner_min_chars"] = _int_between(
                        body["gateway"]["query_planner_min_chars"],
                        16,
                        0,
                        1000,
                    )
                if "query_planner_max_queries" in body["gateway"]:
                    sc_gateway["query_planner_max_queries"] = _int_between(
                        body["gateway"]["query_planner_max_queries"],
                        3,
                        1,
                        3,
                    )
                if "query_planner_max_tokens" in body["gateway"]:
                    sc_gateway["query_planner_max_tokens"] = _int_between(
                        body["gateway"]["query_planner_max_tokens"],
                        360,
                        128,
                        2000,
                    )
                if "word_map_hint_enabled" in body["gateway"]:
                    sc_gateway["word_map_hint_enabled"] = _bool_value(
                        body["gateway"]["word_map_hint_enabled"],
                        False,
                    )
                if "memory_detail_recall_enabled" in body["gateway"]:
                    sc_gateway["memory_detail_recall_enabled"] = _bool_value(
                        body["gateway"]["memory_detail_recall_enabled"],
                        False,
                    )
                if "memory_detail_recall_max_ids" in body["gateway"]:
                    sc_gateway["memory_detail_recall_max_ids"] = _int_between(
                        body["gateway"]["memory_detail_recall_max_ids"],
                        3,
                        1,
                        3,
                    )
                if "memory_detail_recall_budget" in body["gateway"]:
                    sc_gateway["memory_detail_recall_budget"] = _int_between(
                        body["gateway"]["memory_detail_recall_budget"],
                        1200,
                        200,
                        4000,
                    )

            if "self_anchor" in body and isinstance(body["self_anchor"], dict):
                sc_self_anchor = save_config.setdefault("self_anchor", {})
                if "entry_bucket_id" in body["self_anchor"]:
                    sc_self_anchor["entry_bucket_id"] = str(body["self_anchor"].get("entry_bucket_id") or "").strip()

            if "recall" in body:
                sc_recall = save_config.setdefault("recall", {})
                if "query_resurface_enabled" in body["recall"]:
                    sc_recall["query_resurface_enabled"] = _bool_value(
                        body["recall"]["query_resurface_enabled"],
                        False,
                    )

            if "memory_diffusion" in body:
                sc_diffusion = save_config.setdefault("memory_diffusion", {})
                for key, value in _memory_diffusion_dashboard_config(body["memory_diffusion"]).items():
                    sc_diffusion[key] = value

            if "persona" in body:
                sc_persona = save_config.setdefault("persona", {})
                if "enabled" in body["persona"]:
                    sc_persona["enabled"] = bool(body["persona"]["enabled"])
                if "event_recording_enabled" in body["persona"]:
                    sc_persona["event_recording_enabled"] = bool(
                        body["persona"]["event_recording_enabled"]
                    )
                for key in ("model", "base_url"):
                    if key in body["persona"]:
                        sc_persona[key] = str(body["persona"][key] or "").strip()
                # Never persist api_key to yaml (use env var)

            if "reflection" in body:
                sc_reflection = save_config.setdefault("reflection", {})
                for key in (
                    "enabled",
                    "auto_enabled",
                    "daily_enabled",
                    "memory_affect_anchor_enabled",
                    "relationship_weather_affect_anchor_enabled",
                ):
                    if key in body["reflection"]:
                        sc_reflection[key] = bool(body["reflection"][key])
                for key in ("model", "base_url"):
                    if key in body["reflection"]:
                        sc_reflection[key] = str(body["reflection"][key] or "").strip()
                if "daily_min_memory_items" in body["reflection"]:
                    sc_reflection["daily_min_memory_items"] = _int_between(
                        body["reflection"].get("daily_min_memory_items"),
                        5,
                        0,
                        100,
                    )
                if "daily_conversation_turn_limit" in body["reflection"]:
                    sc_reflection["daily_conversation_turn_limit"] = _int_between(
                        body["reflection"].get("daily_conversation_turn_limit"),
                        12,
                        0,
                        80,
                    )
                if "daily_activity_summary_enabled" in body["reflection"]:
                    sc_reflection["daily_activity_summary_enabled"] = bool(
                        body["reflection"].get("daily_activity_summary_enabled")
                    )
                if "daily_activity_summary_turn_limit" in body["reflection"]:
                    sc_reflection["daily_activity_summary_turn_limit"] = _int_between(
                        body["reflection"].get("daily_activity_summary_turn_limit"),
                        0,
                        0,
                        10000,
                    )
                if "daily_activity_summary_max_tokens" in body["reflection"]:
                    sc_reflection["daily_activity_summary_max_tokens"] = _int_between(
                        body["reflection"].get("daily_activity_summary_max_tokens"),
                        320,
                        80,
                        1000,
                    )
                if "daily_chat_memory_mode" in body["reflection"]:
                    mode = str(body["reflection"].get("daily_chat_memory_mode") or "review").strip().lower()
                    sc_reflection["daily_chat_memory_mode"] = mode if mode in {"auto", "review", "off"} else "review"
                if "daily_chat_memory_hour" in body["reflection"]:
                    sc_reflection["daily_chat_memory_hour"] = _int_between(
                        body["reflection"].get("daily_chat_memory_hour"),
                        0,
                        0,
                        23,
                    )
                if "daily_chat_memory_turn_limit" in body["reflection"]:
                    sc_reflection["daily_chat_memory_turn_limit"] = _int_between(
                        body["reflection"].get("daily_chat_memory_turn_limit"),
                        0,
                        0,
                        10000,
                    )
                if "daily_chat_memory_max_per_day" in body["reflection"]:
                    sc_reflection["daily_chat_memory_max_per_day"] = _int_between(
                        body["reflection"].get("daily_chat_memory_max_per_day"),
                        3,
                        0,
                        10,
                    )
                if "thinking_mode" in body["reflection"]:
                    thinking_mode = str(body["reflection"].get("thinking_mode") or "").strip().lower()
                    if thinking_mode not in {"", "enabled", "disabled"}:
                        thinking_mode = ""
                    sc_reflection["thinking_mode"] = thinking_mode
                # Never persist api_key to yaml (use env var)

            if "portrait" in body:
                sc_portrait = save_config.setdefault("portrait", {})
                for key in (
                    "enabled",
                    "auto_enabled",
                    "auto_initial_enabled",
                    "daily_enabled",
                ):
                    if key in body["portrait"]:
                        sc_portrait[key] = bool(body["portrait"][key])
                for key in (
                    "model",
                    "base_url",
                    "state_path",
                    "thinking_mode",
                ):
                    if key in body["portrait"]:
                        sc_portrait[key] = str(body["portrait"][key] or "").strip()
                for key in (
                    "temperature",
                    "max_tokens",
                    "daily_hour",
                    "check_interval_minutes",
                    "material_limit",
                    "first_run_material_limit",
                    "persona_events_limit",
                    "recent_buffer_max",
                    "staging_pool_max",
                    "candidate_max",
                    "user_rewrite_evidence_delta",
                    "manual_suppress_days",
                ):
                    if key in body["portrait"]:
                        sc_portrait[key] = body["portrait"][key]
                # Never persist api_key to yaml (use env var)

            if "dream" in body:
                sc_dream = save_config.setdefault("dream", {})
                for key in (
                    "enabled",
                    "auto_enabled",
                    "surface_enabled",
                    "inject_enabled",
                    "retain_after_inject",
                    "model",
                    "base_url",
                    "temperature",
                    "max_tokens",
                    "daily_hour",
                    "run_window_hours",
                    "daily_probability",
                    "min_material_count",
                    "material_window_hours",
                    "raw_residue_enabled",
                    "raw_residue_turns",
                    "raw_residue_max_chars",
                    "identity_anchor_id",
                ):
                    if key in body["dream"]:
                        sc_dream[key] = body["dream"][key]
                # Never persist api_key to yaml (use env var)
            return save_config

        try:
            save_config = {}
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    save_config = yaml.safe_load(f) or {}
            save_config = _apply_dashboard_config(save_config)

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(save_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            updated.append("persisted_to_yaml")
            if os.path.exists(runtime_config_path):
                runtime_config = {}
                with open(runtime_config_path, "r", encoding="utf-8") as f:
                    runtime_config = yaml.safe_load(f) or {}
                runtime_config = _apply_dashboard_config(runtime_config)
                os.makedirs(os.path.dirname(runtime_config_path), exist_ok=True)
                with open(runtime_config_path, "w", encoding="utf-8") as f:
                    yaml.dump(runtime_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
                updated.append("runtime_yaml_synced")
        except Exception as e:
            try:
                runtime_config = {}
                if os.path.exists(runtime_config_path):
                    with open(runtime_config_path, "r", encoding="utf-8") as f:
                        runtime_config = yaml.safe_load(f) or {}
                runtime_config = _apply_dashboard_config(runtime_config)
                os.makedirs(os.path.dirname(runtime_config_path), exist_ok=True)
                with open(runtime_config_path, "w", encoding="utf-8") as f:
                    yaml.dump(runtime_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
                updated.append("persisted_to_runtime_yaml")
                updated.append(f"config_yaml_unwritable:{type(e).__name__}")
            except Exception as fallback_e:
                return JSONResponse(
                    {"error": f"persist failed: {e}; runtime persist failed: {fallback_e}", "updated": updated},
                    status_code=500,
                )

    if body.get("persist_env", False):
        if "OMBRE_API_KEY" not in env_updates:
            current_dehydration_key = str(config.get("dehydration", {}).get("api_key") or "").strip()
            if current_dehydration_key:
                env_updates["OMBRE_API_KEY"] = current_dehydration_key
                updated.append("dehydration.api_key_from_runtime")
        try:
            env_updated = _write_dashboard_env_values(env_updates)
            if env_updated:
                updated.extend(env_updated)
                updated.append("persisted_to_env")
        except Exception as e:
            return JSONResponse(
                {"error": f"env persist failed: {e}", "updated": updated},
                status_code=500,
            )

    return JSONResponse({"updated": updated, "ok": True})


@mcp.custom_route("/api/status", methods=["GET"])
async def api_status(request):
    """Return dashboard-visible system status."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        stats = await bucket_mgr.get_stats()
        return JSONResponse(
            {
                "decay_engine": "running" if decay_engine.is_running else "stopped",
                "buckets": {
                    "permanent": stats.get("permanent_count", 0),
                    "dynamic": stats.get("dynamic_count", 0),
                    "archive": stats.get("archive_count", 0),
                    "feel": stats.get("feel_count", 0),
                    "total": stats.get("permanent_count", 0)
                    + stats.get("dynamic_count", 0)
                    + stats.get("archive_count", 0)
                    + stats.get("feel_count", 0),
                },
                "mcp_surface": mcp.surface_status(),
                "memory_object_writes": memory_object_write_status(config),
                "using_env_password": bool(os.environ.get("OMBRE_DASHBOARD_PASSWORD", "")),
            }
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# =============================================================
# Import API — conversation history import
# 导入 API — 对话历史导入
# =============================================================

@mcp.custom_route("/api/import/upload", methods=["POST"])
async def api_import_upload(request):
    """Upload a conversation file and start import."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err

    if import_engine.is_running:
        return JSONResponse({"error": "Import already running"}, status_code=409)

    content_type = request.headers.get("content-type", "")
    filename = ""

    try:
        if "multipart/form-data" in content_type:
            form = await request.form()
            file_field = form.get("file")
            if not file_field:
                return JSONResponse({"error": "No file field"}, status_code=400)
            raw_bytes = await file_field.read()
            filename = getattr(file_field, "filename", "upload")
            raw_content = raw_bytes.decode("utf-8", errors="replace")
        else:
            body = await request.body()
            raw_content = body.decode("utf-8", errors="replace")
            # Try to get filename from query params
            filename = request.query_params.get("filename", "upload")

        if not raw_content.strip():
            return JSONResponse({"error": "Empty file"}, status_code=400)

        preserve_raw = request.query_params.get("preserve_raw", "").lower() in ("1", "true")
        resume = request.query_params.get("resume", "").lower() in ("1", "true")
        import_mode = request.query_params.get("import_mode", "auto").strip().lower()
        if import_mode not in ("auto", "operit", "conversation"):
            return JSONResponse({"error": f"Unsupported import mode: {import_mode}"}, status_code=400)
        operit_tagging_value = request.query_params.get("operit_tagging")
        operit_tagging = (
            None
            if operit_tagging_value is None
            else operit_tagging_value.strip().lower() in ("1", "true")
        )

    except Exception as e:
        return JSONResponse({"error": f"Failed to read upload: {e}"}, status_code=400)

    # Start import in background
    async def _run_import():
        try:
            await import_engine.start(
                raw_content,
                filename,
                preserve_raw,
                resume,
                import_mode=import_mode,
                operit_tagging=operit_tagging,
            )
        except Exception as e:
            logger.error(f"Import failed: {e}")

    asyncio.create_task(_run_import())

    return JSONResponse({
        "status": "started",
        "filename": filename,
        "size_bytes": len(raw_content.encode()),
        "import_mode": import_mode,
    })


@mcp.custom_route("/api/import/status", methods=["GET"])
async def api_import_status(request):
    """Get current import progress."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    return JSONResponse(import_engine.get_status())


@mcp.custom_route("/api/import/pause", methods=["POST"])
async def api_import_pause(request):
    """Pause the running import."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    if not import_engine.is_running:
        return JSONResponse({"error": "No import running"}, status_code=400)
    import_engine.pause()
    return JSONResponse({"status": "pause_requested"})


@mcp.custom_route("/api/import/patterns", methods=["GET"])
async def api_import_patterns(request):
    """Detect high-frequency patterns after import."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        patterns = await import_engine.detect_patterns()
        return JSONResponse({"patterns": patterns})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/import/results", methods=["GET"])
async def api_import_results(request):
    """List recently imported/created buckets for review."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        limit = int(request.query_params.get("limit", "50"))
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        # Sort by created time, newest first
        all_buckets.sort(key=lambda b: b["metadata"].get("created", ""), reverse=True)
        results = []
        for b in all_buckets[:limit]:
            results.append({
                "id": b["id"],
                "name": b["metadata"].get("name", ""),
                "content": b["content"][:300],
                "type": b["metadata"].get("type", ""),
                "domain": b["metadata"].get("domain", []),
                "tags": b["metadata"].get("tags", []),
                "importance": b["metadata"].get("importance", 5),
                "created": b["metadata"].get("created", ""),
            })
        return JSONResponse({"buckets": results, "total": len(all_buckets)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/import/review", methods=["POST"])
async def api_import_review(request):
    """Apply review decisions: mark buckets as important/noise/pinned."""
    from starlette.responses import JSONResponse
    err = _require_dashboard_auth(request)
    if err:
        return err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    decisions = body.get("decisions", [])
    if not decisions:
        return JSONResponse({"error": "No decisions provided"}, status_code=400)

    applied = 0
    errors = 0
    for d in decisions:
        bid = d.get("bucket_id", "")
        action = d.get("action", "")
        if not bid or not action:
            continue
        try:
            if action == "important":
                await bucket_mgr.update(bid, importance=9)
            elif action == "pin":
                await bucket_mgr.update(bid, pinned=True)
            elif action == "anchor":
                bucket = await bucket_mgr.get(bid)
                if not bucket:
                    raise ValueError("bucket not found")
                ok, message = await _can_mark_anchor(bid, bucket)
                if not ok:
                    raise ValueError(message)
                await bucket_mgr.update(bid, anchor=True)
            elif action == "noise":
                await bucket_mgr.update(bid, resolved=True, importance=1)
            elif action == "delete":
                result = await _delete_bucket_and_indexes(bid)
                if result.get("status") != "deleted":
                    raise ValueError(result.get("reason") or "bucket not found")
            applied += 1
        except Exception as e:
            logger.warning(f"Review action failed for {bid}: {e}")
            errors += 1

    return JSONResponse({"applied": applied, "errors": errors})


# --- Entry point / 启动入口 ---
if __name__ == "__main__":
    transport = config.get("transport", "stdio")
    logger.info(f"Ombre Brain starting | transport: {transport}")
    recall_warm_started_at = time.perf_counter()
    _recall_query_plan("记忆检索预热")
    logger.info(
        "Recall runtime warmed / 召回运行时已预热: %sms",
        max(0, int((time.perf_counter() - recall_warm_started_at) * 1000)),
    )

    if transport in ("sse", "streamable-http"):
        import threading
        import uvicorn
        from starlette.middleware.cors import CORSMiddleware

        # --- Application-level keepalive: ping /health every 60s ---
        # --- 应用层保活：每 60 秒 ping 一次 /health，防止 Cloudflare Tunnel 空闲断连 ---
        async def _keepalive_loop():
            await _ensure_decay_engine_started_for_transport(transport)
            await asyncio.sleep(10)  # Wait for server to fully start
            async with httpx.AsyncClient() as client:
                while True:
                    try:
                        await client.get("http://localhost:8000/health", timeout=5)
                        logger.debug("Keepalive ping OK / 保活 ping 成功")
                    except Exception as e:
                        logger.warning(f"Keepalive ping failed / 保活 ping 失败: {e}")
                    await asyncio.sleep(60)

        def _start_keepalive():
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_keepalive_loop())

        t = threading.Thread(target=_start_keepalive, daemon=True)
        t.start()

        async def _reflection_loop():
            await asyncio.sleep(20)
            local_bucket_mgr = BucketManager(config)
            local_embedding_engine = EmbeddingEngine(config)
            local_persona_engine = PersonaStateEngine(config)
            local_reflection_engine = ReflectionEngine(config)
            local_portrait_engine = DailyPortraitMaintainer(config)
            local_memory_edge_store = MemoryEdgeStore(config)
            local_gateway_state_store = GatewayStateStore(os.path.join(config["buckets_dir"], "gateway_state.db"))
            while True:
                try:
                    reflection_cfg = config.get("reflection", {}) if isinstance(config.get("reflection", {}), dict) else {}
                    local_reflection_engine.enabled = bool(
                        reflection_cfg.get("enabled", True)
                    )
                    local_reflection_engine.auto_enabled = bool(
                        reflection_cfg.get("auto_enabled", True)
                    )
                    if not local_reflection_engine.enabled or not local_reflection_engine.auto_enabled:
                        await asyncio.sleep(local_reflection_engine.check_interval_minutes * 60)
                        continue
                    local_reflection_engine.daily_enabled = bool(
                        reflection_cfg.get("daily_enabled", True)
                    )
                    local_reflection_engine.memory_affect_anchor_enabled = bool(
                        reflection_cfg.get("memory_affect_anchor_enabled", False)
                    )
                    local_reflection_engine.relationship_weather_affect_anchor_enabled = bool(
                        reflection_cfg.get("relationship_weather_affect_anchor_enabled", False)
                    )
                    local_reflection_engine.daily_min_memory_items = _int_between(
                        reflection_cfg.get("daily_min_memory_items"),
                        5,
                        0,
                        100,
                    )
                    local_reflection_engine.daily_conversation_turn_limit = _int_between(
                        reflection_cfg.get("daily_conversation_turn_limit"),
                        12,
                        0,
                        80,
                    )
                    local_reflection_engine.daily_activity_summary_enabled = bool(
                        reflection_cfg.get("daily_activity_summary_enabled", True)
                    )
                    local_reflection_engine.daily_activity_summary_turn_limit = _int_between(
                        reflection_cfg.get("daily_activity_summary_turn_limit"),
                        getattr(local_reflection_engine, "daily_activity_summary_turn_limit", 0),
                        0,
                        10000,
                    )
                    local_reflection_engine.daily_activity_summary_max_tokens = _int_between(
                        reflection_cfg.get("daily_activity_summary_max_tokens"),
                        getattr(local_reflection_engine, "daily_activity_summary_max_tokens", 320),
                        80,
                        1000,
                    )
                    mode = str(reflection_cfg.get("daily_chat_memory_mode") or "review").strip().lower()
                    local_reflection_engine.daily_chat_memory_mode = (
                        mode if mode in {"auto", "review", "off"} else "review"
                    )
                    local_reflection_engine.daily_chat_memory_hour = _int_between(
                        reflection_cfg.get("daily_chat_memory_hour"),
                        0,
                        0,
                        23,
                    )
                    local_reflection_engine.daily_chat_memory_turn_limit = _int_between(
                        reflection_cfg.get("daily_chat_memory_turn_limit"),
                        0,
                        0,
                        10000,
                    )
                    local_reflection_engine.daily_chat_memory_max_per_day = _int_between(
                        reflection_cfg.get("daily_chat_memory_max_per_day"),
                        3,
                        0,
                        10,
                    )
                    results = await local_reflection_engine.run_due(
                        local_bucket_mgr,
                        local_persona_engine,
                        local_embedding_engine,
                        local_gateway_state_store,
                        raw_event_store,
                    )
                    now_local = local_reflection_engine._local_now()
                    if (
                        getattr(local_reflection_engine, "daily_activity_summary_enabled", True)
                        and now_local.hour >= local_reflection_engine.daily_chat_memory_hour
                    ):
                        activity_date = (now_local - timedelta(days=1)).date().isoformat()
                        timeline_id = f"daily_activity_summary:{activity_date}"
                        if not local_portrait_engine.has_recent_timeline_item(
                            date_key=activity_date,
                            source="daily_activity_summary",
                            timeline_id=timeline_id,
                        ):
                            activity_candidates, activity_daily_impressions = (
                                _daily_activity_materials_from_reflection_results(results)
                            )
                            if not activity_daily_impressions:
                                existing_daily_impression = await _daily_impression_material_for_date(
                                    activity_date,
                                    local_bucket_mgr,
                                )
                                if existing_daily_impression:
                                    activity_daily_impressions.append(existing_daily_impression)
                            activity_result = await local_reflection_engine.run_daily_activity_summary(
                                conversation_turn_store=local_gateway_state_store,
                                raw_event_store=raw_event_store,
                                persona_engine=local_persona_engine,
                                daily_chat_memory_candidates=activity_candidates,
                                daily_impressions=activity_daily_impressions,
                                key=activity_date,
                            )
                            stored_activity = _store_daily_activity_summary_result(
                                activity_result,
                                portrait_engine_arg=local_portrait_engine,
                            )
                            if stored_activity.get("status") not in {"disabled", "skipped"}:
                                results.append(stored_activity)
                    if results:
                        logger.info("Reflection run-due results / 反思定时结果: %s", results)
                    if reflection_cfg.get("enrich_backfill_enabled", True):
                        backfill_result = await _backfill_memory_enrichment(
                            limit=reflection_cfg.get("enrich_backfill_limit", 5),
                            bucket_mgr_arg=local_bucket_mgr,
                            reflection_engine_arg=local_reflection_engine,
                            edge_store_arg=local_memory_edge_store,
                            embedding_engine_arg=local_embedding_engine,
                        )
                        if backfill_result.get("processed"):
                            logger.info(
                                "Memory enrichment backfill / 记忆 enrich 补跑: %s",
                                backfill_result,
                            )
                except Exception as e:
                    logger.warning("Reflection scheduler failed / 反思定时器失败: %s", e)
                await asyncio.sleep(local_reflection_engine.check_interval_minutes * 60)

        def _start_reflection_scheduler():
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_reflection_loop())

        if reflection_engine.enabled and reflection_engine.auto_enabled:
            rt = threading.Thread(target=_start_reflection_scheduler, daemon=True)
            rt.start()
            logger.info("Reflection scheduler enabled / 反思定时器已启用")

        async def _portrait_loop():
            await asyncio.sleep(25)
            local_bucket_mgr = BucketManager(config)
            local_persona_engine = PersonaStateEngine(config)
            local_portrait_engine = DailyPortraitMaintainer(config)
            while True:
                try:
                    results = await local_portrait_engine.run_due(
                        local_bucket_mgr,
                        local_persona_engine,
                    )
                    if results:
                        logger.info("Portrait run-due results / 画像定时结果: %s", results)
                except Exception as e:
                    logger.warning("Portrait scheduler failed / 画像定时器失败: %s", e)
                await asyncio.sleep(local_portrait_engine.check_interval_minutes * 60)

        def _start_portrait_scheduler():
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_portrait_loop())

        if portrait_engine.enabled and portrait_engine.auto_enabled:
            pt = threading.Thread(target=_start_portrait_scheduler, daemon=True)
            pt.start()
            logger.info("Portrait scheduler enabled / 画像定时器已启用")

        async def _word_map_daily_rebuild_loop():
            await asyncio.sleep(35)
            local_bucket_mgr = BucketManager(config)
            local_word_map_store = WordMapStore(config)
            initial_settings = _word_map_daily_rebuild_settings(config)
            initial_now = datetime.now()
            last_run_date = (
                initial_now.date().isoformat()
                if _word_map_should_run_daily_rebuild(initial_now, "", initial_settings)
                else ""
            )
            while True:
                settings = _word_map_daily_rebuild_settings(config)
                try:
                    now = datetime.now()
                    if _word_map_should_run_daily_rebuild(now, last_run_date, settings):
                        result = await _rebuild_word_map_index(
                            local_word_map_store,
                            local_bucket_mgr,
                            include_archive=bool(settings.get("include_archive")),
                        )
                        last_run_date = now.date().isoformat()
                        logger.info("Word Map daily rebuild result / 词图每日重建结果: %s", result)
                except Exception as e:
                    logger.warning("Word Map daily rebuild failed / 词图每日重建失败: %s", e, exc_info=True)
                await asyncio.sleep(int(settings.get("check_interval_seconds") or 900))

        def _start_word_map_daily_rebuild_scheduler():
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_word_map_daily_rebuild_loop())

        word_map_cfg = config.get("word_map", {}) if isinstance(config.get("word_map", {}), dict) else {}
        if _bool_value(word_map_cfg.get("daily_rebuild_enabled"), True):
            wt = threading.Thread(target=_start_word_map_daily_rebuild_scheduler, daemon=True)
            wt.start()
            logger.info("Word Map daily rebuild scheduler started / 词图每日重建定时器已启动")

        async def _dream_loop():
            await asyncio.sleep(30)
            local_bucket_mgr = BucketManager(config)
            while True:
                local_dream_engine = DreamEngine(config)
                local_embedding_engine = EmbeddingEngine(config)
                try:
                    result = await local_dream_engine.run_due(
                        local_bucket_mgr,
                        local_embedding_engine,
                        raw_event_store=raw_event_store,
                    )
                    if result and result.get("status") == "created":
                        logger.info("Dream run-due result / 夜梦定时结果: %s", result)
                except Exception as e:
                    logger.warning("Dream scheduler failed / 夜梦定时器失败: %s", e)
                await asyncio.sleep(local_dream_engine.check_interval_minutes * 60)

        def _start_dream_scheduler():
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_dream_loop())

        dt = threading.Thread(target=_start_dream_scheduler, daemon=True)
        dt.start()
        logger.info("Dream scheduler loop started / 夜梦定时器循环已启动")

        # --- Add CORS middleware so remote clients (Cloudflare Tunnel / ngrok) can connect ---
        # --- 添加 CORS 中间件，让远程客户端（Cloudflare Tunnel / ngrok）能正常连接 ---
        if transport == "streamable-http":
            _app = mcp.streamable_http_app()
        else:
            _app = mcp.sse_app()
        if hasattr(_app, "add_event_handler"):
            async def _start_decay_engine_on_app_startup():
                await _ensure_decay_engine_started_for_transport(transport)

            _app.add_event_handler("startup", _start_decay_engine_on_app_startup)
        _app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"],
        )
        _app.add_middleware(
            OmbreChatGptOAuthMiddleware,
            provider=OMBRE_CHATGPT_OAUTH,
            protected_hosts=OMBRE_CHATGPT_OAUTH_PROTECTED_HOSTS,
        )
        logger.info("CORS middleware enabled for remote transport / 已启用 CORS 中间件")
        if OMBRE_CHATGPT_OAUTH.enabled:
            logger.info(
                "ChatGPT OAuth enabled for Ombre MCP / 已启用 ChatGPT OAuth: protected_hosts=%s",
                sorted(OMBRE_CHATGPT_OAUTH_PROTECTED_HOSTS),
            )
        uvicorn.run(_app, host="0.0.0.0", port=8000)
    else:
        mcp.run(transport=transport)
