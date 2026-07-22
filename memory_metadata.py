"""Read-only normalized metadata view for memory buckets."""

from __future__ import annotations

import os
import re
from typing import Any

DOMAIN_LABELS = {
    "relationship": "关系",
    "intimacy": "亲密",
    "inner": "内心",
    "life": "生活",
    "tech": "技术",
    "project": "项目",
    "general": "通用",
}

DOMAIN_PARENT_LABELS = {
    "relationship": "关系",
    "intimacy": "亲密",
    "inner": "内心",
    "life": "生活",
    "tech": "技术",
    "project": "项目",
    "general": "通用",
}

CANONICAL_DOMAINS = set(DOMAIN_LABELS)

MEMORY_KINDS = {
    "event",
    "preference",
    "profile_fact",
    "reflection",
    "affect_anchor",
    "daily_impression",
    "source_record",
    "raw_import",
    "relationship_weather",
}

STATUS_VIEWS = {"active", "unresolved", "digested", "archived", "protected"}

LEGACY_DOMAIN_MAP = {
    "relationship.identity": "relationship",
    "relationship.symbol": "relationship",
    "relationship.communication": "relationship",
    "relationship.weather": "relationship",
    "relationship.intimacy": "intimacy",
    "project_code": "tech",
    "ai_tools": "tech",
    "hardware_protocol": "tech",
    "intimacy": "intimacy",
    "inner_state": "inner",
    "daily_life": "life",
    "life.health": "life",
    "life.sleep": "life",
    "life.food": "life",
    "life.outing": "life",
    "life.mood": "inner",
    "life.schedule": "life",
    "life.social": "life",
    "social": "life",
    "study_work": "project",
    "craft_body": "project",
    "project.companion_system": "project",
    "project.work": "project",
    "project.academic": "project",
    "project.personal": "project",
    "日常": "life",
    "生活": "life",
    "饮食": "life",
    "出行": "life",
    "健康": "life",
    "睡眠": "life",
    "事务": "life",
    "计划": "life",
    "待办": "life",
    "人际": "relationship",
    "关系": "relationship",
    "恋爱": "relationship",
    "亲密": "intimacy",
    "沟通": "relationship",
    "暗号": "relationship",
    "意象": "relationship",
    "内心": "inner",
    "情绪": "inner",
    "自省": "inner",
    "心理": "inner",
    "数字": "tech",
    "编程": "tech",
    "技术": "tech",
    "项目": "project",
    "工作": "project",
    "学业": "project",
    "学习": "project",
    "个人项目": "project",
    "未分类": "general",
    "通用": "general",
}

DOMAIN_PROMPT_CHOICES = [
    ("relationship", "关系身份、称呼、承诺、边界、沟通方式、暗号、意象、关系锚点"),
    ("intimacy", "亲密、身体、欲望、具身互动"),
    ("inner", "心情、情绪、内心活动、自省、心理状态、梦后感、非关系专属反思"),
    ("life", "健康、睡眠、饮食、出行、日程、现实人际、梦境事实"),
    ("tech", "代码、技术实现、模型、MCP、硬件、部署、数据库、调试"),
    ("project", "我们的项目、工作、学业、个人项目、创作、求职、论文"),
    ("general", "不确定或不属于前面几类"),
]

DOMAIN_ALIASES = {
    "relationship": {
        "relationship",
        "love",
        "romance",
        "partner",
        "恋爱",
        "关系",
        "爱",
        "陪伴",
        "relationship.identity",
        "relationship_identity",
        "identity",
        "human-ai relationship",
        "ai relationship",
        "人机恋",
        "人机关系",
        "关系身份",
        "关系定位",
        "称呼",
        "承诺",
        "边界",
        "身份",
        "relationship.symbol",
        "symbol",
        "private_signal",
        "signal",
        "暗号",
        "意象",
        "象征",
        "relationship.communication",
        "communication",
        "communication_preference",
        "tone",
        "repair",
        "语气",
        "沟通",
        "回应方式",
        "承接",
        "修复",
        "吵架",
        "情绪承接",
        "relationship.weather",
        "relationship_weather",
        "daily_impression",
        "weekly_impression",
        "关系天气",
        "日印象",
        "周印象",
    },
    "intimacy": {
        "intimacy",
        "relationship.intimacy",
        "body",
        "desire",
        "亲密",
        "亲密关系",
        "身体",
        "欲望",
        "具身",
        "色色",
    },
    "life": {
        "life",
        "daily_life",
        "daily",
        "生活",
        "日常",
        "life.health",
        "health",
        "身体状态",
        "健康",
        "生病",
        "病",
        "life.sleep",
        "sleep",
        "睡眠",
        "作息",
        "熬夜",
        "睡觉",
        "life.food",
        "food",
        "meal",
        "饮食",
        "吃饭",
        "午饭",
        "晚饭",
        "餐厅",
        "life.outing",
        "outing",
        "travel",
        "commute",
        "出行",
        "通勤",
        "地铁",
        "高铁",
        "旅行",
        "外出",
        "life.schedule",
        "schedule",
        "todo",
        "followup",
        "日程",
        "安排",
        "待办",
        "未完成",
        "deadline",
        "life.social",
        "social",
        "friend",
        "school_group",
        "人际",
        "社交",
        "朋友",
        "群聊",
        "事务",
    },
    "inner": {
        "inner",
        "inner_state",
        "life.mood",
        "mood",
        "emotion",
        "reflection",
        "self_reflection",
        "feel",
        "feeling",
        "affect",
        "mental",
        "psychology",
        "introspection",
        "心情",
        "情绪",
        "内心",
        "自省",
        "心理",
        "感受",
        "反思",
        "梦后感",
    },
    "project": {
        "project",
        "项目",
        "craft_body",
        "companion_system",
        "memory_system",
        "mist-room",
        "我们的项目",
        "陪伴系统",
        "记忆系统",
        "开发",
        "设备",
        "实体",
        "身体项目",
        "工具",
        "客户端",
        "平台",
        "project.work",
        "work",
        "resume",
        "job",
        "工作",
        "实习",
        "求职",
        "简历",
        "boss",
        "职场",
        "project.academic",
        "academic",
        "study",
        "paper",
        "school",
        "学业",
        "论文",
        "课程",
        "作业",
        "学校",
        "答辩",
        "project.personal",
        "personal_project",
        "个人项目",
        "创作",
        "阅读",
        "手工",
    },
    "tech": {
        "tech",
        "technical",
        "technology",
        "技术",
        "技术计划",
        "编程",
        "代码",
        "调试",
        "debug",
        "debugging",
        "project.companion_system",
        "project_code",
        "ai_tools",
        "hardware_protocol",
        "code",
        "coding",
        "programming",
        "dev",
        "repo",
        "repository",
        "仓库",
        "工程",
        "开发",
        "硬件",
        "协议",
        "数据库",
        "部署",
        "报错",
        "bug",
        "api",
        "sdk",
        "sql",
        "sqlite",
        "server",
        "gateway",
        "ombre",
        "ombre-brain",
        "haven_bridge",
        "bridge",
        "voice",
        "tts",
        "mcp",
        "recall",
        "codex",
        "模型",
        "embedding",
        "reranker",
        "deepseek",
        "qwen",
        "glm",
        "chatgpt",
        "ai",
        "数字",
        "语音",
    },
    "general": {
        "general",
        "other",
        "uncategorized",
        "未分类",
        "通用",
    },
}


def normalize_memory_metadata(bucket: dict[str, Any] | None) -> dict[str, Any]:
    """Return a normalized, read-only metadata view without mutating the bucket."""

    bucket = bucket if isinstance(bucket, dict) else {}
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    legacy_domain = _string_list(meta.get("domain"))
    tags = _string_list(meta.get("tags"))
    type_value = _clean(meta.get("type") or bucket.get("type"))
    path_value = _clean(bucket.get("path") or bucket.get("file_path"))
    name_value = _clean(meta.get("name") or bucket.get("name") or bucket.get("id"))
    text_blob = " ".join(
        item
        for item in [
            type_value,
            path_value,
            name_value,
            " ".join(legacy_domain),
            " ".join(tags),
            _clean(meta.get("memory_layer")),
            _clean(meta.get("profile_kind")),
        ]
        if item
    )

    flags = _flags(meta, type_value, path_value, name_value, legacy_domain, tags)
    kind = _normalize_kind(meta.get("kind")) or _infer_kind(meta, text_blob, flags)
    if kind == "profile_fact" and "profile_fact" not in flags:
        flags.append("profile_fact")
    if kind == "source_record" and "source_record" not in flags:
        flags.append("source_record")
    status_view = _normalize_status(meta.get("status") or meta.get("status_view")) or _infer_status(
        meta,
        type_value,
        path_value,
        legacy_domain,
        tags,
    )
    canonical_domain = (
        _normalize_domain(meta.get("canonical_domain"))
        or _infer_domain(legacy_domain, tags, type_value, path_value, kind)
        or "general"
    )

    parent = domain_parent(canonical_domain)
    return {
        "canonical_domain": canonical_domain,
        "domain_parent": parent,
        "domain_label": domain_label(canonical_domain),
        "domain_parent_label": DOMAIN_PARENT_LABELS.get(parent, parent),
        "kind": kind,
        "status_view": status_view,
        "flags": flags,
        "legacy_domain": legacy_domain,
    }


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw = value
    else:
        raw = [value]
    return [text for text in (_clean(item) for item in raw) if text]


def _compact(value: Any) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff_.:-]+", "", _clean(value).lower())


def _normalize_domain(value: Any) -> str:
    compact = _compact(value)
    if compact in LEGACY_DOMAIN_MAP:
        return LEGACY_DOMAIN_MAP[compact]
    if compact in CANONICAL_DOMAINS:
        return compact
    for domain, aliases in DOMAIN_ALIASES.items():
        if compact in {_compact(alias) for alias in aliases}:
            return domain
    return ""


def normalize_domain_key(value: Any) -> str:
    return _normalize_domain(value)


def domain_parent(domain: str) -> str:
    value = _clean(domain)
    if not value or value == "general":
        return "general"
    return value.split(".", 1)[0]


def domain_label(domain: str) -> str:
    value = _clean(domain)
    return DOMAIN_LABELS.get(value, value or DOMAIN_LABELS["general"])


def domain_prompt_options_text() -> str:
    return "\n".join(f"- {key}：{description}" for key, description in DOMAIN_PROMPT_CHOICES)


def domain_options() -> list[dict[str, str]]:
    return [
        {
            "key": key,
            "label": label,
            "parent": domain_parent(key),
            "parent_label": DOMAIN_PARENT_LABELS.get(domain_parent(key), domain_parent(key)),
        }
        for key, label in DOMAIN_LABELS.items()
    ]


def _normalize_kind(value: Any) -> str:
    compact = _compact(value)
    return compact if compact in MEMORY_KINDS else ""


def _normalize_status(value: Any) -> str:
    compact = _compact(value)
    return compact if compact in STATUS_VIEWS else ""


def _infer_domain(
    legacy_domain: list[str],
    tags: list[str],
    type_value: str,
    path_value: str,
    kind: str,
) -> str:
    candidates = legacy_domain + tags + [type_value, path_value]
    for item in candidates:
        domain = _normalize_domain(item)
        if domain:
            return domain
    if kind in {"relationship_weather"}:
        return "relationship"
    if kind in {"profile_fact", "daily_impression", "reflection", "affect_anchor"}:
        return "life"
    return ""


def _infer_kind(meta: dict[str, Any], text_blob: str, flags: list[str]) -> str:
    compact = _compact(text_blob)
    memory_layer = _compact(meta.get("memory_layer"))
    profile_kind = _compact(meta.get("profile_kind"))
    if "source_record" in compact or "sourcerecord" in compact or "source_record" in flags:
        return "source_record"
    if "relationship_weather" in compact or "relationshipweather" in compact:
        return "relationship_weather"
    if "profile_fact" in compact or "profilefact" in compact or profile_kind:
        return "profile_fact"
    if "daily_impression" in compact or "dailyimpression" in compact or "日印象" in text_blob:
        return "daily_impression"
    if "affect_anchor" in compact or "affectanchor" in compact:
        return "affect_anchor"
    if "reflection" in compact or memory_layer == "reflection":
        return "reflection"
    if "preference" in compact or "偏好" in text_blob:
        return "preference"
    if "raw_import" in compact or "rawimport" in compact:
        return "raw_import"
    return "event"


def _infer_status(
    meta: dict[str, Any],
    type_value: str,
    path_value: str,
    legacy_domain: list[str],
    tags: list[str],
) -> str:
    blob = " ".join([type_value, path_value, " ".join(legacy_domain), " ".join(tags)])
    compact = _compact(blob)
    path_parts = {part.lower() for part in re.split(r"[\\/]+", path_value) if part}
    if type_value == "archived" or "archive" in path_parts or "archived" in path_parts or "归档" in blob:
        return "archived"
    if _truthy(meta.get("protected")) or _truthy(meta.get("pinned")):
        return "protected"
    if _truthy(meta.get("digested")) or "digested" in compact or "已消化" in blob:
        return "digested"
    if meta.get("resolved") is False or "unresolved" in compact or "未解决" in blob:
        return "unresolved"
    return "active"


def _flags(
    meta: dict[str, Any],
    type_value: str,
    path_value: str,
    name_value: str,
    legacy_domain: list[str],
    tags: list[str],
) -> list[str]:
    blob = " ".join([type_value, path_value, " ".join(legacy_domain), " ".join(tags)])
    compact = _compact(blob)
    flags: list[str] = []

    def add(flag: str, condition: bool) -> None:
        if condition and flag not in flags:
            flags.append(flag)

    add("pinned", _truthy(meta.get("pinned")))
    add("protected", _truthy(meta.get("protected")))
    add("anchor", _truthy(meta.get("anchor")) or _has_exact_marker(legacy_domain, tags, {"anchor", "锚点"}))
    add(
        "self_anchor",
        _truthy(meta.get("self_anchor"))
        or _has_exact_marker(
            legacy_domain,
            tags,
            {"self_anchor", "first_person_anchor", "firstpersonanchor", "自我"},
        )
        or _has_exact_marker(
            [],
            [
                _clean(meta.get("profile_kind")),
                _clean(meta.get("bucket_profile_kind")),
                _clean(meta.get("anchor_kind")),
                _clean(meta.get("kind")),
                _clean(meta.get("source")),
            ],
            {"self_anchor", "first_person_anchor", "firstpersonanchor", "自我"},
        ),
    )
    add("favorite", "favorite" in compact or "最爱" in blob)
    add("source_record", "source_record" in compact or "sourcerecord" in compact)
    add("profile_fact", "profile_fact" in compact or "profilefact" in compact or bool(_clean(meta.get("profile_kind"))))
    add("archived", _infer_status(meta, type_value, path_value, legacy_domain, tags) == "archived")
    return flags


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _has_exact_marker(legacy_domain: list[str], tags: list[str], markers: set[str]) -> bool:
    compact_markers = {_compact(marker) for marker in markers}
    return any(_compact(item) in compact_markers for item in [*legacy_domain, *tags])
