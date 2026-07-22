from __future__ import annotations

import os
from typing import Any


RETIRED_WRITE_KINDS = frozenset(
    {
        "feel",
        "whisper",
        "daily_impression",
        "weekly_impression",
        "relationship_weather",
        "profile_fact",
    }
)


def legacy_memory_writes_enabled(config: dict[str, Any] | None = None) -> bool:
    """Return the reversible compatibility switch for retired object writes."""

    env_value = str(os.environ.get("OMBRE_LEGACY_MEMORY_WRITES") or "").strip().lower()
    if env_value:
        return env_value in {"1", "true", "yes", "on", "enabled"}
    config = config or {}
    object_config = config.get("memory_objects", {})
    if not isinstance(object_config, dict):
        object_config = {}
    return bool(object_config.get("legacy_writes_enabled", False))


def retired_write_payload(kind: str) -> dict[str, Any]:
    safe_kind = str(kind or "legacy_memory").strip().lower() or "legacy_memory"
    return {
        "status": "retired",
        "reason": "legacy_memory_write_disabled",
        "object_type": safe_kind,
        "legacy_data_preserved": True,
        "read_compatibility": True,
        "rollback_env": "OMBRE_LEGACY_MEMORY_WRITES=1",
    }


def memory_object_write_status(config: dict[str, Any] | None = None) -> dict[str, Any]:
    enabled = legacy_memory_writes_enabled(config)
    return {
        "legacy_writes_enabled": enabled,
        "mode": "compatibility" if enabled else "retired",
        "retired_object_types": sorted(RETIRED_WRITE_KINDS),
        "legacy_data_preserved": True,
        "read_compatibility": True,
        "rollback_env": "OMBRE_LEGACY_MEMORY_WRITES=1",
    }


def retired_write_message(kind: str) -> str:
    safe_kind = str(kind or "legacy memory").strip()
    return (
        f"写入被拒绝：{safe_kind} 已停止新增；旧数据仍可只读。"
        "有来源的新理解请用 annotate，具体长期时刻请用 write_scene，"
        "本窗心绪请写进 Window Shadow，长篇私密正文交给 Diary/Darkroom。"
        "兼容回退可显式设置 OMBRE_LEGACY_MEMORY_WRITES=1。"
    )
