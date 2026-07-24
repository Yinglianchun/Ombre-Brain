from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from favorite_tags import GENERIC_FAVORITE_TAG, has_favorite_memory_tag

LAYER_CORE = "core_memory"
LAYER_ANCHOR = "long_term_anchor"
LAYER_DYNAMIC = "dynamic_memory"
LAYER_RELATIONSHIP_WEATHER = "relationship_weather"
LAYER_AFFECT_CONTEXT = "affect_context"
LAYER_FAVORITE = "favorite_memory"
LAYER_DREAM = "dream"
LAYER_SOURCE_RECORD = "source_record"
LAYER_PROFILE_INDEX = "profile_index"
LAYER_ARCHIVE = "archive"

WRITE_SUBJECT_USER = "user"
WRITE_SUBJECT_RELATIONSHIP = "relationship"
WRITE_SUBJECT_EVENT = "event"

WRITE_LAYER_STABLE_BOUNDARY = "stable_boundary"
WRITE_LAYER_SHORT_STATE = "short_state"
WRITE_LAYER_PROCESS_EVENT = "process_event"
WRITE_LAYER_RELATIONSHIP_LESSON = "relationship_lesson"

WRITE_SUBJECTS = frozenset(
    {
        WRITE_SUBJECT_USER,
        WRITE_SUBJECT_RELATIONSHIP,
        WRITE_SUBJECT_EVENT,
    }
)
WRITE_LAYERS = frozenset(
    {
        WRITE_LAYER_STABLE_BOUNDARY,
        WRITE_LAYER_SHORT_STATE,
        WRITE_LAYER_PROCESS_EVENT,
        WRITE_LAYER_RELATIONSHIP_LESSON,
    }
)

DIRECT_CONTENT = "content_only"
DIRECT_EXPLICIT = "explicit_only"
DIRECT_EXPLICIT_OR_CONTENT = "explicit_or_content"
DIRECT_RESONANCE = "resonance_only"
DIRECT_NEVER = "never"

RENDER_DIRECT_AUTO = "direct_auto"
RENDER_SUMMARY = "summary"
RENDER_STABLE = "stable_rule_or_original"
RENDER_WEATHER = "weather"
RENDER_AUXILIARY = "auxiliary_context"
RENDER_FAVORITE = "favorite_card"
RENDER_DREAM_ORIGINAL = "dream_original"
RENDER_SOURCE_ONLY = "source_only"

DIFFUSE_SOURCE = "source"
DIFFUSE_CAREFUL_SOURCE = "careful_source"
DIFFUSE_CHAIN_ONLY = "chain_only"
DIFFUSE_NEVER = "never"

CONTEXT_ONLY_SECTIONS = frozenset(
    {
        "comment",
        "reflection",
        "feeling",
        "affect_anchor",
        "favorite_reason",
        "followup",
        "followup_log",
    }
)
RELATIONSHIP_WEATHER_TAGS = frozenset(
    {"relationship_weather", "daily_impression", "weekly_impression"}
)
RAW_SOURCE_TAGS = frozenset({"raw_source", "chat_log", "diary_source", "source_record"})
FAVORITE_TAG = GENERIC_FAVORITE_TAG

SUBJECT_ALIASES = {
    "user": WRITE_SUBJECT_USER,
    "xiaoyu": WRITE_SUBJECT_USER,
    "rain": WRITE_SUBJECT_USER,
    "person": WRITE_SUBJECT_USER,
    "profile": WRITE_SUBJECT_USER,
    "state": WRITE_SUBJECT_USER,
    "preference": WRITE_SUBJECT_USER,
    "boundary": WRITE_SUBJECT_USER,
    "relationship": WRITE_SUBJECT_RELATIONSHIP,
    "relation": WRITE_SUBJECT_RELATIONSHIP,
    "ai": WRITE_SUBJECT_RELATIONSHIP,
    "assistant": WRITE_SUBJECT_RELATIONSHIP,
    "haven": WRITE_SUBJECT_RELATIONSHIP,
    "event": WRITE_SUBJECT_EVENT,
    "process": WRITE_SUBJECT_EVENT,
    "project": WRITE_SUBJECT_EVENT,
    "task": WRITE_SUBJECT_EVENT,
}
LAYER_ALIASES = {
    "stable_boundary": WRITE_LAYER_STABLE_BOUNDARY,
    "stable": WRITE_LAYER_STABLE_BOUNDARY,
    "boundary": WRITE_LAYER_STABLE_BOUNDARY,
    "preference": WRITE_LAYER_STABLE_BOUNDARY,
    "habit": WRITE_LAYER_STABLE_BOUNDARY,
    "identity": WRITE_LAYER_STABLE_BOUNDARY,
    "core_boundary": WRITE_LAYER_STABLE_BOUNDARY,
    "short_state": WRITE_LAYER_SHORT_STATE,
    "state": WRITE_LAYER_SHORT_STATE,
    "current_state": WRITE_LAYER_SHORT_STATE,
    "temporary_state": WRITE_LAYER_SHORT_STATE,
    "short-term_state": WRITE_LAYER_SHORT_STATE,
    "process_event": WRITE_LAYER_PROCESS_EVENT,
    "event": WRITE_LAYER_PROCESS_EVENT,
    "project_state": WRITE_LAYER_PROCESS_EVENT,
    "active_project": WRITE_LAYER_PROCESS_EVENT,
    "task_state": WRITE_LAYER_PROCESS_EVENT,
    "relationship_lesson": WRITE_LAYER_RELATIONSHIP_LESSON,
    "relationship": WRITE_LAYER_RELATIONSHIP_LESSON,
    "response_rule": WRITE_LAYER_RELATIONSHIP_LESSON,
    "promise": WRITE_LAYER_RELATIONSHIP_LESSON,
    "agreement": WRITE_LAYER_RELATIONSHIP_LESSON,
    "commitment": WRITE_LAYER_RELATIONSHIP_LESSON,
}

WRITER_RUNTIME_LAYERS = {
    WRITE_LAYER_STABLE_BOUNDARY: LAYER_ANCHOR,
    WRITE_LAYER_RELATIONSHIP_LESSON: LAYER_ANCHOR,
    WRITE_LAYER_SHORT_STATE: LAYER_DYNAMIC,
    WRITE_LAYER_PROCESS_EVENT: LAYER_DYNAMIC,
}


@dataclass(frozen=True)
class MemoryLayerPolicy:
    layer: str
    direct_seed_policy: str
    render_policy: str
    gateway_section: str
    cooldown_policy: str
    diffusion_policy: str
    preserves_original: bool

    @property
    def can_direct_seed(self) -> bool:
        return self.direct_seed_policy != DIRECT_NEVER

    @property
    def can_diffuse(self) -> bool:
        return self.diffusion_policy != DIFFUSE_NEVER


LAYER_POLICIES: dict[str, MemoryLayerPolicy] = {
    LAYER_CORE: MemoryLayerPolicy(
        layer=LAYER_CORE,
        direct_seed_policy=DIRECT_EXPLICIT_OR_CONTENT,
        render_policy=RENDER_STABLE,
        gateway_section="Core Memory",
        cooldown_policy="rare",
        diffusion_policy=DIFFUSE_CAREFUL_SOURCE,
        preserves_original=True,
    ),
    LAYER_ANCHOR: MemoryLayerPolicy(
        layer=LAYER_ANCHOR,
        direct_seed_policy=DIRECT_CONTENT,
        render_policy=RENDER_DIRECT_AUTO,
        gateway_section="Recalled Memory",
        cooldown_policy="normal",
        diffusion_policy=DIFFUSE_SOURCE,
        preserves_original=True,
    ),
    LAYER_DYNAMIC: MemoryLayerPolicy(
        layer=LAYER_DYNAMIC,
        direct_seed_policy=DIRECT_CONTENT,
        render_policy=RENDER_DIRECT_AUTO,
        gateway_section="Recalled Memory",
        cooldown_policy="normal",
        diffusion_policy=DIFFUSE_SOURCE,
        preserves_original=True,
    ),
    LAYER_RELATIONSHIP_WEATHER: MemoryLayerPolicy(
        layer=LAYER_RELATIONSHIP_WEATHER,
        direct_seed_policy=DIRECT_NEVER,
        render_policy=RENDER_WEATHER,
        gateway_section="Relationship Weather",
        cooldown_policy="interval_or_config",
        diffusion_policy=DIFFUSE_NEVER,
        preserves_original=True,
    ),
    LAYER_AFFECT_CONTEXT: MemoryLayerPolicy(
        layer=LAYER_AFFECT_CONTEXT,
        direct_seed_policy=DIRECT_NEVER,
        render_policy=RENDER_AUXILIARY,
        gateway_section="attached_to_reliable_memory",
        cooldown_policy="parent",
        diffusion_policy=DIFFUSE_NEVER,
        preserves_original=True,
    ),
    LAYER_FAVORITE: MemoryLayerPolicy(
        layer=LAYER_FAVORITE,
        direct_seed_policy=DIRECT_CONTENT,
        render_policy=RENDER_FAVORITE,
        gateway_section="Favorite Memory",
        cooldown_policy="separate_budget",
        diffusion_policy=DIFFUSE_CAREFUL_SOURCE,
        preserves_original=True,
    ),
    LAYER_DREAM: MemoryLayerPolicy(
        layer=LAYER_DREAM,
        direct_seed_policy=DIRECT_RESONANCE,
        render_policy=RENDER_DREAM_ORIGINAL,
        gateway_section="Dream",
        cooldown_policy="dream_surface_rules",
        diffusion_policy=DIFFUSE_CHAIN_ONLY,
        preserves_original=True,
    ),
    LAYER_SOURCE_RECORD: MemoryLayerPolicy(
        layer=LAYER_SOURCE_RECORD,
        direct_seed_policy=DIRECT_NEVER,
        render_policy=RENDER_SOURCE_ONLY,
        gateway_section="none",
        cooldown_policy="not_injected",
        diffusion_policy=DIFFUSE_NEVER,
        preserves_original=True,
    ),
    LAYER_PROFILE_INDEX: MemoryLayerPolicy(
        layer=LAYER_PROFILE_INDEX,
        direct_seed_policy=DIRECT_NEVER,
        render_policy=RENDER_SOURCE_ONLY,
        gateway_section="none",
        cooldown_policy="not_recalled",
        diffusion_policy=DIFFUSE_NEVER,
        preserves_original=True,
    ),
    LAYER_ARCHIVE: MemoryLayerPolicy(
        layer=LAYER_ARCHIVE,
        direct_seed_policy=DIRECT_EXPLICIT,
        render_policy=RENDER_SUMMARY,
        gateway_section="explicit_lookup_only",
        cooldown_policy="sinks",
        diffusion_policy=DIFFUSE_NEVER,
        preserves_original=True,
    ),
}


def policy_for_layer(layer: str) -> MemoryLayerPolicy:
    return LAYER_POLICIES.get(str(layer or ""), LAYER_POLICIES[LAYER_DYNAMIC])


def runtime_layer_from_write_classification(memory_layer: object, memory_subject: object = "") -> str:
    layer = normalize_write_layer(memory_layer)
    if layer == WRITE_LAYER_RELATIONSHIP_LESSON:
        return LAYER_ANCHOR
    if layer == WRITE_LAYER_STABLE_BOUNDARY:
        return LAYER_ANCHOR
    if layer in WRITER_RUNTIME_LAYERS:
        return WRITER_RUNTIME_LAYERS[layer]
    subject = normalize_write_subject(memory_subject)
    if subject == WRITE_SUBJECT_RELATIONSHIP and layer:
        return LAYER_ANCHOR
    return ""


def infer_bucket_layer(bucket: dict[str, Any] | None) -> str:
    bucket = bucket if isinstance(bucket, dict) else {}
    meta = _metadata(bucket)
    tags = _tags(meta)
    bucket_type = _lower(meta.get("type") or meta.get("bucket_type"))

    if _truthy(meta.get("archived")) or _truthy(meta.get("digested")) or _truthy(meta.get("resolved")):
        return LAYER_ARCHIVE
    if bucket_type == "archived":
        return LAYER_ARCHIVE
    if bucket_type in {"source", "raw", "chat_log", "diary_source"} or tags & RAW_SOURCE_TAGS:
        return LAYER_SOURCE_RECORD
    if "profile_fact" in tags or bool(meta.get("profile_kind")):
        return LAYER_PROFILE_INDEX
    if bucket_type == "dream" or "dream" in tags or "night_dream" in tags:
        return LAYER_DREAM
    if bucket_type == "feel" and tags & RELATIONSHIP_WEATHER_TAGS:
        return LAYER_RELATIONSHIP_WEATHER
    if _truthy(meta.get("pinned")) or _truthy(meta.get("protected")) or bucket_type == "permanent":
        return LAYER_CORE
    if _truthy(meta.get("anchor")) or _truthy(meta.get("bucket_anchor")):
        return LAYER_ANCHOR
    if _has_favorite_tag(tags):
        return LAYER_FAVORITE
    if bucket_type == "feel":
        return LAYER_AFFECT_CONTEXT
    writer_layer = runtime_layer_from_write_classification(
        meta.get("memory_layer") or meta.get("bucket_memory_layer"),
        meta.get("memory_subject") or meta.get("bucket_memory_subject"),
    )
    if writer_layer:
        return writer_layer
    return LAYER_DYNAMIC


def policy_for_bucket(bucket: dict[str, Any] | None) -> MemoryLayerPolicy:
    return policy_for_layer(infer_bucket_layer(bucket))


def infer_moment_layer(moment: dict[str, Any] | None) -> str:
    moment = moment if isinstance(moment, dict) else {}
    section = _lower(moment.get("section"))
    if section in CONTEXT_ONLY_SECTIONS:
        return LAYER_AFFECT_CONTEXT
    return infer_bucket_layer({"metadata": _moment_metadata(moment), "id": moment.get("bucket_id")})


def policy_for_moment(moment: dict[str, Any] | None) -> MemoryLayerPolicy:
    return policy_for_layer(infer_moment_layer(moment))


def _parent_policy_for_moment(moment: dict[str, Any] | None) -> MemoryLayerPolicy:
    moment = moment if isinstance(moment, dict) else {}
    return policy_for_layer(
        infer_bucket_layer({"metadata": _moment_metadata(moment), "id": moment.get("bucket_id")})
    )


def can_moment_be_direct_seed(moment: dict[str, Any] | None, *, explicit_lookup: bool = False) -> bool:
    policy = policy_for_moment(moment)
    if policy.direct_seed_policy in {DIRECT_NEVER, DIRECT_RESONANCE}:
        return False
    if policy.direct_seed_policy == DIRECT_EXPLICIT:
        return bool(explicit_lookup)
    return True


def can_bucket_diffuse(bucket: dict[str, Any] | None) -> bool:
    return policy_for_bucket(bucket).can_diffuse


def can_bucket_be_related_target(bucket: dict[str, Any] | None, *, explicit_lookup: bool = False) -> bool:
    policy = policy_for_bucket(bucket)
    if policy.layer == LAYER_ARCHIVE:
        return bool(explicit_lookup)
    if policy.layer in {
        LAYER_DREAM,
        LAYER_SOURCE_RECORD,
        LAYER_PROFILE_INDEX,
        LAYER_RELATIONSHIP_WEATHER,
        LAYER_AFFECT_CONTEXT,
    }:
        return False
    return policy.can_diffuse


def can_moment_be_recall_context(moment: dict[str, Any] | None) -> bool:
    policy = _parent_policy_for_moment(moment)
    return policy.layer in {LAYER_CORE, LAYER_ANCHOR, LAYER_DYNAMIC, LAYER_FAVORITE, LAYER_ARCHIVE}


def can_moment_be_related_target(moment: dict[str, Any] | None, *, explicit_lookup: bool = False) -> bool:
    if not can_moment_be_recall_context(moment):
        return False
    if is_context_only_section((moment or {}).get("section") if isinstance(moment, dict) else ""):
        return False
    policy = _parent_policy_for_moment(moment)
    if policy.layer == LAYER_ARCHIVE:
        return bool(explicit_lookup)
    if policy.layer in {
        LAYER_DREAM,
        LAYER_SOURCE_RECORD,
        LAYER_PROFILE_INDEX,
        LAYER_RELATIONSHIP_WEATHER,
        LAYER_AFFECT_CONTEXT,
    }:
        return False
    return policy.can_diffuse


def can_bucket_be_recent_context(bucket: dict[str, Any] | None, *, explicit_lookup: bool = False) -> bool:
    layer = infer_bucket_layer(bucket)
    if explicit_lookup:
        return layer in {LAYER_ANCHOR, LAYER_DYNAMIC, LAYER_FAVORITE}
    return layer == LAYER_DYNAMIC


def _gate_payload(allowed: bool, reason: str) -> dict[str, Any]:
    return {"allowed": bool(allowed), "reason": reason}


def bucket_diffusion_source_gate(bucket: dict[str, Any] | None) -> dict[str, Any]:
    layer = infer_bucket_layer(bucket)
    policy = policy_for_layer(layer)
    if policy.can_diffuse:
        return _gate_payload(True, "allowed")
    return _gate_payload(False, "diffusion_policy_never")


def bucket_related_target_gate(
    bucket: dict[str, Any] | None,
    *,
    explicit_lookup: bool = False,
) -> dict[str, Any]:
    layer = infer_bucket_layer(bucket)
    policy = policy_for_layer(layer)
    if layer == LAYER_ARCHIVE:
        if explicit_lookup:
            return _gate_payload(True, "archive_explicit_lookup_allowed")
        return _gate_payload(False, "archive_requires_explicit_lookup")
    if layer in {
        LAYER_DREAM,
        LAYER_SOURCE_RECORD,
        LAYER_PROFILE_INDEX,
        LAYER_RELATIONSHIP_WEATHER,
        LAYER_AFFECT_CONTEXT,
    }:
        return _gate_payload(False, f"{layer}_not_related_target")
    if not policy.can_diffuse:
        return _gate_payload(False, "diffusion_policy_never")
    return _gate_payload(True, "allowed")


def bucket_recent_context_gate(
    bucket: dict[str, Any] | None,
    *,
    explicit_lookup: bool = False,
) -> dict[str, Any]:
    layer = infer_bucket_layer(bucket)
    if explicit_lookup:
        if layer in {LAYER_ANCHOR, LAYER_DYNAMIC, LAYER_FAVORITE}:
            return _gate_payload(True, "explicit_recent_allowed")
        return _gate_payload(False, "explicit_recent_layer_blocked")
    if layer == LAYER_DYNAMIC:
        return _gate_payload(True, "automatic_recent_dynamic_allowed")
    return _gate_payload(False, "automatic_recent_dynamic_only")


def moment_direct_seed_gate(
    moment: dict[str, Any] | None,
    *,
    explicit_lookup: bool = False,
) -> dict[str, Any]:
    moment = moment if isinstance(moment, dict) else {}
    layer = infer_moment_layer(moment)
    policy = policy_for_layer(layer)
    if is_context_only_section(moment.get("section")):
        return _gate_payload(False, "context_only_section")
    if policy.direct_seed_policy == DIRECT_NEVER:
        return _gate_payload(False, "direct_seed_policy_never")
    if policy.direct_seed_policy == DIRECT_RESONANCE:
        return _gate_payload(False, "resonance_only_not_normal_direct")
    if policy.direct_seed_policy == DIRECT_EXPLICIT and not explicit_lookup:
        return _gate_payload(False, "explicit_lookup_required")
    if policy.direct_seed_policy == DIRECT_EXPLICIT:
        return _gate_payload(True, "explicit_lookup_allowed")
    return _gate_payload(True, "allowed")


def moment_related_target_gate(
    moment: dict[str, Any] | None,
    *,
    explicit_lookup: bool = False,
) -> dict[str, Any]:
    moment = moment if isinstance(moment, dict) else {}
    if not can_moment_be_recall_context(moment):
        return _gate_payload(False, "parent_layer_not_recall_context")
    if is_context_only_section(moment.get("section")):
        return _gate_payload(False, "context_only_section")
    policy = _parent_policy_for_moment(moment)
    if policy.layer == LAYER_ARCHIVE:
        if explicit_lookup:
            return _gate_payload(True, "archive_explicit_lookup_allowed")
        return _gate_payload(False, "archive_requires_explicit_lookup")
    if policy.layer in {
        LAYER_DREAM,
        LAYER_SOURCE_RECORD,
        LAYER_PROFILE_INDEX,
        LAYER_RELATIONSHIP_WEATHER,
        LAYER_AFFECT_CONTEXT,
    }:
        return _gate_payload(False, f"{policy.layer}_not_related_target")
    if not policy.can_diffuse:
        return _gate_payload(False, "diffusion_policy_never")
    return _gate_payload(True, "allowed")


def bucket_runtime_gate_debug(
    bucket: dict[str, Any] | None,
    *,
    explicit_lookup: bool = False,
) -> dict[str, Any]:
    source = bucket_diffusion_source_gate(bucket)
    related = bucket_related_target_gate(bucket, explicit_lookup=explicit_lookup)
    recent = bucket_recent_context_gate(bucket, explicit_lookup=explicit_lookup)
    return {
        "layer": infer_bucket_layer(bucket),
        "diffusion_source": source,
        "related_target": related,
        "recent_context": recent,
        "would_diffuse_from": source["allowed"],
        "would_inject_related": related["allowed"],
        "would_inject_recent_context": recent["allowed"],
    }


def moment_runtime_gate_debug(
    moment: dict[str, Any] | None,
    *,
    explicit_lookup: bool = False,
) -> dict[str, Any]:
    direct = moment_direct_seed_gate(moment, explicit_lookup=explicit_lookup)
    related = moment_related_target_gate(moment, explicit_lookup=explicit_lookup)
    return {
        "layer": infer_moment_layer(moment),
        "parent_layer": _parent_policy_for_moment(moment).layer,
        "section": str((moment or {}).get("section") or "") if isinstance(moment, dict) else "",
        "direct_seed": direct,
        "recall_context": _gate_payload(
            can_moment_be_recall_context(moment),
            "allowed" if can_moment_be_recall_context(moment) else "parent_layer_not_recall_context",
        ),
        "related_target": related,
        "would_inject_direct": direct["allowed"],
        "would_inject_related": related["allowed"],
    }


def bucket_layer_debug(bucket: dict[str, Any] | None, *, explicit_lookup: bool = False) -> dict[str, Any]:
    bucket = bucket if isinstance(bucket, dict) else {}
    meta = _metadata(bucket)
    layer = infer_bucket_layer(bucket)
    policy = policy_for_layer(layer)
    return {
        **_policy_debug(policy),
        "can_related_target": can_bucket_be_related_target(bucket, explicit_lookup=explicit_lookup),
        "can_recent_context": can_bucket_be_recent_context(bucket, explicit_lookup=explicit_lookup),
        "writer": _writer_debug(
            meta.get("memory_subject") or meta.get("bucket_memory_subject"),
            meta.get("memory_layer") or meta.get("bucket_memory_layer"),
            meta.get("memory_classification_source") or meta.get("bucket_memory_classification_source"),
        ),
    }


def moment_layer_debug(moment: dict[str, Any] | None, *, explicit_lookup: bool = False) -> dict[str, Any]:
    moment = moment if isinstance(moment, dict) else {}
    meta = _moment_metadata(moment)
    layer = infer_moment_layer(moment)
    policy = policy_for_layer(layer)
    parent_policy = _parent_policy_for_moment(moment)
    return {
        **_policy_debug(policy),
        "parent_layer": parent_policy.layer,
        "section": str(moment.get("section") or ""),
        "context_only": is_context_only_section(moment.get("section")),
        "can_direct_seed": can_moment_be_direct_seed(moment, explicit_lookup=explicit_lookup),
        "can_recall_context": can_moment_be_recall_context(moment),
        "can_related_target": can_moment_be_related_target(moment, explicit_lookup=explicit_lookup),
        "writer": _writer_debug(
            meta.get("memory_subject") or meta.get("bucket_memory_subject"),
            meta.get("memory_layer") or meta.get("bucket_memory_layer"),
            meta.get("memory_classification_source") or meta.get("bucket_memory_classification_source"),
        ),
    }


def _policy_debug(policy: MemoryLayerPolicy) -> dict[str, Any]:
    return {
        "layer": policy.layer,
        "direct_seed_policy": policy.direct_seed_policy,
        "render_policy": policy.render_policy,
        "gateway_section": policy.gateway_section,
        "cooldown_policy": policy.cooldown_policy,
        "diffusion_policy": policy.diffusion_policy,
        "can_diffuse": policy.can_diffuse,
        "preserves_original": policy.preserves_original,
    }


def _writer_debug(subject: object, layer: object, source: object = "") -> dict[str, str]:
    subject_text = normalize_write_subject(subject)
    layer_text = normalize_write_layer(layer)
    return {
        "memory_subject": subject_text,
        "memory_layer": layer_text,
        "memory_classification_source": str(source or ""),
        "runtime_layer_hint": runtime_layer_from_write_classification(layer_text, subject_text),
    }


def is_context_only_section(section: object) -> bool:
    return _lower(section) in CONTEXT_ONLY_SECTIONS


def normalize_write_subject(value: object) -> str:
    text = _write_key(value)
    return SUBJECT_ALIASES.get(text, text if text in WRITE_SUBJECTS else "")


def normalize_write_layer(value: object) -> str:
    text = _write_key(value)
    return LAYER_ALIASES.get(text, text if text in WRITE_LAYERS else "")


def normalize_write_classification(
    *,
    memory_subject: object = "",
    memory_layer: object = "",
    tags: object = None,
    content: str = "",
) -> dict[str, str]:
    """Return conservative writer-side classification metadata."""
    tag_set = _tags({"tags": tags or []})
    subject = normalize_write_subject(memory_subject)
    layer = normalize_write_layer(memory_layer)
    source = "model" if subject and layer else "rule"

    inferred_subject, inferred_layer = _infer_write_classification(tag_set, content)
    hard_subject, hard_layer = _hard_write_classification(tag_set)
    if hard_subject and hard_layer and (subject or layer):
        if subject != hard_subject or layer != hard_layer:
            source = "model_adjusted"
        subject, layer = hard_subject, hard_layer
    if not subject:
        subject = inferred_subject
    if not layer:
        layer = inferred_layer

    if layer == WRITE_LAYER_RELATIONSHIP_LESSON:
        subject = WRITE_SUBJECT_RELATIONSHIP
    elif layer in {WRITE_LAYER_STABLE_BOUNDARY, WRITE_LAYER_SHORT_STATE} and not subject:
        subject = WRITE_SUBJECT_USER
    elif layer == WRITE_LAYER_PROCESS_EVENT and not subject:
        subject = WRITE_SUBJECT_EVENT

    if subject not in WRITE_SUBJECTS:
        subject = WRITE_SUBJECT_EVENT
    if layer not in WRITE_LAYERS:
        layer = WRITE_LAYER_PROCESS_EVENT

    if source == "model" and (
        normalize_write_subject(memory_subject) != subject
        or normalize_write_layer(memory_layer) != layer
    ):
        source = "model_adjusted"

    return {
        "memory_subject": subject,
        "memory_layer": layer,
        "memory_classification_source": source,
    }


def _metadata(item: dict[str, Any]) -> dict[str, Any]:
    meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    return meta


def _moment_metadata(moment: dict[str, Any]) -> dict[str, Any]:
    meta = _metadata(moment)
    mapped = dict(meta)
    if "bucket_type" in meta and "type" not in mapped:
        mapped["type"] = meta.get("bucket_type")
    if "bucket_anchor" in meta and "anchor" not in mapped:
        mapped["anchor"] = meta.get("bucket_anchor")
    if "bucket_pinned" in meta and "pinned" not in mapped:
        mapped["pinned"] = meta.get("bucket_pinned")
    if "bucket_protected" in meta and "protected" not in mapped:
        mapped["protected"] = meta.get("bucket_protected")
    if "bucket_favorite_tags" in meta and "tags" not in mapped:
        mapped["tags"] = meta.get("bucket_favorite_tags")
    if meta.get("bucket_favorite") and "tags" not in mapped:
        mapped["tags"] = [FAVORITE_TAG]
    return mapped


def _tags(meta: dict[str, Any]) -> set[str]:
    raw = meta.get("tags") or meta.get("bucket_tags") or []
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",")]
    if not isinstance(raw, (list, tuple, set)):
        return set()
    return {_lower(tag) for tag in raw if str(tag or "").strip()}


def _has_favorite_tag(tags: set[str]) -> bool:
    return has_favorite_memory_tag(tags)


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _lower(value: object) -> str:
    return str(value or "").strip().lower()


def _write_key(value: object) -> str:
    return re.sub(r"[^0-9a-zA-Z_\-\u4e00-\u9fff]+", "_", _lower(value)).strip("_")


def _infer_write_classification(tags: set[str], content: str) -> tuple[str, str]:
    hard_subject, hard_layer = _hard_write_classification(tags)
    if hard_subject and hard_layer:
        return hard_subject, hard_layer
    text = _lower(content)
    if any(term in text for term in ("不喜欢", "边界", "习惯", "偏好", "害怕", "讨厌")):
        return WRITE_SUBJECT_USER, WRITE_LAYER_STABLE_BOUNDARY
    if any(term in text for term in ("头疼", "睡不好", "今天", "这几天", "最近状态")):
        return WRITE_SUBJECT_USER, WRITE_LAYER_SHORT_STATE
    if any(term in text for term in ("以后要", "下次", "需要先", "承诺", "约定")):
        return WRITE_SUBJECT_RELATIONSHIP, WRITE_LAYER_RELATIONSHIP_LESSON
    return WRITE_SUBJECT_EVENT, WRITE_LAYER_PROCESS_EVENT


def _hard_write_classification(tags: set[str]) -> tuple[str, str]:
    if tags & {"boundary", "stable_preference", "profile_fact"}:
        return WRITE_SUBJECT_USER, WRITE_LAYER_STABLE_BOUNDARY
    if tags & {"identity", "signal", "relationship_event", "commitment", "wish"}:
        return WRITE_SUBJECT_RELATIONSHIP, WRITE_LAYER_RELATIONSHIP_LESSON
    if tags & {"project_event", "todo", "memory_system"}:
        return WRITE_SUBJECT_EVENT, WRITE_LAYER_PROCESS_EVENT
    return "", ""
