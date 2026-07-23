from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from memory_diffusion import (
    diffuse_memory,
    path_has_caution,
    path_has_old_version,
    should_suppress_context_candidate,
)
from .diffusion_admission import MemoryDiffusionAdmissionPolicy
from .diffusion_formatter import MemoryDiffusionFormatter
from memory_layers import can_moment_be_direct_seed, can_moment_be_related_target
from .runtime import (
    MemoryDiffusionCandidatePool,
    MemoryDiffusionPlan,
    MemoryDiffusionSelection,
)
from memory_relevance import recall_rank
from recall_policy import RecallPolicy
from utils import count_tokens_approx


@dataclass(frozen=True)
class MemoryDiffusionCandidateRequest:
    diffusion_plan: MemoryDiffusionPlan
    diffusion_options: Any
    seed_moments: list[dict]
    moment_candidates: list[dict]
    moments: list[dict]
    edges: list[dict]
    query_text: str
    query_plan: Any
    seed_bucket_ids: frozenset[str]
    allow_caution_paths: bool
    allow_archive_targets: bool
    session_hard_excluded_ids: frozenset[str]
    explicit_bucket_ids: frozenset[str]
    explicit_moment_ids: frozenset[str]
    direct_detail_requested: bool
    dynamic_anchor_plan: dict[str, Any] = field(default_factory=dict)
    representatives: dict[str, dict] = field(default_factory=dict)
    edge_min_confidence: float = 0.55
    inject_max_cards: int = 2
    dynamic_anchor_payload: Callable[[dict, dict[str, Any]], dict[str, Any]] | None = None


class MemoryDiffusionCandidateExplorer:
    """Explore secondary/direct and graph sources into admitted candidates."""

    def __init__(
        self,
        *,
        recall_policy: RecallPolicy,
        admission_policy: MemoryDiffusionAdmissionPolicy,
        relevance_options: Any,
    ) -> None:
        self.recall_policy = recall_policy
        self.admission_policy = admission_policy
        self.relevance_options = relevance_options

    def explore(self, request: MemoryDiffusionCandidateRequest) -> MemoryDiffusionSelection:
        plan = request.diffusion_plan
        candidate_pool = MemoryDiffusionCandidatePool(
            excluded_same_event_clusters=plan.same_event_seed_clusters,
        )

        def add_candidate(row: dict[str, Any]) -> None:
            moment = row.get("moment")
            if not isinstance(moment, dict):
                return
            bucket_id = str(moment.get("bucket_id") or "")
            moment_id = str(moment.get("moment_id") or "")
            if not bucket_id or not moment_id or bucket_id in request.seed_bucket_ids:
                return
            if self.moment_is_caution_or_old(moment) and not request.allow_caution_paths:
                return

            row["bucket_id"] = bucket_id
            row["moment_id"] = moment_id
            topic_evidence_terms = self.moment_query_topic_evidence_terms(
                request.query_text,
                moment,
            )
            row["topic_evidence_terms"] = topic_evidence_terms
            row["strong_topic_evidence"] = self.topic_evidence_terms_are_strong(
                topic_evidence_terms
            )
            row["has_topic_evidence"] = bool(
                topic_evidence_terms
            ) or self.recall_policy.moment_has_topic_evidence(request.query_text, moment)

            dynamic_plan = request.dynamic_anchor_plan
            if dynamic_plan and request.dynamic_anchor_payload is not None:
                row.update(request.dynamic_anchor_payload(moment, dynamic_plan))
                if dynamic_plan.get("strict_diffusion"):
                    row["dynamic_anchor_required_terms"] = list(
                        dynamic_plan.get("required_terms") or []
                    )
                    row["dynamic_anchor_category_terms"] = list(
                        dynamic_plan.get("category_terms") or []
                    )
                    row["dynamic_anchor_category_overview"] = bool(
                        dynamic_plan.get("category_overview")
                    )

            row["runtime_allowed"] = can_moment_be_related_target(
                moment,
                explicit_lookup=request.allow_archive_targets,
            )
            if (
                bucket_id in request.session_hard_excluded_ids
                and not self._session_exclusion_bypass(request, moment)
            ):
                allowed, reason = False, "session_hard_exclude"
            else:
                allowed, reason = self.admission_policy.decide(row, request.query_plan)
            row["gate_allowed"] = allowed
            row["gate_reason"] = "" if allowed else reason
            row["injectable"] = allowed
            row["suppression_reason"] = "" if allowed else reason
            row["injected"] = False
            candidate_pool.add(row)

        for moment in self.secondary_direct_moments(request):
            confidence, confidence_source = self.secondary_direct_candidate_confidence(moment)
            add_candidate(
                {
                    "moment": moment,
                    "why": "semantic_neighbor",
                    "confidence": confidence,
                    "confidence_source": confidence_source,
                    "confidence_defaulted": confidence_source == "default",
                    "note": "related_query_hit",
                    "source": "secondary_direct",
                    "path": None,
                    "path_len": 0,
                    "activation": _safe_float(moment.get("score"), 0.0),
                    "chain_bundle": False,
                }
            )

        if request.diffusion_options.enabled and request.diffusion_options.top_k > 0:
            self._add_graph_candidates(request, add_candidate)

        return candidate_pool.select(plan.inject_limit)

    def _add_graph_candidates(
        self,
        request: MemoryDiffusionCandidateRequest,
        add_candidate: Callable[[dict[str, Any]], None],
    ) -> None:
        plan = request.diffusion_plan
        filtered_edges = [
            edge
            for edge in request.edges
            if _safe_float(edge.get("confidence"), 0.0) >= request.edge_min_confidence
        ]
        source_record_seed_terms = self.source_record_seed_terms_by_id(request.seed_moments)
        if source_record_seed_terms:
            filtered_edges.extend(self.source_record_fragment_seed_edges(request))

        hits = diffuse_memory(
            plan.seed_scores,
            filtered_edges,
            plan.moment_map,
            options=plan.explore_options,
            exclude_ids={
                moment["moment_id"]
                for moment in request.seed_moments
                if moment.get("moment_id")
            },
            query_text=request.query_text,
        )
        seen_moment_ids: set[str] = set()
        for hit in hits:
            moment = plan.moment_map.get(hit.bucket_id)
            if not moment or hit.bucket_id in seen_moment_ids:
                continue
            bucket_id = str(moment.get("bucket_id") or "")
            if bucket_id in request.seed_bucket_ids:
                continue
            if not can_moment_be_related_target(
                moment,
                explicit_lookup=request.allow_archive_targets,
            ):
                replacement = request.representatives.get(bucket_id)
                if not replacement:
                    continue
                moment = replacement
                if moment.get("moment_id") in seen_moment_ids:
                    continue
                if not can_moment_be_related_target(
                    moment,
                    explicit_lookup=request.allow_archive_targets,
                ):
                    continue
            path = self.select_path(
                hit.paths,
                plan.moment_map,
                request.allow_caution_paths,
            )
            if path is None:
                continue
            source_record_terms = self.source_record_path_topic_terms(
                path,
                source_record_seed_terms,
            )
            if source_record_terms and not self.moment_matches_source_record_topic_terms(
                moment,
                source_record_terms,
            ):
                continue
            add_candidate(
                {
                    "moment": moment,
                    "why": self.admission_policy.path_why(path),
                    "confidence": self.admission_policy.path_confidence(
                        path,
                        default=hit.activation,
                    ),
                    "note": self.diffused_path_note(path, plan.moment_map),
                    "source": "graph",
                    "path": path,
                    "path_len": self.admission_policy.path_cross_bucket_hops(
                        path,
                        plan.moment_map,
                    ),
                    "activation": _safe_float(hit.activation, 0.0),
                    "chain_bundle": (
                        request.diffusion_options.chain_walk_enabled
                        and len(getattr(path, "steps", ()) or ()) >= 2
                    ),
                }
            )
            seen_moment_ids.add(str(moment.get("moment_id") or hit.bucket_id))

    def secondary_direct_moments(
        self,
        request: MemoryDiffusionCandidateRequest,
    ) -> list[dict]:
        hidden: list[dict] = []
        seen = set(request.seed_bucket_ids)
        for moment in request.moment_candidates:
            bucket_id = str(moment.get("bucket_id") or "")
            if not bucket_id or bucket_id in seen:
                continue
            if not can_moment_be_direct_seed(moment):
                continue
            if should_suppress_context_candidate(
                request.query_text,
                moment,
                self.relevance_options,
            ):
                continue
            if (
                request.query_plan.enforce_topic_evidence
                and not self.recall_policy.moment_has_topic_evidence(
                    request.query_text,
                    moment,
                )
            ):
                continue
            hidden.append(moment)
            seen.add(bucket_id)
        if request.query_plan.wants_body_chain:
            hidden.sort(
                key=lambda moment: recall_rank(
                    request.query_text,
                    moment,
                    self.relevance_options,
                )
            )
            return hidden[: max(0, request.diffusion_plan.explore_limit)]
        default_limit = max(0, min(2, request.inject_max_cards))
        limit = request.diffusion_plan.explore_limit
        return hidden[: max(0, limit if limit is not None else default_limit)]

    @staticmethod
    def secondary_direct_candidate_confidence(
        moment: dict,
        *,
        default: float = 0.72,
    ) -> tuple[float, str]:
        for key in ("rerank_score", "semantic_score"):
            value = moment.get(key)
            if value is None:
                continue
            confidence = _safe_float(value, -1.0)
            if confidence > 0:
                return _clamp(confidence), key
        return _clamp(default), "default"

    def moment_query_topic_evidence_terms(self, query: str, moment: dict) -> list[str]:
        field_key = _compact_lookup_key(self.moment_search_fields(moment))
        if not field_key:
            return []
        matched: list[str] = []
        seen: set[str] = set()
        for term in self.recall_policy.specific_query_terms(query):
            cleaned = str(term or "").strip()
            key = _compact_lookup_key(cleaned)
            if not key or key in seen:
                continue
            if len(key) < 2 and not re.search(r"\d", key):
                continue
            if key in field_key:
                matched.append(cleaned)
                seen.add(key)
        return matched

    @staticmethod
    def topic_evidence_terms_are_strong(terms: list[str]) -> bool:
        keys = [_compact_lookup_key(term) for term in terms if _compact_lookup_key(term)]
        if any(len(key) >= 4 for key in keys):
            return True
        return len({key for key in keys if len(key) >= 2}) >= 2

    @staticmethod
    def source_record_seed_terms_by_id(seed_moments: list[dict]) -> dict[str, list[str]]:
        terms_by_id: dict[str, list[str]] = {}
        for moment in seed_moments or []:
            if not MemoryDiffusionCandidateExplorer.is_source_record_fragment_seed(moment):
                continue
            moment_id = str(moment.get("moment_id") or "")
            if not moment_id:
                continue
            metadata = moment.get("metadata", {}) if isinstance(moment.get("metadata"), dict) else {}
            terms = [
                str(term).strip()
                for term in metadata.get("source_record_topic_terms", []) or []
                if str(term).strip()
            ]
            if terms:
                terms_by_id[moment_id] = terms
        return terms_by_id

    def source_record_fragment_seed_edges(
        self,
        request: MemoryDiffusionCandidateRequest,
    ) -> list[dict]:
        edges: list[dict] = []
        seed_terms = self.source_record_seed_terms_by_id(request.seed_moments)
        if not seed_terms:
            return edges
        candidates = [
            moment for moment in request.moments if can_moment_be_related_target(moment)
        ]
        for seed in request.seed_moments:
            seed_id = str(seed.get("moment_id") or "")
            terms = seed_terms.get(seed_id) or []
            if not seed_id or not terms:
                continue
            seed_bucket_id = str(seed.get("bucket_id") or "")
            term_document_counts = self.source_record_term_document_counts(
                candidates,
                terms,
                seed_bucket_id=seed_bucket_id,
            )
            query_keys = {
                _compact_lookup_key(term)
                for term in self.recall_policy.specific_query_terms(request.query_text)
                if _compact_lookup_key(term)
            }
            ranked: list[tuple[tuple[int, float], list[str], bool, dict]] = []
            for moment in candidates:
                if str(moment.get("bucket_id") or "") == seed_bucket_id:
                    continue
                matched_terms = self.matched_source_record_topic_terms(moment, terms)
                if not matched_terms:
                    continue
                strong_match = self.source_record_match_is_strong(
                    matched_terms,
                    term_document_counts,
                    query_keys,
                )
                rank_query = " ".join([request.query_text, *matched_terms]).strip()
                ranked.append(
                    (
                        recall_rank(rank_query, moment, self.relevance_options),
                        matched_terms,
                        strong_match,
                        moment,
                    )
                )
            ranked.sort(key=lambda item: item[0])
            limit = max(4, min(12, request.diffusion_options.top_k * 3))
            for _rank, matched_terms, strong_match, moment in ranked[:limit]:
                target_id = str(moment.get("moment_id") or "")
                if not target_id:
                    continue
                relation_type = "same_topic" if strong_match else "relates_to"
                confidence = 0.92 if strong_match else 0.54
                reason_prefix = (
                    "source_record_fragment_topic_evidence"
                    if strong_match
                    else "source_record_fragment_weak_evidence"
                )
                edges.append(
                    {
                        "source": seed_id,
                        "target": target_id,
                        "bucket_id": seed_bucket_id,
                        "relation_type": relation_type,
                        "confidence": confidence,
                        "reason": f"{reason_prefix}:" + ",".join(matched_terms[:3]),
                    }
                )
        return edges

    def source_record_term_document_counts(
        self,
        candidates: list[dict],
        terms: list[str],
        *,
        seed_bucket_id: str,
    ) -> dict[str, int]:
        counts: dict[str, set[str]] = {}
        for moment in candidates:
            bucket_id = str(moment.get("bucket_id") or "")
            if not bucket_id or bucket_id == seed_bucket_id:
                continue
            for term in self.matched_source_record_topic_terms(moment, terms):
                key = _compact_lookup_key(term)
                if key:
                    counts.setdefault(key, set()).add(bucket_id)
        return {key: len(bucket_ids) for key, bucket_ids in counts.items()}

    @staticmethod
    def source_record_match_is_strong(
        matched_terms: list[str],
        term_document_counts: dict[str, int],
        query_keys: set[str],
    ) -> bool:
        keys: list[str] = []
        for term in matched_terms:
            key = _compact_lookup_key(term)
            if key and key not in keys:
                keys.append(key)
        if not keys:
            return False
        if len(keys) >= 2:
            return True
        key = keys[0]
        if len(key) >= 4 or re.search(r"\d", key) or re.fullmatch(r"[a-z0-9_.:-]{3,}", key):
            return True
        if key in query_keys and any(
            other != key and len(other) > len(key) and key in other
            for other in query_keys
        ):
            return False
        if key in query_keys and len(key) >= 3:
            return False
        return term_document_counts.get(key, 0) <= 3

    @staticmethod
    def source_record_path_topic_terms(
        path: Any,
        terms_by_id: dict[str, list[str]],
    ) -> list[str]:
        nodes = tuple(str(node_id) for node_id in (getattr(path, "nodes", ()) or ()))
        return terms_by_id.get(nodes[0], []) if nodes else []

    @classmethod
    def matched_source_record_topic_terms(
        cls,
        moment: dict,
        terms: list[str],
    ) -> list[str]:
        fields = cls.moment_search_fields(moment)
        matched: list[str] = []
        seen: set[str] = set()
        for term in terms:
            cleaned = str(term or "").strip()
            key = cleaned.lower()
            if not key or key in seen:
                continue
            if key in fields:
                matched.append(cleaned)
                seen.add(key)
        return matched

    @classmethod
    def moment_matches_source_record_topic_terms(
        cls,
        moment: dict,
        terms: list[str],
    ) -> bool:
        return bool(cls.matched_source_record_topic_terms(moment, terms))

    @classmethod
    def select_path(
        cls,
        paths: tuple[Any, ...],
        moment_map: dict[str, dict],
        allow_caution_paths: bool,
    ) -> Any | None:
        for path in paths or ():
            if allow_caution_paths or not cls.path_is_caution_or_old(path, moment_map):
                return path
        return None

    @classmethod
    def path_is_caution_or_old(cls, path: Any, moment_map: dict[str, dict]) -> bool:
        return (
            path_has_caution(path)
            or path_has_old_version(path)
            or cls.path_has_old_moment(path, moment_map)
        )

    @classmethod
    def path_has_old_moment(cls, path: Any, moment_map: dict[str, dict]) -> bool:
        return any(
            cls.moment_is_caution_or_old(moment_map.get(str(node_id)))
            for node_id in getattr(path, "nodes", ()) or ()
        )

    @staticmethod
    def moment_is_caution_or_old(moment: dict | None) -> bool:
        if not isinstance(moment, dict):
            return False
        metadata = moment.get("metadata", {}) if isinstance(moment.get("metadata"), dict) else {}
        if (
            metadata.get("resolved")
            or metadata.get("digested")
            or metadata.get("bucket_resolved")
            or metadata.get("bucket_digested")
        ):
            return True
        if str(metadata.get("type") or metadata.get("bucket_type") or "").lower() == "archived":
            return True
        haystack = " ".join(
            [
                str(metadata.get("name") or metadata.get("bucket_name") or ""),
                " ".join(
                    str(item)
                    for item in metadata.get("tags", []) or metadata.get("bucket_tags", []) or []
                ),
                " ".join(
                    str(item)
                    for item in metadata.get("domain", []) or metadata.get("bucket_domain", []) or []
                ),
                str(moment.get("text") or ""),
            ]
        ).lower()
        return any(
            marker in haystack
            for marker in (
                "冲突",
                "吵架",
                "争吵",
                "矛盾",
                "误会",
                "旧版本",
                "旧版",
                "旧链",
                "旧窗口",
                "已解决",
                "过期",
                "归档",
                "conflict",
                "fight",
                "argument",
                "old version",
                "old path",
                "old chain",
                "resolved",
                "archived",
                "deprecated",
                "obsolete",
            )
        )

    @classmethod
    def diffused_path_note(cls, path: Any, moment_map: dict[str, dict]) -> str:
        if path_has_caution(path):
            return "conflict_or_blocking_path"
        if path_has_old_version(path) or cls.path_has_old_moment(path, moment_map):
            return "old_or_resolved_path"
        return "background_association_not_current_fact"

    @staticmethod
    def moment_search_fields(moment: dict) -> str:
        metadata = moment.get("metadata", {}) if isinstance(moment.get("metadata"), dict) else {}
        return " ".join(
            [
                str(moment.get("text") or ""),
                str(metadata.get("bucket_name") or ""),
                " ".join(str(item) for item in metadata.get("bucket_tags", []) or []),
                " ".join(str(item) for item in metadata.get("bucket_domain", []) or []),
            ]
        ).lower()

    @staticmethod
    def is_source_record_fragment_seed(moment: dict | None) -> bool:
        metadata = (
            moment.get("metadata", {})
            if isinstance(moment, dict) and isinstance(moment.get("metadata"), dict)
            else {}
        )
        synthetic = bool(
            metadata.get("source_record_direct")
            or (moment or {}).get("_source_record_synthetic")
        )
        return synthetic and bool(metadata.get("source_record_fragment_seed"))

    @staticmethod
    def _session_exclusion_bypass(
        request: MemoryDiffusionCandidateRequest,
        moment: dict,
    ) -> bool:
        bucket_id = str(moment.get("bucket_id") or "")
        moment_id = str(moment.get("moment_id") or "")
        return bool(
            (bucket_id and bucket_id in request.explicit_bucket_ids)
            or (moment_id and moment_id in request.explicit_moment_ids)
            or request.direct_detail_requested
            or MemoryDiffusionCandidateExplorer.is_source_record_fragment_seed(moment)
        )


@dataclass(frozen=True)
class MemoryDiffusionRenderRequest:
    selection: MemoryDiffusionSelection
    moment_map: dict[str, dict]
    query_text: str
    context_mode: str
    explicit_lookup: bool
    related_max_chars: int
    token_budget: int


@dataclass(frozen=True)
class MemoryDiffusionRenderHooks:
    build_reading_note: Callable[[str, dict, str, str], dict]
    format_reading_note: Callable[[dict], str]
    trim_text: Callable[[str, int], str]


@dataclass(frozen=True)
class MemoryDiffusionRenderResult:
    context: str
    debug_rows: list[dict[str, Any]]


class MemoryDiffusionContextRenderer:
    """Apply token budget, render admitted candidates, and emit debug receipts."""

    def __init__(self, *, formatter: MemoryDiffusionFormatter) -> None:
        self.formatter = formatter

    def render(
        self,
        request: MemoryDiffusionRenderRequest,
        hooks: MemoryDiffusionRenderHooks,
    ) -> MemoryDiffusionRenderResult:
        remaining = request.token_budget
        parts: list[str] = []
        for row in request.selection.selected:
            if remaining <= 0:
                row["suppression_reason"] = "budget_exhausted"
                continue
            moment = row["moment"]
            reading_note = hooks.build_reading_note(
                request.query_text,
                moment,
                request.context_mode,
                "diffused",
            )
            row["reading_note"] = reading_note
            block = self.formatter.format_moment_line(
                moment,
                max_chars=request.related_max_chars,
                note=self.display_note(row),
                path=row.get("path"),
                moment_map=request.moment_map,
                chain_bundle=bool(row.get("chain_bundle")),
            )
            block = f"{block}\n  {hooks.format_reading_note(reading_note)}"
            tokens = count_tokens_approx(block)
            if tokens > remaining and parts:
                row["suppression_reason"] = "budget_exhausted"
                break
            if tokens > remaining:
                block = hooks.trim_text(block, remaining)
                tokens = count_tokens_approx(block)
            if tokens <= 0:
                row["suppression_reason"] = "budget_exhausted"
                continue
            parts.append(block)
            row["injected"] = True
            row["suppression_reason"] = ""
            remaining -= tokens

        debug_rows = [
            self.formatter.format_candidate_debug(
                row,
                moment_map=request.moment_map,
                explicit_lookup=request.explicit_lookup,
                query=request.query_text,
            )
            for row in request.selection.candidates[:20]
        ]
        return MemoryDiffusionRenderResult(
            context="\n".join(parts),
            debug_rows=debug_rows,
        )

    @staticmethod
    def display_note(row: dict[str, Any]) -> str:
        why = str(row.get("why") or "explicit_edge")
        confidence = _safe_float(row.get("confidence"), 0.0)
        note = str(row.get("note") or "").strip()
        prefix = f"why:{why} confidence:{confidence:.2f}"
        return f"{prefix}; {note}" if note else prefix


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp(value: Any) -> float:
    return max(0.0, min(1.0, _safe_float(value, 0.0)))


def _compact_lookup_key(value: object) -> str:
    return re.sub(
        r"[^0-9a-z\u4e00-\u9fff_.:-]+",
        "",
        str(value or "").strip().lower(),
    )
