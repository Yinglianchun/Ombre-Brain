from __future__ import annotations

import re
from typing import Any, Callable

from .axis_policy import MemoryAxisPolicy
from memory_diffusion import path_has_caution


class MemoryDiffusionAdmissionPolicy:
    """Decide whether an already discovered diffusion candidate may inject."""

    def __init__(
        self,
        *,
        min_confidence: Callable[[], float],
        high_semantic_score: Callable[[], float],
        axis_policy: MemoryAxisPolicy,
        has_axis_relation_marker: Callable[[str], bool],
    ) -> None:
        self._min_confidence = min_confidence
        self._high_semantic_score = high_semantic_score
        self._axis_policy = axis_policy
        self._has_axis_relation_marker = has_axis_relation_marker

    @staticmethod
    def requires_reliable_direct_seed(query_plan: Any) -> bool:
        return (
            not bool(getattr(query_plan, "requires_topic_evidence", False))
            and not bool(getattr(query_plan, "wants_body_chain", False))
        )

    def decide(self, row: dict[str, Any], query_plan: Any) -> tuple[bool, str]:
        moment = row.get("moment")
        if not isinstance(moment, dict):
            return False, "invalid_candidate"
        if not row.get("runtime_allowed"):
            return False, "layer_gate_denied"

        min_confidence = _clamp(self._min_confidence())
        confidence = _safe_float(row.get("confidence"), 0.0)
        if confidence < min_confidence:
            return False, "low_confidence"

        why = str(row.get("why") or "")
        path_len = int(row.get("path_len") or 0)
        high_confidence_explicit_edge = (
            why == "explicit_edge"
            and path_len <= 1
            and confidence >= max(min_confidence + 0.2, 0.80)
        )
        strong_explicit_edge = (
            high_confidence_explicit_edge
            and not self._axis_policy.has_technical_axis(query_plan)
        )
        if (
            row.get("dynamic_anchor_required_terms")
            and not row.get("distinctive_anchor_match")
            and not high_confidence_explicit_edge
        ):
            return False, "discriminative_anchor_missing"
        if (
            row.get("dynamic_anchor_category_overview")
            and row.get("dynamic_anchor_category_terms")
            and not row.get("category_overview_item")
        ):
            return False, "category_overview_item_missing"

        path = row.get("path")
        has_caution_path = bool(path is not None and path_has_caution(path))
        has_source_record_topic_evidence = self.path_source_record_evidence_extends_axis(
            path,
            query_plan,
        )
        explicit_edge_axis_bypass = (
            strong_explicit_edge
            and self.explicit_edge_can_bridge_axis(query_plan)
        )
        strong_local_chain = (
            bool(row.get("chain_bundle"))
            and path_len <= 2
            and confidence >= 0.85
            and not self._axis_policy.has_technical_axis(query_plan)
        )
        activated_axis_groups = getattr(query_plan, "activated_axis_groups", ()) or ()
        if activated_axis_groups and not self._axis_policy.candidate_matches(query_plan, moment):
            if not (
                has_caution_path
                or has_source_record_topic_evidence
                or explicit_edge_axis_bypass
                or strong_local_chain
                or self.semantic_neighbor_has_strong_confidence(row)
            ):
                return False, "activated_axis_mismatch"
        if activated_axis_groups and self._axis_policy.domain_mismatch(query_plan, moment):
            if not (
                has_caution_path
                or has_source_record_topic_evidence
                or explicit_edge_axis_bypass
            ):
                return False, "activated_axis_mismatch"
        if strong_local_chain:
            return True, ""
        if why in {"same_topic", "date_neighbor"}:
            return True, ""
        if row.get("has_topic_evidence"):
            if (
                why == "semantic_neighbor"
                and row.get("confidence_defaulted")
                and not row.get("strong_topic_evidence")
            ):
                return False, "query_topic_evidence_missing"
            return True, ""
        if why == "semantic_neighbor":
            if self.semantic_neighbor_has_strong_confidence(row):
                return True, ""
            return False, "query_topic_evidence_missing"
        if why == "explicit_edge":
            if explicit_edge_axis_bypass:
                return True, ""
            return False, "query_topic_evidence_missing"
        return False, "unknown_diffusion_reason"

    def explicit_edge_can_bridge_axis(self, query_plan: Any) -> bool:
        if not bool(getattr(query_plan, "activated_axis_multi", False)):
            return False
        query = str(getattr(query_plan, "query", "") or "")
        return bool(query and self._has_axis_relation_marker(query))

    def semantic_neighbor_has_strong_confidence(self, row: dict[str, Any]) -> bool:
        if str(row.get("why") or "") != "semantic_neighbor":
            return False
        if row.get("confidence_defaulted"):
            return False
        return _safe_float(row.get("confidence"), 0.0) >= _clamp(
            self._high_semantic_score()
        )

    @staticmethod
    def path_why(path: Any) -> str:
        steps = tuple(getattr(path, "steps", ()) or ())
        for step in steps:
            relation = str(getattr(step, "relation_type", "") or "").lower()
            reason = str(getattr(step, "reason", "") or "").lower()
            if (
                relation == "same_topic"
                or "same_topic" in reason
                or "source_record_fragment_topic_evidence" in reason
                or ("topic" in reason and "off-topic" not in reason)
            ):
                return "same_topic"
            if relation in {"date_neighbor", "same_date", "same_day"} or any(
                marker in reason
                for marker in ("date_neighbor", "same_date", "same day", "same-day", "same_day")
            ):
                return "date_neighbor"
        return "explicit_edge" if steps else "semantic_neighbor"

    @staticmethod
    def path_source_record_evidence_extends_axis(path: Any, query_plan: Any) -> bool:
        terms: list[str] = []
        for step in tuple(getattr(path, "steps", ()) or ()):
            reason = str(getattr(step, "reason", "") or "")
            marker = "source_record_fragment_topic_evidence:"
            if marker not in reason:
                continue
            tail = reason.split(marker, 1)[1]
            terms.extend(part.strip() for part in re.split(r"[,，、/|]", tail) if part.strip())
        if not terms:
            return False
        axis_keys = {
            MemoryAxisPolicy.compact_text(term)
            for term in (getattr(query_plan, "activated_axis_terms", ()) or ())
            if MemoryAxisPolicy.compact_text(term)
        }
        if not axis_keys:
            return True
        for term in terms:
            key = MemoryAxisPolicy.compact_text(term)
            if key and not any(key in axis_key or axis_key in key for axis_key in axis_keys):
                return True
        return False

    @staticmethod
    def path_cross_bucket_hops(path: Any, moment_map: dict[str, dict]) -> int:
        count = 0
        for step in tuple(getattr(path, "steps", ()) or ()):
            source_id = str(getattr(step, "source", "") or "")
            target_id = str(getattr(step, "target", "") or "")
            source_bucket = str((moment_map.get(source_id) or {}).get("bucket_id") or "")
            target_bucket = str((moment_map.get(target_id) or {}).get("bucket_id") or "")
            if source_bucket and target_bucket and source_bucket == target_bucket:
                continue
            count += 1
        return count

    @staticmethod
    def path_confidence(path: Any, *, default: float = 0.65) -> float:
        steps = tuple(getattr(path, "steps", ()) or ())
        if not steps:
            return _clamp(default or 0.65)
        values = [
            _safe_float(getattr(step, "confidence", 0.0), 0.0)
            for step in steps
        ]
        values = [value for value in values if value > 0]
        if not values:
            return _clamp(default or 0.65)
        return _clamp(min(values))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp(value: Any) -> float:
    return max(0.0, min(1.0, _safe_float(value, 0.0)))
