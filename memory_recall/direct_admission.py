from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .axis_policy import MemoryAxisPolicy
from .diffusion_formatter import MOMENT_TEMPERATURE_SECTIONS
from memory_metadata import normalize_memory_metadata
from recall_policy import QueryAnchorPlan, RecallPolicy
from self_anchor import is_self_anchor_metadata
from utils import bucket_content_for_recall


@dataclass(frozen=True)
class DirectAdmissionSignals:
    """Evidence already measured by candidate formation for one direct gate."""

    planner_lexical: bool = False
    word_map: bool = False
    entity_edge: bool = False
    identity_name: bool = False
    direct_detail_requested: bool = False
    tech_query_has_anchor: bool = False
    self_anchor_excluded: bool = False
    evidence_labels: tuple[str, ...] = ()
    hard_evidence_labels: tuple[str, ...] = ()
    locatable_terms: tuple[str, ...] = ()


class MemoryDirectAdmissionPolicy:
    """Own direct bucket/Moment admission without owning candidate retrieval."""

    def __init__(
        self,
        *,
        recall_policy: RecallPolicy,
        axis_policy: MemoryAxisPolicy,
        first_card_min_score: float,
        second_card_min_score: float,
        high_confidence_semantic_score: float,
        recall_admission_semantic_score: float,
    ) -> None:
        self.recall_policy = recall_policy
        self.axis_policy = axis_policy
        self.first_card_min_score = float(first_card_min_score)
        self.second_card_min_score = float(second_card_min_score)
        self.high_confidence_semantic_score = float(high_confidence_semantic_score)
        self.recall_admission_semantic_score = float(recall_admission_semantic_score)

    def admit_bucket(
        self,
        query: str,
        item: dict,
        *,
        signals: DirectAdmissionSignals,
    ) -> bool:
        bucket = item.get("bucket") if isinstance(item, dict) else None
        if not isinstance(bucket, dict) or signals.self_anchor_excluded:
            return False

        evidence_labels = list(signals.evidence_labels)
        hard_evidence_labels = list(signals.hard_evidence_labels)
        item["evidence_labels"] = evidence_labels
        item["hard_evidence_labels"] = hard_evidence_labels
        dynamic_plan = (
            item.get("dynamic_anchor_plan")
            if isinstance(item.get("dynamic_anchor_plan"), dict)
            else {}
        )
        independent_anchor_evidence = bool(
            signals.planner_lexical
            or item.get("exact_anchor_match")
            or item.get("explicit_relation_edge_match")
            or signals.entity_edge
            or signals.identity_name
            or "title_anchor" in hard_evidence_labels
            or "semantic_rescue_direct_span" in hard_evidence_labels
            or "strong_semantic" in hard_evidence_labels
            or "strong_rerank" in hard_evidence_labels
        )
        dynamic_anchor_missing = bool(
            dynamic_plan.get("required_terms")
            and not item.get("distinctive_anchor_match")
            and not independent_anchor_evidence
        )
        category_overview_missing = bool(
            dynamic_plan.get("category_overview")
            and dynamic_plan.get("category_terms")
            and not item.get("category_overview_item")
        )
        query_plan = self.recall_policy.plan_query(query)
        rejection = self.anchor_rejection(
            bucket,
            self.recall_policy.build_query_anchor_plan(query),
        )
        if rejection:
            reason, debug = rejection
            if reason == "anchor_must_group_missing" and self.can_bypass_anchor_with_strong_model_score(
                query,
                semantic_score=item.get("semantic_score"),
                rerank_score=item.get("rerank_score"),
            ):
                item["recall_policy_debug"] = {
                    **debug,
                    "anchor_bypassed_by_strong_model_score": True,
                }
            else:
                item["admission_reason"] = reason
                item["recall_policy_debug"] = debug
                return False
        else:
            item.pop("recall_policy_debug", None)

        axis_rejection = self.axis_rejection(
            query_plan,
            bucket,
            item,
            signals=signals,
        )
        if axis_rejection:
            reason, debug = axis_rejection
            item["admission_reason"] = reason
            item["recall_policy_debug"] = debug
            return False

        decision = self.recall_policy.assess(
            query,
            self.bucket_relevance_node(bucket),
            query_plan=query_plan,
            has_topic_evidence=self.recall_policy.bucket_has_topic_evidence(query, bucket),
            semantic_score=item.get("semantic_score"),
            rerank_score=item.get("rerank_score"),
            high_confidence_edge=bool(
                signals.planner_lexical
                or item.get("exact_anchor_match")
                or signals.word_map
                or signals.entity_edge
                or item.get("semantic_rescue_direct_span")
                or "title_anchor" in hard_evidence_labels
            ),
            auto=True,
        )
        item["admission_reason"] = decision.reason
        if item.get("recall_policy_debug"):
            item["recall_policy_debug"] = {
                **item["recall_policy_debug"],
                "decision": decision.debug,
            }
        else:
            item["recall_policy_debug"] = decision.debug

        if decision.admit_direct and self.bucket_is_tech_domain(bucket):
            tech_rejection = self.tech_domain_rejection(
                item,
                signals=signals,
                node=bucket,
            )
            if tech_rejection:
                item["admission_reason"] = "tech_domain_without_query_anchor"
                item["recall_policy_debug"] = {
                    **self._debug(item),
                    **tech_rejection,
                }
                return False
        if decision.admit_direct and decision.reason == "non_explicit_query":
            if not self.bucket_has_reliable_signal(query, item, signals=signals):
                item["admission_reason"] = "low_recall_evidence"
                return False
        if decision.admit_direct and dynamic_anchor_missing:
            item["admission_reason"] = "discriminative_anchor_missing"
            item["blocked_reason"] = "discriminative_anchor_missing"
            item["recall_policy_debug"] = {
                **self._debug(item),
                "required_terms": list(dynamic_plan.get("required_terms") or []),
                "matched_terms": list(item.get("distinctive_anchor_terms") or []),
                "missing_terms": list(item.get("distinctive_anchor_missing_terms") or []),
                "anchor_coverage": self.safe_float(item.get("anchor_coverage"), 0.0),
                "auto": True,
            }
            return False
        if decision.admit_direct and category_overview_missing:
            item["admission_reason"] = "category_overview_item_missing"
            item["blocked_reason"] = "category_overview_item_missing"
            item["recall_policy_debug"] = {
                **self._debug(item),
                "category_terms": list(dynamic_plan.get("category_terms") or []),
                "matched_terms": list(item.get("category_overview_terms") or []),
                "auto": True,
            }
            return False
        if decision.admit_direct and not hard_evidence_labels:
            reason = self.weak_evidence_block_reason(evidence_labels)
            item["admission_reason"] = reason
            item["blocked_reason"] = reason
            item["recall_policy_debug"] = {
                **self._debug(item),
                "evidence_labels": evidence_labels,
                "hard_evidence_labels": hard_evidence_labels,
                "blocked_reason": reason,
            }
            return False
        return decision.admit_direct

    def admit_moment(
        self,
        query: str,
        moment: dict,
        *,
        signals: DirectAdmissionSignals,
        admitted_bucket_ids: set[str] | None = None,
    ) -> bool:
        if is_self_anchor_metadata(moment.get("metadata", {})):
            return False
        bucket_id = str(moment.get("bucket_id") or "")
        query_plan = self.recall_policy.plan_query(query)
        if admitted_bucket_ids is not None:
            if bucket_id in admitted_bucket_ids:
                moment["admission_reason"] = "admitted_bucket"
                return True
            moment["admission_reason"] = "bucket_not_admitted"
            moment["recall_policy_debug"] = {
                "bucket_id": bucket_id,
                "bucket_admitted": False,
                "auto": True,
            }
            return False

        dynamic_plan = (
            moment.get("dynamic_anchor_plan")
            if isinstance(moment.get("dynamic_anchor_plan"), dict)
            else {}
        )
        dynamic_anchor_missing = bool(
            dynamic_plan.get("required_terms")
            and not moment.get("distinctive_anchor_match")
        )
        category_overview_missing = bool(
            dynamic_plan.get("category_overview")
            and dynamic_plan.get("category_terms")
            and not moment.get("category_overview_item")
        )
        rejection = self.anchor_rejection(
            moment,
            self.recall_policy.build_query_anchor_plan(query),
        )
        if rejection:
            reason, debug = rejection
            if reason == "anchor_must_group_missing" and self.can_bypass_anchor_with_strong_model_score(
                query,
                semantic_score=moment.get("semantic_score"),
                rerank_score=moment.get("rerank_score"),
            ):
                moment["recall_policy_debug"] = {
                    **debug,
                    "anchor_bypassed_by_strong_model_score": True,
                }
            else:
                moment["admission_reason"] = reason
                moment["recall_policy_debug"] = debug
                return False
        else:
            moment.pop("recall_policy_debug", None)

        decision = self.recall_policy.assess(
            query,
            moment,
            query_plan=query_plan,
            has_topic_evidence=self.recall_policy.moment_has_topic_evidence(query, moment),
            semantic_score=moment.get("semantic_score"),
            rerank_score=moment.get("rerank_score"),
            context_only=moment.get("section") in MOMENT_TEMPERATURE_SECTIONS,
            auto=True,
        )
        moment["admission_reason"] = decision.reason
        if moment.get("recall_policy_debug"):
            moment["recall_policy_debug"] = {
                **moment["recall_policy_debug"],
                "decision": decision.debug,
            }
        else:
            moment["recall_policy_debug"] = decision.debug

        if decision.admit_direct and self.moment_is_tech_domain(moment):
            tech_rejection = self.tech_domain_rejection(moment, signals=signals)
            if tech_rejection:
                moment["admission_reason"] = "tech_domain_without_query_anchor"
                moment["recall_policy_debug"] = {
                    **self._debug(moment),
                    **tech_rejection,
                }
                return False
        if (
            decision.admit_direct
            and decision.reason == "non_explicit_query"
            and not self.moment_has_reliable_signal(query, moment, signals=signals)
        ):
            moment["admission_reason"] = "non_explicit_query_score_too_low"
            moment["recall_policy_debug"] = {
                **decision.debug,
                "unselected_moment_score": self.safe_float(
                    moment.get("combined_score", moment.get("score")),
                    0.0,
                ),
                "unselected_moment_min_score": self.unselected_moment_min_score(),
                "has_topic_evidence": self.recall_policy.moment_has_topic_evidence(
                    query,
                    moment,
                ),
            }
            return False
        if decision.admit_direct and dynamic_anchor_missing:
            moment["admission_reason"] = "discriminative_anchor_missing"
            moment["recall_policy_debug"] = {
                **self._debug(moment),
                "required_terms": list(dynamic_plan.get("required_terms") or []),
                "matched_terms": list(moment.get("distinctive_anchor_terms") or []),
                "missing_terms": list(moment.get("distinctive_anchor_missing_terms") or []),
                "auto": True,
            }
            return False
        if decision.admit_direct and category_overview_missing:
            moment["admission_reason"] = "category_overview_item_missing"
            moment["recall_policy_debug"] = {
                **self._debug(moment),
                "category_terms": list(dynamic_plan.get("category_terms") or []),
                "matched_terms": list(moment.get("category_overview_terms") or []),
                "auto": True,
            }
            return False
        if decision.admit_direct:
            axis_rejection = self.axis_rejection(
                query_plan,
                moment,
                moment,
                signals=signals,
            )
            if axis_rejection:
                reason, debug = axis_rejection
                moment["admission_reason"] = reason
                moment["recall_policy_debug"] = {
                    **self._debug(moment),
                    **debug,
                }
                return False
        return decision.admit_direct

    def axis_rejection(
        self,
        query_plan: Any,
        node: dict,
        item: dict,
        *,
        signals: DirectAdmissionSignals,
    ) -> tuple[str, dict[str, Any]] | None:
        hard_labels = {str(label) for label in signals.hard_evidence_labels}
        bypass = bool(
            signals.direct_detail_requested
            or hard_labels & {"title_anchor", "semantic_rescue_direct_span"}
            or signals.planner_lexical
            or item.get("exact_anchor_match")
            or signals.word_map
            or item.get("distinctive_anchor_match")
            or item.get("category_overview_item")
            or item.get("semantic_rescue_direct_span")
            or signals.entity_edge
            or self.recall_policy.has_strong_score(
                semantic_score=item.get("semantic_score"),
                rerank_score=item.get("rerank_score"),
            )
        )
        return self.axis_policy.rejection(query_plan, node, bypass=bypass)

    def tech_domain_rejection(
        self,
        item: dict,
        *,
        signals: DirectAdmissionSignals,
        node: dict | None = None,
    ) -> dict[str, Any] | None:
        if (
            signals.planner_lexical
            or item.get("exact_anchor_match")
            or item.get("explicit_relation_edge_match")
            or signals.entity_edge
            or item.get("distinctive_anchor_match")
        ):
            return None
        if self._has_high_confidence_direct_semantic_evidence(item):
            return None
        if signals.tech_query_has_anchor:
            return None
        return {
            "tech_domain_guard": True,
            "reason": "tech_domain_without_query_anchor",
            "locatable_terms": list(signals.locatable_terms),
            "canonical_domain": "tech" if node is None else str(
                normalize_memory_metadata(node).get("canonical_domain") or ""
            ),
        }

    def bucket_has_reliable_signal(
        self,
        query: str,
        item: dict,
        *,
        signals: DirectAdmissionSignals,
    ) -> bool:
        if (
            signals.planner_lexical
            or item.get("exact_anchor_match")
            or item.get("distinctive_anchor_match")
            or item.get("category_overview_item")
            or item.get("semantic_rescue_direct_span")
            or signals.entity_edge
        ):
            return True
        if self.recall_policy.has_strong_score(
            semantic_score=item.get("semantic_score"),
            rerank_score=item.get("rerank_score"),
        ):
            return True
        bucket = item.get("bucket") if isinstance(item.get("bucket"), dict) else None
        if (
            bucket
            and item.get("entity_edge_match")
            and self.safe_float(item.get("entity_edge_score"), 0.0) >= 0.62
            and self.recall_policy.bucket_has_topic_evidence(query, bucket)
        ):
            return True
        return bool(
            bucket and self.recall_policy.bucket_has_topic_evidence(query, bucket)
        )

    def moment_has_reliable_signal(
        self,
        query: str,
        moment: dict,
        *,
        signals: DirectAdmissionSignals,
    ) -> bool:
        if (
            moment.get("exact_anchor_match")
            or signals.planner_lexical
            or moment.get("distinctive_anchor_match")
            or moment.get("category_overview_item")
        ):
            return True
        if not self.recall_policy.moment_has_topic_evidence(query, moment):
            return False
        return self.recall_policy.has_strong_score(
            semantic_score=moment.get("semantic_score"),
            rerank_score=moment.get("rerank_score"),
        )

    def can_bypass_anchor_with_strong_model_score(
        self,
        query: str,
        *,
        semantic_score: Any = None,
        rerank_score: Any = None,
    ) -> bool:
        if not self.recall_policy.has_strong_score(
            semantic_score=semantic_score,
            rerank_score=rerank_score,
        ):
            return False
        affect_only = {
            "哭",
            "哭了",
            "难过",
            "伤心",
            "开心",
            "激动",
            "生气",
            "委屈",
            "情绪",
            "感觉",
            "emo",
        }
        terms = [
            str(term).strip().lower()
            for term in self.recall_policy.specific_query_terms(query)
            if str(term).strip()
        ]
        concrete_terms = [
            term
            for term in terms
            if term not in affect_only
            and not any(marker in term for marker in affect_only)
        ]
        return len("".join(concrete_terms)) >= 3

    def anchor_rejection(
        self,
        node: dict,
        plan: QueryAnchorPlan,
    ) -> tuple[str, dict[str, Any]] | None:
        if not plan.has_direct_constraints:
            return None
        if self.recall_policy.direct_candidate_satisfies_anchor_plan(node, plan):
            return None
        reason = "anchor_direct_disallowed" if not plan.allow_direct else "anchor_must_group_missing"
        return reason, {
            "query_anchor_plan": self.anchor_plan_debug(plan),
            "must_groups_matched": False,
            "auto": True,
        }

    @staticmethod
    def anchor_plan_debug(plan: QueryAnchorPlan) -> dict[str, Any]:
        return {
            "route": plan.route,
            "focus_query": plan.focus_query,
            "strong_terms": list(plan.strong_terms),
            "weak_terms": list(plan.weak_terms),
            "must_groups": [list(group) for group in plan.must_groups],
            "allow_direct": plan.allow_direct,
            "allow_diffusion_seed": plan.allow_diffusion_seed,
            "debug": dict(plan.debug or {}),
        }

    def unselected_moment_min_score(self) -> float:
        return min(
            self.second_card_min_score,
            max(0.30, self.first_card_min_score * 0.55),
        )

    @staticmethod
    def bucket_relevance_node(bucket: dict) -> dict:
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        return {
            "content": bucket_content_for_recall(bucket),
            "name": meta.get("name") or bucket.get("id") or "",
            "metadata": meta,
        }

    @staticmethod
    def bucket_is_tech_domain(bucket: dict | None) -> bool:
        if not isinstance(bucket, dict):
            return False
        view = normalize_memory_metadata(bucket)
        return str(view.get("domain_parent") or view.get("canonical_domain") or "") == "tech"

    @staticmethod
    def moment_is_tech_domain(moment: dict | None) -> bool:
        if not isinstance(moment, dict):
            return False
        meta = moment.get("metadata", {}) if isinstance(moment.get("metadata"), dict) else {}
        bucket_view = {
            "id": moment.get("bucket_id"),
            "metadata": {
                "name": meta.get("bucket_name") or meta.get("name"),
                "domain": meta.get("bucket_domain") or meta.get("domain") or [],
                "tags": meta.get("bucket_tags") or meta.get("tags") or [],
                "type": meta.get("type") or meta.get("bucket_type"),
                "path": meta.get("path") or meta.get("bucket_path"),
                "pinned": meta.get("pinned") or meta.get("bucket_pinned"),
                "protected": meta.get("protected") or meta.get("bucket_protected"),
                "resolved": (
                    meta.get("resolved")
                    if "resolved" in meta
                    else meta.get("bucket_resolved")
                ),
                "digested": meta.get("digested") or meta.get("bucket_digested"),
            },
        }
        view = normalize_memory_metadata(bucket_view)
        return str(view.get("domain_parent") or view.get("canonical_domain") or "") == "tech"

    def _has_high_confidence_direct_semantic_evidence(self, item: dict) -> bool:
        if not isinstance(item, dict):
            return False
        semantic_score = self.safe_float(item.get("semantic_score"), 0.0)
        final_value = item.get("score")
        if final_value is None:
            final_value = item.get("combined_score")
        final_score = self.safe_float(final_value, 0.0)
        semantic_threshold = max(
            self.high_confidence_semantic_score,
            self.recall_admission_semantic_score,
        )
        final_threshold = max(self.first_card_min_score, semantic_threshold)
        if semantic_score < semantic_threshold or final_score < final_threshold:
            return False
        item["recall_policy_debug"] = {
            **self._debug(item),
            "tech_domain_high_confidence_semantic_bypass": True,
            "tech_domain_semantic_score": round(semantic_score, 4),
            "tech_domain_final_score": round(final_score, 4),
            "tech_domain_semantic_threshold": round(semantic_threshold, 4),
            "tech_domain_final_threshold": round(final_threshold, 4),
        }
        return True

    @staticmethod
    def weak_evidence_block_reason(labels: list[str]) -> str:
        label_set = {
            str(label or "").strip()
            for label in labels or []
            if str(label or "").strip()
        }
        if not label_set:
            return "no_hard_evidence"
        if label_set == {"semantic_hit"}:
            return "semantic_only"
        if label_set.issubset({"retrieval_alias", "semantic_hit", "graph_related"}):
            return "retrieval_alias_only"
        if "category_seed" in label_set and label_set.issubset(
            {"category_seed", "semantic_hit", "graph_related"}
        ):
            return "generic_category_only"
        if label_set.issubset({"semantic_hit", "graph_related"}):
            return "weak_evidence_only"
        return "no_hard_evidence"

    @staticmethod
    def _debug(item: dict) -> dict[str, Any]:
        value = item.get("recall_policy_debug")
        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)
