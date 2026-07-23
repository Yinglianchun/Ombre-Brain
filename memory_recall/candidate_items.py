from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .candidate_fusion import CandidateFusionReceipt


@dataclass(frozen=True)
class BucketCandidateItemInput:
    """Already measured values needed to assemble one bucket candidate item."""

    bucket: dict[str, Any]
    semantic_score: float
    keyword_score: float
    exact_anchor_score: float
    exact_anchor_match: bool
    exact_anchor_terms: Sequence[str]
    exact_anchor_fields: Sequence[str]
    word_map_score: float
    word_map_hint: bool
    word_map_terms: Sequence[str]
    word_map_variant_terms: Sequence[str]
    word_map_neighbor_terms: Sequence[str]
    word_map_activation_terms: Sequence[Mapping[str, Any]]
    word_map_category_seed_terms: Sequence[str]
    low_frequency_match: bool
    low_frequency_direct_match: bool
    low_frequency_terms: Sequence[str]
    low_frequency_direct_terms: Sequence[str]
    low_frequency_category_terms: Sequence[str]
    low_frequency_sources: Sequence[str]
    rare_name_match: bool
    rare_name_terms: Sequence[str]
    rare_name_sources: Sequence[str]
    entity_edge_score: float
    entity_edge_subject: str
    entity_edge_relation: str
    entity_edge_object: str
    importance_score: float
    freshness_score: float
    cooldown_multiplier: float
    fusion: CandidateFusionReceipt
    planner_lexical_match: bool
    planner_lexical_direct_match: bool
    planner_query: Mapping[str, Any] | None
    matched_query_terms: Sequence[str]
    dynamic_anchor: Mapping[str, Any]


class MemoryCandidateItemAssembler:
    """Assemble and project measured candidate receipts without runtime policy."""

    _MOMENT_SIGNAL_FIELDS = (
        "score",
        "semantic_score",
        "keyword_score",
        "rerank_score",
        "exact_anchor_score",
        "planner_lexical_match",
        "planner_lexical_direct_match",
        "exact_anchor_match",
        "exact_anchor_terms",
        "exact_anchor_fields",
        "word_map_score",
        "word_map_hint",
        "word_map_terms",
        "word_map_variant_terms",
        "word_map_neighbor_terms",
        "word_map_category_seed_terms",
        "low_frequency_match",
        "low_frequency_direct_match",
        "low_frequency_terms",
        "low_frequency_direct_terms",
        "low_frequency_category_terms",
        "low_frequency_sources",
        "rare_name_match",
        "rare_name_terms",
        "rare_name_sources",
        "entity_edge_match",
        "entity_edge_score",
        "entity_edge_subject",
        "entity_edge_relation",
        "entity_edge_object",
        "explicit_relation_edge_match",
        "explicit_relation_edge_confidence",
        "explicit_relation_edge_peer_bucket_id",
        "explicit_relation_edge_type",
        "explicit_relation_edge_focused",
        "fusion_mode",
        "fusion_score",
        "vector_norm",
        "keyword_norm",
        "dynamic_alpha",
        "dynamic_alpha_confidence",
        "metadata_adjustment",
        "cooldown_penalty",
        "matched_query_terms",
        "dynamic_anchor_plan",
        "distinctive_anchor_match",
        "distinctive_anchor_terms",
        "distinctive_anchor_missing_terms",
        "anchor_coverage",
        "category_overview_item",
        "category_overview_terms",
        "retrieval_alias_match",
        "retrieval_alias_score",
        "retrieval_alias_terms",
        "retrieval_alias_sources",
        "retrieval_alias_moment_ids",
        "retrieval_alias_bucket_count",
    )

    _BUCKET_RECALL_SIGNAL_FIELDS = (
        "evidence_labels",
        "hard_evidence_labels",
        "blocked_reason",
        "score",
        "semantic_score",
        "keyword_score",
        "rerank_score",
        "exact_anchor_score",
        "planner_lexical_match",
        "planner_lexical_direct_match",
        "exact_anchor_match",
        "exact_anchor_terms",
        "exact_anchor_fields",
        "word_map_score",
        "word_map_hint",
        "word_map_terms",
        "word_map_variant_terms",
        "word_map_neighbor_terms",
        "word_map_category_seed_terms",
        "low_frequency_match",
        "low_frequency_direct_match",
        "low_frequency_terms",
        "low_frequency_direct_terms",
        "low_frequency_category_terms",
        "low_frequency_sources",
        "rare_name_match",
        "rare_name_terms",
        "rare_name_sources",
        "entity_edge_match",
        "entity_edge_score",
        "entity_edge_subject",
        "entity_edge_relation",
        "entity_edge_object",
        "explicit_relation_edge_match",
        "explicit_relation_edge_confidence",
        "explicit_relation_edge_peer_bucket_id",
        "explicit_relation_edge_type",
        "explicit_relation_edge_focused",
        "fusion_mode",
        "fusion_score",
        "vector_norm",
        "keyword_norm",
        "dynamic_alpha",
        "dynamic_alpha_confidence",
        "metadata_adjustment",
        "cooldown_penalty",
        "admission_reason",
        "matched_query_terms",
        "recall_policy_debug",
        "dynamic_anchor_plan",
        "distinctive_anchor_match",
        "distinctive_anchor_terms",
        "distinctive_anchor_missing_terms",
        "anchor_coverage",
        "category_overview_item",
        "category_overview_terms",
        "retrieval_alias_match",
        "retrieval_alias_score",
        "retrieval_alias_terms",
        "retrieval_alias_sources",
        "retrieval_alias_moment_ids",
        "retrieval_alias_bucket_count",
        "semantic_rescue",
        "semantic_rescue_direct_span",
        "semantic_rescue_matched_axis",
        "semantic_rescue_no_diffusion",
    )

    @staticmethod
    def build_bucket_item(inputs: BucketCandidateItemInput) -> dict[str, Any]:
        fusion = inputs.fusion
        item = {
            "bucket": inputs.bucket,
            "score": fusion.score,
            "semantic_score": inputs.semantic_score,
            "keyword_score": inputs.keyword_score,
            "exact_anchor_score": inputs.exact_anchor_score,
            "exact_anchor_match": inputs.exact_anchor_match,
            "exact_anchor_terms": list(inputs.exact_anchor_terms),
            "exact_anchor_fields": list(inputs.exact_anchor_fields),
            "word_map_score": inputs.word_map_score,
            "word_map_hint": inputs.word_map_hint,
            "entity_edge_match": bool(inputs.entity_edge_score > 0),
            "entity_edge_score": inputs.entity_edge_score,
            "entity_edge_subject": inputs.entity_edge_subject,
            "entity_edge_relation": inputs.entity_edge_relation,
            "entity_edge_object": inputs.entity_edge_object,
            "word_map_terms": list(inputs.word_map_terms),
            "word_map_variant_terms": list(inputs.word_map_variant_terms),
            "word_map_neighbor_terms": list(inputs.word_map_neighbor_terms),
            "word_map_activation_terms": [
                dict(row)
                for row in inputs.word_map_activation_terms
                if isinstance(row, dict)
            ],
            "low_frequency_match": inputs.low_frequency_match,
            "low_frequency_direct_match": inputs.low_frequency_direct_match,
            "low_frequency_terms": list(inputs.low_frequency_terms),
            "low_frequency_direct_terms": list(inputs.low_frequency_direct_terms),
            "low_frequency_category_terms": list(inputs.low_frequency_category_terms),
            "low_frequency_sources": list(inputs.low_frequency_sources),
            "word_map_category_seed_terms": list(inputs.word_map_category_seed_terms),
            "rare_name_match": inputs.rare_name_match,
            "rare_name_terms": list(inputs.rare_name_terms),
            "rare_name_sources": list(inputs.rare_name_sources),
            "importance_score": inputs.importance_score,
            "freshness_score": inputs.freshness_score,
            "cooldown_multiplier": inputs.cooldown_multiplier,
            "fusion_mode": fusion.fusion_mode,
            "fusion_score": fusion.fusion_score,
            "vector_norm": fusion.vector_norm,
            "keyword_norm": fusion.keyword_norm,
            "dynamic_alpha": fusion.dynamic_alpha,
            "dynamic_alpha_confidence": fusion.dynamic_alpha_confidence,
            "metadata_adjustment": fusion.metadata_adjustment,
            "word_map_adjustment": fusion.word_map_adjustment,
            "cooldown_penalty": fusion.cooldown_penalty,
            "dynamic_alpha_debug": fusion.dynamic_alpha_debug,
            "planner_lexical_match": inputs.planner_lexical_match,
            "planner_lexical_direct_match": inputs.planner_lexical_direct_match,
            "planner_queries": [inputs.planner_query] if inputs.planner_query else [],
            "matched_query_terms": list(inputs.matched_query_terms),
        }
        item.update(dict(inputs.dynamic_anchor))
        return item

    @classmethod
    def bucket_recall_signal(cls, item: dict[str, Any]) -> dict[str, Any]:
        return {
            key: item.get(key)
            for key in cls._BUCKET_RECALL_SIGNAL_FIELDS
            if isinstance(item, dict) and key in item
        }

    @classmethod
    def decorate_moment(
        cls,
        moment: dict[str, Any],
        signal: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not isinstance(moment, dict) or not isinstance(signal, dict) or not signal:
            return moment
        enriched = dict(moment)
        for key in cls._MOMENT_SIGNAL_FIELDS:
            value = signal.get(key)
            if value is not None and enriched.get(key) is None:
                enriched[key] = value
        reason = str(signal.get("admission_reason") or "").strip()
        if reason and not enriched.get("_admission_reason"):
            enriched["_admission_reason"] = reason
        return enriched
