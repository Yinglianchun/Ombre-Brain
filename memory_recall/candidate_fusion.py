from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class CandidateFusionConfig:
    mode: str
    semantic_weight: float
    keyword_weight: float
    importance_weight: float
    freshness_weight: float
    word_map_hint_weight: float
    first_card_min_score: float
    dynamic_alpha_conf_lo: float
    dynamic_alpha_conf_hi: float
    dynamic_alpha_margin_ref: float
    dynamic_alpha_min: float
    dynamic_alpha_max: float


@dataclass(frozen=True)
class CandidateFusionBatch:
    config: CandidateFusionConfig
    semantic_norms: Mapping[str, float]
    keyword_norms: Mapping[str, float]
    alpha: float
    alpha_debug: Mapping[str, Any]


@dataclass(frozen=True)
class CandidateFusionInputs:
    bucket_id: str
    semantic_score: float
    keyword_score: float
    word_map_score: float
    entity_edge_score: float
    importance_score: float
    freshness_score: float
    relevance_score: float
    cooldown_multiplier: float
    anchored: bool = False


@dataclass(frozen=True)
class CandidateFusionReceipt:
    score: float
    fusion_mode: str
    fusion_score: float
    vector_norm: float
    keyword_norm: float
    dynamic_alpha: float | None
    dynamic_alpha_confidence: float | None
    metadata_adjustment: float
    word_map_adjustment: float
    cooldown_penalty: float
    dynamic_alpha_debug: Mapping[str, Any]


class MemoryCandidateFusionScorer:
    """Fuse already measured candidate signals without reading runtime state."""

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clamp(value: Any, lower: float = 0.0, upper: float = 1.0) -> float:
        return max(lower, min(upper, float(value)))

    @classmethod
    def config(
        cls,
        *,
        mode: str,
        semantic_weight: float,
        keyword_weight: float,
        importance_weight: float,
        freshness_weight: float,
        word_map_hint_weight: float,
        first_card_min_score: float,
        high_confidence_semantic_score: float,
        recall_thresholds: Mapping[str, Any] | None = None,
    ) -> CandidateFusionConfig:
        thresholds = recall_thresholds if isinstance(recall_thresholds, Mapping) else {}
        conf_lo = cls._clamp(
            cls._safe_float(
                thresholds.get(
                    "dynamic_alpha_conf_lo",
                    thresholds.get("vector_min_score", 0.50),
                ),
                0.50,
            )
        )
        conf_hi = cls._clamp(
            cls._safe_float(
                thresholds.get(
                    "dynamic_alpha_conf_hi",
                    high_confidence_semantic_score,
                ),
                high_confidence_semantic_score,
            )
        )
        if conf_hi <= conf_lo:
            conf_hi = min(1.0, conf_lo + 0.01)
        margin_ref = max(
            0.001,
            cls._safe_float(thresholds.get("dynamic_alpha_margin_ref"), 0.08),
        )
        alpha_min = cls._clamp(
            cls._safe_float(thresholds.get("dynamic_alpha_min"), 0.35)
        )
        alpha_max = cls._clamp(
            cls._safe_float(thresholds.get("dynamic_alpha_max"), 0.85)
        )
        if alpha_max < alpha_min:
            alpha_min, alpha_max = alpha_max, alpha_min
        return CandidateFusionConfig(
            mode=str(mode or "dynamic"),
            semantic_weight=float(semantic_weight),
            keyword_weight=float(keyword_weight),
            importance_weight=float(importance_weight),
            freshness_weight=float(freshness_weight),
            word_map_hint_weight=cls._clamp(word_map_hint_weight),
            first_card_min_score=float(first_card_min_score),
            dynamic_alpha_conf_lo=conf_lo,
            dynamic_alpha_conf_hi=conf_hi,
            dynamic_alpha_margin_ref=margin_ref,
            dynamic_alpha_min=alpha_min,
            dynamic_alpha_max=alpha_max,
        )

    @classmethod
    def normalized_score_map(cls, scores: Mapping[str, float]) -> dict[str, float]:
        cleaned = {
            str(key): max(0.0, cls._safe_float(value, 0.0))
            for key, value in (scores or {}).items()
            if str(key or "").strip()
        }
        if not cleaned:
            return {}
        max_score = max(cleaned.values())
        if max_score <= 0:
            return {key: 0.0 for key in cleaned}
        return {
            key: cls._clamp(value / max_score)
            for key, value in cleaned.items()
        }

    @classmethod
    def prepare_batch(
        cls,
        *,
        config: CandidateFusionConfig,
        semantic_scores: Mapping[str, float],
        keyword_scores: Mapping[str, float],
        exact_scores: Mapping[str, float],
        lexical_ids: set[str],
        retrieval_alias_scores: Mapping[str, float],
    ) -> CandidateFusionBatch:
        keyword_basis = {
            str(bucket_id): cls._clamp(cls._safe_float(score, 0.0))
            for bucket_id, score in (keyword_scores or {}).items()
        }
        for bucket_id, score in (exact_scores or {}).items():
            key = str(bucket_id)
            keyword_basis[key] = max(
                keyword_basis.get(key, 0.0),
                cls._clamp(cls._safe_float(score, 0.0)),
            )
        for bucket_id in lexical_ids:
            key = str(bucket_id)
            keyword_basis[key] = max(keyword_basis.get(key, 0.0), 1.0)
        for bucket_id, score in retrieval_alias_scores.items():
            key = str(bucket_id)
            keyword_basis[key] = max(
                keyword_basis.get(key, 0.0),
                cls._clamp(cls._safe_float(score, 0.0)),
            )

        sorted_scores = sorted(
            (
                cls._clamp(cls._safe_float(score, 0.0))
                for score in (semantic_scores or {}).values()
            ),
            reverse=True,
        )
        top1 = sorted_scores[0] if sorted_scores else 0.0
        top2 = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        reference_scores = sorted_scores[1:6]
        reference_score = (
            sum(reference_scores) / len(reference_scores)
            if reference_scores
            else 0.0
        )
        margin = max(0.0, top1 - reference_score)
        confidence_component = cls._clamp(
            (top1 - config.dynamic_alpha_conf_lo)
            / (config.dynamic_alpha_conf_hi - config.dynamic_alpha_conf_lo)
        )
        margin_component = cls._clamp(margin / config.dynamic_alpha_margin_ref)
        confidence = cls._clamp(confidence_component * margin_component)
        alpha = round(
            config.dynamic_alpha_min
            + (config.dynamic_alpha_max - config.dynamic_alpha_min) * confidence,
            4,
        )
        alpha_debug = {
            "alpha": alpha,
            "confidence": round(confidence, 4),
            "top1": round(top1, 4),
            "top2": round(top2, 4),
            "reference_score": round(reference_score, 4),
            "reference_count": len(reference_scores),
            "margin": round(margin, 4),
            "conf_lo": round(config.dynamic_alpha_conf_lo, 4),
            "conf_hi": round(config.dynamic_alpha_conf_hi, 4),
            "margin_ref": config.dynamic_alpha_margin_ref,
            "alpha_min": config.dynamic_alpha_min,
            "alpha_max": config.dynamic_alpha_max,
        }
        return CandidateFusionBatch(
            config=config,
            semantic_norms=cls.normalized_score_map(semantic_scores),
            keyword_norms=cls.normalized_score_map(keyword_basis),
            alpha=alpha,
            alpha_debug=alpha_debug,
        )

    @classmethod
    def score(
        cls,
        batch: CandidateFusionBatch,
        inputs: CandidateFusionInputs,
    ) -> CandidateFusionReceipt:
        config = batch.config
        vector_norm = cls._clamp(batch.semantic_norms.get(inputs.bucket_id, 0.0))
        keyword_norm = cls._clamp(batch.keyword_norms.get(inputs.bucket_id, 0.0))
        word_map_adjustment = round(
            min(
                config.word_map_hint_weight,
                inputs.word_map_score * config.word_map_hint_weight,
            ),
            4,
        )
        metadata_adjustment = 0.0
        cooldown_penalty = 0.0
        if config.mode == "dynamic":
            fusion_score = cls._clamp(
                (
                    batch.alpha * vector_norm
                    + (1.0 - batch.alpha) * keyword_norm
                )
                * inputs.relevance_score
            )
            metadata_adjustment = round(
                0.02 * inputs.importance_score + 0.02 * inputs.freshness_score,
                4,
            )
            cooldown_penalty = round(
                (1.0 - cls._clamp(inputs.cooldown_multiplier)) * 0.03,
                4,
            )
            final_score = round(
                cls._clamp(
                    fusion_score
                    + word_map_adjustment
                    + metadata_adjustment
                    - cooldown_penalty
                ),
                4,
            )
        else:
            fusion_score = (
                inputs.semantic_score * config.semantic_weight
                + inputs.keyword_score * config.keyword_weight
                + word_map_adjustment
                + inputs.entity_edge_score * 0.08
                + inputs.importance_score * config.importance_weight
                + inputs.freshness_score * config.freshness_weight
            ) * inputs.relevance_score
            final_score = round(fusion_score * inputs.cooldown_multiplier, 4)
        if inputs.entity_edge_score > 0:
            final_score = round(
                cls._clamp(
                    final_score
                    + min(0.08, inputs.entity_edge_score * 0.08)
                ),
                4,
            )
        if inputs.anchored:
            final_score = max(final_score, config.first_card_min_score)

        dynamic = config.mode == "dynamic"
        return CandidateFusionReceipt(
            score=final_score,
            fusion_mode=config.mode,
            fusion_score=round(fusion_score, 4),
            vector_norm=round(vector_norm, 4),
            keyword_norm=round(keyword_norm, 4),
            dynamic_alpha=batch.alpha if dynamic else None,
            dynamic_alpha_confidence=(
                cls._safe_float(batch.alpha_debug.get("confidence"), 0.0)
                if dynamic
                else None
            ),
            metadata_adjustment=metadata_adjustment,
            word_map_adjustment=word_map_adjustment,
            cooldown_penalty=cooldown_penalty,
            dynamic_alpha_debug=batch.alpha_debug if dynamic else {},
        )
