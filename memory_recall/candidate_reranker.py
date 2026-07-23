from __future__ import annotations

from typing import Any, Callable


class MemoryCandidateReranker:
    """Apply the configured reranker without owning recall admission policy.

    Gateway/runtime callers keep their rank keys and candidate gates.  This
    collaborator owns only candidate slicing, document submission, score
    fusion, and deterministic head/tail reconstruction.
    """

    def __init__(self, engine_getter: Callable[[], Any]) -> None:
        self._engine_getter = engine_getter

    async def rerank_moments(
        self,
        query: str,
        candidates: list[dict],
        *,
        rank_key: Callable[[str, dict], tuple],
    ) -> list[dict]:
        engine = self._engine_getter()
        if not candidates or not getattr(engine, "enabled", False):
            return candidates
        candidate_limit = min(
            len(candidates),
            max(1, int(getattr(engine, "candidate_limit", 20) or 20)),
        )
        head = candidates[:candidate_limit]
        tail = candidates[candidate_limit:]
        documents = [self.moment_document(moment) for moment in head]
        results = await engine.rerank(query, documents, top_n=len(head))
        if not results:
            return candidates

        by_index = {result.index: result.score for result in results}
        weight = max(0.0, min(1.0, float(getattr(engine, "score_weight", 0.65))))
        reranked = []
        for index, moment in enumerate(head):
            item = dict(moment)
            rerank_score = by_index.get(index)
            base_score = self._safe_float(item.get("score"), 0.0)
            if rerank_score is None:
                item["rerank_score"] = None
                item["combined_score"] = base_score
            else:
                item["rerank_score"] = round(rerank_score, 4)
                item["combined_score"] = round(
                    base_score * (1.0 - weight) + rerank_score * weight,
                    4,
                )
                item["score"] = item["combined_score"]
            reranked.append(item)
        reranked.sort(
            key=lambda item: (
                rank_key(query, item)[0],
                item.get("rerank_score") is None,
                -self._safe_float(item.get("combined_score", item.get("score")), 0.0),
                -self._safe_float(item.get("score"), 0.0),
            )
        )
        return reranked + tail

    async def rerank_buckets(
        self,
        query: str,
        scored_candidates: list[dict],
        *,
        priority_key: Callable[[str, dict], tuple],
        reranked_key: Callable[[str, dict], tuple],
        document_for_item: Callable[[dict], str],
    ) -> list[dict]:
        engine = self._engine_getter()
        if not scored_candidates or not getattr(engine, "enabled", False):
            return scored_candidates
        candidate_limit = min(
            len(scored_candidates),
            max(1, int(getattr(engine, "candidate_limit", 20) or 20)),
        )
        ranked_pool = sorted(
            enumerate(scored_candidates),
            key=lambda pair: priority_key(query, pair[1]),
        )
        head_indices = {index for index, _item in ranked_pool[:candidate_limit]}
        head = [
            item
            for index, item in enumerate(scored_candidates)
            if index in head_indices
        ]
        tail = [
            item
            for index, item in enumerate(scored_candidates)
            if index not in head_indices
        ]
        documents = [document_for_item(item) for item in head]
        results = await engine.rerank(query, documents, top_n=len(head))
        if not results:
            return scored_candidates

        by_index = {result.index: result.score for result in results}
        weight = max(0.0, min(1.0, float(getattr(engine, "score_weight", 0.65))))
        reranked = []
        for index, item in enumerate(head):
            new_item = dict(item)
            rerank_score = by_index.get(index)
            if rerank_score is None:
                new_item["rerank_score"] = None
                new_item["combined_score"] = item["score"]
            else:
                new_item["rerank_score"] = round(rerank_score, 4)
                new_item["combined_score"] = round(
                    item["score"] * (1.0 - weight) + rerank_score * weight,
                    4,
                )
                new_item["score"] = new_item["combined_score"]
            reranked.append(new_item)
        reranked.sort(key=lambda item: reranked_key(query, item))
        return reranked + tail

    @classmethod
    def moment_document(cls, moment: dict) -> str:
        meta = moment.get("metadata", {}) if isinstance(moment.get("metadata"), dict) else {}
        fields = [
            f"title: {meta.get('bucket_name') or moment.get('bucket_id') or ''}",
            f"section: {moment.get('section') or ''}",
            f"domain: {' '.join(str(item) for item in meta.get('bucket_domain', []) or [])}",
            f"tags: {' '.join(str(item) for item in meta.get('bucket_tags', []) or [])}",
            f"summary: {meta.get('annotation_summary') or meta.get('summary') or ''}",
            f"facets: {cls._format_annotation_facets(meta)}",
            f"evidence: {cls._format_evidence_spans(meta)}",
            f"text: {moment.get('text') or ''}",
        ]
        return "\n".join(fields)[:4000]

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
