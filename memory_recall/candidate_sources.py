from __future__ import annotations

import asyncio
import logging
from typing import Any


logger = logging.getLogger(__name__)


class MemoryCandidateSourceCollector:
    """Read direct-recall candidates from stores without owning recall policy."""

    def __init__(
        self,
        *,
        bucket_manager: Any,
        embedding_engine: Any,
        moment_store: Any,
        entity_edge_store: Any,
        identity: dict[str, Any],
        dynamic_top_k: int,
        semantic_candidate_top_k: int,
        embedding_query_timeout_seconds: float,
    ) -> None:
        self.bucket_manager = bucket_manager
        self.embedding_engine = embedding_engine
        self.moment_store = moment_store
        self.entity_edge_store = entity_edge_store
        self.identity = dict(identity or {})
        self.dynamic_top_k = max(0, int(dynamic_top_k))
        self.semantic_candidate_top_k = max(0, int(semantic_candidate_top_k))
        self.embedding_query_timeout_seconds = max(
            0.0,
            float(embedding_query_timeout_seconds),
        )

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clamp(value: Any, lower: float = 0.0, upper: float = 1.0) -> float:
        return max(lower, min(upper, float(value)))

    @staticmethod
    def _debug_str_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            stripped = value.strip()
            return [stripped] if stripped else []
        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value if str(item).strip()]
        return [str(value)]

    def keyword_candidates(self, query: str, buckets: list[dict]) -> dict[str, float]:
        if hasattr(self.bucket_manager, "calc_topic_scores"):
            raw_scores = self.bucket_manager.calc_topic_scores(query, buckets)
            scored = [
                (str(bucket_id), self._clamp(score))
                for bucket_id, score in raw_scores.items()
                if self._clamp(score) > 0
            ]
        else:
            scored = []
            for bucket in buckets:
                keyword_score = self._clamp(
                    self.bucket_manager._calc_topic_score(query, bucket)
                )
                if keyword_score > 0:
                    scored.append((str(bucket["id"]), keyword_score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return {
            bucket_id: score
            for bucket_id, score in scored[: self.dynamic_top_k]
        }

    async def semantic_candidates(
        self,
        query: str,
        eligible_ids: set[str],
    ) -> dict[str, float]:
        if not getattr(self.embedding_engine, "enabled", False):
            return {}

        try:
            search = self.embedding_engine.search_similar(
                query,
                top_k=self.semantic_candidate_top_k,
            )
            if self.embedding_query_timeout_seconds > 0:
                results = await asyncio.wait_for(
                    search,
                    timeout=self.embedding_query_timeout_seconds,
                )
            else:
                results = await search
        except asyncio.TimeoutError:
            logger.warning(
                "Memory candidate semantic search timed out | query_chars=%s timeout_seconds=%.2f",
                len(str(query or "")),
                self.embedding_query_timeout_seconds,
            )
            return {}
        except Exception as exc:
            logger.warning("Memory candidate semantic search failed: %s", exc)
            return {}

        semantic_scores: dict[str, float] = {}
        for bucket_id, similarity in results:
            if bucket_id not in eligible_ids:
                continue
            semantic_scores[str(bucket_id)] = self._clamp(similarity)
        return semantic_scores

    def retrieval_alias_hits(
        self,
        query: str,
        eligible_ids: set[str],
    ) -> list[dict[str, Any]]:
        search = getattr(self.moment_store, "search_retrieval_aliases", None)
        if not callable(search) or not str(query or "").strip() or not eligible_ids:
            return []
        try:
            rows = search(
                query,
                limit=max(
                    self.semantic_candidate_top_k,
                    self.dynamic_top_k * 4,
                    20,
                ),
            )
        except Exception as exc:
            logger.warning("Memory candidate retrieval alias lookup failed: %s", exc)
            return []

        output: list[dict[str, Any]] = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            bucket_id = str(row.get("bucket_id") or "").strip()
            if not bucket_id or bucket_id not in eligible_ids:
                continue
            output.append(
                {
                    "bucket_id": bucket_id,
                    "moment_id": str(row.get("moment_id") or ""),
                    "alias_text": str(row.get("alias_text") or ""),
                    "source": str(row.get("source") or ""),
                    "bucket_count": max(1, int(row.get("bucket_count") or 1)),
                    "score": self._clamp(self._safe_float(row.get("score"), 0.0)),
                    "matched_terms": self._debug_str_list(row.get("matched_terms")),
                }
            )
        return output

    def entity_edge_boosts(
        self,
        query: str,
        candidate_ids: set[str],
    ) -> dict[str, dict[str, Any]]:
        if not query or not candidate_ids:
            return {}
        try:
            return self.entity_edge_store.match_query(
                query,
                self.identity,
                bucket_ids=candidate_ids,
                min_score=0.48,
            )
        except Exception as exc:
            logger.warning("Memory candidate entity edge lookup failed: %s", exc)
            return {}
