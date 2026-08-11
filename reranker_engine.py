from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger("ombre_brain.reranker")


@dataclass(frozen=True)
class RerankResult:
    index: int
    score: float


class RerankerEngine:
    """Small SiliconFlow-compatible rerank client."""

    def __init__(self, config: dict):
        config = config or {}
        embed_cfg = config.get("embedding", {}) or {}
        rerank_cfg = config.get("reranker", {}) or {}
        dehy_cfg = config.get("dehydration", {}) or {}

        self.model = str(rerank_cfg.get("model") or "Qwen/Qwen3-Reranker-4B")
        self.base_url = str(
            rerank_cfg.get("base_url")
            or embed_cfg.get("base_url")
            or dehy_cfg.get("base_url")
            or ""
        ).rstrip("/")
        self.api_key = str(
            rerank_cfg.get("api_key")
            or embed_cfg.get("api_key")
            or dehy_cfg.get("api_key")
            or ""
        )
        self.enabled = bool(self.api_key and self.base_url) and _bool_value(
            rerank_cfg.get("enabled", True)
        )
        self.simulation_shadow_enabled = _bool_value(
            rerank_cfg.get("simulation_shadow_enabled", False),
            False,
        )
        self.shadow_ready = bool(
            self.api_key and self.base_url and self.simulation_shadow_enabled
        )
        self.timeout = _float_between(rerank_cfg.get("timeout_seconds", 12), 12, 1, 120)
        self.candidate_limit = _int_between(rerank_cfg.get("candidate_limit", 5), 5, 1, 100)
        self.score_weight = _float_between(rerank_cfg.get("score_weight", 0.65), 0.65, 0.0, 1.0)

    async def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[RerankResult]:
        return await self._rerank(
            query,
            documents,
            top_n=top_n,
            enabled=self.enabled,
        )

    async def rerank_shadow(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """Run the separately gated explicit-simulation shadow client."""

        return await self._rerank(
            query,
            documents,
            top_n=top_n,
            enabled=self.shadow_ready,
        )

    async def _rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int | None,
        enabled: bool,
    ) -> list[RerankResult]:
        if not enabled or not query or not documents:
            return []
        endpoint = f"{self.base_url}/rerank"
        payload: dict[str, Any] = {
            "model": self.model,
            "query": str(query),
            "documents": [str(document or "") for document in documents],
            "return_documents": False,
        }
        if top_n is not None:
            payload["top_n"] = max(1, min(int(top_n), len(documents)))

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                body = response.json()
        except Exception as exc:
            logger.warning("Reranker request failed: %s", exc)
            return []

        results = []
        for item in body.get("results", []) if isinstance(body, dict) else []:
            try:
                index = int(item.get("index"))
                score = float(item.get("relevance_score", 0.0))
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(documents):
                results.append(RerankResult(index=index, score=max(0.0, min(1.0, score))))
        results.sort(key=lambda item: item.score, reverse=True)
        return results


def _bool_value(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _int_between(value: Any, default: int, min_value: int, max_value: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(min_value, min(max_value, number))


def _float_between(value: Any, default: float, min_value: float, max_value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(min_value, min(max_value, number))
