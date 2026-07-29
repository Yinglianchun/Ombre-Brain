from __future__ import annotations

import asyncio
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROUTE_SOURCE_SCHEMA_VERSION = 1
ROUTE_INDEX_SCHEMA_VERSION = 1
ROUTE_ACTIONS = frozenset({"skip", "recall"})


def _resolve_path(value: Any, *, default: Path, base_dir: Path) -> Path:
    text = str(value or "").strip()
    path = Path(text) if text else default
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _embedding_profile(embedding_engine: Any) -> dict[str, Any]:
    return {
        "model": str(getattr(embedding_engine, "model", "") or "").strip(),
        "query_instruction": str(
            getattr(embedding_engine, "query_instruction", "") or ""
        ).strip(),
        "max_chars": int(getattr(embedding_engine, "max_chars", 0) or 0),
    }


def load_route_source(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("route_source_not_object")
    if int(raw.get("schema_version") or 0) != ROUTE_SOURCE_SCHEMA_VERSION:
        raise ValueError("route_source_schema_mismatch")
    routes = raw.get("routes")
    if not isinstance(routes, list) or not routes:
        raise ValueError("route_source_routes_missing")

    seen_names: set[str] = set()
    seen_texts: set[str] = set()
    normalized_routes: list[dict[str, Any]] = []
    for route in routes:
        if not isinstance(route, dict):
            raise ValueError("route_source_route_not_object")
        name = str(route.get("name") or "").strip()
        action = str(route.get("action") or "").strip().lower()
        if not name or name in seen_names:
            raise ValueError("route_source_name_invalid")
        if action not in ROUTE_ACTIONS:
            raise ValueError(f"route_source_action_invalid:{name}")
        seen_names.add(name)

        utterances = route.get("utterances")
        enabled = bool(route.get("enabled", True))
        if not isinstance(utterances, list):
            raise ValueError(f"route_source_utterances_missing:{name}")
        if enabled and not utterances:
            raise ValueError(f"route_source_utterances_missing:{name}")
        normalized_utterances: list[dict[str, str]] = []
        for item in utterances:
            if isinstance(item, str):
                text = item.strip()
                source = "seed"
            elif isinstance(item, dict):
                text = str(item.get("text") or "").strip()
                source = str(item.get("source") or "seed").strip()
            else:
                raise ValueError(f"route_source_utterance_invalid:{name}")
            text_key = " ".join(text.split()).lower()
            if not text_key or text_key in seen_texts:
                raise ValueError(f"route_source_utterance_duplicate:{name}")
            seen_texts.add(text_key)
            normalized_utterances.append({"text": text, "source": source or "seed"})

        threshold = route.get("threshold")
        normalized_routes.append(
            {
                "name": name,
                "action": action,
                "enabled": enabled,
                "threshold": float(threshold) if threshold is not None else None,
                "utterances": normalized_utterances,
            }
        )
    return {
        "schema_version": ROUTE_SOURCE_SCHEMA_VERSION,
        "dataset_version": int(raw.get("dataset_version") or 1),
        "routes": normalized_routes,
    }


async def build_route_index(
    *,
    source_path: Path,
    output_path: Path,
    embedding_engine: Any,
    concurrency: int = 3,
) -> dict[str, Any]:
    if not getattr(embedding_engine, "enabled", False):
        raise RuntimeError("embedding_engine_disabled")
    embed_query = getattr(embedding_engine, "embed_query", None)
    if not callable(embed_query):
        raise RuntimeError("embedding_engine_missing_embed_query")

    source = load_route_source(source_path)
    semaphore = asyncio.Semaphore(max(1, min(8, int(concurrency or 1))))

    async def embed(text: str) -> list[float]:
        async with semaphore:
            vector = await embed_query(text)
        if not isinstance(vector, list) or not vector:
            raise RuntimeError(f"route_embedding_empty:{text[:40]}")
        return [float(value) for value in vector]

    pending: list[tuple[dict[str, Any], dict[str, str], Any]] = []
    active_routes = [route for route in source["routes"] if route.get("enabled", True)]
    for route in active_routes:
        for utterance in route["utterances"]:
            pending.append((route, utterance, asyncio.create_task(embed(utterance["text"]))))
    if not pending:
        raise RuntimeError("route_source_active_utterances_missing")

    dimensions: set[int] = set()
    route_rows: dict[str, dict[str, Any]] = {
        route["name"]: {
            "name": route["name"],
            "action": route["action"],
            "threshold": route.get("threshold"),
            "utterances": [],
        }
        for route in active_routes
    }
    for route, utterance, task in pending:
        vector = await task
        dimensions.add(len(vector))
        route_rows[route["name"]]["utterances"].append(
            {
                "text": utterance["text"],
                "source": utterance["source"],
                "embedding": vector,
            }
        )
    if len(dimensions) != 1:
        raise RuntimeError("route_embedding_dimension_mismatch")

    payload = {
        "schema_version": ROUTE_INDEX_SCHEMA_VERSION,
        "dataset_version": source["dataset_version"],
        "source_sha256": _source_sha256(source_path),
        "embedding": {
            **_embedding_profile(embedding_engine),
            "dimension": next(iter(dimensions)),
            "kind": "query",
        },
        "routes": list(route_rows.values()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    return payload


class SemanticRecallRouter:
    """Shadow-only semantic decision layer for long-term memory recall."""

    def __init__(self, config: dict[str, Any], embedding_engine: Any):
        self.embedding_engine = embedding_engine
        gateway_cfg = config.get("gateway", {})
        gateway_cfg = gateway_cfg if isinstance(gateway_cfg, dict) else {}
        cfg = gateway_cfg.get("semantic_recall_router", {})
        cfg = cfg if isinstance(cfg, dict) else {}
        project_dir = Path(__file__).resolve().parent.parent
        state_dir = Path(str(config.get("buckets_dir") or project_dir)).resolve()
        self.shadow_enabled = bool(cfg.get("shadow_enabled", False))
        self.min_score = max(0.0, min(1.0, float(cfg.get("min_score", 0.72))))
        self.min_margin = max(0.0, min(1.0, float(cfg.get("min_margin", 0.04))))
        self.aggregation_top_k = max(
            1,
            min(8, int(cfg.get("aggregation_top_k", 1))),
        )
        self.source_path = _resolve_path(
            cfg.get("routes_path"),
            default=project_dir / "resources" / "semantic_recall_routes.json",
            base_dir=project_dir,
        )
        self.index_path = _resolve_path(
            cfg.get("index_path"),
            default=state_dir / "semantic_recall_routes.v1.json",
            base_dir=state_dir,
        )
        self._loaded_index: dict[str, Any] | None = None
        self._loaded_index_mtime_ns = -1

    def debug_base(self, query: str) -> dict[str, Any]:
        return {
            "enabled": self.shadow_enabled,
            "shadow_only": True,
            "called": False,
            "query_preview": str(query or "")[:500],
            "route": "",
            "route_action": "recall",
            "recommended_action": "recall",
            "would_skip": False,
            "confidence": 0.0,
            "margin": 0.0,
            "threshold": self.min_score,
            "reason": "disabled" if not self.shadow_enabled else "",
            "model": str(getattr(self.embedding_engine, "model", "") or ""),
            "dimension": 0,
            "query_vector_ready": False,
            "scores": [],
            "errors": [],
        }

    async def route(self, query: str) -> dict[str, Any]:
        debug, _query_vector = await self.route_with_vector(query)
        return debug

    async def route_with_vector(
        self,
        query: str,
    ) -> tuple[dict[str, Any], list[float] | None]:
        debug = self.debug_base(query)
        text = str(query or "").strip()
        if not self.shadow_enabled:
            return debug, None
        if not text:
            debug["reason"] = "empty_query"
            return debug, None
        if not getattr(self.embedding_engine, "enabled", False):
            debug["reason"] = "embedding_disabled"
            return debug, None

        index, error = self._load_index()
        if error:
            debug["reason"] = error
            debug["errors"].append(error)
            return debug, None

        embed_query = getattr(self.embedding_engine, "embed_query", None)
        if not callable(embed_query):
            debug["reason"] = "embedding_engine_missing_embed_query"
            debug["errors"].append(debug["reason"])
            return debug, None
        debug["called"] = True
        try:
            query_vector = await embed_query(text)
        except Exception as exc:
            debug["reason"] = f"query_embedding_failed:{type(exc).__name__}"
            debug["errors"].append(debug["reason"])
            return debug, None
        if not isinstance(query_vector, list) or not query_vector:
            debug["reason"] = "query_embedding_empty"
            debug["errors"].append(debug["reason"])
            return debug, None

        expected_dimension = int((index.get("embedding") or {}).get("dimension") or 0)
        debug["dimension"] = len(query_vector)
        if expected_dimension <= 0 or len(query_vector) != expected_dimension:
            debug["reason"] = "query_embedding_dimension_mismatch"
            debug["errors"].append(debug["reason"])
            return debug, None

        debug["query_vector_ready"] = True
        scored_routes = self._score_routes(index, query_vector)
        debug["scores"] = [
            {
                "route": row["name"],
                "action": row["action"],
                "score": round(row["score"], 6),
                "threshold": round(row["threshold"], 6),
                "top_examples": row["top_examples"],
            }
            for row in scored_routes
        ]
        if not scored_routes:
            debug["reason"] = "route_index_empty"
            debug["errors"].append(debug["reason"])
            return debug, query_vector

        winner = scored_routes[0]
        opposite_score = max(
            (
                row["score"]
                for row in scored_routes[1:]
                if row["action"] != winner["action"]
            ),
            default=0.0,
        )
        margin = winner["score"] - opposite_score
        debug.update(
            route=winner["name"],
            route_action=winner["action"],
            confidence=round(winner["score"], 6),
            margin=round(margin, 6),
            threshold=round(winner["threshold"], 6),
        )
        if winner["action"] != "skip":
            debug["reason"] = "recall_route_won"
            return debug, query_vector
        if winner["score"] < winner["threshold"]:
            debug["reason"] = "below_threshold"
            return debug, query_vector
        if margin < self.min_margin:
            debug["reason"] = "insufficient_margin"
            return debug, query_vector

        debug["recommended_action"] = "skip"
        debug["would_skip"] = True
        debug["reason"] = "matched_skip_route"
        return debug, query_vector

    def _load_index(self) -> tuple[dict[str, Any], str]:
        if not self.source_path.exists():
            return {}, "route_source_missing"
        if not self.index_path.exists():
            return {}, "route_index_missing"
        try:
            mtime_ns = self.index_path.stat().st_mtime_ns
            if self._loaded_index is None or mtime_ns != self._loaded_index_mtime_ns:
                raw = json.loads(self.index_path.read_text(encoding="utf-8"))
                error = self._validate_index(raw)
                if error:
                    return {}, error
                self._loaded_index = raw
                self._loaded_index_mtime_ns = mtime_ns
            return self._loaded_index, ""
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return {}, "route_index_invalid"

    def _validate_index(self, raw: Any) -> str:
        if not isinstance(raw, dict):
            return "route_index_not_object"
        if int(raw.get("schema_version") or 0) != ROUTE_INDEX_SCHEMA_VERSION:
            return "route_index_schema_mismatch"
        if str(raw.get("source_sha256") or "") != _source_sha256(self.source_path):
            return "route_index_stale"
        profile = raw.get("embedding")
        if not isinstance(profile, dict):
            return "route_index_embedding_profile_missing"
        current_profile = _embedding_profile(self.embedding_engine)
        for key in ("model", "query_instruction", "max_chars"):
            if profile.get(key) != current_profile.get(key):
                return f"route_index_embedding_{key}_mismatch"
        dimension = int(profile.get("dimension") or 0)
        if dimension <= 0:
            return "route_index_dimension_invalid"
        routes = raw.get("routes")
        if not isinstance(routes, list) or not routes:
            return "route_index_routes_missing"
        for route in routes:
            if not isinstance(route, dict):
                return "route_index_route_invalid"
            if str(route.get("action") or "") not in ROUTE_ACTIONS:
                return "route_index_action_invalid"
            utterances = route.get("utterances")
            if not isinstance(utterances, list) or not utterances:
                return "route_index_utterances_missing"
            for utterance in utterances:
                vector = utterance.get("embedding") if isinstance(utterance, dict) else None
                if not isinstance(vector, list) or len(vector) != dimension:
                    return "route_index_vector_dimension_mismatch"
        return ""

    def _score_routes(
        self,
        index: dict[str, Any],
        query_vector: list[float],
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for route in index.get("routes") or []:
            example_scores = [
                (
                    _cosine_similarity(query_vector, utterance["embedding"]),
                    str(utterance.get("text") or ""),
                )
                for utterance in route.get("utterances") or []
            ]
            example_scores.sort(key=lambda row: row[0], reverse=True)
            top_rows = example_scores[: self.aggregation_top_k]
            if not top_rows:
                continue
            threshold = route.get("threshold")
            output.append(
                {
                    "name": str(route.get("name") or ""),
                    "action": str(route.get("action") or "recall"),
                    "score": sum(row[0] for row in top_rows) / len(top_rows),
                    "threshold": (
                        max(0.0, min(1.0, float(threshold)))
                        if threshold is not None
                        else self.min_score
                    ),
                    "top_examples": [
                        {"text": text, "score": round(score, 6)}
                        for score, text in top_rows
                    ],
                }
            )
        output.sort(key=lambda row: (-row["score"], row["name"]))
        return output
