from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4


ROUTE_SOURCE_SCHEMA_VERSION = 1
ROUTE_INDEX_SCHEMA_VERSION = 1
ROUTE_ACTIONS = frozenset({"skip", "recall"})
ROUTE_EXAMPLE_ROLES = frozenset({"typical", "boundary"})
ROUTE_EXAMPLE_ORIGINS = frozenset(
    {"manual", "online_false_positive", "online_false_negative", "import"}
)
ROUTE_EXAMPLE_STATUSES = frozenset({"draft", "published", "retired"})
ROUTE_PUBLISH_CONFIRMATION = "PUBLISH_SEMANTIC_ROUTES"


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


def _legacy_example_metadata(source: str) -> tuple[str, str]:
    if source == "hard_negative":
        return "boundary", "import"
    if source == "historical_false_positive":
        return "typical", "online_false_positive"
    if source == "historical_false_negative":
        return "typical", "online_false_negative"
    return "typical", "import"


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
                role, origin = _legacy_example_metadata(source)
                status = "published"
            elif isinstance(item, dict):
                text = str(item.get("text") or "").strip()
                source = str(item.get("source") or "seed").strip()
                legacy_role, legacy_origin = _legacy_example_metadata(source)
                role = str(item.get("role") or legacy_role).strip().lower()
                origin = str(item.get("origin") or legacy_origin).strip().lower()
                status = str(item.get("status") or "published").strip().lower()
            else:
                raise ValueError(f"route_source_utterance_invalid:{name}")
            if role not in ROUTE_EXAMPLE_ROLES:
                raise ValueError(f"route_source_utterance_role_invalid:{name}")
            if origin not in ROUTE_EXAMPLE_ORIGINS:
                raise ValueError(f"route_source_utterance_origin_invalid:{name}")
            if status not in ROUTE_EXAMPLE_STATUSES:
                raise ValueError(f"route_source_utterance_status_invalid:{name}")
            text_key = " ".join(text.split()).lower()
            if not text_key or text_key in seen_texts:
                raise ValueError(f"route_source_utterance_duplicate:{name}")
            seen_texts.add(text_key)
            normalized_utterances.append(
                {
                    "text": text,
                    "source": source or "seed",
                    "role": role,
                    "origin": origin,
                    "status": status,
                }
            )

        threshold = route.get("threshold")
        normalized_routes.append(
            {
                "name": name,
                "label": str(route.get("label") or "").strip(),
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

    pending: list[tuple[str, dict[str, Any], dict[str, str], Any]] = []
    active_routes = [route for route in source["routes"] if route.get("enabled", True)]
    route_centers_by_name: dict[str, list[dict[str, str]]] = {}
    for route in active_routes:
        route_centers = [
            utterance
            for utterance in route["utterances"]
            if utterance.get("role") == "typical"
            and utterance.get("status") == "published"
        ]
        if not route_centers:
            raise RuntimeError(f"route_source_active_center_missing:{route['name']}")
        route_centers_by_name[route["name"]] = route_centers
    for route in active_routes:
        for utterance in route_centers_by_name[route["name"]]:
            pending.append(
                ("center", route, utterance, asyncio.create_task(embed(utterance["text"])))
            )
        for utterance in route["utterances"]:
            if (
                utterance.get("role") == "boundary"
                and utterance.get("status") == "published"
            ):
                pending.append(
                    (
                        "boundary",
                        route,
                        utterance,
                        asyncio.create_task(embed(utterance["text"])),
                    )
                )
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
    boundary_rows: list[dict[str, Any]] = []
    for kind, route, utterance, task in pending:
        vector = await task
        dimensions.add(len(vector))
        row = {
            "text": utterance["text"],
            "source": utterance["source"],
            "origin": utterance["origin"],
            "embedding": vector,
        }
        if kind == "center":
            route_rows[route["name"]]["utterances"].append(row)
        else:
            boundary_rows.append(
                {
                    "route": route["name"],
                    "action": route["action"],
                    **row,
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
        "boundaries": boundary_rows,
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
    """Semantic decision layer for long-term memory recall."""

    def __init__(self, config: dict[str, Any], embedding_engine: Any):
        self.embedding_engine = embedding_engine
        gateway_cfg = config.get("gateway", {})
        gateway_cfg = gateway_cfg if isinstance(gateway_cfg, dict) else {}
        cfg = gateway_cfg.get("semantic_recall_router", {})
        cfg = cfg if isinstance(cfg, dict) else {}
        project_dir = Path(__file__).resolve().parent.parent
        state_dir = Path(str(config.get("buckets_dir") or project_dir)).resolve()
        configured_mode = str(cfg.get("mode") or "").strip().lower()
        if not configured_mode:
            configured_mode = "shadow" if bool(cfg.get("shadow_enabled", False)) else "off"
        if configured_mode not in {"off", "shadow", "active"}:
            configured_mode = "off"
        self.mode = configured_mode
        self.enabled = self.mode in {"shadow", "active"}
        self.active = self.mode == "active"
        # Compatibility for callers and old config readers.
        self.shadow_enabled = self.enabled
        self.min_score = max(0.0, min(1.0, float(cfg.get("min_score", 0.72))))
        self.min_margin = max(0.0, min(1.0, float(cfg.get("min_margin", 0.04))))
        self.aggregation_top_k = max(
            1,
            min(8, int(cfg.get("aggregation_top_k", 1))),
        )
        self.boundary_veto_enabled = bool(cfg.get("boundary_veto_enabled", True))
        self.boundary_veto_min_score = max(
            0.0,
            min(1.0, float(cfg.get("boundary_veto_min_score", 0.72))),
        )
        self.boundary_veto_max_deficit = max(
            0.0,
            min(1.0, float(cfg.get("boundary_veto_max_deficit", 0.0))),
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
        self.publish_dir = _resolve_path(
            cfg.get("publish_dir"),
            default=state_dir / "semantic_recall_routes",
            base_dir=state_dir,
        )
        self.active_manifest_path = self.publish_dir / "active.json"
        self._publish_lock = asyncio.Lock()
        self._loaded_index: dict[str, Any] | None = None
        self._loaded_index_signature: tuple[str, int, int] | None = None

    def debug_base(self, query: str) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "active": self.active,
            "shadow_only": not self.active,
            "called": False,
            "query_preview": str(query or "")[:500],
            "route": "",
            "route_action": "recall",
            "recommended_action": "recall",
            "would_skip": False,
            "applied_action": "recall",
            "skip_applied": False,
            "confidence": 0.0,
            "margin": 0.0,
            "threshold": self.min_score,
            "reason": "disabled" if not self.enabled else "",
            "model": str(getattr(self.embedding_engine, "model", "") or ""),
            "dimension": 0,
            "query_vector_ready": False,
            "scores": [],
            "boundary_veto": {
                "enabled": self.boundary_veto_enabled,
                "applied": False,
                "threshold": self.boundary_veto_min_score,
                "max_deficit": self.boundary_veto_max_deficit,
                "candidate": None,
            },
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
        if not self.enabled:
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
        boundary = self._best_boundary_veto(index, query_vector, winner)
        if boundary is not None:
            debug["boundary_veto"]["candidate"] = boundary
            if boundary["passes_threshold"] and boundary["within_deficit"]:
                debug["boundary_veto"]["applied"] = True
                debug["reason"] = "boundary_veto"
                return debug, query_vector

        debug["recommended_action"] = "skip"
        debug["would_skip"] = True
        debug["threshold_met"] = winner["score"] >= winner["threshold"]
        debug["margin_met"] = margin >= self.min_margin
        debug["reason"] = "matched_skip_route"
        return debug, query_vector

    def should_apply_skip(self, debug: dict[str, Any] | None) -> bool:
        return bool(
            self.active
            and isinstance(debug, dict)
            and debug.get("would_skip")
        )

    def _active_paths(self) -> tuple[Path, Path, int, str]:
        if not self.active_manifest_path.exists():
            return self.source_path, self.index_path, -1, ""
        try:
            manifest = json.loads(self.active_manifest_path.read_text(encoding="utf-8"))
            generation = str(manifest.get("generation") or "").strip()
            if not generation or Path(generation).name != generation:
                return self.source_path, self.index_path, -1, "route_publish_manifest_invalid"
            generation_dir = (self.publish_dir / generation).resolve()
            if generation_dir.parent != self.publish_dir.resolve():
                return self.source_path, self.index_path, -1, "route_publish_manifest_invalid"
            return (
                generation_dir / "source.json",
                generation_dir / "index.json",
                self.active_manifest_path.stat().st_mtime_ns,
                "",
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return self.source_path, self.index_path, -1, "route_publish_manifest_invalid"

    def _load_index(self) -> tuple[dict[str, Any], str]:
        source_path, index_path, manifest_mtime_ns, manifest_error = self._active_paths()
        if manifest_error:
            return {}, manifest_error
        if not source_path.exists():
            return {}, "route_source_missing"
        if not index_path.exists():
            return {}, "route_index_missing"
        try:
            signature = (str(index_path), index_path.stat().st_mtime_ns, manifest_mtime_ns)
            if self._loaded_index is None or signature != self._loaded_index_signature:
                raw = json.loads(index_path.read_text(encoding="utf-8"))
                error = self._validate_index(raw, source_path=source_path)
                if error:
                    return {}, error
                self._loaded_index = raw
                self._loaded_index_signature = signature
            return self._loaded_index, ""
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return {}, "route_index_invalid"

    def _validate_index(self, raw: Any, *, source_path: Path | None = None) -> str:
        source_path = source_path or self.source_path
        if not isinstance(raw, dict):
            return "route_index_not_object"
        if int(raw.get("schema_version") or 0) != ROUTE_INDEX_SCHEMA_VERSION:
            return "route_index_schema_mismatch"
        if str(raw.get("source_sha256") or "") != _source_sha256(source_path):
            return "route_index_stale"
        source = load_route_source(source_path)
        if int(raw.get("dataset_version") or 0) != int(source["dataset_version"]):
            return "route_index_dataset_version_mismatch"
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
        boundaries = raw.get("boundaries", [])
        if not isinstance(boundaries, list):
            return "route_index_boundaries_invalid"
        for boundary in boundaries:
            if not isinstance(boundary, dict):
                return "route_index_boundary_invalid"
            if str(boundary.get("action") or "") not in ROUTE_ACTIONS:
                return "route_index_boundary_action_invalid"
            vector = boundary.get("embedding")
            if not isinstance(vector, list) or len(vector) != dimension:
                return "route_index_boundary_vector_dimension_mismatch"
        return ""

    def dataset_payload(self) -> dict[str, Any]:
        source_path, index_path, _manifest_mtime_ns, manifest_error = self._active_paths()
        if manifest_error:
            raise RuntimeError(manifest_error)
        source = load_route_source(source_path)
        index_error = ""
        index: dict[str, Any] = {}
        if index_path.exists():
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
                index_error = self._validate_index(index, source_path=source_path)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                index_error = "route_index_invalid"
        else:
            index_error = "route_index_missing"
        boundary_count = sum(
            1
            for route in source["routes"]
            for item in route["utterances"]
            if item.get("role") == "boundary" and item.get("status") == "published"
        )
        indexed_boundary_count = len(index.get("boundaries") or [])
        boundary_index_ready = (
            not index_error
            and "boundaries" in index
            and indexed_boundary_count == boundary_count
        )
        return {
            "ok": True,
            "schema_version": source["schema_version"],
            "dataset_version": source["dataset_version"],
            "deployment_state": "production",
            "source_kind": "published" if self.active_manifest_path.exists() else "seed",
            "routes": source["routes"],
            "embedding": index.get("embedding") or _embedding_profile(self.embedding_engine),
            "index_ready": not index_error,
            "index_error": index_error,
            "boundary_example_count": boundary_count,
            "indexed_boundary_example_count": indexed_boundary_count,
            "boundary_index_ready": boundary_index_ready,
        }

    @staticmethod
    def _published_source_payload(routes: Any, dataset_version: int) -> dict[str, Any]:
        if not isinstance(routes, list):
            raise ValueError("route_publish_routes_missing")
        published_routes = []
        for route in routes:
            if not isinstance(route, dict):
                raise ValueError("route_source_route_not_object")
            published_items = []
            for item in route.get("utterances") or []:
                if isinstance(item, str):
                    published_items.append({"text": item, "status": "published"})
                    continue
                if not isinstance(item, dict):
                    raise ValueError("route_source_utterance_invalid")
                normalized = dict(item)
                if str(normalized.get("status") or "draft").strip().lower() != "retired":
                    normalized["status"] = "published"
                published_items.append(normalized)
            published_routes.append(
                {
                    "name": route.get("name"),
                    "label": route.get("label"),
                    "action": route.get("action"),
                    "enabled": route.get("enabled", True),
                    "threshold": route.get("threshold"),
                    "utterances": published_items,
                }
            )
        return {
            "schema_version": ROUTE_SOURCE_SCHEMA_VERSION,
            "dataset_version": dataset_version,
            "routes": published_routes,
        }

    async def publish_dataset(
        self,
        *,
        routes: Any,
        expected_dataset_version: int,
        confirmation: str,
        concurrency: int = 3,
    ) -> dict[str, Any]:
        if confirmation != ROUTE_PUBLISH_CONFIRMATION:
            raise ValueError("route_publish_confirmation_required")
        async with self._publish_lock:
            current = self.dataset_payload()
            current_version = int(current["dataset_version"])
            if int(expected_dataset_version) != current_version:
                raise ValueError(f"route_publish_version_conflict:{current_version}")

            next_version = current_version + 1
            source_payload = self._published_source_payload(routes, next_version)
            self.publish_dir.mkdir(parents=True, exist_ok=True)
            staging_name = f".staging-{uuid4().hex}"
            staging_dir = self.publish_dir / staging_name
            staging_dir.mkdir()
            source_path = staging_dir / "source.json"
            index_path = staging_dir / "index.json"
            generation_dir: Path | None = None
            activated = False
            try:
                source_path.write_text(
                    json.dumps(source_payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                index = await build_route_index(
                    source_path=source_path,
                    output_path=index_path,
                    embedding_engine=self.embedding_engine,
                    concurrency=concurrency,
                )
                index_error = self._validate_index(index, source_path=source_path)
                if index_error:
                    raise RuntimeError(index_error)

                generation = f"v{next_version:06d}-{_source_sha256(source_path)[:12]}-{uuid4().hex[:8]}"
                generation_dir = self.publish_dir / generation
                staging_dir.replace(generation_dir)
                manifest = {
                    "schema_version": 1,
                    "dataset_version": next_version,
                    "generation": generation,
                    "source_sha256": index["source_sha256"],
                    "published_at": datetime.now(timezone.utc).isoformat(),
                    "embedding": index["embedding"],
                    "route_count": len(index["routes"]),
                    "center_example_count": sum(
                        len(route.get("utterances") or []) for route in index["routes"]
                    ),
                    "boundary_example_count": len(index.get("boundaries") or []),
                }
                manifest_tmp = self.publish_dir / f".active-{uuid4().hex}.tmp"
                manifest_tmp.write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                manifest_tmp.replace(self.active_manifest_path)
                activated = True
                self._loaded_index = None
                self._loaded_index_signature = None
                result = self.dataset_payload()
                result["published"] = manifest
                return result
            except Exception:
                if staging_dir.exists():
                    shutil.rmtree(staging_dir, ignore_errors=True)
                if not activated and generation_dir is not None and generation_dir.exists():
                    shutil.rmtree(generation_dir, ignore_errors=True)
                raise

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

    def _best_boundary_veto(
        self,
        index: dict[str, Any],
        query_vector: list[float],
        winner: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not self.boundary_veto_enabled or winner.get("action") != "skip":
            return None
        candidates: list[tuple[float, dict[str, Any]]] = []
        for boundary in index.get("boundaries") or []:
            if boundary.get("action") == winner.get("action"):
                continue
            score = _cosine_similarity(query_vector, boundary.get("embedding") or [])
            candidates.append((score, boundary))
        if not candidates:
            return None
        score, boundary = max(candidates, key=lambda row: row[0])
        winner_score = float(winner.get("score") or 0.0)
        deficit = max(0.0, winner_score - score)
        return {
            "route": str(boundary.get("route") or ""),
            "action": str(boundary.get("action") or "recall"),
            "text": str(boundary.get("text") or ""),
            "score": round(score, 6),
            "passes_threshold": score >= self.boundary_veto_min_score,
            "beats_skip": score >= winner_score,
            "deficit": round(deficit, 6),
            "within_deficit": deficit <= self.boundary_veto_max_deficit,
        }
