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

from self_anchor import is_self_anchor_bucket
from utils import normalize_scene_cues


CUE_SOURCE_SCHEMA_VERSION = 1
CUE_INDEX_SCHEMA_VERSION = 1
CUE_INDEX_REBUILD_CONFIRMATION = "BUILD_CUE_SEMANTIC_SHADOW"


def scene_cues_are_reviewed(metadata: dict[str, Any]) -> bool:
    """Mirror the authored-cue review contract used by Gateway lexical recall."""
    if str(metadata.get("scene_cues_reviewed_at") or "").strip():
        return True
    if str(metadata.get("last_edit_source") or "") != "edit_scene":
        return False
    history = metadata.get("scene_revision_history")
    if not isinstance(history, list) or not history:
        return False
    revisions = [
        normalize_scene_cues(item.get("cues"))
        for item in history
        if isinstance(item, dict)
    ]
    revisions.append(normalize_scene_cues(metadata.get("scene_cues")))
    return any(
        previous != current
        for previous, current in zip(revisions, revisions[1:])
    )


def scene_cue_hash(cues: Any) -> str:
    normalized = normalize_scene_cues(cues)
    payload = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def scene_cue_indexability_reason(bucket: dict[str, Any]) -> str:
    if not isinstance(bucket, dict) or not str(bucket.get("id") or "").strip():
        return "invalid_bucket"
    if is_self_anchor_bucket(bucket):
        return "self_anchor"
    metadata = bucket.get("metadata") if isinstance(bucket.get("metadata"), dict) else {}
    if str(metadata.get("memory_value_source") or "") != "authored_scene":
        return "not_authored_scene"
    if str(metadata.get("type") or "") in {"archived", "feel"}:
        return "inactive"
    if (
        metadata.get("resolved")
        or metadata.get("digested")
        or metadata.get("deprecated")
        or metadata.get("active") is False
    ):
        return "inactive"
    cues = normalize_scene_cues(metadata.get("scene_cues"))
    if not cues:
        return "missing_cues"
    migrated = bool(
        str(metadata.get("source") or "") == "scene_migration"
        or metadata.get("migration_source_bucket_id")
        or metadata.get("migration_source_tags")
    )
    if migrated and not scene_cues_are_reviewed(metadata):
        return "unreviewed_migration"
    return "eligible"


def scene_is_cue_indexable(bucket: dict[str, Any]) -> bool:
    return scene_cue_indexability_reason(bucket) == "eligible"


def cue_index_selection_summary(buckets: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: dict[str, int] = {}
    for bucket in buckets or []:
        reason = scene_cue_indexability_reason(bucket)
        reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "total_buckets": len(buckets or []),
        "eligible_scenes": reasons.get("eligible", 0),
        "excluded_scenes": sum(
            count
            for reason, count in reasons.items()
            if reason not in {"eligible", "not_authored_scene", "invalid_bucket"}
        ),
        "reasons": dict(sorted(reasons.items())),
    }


def build_cue_source(
    buckets: list[dict[str, Any]],
    *,
    dataset_version: int,
) -> dict[str, Any]:
    scenes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for bucket in buckets or []:
        if not scene_is_cue_indexable(bucket):
            continue
        scene_id = str(bucket.get("id") or "").strip()
        if scene_id in seen:
            raise ValueError(f"cue_source_duplicate_scene:{scene_id}")
        seen.add(scene_id)
        metadata = bucket.get("metadata") if isinstance(bucket.get("metadata"), dict) else {}
        cues = normalize_scene_cues(metadata.get("scene_cues"))
        scenes.append(
            {
                "scene_id": scene_id,
                "title": str(metadata.get("name") or bucket.get("name") or "").strip(),
                "cues": cues,
                "cue_sha256": scene_cue_hash(cues),
            }
        )
    scenes.sort(key=lambda item: item["scene_id"])
    return {
        "schema_version": CUE_SOURCE_SCHEMA_VERSION,
        "dataset_version": max(1, int(dataset_version)),
        "scenes": scenes,
    }


def _embedding_profile(embedding_engine: Any) -> dict[str, Any]:
    return {
        "model": str(getattr(embedding_engine, "model", "") or "").strip(),
        "query_instruction": str(
            getattr(embedding_engine, "query_instruction", "") or ""
        ).strip(),
        "document_instruction": str(
            getattr(embedding_engine, "document_instruction", "") or ""
        ).strip(),
        "max_chars": int(getattr(embedding_engine, "max_chars", 0) or 0),
    }


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


async def build_cue_index(
    *,
    source_path: Path,
    output_path: Path,
    embedding_engine: Any,
    concurrency: int = 3,
) -> dict[str, Any]:
    if not getattr(embedding_engine, "enabled", False):
        raise RuntimeError("embedding_engine_disabled")
    embed_document = getattr(embedding_engine, "embed_document", None)
    if not callable(embed_document):
        raise RuntimeError("embedding_engine_missing_embed_document")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if int(source.get("schema_version") or 0) != CUE_SOURCE_SCHEMA_VERSION:
        raise RuntimeError("cue_source_schema_mismatch")
    scenes = source.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise RuntimeError("cue_source_scenes_missing")

    semaphore = asyncio.Semaphore(max(1, min(8, int(concurrency or 1))))

    async def embed(text: str) -> list[float]:
        async with semaphore:
            vector = await embed_document(text)
        if not isinstance(vector, list) or not vector:
            raise RuntimeError(f"cue_embedding_empty:{text[:40]}")
        return [float(value) for value in vector]

    pending: list[tuple[dict[str, Any], int, str, Any]] = []
    for scene in scenes:
        scene_id = str(scene.get("scene_id") or "").strip()
        cues = normalize_scene_cues(scene.get("cues"))
        cue_hash = str(scene.get("cue_sha256") or "")
        if not scene_id or not cues or cue_hash != scene_cue_hash(cues):
            raise RuntimeError(f"cue_source_scene_invalid:{scene_id or 'unknown'}")
        for ordinal, cue in enumerate(cues):
            pending.append((scene, ordinal, cue, asyncio.create_task(embed(cue))))

    rows: list[dict[str, Any]] = []
    dimensions: set[int] = set()
    vectors = await asyncio.gather(*(task for *_row, task in pending))
    for (scene, ordinal, cue, _task), vector in zip(pending, vectors):
        dimensions.add(len(vector))
        rows.append(
            {
                "scene_id": str(scene["scene_id"]),
                "title": str(scene.get("title") or ""),
                "cue_ordinal": ordinal,
                "cue": cue,
                "cue_sha256": str(scene["cue_sha256"]),
                "embedding": vector,
            }
        )
    if len(dimensions) != 1:
        raise RuntimeError("cue_embedding_dimension_mismatch")
    payload = {
        "schema_version": CUE_INDEX_SCHEMA_VERSION,
        "dataset_version": int(source.get("dataset_version") or 0),
        "source_sha256": _source_sha256(source_path),
        "embedding": {
            **_embedding_profile(embedding_engine),
            "dimension": next(iter(dimensions)),
            "kind": "document_per_cue",
        },
        "cues": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    return payload


class CueSemanticIndex:
    """Versioned cue-vector index used only by explicit simulation shadow."""

    def __init__(self, config: dict[str, Any], embedding_engine: Any):
        project_dir = Path(__file__).resolve().parent.parent
        buckets_dir = Path(str(config.get("buckets_dir") or project_dir)).resolve()
        gateway_cfg = config.get("gateway", {})
        gateway_cfg = gateway_cfg if isinstance(gateway_cfg, dict) else {}
        cfg = gateway_cfg.get("cue_semantic_shadow", {})
        cfg = cfg if isinstance(cfg, dict) else {}
        configured_dir = str(cfg.get("publish_dir") or "").strip()
        self.publish_dir = (
            Path(configured_dir).expanduser().resolve()
            if configured_dir
            else buckets_dir / "cue_semantic_shadow"
        )
        self.active_manifest_path = self.publish_dir / "active.json"
        self.embedding_engine = embedding_engine
        self._publish_lock = asyncio.Lock()

    def _active_paths(self) -> tuple[Path, Path, dict[str, Any]]:
        if not self.active_manifest_path.exists():
            raise RuntimeError("cue_semantic_index_not_built")
        try:
            manifest = json.loads(self.active_manifest_path.read_text(encoding="utf-8"))
            generation = str(manifest.get("generation") or "").strip()
            if not generation or Path(generation).name != generation:
                raise RuntimeError("cue_semantic_manifest_invalid")
            generation_dir = (self.publish_dir / generation).resolve()
            if generation_dir.parent != self.publish_dir.resolve():
                raise RuntimeError("cue_semantic_manifest_invalid")
            return generation_dir / "source.json", generation_dir / "index.json", manifest
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("cue_semantic_manifest_invalid") from exc

    def status(self) -> dict[str, Any]:
        try:
            source_path, index_path, manifest = self._active_paths()
            source = json.loads(source_path.read_text(encoding="utf-8"))
            index = json.loads(index_path.read_text(encoding="utf-8"))
            self._validate_active(source_path, source, index)
            return {
                "status": "available",
                "dataset_version": int(source.get("dataset_version") or 0),
                "scene_count": len(source.get("scenes") or []),
                "cue_count": len(index.get("cues") or []),
                "embedding": dict(index.get("embedding") or {}),
                "published": manifest,
            }
        except RuntimeError as exc:
            return {"status": "unavailable", "reason": str(exc)}
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return {"status": "unavailable", "reason": "cue_semantic_index_invalid"}

    def current_dataset_version(self) -> int:
        """Read the publish CAS version without requiring profile compatibility."""
        if not self.active_manifest_path.exists():
            return 0
        _source_path, _index_path, manifest = self._active_paths()
        try:
            version = int(manifest.get("dataset_version") or 0)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("cue_semantic_manifest_invalid") from exc
        if version <= 0:
            raise RuntimeError("cue_semantic_manifest_invalid")
        return version

    def _validate_active(
        self,
        source_path: Path,
        source: dict[str, Any],
        index: dict[str, Any],
    ) -> None:
        if int(source.get("schema_version") or 0) != CUE_SOURCE_SCHEMA_VERSION:
            raise RuntimeError("cue_source_schema_mismatch")
        if int(index.get("schema_version") or 0) != CUE_INDEX_SCHEMA_VERSION:
            raise RuntimeError("cue_index_schema_mismatch")
        if int(index.get("dataset_version") or 0) != int(source.get("dataset_version") or -1):
            raise RuntimeError("cue_index_dataset_version_mismatch")
        if str(index.get("source_sha256") or "") != _source_sha256(source_path):
            raise RuntimeError("cue_index_source_stale")
        profile = index.get("embedding") if isinstance(index.get("embedding"), dict) else {}
        current_profile = _embedding_profile(self.embedding_engine)
        for key, value in current_profile.items():
            if profile.get(key) != value:
                raise RuntimeError("cue_index_embedding_profile_mismatch")
        if int(profile.get("dimension") or 0) <= 0:
            raise RuntimeError("cue_index_embedding_dimension_missing")

    async def rebuild(
        self,
        *,
        buckets: list[dict[str, Any]],
        expected_dataset_version: int,
        confirmation: str,
        concurrency: int = 3,
    ) -> dict[str, Any]:
        if confirmation != CUE_INDEX_REBUILD_CONFIRMATION:
            raise ValueError("cue_semantic_rebuild_confirmation_required")
        async with self._publish_lock:
            current_version = self.current_dataset_version()
            if int(expected_dataset_version) != current_version:
                raise ValueError(f"cue_semantic_rebuild_version_conflict:{current_version}")
            next_version = current_version + 1
            source = build_cue_source(buckets, dataset_version=next_version)
            if not source["scenes"]:
                raise RuntimeError("cue_source_scenes_missing")
            self.publish_dir.mkdir(parents=True, exist_ok=True)
            staging_dir = self.publish_dir / f".staging-{uuid4().hex}"
            staging_dir.mkdir()
            generation_dir: Path | None = None
            activated = False
            try:
                source_path = staging_dir / "source.json"
                index_path = staging_dir / "index.json"
                source_path.write_text(
                    json.dumps(source, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                index = await build_cue_index(
                    source_path=source_path,
                    output_path=index_path,
                    embedding_engine=self.embedding_engine,
                    concurrency=concurrency,
                )
                generation = f"v{next_version:06d}-{uuid4().hex[:12]}"
                generation_dir = self.publish_dir / generation
                staging_dir.replace(generation_dir)
                manifest = {
                    "schema_version": 1,
                    "dataset_version": next_version,
                    "generation": generation,
                    "published_at": datetime.now(timezone.utc).isoformat(),
                    "scene_count": len(source["scenes"]),
                    "cue_count": len(index["cues"]),
                    "mode": "simulation_shadow",
                }
                manifest_tmp = self.publish_dir / f".active-{uuid4().hex}.tmp"
                manifest_tmp.write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                manifest_tmp.replace(self.active_manifest_path)
                activated = True
                return self.status()
            except Exception:
                if staging_dir.exists():
                    shutil.rmtree(staging_dir, ignore_errors=True)
                if not activated and generation_dir is not None and generation_dir.exists():
                    shutil.rmtree(generation_dir, ignore_errors=True)
                raise

    def search_by_vector(
        self,
        query_embedding: list[float],
        *,
        current_cue_hashes: dict[str, str],
        top_k: int,
    ) -> dict[str, Any]:
        if not query_embedding:
            return {"status": "unavailable", "reason": "query_embedding_unavailable", "matches": []}
        try:
            source_path, index_path, manifest = self._active_paths()
            source = json.loads(source_path.read_text(encoding="utf-8"))
            index = json.loads(index_path.read_text(encoding="utf-8"))
            self._validate_active(source_path, source, index)
            dimension = int((index.get("embedding") or {}).get("dimension") or 0)
            if len(query_embedding) != dimension:
                raise RuntimeError("cue_query_embedding_dimension_mismatch")
            best: dict[str, dict[str, Any]] = {}
            stale_scene_ids: set[str] = set()
            for row in index.get("cues") or []:
                scene_id = str(row.get("scene_id") or "")
                expected_hash = str(row.get("cue_sha256") or "")
                if not scene_id or current_cue_hashes.get(scene_id) != expected_hash:
                    if scene_id:
                        stale_scene_ids.add(scene_id)
                    continue
                vector = row.get("embedding")
                if not isinstance(vector, list) or len(vector) != dimension:
                    continue
                score = _cosine_similarity(query_embedding, [float(value) for value in vector])
                current = best.get(scene_id)
                if current is None or score > float(current["score"]):
                    best[scene_id] = {
                        "scene_id": scene_id,
                        "title": str(row.get("title") or ""),
                        "score": score,
                        "matched_cues": [str(row.get("cue") or "")],
                    }
            matches = sorted(best.values(), key=lambda item: item["score"], reverse=True)
            return {
                "status": "available",
                "dataset_version": int(source.get("dataset_version") or 0),
                "profile": dict(index.get("embedding") or {}),
                "stale_scene_count": len(stale_scene_ids),
                "matches": matches[: max(1, min(50, int(top_k or 1)))],
                "published": manifest,
            }
        except RuntimeError as exc:
            return {"status": "unavailable", "reason": str(exc), "matches": []}
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return {"status": "unavailable", "reason": "cue_semantic_index_invalid", "matches": []}
