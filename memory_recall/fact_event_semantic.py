from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from fact_events import FactEventStore


INDEX_SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _profile(engine: Any) -> dict[str, Any]:
    return {
        "model": str(getattr(engine, "model", "") or ""),
        "document_instruction": str(
            getattr(engine, "document_instruction", "") or ""
        ),
        "max_chars": int(getattr(engine, "max_chars", 0) or 0),
    }


def _source_hash(item: dict[str, Any], profile: dict[str, Any]) -> str:
    payload = {
        "schema": INDEX_SCHEMA_VERSION,
        "item_id": str(item.get("item_id") or ""),
        "item_type": str(item.get("item_type") or ""),
        "fingerprint": str(item.get("fingerprint") or ""),
        "body": str(item.get("body") or ""),
        "profile": profile,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


class FactEventSemanticIndex:
    """Rebuildable body-only shadow index for canonical Fact/Event rows."""

    def __init__(self, config: dict[str, Any], embedding_engine: Any):
        state_dir = str(
            config.get("state_dir")
            or os.path.join(
                os.path.dirname(os.path.abspath(config.get("buckets_dir", "buckets"))),
                "state",
            )
        )
        gateway = config.get("gateway") if isinstance(config.get("gateway"), dict) else {}
        self.enabled = bool(gateway.get("fact_event_recall_shadow_enabled", False))
        self.db_path = os.path.join(state_dir, "fact_event_embeddings.sqlite")
        self.embedding_engine = embedding_engine

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _connect_readonly(self) -> sqlite3.Connection:
        conn = sqlite3.connect(f"{Path(self.db_path).resolve().as_uri()}?mode=ro", uri=True, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def read_availability(self) -> dict[str, Any]:
        if not self.enabled:
            return {"status": "unavailable", "reason": "index_disabled"}
        if not os.path.exists(self.db_path):
            return {"status": "unavailable", "reason": "index_missing"}
        try:
            with closing(self._connect_readonly()) as conn:
                conn.execute("SELECT 1 FROM fact_event_embeddings LIMIT 1").fetchone()
        except sqlite3.Error as exc:
            return {"status": "unavailable", "reason": "index_unreadable", "error": str(exc)}
        return {"status": "ok"}

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS fact_event_embeddings (
                    item_id TEXT PRIMARY KEY,
                    item_type TEXT NOT NULL CHECK(item_type IN ('fact', 'event')),
                    source_hash TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    importance INTEGER NOT NULL,
                    local_date TEXT NOT NULL,
                    local_start_time TEXT NOT NULL,
                    covered_by_scene_id TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_fact_event_embeddings_kind
                ON fact_event_embeddings(item_type, importance, local_date);
                """
            )

    @staticmethod
    def _active_items(store: FactEventStore) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        offset = 0
        while True:
            page = store.list(status="active", limit=500, offset=offset)
            items = list(page.get("items") or [])
            output.extend(items)
            offset += len(items)
            if not items or offset >= int(page.get("count") or 0):
                return output

    async def sync(
        self,
        store: FactEventStore,
        *,
        dry_run: bool = False,
        refresh_all: bool = False,
    ) -> dict[str, Any]:
        items = self._active_items(store)
        profile = _profile(self.embedding_engine)
        existing: dict[str, sqlite3.Row] = {}
        if os.path.exists(self.db_path):
            with closing(self._connect()) as conn:
                existing = {
                    str(row["item_id"]): row
                    for row in conn.execute("SELECT * FROM fact_event_embeddings")
                }
        desired_ids = {str(item.get("item_id") or "") for item in items}
        stale_ids = sorted(set(existing) - desired_ids)
        pending = [
            item
            for item in items
            if refresh_all
            or str(item.get("item_id") or "") not in existing
            or str(existing[str(item["item_id"])]["source_hash"])
            != _source_hash(item, profile)
        ]
        if dry_run:
            return {
                "status": "dry_run",
                "active_items": len(items),
                "to_embed": len(pending),
                "stale_rows": len(stale_ids),
                "memory_kinds": {
                    kind: sum(1 for item in items if item.get("item_type") == kind)
                    for kind in ("fact", "event")
                },
            }
        if not getattr(self.embedding_engine, "enabled", False):
            raise RuntimeError("embedding_engine_disabled")
        self._init_db()
        embedded = 0
        failed = 0
        with closing(self._connect()) as conn:
            for item in items:
                item_id = str(item.get("item_id") or "")
                source_hash = _source_hash(item, profile)
                current = existing.get(item_id)
                vector: list[float] | None = None
                if (
                    not refresh_all
                    and current is not None
                    and str(current["source_hash"]) == source_hash
                ):
                    try:
                        vector = [float(value) for value in json.loads(current["embedding"])]
                    except (TypeError, ValueError, json.JSONDecodeError):
                        vector = None
                if vector is None:
                    vector = await self.embedding_engine.embed_document(
                        str(item.get("body") or "")
                    )
                    if not vector:
                        failed += 1
                        conn.execute(
                            "DELETE FROM fact_event_embeddings WHERE item_id=?",
                            (item_id,),
                        )
                        continue
                    embedded += 1
                conn.execute(
                    """
                    INSERT OR REPLACE INTO fact_event_embeddings(
                        item_id, item_type, source_hash, embedding, model, dimension,
                        importance, local_date, local_start_time,
                        covered_by_scene_id, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item_id,
                        str(item.get("item_type") or ""),
                        source_hash,
                        json.dumps(vector),
                        str(profile.get("model") or ""),
                        len(vector),
                        int(item.get("importance") or 1),
                        str(item.get("local_date") or ""),
                        str(item.get("local_start_time") or ""),
                        str(item.get("covered_by_scene_id") or ""),
                        _now(),
                    ),
                )
            if stale_ids:
                placeholders = ",".join("?" for _ in stale_ids)
                conn.execute(
                    f"DELETE FROM fact_event_embeddings WHERE item_id IN ({placeholders})",
                    stale_ids,
                )
            conn.commit()
        return {
            "status": "ok" if not failed else "partial",
            "active_items": len(items),
            "embedded": embedded,
            "reused": len(items) - embedded - failed,
            "failed": failed,
            "removed": len(stale_ids),
        }

    def search_by_embedding(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 8,
        memory_kinds: Iterable[str] = ("fact", "event"),
        min_importance: int = 1,
        min_importance_by_kind: dict[str, int] | None = None,
        allowed_memory_ids: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"status": "disabled", "matches": []}
        if not query_embedding:
            return {"status": "unavailable", "reason": "query_embedding_empty", "matches": []}
        availability = self.read_availability()
        if availability.get("status") != "ok":
            return {**availability, "matches": []}
        kinds = {str(value).strip().lower() for value in memory_kinds}
        kinds &= {"fact", "event"}
        if not kinds:
            return {"status": "ok", "matches": []}
        default_min_importance = max(1, int(min_importance or 1))
        importance_floor = {
            kind: max(
                1,
                int((min_importance_by_kind or {}).get(kind, default_min_importance) or 1),
            )
            for kind in kinds
        }
        allowed_ids = (
            {
                str(value or "").strip()
                for value in allowed_memory_ids
                if str(value or "").strip()
            }
            if allowed_memory_ids is not None
            else None
        )
        if allowed_ids is not None and not allowed_ids:
            return {
                "status": "ok",
                "candidate_count": 0,
                "indexed_memory_ids": [],
                "matches": [],
            }
        model = str(getattr(self.embedding_engine, "model", "") or "")
        with closing(self._connect_readonly()) as conn:
            if allowed_ids is None:
                rows = conn.execute("SELECT * FROM fact_event_embeddings").fetchall()
            else:
                rows = []
                ordered_ids = sorted(allowed_ids)
                for offset in range(0, len(ordered_ids), 900):
                    chunk = ordered_ids[offset : offset + 900]
                    placeholders = ",".join("?" for _ in chunk)
                    rows.extend(
                        conn.execute(
                            f"SELECT * FROM fact_event_embeddings WHERE item_id IN ({placeholders})",
                            chunk,
                        ).fetchall()
                    )
        matches: list[dict[str, Any]] = []
        for row in rows:
            item_type = str(row["item_type"])
            if item_type not in kinds or str(row["model"]) != model:
                continue
            if int(row["importance"] or 0) < importance_floor[item_type]:
                continue
            try:
                vector = [float(value) for value in json.loads(row["embedding"])]
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if int(row["dimension"] or 0) != len(query_embedding):
                continue
            matches.append(
                {
                    "memory_id": str(row["item_id"]),
                    "memory_kind": str(row["item_type"]),
                    "score": round(_cosine(query_embedding, vector), 4),
                    "importance": int(row["importance"] or 1),
                    "local_date": str(row["local_date"] or ""),
                    "local_start_time": str(row["local_start_time"] or ""),
                    "covered_by_scene_id": str(row["covered_by_scene_id"] or ""),
                }
            )
        matches.sort(key=lambda item: (-float(item["score"]), item["memory_id"]))
        return {
            "status": "ok",
            "candidate_count": len(matches),
            "indexed_memory_ids": sorted(str(item["memory_id"]) for item in matches),
            "min_importance": default_min_importance,
            "min_importance_by_kind": importance_floor,
            "matches": matches[: max(1, min(30, int(top_k or 8)))],
        }
