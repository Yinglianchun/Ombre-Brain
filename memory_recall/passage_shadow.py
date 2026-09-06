from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PASSAGE_SCHEMA_VERSION = 4
_STRONG_BOUNDARIES = frozenset("。！？!?；;\n")
_SOFT_BOUNDARIES = frozenset("，,：:")
_CLOSING_MARKS = frozenset("”’」』】）》\"'")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class PassageConfig:
    min_owner_chars: int = 290
    target_chars: int = 160
    max_chars: int = 240
    min_chars: int = 40
    overlap_sentences: int = 1

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PassageConfig":
        raw = config.get("passage_shadow")
        raw = raw if isinstance(raw, dict) else {}
        min_owner_chars = _bounded_int(raw.get("min_owner_chars"), 290, 0, 2000)
        target = _bounded_int(raw.get("target_chars"), 160, 80, 400)
        maximum = _bounded_int(raw.get("max_chars"), 240, target, 800)
        minimum = _bounded_int(raw.get("min_chars"), 40, 20, target)
        overlap = _bounded_int(raw.get("overlap_sentences"), 1, 0, 2)
        return cls(
            min_owner_chars=min_owner_chars,
            target_chars=target,
            max_chars=maximum,
            min_chars=minimum,
            overlap_sentences=overlap,
        )


def _bounded_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def _trimmed_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if end > start else None


def _section_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for separator in re.finditer(r"(?:\r?\n)[ \t]*(?:\r?\n)+", text):
        span = _trimmed_span(text, start, separator.start())
        if span:
            spans.append(span)
        start = separator.end()
    span = _trimmed_span(text, start, len(text))
    if span:
        spans.append(span)
    return spans


def _sentence_spans(
    text: str,
    start: int,
    end: int,
    *,
    min_chars: int,
    max_chars: int,
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = start
    index = start
    while index < end:
        char = text[index]
        length = index + 1 - cursor
        boundary = char in _STRONG_BOUNDARIES or (
            char in _SOFT_BOUNDARIES and length >= max(min_chars, max_chars // 2)
        )
        if boundary:
            unit_end = index + 1
            while unit_end < end and text[unit_end] in _CLOSING_MARKS:
                unit_end += 1
            span = _trimmed_span(text, cursor, unit_end)
            if span:
                spans.append(span)
            cursor = unit_end
            index = unit_end
            continue
        index += 1
    span = _trimmed_span(text, cursor, end)
    if span:
        spans.append(span)
    return spans


def _split_oversized_span(
    text: str,
    span: tuple[int, int],
    *,
    max_chars: int,
) -> list[tuple[int, int]]:
    start, end = span
    if end - start <= max_chars:
        return [span]
    output: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        hard_end = min(end, cursor + max_chars)
        split_at = hard_end
        if hard_end < end:
            floor = cursor + max_chars // 2
            candidates = [text.rfind(mark, floor, hard_end) for mark in _SOFT_BOUNDARIES]
            boundary = max(candidates)
            if boundary >= floor:
                split_at = boundary + 1
        trimmed = _trimmed_span(text, cursor, split_at)
        if trimmed:
            output.append(trimmed)
        cursor = max(cursor + 1, split_at)
    return output


def build_verbatim_passages(
    content: str,
    passage_config: PassageConfig | None = None,
) -> list[dict[str, Any]]:
    """Split exact text into sentence-aware, source-offset-preserving passages."""

    text = str(content or "")
    config = passage_config or PassageConfig()
    if not text.strip():
        return []

    passage_spans: list[tuple[int, int]] = []
    for section_start, section_end in _section_spans(text):
        units: list[tuple[int, int]] = []
        for unit in _sentence_spans(
            text,
            section_start,
            section_end,
            min_chars=config.min_chars,
            max_chars=config.max_chars,
        ):
            units.extend(
                _split_oversized_span(text, unit, max_chars=config.max_chars)
            )
        if not units:
            continue

        index = 0
        while index < len(units):
            passage_start = units[index][0]
            stop = index + 1
            passage_end = units[index][1]
            while stop < len(units):
                next_end = units[stop][1]
                if next_end - passage_start > config.max_chars:
                    break
                passage_end = next_end
                stop += 1
                if passage_end - passage_start >= config.target_chars:
                    break

            passage_spans.append((passage_start, passage_end))
            if stop >= len(units):
                break
            index = max(index + 1, stop - config.overlap_sentences)

    if not passage_spans:
        span = _trimmed_span(text, 0, len(text))
        passage_spans = [span] if span else []

    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    output: list[dict[str, Any]] = []
    for ordinal, (start, end) in enumerate(passage_spans):
        passage_text = text[start:end]
        output.append(
            {
                "ordinal": len(output),
                "start_offset": start,
                "end_offset": end,
                "text": passage_text,
                "content_hash": content_hash,
            }
        )
    if output:
        return output

    span = _trimmed_span(text, 0, len(text))
    if not span:
        return []
    start, end = span
    return [
        {
            "ordinal": 0,
            "start_offset": start,
            "end_offset": end,
            "text": text[start:end],
            "content_hash": content_hash,
        }
    ]


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


class PassageShadowIndex:
    """Independent, rebuildable Scene/Event passage index with no recall authority."""

    def __init__(self, config: dict[str, Any], embedding_engine: Any):
        state_dir = str(
            config.get("state_dir")
            or os.path.join(
                os.path.dirname(os.path.abspath(config.get("buckets_dir", "buckets"))),
                "state",
            )
        )
        self.db_path = os.path.join(state_dir, "memory_passage_embeddings.sqlite")
        self.embedding_engine = embedding_engine
        self.passage_config = PassageConfig.from_config(config)
        raw_config = config.get("passage_shadow")
        raw_config = raw_config if isinstance(raw_config, dict) else {}
        self.embedding_concurrency = _bounded_int(
            raw_config.get("embedding_concurrency"), 6, 1, 12
        )
        self.backfill_embedding_concurrency = _bounded_int(
            raw_config.get("backfill_embedding_concurrency"), 2, 1, 4
        )
        self.backfill_request_delay_ms = _bounded_int(
            raw_config.get("backfill_request_delay_ms"), 150, 0, 5000
        )

    def _connect(self, db_path: str | None = None) -> sqlite3.Connection:
        conn = sqlite3.connect(db_path or self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _connect_readonly(self) -> sqlite3.Connection:
        conn = sqlite3.connect(f"{Path(self.db_path).resolve().as_uri()}?mode=ro", uri=True, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def read_availability(self) -> dict[str, Any]:
        if not os.path.exists(self.db_path):
            return {"status": "unavailable", "reason": "index_missing"}
        try:
            with closing(self._connect_readonly()) as conn:
                conn.execute("SELECT 1 FROM memory_passage_embeddings LIMIT 1").fetchone()
        except sqlite3.Error as exc:
            return {"status": "unavailable", "reason": "index_unreadable", "error": str(exc)}
        return {"status": "ok"}

    def _init_db(self, db_path: str | None = None) -> None:
        target_path = db_path or self.db_path
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with closing(self._connect(target_path)) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory_passage_embeddings (
                    owner_kind TEXT NOT NULL CHECK(owner_kind IN ('scene', 'event')),
                    owner_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    source_hash TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    start_offset INTEGER NOT NULL,
                    end_offset INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(owner_kind, owner_id, ordinal)
                );
                CREATE INDEX IF NOT EXISTS idx_memory_passages_owner
                ON memory_passage_embeddings(owner_kind, owner_id);

                CREATE TABLE IF NOT EXISTS memory_passage_owner_state (
                    owner_kind TEXT NOT NULL CHECK(owner_kind IN ('scene', 'event')),
                    owner_id TEXT NOT NULL,
                    source_hash TEXT NOT NULL,
                    passage_count INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(owner_kind, owner_id)
                );
                """
            )

    def _source_hash(
        self,
        owner_kind: str,
        owner_id: str,
        content: str,
        title: str = "",
    ) -> str:
        profile = {
            "schema": PASSAGE_SCHEMA_VERSION,
            "owner_kind": owner_kind,
            "owner_id": owner_id,
            "title": title,
            "content": content,
            "model": str(getattr(self.embedding_engine, "model", "") or ""),
            "document_instruction": str(
                getattr(self.embedding_engine, "document_instruction", "") or ""
            ),
            "passage_config": self.passage_config.__dict__,
        }
        return hashlib.sha256(
            json.dumps(profile, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _owners(
        scenes: Iterable[dict[str, Any]],
        events: Iterable[dict[str, Any]],
    ) -> list[tuple[str, str, str, str]]:
        owners: list[tuple[str, str, str, str]] = []
        for item in scenes:
            owner_id = str(item.get("id") or item.get("scene_id") or "").strip()
            content = str(item.get("content") or item.get("body") or "")
            if owner_id and content.strip():
                owners.append(("scene", owner_id, content, ""))
        for item in events:
            owner_id = str(item.get("item_id") or item.get("event_id") or "").strip()
            content = str(item.get("body") or item.get("content") or "")
            if owner_id and content.strip():
                owners.append(("event", owner_id, content, str(item.get("title") or "").strip()))
        return owners

    async def sync(
        self,
        *,
        scenes: Iterable[dict[str, Any]],
        events: Iterable[dict[str, Any]],
        dry_run: bool = False,
        refresh_all: bool = False,
        embedding_concurrency: int | None = None,
        request_delay_ms: int = 0,
        _db_path: str | None = None,
    ) -> dict[str, Any]:
        target_db_path = _db_path or self.db_path
        owners = self._owners(scenes, events)
        existing: dict[tuple[str, str], list[sqlite3.Row]] = {}
        existing_state: dict[tuple[str, str], sqlite3.Row] = {}
        if os.path.exists(target_db_path):
            with closing(self._connect(target_db_path)) as conn:
                for row in conn.execute("SELECT * FROM memory_passage_embeddings"):
                    existing.setdefault(
                        (str(row["owner_kind"]), str(row["owner_id"])), []
                    ).append(row)
                state_table = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                    ("memory_passage_owner_state",),
                ).fetchone()
                if state_table:
                    existing_state = {
                        (str(row["owner_kind"]), str(row["owner_id"])): row
                        for row in conn.execute("SELECT * FROM memory_passage_owner_state")
                    }

        desired_keys = {(kind, owner_id) for kind, owner_id, _content, _title in owners}
        stale_keys = sorted((set(existing) | set(existing_state)) - desired_keys)
        plans: list[tuple[str, str, str, str, str, list[dict[str, Any]], bool]] = []
        for kind, owner_id, content, title in owners:
            source_hash = self._source_hash(kind, owner_id, content, title)
            passages = (
                build_verbatim_passages(content, self.passage_config)
                if len(content.strip()) > self.passage_config.min_owner_chars
                else []
            )
            if len(passages) <= 1:
                passages = []
            state = existing_state.get((kind, owner_id))
            reusable = bool(state) and not refresh_all and (
                str(state["source_hash"]) == source_hash
                and int(state["passage_count"] or 0) == len(passages)
            )
            plans.append((kind, owner_id, content, title, source_hash, passages, reusable))

        kind_counts = {
            kind: sum(
                1
                for owner_kind, _owner_id, _content, _title in owners
                if owner_kind == kind
            )
            for kind in ("scene", "event")
        }
        passage_count = sum(len(plan[5]) for plan in plans)
        to_embed = sum(len(plan[5]) for plan in plans if not plan[6])
        owners_to_refresh = sum(1 for plan in plans if not plan[6])
        whole_only_owners = sum(1 for plan in plans if not plan[5])
        if dry_run:
            return {
                "status": "dry_run",
                "owners": len(owners),
                "owner_kinds": kind_counts,
                "passages": passage_count,
                "to_embed": to_embed,
                "owners_to_refresh": owners_to_refresh,
                "whole_only_owners": whole_only_owners,
                "stale_owners": len(stale_keys),
                "passage_config": self.passage_config.__dict__,
            }
        pending_plans = [plan for plan in plans if not plan[6]]
        if pending_plans and not getattr(self.embedding_engine, "enabled", False):
            raise RuntimeError("embedding_engine_disabled")

        self._init_db(target_db_path)
        embedded = 0
        failed: list[str] = []
        effective_concurrency = _bounded_int(
            embedding_concurrency,
            self.embedding_concurrency,
            1,
            12,
        )
        effective_delay_ms = _bounded_int(request_delay_ms, 0, 0, 5000)
        semaphore = asyncio.Semaphore(effective_concurrency)

        async def embed_plan(
            plan: tuple[str, str, str, str, str, list[dict[str, Any]], bool],
        ) -> tuple[str, str, list[tuple[dict[str, Any], list[float]]]]:
            kind, owner_id, _content, title, _source_hash, passages, reusable = plan
            if reusable:
                return kind, owner_id, []
            rows_to_write: list[tuple[dict[str, Any], list[float]]] = []
            for passage in passages:
                embedding_text = (
                    f"{title}\n{passage['text']}" if kind == "event" and title else passage["text"]
                )
                async with semaphore:
                    if effective_delay_ms:
                        await asyncio.sleep(effective_delay_ms / 1000)
                    vector = await self.embedding_engine.embed_document(embedding_text)
                if not vector:
                    return kind, owner_id, []
                rows_to_write.append((passage, vector))
            return kind, owner_id, rows_to_write

        embedded_plans = await asyncio.gather(
            *(embed_plan(plan) for plan in pending_plans)
        )
        embedded_by_owner = {
            (kind, owner_id): rows for kind, owner_id, rows in embedded_plans
        }
        with closing(self._connect(target_db_path)) as conn:
            for kind, owner_id, _content, _title, source_hash, passages, reusable in plans:
                if reusable:
                    continue
                rows_to_write = embedded_by_owner.get((kind, owner_id), [])
                if not rows_to_write and passages:
                    failed.append(f"{kind}:{owner_id}")
                    continue
                conn.execute(
                    "DELETE FROM memory_passage_embeddings WHERE owner_kind=? AND owner_id=?",
                    (kind, owner_id),
                )
                for passage, vector in rows_to_write:
                    conn.execute(
                        """
                        INSERT INTO memory_passage_embeddings(
                            owner_kind, owner_id, ordinal, source_hash, content_hash,
                            start_offset, end_offset, text, embedding, model, dimension,
                            updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            kind,
                            owner_id,
                            int(passage["ordinal"]),
                            source_hash,
                            str(passage["content_hash"]),
                            int(passage["start_offset"]),
                            int(passage["end_offset"]),
                            str(passage["text"]),
                            json.dumps(vector),
                            str(getattr(self.embedding_engine, "model", "") or ""),
                            len(vector),
                            _now(),
                        ),
                    )
                    embedded += 1
                conn.execute(
                    """
                    INSERT INTO memory_passage_owner_state(
                        owner_kind, owner_id, source_hash, passage_count, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(owner_kind, owner_id) DO UPDATE SET
                        source_hash=excluded.source_hash,
                        passage_count=excluded.passage_count,
                        updated_at=excluded.updated_at
                    """,
                    (kind, owner_id, source_hash, len(passages), _now()),
                )
            for kind, owner_id in stale_keys:
                conn.execute(
                    "DELETE FROM memory_passage_embeddings WHERE owner_kind=? AND owner_id=?",
                    (kind, owner_id),
                )
                conn.execute(
                    "DELETE FROM memory_passage_owner_state WHERE owner_kind=? AND owner_id=?",
                    (kind, owner_id),
                )
            conn.commit()
        return {
            "status": "partial" if failed else "ok",
            "owners": len(owners),
            "owner_kinds": kind_counts,
            "passages": passage_count,
            "owners_to_refresh": owners_to_refresh,
            "whole_only_owners": whole_only_owners,
            "embedded": embedded,
            "reused_owners": sum(1 for plan in plans if plan[6]),
            "failed_owners": sorted(set(failed)),
            "removed_owners": len(stale_keys),
            "embedding_concurrency": effective_concurrency,
            "request_delay_ms": effective_delay_ms,
        }

    async def rebuild_atomic(
        self,
        *,
        scenes: Iterable[dict[str, Any]],
        events: Iterable[dict[str, Any]],
        refresh_all: bool = False,
    ) -> dict[str, Any]:
        """Build a complete candidate DB and activate it only after full success."""

        scene_rows = list(scenes)
        event_rows = list(events)
        plan = await self.sync(
            scenes=scene_rows,
            events=event_rows,
            dry_run=True,
            refresh_all=refresh_all,
        )
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        descriptor, staging_path = tempfile.mkstemp(
            prefix="memory_passage_embeddings-",
            suffix=".next.sqlite",
            dir=os.path.dirname(self.db_path),
        )
        os.close(descriptor)
        os.unlink(staging_path)
        previous_index_available = os.path.exists(self.db_path)
        if previous_index_available:
            shutil.copy2(self.db_path, staging_path)
        try:
            result = await self.sync(
                scenes=scene_rows,
                events=event_rows,
                refresh_all=refresh_all,
                embedding_concurrency=self.backfill_embedding_concurrency,
                request_delay_ms=self.backfill_request_delay_ms,
                _db_path=staging_path,
            )
            if result.get("status") != "ok":
                return {
                    **result,
                    "plan": plan,
                    "activated": False,
                    "previous_index_preserved": previous_index_available,
                }
            os.replace(staging_path, self.db_path)
            return {
                **result,
                "plan": plan,
                "activated": True,
                "previous_index_preserved": False,
            }
        finally:
            if os.path.exists(staging_path):
                os.unlink(staging_path)

    def search_by_embedding(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        owner_kinds: Iterable[str] = ("scene", "event"),
        passages_per_owner: int = 2,
        allowed_owner_ids: set[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        if not query_embedding:
            return {"status": "unavailable", "reason": "query_embedding_empty", "matches": []}
        availability = self.read_availability()
        if availability.get("status") != "ok":
            return {**availability, "matches": []}
        kinds = {str(kind).strip().lower() for kind in owner_kinds}
        kinds &= {"scene", "event"}
        if not kinds:
            return {"status": "ok", "candidate_count": 0, "matches": []}

        model = str(getattr(self.embedding_engine, "model", "") or "")
        with closing(self._connect_readonly()) as conn:
            rows = conn.execute("SELECT * FROM memory_passage_embeddings").fetchall()
        by_owner: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in rows:
            kind = str(row["owner_kind"])
            owner_key = (kind, str(row["owner_id"]))
            if allowed_owner_ids is not None and owner_key not in allowed_owner_ids:
                continue
            if kind not in kinds or str(row["model"]) != model:
                continue
            try:
                vector = [float(value) for value in json.loads(row["embedding"])]
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if int(row["dimension"] or 0) != len(query_embedding):
                continue
            by_owner.setdefault((kind, str(row["owner_id"])), []).append(
                {
                    "ordinal": int(row["ordinal"]),
                    "start_offset": int(row["start_offset"]),
                    "end_offset": int(row["end_offset"]),
                    "text": str(row["text"]),
                    "score": round(_cosine(query_embedding, vector), 4),
                }
            )

        matches: list[dict[str, Any]] = []
        keep = max(1, min(4, int(passages_per_owner or 2)))
        for (kind, owner_id), passages in by_owner.items():
            passages.sort(key=lambda item: (-float(item["score"]), int(item["ordinal"])))
            matches.append(
                {
                    "owner_kind": kind,
                    "owner_id": owner_id,
                    "score": passages[0]["score"],
                    "passages": passages[:keep],
                }
            )
        matches.sort(
            key=lambda item: (-float(item["score"]), item["owner_kind"], item["owner_id"])
        )
        return {
            "status": "ok",
            "candidate_count": len(matches),
            "indexed_owner_ids": [
                {"owner_kind": kind, "owner_id": owner_id}
                for kind, owner_id in sorted(by_owner)
            ],
            "matches": matches[: max(1, min(100, int(top_k or 10)))],
        }

    def owner_ids(self) -> set[tuple[str, str]]:
        if not os.path.exists(self.db_path):
            return set()
        with closing(self._connect()) as conn:
            return {
                (str(row["owner_kind"]), str(row["owner_id"]))
                for row in conn.execute(
                    "SELECT DISTINCT owner_kind, owner_id FROM memory_passage_embeddings"
                )
            }

    def passages_for_owners(
        self,
        owners: Iterable[tuple[str, str]],
        *,
        limit_per_owner: int = 2,
    ) -> dict[tuple[str, str], list[dict[str, Any]]]:
        keys = {
            (str(kind).strip().lower(), str(owner_id).strip())
            for kind, owner_id in owners
            if str(kind).strip().lower() in {"scene", "event"}
            and str(owner_id).strip()
        }
        if not keys or not os.path.exists(self.db_path):
            return {}
        output: dict[tuple[str, str], list[dict[str, Any]]] = {}
        with closing(self._connect()) as conn:
            for kind, owner_id in sorted(keys):
                rows = conn.execute(
                    """
                    SELECT ordinal, start_offset, end_offset, text
                    FROM memory_passage_embeddings
                    WHERE owner_kind=? AND owner_id=?
                    ORDER BY ordinal
                    LIMIT ?
                    """,
                    (kind, owner_id, max(1, min(100, int(limit_per_owner or 2)))),
                ).fetchall()
                output[(kind, owner_id)] = [
                    {
                        "ordinal": int(row["ordinal"]),
                        "start_offset": int(row["start_offset"]),
                        "end_offset": int(row["end_offset"]),
                        "text": str(row["text"]),
                    }
                    for row in rows
                ]
        return output
