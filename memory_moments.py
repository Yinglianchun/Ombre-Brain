from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any

from favorite_tags import favorite_policy_tags
from memory_relevance import (
    MemoryRelevanceOptions,
    content_terms_for_query,
    expanded_terms_for_query,
    facets_for_node,
    memory_relevance_options_from_config,
)
from memory_retrieval_aliases import (
    DEFAULT_COMPACT_RETRIEVAL_ALIAS_PATTERNS,
    GENERIC_RETRIEVAL_ALIAS_KEYS,
    GENERIC_RETRIEVAL_ALIAS_STOP_KEYS,
    MAX_RETRIEVAL_ALIAS_CHARS,
    MAX_RETRIEVAL_ALIASES_PER_BUCKET,
    MAX_RETRIEVAL_ALIASES_PER_MOMENT,
    RETRIEVAL_ALIAS_SECTIONS,
    MemoryRetrievalAliasIndex,
    _build_retrieval_aliases,
    _clean_retrieval_alias_text,
    _compact_retrieval_alias_patterns,
    _retrieval_alias_is_date,
    _retrieval_alias_is_identifier,
    _retrieval_alias_key,
    _retrieval_alias_query_terms,
    _retrieval_alias_variants,
    _retrieval_phrase_candidates,
    _valid_retrieval_alias,
)
from utils import strip_wikilinks


SECTION_ALIASES = {
    "scene": "scene",
    "moment": "moment",
    "memory": "moment",
    "fact": "fact",
    "facts": "fact",
    "profile_fact": "fact",
    "profile fact": "fact",
    "original": "original",
    "raw": "original",
    "quote": "original",
    "quotes": "original",
    "context": "context",
    "evidence_context": "evidence_context",
    "evidence context": "evidence_context",
    "evidence": "evidence_context",
    "background": "context",
    "feeling": "feeling",
    "feel": "feeling",
    "reflection": "reflection",
    "assistant_reflection": "reflection",
    "assistant reflection": "reflection",
    "haven_reflection": "reflection",
    "haven reflection": "reflection",
    "followup": "followup",
    "follow-up": "followup",
    "followup_log": "followup_log",
    "followup log": "followup_log",
    "follow-up log": "followup_log",
    "next": "followup",
    "todo": "followup",
    "todo_log": "followup_log",
    "todo log": "followup_log",
    "affect_anchor": "affect_anchor",
    "affect anchor": "affect_anchor",
    "favorite_reason": "favorite_reason",
    "favorite reason": "favorite_reason",
    "\u7247\u6bb5": "moment",
    "\u573a\u666f": "scene",
    "\u8bb0\u5fc6\u7247\u6bb5": "moment",
    "\u4e8b\u5b9e": "fact",
    "\u5bf9\u8bdd\u4e8b\u5b9e": "fact",
    "\u539f\u6587": "original",
    "\u5bf9\u8bdd\u539f\u6587": "original",
    "\u5f15\u7528": "original",
    "\u4e0a\u4e0b\u6587": "context",
    "\u8bc1\u636e\u4e0a\u4e0b\u6587": "evidence_context",
    "\u8bc1\u636e": "evidence_context",
    "\u80cc\u666f": "context",
    "\u8bed\u5883": "context",
    "\u611f\u53d7": "feeling",
    "\u60c5\u7eea": "feeling",
    "\u53cd\u601d": "reflection",
    "\u540e\u7eed": "followup",
    "\u540e\u7eed\u8bb0\u5f55": "followup_log",
    "\u5f85\u529e": "followup",
    "\u5f85\u529e\u8bb0\u5f55": "followup_log",
    "\u559c\u6b22\u5b83\u7684\u539f\u56e0": "favorite_reason",
    "\u559c\u6b22\u7684\u539f\u56e0": "favorite_reason",
}

HEADING_RE = re.compile(r"^(#{2,6})\s+(.+?)\s*$")

DEFAULT_ANNOTATION_OPTIONS = {
    "enabled": True,
    "max_summary_chars": 160,
    "max_evidence_spans": 3,
    "max_evidence_chars": 120,
}
DEFAULT_CONTENT_START_LINE = 1
SHADOW_CHUNKABLE_CONTENT_SECTIONS = frozenset(
    {
        "body",
        "scene",
        "moment",
        "fact",
        "original",
        "context",
        "evidence_context",
        "reflection",
        "feeling",
    }
)
SENTENCE_END_RE = re.compile(r"[\u3002\uff01\uff1f\uff1b!?;]+|[.]+(?=\s|$)")


class MemoryMomentStore:
    """SQLite index of bucket body/comment moments."""

    def __init__(self, config: dict):
        config = config or {}
        self.relevance_options = memory_relevance_options_from_config(config)
        self.annotation_options = _annotation_options_from_config(config)
        state_dir = config.get("state_dir") or os.path.join(
            os.path.dirname(os.path.abspath(config.get("buckets_dir", "buckets"))),
            "state",
        )
        self.db_path = os.path.join(state_dir, "memory_moments.sqlite")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.retrieval_alias_index = MemoryRetrievalAliasIndex(
            config,
            db_path=self.db_path,
            create=False,
        )
        # Compatibility attributes for local diagnostics and older callers.
        self.retrieval_alias_stop_keys = self.retrieval_alias_index.stop_keys
        self.compact_retrieval_alias_patterns = self.retrieval_alias_index.patterns
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_moments (
                moment_id TEXT PRIMARY KEY,
                bucket_id TEXT NOT NULL,
                section TEXT NOT NULL,
                text TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                source TEXT NOT NULL,
                source_id TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_moments_bucket ON memory_moments(bucket_id, ordinal)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_moment_edges (
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                bucket_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(source, target, relation_type)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_moment_edges_source ON memory_moment_edges(source)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_moment_edges_target ON memory_moment_edges(target)"
        )
        self.retrieval_alias_index.ensure_schema(conn)
        conn.commit()
        conn.close()

    def upsert_bucket(self, bucket: dict) -> list[dict]:
        moments = parse_bucket_moments(bucket, self.relevance_options, self.annotation_options)
        bucket_id = _bucket_id(bucket)
        conn = self._connect()
        self._replace_bucket(
            conn,
            bucket_id,
            moments,
            _bucket_title(bucket),
            explicit_aliases=_bucket_scene_cues(bucket),
        )
        conn.commit()
        conn.close()
        return [dict(moment) for moment in moments]

    def bulk_upsert(self, buckets: list[dict]) -> dict:
        conn = self._connect()
        indexed_buckets = 0
        indexed_moments = 0
        for bucket in buckets:
            bucket_id = _bucket_id(bucket)
            moments = parse_bucket_moments(bucket, self.relevance_options, self.annotation_options)
            self._replace_bucket(
                conn,
                bucket_id,
                moments,
                _bucket_title(bucket),
                explicit_aliases=_bucket_scene_cues(bucket),
            )
            indexed_buckets += 1
            indexed_moments += len(moments)
        conn.commit()
        conn.close()
        return {"buckets": indexed_buckets, "moments": indexed_moments}

    def list_for_bucket(self, bucket_id: str, limit: int = 100) -> list[dict]:
        bucket_id = str(bucket_id or "").strip()
        if not bucket_id:
            return []
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT * FROM memory_moments
            WHERE bucket_id = ?
            ORDER BY ordinal ASC
            LIMIT ?
            """,
            (bucket_id, max(1, int(limit))),
        ).fetchall()
        conn.close()
        return [self._row_to_moment(row) for row in rows]

    def list_all(self, limit: int = 10000) -> list[dict]:
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT * FROM memory_moments
            ORDER BY bucket_id ASC, ordinal ASC
            LIMIT ?
            """,
            (max(1, int(limit)),),
        ).fetchall()
        conn.close()
        return [self._row_to_moment(row) for row in rows]

    def list_for_bucket_aliases(self, bucket_id: str, limit: int = 100) -> list[dict]:
        return self.retrieval_alias_index.list_for_bucket(bucket_id, limit=limit)

    def get(self, moment_id: str) -> dict | None:
        moment_id = str(moment_id or "").strip()
        if not moment_id:
            return None
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM memory_moments WHERE moment_id = ?",
            (moment_id,),
        ).fetchone()
        conn.close()
        return self._row_to_moment(row) if row else None

    def list_edges(self, bucket_id: str = "") -> list[dict]:
        conn = self._connect()
        if bucket_id:
            rows = conn.execute(
                """
                SELECT * FROM memory_moment_edges
                WHERE bucket_id = ?
                ORDER BY source ASC, target ASC
                """,
                (str(bucket_id),),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM memory_moment_edges
                ORDER BY bucket_id ASC, source ASC, target ASC
                """
            ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def replace_generated_edges(
        self,
        edges: list[dict],
        *,
        reason_prefix: str = "local_graph:",
    ) -> int:
        prefix = str(reason_prefix or "").strip()
        if not prefix:
            raise ValueError("reason_prefix is required")
        conn = self._connect()
        conn.execute(
            "DELETE FROM memory_moment_edges WHERE reason LIKE ?",
            (f"{prefix}%",),
        )
        written = 0
        for edge in edges or []:
            if not isinstance(edge, dict):
                continue
            source = str(edge.get("source") or "").strip()
            target = str(edge.get("target") or "").strip()
            bucket_id = str(edge.get("bucket_id") or "").strip()
            relation_type = str(edge.get("relation_type") or "relates_to").strip()
            if not source or not target or source == target or not bucket_id:
                continue
            reason = str(edge.get("reason") or "").strip()
            if not reason.startswith(prefix):
                reason = f"{prefix}{reason}"
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_moment_edges
                (source, target, bucket_id, relation_type, confidence, reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source,
                    target,
                    bucket_id,
                    relation_type,
                    _clamp_float(edge.get("confidence", 0.5), 0.0, 1.0),
                    reason[:240],
                    str(edge.get("created_at") or datetime.now(timezone.utc).isoformat(timespec="seconds")),
                ),
            )
            written += 1
        conn.commit()
        conn.close()
        return written

    def delete_bucket(self, bucket_id: str) -> dict:
        bucket_id = str(bucket_id or "").strip()
        if not bucket_id:
            return {"moments": 0, "edges": 0, "aliases": 0}

        escaped = (
            bucket_id
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        moment_prefix = f"{escaped}:%"
        conn = self._connect()
        edge_cursor = conn.execute(
            """
            DELETE FROM memory_moment_edges
            WHERE bucket_id = ?
               OR source LIKE ? ESCAPE '\\'
               OR target LIKE ? ESCAPE '\\'
            """,
            (bucket_id, moment_prefix, moment_prefix),
        )
        moment_cursor = conn.execute(
            "DELETE FROM memory_moments WHERE bucket_id = ?",
            (bucket_id,),
        )
        alias_count = self.retrieval_alias_index.delete_bucket(conn, bucket_id)
        conn.commit()
        conn.close()
        return {
            "moments": max(0, int(moment_cursor.rowcount or 0)),
            "edges": max(0, int(edge_cursor.rowcount or 0)),
            "aliases": alias_count,
        }

    def search_retrieval_aliases(self, query: str, limit: int = 20) -> list[dict]:
        return self.retrieval_alias_index.search(query, limit=limit)

    def sync_alias_projection(self, bucket: dict, items: list[dict]) -> dict:
        """Replace legacy Moments with a non-Moment recall projection atomically."""
        bucket_id = _bucket_id(bucket)
        desired_aliases = self.retrieval_alias_index.build_rows(
            bucket_id,
            _bucket_title(bucket),
            items,
            explicit_aliases=_bucket_scene_cues(bucket),
        )
        escaped = (
            bucket_id
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        moment_prefix = f"{escaped}:%"
        conn = self._connect()
        try:
            moment_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM memory_moments WHERE bucket_id = ?",
                    (bucket_id,),
                ).fetchone()[0]
                or 0
            )
            edge_count = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM memory_moment_edges
                    WHERE bucket_id = ?
                       OR source LIKE ? ESCAPE '\\'
                       OR target LIKE ? ESCAPE '\\'
                    """,
                    (bucket_id, moment_prefix, moment_prefix),
                ).fetchone()[0]
                or 0
            )
            aliases_changed = not self.retrieval_alias_index.bucket_matches(
                conn,
                bucket_id,
                desired_aliases,
            )
            if not moment_count and not edge_count and not aliases_changed:
                return {"moments": 0, "edges": 0, "aliases": 0, "changed": False}
            edge_cursor = conn.execute(
                """
                DELETE FROM memory_moment_edges
                WHERE bucket_id = ?
                   OR source LIKE ? ESCAPE '\\'
                   OR target LIKE ? ESCAPE '\\'
                """,
                (bucket_id, moment_prefix, moment_prefix),
            )
            moment_cursor = conn.execute(
                "DELETE FROM memory_moments WHERE bucket_id = ?",
                (bucket_id,),
            )
            alias_count = 0
            if aliases_changed:
                alias_count = self.retrieval_alias_index.replace_rows(
                    conn,
                    bucket_id,
                    desired_aliases,
                )
            conn.commit()
            return {
                "moments": max(0, int(moment_cursor.rowcount or 0)),
                "edges": max(0, int(edge_cursor.rowcount or 0)),
                "aliases": alias_count,
                "changed": True,
            }
        finally:
            conn.close()

    def search_moments(
        self,
        query: str,
        *,
        limit: int = 20,
        bucket_boosts: dict[str, float] | None = None,
        include_sections: set[str] | list[str] | tuple[str, ...] | None = None,
        exclude_sections: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> list[dict]:
        return self.search_moment_items(
            query,
            self.list_all(),
            limit=limit,
            bucket_boosts=bucket_boosts,
            include_sections=include_sections,
            exclude_sections=exclude_sections,
        )

    def search_moment_items(
        self,
        query: str,
        moments: list[dict],
        *,
        limit: int = 20,
        bucket_boosts: dict[str, float] | None = None,
        include_sections: set[str] | list[str] | tuple[str, ...] | None = None,
        exclude_sections: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> list[dict]:
        query = str(query or "").strip()
        if not query:
            return []
        bucket_boosts = bucket_boosts or {}
        included = {str(section or "") for section in (include_sections or []) if str(section or "")}
        excluded = {str(section or "") for section in (exclude_sections or []) if str(section or "")}
        query_terms = content_terms_for_query(query, self.relevance_options)
        expanded_query_terms = _expanded_query_terms(query, self.relevance_options)
        scored = []
        for moment in moments or []:
            section = str(moment.get("section") or "")
            if included and section not in included:
                continue
            if excluded and section in excluded:
                continue
            score = _moment_query_score(
                moment,
                query,
                self.relevance_options,
                query_terms=query_terms,
                expanded_query_terms=expanded_query_terms,
            )
            bucket_id = str(moment.get("bucket_id") or "")
            try:
                boost = float(bucket_boosts.get(bucket_id, 0.0))
            except (TypeError, ValueError):
                boost = 0.0
            if boost > 0:
                score = max(score, min(boost, 1.0) * 0.75)
            if score <= 0:
                continue
            item = dict(moment)
            item["score"] = round(score, 4)
            scored.append(item)

        scored.sort(
            key=lambda item: (
                item.get("score", 0.0),
                _moment_section_weight(item.get("section")),
            ),
            reverse=True,
        )
        return scored[: max(1, int(limit))]

    def sample(self, limit: int = 20) -> list[dict]:
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT * FROM memory_moments
            ORDER BY updated_at DESC, bucket_id ASC, ordinal ASC
            LIMIT ?
            """,
            (max(1, int(limit)),),
        ).fetchall()
        conn.close()
        return [self._row_to_moment(row) for row in rows]

    def stats(self) -> dict:
        conn = self._connect()
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS moment_count,
                COUNT(DISTINCT bucket_id) AS bucket_count
            FROM memory_moments
            """
        ).fetchone()
        edge_row = conn.execute(
            "SELECT COUNT(*) AS edge_count FROM memory_moment_edges"
        ).fetchone()
        conn.close()
        return {
            "buckets": int(row["bucket_count"] or 0),
            "moments": int(row["moment_count"] or 0),
            "edges": int(edge_row["edge_count"] or 0),
        }

    def _replace_bucket(
        self,
        conn: sqlite3.Connection,
        bucket_id: str,
        moments: list[dict],
        bucket_title: str,
        *,
        explicit_aliases: Any = (),
    ) -> None:
        conn.execute("DELETE FROM memory_moments WHERE bucket_id = ?", (bucket_id,))
        conn.execute(
            """
            DELETE FROM memory_moment_edges
            WHERE bucket_id = ? AND reason NOT LIKE 'local_graph:%'
            """,
            (bucket_id,),
        )
        for moment in moments:
            conn.execute(
                """
                INSERT INTO memory_moments
                (moment_id, bucket_id, section, text, ordinal, source, source_id,
                 text_hash, metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    moment["moment_id"],
                    moment["bucket_id"],
                    moment["section"],
                    moment["text"],
                    moment["ordinal"],
                    moment["source"],
                    moment["source_id"],
                    moment["text_hash"],
                    moment["metadata_json"],
                    moment["created_at"],
                    moment["updated_at"],
                ),
            )
        for edge in build_moment_edges(moments):
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_moment_edges
                (source, target, bucket_id, relation_type, confidence, reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    edge["source"],
                    edge["target"],
                    edge["bucket_id"],
                    edge["relation_type"],
                    edge["confidence"],
                    edge["reason"],
                    edge["created_at"],
                ),
            )
        self.retrieval_alias_index.replace_bucket(
            conn,
            bucket_id,
            bucket_title,
            moments,
            explicit_aliases=explicit_aliases,
        )

    def _row_to_moment(self, row: sqlite3.Row) -> dict:
        moment = dict(row)
        try:
            metadata = json.loads(moment.get("metadata_json") or "{}")
        except json.JSONDecodeError:
            metadata = {}
        moment["metadata"] = metadata if isinstance(metadata, dict) else {}
        return moment


def _bucket_title(bucket: dict) -> str:
    meta = bucket.get("metadata") if isinstance(bucket.get("metadata"), dict) else {}
    return _clean_text(
        meta.get("name")
        or meta.get("title")
        or bucket.get("name")
        or bucket.get("title")
        or ""
    )


def _bucket_scene_cues(bucket: dict) -> list[str]:
    meta = bucket.get("metadata") if isinstance(bucket.get("metadata"), dict) else {}
    return _list_text(meta.get("scene_cues"))


def _annotation_options_from_config(config: dict | None) -> dict:
    raw = (config or {}).get("moment_annotations", {})
    options = dict(DEFAULT_ANNOTATION_OPTIONS)
    if isinstance(raw, dict):
        options.update(raw)
    return {
        "enabled": _bool_value(options.get("enabled"), True),
        "max_summary_chars": _int_between(options.get("max_summary_chars"), 160, 40, 500),
        "max_evidence_spans": _int_between(options.get("max_evidence_spans"), 3, 0, 10),
        "max_evidence_chars": _int_between(options.get("max_evidence_chars"), 120, 30, 500),
    }


def parse_bucket_moments(
    bucket: dict,
    relevance_options: MemoryRelevanceOptions | None = None,
    annotation_options: dict | None = None,
) -> list[dict]:
    if not isinstance(bucket, dict):
        raise ValueError("bucket must be a dict")

    relevance_options = relevance_options or memory_relevance_options_from_config()
    annotation_options = {**DEFAULT_ANNOTATION_OPTIONS, **(annotation_options or {})}
    bucket_id = _bucket_id(bucket)
    meta = bucket.get("metadata") if isinstance(bucket.get("metadata"), dict) else {}
    base_meta = _bucket_metadata(meta, bucket)
    updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    moments: list[dict] = []
    ordinal = 0

    raw_content = str(bucket.get("content") or "")
    source_base = _source_ref_base(bucket)
    content = _clean_text(raw_content)
    if content:
        structured = _content_moments(
            bucket_id,
            raw_content,
            base_meta,
            updated_at,
            relevance_options,
            annotation_options,
            source_base,
        )
        if structured:
            for moment in structured:
                moment["ordinal"] = ordinal
                moment["moment_id"] = _moment_id(bucket_id, moment["source"], moment["section"], ordinal, moment["source_id"])
                moments.append(moment)
                ordinal += 1
        else:
            moments.append(
                _make_moment(
                    bucket_id=bucket_id,
                    section="body",
                    text=content,
                    ordinal=ordinal,
                    source="content",
                    source_id="body",
                    metadata=base_meta,
                    source_ref=_source_ref_for_body(raw_content, source_base),
                    created_at=str(meta.get("created") or meta.get("updated_at") or ""),
                    updated_at=updated_at,
                    relevance_options=relevance_options,
                    annotation_options=annotation_options,
                )
            )
            ordinal += 1

    comments = meta.get("comments", [])
    if isinstance(comments, list):
        for index, comment in enumerate(comments):
            if not isinstance(comment, dict):
                continue
            text = _clean_text(comment.get("content", ""))
            if not text:
                continue
            source_id = str(comment.get("id") or f"comment-{index}")
            metadata = _clean_metadata(
                {
                    **base_meta,
                    "comment_id": source_id,
                    "comment_author": comment.get("author"),
                    "comment_kind": comment.get("kind"),
                    "comment_source": comment.get("source"),
                    "comment_valence": comment.get("valence"),
                    "comment_arousal": comment.get("arousal"),
                }
            )
            moments.append(
                _make_moment(
                    bucket_id=bucket_id,
                    section="comment",
                    text=text,
                    ordinal=ordinal,
                    source="comment",
                    source_id=source_id,
                    metadata=metadata,
                    created_at=str(comment.get("created") or meta.get("updated_at") or ""),
                    updated_at=updated_at,
                    relevance_options=relevance_options,
                    annotation_options=annotation_options,
                )
            )
            ordinal += 1

    return moments


def build_bucket_recall_item(
    bucket: dict,
    *,
    node_id: str,
    node_kind: str,
    section: str,
    source: str,
    source_id: str,
    extra_metadata: dict | None = None,
    relevance_options: MemoryRelevanceOptions | None = None,
    annotation_options: dict | None = None,
) -> dict:
    """Build one in-memory recall item without parsing or persisting a Moment.

    Canonical objects use this projection to participate in the current recall
    policy while retaining their own identity.  Legacy bucket parsing remains
    exclusively in ``parse_bucket_moments``.
    """
    if not isinstance(bucket, dict):
        raise ValueError("bucket must be a dict")
    normalized_node_id = str(node_id or "").strip()
    normalized_kind = str(node_kind or "").strip()
    if not normalized_node_id or not normalized_kind:
        raise ValueError("node_id and node_kind are required")

    bucket_id = _bucket_id(bucket)
    meta = bucket.get("metadata") if isinstance(bucket.get("metadata"), dict) else {}
    metadata = _clean_metadata(
        {
            **_bucket_metadata(meta, bucket),
            **(extra_metadata or {}),
            "recall_node_kind": normalized_kind,
        }
    )
    raw_content = str(bucket.get("content") or "")
    item = _make_moment(
        bucket_id=bucket_id,
        section=str(section or "body"),
        text=_clean_text(raw_content),
        ordinal=0,
        source=str(source or normalized_kind),
        source_id=str(source_id or bucket_id),
        metadata=metadata,
        source_ref=_source_ref_for_body(raw_content, _source_ref_base(bucket)),
        created_at=str(meta.get("created") or meta.get("updated_at") or ""),
        updated_at=str(meta.get("updated_at") or meta.get("created") or ""),
        relevance_options=relevance_options or memory_relevance_options_from_config(),
        annotation_options={**DEFAULT_ANNOTATION_OPTIONS, **(annotation_options or {})},
    )
    item["moment_id"] = normalized_node_id
    item["node_id"] = normalized_node_id
    item["node_kind"] = normalized_kind
    return item


def preview_bucket_moment_chunks(
    bucket: dict,
    target_chars: int = 320,
    max_chars: int = 520,
    min_tail_chars: int = 100,
) -> dict:
    """Preview line-aware content chunks without changing indexed moments."""
    target_chars = int(target_chars)
    max_chars = int(max_chars)
    min_tail_chars = int(min_tail_chars)
    if target_chars <= 0 or max_chars <= 0:
        raise ValueError("target_chars and max_chars must be positive")
    if target_chars > max_chars:
        raise ValueError("target_chars must not exceed max_chars")
    if min_tail_chars < 0 or min_tail_chars > max_chars:
        raise ValueError("min_tail_chars must be between 0 and max_chars")

    current = parse_bucket_moments(bucket)
    content_text_start_lines = _content_moment_text_start_lines(bucket)
    current_debug = [_current_moment_preview(moment) for moment in current]
    shadow: list[dict] = []

    for moment in current:
        text = str(moment.get("text") or "")
        should_split = (
            moment.get("source") == "content"
            and moment.get("section") in SHADOW_CHUNKABLE_CONTENT_SECTIONS
            and len(text) > max_chars
        )
        chunks = (
            _split_shadow_moment_text(text, target_chars, max_chars, min_tail_chars)
            if should_split
            else [{"text": text, "start_line": 1, "end_line": max(1, text.count("\n") + 1)}]
        )
        parent_ref = _moment_source_ref(moment)
        text_start_line = content_text_start_lines.get(str(moment.get("source_id") or ""))
        for chunk_index, chunk in enumerate(chunks):
            source_ref = parent_ref
            if should_split:
                source_ref = _shadow_chunk_source_ref(parent_ref, text_start_line, chunk)
            chunk_text = str(chunk["text"])
            shadow.append(
                {
                    "parent_moment_id": str(moment.get("moment_id") or ""),
                    "section": str(moment.get("section") or ""),
                    "chunk_index": chunk_index,
                    "chars": len(chunk_text),
                    "text_preview": _clip_text(chunk_text, 180),
                    "source_ref": source_ref,
                    "source": str(moment.get("source") or ""),
                    "source_id": str(moment.get("source_id") or ""),
                    "parent_ordinal": int(moment.get("ordinal") or 0),
                    "split": should_split,
                }
            )

    current_content_count = sum(1 for moment in current if moment.get("source") == "content")
    shadow_content_count = sum(1 for moment in shadow if moment.get("source") == "content")
    return {
        "mode": "shadow_preview",
        "strategy": "line_then_sentence_with_short_tail_merge",
        "thresholds": {
            "target_chars": target_chars,
            "max_chars": max_chars,
            "min_tail_chars": min_tail_chars,
        },
        "current_content_moment_count": current_content_count,
        "shadow_content_moment_count": shadow_content_count,
        "changed": shadow_content_count != current_content_count,
        "current_moments": current_debug,
        "shadow_moments": shadow,
    }


def build_moment_edges(moments: list[dict]) -> list[dict]:
    ordered = sorted(
        [moment for moment in moments if moment.get("moment_id")],
        key=lambda item: int(item.get("ordinal", 0)),
    )
    if not ordered:
        return []

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    edges: list[dict] = []
    for left, right in zip(ordered, ordered[1:]):
        edges.append(
            _make_edge(
                left,
                right,
                "next_context",
                0.85,
                "same bucket next moment",
                now,
            )
        )
        edges.append(
            _make_edge(
                right,
                left,
                "previous_context",
                0.75,
                "same bucket previous moment",
                now,
            )
        )

    anchor = _first_content_moment(ordered)
    if anchor:
        for moment in ordered:
            section = str(moment.get("section") or "")
            if section not in {"affect_anchor", "favorite_reason", "comment"}:
                continue
            if moment["moment_id"] == anchor["moment_id"]:
                continue
            edges.append(
                _make_edge(
                    moment,
                    anchor,
                    "emotional_echo",
                    0.9,
                    f"{section} points back to source moment",
                    now,
                )
            )
        for moment in ordered:
            section = str(moment.get("section") or "")
            if section not in {"feeling", "reflection"} or moment["moment_id"] == anchor["moment_id"]:
                continue
            edges.append(
                _make_edge(
                    moment,
                    anchor,
                    "reflects_on",
                    0.8,
                    "feeling reflects source moment",
                    now,
                )
            )
    return _dedupe_edges(edges)


def _content_moments(
    bucket_id: str,
    content: str,
    base_meta: dict,
    updated_at: str,
    relevance_options: MemoryRelevanceOptions,
    annotation_options: dict,
    source_base: dict | None = None,
) -> list[dict]:
    blocks = _split_markdown_blocks(content)
    if not any(_canonical_section(block["heading"]) for block in blocks if block["heading"]):
        return []

    moments: list[dict] = []
    ordinal = 0
    for block_index, block in enumerate(blocks):
        heading = block["heading"]
        text = block["text"].strip()
        section = _canonical_section(heading) if heading else "body"
        if not section:
            section = "body"
            text = f"{block['heading_line']}\n{text}".strip()
        if not text:
            continue
        moments.append(
            _make_moment(
                bucket_id=bucket_id,
                section=section,
                text=text,
                ordinal=ordinal,
                source="content",
                source_id=f"{section}-{block_index}",
                metadata=base_meta,
                source_ref=_source_ref_for_block(block, source_base),
                created_at=str(base_meta.get("bucket_created") or ""),
                updated_at=updated_at,
                relevance_options=relevance_options,
                annotation_options=annotation_options,
            )
        )
        ordinal += 1
    return moments


def _split_markdown_blocks(content: str) -> list[dict]:
    lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks = []
    current = {"heading": "", "heading_line": "", "heading_line_no": 0, "start_line": 1, "lines": []}
    for line_no, line in enumerate(lines, start=1):
        match = HEADING_RE.match(line)
        if match:
            if current["heading"] or any(str(item).strip() for item in current["lines"]):
                blocks.append(_block_from_split_state(current, line_no - 1))
            current = {
                "heading": match.group(2),
                "heading_line": line,
                "heading_line_no": line_no,
                "start_line": line_no,
                "lines": [],
            }
        else:
            current["lines"].append(line)
    if current["heading"] or any(str(item).strip() for item in current["lines"]):
        blocks.append(_block_from_split_state(current, len(lines)))
    return blocks


def _block_from_split_state(current: dict, fallback_end_line: int) -> dict:
    heading = str(current.get("heading") or "")
    heading_line = str(current.get("heading_line") or "")
    raw_lines = list(current.get("lines") or [])
    text = "\n".join(raw_lines).strip()
    start_line = _safe_int(current.get("start_line"), 1)
    end_line = max(start_line, int(fallback_end_line))
    text_start_line = start_line
    if heading:
        nonblank = [
            index
            for index, line in enumerate(raw_lines, start=start_line + 1)
            if str(line).strip()
        ]
        if nonblank:
            text_start_line = nonblank[0]
            end_line = nonblank[-1]
    else:
        nonblank = [
            index
            for index, line in enumerate(raw_lines, start=start_line)
            if str(line).strip()
        ]
        if nonblank:
            start_line = nonblank[0]
            text_start_line = start_line
            end_line = nonblank[-1]
    return {
        "heading": heading,
        "heading_line": heading_line,
        "text": text,
        "start_line": start_line,
        "end_line": end_line,
        "text_start_line": text_start_line,
    }


def _current_moment_preview(moment: dict) -> dict:
    text = str(moment.get("text") or "")
    return {
        "moment_id": str(moment.get("moment_id") or ""),
        "section": str(moment.get("section") or ""),
        "ordinal": int(moment.get("ordinal") or 0),
        "source": str(moment.get("source") or ""),
        "source_id": str(moment.get("source_id") or ""),
        "chars": len(text),
        "text_preview": _clip_text(text, 180),
        "source_ref": _moment_source_ref(moment),
    }


def _moment_source_ref(moment: dict) -> dict | None:
    metadata = moment.get("metadata") if isinstance(moment.get("metadata"), dict) else {}
    source_ref = metadata.get("source_ref")
    return dict(source_ref) if isinstance(source_ref, dict) else None


def _content_moment_text_start_lines(bucket: dict) -> dict[str, int]:
    source_base = _source_ref_base(bucket)
    if not source_base:
        return {}

    content = str(bucket.get("content") or "")
    blocks = _split_markdown_blocks(content)
    offset = int(source_base["content_start_line"]) - 1
    if not any(_canonical_section(block["heading"]) for block in blocks if block["heading"]):
        lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        first_nonblank = next(
            (line_no for line_no, line in enumerate(lines, start=1) if str(line).strip()),
            1,
        )
        return {"body": offset + first_nonblank}

    starts: dict[str, int] = {}
    for block_index, block in enumerate(blocks):
        heading = block["heading"]
        text = str(block["text"] or "").strip()
        canonical_section = _canonical_section(heading) if heading else "body"
        section = canonical_section or "body"
        if not canonical_section and heading:
            text = f"{block['heading_line']}\n{text}".strip()
        if not text:
            continue
        relative_line = (
            _safe_int(block.get("text_start_line"), _safe_int(block.get("start_line"), 1))
            if canonical_section
            else _safe_int(block.get("start_line"), 1)
        )
        starts[f"{section}-{block_index}"] = offset + relative_line
    return starts


def _split_shadow_moment_text(
    text: str,
    target_chars: int,
    max_chars: int,
    min_tail_chars: int,
) -> list[dict]:
    units = _shadow_text_units(text, max_chars)
    if not units:
        return [{"text": text, "start_line": 1, "end_line": 1}]

    packed: list[list[dict]] = []
    current: list[dict] = []
    for unit in units:
        candidate = [*current, unit]
        if current and (
            len(_render_shadow_units(current)) >= target_chars
            or len(_render_shadow_units(candidate)) > max_chars
        ):
            packed.append(current)
            current = [unit]
        else:
            current = candidate
    if current:
        packed.append(current)

    if len(packed) > 1 and len(_render_shadow_units(packed[-1])) < min_tail_chars:
        merged = [*packed[-2], *packed[-1]]
        if len(_render_shadow_units(merged)) <= max_chars:
            packed[-2:] = [merged]

    return [
        {
            "text": _render_shadow_units(chunk_units),
            "start_line": int(chunk_units[0]["line"]),
            "end_line": int(chunk_units[-1]["line"]),
        }
        for chunk_units in packed
    ]


def _shadow_text_units(text: str, max_chars: int) -> list[dict]:
    lines = str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    units: list[dict] = []
    previous_line = 0
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        fragments = _sentence_fragments(line) if len(line) > max_chars else [line]
        for fragment_index, fragment in enumerate(fragments):
            if not fragment:
                continue
            pieces = [fragment[index : index + max_chars] for index in range(0, len(fragment), max_chars)]
            for piece_index, piece in enumerate(pieces):
                separator = ""
                if fragment_index == 0 and piece_index == 0 and previous_line:
                    separator = "\n" * max(1, line_no - previous_line)
                units.append({"text": piece, "separator": separator, "line": line_no})
        previous_line = line_no
    return units


def _sentence_fragments(line: str) -> list[str]:
    fragments = []
    start = 0
    for match in SENTENCE_END_RE.finditer(line):
        end = match.end()
        if end > start:
            fragments.append(line[start:end])
        start = end
    if start < len(line):
        fragments.append(line[start:])
    return fragments or [line]


def _render_shadow_units(units: list[dict]) -> str:
    if not units:
        return ""
    parts = [str(units[0]["text"])]
    for unit in units[1:]:
        parts.append(str(unit.get("separator") or ""))
        parts.append(str(unit["text"]))
    return "".join(parts).strip()


def _shadow_chunk_source_ref(
    parent_ref: dict | None,
    text_start_line: int | None,
    chunk: dict,
) -> dict | None:
    if not parent_ref:
        return None
    source_ref = dict(parent_ref)
    if text_start_line is None:
        return source_ref
    source_ref["start_line"] = text_start_line + int(chunk["start_line"]) - 1
    source_ref["end_line"] = text_start_line + int(chunk["end_line"]) - 1
    return source_ref


def _make_edge(
    source: dict,
    target: dict,
    relation_type: str,
    confidence: float,
    reason: str,
    created_at: str,
) -> dict:
    return {
        "source": source["moment_id"],
        "target": target["moment_id"],
        "bucket_id": source["bucket_id"],
        "relation_type": relation_type,
        "confidence": confidence,
        "reason": reason,
        "created_at": created_at,
    }


def _first_content_moment(moments: list[dict]) -> dict | None:
    # New Scene is canonical. For legacy buckets prefer their authored body over
    # a possibly dehydrated `### moment`; the old heading remains a search hint.
    for section in ("scene", "body", "original", "moment", "fact", "evidence_context", "context"):
        for moment in moments:
            if moment.get("section") == section:
                return moment
    return moments[0] if moments else None


def _dedupe_edges(edges: list[dict]) -> list[dict]:
    deduped: dict[tuple[str, str, str], dict] = {}
    for edge in edges:
        key = (edge["source"], edge["target"], edge["relation_type"])
        existing = deduped.get(key)
        if not existing or float(edge.get("confidence", 0.0)) > float(existing.get("confidence", 0.0)):
            deduped[key] = edge
    return list(deduped.values())


def _canonical_section(heading: str) -> str:
    raw = _clean_text(heading).lower()
    if not raw:
        return ""
    cleaned = re.sub(r"^[\d.\-\s\u3001]+", "", raw)
    cleaned = re.split(r"[:\uff1a(/|\s]", cleaned, maxsplit=1)[0].strip()
    direct = SECTION_ALIASES.get(raw) or SECTION_ALIASES.get(cleaned)
    if direct:
        return direct

    compact = re.sub(r"[\s_\-·:：/|（）()【】\[\]]+", "", cleaned)
    if ("喜欢" in compact and ("原因" in compact or "为什么" in compact)) or (
        "favorite" in compact and ("reason" in compact or "why" in compact)
    ):
        return "favorite_reason"
    if ("affect" in compact and "anchor" in compact) or ("情感" in compact and "锚" in compact):
        return "affect_anchor"
    return ""


def _make_moment(
    *,
    bucket_id: str,
    section: str,
    text: str,
    ordinal: int,
    source: str,
    source_id: str,
    metadata: dict,
    source_ref: dict | None = None,
    created_at: str,
    updated_at: str,
    relevance_options: MemoryRelevanceOptions | None = None,
    annotation_options: dict | None = None,
) -> dict:
    text = _clean_text(text)
    metadata = _clean_metadata(metadata)
    if source_ref:
        metadata = _clean_metadata({**metadata, "source_ref": source_ref})
    metadata = _annotated_metadata(
        text,
        section,
        metadata,
        relevance_options or memory_relevance_options_from_config(),
        annotation_options or DEFAULT_ANNOTATION_OPTIONS,
    )
    moment_id = _moment_id(bucket_id, source, section, ordinal, source_id)
    metadata_json = json.dumps(metadata, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return {
        "moment_id": moment_id,
        "bucket_id": bucket_id,
        "section": section,
        "text": text,
        "ordinal": int(ordinal),
        "source": source,
        "source_id": str(source_id or ""),
        "text_hash": _sha1(text),
        "metadata": metadata,
        "metadata_json": metadata_json,
        "created_at": str(created_at or ""),
        "updated_at": updated_at,
    }


def _bucket_metadata(meta: dict, bucket: dict) -> dict:
    tags = _list_text(meta.get("tags"))
    content = str(bucket.get("content") or "")
    favorite_tags = favorite_policy_tags(tags)
    return _clean_metadata(
        {
            "bucket_name": meta.get("name") or bucket.get("name") or "",
            "bucket_type": meta.get("type"),
            "bucket_tags": tags,
            "bucket_domain": _list_text(meta.get("domain")),
            "bucket_scene_cues": _list_text(meta.get("scene_cues")),
            "bucket_valence": meta.get("valence"),
            "bucket_arousal": meta.get("arousal"),
            "bucket_anchor": meta.get("anchor"),
            "bucket_pinned": meta.get("pinned"),
            "bucket_protected": meta.get("protected"),
            "bucket_resolved": meta.get("resolved"),
            "bucket_digested": meta.get("digested"),
            "bucket_memory_subject": meta.get("memory_subject"),
            "bucket_memory_layer": meta.get("memory_layer"),
            "bucket_memory_classification_source": meta.get("memory_classification_source"),
            "bucket_favorite": bool(favorite_tags),
            "bucket_favorite_tags": favorite_tags,
            "bucket_has_affect_anchor": "### affect_anchor" in content,
            "bucket_date": meta.get("date"),
            "bucket_created": meta.get("created"),
            "bucket_updated_at": meta.get("updated_at"),
            "bucket_path": bucket.get("path"),
        }
    )


def _source_ref_base(bucket: dict) -> dict | None:
    meta = bucket.get("metadata") if isinstance(bucket.get("metadata"), dict) else {}
    path = str(bucket.get("path") or meta.get("path") or meta.get("bucket_path") or "").strip()
    if not path:
        return None
    start_line = _safe_int(
        bucket.get("content_start_line") or meta.get("content_start_line"),
        DEFAULT_CONTENT_START_LINE,
    )
    return {
        "path": path,
        "content_start_line": max(1, start_line),
        "source": "bucket_content",
    }


def _source_ref_for_body(content: str, source_base: dict | None) -> dict | None:
    if not source_base:
        return None
    line_count = max(1, len(str(content or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")))
    start = int(source_base["content_start_line"])
    return {
        "path": source_base["path"],
        "content_start_line": start,
        "start_line": start,
        "end_line": start + line_count - 1,
        "source": source_base.get("source") or "bucket_content",
    }


def _source_ref_for_block(block: dict, source_base: dict | None) -> dict | None:
    if not source_base:
        return None
    start_line = _safe_int(block.get("start_line"), 1)
    end_line = _safe_int(block.get("end_line"), start_line)
    offset = int(source_base["content_start_line"]) - 1
    content_start_line = int(source_base["content_start_line"])
    return {
        "path": source_base["path"],
        "content_start_line": content_start_line,
        "start_line": max(1, offset + start_line),
        "end_line": max(1, offset + end_line),
        "source": source_base.get("source") or "bucket_content",
    }


def _annotated_metadata(
    text: str,
    section: str,
    metadata: dict,
    relevance_options: MemoryRelevanceOptions,
    annotation_options: dict,
) -> dict:
    if not _bool_value(annotation_options.get("enabled", True), True):
        return metadata

    max_summary_chars = _int_between(annotation_options.get("max_summary_chars", 160), 160, 40, 500)
    max_spans = _int_between(annotation_options.get("max_evidence_spans", 3), 3, 0, 10)
    max_evidence_chars = _int_between(annotation_options.get("max_evidence_chars", 120), 120, 30, 500)
    annotated = dict(metadata)
    summary = _moment_summary(text, max_summary_chars)
    if summary:
        annotated.setdefault("annotation_summary", summary)
        annotated.setdefault("summary", summary)

    node = {
        "section": section,
        "text": text,
        "metadata": annotated,
    }
    facets = facets_for_node(node, relevance_options)
    active = {}
    for facet, score in facets.items():
        safe_score = _safe_float(score)
        if safe_score is not None and safe_score >= 0.3:
            active[facet] = round(safe_score, 3)
    if active:
        annotated.setdefault("annotation_facets", active)
        evidence_spans = _evidence_spans_for_facets(text, active, relevance_options, max_spans, max_evidence_chars)
        if evidence_spans:
            annotated.setdefault("evidence_spans", evidence_spans)
    return _clean_metadata(annotated)


def _moment_summary(text: str, max_chars: int) -> str:
    compact = " ".join(_clean_text(text).split())
    if not compact:
        return ""
    sentences = re.split(r"(?<=[。！？!?；;])\s*", compact)
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            return _clip_text(sentence, max_chars)
    return _clip_text(compact, max_chars)


def _evidence_spans_for_facets(
    text: str,
    facets: dict[str, float],
    relevance_options: MemoryRelevanceOptions,
    max_spans: int,
    max_chars: int,
) -> list[dict]:
    if max_spans <= 0:
        return []
    compact = " ".join(_clean_text(text).split())
    if not compact:
        return []
    spans = []
    for facet, _score in sorted(facets.items(), key=lambda item: item[1], reverse=True):
        aliases = relevance_options.aliases.get(facet, ())
        span = _best_evidence_span(compact, aliases, max_chars)
        if not span:
            continue
        item = {"facet": facet, "text": span}
        if item not in spans:
            spans.append(item)
        if len(spans) >= max_spans:
            break
    if not spans:
        spans.append({"facet": "summary", "text": _clip_text(compact, max_chars)})
    return spans


def _best_evidence_span(text: str, aliases: tuple[str, ...], max_chars: int) -> str:
    text_lower = text.lower()
    for alias in sorted((str(item) for item in aliases or ()), key=len, reverse=True):
        alias = alias.strip()
        if not alias:
            continue
        index = text_lower.find(alias.lower())
        if index < 0:
            continue
        start = max(0, index - max_chars // 3)
        end = min(len(text), index + len(alias) + max_chars // 2)
        return _clip_text(text[start:end].strip(), max_chars)
    return ""


def _join_evidence_spans(raw: Any) -> str:
    if not isinstance(raw, list):
        return ""
    parts = []
    for item in raw:
        if isinstance(item, dict):
            text = item.get("text") or item.get("span") or ""
        else:
            text = item
        if str(text).strip():
            parts.append(str(text))
    return " ".join(parts)


def _clean_metadata(metadata: dict) -> dict:
    cleaned = {}
    for key, value in (metadata or {}).items():
        cleaned_value = _clean_metadata_value(value)
        if cleaned_value is not None:
            cleaned[key] = cleaned_value
    return cleaned


def _clean_metadata_value(value: Any):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        cleaned = {
            str(key): cleaned_value
            for key, item in value.items()
            if (cleaned_value := _clean_metadata_value(item)) is not None
        }
        return cleaned
    if isinstance(value, (list, tuple)):
        return [
            cleaned_item
            for item in value
            if (cleaned_item := _clean_metadata_value(item)) is not None
        ]
    return str(value)


def _bucket_id(bucket: dict) -> str:
    meta = bucket.get("metadata") if isinstance(bucket.get("metadata"), dict) else {}
    bucket_id = str(bucket.get("id") or meta.get("id") or "").strip()
    if not bucket_id:
        raise ValueError("bucket id is required")
    return bucket_id


def _list_text(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    if value:
        return [str(value)]
    return []


def _moment_query_score(
    moment: dict,
    query: str,
    relevance_options: MemoryRelevanceOptions | None = None,
    *,
    query_terms: list[str] | None = None,
    expanded_query_terms: list[str] | None = None,
) -> float:
    query = str(query or "").strip()
    if not query:
        return 0.0
    text = str(moment.get("text") or "")
    meta = moment.get("metadata", {}) if isinstance(moment.get("metadata"), dict) else {}
    fields = " ".join(
        [
            text,
            str(meta.get("annotation_summary") or ""),
            _join_evidence_spans(meta.get("evidence_spans")),
            " ".join(str(facet) for facet in (meta.get("annotation_facets") or {}).keys())
            if isinstance(meta.get("annotation_facets"), dict)
            else "",
            str(meta.get("bucket_name") or ""),
            " ".join(_list_text(meta.get("bucket_tags"))),
            " ".join(_list_text(meta.get("bucket_domain"))),
        ]
    ).lower()
    query_lower = query.lower()
    score = 0.0
    if _term_matches_fields(query_lower, fields):
        score += 0.65
    terms = query_terms if query_terms is not None else content_terms_for_query(query, relevance_options)
    if terms:
        matched = sum(1 for term in terms if _term_matches_fields(term.lower(), fields))
        score += min(0.5, matched / max(1, len(terms)) * 0.5)
    expanded_terms = (
        expanded_query_terms
        if expanded_query_terms is not None
        else _expanded_query_terms(query, relevance_options)
    )
    if expanded_terms:
        matched_expanded = sum(
            1 for term in expanded_terms
            if _term_matches_fields(term.lower(), fields)
        )
        if matched_expanded:
            score += min(0.38, matched_expanded / max(1, len(expanded_terms)) * 0.38)
    if score <= 0:
        return 0.0
    score *= _moment_section_weight(moment.get("section"))
    if meta.get("bucket_favorite") or meta.get("bucket_anchor"):
        score += 0.06
    return round(min(score, 1.5), 4)


def _query_terms(query: str) -> list[str]:
    raw = str(query or "").strip()
    terms = [part for part in re.split(r"[\s,，。！？!?;；:：/\\|]+", raw) if part]
    terms.extend(re.findall(r"[A-Za-z0-9_\-]+|[\u4e00-\u9fff]{1,}", raw))
    seen = set()
    unique = []
    for term in terms:
        key = term.lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(term)
    return unique


def _expanded_query_terms(
    query: str,
    relevance_options: MemoryRelevanceOptions | None = None,
) -> list[str]:
    return expanded_terms_for_query(query, relevance_options)


def _term_matches_fields(term: str, fields: str) -> bool:
    term = str(term or "").lower()
    fields = str(fields or "").lower()
    if not term:
        return False
    if re.fullmatch(r"[a-z0-9_]", term):
        return re.search(rf"(?<![a-z0-9_]){re.escape(term)}(?![a-z0-9_])", fields) is not None
    return term in fields


def _moment_section_weight(section: Any) -> float:
    return {
        "scene": 1.12,
        "body": 1.08,
        "original": 1.02,
        "moment": 0.96,
        "fact": 0.94,
        "context": 0.95,
        "evidence_context": 0.94,
        "reflection": 0.9,
        "feeling": 0.9,
        "followup": 0.88,
        "followup_log": 0.6,
        "affect_anchor": 0.82,
        "favorite_reason": 0.82,
        "comment": 0.78,
    }.get(str(section or ""), 0.85)


def _metadata_float(meta: dict, key: str, default: float) -> float:
    try:
        return float(meta.get(key, default))
    except (TypeError, ValueError):
        return default


def _safe_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp_float(value: Any, low: float, high: float) -> float:
    parsed = _safe_float(value)
    if parsed is None:
        parsed = low
    return max(low, min(high, parsed))


def _clip_text(text: str, max_chars: int) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rstrip() + "..."


def _bool_value(value: Any, default: bool = False) -> bool:
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


def _clean_text(value: Any) -> str:
    return strip_wikilinks(str(value or "")).strip()


def _moment_id(bucket_id: str, source: str, section: str, ordinal: int, source_id: str) -> str:
    digest = _sha1(f"{bucket_id}|{source}|{section}|{ordinal}|{source_id}")[:16]
    return f"{bucket_id}:{digest}"


def _sha1(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()
