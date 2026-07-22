from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import unicodedata
from datetime import datetime, timezone
from typing import Any

from identity import identity_names
from query_terms import GENERIC_LEXICAL_STOPWORDS
from utils import strip_wikilinks


RETRIEVAL_ALIAS_SECTIONS = frozenset({"scene", "body", "moment", "fact", "original"})
MAX_RETRIEVAL_ALIASES_PER_BUCKET = 24
MAX_RETRIEVAL_ALIASES_PER_MOMENT = 4
MAX_EXPLICIT_RETRIEVAL_ALIASES_PER_BUCKET = 12
MAX_RETRIEVAL_ALIAS_CHARS = 72
GENERIC_RETRIEVAL_ALIAS_KEYS = frozenset(
    {
        "memory",
        "memories",
        "moment",
        "scene",
        "moments",
        "fact",
        "facts",
        "original",
        "record",
        "records",
        "conversation",
        "conversations",
        "daily",
        "game",
        "games",
        "note",
        "notes",
        "momentbucket",
        "事情",
        "事实",
        "哥哥",
        "今天",
        "以前",
        "原文",
        "对话",
        "我们",
        "日常",
        "游戏",
        "记录",
        "记忆",
        "片段",
    }
)


def _compact_retrieval_alias_patterns(identity_terms: Any = ()) -> tuple[re.Pattern, ...]:
    subjects = ["哥哥", "我", "我们", "她", "他", "i", "we", "she", "he"]
    subjects.extend(str(term or "").strip() for term in (identity_terms or ()))
    subject_pattern = "|".join(
        sorted(
            {re.escape(term) for term in subjects if term},
            key=len,
            reverse=True,
        )
    )
    return (
        re.compile(
            rf"^(?:{subject_pattern})(?:和|与)(?:{subject_pattern})"
            r"(?:关于|有关)?(.+?)(?:的)?"
            r"(?:约定|对话|记忆|记录|事情|片段)$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^(?:关于|有关)(.+?)(?:的)?"
            r"(?:约定|对话|记忆|记录|事情|片段)?$",
            re.IGNORECASE,
        ),
        re.compile(
            rf"^(?:{subject_pattern})"
            r"(?:曾经|当时|后来|现在|一直)?"
            r"(?:说过|说|觉得|认为|记得|希望|想要|想|决定|约定|喜欢|提到)"
            r"[\s,，:：]*(.+)$",
            re.IGNORECASE,
        ),
        re.compile(
            rf"^(?:{subject_pattern})\s+"
            r"(?:said|says|thought|thinks|wanted|wants|remembered|remembers|agreed|decided|mentioned)"
            r"\s+(?:that\s+)?(.+)$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^(?:memory|moment|note|record|conversation)\s+(?:about|of)\s+(.+)$",
            re.IGNORECASE,
        ),
        re.compile(r"^(?:about|regarding)\s+(.+)$", re.IGNORECASE),
    )


DEFAULT_COMPACT_RETRIEVAL_ALIAS_PATTERNS = _compact_retrieval_alias_patterns()


def _clean_text(value: Any) -> str:
    return strip_wikilinks(str(value or "")).strip()


def _clean_retrieval_alias_text(value: Any) -> str:
    text = _clean_text(value)
    text = re.sub(r"^(?:(?:#{1,6}|[-*+]|>)\s*)+", "", text).strip()
    return text.strip(
        " \t\r\n`\"'.,!?;:()[]{}"
        "\u2018\u2019\u201c\u201d\u3001\u3002\uff01\uff1f\uff0c\uff1b\uff1a"
        "\uff08\uff09\u3010\u3011"
    )


def _retrieval_alias_key(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).lower()
    return re.sub(r"[\W_]+", "", normalized, flags=re.UNICODE)


GENERIC_RETRIEVAL_ALIAS_STOP_KEYS = frozenset(
    {
        *GENERIC_RETRIEVAL_ALIAS_KEYS,
        *(
            _retrieval_alias_key(term)
            for term in GENERIC_LEXICAL_STOPWORDS
            if _retrieval_alias_key(term)
        ),
    }
)


def _retrieval_alias_is_date(value: str) -> bool:
    text = str(value or "").strip()
    return bool(
        re.fullmatch(
            r"\d{4}(?:[-/.]\d{1,2}){1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?",
            text,
        )
        or re.fullmatch(r"\d{4}年\d{1,2}月(?:\d{1,2}日)?", text)
        or re.fullmatch(r"\d{8}", text)
    )


def _retrieval_alias_is_identifier(value: str) -> bool:
    text = str(value or "").strip().lower()
    compact = re.sub(r"[-_:{}\s]", "", text)
    if compact.isdigit():
        return True
    if re.fullmatch(r"[0-9a-f]{8,64}", compact):
        return True
    if re.fullmatch(r"(?:id|uuid|bucket|moment|comment)[-_: #]*[a-z0-9-]+", text):
        return True
    return bool(
        re.fullmatch(r"[a-z]+[-_]?[a-z0-9_-]*\d[a-z0-9_-]*", text)
        and len(compact) >= 12
    )


def _valid_retrieval_alias(
    alias_text: str,
    *,
    stop_keys: frozenset[str] = GENERIC_RETRIEVAL_ALIAS_STOP_KEYS,
) -> bool:
    alias_key = _retrieval_alias_key(alias_text)
    if len(alias_key) < 3 or len(alias_text) > MAX_RETRIEVAL_ALIAS_CHARS:
        return False
    if alias_key in stop_keys:
        return False
    if len(re.findall(r"[A-Za-z0-9]+", alias_text)) > 14:
        return False
    if _retrieval_alias_is_date(alias_text) or _retrieval_alias_is_identifier(alias_text):
        return False
    return True


def _retrieval_alias_variants(
    value: Any,
    *,
    patterns: tuple[re.Pattern, ...] = DEFAULT_COMPACT_RETRIEVAL_ALIAS_PATTERNS,
) -> list[str]:
    base = _clean_retrieval_alias_text(value)
    if not base:
        return []
    variants = [base]
    seen = {_retrieval_alias_key(base)}
    queue = [(base, 0)]
    while queue:
        current, depth = queue.pop(0)
        for pattern in patterns:
            match = pattern.match(current)
            if not match:
                continue
            compact = _clean_retrieval_alias_text(match.group(1))
            key = _retrieval_alias_key(compact)
            if not key or key in seen:
                continue
            seen.add(key)
            variants.append(compact)
            if depth < 1:
                queue.append((compact, depth + 1))
    return variants


def _retrieval_phrase_candidates(value: Any) -> list[str]:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    candidates: list[str] = []
    for line in text.split("\n"):
        fragments = [
            fragment
            for fragment in re.split(r"(?<=[\u3002\uff01\uff1f\uff1b!?;])\s*", line)
            if _clean_retrieval_alias_text(fragment)
        ]
        candidates.extend([line] if len(fragments) <= 1 else fragments)

    unique: list[str] = []
    seen = set()
    for candidate in candidates:
        cleaned = _clean_retrieval_alias_text(candidate)
        key = _retrieval_alias_key(cleaned)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(cleaned)
    return unique


def _retrieval_alias_query_terms(
    query: Any,
    *,
    patterns: tuple[re.Pattern, ...] = DEFAULT_COMPACT_RETRIEVAL_ALIAS_PATTERNS,
    stop_keys: frozenset[str] = GENERIC_RETRIEVAL_ALIAS_STOP_KEYS,
) -> list[tuple[str, str]]:
    cleaned = _clean_retrieval_alias_text(query)
    if not cleaned:
        return []
    candidates = _retrieval_alias_variants(cleaned, patterns=patterns)
    candidates.extend(
        part
        for part in re.split(r"[\s,\uff0c\u3002\uff01\uff1f!?;\uff1b:\uff1a/\\|]+", cleaned)
        if part
    )

    terms: list[tuple[str, str]] = []
    seen = set()
    for candidate in candidates:
        text = _clean_retrieval_alias_text(candidate)
        key = _retrieval_alias_key(text)
        if len(key) < 2 or key in seen or key in stop_keys:
            continue
        seen.add(key)
        terms.append((text, key))
    return terms[:8]


def _build_retrieval_aliases(
    bucket_id: str,
    bucket_title: str,
    items: list[dict],
    *,
    explicit_aliases: Any = (),
    patterns: tuple[re.Pattern, ...] = DEFAULT_COMPACT_RETRIEVAL_ALIAS_PATTERNS,
    stop_keys: frozenset[str] | None = None,
) -> list[dict]:
    stop_keys = stop_keys or GENERIC_RETRIEVAL_ALIAS_STOP_KEYS
    updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    aliases: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    def add_alias(text: str, source: str, item_id: str) -> bool:
        alias_text = _clean_retrieval_alias_text(text)
        if not _valid_retrieval_alias(alias_text, stop_keys=stop_keys):
            return False
        alias_key = _retrieval_alias_key(alias_text)
        identity = (item_id, alias_key, source)
        if identity in seen:
            return False
        seen.add(identity)
        aliases.append(
            {
                "bucket_id": bucket_id,
                # The column name remains moment_id for on-disk compatibility;
                # canonical Scene rows carry their typed recall node id here.
                "moment_id": item_id,
                "alias_text": alias_text,
                "alias_key": alias_key,
                "source": source,
                "text_hash": hashlib.sha1(alias_text.encode("utf-8")).hexdigest(),
                "updated_at": updated_at,
            }
        )
        return True

    for variant in _retrieval_alias_variants(bucket_title, patterns=patterns):
        if len(aliases) >= MAX_RETRIEVAL_ALIASES_PER_BUCKET:
            return aliases
        add_alias(variant, "title", "")

    ordered_items = sorted(
        [item for item in items if item.get("section") in RETRIEVAL_ALIAS_SECTIONS],
        key=lambda item: int(item.get("ordinal", 0)),
    )
    explicit_item_id = str(
        (
            ordered_items[0].get("node_id")
            or ordered_items[0].get("moment_id")
        )
        if ordered_items
        else ""
    )
    explicit_count = 0
    for value in explicit_aliases or ():
        for variant in _retrieval_alias_variants(value, patterns=patterns):
            if len(aliases) >= MAX_RETRIEVAL_ALIASES_PER_BUCKET:
                return aliases
            if add_alias(variant, "moment", explicit_item_id):
                explicit_count += 1
            if explicit_count >= MAX_EXPLICIT_RETRIEVAL_ALIASES_PER_BUCKET:
                break
        if explicit_count >= MAX_EXPLICIT_RETRIEVAL_ALIASES_PER_BUCKET:
            break

    for item in ordered_items:
        item_id = str(item.get("node_id") or item.get("moment_id") or "")
        if not item_id:
            continue
        added_for_item = 0
        for phrase in _retrieval_phrase_candidates(item.get("text")):
            for variant in _retrieval_alias_variants(phrase, patterns=patterns):
                if len(aliases) >= MAX_RETRIEVAL_ALIASES_PER_BUCKET:
                    return aliases
                if add_alias(variant, "moment", item_id):
                    added_for_item += 1
                if added_for_item >= MAX_RETRIEVAL_ALIASES_PER_MOMENT:
                    break
            if added_for_item >= MAX_RETRIEVAL_ALIASES_PER_MOMENT:
                break
    return aliases


class MemoryRetrievalAliasIndex:
    """Derived title/content aliases shared by Scene and legacy recall sources."""

    def __init__(self, config: dict, *, db_path: str, create: bool = True):
        names = identity_names(config or {})
        self.stop_keys = frozenset(
            {
                *GENERIC_RETRIEVAL_ALIAS_STOP_KEYS,
                *(
                    _retrieval_alias_key(term)
                    for term in names["relationship_terms"]
                    if _retrieval_alias_key(term)
                ),
            }
        )
        self.patterns = _compact_retrieval_alias_patterns(names["relationship_terms"])
        self.db_path = str(db_path)
        if create:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            conn = self._connect()
            try:
                self.ensure_schema(conn)
                conn.commit()
            finally:
                conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def ensure_schema(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_retrieval_aliases (
                bucket_id TEXT NOT NULL,
                moment_id TEXT NOT NULL DEFAULT '',
                alias_text TEXT NOT NULL,
                alias_key TEXT NOT NULL,
                source TEXT NOT NULL CHECK(source IN ('title', 'moment')),
                text_hash TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(bucket_id, moment_id, alias_key, source)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memory_retrieval_aliases_alias_key
            ON memory_retrieval_aliases(alias_key, bucket_id)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memory_retrieval_aliases_bucket
            ON memory_retrieval_aliases(bucket_id)
            """
        )

    def build_rows(
        self,
        bucket_id: str,
        bucket_title: str,
        items: list[dict],
        *,
        explicit_aliases: Any = (),
    ) -> list[dict]:
        return _build_retrieval_aliases(
            bucket_id,
            bucket_title,
            items,
            explicit_aliases=explicit_aliases,
            patterns=self.patterns,
            stop_keys=self.stop_keys,
        )

    @staticmethod
    def _row_signature(row: dict) -> tuple[str, str, str, str, str, str]:
        return (
            str(row.get("bucket_id") or ""),
            str(row.get("moment_id") or ""),
            str(row.get("alias_text") or ""),
            str(row.get("alias_key") or ""),
            str(row.get("source") or ""),
            str(row.get("text_hash") or ""),
        )

    def bucket_matches(
        self,
        conn: sqlite3.Connection,
        bucket_id: str,
        desired_rows: list[dict],
    ) -> bool:
        current = conn.execute(
            "SELECT * FROM memory_retrieval_aliases WHERE bucket_id = ?",
            (str(bucket_id),),
        ).fetchall()
        current_signatures = sorted(self._row_signature(dict(row)) for row in current)
        desired_signatures = sorted(self._row_signature(row) for row in desired_rows)
        return current_signatures == desired_signatures

    def replace_rows(
        self,
        conn: sqlite3.Connection,
        bucket_id: str,
        rows: list[dict],
    ) -> int:
        conn.execute(
            "DELETE FROM memory_retrieval_aliases WHERE bucket_id = ?",
            (str(bucket_id),),
        )
        conn.executemany(
            """
            INSERT INTO memory_retrieval_aliases
            (bucket_id, moment_id, alias_text, alias_key, source, text_hash, updated_at)
            VALUES (:bucket_id, :moment_id, :alias_text, :alias_key, :source, :text_hash, :updated_at)
            """,
            rows,
        )
        return len(rows)

    def replace_bucket(
        self,
        conn: sqlite3.Connection,
        bucket_id: str,
        bucket_title: str,
        items: list[dict],
        *,
        explicit_aliases: Any = (),
    ) -> int:
        rows = self.build_rows(
            bucket_id,
            bucket_title,
            items,
            explicit_aliases=explicit_aliases,
        )
        return self.replace_rows(conn, bucket_id, rows)

    def delete_bucket(self, conn: sqlite3.Connection, bucket_id: str) -> int:
        cursor = conn.execute(
            "DELETE FROM memory_retrieval_aliases WHERE bucket_id = ?",
            (str(bucket_id),),
        )
        return max(0, int(cursor.rowcount or 0))

    def list_for_bucket(self, bucket_id: str, limit: int = 100) -> list[dict]:
        normalized = str(bucket_id or "").strip()
        if not normalized:
            return []
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT * FROM memory_retrieval_aliases
                WHERE bucket_id = ?
                ORDER BY source ASC, moment_id ASC, alias_key ASC
                LIMIT ?
                """,
                (normalized, max(1, int(limit))),
            ).fetchall()
        finally:
            conn.close()
        return [dict(row) for row in rows]

    def search(self, query: str, limit: int = 20) -> list[dict]:
        query_terms = _retrieval_alias_query_terms(
            query,
            patterns=self.patterns,
            stop_keys=self.stop_keys,
        )
        if not query_terms:
            return []

        conditions = ["a.alias_key LIKE ?" for _ in query_terms]
        params: list[Any] = [f"%{key}%" for _, key in query_terms]
        full_query_key = _retrieval_alias_key(query)
        if full_query_key:
            conditions.append("? LIKE '%' || a.alias_key || '%'")
            params.append(full_query_key)

        conn = self._connect()
        try:
            rows = conn.execute(
                f"""
                SELECT a.*, counts.bucket_count
                FROM memory_retrieval_aliases AS a
                JOIN (
                    SELECT alias_key, COUNT(DISTINCT bucket_id) AS bucket_count
                    FROM memory_retrieval_aliases
                    GROUP BY alias_key
                ) AS counts ON counts.alias_key = a.alias_key
                WHERE {' OR '.join(conditions)}
                """,
                params,
            ).fetchall()
        finally:
            conn.close()

        results = []
        for row in rows:
            alias = dict(row)
            alias_key = str(alias.get("alias_key") or "")
            if alias_key in self.stop_keys:
                continue
            matched_terms = [
                text
                for text, key in query_terms
                if key in alias_key or alias_key in key
            ]
            if not matched_terms:
                continue
            matched_keys = {
                key for _, key in query_terms if key in alias_key or alias_key in key
            }
            coverage = len(matched_keys) / max(1, len({key for _, key in query_terms}))
            if full_query_key == alias_key:
                score = 1.0
            elif full_query_key and (full_query_key in alias_key or alias_key in full_query_key):
                score = 0.92
            else:
                specificity = max(len(key) for key in matched_keys) / max(1, len(alias_key))
                score = min(0.9, 0.5 + coverage * 0.28 + min(1.0, specificity) * 0.12)
            results.append(
                {
                    "bucket_id": alias["bucket_id"],
                    "moment_id": alias["moment_id"],
                    "alias_text": alias["alias_text"],
                    "source": alias["source"],
                    "bucket_count": int(alias["bucket_count"] or 0),
                    "score": round(score, 4),
                    "matched_terms": matched_terms,
                }
            )

        results.sort(
            key=lambda item: (
                -float(item["score"]),
                int(item["bucket_count"]),
                0 if item["source"] == "title" else 1,
                item["bucket_id"],
                item["moment_id"],
                item["alias_text"],
            )
        )
        return results[: max(1, int(limit))]
