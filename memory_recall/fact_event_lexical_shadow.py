"""Incremental, rebuildable Fact/Event lexical candidate index."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from typing import Any, Iterable

import jieba


SCHEMA_VERSION = 1
FIELD_WEIGHTS = {"title": 1.5, "atomic_question": 2.0, "body": 1.0}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_term(value: Any) -> str:
    return "".join(re.findall(r"[a-z0-9_.:-]+|[\u4e00-\u9fff]+", str(value or "").lower()))


def lexical_terms(text: str) -> list[str]:
    """Tokenize for search without a hand-written stop-word list."""

    output: list[str] = []
    for raw in jieba.lcut_for_search(str(text or ""), HMM=True):
        term = _normalize_term(raw)
        if not term:
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]", term):
            continue
        if re.fullmatch(r"[a-z]", term):
            continue
        output.append(term)
    return output


def _source_hash(item: dict[str, Any]) -> str:
    payload = {
        "schema": SCHEMA_VERSION,
        "item_id": str(item.get("item_id") or ""),
        "item_type": str(item.get("item_type") or ""),
        "title": str(item.get("title") or ""),
        "atomic_question": str(item.get("atomic_question") or ""),
        "body": str(item.get("body") or ""),
        "importance": int(item.get("importance") or 0),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


class FactEventLexicalShadowIndex:
    """Candidate-only BM25 index over eligible canonical Fact/Event text."""

    def __init__(self, config: dict[str, Any]):
        state_dir = str(
            config.get("state_dir")
            or os.path.join(
                os.path.dirname(os.path.abspath(config.get("buckets_dir", "buckets"))),
                "state",
            )
        )
        raw = config.get("fact_event_lexical_shadow")
        raw = raw if isinstance(raw, dict) else {}
        self.db_path = os.path.join(state_dir, "fact_event_lexical_shadow.sqlite")
        self.k1 = float(raw.get("k1") or 1.2)
        self.b = float(raw.get("b") or 0.75)
        self.max_df_ratio = float(raw.get("max_df_ratio") or 0.2)
        self.max_short_term_df = max(1, int(raw.get("max_short_term_df") or 3))

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS lexical_documents (
                    item_id TEXT PRIMARY KEY,
                    item_type TEXT NOT NULL CHECK(item_type IN ('fact', 'event')),
                    source_hash TEXT NOT NULL,
                    importance INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    atomic_question TEXT NOT NULL,
                    body TEXT NOT NULL,
                    doc_len INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS lexical_terms (
                    item_id TEXT NOT NULL,
                    term TEXT NOT NULL,
                    weighted_tf REAL NOT NULL,
                    fields_json TEXT NOT NULL,
                    PRIMARY KEY(item_id, term),
                    FOREIGN KEY(item_id) REFERENCES lexical_documents(item_id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_lexical_terms_term
                    ON lexical_terms(term);
                CREATE TABLE IF NOT EXISTS lexical_stats (
                    term TEXT PRIMARY KEY,
                    df INTEGER NOT NULL
                );
                """
            )

    @staticmethod
    def _eligible(
        items: Iterable[dict[str, Any]],
        min_importance: int,
    ) -> list[dict[str, Any]]:
        return [
            item
            for item in items
            if str(item.get("item_type") or "") in {"fact", "event"}
            and str(item.get("status") or "active") == "active"
            and int(item.get("importance") or 0) >= min_importance
            and str(item.get("item_id") or "")
            and str(item.get("body") or "").strip()
        ]

    @staticmethod
    def _document_terms(item: dict[str, Any]) -> tuple[int, dict[str, dict[str, Any]]]:
        combined: dict[str, dict[str, Any]] = {}
        doc_len = 0
        for field, weight in FIELD_WEIGHTS.items():
            counts = Counter(lexical_terms(str(item.get(field) or "")))
            doc_len += sum(counts.values())
            for term, count in counts.items():
                row = combined.setdefault(term, {"weighted_tf": 0.0, "fields": []})
                row["weighted_tf"] += float(count) * weight
                row["fields"].append(field)
        return max(1, doc_len), combined

    def sync(
        self,
        items: Iterable[dict[str, Any]],
        *,
        min_importance: int = 3,
        dry_run: bool = False,
        refresh_all: bool = False,
    ) -> dict[str, Any]:
        eligible = self._eligible(items, max(1, int(min_importance or 1)))
        existing: dict[str, sqlite3.Row] = {}
        if os.path.exists(self.db_path):
            with closing(self._connect()) as conn:
                existing = {
                    str(row["item_id"]): row
                    for row in conn.execute("SELECT item_id, source_hash FROM lexical_documents")
                }
        desired_ids = {str(item["item_id"]) for item in eligible}
        stale_ids = sorted(set(existing) - desired_ids)
        pending = [
            item
            for item in eligible
            if refresh_all
            or str(item["item_id"]) not in existing
            or str(existing[str(item["item_id"])]["source_hash"]) != _source_hash(item)
        ]
        result = {
            "status": "dry_run" if dry_run else "ok",
            "min_importance": max(1, int(min_importance or 1)),
            "eligible": len(eligible),
            "to_index": len(pending),
            "stale_rows": len(stale_ids),
            "memory_kinds": {
                kind: sum(1 for item in eligible if item.get("item_type") == kind)
                for kind in ("fact", "event")
            },
        }
        if dry_run:
            return result
        self._init_db()
        with closing(self._connect()) as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            for item in pending:
                item_id = str(item["item_id"])
                doc_len, terms = self._document_terms(item)
                conn.execute("DELETE FROM lexical_terms WHERE item_id=?", (item_id,))
                conn.execute(
                    """
                    INSERT OR REPLACE INTO lexical_documents(
                        item_id, item_type, source_hash, importance, title,
                        atomic_question, body, doc_len, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item_id,
                        str(item.get("item_type") or ""),
                        _source_hash(item),
                        int(item.get("importance") or 0),
                        str(item.get("title") or ""),
                        str(item.get("atomic_question") or ""),
                        str(item.get("body") or ""),
                        doc_len,
                        _now(),
                    ),
                )
                conn.executemany(
                    "INSERT INTO lexical_terms(item_id, term, weighted_tf, fields_json) VALUES (?, ?, ?, ?)",
                    [
                        (item_id, term, row["weighted_tf"], json.dumps(row["fields"]))
                        for term, row in terms.items()
                    ],
                )
            for item_id in stale_ids:
                conn.execute("DELETE FROM lexical_terms WHERE item_id=?", (item_id,))
                conn.execute("DELETE FROM lexical_documents WHERE item_id=?", (item_id,))
            if pending or stale_ids:
                conn.execute("DELETE FROM lexical_stats")
                conn.execute(
                    "INSERT INTO lexical_stats(term, df) SELECT term, COUNT(*) FROM lexical_terms GROUP BY term"
                )
            conn.commit()
        return {**result, "indexed": len(pending), "removed": len(stale_ids)}

    @staticmethod
    def _match_span(term: str, row: sqlite3.Row, fields: list[str]) -> dict[str, Any]:
        for field in ("body", "atomic_question", "title"):
            if field not in fields:
                continue
            text = str(row[field] or "")
            start = text.lower().find(term.lower())
            if start >= 0:
                return {
                    "field": field,
                    "start_offset": start,
                    "end_offset": start + len(term),
                    "text": text[start : start + len(term)],
                }
        return {}

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        memory_kinds: Iterable[str] = ("fact", "event"),
        min_importance: int = 1,
        min_importance_by_kind: dict[str, int] | None = None,
        allowed_memory_ids: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        query_terms = list(dict.fromkeys(lexical_terms(query)))
        if not query_terms:
            return {"status": "ok", "candidate_count": 0, "matches": []}
        if not os.path.exists(self.db_path):
            return {"status": "unavailable", "reason": "index_missing", "matches": []}
        kinds = {
            str(value or "").strip().lower() for value in memory_kinds
        }.intersection({"fact", "event"})
        if not kinds:
            return {"status": "ok", "candidate_count": 0, "matches": []}
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
            return {"status": "ok", "candidate_count": 0, "matches": []}
        placeholders = ",".join("?" for _ in query_terms)
        with closing(self._connect()) as conn:
            corpus = conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(AVG(doc_len), 1) AS avgdl FROM lexical_documents"
            ).fetchone()
            n_docs = int(corpus["n"] or 0)
            avgdl = float(corpus["avgdl"] or 1.0)
            stats = {
                str(row["term"]): int(row["df"])
                for row in conn.execute(
                    f"SELECT term, df FROM lexical_stats WHERE term IN ({placeholders})",
                    query_terms,
                )
            }
            if not stats or not n_docs:
                return {"status": "ok", "candidate_count": 0, "matches": []}
            rows = conn.execute(
                f"""
                SELECT d.*, t.term, t.weighted_tf, t.fields_json
                FROM lexical_terms t
                JOIN lexical_documents d ON d.item_id=t.item_id
                WHERE t.term IN ({placeholders})
                """,
                query_terms,
            ).fetchall()
        by_item: dict[str, dict[str, Any]] = {}
        for row in rows:
            item_type = str(row["item_type"])
            if item_type not in kinds:
                continue
            if allowed_ids is not None and str(row["item_id"]) not in allowed_ids:
                continue
            if int(row["importance"] or 0) < importance_floor[item_type]:
                continue
            term = str(row["term"])
            df = stats.get(term, n_docs)
            tf = float(row["weighted_tf"] or 0.0)
            doc_len = float(row["doc_len"] or 1.0)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            denominator = tf + self.k1 * (1.0 - self.b + self.b * doc_len / avgdl)
            score = idf * (tf * (self.k1 + 1.0)) / max(denominator, 1e-9)
            current = by_item.setdefault(
                str(row["item_id"]),
                {
                    "owner_kind": str(row["item_type"]),
                    "owner_id": str(row["item_id"]),
                    "importance": int(row["importance"] or 0),
                    "score": 0.0,
                    "matched_terms": [],
                    "specific_terms": [],
                    "matched_term_df": {},
                    "matched_spans": [],
                    "body": str(row["body"] or ""),
                },
            )
            current["score"] += score
            current["matched_terms"].append(term)
            current["matched_term_df"][term] = df
            if df / n_docs <= self.max_df_ratio and (
                len(term) >= 3 or df <= self.max_short_term_df
            ):
                current["specific_terms"].append(term)
            span = self._match_span(term, row, json.loads(row["fields_json"] or "[]"))
            if span:
                current["matched_spans"].append(span)
        matches = [row for row in by_item.values() if row["specific_terms"]]
        matches.sort(
            key=lambda row: (-float(row["score"]), -len(row["specific_terms"]), row["owner_id"])
        )
        output = []
        for row in matches[: max(1, int(top_k or 1))]:
            body = row.pop("body")
            row["score"] = round(float(row["score"]), 4)
            row["matched_terms"] = sorted(set(row["matched_terms"]))
            row["specific_terms"] = sorted(set(row["specific_terms"]))
            row["passages"] = [{
                "ordinal": 0,
                "start_offset": 0,
                "end_offset": len(body),
                "text": body,
                "score": row["score"],
            }]
            row["candidate_sources"] = ["fact_event_lexical"]
            row["candidate_only"] = True
            row["decision_applied"] = False
            output.append(row)
        return {
            "status": "ok",
            "candidate_count": len(matches),
            "corpus_documents": n_docs,
            "query_terms": query_terms,
            "matches": output,
            "min_importance_by_kind": importance_floor,
            "candidate_only": True,
            "decision_applied": False,
        }
