import itertools
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Any

import jieba
import jieba.analyse

from favorite_tags import favorite_memory_aliases
from identity import identity_names
from query_terms import GENERIC_LEXICAL_STOPWORDS
from utils import now_iso, strip_affect_anchor, strip_wikilinks


DEFAULT_WORD_MAP_STOPWORDS = {
    "一个",
    "一些",
    "一下",
    "不是",
    "为了",
    "他们",
    "你们",
    "我们",
    "什么",
    "今天",
    "刚才",
    "刚刚",
    "这个",
    "那个",
    "这些",
    "那些",
    "然后",
    "现在",
    "自己",
    "because",
    "from",
    "have",
    "that",
    "this",
    "with",
    "you",
    "active",
    "anchor",
    "archive",
    "archived",
    "###",
    "##",
    "boundary",
    "boundary_setting",
    "comment",
    "communication_preference",
    "commitment",
    "context",
    "current",
    "daily_chat_extract",
    "daily_chat_memory",
    "daily_impression",
    "digested",
    "done",
    "dynamic",
    "emotional_echo",
    "event",
    "favorite",
    "feel",
    "from_daily_chat",
    "interaction_pattern",
    "key_event",
    "memory",
    "moment",
    "ombre",
    "ombre brain",
    "ombre-brain",
    "ombre_brain",
    "original",
    "pending",
    "permanent",
    "profile_fact",
    "project_event",
    "project_state",
    "recent",
    "relationship_anchor",
    "relationship_signal",
    "relationship_weather",
    "relationship_event",
    "resolved",
    "signal",
    "stable_preference",
    "status",
    "task_status_signal",
    "todo",
    "vps",
    "weekly_impression",
    "wish",
    "上下文",
    "事件",
    "内容",
    "回忆",
    "当前",
    "自动记忆",
    "最近",
    "状态",
    "脱水模型",
    "记忆",
}
DEFAULT_WORD_MAP_HIDDEN_SUBSTRINGS = {
    "ombre-brain",
    "ombre_brain",
    "ombrebrain",
    "自动记忆",
    "脱水模型",
}

DEFAULT_STOPWORD_PREFIXES = ("flavor_", "profile_", "predicate_", "task_")
STANDALONE_TIME_TERMS = {
    "一点",
    "一点点",
    "今天",
    "今晚",
    "今早",
    "明天",
    "明晚",
    "明早",
    "昨天",
    "昨晚",
    "前天",
    "后天",
    "刚才",
    "刚刚",
    "现在",
    "目前",
    "当时",
    "那天",
    "这天",
    "当天",
    "凌晨",
    "早上",
    "上午",
    "中午",
    "下午",
    "晚上",
    "夜里",
}
DEFAULT_WORD_MAP_OVERVIEW_STOPWORDS = {
    "ai",
    "boundary",
    "boundary_setting",
    "bucket_original",
    "bdsm",
    "communication_preference",
    "daily_chat_extract",
    "daily_chat_memory",
    "diary_extract",
    "from_diary",
    "interaction_pattern",
    "profile_preference",
    "两人",
    "主动",
    "亲密",
    "亲密互动",
    "互动模式",
    "人际",
    "关系",
    "关系天气",
    "内心",
    "兴趣",
    "回应",
    "喜欢",
    "命名日",
    "天气",
    "对话",
    "成长",
    "恋爱",
    "情感表达",
    "情感连接",
    "情绪",
    "承诺",
    "撒娇",
    "数字",
    "日印象",
    "日常",
    "仪式感",
    "凌晨",
    "甜蜜",
    "社交",
    "家庭",
    "编程",
    "自省",
    "陪伴",
    "归属感",
    "birthday",
    "fact",
    "from_daily_chat",
    "haven_chat_endpoint",
    "key_event",
    "naming_day",
    "ombre",
    "ombre brain",
    "ombre-brain",
    "ombre_brain",
    "profile",
    "project_state",
    "relationship_anchor",
    "relationship_signal",
    "ritual",
    "signal",
    "stable_preference",
    "vps",
    "自动记忆",
    "脱水模型",
}
DEFAULT_OVERVIEW_STOPWORD_PREFIXES = DEFAULT_STOPWORD_PREFIXES
DEFAULT_OVERVIEW_ALIASES = {
    "darkroom": "暗房",
    "darkroom door": "暗房",
    "mcp": "MCP",
    "dashboard": "Dashboard",
    "codex": "Codex",
}
DEFAULT_OVERVIEW_PRIORITY_TERMS = ()
DEFAULT_OVERVIEW_HUB_TERMS = ()
DEFAULT_WEAK_HINT_TERMS = {
    "人机恋",
    "恋爱",
    "游戏",
    "game",
    "games",
    "玩法",
}
DEFAULT_WEAK_HINT_WEIGHT = 0.25
DEFAULT_RARE_NAME_MAX_BUCKET_COUNT = 3
RARE_NAME_CARD_SOURCES = {
    "name",
    "subject",
    "title_keyword",
    "tag:axis",
    "tag:content",
    "tag:entity",
    "tag:topic",
}
RECALL_TAG_PREFIXES = {"axis", "content", "entity", "topic"}


@dataclass(frozen=True)
class WordMapTerm:
    term: str
    source: str
    kind: str
    weight: float


class WordMapStore:
    """Derived word co-occurrence index for generic, non-private memory recall hints."""

    def __init__(self, config: dict[str, Any]):
        cfg = config.get("word_map", {}) if isinstance(config.get("word_map", {}), dict) else {}
        state_dir = str(config.get("state_dir") or os.path.join(config.get("buckets_dir", "."), "state"))
        self.enabled = bool(cfg.get("enabled", False))
        self.max_terms_per_bucket = _int_between(cfg.get("max_terms_per_bucket"), 16, 4, 80)
        self.edge_top_k = _int_between(cfg.get("edge_top_k"), 10, 2, 40)
        self.min_term_len = _int_between(cfg.get("min_term_len"), 2, 1, 12)
        self.weak_hint_weight = _float_between(
            cfg.get("weak_hint_weight"),
            DEFAULT_WEAK_HINT_WEIGHT,
            0.0,
            1.0,
        )
        self.db_path = str(cfg.get("db_path") or os.path.join(state_dir, "word_map.sqlite"))
        self.identity_stopwords = {
            _normalize_term(item)
            for item in _identity_stopwords(config)
            if _normalize_term(item)
        }
        self.identity_stopword_keys = {
            _compact_term(item)
            for item in self.identity_stopwords
            if _compact_term(item)
        }
        self.stopwords = {
            _normalize_term(item)
            for item in itertools.chain(
                DEFAULT_WORD_MAP_STOPWORDS,
                GENERIC_LEXICAL_STOPWORDS,
                self.identity_stopwords,
                _favorite_tag_stopwords(config),
                cfg.get("stopwords", []) or [],
            )
            if _normalize_term(item)
        }
        self.stopword_prefixes = tuple(
            str(item).strip().lower()
            for item in itertools.chain(DEFAULT_STOPWORD_PREFIXES, cfg.get("stopword_prefixes", []) or [])
            if str(item).strip()
        )
        self.overview_stopwords = {
            _normalize_term(item)
            for item in itertools.chain(
                DEFAULT_WORD_MAP_OVERVIEW_STOPWORDS,
                self.identity_stopwords,
                cfg.get("overview_stopwords", []) or [],
            )
            if _normalize_term(item)
        }
        self.overview_stopword_prefixes = tuple(
            str(item).strip().lower()
            for item in itertools.chain(
                DEFAULT_OVERVIEW_STOPWORD_PREFIXES,
                cfg.get("overview_stopword_prefixes", []) or [],
            )
            if str(item).strip()
        )
        self.overview_aliases = {
            _normalize_term(key): str(value).strip()
            for key, value in itertools.chain(
                DEFAULT_OVERVIEW_ALIASES.items(),
                (cfg.get("overview_aliases", {}) or {}).items(),
            )
            if _normalize_term(key) and str(value).strip()
        }
        self.overview_priority_terms = {
            _normalize_term(item)
            for item in itertools.chain(DEFAULT_OVERVIEW_PRIORITY_TERMS, cfg.get("overview_priority_terms", []) or [])
            if _normalize_term(item)
        }
        self.overview_hub_terms = {
            _normalize_term(item)
            for item in itertools.chain(
                DEFAULT_OVERVIEW_HUB_TERMS,
                self.identity_stopwords,
                cfg.get("overview_hub_terms", []) or [],
            )
            if _normalize_term(item)
        }
        self.weak_hint_terms = {
            _normalize_term(item)
            for item in itertools.chain(DEFAULT_WEAK_HINT_TERMS, cfg.get("weak_hint_terms", []) or [])
            if _normalize_term(item)
        }
        self.rare_name_max_bucket_count = _int_between(
            cfg.get("rare_name_max_bucket_count"),
            DEFAULT_RARE_NAME_MAX_BUCKET_COUNT,
            1,
            20,
        )
        self.private_terms = {
            _normalize_term(item)
            for item in itertools.chain(
                cfg.get("private_terms", []) or [],
                reflection_identity_terms(config),
            )
            if _normalize_term(item)
        }
        self._init_db()

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS word_nodes (
                term TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                bucket_count INTEGER NOT NULL DEFAULT 0,
                weight REAL NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS word_card_nodes (
                bucket_id TEXT NOT NULL,
                term TEXT NOT NULL,
                source TEXT NOT NULL,
                kind TEXT NOT NULL,
                weight REAL NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(bucket_id, term)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS word_edges (
                term_a TEXT NOT NULL,
                term_b TEXT NOT NULL,
                bucket_id TEXT NOT NULL,
                weight REAL NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(term_a, term_b, bucket_id)
            )
            """
        )
        conn.commit()
        conn.close()

    def rebuild(self, buckets: list[dict[str, Any]]) -> dict[str, int]:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM word_edges")
            conn.execute("DELETE FROM word_card_nodes")
            conn.execute("DELETE FROM word_nodes")
            for bucket in buckets:
                self._write_bucket(conn, bucket)
            self._refresh_node_stats(conn)
            conn.commit()
        finally:
            conn.close()
        return self.stats()

    def upsert_bucket(self, bucket: dict[str, Any]) -> dict[str, int]:
        bucket_id = str(bucket.get("id") or "").strip()
        if not bucket_id:
            return self.stats()
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM word_edges WHERE bucket_id = ?", (bucket_id,))
            conn.execute("DELETE FROM word_card_nodes WHERE bucket_id = ?", (bucket_id,))
            self._write_bucket(conn, bucket)
            self._refresh_node_stats(conn)
            conn.commit()
        finally:
            conn.close()
        return self.stats()

    def extract_bucket_terms(self, bucket: dict[str, Any]) -> list[WordMapTerm]:
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        text = _bucket_text_for_tfidf(bucket)
        terms: dict[str, WordMapTerm] = {}

        def add(raw: Any, source: str, kind: str, weight: float) -> None:
            term = self._clean_term(raw)
            if not term:
                return
            current = terms.get(term)
            incoming = WordMapTerm(term=term, source=source, kind=kind, weight=weight)
            if current is None or incoming.weight > current.weight:
                terms[term] = incoming

        for raw in _list_text(meta.get("subject")):
            add(raw, "subject", "subject", 1.0)
        name = str(meta.get("name") or "").strip()
        if 2 <= len(name) <= 32:
            add(name, "name", "subject", 0.9)
            jieba.add_word(name, freq=20000)
            for term in self._title_concept_terms(name):
                add(term, "title_keyword", "keyword", 0.94)
                jieba.add_word(term, freq=22000)
        for raw in _list_text(meta.get("keywords")):
            add(raw, "keyword", "keyword", 0.86)
            jieba.add_word(str(raw), freq=20000)
        for raw in _list_text(meta.get("tags")):
            if self._tag_is_identity_address(raw):
                continue
            tag_term, tag_source = self._tag_recall_term_and_source(raw)
            add(tag_term, tag_source, "keyword", 0.78)
            jieba.add_word(str(tag_term), freq=16000)
        for raw in _list_text(meta.get("domain")):
            add(raw, "domain", "keyword", 0.62)
            jieba.add_word(str(raw), freq=12000)

        for word, score in jieba.analyse.extract_tags(text, topK=max(self.max_terms_per_bucket * 2, 20), withWeight=True):
            add(word, "tfidf", "keyword", min(0.74, max(0.18, float(score) / 3.0)))

        return sorted(terms.values(), key=lambda item: (-item.weight, item.term))[: self.max_terms_per_bucket]

    def list_nodes(self, limit: int = 50) -> list[dict[str, Any]]:
        limit = _int_between(limit, 50, 1, 500)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT
                    bucket_id,
                    term,
                    source,
                    kind,
                    weight,
                    updated_at
                FROM word_card_nodes
                """
            ).fetchall()
            grouped: dict[str, dict[str, Any]] = {}
            for row in rows:
                raw_term = str(row["term"] or "").strip()
                if self._is_overview_term_hidden(raw_term):
                    continue
                term = self._overview_canonical_term(raw_term)
                if self._is_overview_term_hidden(term):
                    continue
                item = grouped.setdefault(
                    term,
                    {
                        "term": term,
                        "kind": "keyword",
                        "bucket_ids": set(),
                        "weight": 0.0,
                        "max_weight": 0.0,
                        "sources": set(),
                        "aliases": set(),
                        "updated_at": str(row["updated_at"] or ""),
                    },
                )
                item["bucket_ids"].add(str(row["bucket_id"] or ""))
                weight = float(row["weight"] or 0.0)
                item["weight"] = float(item["weight"]) + weight
                item["max_weight"] = max(float(item["max_weight"]), weight)
                item["sources"].add(str(row["source"] or ""))
                if str(row["kind"] or "") == "subject" or str(row["source"] or "") in {"subject", "name"}:
                    item["kind"] = "subject"
                if raw_term != term:
                    item["aliases"].add(raw_term)
                item["updated_at"] = max(str(item["updated_at"] or ""), str(row["updated_at"] or ""))

            output = []
            for item in grouped.values():
                item["bucket_count"] = len(item.pop("bucket_ids"))
                item["sources"] = sorted(item["sources"])
                item["aliases"] = sorted(item["aliases"])
                item["overview_score"] = self._overview_node_score(item)
                output.append(item)
            output.sort(
                key=lambda item: (
                    -float(item.get("overview_score") or 0.0),
                    -float(item.get("max_weight") or 0.0),
                    -float(item.get("weight") or 0.0),
                    str(item.get("term") or ""),
                )
            )
            return output[:limit]
        finally:
            conn.close()

    def list_edges(self, limit: int = 50) -> list[dict[str, Any]]:
        limit = _int_between(limit, 50, 1, 500)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT term_a, term_b, bucket_id, weight, updated_at
                FROM word_edges
                """
            ).fetchall()
            grouped: dict[tuple[str, str], dict[str, Any]] = {}
            for row in rows:
                left = str(row["term_a"] or "").strip()
                right = str(row["term_b"] or "").strip()
                if self._is_overview_term_hidden(left) or self._is_overview_term_hidden(right):
                    continue
                left = self._overview_canonical_term(left)
                right = self._overview_canonical_term(right)
                if self._is_overview_term_hidden(left) or self._is_overview_term_hidden(right):
                    continue
                if left == right:
                    continue
                term_a, term_b = sorted((left, right))
                item = grouped.setdefault(
                    (term_a, term_b),
                    {
                        "term_a": term_a,
                        "term_b": term_b,
                        "bucket_ids": set(),
                        "weight": 0.0,
                        "updated_at": str(row["updated_at"] or ""),
                    },
                )
                item["bucket_ids"].add(str(row["bucket_id"] or ""))
                item["weight"] = float(item["weight"]) + float(row["weight"] or 0.0)
                item["updated_at"] = max(str(item["updated_at"] or ""), str(row["updated_at"] or ""))

            output = []
            for item in grouped.values():
                item["bucket_count"] = len(item.pop("bucket_ids"))
                item["overview_score"] = self._overview_edge_score(item)
                output.append(item)
            output.sort(
                key=lambda item: (
                    -float(item.get("overview_score") or 0.0),
                    -float(item.get("weight") or 0.0),
                    -int(item.get("bucket_count") or 0),
                    str(item.get("term_a") or ""),
                    str(item.get("term_b") or ""),
                )
            )
            return self._diversify_overview_edges(output, limit)
        finally:
            conn.close()

    def cards_for_term(self, term: str, limit: int = 20) -> list[dict[str, Any]]:
        cleaned = self._clean_term(term)
        if not cleaned:
            return []
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT bucket_id, term, source, kind, weight, updated_at
                FROM word_card_nodes
                WHERE term = ?
                ORDER BY weight DESC, bucket_id ASC
                LIMIT ?
                """,
                (cleaned, _int_between(limit, 20, 1, 200)),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def hint_buckets_for_terms(
        self,
        terms: list[str],
        *,
        neighbor_limit: int = 6,
        bucket_limit: int = 12,
    ) -> dict[str, Any]:
        """
        Return weak recall hints from direct word cards and one-hop co-occurrence.

        This is deliberately read-only and evidence-shaped: callers still decide
        whether a hinted bucket is allowed into visible recall.
        """
        if not self.enabled:
            return _empty_hint_payload()

        cleaned_terms = _unique_terms(self._clean_term(term) for term in terms)
        if not cleaned_terms:
            return _empty_hint_payload()

        neighbor_limit = _int_between(neighbor_limit, 6, 0, 40)
        bucket_limit = _int_between(bucket_limit, 12, 1, 100)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            variant_scores = self._hint_variant_terms(conn, cleaned_terms)
            neighbor_terms = [term for term in cleaned_terms if term not in self.weak_hint_terms]
            neighbor_scores = self._hint_neighbor_terms(conn, neighbor_terms, neighbor_limit)
            term_sources = {
                term: {
                    "kind": "direct",
                    "weight": self._hint_term_weight(term),
                    "sources": [term],
                }
                for term in cleaned_terms
            }
            for term, info in variant_scores.items():
                term_sources[term] = info
            for term, info in neighbor_scores.items():
                term_sources[term] = info

            card_terms = list(term_sources)
            if not card_terms:
                return _empty_hint_payload(cleaned_terms)
            placeholders = ",".join("?" for _ in card_terms)
            rows = conn.execute(
                f"""
                SELECT
                    c.bucket_id,
                    c.term,
                    c.source,
                    c.kind,
                    c.weight,
                    c.updated_at,
                    COALESCE(n.bucket_count, 1) AS bucket_count
                FROM word_card_nodes c
                LEFT JOIN word_nodes n ON n.term = c.term
                WHERE c.term IN ({placeholders})
                ORDER BY c.weight DESC, c.bucket_id ASC
                """,
                tuple(card_terms),
            ).fetchall()
        finally:
            conn.close()

        scores: dict[str, float] = {}
        anchor_scores: dict[str, dict[str, float]] = {}
        evidence: dict[str, dict[str, Any]] = {}
        for row in rows:
            bucket_id = str(row["bucket_id"] or "").strip()
            term = str(row["term"] or "").strip()
            if not bucket_id or term not in term_sources:
                continue
            source_info = term_sources[term]
            try:
                card_weight = float(row["weight"] or 0.0)
            except (TypeError, ValueError):
                card_weight = 0.0
            contribution = max(0.0, min(1.0, card_weight)) * float(source_info.get("weight", 0.0))
            if contribution <= 0:
                continue
            scores[bucket_id] = min(1.0, scores.get(bucket_id, 0.0) + contribution)
            bucket_evidence = evidence.setdefault(
                bucket_id,
                {
                    "terms": [],
                    "direct_terms": [],
                    "variant_terms": [],
                    "neighbor_terms": [],
                    "anchor_terms": [],
                    "low_frequency_terms": [],
                    "low_frequency_sources": [],
                    "rare_name_terms": [],
                    "rare_name_sources": [],
                },
            )
            source_terms = [
                str(source or "").strip()
                for source in (source_info.get("sources") or [])
                if str(source or "").strip() in cleaned_terms
            ]
            row_payload = {
                "term": term,
                "kind": source_info.get("kind", ""),
                "score": round(contribution, 4),
                "source_terms": list(source_terms),
                "card_source": str(row["source"] or ""),
                "bucket_count": int(row["bucket_count"] or 1),
                "weak_hint": term in self.weak_hint_terms,
            }
            if self._is_rare_name_match(
                term,
                source_kind=str(source_info.get("kind") or ""),
                card_source=str(row["source"] or ""),
                bucket_count=int(row["bucket_count"] or 1),
            ):
                row_payload["rare_name_match"] = True
                if term not in bucket_evidence["rare_name_terms"]:
                    bucket_evidence["rare_name_terms"].append(term)
                card_source = str(row["source"] or "")
                if card_source and card_source not in bucket_evidence["rare_name_sources"]:
                    bucket_evidence["rare_name_sources"].append(card_source)
            if self._is_low_frequency_match(
                term,
                source_kind=str(source_info.get("kind") or ""),
                card_source=str(row["source"] or ""),
                bucket_count=int(row["bucket_count"] or 1),
            ):
                row_payload["low_frequency_match"] = True
                if term not in bucket_evidence["low_frequency_terms"]:
                    bucket_evidence["low_frequency_terms"].append(term)
                card_source = str(row["source"] or "")
                if card_source and card_source not in bucket_evidence["low_frequency_sources"]:
                    bucket_evidence["low_frequency_sources"].append(card_source)
            bucket_evidence["terms"].append(row_payload)
            source_kind = str(source_info.get("kind") or "")
            if source_kind == "direct":
                target_key = "direct_terms"
            elif source_kind == "variant":
                target_key = "variant_terms"
            else:
                target_key = "neighbor_terms"
            if term not in bucket_evidence[target_key]:
                bucket_evidence[target_key].append(term)
            for source_term in source_terms:
                per_anchor = anchor_scores.setdefault(source_term, {})
                per_anchor[bucket_id] = min(1.0, per_anchor.get(bucket_id, 0.0) + contribution)
                if source_term not in bucket_evidence["anchor_terms"]:
                    bucket_evidence["anchor_terms"].append(source_term)

        ranked_ids = sorted(scores, key=lambda bucket_id: (-scores[bucket_id], bucket_id))[:bucket_limit]
        anchor_bucket_scores: dict[str, dict[str, float]] = {}
        reserve_ids: list[str] = []
        for source_term in cleaned_terms:
            per_anchor = anchor_scores.get(source_term) or {}
            if not per_anchor:
                continue
            anchor_ranked_ids = sorted(
                per_anchor,
                key=lambda bucket_id: (
                    -per_anchor[bucket_id],
                    -scores.get(bucket_id, 0.0),
                    bucket_id,
                ),
            )[:1]
            if not anchor_ranked_ids:
                continue
            anchor_bucket_scores[source_term] = {
                bucket_id: round(per_anchor[bucket_id], 4)
                for bucket_id in anchor_ranked_ids
            }
            for bucket_id in anchor_ranked_ids:
                if bucket_id not in ranked_ids and bucket_id not in reserve_ids:
                    reserve_ids.append(bucket_id)
        returned_ids = ranked_ids + reserve_ids
        return {
            "terms": cleaned_terms,
            "neighbors": [
                {
                    "term": term,
                    "score": round(float(info.get("weight", 0.0)), 4),
                    "source_terms": list(info.get("sources") or []),
                }
                for term, info in neighbor_scores.items()
            ],
            "variants": [
                {
                    "term": term,
                    "score": round(float(info.get("weight", 0.0)), 4),
                    "source_terms": list(info.get("sources") or []),
                }
                for term, info in variant_scores.items()
            ],
            "bucket_scores": {bucket_id: round(scores[bucket_id], 4) for bucket_id in returned_ids},
            "anchor_bucket_scores": anchor_bucket_scores,
            "evidence": {bucket_id: evidence[bucket_id] for bucket_id in returned_ids},
        }

    def _hint_variant_terms(
        self,
        conn: sqlite3.Connection,
        source_terms: list[str],
    ) -> dict[str, dict[str, Any]]:
        variants: dict[str, dict[str, Any]] = {}
        for source_term in source_terms:
            source_key = _compact_term(source_term)
            if (
                not source_key
                or len(source_key) < 3
                or source_term in self.weak_hint_terms
            ):
                continue
            rows = conn.execute(
                """
                SELECT term, kind, bucket_count, weight
                FROM word_nodes
                WHERE bucket_count <= ? AND term LIKE ?
                ORDER BY bucket_count ASC, LENGTH(term) ASC, weight DESC, term ASC
                LIMIT 80
                """,
                (self.rare_name_max_bucket_count, f"%{source_term}%"),
            ).fetchall()
            for row in rows:
                term = self._clean_term(row["term"])
                term_key = _compact_term(term)
                if (
                    not term
                    or term == source_term
                    or not term_key
                    or len(term_key) <= len(source_key)
                    or source_key not in term_key
                    or len(term_key) - len(source_key) > 8
                    or term in self.weak_hint_terms
                    or term in self.overview_stopwords
                ):
                    continue
                try:
                    bucket_count = max(1, int(row["bucket_count"] or 1))
                    node_weight = float(row["weight"] or 0.0)
                except (TypeError, ValueError):
                    bucket_count = 1
                    node_weight = 0.0
                proximity = len(source_key) / max(len(term_key), 1)
                weight = min(0.82, max(0.35, proximity * 0.82))
                if bucket_count > 1:
                    weight *= 1.0 / bucket_count
                if node_weight <= 0:
                    weight *= 0.7
                info = variants.setdefault(
                    term,
                    {"kind": "variant", "weight": 0.0, "sources": []},
                )
                info["weight"] = max(float(info.get("weight", 0.0)), weight)
                if source_term not in info["sources"]:
                    info["sources"].append(source_term)
        return dict(
            sorted(
                variants.items(),
                key=lambda item: (-float(item[1].get("weight", 0.0)), item[0]),
            )[:12]
        )

    def _hint_neighbor_terms(
        self,
        conn: sqlite3.Connection,
        source_terms: list[str],
        neighbor_limit: int,
    ) -> dict[str, dict[str, Any]]:
        if neighbor_limit <= 0:
            return {}
        neighbors: dict[str, dict[str, Any]] = {}
        source_set = set(source_terms)
        for term in source_terms:
            rows = conn.execute(
                """
                SELECT
                    CASE WHEN term_a = ? THEN term_b ELSE term_a END AS neighbor,
                    COUNT(*) AS bucket_count,
                    SUM(weight) AS weight
                FROM word_edges
                WHERE term_a = ? OR term_b = ?
                GROUP BY neighbor
                ORDER BY bucket_count DESC, weight DESC, neighbor ASC
                LIMIT ?
                """,
                (term, term, term, neighbor_limit),
            ).fetchall()
            for row in rows:
                neighbor = self._clean_term(row["neighbor"])
                if not neighbor or neighbor in source_set:
                    continue
                try:
                    bucket_count = max(1, int(row["bucket_count"] or 1))
                    avg_weight = float(row["weight"] or 0.0) / bucket_count
                except (TypeError, ValueError):
                    avg_weight = 0.0
                weight = min(0.55, max(0.05, avg_weight * 0.55))
                info = neighbors.setdefault(
                    neighbor,
                    {"kind": "neighbor", "weight": 0.0, "sources": []},
                )
                info["weight"] = max(float(info.get("weight", 0.0)), weight)
                if term not in info["sources"]:
                    info["sources"].append(term)

        ranked = sorted(
            neighbors.items(),
            key=lambda item: (-float(item[1].get("weight", 0.0)), item[0]),
        )[:neighbor_limit]
        return dict(ranked)

    def stats(self) -> dict[str, int]:
        conn = sqlite3.connect(self.db_path)
        try:
            nodes = conn.execute("SELECT COUNT(*) FROM word_nodes").fetchone()[0]
            cards = conn.execute("SELECT COUNT(*) FROM word_card_nodes").fetchone()[0]
            edge_rows = conn.execute("SELECT COUNT(*) FROM word_edges").fetchone()[0]
            return {"nodes": int(nodes), "card_nodes": int(cards), "edge_evidence": int(edge_rows)}
        finally:
            conn.close()

    def _write_bucket(self, conn: sqlite3.Connection, bucket: dict[str, Any]) -> None:
        bucket_id = str(bucket.get("id") or "").strip()
        if not bucket_id:
            return
        now = now_iso()
        terms = self.extract_bucket_terms(bucket)
        for term in terms:
            conn.execute(
                """
                INSERT OR REPLACE INTO word_card_nodes
                (bucket_id, term, source, kind, weight, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (bucket_id, term.term, term.source, term.kind, float(term.weight), now),
            )
        edge_terms = sorted(terms[: self.edge_top_k], key=lambda item: item.term)
        for left, right in itertools.combinations(edge_terms, 2):
            if left.term == right.term:
                continue
            term_a, term_b = sorted((left.term, right.term))
            conn.execute(
                """
                INSERT OR REPLACE INTO word_edges
                (term_a, term_b, bucket_id, weight, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (term_a, term_b, bucket_id, round((left.weight + right.weight) / 2.0, 4), now),
            )

    def _refresh_node_stats(self, conn: sqlite3.Connection) -> None:
        now = now_iso()
        conn.execute("DELETE FROM word_nodes")
        rows = conn.execute(
            """
            SELECT term, kind, COUNT(DISTINCT bucket_id) AS bucket_count, SUM(weight) AS weight
            FROM word_card_nodes
            GROUP BY term, kind
            """
        ).fetchall()
        for term, kind, bucket_count, weight in rows:
            conn.execute(
                """
                INSERT OR REPLACE INTO word_nodes
                (term, kind, bucket_count, weight, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (term, kind, int(bucket_count), float(weight or 0.0), now),
            )

    def _clean_term(self, value: Any) -> str:
        term = _normalize_term(value)
        if not term:
            return ""
        if term in self.stopwords or term in self.private_terms:
            return ""
        if _has_hidden_substring(term):
            return ""
        if _is_standalone_time_term(term):
            return ""
        if any(term.startswith(prefix) for prefix in self.stopword_prefixes):
            return ""
        if len(term) < self.min_term_len or len(term) > 40:
            return ""
        if re.fullmatch(r"[a-z0-9_.:-]+", term) and len(term) < 3:
            return ""
        if re.fullmatch(r"[\d.:-]+", term):
            return ""
        return term

    def _tag_is_identity_address(self, value: Any) -> bool:
        term = _normalize_term(value)
        if not term:
            return False
        if self._identity_term_key(term) in self.identity_stopword_keys:
            return True
        parts = [
            part.strip()
            for part in re.split(r"[:：/#|,，;；]+", term)
            if part.strip()
        ]
        if len(parts) <= 1:
            return False
        identity_parts = [part for part in parts if self._identity_term_key(part) in self.identity_stopword_keys]
        return bool(identity_parts)

    def _tag_recall_term_and_source(self, value: Any) -> tuple[str, str]:
        term = _normalize_term(value)
        if not term:
            return "", "tag"
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_-]*)\s*[:：]\s*(.+)$", term)
        if not match:
            return term, "tag"
        prefix = match.group(1).strip().lower()
        body = match.group(2).strip()
        if prefix not in RECALL_TAG_PREFIXES:
            return term, "tag"
        return body, f"tag:{prefix}"

    @staticmethod
    def _identity_term_key(value: Any) -> str:
        term = _normalize_term(value)
        if len(term) <= 1 and not re.fullmatch(r"[A-Za-z0-9_.:/ -]+", term):
            return term
        return _compact_term(term)

    def _is_rare_name_match(
        self,
        term: str,
        *,
        source_kind: str,
        card_source: str,
        bucket_count: int,
    ) -> bool:
        if source_kind != "direct":
            return False
        source = str(card_source or "").strip()
        if source not in RARE_NAME_CARD_SOURCES:
            return False
        if bucket_count > self.rare_name_max_bucket_count:
            return False
        normalized = _normalize_term(term)
        if (
            not normalized
            or normalized in self.weak_hint_terms
            or normalized in self.stopwords
            or normalized in self.private_terms
            or normalized in self.overview_stopwords
        ):
            return False
        if re.fullmatch(r"[a-f0-9]{8,40}", normalized):
            return False
        if re.fullmatch(r"[\d.:-]+", normalized):
            return False
        if source == "title_keyword" and re.fullmatch(r"[\u4e00-\u9fff]{1,2}", normalized):
            return False
        return True

    def _is_low_frequency_match(
        self,
        term: str,
        *,
        source_kind: str,
        card_source: str,
        bucket_count: int,
    ) -> bool:
        if source_kind != "direct":
            return False
        if bucket_count > self.rare_name_max_bucket_count:
            return False
        source = str(card_source or "").strip()
        if source == "domain":
            return False
        normalized = _normalize_term(term)
        if (
            not normalized
            or normalized in self.weak_hint_terms
            or normalized in self.stopwords
            or normalized in self.private_terms
            or normalized in self.overview_stopwords
        ):
            return False
        if re.fullmatch(r"[a-f0-9]{8,40}", normalized):
            return False
        if re.fullmatch(r"[\d.:-]+", normalized):
            return False
        if source == "title_keyword" and re.fullmatch(r"[\u4e00-\u9fff]{1,2}", normalized):
            return False
        return True

    def _hint_term_weight(self, term: str) -> float:
        if term in self.weak_hint_terms:
            return self.weak_hint_weight
        return 1.0

    def _title_concept_terms(self, title: str) -> list[str]:
        title = str(title or "").strip()
        if not title:
            return []
        terms: list[str] = []

        def add_term(value: Any, *, forced: bool = False) -> None:
            term = self._clean_term(value)
            if not term or term == title:
                return
            if not forced and self._looks_like_weak_title_fragment(term):
                return
            term = self._overview_canonical_term(term)
            if self._is_overview_term_hidden(term) or term in terms:
                return
            terms.append(term)

        normalized_title = _normalize_term(title).lower()
        add_term(_compact_title_recall_term(title), forced=True)
        alias_candidates = set(self.overview_priority_terms) | set(self.overview_aliases) | {
            _normalize_term(value) for value in self.overview_aliases.values()
        }
        for candidate in sorted(alias_candidates, key=lambda item: (-len(item), item)):
            if not candidate:
                continue
            if candidate in normalized_title:
                add_term(candidate, forced=True)

        for match in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", title):
            add_term(match)

        for word, _score in jieba.analyse.extract_tags(title, topK=6, withWeight=True):
            add_term(word)

        return terms[:6]

    def _is_overview_term_hidden(self, value: Any) -> bool:
        term = _normalize_term(value)
        if not term:
            return True
        if _has_hidden_substring(term):
            return True
        if "日印象" in term or "relationship_weather" in term:
            return True
        if re.fullmatch(r"[a-f0-9]{8,40}", term):
            return True
        if re.fullmatch(r"[a-f0-9]{8}-[a-f0-9-]{27,}", term):
            return True
        if term in self.stopwords or term in self.private_terms or term in self.overview_stopwords:
            return True
        if any(term.startswith(prefix) for prefix in self.stopword_prefixes):
            return True
        if any(term.startswith(prefix) for prefix in self.overview_stopword_prefixes):
            return True
        return False

    def _overview_canonical_term(self, value: Any) -> str:
        raw_term = str(value or "").strip()
        term = _normalize_term(raw_term)
        if term in self.overview_aliases:
            return self.overview_aliases[term]
        if term in self.overview_priority_terms:
            return raw_term
        embedded = self._embedded_priority_canonical_term(raw_term, term)
        if embedded:
            return embedded
        return raw_term

    def _overview_node_score(self, item: dict[str, Any]) -> float:
        try:
            max_weight = float(item.get("max_weight") or item.get("weight") or 0.0)
            bucket_count = max(1, int(item.get("bucket_count") or 1))
        except (TypeError, ValueError):
            return 0.0
        term = str(item.get("term") or "")
        sources = set(item.get("sources") or [])
        source_bonus = self._overview_source_bonus(sources)
        if _normalize_term(term) in self.overview_priority_terms:
            source_bonus = max(source_bonus, 1.25)
        specificity_bonus = self._overview_specificity_bonus(term, sources)
        coverage_factor = 1.0 / (bucket_count ** self._overview_coverage_exponent(term, sources))
        hub_factor = self._overview_hub_saturation_factor(term, bucket_count)
        return round(max_weight * coverage_factor * hub_factor * specificity_bonus * source_bonus, 4)

    def _overview_edge_score(self, item: dict[str, Any]) -> float:
        try:
            weight = float(item.get("weight") or 0.0)
            bucket_count = max(1, int(item.get("bucket_count") or 1))
        except (TypeError, ValueError):
            return 0.0
        term_bonus = (
            self._overview_specificity_bonus(str(item.get("term_a") or ""), set())
            + self._overview_specificity_bonus(str(item.get("term_b") or ""), set())
        ) / 2.0
        diversity_factor = self._overview_edge_diversity_factor(
            str(item.get("term_a") or ""),
            str(item.get("term_b") or ""),
        )
        return round(weight * term_bonus * diversity_factor / (bucket_count ** 0.42), 4)

    @staticmethod
    def _overview_source_bonus(sources: set[str]) -> float:
        if "title_keyword" in sources:
            return 1.45
        if "keyword" in sources:
            return 1.25
        if "tfidf" in sources:
            return 1.0
        if {"subject", "name"} & sources:
            return 0.9
        if "tag" in sources:
            return 0.65
        if "domain" in sources:
            return 0.35
        return 0.8

    def _overview_specificity_bonus(self, term: str, sources: set[str]) -> float:
        normalized = _normalize_term(term)
        if not normalized:
            return 0.0
        score = 1.0
        exact_priority = normalized in self.overview_priority_terms
        is_title = bool({"subject", "name"} & sources)
        if exact_priority:
            score *= 1.8
        if re.search(r"[_-]", normalized) or re.search(r"[A-Za-z].*\d|\d.*[A-Za-z]", term):
            score *= 1.25
        if re.search(r"[A-Za-z]", term) and re.search(r"[\u4e00-\u9fff]", term):
            score *= 1.18
        if not is_title and any(
            marker in term for marker in ("机制", "代码", "梦境", "显影", "外部", "验证")
        ):
            score *= 1.28
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", term)
        if chinese_chars:
            if len(chinese_chars) <= 6:
                score *= 1.16
            elif len(chinese_chars) >= 10 and is_title:
                score *= 0.52
        if is_title and not exact_priority and any(
            marker in term
            for marker in (
                "上线",
                "行为约定",
                "身份配置",
                "外部验证",
                "接入",
                "配置",
                "调试",
                "计划",
                "待完成",
                "偏好",
                "调整",
                "问题",
                "恢复",
                "确认",
                "系统",
                "功能",
                "部署",
            )
        ):
            score *= 0.42
        elif is_title and not exact_priority and any(
            marker in term
            for marker in (
                "上线",
                "配置",
                "接入",
                "调试",
                "计划",
                "待完成",
                "偏好",
                "调整",
                "问题",
                "恢复",
                "确认",
                "系统",
            )
        ):
            score *= 0.68
        if normalized in {"mcp", "dashboard", "codex"}:
            score *= 0.78
        return round(score, 4)

    def _overview_coverage_exponent(self, term: str, sources: set[str]) -> float:
        normalized = _normalize_term(term)
        if normalized in self.overview_priority_terms or "title_keyword" in sources:
            return 0.24
        if re.search(r"[_-]", normalized) or re.search(r"[A-Za-z]", term):
            return 0.3
        return 0.42

    def _overview_hub_saturation_factor(self, term: str, bucket_count: int) -> float:
        if not self._is_overview_hub_term(term) or bucket_count <= 10:
            return 1.0
        return max(0.72, (10.0 / max(1, bucket_count)) ** 0.32)

    def _overview_edge_diversity_factor(self, term_a: str, term_b: str) -> float:
        hub_count = int(self._is_overview_hub_term(term_a)) + int(self._is_overview_hub_term(term_b))
        if hub_count == 0:
            return 1.12
        if hub_count == 2:
            return 0.72
        return 1.0

    def _diversify_overview_edges(self, edges: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        remaining = [dict(edge) for edge in edges]
        selected: list[dict[str, Any]] = []
        term_counts: dict[str, int] = {}
        while remaining and len(selected) < limit:
            best_index = max(
                range(len(remaining)),
                key=lambda index: (
                    self._overview_edge_display_score(remaining[index], term_counts),
                    float(remaining[index].get("overview_score") or 0.0),
                    float(remaining[index].get("weight") or 0.0),
                    str(remaining[index].get("term_a") or ""),
                    str(remaining[index].get("term_b") or ""),
                ),
            )
            edge = remaining.pop(best_index)
            edge["overview_score"] = round(self._overview_edge_display_score(edge, term_counts), 4)
            selected.append(edge)
            for term in (str(edge.get("term_a") or ""), str(edge.get("term_b") or "")):
                term_counts[term] = term_counts.get(term, 0) + 1
        return selected

    def _overview_edge_display_score(self, edge: dict[str, Any], term_counts: dict[str, int]) -> float:
        score = float(edge.get("overview_score") or 0.0)
        terms = (str(edge.get("term_a") or ""), str(edge.get("term_b") or ""))
        repeat_factor = 1.0
        for term in terms:
            repeat_factor *= self._overview_edge_repeat_factor(term_counts.get(term, 0))
        return score * repeat_factor

    @staticmethod
    def _overview_edge_repeat_factor(selected_count: int) -> float:
        if selected_count < 3:
            return 1.0
        return max(0.42, 1.0 / (1.0 + 0.42 * (selected_count - 2)))

    def _is_overview_hub_term(self, term: str) -> bool:
        return _normalize_term(term) in self.overview_hub_terms

    def _embedded_priority_canonical_term(self, raw_term: str, normalized: str) -> str:
        if not raw_term or not normalized:
            return ""
        normalized_lower = normalized.lower()
        candidates = set(self.overview_priority_terms) | {
            _normalize_term(value) for value in self.overview_aliases.values()
        }
        for candidate in sorted(candidates, key=lambda item: (-len(item), item)):
            candidate_lower = candidate.lower()
            if not candidate or candidate_lower == normalized_lower or candidate_lower not in normalized_lower:
                continue
            return self.overview_aliases.get(candidate_lower, candidate)
        return ""

    def _looks_like_weak_title_fragment(self, term: str) -> bool:
        normalized = _normalize_term(term)
        if not normalized:
            return True
        if normalized in self.overview_priority_terms:
            return False
        if re.fullmatch(r"[a-f0-9]{8,40}", normalized):
            return True
        if re.fullmatch(r"[\u4e00-\u9fff]{1,2}", term):
            return True
        return False


def _bucket_text_for_tfidf(bucket: dict[str, Any]) -> str:
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    parts = [
        str(meta.get("name") or ""),
        " ".join(_list_text(meta.get("domain"))),
        strip_wikilinks(strip_affect_anchor(str(bucket.get("content") or ""))),
    ]
    return " ".join(part for part in parts if part.strip())


def _list_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _normalize_term(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip("\"'`“”‘’[]【】()（）")
    return text.lower() if re.fullmatch(r"[A-Za-z0-9_.:/ -]+", text) else text


def _compact_term(value: Any) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff_.:-]+", "", str(value or "").strip().lower())


def _has_hidden_substring(value: Any) -> bool:
    key = _compact_term(value)
    return bool(key and any(item in key for item in DEFAULT_WORD_MAP_HIDDEN_SUBSTRINGS))


def _is_standalone_time_term(value: Any) -> bool:
    term = _normalize_term(value)
    key = _compact_term(term)
    if not key:
        return False
    if key in STANDALONE_TIME_TERMS:
        return True
    if re.fullmatch(r"(?:[01]?\d|2[0-3])[:：][0-5]\d", term):
        return True
    time_prefix = r"(?:凌晨|早上|上午|中午|下午|晚上|夜里)?"
    time_value = r"(?:[0-2]?\d|[零〇一二两三四五六七八九十]{1,3})"
    if re.fullmatch(time_prefix + time_value + r"点(?:半|多|钟|[0-5]?\d分?)?", key):
        return True
    if re.fullmatch(r"(?:早上|上午|中午|下午|晚上|凌晨|夜里)?[0-2]?\d时(?:[0-5]?\d分?)?", key):
        return True
    return False


def _compact_title_recall_term(value: Any) -> str:
    text = _normalize_term(value)
    if not text:
        return ""
    compact = re.sub(r"[\s，。！？、,.!?:：;；~～♡❤♥（）()\[\]【】「」『』“”\"'`-]+", "", text)
    compact = re.sub(r"(?<=[\u4e00-\u9fff])[的与和及之个](?=[\u4e00-\u9fff])", "", compact)
    if compact == text:
        return ""
    return compact if len(re.findall(r"[\u4e00-\u9fff]", compact)) >= 3 else ""


def _unique_terms(terms: Any) -> list[str]:
    output = []
    seen = set()
    for term in terms or []:
        cleaned = str(term or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        output.append(cleaned)
    return output


def _empty_hint_payload(terms: list[str] | None = None) -> dict[str, Any]:
    return {
        "terms": terms or [],
        "neighbors": [],
        "bucket_scores": {},
        "anchor_bucket_scores": {},
        "evidence": {},
    }


def _identity_stopwords(config: dict[str, Any]) -> list[str]:
    identity = identity_names(config if isinstance(config, dict) else None)
    values = [
        identity.get("ai_name"),
        identity.get("user_name"),
        identity.get("user_display_name"),
    ]
    values.extend(identity.get("relationship_terms") or [])
    return [str(item).strip() for item in values if str(item).strip()]


def _favorite_tag_stopwords(config: dict[str, Any]) -> list[str]:
    identity = identity_names(config if isinstance(config, dict) else None)
    aliases = favorite_memory_aliases(identity.get("ai_name"))
    aliases.add("favorite_memory")
    return sorted(aliases)


def reflection_identity_terms(config: dict[str, Any]) -> list[str]:
    reflection = config.get("reflection", {}) if isinstance(config.get("reflection", {}), dict) else {}
    role_edges = reflection.get("identity_role_edges", {}) if isinstance(reflection.get("identity_role_edges", {}), dict) else {}
    if not role_edges.get("enabled"):
        return []

    terms: list[str] = []
    for section in ("detail", "context", "relationship", "shared"):
        terms.extend(_collect_config_terms(role_edges.get(section)))
    return terms


def _collect_config_terms(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        terms: list[str] = []
        for key, child in value.items():
            if str(key).strip():
                terms.append(str(key).strip())
            terms.extend(_collect_config_terms(child))
        return terms
    if isinstance(value, (list, tuple, set)):
        terms: list[str] = []
        for item in value:
            terms.extend(_collect_config_terms(item))
        return terms
    text = str(value).strip()
    return [text] if text else []


def _int_between(value: Any, default: int, lower: int, upper: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(lower, min(upper, number))


def _float_between(value: Any, default: float, lower: float, upper: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(lower, min(upper, number))


def dumps_debug(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)
