from __future__ import annotations

import json
import hashlib
import os
import re
import sqlite3
import unicodedata
import warnings
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from identity import identity_names
from query_terms import GENERIC_LEXICAL_STOPWORDS

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import jieba
        import jieba.posseg as jieba_posseg

        jieba.setLogLevel(20)
except Exception:  # pragma: no cover - reduced runtimes may ship jieba without its dictionary
    jieba_posseg = None


SCHEMA_VERSION = 2
_DYNAMIC_POS = frozenset({"nr", "ns", "nt", "nz", "eng"})
_QUOTED_WORK = re.compile(r"《([^》]{2,40})》")
_QUOTED_NAME = re.compile(r"[“「『]([^”」』]{2,24})[”」』]")
_LATIN_NAME = re.compile(
    r"(?<![A-Za-z0-9_-])[A-Za-z][A-Za-z0-9_.-]{1,31}(?![A-Za-z0-9_-])"
)
_INTENT_PATTERNS = (
    ("exact_evidence", "exact_evidence", re.compile(r"原话|逐字|哪天|哪一[天次]|什么时候|具体日期|怎么说的")),
    ("arc_narrative", "narrative_read", re.compile(r"整体|完整(?:剧情|故事|经过)|从头|整条线|讲讲(?:剧情|故事)")),
    ("progress", "latest_relevant_member", re.compile(r"看到哪|读到哪|做到哪|进行到哪|进展(?:到哪|如何|怎么样)|追到哪")),
    ("timeline", "timeline", re.compile(r"后来|后续|之后|怎么发展|如何发展|发展成|演变|时间线")),
    ("member_search", "member_search", re.compile(r"哪一段|那一段|这段|其中一段|某一段|提到.+(?:那段|一段)")),
    ("recall_reference", "arc_index", re.compile(r"还记得|记得|想起|回忆|上次(?:聊|说|看|读|做)")),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _key(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[\W_]+", "", normalized, flags=re.UNICODE)


def _surface(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _stable_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_fingerprint(
    source_refs: Iterable[dict[str, Any]],
    *,
    extraction_contract_hash: str,
) -> str:
    sources: list[dict[str, Any]] = []
    for ordinal, ref in enumerate(source_refs):
        if not isinstance(ref, dict):
            continue
        content = str(ref.get("content") or "")
        if not content:
            continue
        sources.append(
            {
                "ordinal": ordinal,
                "source_system": str(ref.get("source_system") or ""),
                "session_id": str(ref.get("session_id") or ref.get("thread_id") or ""),
                "message_id": str(ref.get("message_id") or ordinal),
                "content": content,
            }
        )
    return _stable_hash(
        {
            "contract": extraction_contract_hash,
            "sources": sources,
        }
    )


def _term_count(text: str, term: str) -> int:
    if not text or not term:
        return 0
    if re.fullmatch(r"[A-Za-z0-9_.-]+", term):
        return len(
            re.findall(
                rf"(?<![A-Za-z0-9_.-]){re.escape(term)}(?![A-Za-z0-9_.-])",
                text,
                re.IGNORECASE,
            )
        )
    return len(re.findall(re.escape(term), text, re.IGNORECASE))


def _term_spans(text: str, term: str, *, limit: int = 8) -> list[tuple[int, int]]:
    if not text or not term:
        return []
    pattern = (
        rf"(?<![A-Za-z0-9_.-]){re.escape(term)}(?![A-Za-z0-9_.-])"
        if re.fullmatch(r"[A-Za-z0-9_.-]+", term)
        else re.escape(term)
    )
    return [match.span() for match in list(re.finditer(pattern, text, re.IGNORECASE))[:limit]]


def _normalize_arc_profiles(raw_profiles: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for raw in raw_profiles:
        arc_key = str(raw.get("arc_key") or "").strip()
        if not arc_key:
            continue
        members: list[dict[str, str]] = []
        for member in raw.get("members") or []:
            if not isinstance(member, dict):
                continue
            kind = str(member.get("owner_kind") or "").strip().lower()
            owner_id = str(member.get("owner_id") or "").strip()
            if kind in {"scene", "event"} and owner_id:
                members.append({"owner_kind": kind, "owner_id": owner_id})
        output.append(
            {
                "arc_key": arc_key,
                "title": _surface(raw.get("title")),
                "title_aliases": list(
                    dict.fromkeys(
                        _surface(value)
                        for value in raw.get("title_aliases") or []
                        if _surface(value)
                    )
                ),
                "primary_entities": list(
                    dict.fromkeys(
                        _surface(value)
                        for value in raw.get("primary_entities") or []
                        if _surface(value)
                    )
                ),
                "supporting_entities": list(
                    dict.fromkeys(
                        _surface(value)
                        for value in raw.get("supporting_entities") or []
                        if _surface(value)
                    )
                ),
                "members": members,
            }
        )
    return output


def _known_terms(profiles: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    fields = (
        ("title", "title"),
        ("title_aliases", "title_alias"),
        ("primary_entities", "primary_entity"),
        ("supporting_entities", "supporting_entity"),
    )
    for profile in profiles:
        for field, role in fields:
            values = [profile.get(field)] if field == "title" else profile.get(field) or []
            for value in values:
                text = _surface(value)
                key = _key(text)
                if not key:
                    continue
                row = output.setdefault(
                    key,
                    {"entity_key": key, "entity_text": text, "roles": [], "arc_keys": []},
                )
                if role not in row["roles"]:
                    row["roles"].append(role)
                if profile["arc_key"] not in row["arc_keys"]:
                    row["arc_keys"].append(profile["arc_key"])
    return output


def _dynamic_terms(text: str) -> set[str]:
    output = set(_QUOTED_WORK.findall(text))
    output.update(_QUOTED_NAME.findall(text))
    output.update(_LATIN_NAME.findall(text))
    if jieba_posseg is not None:
        for word, flag in jieba_posseg.cut(text):
            value = _surface(word)
            if flag in _DYNAMIC_POS and 2 <= len(value) <= 24:
                output.add(value)
    return output


def extract_observed_entities(
    source_refs: Iterable[dict[str, Any]],
    *,
    known_terms: dict[str, dict[str, Any]],
    stop_keys: frozenset[str],
) -> list[dict[str, Any]]:
    """Extract bounded entity observations from exact bound source text.

    The output stores only entity spans and source identifiers. It never stores
    or exposes the source transcript as a query-time recall route.
    """

    sources: list[dict[str, Any]] = []
    for ordinal, ref in enumerate(source_refs):
        content = str(ref.get("content") or "") if isinstance(ref, dict) else ""
        if not content:
            continue
        sources.append(
            {
                "ordinal": ordinal,
                "source_system": str(ref.get("source_system") or ""),
                "session_id": str(ref.get("session_id") or ref.get("thread_id") or ""),
                "message_id": str(ref.get("message_id") or ordinal),
                "content": content,
            }
        )
    if not sources:
        return []

    candidates: dict[str, str] = {
        key: str(row["entity_text"]) for key, row in known_terms.items()
    }
    for source in sources:
        for value in _dynamic_terms(source["content"]):
            key = _key(value)
            if key and key not in candidates:
                candidates[key] = _surface(value)

    rows: list[dict[str, Any]] = []
    for entity_key, entity_text in candidates.items():
        if entity_key in stop_keys or len(entity_key) < 2 or entity_key.isdigit():
            continue
        total = 0
        support_sources = 0
        supports: list[dict[str, Any]] = []
        for source in sources:
            count = _term_count(source["content"], entity_text)
            if not count:
                continue
            total += count
            support_sources += 1
            for start, end in _term_spans(source["content"], entity_text):
                supports.append(
                    {
                        "source_ordinal": source["ordinal"],
                        "source_system": source["source_system"],
                        "session_id": source["session_id"],
                        "message_id": source["message_id"],
                        "start_offset": start,
                        "end_offset": end,
                        "text": source["content"][start:end],
                    }
                )
        explicit_work = any(
            _key(value) == entity_key
            for source in sources
            for value in _QUOTED_WORK.findall(source["content"])
        )
        known = known_terms.get(entity_key) or {}
        known_title = bool(set(known.get("roles") or []).intersection({"title", "title_alias"}))
        repeated = total >= 2 or support_sources >= 2
        if not (repeated or explicit_work or known_title):
            continue
        if explicit_work:
            basis = "explicit_work_title"
        elif repeated:
            basis = "repeated_bound_source"
        else:
            basis = "known_arc_title"
        rows.append(
            {
                "entity_key": entity_key,
                "entity_text": entity_text,
                "occurrence_count": total,
                "source_count": support_sources,
                "confidence_basis": basis,
                "known_roles": sorted(known.get("roles") or []),
                "known_arc_keys": sorted(known.get("arc_keys") or []),
                "supports": supports[:8],
            }
        )
    rows.sort(
        key=lambda row: (
            -int(row["source_count"]),
            -int(row["occurrence_count"]),
            row["entity_key"],
        )
    )
    return rows[:80]


def _query_intent(query: str) -> tuple[str, str]:
    for intent, operator, pattern in _INTENT_PATTERNS:
        if pattern.search(query):
            return intent, operator
    return "none", "none"


class ObservedEntityShadowIndex:
    """Incremental observed-entity, Arc-link-candidate, and scope sidecar."""

    def __init__(self, config: dict[str, Any]):
        state_dir = str(
            config.get("state_dir")
            or os.path.join(
                os.path.dirname(os.path.abspath(config.get("buckets_dir", "buckets"))),
                "state",
            )
        )
        self.db_path = os.path.join(state_dir, "observed_entity_shadow.sqlite")
        names = identity_names(config or {})
        stop_values = {
            *GENERIC_LEXICAL_STOPWORDS,
            "我",
            "你",
            "她",
            "他",
            "我们",
            "你们",
            "他们",
            "哥哥",
            "老婆",
            "老公",
            "小雨",
            "Haven",
            "事情",
            "内容",
            "记忆",
            "片段",
            "问题",
            "今天",
            "昨天",
            "明天",
        }
        for field in ("relationship_terms", "user_aliases", "assistant_aliases"):
            stop_values.update(names.get(field) or [])
        self.stop_keys = frozenset(_key(value) for value in stop_values if _key(value))

    def _connect(self, *, readonly: bool = False) -> sqlite3.Connection:
        if readonly:
            conn = sqlite3.connect(
                f"{Path(self.db_path).resolve().as_uri()}?mode=ro", uri=True, timeout=10.0
            )
        else:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS observed_entities (
                    owner_kind TEXT NOT NULL CHECK(owner_kind IN ('scene', 'event')),
                    owner_id TEXT NOT NULL,
                    entity_key TEXT NOT NULL,
                    entity_text TEXT NOT NULL,
                    occurrence_count INTEGER NOT NULL,
                    source_count INTEGER NOT NULL,
                    confidence_basis TEXT NOT NULL,
                    known_roles_json TEXT NOT NULL,
                    known_arc_keys_json TEXT NOT NULL,
                    supports_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(owner_kind, owner_id, entity_key)
                );
                CREATE INDEX IF NOT EXISTS idx_observed_entity_key
                    ON observed_entities(entity_key, owner_kind, owner_id);

                CREATE TABLE IF NOT EXISTS observed_entity_owner_state (
                    owner_kind TEXT NOT NULL CHECK(owner_kind IN ('scene', 'event')),
                    owner_id TEXT NOT NULL,
                    source_fingerprint TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(owner_kind, owner_id)
                );

                CREATE TABLE IF NOT EXISTS arc_observed_entities (
                    arc_key TEXT NOT NULL,
                    entity_key TEXT NOT NULL,
                    entity_text TEXT NOT NULL,
                    member_count INTEGER NOT NULL,
                    occurrence_count INTEGER NOT NULL,
                    members_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(arc_key, entity_key)
                );
                CREATE INDEX IF NOT EXISTS idx_arc_observed_entity_key
                    ON arc_observed_entities(entity_key, arc_key);

                CREATE TABLE IF NOT EXISTS scope_anchors (
                    entity_key TEXT NOT NULL,
                    entity_text TEXT NOT NULL,
                    arc_key TEXT NOT NULL,
                    source_kind TEXT NOT NULL,
                    trusted INTEGER NOT NULL CHECK(trusted IN (0, 1)),
                    arc_count INTEGER NOT NULL,
                    member_count INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(entity_key, arc_key, source_kind)
                );
                CREATE INDEX IF NOT EXISTS idx_scope_anchor_key
                    ON scope_anchors(entity_key, trusted, arc_key);

                CREATE TABLE IF NOT EXISTS owner_arc_link_candidates (
                    owner_kind TEXT NOT NULL CHECK(owner_kind IN ('scene', 'event')),
                    owner_id TEXT NOT NULL,
                    arc_key TEXT NOT NULL,
                    score INTEGER NOT NULL,
                    signals_json TEXT NOT NULL,
                    admission_eligible INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(owner_kind, owner_id, arc_key)
                );
                """
            )

    def _owner_state(self) -> dict[tuple[str, str], str]:
        if not os.path.exists(self.db_path):
            return {}
        with closing(self._connect(readonly=True)) as conn:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                ("observed_entity_owner_state",),
            ).fetchone()
            if not exists:
                return {}
            rows = conn.execute(
                "SELECT owner_kind, owner_id, source_fingerprint FROM observed_entity_owner_state"
            ).fetchall()
        return {
            (str(row["owner_kind"]), str(row["owner_id"])): str(row["source_fingerprint"])
            for row in rows
        }

    def _stored_owner_entities(
        self,
        owner_keys: set[tuple[str, str]],
    ) -> dict[tuple[str, str], list[dict[str, Any]]]:
        output: dict[tuple[str, str], list[dict[str, Any]]] = {
            owner_key: [] for owner_key in owner_keys
        }
        if not owner_keys or not os.path.exists(self.db_path):
            return output
        with closing(self._connect(readonly=True)) as conn:
            rows = conn.execute(
                "SELECT * FROM observed_entities ORDER BY owner_kind, owner_id, entity_key"
            ).fetchall()
        for row in rows:
            owner_key = (str(row["owner_kind"]), str(row["owner_id"]))
            if owner_key not in output:
                continue
            output[owner_key].append(
                {
                    "entity_key": str(row["entity_key"]),
                    "entity_text": str(row["entity_text"]),
                    "occurrence_count": int(row["occurrence_count"]),
                    "source_count": int(row["source_count"]),
                    "confidence_basis": str(row["confidence_basis"]),
                    "known_roles": json.loads(str(row["known_roles_json"] or "[]")),
                    "known_arc_keys": json.loads(
                        str(row["known_arc_keys_json"] or "[]")
                    ),
                    "supports": json.loads(str(row["supports_json"] or "[]")),
                }
            )
        return output

    def _stored_owner_keys(self) -> set[tuple[str, str]]:
        if not os.path.exists(self.db_path):
            return set()
        with closing(self._connect(readonly=True)) as conn:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                ("observed_entities",),
            ).fetchone()
            if not exists:
                return set()
            rows = conn.execute(
                "SELECT DISTINCT owner_kind, owner_id FROM observed_entities"
            ).fetchall()
        return {(str(row["owner_kind"]), str(row["owner_id"])) for row in rows}

    @staticmethod
    def _reconcile_rows(
        conn: sqlite3.Connection,
        *,
        table: str,
        key_columns: tuple[str, ...],
        value_columns: tuple[str, ...],
        rows: Iterable[dict[str, Any]],
        now: str,
    ) -> dict[str, int]:
        desired = {
            tuple(row[column] for column in key_columns): tuple(
                row[column] for column in value_columns
            )
            for row in rows
        }
        selected = ", ".join((*key_columns, *value_columns))
        existing = {
            tuple(row[column] for column in key_columns): tuple(
                row[column] for column in value_columns
            )
            for row in conn.execute(f"SELECT {selected} FROM {table}").fetchall()
        }
        removed_keys = existing.keys() - desired.keys()
        added_keys = desired.keys() - existing.keys()
        updated_keys = {
            key for key in desired.keys() & existing.keys() if desired[key] != existing[key]
        }
        where = " AND ".join(f"{column}=?" for column in key_columns)
        conn.executemany(
            f"DELETE FROM {table} WHERE {where}",
            sorted(removed_keys),
        )
        insert_columns = (*key_columns, *value_columns, "updated_at")
        placeholders = ", ".join("?" for _ in insert_columns)
        conn.executemany(
            f"INSERT INTO {table} ({', '.join(insert_columns)}) VALUES ({placeholders})",
            [(*key, *desired[key], now) for key in sorted(added_keys)],
        )
        assignments = ", ".join(
            [*(f"{column}=?" for column in value_columns), "updated_at=?"]
        )
        conn.executemany(
            f"UPDATE {table} SET {assignments} WHERE {where}",
            [(*desired[key], now, *key) for key in sorted(updated_keys)],
        )
        return {
            "inserted": len(added_keys),
            "updated": len(updated_keys),
            "deleted": len(removed_keys),
            "unchanged": len(desired) - len(added_keys) - len(updated_keys),
        }

    def sync(
        self,
        *,
        owners: Iterable[dict[str, Any]],
        arc_profiles: Iterable[dict[str, Any]],
        dry_run: bool = False,
    ) -> dict[str, Any]:
        profiles = _normalize_arc_profiles(arc_profiles)
        known = _known_terms(profiles)
        extraction_contract_hash = _stable_hash(
            {
                "schema_version": SCHEMA_VERSION,
                "known_terms": known,
                "stop_keys": sorted(self.stop_keys),
            }
        )
        normalized_owners: dict[tuple[str, str], dict[str, Any]] = {}
        for raw in owners:
            kind = str(raw.get("owner_kind") or "").strip().lower()
            owner_id = str(raw.get("owner_id") or "").strip()
            if kind not in {"scene", "event"} or not owner_id:
                continue
            source_refs = raw.get("source_refs") or []
            normalized_owners[(kind, owner_id)] = {
                "owner_kind": kind,
                "owner_id": owner_id,
                "source_refs": source_refs,
                "source_fingerprint": _source_fingerprint(
                    source_refs,
                    extraction_contract_hash=extraction_contract_hash,
                ),
            }
        stored_state = self._owner_state()
        stored_owner_keys = self._stored_owner_keys()
        current_keys = set(normalized_owners)
        changed_keys = {
            owner_key
            for owner_key, owner in normalized_owners.items()
            if stored_state.get(owner_key) != owner["source_fingerprint"]
        }
        unchanged_keys = current_keys - changed_keys
        deleted_keys = (set(stored_state) | stored_owner_keys) - current_keys
        owner_rows = self._stored_owner_entities(unchanged_keys)
        for owner_key in changed_keys:
            owner = normalized_owners[owner_key]
            owner_rows[owner_key] = extract_observed_entities(
                owner["source_refs"],
                known_terms=known,
                stop_keys=self.stop_keys,
            )
        entity_count = sum(len(entities) for entities in owner_rows.values())
        if dry_run:
            return {
                "status": "dry_run",
                "owners": len(normalized_owners),
                "owners_extracted": len(changed_keys),
                "owners_unchanged": len(unchanged_keys),
                "owners_deleted": len(deleted_keys),
                "observed_entities": entity_count,
                "arcs": len(profiles),
                "decision_applied": False,
                "canonical_writes": False,
            }

        self._init_db()
        now = _now()
        profile_by_arc = {profile["arc_key"]: profile for profile in profiles}
        arc_aggregates: dict[tuple[str, str], dict[str, Any]] = {}
        for profile in profiles:
            for member in profile["members"]:
                member_key = (member["owner_kind"], member["owner_id"])
                for entity in owner_rows.get(member_key, []):
                    key = (profile["arc_key"], entity["entity_key"])
                    aggregate = arc_aggregates.setdefault(
                        key,
                        {
                            "entity_text": entity["entity_text"],
                            "members": set(),
                            "occurrence_count": 0,
                        },
                    )
                    aggregate["members"].add(member_key)
                    aggregate["occurrence_count"] += int(entity["occurrence_count"])

        authored_by_key: dict[str, list[dict[str, str]]] = defaultdict(list)
        for profile in profiles:
            for field, role in (
                ("title", "title"),
                ("title_aliases", "title_alias"),
                ("primary_entities", "primary_entity"),
                ("supporting_entities", "supporting_entity"),
            ):
                values = [profile[field]] if field == "title" else profile[field]
                for value in values:
                    entity_key = _key(value)
                    if entity_key and entity_key not in self.stop_keys:
                        authored_by_key[entity_key].append(
                            {"arc_key": profile["arc_key"], "entity_text": value, "role": role}
                        )
        observed_arcs_by_key: dict[str, set[str]] = defaultdict(set)
        for arc_key, entity_key in arc_aggregates:
            observed_arcs_by_key[entity_key].add(arc_key)

        scope_rows: list[dict[str, Any]] = []
        for entity_key, authored_rows in authored_by_key.items():
            arc_keys = {row["arc_key"] for row in authored_rows}
            for row in authored_rows:
                trusted = len(arc_keys) == 1
                scope_rows.append(
                    {
                        **row,
                        "entity_key": entity_key,
                        "source_kind": row["role"],
                        "trusted": trusted,
                        "arc_count": len(arc_keys),
                        "member_count": 0,
                    }
                )
        for (arc_key, entity_key), aggregate in arc_aggregates.items():
            member_count = len(aggregate["members"])
            if member_count < 2 or entity_key in authored_by_key:
                continue
            arc_count = len(observed_arcs_by_key[entity_key])
            scope_rows.append(
                {
                    "entity_key": entity_key,
                    "entity_text": aggregate["entity_text"],
                    "arc_key": arc_key,
                    "source_kind": "observed",
                    "trusted": arc_count == 1,
                    "arc_count": arc_count,
                    "member_count": member_count,
                }
            )

        link_rows: list[dict[str, Any]] = []
        for owner_key, entities in owner_rows.items():
            entity_by_key = {entity["entity_key"]: entity for entity in entities}
            for arc_key, profile in profile_by_arc.items():
                signals: list[dict[str, Any]] = []
                score = 0
                for entity_key in sorted(entity_by_key):
                    entity = entity_by_key[entity_key]
                    authored_matches = [
                        row
                        for row in authored_by_key.get(entity_key, [])
                        if row["arc_key"] == arc_key
                    ]
                    for authored in authored_matches:
                        role = authored["role"]
                        weight = 100 if role in {"title", "title_alias"} else 35
                        if role not in {"title", "title_alias"} and int(entity["occurrence_count"]) < 2:
                            continue
                        score += weight
                        signals.append(
                            {
                                "kind": f"authored_{role}",
                                "entity": entity["entity_text"],
                                "occurrence_count": entity["occurrence_count"],
                            }
                        )
                    aggregate = arc_aggregates.get((arc_key, entity_key))
                    if aggregate:
                        other_members = {
                            member for member in aggregate["members"] if member != owner_key
                        }
                        if other_members:
                            weight = 20 + min(20, len(other_members) * 5)
                            score += weight
                            signals.append(
                                {
                                    "kind": "repeated_observed_entity",
                                    "entity": entity["entity_text"],
                                    "candidate_occurrence_count": entity["occurrence_count"],
                                    "supporting_member_count": len(other_members),
                                }
                            )
                if not signals:
                    continue
                link_rows.append(
                    {
                        "owner_kind": owner_key[0],
                        "owner_id": owner_key[1],
                        "arc_key": arc_key,
                        "score": score,
                        "signals": signals,
                    }
                )

        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            for owner_key in sorted(changed_keys | deleted_keys):
                conn.execute(
                    "DELETE FROM observed_entities WHERE owner_kind=? AND owner_id=?",
                    owner_key,
                )
            for owner_key in sorted(changed_keys):
                for entity in owner_rows[owner_key]:
                    conn.execute(
                        """
                        INSERT INTO observed_entities(
                            owner_kind, owner_id, entity_key, entity_text,
                            occurrence_count, source_count, confidence_basis,
                            known_roles_json, known_arc_keys_json, supports_json, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            owner_key[0],
                            owner_key[1],
                            entity["entity_key"],
                            entity["entity_text"],
                            entity["occurrence_count"],
                            entity["source_count"],
                            entity["confidence_basis"],
                            json.dumps(entity["known_roles"], ensure_ascii=False),
                            json.dumps(entity["known_arc_keys"], ensure_ascii=False),
                            json.dumps(entity["supports"], ensure_ascii=False),
                            now,
                        ),
                    )
            conn.executemany(
                "DELETE FROM observed_entity_owner_state WHERE owner_kind=? AND owner_id=?",
                sorted(deleted_keys),
            )
            conn.executemany(
                """
                INSERT INTO observed_entity_owner_state(
                    owner_kind, owner_id, source_fingerprint, updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(owner_kind, owner_id) DO UPDATE SET
                    source_fingerprint=excluded.source_fingerprint,
                    updated_at=excluded.updated_at
                """,
                [
                    (*owner_key, normalized_owners[owner_key]["source_fingerprint"], now)
                    for owner_key in sorted(changed_keys)
                ],
            )
            arc_rows = []
            for (arc_key, entity_key), aggregate in arc_aggregates.items():
                members = [
                    {"owner_kind": kind, "owner_id": owner_id}
                    for kind, owner_id in sorted(aggregate["members"])
                ]
                arc_rows.append(
                    {
                        "arc_key": arc_key,
                        "entity_key": entity_key,
                        "entity_text": aggregate["entity_text"],
                        "member_count": len(members),
                        "occurrence_count": aggregate["occurrence_count"],
                        "members_json": json.dumps(members, ensure_ascii=False),
                    }
                )
            reconciliation = {
                "arc_observed_entities": self._reconcile_rows(
                    conn,
                    table="arc_observed_entities",
                    key_columns=("arc_key", "entity_key"),
                    value_columns=(
                        "entity_text",
                        "member_count",
                        "occurrence_count",
                        "members_json",
                    ),
                    rows=arc_rows,
                    now=now,
                ),
                "scope_anchors": self._reconcile_rows(
                    conn,
                    table="scope_anchors",
                    key_columns=("entity_key", "arc_key", "source_kind"),
                    value_columns=("entity_text", "trusted", "arc_count", "member_count"),
                    rows=(
                        {
                            **row,
                            "trusted": int(bool(row["trusted"])),
                        }
                        for row in scope_rows
                    ),
                    now=now,
                ),
                "owner_arc_link_candidates": self._reconcile_rows(
                    conn,
                    table="owner_arc_link_candidates",
                    key_columns=("owner_kind", "owner_id", "arc_key"),
                    value_columns=("score", "signals_json", "admission_eligible"),
                    rows=(
                        {
                            **row,
                            "signals_json": json.dumps(row["signals"], ensure_ascii=False),
                            "admission_eligible": 0,
                        }
                        for row in link_rows
                    ),
                    now=now,
                ),
            }
            conn.commit()
        return {
            "status": "ok",
            "schema_version": SCHEMA_VERSION,
            "owners": len(normalized_owners),
            "owners_extracted": len(changed_keys),
            "owners_unchanged": len(unchanged_keys),
            "owners_deleted": len(deleted_keys),
            "observed_entities": entity_count,
            "arcs": len(profiles),
            "arc_observed_entities": len(arc_aggregates),
            "scope_anchors": len(scope_rows),
            "link_candidates": len(link_rows),
            "reconciliation": reconciliation,
            "decision_applied": False,
            "canonical_writes": False,
            "source_text_queryable": False,
        }

    def owner_entities(self, owner_kind: str, owner_id: str) -> list[dict[str, Any]]:
        if not os.path.exists(self.db_path):
            return []
        with closing(self._connect(readonly=True)) as conn:
            rows = conn.execute(
                """
                SELECT * FROM observed_entities
                WHERE owner_kind=? AND owner_id=?
                ORDER BY source_count DESC, occurrence_count DESC, entity_key
                """,
                (str(owner_kind or "").lower(), str(owner_id or "")),
            ).fetchall()
        return [
            {
                "entity": str(row["entity_text"]),
                "occurrence_count": int(row["occurrence_count"]),
                "source_count": int(row["source_count"]),
                "confidence_basis": str(row["confidence_basis"]),
                "supports": json.loads(str(row["supports_json"] or "[]")),
            }
            for row in rows
        ]

    def link_candidates(self, owner_kind: str, owner_id: str) -> list[dict[str, Any]]:
        if not os.path.exists(self.db_path):
            return []
        with closing(self._connect(readonly=True)) as conn:
            rows = conn.execute(
                """
                SELECT * FROM owner_arc_link_candidates
                WHERE owner_kind=? AND owner_id=?
                ORDER BY score DESC, arc_key
                """,
                (str(owner_kind or "").lower(), str(owner_id or "")),
            ).fetchall()
        return [
            {
                "arc_key": str(row["arc_key"]),
                "score": int(row["score"]),
                "signals": json.loads(str(row["signals_json"] or "[]")),
                "admission_eligible": False,
                "candidate_only": True,
            }
            for row in rows
        ]

    def resolve_query(self, query: str) -> dict[str, Any]:
        text = str(query or "").strip()
        intent, operator = _query_intent(text)
        if not text:
            return {
                "status": "no_scope",
                "intent": intent,
                "operator": operator,
                "scope_anchor": None,
                "retrieval_allowed": False,
                "decision_applied": False,
            }
        if not os.path.exists(self.db_path):
            return {
                "status": "scope_index_unavailable",
                "intent": intent,
                "operator": operator,
                "scope_anchor": None,
                "retrieval_allowed": False,
                "decision_applied": False,
            }
        with closing(self._connect(readonly=True)) as conn:
            rows = conn.execute(
                "SELECT * FROM scope_anchors ORDER BY LENGTH(entity_text) DESC, entity_key, arc_key"
            ).fetchall()
        matches: list[dict[str, Any]] = []
        occupied: list[tuple[int, int]] = []
        grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
        entity_order: list[str] = []
        for row in rows:
            entity_key = str(row["entity_key"])
            if entity_key not in grouped:
                entity_order.append(entity_key)
            grouped[entity_key].append(row)
        entity_order.sort(
            key=lambda key: (-len(str(grouped[key][0]["entity_text"])), key)
        )
        for entity_key in entity_order:
            mappings = grouped[entity_key]
            entity = str(mappings[0]["entity_text"])
            for start, end in _term_spans(text, entity, limit=4):
                if any(not (end <= old_start or start >= old_end) for old_start, old_end in occupied):
                    continue
                occupied.append((start, end))
                for row in mappings:
                    matches.append(
                        {
                            "entity": text[start:end],
                            "start_offset": start,
                            "end_offset": end,
                            "arc_key": str(row["arc_key"]),
                            "source_kind": str(row["source_kind"]),
                            "trusted": bool(row["trusted"]),
                            "arc_count": int(row["arc_count"]),
                        }
                    )
                break
        trusted_arc_keys = sorted({row["arc_key"] for row in matches if row["trusted"]})
        residue = text
        for start, end in sorted(
            {(row["start_offset"], row["end_offset"]) for row in matches}, reverse=True
        ):
            residue = residue[:start] + " " * (end - start) + residue[end:]
        residue = " ".join(residue.split())
        if len(trusted_arc_keys) == 1:
            chosen = next(
                row for row in matches if row["trusted"] and row["arc_key"] == trusted_arc_keys[0]
            )
            scope_anchor = {
                "entity": chosen["entity"],
                "arc_key": chosen["arc_key"],
                "source_kind": chosen["source_kind"],
            }
            return {
                "status": "scoped_recall" if intent != "none" else "scope_only",
                "intent": intent,
                "operator": operator,
                "intent_view": residue,
                "scope_anchor": scope_anchor,
                "matches": matches,
                "retrieval_allowed": intent != "none",
                "decision_applied": False,
            }
        if matches:
            return {
                "status": "ambiguous_scope",
                "intent": intent,
                "operator": operator,
                "intent_view": residue,
                "scope_anchor": None,
                "candidate_arc_keys": sorted({row["arc_key"] for row in matches}),
                "matches": matches,
                "retrieval_allowed": False,
                "decision_applied": False,
            }
        return {
            "status": "insufficient_scope" if intent != "none" else "no_scope",
            "intent": intent,
            "operator": operator,
            "intent_view": text,
            "scope_anchor": None,
            "matches": [],
            "retrieval_allowed": False,
            "decision_applied": False,
        }
