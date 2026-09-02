from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from typing import Any

from query_terms import GENERIC_LEXICAL_STOPWORDS


_SOURCE_TYPES = {"event", "scene"}
_GLOBAL_STOP_TERMS = {
    *GENERIC_LEXICAL_STOPWORDS,
    "haven",
    "小雨",
    "老公",
    "老婆",
    "event",
    "scene",
    "arc",
    "token",
    "tokens",
}
_ASCII_TERM_RE = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_.+/#:-]{2,40}(?![A-Za-z0-9_])")
_BOOK_TERM_RE = re.compile(r"《([^》\r\n]{2,40})》")
_CJK_RUN_RE = re.compile(r"[\u3400-\u9fff]{2,24}")
_CJK_SPLIT_RE = re.compile(
    r"(?:我们|你们|他们|她们|这个|那个|这些|那些|已经|还是|然后|但是|因为|所以|以及|关于|通过|继续|开始|后来|当前|今天|昨天|明天|一次|一种|一个|一些|可以|需要|觉得|发现|讨论|记录|进行|完成|问题|事情|东西|内容|时候|里面|的话|之后|之前|现在|没有|不是|终于|一起)"
)


def _compact(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split()).strip()


def _material_key(item: dict[str, Any]) -> str:
    source_type = str(item.get("source_type") or "").strip().lower()
    source_id = str(item.get("source_id") or "").strip()
    return f"{source_type}:{source_id}" if source_type in _SOURCE_TYPES and source_id else ""


def _material_text(item: dict[str, Any]) -> str:
    return "\n".join(
        value
        for value in (
            _compact(item.get("title")),
            _compact(item.get("summary")),
            _compact(item.get("source_excerpt")),
            _compact(item.get("search_text")),
        )
        if value
    )


def _normalized_term(value: Any) -> str:
    return _compact(value).strip("_-—·:：,，。.!！?？/ ").casefold()


def _term_allowed(term: str) -> bool:
    normalized = _normalized_term(term)
    if len(normalized) < 2 or normalized in _GLOBAL_STOP_TERMS:
        return False
    if normalized.isdigit() or re.fullmatch(r"20\d{2}(?:[-/.]\d{1,2}){0,2}", normalized):
        return False
    if len(normalized) == 2 and normalized.endswith(("的", "了", "是", "在", "和", "与")):
        return False
    return True


def _raw_terms(text: str, *, title: str = "") -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for value in _BOOK_TERM_RE.findall(text):
        term = _normalized_term(value)
        if _term_allowed(term):
            rows.append((term, "explicit_work"))
    for value in _ASCII_TERM_RE.findall(text):
        term = _normalized_term(value)
        if _term_allowed(term):
            rows.append((term, "ascii"))
    for run in _CJK_RUN_RE.findall(text):
        chunks = [part for part in _CJK_SPLIT_RE.split(run) if part]
        for chunk in chunks:
            if 2 <= len(chunk) <= 10 and _term_allowed(chunk):
                rows.append((_normalized_term(chunk), "cjk_phrase"))
            if len(chunk) > 4:
                for width in (4, 3, 2):
                    for start in range(0, len(chunk) - width + 1):
                        term = _normalized_term(chunk[start : start + width])
                        if _term_allowed(term):
                            rows.append((term, "cjk_ngram"))
    compact_title = _normalized_term(title)
    if 2 <= len(compact_title) <= 32 and _term_allowed(compact_title):
        rows.append((compact_title, "title"))
    return rows


def build_material_document_frequency(materials: list[dict[str, Any]]) -> dict[str, int]:
    frequency: Counter[str] = Counter()
    for item in materials:
        if not _material_key(item):
            continue
        unique = {term for term, _kind in _raw_terms(_material_text(item), title=str(item.get("title") or ""))}
        frequency.update(unique)
    return dict(frequency)


def extract_seed_keywords(
    seed: dict[str, Any],
    document_frequency: dict[str, int],
    *,
    max_keywords: int = 8,
) -> list[dict[str, Any]]:
    text = _material_text(seed)
    occurrences = Counter(term for term, _kind in _raw_terms(text, title=str(seed.get("title") or "")))
    kinds: dict[str, set[str]] = {}
    for term, kind in _raw_terms(text, title=str(seed.get("title") or "")):
        kinds.setdefault(term, set()).add(kind)
    candidates = []
    for term, count in occurrences.items():
        df = int(document_frequency.get(term) or 0)
        # A keyword must lead somewhere else in the active library.  Unique words
        # remain useful prose, but cannot define a one-hop review corridor.
        if df < 2:
            continue
        kind_bonus = 6 if "explicit_work" in kinds.get(term, set()) else 3 if "title" in kinds.get(term, set()) else 0
        score = kind_bonus + min(len(term), 12) + math.log1p(count) * 2 - math.log1p(df) * 1.5
        candidates.append(
            {
                "term": term,
                "document_frequency": df,
                "occurrences": count,
                "origins": sorted(kinds.get(term, set())),
                "score": round(score, 6),
            }
        )
    candidates.sort(
        key=lambda row: (
            -int("explicit_work" in row["origins"]),
            -float(row["score"]),
            int(row["document_frequency"]),
            -len(row["term"]),
            row["term"],
        )
    )
    return candidates[: max(1, min(int(max_keywords), 16))]


def build_keyword_corridors(
    materials: list[dict[str, Any]],
    seed_keys: list[str],
    *,
    max_keywords: int = 8,
    max_candidates_per_seed: int = 12,
) -> list[dict[str, Any]]:
    """Build bounded one-hop lexical corridors over the complete active inventory."""

    material_by_key = {_material_key(item): dict(item) for item in materials if _material_key(item)}
    document_frequency = build_material_document_frequency(list(material_by_key.values()))
    corridors: list[dict[str, Any]] = []
    for seed_key in list(dict.fromkeys(seed_keys)):
        seed = material_by_key.get(seed_key)
        if not seed or list(seed.get("bound_narrative_ids") or []):
            continue
        keywords = extract_seed_keywords(seed, document_frequency, max_keywords=max_keywords)
        if not keywords:
            continue
        ranked: list[tuple[tuple[Any, ...], dict[str, Any], list[str]]] = []
        for candidate_key, candidate in material_by_key.items():
            if candidate_key == seed_key:
                continue
            if list(candidate.get("bound_narrative_ids") or []):
                continue
            normalized_text = _normalized_term(_material_text(candidate))
            matched = [row["term"] for row in keywords if row["term"] in normalized_text]
            if not matched:
                continue
            explicit_count = sum(
                1
                for row in keywords
                if row["term"] in matched and "explicit_work" in row["origins"]
            )
            candidate_date = re.sub(
                r"\D",
                "",
                str(candidate.get("updated_at") or candidate.get("date") or ""),
            )[:14]
            rank = (-explicit_count, -len(matched), -int(candidate_date or 0), candidate_key)
            ranked.append((rank, candidate, matched))
        ranked.sort(key=lambda row: (row[0][0], row[0][1], row[0][2], row[0][3]), reverse=False)
        selected = ranked[: max(1, min(int(max_candidates_per_seed), 24))]
        if not selected:
            continue
        candidate_rows = []
        for _rank, candidate, matched in selected:
            candidate_rows.append({**candidate, "matched_keywords": matched})
        corridors.append(
            {
                "seed": dict(seed),
                "seed_key": seed_key,
                "keywords": keywords,
                "candidates": candidate_rows,
                "inventory_size": len(material_by_key),
            }
        )
    return corridors


def _prompt_material(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_type": str(item.get("source_type") or ""),
        "source_id": str(item.get("source_id") or ""),
        "date": str(item.get("date") or ""),
        "title": str(item.get("title") or "")[:160],
        "summary": str(item.get("summary") or "")[:700],
        "source_excerpt": str(item.get("source_excerpt") or "")[:1400],
        "bound_narrative_ids": list(item.get("bound_narrative_ids") or [])[:8],
        "matched_keywords": list(item.get("matched_keywords") or [])[:8],
    }


def build_new_roll_candidate_prompt(
    corridors: list[dict[str, Any]],
    *,
    role_rules: str,
) -> list[dict[str, str]]:
    """Build a bounded Terra task that can only propose review candidates."""

    payload = []
    for corridor in corridors[:24]:
        seed = corridor.get("seed") if isinstance(corridor.get("seed"), dict) else {}
        payload.append(
            {
                "seed": _prompt_material(seed),
                "keywords": list(corridor.get("keywords") or [])[:8],
                "one_hop_candidates": [
                    _prompt_material(item)
                    for item in corridor.get("candidates") or []
                    if isinstance(item, dict)
                ][:24],
            }
        )
    return [
        {"role": "system", "content": str(role_rules or "").strip()},
        {
            "role": "user",
            "content": (
                "只返回 JSON：{\"candidates\":[{\"seed_source_type\":\"event|scene\","
                "\"seed_source_id\":\"...\",\"title\":\"暂定卷名\","
                "\"reason\":\"为什么是一条持续叙事\",\"materials\":["
                "{\"source_type\":\"event|scene\",\"source_id\":\"...\"}],"
                "\"confidence\":\"high|medium\",\"latest_date\":\"YYYY-MM-DD\"}]}。"
                "没有可信候选时返回 {\"candidates\":[]}。\n\n"
                f"<keyword_corridors_json>{json.dumps(payload, ensure_ascii=False)}</keyword_corridors_json>"
            ),
        },
    ]


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = str(value or "").strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("new roll scout output must be an object")
    return parsed


def normalize_new_roll_candidates(
    value: Any,
    corridors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    raw = _json_object(value)
    candidates = raw.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("new roll scout candidates must be a list")

    allowed_by_seed: dict[str, dict[str, dict[str, Any]]] = {}
    seed_materials: dict[str, dict[str, Any]] = {}
    keyword_by_seed: dict[str, list[str]] = {}
    for corridor in corridors:
        seed = corridor.get("seed") if isinstance(corridor.get("seed"), dict) else {}
        seed_key = _material_key(seed)
        if not seed_key:
            continue
        rows = [seed, *(corridor.get("candidates") or [])]
        allowed_by_seed[seed_key] = {
            _material_key(row): row
            for row in rows
            if isinstance(row, dict)
            and _material_key(row)
            and not list(row.get("bound_narrative_ids") or [])
        }
        seed_materials[seed_key] = seed
        keyword_by_seed[seed_key] = [
            str(item.get("term") or "")
            for item in corridor.get("keywords") or []
            if isinstance(item, dict) and str(item.get("term") or "")
        ]

    normalized: list[dict[str, Any]] = []
    claimed: set[str] = set()
    for item in candidates[:24]:
        if not isinstance(item, dict):
            continue
        seed_key = f"{str(item.get('seed_source_type') or '').strip().lower()}:{str(item.get('seed_source_id') or '').strip()}"
        allowed = allowed_by_seed.get(seed_key)
        if not allowed:
            continue
        material_keys = []
        for material in item.get("materials") or []:
            if not isinstance(material, dict):
                continue
            key = f"{str(material.get('source_type') or '').strip().lower()}:{str(material.get('source_id') or '').strip()}"
            if key in allowed and key not in material_keys:
                material_keys.append(key)
        if seed_key not in material_keys or len(material_keys) < 2 or claimed.intersection(material_keys):
            continue
        title = str(item.get("title") or "").strip()[:120]
        reason = str(item.get("reason") or "").strip()[:500]
        confidence = str(item.get("confidence") or "").strip().lower()
        if not title or not reason or confidence not in {"high", "medium"}:
            continue
        source_event_ids = [key.split(":", 1)[1] for key in material_keys if key.startswith("event:")]
        source_scene_ids = [key.split(":", 1)[1] for key in material_keys if key.startswith("scene:")]
        if len(source_event_ids) + len(source_scene_ids) < 2:
            continue
        claimed.update(material_keys)
        seed = seed_materials[seed_key]
        normalized.append(
            {
                "title": title,
                "reason": reason,
                "source_event_ids": source_event_ids,
                "source_scene_ids": source_scene_ids,
                "seed_source_type": str(seed.get("source_type") or ""),
                "seed_source_id": str(seed.get("source_id") or ""),
                "matched_keywords": keyword_by_seed.get(seed_key, []),
                "confidence": confidence,
                "latest_date": str(item.get("latest_date") or "").strip()[:10],
            }
        )
    return normalized


async def propose_new_roll_candidates(
    *,
    client: Any,
    model: str,
    corridors: list[dict[str, Any]],
    role_rules: str,
    completion_options: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if client is None or not str(model or "").strip() or not corridors or not str(role_rules or "").strip():
        return []
    response = await client.chat.completions.create(
        model=str(model),
        messages=build_new_roll_candidate_prompt(corridors, role_rules=role_rules),
        **(completion_options or {}),
    )
    content = response.choices[0].message.content if response.choices else ""
    return normalize_new_roll_candidates(content, corridors)
