from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Iterable


_RECENT_QUERY_MARKERS = (
    "最近",
    "近期",
    "近来",
    "最新",
    "刚才",
    "刚刚",
    "上次",
    "后来",
    "后续",
    "现在",
    "目前",
)


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("owner_kind") or "").strip().lower(),
        str(row.get("owner_id") or "").strip(),
    )


def _score(row: dict[str, Any]) -> float | None:
    try:
        value = row.get("score")
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _memory_time(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.strptime(text[:10], "%Y-%m-%d")
        except ValueError:
            return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def rerank_lane_with_freshness(
    rows: Iterable[dict[str, Any]],
    *,
    query: str,
    default_weight: float = 0.015,
    recent_weight: float = 0.05,
) -> list[dict[str, Any]]:
    """Apply a bounded within-lane recency prior without changing evidence scores."""

    output = [deepcopy(row) for row in rows]
    explicit_recent = any(marker in str(query or "") for marker in _RECENT_QUERY_MARKERS)
    weight = max(0.0, min(0.1, recent_weight if explicit_recent else default_weight))
    dated = sorted(
        {
            _memory_time(row.get("memory_date"))
            for row in output
            if _memory_time(row.get("memory_date")) > 0
        }
    )
    date_rank = {
        value: (index / max(1, len(dated) - 1) if len(dated) > 1 else 1.0)
        for index, value in enumerate(dated)
    }
    for row in output:
        timestamp = _memory_time(row.get("memory_date"))
        freshness = date_rank.get(timestamp, 0.0)
        semantic_score = _score(row)
        row["freshness"] = {
            "memory_date": str(row.get("memory_date") or ""),
            "relative_rank": round(freshness, 4),
            "weight": round(weight, 4),
            "explicit_recent_intent": explicit_recent,
            "candidate_only": True,
        }
        row["rerank_score"] = (
            round(semantic_score + weight * freshness, 4)
            if semantic_score is not None
            else None
        )
    output.sort(
        key=lambda row: (
            row.get("rerank_score") is None,
            -float(row.get("rerank_score") or 0.0),
            -_memory_time(row.get("memory_date")),
            -float(row.get("score") or 0.0),
            str(row.get("owner_id") or ""),
        )
    )
    return output


def _extend_unique(target: dict[str, Any], source: dict[str, Any], field: str) -> None:
    values = list(target.get(field) or [])
    for value in source.get(field) or []:
        if value not in values:
            values.append(value)
    if values:
        target[field] = values


def _merge_scored_rows(
    rows: Iterable[dict[str, Any]],
    *,
    owner_kind: str,
    score_source: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in rows:
        row = deepcopy(raw)
        key = _key(row)
        if key[0] != owner_kind or not key[1]:
            continue
        score = _score(row)
        if score is None:
            continue
        current = merged.get(key)
        if current is None:
            row["candidate_sources"] = list(
                dict.fromkeys([*(row.get("candidate_sources") or []), score_source])
            )
            row["score_components"] = {score_source: round(score, 4)}
            row["score"] = round(score, 4)
            row["candidate_only"] = True
            row["decision_applied"] = False
            merged[key] = row
            continue
        _extend_unique(current, row, "candidate_sources")
        _extend_unique(current, {"candidate_sources": [score_source]}, "candidate_sources")
        current.setdefault("score_components", {})[score_source] = round(score, 4)
        if score > float(current.get("score") or 0.0):
            current["score"] = round(score, 4)
            if row.get("passages"):
                current["passages"] = deepcopy(row["passages"])
    return merged


def _attach_candidate_only_rows(
    merged: dict[tuple[str, str], dict[str, Any]],
    rows: Iterable[dict[str, Any]],
    *,
    owner_kind: str,
    source: str,
    strip_passages: bool,
) -> list[tuple[str, str]]:
    added: list[tuple[str, str]] = []
    for raw in rows:
        row = deepcopy(raw)
        key = _key(row)
        if key[0] != owner_kind or not key[1]:
            continue
        current = merged.get(key)
        if current is None:
            current = {
                "owner_kind": key[0],
                "owner_id": key[1],
                "score": None,
                "passages": [],
                "candidate_sources": [source],
                "score_components": {},
                "candidate_only": True,
                "decision_applied": False,
            }
            merged[key] = current
            added.append(key)
        else:
            _extend_unique(current, {"candidate_sources": [source]}, "candidate_sources")
        for field in (
            "matched_cues",
            "matched_terms",
            "specific_terms",
            "matched_spans",
        ):
            _extend_unique(current, row, field)
        if not strip_passages and not current.get("passages") and row.get("passages"):
            current["passages"] = deepcopy(row["passages"])
    return added


def _merge_score_maps(
    merged: dict[tuple[str, str], dict[str, Any]],
    incoming: dict[tuple[str, str], dict[str, Any]],
) -> None:
    for key, row in incoming.items():
        current = merged.get(key)
        if current is None:
            merged[key] = row
            continue
        _extend_unique(current, row, "candidate_sources")
        current.setdefault("score_components", {}).update(row.get("score_components") or {})
        if float(row.get("score") or 0.0) > float(current.get("score") or 0.0):
            current["score"] = row["score"]
            current["passages"] = deepcopy(row.get("passages") or [])


def build_scene_lane(
    passage_rows: Iterable[dict[str, Any]],
    cue_rows: Iterable[dict[str, Any]],
    whole_rows: Iterable[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Build a Scene lane from max(whole, passage); cues remain score-free."""

    merged = _merge_scored_rows(
        passage_rows,
        owner_kind="scene",
        score_source="scene_passage_embedding",
    )
    whole = _merge_scored_rows(
        whole_rows,
        owner_kind="scene",
        score_source="scene_whole_embedding",
    )
    _merge_score_maps(merged, whole)
    cue_only = _attach_candidate_only_rows(
        merged,
        cue_rows,
        owner_kind="scene",
        source="scene_cue_candidate",
        strip_passages=True,
    )
    cue_only_set = set(cue_only)
    rows = list(merged.values())
    rows.sort(
        key=lambda row: (
            _key(row) in cue_only_set,
            -float(row.get("score") or 0.0),
            str(row.get("owner_id") or ""),
        )
    )
    return rows


def build_event_lane(
    passage_rows: Iterable[dict[str, Any]],
    body_rows: Iterable[dict[str, Any]],
    lexical_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build an Event lane from max(whole, passage) cosine scores.

    Lexical matches are candidate-only metadata. Their BM25 values are not mixed
    with cosine similarity and therefore never replace the Event evidence score.
    """

    merged = _merge_scored_rows(
        passage_rows,
        owner_kind="event",
        score_source="event_passage_embedding",
    )
    body = _merge_scored_rows(
        body_rows,
        owner_kind="event",
        score_source="event_whole_embedding",
    )
    _merge_score_maps(merged, body)
    lexical_only = _attach_candidate_only_rows(
        merged,
        lexical_rows,
        owner_kind="event",
        source="event_lexical_candidate",
        strip_passages=True,
    )
    lexical_only_set = set(lexical_only)
    rows = list(merged.values())
    rows.sort(
        key=lambda row: (
            _key(row) in lexical_only_set,
            -float(row.get("score") or 0.0),
            str(row.get("owner_id") or ""),
        )
    )
    return rows


def balanced_typed_pool(
    lanes: list[tuple[str, list[dict[str, Any]], int]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Interleave typed lanes without comparing scores across object kinds."""

    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(lane: str, lane_rank: int, row: dict[str, Any]) -> None:
        key = _key(row)
        if not all(key) or key in seen or len(output) >= limit:
            return
        seen.add(key)
        output.append(
            {
                **deepcopy(row),
                "candidate_lane": lane,
                "lane_rank": lane_rank,
                "decision_applied": False,
            }
        )

    for lane, rows, quota in lanes:
        for lane_rank, row in enumerate(rows[: max(0, quota)], start=1):
            add(lane, lane_rank, row)
    for lane, rows, _quota in lanes:
        for lane_rank, row in enumerate(rows, start=1):
            add(lane, lane_rank, row)
            if len(output) >= limit:
                return output
    return output
