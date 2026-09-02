from __future__ import annotations

import json
import re
from typing import Any


def build_new_roll_candidate_prompt(events: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build a bounded external-model task that can only propose groupings."""

    materials = [
        {
            "event_id": str(item.get("event_id") or ""),
            "date": str(item.get("date") or ""),
            "title": str(item.get("title") or "")[:160],
            "summary": str(item.get("summary") or "")[:360],
        }
        for item in events[:80]
        if str(item.get("event_id") or "").strip()
    ]
    return [
        {
            "role": "system",
            "content": (
                "你是叙事卷修订箱的成卷候选侦察器，不是 Writer。只判断哪些尚未归卷的 Event "
                "可能属于同一条持续叙事。不得写叙事正文，不得添加输入中没有的 Event ID。"
                "只有至少 2 个 Event 共享持续对象、因果线或反复推进的问题时才提案；仅仅主题相似不算。"
                "拿不准就不提。一个 Event 在本轮最多属于一个候选。"
            ),
        },
        {
            "role": "user",
            "content": (
                "只返回 JSON：{\"candidates\":[{\"title\":\"暂定卷名\","
                "\"reason\":\"为什么是一条持续叙事\",\"source_event_ids\":[\"event_...\"],"
                "\"confidence\":\"high|medium\",\"latest_date\":\"YYYY-MM-DD\"}]}。"
                "没有可信候选时返回 {\"candidates\":[]}。\n\n"
                f"<unbound_events_json>{json.dumps(materials, ensure_ascii=False)}</unbound_events_json>"
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


def normalize_new_roll_candidates(value: Any, eligible_event_ids: set[str]) -> list[dict[str, Any]]:
    raw = _json_object(value)
    candidates = raw.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("new roll scout candidates must be a list")

    normalized: list[dict[str, Any]] = []
    claimed: set[str] = set()
    for item in candidates[:12]:
        if not isinstance(item, dict):
            continue
        event_ids = list(
            dict.fromkeys(
                str(event_id or "").strip()
                for event_id in item.get("source_event_ids") or []
                if str(event_id or "").strip() in eligible_event_ids
            )
        )
        if len(event_ids) < 2 or claimed.intersection(event_ids):
            continue
        title = str(item.get("title") or "").strip()[:120]
        reason = str(item.get("reason") or "").strip()[:500]
        confidence = str(item.get("confidence") or "").strip().lower()
        if not title or not reason or confidence not in {"high", "medium"}:
            continue
        claimed.update(event_ids)
        normalized.append(
            {
                "title": title,
                "reason": reason,
                "source_event_ids": event_ids,
                "confidence": confidence,
                "latest_date": str(item.get("latest_date") or "").strip()[:10],
            }
        )
    return normalized


async def propose_new_roll_candidates(
    *,
    client: Any,
    model: str,
    events: list[dict[str, Any]],
    completion_options: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if client is None or not str(model or "").strip() or len(events) < 2:
        return []
    response = await client.chat.completions.create(
        model=str(model),
        messages=build_new_roll_candidate_prompt(events),
        **(completion_options or {}),
    )
    content = response.choices[0].message.content if response.choices else ""
    eligible = {str(item.get("event_id") or "") for item in events}
    return normalize_new_roll_candidates(content, eligible)
