from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_recall.typed_admission_shadow import evaluate_typed_admission_shadow
from memory_recall.typed_reading_view_shadow import build_typed_reading_view_shadow


def row(ref: str, date: str, text: str, score: float = 0.7) -> dict:
    kind, owner_id = ref.split(":", 1)
    return {
        "owner_kind": kind,
        "owner_id": owner_id,
        "title": text,
        "memory_date": date,
        "score": score,
        "passages": [{"text": text, "score": score}],
    }


def with_arc(item: dict, arc_key: str = "work:spy") -> dict:
    return {
        **item,
        "arc_cards": [
            {
                "arc_key": arc_key,
                "narrative_id": "narrative_spy",
                "title": "间谍过家家",
                "revision": 3,
                "member_count": 9,
                "narrative_available": True,
                "read_hint": "可按需读取",
                "body": "must not leak",
            }
        ],
    }


events = [
    row("event:e139", "2026-08-03", "一起追到第139话"),
    row("event:e140", "2026-08-31", "第140话留到明天"),
]
progress = evaluate_typed_admission_shadow(
    "间谍过家家看到哪了",
    {"intent": "progress", "operator": "latest_relevant_member"},
    events,
)
assert progress["selected_refs"] == ["event:e140"], progress
progress_view = build_typed_reading_view_shadow(
    {"scope_anchor": {"arc_key": "work:spy"}},
    [with_arc(item) for item in events],
    progress,
)
assert progress_view["reading_depth"] == "event", progress_view
assert progress_view["arc_card_attached"] is True, progress_view
assert "body" not in progress_view["arc_cards"][0], progress_view

recent = evaluate_typed_admission_shadow(
    "间谍过家家最近怎么样",
    {"intent": "recent", "operator": "latest_relevant_member"},
    events,
)
assert recent["selected_refs"] == ["event:e140"], recent

member = evaluate_typed_admission_shadow(
    "我们第一次一起看间谍过家家的时候",
    {
        "status": "scoped_recall",
        "intent": "member_search",
        "operator": "member_search",
        "intent_view": "我们第一次一起看 的时候",
    },
    events,
    rerank_scores={"event:e139": 0.02, "event:e140": 0.03},
)
assert member["selected_refs"] == [], member
assert member["rerank_query"] == "我们第一次一起看 的时候", member

detail = evaluate_typed_admission_shadow(
    "具体细节",
    {"intent": "none", "operator": "none"},
    events,
    rerank_scores={"event:e139": 0.99, "event:e140": 0.3},
)
assert detail["selected_refs"] == ["event:e139"], detail
detail_view = build_typed_reading_view_shadow({}, events, detail)
assert detail_view["reading_depth"] == "event", detail_view
assert detail_view["arc_cards"] == [], detail_view

timeline = evaluate_typed_admission_shadow(
    "蓝牙后来怎么发展",
    {"intent": "timeline", "operator": "timeline"},
    events,
)
assert timeline["selected_refs"] == [], timeline
assert timeline["material_refs"] == ["event:e139", "event:e140"], timeline
timeline_view = build_typed_reading_view_shadow(
    {"scope_anchor": {"arc_key": "work:spy"}},
    [with_arc(item) for item in reversed(events)],
    timeline,
)
assert timeline_view["reading_depth"] == "arc_timeline", timeline_view
assert [item["ref"] for item in timeline_view["read_items"]] == [
    "event:e139",
    "event:e140",
], timeline_view

narrative = evaluate_typed_admission_shadow(
    "整体剧情",
    {"intent": "arc_narrative", "operator": "narrative_read"},
    events,
)
assert narrative["mode"] == "defer_to_narrative", narrative
assert narrative["selected_refs"] == [], narrative
assert narrative["decision_applied"] is False, narrative
assert narrative["live_injection_enabled"] is False, narrative
narrative_view = build_typed_reading_view_shadow(
    {"scope_anchor": {"arc_key": "work:spy"}},
    [with_arc(item) for item in events],
    narrative,
)
assert narrative_view["reading_depth"] == "arc_narrative", narrative_view
assert narrative_view["content_included"] is False, narrative_view
assert narrative_view["narrative_body_included"] is False, narrative_view

exact = evaluate_typed_admission_shadow(
    "原话怎么说",
    {"intent": "exact_evidence", "operator": "exact_evidence"},
    [with_arc(item) for item in events],
)
exact_view = build_typed_reading_view_shadow(
    {"scope_anchor": {"arc_key": "work:spy"}},
    [with_arc(item) for item in events],
    exact,
)
assert exact_view["reading_depth"] == "exact_evidence", exact_view
assert exact_view["reason"] == "bridge_raw_source_route_disabled", exact_view
assert exact_view["arc_cards"] == [], exact_view
assert exact_view["raw_source_query_enabled"] is False, exact_view

print("TYPED_ADMISSION_SHADOW_OK")
