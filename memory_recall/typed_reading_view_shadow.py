from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable


_ARC_CARD_FIELDS = (
    "arc_key",
    "narrative_id",
    "title",
    "publication_status",
    "revision",
    "member_count",
    "narrative_available",
    "latest_member_date",
    "read_hint",
)


def _ref(row: dict[str, Any]) -> str:
    kind = str(row.get("owner_kind") or "").strip().lower()
    owner_id = str(row.get("owner_id") or "").strip()
    return f"{kind}:{owner_id}" if kind and owner_id else ""


def _timestamp(value: Any) -> float:
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


def _safe_card(card: dict[str, Any]) -> dict[str, Any]:
    return {
        key: card.get(key)
        for key in _ARC_CARD_FIELDS
        if card.get(key) not in (None, "")
    }


def _cards_for_rows(
    rows: Iterable[dict[str, Any]],
    *,
    scope_arc_key: str = "",
) -> list[dict[str, Any]]:
    cards: dict[str, dict[str, Any]] = {}
    for row in rows:
        for raw in row.get("arc_cards") or []:
            if not isinstance(raw, dict):
                continue
            arc_key = str(raw.get("arc_key") or "").strip()
            if not arc_key or (scope_arc_key and arc_key != scope_arc_key):
                continue
            cards.setdefault(arc_key, _safe_card(raw))
    return [cards[key] for key in sorted(cards)[:3]]


def build_typed_reading_view_shadow(
    entity_scope: dict[str, Any] | None,
    candidates: Iterable[dict[str, Any]],
    admission: dict[str, Any],
    *,
    timeline_limit: int = 6,
) -> dict[str, Any]:
    """Build a body-free reading receipt without applying a read or injection."""

    scope = entity_scope if isinstance(entity_scope, dict) else {}
    rows = [dict(row) for row in candidates if isinstance(row, dict) and _ref(row)]
    by_ref = {_ref(row): row for row in rows}
    mode = str(admission.get("mode") or "")
    selected_refs = [str(ref) for ref in admission.get("selected_refs") or []]
    material_refs = [str(ref) for ref in admission.get("material_refs") or []]
    scope_arc_key = str((scope.get("scope_anchor") or {}).get("arc_key") or "")

    reading_depth = "none"
    read_items: list[dict[str, Any]] = []
    arc_cards: list[dict[str, Any]] = []
    reason = "no_admitted_read"

    if mode == "timeline_scope_material":
        material_rows = [by_ref[ref] for ref in material_refs if ref in by_ref]
        material_rows.sort(
            key=lambda row: (_timestamp(row.get("memory_date")), _ref(row))
        )
        material_rows = material_rows[-max(1, min(int(timeline_limit or 6), 12)) :]
        read_items = [
            {
                "ref": _ref(row),
                "reading_depth": "event"
                if str(row.get("owner_kind") or "").lower() == "event"
                else "scene",
                "memory_date": str(row.get("memory_date") or ""),
            }
            for row in material_rows
        ]
        arc_cards = _cards_for_rows(material_rows, scope_arc_key=scope_arc_key)
        if scope_arc_key and read_items and arc_cards:
            reading_depth = "arc_timeline"
            reason = "bounded_arc_members_in_chronological_order"
        else:
            read_items = []
            arc_cards = []
            reason = "timeline_requires_confirmed_arc_card"
    elif mode == "defer_to_narrative":
        arc_cards = _cards_for_rows(rows, scope_arc_key=scope_arc_key)
        available = [card for card in arc_cards if card.get("narrative_available")]
        if scope_arc_key and available:
            reading_depth = "arc_narrative"
            reason = "narrative_available_on_demand"
            read_items = [
                {
                    "ref": f"narrative:{card['narrative_id']}",
                    "reading_depth": "arc_narrative",
                }
                for card in available
                if str(card.get("narrative_id") or "")
            ][:1]
        elif scope_arc_key and arc_cards:
            reading_depth = "arc_index"
            reason = "arc_found_without_readable_narrative"
        else:
            arc_cards = []
            reason = "narrative_requires_confirmed_arc_card"
    elif mode == "defer_to_exact_evidence":
        reading_depth = "exact_evidence"
        reason = "bridge_raw_source_route_disabled"
    elif selected_refs:
        selected_rows = [by_ref[ref] for ref in selected_refs if ref in by_ref]
        read_items = [
            {
                "ref": _ref(row),
                "reading_depth": "event"
                if str(row.get("owner_kind") or "").lower() == "event"
                else "scene",
                "memory_date": str(row.get("memory_date") or ""),
            }
            for row in selected_rows
        ]
        if read_items:
            reading_depth = str(read_items[0]["reading_depth"])
        reason = "admitted_owner_read" if read_items else reason
        arc_cards = _cards_for_rows(selected_rows, scope_arc_key=scope_arc_key)

    return {
        "status": "ok",
        "reading_depth": reading_depth,
        "reason": reason,
        "read_items": read_items,
        "arc_cards": arc_cards,
        "arc_card_attached": bool(arc_cards),
        "content_included": False,
        "narrative_body_included": False,
        "raw_source_query_enabled": False,
        "read_applied": False,
        "live_injection_enabled": False,
    }
