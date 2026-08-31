from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable


MAX_ARC_MATERIAL_PICKS = 5
ARC_MATERIAL_MENU_LIMIT = 10


def _clean_material(row: dict[str, Any]) -> dict[str, Any] | None:
    kind = str(row.get("kind") or "").strip().lower()
    material_id = str(row.get("id") or "").strip()
    if kind not in {"narrative", "event", "scene", "diary"} or not material_id:
        return None
    return {
        "kind": kind,
        "id": material_id,
        "title": str(row.get("title") or material_id).strip() or material_id,
        "date": str(row.get("date") or "").strip()[:10],
    }


def build_arc_materials(
    narrative: dict[str, Any] | None,
    materials: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create stable positions for one body-free Arc materials snapshot."""

    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in materials:
        cleaned = _clean_material(raw) if isinstance(raw, dict) else None
        if cleaned and cleaned["kind"] != "narrative":
            rows.setdefault((cleaned["kind"], cleaned["id"]), cleaned)
    ordered = sorted(
        rows.values(),
        key=lambda row: (
            not bool(row["date"]),
            row["date"],
            row["kind"],
            row["id"],
        ),
    )

    result: list[dict[str, Any]] = []
    cleaned_narrative = (
        _clean_material(narrative) if isinstance(narrative, dict) else None
    )
    if cleaned_narrative and cleaned_narrative["kind"] == "narrative":
        result.append({"index": 0, **cleaned_narrative})
    result.extend(
        {"index": index, **row}
        for index, row in enumerate(ordered, start=1)
    )
    return result


def displayed_arc_materials(
    materials: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in materials if isinstance(row, dict)]
    if len(rows) <= ARC_MATERIAL_MENU_LIMIT:
        return rows
    narrative = [row for row in rows if int(row.get("index", -1)) == 0]
    members = [row for row in rows if int(row.get("index", -1)) != 0]
    if narrative:
        visible = [*narrative[:1], *members[:4], *members[-5:]]
    else:
        visible = [*members[:5], *members[-5:]]
    by_index = {int(row["index"]): row for row in visible}
    return [by_index[index] for index in sorted(by_index)]


def arc_materials_fingerprint(materials: Iterable[dict[str, Any]]) -> str:
    payload = [
        {
            key: row.get(key)
            for key in ("index", "kind", "id", "title", "date")
        }
        for row in materials
        if isinstance(row, dict)
    ]
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]


def normalize_arc_material_picks(picks: Iterable[Any]) -> list[int]:
    raw = list(picks or [])
    if not raw:
        raise ValueError("at_least_one_pick_required")
    if len(raw) > MAX_ARC_MATERIAL_PICKS:
        raise ValueError("too_many_picks")
    normalized: list[int] = []
    for value in raw:
        if isinstance(value, bool):
            raise ValueError("pick_must_be_non_negative_integer")
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("pick_must_be_non_negative_integer") from exc
        if number < 0 or str(value).strip() != str(number):
            raise ValueError("pick_must_be_non_negative_integer")
        if number not in normalized:
            normalized.append(number)
    return normalized
