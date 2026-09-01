from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Iterable


_PROGRESS_EVIDENCE = re.compile(
    r"第?\s*\d+\s*(?:话|章|集)|(?:MISSION|EPISODE)\s*\d+|看到|读到|追到|留到明天",
    re.IGNORECASE,
)


def _ref(row: dict[str, Any]) -> str:
    kind = str(row.get("owner_kind") or "").strip().lower()
    owner_id = str(row.get("owner_id") or "").strip()
    return f"{kind}:{owner_id}" if kind and owner_id else ""


def _candidate_text(row: dict[str, Any]) -> str:
    fields = [str(row.get("title") or "").strip()]
    fields.extend(
        str(passage.get("text") or "").strip()
        for passage in row.get("passages") or []
        if isinstance(passage, dict) and str(passage.get("text") or "").strip()
    )
    return "\n".join(field for field in fields if field)


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


def typed_admission_rerank_query(
    query: str,
    entity_scope: dict[str, Any] | None,
) -> str:
    # Entity masking is only a routing aid.  The reranker must see the full
    # evidence question or an entity-detail query such as “巧克蕾是谁” collapses
    # into the meaningless residue “谁”.
    _ = entity_scope
    return str(query or "").strip()


def evaluate_typed_admission_shadow(
    query: str,
    entity_scope: dict[str, Any] | None,
    candidates: Iterable[dict[str, Any]],
    *,
    rerank_scores: dict[str, float] | None = None,
    direct_threshold: float = 0.65,
) -> dict[str, Any]:
    """Classify typed candidates without applying an admission or injection."""

    scope = entity_scope if isinstance(entity_scope, dict) else {}
    operator = str(scope.get("operator") or "none")
    intent = str(scope.get("intent") or "none")
    rows = []
    for candidate in candidates:
        ref = _ref(candidate)
        if not ref:
            continue
        rows.append(
            {
                "ref": ref,
                "owner_kind": str(candidate.get("owner_kind") or ""),
                "owner_id": str(candidate.get("owner_id") or ""),
                "memory_date": str(candidate.get("memory_date") or ""),
                "candidate_score": candidate.get("score"),
                "has_scored_evidence": bool(
                    candidate.get("score") is not None and _candidate_text(candidate)
                ),
                "candidate_text": _candidate_text(candidate),
                "disposition": "reject",
                "reason": "not_evaluated",
                "rerank_score": None,
                "decision_applied": False,
            }
        )

    if operator == "narrative_read":
        mode = "defer_to_narrative"
        for row in rows:
            row.update(disposition="defer", reason=mode)
    elif operator == "exact_evidence":
        mode = "defer_to_exact_evidence"
        for row in rows:
            row.update(disposition="defer", reason=mode)
    elif operator == "timeline":
        mode = "timeline_scope_material"
        for row in rows:
            row.update(
                disposition="scope_material" if row["has_scored_evidence"] else "reject",
                reason=(
                    "timeline_candidate_material"
                    if row["has_scored_evidence"]
                    else "candidate_without_scored_evidence"
                ),
            )
    elif operator == "latest_relevant_member":
        mode = "structured_latest"
        eligible = [row for row in rows if row["has_scored_evidence"]]
        if intent == "progress":
            eligible = [row for row in eligible if _PROGRESS_EVIDENCE.search(row["candidate_text"])]
        selected = max(
            eligible,
            key=lambda row: (
                _timestamp(row["memory_date"]),
                float(row["candidate_score"] or 0.0),
                row["ref"],
            ),
            default=None,
        )
        for row in rows:
            if row is selected:
                row.update(disposition="structured_latest", reason="latest_dated_relevant_member")
            else:
                row.update(disposition="reject", reason="not_latest_relevant_member")
    else:
        mode = "direct_evidence_rerank"
        scores = rerank_scores or {}
        threshold = max(0.0, min(1.0, float(direct_threshold)))
        for row in rows:
            score = scores.get(row["ref"])
            row["rerank_score"] = round(float(score), 4) if score is not None else None
            if not row["has_scored_evidence"]:
                row.update(disposition="reject", reason="candidate_without_scored_evidence")
            elif score is None:
                row.update(disposition="reject", reason="reranker_score_missing")
            elif float(score) >= threshold:
                row.update(disposition="direct_evidence", reason="reranker_direct_evidence")
            else:
                row.update(disposition="reject", reason="reranker_below_direct_threshold")

    public_rows = []
    for row in rows:
        public_rows.append({key: value for key, value in row.items() if key != "candidate_text"})
    selected_refs = [
        row["ref"]
        for row in public_rows
        if row["disposition"] in {"direct_evidence", "structured_latest"}
    ]
    material_refs = [
        row["ref"] for row in public_rows if row["disposition"] == "scope_material"
    ]
    return {
        "status": "ok",
        "mode": mode,
        "operator": operator,
        "intent": intent,
        "rerank_query": typed_admission_rerank_query(query, scope),
        "direct_threshold": round(max(0.0, min(1.0, float(direct_threshold))), 4),
        "selected_refs": selected_refs,
        "material_refs": material_refs,
        "candidates": public_rows,
        "decision_applied": False,
        "live_injection_enabled": False,
    }
