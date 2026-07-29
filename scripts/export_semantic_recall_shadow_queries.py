from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any


DATASET_SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a private, fixed real-query set for semantic recall shadow evaluation."
    )
    parser.add_argument("--database", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--max-query-chars", type=int, default=1000)
    return parser.parse_args()


def _legacy_snapshot(payload_text: str) -> dict[str, Any]:
    try:
        payload = json.loads(payload_text) if payload_text else {}
    except (TypeError, json.JSONDecodeError):
        payload = {}
    payload = payload if isinstance(payload, dict) else {}
    memory_sentinel = payload.get("memory_sentinel_debug")
    memory_sentinel = memory_sentinel if isinstance(memory_sentinel, dict) else {}
    domain_sentinel = payload.get("domain_sentinel_debug")
    domain_sentinel = domain_sentinel if isinstance(domain_sentinel, dict) else {}
    query_planner = payload.get("query_planner_debug")
    query_planner = query_planner if isinstance(query_planner, dict) else {}
    timing = payload.get("prepare_timing_debug")
    timing = timing if isinstance(timing, dict) else {}
    return {
        "available": bool(payload),
        "memory_sentinel_called": bool(memory_sentinel.get("called")),
        "memory_sentinel_route": str(memory_sentinel.get("route") or ""),
        "domain_sentinel_called": bool(domain_sentinel.get("called")),
        "query_planner_triggered": bool(query_planner.get("triggered")),
        "query_planner_skip_reason": str(query_planner.get("skip_reason") or ""),
        "skip_broad_dynamic_recall": (
            bool(timing.get("skip_broad_dynamic_recall"))
            if "skip_broad_dynamic_recall" in timing
            else None
        ),
        "injected_bucket_ids": [
            str(item)
            for item in (payload.get("injected_bucket_ids") or [])
            if str(item or "").strip()
        ],
    }


def export_query_set(
    database_path: Path,
    *,
    limit: int,
    max_query_chars: int,
) -> dict[str, Any]:
    database_path = database_path.resolve()
    if not database_path.is_file():
        raise FileNotFoundError(database_path)
    connection = sqlite3.connect(
        f"file:{database_path.as_posix()}?mode=ro",
        uri=True,
    )
    try:
        rows = connection.execute(
            """
            SELECT
                turns.created_at,
                turns.user_text,
                COALESCE(
                    (
                        SELECT debug.payload_json
                        FROM injection_debug AS debug
                        WHERE debug.session_id = turns.session_id
                          AND debug.round_id = turns.round_id
                        ORDER BY debug.id DESC
                        LIMIT 1
                    ),
                    ''
                )
            FROM conversation_turns AS turns
            WHERE trim(turns.user_text) <> ''
            ORDER BY turns.created_at DESC, turns.id DESC
            LIMIT ?
            """,
            (max(1, int(limit)) * 5,),
        ).fetchall()
    finally:
        connection.close()

    cases: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    for created_at, raw_text, debug_payload in rows:
        text = str(raw_text or "").strip()
        normalized = " ".join(text.split()).casefold()
        if (
            not normalized
            or normalized in seen_queries
            or len(text) > max(1, int(max_query_chars))
        ):
            continue
        seen_queries.add(normalized)
        cases.append(
            {
                "id": hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16],
                "created_at": str(created_at or ""),
                "query": text,
                "legacy": _legacy_snapshot(str(debug_payload or "")),
            }
        )
        if len(cases) >= max(1, int(limit)):
            break

    cases.reverse()
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "gateway_state.conversation_turns",
        "case_count": len(cases),
        "legacy_debug_count": sum(
            1 for case in cases if case["legacy"]["available"]
        ),
        "cases": cases,
    }


def main() -> int:
    args = parse_args()
    payload = export_query_set(
        Path(args.database),
        limit=args.limit,
        max_query_chars=args.max_query_chars,
    )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"exported cases={payload['case_count']} "
        f"legacy_debug={payload['legacy_debug_count']} output={output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
