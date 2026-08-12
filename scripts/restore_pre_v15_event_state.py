from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_scope(paths: list[str]) -> tuple[set[str], set[str], set[str]]:
    replace: set[str] = set()
    keep: set[str] = set()
    retire: set[str] = set()
    for path in paths:
        artifact = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        for event in artifact.get("events") or []:
            for link in event.get("old_events") or []:
                replace.add(str(link["item_id"]))
        decisions = artifact.get("migration_decisions") or {}
        keep.update(str(value) for value in decisions.get("keep_old_event_ids") or [])
        retire.update(str(value) for value in decisions.get("retire_old_event_ids") or [])
    if replace.intersection(keep | retire) or keep.intersection(retire):
        raise ValueError("migration scope overlaps")
    return replace, keep, retire


def rows_by_id(conn: sqlite3.Connection, ids: set[str]) -> dict[str, sqlite3.Row]:
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT * FROM fact_events WHERE item_id IN ({placeholders})",
        sorted(ids),
    ).fetchall()
    return {str(row["item_id"]): row for row in rows}


def replacement_ids(
    conn: sqlite3.Connection,
    roots: set[str],
    *,
    created_after: str,
) -> set[str]:
    result: set[str] = set()
    for root in roots:
        rows = conn.execute(
            """
            WITH RECURSIVE family AS (
              SELECT item_id, supersedes_item_id, created_at
              FROM fact_events WHERE supersedes_item_id=?
              UNION ALL
              SELECT child.item_id, child.supersedes_item_id, child.created_at
              FROM fact_events child JOIN family parent
                ON child.supersedes_item_id=parent.item_id
            )
            SELECT item_id FROM family WHERE created_at>=?
            """,
            (root, created_after),
        ).fetchall()
        result.update(str(row[0]) for row in rows)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Restore pre-v15 Event statuses without rolling back later data.")
    parser.add_argument("--before-db", required=True)
    parser.add_argument("--current-db", required=True)
    parser.add_argument("--artifact", action="append", required=True)
    parser.add_argument("--created-after", default="2026-08-11T18:11:33+00:00")
    parser.add_argument("--backup")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    replace, keep, retire = load_scope(args.artifact)
    old_ids = replace | keep | retire
    before = sqlite3.connect(args.before_db)
    before.row_factory = sqlite3.Row
    current = sqlite3.connect(args.current_db)
    current.row_factory = sqlite3.Row
    before_rows = rows_by_id(before, old_ids)
    current_rows = rows_by_id(current, old_ids)
    missing_before = sorted(old_ids.difference(before_rows))
    missing_current = sorted(old_ids.difference(current_rows))
    if missing_before or missing_current:
        raise ValueError(
            f"scope rows missing: before={missing_before[:5]} current={missing_current[:5]}"
        )
    non_active_before = sorted(
        item_id for item_id, row in before_rows.items() if str(row["status"]) != "active"
    )
    if non_active_before:
        raise ValueError(f"pre-v15 scope contains non-active rows: {non_active_before[:5]}")

    replacements = replacement_ids(
        current,
        replace,
        created_after=args.created_after,
    )
    summary: dict[str, Any] = {
        "replace_old_events": len(replace),
        "keep_old_events": len(keep),
        "retire_old_events": len(retire),
        "replacement_events_to_archive": len(replacements),
        "apply": bool(args.apply),
    }
    if not args.apply:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if not args.backup:
        raise ValueError("--backup is required with --apply")

    backup = Path(args.backup)
    if backup.exists():
        raise ValueError("backup target already exists")
    current.close()
    shutil.copy2(args.current_db, backup)
    current = sqlite3.connect(args.current_db, timeout=30)
    current.row_factory = sqlite3.Row
    current.execute("BEGIN IMMEDIATE")
    try:
        for item_id in sorted(replace | keep):
            current.execute(
                "UPDATE fact_events SET status='active', updated_at=datetime('now') WHERE item_id=?",
                (item_id,),
            )
        for item_id in sorted(retire | replacements):
            current.execute(
                "UPDATE fact_events SET status='archived', updated_at=datetime('now') WHERE item_id=?",
                (item_id,),
            )
        current.commit()
    except Exception:
        current.rollback()
        raise

    restored = rows_by_id(current, replace | keep)
    retired = rows_by_id(current, retire | replacements)
    if any(str(row["status"]) != "active" for row in restored.values()):
        raise RuntimeError("not every pre-v15 Event was restored")
    if any(str(row["status"]) != "archived" for row in retired.values()):
        raise RuntimeError("not every replacement or explicit retirement was archived")
    summary["backup"] = str(backup)
    summary["completed_at"] = datetime.now(timezone.utc).isoformat()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
