#!/usr/bin/env python3
"""Verify that RiJi and legacy Darkroom records survived migration exactly."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any


DIARY_FIELDS = (
    "id",
    "date",
    "title",
    "content",
    "author",
    "emotion_tags",
    "created_at",
    "updated_at",
)
COMMENT_FIELDS = ("id", "diary_id", "content", "author", "created_at")


def _rows(db_path: Path, table: str, fields: tuple[str, ...]) -> list[tuple[Any, ...]]:
    with sqlite3.connect(db_path) as conn:
        return conn.execute(
            f"SELECT {', '.join(fields)} FROM {table} ORDER BY id"
        ).fetchall()


def _digest(rows: list[tuple[Any, ...]]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_darkroom(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not path.exists():
        return entries
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid Darkroom JSON at line {line_number}") from exc
        if isinstance(value, dict):
            entries.append(value)
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare the old RiJi DB and Darkroom JSONL with the unified DB."
    )
    parser.add_argument("--source-db", required=True)
    parser.add_argument("--target-db", required=True)
    parser.add_argument("--darkroom-entries", required=True)
    args = parser.parse_args()

    source_db = Path(args.source_db).resolve()
    target_db = Path(args.target_db).resolve()
    darkroom_path = Path(args.darkroom_entries).resolve()
    for path in (source_db, target_db, darkroom_path):
        if not path.exists():
            raise SystemExit(f"required migration source is missing: {path}")

    source_diaries = _rows(source_db, "diaries", DIARY_FIELDS)
    target_diaries = _rows(target_db, "diaries", DIARY_FIELDS)
    source_comments = _rows(source_db, "comments", COMMENT_FIELDS)
    target_comments = _rows(target_db, "comments", COMMENT_FIELDS)
    target_diary_by_id = {int(row[0]): row for row in target_diaries}
    target_comment_by_id = {int(row[0]): row for row in target_comments}

    missing_diaries = [
        int(row[0]) for row in source_diaries if target_diary_by_id.get(int(row[0])) != row
    ]
    missing_comments = [
        int(row[0]) for row in source_comments if target_comment_by_id.get(int(row[0])) != row
    ]

    darkroom_entries = _load_darkroom(darkroom_path)
    with sqlite3.connect(target_db) as conn:
        migrated_darkroom = {
            str(row[0]): (str(row[1]), str(row[2]))
            for row in conn.execute(
                """
                SELECT source_id, content, entry_type
                FROM diaries
                WHERE source_id LIKE 'legacy_darkroom:%'
                """
            ).fetchall()
        }

    darkroom_mismatches: list[str] = []
    valid_darkroom = 0
    for entry in darkroom_entries:
        entry_id = str(entry.get("id") or "").strip()
        content = str(entry.get("note") or "")
        created_at = str(entry.get("created_at") or "").strip()
        if not entry_id or not content.strip() or not created_at:
            continue
        valid_darkroom += 1
        source_id = f"legacy_darkroom:{entry_id}"
        migrated = migrated_darkroom.get(source_id)
        if not migrated or migrated[0] != content or migrated[1] != "darkroom":
            darkroom_mismatches.append(entry_id)

    ok = not missing_diaries and not missing_comments and not darkroom_mismatches
    result = {
        "status": "ok" if ok else "mismatch",
        "source_diaries": len(source_diaries),
        "source_comments": len(source_comments),
        "valid_darkroom_entries": valid_darkroom,
        "matched_darkroom_entries": valid_darkroom - len(darkroom_mismatches),
        "source_diaries_sha256": _digest(source_diaries),
        "source_comments_sha256": _digest(source_comments),
        "missing_or_changed_diary_ids": missing_diaries,
        "missing_or_changed_comment_ids": missing_comments,
        "mismatched_darkroom_entry_ids": darkroom_mismatches,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
