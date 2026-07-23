#!/usr/bin/env python3
"""Import legacy Ombre Darkroom JSONL into the unified Diary SQLite backend."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diary_store import DiaryStore


def load_entries(path: Path) -> list[dict]:
    entries: list[dict] = []
    if not path.exists():
        return entries
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            entries.append(value)
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import each legacy Darkroom entry verbatim as entry_type=darkroom."
    )
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--entries-path", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    db_path = Path(args.db_path).resolve()
    entries_path = Path(args.entries_path).resolve()
    entries = load_entries(entries_path)
    summary = {
        "mode": "apply" if args.apply else "dry_run",
        "db_path": str(db_path),
        "entries_path": str(entries_path),
        "source_entries": len(entries),
    }
    if not args.apply:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if not db_path.exists():
        raise SystemExit(f"diary database does not exist: {db_path}")
    backup = db_path.with_name(
        f"{db_path.name}.pre-darkroom-{datetime.now().strftime('%Y%m%d-%H%M%S')}.bak"
    )
    shutil.copy2(db_path, backup)
    store = DiaryStore(db_path=db_path)
    result = store.import_legacy_darkroom(entries)
    print(
        json.dumps(
            {
                **summary,
                **result,
                "backup": str(backup),
                "stats": store.stats(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
