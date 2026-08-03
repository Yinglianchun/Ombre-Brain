#!/usr/bin/env python3
"""Restricted SSH projection for Serein's read-only Hook observation view."""

from __future__ import annotations

import json
import sqlite3


DB_PATH = "/opt/haven_bridge/data/haven.db"
LIMIT = 80


def main() -> None:
    connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT id, session_id, created_at, content, metadata_json
        FROM messages
        WHERE role='user'
        ORDER BY id DESC
        LIMIT 400
        """
    ).fetchall()
    items = []
    for row in rows:
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            metadata = {}
        if "hook_memory_outcome" not in metadata:
            continue
        raw_ids = metadata.get("gateway_memory_injected_ids")
        injected_ids = list(
            dict.fromkeys(
                str(item).strip()
                for item in (raw_ids if isinstance(raw_ids, list) else [])
                if str(item).strip()
            )
        )
        memory_items = []
        raw_items = metadata.get("gateway_memory_items")
        for item in raw_items if isinstance(raw_items, list) else []:
            if not isinstance(item, dict):
                continue
            compact = {}
            for key in ("id", "title", "domain", "date", "moment_id", "source_kind"):
                value = str(item.get(key) or "").strip()
                if value:
                    compact[key] = value[:200]
            score = item.get("score")
            if isinstance(score, (int, float)):
                compact["score"] = float(score)
            if compact:
                memory_items.append(compact)
        items.append(
            {
                "id": int(row["id"]),
                "session_id": int(row["session_id"]) if row["session_id"] is not None else None,
                "created_at": str(row["created_at"] or ""),
                "query": str(row["content"] or "")[:500],
                "hook_memory_outcome": str(metadata.get("hook_memory_outcome") or ""),
                "hook_memory_sources": metadata.get("hook_memory_sources")
                if isinstance(metadata.get("hook_memory_sources"), list)
                else [],
                "gateway_memory_trigger": str(metadata.get("gateway_memory_trigger") or ""),
                "gateway_memory_route": str(metadata.get("gateway_memory_route") or ""),
                "gateway_memory_injected_ids": injected_ids,
                "gateway_memory_items": memory_items,
            }
        )
        if len(items) >= LIMIT:
            break
    connection.close()
    print(json.dumps({"status": "ok", "items": items}, ensure_ascii=False))


if __name__ == "__main__":
    main()
