#!/usr/bin/env python3
"""Plan or atomically build the Scene/Event passage shadow index."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bucket_manager import BucketManager
from embedding_engine import EmbeddingEngine
from fact_events import FactEventStore
from memory_recall.passage_shadow import PassageShadowIndex
from utils import bucket_text_for_embedding, load_config


def active_scene(bucket: dict[str, Any]) -> dict[str, Any] | None:
    metadata = bucket.get("metadata") if isinstance(bucket.get("metadata"), dict) else {}
    authored = (
        str(metadata.get("memory_value_source") or "") == "authored_scene"
        or str(metadata.get("object_kind") or "").strip().lower() == "scene"
    )
    archived = (
        str(metadata.get("type") or "").strip().lower() == "archived"
        or str(metadata.get("scene_status") or "").strip().lower() == "archived"
        or metadata.get("active") is False
    )
    content = bucket_text_for_embedding(bucket)
    owner_id = str(bucket.get("id") or "").strip()
    if not authored or archived or not owner_id or not content.strip():
        return None
    return {
        "id": owner_id,
        "title": str(metadata.get("name") or bucket.get("name") or "").strip(),
        "content": content,
    }


def active_events(store: FactEventStore) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = store.list(
            item_type="event",
            status="active",
            limit=500,
            offset=offset,
        )
        items = list(page.get("items") or [])
        output.extend(items)
        offset += len(items)
        if not items or offset >= int(page.get("count") or 0):
            return output


async def run(args: argparse.Namespace) -> int:
    config = load_config(args.config or None)
    engine = EmbeddingEngine(config)
    index = PassageShadowIndex(config, engine)
    buckets = await BucketManager(config).list_all(include_archive=True)
    scenes = [scene for bucket in buckets if (scene := active_scene(bucket)) is not None]
    events = active_events(FactEventStore(config, create=False))
    plan = await index.sync(
        scenes=scenes,
        events=events,
        dry_run=True,
        refresh_all=args.refresh_all,
    )
    print(json.dumps({"mode": "plan", **plan}, ensure_ascii=False, indent=2))
    if not args.apply:
        return 0
    if int(plan.get("to_embed") or 0) and not engine.enabled:
        print("Embedding engine is disabled; current passage index was not changed.", file=sys.stderr)
        return 2
    result = await index.rebuild_atomic(
        scenes=scenes,
        events=events,
        refresh_all=args.refresh_all,
    )
    print(
        json.dumps(
            {
                "mode": "apply",
                **result,
                "canonical_writes": False,
                "live_injection_enabled": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.get("activated") else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plan or atomically build the Scene/Event passage shadow index."
    )
    parser.add_argument("--config", default="")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Build in a staging SQLite file and atomically activate only on full success.",
    )
    parser.add_argument("--refresh-all", action="store_true")
    raise SystemExit(asyncio.run(run(parser.parse_args())))
