#!/usr/bin/env python3
"""Backfill whole-Scene embeddings and their owner-only verbatim span index.

Dry-run is the default. ``--apply`` performs remote embedding calls and replaces
only derived embedding rows; it never rewrites Scene files or creates child IDs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bucket_manager import BucketManager
from embedding_engine import EmbeddingEngine
from utils import bucket_text_for_embedding, load_config


def _is_authored_scene(bucket: dict) -> bool:
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    return (
        str(meta.get("memory_value_source") or "") == "authored_scene"
        or str(meta.get("object_kind") or "").strip().lower() == "scene"
    ) and str(meta.get("status") or "active").strip().lower() != "archived"


async def run(args: argparse.Namespace) -> int:
    config = load_config(args.config or None)
    manager = BucketManager(config)
    engine = EmbeddingEngine(config)
    buckets = await manager.list_all(include_archive=False)

    candidates: list[tuple[str, str]] = []
    for bucket in buckets:
        if not _is_authored_scene(bucket):
            continue
        scene_id = str(bucket.get("id") or "").strip()
        content = bucket_text_for_embedding(bucket)
        if scene_id and len(content.strip()) > engine.scene_chunk_chars:
            candidates.append((scene_id, content))
    if args.limit > 0:
        candidates = candidates[: args.limit]

    report = {
        "mode": "apply" if args.apply else "dry_run",
        "scene_chunk_chars": engine.scene_chunk_chars,
        "candidate_count": len(candidates),
        "candidate_scene_ids": [scene_id for scene_id, _ in candidates],
        "updated": [],
        "failed": [],
    }
    if not args.apply:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if not engine.enabled:
        report["error"] = "embedding is disabled or missing an API key"
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2

    for scene_id, content in candidates:
        if await engine.generate_and_store_scene(scene_id, content):
            report["updated"].append(scene_id)
        else:
            report["failed"].append(scene_id)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["failed"] else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--apply", action="store_true")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
