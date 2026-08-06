from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bucket_manager import BucketManager
from embedding_engine import EmbeddingEngine
from memory_recall.cue_semantic import (
    CUE_INDEX_REBUILD_CONFIRMATION,
    CueSemanticIndex,
    build_cue_source,
    cue_index_selection_summary,
)
from utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview or atomically build the simulation-only Scene cue embedding index.",
    )
    parser.add_argument("--apply", action="store_true", help="Build and activate a new generation.")
    parser.add_argument("--expected-version", type=int, default=None)
    parser.add_argument("--confirm", default="")
    parser.add_argument("--concurrency", type=int, default=3)
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    config = load_config()
    manager = BucketManager(config)
    buckets = await manager.list_all(include_archive=False)
    engine = EmbeddingEngine(config)
    index = CueSemanticIndex(config, engine)
    current = index.status()
    current_version = index.current_dataset_version()
    source = build_cue_source(buckets, dataset_version=current_version + 1)
    preview = {
        "mode": "apply" if args.apply else "dry_run",
        "current": current,
        "next_dataset_version": current_version + 1,
        "scene_count": len(source["scenes"]),
        "cue_count": sum(len(scene["cues"]) for scene in source["scenes"]),
        "selection": cue_index_selection_summary(buckets),
        "embedding_profile": {
            "model": engine.model,
            "query_instruction": engine.query_instruction,
            "document_instruction": engine.document_instruction,
            "max_chars": engine.max_chars,
        },
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    if not args.apply:
        return 0
    if args.expected_version is None:
        raise SystemExit("--expected-version is required with --apply")
    if args.confirm != CUE_INDEX_REBUILD_CONFIRMATION:
        raise SystemExit(
            f"--confirm {CUE_INDEX_REBUILD_CONFIRMATION} is required with --apply"
        )
    built = await index.rebuild(
        buckets=buckets,
        expected_dataset_version=args.expected_version,
        confirmation=args.confirm,
        concurrency=max(1, min(8, args.concurrency)),
    )
    print(json.dumps(built, ensure_ascii=False, indent=2))
    return 0


raise SystemExit(asyncio.run(main()))
