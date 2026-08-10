from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embedding_engine import EmbeddingEngine
from fact_events import FactEventStore
from memory_recall.fact_event_semantic import FactEventSemanticIndex
from utils import load_config


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--refresh-all", action="store_true")
    args = parser.parse_args()
    config = load_config()
    engine = EmbeddingEngine(config)
    index = FactEventSemanticIndex(config, engine)
    result = await index.sync(
        FactEventStore(config, create=False),
        dry_run=args.dry_run,
        refresh_all=args.refresh_all,
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
