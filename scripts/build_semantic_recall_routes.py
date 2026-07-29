from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embedding_engine import EmbeddingEngine
from memory_recall.semantic_router import build_route_index, load_route_source
from utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the offline semantic recall route vector index."
    )
    parser.add_argument("--config", default="", help="Ombre config YAML path")
    parser.add_argument(
        "--routes",
        default=str(ROOT / "resources" / "semantic_recall_routes.json"),
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate approved route examples without calling the embedding API.",
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> int:
    source_path = Path(args.routes).resolve()
    source = load_route_source(source_path)
    active_routes = [
        route for route in source["routes"] if route.get("enabled", True)
    ]
    example_count = sum(len(route["utterances"]) for route in active_routes)
    template_count = len(source["routes"]) - len(active_routes)
    if args.validate_only:
        print(
            f"valid active_routes={len(active_routes)} templates={template_count} "
            f"examples={example_count} "
            f"dataset_version={source['dataset_version']}"
        )
        return 0

    config = load_config(args.config or None)
    engine = EmbeddingEngine(config)
    if not engine.enabled:
        raise RuntimeError("embedding engine is disabled; configure the API before building")
    output_path = (
        Path(args.output).resolve()
        if args.output
        else Path(config["buckets_dir"]).resolve() / "semantic_recall_routes.v1.json"
    )
    payload = await build_route_index(
        source_path=source_path,
        output_path=output_path,
        embedding_engine=engine,
        concurrency=args.concurrency,
    )
    print(
        f"built routes={len(payload['routes'])} examples={example_count} "
        f"model={payload['embedding']['model']} "
        f"dimension={payload['embedding']['dimension']} output={output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
