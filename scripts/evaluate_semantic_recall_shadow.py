from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embedding_engine import EmbeddingEngine
from memory_recall.semantic_router import SemanticRecallRouter
from utils import load_config


REPORT_SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the semantic recall router against a private real-query set."
    )
    parser.add_argument("--config", default="")
    parser.add_argument("--queries", required=True)
    parser.add_argument("--routes", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--aggregation-top-k", type=int, default=0)
    parser.add_argument("--min-score", type=float, default=-1.0)
    parser.add_argument("--min-margin", type=float, default=-1.0)
    return parser.parse_args()


async def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    query_payload = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    cases = query_payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("shadow_query_cases_missing")

    config = load_config(args.config or None)
    router_config = config.setdefault("gateway", {}).setdefault(
        "semantic_recall_router",
        {},
    )
    router_config.update(
        shadow_enabled=True,
        routes_path=str(Path(args.routes).resolve()),
        index_path=str(Path(args.index).resolve()),
        publish_dir=str(
            Path(args.output).resolve().parent
            / f".{Path(args.output).name}.publish-disabled"
        ),
    )
    if args.aggregation_top_k > 0:
        router_config["aggregation_top_k"] = args.aggregation_top_k
    if args.min_score >= 0:
        router_config["min_score"] = args.min_score
    if args.min_margin >= 0:
        router_config["min_margin"] = args.min_margin
    engine = EmbeddingEngine(config)
    if not engine.enabled:
        raise RuntimeError("embedding engine is disabled; configure the API before evaluation")
    router = SemanticRecallRouter(config, engine)
    semaphore = asyncio.Semaphore(max(1, min(8, int(args.concurrency or 1))))

    async def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
        query = str(case.get("query") or "")
        async with semaphore:
            debug = await router.route(query)
        legacy = case.get("legacy")
        legacy = legacy if isinstance(legacy, dict) else {}
        legacy_skip = legacy.get("skip_broad_dynamic_recall")
        semantic_action = str(debug.get("recommended_action") or "recall")
        expected_action = str(case.get("expected_action") or "").strip().lower()
        expected_available = expected_action in {"skip", "recall"}
        comparison_available = isinstance(legacy_skip, bool)
        return {
            "id": str(case.get("id") or ""),
            "created_at": str(case.get("created_at") or ""),
            "query": query,
            "semantic": {
                "action": semantic_action,
                "route": str(debug.get("route") or ""),
                "confidence": debug.get("confidence"),
                "margin": debug.get("margin"),
                "reason": str(debug.get("reason") or ""),
                "errors": list(debug.get("errors") or []),
                "scores": list(debug.get("scores") or []),
            },
            "legacy": legacy,
            "expected": {
                "available": expected_available,
                "action": expected_action if expected_available else None,
                "agrees": (
                    semantic_action == expected_action
                    if expected_available
                    else None
                ),
            },
            "comparison": {
                "available": comparison_available,
                "agrees": (
                    semantic_action == ("skip" if legacy_skip else "recall")
                    if comparison_available
                    else None
                ),
            },
        }

    results = await asyncio.gather(
        *(evaluate_case(case) for case in cases if isinstance(case, dict))
    )
    comparable = [row for row in results if row["comparison"]["available"]]
    disagreements = [
        row for row in comparable if row["comparison"]["agrees"] is False
    ]
    expected_rows = [row for row in results if row["expected"]["available"]]
    expected_failures = [
        row for row in expected_rows if row["expected"]["agrees"] is False
    ]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "query_dataset_schema_version": query_payload.get("schema_version"),
        "summary": {
            "cases": len(results),
            "semantic_skip": sum(
                1 for row in results if row["semantic"]["action"] == "skip"
            ),
            "semantic_recall": sum(
                1 for row in results if row["semantic"]["action"] == "recall"
            ),
            "errors": sum(1 for row in results if row["semantic"]["errors"]),
            "expected_comparable": len(expected_rows),
            "expected_passes": len(expected_rows) - len(expected_failures),
            "expected_failures": len(expected_failures),
            "legacy_comparable": len(comparable),
            "legacy_agreements": len(comparable) - len(disagreements),
            "legacy_disagreements": len(disagreements),
            "policy": {
                "aggregation_top_k": router.aggregation_top_k,
                "min_score": router.min_score,
                "min_margin": router.min_margin,
            },
        },
        "results": results,
    }


async def run(args: argparse.Namespace) -> int:
    report = await evaluate(args)
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False))
    print(f"output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
