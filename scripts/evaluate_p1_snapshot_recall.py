#!/usr/bin/env python3
"""Evaluate P1 recall semantics against a private bucket snapshot.

The report contains ids and structural outcomes, not bucket prose or generated
queries. The snapshot and embedding database are opened locally; no bucket is
edited and no live Gateway state is used.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bucket_manager import BucketManager
from gateway import GatewayService
from gateway_state import GatewayStateStore
from memory_moments import parse_bucket_moments
from memory_layers import LAYER_SOURCE_RECORD, infer_bucket_layer
from scripts.compare_dynamic_alpha_rrf import DisabledReranker, OfflineEmbeddingEngine
from utils import load_config


DEFAULT_API_KEY_ENV = "HANDOFF_SUMMARIZER_API_KEY_2"
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-4B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buckets-dir", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--embedding-api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--embedding-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--annotation-samples", type=int, default=5)
    parser.add_argument("--daily-samples", type=int, default=5)
    return parser.parse_args()


def compact_text(value: Any, limit: int = 120) -> str:
    return " ".join(str(value or "").split())[:limit]


def query_hash(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]


def bucket_id(bucket: dict | None) -> str:
    return str((bucket or {}).get("id") or "")


def content_section_text(bucket: dict, sections: set[str]) -> str:
    for moment in parse_bucket_moments(bucket):
        if moment.get("source") == "content" and moment.get("section") in sections:
            text = compact_text(moment.get("text"))
            if text:
                return text
    return ""


def is_daily_bucket(bucket: dict) -> bool:
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    tags = {str(tag).strip().lower() for tag in meta.get("tags", []) or [] if str(tag).strip()}
    return bool({"daily_impression", "weekly_impression", "relationship_weather"} & tags)


async def select_buckets(
    service: GatewayService,
    query: str,
    all_buckets: list[dict],
    case_id: str,
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    selected, suppressed, debug = await service._select_dynamic_buckets(
        query,
        f"p1-snapshot-{case_id}",
        all_buckets,
        search_query=service._dynamic_recall_search_query(query),
        include_query_planner_debug=True,
        allow_semantic_session_dedupe=False,
        allow_rerank=False,
    )
    return selected, suppressed, debug


def suppressed_reasons(items: list[dict], target_bucket_id: str) -> list[str]:
    reasons = []
    for item in items:
        bucket = item.get("bucket") if isinstance(item.get("bucket"), dict) else {}
        if bucket_id(bucket) != target_bucket_id:
            continue
        reason = str(item.get("admission_reason") or item.get("suppression_reason") or "unknown")
        if reason not in reasons:
            reasons.append(reason)
    return reasons


async def format_selected(
    service: GatewayService,
    selected: list[dict],
    all_buckets: list[dict],
    query: str,
) -> tuple[str, list[str]]:
    recalled: list[dict] = []
    grouped: dict[str, list[dict]] = {}
    sections: list[str] = []
    for bucket in selected:
        current_id = bucket_id(bucket)
        moments = service._direct_moments_for_bucket(bucket, query)
        representative = service._representative_moment(moments)
        if representative:
            grouped[current_id] = service._context_moments_for_bucket(bucket)
            recalled.append(representative)
            sections.append(str(representative.get("section") or ""))
    rendered = await service._format_recalled_moments(
        recalled,
        grouped,
        all_buckets,
        service.recalled_budget,
        query,
        context_mode="",
    )
    return rendered, sections


async def evaluate_facts(service: GatewayService, all_buckets: list[dict]) -> list[dict[str, Any]]:
    rows = []
    for index, fact in enumerate(
        [bucket for bucket in all_buckets if service._is_profile_fact_bucket(bucket)],
        start=1,
    ):
        fact_text = content_section_text(fact, {"fact", "body"})
        if not fact_text:
            continue
        query = f"关于这个偏好或边界，你还记得吗？{fact_text}"
        expected_refs = service._profile_fact_evidence_refs(fact)
        expected_ids = sorted({row["bucket_id"] for row in expected_refs if row.get("bucket_id")})
        selected, _suppressed, debug = await select_buckets(service, query, all_buckets, f"fact-{index}")
        selected_ids = [bucket_id(bucket) for bucket in selected]
        routes = list(debug.get("profile_fact_routes") or [])
        rows.append(
            {
                "case": f"fact-{index}",
                "query_hash": query_hash(query),
                "fact_bucket_id": bucket_id(fact),
                "expected_evidence_ids": expected_ids,
                "selected_bucket_ids": selected_ids,
                "routed_fact_ids": [str(row.get("fact_id") or "") for row in routes],
                "legacy_routes": sum(1 for row in routes if row.get("legacy_evidence")),
                "fact_body_selected": bucket_id(fact) in selected_ids,
                "evidence_selected": any(item in selected_ids for item in expected_ids),
            }
        )
    return rows


async def evaluate_annotations(
    service: GatewayService,
    all_buckets: list[dict],
    limit: int,
) -> list[dict[str, Any]]:
    candidates = []
    for bucket in all_buckets:
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        if (
            meta.get("type") == "feel"
            or infer_bucket_layer(bucket) in {"archive", LAYER_SOURCE_RECORD}
            or service._is_profile_fact_bucket(bucket)
        ):
            continue
        for comment in meta.get("comments", []) if isinstance(meta.get("comments"), list) else []:
            if not isinstance(comment, dict) or not compact_text(comment.get("content")):
                continue
            candidates.append((bucket, comment))
    candidates.sort(
        key=lambda pair: (
            0 if str(pair[1].get("kind") or "") == "year_ring" else 1,
            str(pair[1].get("created") or ""),
            bucket_id(pair[0]),
        )
    )
    rows = []
    for index, (bucket, comment) in enumerate(candidates[: max(0, limit)], start=1):
        query = f"后来或现在再看这件事呢？{compact_text(comment.get('content'))}"
        selected, suppressed, _debug = await select_buckets(service, query, all_buckets, f"annotation-{index}")
        rendered, sections = await format_selected(service, selected, all_buckets, query)
        parent_id = bucket_id(bucket)
        selected_ids = [bucket_id(item) for item in selected]
        rows.append(
            {
                "case": f"annotation-{index}",
                "query_hash": query_hash(query),
                "comment_id": str(comment.get("id") or ""),
                "comment_kind": str(comment.get("kind") or "comment"),
                "parent_bucket_id": parent_id,
                "selected_bucket_ids": selected_ids,
                "representative_sections": sections,
                "parent_selected": parent_id in selected_ids,
                "parent_suppression_reasons": suppressed_reasons(suppressed, parent_id),
                "one_annotation_attached": rendered.count("[year_ring]") == 1,
            }
        )
    return rows


async def evaluate_daily_exclusion(
    service: GatewayService,
    all_buckets: list[dict],
    limit: int,
) -> list[dict[str, Any]]:
    rows = []
    daily = sorted(
        [bucket for bucket in all_buckets if is_daily_bucket(bucket)],
        key=bucket_id,
    )[: max(0, limit)]
    for index, bucket in enumerate(daily, start=1):
        material = content_section_text(bucket, {"body", "moment", "reflection"})
        if not material:
            continue
        query = f"你还记得这段日印象吗？{material}"
        selected, _suppressed, _debug = await select_buckets(service, query, all_buckets, f"daily-{index}")
        selected_ids = [bucket_id(item) for item in selected]
        rows.append(
            {
                "case": f"daily-{index}",
                "query_hash": query_hash(query),
                "daily_bucket_id": bucket_id(bucket),
                "selected_bucket_ids": selected_ids,
                "daily_bucket_excluded": bucket_id(bucket) not in selected_ids,
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return {
        "cases": len(rows),
        "passed": sum(1 for row in rows if row.get(key) is True),
        "failed": sum(1 for row in rows if row.get(key) is not True),
    }


async def main() -> int:
    args = parse_args()
    buckets_dir = Path(args.buckets_dir).expanduser().resolve()
    embeddings_db = buckets_dir / "embeddings.db"
    api_key = os.environ.get(args.embedding_api_key_env, "")
    if not buckets_dir.is_dir() or not embeddings_db.is_file() or not api_key:
        print("Missing buckets directory, embeddings.db, or embedding API key.", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="ombre-p1-recall-") as temp_dir:
        config = deepcopy(load_config())
        config["buckets_dir"] = str(buckets_dir)
        config["state_dir"] = str(Path(temp_dir) / "state")
        config["word_map"] = {**dict(config.get("word_map") or {}), "enabled": False}
        config["reranker"] = {**dict(config.get("reranker") or {}), "enabled": False}
        config["gateway"] = {
            **dict(config.get("gateway") or {}),
            "retrieval_mode": "bucket",
            "word_map_hint_enabled": False,
        }
        embedding = dict(config.get("embedding") or {})
        embedding.update(
            {
                "enabled": True,
                "api_key": api_key,
                "base_url": args.embedding_base_url,
                "model": args.embedding_model,
            }
        )
        config["embedding"] = embedding
        manager = BucketManager(config)
        service = GatewayService(
            config,
            bucket_mgr=manager,
            embedding_engine=OfflineEmbeddingEngine(
                db_path=embeddings_db,
                api_key=api_key,
                base_url=args.embedding_base_url,
                model=args.embedding_model,
                max_chars=int(embedding.get("max_chars") or 6000),
                query_instruction=str(embedding.get("query_instruction") or ""),
            ),
            reranker_engine=DisabledReranker(),
            state_store=GatewayStateStore(str(Path(temp_dir) / "state" / "gateway_state.db")),
        )
        all_buckets = await manager.list_all(include_archive=False)
        fact_rows = await evaluate_facts(service, all_buckets)
        annotation_rows = await evaluate_annotations(service, all_buckets, args.annotation_samples)
        daily_rows = await evaluate_daily_exclusion(service, all_buckets, args.daily_samples)

    payload = {
        "mode": "private_snapshot_read_only",
        "buckets_dir": str(buckets_dir),
        "retrieval_mode": "bucket",
        "query_planner": False,
        "reranker": False,
        "word_map": False,
        "summary": {
            "fact_evidence_routing": summarize(fact_rows, "evidence_selected"),
            "fact_body_hidden": summarize(
                [{**row, "passed": not row["fact_body_selected"]} for row in fact_rows],
                "passed",
            ),
            "annotation_parent_routing": summarize(annotation_rows, "parent_selected"),
            "annotation_single_attachment": summarize(annotation_rows, "one_annotation_attached"),
            "daily_exclusion": summarize(daily_rows, "daily_bucket_excluded"),
        },
        "fact_cases": fact_rows,
        "annotation_cases": annotation_rows,
        "daily_cases": daily_rows,
    }
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    failures = sum(group["failed"] for group in payload["summary"].values())
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
