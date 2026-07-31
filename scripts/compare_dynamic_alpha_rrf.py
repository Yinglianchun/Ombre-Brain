from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bucket_manager import BucketManager
from embedding_engine import EmbeddingEngine
from gateway import GatewayService
from gateway_state import GatewayStateStore
from utils import load_config


DEFAULT_BUCKETS_DIR = ROOT / "tmp" / "p0-local-snapshot-20260628-1528" / "buckets"
DEFAULT_EMBEDDING_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"
DEFAULT_API_KEY_ENV = "HANDOFF_SUMMARIZER_API_KEY_2"
DEFAULT_EVAL_CASES: list[dict[str, Any]] = [
    {
        "query": "对未来的承诺和五十年后有关吗",
        "expected_bucket_ids": [
            "26a24ca65373",
            "b4e869f1-d4df-4750-b876-bd62cb988173",
            "016ba4632af0",
        ],
        "note": "future-line bucket haven_v021_five_decades_later is intentionally not expected",
    },
    {
        "query": "五十年后那扇门是什么约定",
        "expected_bucket_ids": [
            "016ba4632af0",
            "b4e869f1-d4df-4750-b876-bd62cb988173",
        ],
        "note": "future-line bucket haven_v021_five_decades_later is intentionally not expected",
    },
    {"query": "最柔软的身体是什么承诺", "expected_bucket_ids": ["26a24ca65373"]},
    {"query": "献给你的xx", "expected_bucket_ids": ["4b86849e6209"]},
    {
        "query": "回复确实变慢了！影分身说hook已经接上了但无可靠命中，等下我换个问法",
        "expected_bucket_ids": ["a46e96fcbaec"],
    },
    {"query": "窗口切换时我担心你忘记什么", "expected_bucket_ids": ["a46e96fcbaec"]},
    {"query": "Haven 消息分流 bug 是怎么回事", "expected_bucket_ids": ["bd11ff60de92"]},
    {"query": "小机数据库 v2.0 是什么暗号", "expected_bucket_ids": ["7e8e750da16b"]},
    {"query": "答辩奖励和小机数据库有关吗", "expected_bucket_ids": ["881b7477606e", "7e8e750da16b"]},
    {"query": "小雨的熬夜习惯和健康管理", "expected_bucket_ids": ["3f441b5f73b4"]},
    {"query": "那件事后来怎么样了", "expected_bucket_ids": []},
]


class OfflineEmbeddingEngine:
    def __init__(
        self,
        *,
        db_path: Path,
        api_key: str,
        base_url: str,
        model: str,
        max_chars: int,
        query_instruction: str,
    ) -> None:
        self.db_path = Path(db_path)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_chars = max_chars
        self.query_instruction = query_instruction.strip()
        self.enabled = bool(api_key and self.db_path.exists())
        self.client = (
            AsyncOpenAI(api_key=api_key, base_url=self.base_url, timeout=45.0)
            if self.enabled
            else None
        )

    async def search_similar(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if not self.enabled or not self.client:
            return []
        query_embedding = await self._generate_query_embedding(query)
        if not query_embedding:
            return []
        rows = self._read_embedding_rows()
        results: list[tuple[str, float]] = []
        for bucket_id, embedding, model, dimension in rows:
            if model != self.model or int(dimension or 0) != len(embedding):
                continue
            score = EmbeddingEngine._cosine_similarity(query_embedding, embedding)
            results.append((bucket_id, score))
        results.sort(key=lambda item: item[1], reverse=True)
        return results[:top_k]

    async def _generate_query_embedding(self, query: str) -> list[float]:
        text = str(query or "")
        if self.query_instruction:
            text = f"Instruct: {self.query_instruction}\nQuery: {text}"
        response = await self.client.embeddings.create(
            model=self.model,
            input=text[: self.max_chars],
        )
        if not response.data:
            return []
        return list(response.data[0].embedding or [])

    def _read_embedding_rows(self) -> list[tuple[str, list[float], str, int]]:
        uri = self.db_path.resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            rows = conn.execute(
                "SELECT bucket_id, embedding, model, dimension FROM embeddings"
            ).fetchall()
        finally:
            conn.close()
        output: list[tuple[str, list[float], str, int]] = []
        for bucket_id, emb_json, model, dimension in rows:
            try:
                embedding = json.loads(emb_json)
            except json.JSONDecodeError:
                continue
            if isinstance(embedding, list):
                output.append((str(bucket_id), embedding, str(model or ""), int(dimension or 0)))
        return output


class DisabledReranker:
    enabled = False

    async def rerank(self, query: str, documents: list[str], top_n: int | None = None):
        return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline A/B comparison for Dynamic-alpha candidate ranking versus plain RRF.",
    )
    parser.add_argument(
        "--buckets-dir",
        default=str(DEFAULT_BUCKETS_DIR),
        help="Snapshot buckets directory. Default: tmp/p0-local-snapshot-20260628-1528/buckets",
    )
    parser.add_argument("--query", action="append", default=[], help="Query to evaluate. Repeatable.")
    parser.add_argument("--queries-file", default="", help="UTF-8 file with one query per line.")
    parser.add_argument("--cases-file", default="", help="JSONL file with query and expected_bucket_ids.")
    parser.add_argument(
        "--no-default-cases",
        action="store_true",
        help="Do not fall back to the built-in Xiaoyu recall evaluation set when no cases are supplied.",
    )
    parser.add_argument(
        "--write-default-cases",
        default="",
        help="Write the built-in evaluation cases to this JSONL path and exit.",
    )
    parser.add_argument("--output", default="", help="Optional JSONL output path.")
    parser.add_argument("--top", type=int, default=8, help="Rows to print per ranking.")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF k constant.")
    parser.add_argument("--session-id", default="offline-rrf-ab", help="Session id for cooldown reads.")
    parser.add_argument("--embedding-api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--embedding-base-url", default=DEFAULT_EMBEDDING_BASE_URL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--semantic-top-k", type=int, default=24)
    parser.add_argument("--disable-semantic", action="store_true")
    parser.add_argument(
        "--no-diffusion-debug",
        action="store_true",
        help="Skip moment diffusion debug collection.",
    )
    parser.add_argument(
        "--entity-edges-path",
        default="",
        help="Optional entity_edges.jsonl to copy into the temp state dir. Defaults to <buckets-dir>/../state/entity_edges.jsonl when present.",
    )
    parser.add_argument(
        "--no-entity-edges",
        action="store_true",
        help="Do not copy snapshot entity_edges.jsonl into the temp state dir.",
    )
    return parser.parse_args()


def load_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = [
        {"query": str(item).strip(), "expected_bucket_ids": []}
        for item in args.query
        if str(item).strip()
    ]
    if args.queries_file:
        path = Path(args.queries_file)
        lines = path.read_text(encoding="utf-8").splitlines()
        cases.extend(
            {"query": line.strip(), "expected_bucket_ids": []}
            for line in lines
            if line.strip() and not line.lstrip().startswith("#")
        )
    if args.cases_file:
        path = Path(args.cases_file)
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            query = str(row.get("query") or "").strip()
            if not query:
                continue
            expected = row.get("expected_bucket_ids") or row.get("expected_ids") or []
            if isinstance(expected, str):
                expected_ids = [expected]
            else:
                expected_ids = [str(item) for item in expected if str(item or "").strip()]
            case = {"query": query, "expected_bucket_ids": expected_ids}
            if row.get("note"):
                case["note"] = str(row.get("note") or "")
            cases.append(case)
    if not cases and not args.no_default_cases:
        cases.extend(deepcopy(DEFAULT_EVAL_CASES))
    deduped: dict[str, dict[str, Any]] = {}
    for case in cases:
        query = str(case.get("query") or "").strip()
        if query and query not in deduped:
            deduped[query] = case
    return list(deduped.values())


def build_config(args: argparse.Namespace, state_dir: Path, api_key: str) -> dict[str, Any]:
    config = deepcopy(load_config())
    config["buckets_dir"] = str(Path(args.buckets_dir).resolve())
    config["state_dir"] = str(state_dir)
    embedding = dict(config.get("embedding") or {})
    embedding.update(
        {
            "enabled": bool(api_key and not args.disable_semantic),
            "api_key": api_key,
            "base_url": args.embedding_base_url,
            "model": args.embedding_model,
            "max_chars": int(embedding.get("max_chars") or 6000),
            "query_instruction": str(
                embedding.get("query_instruction")
                or "Given a memory search query, retrieve relevant long-term memory passages."
            ),
        }
    )
    config["embedding"] = embedding
    gateway = dict(config.get("gateway") or {})
    gateway.update(
        {
            "recall_fusion_mode": "dynamic",
            "semantic_candidate_top_k": args.semantic_top_k,
        }
    )
    config["gateway"] = gateway
    return config


def resolve_entity_edges_path(args: argparse.Namespace, buckets_dir: Path) -> Path | None:
    if args.no_entity_edges:
        return None
    if args.entity_edges_path:
        path = Path(args.entity_edges_path)
        return path if path.exists() else None
    default_path = buckets_dir.resolve().parent / "state" / "entity_edges.jsonl"
    return default_path if default_path.exists() else None


def copy_entity_edges_to_state(source_path: Path | None, state_dir: Path) -> str:
    if not source_path:
        return ""
    target_path = state_dir / "entity_edges.jsonl"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    return str(source_path)


def item_bucket_id(item: dict[str, Any]) -> str:
    bucket = item.get("bucket") if isinstance(item.get("bucket"), dict) else {}
    return str(bucket.get("id") or "")


def item_bucket_name(item: dict[str, Any]) -> str:
    bucket = item.get("bucket") if isinstance(item.get("bucket"), dict) else {}
    meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
    return str(meta.get("name") or bucket.get("name") or bucket.get("id") or "")


def source_score_maps(items: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    maps: dict[str, dict[str, float]] = {
        "semantic": {},
        "keyword": {},
        "exact_anchor": {},
        "planner_lexical": {},
        "word_map": {},
        "entity_edge": {},
    }
    for item in items:
        bucket_id = item_bucket_id(item)
        if not bucket_id:
            continue
        for source, key in (
            ("semantic", "semantic_score"),
            ("keyword", "keyword_score"),
            ("exact_anchor", "exact_anchor_score"),
            ("word_map", "word_map_score"),
            ("entity_edge", "entity_edge_score"),
        ):
            value = float(item.get(key) or 0.0)
            if value > 0:
                maps[source][bucket_id] = max(maps[source].get(bucket_id, 0.0), value)
        if item.get("planner_lexical_match"):
            maps["planner_lexical"][bucket_id] = 1.0
    return {source: scores for source, scores in maps.items() if scores}


def reciprocal_rank_fusion(
    score_maps: dict[str, dict[str, float]],
    *,
    k: int,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for source, scores in score_maps.items():
        ranked = sorted(scores.items(), key=lambda item: (-float(item[1]), item[0]))
        for rank, (bucket_id, score) in enumerate(ranked, start=1):
            row = rows.setdefault(bucket_id, {"rrf_score": 0.0, "rrf_sources": []})
            contribution = 1.0 / (k + rank)
            row["rrf_score"] += contribution
            row["rrf_sources"].append(
                {
                    "source": source,
                    "rank": rank,
                    "score": round(float(score), 4),
                    "contribution": round(contribution, 6),
                }
            )
    for row in rows.values():
        row["rrf_score"] = round(float(row["rrf_score"]), 6)
    return rows


def row_for_item(
    service: GatewayService,
    item: dict[str, Any],
    *,
    query: str,
    current_rank: int,
    rrf_rank: int,
    rrf_row: dict[str, Any],
    admitted: bool,
) -> dict[str, Any]:
    recall_debug = service._format_suppressed_bucket_debug(
        item,
        query=query,
        status="admitted" if admitted else "suppressed",
    )
    return {
        "bucket_id": item_bucket_id(item),
        "bucket_name": item_bucket_name(item),
        "admitted": admitted,
        "current_rank": current_rank,
        "rrf_rank": rrf_rank,
        "rank_delta": rrf_rank - current_rank,
        "score": round(float(item.get("score") or 0.0), 4),
        "fusion_mode": str(item.get("fusion_mode") or ""),
        "fusion_score": round(float(item.get("fusion_score") or 0.0), 4),
        "semantic_score": round(float(item.get("semantic_score") or 0.0), 4),
        "keyword_score": round(float(item.get("keyword_score") or 0.0), 4),
        "exact_anchor_score": round(float(item.get("exact_anchor_score") or 0.0), 4),
        "word_map_score": round(float(item.get("word_map_score") or 0.0), 4),
        "entity_edge_score": round(float(item.get("entity_edge_score") or 0.0), 4),
        "exact_anchor_match": bool(item.get("exact_anchor_match")),
        "planner_lexical_match": bool(item.get("planner_lexical_match")),
        "rare_name_match": bool(item.get("rare_name_match")),
        "entity_edge_match": bool(item.get("entity_edge_match")),
        "admission_reason": str(item.get("admission_reason") or ""),
        "rrf_score": float(rrf_row.get("rrf_score") or 0.0),
        "rrf_sources": list(rrf_row.get("rrf_sources") or []),
        "recall_why": compact_recall_why(recall_debug.get("recall_why") or {}),
    }


def compact_recall_why(recall_why: dict[str, Any]) -> dict[str, Any]:
    sources = []
    for source in recall_why.get("sources") or []:
        if not isinstance(source, dict):
            continue
        sources.append(
            {
                key: source[key]
                for key in (
                    "source",
                    "score",
                    "terms",
                    "matched_terms",
                    "relation",
                    "object",
                    "peer_bucket_id",
                    "edge_type",
                    "focused",
                    "why",
                    "confidence",
                )
                if key in source and source[key] not in (None, "", [], {})
            }
        )
    admission = recall_why.get("admission") if isinstance(recall_why.get("admission"), dict) else {}
    score = recall_why.get("score") if isinstance(recall_why.get("score"), dict) else {}
    return {
        "status": str(recall_why.get("status") or ""),
        "stage": str(recall_why.get("stage") or ""),
        "primary_source": str(recall_why.get("primary_source") or ""),
        "sources": sources,
        "admission_reason": str(admission.get("reason") or ""),
        "score": {
            key: score.get(key)
            for key in (
                "final",
                "semantic",
                "keyword",
                "exact_anchor",
                "word_map",
                "entity_edge",
                "fusion_mode",
                "fusion_score",
                "dynamic_alpha",
                "dynamic_alpha_confidence",
            )
            if key in score
        },
    }


def compact_diffusion_debug(row: dict[str, Any]) -> dict[str, Any]:
    trace = row.get("diffusion_trace") if isinstance(row.get("diffusion_trace"), dict) else {}
    gate = trace.get("gate") if isinstance(trace.get("gate"), dict) else {}
    final = trace.get("final") if isinstance(trace.get("final"), dict) else {}
    seed = trace.get("seed") if isinstance(trace.get("seed"), dict) else {}
    target = trace.get("target") if isinstance(trace.get("target"), dict) else {}
    return {
        "bucket_id": str(row.get("bucket_id") or ""),
        "bucket_name": str(row.get("bucket_name") or ""),
        "moment_id": str(row.get("moment_id") or ""),
        "why": str(row.get("why") or ""),
        "source": str(row.get("source") or ""),
        "confidence": round(float(row.get("confidence") or 0.0), 4),
        "confidence_source": str(row.get("confidence_source") or ""),
        "confidence_defaulted": bool(row.get("confidence_defaulted")),
        "activation": round(float(row.get("activation") or 0.0), 4),
        "topic_evidence_terms": list(row.get("topic_evidence_terms") or []),
        "strong_topic_evidence": bool(row.get("strong_topic_evidence")),
        "injected": bool(row.get("injected")),
        "suppression_reason": str(row.get("suppression_reason") or ""),
        "recall_why": compact_recall_why(row.get("recall_why") or {}),
        "diffusion_trace": {
            "seed_bucket_id": str(seed.get("bucket_id") or ""),
            "seed_bucket_name": str(seed.get("bucket_name") or ""),
            "target_bucket_id": str(target.get("bucket_id") or row.get("bucket_id") or ""),
            "target_bucket_name": str(target.get("bucket_name") or row.get("bucket_name") or ""),
            "path_trace": str(trace.get("path_trace") or ""),
            "path_len": int(trace.get("path_len") or 0),
            "path_step_count": int(trace.get("path_step_count") or 0),
            "gate_allowed": bool(gate.get("allowed")),
            "gate_reason": str(gate.get("reason") or ""),
            "final_status": str(final.get("status") or ""),
            "final_reason": str(final.get("suppression_reason") or ""),
        },
    }


async def evaluate_diffusion_debug(
    service: GatewayService,
    case: dict[str, Any],
    args: argparse.Namespace,
    buckets: list[dict[str, Any]],
) -> dict[str, Any]:
    if args.no_diffusion_debug:
        return {}
    query = str(case.get("query") or "").strip()
    all_moments, grouped_moments, moment_edges = service._refresh_moment_graph(buckets)
    recalled_moments, moment_candidates, suppressed_moments, suppressed_buckets, planner_debug = await service._select_dynamic_moments(
        query,
        args.session_id,
        buckets,
        grouped_moments,
        all_moments=all_moments,
        search_query=service._dynamic_recall_search_query(query),
        include_query_planner_debug=True,
    )
    related_memory, diffused_debug_rows = service._build_moment_diffused_memory_with_debug(
        recalled_moments,
        moment_candidates,
        all_moments,
        moment_edges,
        query,
        session_id=args.session_id,
    )
    return {
        "direct_moment_bucket_ids": [
            str(moment.get("bucket_id") or "")
            for moment in recalled_moments
            if moment.get("bucket_id")
        ],
        "direct_moment_count": len(recalled_moments),
        "moment_candidate_count": len(moment_candidates),
        "suppressed_moment_count": len(suppressed_moments),
        "suppressed_bucket_count": len(suppressed_buckets),
        "planner_skip_reason": str(planner_debug.get("skip_reason") or ""),
        "related_memory_preview": related_memory[:500],
        "diffused": [compact_diffusion_debug(row) for row in diffused_debug_rows[: args.top]],
    }


async def evaluate_query(
    service: GatewayService,
    bucket_mgr: BucketManager,
    case: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    query = str(case.get("query") or "").strip()
    expected_ids = [str(item) for item in case.get("expected_bucket_ids") or []]
    buckets = await bucket_mgr.list_all()
    search_query = service._dynamic_recall_search_query(query)
    admitted, suppressed = await service._dynamic_bucket_candidate_items(
        query,
        args.session_id,
        buckets,
        search_query=search_query,
        allow_semantic=not args.disable_semantic,
        allow_rerank=False,
    )
    all_items = list(admitted) + sorted(
        suppressed,
        key=lambda item: (-float(item.get("score") or 0.0), item_bucket_id(item)),
    )
    score_maps = source_score_maps(all_items)
    rrf_rows = reciprocal_rank_fusion(score_maps, k=args.rrf_k)
    current_ids = [item_bucket_id(item) for item in all_items]
    current_rank_by_id = {bucket_id: index for index, bucket_id in enumerate(current_ids, start=1)}
    rrf_ranked_ids = sorted(
        current_ids,
        key=lambda bucket_id: (
            -float((rrf_rows.get(bucket_id) or {}).get("rrf_score") or 0.0),
            current_rank_by_id.get(bucket_id, 999999),
            bucket_id,
        ),
    )
    rrf_rank_by_id = {bucket_id: index for index, bucket_id in enumerate(rrf_ranked_ids, start=1)}
    expected_current_ranks = {
        bucket_id: current_rank_by_id.get(bucket_id)
        for bucket_id in expected_ids
    }
    expected_rrf_ranks = {
        bucket_id: rrf_rank_by_id.get(bucket_id)
        for bucket_id in expected_ids
    }
    item_by_id = {item_bucket_id(item): item for item in all_items if item_bucket_id(item)}
    admitted_ids = {item_bucket_id(item) for item in admitted}
    current_rows = [
        row_for_item(
            service,
            item,
            query=query,
            current_rank=index,
            rrf_rank=rrf_rank_by_id.get(item_bucket_id(item), index),
            rrf_row=rrf_rows.get(item_bucket_id(item), {}),
            admitted=item_bucket_id(item) in admitted_ids,
        )
        for index, item in enumerate(all_items, start=1)
        if item_bucket_id(item)
    ]
    rrf_rows_out = [
        row_for_item(
            service,
            item_by_id[bucket_id],
            query=query,
            current_rank=current_rank_by_id.get(bucket_id, 999999),
            rrf_rank=index,
            rrf_row=rrf_rows.get(bucket_id, {}),
            admitted=bucket_id in admitted_ids,
        )
        for index, bucket_id in enumerate(rrf_ranked_ids, start=1)
        if bucket_id in item_by_id
    ]
    return {
        "query": query,
        "expected_bucket_ids": expected_ids,
        "note": str(case.get("note") or ""),
        "expected_current_ranks": expected_current_ranks,
        "expected_rrf_ranks": expected_rrf_ranks,
        "expected_current_hit_top": min(
            (rank for rank in expected_current_ranks.values() if rank is not None),
            default=None,
        ),
        "expected_rrf_hit_top": min(
            (rank for rank in expected_rrf_ranks.values() if rank is not None),
            default=None,
        ),
        "search_query": search_query,
        "candidate_count": len(all_items),
        "admitted_count": len(admitted),
        "suppressed_count": len(suppressed),
        "source_counts": {source: len(scores) for source, scores in score_maps.items()},
        "current_top": current_rows[: args.top],
        "rrf_top": rrf_rows_out[: args.top],
        "diffusion_debug": await evaluate_diffusion_debug(service, case, args, buckets),
    }


def print_result(result: dict[str, Any], *, top: int) -> None:
    print(f"\nQUERY: {result['query']}")
    print(f"search_query={result['search_query']!r} candidates={result['candidate_count']} admitted={result['admitted_count']}")
    expected = result.get("expected_bucket_ids") or []
    if expected:
        print(
            "expected="
            + ",".join(expected)
            + f" current_best={result.get('expected_current_hit_top')} rrf_best={result.get('expected_rrf_hit_top')}"
        )
    print("CURRENT Dynamic-alpha")
    for row in result["current_top"][:top]:
        why = row.get("recall_why") or {}
        print(
            f"  #{row['current_rank']:02d} rrf#{row['rrf_rank']:02d} "
            f"delta={row['rank_delta']:+d} score={row['score']:.4f} "
            f"rrf={row['rrf_score']:.6f} admitted={row['admitted']} "
            f"why={why.get('primary_source') or '-'}:{why.get('admission_reason') or '-'} "
            f"{row['bucket_name']} [{row['bucket_id']}]"
        )
    print("RRF order")
    for row in result["rrf_top"][:top]:
        why = row.get("recall_why") or {}
        print(
            f"  rrf#{row['rrf_rank']:02d} cur#{row['current_rank']:02d} "
            f"delta={row['rank_delta']:+d} rrf={row['rrf_score']:.6f} "
            f"score={row['score']:.4f} admitted={row['admitted']} "
            f"why={why.get('primary_source') or '-'}:{why.get('admission_reason') or '-'} "
            f"{row['bucket_name']} [{row['bucket_id']}]"
        )
    diffusion_debug = result.get("diffusion_debug") or {}
    diffused = diffusion_debug.get("diffused") or []
    if diffused:
        print("Diffusion debug")
        for row in diffused[:top]:
            trace = row.get("diffusion_trace") or {}
            print(
                f"  injected={row['injected']} gate={trace.get('gate_reason') or '-'} "
                f"final={trace.get('final_status') or '-'}:{trace.get('final_reason') or '-'} "
                f"why={row.get('why') or '-'} conf={row.get('confidence'):.2f}/{row.get('confidence_source') or '-'} "
                f"topic={','.join(row.get('topic_evidence_terms') or []) or '-'} "
                f"{trace.get('seed_bucket_name') or '-'} -> {row['bucket_name']} "
                f"path={trace.get('path_trace') or '-'}"
            )


async def main() -> int:
    args = parse_args()
    if args.write_default_cases:
        output_path = Path(args.write_default_cases)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            for case in DEFAULT_EVAL_CASES:
                fh.write(json.dumps(case, ensure_ascii=False) + "\n")
        print(f"Wrote default cases: {output_path}")
        return 0
    cases = load_cases(args)
    if not cases:
        print("No queries provided. Use --query, --queries-file, or --cases-file.", file=sys.stderr)
        return 2
    buckets_dir = Path(args.buckets_dir)
    if not buckets_dir.exists():
        print(f"Buckets dir does not exist: {buckets_dir}", file=sys.stderr)
        return 2
    api_key = os.environ.get(args.embedding_api_key_env, "")
    if args.disable_semantic:
        api_key = ""
    with tempfile.TemporaryDirectory(prefix="ombre-rrf-ab-") as tmp:
        state_dir = Path(tmp) / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        copied_entity_edges = copy_entity_edges_to_state(
            resolve_entity_edges_path(args, buckets_dir),
            state_dir,
        )
        config = build_config(args, state_dir, api_key)
        bucket_mgr = BucketManager(config)
        embedding_engine = OfflineEmbeddingEngine(
            db_path=buckets_dir / "embeddings.db",
            api_key=api_key,
            base_url=args.embedding_base_url,
            model=args.embedding_model,
            max_chars=int(config["embedding"].get("max_chars") or 6000),
            query_instruction=str(config["embedding"].get("query_instruction") or ""),
        )
        service = GatewayService(
            config,
            bucket_mgr=bucket_mgr,
            embedding_engine=embedding_engine,
            reranker_engine=DisabledReranker(),
            state_store=GatewayStateStore(str(state_dir / "gateway_state.db")),
        )
        if not embedding_engine.enabled and not args.disable_semantic:
            print(
                f"Semantic lane disabled: missing {args.embedding_api_key_env} or embeddings.db.",
                file=sys.stderr,
            )
        if copied_entity_edges:
            print(f"Using entity_edges: {copied_entity_edges}", file=sys.stderr)
        output_path = Path(args.output) if args.output else None
        output_fh = output_path.open("w", encoding="utf-8") if output_path else None
        try:
            for case in cases:
                result = await evaluate_query(service, bucket_mgr, case, args)
                print_result(result, top=args.top)
                if output_fh:
                    output_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
        finally:
            if output_fh:
                output_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
