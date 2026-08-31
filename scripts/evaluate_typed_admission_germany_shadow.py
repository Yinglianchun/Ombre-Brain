#!/usr/bin/env python3
"""Evaluate operator-aware typed admission without changing live decisions."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_recall.typed_admission_shadow import (
    evaluate_typed_admission_shadow,
    typed_admission_rerank_query,
)
from reranker_engine import RerankerEngine
from utils import load_config


DEFAULT_CASES = ROOT / "resources" / "typed_admission_germany_gold_v1.json"


def _post_json(url: str, token: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def _candidate_ref(row: dict[str, Any]) -> str:
    kind = str(row.get("owner_kind") or "")
    owner_id = str(row.get("owner_id") or "")
    return f"{kind}:{owner_id}" if kind and owner_id else ""


def _reranker_document(row: dict[str, Any]) -> str:
    title = str(row.get("title") or "").strip()
    passages = [
        str(passage.get("text") or "").strip()
        for passage in row.get("passages") or []
        if isinstance(passage, dict) and str(passage.get("text") or "").strip()
    ]
    return f"title: {title}\nbody: {'\n'.join(passages[:2])}"[:4000]


def _typed_candidates(
    query: str,
    case_id: str,
    *,
    endpoint: str,
    token: str,
    timeout: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = {
        "query": query,
        "messages": [{"role": "user", "content": query}],
        "session_id": f"typed-admission-germany-{case_id}",
        "recall_mode": "full",
        "simulation": True,
        "simulation_scope": "full_shadow",
        "include_debug": True,
        "include_context": False,
        "include_recent_context": False,
        "include_diffused": False,
        "allow_rerank": False,
        "max_cards": 0,
    }
    response = _post_json(endpoint, token, payload, timeout)
    debug = response.get("debug") if isinstance(response.get("debug"), dict) else {}
    semantic = (
        debug.get("semantic_recall_debug")
        if isinstance(debug.get("semantic_recall_debug"), dict)
        else {}
    )
    budget = (
        semantic.get("retrieval_budget")
        if isinstance(semantic.get("retrieval_budget"), dict)
        else {}
    )
    shadow = (
        budget.get("passage_candidate_shadow")
        if isinstance(budget.get("passage_candidate_shadow"), dict)
        else {}
    )
    scope = shadow.get("entity_scope") if isinstance(shadow.get("entity_scope"), dict) else {}
    return scope, list(shadow.get("candidates") or [])


async def _evaluate_case(
    case: dict[str, Any],
    *,
    endpoint: str,
    token: str,
    timeout: float,
    engine: RerankerEngine,
    threshold: float,
) -> dict[str, Any]:
    query = str(case.get("query") or "")
    scope, candidates = _typed_candidates(
        query,
        str(case.get("id") or ""),
        endpoint=endpoint,
        token=token,
        timeout=timeout,
    )
    probe = evaluate_typed_admission_shadow(
        query,
        scope,
        candidates,
        direct_threshold=threshold,
    )
    rerank_scores: dict[str, float] = {}
    reranker_called = probe["mode"] == "direct_evidence_rerank" and bool(candidates)
    if reranker_called:
        rerank_query = typed_admission_rerank_query(query, scope)
        documents = [_reranker_document(row) for row in candidates]
        results = await engine.rerank_shadow(rerank_query, documents, top_n=len(documents))
        for result in results:
            if 0 <= int(result.index) < len(candidates):
                ref = _candidate_ref(candidates[int(result.index)])
                if ref:
                    rerank_scores[ref] = float(result.score)
    admission = evaluate_typed_admission_shadow(
        query,
        scope,
        candidates,
        rerank_scores=rerank_scores,
        direct_threshold=threshold,
    )

    failures = []
    if admission["mode"] != str(case.get("expected_mode") or ""):
        failures.append(
            f"mode expected={case.get('expected_mode')!r} actual={admission['mode']!r}"
        )
    expected_selected = {str(ref) for ref in case.get("expected_selected_refs") or []}
    selected = set(admission["selected_refs"])
    if selected != expected_selected:
        failures.append(
            f"selected expected={sorted(expected_selected)} actual={sorted(selected)}"
        )
    expected_rerank_query = case.get("expected_rerank_query")
    if expected_rerank_query is not None and admission["rerank_query"] != expected_rerank_query:
        failures.append(
            f"rerank_query expected={expected_rerank_query!r} actual={admission['rerank_query']!r}"
        )
    required_material = {str(ref) for ref in case.get("required_material_refs") or []}
    missing_material = sorted(required_material - set(admission["material_refs"]))
    if missing_material:
        failures.append(f"missing_material_refs={missing_material}")
    if case.get("closed_world_selection") and selected - expected_selected:
        failures.append(f"unexpected_selected_refs={sorted(selected - expected_selected)}")
    if admission.get("decision_applied") is not False or admission.get("live_injection_enabled") is not False:
        failures.append("shadow_contract_violation")

    return {
        "id": str(case.get("id") or ""),
        "query": query,
        "passed": not failures,
        "failures": failures,
        "mode": admission["mode"],
        "operator": admission["operator"],
        "intent": admission["intent"],
        "rerank_query": admission["rerank_query"],
        "reranker_called": reranker_called,
        "selected_refs": admission["selected_refs"],
        "material_refs": admission["material_refs"],
        "expected_selected_refs": sorted(expected_selected),
        "candidate_count": len(candidates),
        "candidates": admission["candidates"],
        "decision_applied": admission["decision_applied"],
        "live_injection_enabled": admission["live_injection_enabled"],
    }


async def run(args: argparse.Namespace) -> int:
    token = os.environ.get(args.token_env, "").strip()
    if not token:
        raise SystemExit(f"missing token environment variable: {args.token_env}")
    dataset = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    cases = list(dataset.get("cases") or [])
    if args.limit > 0:
        cases = cases[: args.limit]
    threshold = float(dataset.get("direct_threshold") or 0.65)
    endpoint = args.base_url.rstrip("/") + "/api/hook/recall"
    engine = RerankerEngine(load_config(args.config or None))
    rows = []
    for case in cases:
        rows.append(
            await _evaluate_case(
                case,
                endpoint=endpoint,
                token=token,
                timeout=args.timeout,
                engine=engine,
                threshold=threshold,
            )
        )
    failed = [row["id"] for row in rows if not row["passed"]]
    rerank_rows = [row for row in rows if row["mode"] == "direct_evidence_rerank"]
    rerank_pair_count = sum(row["candidate_count"] for row in rerank_rows)
    rerank_positive_count = sum(
        len(row["expected_selected_refs"]) for row in rerank_rows
    )
    report = {
        "dataset": dataset.get("dataset"),
        "frozen_from_live_head": dataset.get("frozen_from_live_head"),
        "simulation": True,
        "decision_applied": False,
        "live_injection_enabled": False,
        "reranker": {
            "model": engine.model,
            "shadow_ready": engine.shadow_ready,
            "direct_threshold": threshold,
            "role": "direct_evidence_only",
        },
        "metrics": {
            "case_count": len(rows),
            "passed": len(rows) - len(failed),
            "failed": len(failed),
            "reranker_call_count": sum(row["reranker_called"] for row in rows),
            "expected_selected_count": sum(
                len(row["expected_selected_refs"]) for row in rows
            ),
            "selected_count": sum(len(row["selected_refs"]) for row in rows),
            "direct_rerank_pair_count": rerank_pair_count,
            "direct_rerank_positive_pair_count": rerank_positive_count,
            "direct_rerank_hard_negative_pair_count": (
                rerank_pair_count - rerank_positive_count
            ),
            "false_direct_admission_count": sum(
                len(set(row["selected_refs"]) - set(row["expected_selected_refs"]))
                for row in rows
            ),
            "shadow_contract_violations": sum(
                row["decision_applied"] is not False
                or row["live_injection_enabled"] is not False
                for row in rows
            ),
        },
        "failed_cases": failed,
        "cases": rows,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
        print(json.dumps({"output": args.output, "metrics": report["metrics"]}, ensure_ascii=False))
    else:
        print(rendered, end="")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--config", default="")
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--token-env", default="OMBRE_GATEWAY_TOKEN")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
