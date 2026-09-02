#!/usr/bin/env python3
"""Run fixed Germany Event/Scene scope and candidate fixtures via full shadow.

The runner requests ``simulation=true`` and only evaluates candidate/debug
receipts.  It never records an injection or applies an admission decision.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "resources" / "typed_recall_germany_shadow_gold_v1.json"


def _post_json(url: str, token: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def _ref(row: dict[str, Any]) -> str:
    kind = str(row.get("owner_kind") or "")
    owner_id = str(row.get("owner_id") or "")
    return f"{kind}:{owner_id}" if kind and owner_id else ""


def _evaluate_case(
    case: dict[str, Any],
    *,
    endpoint: str,
    token: str,
    timeout: float,
) -> dict[str, Any]:
    query = str(case.get("query") or "")
    payload = {
        "query": query,
        "messages": [{"role": "user", "content": query}],
        "session_id": f"typed-germany-shadow-{case['id']}",
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
    started = time.perf_counter()
    response = _post_json(endpoint, token, payload, timeout)
    wall_ms = round((time.perf_counter() - started) * 1000)
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
    candidates = list(shadow.get("candidates") or [])
    candidate_refs = [_ref(row) for row in candidates if _ref(row)]
    by_ref = {_ref(row): row for row in candidates if _ref(row)}
    expected = case.get("expected") if isinstance(case.get("expected"), dict) else {}

    actual = {
        "shadow_status": str(shadow.get("status") or ""),
        "reason": str(shadow.get("reason") or ""),
        "scope_status": str(scope.get("status") or ""),
        "intent": str(scope.get("intent") or ""),
        "operator": str(scope.get("operator") or ""),
        "arc_key": str((scope.get("scope_anchor") or {}).get("arc_key") or ""),
        "candidate_count": int(shadow.get("candidate_count") or 0),
    }
    failures: list[str] = []
    for field in (
        "shadow_status",
        "reason",
        "scope_status",
        "intent",
        "operator",
        "arc_key",
        "candidate_count",
    ):
        if field in expected and actual[field] != expected[field]:
            failures.append(f"{field}: expected={expected[field]!r} actual={actual[field]!r}")
    if (
        "min_candidate_count" in expected
        and actual["candidate_count"] < int(expected["min_candidate_count"])
    ):
        failures.append(
            "candidate_count: "
            f"expected>={expected['min_candidate_count']!r} "
            f"actual={actual['candidate_count']!r}"
        )

    required_refs = {str(value) for value in expected.get("required_candidate_refs") or []}
    missing_refs = sorted(required_refs - set(candidate_refs))
    if missing_refs:
        failures.append(f"missing_candidate_refs={missing_refs}")

    forbidden_refs = {str(value) for value in expected.get("forbidden_candidate_refs") or []}
    present_forbidden = sorted(forbidden_refs.intersection(candidate_refs))
    if present_forbidden:
        failures.append(f"forbidden_candidate_refs={present_forbidden}")

    for target_ref, source in (expected.get("required_candidate_sources") or {}).items():
        sources = list((by_ref.get(str(target_ref)) or {}).get("candidate_sources") or [])
        if str(source) not in sources:
            failures.append(f"missing_source[{target_ref}]={source}")

    if expected.get("all_candidates_in_scope"):
        expected_arc = str(expected.get("arc_key") or "")
        leaked = []
        for row in candidates:
            card_keys = {
                str(card.get("arc_key") or "")
                for card in row.get("arc_cards") or []
                if isinstance(card, dict)
            }
            if expected_arc not in card_keys:
                leaked.append(_ref(row))
        if leaked:
            failures.append(f"out_of_scope_candidates={leaked}")

    shadow_contract_ok = bool(
        shadow.get("decision_applied") is False
        and shadow.get("live_injection_enabled") is False
        and all(row.get("decision_applied") is False for row in candidates)
    )
    if not shadow_contract_ok:
        failures.append("shadow_contract_violation")

    return {
        "id": str(case.get("id") or ""),
        "query": query,
        "passed": not failures,
        "failures": failures,
        "actual": actual,
        "candidate_refs": candidate_refs,
        "candidate_sources": {
            ref: list((by_ref.get(ref) or {}).get("candidate_sources") or [])
            for ref in candidate_refs
        },
        "required_candidate_refs": sorted(required_refs),
        "wall_ms": wall_ms,
        "shadow_contract_ok": shadow_contract_ok,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--token-env", default="OMBRE_GATEWAY_TOKEN")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    token = os.environ.get(args.token_env, "").strip()
    if not token:
        raise SystemExit(f"missing token environment variable: {args.token_env}")
    dataset = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    cases = list(dataset.get("cases") or [])
    if args.limit > 0:
        cases = cases[: args.limit]
    endpoint = args.base_url.rstrip("/") + "/api/hook/recall"
    rows = [
        _evaluate_case(case, endpoint=endpoint, token=token, timeout=args.timeout)
        for case in cases
    ]
    failed = [row["id"] for row in rows if not row["passed"]]
    report = {
        "dataset": dataset.get("dataset"),
        "frozen_from_live_head": dataset.get("frozen_from_live_head"),
        "endpoint": endpoint,
        "simulation": True,
        "simulation_scope": "full_shadow",
        "decision_applied": False,
        "live_injection_enabled": False,
        "metrics": {
            "case_count": len(rows),
            "passed": len(rows) - len(failed),
            "failed": len(failed),
            "target_case_count": sum(bool(row["required_candidate_refs"]) for row in rows),
            "target_recalled": sum(
                bool(set(row["required_candidate_refs"]).issubset(row["candidate_refs"]))
                for row in rows
                if row["required_candidate_refs"]
            ),
            "shadow_contract_violations": sum(
                not row["shadow_contract_ok"] for row in rows
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


if __name__ == "__main__":
    raise SystemExit(main())
