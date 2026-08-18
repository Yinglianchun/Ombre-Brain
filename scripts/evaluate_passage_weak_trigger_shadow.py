#!/usr/bin/env python3
"""Evaluate weak-trigger-gated query-view through the read-only Gateway simulation API.

The runner never requests live injection. It sends ``simulation=true`` with
``simulation_scope=full_shadow`` and reports the ordinary formal recall and the
candidate-only passage/query-view diagnostics separately.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


CLEAR_CURRENT_OR_CHITCHAT_CONFOUNDERS = frozenset(
    {
        "现在 token/字节预算怎么配？",
        "今天又下雨了。",
        "这个回复太文艺了，改直白点。",
        "现在官方又更新模型了吗？",
        "现在 Chat 端能用 MCP 吗？",
        "现在 Haven profile 已验证好吗？",
        "现在小红书登录成功了吗？",
        "暗房现在有几条？",
        "现在欲望系统在干嘛？",
        "现在还会发两遍吗？",
    }
)


def _load_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    if args.cases:
        payload = json.loads(Path(args.cases).read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise SystemExit("cases file must contain a JSON list")
        cases.extend(payload)

    if args.gold:
        payload = json.loads(Path(args.gold).read_text(encoding="utf-8"))
        for row in payload.get("cases") or []:
            verdict = str(row.get("action_verdict") or "")
            observation_id = str(row.get("observation_id") or "")
            tags = ["terra_fixed_gold"]
            if observation_id == "hook-8104":
                tags.extend(["focused_long_query", "multi_clause"])
            if observation_id == "hook-9179":
                tags.append("present_chitchat")
            if observation_id == "hook-7635":
                tags.extend(["volatile_current", "token_or_count_current_state"])
            cases.append(
                {
                    "id": f"gold-{observation_id}",
                    "suite": "terra_gold_22",
                    "case_kind": verdict,
                    "tags": tags,
                    "query": row.get("query"),
                    "target_refs": [
                        f"scene:{item}"
                        for item in row.get("labeled_relevant_ids") or []
                    ],
                    "irrelevant_refs": [
                        f"scene:{item}"
                        for item in row.get("labeled_irrelevant_ids") or []
                    ],
                    "expected_trigger": (
                        "if_target_not_formally_recalled"
                        if verdict == "correct"
                        else False
                        if verdict == "false_positive"
                        else None
                    ),
                }
            )

    if args.live_pairs:
        payload = json.loads(Path(args.live_pairs).read_text(encoding="utf-8"))
        for pair_index, pair in enumerate(payload, start=1):
            target_ref = f"{pair['owner_kind']}:{pair['owner_id']}"
            for case_kind in ("probe", "confounder"):
                query = str(pair[case_kind])
                tags = ["terra_live_pair", case_kind]
                if query == "现在 token/字节预算怎么配？":
                    tags.extend(["token_byte", "volatile_current"])
                if query in {"今天又下雨了。", "这个回复太文艺了，改直白点。"}:
                    tags.append("present_chitchat")
                if query in {
                    "我是更像小猫还是狐狸？",
                    "最初那只小狐狸指谁？",
                }:
                    tags.append("cat_fox")
                cases.append(
                    {
                        "id": f"live-{pair_index:02d}-{case_kind}",
                        "suite": "terra_live_pairs_20",
                        "case_kind": case_kind,
                        "tags": tags,
                        "query": query,
                        "target_refs": [target_ref],
                        "expected_trigger": (
                            "if_target_not_formally_recalled"
                            if case_kind == "probe"
                            else False
                        ),
                        "target_is_clearly_irrelevant": (
                            case_kind == "confounder"
                            and query in CLEAR_CURRENT_OR_CHITCHAT_CONFOUNDERS
                        ),
                    }
                )

    if args.planner_cases:
        payload = json.loads(Path(args.planner_cases).read_text(encoding="utf-8"))
        cat = next(
            (row for row in payload if row.get("id") == "cat_fox_correction"),
            None,
        )
        if cat:
            for with_context in (False, True):
                cases.append(
                    {
                        "id": f"focused-cat-fox-{'with' if with_context else 'without'}-context",
                        "suite": "focused",
                        "case_kind": "probe",
                        "tags": [
                            "cat_fox",
                            "previous_turn_coreference",
                            "with_context" if with_context else "without_context",
                        ],
                        "query": cat.get("current_message"),
                        "previous_turn": cat.get("previous_turn") if with_context else "",
                        "target_refs": [f"scene:{item}" for item in cat.get("target_ids") or []],
                        "expected_trigger": "if_target_not_formally_recalled",
                    }
                )
    return cases


def _percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _refs(rows: list[dict[str, Any]]) -> list[str]:
    output: list[str] = []
    for row in rows:
        kind = str(row.get("owner_kind") or "")
        owner_id = str(row.get("owner_id") or "")
        if kind and owner_id:
            output.append(f"{kind}:{owner_id}")
    return output


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


def _messages(case: dict[str, Any]) -> list[dict[str, str]]:
    previous_turn = str(case.get("previous_turn") or "").strip()
    query = str(case.get("query") or "").strip()
    if not previous_turn:
        return [{"role": "user", "content": query}]
    return [
        {"role": "user", "content": previous_turn},
        {"role": "user", "content": query},
    ]


def evaluate_case(
    case: dict[str, Any],
    *,
    endpoint: str,
    token: str,
    timeout: float,
) -> dict[str, Any]:
    query = str(case.get("query") or "").strip()
    payload = {
        "query": query,
        "messages": _messages(case),
        "session_id": f"passage-weak-trigger-shadow-{case['id']}",
        "recall_mode": "full",
        "simulation": True,
        "simulation_scope": "full_shadow",
        "include_debug": True,
        "include_context": False,
        "include_recent_context": False,
        "max_cards": 2,
    }
    started_at = time.perf_counter()
    response = _post_json(endpoint, token, payload, timeout)
    wall_ms = max(0, round((time.perf_counter() - started_at) * 1000))

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
    passage = (
        budget.get("passage_candidate_shadow")
        if isinstance(budget.get("passage_candidate_shadow"), dict)
        else {}
    )
    trigger = (
        passage.get("weak_candidate_trigger_shadow")
        if isinstance(passage.get("weak_candidate_trigger_shadow"), dict)
        else {}
    )
    query_view = (
        passage.get("query_view_shadow")
        if isinstance(passage.get("query_view_shadow"), dict)
        else {}
    )

    baseline_refs = set(_refs(list(passage.get("candidates") or [])))
    expanded_refs = set(_refs(list(query_view.get("candidates") or [])))
    added_refs = sorted(expanded_refs - baseline_refs)
    recalled_ids = {
        str(item)
        for item in (response.get("recalled_ids") or debug.get("injected_bucket_ids") or [])
        if str(item)
    }
    target_refs = {
        str(item) for item in case.get("target_refs") or [] if str(item)
    }
    target_ids = {item.split(":", 1)[-1] for item in target_refs}
    irrelevant_refs = {
        str(item) for item in case.get("irrelevant_refs") or [] if str(item)
    }

    query_view_count = int(trigger.get("query_view_count") or 0)
    query_view_executable = query_view_count > 1
    expected_trigger = case.get("expected_trigger")
    rescue_needed: bool | None = None
    if expected_trigger == "if_target_not_formally_recalled":
        rescue_needed = None if not target_ids else not bool(target_ids & recalled_ids)
        expected_trigger = (
            None
            if rescue_needed is None
            else bool(rescue_needed and query_view_executable)
        )
    elif expected_trigger is not None:
        expected_trigger = bool(expected_trigger)

    would_trigger = bool(trigger.get("would_trigger"))
    return {
        "id": case["id"],
        "suite": case.get("suite"),
        "case_kind": case.get("case_kind"),
        "tags": list(case.get("tags") or []),
        "query": query,
        "previous_turn": str(case.get("previous_turn") or ""),
        "expected_trigger": expected_trigger,
        "rescue_needed": rescue_needed,
        "query_view_executable": query_view_executable,
        "would_trigger": would_trigger,
        "trigger_correct": (
            None if expected_trigger is None else would_trigger == expected_trigger
        ),
        "trigger_reason": trigger.get("reason"),
        "weak_candidate_detected": trigger.get("weak_candidate_detected"),
        "weak_candidate_reason": trigger.get("weak_candidate_reason"),
        "execution_gate": trigger.get("execution_gate") or {},
        "trigger_decision_applied": trigger.get("decision_applied"),
        "query_view_execution_changed": trigger.get("query_view_execution_changed"),
        "live_execution_changed": trigger.get("live_execution_changed"),
        "query_view_status": query_view.get("status"),
        "query_view_executed": query_view.get("status") == "ok",
        "query_view_live_injection_enabled": query_view.get("live_injection_enabled"),
        "formal_recalled_ids": sorted(recalled_ids),
        "target_refs": sorted(target_refs),
        "irrelevant_refs": sorted(irrelevant_refs),
        "baseline_candidate_refs": sorted(baseline_refs),
        "expanded_candidate_refs": sorted(expanded_refs),
        "added_candidate_refs": added_refs,
        "new_correct_refs": sorted(target_refs & set(added_refs)),
        "new_labeled_irrelevant_refs": sorted(irrelevant_refs & set(added_refs)),
        "historical_target_added_on_confounder": bool(
            case.get("case_kind") == "confounder" and target_refs & set(added_refs)
        ),
        "clearly_irrelevant_target_added": bool(
            case.get("target_is_clearly_irrelevant") and target_refs & set(added_refs)
        ),
        "query_view_incremental_ms": query_view.get("timing_ms"),
        "top_body_semantic": trigger.get("top_body_semantic"),
        "formal_recalled_count": trigger.get("formal_recalled_count"),
        "query_view_count": query_view_count,
        "request_wall_ms": wall_ms,
        "prepare_timing_debug": debug.get("prepare_timing_debug") or {},
        "passage_status": passage.get("status"),
        "passage_live_injection_enabled": passage.get("live_injection_enabled"),
        "response_ok": response.get("ok"),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    judged = [row for row in rows if row["expected_trigger"] is not None]
    expected_yes = [row for row in judged if row["expected_trigger"]]
    expected_no = [row for row in judged if not row["expected_trigger"]]
    triggered = [row for row in rows if row["would_trigger"]]
    rescue_needed = [row for row in rows if row.get("rescue_needed") is True]
    rescue_executable = [row for row in rescue_needed if row["query_view_executable"]]
    incremental = [
        int(row["query_view_incremental_ms"])
        for row in triggered
        if isinstance(row.get("query_view_incremental_ms"), (int, float))
    ]
    wall = [int(row["request_wall_ms"]) for row in rows]
    return {
        "case_count": len(rows),
        "judged_trigger_count": len(judged),
        "expected_trigger_and_triggered": sum(row["would_trigger"] for row in expected_yes),
        "expected_trigger_count": len(expected_yes),
        "unexpected_trigger": sum(row["would_trigger"] for row in expected_no),
        "expected_no_trigger_count": len(expected_no),
        "rescue_needed_count": len(rescue_needed),
        "rescue_executable_count": len(rescue_executable),
        "rescue_executable_and_triggered": sum(
            row["would_trigger"] for row in rescue_executable
        ),
        "rescue_single_view_count": sum(
            not row["query_view_executable"] for row in rescue_needed
        ),
        "false_positive_query_view_execution_count": sum(
            row.get("case_kind") == "false_positive"
            and row.get("query_view_executed")
            for row in rows
        ),
        "trigger_accuracy": (
            round(sum(bool(row["trigger_correct"]) for row in judged) / len(judged), 4)
            if judged
            else None
        ),
        "new_correct_parent_count": sum(len(row["new_correct_refs"]) for row in rows),
        "new_labeled_irrelevant_parent_count": sum(
            len(row["new_labeled_irrelevant_refs"]) for row in rows
        ),
        "historical_target_added_on_confounder_count": sum(
            row["historical_target_added_on_confounder"] for row in rows
        ),
        "clearly_irrelevant_target_added_count": sum(
            row["clearly_irrelevant_target_added"] for row in rows
        ),
        "triggered_query_view_incremental_ms": {
            "count": len(incremental),
            "mean": round(statistics.mean(incremental), 1) if incremental else None,
            "median": round(statistics.median(incremental), 1) if incremental else None,
            "p95": _percentile(incremental, 0.95),
            "max": max(incremental) if incremental else None,
        },
        "request_wall_ms": {
            "mean": round(statistics.mean(wall), 1) if wall else None,
            "median": round(statistics.median(wall), 1) if wall else None,
            "p95": _percentile(wall, 0.95),
            "max": max(wall) if wall else None,
        },
        "shadow_contract_violations": [
            row["id"]
            for row in rows
            if row.get("passage_live_injection_enabled") is not False
            or (
                row.get("query_view_status") not in {None, "not_needed"}
                and row.get("query_view_live_injection_enabled") is not False
            )
            or row.get("live_execution_changed") is not False
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="")
    parser.add_argument("--gold", default="")
    parser.add_argument("--live-pairs", default="")
    parser.add_argument("--planner-cases", default="")
    parser.add_argument("--base-url", default="http://127.0.0.1:18002")
    parser.add_argument("--token-env", default="OMBRE_GATEWAY_TOKEN")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--output", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    token = os.environ.get(args.token_env, "").strip()
    if not token:
        raise SystemExit(f"missing token environment variable: {args.token_env}")
    cases = _load_cases(args)
    if not cases:
        raise SystemExit("no cases loaded")
    if args.limit > 0:
        cases = cases[: args.limit]

    endpoint = args.base_url.rstrip("/") + "/api/hook/recall"
    rows = [
        evaluate_case(case, endpoint=endpoint, token=token, timeout=args.timeout)
        for case in cases
    ]
    report = {
        "endpoint": endpoint,
        "simulation": True,
        "simulation_scope": "full_shadow",
        "live_decision_applied": False,
        "query_view_execution_gate_applied": True,
        "metrics": summarize(rows),
        "cases": rows,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
        print(json.dumps({"output": args.output, "metrics": report["metrics"]}, ensure_ascii=False, indent=2))
    else:
        print(rendered, end="")
    return 1 if report["metrics"]["shadow_contract_violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
