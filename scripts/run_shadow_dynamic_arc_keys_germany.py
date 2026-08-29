from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.shadow_arc_candidates_germany import CARD_SPECS, build_nodes, fetch_snapshot
from scripts.shadow_dynamic_arc_keys import ReviewedAnchor, propose_dynamic_arc_keys


BRIDGE_VISUAL_ANCHOR = ReviewedAnchor(
    key_kind="project",
    label="bridge visual direction",
    terms=("Haven Bridge", "视觉方向", "Unseen"),
    basis="compound",
)
BRIDGE_VISUAL_ARC_KEY = "project:bridge-visual-direction"
KNOWN_ARC_KEYS = frozenset(
    {str(arc_key) for arc_key, _title, _anchors in CARD_SPECS}
    | {BRIDGE_VISUAL_ARC_KEY}
)


def fetch_target_receipt(host: str, key: Path) -> dict[str, Any]:
    remote = r'''import json,os,subprocess,urllib.request
source='/opt/haven_bridge-src'
pid=subprocess.check_output(['systemctl','show','haven-bridge.service','-p','MainPID','--value'],text=True).strip()
status=json.loads(urllib.request.urlopen('http://127.0.0.1:18789/api/status',timeout=10).read().decode('utf-8'))
print(json.dumps({
  'hostname':subprocess.check_output(['hostname'],text=True).strip(),
  'source_path':source,
  'git_status':subprocess.check_output(['git','-C',source,'status','--short','--branch'],text=True).strip(),
  'git_head':subprocess.check_output(['git','-C',source,'rev-parse','--short','HEAD'],text=True).strip(),
  'service_active':subprocess.check_output(['systemctl','is-active','haven-bridge.service'],text=True).strip(),
  'service_pid':pid,
  'service_cwd':os.path.realpath('/proc/'+pid+'/cwd'),
  'bridge_backend':status.get('backend'),
  'bridge_root':status.get('root'),
  'bridge_db':status.get('db'),
  'memory_review_status':(status.get('memory_review') or {}).get('status'),
  'track_router_cursor_message_id':(status.get('memory_review') or {}).get('track_router_cursor_message_id'),
},ensure_ascii=False))
'''
    result = subprocess.run(
        ["ssh", "-i", str(key), f"root@{host}", "python3", "-"],
        input=remote,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Germany target receipt failed: {result.stderr.strip()[:500]}")
    receipt = json.loads(result.stdout)
    if receipt.get("service_active") != "active" or receipt.get("bridge_backend") != "ok":
        raise RuntimeError("Germany target receipt is not healthy")
    return receipt


def scan_dynamic_keys(
    snapshot: dict[str, Any],
    *,
    session_id: int,
    target_receipt: dict[str, Any],
) -> dict[str, Any]:
    nodes = build_nodes(snapshot)
    recent_ids = {node.node_id for node in nodes if session_id in node.session_ids}
    dynamic = propose_dynamic_arc_keys(
        nodes,
        reviewed_anchors=(BRIDGE_VISUAL_ANCHOR,),
    )
    proposals: list[dict[str, Any]] = []
    for item in dynamic["proposals"]:
        recent_seed_ids = sorted(set(item["seed_ids"]).intersection(recent_ids))
        proposals.append(
            {
                **item,
                "scope_priority": "session_recent_seed" if recent_seed_ids else "global_seed",
                "session_recent_seed_ids": recent_seed_ids,
                "discovery_status": (
                    "known_confirmed" if item["arc_key"] in KNOWN_ARC_KEYS else "new_discovery"
                ),
            }
        )
    proposals.sort(
        key=lambda item: (
            item["scope_priority"] != "session_recent_seed",
            item["discovery_status"] != "new_discovery",
            item["arc_key"],
        )
    )
    blocked_reasons = {
        "parked_source",
        "routine_only",
        "routine_text",
        "event_rewrite_required",
        "intimacy_excluded",
    }
    blocked_seed_ids = {
        item["seed_id"]
        for item in dynamic["rejected_seed_receipts"]
        if item["reason"] in blocked_reasons
    }
    proposed_support_ids = {
        source_id for item in proposals for source_id in item["supporting_material_ids"]
    }
    if blocked_seed_ids.intersection(proposed_support_ids):
        raise RuntimeError("blocked intimacy/routine/parked material entered a key proposal")
    rejected_reason_counts = {
        reason: sum(item["reason"] == reason for item in dynamic["rejected_seed_receipts"])
        for reason in sorted({item["reason"] for item in dynamic["rejected_seed_receipts"]})
    }
    return {
        "schema_version": "shadow-dynamic-arc-key-live-scan-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "live_write": False,
        "target_receipt": target_receipt,
        "scope": {
            "session_id": session_id,
            "active_event_count": len(snapshot.get("events") or []),
            "active_scene_count": len(snapshot.get("scenes") or []),
            "active_node_count": len(nodes),
            "session_recent_node_count": len(recent_ids),
            "reviewed_anchor_keys": [BRIDGE_VISUAL_ARC_KEY],
            "other_dynamic_keys": "explicit_work_only",
            "model_called": False,
            "embedding_used": False,
            "date_used": False,
        },
        "summary": {
            "proposal_count": len(proposals),
            "session_recent_proposal_count": sum(
                item["scope_priority"] == "session_recent_seed" for item in proposals
            ),
            "known_confirmed_count": sum(
                item["discovery_status"] == "known_confirmed" for item in proposals
            ),
            "new_discovery_count": sum(
                item["discovery_status"] == "new_discovery" for item in proposals
            ),
        },
        "guard_verification": {
            "blocked_seed_count": len(blocked_seed_ids),
            "blocked_seed_ids_in_proposals": [],
            "rejected_reason_counts": rejected_reason_counts,
            "intimacy_routine_parked_keys_minted": False,
        },
        "proposals": proposals,
        "proposal_receipts": dynamic["proposal_receipts"],
        "rejected_seed_receipts": dynamic["rejected_seed_receipts"],
        "proposal_contract": dynamic["proposal_contract"],
        "writes_performed": [],
    }


def render_markdown(result: dict[str, Any]) -> str:
    scope = result["scope"]
    summary = result["summary"]
    receipt = result["target_receipt"]
    lines = [
        "# Germany dynamic Arc key shadow scan",
        "",
        "- Mode: live read-only; no model, embedding, date signal, admission, publication, or canonical write.",
        f"- Target: `{receipt['hostname']}` / `{receipt['source_path']}` / `{receipt['git_head']}` / service `{receipt['service_active']}`.",
        f"- Inventory: `{scope['active_event_count']}` active Events + `{scope['active_scene_count']}` active Scenes = `{scope['active_node_count']}` nodes.",
        f"- Session {scope['session_id']} recent nodes: `{scope['session_recent_node_count']}`.",
        f"- Proposals: `{summary['proposal_count']}` total; `{summary['session_recent_proposal_count']}` with recent seed; `{summary['new_discovery_count']}` new; `{summary['known_confirmed_count']}` known.",
        f"- Guard: `{result['guard_verification']['blocked_seed_count']}` intimacy/routine/parked-or-rewrite seeds blocked; none entered a proposal.",
        "",
        "## Proposals",
        "",
    ]
    if not result["proposals"]:
        lines.append("- None.")
    for item in result["proposals"]:
        lines.extend(
            [
                f"### `{item['arc_key']}`",
                "",
                f"- Discovery: `{item['discovery_status']}`",
                f"- Scope priority: `{item['scope_priority']}`",
                "- Session recent seeds: "
                + (", ".join(f"`{value}`" for value in item["session_recent_seed_ids"]) or "none"),
                "- Supporting materials: "
                + ", ".join(f"`{value}`" for value in item["supporting_material_ids"]),
                "- Decision: proposal only; admission not run; publication not attempted.",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Germany dynamic Arc key shadow scan")
    parser.add_argument("--host", default="168.119.228.217")
    parser.add_argument("--key", type=Path, default=Path.home() / ".ssh" / "id_ed25519")
    parser.add_argument("--session-id", type=int, default=22)
    parser.add_argument(
        "--artifact-base",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "narrative-arc-preview"
            / "2026-08-29-session22-dynamic-arc-key-live-scan-v1"
        ),
    )
    args = parser.parse_args()
    target_receipt = fetch_target_receipt(args.host, args.key)
    snapshot = fetch_snapshot(args.host, args.key, args.session_id)
    result = scan_dynamic_keys(
        snapshot,
        session_id=args.session_id,
        target_receipt=target_receipt,
    )
    json_path = args.artifact_base.with_suffix(".json")
    markdown_path = args.artifact_base.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    markdown_path.write_text(render_markdown(result), encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "status": "completed",
                "json_artifact": str(json_path),
                "markdown_artifact": str(markdown_path),
                **result["summary"],
                "writes_performed": [],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
