from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.shadow_arc_candidates_germany import (
    ANCHOR_STOP_WORDS,
    GENERIC_RECALL_ONLY,
    GLOBAL_STOP_ENTITIES,
    PERSON_RECALL_ONLY,
    Node,
    build_anchor_document_frequency,
    build_nodes,
    extract_anchor_candidates,
    fetch_snapshot,
    gate_decision,
)


CANDIDATE_CAP = 60
RARE_TERM_MAX_DF = 6
REVIEWED_SEARCHES = {
    "stone_memory": ("Stone Memory",),
    "cyberboss": ("CyberBoss",),
    "obsidian_memory_archive": ("Obsidian",),
    "atomoxetine_adjustment": ("托莫西汀", "25mg", "50mg"),
    "android_companion": ("Companion", "Shizuku"),
    "sticker_tool": ("表情包", "sticker"),
    "sensory_doll": ("共感娃娃", "ESP32", "MPR121"),
    "nowhere_places": ("Nowhere", "设得兰"),
}


def fetch_registry_receipt(host: str, key: Path) -> dict[str, Any]:
    remote = r'''import hashlib,json,pathlib,subprocess,urllib.request
root=pathlib.Path('/opt/Ombre-Brain-src')
registry_path=pathlib.Path('/srv/ombre-brain/state/narrative_rolls/registry.json')
raw=registry_path.read_bytes(); registry=json.loads(raw.decode('utf-8'))
health=json.loads(urllib.request.urlopen('http://127.0.0.1:18001/health',timeout=10).read().decode('utf-8'))
rolls=[]
for row in registry.get('rolls') or []:
    rolls.append({key:row.get(key) for key in (
        'narrative_id','title','arc_key','parent_narrative_id','publication_status','lifecycle',
        'linked_event_ids','linked_scene_ids','title_aliases','primary_entities','supporting_entities',
        'intent_tags','query_cues','revision'
    )})
print(json.dumps({
  'source_path':str(root),
  'git_status':subprocess.check_output(['git','-C',str(root),'status','--short','--branch'],text=True).strip(),
  'git_head':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
  'container':subprocess.check_output(['docker','inspect','ombre-brain','--format','{{.State.Status}} {{.Id}}'],text=True).strip(),
  'health':{'status':health.get('status'),'buckets':health.get('buckets'),'active_events':(health.get('fact_events') or {}).get('active')},
  'registry_sha256':hashlib.sha256(raw).hexdigest(),
  'rolls':rolls,
},ensure_ascii=False))
'''
    result = subprocess.run(
        ["ssh", "-i", str(key), f"root@{host}", "python3", "-"],
        input=remote,
        text=True,
        encoding="utf-8",
        errors="strict",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=45,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip()[:500])
    receipt = json.loads(result.stdout)
    if receipt.get("health", {}).get("status") != "ok" or not str(receipt.get("container", "")).startswith("running "):
        raise RuntimeError("Ombre runtime is not healthy")
    return receipt


def _independent(nodes: list[Node]) -> list[Node]:
    selected: list[Node] = []
    for node in sorted(nodes, key=lambda item: (item.kind != "event", item.node_id)):
        source_ids = set(node.source_ids)
        if source_ids and any(source_ids.intersection(other.source_ids) for other in selected):
            continue
        selected.append(node)
    return selected


def _usable_term(term: str) -> bool:
    value = " ".join(str(term or "").split())
    if not value or value in ANCHOR_STOP_WORDS or value in GENERIC_RECALL_ONLY:
        return False
    if value in GLOBAL_STOP_ENTITIES or value in PERSON_RECALL_ONLY:
        return False
    if re.fullmatch(r"[\u4e00-\u9fff]+", value):
        return 3 <= len(value) <= 24
    return 4 <= len(value) <= 40 and bool(re.search(r"[A-Za-z]", value))


def _node_preview(node: Node) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "kind": node.kind,
        "date": node.date,
        "title": node.title,
        "excerpt": " ".join(node.text.split())[:220],
        "source_message_ids": list(node.source_ids),
    }


def _roll_member_ids(rolls: list[dict[str, Any]]) -> set[str]:
    return {
        str(value)
        for roll in rolls
        for field in ("linked_event_ids", "linked_scene_ids")
        for value in (roll.get(field) or [])
        if str(value)
    }


def build_unassigned_inventory(nodes: list[Node], rolls: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {node.node_id: node for node in nodes}
    assigned_ids = _roll_member_ids(rolls)
    unassigned_nodes = [node for node in nodes if node.node_id not in assigned_ids]
    eligible: list[Node] = []
    excluded: list[dict[str, str]] = []
    for node in unassigned_nodes:
        action, reason = gate_decision(node, "topic:__unassigned_arc_discovery__")
        if action == "eligible":
            eligible.append(node)
        else:
            excluded.append({"node_id": node.node_id, "decision": action, "reason": reason})

    corpus_df = build_anchor_document_frequency(nodes)
    receipts = {node.node_id: extract_anchor_candidates(node, corpus_df) for node in eligible}
    explicit_groups: dict[str, list[Node]] = defaultdict(list)
    rare_groups: dict[str, list[Node]] = defaultdict(list)
    compound_groups: dict[tuple[str, str], list[Node]] = defaultdict(list)
    for node in eligible:
        receipt = receipts[node.node_id]
        explicit = {str(item.get("term") or "").strip() for item in receipt.get("explicit_work") or []}
        for term in explicit:
            if _usable_term(term):
                explicit_groups[term].append(node)
        for term in receipt.get("anchor_candidates") or []:
            term = str(term or "").strip()
            if not _usable_term(term) or term in explicit:
                continue
            if not 2 <= int(corpus_df.get(term, 0)) <= RARE_TERM_MAX_DF:
                continue
            occurrences = receipt.get("occurrences", {}).get(term) or []
            salient = any(item.get("origin") == "title" or int(item.get("count") or 0) >= 2 for item in occurrences)
            if salient:
                rare_groups[term].append(node)
        for pair in receipt.get("compound") or []:
            terms = sorted({str(value or "").strip() for value in pair.get("terms") or []})
            if len(terms) != 2 or not all(_usable_term(term) for term in terms):
                continue
            if terms[0].lower() in terms[1].lower() or terms[1].lower() in terms[0].lower():
                continue
            compound_groups[(terms[0], terms[1])].append(node)

    candidates: list[dict[str, Any]] = []
    claimed_support_sets: set[tuple[str, ...]] = set()
    for term, grouped in sorted(explicit_groups.items()):
        independent = _independent(grouped)
        if len(independent) < 2:
            continue
        support = tuple(node.node_id for node in independent)
        claimed_support_sets.add(support)
        candidates.append({
            "candidate_kind": "explicit_work",
            "suggested_arc_key": f"work:{term.lower()}",
            "anchor_terms": [term],
            "supporting_materials": [_node_preview(node) for node in independent],
            "support_count": len(independent),
            "decision": "review_required",
        })

    for term, grouped in sorted(rare_groups.items()):
        independent = _independent(grouped)
        support = tuple(node.node_id for node in independent)
        if len(independent) < 2 or support in claimed_support_sets:
            continue
        candidates.append({
            "candidate_kind": "rare_repeated_phrase",
            "suggested_arc_key": "",
            "anchor_terms": [term],
            "document_frequency": int(corpus_df.get(term, 0)),
            "supporting_materials": [_node_preview(node) for node in independent],
            "support_count": len(independent),
            "decision": "review_required",
        })

    for terms, grouped in sorted(compound_groups.items()):
        independent = _independent(grouped)
        support = tuple(node.node_id for node in independent)
        if len(independent) < 2 or support in claimed_support_sets:
            continue
        candidates.append({
            "candidate_kind": "repeated_compound",
            "suggested_arc_key": "",
            "anchor_terms": list(terms),
            "document_frequency": [int(corpus_df.get(term, 0)) for term in terms],
            "supporting_materials": [_node_preview(node) for node in independent],
            "support_count": len(independent),
            "decision": "review_required",
        })

    kind_priority = {"explicit_work": 0, "rare_repeated_phrase": 1, "repeated_compound": 2}
    candidates.sort(
        key=lambda item: (
            kind_priority[item["candidate_kind"]],
            -item["support_count"],
            tuple(item["anchor_terms"]),
        )
    )
    candidates = candidates[:CANDIDATE_CAP]

    coverage: list[dict[str, Any]] = []
    extension_candidates: list[dict[str, Any]] = []
    for roll in rolls:
        arc_key = str(roll.get("arc_key") or "")
        if not arc_key:
            continue
        label = arc_key.split(":", 1)[-1]
        direct = [node.node_id for node in nodes if label.lower() in node.searchable.lower()]
        coverage.append({
            "arc_key": arc_key,
            "registry_member_ids": sorted(
                str(value)
                for field in ("linked_event_ids", "linked_scene_ids")
                for value in (roll.get(field) or [])
            ),
            "direct_label_support_ids": sorted(direct),
            "unassigned_direct_label_support_ids": sorted(set(direct) - assigned_ids),
        })

        member_ids = {
            str(value)
            for field in ("linked_event_ids", "linked_scene_ids")
            for value in (roll.get(field) or [])
        }
        member_nodes = [by_id[node_id] for node_id in sorted(member_ids) if node_id in by_id]
        metadata_text = "\n".join(
            str(value)
            for field in (
                "title",
                "title_aliases",
                "primary_entities",
                "supporting_entities",
                "query_cues",
            )
            for value in ([roll.get(field)] if isinstance(roll.get(field), str) else (roll.get(field) or []))
        )
        seed_term_nodes: dict[str, set[str]] = defaultdict(set)
        explicit_seed_terms: set[str] = set()
        for member in member_nodes:
            member_receipt = extract_anchor_candidates(member, corpus_df)
            explicit_seed_terms.update(
                str(item.get("term") or "").strip()
                for item in member_receipt.get("explicit_work") or []
            )
            for term in member_receipt.get("anchor_candidates") or []:
                term = str(term or "").strip()
                if _usable_term(term) and int(corpus_df.get(term, 0)) <= 12:
                    seed_term_nodes[term].add(member.node_id)
        stable_seed_terms = {
            term
            for term, supporting_members in seed_term_nodes.items()
            if len(supporting_members) >= 2
            or term in explicit_seed_terms
            or term.lower() in metadata_text.lower()
        }
        stable_seed_terms.add(label)
        recalled: list[dict[str, Any]] = []
        for node in eligible:
            matched = sorted(
                term for term in stable_seed_terms
                if term and term.lower() in node.searchable.lower()
            )
            exact_work = label.lower() in node.searchable.lower() or bool(explicit_seed_terms.intersection(matched))
            if not exact_work and len(matched) < 2:
                continue
            source_overlap = sorted(
                set(node.source_ids).intersection(
                    source_id for member in member_nodes for source_id in member.source_ids
                )
            )
            recalled.append({
                **_node_preview(node),
                "matched_seed_terms": matched,
                "recall_reason": "exact_work_anchor" if exact_work else "multi_rare_seed_anchor",
                "source_overlap_with_existing_members": source_overlap,
                "decision": "defer_source_overlap" if source_overlap else "review_required",
            })
        if recalled:
            extension_candidates.append({
                "arc_key": arc_key,
                "narrative_id": str(roll.get("narrative_id") or ""),
                "seed_member_ids": sorted(member_ids),
                "stable_seed_terms": sorted(stable_seed_terms),
                "recalled_unassigned_materials": recalled,
            })

    stone = [
        item for item in candidates
        if any("stone" in term.lower() and "memory" in term.lower() for term in item["anchor_terms"])
    ]
    spy = next((item for item in coverage if item["arc_key"] == "work:间谍过家家"), None)
    spy_extension = next(
        (item for item in extension_candidates if item["arc_key"] == "work:间谍过家家"),
        None,
    )
    reviewed_searches: list[dict[str, Any]] = []
    for search_key, terms in REVIEWED_SEARCHES.items():
        matches = []
        for node in eligible:
            matched = [term for term in terms if term.lower() in node.searchable.lower()]
            if matched:
                matches.append({**_node_preview(node), "matched_terms": matched})
        reviewed_searches.append({
            "search_key": search_key,
            "terms": list(terms),
            "matches": matches[:30],
            "match_count": len(matches),
            "truncated": len(matches) > 30,
        })
    return {
        "counts": {
            "active_nodes": len(nodes),
            "assigned_unique_nodes": len(set(by_id).intersection(assigned_ids)),
            "unassigned_nodes": len(unassigned_nodes),
            "eligible_unassigned_nodes": len(eligible),
            "gated_unassigned_nodes": len(excluded),
            "candidate_groups": len(candidates),
        },
        "requested_checks": {
            "stone_memory_candidates": stone,
            "spy_family_coverage": spy,
            "spy_family_extension_candidates": spy_extension,
        },
        "candidates": candidates,
        "existing_arc_coverage": coverage,
        "existing_arc_extension_candidates": extension_candidates,
        "reviewed_anchor_searches": reviewed_searches,
        "gated_receipts": excluded,
    }


def render_markdown(result: dict[str, Any]) -> str:
    counts = result["counts"]
    lines = [
        "# Germany unassigned Event / Scene Arc candidate shadow",
        "",
        "- Read-only discovery. No admission, publication, canonical, Event ownership, or cursor write.",
        f"- Runtime: `{result['runtime_receipt']['git_head'][:8]}` / registry `{result['runtime_receipt']['registry_sha256']}` / `{result['runtime_receipt']['health']['status']}`.",
        f"- Inventory: `{counts['active_nodes']}` active nodes; `{counts['assigned_unique_nodes']}` already referenced; `{counts['unassigned_nodes']}` unassigned.",
        f"- Discovery pool: `{counts['eligible_unassigned_nodes']}` eligible; `{counts['gated_unassigned_nodes']}` gated; `{counts['candidate_groups']}` bounded candidate groups.",
        "",
        "## Candidate groups",
        "",
    ]
    for index, item in enumerate(result["candidates"], 1):
        lines.extend([
            f"### {index}. {' + '.join(item['anchor_terms'])}",
            "",
            f"- Kind: `{item['candidate_kind']}`; support: `{item['support_count']}`; decision: `{item['decision']}`.",
            f"- Suggested key: `{item['suggested_arc_key']}`" if item["suggested_arc_key"] else "- Suggested key: pending semantic review.",
        ])
        for member in item["supporting_materials"]:
            lines.append(
                f"- `{member['node_id']}` ({member['kind']}, {member['date']}): {member['title'] or member['excerpt']}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only all-unassigned Arc candidate discovery")
    parser.add_argument("--host", default="168.119.228.217")
    parser.add_argument("--key", type=Path, default=Path.home() / ".ssh" / "id_ed25519")
    parser.add_argument("--session-id", type=int, default=22)
    parser.add_argument(
        "--artifact-base",
        type=Path,
        default=ROOT / "artifacts" / "narrative-arc-preview" / "2026-08-29-germany-unassigned-arc-candidates-v1",
    )
    args = parser.parse_args()
    runtime = fetch_registry_receipt(args.host, args.key)
    snapshot = fetch_snapshot(args.host, args.key, args.session_id)
    nodes = build_nodes(snapshot)
    inventory = build_unassigned_inventory(nodes, runtime["rolls"])
    result = {
        "schema_version": "shadow-unassigned-arc-candidates-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "live_write": False,
        "runtime_receipt": {key: value for key, value in runtime.items() if key != "rolls"},
        "registry_roll_count": len(runtime["rolls"]),
        **inventory,
        "writes_performed": [],
    }
    json_path = args.artifact_base.with_suffix(".json")
    md_path = args.artifact_base.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    md_path.write_text(render_markdown(result), encoding="utf-8", newline="\n")
    print(json.dumps({"status": "completed", **result["counts"], "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
