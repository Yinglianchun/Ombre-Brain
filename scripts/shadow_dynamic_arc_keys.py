from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from scripts.shadow_arc_candidates_germany import (
    ARC_BATCH_CAP,
    ARC_MEMBER_CAP,
    GLOBAL_STOP_ENTITIES,
    PERSON_RECALL_ONLY,
    RECALL_TOP_K_PER_SEED,
    Node,
    build_anchor_document_frequency,
    extract_anchor_candidates,
    gate_decision,
    node_paragraphs,
)


ARC_KEY_KINDS = frozenset({"work", "project", "milestone", "topic"})


@dataclass(frozen=True)
class ReviewedAnchor:
    key_kind: str
    label: str
    terms: tuple[str, ...]
    basis: str


def stable_arc_key(kind: str, label: str) -> str:
    safe_kind = str(kind or "").strip().lower()
    if safe_kind not in ARC_KEY_KINDS:
        raise ValueError("arc key kind must be work, project, milestone, or topic")
    compact_label = " ".join(str(label or "").split()).strip()
    if not compact_label:
        raise ValueError("arc key label is required")
    slug = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "-", compact_label.lower()).strip("-")
    if not slug:
        raise ValueError("arc key label has no stable characters")
    return f"{safe_kind}:{slug}"


def _eligible_for_key_proposal(node: Node) -> tuple[bool, str]:
    decision, reason = gate_decision(node, "topic:__dynamic_arc_key_proposal__")
    return decision == "eligible", reason


def _same_paragraph_origins(node: Node, terms: tuple[str, ...]) -> list[str]:
    lowered = tuple(str(term or "").strip().lower() for term in terms if str(term or "").strip())
    if not lowered:
        return []
    return [
        paragraph["origin"]
        for paragraph in node_paragraphs(node)
        if all(term in paragraph["text"].lower() for term in lowered)
    ]


def _reviewed_anchor_match(node: Node, spec: ReviewedAnchor) -> dict[str, Any] | None:
    if not isinstance(spec, ReviewedAnchor):
        raise ValueError("reviewed anchors must be ReviewedAnchor values")
    if spec.key_kind not in ARC_KEY_KINDS:
        raise ValueError("reviewed anchor has an invalid key kind")
    if spec.basis not in {"compound", "repeated_phrase"}:
        raise ValueError("reviewed anchor basis must be compound or repeated_phrase")
    terms = tuple(dict.fromkeys(" ".join(str(term or "").split()) for term in spec.terms))
    terms = tuple(term for term in terms if term)
    if not terms:
        raise ValueError("reviewed anchor needs terms")
    if all(term in GLOBAL_STOP_ENTITIES or term in PERSON_RECALL_ONLY for term in terms):
        raise ValueError("global names or person-only anchors cannot mint an Arc key")

    origins = _same_paragraph_origins(node, terms)
    if not origins:
        return None
    if spec.basis == "repeated_phrase" and len(terms) != 1:
        raise ValueError("repeated_phrase reviewed anchors require exactly one phrase")
    return {
        "kind": f"reviewed_{spec.basis}",
        "key_kind": spec.key_kind,
        "label": " ".join(str(spec.label or "").split()),
        "terms": list(terms),
        "origins": origins,
    }


def _explicit_work_matches(node: Node, anchor_receipt: dict[str, Any]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for item in anchor_receipt.get("explicit_work") or []:
        term = " ".join(str(item.get("term") or "").split())
        if not term or term in GLOBAL_STOP_ENTITIES or term in PERSON_RECALL_ONLY:
            continue
        matches.append(
            {
                "kind": "explicit_work",
                "key_kind": "work",
                "label": term,
                "terms": [term],
                "origins": [
                    occurrence["origin"]
                    for occurrence in (anchor_receipt.get("occurrences") or {}).get(term, [])
                ],
            }
        )
    return matches


def _direct_anchor_match(node: Node, anchor: dict[str, Any]) -> list[str]:
    return _same_paragraph_origins(node, tuple(anchor["terms"]))


def _independent_materials(nodes: list[Node]) -> list[Node]:
    selected: list[Node] = []
    for node in sorted(nodes, key=lambda item: (item.kind != "event", item.node_id)):
        source_ids = set(node.source_ids)
        if source_ids and any(source_ids.intersection(other.source_ids) for other in selected):
            continue
        selected.append(node)
    return selected


def propose_dynamic_arc_keys(
    nodes: list[Node],
    *,
    reviewed_anchors: tuple[ReviewedAnchor, ...] = (),
) -> dict[str, Any]:
    ordered_nodes = sorted(nodes, key=lambda item: item.node_id)
    corpus_df = build_anchor_document_frequency(ordered_nodes)
    anchor_receipts = {
        node.node_id: extract_anchor_candidates(node, corpus_df) for node in ordered_nodes
    }
    eligible_nodes: list[Node] = []
    rejected_seeds: list[dict[str, str]] = []
    for node in ordered_nodes:
        eligible, reason = _eligible_for_key_proposal(node)
        if eligible:
            eligible_nodes.append(node)
        else:
            rejected_seeds.append({"seed_id": node.node_id, "reason": reason})

    receipts: list[dict[str, Any]] = []
    for seed in eligible_nodes:
        anchors = _explicit_work_matches(seed, anchor_receipts[seed.node_id])
        for spec in reviewed_anchors:
            matched = _reviewed_anchor_match(seed, spec)
            if matched:
                anchors.append(matched)
        anchors.sort(key=lambda item: (item["kind"], item["label"], item["terms"]))

        seen_keys: set[str] = set()
        for anchor in anchors:
            arc_key = stable_arc_key(anchor["key_kind"], anchor["label"])
            if arc_key in seen_keys:
                continue
            seen_keys.add(arc_key)

            direct_nodes = [
                node for node in eligible_nodes if _direct_anchor_match(node, anchor)
            ]
            independent = _independent_materials(direct_nodes)
            status = (
                "deferred_member_cap"
                if len(independent) > ARC_MEMBER_CAP
                else "proposed_for_review"
                if len(independent) >= 2
                else "insufficient_independent_materials"
            )
            receipts.append(
                {
                    "seed_id": seed.node_id,
                    "arc_key": arc_key,
                    "anchor_kind": anchor["kind"],
                    "anchor_label": anchor["label"],
                    "anchor_terms": anchor["terms"],
                    "seed_origins": anchor["origins"],
                    "direct_support_ids": [node.node_id for node in direct_nodes],
                    "independent_support_ids": [node.node_id for node in independent],
                    "independent_material_count": len(independent),
                    "one_hop": True,
                    "embedding_used": False,
                    "date_used": False,
                    "status": status,
                    "admission_decision": "not_run",
                    "publish_attempted": False,
                }
            )
        if len(seen_keys) > RECALL_TOP_K_PER_SEED:
            raise ValueError("dynamic Arc key proposals exceeded the per-seed cap")

    receipts.sort(key=lambda item: (item["arc_key"], item["seed_id"]))
    proposed_receipts = [item for item in receipts if item["status"] == "proposed_for_review"]
    proposals: list[dict[str, Any]] = []
    for arc_key in sorted({item["arc_key"] for item in proposed_receipts}):
        grouped = [item for item in proposed_receipts if item["arc_key"] == arc_key]
        supporting_ids = sorted(
            {source_id for item in grouped for source_id in item["independent_support_ids"]}
        )
        proposals.append(
            {
                "arc_key": arc_key,
                "seed_ids": sorted(item["seed_id"] for item in grouped),
                "supporting_material_ids": supporting_ids,
                "status": "proposed_for_review",
                "admission_decision": "not_run",
                "publication_status": "not_published",
            }
        )
    if len(proposals) > ARC_BATCH_CAP:
        raise ValueError("dynamic Arc key proposals exceeded the batch cap")

    return {
        "schema_version": "shadow-dynamic-arc-key-proposals-v1",
        "proposal_contract": {
            "minimum_independent_materials": 2,
            "one_hop": True,
            "rare_single_can_mint": False,
            "person_can_mint": False,
            "embedding_can_mint": False,
            "date_can_mint": False,
            "automatic_admission": False,
            "automatic_publication": False,
        },
        "proposals": proposals,
        "proposal_receipts": receipts,
        "rejected_seed_receipts": rejected_seeds,
        "anchor_receipts": [anchor_receipts[node_id] for node_id in sorted(anchor_receipts)],
        "writes_performed": [],
    }
