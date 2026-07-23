from __future__ import annotations

from typing import Any, Callable

from memory_layers import (
    CONTEXT_ONLY_SECTIONS,
    moment_layer_debug,
    moment_runtime_gate_debug,
)
from recall_policy import RecallPolicy
from utils import strip_temperature_meaning_lines, strip_wikilinks


MOMENT_SECTION_LABELS = {
    "body": "body",
    "moment": "moment",
    "fact": "fact",
    "original": "original",
    "evidence_context": "evidence_context",
    "context": "context",
    "reflection": "reflection",
    "feeling": "feeling",
    "followup": "followup",
    "affect_anchor": "affect_anchor",
    "favorite_reason": "favorite_reason",
    "comment": "year_ring",
}
MOMENT_TEMPERATURE_SECTIONS = CONTEXT_ONLY_SECTIONS - {"followup", "followup_log"}

RecallWhyBuilder = Callable[[dict[str, Any], str, str], dict[str, Any]]


class MemoryDiffusionFormatter:
    """Render diffused memory and its debug receipt without owning Gateway."""

    def __init__(
        self,
        *,
        recall_policy: RecallPolicy,
        recall_why_builder: RecallWhyBuilder,
    ) -> None:
        self.recall_policy = recall_policy
        self.recall_why_builder = recall_why_builder

    def format_candidate_debug(
        self,
        row: dict[str, Any],
        *,
        moment_map: dict[str, dict],
        explicit_lookup: bool,
        query: str,
    ) -> dict[str, Any]:
        payload = self.format_moment_debug(
            row["moment"],
            note=str(row.get("note") or ""),
            path=row.get("path"),
            moment_map=moment_map,
            explicit_lookup=explicit_lookup,
            query=query,
            chain_bundle=bool(row.get("chain_bundle")),
        )
        payload.update(
            {
                "why": str(row.get("why") or ""),
                "confidence": self._safe_float(row.get("confidence"), 0.0),
                "confidence_source": str(row.get("confidence_source") or ""),
                "confidence_defaulted": bool(row.get("confidence_defaulted")),
                "activation": self._safe_float(row.get("activation"), 0.0),
                "source": str(row.get("source") or ""),
                "injected": bool(row.get("injected")),
                "suppression_reason": str(row.get("suppression_reason") or ""),
                "has_topic_evidence": bool(row.get("has_topic_evidence")),
                "topic_evidence_terms": list(row.get("topic_evidence_terms") or []),
                "strong_topic_evidence": bool(row.get("strong_topic_evidence")),
                "distinctive_anchor_match": bool(row.get("distinctive_anchor_match")),
                "distinctive_anchor_terms": list(row.get("distinctive_anchor_terms") or []),
                "distinctive_anchor_missing_terms": list(
                    row.get("distinctive_anchor_missing_terms") or []
                ),
                "category_overview_item": bool(row.get("category_overview_item")),
                "category_overview_terms": list(row.get("category_overview_terms") or []),
                "reading_note": (
                    row.get("reading_note")
                    if isinstance(row.get("reading_note"), dict)
                    else {}
                ),
                "diffusion_trace": self.format_candidate_trace(row, moment_map),
            }
        )
        payload["recall_why"] = self.recall_why_builder(
            payload,
            "injected_diffused" if payload["injected"] else "suppressed_diffused",
            "diffusion_candidate",
        )
        return payload

    def format_candidate_trace(
        self,
        row: dict[str, Any],
        moment_map: dict[str, dict],
    ) -> dict[str, Any]:
        path = row.get("path")
        path_nodes = tuple(str(node_id) for node_id in (getattr(path, "nodes", ()) or ()))
        path_steps = tuple(getattr(path, "steps", ()) or ())
        moment = row.get("moment") if isinstance(row.get("moment"), dict) else {}
        target_id = str(row.get("moment_id") or moment.get("moment_id") or "")
        target_node_id = path_nodes[-1] if path_nodes else target_id
        seed_node_id = path_nodes[0] if path_nodes else ""
        gate_allowed = bool(row.get("gate_allowed", row.get("injectable")))
        gate_reason = str(row.get("gate_reason") or "")
        suppression_reason = str(row.get("suppression_reason") or "")
        injected = bool(row.get("injected"))
        if injected:
            final_status = "injected"
        elif suppression_reason or not gate_allowed:
            final_status = "suppressed"
        else:
            final_status = "eligible"

        return {
            "source": str(row.get("source") or ""),
            "why": str(row.get("why") or ""),
            "confidence": self._safe_float(row.get("confidence"), 0.0),
            "confidence_source": str(row.get("confidence_source") or ""),
            "confidence_defaulted": bool(row.get("confidence_defaulted")),
            "activation": self._safe_float(row.get("activation"), 0.0),
            "path_len": int(row.get("path_len") or 0),
            "path_step_count": len(path_steps),
            "path_trace": self.path_summary(path, moment_map) if path is not None else "",
            "seed": (
                self.format_path_node_debug(seed_node_id, moment_map.get(seed_node_id))
                if seed_node_id
                else {}
            ),
            "target": self.format_path_node_debug(
                target_node_id,
                moment_map.get(target_node_id) or moment,
            ),
            "gate": {
                "allowed": gate_allowed,
                "reason": gate_reason,
                "runtime_allowed": bool(row.get("runtime_allowed")),
                "has_topic_evidence": bool(row.get("has_topic_evidence")),
                "topic_evidence_terms": list(row.get("topic_evidence_terms") or []),
                "strong_topic_evidence": bool(row.get("strong_topic_evidence")),
            },
            "final": {
                "status": final_status,
                "injected": injected,
                "suppression_reason": suppression_reason,
            },
        }

    def format_moment_line(
        self,
        moment: dict,
        *,
        max_chars: int,
        note: str,
        path: Any | None = None,
        moment_map: dict[str, dict] | None = None,
        chain_bundle: bool = False,
    ) -> str:
        if chain_bundle and path is not None and len(getattr(path, "steps", ()) or ()) >= 2:
            return self.format_chain_bundle(
                moment,
                max_chars=max_chars,
                note=note,
                path=path,
                moment_map=moment_map or {},
            )
        summary = self.moment_summary(
            moment,
            max_chars=max_chars,
            path=path,
            moment_map=moment_map or {},
        )
        context = self.temperature_context(
            moment,
            path=path,
            moment_map=moment_map or {},
        )
        context_part = f"; context: {context}" if context else ""
        suffix = f" ({note})" if note else ""
        return (
            f"- [bucket_id:{moment.get('bucket_id') or ''}] "
            f"[moment_id:{moment.get('moment_id') or ''}] "
            f"{summary}{context_part}{suffix}"
        )

    def format_chain_bundle(
        self,
        moment: dict,
        *,
        max_chars: int,
        note: str,
        path: Any,
        moment_map: dict[str, dict],
    ) -> str:
        nodes = tuple(str(node_id) for node_id in (getattr(path, "nodes", ()) or ()))
        seed_id = nodes[0] if nodes else ""
        seed_label = self.node_label(moment_map.get(seed_id), seed_id)
        chain = self.path_summary(path, moment_map)
        target = self.moment_summary(
            moment,
            max_chars=max_chars,
            path=None,
            moment_map=moment_map,
        )
        temperature = self.temperature_context(
            moment,
            path=path,
            moment_map=moment_map,
        )
        temperature_part = f"; temperature: {temperature}" if temperature else ""
        suffix = f" ({note})" if note else ""
        return (
            f"- Chain Bundle: seed {seed_label}; chain: {chain}; "
            f"target: {target}{temperature_part}{suffix}"
        )

    def temperature_context(
        self,
        moment: dict,
        *,
        path: Any | None = None,
        moment_map: dict[str, dict] | None = None,
        max_items: int = 2,
        max_chars: int = 90,
    ) -> str:
        return self.format_temperature_context_items(
            self.temperature_context_items(
                moment,
                path=path,
                moment_map=moment_map,
                max_items=max_items,
                max_chars=max_chars,
            )
        )

    def temperature_context_items(
        self,
        moment: dict,
        *,
        path: Any | None = None,
        moment_map: dict[str, dict] | None = None,
        max_items: int = 2,
        max_chars: int = 90,
    ) -> list[dict[str, Any]]:
        moment_map = moment_map or {}
        bucket_id = str(moment.get("bucket_id") or "")
        if not bucket_id:
            return []
        contexts: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add_context(candidate: dict | None) -> None:
            if len(contexts) >= max_items or not isinstance(candidate, dict):
                return
            if str(candidate.get("bucket_id") or "") != bucket_id:
                return
            if candidate.get("section") not in MOMENT_TEMPERATURE_SECTIONS:
                return
            moment_id = str(candidate.get("moment_id") or "")
            if (
                not moment_id
                or moment_id == str(moment.get("moment_id") or "")
                or moment_id in seen
            ):
                return
            if not self.moment_text(candidate, max_chars):
                return
            seen.add(moment_id)
            section = str(candidate.get("section") or "")
            contexts.append(
                {
                    "bucket_id": bucket_id,
                    "bucket_name": self.moment_bucket_title(candidate),
                    "moment_id": moment_id,
                    "section": section,
                    "label": MOMENT_SECTION_LABELS.get(section, section or "moment"),
                    "text_preview": self.moment_text(candidate, max_chars),
                }
            )

        for node_id in getattr(path, "nodes", ()) or ():
            add_context(moment_map.get(str(node_id)))
        for candidate in sorted(
            moment_map.values(),
            key=lambda item: int(item.get("ordinal") or 0) if isinstance(item, dict) else 0,
        ):
            add_context(candidate)
            if len(contexts) >= max_items:
                break
        return contexts

    @staticmethod
    def format_temperature_context_items(items: list[dict[str, Any]]) -> str:
        return " / ".join(
            f"[{item.get('label') or item.get('section') or 'moment'}] "
            f"{item.get('text_preview') or ''}"
            for item in items
            if item.get("text_preview")
        )

    def moment_summary(
        self,
        moment: dict,
        *,
        max_chars: int,
        path: Any | None = None,
        moment_map: dict[str, dict],
    ) -> str:
        section = str(moment.get("section") or "")
        label = MOMENT_SECTION_LABELS.get(section, section or "moment")
        title = self.moment_bucket_title(moment) or str(moment.get("bucket_id") or "memory")
        status = self.moment_status_label(moment)
        parts = [f"{label} summary from {title}"]
        if status:
            parts.append(status)
        path_summary = self.path_summary(path, moment_map) if path is not None else ""
        if path_summary:
            parts.append(f"path {path_summary}")
        return self._clip_text("; ".join(parts), max_chars)

    @staticmethod
    def moment_status_label(moment: dict) -> str:
        meta = moment.get("metadata", {}) if isinstance(moment.get("metadata"), dict) else {}
        if meta.get("resolved") or meta.get("bucket_resolved"):
            return "resolved"
        if meta.get("digested") or meta.get("bucket_digested"):
            return "digested"
        if str(meta.get("type") or meta.get("bucket_type") or "").lower() == "archived":
            return "archived"
        return ""

    def path_summary(self, path: Any, moment_map: dict[str, dict]) -> str:
        steps = getattr(path, "steps", ()) or ()
        nodes = tuple(str(node_id) for node_id in (getattr(path, "nodes", ()) or ()))
        if not nodes:
            return ""
        labels = [self.node_label(moment_map.get(nodes[0]), nodes[0])]
        for step in steps:
            target_id = str(getattr(step, "target", "") or "")
            relation = str(getattr(step, "relation_type", "") or "relates_to")
            arrow = "<-" if getattr(step, "direction", "") == "incoming" else "->"
            labels.append(
                f"{arrow}{relation}-> {self.node_label(moment_map.get(target_id), target_id)}"
            )
        return self._clip_text(" ".join(labels), 140)

    def node_label(self, moment: dict | None, fallback_id: str) -> str:
        if isinstance(moment, dict):
            return self._clip_text(
                self.moment_bucket_title(moment)
                or str(moment.get("bucket_id") or fallback_id),
                48,
            )
        return self._clip_text(fallback_id, 48)

    @classmethod
    def moment_text(cls, moment: dict, max_chars: int = 220) -> str:
        text = strip_temperature_meaning_lines(str(moment.get("text") or ""))
        return cls._clip_text(" ".join(text.split()), max_chars)

    @staticmethod
    def moment_bucket_title(moment: dict) -> str:
        meta = moment.get("metadata", {}) if isinstance(moment.get("metadata"), dict) else {}
        title = str(meta.get("bucket_name") or "").strip()
        bucket_id = str(moment.get("bucket_id") or "")
        return "" if title == bucket_id else title

    def related_runtime_gate_payload(
        self,
        moment: dict,
        *,
        explicit_lookup: bool = False,
        query: str = "",
    ) -> dict[str, Any]:
        gate = moment_runtime_gate_debug(moment, explicit_lookup=explicit_lookup)
        query_plan = self.recall_policy.plan_query(query)
        topic_required = bool(query_plan.enforce_topic_evidence)
        has_topic_evidence = (
            self.recall_policy.moment_has_topic_evidence(query, moment)
            if topic_required and isinstance(moment, dict)
            else False
        )
        related_allowed = bool(gate["related_target"]["allowed"])
        related_reason = str(gate["related_target"]["reason"])
        if related_allowed and topic_required and not has_topic_evidence:
            related_allowed = False
            related_reason = "query_topic_evidence_missing"
        gate["topic_evidence"] = {
            "required": topic_required,
            "present": has_topic_evidence if topic_required else None,
        }
        gate["related_injection"] = {
            "allowed": related_allowed,
            "reason": related_reason,
        }
        gate["would_inject_related"] = related_allowed
        return gate

    def format_moment_debug(
        self,
        moment: dict,
        *,
        note: str = "",
        path: Any | None = None,
        moment_map: dict[str, dict] | None = None,
        explicit_lookup: bool = False,
        query: str = "",
        chain_bundle: bool = False,
    ) -> dict[str, Any]:
        moment_map = moment_map or {}
        payload = {
            "bucket_id": str(moment.get("bucket_id") or ""),
            "bucket_name": self.moment_bucket_title(moment),
            "moment_id": str(moment.get("moment_id") or ""),
            "section": moment.get("section"),
            "note": str(note or ""),
            "chain_bundle": bool(chain_bundle),
            "layer_debug": moment_layer_debug(moment, explicit_lookup=explicit_lookup),
            "runtime_gate": self.related_runtime_gate_payload(
                moment,
                explicit_lookup=explicit_lookup,
                query=query,
            ),
            "temperature_context": self.temperature_context_items(
                moment,
                path=path,
                moment_map=moment_map,
            ),
            "text_preview": self.moment_text(moment, 180),
        }
        if path is not None:
            payload["path"] = self.format_path_debug(path, moment_map)
        return payload

    def format_path_debug(self, path: Any, moment_map: dict[str, dict]) -> dict[str, Any]:
        nodes = tuple(str(node_id) for node_id in (getattr(path, "nodes", ()) or ()))
        steps = tuple(getattr(path, "steps", ()) or ())
        return {
            "trace": self.path_summary(path, moment_map),
            "score": self._safe_float(getattr(path, "score", 0.0), 0.0),
            "nodes": [
                self.format_path_node_debug(node_id, moment_map.get(node_id))
                for node_id in nodes
            ],
            "steps": [
                {
                    "source": str(getattr(step, "source", "") or ""),
                    "source_label": self.node_label(
                        moment_map.get(str(getattr(step, "source", "") or "")),
                        str(getattr(step, "source", "") or ""),
                    ),
                    "target": str(getattr(step, "target", "") or ""),
                    "target_label": self.node_label(
                        moment_map.get(str(getattr(step, "target", "") or "")),
                        str(getattr(step, "target", "") or ""),
                    ),
                    "relation_type": str(
                        getattr(step, "relation_type", "") or "relates_to"
                    ),
                    "confidence": self._safe_float(
                        getattr(step, "confidence", 0.0), 0.0
                    ),
                    "direction": str(getattr(step, "direction", "") or "outgoing"),
                    "reason": str(getattr(step, "reason", "") or ""),
                }
                for step in steps
            ],
        }

    def format_path_node_debug(
        self,
        moment_id: str,
        moment: dict | None,
    ) -> dict[str, Any]:
        if not isinstance(moment, dict):
            return {
                "moment_id": str(moment_id or ""),
                "bucket_id": "",
                "bucket_name": str(moment_id or ""),
                "section": "",
            }
        return {
            "moment_id": str(moment.get("moment_id") or moment_id or ""),
            "bucket_id": str(moment.get("bucket_id") or ""),
            "bucket_name": self.moment_bucket_title(moment),
            "section": str(moment.get("section") or ""),
        }

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _clip_text(text: str, max_chars: int) -> str:
        compact = " ".join(strip_wikilinks(str(text or "")).split())
        if len(compact) <= max_chars:
            return compact
        return compact[:max_chars].rstrip() + "..."
