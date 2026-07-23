from __future__ import annotations

import re
from dataclasses import dataclass

from .direct_admission import DirectAdmissionSignals
from memory_layers import LAYER_SOURCE_RECORD, infer_bucket_layer
from recall_policy import RecallPolicy
from utils import (
    strip_display_temperature_sections,
    strip_followup_sections,
    strip_temperature_meaning_lines,
    strip_wikilinks,
)
from memory_relevance import extract_protected_phrases


EXPLICIT_BUCKET_ID_RE = re.compile(
    r"\[bucket_id:(?P<bracket>[^\]\s]+)\]"
    r"|(?:bucket_id|bucket id|bucket-id|记忆桶|桶id|桶ID)\s*[:=：]\s*(?P<plain>[A-Za-z0-9_.:-]+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DirectEvidenceContext:
    query_terms: tuple[str, ...] = ()
    category_term_keys: frozenset[str] = frozenset()
    category_overview: bool = False
    planner_lexical: bool = False
    word_map: bool = False
    word_map_category_seed: bool = False
    matched_query_terms_specific: bool = False
    identity_name: bool = False
    direct_detail_requested: bool = False
    tech_query_has_anchor: bool = False
    self_anchor_excluded: bool = False
    locatable_terms: tuple[str, ...] = ()


class MemoryDirectEvidenceBuilder:
    """Classify measured direct evidence and build admission receipts."""

    def __init__(
        self,
        *,
        recall_policy: RecallPolicy,
        high_confidence_keyword_score: float,
    ) -> None:
        self.recall_policy = recall_policy
        self.high_confidence_keyword_score = float(high_confidence_keyword_score)

    def bucket_signals(
        self,
        query: str,
        item: dict,
        *,
        context: DirectEvidenceContext,
    ) -> DirectAdmissionSignals:
        labels = self.bucket_evidence_labels(query, item, context=context)
        return self.signals(
            item,
            context=context,
            evidence_labels=labels,
            hard_evidence_labels=self.hard_evidence_labels(labels),
        )

    def moment_signals(
        self,
        item: dict,
        *,
        context: DirectEvidenceContext,
    ) -> DirectAdmissionSignals:
        return self.signals(item, context=context)

    def signals(
        self,
        item: dict,
        *,
        context: DirectEvidenceContext,
        evidence_labels: list[str] | tuple[str, ...] = (),
        hard_evidence_labels: list[str] | tuple[str, ...] = (),
    ) -> DirectAdmissionSignals:
        return DirectAdmissionSignals(
            planner_lexical=context.planner_lexical,
            word_map=context.word_map,
            entity_edge=self.entity_edge_direct_signal(item),
            identity_name=context.identity_name,
            direct_detail_requested=context.direct_detail_requested,
            tech_query_has_anchor=context.tech_query_has_anchor,
            self_anchor_excluded=context.self_anchor_excluded,
            evidence_labels=tuple(evidence_labels),
            hard_evidence_labels=tuple(hard_evidence_labels),
            locatable_terms=context.locatable_terms,
        )

    def bucket_evidence_labels(
        self,
        query: str,
        item: dict,
        *,
        context: DirectEvidenceContext,
    ) -> list[str]:
        if not isinstance(item, dict):
            return []
        bucket = item.get("bucket") if isinstance(item.get("bucket"), dict) else {}
        labels: list[str] = []
        title_anchor_terms = self.bucket_title_anchor_terms(
            query,
            bucket,
            context=context,
        )
        if title_anchor_terms:
            item["title_anchor_terms"] = title_anchor_terms
            labels.append("title_anchor")
        definition_literal_terms = self.definition_query_literal_terms(
            query,
            bucket,
            category_overview=context.category_overview,
        )
        if definition_literal_terms:
            item["definition_literal_terms"] = definition_literal_terms
            labels.append("definition_literal_span")
        if item.get("exact_anchor_match") or self.safe_float(
            item.get("exact_anchor_score"), 0.0
        ) > 0:
            labels.append("exact_anchor")
        protected_phrases = extract_protected_phrases(query)
        if protected_phrases and isinstance(bucket, dict):
            bucket_text_key = self.compact_lookup_key(self.bucket_evidence_text(bucket))
            if any(
                self.compact_lookup_key(phrase) in bucket_text_key
                for phrase in protected_phrases
            ):
                labels.append("protected_phrase")
        if context.planner_lexical:
            labels.append("entity_match")
        if item.get("explicit_relation_edge_match") or self.entity_edge_direct_signal(item):
            labels.append("entity_match")
        if context.identity_name:
            labels.append("identity_name_match")
        if self.source_record_explicit_match_reason(query, bucket):
            labels.append("source_record_exact")
        if (
            isinstance(bucket, dict)
            and self.recall_policy._short_taste_query_terms(query)
            and self.recall_policy.bucket_has_topic_evidence(query, bucket)
        ):
            labels.append("taste_evidence")
        if item.get("distinctive_anchor_match"):
            labels.append("distinctive_anchor")
        if item.get("category_overview_item"):
            labels.append("category_overview_item")
        if item.get("retrieval_alias_match"):
            labels.append("retrieval_alias")
        if item.get("semantic_rescue_direct_span"):
            labels.append("semantic_rescue_direct_span")
        if item.get("word_map_category_seed_terms") or context.word_map_category_seed:
            labels.append("category_seed")
        if (
            isinstance(bucket, dict)
            and self.safe_float(item.get("keyword_score"), 0.0) > 0
            and item.get("distinctive_anchor_match")
            and self.recall_policy.bucket_has_topic_evidence(query, bucket)
            and context.matched_query_terms_specific
        ):
            labels.append("keyword_match")
        if self.safe_float(item.get("semantic_score"), 0.0) > 0:
            labels.append("semantic_hit")
        if (
            isinstance(bucket, dict)
            and self.safe_float(item.get("semantic_score"), 0.0) >= 0.50
            and self.safe_float(item.get("keyword_score"), 0.0)
            >= max(0.80, self.high_confidence_keyword_score)
            and self.safe_float(item.get("score"), 0.0) >= 0.95
            and self.recall_policy.bucket_has_topic_evidence(query, bucket)
        ):
            labels.append("hybrid_topic_match")
        if not self.is_source_record_bucket(bucket):
            if (
                self.safe_float(item.get("semantic_score"), 0.0)
                >= self.recall_policy.semantic_threshold
            ):
                labels.append("strong_semantic")
            if (
                self.safe_float(item.get("rerank_score"), 0.0)
                >= self.recall_policy.rerank_threshold
            ):
                labels.append("strong_rerank")
        if (
            item.get("word_map_hint")
            or self.safe_float(item.get("word_map_score"), 0.0) > 0
            or item.get("entity_edge_match")
        ):
            labels.append("graph_related")
        return self.dedupe_labels(labels)

    def bucket_title_anchor_terms(
        self,
        query: str,
        bucket: dict,
        *,
        context: DirectEvidenceContext,
    ) -> list[str]:
        if not query or not isinstance(bucket, dict):
            return []
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        title = str(meta.get("name") or bucket.get("name") or "").strip()
        title_key = self.compact_lookup_key(title)
        if not title_key:
            return []
        output: list[str] = []
        for term in context.query_terms:
            key = self.compact_lookup_key(term)
            if not key or len(key) < 3 or key in context.category_term_keys:
                continue
            if key in title_key and term not in output:
                output.append(term)
        query_key = self.compact_lookup_key(query)
        for fragment in re.split(r"[与和及、/|：:—-]+", title):
            cleaned = fragment.strip()
            key = self.compact_lookup_key(cleaned)
            if (
                len(key) >= 4
                and key in query_key
                and key not in context.category_term_keys
                and cleaned not in output
            ):
                output.append(cleaned)
        return output

    def definition_query_literal_terms(
        self,
        query: str,
        bucket: dict,
        *,
        category_overview: bool,
    ) -> list[str]:
        if not query or not isinstance(bucket, dict) or category_overview:
            return []
        text = " ".join(str(query or "").split()).strip()
        match = re.search(
            r"^(?P<target>.+?)(?:(?:到底)?(?:是|叫)|指的(?:是)?)(?:什么|啥)",
            text,
            re.IGNORECASE,
        )
        if not match:
            return []
        target = str(match.group("target") or "").strip(
            " \t，。！？、,.!?:：;；~～"
        )
        prefixes = ("请问", "我想问", "想问", "你还记得", "还记得", "关于")
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if target.startswith(prefix):
                    target = target[len(prefix) :].strip()
                    changed = True
        target_key = self.compact_lookup_key(target)
        if not target_key or target_key in {"这件事", "那件事", "这个", "那个", "它"}:
            return []
        cjk_len = len(re.findall(r"[\u4e00-\u9fff]", target_key))
        if cjk_len < 4 and len(target_key) < 6:
            return []
        bucket_key = self.compact_lookup_key(self.bucket_evidence_text(bucket))
        return [target] if target_key in bucket_key else []

    def source_record_explicit_match_reason(self, query: str, bucket: dict) -> str:
        if not query or not isinstance(bucket, dict):
            return ""
        bucket_id = str(bucket.get("id") or "")
        explicit_ids = {
            str(match.group("bracket") or match.group("plain") or "").strip()
            for match in EXPLICIT_BUCKET_ID_RE.finditer(str(query or ""))
        }
        if bucket_id and bucket_id in explicit_ids:
            return "explicit_bucket_id"
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        title = str(meta.get("name") or bucket_id or "").strip()
        title_key = self.compact_lookup_key(title)
        query_key = self.compact_lookup_key(query)
        if title_key and (query_key == title_key or title_key in query_key):
            return "explicit_bucket_title"
        for term in self.recall_policy.specific_query_terms(query):
            term_key = self.compact_lookup_key(term)
            if not term_key or len(term_key) < 2:
                continue
            if term_key == title_key or (len(term_key) >= 3 and term_key in title_key):
                return "explicit_bucket_title"
        return ""

    @staticmethod
    def bucket_evidence_text(bucket: dict) -> str:
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        return " ".join(
            [
                str(meta.get("name") or bucket.get("id") or ""),
                str(meta.get("annotation_summary") or meta.get("summary") or ""),
                " ".join(str(tag) for tag in meta.get("tags", []) or []),
                " ".join(str(item) for item in meta.get("domain", []) or []),
                MemoryDirectEvidenceBuilder.rendered_bucket_content(bucket),
            ]
        )

    @staticmethod
    def rendered_bucket_content(bucket: dict) -> str:
        text = strip_wikilinks(str(bucket.get("content") or ""))
        text = strip_display_temperature_sections(text)
        text = strip_followup_sections(text)
        text = strip_temperature_meaning_lines(text).strip()
        if "### moment" in text:
            body, moment_tail = text.split("### moment", 1)
            body = body.strip()
            rest = "### moment" + moment_tail
            moment_line = (
                rest.split("\n", 1)[-1].split("\n")[0].strip()
                if "\n" in rest
                else ""
            )
            if body and moment_line:
                first_sentence = re.split(r"[。！？!?]", body, maxsplit=1)[0].strip()
                if (
                    first_sentence
                    and len(first_sentence) >= 8
                    and (first_sentence in moment_line or moment_line in first_sentence)
                ):
                    body = body[len(first_sentence) :].lstrip("。！？!?\n ")
                    text = (body + "\n\n" + rest).strip()
        return text

    @staticmethod
    def hard_evidence_labels(labels: list[str]) -> list[str]:
        hard = {
            "raw_transcript_exact",
            "protected_phrase",
            "same_day_metadata",
            "exact_anchor",
            "entity_match",
            "keyword_match",
            "distinctive_anchor",
            "category_overview_item",
            "identity_name_match",
            "source_record_exact",
            "taste_evidence",
            "title_anchor",
            "definition_literal_span",
            "semantic_rescue_direct_span",
            "hybrid_topic_match",
            "strong_semantic",
            "strong_rerank",
        }
        return [label for label in labels or [] if label in hard]

    @staticmethod
    def entity_edge_direct_signal(item: dict) -> bool:
        if not isinstance(item, dict) or not item.get("entity_edge_match"):
            return False
        relation = str(item.get("entity_edge_relation") or "")
        if relation not in {
            "likes",
            "dislikes",
            "prefers",
            "boundary",
            "participates_in",
            "shared_anchor",
        }:
            return False
        return MemoryDirectEvidenceBuilder.safe_float(
            item.get("entity_edge_score"), 0.0
        ) >= 0.72

    @staticmethod
    def is_source_record_bucket(bucket: dict) -> bool:
        return infer_bucket_layer(bucket) == LAYER_SOURCE_RECORD

    @staticmethod
    def dedupe_labels(labels: list[str]) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for label in labels or []:
            text = str(label or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            output.append(text)
        return output

    @staticmethod
    def compact_lookup_key(value: object) -> str:
        return re.sub(
            r"[^0-9a-z\u4e00-\u9fff]+",
            "",
            str(value or "").strip().lower(),
        )

    @staticmethod
    def safe_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)
