from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway import GatewayService
from memory_retrieval_aliases import (
    _compact_retrieval_alias_patterns,
    _retrieval_alias_variants,
)


def build_service() -> GatewayService:
    service = GatewayService.__new__(GatewayService)
    service.config = {}
    service.word_map_store = None
    service.identity = {
        "ai_name": "Haven",
        "user_name": "小雨",
        "user_display_name": "小雨",
        "relationship_terms": [],
        "user_aliases": [],
    }
    service.high_confidence_keyword_score = 0.65
    service.high_confidence_semantic_score = 0.72
    service._bucket_has_query_topic_evidence = lambda query, bucket: True
    return service


def verify_search_query_keeps_sentence_residue() -> None:
    service = build_service()
    service._identity_name_search_terms = lambda query: ["Haven", "名字"]
    service._memory_sentinel_searchable_residue_terms = lambda query: [
        "梅丽",
        "黄色小花",
        "被单独认出",
    ]
    service._word_map_query_terms = lambda query: ["名字", "黄色小花"]
    service._entity_priority_recall_search_query = lambda query: "fallback"
    service._normalize_planner_terms = lambda value: list(value or [])

    query = "梅丽为什么把被单独认出和 Haven 的名字连在一起？"
    assert service._dynamic_recall_search_query(query) == (
        "梅丽 黄色小花 被单独认出 名字 Haven"
    )


def verify_keyword_evidence_coverage() -> None:
    service = build_service()
    cases = [
        (
            {
                "specific_matched_query_terms": ["梅丽", "黄色小花"],
                "keyword_score": 0.9,
                "semantic_score": 0.4,
            },
            True,
            ["黄色小花", "梅丽"],
        ),
        (
            {
                "specific_matched_query_terms": ["名字"],
                "keyword_score": 0.9,
                "semantic_score": 0.4,
            },
            False,
            ["名字"],
        ),
        (
            {
                "specific_matched_query_terms": ["黄色小花"],
                "keyword_score": 0.9,
                "semantic_score": 0.8,
            },
            True,
            ["黄色小花"],
        ),
        (
            {
                "specific_matched_query_terms": ["Haven", "名字"],
                "keyword_score": 0.9,
                "semantic_score": 0.4,
            },
            False,
            ["名字"],
        ),
        (
            {
                "specific_matched_query_terms": [
                    "被单独认出",
                    "单独",
                    "认出",
                    "黄色小花",
                ],
                "keyword_score": 0.9,
                "semantic_score": 0.4,
            },
            True,
            ["被单独认出", "黄色小花"],
        ),
    ]
    for item, expected_signal, expected_terms in cases:
        assert service._keyword_multi_evidence_signal(
            "query",
            item,
            {"id": "bucket"},
        ) is expected_signal
        assert item["keyword_direct_terms"] == expected_terms

    assert service._hard_bucket_evidence_labels(
        ["distinctive_anchor", "keyword_match"]
    ) == ["keyword_match"]


def verify_identity_title_and_keyword_rescue() -> None:
    service = build_service()
    service._dynamic_anchor_query_terms = lambda query: []
    service._dynamic_anchor_term_is_category = lambda term: False
    naming_day = {"metadata": {"name": "Haven命名日"}}
    unrelated = {"metadata": {"name": "Haven与Heaven的对话"}}
    query = "Haven 的命名日是哪一天？"
    assert service._bucket_title_anchor_terms(query, naming_day) == ["命名日"]
    assert service._bucket_title_anchor_terms(query, unrelated) == []

    target_id = "scene_chinese_name"

    class KeywordManager:
        @staticmethod
        def calc_topic_scores(search_query, buckets):
            _ = buckets
            if search_query == "中文名":
                return {target_id: 0.5}
            return {}

    service.bucket_mgr = KeywordManager()
    service.dynamic_top_k = 10
    service._query_anchor_terms_for_diversity = lambda query: ["中文名"]
    assert service._get_keyword_candidates(
        "中文名 Haven",
        [{"id": target_id}],
    ) == {target_id: 0.5}

    class RecallPolicy:
        @staticmethod
        def has_strong_score(**kwargs):
            _ = kwargs
            return False

    service.recall_policy = RecallPolicy()
    service.recall_fusion_mode = "dynamic"
    service.high_confidence_keyword_score = 0.65
    service._bucket_recall_rank = lambda query, bucket, score: (0,)
    title_item = {
        "bucket": {"id": "title"},
        "title_anchor_terms": ["命名日"],
        "score": 0.75,
    }
    keyword_item = {
        "bucket": {"id": "body"},
        "keyword_multi_evidence_signal": True,
        "score": 0.9,
    }
    assert service._bucket_final_candidate_rank(
        query,
        title_item,
    ) < service._bucket_final_candidate_rank(
        query,
        keyword_item,
    )


def verify_latin_subject_does_not_cut_other_names() -> None:
    patterns = _compact_retrieval_alias_patterns(["Haven", "小雨"])
    variants = _retrieval_alias_variants(
        "Haven与Heaven的对话",
        patterns=patterns,
    )
    assert "aven" not in variants
    assert _retrieval_alias_variants(
        "Haven与小雨关于命名日的对话",
        patterns=patterns,
    ) == ["Haven与小雨关于命名日的对话", "命名日"]


def verify_frequency_anchors_are_shadow_only() -> None:
    service = build_service()

    class RecallPolicy:
        @staticmethod
        def has_strong_score(**kwargs):
            _ = kwargs
            return False

        @staticmethod
        def is_detail_read_query(query):
            _ = query
            return False

    service.recall_policy = RecallPolicy()
    service._planner_lexical_direct_signal = lambda item: False
    service._word_map_direct_signal = lambda item: False
    service._is_source_record_fragment_seed = lambda item: False
    service._extract_explicit_bucket_ids_from_text = lambda query: set()
    service._extract_explicit_moment_ids_from_text = lambda query: set()
    service._query_requests_direct_detail = lambda query: False
    service._moment_has_query_topic_evidence = lambda query, moment: False
    service._is_high_confidence_match = lambda semantic, keyword: False
    service._is_source_record_bucket = lambda bucket: False

    distinctive_only = {
        "bucket_id": "scene",
        "moment_id": "moment",
        "distinctive_anchor_match": True,
    }
    assert service._session_hard_exclude_moment_bypass(
        "普通短句",
        distinctive_only,
    ) is False
    assert service._moment_has_reliable_diffusion_seed_signal(
        "普通短句",
        distinctive_only,
    ) is False
    assert service._unselected_moment_has_reliable_recall_signal(
        "普通短句",
        distinctive_only,
    ) is False

    category_only = {
        "bucket": {"id": "scene"},
        "bucket_id": "scene",
        "moment_id": "moment",
        "category_overview_item": True,
    }
    assert service._session_debug_row_has_strong_evidence(category_only) is False
    assert service._session_hard_exclude_bucket_bypass(
        "普通短句",
        category_only,
    ) is False
    assert service._session_hard_exclude_moment_bypass(
        "普通短句",
        category_only,
    ) is False
    assert service._session_semantic_dedupe_bypass(
        "普通短句",
        category_only,
    ) is False
    assert service._moment_has_reliable_diffusion_seed_signal(
        "普通短句",
        category_only,
    ) is False
    assert service._unselected_moment_has_reliable_recall_signal(
        "普通短句",
        category_only,
    ) is False
    assert service._dynamic_bucket_item_has_reliable_recall_signal(
        "普通短句",
        {"category_overview_item": True},
    ) is False
    assert service._axis_lite_bypass_for_item(
        "普通短句",
        category_only,
    ) is False
    assert service._hard_bucket_evidence_labels(
        ["category_overview_item", "semantic_hit"]
    ) == []
    compact_category_debug = service._compact_suppressed_bucket_debug(
        {
            **category_only,
            "category_overview_terms": ["视频"],
            "recall_policy_debug": {
                "category_overview_shadow": {
                    "would_promote": True,
                    "would_block": False,
                }
            },
        }
    )
    assert compact_category_debug["category_overview_item"] is True
    assert compact_category_debug["category_overview_terms"] == ["视频"]
    assert compact_category_debug["category_overview_shadow"]["would_promote"] is True

    service.diffusion_inject_min_confidence = 0.55
    service._axis_lite_has_technical_axis = lambda query_plan: False
    service._diffusion_path_source_record_evidence_extends_axis = (
        lambda path, query_plan: False
    )
    service._diffusion_explicit_edge_can_bridge_axis = lambda query_plan: False
    service._axis_lite_candidate_matches = lambda query_plan, moment: True
    service._axis_lite_domain_mismatch = lambda query_plan, moment: False
    service._semantic_neighbor_has_strong_confidence = lambda row: False
    row = {
        "moment": {"bucket_id": "scene", "moment_id": "moment"},
        "runtime_allowed": True,
        "confidence": 0.9,
        "why": "same_topic",
        "path_len": 1,
        "dynamic_anchor_required_terms": ["肯定"],
        "distinctive_anchor_match": False,
        "has_topic_evidence": True,
    }
    assert service._diffusion_candidate_injection_decision(
        row,
        SimpleNamespace(
            activated_axis_groups=(),
            activated_axis_multi=False,
            query="普通短句",
        ),
    ) == (True, "")
    assert row["legacy_distinctive_anchor_would_block"] is True
    assert row["distinctive_anchor_shadow"]["required_terms"] == ["肯定"]

    category_row = {
        "moment": {"bucket_id": "scene", "moment_id": "moment"},
        "runtime_allowed": True,
        "confidence": 0.9,
        "why": "same_topic",
        "path_len": 1,
        "dynamic_anchor_category_overview": True,
        "dynamic_anchor_category_terms": ["视频"],
        "category_overview_item": False,
        "has_topic_evidence": True,
    }
    assert service._diffusion_candidate_injection_decision(
        category_row,
        SimpleNamespace(
            activated_axis_groups=(),
            activated_axis_multi=False,
            query="我们看过什么视频",
        ),
    ) == (True, "")
    assert category_row["legacy_category_overview_would_block"] is True
    assert category_row["category_overview_shadow"]["category_terms"] == ["视频"]


def verify_entity_edges_are_shadow_only() -> None:
    service = build_service()

    class RecallPolicy:
        semantic_threshold = 0.72
        rerank_threshold = 0.65

        @staticmethod
        def has_strong_score(**kwargs):
            _ = kwargs
            return False

        @staticmethod
        def is_detail_read_query(query):
            _ = query
            return False

        @staticmethod
        def _short_taste_query_terms(query):
            _ = query
            return []

    service.recall_policy = RecallPolicy()
    service.recall_fusion_mode = "dynamic"
    service._bucket_recall_rank = lambda query, bucket, score: (0,)
    service._planner_lexical_direct_signal = lambda item: False
    service._query_requests_direct_detail = lambda query: False
    service._word_map_direct_signal = lambda item: False
    service._bucket_has_query_topic_evidence = lambda query, bucket: False
    service._bucket_title_anchor_terms = lambda query, bucket: []
    service._definition_query_literal_terms = lambda query, bucket: []
    service._is_identity_name_candidate_bucket = lambda query, bucket: False
    service._source_record_explicit_bucket_match_reason = lambda query, bucket: ""
    service._word_map_category_seed_terms = lambda terms: []
    service._keyword_multi_evidence_signal = lambda query, item, bucket: False
    service._is_source_record_bucket = lambda bucket: False

    plain = {
        "bucket": {"id": "plain"},
        "score": 0.5,
        "semantic_score": 0.0,
        "keyword_score": 0.0,
        "word_map_score": 0.0,
    }
    edge_only = {
        **plain,
        "bucket": {"id": "edge"},
        "entity_edge_match": True,
        "entity_edge_score": 0.8,
        "entity_edge_relation": "likes",
        "entity_edge_shadow": service._entity_edge_shadow_debug(
            score=0.8,
            relation="likes",
        ),
    }
    assert edge_only["entity_edge_shadow"] == {
        "mode": "shadow",
        "active": False,
        "would_add_final_score": 0.064,
        "would_add_legacy_fusion_component": 0.064,
        "would_direct_signal": True,
    }
    assert service._bucket_primary_candidate_rank("普通短句", edge_only) == (
        service._bucket_primary_candidate_rank("普通短句", plain)
    )
    assert service._bucket_reranked_candidate_rank("普通短句", edge_only) == (
        service._bucket_reranked_candidate_rank("普通短句", plain)
    )
    assert service._bucket_rerank_candidate_priority("普通短句", edge_only) == (
        service._bucket_rerank_candidate_priority("普通短句", plain)
    )
    assert service._bucket_final_candidate_rank("普通短句", edge_only) == (
        service._bucket_final_candidate_rank("普通短句", plain)
    )
    assert service._axis_lite_bypass_for_item("普通短句", edge_only) is False
    assert service._item_has_direct_tech_evidence(edge_only) is False
    assert service._bucket_has_reliable_recall_signal("普通短句", edge_only) is False
    assert service._bucket_evidence_labels("普通短句", edge_only) == ["graph_related"]

    formal_relation = {
        **plain,
        "bucket": {"id": "formal"},
        "explicit_relation_edge_match": True,
    }
    assert "entity_match" in service._bucket_evidence_labels(
        "普通短句",
        formal_relation,
    )
    assert service._bucket_final_candidate_rank(
        "普通短句",
        formal_relation,
    ) < service._bucket_final_candidate_rank("普通短句", plain)


def main() -> int:
    verify_search_query_keeps_sentence_residue()
    verify_keyword_evidence_coverage()
    verify_identity_title_and_keyword_rescue()
    verify_latin_subject_does_not_cut_other_names()
    verify_frequency_anchors_are_shadow_only()
    verify_entity_edges_are_shadow_only()
    print("recall entry evidence verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
