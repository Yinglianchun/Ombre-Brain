from __future__ import annotations

import sys
from pathlib import Path


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


def main() -> int:
    verify_search_query_keeps_sentence_residue()
    verify_keyword_evidence_coverage()
    verify_identity_title_and_keyword_rescue()
    verify_latin_subject_does_not_cut_other_names()
    print("recall entry evidence verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
