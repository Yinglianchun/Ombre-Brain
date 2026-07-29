from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway import GatewayService


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


def main() -> int:
    verify_search_query_keeps_sentence_residue()
    verify_keyword_evidence_coverage()
    print("recall entry evidence verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
