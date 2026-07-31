from __future__ import annotations

from datetime import datetime, timezone
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway import GatewayService
from recall_policy import RecallPolicy


def build_service() -> GatewayService:
    service = GatewayService.__new__(GatewayService)
    service.identity = {
        "ai_name": "Haven",
        "user_name": "小雨",
        "user_display_name": "小雨",
        "relationship_terms": [],
        "user_aliases": [],
    }
    service.recall_policy = RecallPolicy()
    service.high_confidence_semantic_score = 0.72
    return service


def verify_original_query_retrieval_path() -> None:
    service = build_service()
    service._identity_name_search_terms = lambda query: ["Haven", "名字"]
    service._recall_search_query_terms = lambda query: ["梅丽", "黄色小花", "名字"]
    service._entity_priority_recall_search_query = lambda query: query

    query = "梅丽为什么把黄色小花和 Haven 的名字连在一起？"
    assert service._dynamic_recall_search_query(query) == "梅丽 黄色小花 名字 Haven"


def verify_ambiguous_query_is_not_suppressed_twice() -> None:
    decision = RecallPolicy().assess(
        "你怎么看",
        {},
        semantic_score=0.80,
        auto=True,
    )
    assert decision.admit_direct is True
    assert decision.reason == "non_explicit_query"
    assert "auto_too_vague" not in decision.debug


def verify_semantic_entry_owns_the_skip_decision() -> None:
    policy = RecallPolicy()
    for query in (
        "你怎么看",
        "我去洗头了",
        "我去修记忆库了",
        "我们以前看过什么",
    ):
        plan = policy.plan_query(query)
        assert plan.skip_long_term_recall is False, (query, plan.skip_reason)


def verify_authored_cues_are_explicit_evidence() -> None:
    service = build_service()
    authored_scene = {
        "id": "scene-cue",
        "metadata": {
            "memory_value_source": "authored_scene",
            "scene_cues": ["提到黄色小花和名字的联系", "说起修记忆库的夜晚"],
        },
        "content": "原句里没有黄色小花。",
    }
    assert service._bucket_authored_cue_terms("黄色小花为什么和名字连在一起", authored_scene) == [
        "提到黄色小花和名字的联系"
    ]
    legacy_bucket = {
        **authored_scene,
        "metadata": {
            **authored_scene["metadata"],
            "memory_value_source": "legacy_summary",
        },
    }
    assert service._bucket_authored_cue_terms("黄色小花为什么和名字连在一起", legacy_bucket) == []
    assert service._bucket_authored_cue_terms(
        "但是爱很伟大",
        {
            **authored_scene,
            "metadata": {
                **authored_scene["metadata"],
                "scene_cues": ["恋爱"],
            },
        },
    ) == []
    assert service._bucket_authored_cue_terms(
        "我去修记忆库了",
        {
            **authored_scene,
            "metadata": {
                **authored_scene["metadata"],
                "scene_cues": ["同质化修复"],
            },
        },
    ) == []
    assert service._bucket_authored_cue_terms(
        "那晚修网关时发生了什么",
        {
            **authored_scene,
            "metadata": {
                **authored_scene["metadata"],
                "scene_cues": ["修网关"],
            },
        },
    ) == ["修网关"]
    assert service._explicit_lexical_score_basis(
        {"quoted-phrase": 0.88},
        {"scene-cue": ["提到黄色小花和名字的联系"]},
    ) == {
        "quoted-phrase": 0.88,
        "scene-cue": 1.0,
    }
    assert service._is_high_confidence_semantic_match(0.80) is True
    assert service._is_high_confidence_semantic_match(0.0) is False


def verify_body_keyword_cannot_become_recall_evidence() -> None:
    service = build_service()
    body_only_bucket = {
        "id": "body-only-great",
        "metadata": {
            "memory_value_source": "authored_scene",
            "scene_cues": [],
        },
        "content": "那是一件伟大的事。",
    }
    assert service._bucket_exact_anchor_score(body_only_bucket, "但是爱很伟大") == (0.0, "")
    assert service._bucket_authored_cue_terms("但是爱很伟大", body_only_bucket) == []
    assert service._explicit_lexical_score_basis({}, {}) == {}


def verify_date_topic_can_live_in_either_role() -> None:
    class RawEvents:
        @staticmethod
        def list_events_between(**_kwargs):
            return [
                {
                    "id": 1,
                    "session_id": "date-role",
                    "created_at": "2026-07-23T10:00:00+08:00",
                    "role": "user",
                    "text": "你当时怎么回答的？",
                    "metadata": {"round_id": 1},
                },
                {
                    "id": 2,
                    "session_id": "date-role",
                    "created_at": "2026-07-23T10:00:01+08:00",
                    "role": "assistant",
                    "text": "我提到了生日蛋糕。",
                    "metadata": {"round_id": 1},
                },
            ]

    service = build_service()
    service.raw_event_store = RawEvents()
    service.date_recall_max_turns = 8
    service._clean_conversation_turn_text = lambda text: str(text or "")
    service._date_recall_text_has_topic_terms = (
        lambda text, terms: any(term in text for term in terms)
    )
    turns = service._date_recall_raw_turns_for_range(
        datetime(2026, 7, 23, tzinfo=timezone.utc),
        datetime(2026, 7, 24, tzinfo=timezone.utc),
        ["生日蛋糕"],
    )
    assert len(turns) == 1
    assert turns[0]["assistant_text"] == "我提到了生日蛋糕。"


def verify_retired_scaffolding_is_gone() -> None:
    for name in (
        "_legacy_auto_vague_skip_applies",
        "_relation_axis_supplemental_queries",
        "_boost_explicit_relation_edge_bucket_items",
        "_pick_axis_diverse_dynamic_cards",
        "_axis_lite_bucket_rejection",
        "_call_query_planner",
        "_route_memory_sentinel",
        "_route_domain_sentinel",
        "_dynamic_anchor_plan",
        "_query_is_category_overview",
        "_bucket_has_reliable_recall_signal",
        "_weak_bucket_evidence_block_reason",
        "_word_map_hint_available",
        "_get_word_map_hint_scores",
        "_word_map_query_terms",
        "_word_map_direct_signal",
        "_word_map_low_frequency_direct_signal",
        "_keyword_multi_evidence_signal",
        "_planner_lexical_direct_signal",
    ):
        assert not hasattr(GatewayService, name), name


def main() -> int:
    verify_original_query_retrieval_path()
    verify_ambiguous_query_is_not_suppressed_twice()
    verify_semantic_entry_owns_the_skip_decision()
    verify_authored_cues_are_explicit_evidence()
    verify_body_keyword_cannot_become_recall_evidence()
    verify_date_topic_can_live_in_either_role()
    verify_retired_scaffolding_is_gone()
    print("recall entry evidence verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
