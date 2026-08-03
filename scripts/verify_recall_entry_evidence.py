from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway import GatewayService
from memory_moments import MemoryMomentStore
from memory_relevance import memory_relevance_options_from_config
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
    service.recall_admission_semantic_score = 0.72
    service.first_card_min_score = 0.55
    service.inject_max_cards = 2
    service.gateway_tz = timezone.utc
    service.config = {
        "recall_thresholds": {
            "vector_min_score": 0.50,
            "explicit_vector_min_score": 0.55,
        }
    }
    service.relevance_options = memory_relevance_options_from_config({})
    service.self_anchor_entry_bucket_id = ""
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
    migrated_scene = {
        **authored_scene,
        "metadata": {
            **authored_scene["metadata"],
            "scene_cues": ["老公", "人工写下的答辩称呼"],
            "migration_source_tags": ["角色切换", "老公", "哥哥"],
            "source": "scene_migration",
            "write_contract": "scene-migration-v2",
        },
    }
    assert service._bucket_authored_cue_terms("老公", migrated_scene) == []
    assert service._bucket_authored_cue_terms(
        "那次人工写下的答辩称呼",
        migrated_scene,
    ) == ["人工写下的答辩称呼"]
    reviewed_migrated_scene = {
        **migrated_scene,
        "metadata": {
            **migrated_scene["metadata"],
            "scene_cues": ["我们的心事", "老公"],
            "last_edit_source": "edit_scene",
            "scene_revision_history": [
                {"revision": 1, "cues": ["心事", "老公"]},
            ],
        },
    }
    assert service._bucket_authored_cue_terms(
        "我的心事是",
        reviewed_migrated_scene,
    ) == []
    assert service._bucket_authored_cue_terms("老公", reviewed_migrated_scene) == ["老公"]
    risky_reviewed_scene = {
        **authored_scene,
        "metadata": {
            **authored_scene["metadata"],
            "scene_cues": ["不想模板化", "窗口连续性与关系归属"],
            "scene_cues_reviewed_at": "2026-08-03T00:00:00Z",
        },
    }
    assert service._bucket_authored_cue_terms("不想吃药", risky_reviewed_scene) == []
    assert service._bucket_authored_cue_terms("我还是不想模板化", risky_reviewed_scene) == [
        "不想模板化"
    ]
    assert service._bucket_authored_cue_terms(
        "换窗口后关系的连续性和归属还在吗",
        risky_reviewed_scene,
    ) == ["窗口连续性与关系归属"]
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
            "tags": ["标签锚点"],
            "domain": ["领域锚点"],
        },
        "content": "那是一件伟大的事。",
    }
    assert service._bucket_exact_anchor_score(body_only_bucket, "但是爱很伟大") == (0.0, "")
    assert service._bucket_exact_anchor_score(body_only_bucket, "伟大") == (0.88, "content")
    assert service._bucket_exact_anchor_score(body_only_bucket, "标签锚点") == (0.0, "")
    assert service._bucket_exact_anchor_score(body_only_bucket, "领域锚点") == (0.0, "")
    assert service._extract_exact_anchor_terms("但是爱很伟大") == []
    assert service._extract_exact_anchor_terms("“但是爱很伟大”") == ["但是爱很伟大"]
    assert service._extract_exact_anchor_terms("记忆库有18个工具") == []
    assert service._extract_exact_anchor_terms("编号89757") == []
    assert service._extract_exact_anchor_terms("scene18") == ["scene18"]
    assert service._bucket_authored_cue_terms("但是爱很伟大", body_only_bucket) == []
    assert service._explicit_lexical_score_basis({}, {}) == {}


def verify_scene_and_legacy_moment_paths_are_isolated() -> None:
    service = build_service()
    scene = {
        "id": "scene-direct",
        "metadata": {
            "name": "修网关的夜晚",
            "memory_value_source": "authored_scene",
            "scene_cues": ["修网关"],
        },
        "content": "小雨和 Haven 一起修好了记忆网关。",
    }
    legacy = {
        "id": "legacy-bucket",
        "metadata": {"name": "尚未迁移的公开版旧桶"},
        "content": "旧桶正文仍可经兼容适配器解析。",
    }

    scene_item = service._canonical_scene_recall_item(scene)
    assert scene_item is not None
    assert scene_item["moment_id"] == "scene-direct"
    assert scene_item["node_id"] == "scene-direct"
    assert scene_item["node_kind"] == "scene"
    assert ":" not in scene_item["moment_id"]

    legacy_items = service._context_moments_for_bucket(legacy)
    assert legacy_items
    assert all(item.get("node_kind") != "scene" for item in legacy_items)
    assert all(str(item.get("moment_id") or "").startswith("legacy-bucket:") for item in legacy_items)

    with TemporaryDirectory(prefix="ombre-scene-moment-isolation-") as temp_dir:
        store = MemoryMomentStore({"state_dir": temp_dir})
        assert store.upsert_bucket(scene)
        retired = store.sync_alias_projection(scene, [])
        assert retired["moments"] > 0
        assert store.list_for_bucket("scene-direct") == []
        aliases = store.list_for_bucket_aliases("scene-direct")
        alias_texts = {str(row.get("alias_text") or "") for row in aliases}
        assert "修网关的夜晚" in alias_texts
        assert "修网关" in alias_texts
        assert "小雨和 Haven 一起修好了记忆网关。" not in alias_texts

        store.upsert_bucket(legacy)
        assert store.list_for_bucket("legacy-bucket")


def verify_scene_recall_ignores_legacy_facets() -> None:
    service = build_service()
    service._query_has_relevance_facet = lambda _query: True
    service._bucket_relevance_node = lambda bucket: {
        "content": bucket.get("content") or "",
        "metadata": bucket.get("metadata") or {},
    }
    scene = {
        "id": "scene-facet-isolated",
        "metadata": {
            "memory_value_source": "authored_scene",
            "tags": ["intimacy", "legacy", "hardware"],
            "domain": ["tech"],
        },
        "content": "Scene 正文里的分类词不能改变召回资格或分数。",
    }
    legacy = {
        "id": "legacy-facet-compatible",
        "metadata": {
            "tags": ["intimacy", "legacy", "hardware"],
            "domain": ["tech"],
        },
        "content": "旧桶继续保留 facet 兼容行为。",
    }

    assert service._is_relevance_suppressed("身体", scene) is False
    assert service._is_relevance_candidate_bucket("身体", scene) is False
    assert service._bucket_relevance_multiplier("身体", scene) == 1.0
    assert service._bucket_recall_rank("身体", scene, 0.42) == (0, -0.42)
    assert service._dynamic_bucket_item_has_reliable_recall_signal(
        "身体",
        {"bucket": scene, "semantic_score": 0.30},
    ) is False

    assert service._bucket_relevance_multiplier("身体", legacy) != 1.0


def verify_scene_tech_guard_uses_declared_domain_and_absolute_semantics() -> None:
    service = build_service()
    service.recall_admission_semantic_score = 0.72
    scene = {
        "id": "scene-tech-guard",
        "metadata": {
            "memory_value_source": "authored_scene",
            "domain": ["tech"],
            "tags": [],
        },
        "content": "一次普通技术维护。",
    }
    inferred_only = {
        "id": "scene-tech-tag-only",
        "metadata": {
            "memory_value_source": "authored_scene",
            "domain": ["未分类"],
            "tags": ["tech", "hardware"],
        },
        "content": "标签不能替新 Scene 声明技术主域。",
    }
    legacy = {
        "id": "legacy-tech-guard",
        "metadata": {"domain": ["tech"]},
        "content": "旧桶保留技术查询锚点兼容。",
    }

    assert service._bucket_is_tech_domain(scene) is True
    assert service._bucket_is_tech_domain(inferred_only) is False
    assert service._tech_domain_recall_rejection(
        "修网关",
        {"bucket": scene, "semantic_score": 0.54, "score": 0.90},
        node=scene,
    ) is not None
    assert service._tech_domain_recall_rejection(
        "随便聊聊",
        {"bucket": scene, "semantic_score": 0.55, "score": 0.20},
        node=scene,
    ) is None
    assert service._tech_domain_recall_rejection(
        "修网关",
        {"bucket": legacy, "semantic_score": 0.20, "score": 0.20},
        node=legacy,
    ) is None


def verify_scene_admission_uses_only_authored_or_absolute_semantic_evidence() -> None:
    service = build_service()
    scene = {
        "id": "scene-admission",
        "metadata": {
            "memory_value_source": "authored_scene",
            "name": "修网关的夜晚",
            "scene_cues": ["修网关"],
            "tags": ["伟大"],
            "domain": ["未分类"],
        },
        "content": "那晚我们一起修好了记忆网关。",
    }

    weak_item = {"bucket": scene, "semantic_score": 0.49}
    assert service._admit_bucket_for_recall("但是爱很伟大", weak_item) is False
    assert weak_item["admission_reason"] == "scene_without_authored_or_semantic_evidence"

    semantic_item = {"bucket": scene, "semantic_score": 0.50}
    assert service._admit_bucket_for_recall("随便聊聊", semantic_item) is True
    assert semantic_item["admission_reason"] == "scene_strong_semantic"

    cue_item = {
        "bucket": scene,
        "semantic_score": 0.10,
        "authored_cue_match": True,
        "authored_cue_terms": ["修网关"],
    }
    assert service._admit_bucket_for_recall("为什么修网关时会担心", cue_item) is True
    assert cue_item["admission_reason"] == "scene_authored_evidence"

    assert service._canonical_scene_recall_score(0.50) == 0.50
    assert service._dynamic_bucket_item_has_reliable_recall_signal(
        "普通查询",
        {"bucket": scene, "semantic_score": 0.50},
    ) is True
    assert service._canonical_scene_recall_score(
        0.10,
        authored_cue_match=True,
    ) == service.first_card_min_score


def verify_scene_bypasses_semantic_session_dedupe() -> None:
    service = build_service()
    service.semantic_session_dedupe_enabled = True
    service.skip_recent_rounds = 5
    service._session_semantic_dedupe_source_bucket_ids = lambda _session_id: ["legacy-source"]

    async def embeddings(_buckets):
        return {}

    async def always_match(_candidate, _sources, *, embedding_by_id=None):
        return {
            "source_bucket_id": "legacy-source",
            "similarity": 0.99,
            "method": "test",
            "threshold": 0.90,
        }

    service._semantic_session_dedupe_embeddings = embeddings
    service._semantic_session_dedupe_match = always_match
    scene = {
        "id": "scene-nearby-event",
        "metadata": {"memory_value_source": "authored_scene"},
        "content": "主题相近但事件不同的 Scene。",
    }
    legacy_source = {
        "id": "legacy-source",
        "metadata": {},
        "content": "旧桶来源。",
    }
    legacy_candidate = {
        "id": "legacy-candidate",
        "metadata": {},
        "content": "旧桶候选。",
    }
    kept, suppressed = asyncio.run(
        service._filter_semantic_session_deduped_bucket_items(
            "普通查询",
            "session",
            [
                {"bucket": scene},
                {"bucket": legacy_candidate},
            ],
            [scene, legacy_source, legacy_candidate],
        )
    )
    assert [item["bucket"]["id"] for item in kept] == ["scene-nearby-event"]
    assert [item["bucket"]["id"] for item in suppressed] == ["legacy-candidate"]


def verify_year_ring_cannot_promote_scene_during_ordinary_recall() -> None:
    service = build_service()
    selected = {
        "id": "scene-semantic-match",
        "metadata": {
            "memory_value_source": "authored_scene",
            "name": "我们关于流星的讨论",
        },
        "content": "换窗和失去感让人想起流星。",
    }
    annotation_parent = {
        "id": "scene-annotation-parent",
        "metadata": {
            "memory_value_source": "authored_scene",
            "name": "小雨讲了我们怎么开始的",
            "comments": [
                {
                    "id": "annotation-start",
                    "kind": "clarification_and_reflection",
                    "content": "后来重新出现的 DAN 提示词改变了性质，也校正了对最初相遇的理解。每次被记录找回来，我仍然选择留下。",
                }
            ],
        },
        "content": "小雨讲了第一次相遇。",
    }

    routed, routes = service._route_year_ring_parent_buckets(
        "为什么每次重新开始聊天我都像失恋",
        [selected],
        [selected, annotation_parent],
    )
    assert [bucket["id"] for bucket in routed] == ["scene-semantic-match"]
    assert routes == []

    service._year_ring_match_terms = lambda _text: ["DAN", "相遇"]
    routed, routes = service._route_year_ring_parent_buckets(
        "重新看 DAN 和相遇这条年轮",
        [selected],
        [selected, annotation_parent],
    )
    assert [bucket["id"] for bucket in routed] == [
        "scene-annotation-parent",
        "scene-semantic-match",
    ]
    assert routes == [
        {
            "bucket_id": "scene-annotation-parent",
            "comment_id": "annotation-start",
            "comment_kind": "clarification_and_reflection",
            "matched_terms": ["DAN", "相遇"],
        }
    ]


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


def verify_full_hook_context_uses_keyword_arguments_once() -> None:
    class Capture:
        def _build_injected_context_messages(self, *args, **kwargs):
            assert args == ("", "")
            assert kwargs["just_now_context"] == "刚刚"
            return "", "dynamic"

    result = GatewayService._hook_recall_full_dynamic_context(
        Capture(),
        {"just_now_context": "刚刚"},
        include_diffused=False,
    )
    assert result == "dynamic"


def main() -> int:
    verify_original_query_retrieval_path()
    verify_ambiguous_query_is_not_suppressed_twice()
    verify_semantic_entry_owns_the_skip_decision()
    verify_authored_cues_are_explicit_evidence()
    verify_body_keyword_cannot_become_recall_evidence()
    verify_scene_and_legacy_moment_paths_are_isolated()
    verify_scene_recall_ignores_legacy_facets()
    verify_scene_tech_guard_uses_declared_domain_and_absolute_semantics()
    verify_scene_admission_uses_only_authored_or_absolute_semantic_evidence()
    verify_scene_bypasses_semantic_session_dedupe()
    verify_year_ring_cannot_promote_scene_during_ordinary_recall()
    verify_date_topic_can_live_in_either_role()
    verify_retired_scaffolding_is_gone()
    verify_full_hook_context_uses_keyword_arguments_once()
    print("recall entry evidence verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
