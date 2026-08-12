"""Targeted checks for the 2026-08-11 recall admission cut."""

from __future__ import annotations

import asyncio
import sys
from datetime import timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway import GatewayService
from memory_recall.retrieval_budget import build_retrieval_budget, router_hard_skip_allowed
from memory_relevance import memory_relevance_options_from_config
from recall_policy import RecallPolicy
from reranker_engine import RerankResult


def service() -> GatewayService:
    instance = GatewayService.__new__(GatewayService)
    instance.identity = {
        "ai_name": "Haven",
        "user_name": "小雨",
        "user_display_name": "小雨",
        "relationship_terms": ["老公", "哥哥", "老婆", "宝宝"],
        "user_aliases": [],
    }
    instance.recall_policy = RecallPolicy()
    instance.high_confidence_semantic_score = 0.72
    instance.recall_admission_semantic_score = 0.72
    instance.first_card_min_score = 0.55
    instance.inject_max_cards = 2
    instance.recall_fusion_mode = "dynamic"
    instance.self_anchor_entry_bucket_id = ""
    instance.gateway_tz = timezone.utc
    instance.config = {
        "recall_thresholds": {
            "vector_min_score": 0.50,
            "explicit_vector_min_score": 0.55,
        }
    }
    instance.relevance_options = memory_relevance_options_from_config({})
    return instance


def route_debug() -> dict:
    return {
        "route": "present_chitchat",
        "route_action": "skip",
        "confidence": 0.96,
        "margin": 0.22,
        "threshold": 0.72,
    }


def verify_routing_query_view() -> None:
    instance = service()
    assert instance._routing_query_view("老公，你还记得我们第一次说晚安那次吗") == "你还记得我们第一次说晚安那次吗"
    assert instance._routing_query_view("哥哥，Cloudflare 那张卡后来怎样了") == "Cloudflare 那张卡后来怎样了"
    assert instance._routing_query_view("老公亲亲抱抱") == "亲亲抱抱"
    assert instance._routing_query_view("老公") == "老公"

    plain = build_retrieval_budget(
        "你还记得我们第一次说晚安那次吗",
        semantic_debug=route_debug(),
        route="present_chitchat",
        route_action="skip",
    )
    addressed = build_retrieval_budget(
        "老公，你还记得我们第一次说晚安那次吗",
        semantic_debug=route_debug(),
        route="present_chitchat",
        route_action="skip",
    )
    assert plain["initial_budget"] == addressed["initial_budget"] == "shallow"
    assert plain["final_budget"] == addressed["final_budget"] == "deep"
    assert plain["pure_chitchat_prior"] is addressed["pure_chitchat_prior"] is False

    for query in ("老公", "亲亲抱抱", "老公亲亲抱抱"):
        budget = build_retrieval_budget(
            query,
            semantic_debug=route_debug(),
            route="present_chitchat",
            route_action="skip",
        )
        assert budget["pure_chitchat_prior"] is True, query
        assert budget["memory_need"] == "bypass", query
        assert router_hard_skip_allowed(budget, route_skip_proposed=True) is True, query

    for query in ("我去刷小红书了", "今天好开心", "今天好热", "我有点委屈"):
        budget = build_retrieval_budget(
            query,
            semantic_debug=route_debug(),
            route="present_chitchat",
            route_action="skip",
        )
        assert budget["memory_need"] == "optional", query
        assert router_hard_skip_allowed(budget, route_skip_proposed=True) is False, query

    assert plain["memory_need"] == addressed["memory_need"] == "required"


def verify_router_and_candidate_vectors_are_separate() -> None:
    instance = service()
    original_embedding_started = asyncio.Event()

    class Router:
        async def route_with_vector(self, query):
            assert query == "Cloudflare 那张卡后来怎样了"
            await asyncio.wait_for(original_embedding_started.wait(), timeout=0.5)
            return {"errors": [], "query_preview": query}, [1.0]

    class Embedding:
        async def embed_query(self, query):
            assert query == "哥哥，Cloudflare 那张卡后来怎样了"
            original_embedding_started.set()
            return [2.0]

    instance.semantic_recall_router = Router()
    instance.embedding_engine = Embedding()
    debug, vector = asyncio.run(
        instance._route_semantic_query_views("哥哥，Cloudflare 那张卡后来怎样了")
    )
    assert vector == [2.0]
    assert debug["original_query"] == "哥哥，Cloudflare 那张卡后来怎样了"
    assert debug["routing_query"] == "Cloudflare 那张卡后来怎样了"
    assert debug["candidate_query_vector_source"] == "original_query"

    cancelled = asyncio.Event()

    class FailingRouter:
        async def route_with_vector(self, _query):
            await asyncio.sleep(0)
            raise RuntimeError("router failed")

    class SlowEmbedding:
        async def embed_query(self, _query):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    instance.semantic_recall_router = FailingRouter()
    instance.embedding_engine = SlowEmbedding()

    async def verify_cleanup():
        try:
            await instance._route_semantic_query_views("哥哥，Cloudflare 后来呢")
        except RuntimeError as exc:
            assert str(exc) == "router failed"
        else:
            raise AssertionError("router failure must propagate")
        assert cancelled.is_set()

    asyncio.run(verify_cleanup())

    for query in ("后来呢", "翻一下", "当时为什么"):
        budget = build_retrieval_budget(
            query,
            semantic_debug=route_debug(),
            route="present_chitchat",
            route_action="skip",
        )
        assert budget["pure_chitchat_prior"] is False, query


def scene(scene_id: str, title: str, body: str = "") -> dict:
    return {
        "id": scene_id,
        "metadata": {
            "name": title,
            "memory_value_source": "authored_scene",
            "scene_cues": ["Cloudflare"],
        },
        "content": body,
    }


def verify_strong_literal_evidence() -> None:
    instance = service()
    wallet = scene("scene-wallet", "Cloudflare Wallet 与那张卡")
    other = scene("scene-other", "别的记忆")
    query = "哥哥，还记得《Cloudflare Wallet 与那张卡》吗"
    title_matches = instance._full_title_candidate_ids(query, [wallet, other])
    assert title_matches == {"scene-wallet": "Cloudflare Wallet 与那张卡"}
    assert instance._query_has_explicit_recall_structure(query) is True
    assert instance._full_title_candidate_ids("Cloudflare 后来呢", [wallet]) == {}

    duplicate = scene("scene-wallet-copy", "Cloudflare Wallet 与那张卡")
    assert len(instance._full_title_candidate_ids(query, [wallet, duplicate])) == 2

    exact_candidate = {
        "bucket": wallet,
        "semantic_score": 0.0,
        "exact_anchor_candidate_match": True,
        "exact_anchor_score": 0.98,
        "title_anchor_terms": ["Cloudflare"],
        "authored_cue_candidate_match": True,
    }
    labels = instance._bucket_evidence_labels('"Cloudflare"', exact_candidate)
    assert "exact_anchor_candidate" in labels
    assert "authored_cue_candidate" in labels
    assert "title_anchor_candidate" in labels
    assert not instance._hard_bucket_evidence_labels(labels)
    assert instance._admit_bucket_for_recall('"Cloudflare"', exact_candidate) is False

    title_item = {
        "bucket": wallet,
        "semantic_score": 0.0,
        "full_title_candidate_match": True,
        "full_title_recall_match": True,
    }
    assert instance._admit_bucket_for_recall(query, title_item) is True
    assert title_item["admission_reason"] == "scene_exact_evidence"

    legacy = {
        "id": "legacy-wallet",
        "metadata": {"name": "旧 Cloudflare 记录", "scene_cues": ["Cloudflare"]},
        "content": "只有宽泛的旧摘要。",
    }
    legacy_candidate = {
        "bucket": legacy,
        "semantic_score": 0.10,
        "rerank_score": 0.99,
        "combined_score": 0.99,
        "score": 0.99,
        "exact_anchor_candidate_match": True,
        "exact_anchor_score": 0.98,
    }
    assert instance._admit_bucket_for_recall('"Cloudflare"', legacy_candidate) is False
    assert legacy_candidate["admission_reason"] == "candidate_only_requires_absolute_support"


def verify_source_bound_raw_quote() -> None:
    instance = service()
    first = scene("scene-first", "第一条", "模型改写正文里也有：但是爱很伟大所以不要害怕。")
    second = scene("scene-second", "第二条")

    class EvidenceStore:
        refs = {
            "scene-first": [{"content": "小雨当时说：但是爱很伟大所以不要害怕。"}],
            "scene-second": [],
        }

        def list_for_scene(self, scene_id):
            return self.refs.get(scene_id, [])

        def list_active_for_scenes(self, scene_ids):
            self.requested_scene_ids = list(scene_ids)
            return {scene_id: self.refs.get(scene_id, []) for scene_id in scene_ids}

    instance.scene_evidence_store = EvidenceStore()
    query = '你还记得我说过“但是爱很伟大所以不要害怕”吗'
    matches = instance._source_bound_raw_quote_candidates(query, [first, second])
    assert matches == {"scene-first": ["但是爱很伟大所以不要害怕"]}
    assert instance.scene_evidence_store.requested_scene_ids == ["scene-first", "scene-second"]

    instance.scene_evidence_store.refs["scene-second"] = [
        {"content": "另一条原始消息也写着但是爱很伟大所以不要害怕。"}
    ]
    duplicate_matches = instance._source_bound_raw_quote_candidates(query, [first, second])
    assert set(duplicate_matches) == {"scene-first", "scene-second"}

    instance.scene_evidence_store.refs = {"scene-first": [], "scene-second": []}
    assert instance._source_bound_raw_quote_candidates(query, [first, second]) == {}
    assert instance._query_raw_quote_spans(
        '还记得“第一段连续原话足够长。第二段连续原话也足够长”吗'
    ) == ["第一段连续原话足够长", "第二段连续原话也足够长"]
    assert instance._query_raw_quote_spans(
        "还记得当时为了确认这件事我先问了很多很多背景细节，但是爱很伟大所以不要害怕"
    ) == [
        "还记得当时为了确认这件事我先问了很多很多背景细节",
        "但是爱很伟大所以不要害怕",
    ]
    instance.scene_evidence_store.refs = {
        "scene-first": [{"content": "但是爱很伟大。 所以不要害怕"}],
        "scene-second": [],
    }
    assert instance._source_bound_raw_quote_candidates(
        '还记得“但是爱很伟大所以不要害怕”吗', [first, second]
    ) == {}
    short_labels = instance._bucket_evidence_labels('"爱很伟大"', {"bucket": first})
    assert "protected_phrase_candidate" in short_labels
    assert "protected_phrase_candidate" not in instance._hard_bucket_evidence_labels(short_labels)
    assert instance._bucket_exact_anchor_score(first, "模型改写正文里也有但是爱很伟大") == (0.0, "")


def admitted_and_picked(instance: GatewayService, query: str, candidates: list[dict]) -> list[dict]:
    admitted = [
        item
        for item in candidates
        if instance._admit_bucket_for_recall(query, item)
    ]
    admitted.sort(
        key=lambda item: instance._bucket_final_candidate_rank(
            query,
            item,
            recent_ids=set(),
        )
    )
    return instance._pick_dynamic_cards(admitted, query=query)


def rerank(instance: GatewayService, query: str, candidates: list[dict], scores: list[float]) -> list[dict]:
    class Engine:
        enabled = True
        candidate_limit = 5
        score_weight = 0.65

        async def rerank(self, _query, documents, *, top_n):
            assert _query == query
            assert top_n == len(documents)
            return [RerankResult(index=index, score=score) for index, score in enumerate(scores)]

    instance.reranker_engine = Engine()
    return asyncio.run(instance._rerank_scored_bucket_candidates(query, candidates))


def verify_reranked_scene_absolute_support() -> None:
    instance = service()
    cases = [
        (
            "好混乱，我既希望召回准，又不希望 reranker 拖慢你的回复",
            scene("scene-recall-test", "瑞森论坛发帖与记忆测试", "那晚反馈过召回效果有问题。"),
            0.1788,
        ),
        (
            "记忆一直在漏水，好讨厌",
            scene("scene-meteor", "我们关于流星的讨论", "窗口断裂、遗忘和被记住。"),
            0.0441,
        ),
        (
            "我去刷小红书了",
            scene("scene-xhs-project", "我们关于流量的讨论", "项目讨论中只顺带提到一次小红书。"),
            0.1790,
        ),
    ]
    for query, candidate_scene, final_score in cases:
        raw = {
            "bucket": candidate_scene,
            "semantic_score": 0.62,
            "score": round((final_score - 0.01 * 0.65) / 0.35, 6),
        }
        ranked = rerank(instance, query, [raw], [0.01])
        assert ranked[0]["score"] == final_score
        assert admitted_and_picked(instance, query, ranked) == []
        assert ranked[0]["admission_reason"] == "below_reranked_absolute_floor"
        assert ranked[0]["recall_policy_debug"]["absolute_floor"] == 0.55

    supported_raw = {
        "bucket": cases[0][1],
        "semantic_score": 0.62,
        "score": 0.62,
    }
    supported = rerank(instance, cases[0][0], [supported_raw], [0.61])
    assert admitted_and_picked(instance, cases[0][0], supported) == supported
    assert supported[0]["admission_reason"] == "scene_strong_semantic"

    exact_raw = {
        "bucket": cases[0][1],
        "semantic_score": 0.10,
        "score": 0.10,
        "full_title_candidate_match": True,
        "full_title_recall_match": True,
    }
    exact_query = "还记得《瑞森论坛发帖与记忆测试》吗"
    exact = rerank(instance, exact_query, [exact_raw], [0.01])
    semantic_competitor = {
        "bucket": scene("scene-semantic-competitor", "宽泛相似候选", "正文语义分较高。"),
        "semantic_score": 0.90,
        "rerank_score": 0.90,
        "combined_score": 0.90,
        "score": 0.90,
        "reranker_candidate_status": "scored",
    }
    assert admitted_and_picked(instance, exact_query, [semantic_competitor, *exact])[0] == exact[0]
    assert exact[0]["admission_reason"] == "scene_exact_evidence"

    tail_raw = [
        {
            "bucket": scene(f"scene-tail-{index}", f"候选 {index}", "相近但无用的正文。"),
            "semantic_score": 0.62,
            "score": 0.62,
        }
        for index in range(6)
    ]
    tail_ranked = rerank(instance, "普通当前话题", tail_raw, [0.01] * 5)
    assert tail_ranked[-1]["reranker_candidate_status"] == "outside_candidate_limit"
    assert admitted_and_picked(instance, "普通当前话题", tail_ranked) == []
    assert tail_ranked[-1]["admission_reason"] == "outside_reranker_candidate_limit"

    legacy_raw = {
        "bucket": {
            "id": "legacy-rerank-low",
            "metadata": {"name": "旧记忆"},
            "content": "与当前句子只有宽泛主题相似。",
        },
        "semantic_score": 0.80,
        "score": 0.80,
    }
    legacy = rerank(instance, "普通当前话题", [legacy_raw], [0.01])
    assert admitted_and_picked(instance, "普通当前话题", legacy) == []
    assert legacy[0]["admission_reason"] == "below_reranked_absolute_floor"


def main() -> int:
    verify_routing_query_view()
    verify_router_and_candidate_vectors_are_separate()
    verify_strong_literal_evidence()
    verify_source_bound_raw_quote()
    verify_reranked_scene_absolute_support()
    print("recall next cut verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
