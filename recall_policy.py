from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
from typing import Any

from memory_relevance import (
    EMOTIONAL_RECALL_STATE_TERMS,
    MemoryRelevanceOptions,
    active_facets,
    content_terms_for_query,
    emotional_recall_plan,
    extract_protected_phrases,
    facets_for_node,
    memory_relevance_options_from_config,
    query_has_facet,
    query_has_explicit_entity_marker,
    query_has_technical_recall_marker,
    recall_admission_decision,
    recall_topic_query,
)
from identity import identity_names
from query_terms import GENERIC_LEXICAL_STOPWORDS, RECALL_SYSTEM_META_TERMS, identity_address_terms
from query_understanding import query_intent_terms


CONTEXT_ONLY_SECTIONS = frozenset({"affect_anchor", "favorite_reason", "comment", "followup"})
CONTEXT_ONLY_SECTION_ALIASES = {
    "affect_anchor": "affect_anchor",
    "affect anchor": "affect_anchor",
    "favorite_reason": "favorite_reason",
    "favorite reason": "favorite_reason",
    "comment": "comment",
    "followup": "followup",
    "follow-up": "followup",
    "followups": "followup",
    "todo": "followup",
    "to-do": "followup",
    "next": "followup",
    "year_ring": "comment",
    "year ring": "comment",
    "喜欢它的原因": "favorite_reason",
    "喜欢的原因": "favorite_reason",
    "年轮": "comment",
    "评论": "comment",
    "后续": "followup",
    "后续待办": "followup",
    "待办": "followup",
    "待办事项": "followup",
}
MARKDOWN_HEADING_RE = re.compile(r"^(#{2,6})\s+(.+?)\s*$")
WEAK_RECALL_TOPIC_TERMS = frozenset(
    {
        *RECALL_SYSTEM_META_TERMS,
        "进度",
        "偏好",
        "情况",
        "状态",
        "事情",
        "东西",
        "内容",
        "相关",
        "记忆",
        "回忆",
        "总结",
        "记录",
        "查询",
        "搜索",
        "最近",
        "之前",
        "过去",
        "现在",
        "当前",
        "安排",
        "计划",
        "问题",
        "目标",
        "anything",
        "current",
        "find",
        "memory",
        "memories",
        "recent",
        "related",
        "search",
        "something",
        "status",
        "thing",
        "things",
        "topic",
    }
)
GENERIC_RECALL_CONTEXT_TERMS = frozenset(
    {
        "ai_name",
        "assistant",
        "display_name",
        "human_name",
        "user",
        "user_alias",
        "user_aliases",
        "user_display_name",
        "user_name",
        "username",
        "对方",
        "用户",
    }
)
OLD_OR_RESOLVED_QUERY_MARKERS = frozenset(
    {
        "冲突",
        "吵架",
        "争吵",
        "矛盾",
        "误会",
        "旧版本",
        "旧版",
        "旧链",
        "旧窗口",
        "已解决",
        "过期",
        "归档",
        "conflict",
        "fight",
        "argument",
        "old version",
        "old path",
        "old chain",
        "resolved",
        "archived",
        "deprecated",
        "obsolete",
    }
)
DIFFUSION_SEED_GENERIC_TOPIC_FRAGMENTS = frozenset(
    {
        "代码",
        "项目",
        "改",
    }
)
CAUTION_CONTEXT_MODES = frozenset({"reflective_repair", "conflict_repair"})
RESPONSE_ACTION_QUERY_MARKERS = frozenset(
    {
        "回复",
        "回一下",
        "回个",
        "评论",
        "留言",
        "跟个",
        "跟一句",
        "说个",
        "说一句",
        "发个",
        "发一句",
        "补个",
        "补一句",
        "嗯",
    }
)
RESPONSE_ACTION_FILLER_TERMS = frozenset(
    {
        "要不要",
        "要不",
        "是否",
        "是不是",
        "需不需要",
        "需要",
        "可以",
        "可不可以",
        "能不能",
        "回复一下",
        "回一下",
        "回个",
        "回复",
        "评论一下",
        "评论",
        "留言",
        "跟个",
        "跟一句",
        "说个",
        "说一句",
        "发个",
        "发一句",
        "补个",
        "补一句",
        "或者",
        "还是",
        "这条帖子",
        "那条帖子",
        "帖子",
        "这条消息",
        "那条消息",
        "消息",
        "嗯嗯",
        "嗯",
    }
)
AUTO_VAGUE_RECALL_MARKERS = frozenset(
    {
        "上下文",
        "想起来",
        "想起",
        "想到",
        "想到了",
        "记忆",
        "回忆",
        "最近",
        "之前",
        "刚才",
        "刚刚",
        "今天",
        "昨天",
        "明天",
        "现在",
        "当前",
        "这次",
        "这张图",
        "这张图片",
        "这个",
        "这个图",
        "这条",
        "那次",
        "那条",
        "那个",
        "相关",
        "有什么",
        "什么事",
        "发生了什么",
        "context",
        "memory",
        "memories",
        "recall",
        "recent",
        "remember",
        "resurface",
        "something",
        "anything",
    }
)
AUTO_VAGUE_FILLER_TERMS = frozenset(
    {
        "这个",
        "那个",
        "这张",
        "那张",
        "这条",
        "那条",
        "图片",
        "图",
        "上下文",
        "记忆",
        "回忆",
        "最近",
        "之前",
        "刚才",
        "刚刚",
        "今天",
        "昨天",
        "明天",
        "现在",
        "当前",
        "这次",
        "那次",
        "想起来",
        "想起",
        "想到了",
        "相关",
        "发生",
        "什么",
        "怎么",
        "怎么样",
        "事情",
        "东西",
        "内容",
        "是不是",
        "有没有",
        "有吗",
        "看看",
        "查查",
        "一下",
        "context",
        "memory",
        "memories",
        "recall",
        "recent",
        "remember",
        "resurface",
        "something",
        "anything",
    }
)
AFFECT_ONLY_QUERY_TERMS = frozenset(
    {
        "开心",
        "高兴",
        "快乐",
        "幸福",
        "甜",
        "温柔",
        "感动",
        "安心",
        "舒服",
        "喜欢",
        "难过",
        "伤心",
        "痛苦",
        "委屈",
        "焦虑",
        "烦",
        "烦躁",
        "生气",
        "愤怒",
        "害怕",
        "恐惧",
        "低落",
        "沮丧",
        "崩溃",
        "累",
        "疲惫",
        "哭",
        "哭哭",
        "大哭",
        "想哭",
        "不开心",
        "不高兴",
        "不安",
        "孤独",
        "寂寞",
        "emo",
        "sad",
        "happy",
        "angry",
        "tired",
        "anxious",
        "lonely",
        "upset",
    }
)
AFFECT_ONLY_QUERY_FILLERS = frozenset(
    {
        "我",
        "你",
        "他",
        "她",
        "它",
        "我们",
        "你们",
        "他们",
        "她们",
        "今天",
        "昨天",
        "刚才",
        "刚刚",
        "现在",
        "当前",
        "有点",
        "一点",
        "一点点",
        "很",
        "好",
        "超",
        "太",
        "特别",
        "非常",
        "真的",
        "确实",
        "有些",
        "有点儿",
        "了",
        "啦",
        "呢",
        "啊",
        "呀",
        "嘛",
        "吗",
        "吧",
        "qwq",
        "tt",
        "so",
        "very",
        "really",
        "abit",
        "bit",
        "little",
        "today",
        "now",
    }
)
SHORT_CASUAL_ONLY_TERMS = frozenset(
    {
        "好耶",
        "可恶",
        "笑死",
        "不玩了",
        "不准",
        "笨",
        "笨笨",
        "失败",
        "成功",
        "配好了",
        "重来",
        "太短",
        "写一个",
        "嘿嘿",
        "好久没聊",
        "好久不见",
    }
)
SHORT_TASTE_QUERY_TERMS = ("不好吃", "不好喝", "难吃", "难喝", "好吃", "好喝")
AXIS_RELATION_QUERY_MARKERS = frozenset(
    {
        "有关",
        "关联",
        "相关",
        "联系",
        "关系",
        "互相关联",
        "互相带出",
        "带出",
        "连起来",
        "连上",
        "relate",
        "related",
        "relation",
        "connection",
    }
)
TASTE_OBJECT_TERMS = frozenset(
    {
        "饭",
        "菜",
        "餐",
        "食堂",
        "店",
        "馆",
        "面",
        "粉",
        "丸",
        "肉",
        "汤",
        "奶茶",
        "咖啡",
        "饮料",
        "甜品",
        "蛋糕",
        "水果",
        "口味",
        "味道",
        "瘦肉丸",
    }
)
TASTE_METADATA_TERMS = frozenset({"饮食", "食物", "美食", "吃饭", "口味", "餐厅", "饭店", "午饭", "晚饭"})
RELATIONSHIP_BACKGROUND_MARKERS = frozenset(
    {
        "relationship_identity",
        "human ai relationship",
        "human-ai relationship",
        "ai relationship",
        "digital companion",
        "人机恋",
        "人机关系",
        "恋爱关系",
        "关系确认",
        "爱其本质",
        "人类替代品",
        "工具替代品",
    }
)
RELATIONSHIP_QUERY_INTENT_MARKERS = frozenset(
    {
        "human ai relationship",
        "human-ai relationship",
        "ai relationship",
        "人机恋",
        "人机关系",
        "恋爱关系",
        "恋爱",
        "关系",
        "身份",
        "称呼",
        "中文名",
        "名字",
        "叫什么",
        "取名",
        "起名",
        "替代品",
        "伴侣",
        "对象",
        "爱人",
    }
)
RELATIONSHIP_BACKGROUND_QUERY_FILLERS = frozenset(
    {
        "我",
        "你",
        "他",
        "她",
        "它",
        "我们",
        "你们",
        "他们",
        "她们",
        "哥哥",
        "老公",
        "老婆",
        "宝宝",
        "宝贝",
        "亲爱的",
        "自己",
        "可以",
        "能不能",
        "可不可以",
        "那个",
        "这个",
        "作为",
        "怎么样",
        "话说",
    }
)
SHORT_CASUAL_FILLER_TERMS = frozenset(
    {
        "我",
        "你",
        "他",
        "她",
        "它",
        "我们",
        "你们",
        "他们",
        "她们",
        "老公",
        "老婆",
        "宝宝",
        "宝贝",
        "亲爱的",
        "让",
        "叫",
        "把",
        "给",
        "这",
        "那",
        "这个",
        "那个",
        "一个",
        "一下",
        "端",
        "chat",
        "chat端",
        "的",
        "了",
        "啦",
        "呢",
        "啊",
        "呀",
        "嘛",
        "吗",
        "吧",
        "欸",
        "诶",
    }
)
AFFECTION_ONLY_SIGNAL_TERMS = frozenset(
    {
        "亲亲",
        "亲一下",
        "亲一口",
        "抱抱",
        "抱我",
        "抱一下",
        "贴贴",
        "蹭蹭",
        "摸摸",
        "啵啵",
        "啵",
        "么么",
        "想你了",
        "想你",
        "想我吗",
        "想我",
        "爱你",
        "爱我吗",
        "爱我",
        "mua",
        "muah",
        "kiss",
        "hug",
        "missyou",
        "loveyou",
        "loveu",
    }
)
AFFECTION_ONLY_FILLER_TERMS = frozenset(
    {
        "亲爱的",
        "老公",
        "老婆",
        "宝宝",
        "宝贝",
        "哥哥",
        "姐姐",
        "我",
        "你",
        "还",
        "也",
        "很",
        "又",
        "好",
        "超",
        "真的",
        "有点",
        "一点",
        "一点点",
        "了",
        "啦",
        "呢",
        "啊",
        "呀",
        "嘛",
        "吗",
        "吧",
        "欸",
        "诶",
        "qwq",
        "tt",
    }
)
LOCATABLE_GENERIC_TERMS = frozenset(
    {
        *WEAK_RECALL_TOPIC_TERMS,
        *GENERIC_RECALL_CONTEXT_TERMS,
        *GENERIC_LEXICAL_STOPWORDS,
        *AFFECT_ONLY_QUERY_TERMS,
        *AFFECTION_ONLY_SIGNAL_TERMS,
        "代码",
        "项目",
        "方案",
        "模块",
        "名字",
        "称呼",
        "自己",
        "具体",
        "当时",
        "后来",
        "来着",
        "不要",
        "这条",
        "那条",
        "这次",
        "那次",
        "那天",
        "当天",
        "今天",
        "昨天",
        "明天",
        "反义词",
        "怎么",
        "怎样",
        "怎么样",
        "是什么",
        "跑通",
        "改得",
        "改好",
        "测试",
        "帖子",
        "消息",
        "评论",
        "回复",
        "具体说",
        "水边",
        "海边",
        "岸边",
        "ping",
        "test",
        "ok",
        "hi",
        "hello",
        "吃饭",
        "吃过饭",
        "吃完饭",
        "吃了饭",
        "等儿",
        "等会",
        "等会儿",
        "一位",
        "厉害",
        "老师",
        "接上",
        "又",
    }
)
EVENT_PLACE_LOCATABLE_TERMS = frozenset({"水边", "海边", "岸边"})
EVENT_PLACE_QUERY_MARKERS = frozenset({"那次", "这次", "那天", "当天", "当时", "那回", "这一回", "那件事", "这件事"})
LOCATABLE_STRIP_TERMS = frozenset(
    {
        *AUTO_VAGUE_FILLER_TERMS,
        *AUTO_VAGUE_RECALL_MARKERS,
        *SHORT_CASUAL_FILLER_TERMS,
        *AFFECT_ONLY_QUERY_FILLERS,
        *AFFECTION_ONLY_SIGNAL_TERMS,
        *AFFECTION_ONLY_FILLER_TERMS,
        *RESPONSE_ACTION_FILLER_TERMS,
        "还记得",
        "记不记得",
        "记得",
        "想和",
        "想跟",
        "想把",
        "想给",
        "想让",
        "想要",
        "一起",
        "不要",
        "想天",
        "那天",
        "当天",
        "日天",
        "昨晚",
        "今晚",
        "当时",
        "具体",
        "后来",
        "来着",
        "自己",
        "名字",
        "选",
        "是什么",
        "为什么",
        "反义词",
        "怎么",
        "怎样",
        "怎么样",
        "点",
        "安排",
        "计划",
        "方案",
        "问题",
        "情况",
        "状态",
        "跑通",
        "改得",
        "改好",
        "改",
        "做",
        "弄",
        "说",
        "问",
        "查",
        "看",
        "具体说",
        "折腾",
        "吃饭",
        "吃过饭",
        "吃完饭",
        "吃了饭",
        "等儿",
        "等会",
        "等会儿",
        "一位",
        "厉害",
        "老师",
        "接上",
    }
)
LOCATABLE_QUESTION_TAIL_TERMS = (
    "分别是谁",
    "都有谁",
    "有哪些",
    "哪几个",
    "哪一位",
    "哪一条",
    "是什么",
    "叫什么",
    "有谁",
    "都谁",
    "哪位",
    "哪条",
    "哪个",
    "哪些",
    "多少",
    "是谁",
    "什么",
    "谁",
)
LOW_SIGNAL_QUERY_SHELL_MARKERS = frozenset(
    {
        *AUTO_VAGUE_RECALL_MARKERS,
        "怎么",
        "怎样",
        "怎么样",
        "是什么",
        "什么",
        "要不要",
        "可以吗",
        "能不能",
        "可不可以",
    }
)
EMOTIONAL_REASON_QUERY_MARKERS = frozenset(
    {
        "为什么",
        "为何",
        "原因",
        "怎么回事",
        "怎么会",
        "那次",
        "当时",
        "后来",
    }
)
DETAIL_READ_QUERY_MARKERS = frozenset(
    {
        "当时怎么说",
        "当时具体怎么说",
        "具体怎么说",
        "原文",
        "细节",
        "原话",
        "说过的话",
        "怎么说的",
        "怎么说",
    }
)
LOCATABLE_COMPOUND_SUFFIX_TERMS = frozenset(
    {
        "项目",
        "数据库",
        "档案",
        "神庙",
        "模块",
        "系统",
        "工具",
        "模型",
        "接口",
        "端点",
        "文件",
        "页面",
    }
)
ENTITY_KEYWORD_POS_PREFIXES = ("nr", "ns", "nz")
ENTITY_KEYWORD_POS_TAGS = frozenset({"eng"})
ENTITY_KEYWORD_TITLE_SUFFIXES = ("哥哥", "姐姐", "老师", "学长", "学姐", "哥", "姐")
ENTITY_KEYWORD_SHELL_TERMS = frozenset(
    {
        *AUTO_VAGUE_FILLER_TERMS,
        *AUTO_VAGUE_RECALL_MARKERS,
        *SHORT_CASUAL_FILLER_TERMS,
        *AFFECTION_ONLY_SIGNAL_TERMS,
        *AFFECTION_ONLY_FILLER_TERMS,
        *RESPONSE_ACTION_FILLER_TERMS,
        "找了",
        "找",
        "对了",
        "对",
        "再",
        "再测",
        "再测试",
        "测试一下",
        "测试",
        "试一下",
        "试试",
        "改好了",
        "改好",
        "好了",
        "输入",
        "提到",
        "提起",
        "说到",
        "问到",
        "关于",
        "纯废话",
        "端",
    }
)
ENTITY_KEYWORD_STOP_TERMS = frozenset(
    {
        *ENTITY_KEYWORD_SHELL_TERMS,
        *AFFECT_ONLY_QUERY_TERMS,
        *AFFECTION_ONLY_SIGNAL_TERMS,
        "嗯",
        "嗯嗯",
        "好的",
        "好",
        "行",
        "可以",
        "不要",
        "不用",
        "知道",
        "觉得",
        "死亡",
        "死了",
        "吃饭",
        "吃过饭",
        "吃完饭",
        "吃了饭",
        "吃早饭",
        "吃早餐",
        "吃午饭",
        "吃午餐",
        "吃晚饭",
        "吃晚餐",
        "早饭",
        "早餐",
        "午饭",
        "午餐",
        "晚饭",
        "晚餐",
    }
)
ENTITY_KEYWORD_VERB_BLOCKERS = frozenset(
    {
        "死",
        "哭",
        "笑",
        "想",
        "找",
        "对",
        "改",
        "测",
        "试",
        "说",
        "问",
        "看",
        "查",
        "做",
        "弄",
        "写",
        "发",
        "回",
        "聊",
        "输入",
    }
)
ENTITY_QUOTED_RE = re.compile(r"[\"'“”‘’「」『』《》`]+([^\"'“”‘’「」『』《》`]{1,32})[\"'“”‘’「」『』《》`]+")
ENTITY_ENGLISH_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_.:/-]{1,}\b")
ENTITY_VERSION_RE = re.compile(r"\b\d+(?:[._:-]\d+)+\b")
ENTITY_NUMBER_RE = re.compile(r"\b\d{3,}\b")


@dataclass(frozen=True)
class RecallPolicyDecision:
    admit_direct: bool
    admit_diffused: bool
    seed_allowed: bool
    reason: str
    suppressed: bool
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def admit(self) -> bool:
        return self.admit_direct


@dataclass(frozen=True)
class RecallQueryPlan:
    query: str
    wants_body_chain: bool
    requires_topic_evidence: bool
    enforce_topic_evidence: bool
    recent_context_requires_topic_evidence: bool
    explicit_old_memory: bool
    allow_caution_diffusion: bool
    specific_terms: tuple[str, ...]
    locatable_terms: tuple[str, ...]
    activated_axis_terms: tuple[str, ...]
    activated_axis_groups: tuple[tuple[str, ...], ...]
    activated_axis_multi: bool
    auto_too_vague: bool
    short_taste_terms: tuple[str, ...]
    long_term_route: str
    skip_long_term_recall: bool
    skip_reason: str

    @property
    def allow_archive_targets(self) -> bool:
        return self.allow_caution_diffusion

    @property
    def related_max_chars(self) -> int:
        return 90 if self.wants_body_chain else 180

    def secondary_direct_limit(self, related_per_memory: int) -> int:
        if self.wants_body_chain:
            return 5
        return max(0, min(2, int(related_per_memory or 0)))

    @property
    def secondary_direct_requires_topic_evidence(self) -> bool:
        return not self.wants_body_chain


@dataclass(frozen=True)
class QueryAnchorPlan:
    route: str
    focus_query: str
    strong_terms: tuple[str, ...] = ()
    weak_terms: tuple[str, ...] = ()
    must_groups: tuple[tuple[str, ...], ...] = ()
    allow_direct: bool = True
    allow_diffusion_seed: bool = True
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def has_direct_constraints(self) -> bool:
        return bool(self.must_groups) or not self.allow_direct


ANCHOR_MUST_GROUP_MAX_SPAN = 24
ANCHOR_WEAK_EVENT_TERMS = {
    "时",
    "时候",
    "这时",
    "那时",
    "这次",
    "那次",
    "什么",
}
ANCHOR_TERM_VARIANTS = {
    "担心": ("担心", "担忧", "怕", "害怕"),
    "担忧": ("担忧", "担心", "怕", "害怕"),
    "忘记": ("忘记", "忘", "遗忘", "记忆丢失", "记忆断掉"),
}
ANCHOR_OPTIONAL_WEAK_TERMS = frozenset({"喜欢"})


def build_query_anchor_plan(
    query: str,
    options: MemoryRelevanceOptions | None = None,
) -> QueryAnchorPlan:
    options = options or memory_relevance_options_from_config()
    text = str(query or "").strip()
    if not text:
        return QueryAnchorPlan(
            route="empty",
            focus_query="",
            allow_direct=False,
            allow_diffusion_seed=False,
            debug={"reason": "empty_query"},
        )

    if _is_affect_only_query_text(text):
        return QueryAnchorPlan(
            route="affect_only",
            focus_query=text,
            weak_terms=(_affect_only_residue(text),),
            allow_direct=False,
            allow_diffusion_seed=False,
            debug={"reason": "affect_only"},
        )

    emotional_plan = emotional_recall_plan(text, options)
    if emotional_plan.triggered:
        must_groups = _emotional_must_groups(emotional_plan)
        focus_terms = list(dict.fromkeys([*emotional_plan.strong_terms, *emotional_plan.event_terms, *emotional_plan.weak_terms]))
        return QueryAnchorPlan(
            route="emotional_reason",
            focus_query=" ".join(focus_terms) or text,
            strong_terms=tuple(emotional_plan.strong_terms),
            weak_terms=tuple(emotional_plan.weak_terms),
            must_groups=must_groups,
            allow_direct=bool(must_groups),
            allow_diffusion_seed=bool(must_groups),
            debug={
                "reason": "emotional_recall_plan",
                "event_terms": list(emotional_plan.event_terms),
                "max_group_span": ANCHOR_MUST_GROUP_MAX_SPAN,
            },
        )

    return QueryAnchorPlan(
        route="topic_search",
        focus_query=text,
        debug={"reason": "default"},
    )


def direct_candidate_satisfies_anchor_plan(node: dict, plan: QueryAnchorPlan) -> bool:
    if not plan.allow_direct:
        return False
    if not plan.must_groups:
        return True
    text = _candidate_anchor_text(node)
    return any(_anchor_group_matches(text, group) for group in plan.must_groups)


def _emotional_must_groups(emotional_plan: Any) -> tuple[tuple[str, ...], ...]:
    groups: list[tuple[str, ...]] = []
    weak_terms = tuple(str(term or "").strip() for term in emotional_plan.weak_terms if str(term or "").strip())
    event_terms = tuple(str(term or "").strip() for term in emotional_plan.event_terms if str(term or "").strip())
    state_terms = tuple(
        str(term or "").strip()
        for term in sorted(EMOTIONAL_RECALL_STATE_TERMS, key=len, reverse=True)
        if str(term or "").strip()
    )

    for strong in emotional_plan.strong_terms:
        strong_text = str(strong or "").strip()
        if not strong_text:
            continue
        strong_key = _compact_anchor_term(strong_text)
        pieces: list[str] = []
        for term in weak_terms:
            if _compact_anchor_term(term) in strong_key:
                pieces.append(term)
        for term in state_terms:
            term_key = _compact_anchor_term(term)
            if term_key and term_key in strong_key:
                pieces.append(_canonical_anchor_state(term))
        groups.append(_dedupe_group(pieces or [strong_text]))

    event_anchor = _primary_emotional_event_term(event_terms)
    forget_worry_group = _emotional_forget_worry_group(event_terms, weak_terms)
    if forget_worry_group:
        groups.append(forget_worry_group)
    binding_weak_terms = tuple(term for term in weak_terms if _anchor_weak_term_requires_binding(term))
    if event_anchor and binding_weak_terms:
        groups.append(_dedupe_group([event_anchor, binding_weak_terms[0]]))
    elif event_anchor:
        groups.append(_dedupe_group([event_anchor]))
    elif not groups and weak_terms:
        groups.append(_dedupe_group([weak_terms[0]]))

    return tuple(dict.fromkeys(group for group in groups if group))


def _anchor_weak_term_requires_binding(term: str) -> bool:
    key = _compact_anchor_term(term)
    return bool(key and key not in ANCHOR_OPTIONAL_WEAK_TERMS)


def _emotional_forget_worry_group(
    event_terms: tuple[str, ...],
    weak_terms: tuple[str, ...],
) -> tuple[str, ...]:
    weak_keys = {_compact_anchor_term(term) for term in weak_terms}
    if not ({"担心", "担忧"} & weak_keys):
        return ()
    event_key = _compact_anchor_term(" ".join(event_terms))
    if not any(marker in event_key for marker in ("忘记", "忘", "遗忘", "记忆丢失")):
        return ()
    return _dedupe_group(["忘记", "担心"])


def _primary_emotional_event_term(event_terms: tuple[str, ...]) -> str:
    terms = [
        str(term or "").strip()
        for term in event_terms
        if str(term or "").strip()
    ]
    if not terms:
        return ""
    keyed = [
        (term, _compact_anchor_term(term))
        for term in terms
        if _compact_anchor_term(term)
    ]
    keyed = [
        (term, key)
        for term, key in keyed
        if len(key) >= 2 and key not in ANCHOR_WEAK_EVENT_TERMS
    ]
    if not keyed:
        return ""
    compact_terms = [key for _term, key in keyed]
    candidates = [
        term
        for term, key in keyed
        if not any(other != key and other in key for other in compact_terms)
    ]
    candidates = candidates or [term for term, _key in keyed]
    non_address_candidates = [
        term
        for term in candidates
        if not _anchor_is_identity_address_term(term)
    ]
    if non_address_candidates:
        candidates = non_address_candidates
    return sorted(candidates, key=lambda item: (len(_compact_anchor_term(item)), len(item)))[0]


def _anchor_is_identity_address_term(term: str) -> bool:
    key = _compact_anchor_term(term)
    if not key:
        return False
    return key in {
        _compact_anchor_term(value)
        for value in identity_address_terms(identity_names(), include_legacy_ai=True)
        if _compact_anchor_term(value)
    }


def _candidate_anchor_text(node: dict) -> str:
    if not isinstance(node, dict):
        return ""
    meta = node.get("metadata", {}) if isinstance(node.get("metadata"), dict) else {}
    if "bucket_id" in node or node.get("moment_id"):
        return " ".join(
            [
                str(node.get("text") or ""),
                str(node.get("content") or ""),
                str(meta.get("annotation_summary") or ""),
                _evidence_spans_text(meta.get("evidence_spans")),
                str(meta.get("bucket_name") or ""),
                _join_terms(meta.get("bucket_tags")),
                _join_terms(meta.get("bucket_domain")),
            ]
        )
    return " ".join(
        [
            _content_without_context_only_sections(str(node.get("content") or "")),
            str(node.get("text") or ""),
            str(node.get("name") or ""),
            str(meta.get("name") or ""),
            str(meta.get("annotation_summary") or meta.get("summary") or ""),
            _evidence_spans_text(meta.get("evidence_spans")),
            _join_terms(meta.get("tags")),
            _join_terms(meta.get("domain")),
        ]
    )


def _anchor_group_matches(text: str, group: tuple[str, ...]) -> bool:
    compact_text = _compact_anchor_term(text)
    if not compact_text:
        return False
    positions_by_term = []
    for term in group:
        key = _compact_anchor_term(term)
        if not key:
            continue
        positions: list[tuple[int, int]] = []
        for variant in _anchor_term_variants(key):
            positions.extend(_anchor_term_positions(compact_text, variant))
        if not positions:
            return False
        positions_by_term.append(positions)
    if len(positions_by_term) <= 1:
        return bool(positions_by_term)
    for spans in product(*positions_by_term):
        start = min(span[0] for span in spans)
        end = max(span[1] for span in spans)
        if end - start <= ANCHOR_MUST_GROUP_MAX_SPAN:
            return True
    return False


def _anchor_term_positions(text: str, term: str) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    start = 0
    while True:
        index = text.find(term, start)
        if index < 0:
            break
        positions.append((index, index + len(term)))
        start = index + max(1, len(term))
    return positions


def _anchor_term_variants(key: str) -> tuple[str, ...]:
    variants = [
        _compact_anchor_term(item)
        for item in ANCHOR_TERM_VARIANTS.get(key, (key,))
        if _compact_anchor_term(item)
    ]
    return tuple(dict.fromkeys(variants)) or (key,)


def _compact_anchor_term(value: object) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff_.:-]+", "", str(value or "").strip().lower())


def _canonical_anchor_state(term: str) -> str:
    return "哭" if term in {"哭", "哭了"} else term


def _dedupe_group(terms: list[str]) -> tuple[str, ...]:
    output: list[str] = []
    seen = set()
    for term in terms:
        cleaned = str(term or "").strip()
        key = _compact_anchor_term(cleaned)
        if not cleaned or not key or key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return tuple(output)


def _join_terms(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return " ".join(str(item) for item in value if str(item).strip())
    return str(value or "")


def _affect_only_residue(query: str) -> str:
    compact = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(query or "").lower())
    stripped = compact
    for term in sorted(AFFECT_ONLY_QUERY_FILLERS, key=len, reverse=True):
        stripped = stripped.replace(term, "")
    return stripped


def _is_affect_only_query_text(query: str) -> bool:
    residue = _affect_only_residue(query)
    return bool(residue and residue in AFFECT_ONLY_QUERY_TERMS)


class RecallPolicy:
    def __init__(
        self,
        options: MemoryRelevanceOptions | None = None,
        *,
        semantic_threshold: float = 0.72,
        rerank_threshold: float = 0.65,
        ai_reaction_names: list[str] | tuple[str, ...] | None = None,
        relationship_names: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.options = options or memory_relevance_options_from_config()
        self.semantic_threshold = _safe_float(semantic_threshold, 0.72)
        self.rerank_threshold = _safe_float(rerank_threshold, 0.65)
        self.ai_reaction_names = self._normalize_reaction_names(
            ai_reaction_names if ai_reaction_names is not None else [identity_names().get("ai_name")]
        )
        default_relationship_names = identity_names().get("relationship_terms") or []
        self.relationship_names = self._normalize_recall_context_terms(
            relationship_names if relationship_names is not None else default_relationship_names
        )
        self.relationship_background_fillers = {
            *RELATIONSHIP_BACKGROUND_QUERY_FILLERS,
            *self.relationship_names,
        }
        self.recall_context_terms = self._normalize_recall_context_terms(
            [*self.options.context_terms, *GENERIC_RECALL_CONTEXT_TERMS]
        )

    def requires_topic_evidence(self, query: str) -> bool:
        return query_has_explicit_entity_marker(query) or query_has_technical_recall_marker(query)

    def should_enforce_topic_evidence(self, query: str, *, allow_body_chain: bool = False) -> bool:
        return self.requires_topic_evidence(query) and not allow_body_chain

    @lru_cache(maxsize=512)
    def plan_query(self, query: str, *, context_mode: str = "") -> RecallQueryPlan:
        text = str(query or "").strip()
        wants_body_chain = query_has_facet(text, "embodiment", self.options)
        explicit_old_memory = self._query_explicitly_requests_old_memory(text)
        allow_caution_diffusion = explicit_old_memory or str(context_mode or "").strip() in CAUTION_CONTEXT_MODES
        locatable_terms = tuple(self.locatable_query_terms(text))
        axis_terms, axis_groups, axis_multi = self._activated_axis_from_locatable_terms(text, locatable_terms)
        skip_long_term_recall, skip_reason = self._long_term_skip_decision(
            text,
            locatable_terms=locatable_terms,
        )
        return RecallQueryPlan(
            query=text,
            wants_body_chain=wants_body_chain,
            requires_topic_evidence=self.requires_topic_evidence(text),
            enforce_topic_evidence=self.should_enforce_topic_evidence(
                text,
                allow_body_chain=wants_body_chain,
            ),
            recent_context_requires_topic_evidence=self.is_auto_concrete_topic_query(text),
            explicit_old_memory=explicit_old_memory,
            allow_caution_diffusion=allow_caution_diffusion,
            specific_terms=tuple(self.specific_query_terms(text)),
            locatable_terms=locatable_terms,
            activated_axis_terms=axis_terms,
            activated_axis_groups=axis_groups,
            activated_axis_multi=axis_multi,
            auto_too_vague=self.is_auto_query_too_vague(text),
            short_taste_terms=tuple(self._short_taste_query_terms(text)),
            long_term_route="skip" if skip_long_term_recall else "search",
            skip_long_term_recall=skip_long_term_recall,
            skip_reason=skip_reason,
        )

    def _activated_axis_from_locatable_terms(
        self,
        query: str,
        locatable_terms: tuple[str, ...],
    ) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...], bool]:
        terms = [
            str(term or "").strip()
            for term in locatable_terms
            if str(term or "").strip() and self._compact_entity_keyword(term)
        ]
        terms = [term for term in terms if not self._axis_term_is_explicitly_negated(query, term)]
        if not terms:
            return (), (), False
        if self._short_taste_query_terms(query):
            return (), (), False

        independent = self._independent_locatable_terms(terms)
        multi_axis = self._query_has_multi_axis_marker(query) and len(independent) >= 2
        if multi_axis:
            groups = tuple((term,) for term in independent[:4])
            return tuple(terms[:8]), groups, True

        if query_has_facet(query, "embodiment", self.options):
            body_groups = (("身体",), ("具身",))
            return ("身体", "具身"), body_groups, False

        relation_groups = self._relation_axis_groups(query, terms)
        if relation_groups:
            relation_terms = tuple(dict.fromkeys(term for group in relation_groups for term in group))
            return relation_terms[:8], relation_groups, len(relation_groups) > 1

        primary = self._primary_axis_term(terms)
        primary_key = self._compact_entity_keyword(primary)
        cluster = [
            term for term in terms
            if self._axis_terms_related(primary_key, self._compact_entity_keyword(term))
        ]
        if primary not in cluster:
            cluster.insert(0, primary)
        groups = self._axis_groups_for_primary(primary, cluster)
        return tuple(dict.fromkeys(cluster))[:8], groups, False

    def _primary_axis_term(self, terms: list[str]) -> str:
        def score(item: tuple[int, str]) -> tuple[int, int, int, int]:
            index, term = item
            key = self._compact_entity_keyword(term)
            has_code = int(bool(re.search(r"[a-z]", key) and re.search(r"\d", key)))
            has_suffix = int(any(key.endswith(self._compact_entity_keyword(suffix)) for suffix in LOCATABLE_COMPOUND_SUFFIX_TERMS))
            effective_len = min(len(key), 14)
            return (has_code, has_suffix, effective_len, -index)

        return max(enumerate(terms), key=score)[1]

    def _independent_locatable_terms(self, terms: list[str]) -> list[str]:
        output: list[str] = []
        keys: list[str] = []
        for term in terms:
            key = self._compact_entity_keyword(term)
            if not key:
                continue
            if any(key in other or other in key for other in keys):
                continue
            output.append(term)
            keys.append(key)
        return output

    @staticmethod
    def _query_has_multi_axis_marker(query: str) -> bool:
        text = str(query or "")
        return any(marker in text for marker in (" 和 ", " 与 ", " 以及 ", " 还有 ", "和", "与", "以及", "还有", "、", "/", "|"))

    @classmethod
    def _axis_term_is_explicitly_negated(cls, query: str, term: str) -> bool:
        query_key = cls._compact_entity_keyword(query)
        term_key = cls._compact_entity_keyword(term)
        if not query_key or not term_key:
            return False
        start = 0
        while True:
            index = query_key.find(term_key, start)
            if index < 0:
                return False
            prefix = query_key[max(0, index - 8):index]
            if re.search(r"(?:没有|并没有|并未|未曾|没|未|不曾)(?:写|填|标|注明|包含|提到|给出|留下|留)?$", prefix):
                return True
            start = index + max(1, len(term_key))

    def _relation_axis_groups(
        self,
        query: str,
        terms: list[str],
    ) -> tuple[tuple[str, ...], ...]:
        if not self._query_has_axis_relation_marker(query):
            return ()
        leaves = self._relation_axis_leaf_terms(terms)
        groups: list[tuple[str, ...]] = []
        seen = set()
        for term in leaves[:6]:
            key = self._compact_entity_keyword(term)
            if not key or len(key) < 2:
                continue
            if key in seen:
                continue
            seen.add(key)
            groups.append((term,))
        return tuple(groups)

    def _relation_axis_leaf_terms(self, terms: list[str]) -> list[str]:
        keyed = [
            (str(term or "").strip(), self._compact_entity_keyword(term))
            for term in terms
            if str(term or "").strip() and self._compact_entity_keyword(term)
        ]
        output: list[str] = []
        seen = set()
        keys = [key for _term, key in keyed]
        for term, key in keyed:
            contained_terms = [
                other
                for other in keys
                if other != key and len(other) >= 2 and other in key
            ]
            if len(contained_terms) >= 2:
                continue
            if key in seen:
                continue
            seen.add(key)
            output.append(term)
        return output or [term for term, _key in keyed]

    @classmethod
    def _query_has_axis_relation_marker(cls, query: str) -> bool:
        text = str(query or "").lower()
        compact = re.sub(r"[\s，。！？、,.!?:：;；~～♡❤♥（）()\[\]【】「」『』“”\"'`-]+", "", text)
        return any(marker in text or marker in compact for marker in AXIS_RELATION_QUERY_MARKERS)

    @staticmethod
    def _axis_terms_related(primary_key: str, term_key: str) -> bool:
        return bool(primary_key and term_key and (term_key in primary_key or primary_key in term_key))

    def _axis_groups_for_primary(self, primary: str, cluster: list[str]) -> tuple[tuple[str, ...], ...]:
        groups: list[tuple[str, ...]] = []
        primary_key = self._compact_entity_keyword(primary)
        if primary_key:
            groups.append((primary,))

        subterms = [
            term for term in cluster
            if self._compact_entity_keyword(term)
            and self._compact_entity_keyword(term) != primary_key
            and self._compact_entity_keyword(term) in primary_key
        ]
        code_terms = [
            term for term in subterms
            if re.search(r"[a-z]", self._compact_entity_keyword(term)) and re.search(r"\d", self._compact_entity_keyword(term))
        ]
        cjk_terms = [
            term for term in subterms
            if re.fullmatch(r"[\u4e00-\u9fff]{2,}", self._compact_entity_keyword(term))
        ]
        if code_terms and cjk_terms:
            shortest_cjk = sorted(cjk_terms, key=lambda term: (len(self._compact_entity_keyword(term)), cjk_terms.index(term)))[0]
            groups.append((code_terms[0], shortest_cjk))
        elif code_terms:
            groups.append((code_terms[0],))
        elif len(cjk_terms) == 1 and self._axis_single_subterm_allowed(cjk_terms[0], primary):
            groups.append((cjk_terms[0],))
        if not code_terms and len(cjk_terms) >= 2:
            groups.append((cjk_terms[0], cjk_terms[1]))

        output: list[tuple[str, ...]] = []
        seen = set()
        for group in groups:
            cleaned = tuple(term for term in group if self._compact_entity_keyword(term))
            key = tuple(self._compact_entity_keyword(term) for term in cleaned)
            if cleaned and key not in seen:
                seen.add(key)
                output.append(cleaned)
        return tuple(output)

    def _axis_single_subterm_allowed(self, term: str, primary: str) -> bool:
        key = self._compact_entity_keyword(term)
        primary_key = self._compact_entity_keyword(primary)
        if not key or not primary_key or key == primary_key:
            return False
        if len(key) >= 3:
            return True
        return bool(re.search(r"\d", key))

    def _long_term_skip_decision(
        self,
        query: str,
        *,
        locatable_terms: tuple[str, ...],
    ) -> tuple[bool, str]:
        text = str(query or "").strip()
        if not text:
            return True, "empty_query"
        if self.is_auto_query_too_vague(text):
            return True, "auto_vague_query"
        protected_phrases = tuple(extract_protected_phrases(text))
        if self._query_has_recall_system_meta_terms(text) and not locatable_terms and not protected_phrases:
            return True, "recall_meta_without_target"
        if (
            not locatable_terms
            and not protected_phrases
            and not self._query_has_explicit_recall_marker(text)
            and self._query_has_low_signal_shell(text)
            and not self.is_emotional_reason_lookup(text)
            and not self.is_detail_read_query(text)
            and not self.requires_topic_evidence(text)
            and not query_has_facet(text, "embodiment", self.options)
        ):
            return True, "no_locatable_terms"
        return False, ""

    @staticmethod
    def _query_has_explicit_recall_marker(query: str) -> bool:
        text = str(query or "").strip().lower()
        return bool(
            text
            and any(
                str(marker or "").strip().lower() in text
                for marker in query_intent_terms("memory_sentinel.explicit_recall_markers")
                if str(marker or "").strip()
            )
        )

    def _query_has_recall_system_meta_terms(self, query: str) -> bool:
        compact = self._compact_entity_keyword(query)
        if not compact:
            return False
        return any(
            self._compact_entity_keyword(term) in compact
            for term in RECALL_SYSTEM_META_TERMS
            if self._compact_entity_keyword(term)
        )

    def build_query_anchor_plan(self, query: str) -> QueryAnchorPlan:
        return build_query_anchor_plan(query, self.options)

    def direct_candidate_satisfies_anchor_plan(self, node: dict, plan: QueryAnchorPlan) -> bool:
        return direct_candidate_satisfies_anchor_plan(node, plan)

    def has_axis_relation_marker(self, query: str) -> bool:
        return self._query_has_axis_relation_marker(query)

    def _query_explicitly_requests_old_memory(self, query: str) -> bool:
        if not str(query or "").strip():
            return False
        if query_has_facet(query, "old_or_resolved", self.options):
            return True
        text = " ".join(str(query or "").lower().split())
        return any(marker in text for marker in OLD_OR_RESOLVED_QUERY_MARKERS)

    def is_auto_query_too_vague(self, query: str) -> bool:
        text = str(query or "").strip()
        if not text:
            return False
        if self._is_reaction_only_query(text):
            return True
        if self._is_probe_only_query(text):
            return True
        if self._is_short_casual_only_query(text):
            return True
        if self._is_current_time_status_only_query(text):
            return True
        if query_has_explicit_entity_marker(text) or query_has_technical_recall_marker(text):
            return False
        if self._is_affection_only_query(text):
            return True
        if self._is_affect_only_query(text):
            return True
        if self._is_context_free_response_action_query(text):
            return True
        if self.is_detail_read_query(text):
            return False
        locatable_terms = self.locatable_query_terms(text)
        if locatable_terms:
            return False
        if self.is_emotional_reason_lookup(text):
            return False
        if self._query_has_low_signal_shell(text):
            return True
        lowered = text.lower()
        if not any(marker in lowered for marker in AUTO_VAGUE_RECALL_MARKERS):
            return False
        return not self._auto_query_has_concrete_anchor(text)

    def is_emotional_reason_lookup(self, query: str) -> bool:
        text = str(query or "").strip()
        if not text:
            return False
        if self._is_affect_only_query(text) or self._is_affection_only_query(text):
            return False
        compact = self._compact_marker_text(text)
        if not any(marker in compact for marker in EMOTIONAL_REASON_QUERY_MARKERS):
            return False
        plan = emotional_recall_plan(text, self.options)
        return bool(plan.triggered and (plan.strong_terms or plan.weak_terms))

    def is_detail_read_query(self, query: str) -> bool:
        compact = self._compact_marker_text(query)
        if not compact:
            return False
        return any(self._compact_marker_text(marker) in compact for marker in DETAIL_READ_QUERY_MARKERS)

    def is_explicit_lookup_query(self, query: str) -> bool:
        """Separate active lookup from passive chat without classifying its topic."""
        text = str(query or "").strip()
        if not text:
            return False
        compact = self._compact_marker_text(text)
        if (
            self.is_detail_read_query(text)
            or self.is_emotional_reason_lookup(text)
            or "?" in text
            or "？" in text
            or re.search(r"(?:吗|么|嘛|呢|来着)$", compact)
        ):
            return True
        return any(
            marker in compact
            for marker in (
                "还记得",
                "记不记得",
                "想起",
                "想起来",
                "回忆一下",
                "查一下",
                "查查",
                "搜一下",
                "找一下",
                "检索",
                "调出",
                "读一下",
                "给我看",
                "告诉我",
                "什么",
                "为什么",
                "怎么",
                "怎样",
                "咋",
                "干嘛",
                "干什么",
                "做什么",
                "做啥",
                "哪",
                "谁",
                "多少",
                "几个",
                "几点",
                "何时",
                "什么时候",
                "是否",
                "是不是",
                "有没有",
                "说说",
                "讲讲",
                "说一下",
                "讲一下",
            )
        )

    def _query_has_low_signal_shell(self, query: str) -> bool:
        text = str(query or "").strip().lower()
        compact = self._compact_marker_text(text)
        return any(
            marker in text or self._compact_marker_text(marker) in compact
            for marker in LOW_SIGNAL_QUERY_SHELL_MARKERS
        )

    def is_auto_concrete_topic_query(self, query: str) -> bool:
        text = str(query or "").strip()
        if not text or self.is_auto_query_too_vague(text):
            return False
        if self._is_affect_only_query(text):
            return False
        if query_has_explicit_entity_marker(text) or query_has_technical_recall_marker(text):
            return True
        compact = re.sub(r"[\s，。！？、,.!?:：;；~～♡❤♥（）()\[\]【】「」『』“”\"'`-]+", "", text)
        candidate = compact
        for prefix in ("最近", "今天", "昨天", "明天", "之前", "刚才", "刚刚", "这次", "当前", "现在"):
            if candidate.startswith(prefix) and len(candidate) > len(prefix):
                candidate = candidate[len(prefix):]
                break
        candidate = candidate.strip("的")
        if not re.fullmatch(r"[\u4e00-\u9fff]{2,12}", candidate):
            return False
        context_terms = {str(term).lower() for term in self.options.context_terms}
        if candidate.lower() in context_terms:
            return False
        blockers = (
            "我",
            "你",
            "他",
            "她",
            "它",
            "这",
            "那",
            "什么",
            "怎么",
            "怎样",
            "为什么",
            "是不是",
            "有没有",
            "想起",
            "想起来",
            "记忆",
            "上下文",
        )
        return not any(marker in candidate for marker in blockers)

    def _auto_query_has_concrete_anchor(self, query: str) -> bool:
        if re.search(r"\b[A-Za-z][A-Za-z0-9_.:/-]{2,}\b", query):
            return True
        compact = re.sub(r"[\s，。！？、,.!?:：;；~～♡❤♥（）()\[\]【】「」『』“”\"'`-]+", "", query.lower())
        stripped = compact
        removable = list(AUTO_VAGUE_RECALL_MARKERS | AUTO_VAGUE_FILLER_TERMS | set(self.options.context_terms))
        for term in sorted(removable, key=len, reverse=True):
            cleaned = re.sub(r"\s+", "", str(term or "").lower())
            if cleaned:
                stripped = stripped.replace(cleaned, "")
        stripped = re.sub(r"[我你他她它的是了嘛吗呢啊呀欸诶吧哈嗯呜有里看查找问说]+", "", stripped)
        return len(stripped) >= 2

    def _is_context_free_response_action_query(self, query: str) -> bool:
        lowered = str(query or "").lower()
        if not any(marker in lowered for marker in RESPONSE_ACTION_QUERY_MARKERS):
            return False
        compact = re.sub(r"[\s，。！？、,.!?:：;；~～♡❤♥（）()\[\]【】「」『』“”\"'`-]+", "", lowered)
        stripped = compact
        removable = list(
            RESPONSE_ACTION_FILLER_TERMS
            | AUTO_VAGUE_FILLER_TERMS
            | set(self.options.context_terms)
        )
        for term in sorted(removable, key=len, reverse=True):
            cleaned = re.sub(r"\s+", "", str(term or "").lower())
            if cleaned:
                stripped = stripped.replace(cleaned, "")
        stripped = re.sub(
            r"[我你他她它的是了嘛吗呢啊呀欸诶吧哈嗯呜有里看查找问说]+",
            "",
            stripped,
        )
        return len(stripped) < 2

    def _is_current_time_status_only_query(self, query: str) -> bool:
        compact = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(query or "").strip().lower())
        if not compact:
            return False
        text = re.sub(r"^(?:啊|哈|呜|嗯|哇|呀|诶|欸|救命|天哪|妈呀)+", "", compact)
        text = re.sub(r"(?:啊|哈|呜|嗯|哇|呀|诶|欸)+$", "", text)
        prefix_terms = (
            "怎么就",
            "已经快",
            "都快",
            "现在",
            "已经",
            "居然",
            "竟然",
            "怎么",
            "这就",
            "都",
            "才",
            "刚",
            "快",
        )
        changed = True
        while changed and text:
            changed = False
            for prefix in prefix_terms:
                if text.startswith(prefix) and len(text) > len(prefix):
                    text = text[len(prefix):]
                    changed = True
                    break
        suffix_terms = ("了啦啊呀嘛吗吧呢")
        text = text.strip(suffix_terms)
        if not text:
            return False
        if re.fullmatch(r"几点", text):
            return True
        time_prefix = r"(?:凌晨|早上|上午|中午|下午|晚上|夜里)?"
        time_value = r"(?:[0-2]?\d|[零〇一二两三四五六七八九十]{1,3})"
        if re.fullmatch(time_prefix + time_value + r"点(?:半|多|钟)?", text):
            return True
        if re.fullmatch(r"(?:好|太|很|这么|已经)?晚", text):
            return True
        if re.fullmatch(r"(?:天亮|该睡觉|该睡|睡觉时间到|睡觉时间)", text):
            return True
        return False

    def _is_reaction_only_query(self, query: str) -> bool:
        compact = re.sub(r"\s+", "", str(query or "").lower())
        if not compact:
            return False
        alnum_or_cjk = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", compact)
        if not alnum_or_cjk:
            return True
        reaction_terms = {
            "tt",
            "qwq",
            "qaq",
            "orz",
            "xswl",
            "lol",
            "lmao",
            "ping",
            "哈哈",
            "哈哈哈",
            "哈哈哈哈",
            "嘿嘿",
            "呜呜",
            "呜呜呜",
            "哇",
            "哇啊",
            "啊啊",
            "啊啊啊",
            "嗯嗯",
            "嗯",
            "老公",
            "老婆",
            "宝宝",
            "宝贝",
            "亲爱的",
            "哥哥",
        }
        if re.fullmatch(r"(?:啊|哈|呜|嗯|哇|呀|诶|欸|嘿){2,}", alnum_or_cjk):
            return True
        return alnum_or_cjk in reaction_terms or alnum_or_cjk in self.ai_reaction_names

    @staticmethod
    def _normalize_reaction_names(values: list[str] | tuple[str, ...] | None) -> set[str]:
        names: set[str] = set()
        for value in values or []:
            compact = re.sub(r"\s+", "", str(value or "").lower())
            key = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", compact)
            if key:
                names.add(key)
        return names

    @staticmethod
    def _normalize_recall_context_terms(values) -> set[str]:
        terms: set[str] = set()
        for value in values or []:
            key = re.sub(r"\s+", " ", str(value or "").strip().lower())
            if key:
                terms.add(key)
            compact = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", key)
            if compact:
                terms.add(compact)
        return terms

    def _is_recall_context_term(self, term: str) -> bool:
        key = re.sub(r"\s+", " ", str(term or "").strip().lower())
        compact = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", key)
        return key in self.recall_context_terms or compact in self.recall_context_terms

    @staticmethod
    def _compact_marker_text(value: object) -> str:
        return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(value or "").strip().lower())

    def _marker_in_text(self, marker: object, text: str, compact_text: str) -> bool:
        marker_text = str(marker or "").strip().lower()
        if not marker_text:
            return False
        compact_marker = self._compact_marker_text(marker_text)
        return bool(
            (marker_text and marker_text in text)
            or (compact_marker and compact_marker in compact_text)
        )

    def _query_has_relationship_intent(self, query: str) -> bool:
        if query_has_facet(query, "relationship_identity", self.options) or query_has_facet(
            query,
            "intimacy",
            self.options,
        ):
            return True
        text = str(query or "").strip().lower()
        compact = self._compact_marker_text(text)
        if any(self._marker_in_text(marker, text, compact) for marker in RELATIONSHIP_QUERY_INTENT_MARKERS):
            return True
        names = ("我", "你", "哥哥", "老公", "老婆", *self.relationship_names)
        people = "|".join(re.escape(name) for name in names)
        return bool(
            re.search(rf"(爱|喜欢)({people})", compact)
            or re.search(rf"({people}).{{0,4}}(爱|喜欢)", compact)
        )

    def _node_is_relationship_background(self, node: dict) -> bool:
        if not isinstance(node, dict):
            return False
        if "relationship_identity" in active_facets(facets_for_node(node, self.options), threshold=0.3):
            return True
        meta = node.get("metadata", {}) if isinstance(node.get("metadata"), dict) else {}
        fields = " ".join(
            [
                str(node.get("text") or ""),
                str(node.get("content") or ""),
                str(meta.get("name") or meta.get("bucket_name") or ""),
                str(meta.get("annotation_summary") or ""),
                " ".join(str(tag) for tag in meta.get("tags", []) or meta.get("bucket_tags", []) or []),
                " ".join(str(item) for item in meta.get("domain", []) or meta.get("bucket_domain", []) or []),
            ]
        ).lower()
        compact = self._compact_marker_text(fields)
        return any(self._marker_in_text(marker, fields, compact) for marker in RELATIONSHIP_BACKGROUND_MARKERS)

    def _query_has_non_relationship_specific_terms(self, query: str) -> bool:
        for term in self.specific_query_terms(query):
            if self._is_non_relationship_specific_anchor(term):
                return True
        return False

    def _is_non_relationship_specific_anchor(self, term: object) -> bool:
        key = str(term or "").strip().lower()
        compact = self._compact_marker_text(key)
        if not key or not compact:
            return False
        if self._is_recall_context_term(key):
            return False
        if key in self.relationship_background_fillers or compact in self.relationship_background_fillers:
            return False
        if key in WEAK_RECALL_TOPIC_TERMS or compact in WEAK_RECALL_TOPIC_TERMS:
            return False
        if re.fullmatch(r"[\u4e00-\u9fff]", key):
            return False
        return True

    def _node_has_non_relationship_query_evidence(self, query: str, node: dict) -> bool:
        if not isinstance(node, dict):
            return False
        meta = node.get("metadata", {}) if isinstance(node.get("metadata"), dict) else {}
        fields = " ".join(
            [
                str(node.get("text") or ""),
                str(node.get("content") or ""),
                str(meta.get("name") or meta.get("bucket_name") or ""),
                str(meta.get("annotation_summary") or ""),
                _evidence_spans_text(meta.get("evidence_spans")),
                " ".join(str(tag) for tag in meta.get("tags", []) or meta.get("bucket_tags", []) or []),
                " ".join(str(item) for item in meta.get("domain", []) or meta.get("bucket_domain", []) or []),
            ]
        ).lower()
        return any(
            str(term or "").strip().lower() in fields
            for term in self.specific_query_terms(query)
            if self._is_non_relationship_specific_anchor(term)
        )

    def _relationship_background_off_intent(
        self,
        query: str,
        node: dict,
    ) -> bool:
        return (
            self._node_is_relationship_background(node)
            and not self._query_has_relationship_intent(query)
            and self._query_has_non_relationship_specific_terms(query)
            and not self._node_has_non_relationship_query_evidence(query, node)
        )

    def _is_probe_only_query(self, query: str) -> bool:
        text = str(query or "").strip().lower()
        if not text:
            return False
        probe_markers = (
            "试一下",
            "试试",
            "测试一下",
            "测试",
            "test",
            "try",
        )
        if not any(marker in text for marker in probe_markers):
            return False
        if any(re.search(r"[\u4e00-\u9fff]", term) for term in self.extract_entity_keywords(text)):
            return False
        recall_intent_markers = (
            "记得",
            "记忆",
            "想起",
            "回忆",
            "召回",
            "检索",
            "查一下",
            "找一下",
            "为什么",
            "原因",
            "remember",
            "recall",
            "memory",
            "search",
            "look up",
            "why",
        )
        return not any(marker in text for marker in recall_intent_markers)

    def _is_affect_only_query(self, query: str) -> bool:
        compact = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(query or "").lower())
        if not compact:
            return False
        stripped = compact
        for term in sorted(AFFECT_ONLY_QUERY_FILLERS, key=len, reverse=True):
            stripped = stripped.replace(term, "")
        if not stripped:
            return False
        return stripped in AFFECT_ONLY_QUERY_TERMS

    def _is_short_casual_only_query(self, query: str) -> bool:
        text = str(query or "").strip().lower()
        if not text:
            return False
        if any(marker in text for marker in AUTO_VAGUE_RECALL_MARKERS):
            return False
        if query_has_technical_recall_marker(text):
            return False
        compact = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)
        if not compact or len(compact) > 24:
            return False
        compact = re.sub(r"\d{1,4}$", "", compact)
        if not compact:
            return False
        if compact in SHORT_CASUAL_ONLY_TERMS:
            return True
        has_casual_signal = any(term in compact for term in SHORT_CASUAL_ONLY_TERMS)
        if not has_casual_signal:
            return False
        stripped = compact
        removable = (
            SHORT_CASUAL_ONLY_TERMS
            | SHORT_CASUAL_FILLER_TERMS
            | AFFECT_ONLY_QUERY_FILLERS
            | set(self.options.context_terms)
        )
        for term in sorted(removable, key=len, reverse=True):
            cleaned = re.sub(r"\s+", "", str(term or "").lower())
            if cleaned:
                stripped = stripped.replace(cleaned, "")
        return len(stripped) < 2

    def _is_affection_only_query(self, query: str) -> bool:
        text = str(query or "").strip().lower()
        if not text:
            return False
        if query_has_explicit_entity_marker(text) or query_has_technical_recall_marker(text):
            return False
        compact = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)
        if not compact:
            return False
        if not any(term in compact for term in AFFECTION_ONLY_SIGNAL_TERMS):
            return False
        stripped = compact
        removable = (
            AFFECTION_ONLY_SIGNAL_TERMS
            | AFFECTION_ONLY_FILLER_TERMS
            | SHORT_CASUAL_FILLER_TERMS
            | AFFECT_ONLY_QUERY_FILLERS
            | set(self.options.context_terms)
        )
        for term in sorted(removable, key=len, reverse=True):
            cleaned = re.sub(r"\s+", "", str(term or "").lower())
            if cleaned:
                stripped = stripped.replace(cleaned, "")
        return len(stripped) < 2

    def _short_taste_query_terms(self, query: str) -> list[str]:
        text = str(query or "").strip().lower()
        if not text:
            return []
        compact = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)
        compact = re.sub(r"\d{1,4}$", "", compact)
        if not compact or len(compact) > 12:
            return []
        stripped = compact
        removable = SHORT_CASUAL_FILLER_TERMS | (AFFECT_ONLY_QUERY_FILLERS - {"好"}) | set(self.options.context_terms)
        for term in sorted(removable, key=len, reverse=True):
            cleaned = re.sub(r"\s+", "", str(term or "").lower())
            if cleaned:
                stripped = stripped.replace(cleaned, "")
        return [term for term in SHORT_TASTE_QUERY_TERMS if stripped == term]

    def _fields_have_taste_evidence(
        self,
        taste_terms: list[str],
        fields: str,
        metadata_text: str,
    ) -> bool:
        text = str(fields or "").lower()
        meta = str(metadata_text or "").lower()
        has_food_metadata = any(term in meta for term in TASTE_METADATA_TERMS | TASTE_OBJECT_TERMS)
        for term in taste_terms:
            if term == "好吃":
                pattern = r"(?<!好)好吃"
            elif term == "好喝":
                pattern = r"(?<!好)好喝"
            else:
                pattern = re.escape(term)
            for match in re.finditer(pattern, text):
                start, end = match.span()
                window = text[max(0, start - 18): min(len(text), end + 18)]
                if "隔壁好吃" in window or "隔壁好喝" in window:
                    continue
                if has_food_metadata and any(obj in window for obj in TASTE_OBJECT_TERMS | TASTE_METADATA_TERMS):
                    return True
                if any(obj in window for obj in TASTE_OBJECT_TERMS):
                    return True
                if re.search(r"觉得.{1,16}" + pattern, window):
                    return True
        return False

    def extract_entity_keywords(self, query: str) -> list[str]:
        raw = str(query or "").strip()
        if not raw:
            return []
        keywords: list[str] = []
        seen: set[str] = set()
        strong_keys: set[str] = set()

        def add(value: object, *, strong: bool = False) -> None:
            cleaned = self._normalize_entity_keyword(value)
            if not cleaned or not self._entity_keyword_allowed(cleaned, strong=strong):
                return
            key = self._compact_entity_keyword(cleaned)
            if not key or key in seen:
                return
            seen.add(key)
            if strong:
                strong_keys.add(key)
            keywords.append(cleaned)

        for phrase in extract_protected_phrases(raw):
            add(phrase, strong=True)
        for match in ENTITY_QUOTED_RE.finditer(raw):
            add(match.group(1), strong=True)
        for match in ENTITY_VERSION_RE.finditer(raw):
            add(match.group(0), strong=True)
        for match in ENTITY_ENGLISH_RE.finditer(raw):
            add(match.group(0), strong=True)
        for match in ENTITY_NUMBER_RE.finditer(raw):
            add(match.group(0), strong=True)
        for match in re.finditer(r"[A-Za-z0-9_.:-]*[\u4e00-\u9fff]+[A-Za-z0-9_.:-]+|[A-Za-z0-9_.:-]+[\u4e00-\u9fff]+[A-Za-z0-9_.:-]*", raw):
            mixed = match.group(0)
            stripped = self._strip_entity_shell(mixed)
            value = mixed if stripped == self._compact_entity_keyword(mixed) else stripped
            add(value, strong=True)

        for word, flag in self._posseg_words(raw):
            if (
                flag in ENTITY_KEYWORD_POS_TAGS
                or any(str(flag or "").startswith(prefix) for prefix in ENTITY_KEYWORD_POS_PREFIXES)
            ):
                add(word, strong=True)
                for expanded in self._expand_entity_title_suffixes(raw, word):
                    add(expanded, strong=True)

        for span in re.findall(r"[\u4e00-\u9fff]{2,16}", raw):
            candidate = self._strip_entity_shell(span)
            if 2 <= len(candidate) <= 8 and not self._entity_candidate_has_verb_blocker(candidate):
                add(candidate)

        return self._dedupe_entity_keywords(keywords, strong_keys=strong_keys)

    def _dedupe_entity_keywords(self, values: list[str], *, strong_keys: set[str] | None = None) -> list[str]:
        strong_keys = strong_keys or set()
        pairs: list[tuple[str, str]] = []
        seen: set[str] = set()
        for value in values:
            cleaned = self._normalize_entity_keyword(value)
            key = self._compact_entity_keyword(cleaned)
            if not cleaned or not key or key in seen:
                continue
            seen.add(key)
            pairs.append((cleaned, key))
        output: list[str] = []
        for cleaned, key in pairs:
            if key in strong_keys:
                contained_by_longer = any(
                    other_key != key and other_key in strong_keys and key in other_key
                    for _other, other_key in pairs
                )
                noisy_extension_of_strong = False
            else:
                contained_by_longer = any(
                    other_key != key and other_key not in strong_keys and key in other_key
                    for _other, other_key in pairs
                )
                noisy_extension_of_strong = any(
                    other_key != key and other_key in strong_keys and other_key in key
                    for _other, other_key in pairs
                )
            if contained_by_longer or noisy_extension_of_strong:
                continue
            output.append(cleaned)
        return output

    @staticmethod
    def _posseg_words(text: str) -> list[tuple[str, str]]:
        try:
            import jieba.posseg as pseg
        except Exception:
            return []
        try:
            return [(str(item.word), str(item.flag)) for item in pseg.cut(text)]
        except Exception:
            return []

    @staticmethod
    def _normalize_entity_keyword(value: object) -> str:
        cleaned = str(value or "").strip().strip("\"'`“”‘’「」『』《》")
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip("，。！？、,.!?:：;；~～♡❤♥（）()[]【】")

    @staticmethod
    def _compact_entity_keyword(value: object) -> str:
        return re.sub(r"[^0-9a-z\u4e00-\u9fff_.:-]+", "", str(value or "").strip().lower())

    def _entity_keyword_allowed(self, value: str, *, strong: bool = False) -> bool:
        compact = self._compact_entity_keyword(value)
        if not compact:
            return False
        if compact in ENTITY_KEYWORD_STOP_TERMS or self._is_recall_context_term(compact):
            return False
        if compact in self.ai_reaction_names:
            return False
        if ENTITY_VERSION_RE.fullmatch(compact) or ENTITY_NUMBER_RE.fullmatch(compact):
            return True
        if re.fullmatch(r"[a-z][a-z0-9_.:/-]{1,}", compact):
            return compact not in ENTITY_KEYWORD_STOP_TERMS
        if re.fullmatch(r"[\u4e00-\u9fff]+", compact):
            residue = self._strip_entity_shell(compact)
            if len(residue) < 2 or len(residue) > 8:
                return False
            if residue in ENTITY_KEYWORD_STOP_TERMS or self._is_affect_only_query(residue):
                return False
            if not strong and self._entity_candidate_has_verb_blocker(residue):
                return False
            return True
        return True

    def _strip_entity_shell(self, value: object) -> str:
        residue = self._compact_entity_keyword(value)
        for term in sorted(ENTITY_KEYWORD_SHELL_TERMS, key=len, reverse=True):
            cleaned = self._compact_entity_keyword(term)
            if cleaned:
                residue = residue.replace(cleaned, "")
        residue = re.sub(r"[我你他她它的是了嘛吗呢啊呀欸诶吧哈嗯呜]+", "", residue)
        return residue

    @staticmethod
    def _entity_candidate_has_verb_blocker(value: str) -> bool:
        text = str(value or "").strip()
        if not text:
            return False
        if text in ENTITY_KEYWORD_VERB_BLOCKERS:
            return True
        chars = [char for char in text if re.fullmatch(r"[\u4e00-\u9fff]", char)]
        return bool(chars and len(chars) == len(text) and all(char in ENTITY_KEYWORD_VERB_BLOCKERS for char in chars))

    def _expand_entity_title_suffixes(self, raw: str, entity: str) -> list[str]:
        compact_raw = self._compact_entity_keyword(raw)
        compact_entity = self._compact_entity_keyword(entity)
        if not compact_raw or not compact_entity:
            return []
        output: list[str] = []
        start = 0
        while True:
            index = compact_raw.find(compact_entity, start)
            if index < 0:
                break
            tail = compact_raw[index + len(compact_entity):]
            for suffix in sorted(ENTITY_KEYWORD_TITLE_SUFFIXES, key=len, reverse=True):
                compact_suffix = self._compact_entity_keyword(suffix)
                if compact_suffix and tail.startswith(compact_suffix):
                    output.append(entity + suffix)
                    break
            start = index + max(1, len(compact_entity))
        return output

    def locatable_query_terms(self, query: str) -> list[str]:
        raw = str(query or "").strip()
        if not raw:
            return []
        return list(self._locatable_query_terms_cached(raw))

    @lru_cache(maxsize=512)
    def _locatable_query_terms_cached(self, raw: str) -> tuple[str, ...]:
        output: list[str] = []
        seen: set[str] = set()
        content_term_keys = {
            self._compact_entity_keyword(term)
            for term in content_terms_for_query(raw, self.options)
            if self._compact_entity_keyword(term)
        }

        def add(value: object, *, force: bool = False) -> None:
            cleaned = self._normalize_locatable_query_term(value)
            if not cleaned:
                return
            if not force and not self._locatable_query_term_allowed(cleaned):
                return
            key = self._compact_entity_keyword(cleaned)
            if not key or key in seen:
                return
            seen.add(key)
            output.append(cleaned)

        for phrase in extract_protected_phrases(raw):
            add(phrase)
        for match in ENTITY_QUOTED_RE.finditer(raw):
            add(match.group(1))
        for match in ENTITY_VERSION_RE.finditer(raw):
            add(match.group(0))
        for match in ENTITY_ENGLISH_RE.finditer(raw):
            add(match.group(0))
        for match in ENTITY_NUMBER_RE.finditer(raw):
            add(match.group(0))
        for match in re.finditer(
            r"[A-Za-z0-9_.:-]*[\u4e00-\u9fff]+[A-Za-z0-9_.:-]+|[A-Za-z0-9_.:-]+[\u4e00-\u9fff]+[A-Za-z0-9_.:-]*",
            raw,
        ):
            add(match.group(0))

        for term in self._pos_structural_locatable_terms(raw, content_term_keys=content_term_keys):
            add(term)

        specific_terms = self.specific_query_terms(raw)
        compact_raw = self._compact_entity_keyword(raw)
        structural_terms = list(output)
        for structural_term in structural_terms:
            structural_key = self._compact_entity_keyword(structural_term)
            for term in specific_terms:
                term_key = self._compact_entity_keyword(term)
                if not term_key or term_key == structural_key or term_key not in structural_key:
                    continue
                if self._contained_structural_subterm_allowed(term):
                    add(term)
        for left, right in product(structural_terms, specific_terms):
            left_text = str(left or "").strip()
            right_text = str(right or "").strip()
            if not left_text or not right_text or left_text == right_text:
                continue
            if right_text not in LOCATABLE_COMPOUND_SUFFIX_TERMS:
                continue
            combined = f"{left_text}{right_text}"
            if self._compact_entity_keyword(combined) in compact_raw:
                add(combined)
        if query_has_facet(raw, "embodiment", self.options):
            add("身体")
            add("具身")
        for term in self._relation_axis_locatable_terms(raw, specific_terms):
            add(term)
        for term in self._event_place_locatable_terms(raw, specific_terms):
            add(term, force=True)

        return tuple(output[:8])

    def _pos_structural_locatable_terms(
        self,
        raw: str,
        *,
        content_term_keys: set[str],
    ) -> list[str]:
        tokens = [
            (self._normalize_entity_keyword(word), str(flag or ""))
            for word, flag in self._posseg_words(raw)
            if self._normalize_entity_keyword(word)
        ]
        output: list[str] = []

        def add(value: object) -> None:
            cleaned = self._normalize_entity_keyword(value)
            if cleaned:
                output.append(cleaned)

        for word, flag in tokens:
            if flag in ENTITY_KEYWORD_POS_TAGS or any(flag.startswith(prefix) for prefix in ENTITY_KEYWORD_POS_PREFIXES):
                word = self._strip_leading_axis_conjunction(word, content_term_keys=content_term_keys)
                add(word)
                for expanded in self._expand_entity_title_suffixes(raw, word):
                    add(expanded)
                continue
            if self._standalone_locatable_noun(word, flag, content_term_keys=content_term_keys):
                add(word)

        for index in range(len(tokens)):
            for width in (2, 3):
                window = tokens[index: index + width]
                if len(window) != width:
                    continue
                if not all(self._compound_locatable_token_allowed(word, flag) for word, flag in window):
                    continue
                combined = "".join(word for word, _flag in window)
                combined_key = self._compact_entity_keyword(combined)
                suffix = next(
                    (
                        suffix
                        for suffix in LOCATABLE_COMPOUND_SUFFIX_TERMS
                        if combined_key.endswith(self._compact_entity_keyword(suffix))
                    ),
                    "",
                )
                if not suffix:
                    continue
                add(combined)
                for word, _flag in window:
                    add(word)

        return self._dedupe_entity_keywords(output)

    def _strip_leading_axis_conjunction(self, word: str, *, content_term_keys: set[str]) -> str:
        key = self._compact_entity_keyword(word)
        if len(key) <= 2:
            return word
        for prefix in ("和", "与"):
            if key.startswith(prefix):
                rest = key[len(prefix):]
                if rest in content_term_keys:
                    return rest
        return word

    def _standalone_locatable_noun(
        self,
        word: str,
        flag: str,
        *,
        content_term_keys: set[str],
    ) -> bool:
        key = self._compact_entity_keyword(word)
        if not key or key not in content_term_keys:
            return False
        if key in LOCATABLE_GENERIC_TERMS or self._is_recall_context_term(key):
            return False
        if not (flag == "eng" or flag.startswith("n") or flag in {"s"}):
            return False
        if re.fullmatch(r"[a-z][a-z0-9_.:/-]{2,}", key):
            return True
        if re.search(r"\d", key) and re.search(r"[a-z\u4e00-\u9fff]", key):
            return True
        if re.fullmatch(r"小[\u4e00-\u9fffA-Za-z0-9]{1,4}", key):
            return True
        if re.fullmatch(r"[\u4e00-\u9fff]{2,6}", key):
            return not self._entity_candidate_has_verb_blocker(key)
        return False

    def _compound_locatable_token_allowed(self, word: str, flag: str) -> bool:
        key = self._compact_entity_keyword(word)
        if not key or key in LOCATABLE_GENERIC_TERMS or self._is_recall_context_term(key):
            return False
        if flag == "eng":
            return True
        if flag.startswith("n") or flag in {"s"}:
            return True
        return bool(re.search(r"\d", key) and re.search(r"[a-z\u4e00-\u9fff]", key))

    def _relation_axis_locatable_terms(self, raw: str, specific_terms: list[str]) -> list[str]:
        output: list[str] = []
        if not self._query_has_axis_relation_marker(raw):
            terms = []
        else:
            terms = specific_terms
        for term in terms:
            key = self._compact_entity_keyword(term)
            if not key:
                continue
            if key in LOCATABLE_GENERIC_TERMS or self._is_recall_context_term(key):
                continue
            if re.fullmatch(r"[一二三四五六七八九十百千万两0-9]+年(?:后)?", key):
                output.append(term)
                continue
            if key in {"承诺", "约定", "未来"}:
                output.append(term)
        if not output:
            for match in re.finditer(r"[一二三四五六七八九十百千万两0-9]+年(?:后)?", str(raw or "")):
                value = match.group(0)
                output.append(value[:-1] if value.endswith("后") else value)
        return output

    def _event_place_locatable_terms(self, raw: str, specific_terms: list[str]) -> list[str]:
        compact = self._compact_entity_keyword(raw)
        if not any(self._compact_entity_keyword(marker) in compact for marker in EVENT_PLACE_QUERY_MARKERS):
            return []
        output: list[str] = []
        for term in specific_terms:
            key = self._compact_entity_keyword(term)
            if key in EVENT_PLACE_LOCATABLE_TERMS:
                output.append(term)
        return output

    def _contained_structural_subterm_allowed(self, value: object) -> bool:
        key = self._compact_entity_keyword(value)
        if not key:
            return False
        if key in LOCATABLE_GENERIC_TERMS or self._is_recall_context_term(key):
            return False
        if len(key) < 2:
            return False
        if re.fullmatch(r"[a-z][a-z0-9_.:/-]{2,}", key):
            return True
        if re.search(r"\d", key) and re.search(r"[a-z\u4e00-\u9fff]", key):
            return True
        return bool(re.fullmatch(r"[\u4e00-\u9fff]{2,8}", key))

    def _normalize_locatable_query_term(self, value: object) -> str:
        cleaned = self._normalize_entity_keyword(value)
        if not cleaned:
            return ""
        compact = self._compact_entity_keyword(cleaned)
        if not compact:
            return ""

        strip_terms = set(LOCATABLE_STRIP_TERMS)
        strip_terms.update(str(term or "") for term in self.options.context_terms)
        strip_terms.update(str(term or "") for term in self.ai_reaction_names)
        strip_terms.update(
            str(term or "")
            for term in (
                identity_names().get("ai_name"),
                identity_names().get("user_name"),
                identity_names().get("user_display_name"),
                *(identity_names().get("user_aliases") or []),
            )
        )
        for term in sorted(strip_terms, key=lambda item: len(self._compact_entity_keyword(item)), reverse=True):
            fragment = self._compact_entity_keyword(term)
            if fragment:
                compact = compact.replace(fragment, "")

        compact = re.sub(r"[我你他她它的是了啦呢啊呀嘛吗吧欸诶得]+", "", compact)
        compact = self._strip_locatable_question_tail(compact)
        if re.fullmatch(r"[a-z][a-z0-9_.:/-]{1,}", compact):
            for match in ENTITY_ENGLISH_RE.finditer(cleaned):
                if self._compact_entity_keyword(match.group(0)) == compact:
                    return match.group(0)
        return compact

    @classmethod
    def _strip_locatable_question_tail(cls, compact: str) -> str:
        text = str(compact or "").strip()
        if not text:
            return ""
        tails = sorted(
            (cls._compact_entity_keyword(term) for term in LOCATABLE_QUESTION_TAIL_TERMS),
            key=len,
            reverse=True,
        )
        changed = True
        while changed:
            changed = False
            for tail in tails:
                if tail and text.endswith(tail) and len(text) > len(tail):
                    text = text[: -len(tail)].strip("的")
                    changed = True
                    break
        return text

    def _locatable_query_term_allowed(self, value: str) -> bool:
        key = self._compact_entity_keyword(value)
        if not key:
            return False
        if key in LOCATABLE_GENERIC_TERMS or self._is_recall_context_term(key):
            return False
        if key in self.ai_reaction_names:
            return False
        if self._is_affect_only_query(key) or self._is_affection_only_query(key):
            return False
        if ENTITY_VERSION_RE.fullmatch(key) or ENTITY_NUMBER_RE.fullmatch(key):
            return True
        if re.fullmatch(r"[a-z][a-z0-9_.:/-]{2,}", key):
            return key not in LOCATABLE_GENERIC_TERMS
        if re.search(r"\d", key) and re.search(r"[a-z\u4e00-\u9fff]", key):
            return True
        if re.fullmatch(r"[\u4e00-\u9fff]+", key):
            if len(key) < 2 or len(key) > 18:
                return False
            if key in LOCATABLE_GENERIC_TERMS:
                return False
            if all(char in LOCATABLE_GENERIC_TERMS for char in key):
                return False
            return True
        return bool(re.search(r"[\u4e00-\u9fffA-Za-z0-9]", key))

    def specific_query_terms(self, query: str) -> list[str]:
        raw = str(query or "")
        return list(self._specific_query_terms_cached(raw))

    def _topic_evidence_terms(self, query: str) -> list[str]:
        terms: list[str] = []
        for term in self.specific_query_terms(query):
            key = self._compact_entity_keyword(term)
            if not key:
                continue
            if re.fullmatch(r"[\u4e00-\u9fff]", key):
                continue
            terms.append(term)
        return terms

    @lru_cache(maxsize=512)
    def _specific_query_terms_cached(self, raw: str) -> tuple[str, ...]:
        terms = list(content_terms_for_query(raw, self.options))
        topic_key = recall_topic_query(raw, self.options)
        allow_single_cjk_terms = {
            str(term or "").strip()
            for term in content_terms_for_query(topic_key, self.options)
            if re.fullmatch(r"[\u4e00-\u9fff]", str(term or "").strip())
        }
        terms.extend(re.findall(r"\d+(?:\.\d+)+", raw))
        terms.extend(re.findall(r"[A-Za-z]+[A-Za-z0-9_.:-]*\d[A-Za-z0-9_.:-]*", raw))
        kept = []
        seen = set()
        for term in terms:
            cleaned = str(term or "").strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            if key in WEAK_RECALL_TOPIC_TERMS:
                continue
            if key in RECALL_SYSTEM_META_TERMS:
                continue
            if self._is_recall_context_term(cleaned):
                continue
            if re.fullmatch(r"[a-z0-9_.:-]+", key) and len(key) < 3 and not re.fullmatch(r"\d+(?:\.\d+)+", key):
                continue
            if (
                re.fullmatch(r"[\u4e00-\u9fff]+", cleaned)
                and len(cleaned) < 2
                and cleaned not in allow_single_cjk_terms
            ):
                continue
            if any(_term_subsumes(existing.lower(), key) for existing in kept):
                continue
            kept = [existing for existing in kept if not _term_subsumes(key, existing.lower())]
            seen = {existing.lower() for existing in kept}
            seen.add(key)
            kept.append(cleaned)
        return tuple(kept)

    def moment_has_topic_evidence(self, query: str, moment: dict) -> bool:
        taste_terms = self._short_taste_query_terms(query)
        terms = self._topic_evidence_terms(query)
        if not terms:
            return False
        meta = moment.get("metadata", {}) if isinstance(moment.get("metadata"), dict) else {}
        fields = " ".join(
            [
                str(moment.get("text") or ""),
                str(meta.get("annotation_summary") or ""),
                _evidence_spans_text(meta.get("evidence_spans")),
                str(meta.get("bucket_name") or ""),
                " ".join(str(tag) for tag in (meta.get("bucket_tags") or []) if str(tag).strip()),
                " ".join(str(item) for item in (meta.get("bucket_domain") or []) if str(item).strip()),
            ]
        ).lower()
        if taste_terms:
            metadata_text = " ".join(
                [
                    str(meta.get("bucket_name") or ""),
                    " ".join(str(tag) for tag in (meta.get("bucket_tags") or []) if str(tag).strip()),
                    " ".join(str(item) for item in (meta.get("bucket_domain") or []) if str(item).strip()),
                ]
            ).lower()
            return self._fields_have_taste_evidence(taste_terms, fields, metadata_text)
        return any(term.lower() in fields for term in terms)

    def bucket_has_topic_evidence(self, query: str, bucket: dict) -> bool:
        taste_terms = self._short_taste_query_terms(query)
        terms = self._topic_evidence_terms(query)
        if not terms:
            return False
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        fields = " ".join(
            [
                _content_without_context_only_sections(str(bucket.get("content") or "")),
                str(meta.get("name") or ""),
                str(meta.get("annotation_summary") or ""),
                _evidence_spans_text(meta.get("evidence_spans")),
                " ".join(str(tag) for tag in (meta.get("tags") or []) if str(tag).strip()),
                " ".join(str(item) for item in (meta.get("domain") or []) if str(item).strip()),
            ]
        ).lower()
        if taste_terms:
            metadata_text = " ".join(
                [
                    str(meta.get("name") or ""),
                    " ".join(str(tag) for tag in (meta.get("tags") or []) if str(tag).strip()),
                    " ".join(str(item) for item in (meta.get("domain") or []) if str(item).strip()),
                ]
            ).lower()
            return self._fields_have_taste_evidence(taste_terms, fields, metadata_text)
        return any(term.lower() in fields for term in terms)

    def node_has_topic_evidence(self, query: str, node: dict) -> bool:
        if "bucket_id" in node or node.get("moment_id"):
            return self.moment_has_topic_evidence(query, node)
        return self.bucket_has_topic_evidence(query, node)

    def allows_moment_context(
        self,
        query: str,
        moment: dict,
        *,
        allow_body_chain: bool = False,
    ) -> bool:
        if not self.should_enforce_topic_evidence(query, allow_body_chain=allow_body_chain):
            return True
        return self.moment_has_topic_evidence(query, moment)

    def allows_bucket_context(
        self,
        query: str,
        bucket: dict,
        *,
        allow_body_chain: bool = False,
    ) -> bool:
        if not self.should_enforce_topic_evidence(query, allow_body_chain=allow_body_chain):
            return True
        return self.bucket_has_topic_evidence(query, bucket)

    def has_strong_score(
        self,
        *,
        semantic_score: float | None = None,
        rerank_score: float | None = None,
    ) -> bool:
        return (
            _safe_float(semantic_score, 0.0) >= self.semantic_threshold
            or _safe_float(rerank_score, 0.0) >= self.rerank_threshold
        )

    def assess(
        self,
        query: str,
        node: dict,
        *,
        query_plan: RecallQueryPlan | None = None,
        has_topic_evidence: bool | None = None,
        semantic_score: float | None = None,
        rerank_score: float | None = None,
        high_confidence_edge: bool = False,
        context_only: bool = False,
        auto: bool = False,
        semantic_entry_routed: bool = False,
    ) -> RecallPolicyDecision:
        query_plan = query_plan or self.plan_query(query)
        if has_topic_evidence is None:
            has_topic_evidence = self.node_has_topic_evidence(query, node)
        legacy_auto_vague_would_suppress = bool(
            auto and query_plan.auto_too_vague and not high_confidence_edge
        )
        auto_too_vague = bool(
            legacy_auto_vague_would_suppress and not semantic_entry_routed
        )
        debug = {
            "requires_topic_evidence": query_plan.requires_topic_evidence,
            "has_topic_evidence": bool(has_topic_evidence),
            "specific_query_terms": list(query_plan.specific_terms),
            "short_taste_query_terms": list(query_plan.short_taste_terms),
            "semantic_score": _maybe_float(semantic_score),
            "rerank_score": _maybe_float(rerank_score),
            "high_confidence_edge": bool(high_confidence_edge),
            "context_only": bool(context_only),
            "auto": bool(auto),
            "auto_too_vague": bool(auto_too_vague),
            "semantic_entry_routed": bool(semantic_entry_routed),
            "legacy_auto_vague_would_suppress": legacy_auto_vague_would_suppress,
        }

        if auto_too_vague:
            return RecallPolicyDecision(
                admit_direct=False,
                admit_diffused=False,
                seed_allowed=False,
                reason="auto_vague_query_without_topic",
                suppressed=True,
                debug=debug,
            )

        if context_only:
            return RecallPolicyDecision(
                admit_direct=False,
                admit_diffused=False,
                seed_allowed=False,
                reason="context_only_temperature_moment",
                suppressed=True,
                debug=debug,
            )

        base = recall_admission_decision(
            query,
            node,
            self.options,
            semantic_score=semantic_score,
            rerank_score=rerank_score,
            high_confidence_edge=high_confidence_edge,
            semantic_threshold=self.semantic_threshold,
            rerank_threshold=self.rerank_threshold,
        )
        debug["base_reason"] = base.reason

        if not base.admit:
            return RecallPolicyDecision(
                admit_direct=False,
                admit_diffused=False,
                seed_allowed=False,
                reason=base.reason,
                suppressed=True,
                debug=debug,
            )

        if self._relationship_background_off_intent(
            query,
            node,
        ):
            debug["relationship_background_off_intent"] = True
            return RecallPolicyDecision(
                admit_direct=False,
                admit_diffused=False,
                seed_allowed=False,
                reason="relationship_background_without_query_topic_evidence",
                suppressed=True,
                debug=debug,
            )

        if (
            debug["short_taste_query_terms"]
            and not has_topic_evidence
            and not self.has_strong_score(
                semantic_score=semantic_score,
                rerank_score=rerank_score,
            )
        ):
            return RecallPolicyDecision(
                admit_direct=False,
                admit_diffused=False,
                seed_allowed=False,
                reason="short_taste_query_without_taste_evidence",
                suppressed=True,
                debug=debug,
            )

        if (
            debug["requires_topic_evidence"]
            and not has_topic_evidence
            and not self.has_strong_score(
                semantic_score=semantic_score,
                rerank_score=rerank_score,
            )
            and not high_confidence_edge
        ):
            return RecallPolicyDecision(
                admit_direct=False,
                admit_diffused=False,
                seed_allowed=False,
                reason="query_topic_evidence_missing",
                suppressed=True,
                debug=debug,
            )

        return RecallPolicyDecision(
            admit_direct=True,
            admit_diffused=True,
            seed_allowed=True,
            reason=base.reason,
            suppressed=False,
            debug=debug,
        )


def is_context_only_section(section: Any) -> bool:
    return str(section or "") in CONTEXT_ONLY_SECTIONS


def _evidence_spans_text(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    parts = []
    for item in value:
        if isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            if text:
                parts.append(text)
        elif isinstance(item, str) and item.strip():
            parts.append(item.strip())
    return " ".join(parts)


def _content_without_context_only_sections(content: str) -> str:
    lines = str(content or "").splitlines()
    kept: list[str] = []
    skip_until_level = 0
    for line in lines:
        match = MARKDOWN_HEADING_RE.match(line)
        if match:
            level = len(match.group(1))
            raw_heading = match.group(2).strip()
            if skip_until_level and level > skip_until_level:
                continue
            skip_until_level = 0
            if _context_only_heading(raw_heading):
                skip_until_level = level
                continue
        if skip_until_level:
            continue
        kept.append(line)
    return "\n".join(kept)


def _context_only_heading(heading: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(heading or "").strip().lower())
    normalized = normalized.strip("：: -_")
    normalized = re.sub(r"^\d+[.、]\s*", "", normalized)
    normalized = normalized.replace("-", "_")
    return CONTEXT_ONLY_SECTION_ALIASES.get(normalized, normalized) in CONTEXT_ONLY_SECTIONS


def diffusion_seed_topic_term_has_specific_residue(term: object) -> bool:
    residue = re.sub(
        r"[^0-9a-z\u4e00-\u9fff_.:-]+",
        "",
        str(term or "").strip().lower(),
    )
    if not residue:
        return False
    for generic in sorted(DIFFUSION_SEED_GENERIC_TOPIC_FRAGMENTS, key=len, reverse=True):
        residue = residue.replace(generic, "")
    return bool(residue.strip())


def _term_subsumes(container: str, contained: str) -> bool:
    if container == contained:
        return True
    if not container or not contained:
        return False
    if not re.search(r"\d", contained):
        return False
    return contained in container


def _maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any, default: float) -> float:
    number = _maybe_float(value)
    return default if number is None else number
