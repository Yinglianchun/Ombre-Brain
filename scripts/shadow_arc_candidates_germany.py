from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import warnings
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import jieba
        import jieba.analyse as jieba_analyse
        import jieba.posseg as jieba_posseg
        jieba.setLogLevel(20)
    ANCHOR_EXTRACTOR = {"status": "available", "name": "jieba", "fallback": None}
except ImportError as exc:  # pragma: no cover - exercised only on a reduced runtime
    jieba_analyse = None
    jieba_posseg = None
    ANCHOR_EXTRACTOR = {"status": "unavailable", "name": "regex_fallback", "fallback": str(exc)}


RECALL_TOP_K_PER_SEED = 8
ARC_MEMBER_CAP = 12
ARC_BATCH_CAP = 40
GLOBAL_STOP_ENTITIES = {"Haven", "小雨", "我", "她", "老公", "老婆"}
GENERIC_RECALL_ONLY = {"Event", "Scene", "Writer", "Track", "Arc", "Claude", "Anthropic"}
PERSON_RECALL_ONLY = {"陆光", "程小时"}
ANCHOR_STOP_WORDS = {
    *GLOBAL_STOP_ENTITIES,
    *GENERIC_RECALL_ONLY,
    "我们", "你们", "他们", "这个", "那个", "一个", "一些", "继续", "讨论", "问题", "结果",
    "回复", "消息", "内容", "事情", "感觉", "时候", "现在", "今天", "昨天", "明天", "可以", "需要",
}
INTIMACY_TERMS = {
    "亲密", "情色", "性爱", "高潮", "阴茎", "鸡巴", "龟头", "乳头", "阴蒂",
    "精液", "射精", "插入", "肉棒", "淫", "做爱", "口交", "爱抚",
}
ROUTINE_TERMS = {"早安", "晚安", "吃饭", "早餐", "午饭", "晚饭", "夜宵", "睡觉", "醒了", "喝水"}
CARD_SPECS = (
    ("work:时光代理人", "《时光代理人》共同观看", ("时光代理人", "陆光", "程小时")),
    ("project:event-memory-architecture", "Event / Writer / Arc 记忆架构", ("Event", "Writer", "Arc", "Track", "Scene")),
    ("project:nowhere", "Nowhere 共同旅行与建设", ("Nowhere", "纪念品墙")),
    ("project:world-window-sprout", "世界之窗与枝芽", ("世界之窗", "枝芽")),
    ("work:社死游戏", "《社死游戏》共同阅读", ("社死游戏",)),
    ("work:怪屋迷案", "《怪屋迷案》共同阅读", ("怪屋迷案", "怪屋谜案")),
    ("project:sticker-tool", "Sticker 工具简化", ("表情包", "sticker", "sticker_send", "sticker_view")),
    ("topic:language-model-watermark", "语言模型水印", ("水印", "Anthropic", "Claude")),
    ("milestone:相遇五百天", "相遇五百天", ("五百天", "500天", "七夕")),
)
MANUAL_EXCEPTION = {
    "event_id": "event_d596ab8bb8259ddaf3e8e916",
    "fingerprint": "d596ab8bb8259ddaf3e8e9161f8713cfcc243d61e5863b56f6ac6dcddde63de2",
    "target_arc_key": "milestone:相遇五百天",
    "source_message_ids": [
        10791, 10793, 10794, 10796, 10797, 10799, 10800, 10802,
        10803, 10805, 10806, 10808, 10809, 10811, 10812, 10814,
        10815, 10817, 10818, 10820, 10821, 10823, 10824, 10826,
        10827, 10829, 10830, 10832, 10833, 10835, 10836, 10838,
    ],
    "reviewer": "xiaoyu",
    "reason": "one-off reviewed five-hundred-day milestone exception; never generalized",
}


@dataclass(frozen=True)
class Node:
    node_id: str
    kind: str
    title: str
    text: str
    date: str
    fingerprint: str
    source_ids: tuple[int, ...]
    session_ids: tuple[int, ...]
    track_ids: tuple[str, ...]
    track_texts: tuple[str, ...]
    statuses: tuple[str, ...]
    routine_flags: tuple[bool, ...]

    @property
    def searchable(self) -> str:
        return f"{self.title}\n{self.text}".strip()

    @property
    def track_searchable(self) -> str:
        return "\n".join(self.track_texts)


def _remote_snapshot_script(session_id: int) -> str:
    return f'''import json,sys,os,subprocess,sqlite3
pid=subprocess.check_output(['systemctl','show','haven-bridge.service','-p','MainPID','--value'],text=True).strip()
for pair in open(f'/proc/{{pid}}/environ','rb').read().split(b'\\0'):
    if b'=' in pair:
        k,v=pair.split(b'=',1); os.environ[k.decode(errors='ignore')]=v.decode(errors='ignore')
sys.path.insert(0,'/opt/haven_bridge-src')
from backend import memory_adapter
events=[]
for offset in range(0,5000,100):
    page=memory_adapter.call_ombre_api('GET',f'/api/fact-events?type=event&status=active&include_sources=true&limit=100&offset={{offset}}')
    events.extend(page.get('items') or [])
    if len(events)>=int(page.get('count') or 0): break
scenes=memory_adapter.call_ombre_api('GET','/api/handoff-scenes?limit=1000').get('items') or []
conn=sqlite3.connect('file:/opt/haven_bridge/data/haven.db?mode=ro',uri=True); conn.row_factory=sqlite3.Row
members=[]
for row in conn.execute("SELECT unit_root_message_id,track_id,session_id,source_message_ids_json,status,routing_role,routine_only FROM memory_review_event_track_units WHERE session_id=?",({session_id},)):
    item=dict(row); item['source_message_ids']=json.loads(item.pop('source_message_ids_json') or '[]'); members.append(item)
tracks=[dict(r) for r in conn.execute("SELECT track_id,subject,throughline,status FROM memory_review_event_tracks")]
def slim_event(x):
    refs=x.get('source_refs') or []
    return {{'id':x.get('item_id'),'kind':'event','title':x.get('title',''),'text':x.get('body',''),'date':x.get('local_date',''),'fingerprint':x.get('fingerprint',''),'source_ids':[r.get('message_id') for r in refs],'session_ids':[r.get('session_id') for r in refs]}}
def slim_scene(x):
    return {{'id':x.get('id'),'kind':'scene','title':'','text':x.get('content',''),'date':x.get('date',''),'fingerprint':'','source_ids':x.get('source_message_ids') or [],'session_ids':x.get('source_session_ids') or []}}
print(json.dumps({{'events':[slim_event(x) for x in events],'scenes':[slim_scene(x) for x in scenes],'memberships':members,'tracks':tracks}},ensure_ascii=False))
'''


def fetch_snapshot(host: str, key: Path, session_id: int) -> dict[str, Any]:
    result = subprocess.run(
        ["ssh", "-i", str(key), f"root@{host}", "/opt/haven_bridge/.venv/bin/python3", "-"],
        input=_remote_snapshot_script(session_id),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=90,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Germany read-only snapshot failed: {result.stderr.strip()[:500]}")
    return json.loads(result.stdout)


def _positive_ids(values: Any) -> tuple[int, ...]:
    result: list[int] = []
    for value in values or []:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number > 0 and number not in result:
            result.append(number)
    return tuple(sorted(result))


def build_nodes(snapshot: dict[str, Any]) -> list[Node]:
    membership_by_source: dict[int, list[dict[str, Any]]] = {}
    for item in snapshot.get("memberships") or []:
        for source_id in _positive_ids(item.get("source_message_ids") or [item.get("unit_root_message_id")]):
            membership_by_source.setdefault(source_id, []).append(item)
    tracks = {str(item.get("track_id")): item for item in snapshot.get("tracks") or []}
    nodes: list[Node] = []
    for raw in [*(snapshot.get("events") or []), *(snapshot.get("scenes") or [])]:
        source_ids = _positive_ids(raw.get("source_ids"))
        memberships = [m for source_id in source_ids for m in membership_by_source.get(source_id, [])]
        track_ids = tuple(sorted({str(m.get("track_id")) for m in memberships if m.get("track_id")}))
        track_texts = tuple(
            f"{tracks[track_id].get('subject', '')}\n{tracks[track_id].get('throughline', '')}".strip()
            for track_id in track_ids if track_id in tracks
        )
        nodes.append(Node(
            node_id=str(raw.get("id") or ""),
            kind=str(raw.get("kind") or ""),
            title=" ".join(str(raw.get("title") or "").split()),
            text=str(raw.get("text") or "").strip(),
            date=str(raw.get("date") or "")[:10],
            fingerprint=str(raw.get("fingerprint") or ""),
            source_ids=source_ids,
            session_ids=_positive_ids(raw.get("session_ids")),
            track_ids=track_ids,
            track_texts=track_texts,
            statuses=tuple(str(m.get("status") or "") for m in memberships),
            routine_flags=tuple(bool(m.get("routine_only")) for m in memberships),
        ))
    return sorted((node for node in nodes if node.node_id and node.text), key=lambda node: node.node_id)


def anchor_match(text: str, anchor: str) -> bool:
    if re.fullmatch(r"[A-Za-z0-9_-]+", anchor):
        return bool(re.search(rf"(?<![A-Za-z0-9_-]){re.escape(anchor)}(?![A-Za-z0-9_-])", text, re.I))
    return anchor.lower() in text.lower()


def named_terms(text: str) -> set[str]:
    terms = set(re.findall(r"《([^》]{2,40})》", str(text or "")))
    for _, _, anchors in CARD_SPECS:
        terms.update(anchor for anchor in anchors if anchor_match(text, anchor))
    return {term for term in terms if term not in GLOBAL_STOP_ENTITIES}


def node_paragraphs(node: Node) -> list[dict[str, str]]:
    paragraphs: list[dict[str, str]] = []
    if node.title:
        paragraphs.append({"origin": "title", "text": node.title})
    for index, value in enumerate(re.split(r"\n\s*\n+", node.text)):
        if value.strip():
            paragraphs.append({"origin": f"body:{index}", "text": value.strip()})
    for index, value in enumerate(node.track_texts):
        for part_index, part in enumerate(re.split(r"\n\s*\n+", value)):
            if part.strip():
                paragraphs.append({"origin": f"track:{index}:{part_index}", "text": part.strip()})
    return paragraphs


def _dynamic_anchor_terms(text: str) -> set[str]:
    terms = set(re.findall(r"(?<![A-Za-z0-9_-])[A-Za-z][A-Za-z0-9_-]{1,31}(?![A-Za-z0-9_-])", text))
    # Jieba cannot know every fictional name. A conservative two-character surname
    # fallback keeps unknown names recallable without granting admission.
    terms.update(re.findall(r"[赵钱孙李周吴郑王冯陈蒋沈韩杨朱秦许何吕施张孔曹严华金魏陶姜谢邹苏潘葛范彭鲁韦昌马苗方俞任袁柳史唐薛雷贺倪汤罗毕郝安常乐傅齐康伍余顾孟黄萧尹姚邵汪毛米贝戴宋熊纪舒项祝董梁杜阮蓝季贾江郭梅林钟徐邱高夏蔡田樊胡霍万卢莫房解丁邓洪左石崔龚程陆翁羊甄家封储段巫焦侯秋仲宫宁仇甘武刘龙白赖卓蒙乔曾关查游权][\u4e00-\u9fff]", text))
    if jieba_posseg is not None and jieba_analyse is not None:
        allowed = {"n", "nr", "ns", "nt", "nz", "vn", "eng"}
        for word, flag in jieba_posseg.cut(text):
            value = word.strip()
            if flag in allowed and 2 <= len(value) <= 12:
                terms.add(value)
        terms.update(
            value.strip() for value in jieba_analyse.extract_tags(
                text,
                topK=20,
                withWeight=False,
                allowPOS=("n", "nr", "ns", "nt", "nz", "vn", "eng"),
            )
            if 2 <= len(value.strip()) <= 12
        )
    else:
        terms.update(re.findall(r"[\u4e00-\u9fff]{2,12}", text))
    return {
        term for term in terms
        if term not in ANCHOR_STOP_WORDS and not term.isdigit()
    }


def _raw_anchor_terms(node: Node) -> set[str]:
    terms: set[str] = set()
    for paragraph in node_paragraphs(node):
        text = paragraph["text"]
        terms.update(re.findall(r"《([^》]{2,40})》", text))
        terms.update(re.findall(r"[“「『]([^”」』]{2,30})[”」』]", text))
        terms.update(_dynamic_anchor_terms(text))
        for _, _, anchors in CARD_SPECS:
            terms.update(anchor for anchor in anchors if anchor_match(text, anchor))
        if paragraph["origin"] == "title":
            terms.update(
                fragment.strip() for fragment in re.split(r"[，。！？、：；（）()\s]+", text)
                if 3 <= len(fragment.strip()) <= 20
            )
    return {term for term in terms if term and term not in GLOBAL_STOP_ENTITIES}


def build_anchor_document_frequency(nodes: list[Node]) -> dict[str, int]:
    frequency: Counter[str] = Counter()
    for node in nodes:
        frequency.update(_raw_anchor_terms(node))
    return dict(sorted(frequency.items()))


def extract_anchor_candidates(node: Node, corpus_df: dict[str, int]) -> dict[str, Any]:
    paragraphs = node_paragraphs(node)
    terms = sorted(_raw_anchor_terms(node))
    explicit_work = sorted(set(re.findall(r"《([^》]{2,40})》", "\n".join(item["text"] for item in paragraphs))))
    occurrences: dict[str, list[dict[str, Any]]] = {}
    compounds: list[dict[str, Any]] = []
    repeated: list[dict[str, Any]] = []
    compound_terms: set[str] = set()
    for paragraph in paragraphs:
        present = []
        for term in terms:
            count = len(re.findall(re.escape(term), paragraph["text"], re.I))
            if not count:
                continue
            occurrences.setdefault(term, []).append({"origin": paragraph["origin"], "count": count})
            present.append(term)
            if count >= 2:
                repeated.append({"term": term, "origin": paragraph["origin"], "count": count, "document_frequency": corpus_df.get(term, 0)})
        # Keep this bounded and deterministic; card-specific consumers filter the pairs.
        known_anchors = {anchor for _, _, anchors in CARD_SPECS for anchor in anchors}
        present = sorted(
            present,
            key=lambda term: (term not in explicit_work, term not in known_anchors, term),
        )[:12]
        for left, right in combinations(present, 2):
            compounds.append({"terms": [left, right], "origin": paragraph["origin"]})
            compound_terms.update((left, right))
    rare_single = [
        {"term": term, "document_frequency": corpus_df.get(term, 0)}
        for term in terms
        if corpus_df.get(term, 0) <= 3 and term not in compound_terms and term not in explicit_work
    ]
    return {
        "node_id": node.node_id,
        "extractor": dict(ANCHOR_EXTRACTOR),
        "anchor_candidates": terms,
        "explicit_work": [
            {"term": term, "document_frequency": corpus_df.get(term, 0)} for term in explicit_work
        ],
        "compound": compounds,
        "repeated_phrase": repeated,
        "rare_single": rare_single,
        "occurrences": occurrences,
        "document_frequency": {term: corpus_df.get(term, 0) for term in terms},
    }


def _phrase_like(term: str) -> bool:
    return bool(re.fullmatch(r"[\u4e00-\u9fff]{3,}", term) or " " in term or "-" in term)


def card_anchor_receipt(anchor_receipt: dict[str, Any], anchors: tuple[str, ...]) -> dict[str, Any]:
    anchor_set = set(anchors)
    explicit = sorted(
        item["term"] for item in anchor_receipt["explicit_work"]
        if item["term"] in anchor_set
    )
    compound = []
    for item in anchor_receipt["compound"]:
        matched = sorted(set(item["terms"]).intersection(anchor_set))
        if len(matched) < 2 or all(term in GENERIC_RECALL_ONLY for term in matched):
            continue
        compound.append({"terms": matched, "origin": item["origin"]})
    salient_phrase = []
    for anchor in anchors:
        if anchor in GENERIC_RECALL_ONLY or anchor in PERSON_RECALL_ONLY or not _phrase_like(anchor):
            continue
        occurrences = anchor_receipt["occurrences"].get(anchor) or []
        document_frequency = next(
            (item["document_frequency"] for item in anchor_receipt["explicit_work"] if item["term"] == anchor),
            None,
        )
        if document_frequency is None:
            document_frequency = max(
                [item["document_frequency"] for item in anchor_receipt["repeated_phrase"] if item["term"] == anchor] or [0]
            )
        # Occurrence-only terms still need their corpus DF; it is attached below by the caller.
        document_frequency = max(document_frequency, int(anchor_receipt.get("document_frequency", {}).get(anchor, 0)))
        salient = any(item["origin"] == "title" or item["count"] >= 2 for item in occurrences)
        if document_frequency >= 2 and salient:
            salient_phrase.append({"term": anchor, "document_frequency": document_frequency})
    direct = bool(explicit or compound or salient_phrase)
    return {
        "explicit_work": explicit,
        "compound": compound,
        "salient_repeated_phrase": salient_phrase,
        "direct_admission_evidence": direct,
        "admission_reason": "rare_compound_anchor" if direct else "single_or_generic_anchor_recall_only",
    }


def lexical_tokens(text: str) -> Counter[str]:
    tokens = re.findall(r"[a-z][a-z0-9_-]{2,}|[\u4e00-\u9fff]{2,8}", str(text or "").lower())
    stops = {item.lower() for item in GLOBAL_STOP_ENTITIES}
    return Counter(token for token in tokens if token not in stops)


def cosine_lexical(left: str, right: str) -> float:
    a, b = lexical_tokens(left), lexical_tokens(right)
    if not a or not b:
        return 0.0
    numerator = sum(a[key] * b.get(key, 0) for key in a)
    denominator = math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))
    return round(numerator / denominator, 4) if denominator else 0.0


def is_manual_exception(node: Node, arc_key: str) -> bool:
    return (
        node.node_id == MANUAL_EXCEPTION["event_id"]
        and node.fingerprint == MANUAL_EXCEPTION["fingerprint"]
        and list(node.source_ids) == MANUAL_EXCEPTION["source_message_ids"]
        and arc_key == MANUAL_EXCEPTION["target_arc_key"]
        and bool(MANUAL_EXCEPTION["reviewer"])
        and bool(MANUAL_EXCEPTION["reason"])
    )


def mapped_track_ids(node: Node, anchors: tuple[str, ...]) -> list[str]:
    return [
        track_id for track_id, track_text in zip(node.track_ids, node.track_texts)
        if any(anchor_match(track_text, anchor) for anchor in anchors)
    ]


def gate_decision(node: Node, arc_key: str) -> tuple[str, str]:
    manual = is_manual_exception(node, arc_key)
    if any(status == "parked" for status in node.statuses):
        return "defer", "parked_source"
    if node.routine_flags and any(node.routine_flags) and not all(node.routine_flags):
        return "defer", "event_rewrite_required"
    if node.routine_flags and all(node.routine_flags):
        return "exclude", "routine_only"
    track_intimacy = any(term.lower() in node.track_searchable.lower() for term in INTIMACY_TERMS)
    # "插入" is common technical/plot language, so it cannot mark intimacy by itself.
    body_intimacy = any(
        term.lower() in node.searchable.lower() for term in INTIMACY_TERMS if term != "插入"
    )
    if (track_intimacy or body_intimacy) and not manual:
        return "exclude", "intimacy_excluded"
    routine_hits = sum(term in node.searchable for term in ROUTINE_TERMS)
    if routine_hits >= 2 and not named_terms(node.searchable) and len(node.searchable) < 240:
        return "exclude", "routine_text"
    return "eligible", ""


def retrieval_evidence(
    node: Node,
    arc_key: str,
    anchors: tuple[str, ...],
    seed: Node,
    anchor_receipt: dict[str, Any],
) -> dict[str, Any]:
    matched_terms = [anchor for anchor in anchors if anchor_match(node.searchable, anchor)]
    mapped_tracks = mapped_track_ids(node, anchors)
    track_overlap = sorted(set(node.track_ids).intersection(seed.track_ids))
    source_overlap = sorted(set(node.source_ids).intersection(seed.source_ids))
    lexical = cosine_lexical(seed.searchable + "\n" + " ".join(anchors), node.searchable)
    manual = is_manual_exception(node, arc_key)
    card_receipt = card_anchor_receipt(anchor_receipt, anchors)
    direct = bool(card_receipt["direct_admission_evidence"] or manual)
    return {
        "matched_card_terms": matched_terms,
        "mapped_track_ids": mapped_tracks,
        "exact_track_overlap": track_overlap,
        "exact_source_overlap": source_overlap,
        "lexical_score": lexical,
        "manual_exception": MANUAL_EXCEPTION if manual else None,
        "anchor_receipt": card_receipt,
        "embedding": {"status": "unavailable", "reason": "not called; pending"},
        "direct_admission_evidence": direct,
    }


def overlap_kind(left: Node, right: Node) -> str:
    a, b = set(left.source_ids), set(right.source_ids)
    if not a or not b or not a.intersection(b):
        return ""
    if left.kind == right.kind == "event":
        return "event_event_collision"
    if left.kind == right.kind == "scene" and (a <= b or b <= a):
        return "scene_scene_containment"
    if left.kind != right.kind and (a <= b or b <= a):
        return "event_scene_containment"
    return "partial_cross_kind_overlap"


def _score(evidence: dict[str, Any], recent: bool) -> float:
    anchor_receipt = evidence["anchor_receipt"]
    return (
        80.0 * bool(evidence["manual_exception"])
        + 35.0 * len(anchor_receipt["explicit_work"])
        + 30.0 * len(anchor_receipt["compound"])
        + 20.0 * len(anchor_receipt["salient_repeated_phrase"])
        + 6.0 * len(evidence["matched_card_terms"])
        + 4.0 * len(evidence["mapped_track_ids"])
        + 8.0 * bool(evidence["exact_source_overlap"])
        + 4.0 * bool(evidence["exact_track_overlap"])
        + 10.0 * float(evidence["lexical_score"])
        + 2.0 * recent
    )


def candidate_shadow(nodes: list[Node], *, session_id: int, source_min: int, source_max: int) -> dict[str, Any]:
    nodes = sorted(nodes, key=lambda node: node.node_id)
    anchor_df = build_anchor_document_frequency(nodes)
    anchor_receipts = {node.node_id: extract_anchor_candidates(node, anchor_df) for node in nodes}
    recent_ids = {
        node.node_id for node in nodes
        if session_id in node.session_ids and any(source_min <= source_id <= source_max for source_id in node.source_ids)
    }
    cards: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for arc_key, title, anchors in CARD_SPECS:
        seeds = []
        for node in nodes:
            if node.node_id not in recent_ids or node.node_id == MANUAL_EXCEPTION["event_id"]:
                continue
            action, _ = gate_decision(node, arc_key)
            if action != "eligible":
                continue
            if card_anchor_receipt(anchor_receipts[node.node_id], anchors)["direct_admission_evidence"]:
                seeds.append(node)
        merged: dict[str, tuple[float, Node, dict[str, Any]]] = {}
        for seed in seeds:
            ranked: list[tuple[float, Node, dict[str, Any]]] = []
            for node in nodes:
                evidence = retrieval_evidence(node, arc_key, anchors, seed, anchor_receipts[node.node_id])
                score = _score(evidence, node.node_id in recent_ids)
                retrievable = bool(
                    evidence["direct_admission_evidence"]
                    or evidence["exact_track_overlap"]
                    or evidence["exact_source_overlap"]
                    or evidence["lexical_score"] > 0
                )
                if retrievable:
                    ranked.append((score, node, evidence))
            ranked.sort(key=lambda item: (-item[0], item[1].node_id))
            for score, node, evidence in ranked[:RECALL_TOP_K_PER_SEED]:
                old = merged.get(node.node_id)
                if old is None or score > old[0]:
                    merged[node.node_id] = (score, node, evidence)
        # Exact session-scope audit is retained even when an old corpus hit outranks it.
        # These are direct card/Track matches, not a second retrieval hop.
        for node in nodes:
            if node.node_id not in recent_ids:
                continue
            evidence = retrieval_evidence(node, arc_key, anchors, seeds[0], anchor_receipts[node.node_id]) if seeds else None
            if evidence and evidence["direct_admission_evidence"]:
                score = _score(evidence, True)
                old = merged.get(node.node_id)
                if old is None or score > old[0]:
                    merged[node.node_id] = (score, node, evidence)
        ranked_merged = sorted(merged.values(), key=lambda item: (-item[0], item[1].node_id))[:ARC_BATCH_CAP]
        decisions: list[dict[str, Any]] = []
        recalled_nodes = [node for _, node, _ in ranked_merged]
        collision_eligible = {
            node.node_id for _, node, evidence in ranked_merged
            if gate_decision(node, arc_key)[0] == "eligible" and evidence["direct_admission_evidence"]
        }
        event_collision_ids = {
            node.node_id for index, node in enumerate(recalled_nodes)
            for other in recalled_nodes[index + 1:]
            if overlap_kind(node, other) == "event_event_collision"
            and node.node_id in collision_eligible and other.node_id in collision_eligible
            for node in (node, other)
        }
        accepted: list[Node] = []
        for score, node, evidence in ranked_merged:
            action, reason = gate_decision(node, arc_key)
            if action == "eligible" and node.node_id in event_collision_ids:
                action, reason = "defer", "canonical_source_collision"
            if action == "eligible" and not evidence["direct_admission_evidence"]:
                action, reason = "exclude", "indirect_only"
            covered_by = ""
            if action == "eligible":
                for other in accepted:
                    relation = overlap_kind(node, other)
                    if relation in {"scene_scene_containment", "event_scene_containment"}:
                        # Prefer Event over an overlapping Scene; otherwise keep the first stable node.
                        if node.kind == "event" and other.kind == "scene":
                            previous = next(item for item in decisions if item["node_id"] == other.node_id)
                            previous.update(decision="exclude", reason="covered_by_event", covered_by=node.node_id)
                            accepted.remove(other)
                            continue
                        action, reason, covered_by = "exclude", "covered_by_existing_material", other.node_id
                        break
            if action == "eligible":
                action, reason = "include", "direct_one_hop_match"
                accepted.append(node)
            decision = {
                "arc_key": arc_key,
                "node_id": node.node_id,
                "kind": node.kind,
                "title": node.title or node.text[:80].replace("\n", " "),
                "date": node.date,
                "source_message_ids": list(node.source_ids),
                "decision": action,
                "reason": reason,
                "covered_by": covered_by,
                "score": round(score, 3),
                "retrieval_evidence": evidence,
            }
            decisions.append(decision)
        included = [item for item in decisions if item["decision"] == "include"]
        card_status = "candidate" if len(included) >= 2 else "not_promoted_singleton"
        if len(included) > ARC_MEMBER_CAP:
            for item in included[ARC_MEMBER_CAP:]:
                item.update(decision="defer", reason="arc_member_cap")
            included = included[:ARC_MEMBER_CAP]
        receipts.extend(decisions)
        if len(included) >= 2:
            cards.append({
                "arc_key": arc_key,
                "title": title,
                "anchors": list(anchors),
                "seed_node_ids": [node.node_id for node in seeds],
                "member_count": len(included),
                "members": included,
                "status": card_status,
                "one_hop_only": True,
            })
    cards.sort(key=lambda card: card["arc_key"])
    receipts.sort(key=lambda item: (item["arc_key"], item["node_id"]))
    return {
        "mode": "germany_read_only_arc_candidate_shadow",
        "scope": {"session_id": session_id, "source_min": source_min, "source_max": source_max},
        "caps": {
            "recall_top_k_per_seed": RECALL_TOP_K_PER_SEED,
            "arc_member_cap": ARC_MEMBER_CAP,
            "arc_batch_cap": ARC_BATCH_CAP,
        },
        "embedding": {"status": "unavailable", "reason": "not called; pending"},
        "global_stop_entities": sorted(GLOBAL_STOP_ENTITIES),
        "hard_rules": {
            "one_hop_arc_key": True,
            "transitive_closure": False,
            "track_overlap_only_boosts": True,
            "embedding_date_person_only_admission": False,
            "intimacy_arc": False,
            "manual_exception_not_seed": True,
        },
        "input_counts": {"nodes": len(nodes), "recent_nodes": len(recent_ids)},
        "anchor_receipts": [anchor_receipts[node_id] for node_id in sorted(recent_ids)],
        "cards": cards,
        "recall_receipts": receipts,
        "manual_exception_contract": MANUAL_EXCEPTION,
        "writes_performed": [],
    }


def markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# Germany Arc candidate shadow — session 22 first pass", "",
        "- Strict read-only: no Germany DB/canonical/Arc write and no model/embedding call.",
        f"- Scope: session `{result['scope']['session_id']}`, sources `{result['scope']['source_min']}–{result['scope']['source_max']}`.",
        f"- Caps: recall top-k/seed `{RECALL_TOP_K_PER_SEED}`, members/Arc `{ARC_MEMBER_CAP}`, batch/Arc `{ARC_BATCH_CAP}`.",
        f"- Embedding: `{result['embedding']['status']}` — {result['embedding']['reason']}.",
        f"- Input: `{result['input_counts']['nodes']}` Event/Scene nodes; recent nodes `{result['input_counts']['recent_nodes']}`.",
        "", "## Candidate Arc cards", "",
    ]
    for card in result["cards"]:
        lines.extend([f"### {card['arc_key']} — {card['title']}", "", f"Members: `{card['member_count']}`; one-hop: `true`.", ""])
        for member in card["members"]:
            ev = member["retrieval_evidence"]
            evidence = []
            if ev["matched_card_terms"]: evidence.append("terms=" + ",".join(ev["matched_card_terms"]))
            if ev["mapped_track_ids"]: evidence.append("mapped_track=" + ",".join(ev["mapped_track_ids"]))
            if ev["exact_track_overlap"]: evidence.append("track_boost=" + ",".join(ev["exact_track_overlap"]))
            if ev["exact_source_overlap"]: evidence.append("source_overlap=" + ",".join(map(str, ev["exact_source_overlap"])))
            if ev["manual_exception"]: evidence.append("manual_exception=exact")
            evidence.append(f"lexical={ev['lexical_score']}")
            lines.append(f"- `{member['node_id']}` ({member['kind']}, {member['date']}) — {member['title']}  \n  evidence: {'; '.join(evidence)}")
        lines.append("")
    lines.extend(["## Recall decision receipt", ""])
    for item in result["recall_receipts"]:
        lines.append(f"- `{item['arc_key']}` / `{item['node_id']}`: **{item['decision']}** — `{item['reason']}`")
    lines.extend([
        "", "## Guard receipts", "",
        "- Every member independently has a card entity/term, an explicitly mapped Track card, or the exact manual exception; Track overlap only boosts rank.",
        "- Recall is top-8 per seed, merged batch is at most 40 per Arc, and admitted membership is at most 12 per Arc.",
        "- Parked hits defer; mixed routine/substantive ownership defers for Event rewrite.",
        "- Event/Event source collisions defer; contained Scene material is collapsed and cannot add a second vote.",
        "- No connected-component or transitive snowball; embedding/date/person/lexical-only evidence cannot admit.",
        "- Explicit intimacy is checked in Event/Scene text and Track subject/throughline; no intimacy Arc is built.",
        "- `event_d596ab8bb8259ddaf3e8e916` requires exact sources, fingerprint, target Arc, reviewer and reason, and never seeds.",
        "- `writes_performed=[]`.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Germany Arc candidate shadow")
    parser.add_argument("--host", default="168.119.228.217")
    parser.add_argument("--key", type=Path, default=Path(r"C:\Users\86188\.ssh\id_ed25519"))
    parser.add_argument("--session-id", type=int, default=22)
    parser.add_argument("--source-min", type=int, default=11213)
    parser.add_argument("--source-max", type=int, default=11923)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = candidate_shadow(
        build_nodes(fetch_snapshot(args.host, args.key, args.session_id)),
        session_id=args.session_id,
        source_min=args.source_min,
        source_max=args.source_max,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else markdown_report(result))


if __name__ == "__main__":
    main()
