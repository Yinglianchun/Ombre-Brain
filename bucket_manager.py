# ============================================================
# Module: Memory Bucket Manager (bucket_manager.py)
# 模块：记忆桶管理器
#
# CRUD operations, multi-dimensional index search, activation updates
# for memory buckets.
# 记忆桶的增删改查、多维索引搜索、激活更新。
#
# Core design:
# 核心逻辑：
#   - Each bucket = one Markdown file (YAML frontmatter + body)
#     每个记忆桶 = 一个 Markdown 文件
#   - Storage by type: permanent / dynamic / archive
#     存储按类型分目录
#   - Multi-dimensional soft index: domain + valence/arousal + fuzzy text
#     多维软索引：主题域 + 情感坐标 + 文本模糊匹配
#   - Search strategy: domain pre-filter → weighted multi-dim ranking
#     搜索策略：主题域预筛 → 多维加权精排
#   - Emotion coordinates based on Russell circumplex model:
#     情感坐标基于环形情感模型（Russell circumplex）：
#       valence (0~1): 0=negative → 1=positive
#       arousal (0~1): 0=calm → 1=excited
#
# Depended on by: server.py, decay_engine.py
# 被谁依赖：server.py, decay_engine.py
# ============================================================

import os
import math
import logging
import re
import shutil
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import frontmatter
import jieba

from identity import identity_names
from memory_relevance import content_terms_for_query, memory_relevance_options_from_config, recall_topic_query
from query_terms import GENERIC_LEXICAL_STOPWORDS
from utils import (
    bucket_content_for_recall,
    generate_bucket_id,
    normalize_scene_cues,
    now_iso,
    safe_path,
    sanitize_name,
    strip_wikilinks,
)

logger = logging.getLogger("ombre_brain.bucket")


class BucketManager:
    """
    Memory bucket manager — entry point for all bucket CRUD operations.
    Buckets are stored as Markdown files with YAML frontmatter for metadata
    and body for content. Natively compatible with Obsidian browsing/editing.
    记忆桶管理器 —— 所有桶的 CRUD 操作入口。
    桶以 Markdown 文件存储，YAML frontmatter 存元数据，正文存内容。
    天然兼容 Obsidian 直接浏览和编辑。
    """

    def __init__(self, config: dict):
        self.config = config
        # --- Read storage paths from config / 从配置中读取存储路径 ---
        self.base_dir = config["buckets_dir"]
        self.permanent_dir = os.path.join(self.base_dir, "permanent")
        self.dynamic_dir = os.path.join(self.base_dir, "dynamic")
        self.archive_dir = os.path.join(self.base_dir, "archive")
        self.feel_dir = os.path.join(self.base_dir, "feel")
        self.tombstone_dir = os.path.join(self.base_dir, ".tombstones")
        self.fuzzy_threshold = config.get("matching", {}).get("fuzzy_threshold", 50)
        self.max_results = config.get("matching", {}).get("max_results", 5)

        # --- Wikilink config / 双链配置 ---
        wikilink_cfg = config.get("wikilink", {})
        self.wikilink_enabled = wikilink_cfg.get("enabled", True)
        self.wikilink_use_tags = wikilink_cfg.get("use_tags", False)
        self.wikilink_use_domain = wikilink_cfg.get("use_domain", True)
        self.wikilink_use_auto_keywords = wikilink_cfg.get("use_auto_keywords", True)
        self.wikilink_auto_top_k = wikilink_cfg.get("auto_top_k", 8)
        self.wikilink_min_len = wikilink_cfg.get("min_keyword_len", 2)
        self.wikilink_exclude_keywords = set(wikilink_cfg.get("exclude_keywords", []))
        self.wikilink_stopwords = {
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
            "都", "一个", "上", "也", "很", "到", "说", "要", "去",
            "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
            "我们", "你们", "他们", "然后", "今天", "昨天", "明天", "一下",
            "the", "and", "for", "are", "but", "not", "you", "all", "can",
            "had", "her", "was", "one", "our", "out", "has", "have", "with",
            "this", "that", "from", "they", "been", "said", "will", "each",
        }
        self.wikilink_stopwords |= {w.lower() for w in self.wikilink_exclude_keywords}

        # --- Search scoring weights / 检索权重配置 ---
        scoring = config.get("scoring_weights", {})
        self.w_topic = scoring.get("topic_relevance", 4.0)
        # Legacy valence/arousal fields remain readable, but P1 removes them
        # from ordinary recall ranking. Keep the public attribute for clients.
        self.configured_w_emotion = scoring.get("emotion_resonance", 0.0)
        self.w_emotion = 0.0
        self.w_time = scoring.get("time_proximity", 1.5)
        self.w_importance = scoring.get("importance", 1.0)
        self.content_weight = scoring.get("content_weight", 1.0)  # Added to allow better content-based matching during merge
        self.lexical_stop_terms = self._build_lexical_stop_terms(config)
        self._lexical_profile_cache: dict[
            str,
            tuple[tuple, Counter[str], float, tuple[str, str, str, str, str]],
        ] = {}

    # ---------------------------------------------------------
    # Create a new bucket
    # 创建新桶
    # Write content and metadata into a .md file
    # 将内容和元数据写入一个 .md 文件
    # ---------------------------------------------------------
    async def create(
        self,
        content: str,
        tags: list[str] = None,
        importance: int = 5,
        domain: list[str] = None,
        valence: float = 0.5,
        arousal: float = 0.3,
        bucket_type: str = "dynamic",
        name: str = None,
        pinned: bool = False,
        protected: bool = False,
        bucket_id: str = None,
        source: str = None,
        created: str = None,
        last_active: str = None,
        updated_at: str = None,
        anchor: bool = False,
        resolved: bool = False,
        digested: bool = False,
        confidence: float | None = None,
        period: str | None = None,
        date: str | None = None,
        extra_metadata: dict | None = None,
    ) -> str:
        """
        Create a new memory bucket, return bucket ID.
        创建一个新的记忆桶，返回桶 ID。

        pinned/protected=True: bucket won't be merged, decayed, or have importance changed.
        Importance is locked to 10 for pinned/protected buckets.
        pinned/protected 桶不参与合并与衰减，importance 强制锁定为 10。
        """
        bucket_id = bucket_id or generate_bucket_id()
        bucket_name = sanitize_name(name) if name else bucket_id
        domain = domain or ["未分类"]
        tags = tags or []
        linked_content = content  # wikilink injection disabled; LLM adds [[]] via prompt
        created_at = created or now_iso()
        last_active_at = last_active or created_at
        updated_at_value = updated_at or created_at

        # --- Pinned/protected buckets: lock importance to 10 ---
        # --- 钉选/保护桶：importance 强制锁定为 10 ---
        if pinned or protected:
            importance = 10

        # --- Build YAML frontmatter metadata / 构建元数据 ---
        metadata = {
            "id": bucket_id,
            "name": bucket_name,
            "tags": tags,
            "domain": domain,
            "valence": max(0.0, min(1.0, valence)),
            "arousal": max(0.0, min(1.0, arousal)),
            "importance": max(1, min(10, importance)),
            "type": bucket_type,
            "created": created_at,
            "date": str(date).strip() if date else str(created_at)[:10],
            "last_active": last_active_at,
            "updated_at": updated_at_value,
            "activation_count": 0,
        }
        if confidence is not None:
            metadata["confidence"] = max(0.0, min(1.0, float(confidence)))
        if period:
            metadata["period"] = str(period)
        if pinned:
            metadata["pinned"] = True
        if protected:
            metadata["protected"] = True
        if anchor:
            metadata["anchor"] = True
        if resolved:
            metadata["resolved"] = True
        if digested:
            metadata["digested"] = True
        if source:
            metadata["source"] = source
        if extra_metadata:
            reserved = set(metadata.keys()) | {"content"}
            for key, value in extra_metadata.items():
                if key in reserved or value is None:
                    continue
                metadata[str(key)] = value

        # --- Assemble Markdown file (frontmatter + body) ---
        # --- 组装 Markdown 文件 ---
        post = frontmatter.Post(linked_content, **metadata)

        # --- Choose directory by type + primary domain ---
        # --- 按类型 + 主题域选择存储目录 ---
        if bucket_type == "permanent" or pinned:
            type_dir = self.permanent_dir
            if pinned and bucket_type != "permanent":
                metadata["type"] = "permanent"
        elif bucket_type == "feel":
            type_dir = self.feel_dir
        else:
            type_dir = self.dynamic_dir
        if bucket_type == "feel":
            primary_domain = "沉淀物"  # feel subfolder name
        else:
            primary_domain = sanitize_name(domain[0]) if domain else "未分类"
        target_dir = os.path.join(type_dir, primary_domain)
        os.makedirs(target_dir, exist_ok=True)

        # --- Filename: readable_name_bucketID.md (Obsidian friendly) ---
        # --- 文件名：可读名称_桶ID.md ---
        if bucket_name and bucket_name != bucket_id:
            filename = f"{bucket_name}_{bucket_id}.md"
        else:
            filename = f"{bucket_id}.md"
        file_path = safe_path(target_dir, filename)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(frontmatter.dumps(post))
        except OSError as e:
            logger.error(f"Failed to write bucket file / 写入桶文件失败: {file_path}: {e}")
            raise

        logger.info(
            f"Created bucket / 创建记忆桶: {bucket_id} ({bucket_name}) → {primary_domain}/"
            + (" [PINNED]" if pinned else "") + (" [PROTECTED]" if protected else "")
        )
        return bucket_id

    # ---------------------------------------------------------
    # Read bucket content
    # 读取桶内容
    # Returns {"id", "metadata", "content", "path"} or None
    # ---------------------------------------------------------
    async def get(self, bucket_id: str) -> Optional[dict]:
        """
        Read a single bucket by ID.
        根据 ID 读取单个桶。
        """
        if not bucket_id or not isinstance(bucket_id, str):
            return None
        file_path = self._find_bucket_file(bucket_id)
        if not file_path:
            return None
        return self._load_bucket(file_path)

    # ---------------------------------------------------------
    # Move bucket between directories
    # 在目录间移动桶文件
    # ---------------------------------------------------------
    def _move_bucket(self, file_path: str, target_type_dir: str, domain: list[str] = None) -> str:
        """
        Move a bucket file to a new type directory, preserving domain subfolder.
        Returns new file path.
        """
        primary_domain = sanitize_name(domain[0]) if domain else "未分类"
        target_dir = os.path.join(target_type_dir, primary_domain)
        os.makedirs(target_dir, exist_ok=True)
        filename = os.path.basename(file_path)
        new_path = safe_path(target_dir, filename)
        if os.path.normpath(file_path) != os.path.normpath(new_path):
            os.rename(file_path, new_path)
            logger.info(f"Moved bucket / 移动记忆桶: {filename} → {target_dir}/")
        return new_path

    # ---------------------------------------------------------
    # Update bucket
    # 更新桶
    # Supports: content, tags, facets, importance, valence, arousal, name, resolved
    # ---------------------------------------------------------
    async def update(self, bucket_id: str, **kwargs) -> bool:
        """
        Update bucket content or metadata fields.
        更新桶的内容或元数据字段。
        """
        file_path = self._find_bucket_file(bucket_id)
        if not file_path:
            return False

        try:
            post = frontmatter.load(file_path)
        except Exception as e:
            logger.warning(f"Failed to load bucket for update / 加载桶失败: {file_path}: {e}")
            return False
        if bool(post.get("source_record_immutable", False)):
            logger.warning("Refused immutable source record update / 拒绝修改不可变来源记录: %s", bucket_id)
            return False

        # --- Pinned/protected buckets: lock importance to 10, ignore importance changes ---
        # --- 钉选/保护桶：importance 不可修改，强制保持 10 ---
        is_pinned = post.get("pinned", False) or post.get("protected", False)
        if is_pinned:
            kwargs.pop("importance", None)  # silently ignore importance update

        # --- Update only fields that were passed in / 只改传入的字段 ---
        if "content" in kwargs:
            post.content = kwargs["content"]  # wikilink injection disabled; LLM adds [[]] via prompt
        if "tags" in kwargs:
            post["tags"] = kwargs["tags"]
        if "facets" in kwargs:
            post["facets"] = kwargs["facets"] if isinstance(kwargs["facets"], list) else []
        if "importance" in kwargs:
            post["importance"] = max(1, min(10, int(kwargs["importance"])))
        if "domain" in kwargs:
            post["domain"] = kwargs["domain"]
        if "valence" in kwargs:
            post["valence"] = max(0.0, min(1.0, float(kwargs["valence"])))
        if "arousal" in kwargs:
            post["arousal"] = max(0.0, min(1.0, float(kwargs["arousal"])))
        if "name" in kwargs:
            post["name"] = sanitize_name(kwargs["name"])
        if "resolved" in kwargs:
            post["resolved"] = bool(kwargs["resolved"])
        if "pinned" in kwargs:
            post["pinned"] = bool(kwargs["pinned"])
            if kwargs["pinned"]:
                post["importance"] = 10  # pinned → lock importance to 10
        if "anchor" in kwargs:
            post["anchor"] = bool(kwargs["anchor"])
        if "digested" in kwargs:
            post["digested"] = bool(kwargs["digested"])
        if "model_valence" in kwargs:
            post["model_valence"] = max(0.0, min(1.0, float(kwargs["model_valence"])))
        if "source" in kwargs:
            post["source"] = str(kwargs["source"])
        if "confidence" in kwargs:
            post["confidence"] = max(0.0, min(1.0, float(kwargs["confidence"])))
        if "period" in kwargs:
            post["period"] = str(kwargs["period"])
        if "date" in kwargs:
            post["date"] = str(kwargs["date"])
        if "comments" in kwargs:
            post["comments"] = kwargs["comments"] if isinstance(kwargs["comments"], list) else []
        if "comment_count" in kwargs:
            post["comment_count"] = max(0, int(kwargs["comment_count"]))
        if "active" in kwargs:
            post["active"] = bool(kwargs["active"])
        if "deprecated" in kwargs:
            post["deprecated"] = bool(kwargs["deprecated"])
        if "profile_kind" in kwargs:
            post["profile_kind"] = str(kwargs["profile_kind"])
        if "subject" in kwargs:
            post["subject"] = str(kwargs["subject"])
        if "predicate" in kwargs:
            post["predicate"] = str(kwargs["predicate"])
        if "object" in kwargs:
            post["object"] = str(kwargs["object"])
        if "evidence" in kwargs:
            post["evidence"] = kwargs["evidence"] if isinstance(kwargs["evidence"], list) else []
        if "source_bucket_ids" in kwargs:
            post["source_bucket_ids"] = kwargs["source_bucket_ids"] if isinstance(kwargs["source_bucket_ids"], list) else []
        if "source_persona_event_ids" in kwargs:
            post["source_persona_event_ids"] = (
                kwargs["source_persona_event_ids"] if isinstance(kwargs["source_persona_event_ids"], list) else []
            )
        if "source_conversation_turn_ids" in kwargs:
            post["source_conversation_turn_ids"] = (
                kwargs["source_conversation_turn_ids"] if isinstance(kwargs["source_conversation_turn_ids"], list) else []
            )
        if "extra_metadata" in kwargs and isinstance(kwargs["extra_metadata"], dict):
            reserved = {"id", "name", "content", "created", "last_active", "updated_at"}
            for key, value in kwargs["extra_metadata"].items():
                if key in reserved or value is None:
                    continue
                post[str(key)] = value

        # --- Auto-refresh content update time and activation time ---
        # --- 自动刷新内容更新时间与激活时间 ---
        post["updated_at"] = kwargs.get("updated_at") or now_iso()
        post["last_active"] = kwargs.get("last_active") or now_iso()

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(frontmatter.dumps(post))
        except OSError as e:
            logger.error(f"Failed to write bucket update / 写入桶更新失败: {file_path}: {e}")
            return False

        # --- Auto-move: pinned → permanent/ ---
        # --- 自动移动：钉选 → permanent/ ---
        domain = post.get("domain", ["未分类"])
        if kwargs.get("pinned") and post.get("type") != "permanent":
            post["type"] = "permanent"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(frontmatter.dumps(post))
            self._move_bucket(file_path, self.permanent_dir, domain)
        elif "domain" in kwargs and post.get("type") != "feel":
            bucket_type = str(post.get("type") or "dynamic")
            if bucket_type == "archived":
                target_dir = self.archive_dir
            elif bucket_type == "permanent":
                target_dir = self.permanent_dir
            else:
                target_dir = self.dynamic_dir
            self._move_bucket(file_path, target_dir, domain)

        logger.info(f"Updated bucket / 更新记忆桶: {bucket_id}")
        return True

    async def add_comment(
        self,
        bucket_id: str,
        content: str,
        *,
        author: str | None = None,
        kind: str = "comment",
        valence: float | None = None,
        arousal: float | None = None,
        source: str | None = None,
        created: str | None = None,
        touch: bool = True,
    ) -> Optional[dict]:
        """
        Append a ring/comment to an existing bucket without changing its body.
        给已有桶追加年轮，不改正文。
        """
        file_path = self._find_bucket_file(bucket_id)
        if not file_path or not content or not str(content).strip():
            return None

        try:
            post = frontmatter.load(file_path)
        except Exception as e:
            logger.warning(f"Failed to load bucket for comment / 加载评论目标失败: {file_path}: {e}")
            return None

        comments = post.get("comments", [])
        if not isinstance(comments, list):
            comments = []

        now = now_iso()
        created_at = str(created or now).strip() or now
        default_author = identity_names(self.config).get("ai_name") or "AI"
        entry = {
            "id": generate_bucket_id(),
            "created": created_at,
            "author": str(author or default_author),
            "kind": str(kind or "comment"),
            "content": str(content).strip(),
        }
        if source:
            entry["source"] = str(source)
        if valence is not None:
            entry["valence"] = max(0.0, min(1.0, float(valence)))
            if entry["kind"] == "feel":
                post["model_valence"] = entry["valence"]
        if arousal is not None:
            entry["arousal"] = max(0.0, min(1.0, float(arousal)))

        comments.append(entry)
        post["comments"] = comments
        post["comment_count"] = len(comments)
        post["updated_at"] = now
        if touch:
            post["last_active"] = now
            post["activation_count"] = post.get("activation_count", 0) + 1

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(frontmatter.dumps(post))
        except OSError as e:
            logger.error(f"Failed to write bucket comment / 写入桶评论失败: {file_path}: {e}")
            return None

        if touch:
            current_time = self._parse_iso_datetime(post.get("created", post.get("last_active", "")))
            if current_time is None:
                current_time = datetime.now(timezone.utc).replace(tzinfo=None)
            await self._time_ripple(bucket_id, current_time)

        logger.info(f"Added bucket comment / 已追加年轮: {bucket_id}#{entry['id']}")
        return entry

    async def delete_comment(
        self,
        bucket_id: str,
        comment_id: str,
        *,
        allowed_author: str | None = None,
        allowed_source: str | None = None,
    ) -> dict:
        file_path = self._find_bucket_file(bucket_id)
        if not file_path or not comment_id:
            return {"status": "not_found"}

        try:
            post = frontmatter.load(file_path)
        except Exception as e:
            logger.warning(f"Failed to load bucket for comment delete / 加载评论删除目标失败: {file_path}: {e}")
            return {"status": "not_found"}

        comments = post.get("comments", [])
        if not isinstance(comments, list):
            return {"status": "not_found"}

        kept = []
        target = None
        for comment in comments:
            if isinstance(comment, dict) and str(comment.get("id") or "") == str(comment_id):
                target = comment
            else:
                kept.append(comment)

        if target is None:
            return {"status": "not_found"}
        if allowed_author is not None and target.get("author") != allowed_author:
            return {"status": "forbidden", "comment": target}
        if allowed_source is not None and target.get("source") != allowed_source:
            return {"status": "forbidden", "comment": target}

        post["comments"] = kept
        post["comment_count"] = len(kept)
        post["updated_at"] = now_iso()

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(frontmatter.dumps(post))
        except OSError as e:
            logger.error(f"Failed to delete bucket comment / 删除桶评论失败: {file_path}: {e}")
            return {"status": "failed", "comment": target}

        logger.info(f"Deleted bucket comment / 已删除年轮: {bucket_id}#{comment_id}")
        return {"status": "deleted", "comment": target}

    # ---------------------------------------------------------
    # Wikilink injection — DISABLED
    # 自动添加 Obsidian 双链 — 已禁用
    # Now handled by LLM prompts (Gemini adds [[]] for proper nouns)
    # 现在由 LLM prompt 处理（Gemini 对人名/地名/专有名词加 [[]]）
    # ---------------------------------------------------------
    # def _apply_wikilinks(self, content, tags, domain, name): ...
    # def _collect_wikilink_keywords(self, content, tags, domain, name): ...
    # def _normalize_keywords(self, keywords): ...
    # def _extract_auto_keywords(self, content): ...

    # ---------------------------------------------------------
    # Delete bucket
    # 删除桶
    # ---------------------------------------------------------
    async def delete(self, bucket_id: str) -> bool:
        """
        Delete a memory bucket file.
        删除指定的记忆桶文件。
        """
        file_path = self._find_bucket_file(bucket_id)
        if not file_path:
            return False

        try:
            post = frontmatter.load(file_path)
            if bool(post.get("source_record_immutable", False)):
                logger.warning("Refused immutable source record deletion / 拒绝删除不可变来源记录: %s", bucket_id)
                return False
            tombstone = self._build_tombstone(bucket_id, file_path)
            os.remove(file_path)
            self._write_tombstone(tombstone)
        except Exception as e:
            logger.error(f"Failed to delete bucket file / 删除桶文件失败: {file_path}: {e}")
            return False

        logger.info(f"Deleted bucket / 删除记忆桶: {bucket_id}")
        return True

    def _build_tombstone(self, bucket_id: str, file_path: str) -> dict:
        deleted_at = now_iso()
        tombstone = {
            "id": bucket_id,
            "title": bucket_id,
            "type": "archived",
            "domain": ["deleted"],
            "tags": ["deleted"],
            "content": "",
            "valence": 0.5,
            "arousal": 0.5,
            "importance": 1,
            "pinned": False,
            "resolved": True,
            "digested": True,
            "activation_count": 0,
            "created": deleted_at,
            "last_active": deleted_at,
            "updated_at": deleted_at,
            "source": "deleted",
            "deleted_at": deleted_at,
        }
        try:
            post = frontmatter.load(file_path)
            tombstone.update(
                {
                    "title": post.get("name", bucket_id),
                    "type": post.get("type", "archived"),
                    "domain": post.get("domain", ["deleted"]) or ["deleted"],
                    "created": post.get("created", deleted_at),
                }
            )
        except Exception:
            pass
        return tombstone

    def _write_tombstone(self, tombstone: dict) -> None:
        os.makedirs(self.tombstone_dir, exist_ok=True)
        path = safe_path(self.tombstone_dir, f"{tombstone['id']}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(tombstone, f, ensure_ascii=False, indent=2)

    # ---------------------------------------------------------
    # Touch bucket (refresh activation time + increment count)
    # 触碰桶（刷新激活时间 + 累加激活次数）
    # Called on every recall hit; affects decay score.
    # 每次检索命中时调用，影响衰减得分。
    # ---------------------------------------------------------
    async def touch(self, bucket_id: str) -> None:
        """
        Update a bucket's last activation time and count.
        Also triggers time ripple: nearby memories get a slight activation boost.
        更新桶的最后激活时间和激活次数。
        同时触发时间涟漪：时间上相邻的记忆轻微唤醒。
        """
        file_path = self._find_bucket_file(bucket_id)
        if not file_path:
            return

        try:
            post = frontmatter.load(file_path)
            post["last_active"] = now_iso()
            post["activation_count"] = post.get("activation_count", 0) + 1

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(frontmatter.dumps(post))

            # --- Time ripple: boost nearby memories within ±48h ---
            # --- 时间涟漪：±48小时内的记忆轻微唤醒 ---
            current_time = self._parse_iso_datetime(post.get("created", post.get("last_active", "")))
            if current_time is None:
                current_time = datetime.now(timezone.utc).replace(tzinfo=None)
            await self._time_ripple(bucket_id, current_time)
        except Exception as e:
            logger.warning(f"Failed to touch bucket / 触碰桶失败: {bucket_id}: {e}")

    async def _time_ripple(self, source_id: str, reference_time: datetime, hours: float = 48.0) -> None:
        """
        Slightly boost activation_count of buckets created/activated near the reference time.
        轻微提升时间相邻桶的激活次数（+0.3），不改 last_active 避免递归唤醒。
        Max 5 buckets rippled per touch to bound I/O.
        """
        try:
            all_buckets = await self.list_all(include_archive=False)
        except Exception:
            return

        rippled = 0
        max_ripple = 5
        for bucket in all_buckets:
            if rippled >= max_ripple:
                break
            if bucket["id"] == source_id:
                continue
            meta = bucket.get("metadata", {})
            # Skip pinned/permanent/feel
            if meta.get("pinned") or meta.get("protected") or meta.get("type") in ("permanent", "feel"):
                continue

            created_str = meta.get("created", meta.get("last_active", ""))
            created = self._parse_iso_datetime(created_str)
            if created is None:
                continue
            delta_hours = abs((reference_time - created).total_seconds()) / 3600

            if delta_hours <= hours:
                # Boost activation_count by 0.3 (fractional), don't change last_active
                file_path = self._find_bucket_file(bucket["id"])
                if not file_path:
                    continue
                try:
                    post = frontmatter.load(file_path)
                    current_count = post.get("activation_count", 1)
                    # Store as float for fractional increments; calculate_score handles it
                    post["activation_count"] = round(current_count + 0.3, 1)
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(frontmatter.dumps(post))
                    rippled += 1
                except Exception:
                    continue

    # ---------------------------------------------------------
    # Multi-dimensional search (core feature)
    # 多维搜索（核心功能）
    #
    # Strategy: domain pre-filter → weighted multi-dim ranking
    # 策略：主题域预筛 → 多维加权精排
    #
    # Ranking formula:
    #   total = topic(×w_topic) + time(×w_time) + importance(×w_importance)
    #
    # Per-dimension scores (normalized to 0~1):
    #   topic     = rapidfuzz weighted match (name/tags/domain/body)
    #   time      = e^(-0.02 × days) (recent memories first)
    #   importance = importance / 10
    # ---------------------------------------------------------
    async def search(
        self,
        query: str,
        limit: int = None,
        domain_filter: list[str] = None,
        query_valence: float = None,
        query_arousal: float = None,
        include_archive: bool = True,
    ) -> list[dict]:
        """
        Multi-dimensional indexed search for memory buckets.
        多维索引搜索记忆桶。

        domain_filter: pre-filter by domain (None = search all)
        query_valence/arousal: accepted for legacy API compatibility; ignored by ordinary ranking
        """
        if not query or not query.strip():
            return []

        limit = limit or self.max_results
        all_buckets = await self.list_all(include_archive=include_archive)

        if not all_buckets:
            return []

        # --- Layer 1: domain pre-filter (fast scope reduction) ---
        # --- 第一层：主题域预筛（快速缩小范围）---
        if domain_filter:
            filter_set = {d.lower() for d in domain_filter}
            candidates = [
                b for b in all_buckets
                if {d.lower() for d in b["metadata"].get("domain", [])} & filter_set
            ]
            # Fall back to full search if pre-filter yields nothing
            # 预筛为空则回退全量搜索
            if not candidates:
                candidates = all_buckets
        else:
            candidates = all_buckets

        # --- Layer 2: weighted multi-dim ranking ---
        # --- 第二层：多维加权精排 ---
        topic_scores = self.calc_topic_scores(query, candidates)
        scored = []
        for bucket in candidates:
            meta = bucket.get("metadata", {})

            try:
                # Dim 1: topic relevance (BM25 lexical text, 0~1)
                topic_score = topic_scores.get(str(bucket.get("id") or ""), 0.0)

                # Dim 2: time proximity (exponential decay, 0~1)
                time_score = self._calc_time_score(meta)

                # Dim 3: importance (direct normalization)
                importance_score = max(1, min(10, int(meta.get("importance", 5)))) / 10.0

                # --- Weighted sum / 加权求和 ---
                total = (
                    topic_score * self.w_topic
                    + time_score * self.w_time
                    + importance_score * self.w_importance
                )
                # Normalize to 0~100 for readability
                weight_sum = self.w_topic + self.w_time + self.w_importance
                normalized = (total / weight_sum) * 100 if weight_sum > 0 else 0

                if normalized >= self.fuzzy_threshold:
                    # Resolved buckets get ranking penalty after thresholding.
                    # 已解决桶先按相关性过阈值，再在排序阶段降权。
                    if meta.get("resolved", False):
                        normalized *= 0.3
                    bucket["score"] = round(normalized, 2)
                    scored.append(bucket)
            except Exception as e:
                logger.warning(
                    f"Scoring failed for bucket {bucket.get('id', '?')} / "
                    f"桶评分失败: {e}"
                )
                continue

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    # ---------------------------------------------------------
    # Topic relevance sub-score:
    # BM25 lexical relevance over weighted fields:
    # name(×3) + domain(×2.5) + tags(×2) + body(×content_weight)
    # 文本相关性子分：对加权字段做 BM25，常见词降权，身份词停用。
    # ---------------------------------------------------------
    def calc_topic_scores(self, query: str, buckets: list[dict]) -> dict[str, float]:
        query = recall_topic_query(str(query or "").strip())
        if not query or not buckets:
            return {}

        short_cjk_query = self._short_cjk_topic_query(query)
        if short_cjk_query:
            short_cjk_terms = self._short_cjk_topic_terms(short_cjk_query)
            if not short_cjk_terms:
                return {}
            return {
                str(bucket.get("id") or ""): max(
                    self._calc_short_cjk_topic_score(
                        term,
                        bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {},
                        self._bucket_searchable_content(bucket),
                    )
                    for term in short_cjk_terms
                )
                for bucket in buckets
                if bucket.get("id")
            }

        query_terms = self._lexical_query_terms(query)
        if not query_terms:
            return {}

        query_phrase = self._lexical_query_phrase(query)
        docs = []
        document_frequency: Counter[str] = Counter()
        for bucket in buckets:
            bucket_id = str(bucket.get("id") or "")
            if not bucket_id:
                continue
            term_frequency, doc_length = self._bucket_lexical_profile(bucket)
            phrase_score = self._lexical_phrase_boost(bucket, query_phrase)
            docs.append((bucket_id, term_frequency, doc_length, phrase_score))
            present = set(term_frequency)
            for term in query_terms:
                if term in present:
                    document_frequency[term] += 1

        if not docs:
            return {}

        total_docs = len(docs)
        avg_doc_length = sum(doc_length for _bid, _tf, doc_length, _phrase in docs) / max(1, total_docs)
        avg_doc_length = max(avg_doc_length, 1.0)
        k1 = 1.4
        b = 0.72
        scores: dict[str, float] = {}

        for bucket_id, term_frequency, doc_length, phrase_score in docs:
            raw_score = 0.0
            for term in query_terms:
                tf = float(term_frequency.get(term, 0.0))
                if tf <= 0:
                    continue
                df = max(1, int(document_frequency.get(term, 0)))
                idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
                denominator = tf + k1 * (1.0 - b + b * (doc_length / avg_doc_length))
                if denominator <= 0:
                    continue
                raw_score += idf * (tf * (k1 + 1.0)) / denominator

            if raw_score <= 0 and phrase_score <= 0:
                continue
            base_score = self._normalize_bm25_score(raw_score) if raw_score > 0 else 0.0
            scores[bucket_id] = round(max(base_score, phrase_score), 4)
        return scores

    def _calc_topic_score(self, query: str, bucket: dict) -> float:
        """
        Calculate text dimension relevance score (0~1).
        计算文本维度的相关性得分。
        """
        meta = bucket.get("metadata", {})
        searchable_content = self._bucket_searchable_content(bucket)

        short_cjk_query = self._short_cjk_topic_query(query)
        if short_cjk_query:
            if self._is_lexical_stop_term(short_cjk_query):
                return 0.0
            return self._calc_short_cjk_topic_score(short_cjk_query, meta, searchable_content)

        return self.calc_topic_scores(query, [bucket]).get(str(bucket.get("id") or ""), 0.0)

    def filter_specific_lexical_terms(
        self,
        terms: list[str],
        buckets: list[dict],
        *,
        preserve_terms: set[str] | None = None,
        min_specificity: float = 0.34,
        max_document_ratio: float = 0.45,
    ) -> list[str]:
        preserve = {
            self._compact_lexical_phrase(term)
            for term in (preserve_terms or set())
            if self._compact_lexical_phrase(term)
        }
        ordered = []
        seen = set()
        for term in terms or []:
            cleaned = str(term or "").strip()
            key = self._compact_lexical_phrase(cleaned)
            if not cleaned or not key or key in seen:
                continue
            seen.add(key)
            ordered.append((cleaned, key))
        if not ordered:
            return []
        if not buckets:
            return [term for term, _key in ordered]

        haystacks = [
            self._bucket_lexical_haystack(bucket)
            for bucket in buckets
            if isinstance(bucket, dict) and bucket.get("id")
        ]
        haystacks = [haystack for haystack in haystacks if haystack]
        total_docs = len(haystacks)
        if total_docs <= 0:
            return [term for term, _key in ordered]

        max_df = max(1, int(math.ceil(total_docs * max_document_ratio)))
        if total_docs < 8:
            max_df = max(1, total_docs // 2)
        max_idf = math.log(1.0 + (total_docs + 0.5) / 0.5)
        kept: list[str] = []
        for term, key in ordered:
            if key in preserve:
                kept.append(term)
                continue
            df = sum(1 for haystack in haystacks if key in haystack)
            if df <= 0:
                continue
            idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
            specificity = idf / max(max_idf, 1e-9)
            if df <= max_df or specificity >= min_specificity:
                kept.append(term)
        return kept

    def lexical_term_specificity_stats(self, terms: list[str], buckets: list[dict]) -> dict[str, dict[str, float]]:
        haystacks = [
            self._bucket_lexical_haystack(bucket)
            for bucket in buckets or []
            if isinstance(bucket, dict) and bucket.get("id")
        ]
        haystacks = [haystack for haystack in haystacks if haystack]
        total_docs = len(haystacks)
        max_idf = math.log(1.0 + (total_docs + 0.5) / 0.5) if total_docs else 1.0
        stats: dict[str, dict[str, float]] = {}
        for term in terms or []:
            cleaned = str(term or "").strip()
            key = self._compact_lexical_phrase(cleaned)
            if not key or key in stats:
                continue
            df = sum(1 for haystack in haystacks if key in haystack) if total_docs else 0
            idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5)) if df > 0 else max_idf
            stats[cleaned] = {
                "document_frequency": float(df),
                "document_count": float(total_docs),
                "specificity": round(idf / max(max_idf, 1e-9), 4),
            }
        return stats

    def _bucket_lexical_haystack(self, bucket: dict) -> str:
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        text = " ".join(
            [
                str(meta.get("name") or ""),
                " ".join(str(item) for item in meta.get("domain", []) or []),
                " ".join(str(item) for item in meta.get("tags", []) or []),
                self._bucket_searchable_content(bucket),
            ]
        )
        return self._compact_lexical_phrase(text)

    def _bucket_searchable_content(self, bucket: dict) -> str:
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        comments = meta.get("comments", []) if isinstance(meta.get("comments", []), list) else []
        year_rings = " ".join(
            str(comment.get("content") or "")
            for comment in comments
            if isinstance(comment, dict)
        )
        body = bucket_content_for_recall(bucket)[:1000]
        return f"{body}\n{year_rings[:600]}".strip()

    def _bucket_lexical_profile(self, bucket: dict) -> tuple[Counter[str], float]:
        _signature, term_frequency, weighted_length, _phrase_fields = self._bucket_lexical_cache_entry(bucket)
        return term_frequency, weighted_length

    def warm_lexical_profiles(self, buckets: list[dict]) -> int:
        warmed = 0
        for bucket in buckets or []:
            if not isinstance(bucket, dict) or not bucket.get("id"):
                continue
            self._bucket_lexical_cache_entry(bucket)
            warmed += 1
        return warmed

    def _bucket_lexical_cache_entry(
        self,
        bucket: dict,
    ) -> tuple[tuple, Counter[str], float, tuple[str, str, str, str, str]]:
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        bucket_id = str(bucket.get("id") or meta.get("id") or "")
        name = str(meta.get("name") or "")
        domain = " ".join(str(item) for item in meta.get("domain", []) or [])
        tags = " ".join(str(item) for item in meta.get("tags", []) or [])
        scene_cues = " ".join(normalize_scene_cues(meta.get("scene_cues")))
        content = self._bucket_searchable_content(bucket)
        content_weight = float(self.content_weight or 1.0)
        signature = (bucket_id, name, domain, tags, scene_cues, content, content_weight)
        cache_key = bucket_id or f"anonymous:{hash(signature)}"
        cached = self._lexical_profile_cache.get(cache_key)
        if cached is not None and cached[0] == signature:
            return cached
        fields = (
            ("name", name, 3.0),
            ("scene_cues", scene_cues, 2.8),
            ("domain", domain, 2.5),
            ("tags", tags, 2.0),
            ("content", content, content_weight),
        )
        term_frequency: Counter[str] = Counter()
        weighted_length = 0.0
        for field, text, weight in fields:
            tokens = self._lexical_tokens(text)
            if field != "content":
                compact = self._compact_lexical_phrase(text)
                if compact and not self._is_lexical_stop_term(compact):
                    tokens.append(compact)
            for token in tokens:
                term_frequency[token] += weight
            weighted_length += max(1, len(tokens)) * weight if tokens else 0.0
        value = (
            signature,
            term_frequency,
            max(weighted_length, 1.0),
            tuple(self._compact_lexical_phrase(text) for _field, text, _weight in fields),
        )
        if len(self._lexical_profile_cache) >= 4096:
            self._lexical_profile_cache.clear()
        self._lexical_profile_cache[cache_key] = value
        return value

    def _lexical_query_terms(self, query: str) -> list[str]:
        terms = self._lexical_tokens(query)
        compact = self._lexical_query_phrase(query)
        if compact:
            terms.append(compact)
        return list(dict.fromkeys(terms))

    def _lexical_query_phrase(self, query: str) -> str:
        compact = self._compact_lexical_phrase(query)
        if not compact or len(compact) > 32 or self._is_lexical_stop_term(compact):
            return ""
        if re.fullmatch(r"[\u4e00-\u9fff]+", compact) and len(compact) < 3:
            return ""
        if re.fullmatch(r"[a-z0-9_.:-]+", compact):
            if re.fullmatch(r"[\d.:-]+", compact):
                return ""
            if len(compact) < 3 and not re.search(r"\d", compact):
                return ""
        return compact

    def _lexical_phrase_boost(self, bucket: dict, query_phrase: str) -> float:
        if not query_phrase or len(query_phrase) < 3:
            return 0.0
        _signature, _term_frequency, _weighted_length, phrase_fields = self._bucket_lexical_cache_entry(bucket)
        checks = zip(phrase_fields, (0.95, 0.82, 0.56, 0.54, 0.48))
        best = 0.0
        for compact, score in checks:
            if compact and query_phrase in compact:
                best = max(best, score)
        return best

    def _lexical_tokens(self, text: str) -> list[str]:
        raw = str(text or "")
        if not raw.strip():
            return []
        candidates: list[str] = []
        candidates.extend(jieba.lcut(raw, cut_all=False))
        candidates.extend(re.findall(r"[A-Za-z]+[A-Za-z0-9_.:-]*|\d+(?:\.\d+)+|[\u4e00-\u9fff]{2,}", raw))
        tokens = []
        for candidate in candidates:
            token = self._normalize_lexical_term(candidate)
            if token:
                tokens.append(token)
        return tokens

    def _normalize_lexical_term(self, value: object) -> str:
        term = str(value or "").strip().lower()
        term = re.sub(r"^[\s，。！？、,.!?:：;；~～♡❤♥（）()\[\]【】「」『』“”\"'`-]+", "", term)
        term = re.sub(r"[\s，。！？、,.!?:：;；~～♡❤♥（）()\[\]【】「」『』“”\"'`-]+$", "", term)
        if not term or self._is_lexical_stop_term(term):
            return ""
        compact = re.sub(r"[^0-9a-z\u4e00-\u9fff_.:-]+", "", term)
        if not compact or self._is_lexical_stop_term(compact):
            return ""
        if re.fullmatch(r"[\u4e00-\u9fff]+", compact) and len(compact) < 2:
            return ""
        if re.fullmatch(r"[a-z0-9_.:-]+", compact):
            if re.fullmatch(r"[\d.:-]+", compact):
                return ""
            if len(compact) < 3 and not re.search(r"\d", compact):
                return ""
        return compact

    @staticmethod
    def _compact_lexical_phrase(value: object) -> str:
        return re.sub(
            r"[\s，。！？、,.!?:：;；~～♡❤♥（）()\[\]【】「」『』“”\"'`-]+",
            "",
            str(value or "").strip().lower(),
        )

    def _is_lexical_stop_term(self, value: object) -> bool:
        term = self._compact_lexical_phrase(value)
        return bool(term and term in self.lexical_stop_terms)

    def _build_lexical_stop_terms(self, config: dict) -> set[str]:
        values = set(GENERIC_LEXICAL_STOPWORDS)
        values.update(getattr(self, "wikilink_stopwords", set()))
        try:
            options = memory_relevance_options_from_config(config)
            values.update(options.context_terms)
        except Exception:
            pass
        try:
            identity = identity_names(config)
            values.update(identity.get("relationship_terms") or [])
            values.update(identity.get("user_aliases") or [])
        except Exception:
            pass
        normalized = set()
        for value in values:
            compact = self._compact_lexical_phrase(value)
            if compact:
                normalized.add(compact)
        return normalized

    @staticmethod
    def _normalize_bm25_score(raw_score: float) -> float:
        return max(0.0, min(1.0, 1.0 - math.exp(-float(raw_score) / 4.0)))

    @staticmethod
    def _short_cjk_topic_query(query: str) -> str:
        compact = re.sub(
            r"[\s，。！？、,.!?:：;；~～♡❤♥（）()\[\]【】「」『』“”\"'`-]+",
            "",
            str(query or ""),
        )
        if re.fullmatch(r"[\u4e00-\u9fff]{1,3}", compact):
            return compact
        return ""

    def _short_cjk_topic_terms(self, query: str) -> list[str]:
        terms = []
        for term in [query, *content_terms_for_query(query)]:
            cleaned = str(term or "").strip()
            if not re.fullmatch(r"[\u4e00-\u9fff]{1,3}", cleaned):
                continue
            if self._is_lexical_stop_term(cleaned):
                continue
            if cleaned not in terms:
                terms.append(cleaned)
        return terms

    def _calc_short_cjk_topic_score(self, query: str, meta: dict, searchable_content: str) -> float:
        def evidence(value: object) -> int:
            text = str(value or "")
            if query in text:
                return 100
            if len(query) < 3:
                return 0
            query_chars = {char for char in query if "\u4e00" <= char <= "\u9fff"}
            if len(query_chars) < 3:
                return 0
            overlap = query_chars & {char for char in text if "\u4e00" <= char <= "\u9fff"}
            coverage = len(overlap) / len(query_chars)
            return 70 if len(overlap) >= 2 and coverage >= 0.67 else 0

        name_score = evidence(meta.get("name")) * 3
        domain_score = max((evidence(d) for d in meta.get("domain", []) or []), default=0) * 2.5
        tag_score = max((evidence(tag) for tag in meta.get("tags", []) or []), default=0) * 2
        content_score = evidence(searchable_content) * self.content_weight
        weighted = name_score + domain_score + tag_score + content_score
        if weighted <= 0:
            return 0.0

        score = weighted / (100 * (3 + 2.5 + 2 + self.content_weight))
        if content_score:
            score = max(score, 0.36)
        if domain_score or tag_score:
            score = max(score, 0.45)
        if name_score:
            score = max(score, 0.50)
        return min(1.0, score)

    # ---------------------------------------------------------
    # Legacy emotion-distance diagnostic. Ordinary search no longer calls it;
    # retained so old dashboards/clients can inspect stored coordinates.
    # ---------------------------------------------------------
    def _calc_emotion_score(
        self, q_valence: float, q_arousal: float, meta: dict
    ) -> float:
        """
        Calculate emotion resonance score (0~1, closer = higher).
        计算情感共鸣度（0~1，越近越高）。
        """
        if q_valence is None or q_arousal is None:
            return 0.5  # No emotion coordinates → neutral / 无情感坐标时给中性分

        try:
            b_valence = float(meta.get("valence", 0.5))
            b_arousal = float(meta.get("arousal", 0.3))
        except (ValueError, TypeError):
            return 0.5

        # Euclidean distance, max sqrt(2) ≈ 1.414
        dist = math.sqrt((q_valence - b_valence) ** 2 + (q_arousal - b_arousal) ** 2)
        return max(0.0, 1.0 - dist / 1.414)

    # ---------------------------------------------------------
    # Time proximity sub-score:
    # More recent activation → higher score
    # 时间亲近子分：距上次激活越近分越高
    # ---------------------------------------------------------
    def _calc_time_score(self, meta: dict) -> float:
        """
        Calculate time proximity score (0~1, more recent = higher).
        计算时间亲近度。
        """
        last_active_str = meta.get("last_active", meta.get("created", ""))
        last_active = self._parse_iso_datetime(last_active_str)
        if last_active is None:
            days = 30
        else:
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            days = max(0.0, (now - last_active).total_seconds() / 86400)
        return math.exp(-0.02 * days)

    def _parse_iso_datetime(self, value) -> Optional[datetime]:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
        if parsed.tzinfo is not None:
            return parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed

    # ---------------------------------------------------------
    # List all buckets
    # 列出所有桶
    # ---------------------------------------------------------
    async def list_all(self, include_archive: bool = False) -> list[dict]:
        """
        Recursively walk directories (including domain subdirs), list all buckets.
        递归遍历目录（含域子目录），列出所有记忆桶。
        """
        buckets = []

        dirs = [self.permanent_dir, self.dynamic_dir, self.feel_dir]
        if include_archive:
            dirs.append(self.archive_dir)

        for dir_path in dirs:
            if not os.path.exists(dir_path):
                continue
            for root, _, files in os.walk(dir_path):
                for filename in files:
                    if not filename.endswith(".md"):
                        continue
                    file_path = os.path.join(root, filename)
                    bucket = self._load_bucket(file_path)
                    if bucket:
                        buckets.append(bucket)

        return buckets

    # ---------------------------------------------------------
    # Statistics (counts per category + total size)
    # 统计信息（各分类桶数量 + 总体积）
    # ---------------------------------------------------------
    async def get_stats(self) -> dict:
        """
        Return memory bucket statistics (including domain subdirs).
        返回记忆桶的统计数据。
        """
        stats = {
            "permanent_count": 0,
            "dynamic_count": 0,
            "archive_count": 0,
            "feel_count": 0,
            "total_size_kb": 0.0,
            "domains": {},
        }

        for subdir, key in [
            (self.permanent_dir, "permanent_count"),
            (self.dynamic_dir, "dynamic_count"),
            (self.archive_dir, "archive_count"),
            (self.feel_dir, "feel_count"),
        ]:
            if not os.path.exists(subdir):
                continue
            for root, _, files in os.walk(subdir):
                for f in files:
                    if f.endswith(".md"):
                        stats[key] += 1
                        fpath = os.path.join(root, f)
                        try:
                            stats["total_size_kb"] += os.path.getsize(fpath) / 1024
                        except OSError:
                            pass
                        # Per-domain counts / 每个域的桶数量
                        domain_name = os.path.basename(root)
                        if domain_name != os.path.basename(subdir):
                            stats["domains"][domain_name] = stats["domains"].get(domain_name, 0) + 1

        return stats

    # ---------------------------------------------------------
    # Archive bucket (move from permanent/dynamic into archive)
    # 归档桶（从 permanent/dynamic 移入 archive）
    # Called by decay engine to simulate "forgetting"
    # 由衰减引擎调用，模拟"遗忘"
    # ---------------------------------------------------------
    async def archive(self, bucket_id: str) -> bool:
        """
        Move a bucket into the archive directory (preserving domain subdirs).
        将指定桶移入归档目录（保留域子目录结构）。
        """
        file_path = self._find_bucket_file(bucket_id)
        if not file_path:
            return False

        try:
            # Read once, get domain info and update type / 一次性读取
            post = frontmatter.load(file_path)
            if bool(post.get("source_record_immutable", False)):
                logger.warning("Refused immutable source record archive / 拒绝归档不可变来源记录: %s", bucket_id)
                return False
            if (
                str(post.get("object_kind") or "").strip().lower() == "scene"
                or str(post.get("memory_value_source") or "").strip() == "authored_scene"
            ):
                return await self.set_scene_status(
                    bucket_id,
                    "archived",
                    changed_at=now_iso(),
                )
            domain = post.get("domain", ["未分类"])
            if not isinstance(domain, list):
                domain = [domain]
            primary_domain = sanitize_name(domain[0]) if domain else "未分类"
            archive_subdir = os.path.join(self.archive_dir, primary_domain)
            os.makedirs(archive_subdir, exist_ok=True)

            dest = safe_path(archive_subdir, os.path.basename(file_path))

            # Update type marker then move file / 更新类型标记后移动文件
            post["type"] = "archived"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(frontmatter.dumps(post))

            # Use shutil.move for cross-filesystem safety
            # 使用 shutil.move 保证跨文件系统安全
            shutil.move(file_path, str(dest))
        except Exception as e:
            logger.error(
                f"Failed to archive bucket / 归档桶失败: {bucket_id}: {e}"
            )
            return False

        logger.info(f"Archived bucket / 归档记忆桶: {bucket_id} → archive/{primary_domain}/")
        return True

    async def activate(self, bucket_id: str) -> bool:
        """
        Move an archived bucket back to dynamic storage and mark it active.
        将归档桶移回 dynamic，并标为 active。
        """
        file_path = self._find_bucket_file(bucket_id)
        if not file_path:
            return False

        try:
            post = frontmatter.load(file_path)
            if (
                str(post.get("object_kind") or "").strip().lower() == "scene"
                or str(post.get("memory_value_source") or "").strip() == "authored_scene"
            ):
                return await self.set_scene_status(
                    bucket_id,
                    "active",
                    changed_at=now_iso(),
                )
            domain = post.get("domain", ["未分类"])
            if not isinstance(domain, list):
                domain = [domain]
            primary_domain = sanitize_name(domain[0]) if domain else "未分类"
            target_dir = os.path.join(self.dynamic_dir, primary_domain)
            os.makedirs(target_dir, exist_ok=True)
            dest = safe_path(target_dir, os.path.basename(file_path))

            post["type"] = "dynamic"
            post["active"] = True
            post["deprecated"] = False
            post["resolved"] = False
            post["updated_at"] = now_iso()
            post["last_active"] = post.get("last_active") or post["updated_at"]
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(frontmatter.dumps(post))

            if os.path.normpath(file_path) != os.path.normpath(str(dest)):
                shutil.move(file_path, str(dest))
        except Exception as e:
            logger.error(
                f"Failed to activate bucket / 恢复桶失败: {bucket_id}: {e}"
            )
            return False

        logger.info(f"Activated bucket / 恢复记忆桶: {bucket_id} → dynamic/{primary_domain}/")
        return True

    async def set_scene_status(
        self,
        bucket_id: str,
        status: str,
        *,
        changed_at: str,
    ) -> bool:
        """
        Move one authored Scene between active storage and archive without
        changing its content, source evidence, revisions, or annotations.
        在 active 与 archive 间移动 authored Scene，不改正文、来源证据、修订或年轮。

        Canonical Scene validation and optimistic concurrency live in the
        server façade. This storage primitive only performs the requested,
        metadata-preserving transition.
        """
        safe_status = str(status or "").strip().lower()
        if safe_status not in {"active", "archived"}:
            return False

        file_path = self._find_bucket_file(bucket_id)
        if not file_path:
            return False

        try:
            post = frontmatter.load(file_path)
            if bool(post.get("source_record_immutable", False)):
                logger.warning(
                    "Refused immutable source record Scene status change / "
                    "拒绝修改不可变来源 Scene 状态: %s",
                    bucket_id,
                )
                return False

            domain = post.get("domain", ["未分类"])
            if not isinstance(domain, list):
                domain = [domain]
            primary_domain = sanitize_name(domain[0]) if domain else "未分类"
            current_type = str(post.get("type") or "dynamic").strip().lower()
            current_status = (
                "archived"
                if current_type == "archived"
                or str(post.get("scene_status") or "").strip().lower() == "archived"
                else "active"
            )
            history = post.get("scene_status_history")
            history = list(history) if isinstance(history, list) else []
            history.append(
                {
                    "from": current_status,
                    "to": safe_status,
                    "changed_at": changed_at,
                    "source": "set_scene_status",
                }
            )
            if len(history) > 20:
                history = [history[0], *history[-19:]]

            if safe_status == "archived":
                previous_type = current_type if current_type in {"dynamic", "permanent"} else "dynamic"
                post["scene_pre_archive_type"] = previous_type
                post["type"] = "archived"
                post["active"] = False
                post["archived_at"] = changed_at
                target_root = self.archive_dir
            else:
                previous_type = str(post.get("scene_pre_archive_type") or "dynamic").strip().lower()
                restored_type = previous_type if previous_type in {"dynamic", "permanent"} else "dynamic"
                post["type"] = restored_type
                post["active"] = True
                post["deprecated"] = False
                post["restored_at"] = changed_at
                target_root = self.permanent_dir if restored_type == "permanent" else self.dynamic_dir

            post["scene_status"] = safe_status
            post["scene_status_history"] = history
            post["last_status_change_source"] = "set_scene_status"
            post["updated_at"] = changed_at

            target_dir = os.path.join(target_root, primary_domain)
            os.makedirs(target_dir, exist_ok=True)
            dest = safe_path(target_dir, os.path.basename(file_path))
            same_path = os.path.normpath(file_path) == os.path.normpath(str(dest))
            if not same_path and os.path.exists(dest):
                logger.error(
                    "Refused Scene status move because destination exists / "
                    "Scene 状态迁移目标已存在: %s",
                    dest,
                )
                return False

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(frontmatter.dumps(post))
            if not same_path:
                shutil.move(file_path, str(dest))
        except Exception as e:
            logger.error(
                "Failed to change Scene status / 修改 Scene 状态失败: %s: %s",
                bucket_id,
                e,
            )
            return False

        logger.info(
            "Changed Scene status / 修改 Scene 状态: %s → %s/%s/",
            bucket_id,
            safe_status,
            primary_domain,
        )
        return True

    # ---------------------------------------------------------
    # Internal: find bucket file across all three directories
    # 内部：在三个目录中查找桶文件
    # ---------------------------------------------------------
    def _find_bucket_file(self, bucket_id: str) -> Optional[str]:
        """
        Recursively search permanent/dynamic/archive for a bucket file
        matching the given ID.
        在 permanent/dynamic/archive 中递归查找指定 ID 的桶文件。
        """
        if not bucket_id:
            return None
        for dir_path in [self.permanent_dir, self.dynamic_dir, self.archive_dir, self.feel_dir]:
            if not os.path.exists(dir_path):
                continue
            for root, _, files in os.walk(dir_path):
                for fname in files:
                    if not fname.endswith(".md"):
                        continue
                    # Match by exact ID segment in filename
                    # 通过文件名中的 ID 片段精确匹配
                    name_part = fname[:-3]  # remove .md
                    if name_part == bucket_id or name_part.endswith(f"_{bucket_id}"):
                        return os.path.join(root, fname)
        return None

    # ---------------------------------------------------------
    # Internal: load bucket data from .md file
    # 内部：从 .md 文件加载桶数据
    # ---------------------------------------------------------
    def _load_bucket(self, file_path: str) -> Optional[dict]:
        """
        Parse a Markdown file and return structured bucket data.
        解析 Markdown 文件，返回桶的结构化数据。
        """
        try:
            raw = Path(file_path).read_text(encoding="utf-8")
            post = frontmatter.load(file_path)
            return {
                "id": post.get("id", Path(file_path).stem),
                "metadata": dict(post.metadata),
                "content": post.content,
                "path": file_path,
                "content_start_line": _markdown_body_start_line(raw),
            }
        except Exception as e:
            logger.warning(
                f"Failed to load bucket file / 加载桶文件失败: {file_path}: {e}"
            )
            return None


def _markdown_body_start_line(text: str) -> int:
    lines = str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if not lines or lines[0].strip() != "---":
        return 1
    for index, line in enumerate(lines[1:], start=2):
        if line.strip() == "---":
            body_start = index + 1
            while body_start <= len(lines) and not lines[body_start - 1].strip():
                body_start += 1
            return max(1, body_start)
    return 1
