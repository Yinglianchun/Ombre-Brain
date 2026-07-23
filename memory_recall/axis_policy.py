from __future__ import annotations

import re
from typing import Any

from utils import bucket_content_for_recall


DEFAULT_TECHNICAL_AXIS_TERMS = (
    "esp32",
    "mpr121",
    "sqlite",
    "模块",
    "硬件",
    "接口",
    "端点",
    "api",
    "gateway",
    "bridge",
    "mcp",
    "embedding",
    "rerank",
    "代码",
    "开源项目",
)
DEFAULT_TECHNICAL_DATABASE_TERMS = (
    "schema",
    "端点",
    "接口",
    "代码",
    "实现",
    "导入",
    "索引",
    "查询",
    "字段",
    "表结构",
    "迁移",
    "sqlite",
    "sql",
)
DEFAULT_TECHNICAL_DOMAIN_TERMS = (
    "projectcode",
    "hardwareprotocol",
    "hardware",
    "code",
    "debug",
    "技术",
    "技术计划",
    "项目",
    "工程",
    "代码",
    "硬件",
    "协议",
    "数据库",
    "开发",
)


class MemoryAxisPolicy:
    """Classify query axes and candidate nodes without owning recall evidence."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config if isinstance(config, dict) else {}
        self.technical_axis_terms = self.config_terms(
            self.config,
            "technical_axis_terms",
            DEFAULT_TECHNICAL_AXIS_TERMS,
        )
        self.technical_database_terms = self.config_terms(
            self.config,
            "technical_database_terms",
            DEFAULT_TECHNICAL_DATABASE_TERMS,
        )
        self.technical_domain_terms = self.config_terms(
            self.config,
            "technical_domain_terms",
            DEFAULT_TECHNICAL_DOMAIN_TERMS,
        )

    @staticmethod
    def compact_text(value: object) -> str:
        return re.sub(
            r"[^0-9a-z\u4e00-\u9fff_.:-]+",
            "",
            str(value or "").strip().lower(),
        )

    @staticmethod
    def config_terms(
        config: dict[str, Any],
        key: str,
        fallback: tuple[str, ...],
    ) -> tuple[str, ...]:
        value = config.get(key) if isinstance(config, dict) and key in config else fallback
        raw_terms = [value] if isinstance(value, str) else list(value or [])
        return tuple(str(term).strip() for term in raw_terms if str(term or "").strip())

    def node_text(self, node: dict) -> str:
        if not isinstance(node, dict):
            return ""
        metadata = node.get("metadata", {}) if isinstance(node.get("metadata"), dict) else {}
        if self._is_moment_node(node):
            fields = [
                str(node.get("text") or ""),
                str(node.get("content") or ""),
                str(metadata.get("annotation_summary") or ""),
                str(metadata.get("bucket_name") or ""),
                " ".join(str(tag) for tag in metadata.get("bucket_tags", []) or []),
                " ".join(str(item) for item in metadata.get("bucket_domain", []) or []),
            ]
        else:
            fields = [
                str(metadata.get("name") or node.get("id") or ""),
                str(metadata.get("annotation_summary") or ""),
                " ".join(str(tag) for tag in metadata.get("tags", []) or []),
                " ".join(str(item) for item in metadata.get("domain", []) or []),
                bucket_content_for_recall(node),
            ]
        return self.compact_text(" ".join(fields))

    def candidate_matches(self, query_plan: Any, node: dict) -> bool:
        groups = getattr(query_plan, "activated_axis_groups", ()) or ()
        if not groups:
            return True
        return any(self.group_matches_node(group, node) for group in groups)

    def group_matches_node(self, group: tuple[str, ...], node: dict) -> bool:
        text = self.node_text(node)
        keys = [self.compact_text(term) for term in group if self.compact_text(term)]
        return bool(keys and text and all(key in text for key in keys))

    def has_technical_axis(self, query_plan: Any) -> bool:
        terms = " ".join(
            str(term or "")
            for term in getattr(query_plan, "activated_axis_terms", ()) or ()
        )
        key = self.compact_text(terms)
        if not key:
            return False
        if any(self.compact_text(marker) in key for marker in self.technical_axis_terms):
            return True
        if "数据库" not in key:
            return False
        query_key = self.compact_text(getattr(query_plan, "query", ""))
        return any(
            self.compact_text(marker) in query_key
            for marker in self.technical_database_terms
        )

    def node_has_technical_domain(self, node: dict) -> bool:
        if not isinstance(node, dict):
            return False
        metadata = node.get("metadata", {}) if isinstance(node.get("metadata"), dict) else {}
        domains = (
            metadata.get("bucket_domain")
            if self._is_moment_node(node)
            else metadata.get("domain")
        )
        domain_text = self.compact_text(" ".join(str(item) for item in domains or []))
        if not domain_text:
            return False
        return any(
            self.compact_text(marker) in domain_text
            for marker in self.technical_domain_terms
        )

    def node_name_matches_primary(self, query_plan: Any, node: dict) -> bool:
        groups = getattr(query_plan, "activated_axis_groups", ()) or ()
        if not groups or not groups[0]:
            return False
        primary_key = self.compact_text(groups[0][0])
        if not primary_key:
            return False
        metadata = node.get("metadata", {}) if isinstance(node.get("metadata"), dict) else {}
        if self._is_moment_node(node):
            name = str(metadata.get("bucket_name") or "")
        else:
            name = str(metadata.get("name") or node.get("name") or "")
        return primary_key in self.compact_text(name)

    def domain_mismatch(self, query_plan: Any, node: dict) -> bool:
        if not self.has_technical_axis(query_plan):
            return False
        if self.node_has_technical_domain(node):
            return False
        if self.node_name_matches_primary(query_plan, node):
            return False
        return True

    def debug(self, query_plan: Any, *, matched: bool) -> dict[str, Any]:
        return {
            "activated_axis_terms": list(
                getattr(query_plan, "activated_axis_terms", ()) or ()
            ),
            "activated_axis_groups": [
                list(group)
                for group in (getattr(query_plan, "activated_axis_groups", ()) or ())
            ],
            "activated_axis_multi": bool(
                getattr(query_plan, "activated_axis_multi", False)
            ),
            "activated_axis_matched": bool(matched),
            "activated_axis_technical": self.has_technical_axis(query_plan),
            "auto": True,
        }

    def rejection(
        self,
        query_plan: Any,
        node: dict,
        *,
        bypass: bool = False,
    ) -> tuple[str, dict[str, Any]] | None:
        if not (getattr(query_plan, "activated_axis_groups", ()) or ()) or bypass:
            return None
        matched = self.candidate_matches(query_plan, node)
        if matched:
            if self.domain_mismatch(query_plan, node):
                debug = self.debug(query_plan, matched=True)
                debug["activated_axis_domain_matched"] = False
                return "activated_axis_mismatch", debug
            return None
        return "activated_axis_mismatch", self.debug(query_plan, matched=False)

    @staticmethod
    def _is_moment_node(node: dict) -> bool:
        return "bucket_id" in node or bool(node.get("moment_id"))
