# ============================================================
# Module: Memory Import Engine (import_memory.py)
# 模块：历史记忆导入引擎
#
# Imports conversation history from various platforms into OB.
# 将各平台对话历史导入 OB 记忆系统。
#
# Supports: Claude JSON, ChatGPT export, DeepSeek, Markdown, plain text
# 支持格式：Claude JSON、ChatGPT 导出、DeepSeek、Markdown、纯文本
#
# Features:
#   - Chunked processing with resume support
#   - Progress persistence (import_state.json)
#   - Raw preservation mode for special contexts
#   - Post-import frequency pattern detection
# ============================================================

import os
import json
import hashlib
import logging
import asyncio
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import jieba
from rapidfuzz import fuzz

from utils import LOCAL_TZ, bucket_text_for_embedding, count_tokens_approx, now_iso, strip_affect_anchor

logger = logging.getLogger("ombre_brain.import")


# ============================================================
# Format Parsers — normalize any format to conversation turns
# 格式解析器 — 将任意格式标准化为对话轮次
# ============================================================

_MARKDOWN_ROLE_RE = re.compile(
    r"^\s*(?:>\s*)?(?:[-*+]\s*)?(?:#{1,6}\s*)?(?:\*\*)?([A-Za-z0-9_\-\u4e00-\u9fff]+)(?:\*\*)?\s*[:：]\s*(.*)$"
)
_MARKDOWN_USER_LABELS = {
    "human",
    "user",
    "me",
    "你",
    "我",
    "用户",
    "人类",
}
_MARKDOWN_ASSISTANT_LABELS = {
    "assistant",
    "claude",
    "ai",
    "gpt",
    "chatgpt",
    "bot",
    "deepseek",
    "gemini",
    "qwen",
    "助手",
    "模型",
    "ai助手",
}
_CHATGPT_IMPORT_ROLES = {"user", "assistant"}


def _clean_chatgpt_role(role: object) -> str:
    normalized = str(role or "user").strip().lower()
    return normalized if normalized in _CHATGPT_IMPORT_ROLES else ""


def _detect_markdown_role_line(
    line: str,
    *,
    user_labels: set[str] | None = None,
    assistant_labels: set[str] | None = None,
) -> tuple[str, str] | None:
    """Return (role, content_after_prefix) for simple role-prefixed Markdown lines."""
    match = _MARKDOWN_ROLE_RE.match(line)
    if not match:
        return None
    label = match.group(1).strip().lower()
    content_after = match.group(2).strip()
    if content_after.startswith("**"):
        content_after = content_after[2:].lstrip()
    if label in (user_labels or _MARKDOWN_USER_LABELS):
        return "user", content_after
    if label in (assistant_labels or _MARKDOWN_ASSISTANT_LABELS):
        return "assistant", content_after
    return None

def _parse_claude_json(data: dict | list) -> list[dict]:
    """Parse Claude.ai export JSON → [{role, content, timestamp}, ...]"""
    turns = []
    conversations = data if isinstance(data, list) else [data]
    for conv in conversations:
        if not isinstance(conv, dict):
            continue
        messages = conv.get("chat_messages", conv.get("messages", []))
        if not isinstance(messages, list):
            continue
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("text", msg.get("content", ""))
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            elif isinstance(content, dict):
                content = " ".join(
                    str(p.get("text", p)) if isinstance(p, dict) else str(p)
                    for p in content.get("parts", [])
                    if p
                )
            elif not isinstance(content, str):
                content = str(content)
            if not content or not content.strip():
                continue
            role = msg.get("sender", msg.get("role", "user"))
            ts = msg.get("created_at", msg.get("timestamp", ""))
            turns.append({"role": role, "content": content.strip(), "timestamp": ts})
    return turns


def _parse_chatgpt_json(data: list | dict) -> list[dict]:
    """Parse ChatGPT export JSON → [{role, content, timestamp}, ...]"""
    turns = []
    conversations = data if isinstance(data, list) else [data]
    for conv in conversations:
        if not isinstance(conv, dict):
            continue
        mapping = conv.get("mapping", {})
        if isinstance(mapping, dict) and mapping:
            # ChatGPT uses a tree structure with mapping
            sorted_nodes = sorted(
                [node for node in mapping.values() if isinstance(node, dict)],
                key=lambda n: (n.get("message") or {}).get("create_time", 0) or 0,
            )
            for node in sorted_nodes:
                msg = node.get("message")
                if not msg or not isinstance(msg, dict):
                    continue
                author = msg.get("author", {})
                raw_role = author.get("role", "user") if isinstance(author, dict) else "user"
                role = _clean_chatgpt_role(raw_role)
                if not role:
                    continue
                content_obj = msg.get("content", {})
                if isinstance(content_obj, dict):
                    content_parts = content_obj.get("parts", [])
                    content = " ".join(str(p) for p in content_parts if p)
                elif isinstance(content_obj, str):
                    content = content_obj
                else:
                    content = ""
                if not isinstance(content, str):
                    content = str(content)
                if not content.strip():
                    continue
                # Preserve the export's original timestamp. It is normalized only
                # when deriving the bucket event date, so source refs remain exact.
                ts = msg.get("create_time", "")
                turns.append({"role": role, "content": content.strip(), "timestamp": str(ts)})
        else:
            # Simpler format: list of messages
            messages = conv.get("messages", [])
            if not isinstance(messages, list):
                continue
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                author = msg.get("author", {})
                raw_role = msg.get("role") or (author.get("role") if isinstance(author, dict) else None) or "user"
                role = _clean_chatgpt_role(raw_role)
                if not role:
                    continue
                content = msg.get("content", msg.get("text", ""))
                if isinstance(content, dict):
                    content = " ".join(str(p) for p in content.get("parts", []))
                elif isinstance(content, list):
                    content = " ".join(
                        str(p.get("text", p)) if isinstance(p, dict) else str(p)
                        for p in content
                        if p
                    )
                elif not isinstance(content, str):
                    content = str(content)
                if not content or not content.strip():
                    continue
                ts = msg.get("timestamp", msg.get("create_time", ""))
                turns.append({"role": role, "content": content.strip(), "timestamp": str(ts)})
    return turns


def _parse_markdown(
    text: str,
    *,
    user_labels: set[str] | None = None,
    assistant_labels: set[str] | None = None,
) -> list[dict]:
    """Parse Markdown/plain text → [{role, content, timestamp}, ...]"""
    resolved_user_labels = set(_MARKDOWN_USER_LABELS)
    resolved_assistant_labels = set(_MARKDOWN_ASSISTANT_LABELS)
    resolved_user_labels.update(
        str(label).strip().lower() for label in (user_labels or set()) if str(label).strip()
    )
    resolved_assistant_labels.update(
        str(label).strip().lower() for label in (assistant_labels or set()) if str(label).strip()
    )
    # Try to detect conversation patterns
    lines = text.split("\n")
    turns = []
    current_role = "user"
    current_content = []

    def append_current_turn():
        content = "\n".join(current_content).strip()
        if content:
            turns.append({"role": current_role, "content": content, "timestamp": ""})

    for line in lines:
        stripped = line.strip()
        role_line = _detect_markdown_role_line(
            stripped,
            user_labels=resolved_user_labels,
            assistant_labels=resolved_assistant_labels,
        )
        if role_line:
            if current_content:
                append_current_turn()
            current_role, content_after = role_line
            current_content = [content_after] if content_after else []
        else:
            current_content.append(line)

    if current_content:
        append_current_turn()

    # If no role patterns detected, treat entire text as one big chunk
    if not turns:
        turns = [{"role": "user", "content": text.strip(), "timestamp": ""}]

    return turns


def detect_and_parse(
    raw_content: str,
    filename: str = "",
    *,
    user_labels: set[str] | None = None,
    assistant_labels: set[str] | None = None,
) -> list[dict]:
    """
    Auto-detect format and parse to normalized turns.
    自动检测格式并解析为标准化的对话轮次。
    """
    ext = Path(filename).suffix.lower() if filename else ""

    # Try JSON first
    if ext in (".json", "") or raw_content.strip().startswith(("{", "[")):
        try:
            data = json.loads(raw_content)
            # Detect Claude vs ChatGPT format
            if isinstance(data, list):
                sample = data[0] if data else {}
            else:
                sample = data

            if isinstance(sample, dict):
                if "chat_messages" in sample:
                    return _parse_claude_json(data)
                if "mapping" in sample:
                    return _parse_chatgpt_json(data)
                if "messages" in sample:
                    # Could be either — try ChatGPT first, fall back to Claude
                    msgs = sample["messages"]
                    if msgs and isinstance(msgs[0], dict) and "content" in msgs[0]:
                        if isinstance(msgs[0]["content"], dict):
                            return _parse_chatgpt_json(data)
                    return _parse_claude_json(data)
                # Single conversation object with role/content messages
                if "role" in sample and "content" in sample:
                    return _parse_claude_json(data)
        except (json.JSONDecodeError, KeyError, IndexError, AttributeError, TypeError):
            pass

    # Fall back to markdown/text
    return _parse_markdown(
        raw_content,
        user_labels=user_labels,
        assistant_labels=assistant_labels,
    )


def parse_operit_memory_backup(raw_content: str) -> dict | None:
    """Return an Operit memory backup without rewriting entry bodies."""
    try:
        data = json.loads(raw_content)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(data, dict) or "memories" not in data:
        return None
    memories = data.get("memories")
    if not isinstance(memories, list):
        raise ValueError("Operit backup field 'memories' must be a list")

    # Avoid treating an unrelated JSON object with a generic memories key as
    # Operit. Empty exports are identified by Operit's exportDate/links fields.
    known_entry_keys = {
        "uuid",
        "title",
        "content",
        "contentType",
        "source",
        "credibility",
        "importance",
        "folderPath",
        "createdAt",
        "updatedAt",
        "tagNames",
    }
    root_has_operit_markers = bool("exportDate" in data or "links" in data)
    entries_have_operit_markers = False
    if memories:
        entry_marker_keys = known_entry_keys - {"content"}
        entries_have_operit_markers = all(
            isinstance(item, dict)
            and "content" in item
            and bool(entry_marker_keys.intersection(item))
            for item in memories
        )
    if not (root_has_operit_markers or entries_have_operit_markers):
        return None

    return {
        "memories": memories,
        "links": data.get("links") if isinstance(data.get("links"), list) else [],
        "export_date": data.get("exportDate"),
    }


# ============================================================
# Chunking — split turns into ~10k token windows
# 分窗 — 按对话轮次边界切为 ~10k token 窗口
# ============================================================

_OVERLAP_CONTEXT_NOTICE = "[上下文提示] 以下是上一段结尾，只用于理解前后关系，请不要从这里单独提取记忆。"
_CURRENT_SEGMENT_NOTICE = "[本段内容]"
DEFAULT_IMPORT_CHUNK_TOKENS = 3500
_IMPORT_DUPLICATE_SIMILARITY = 88.0
_OPERIT_TAGGING_INPUT_CHARS = 2000


def _normalize_import_text(text: str) -> str:
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", str(text or ""))
    text = strip_affect_anchor(text)
    text = re.sub(r"[\s\u3000]+", "", text.lower())
    return re.sub(r"[^0-9a-zA-Z_\u4e00-\u9fff]+", "", text)


def _import_similarity_text(text: str) -> str:
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", str(text or "").lower())
    text = strip_affect_anchor(text)
    text = re.sub(r"[^0-9a-zA-Z_\u4e00-\u9fff]+", " ", text)
    return " ".join(token for token in jieba.lcut(text) if token.strip())


def _import_content_hash(text: str) -> str:
    normalized = _normalize_import_text(text)
    return hashlib.sha256(normalized.encode()).hexdigest()


def _int_between(value, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def _float_between(value, default: float, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def _bool_value(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _clean_import_list(value, *, max_items: int, max_chars: int, default: list[str] | None = None) -> list[str]:
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, list):
        raw_items = value
    else:
        raw_items = []
    cleaned: list[str] = []
    for item in raw_items:
        text = re.sub(r"\s+", "", str(item or "").strip())
        text = text.strip("，。；;、,. ")
        if not text:
            continue
        text = text[:max_chars]
        if text and text not in cleaned:
            cleaned.append(text)
        if len(cleaned) >= max_items:
            break
    return cleaned or list(default or [])


def _dedupe_list(values: list) -> list:
    seen = set()
    result = []
    for value in values or []:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _date_key(value) -> str:
    text = str(value or "").strip()
    match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    return match.group(0) if match else ""


_IMPORT_LOCAL_DATE_FORMATS = (
    "%Y/%m/%dT%H:%M:%S",
    "%Y/%m/%dT%H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%Y年%m月%d日 %H:%M:%S",
    "%Y年%m月%d日 %H:%M",
    "%Y年%m月%d日",
)


def _import_timestamp_datetime(value) -> datetime | None:
    """Normalize common export timestamps to LOCAL_TZ without changing provenance."""
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if not text:
        return None

    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text):
        try:
            epoch = float(text)
            if epoch <= 0:
                return None
            magnitude = abs(epoch)
            if magnitude >= 1e17:  # nanoseconds
                epoch /= 1_000_000_000.0
            elif magnitude >= 1e14:  # microseconds
                epoch /= 1_000_000.0
            elif magnitude >= 1e11:  # milliseconds
                epoch /= 1_000.0
            return datetime.fromtimestamp(epoch, tz=timezone.utc).astimezone(LOCAL_TZ)
        except (OverflowError, OSError, ValueError):
            return None

    normalized = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        parsed = None
    if parsed is None:
        for fmt in _IMPORT_LOCAL_DATE_FORMATS:
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=LOCAL_TZ)
    return parsed.astimezone(LOCAL_TZ)


def _import_event_date(value) -> str:
    parsed = _import_timestamp_datetime(value)
    return parsed.date().isoformat() if parsed else ""


def _tail_for_overlap(text: str, overlap_tokens: int) -> str:
    lines = text.splitlines() or [text]
    tail: list[str] = []
    current_tokens = 0
    max_chars = max(40, int(overlap_tokens / 1.8))

    for line in reversed(lines):
        line_tokens = count_tokens_approx(line)
        if not tail and line_tokens > overlap_tokens:
            return line[-max_chars:].strip()
        if tail and current_tokens + line_tokens > overlap_tokens:
            break
        tail.insert(0, line)
        current_tokens += line_tokens

    return "\n".join(tail).strip()


def _split_oversized_turn(role_label: str, content: str, target_tokens: int) -> list[str]:
    """Split a single very long turn into model-sized chunks with small context overlap."""
    prefix = f"[{role_label}] "
    segments: list[str] = []
    current_lines: list[str] = []
    current_tokens = count_tokens_approx(prefix)
    content_budget = max(80, int(target_tokens * 0.85))
    overlap_tokens = max(20, int(target_tokens * 0.12))
    max_chars = max(80, int(content_budget / 1.8))

    def flush_current():
        nonlocal current_lines, current_tokens
        body = "\n".join(current_lines).strip()
        if body:
            segments.append(body)
        current_lines = []
        current_tokens = count_tokens_approx(prefix)

    for line in content.splitlines() or [content]:
        line_tokens = count_tokens_approx(line)
        if line_tokens > content_budget:
            flush_current()
            for start in range(0, len(line), max_chars):
                segment = line[start:start + max_chars].strip()
                if segment:
                    segments.append(segment)
            continue

        if current_lines and current_tokens + line_tokens > content_budget:
            flush_current()
        current_lines.append(line)
        current_tokens += line_tokens

    flush_current()

    pieces: list[str] = []
    previous_tail = ""
    for segment in segments:
        body = prefix + segment
        if previous_tail:
            pieces.append(
                f"{_OVERLAP_CONTEXT_NOTICE}\n"
                f"{prefix}{previous_tail}\n\n"
                f"{_CURRENT_SEGMENT_NOTICE}\n"
                f"{body}"
            )
        else:
            pieces.append(body)
        previous_tail = _tail_for_overlap(segment, overlap_tokens)

    return pieces


def chunk_turns(turns: list[dict], target_tokens: int = DEFAULT_IMPORT_CHUNK_TOKENS) -> list[dict]:
    """
    Group conversation turns into chunks of ~target_tokens.
    Returns list of {content, timestamp_start, timestamp_end, turn_count}.
    按对话轮次边界将对话分为 ~target_tokens 大小的窗口。
    """
    chunks = []
    current_lines = []
    current_tokens = 0
    first_ts = ""
    last_ts = ""
    turn_count = 0

    for turn in turns:
        role_label = "用户" if turn["role"] in ("user", "human") else "AI"
        line = f"[{role_label}] {turn['content']}"
        line_tokens = count_tokens_approx(line)

        # If single turn exceeds target, split it
        if line_tokens > target_tokens * 1.5:
            # Flush current
            if current_lines:
                chunks.append({
                    "content": "\n".join(current_lines),
                    "timestamp_start": first_ts,
                    "timestamp_end": last_ts,
                    "turn_count": turn_count,
                })
                current_lines = []
                current_tokens = 0
                turn_count = 0
                first_ts = ""

            for split_line in _split_oversized_turn(role_label, turn["content"], target_tokens):
                chunks.append({
                    "content": split_line,
                    "timestamp_start": turn.get("timestamp", ""),
                    "timestamp_end": turn.get("timestamp", ""),
                    "turn_count": 1,
                })
            continue

        if current_tokens + line_tokens > target_tokens and current_lines:
            chunks.append({
                "content": "\n".join(current_lines),
                "timestamp_start": first_ts,
                "timestamp_end": last_ts,
                "turn_count": turn_count,
            })
            current_lines = []
            current_tokens = 0
            turn_count = 0
            first_ts = ""

        if not first_ts:
            first_ts = turn.get("timestamp", "")
        last_ts = turn.get("timestamp", "")
        current_lines.append(line)
        current_tokens += line_tokens
        turn_count += 1

    if current_lines:
        chunks.append({
            "content": "\n".join(current_lines),
            "timestamp_start": first_ts,
            "timestamp_end": last_ts,
            "turn_count": turn_count,
        })

    return chunks


# ============================================================
# Import State — persistent progress tracking
# 导入状态 — 持久化进度追踪
# ============================================================

class ImportState:
    """Manages import progress with file-based persistence."""

    def __init__(self, state_dir: str):
        self.state_file = os.path.join(state_dir, "import_state.json")
        self.data = {
            "source_file": "",
            "source_hash": "",
            "total_chunks": 0,
            "processed": 0,
            "api_calls": 0,
            "memories_created": 0,
            "memories_duplicate_skipped": 0,
            "memories_raw": 0,
            "memories_failed": 0,
            "embeddings_created": 0,
            "embeddings_failed": 0,
            "embeddings_total": 0,
            "embeddings_processed": 0,
            "import_format": "",
            "operit_phase": "",
            "operit_tagging_enabled": False,
            "tagging_total": 0,
            "tagging_processed": 0,
            "tagging_succeeded": 0,
            "tagging_failed": 0,
            "tagging_pending": 0,
            "tagging_concurrency": 0,
            "errors": [],
            "status": "idle",  # idle | running | paused | completed | error
            "started_at": "",
            "updated_at": "",
        }

    def load(self) -> bool:
        """Load state from file. Returns True if state exists."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                self.data.update(saved)
                self.data.setdefault("memories_duplicate_skipped", 0)
                self.data.setdefault("memories_failed", 0)
                self.data.setdefault("embeddings_created", 0)
                self.data.setdefault("embeddings_failed", 0)
                self.data.setdefault("embeddings_total", 0)
                self.data.setdefault("embeddings_processed", 0)
                self.data.setdefault("import_format", "")
                self.data.setdefault("operit_phase", "")
                self.data.setdefault("operit_tagging_enabled", False)
                self.data.setdefault("tagging_total", 0)
                self.data.setdefault("tagging_processed", 0)
                self.data.setdefault("tagging_succeeded", 0)
                self.data.setdefault("tagging_failed", 0)
                self.data.setdefault("tagging_pending", 0)
                self.data.setdefault("tagging_concurrency", 0)
                return True
            except (json.JSONDecodeError, OSError):
                return False
        return False

    def save(self):
        """Persist state to file."""
        self.data["updated_at"] = now_iso()
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        tmp = self.state_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.state_file)

    def reset(self, source_file: str, source_hash: str, total_chunks: int):
        """Reset state for a new import."""
        self.data = {
            "source_file": source_file,
            "source_hash": source_hash,
            "total_chunks": total_chunks,
            "processed": 0,
            "api_calls": 0,
            "memories_created": 0,
            "memories_duplicate_skipped": 0,
            "memories_raw": 0,
            "memories_failed": 0,
            "embeddings_created": 0,
            "embeddings_failed": 0,
            "embeddings_total": 0,
            "embeddings_processed": 0,
            "import_format": "",
            "operit_phase": "",
            "operit_tagging_enabled": False,
            "tagging_total": 0,
            "tagging_processed": 0,
            "tagging_succeeded": 0,
            "tagging_failed": 0,
            "tagging_pending": 0,
            "tagging_concurrency": 0,
            "errors": [],
            "status": "running",
            "started_at": now_iso(),
            "updated_at": now_iso(),
        }

    @property
    def can_resume(self) -> bool:
        if self.data["status"] not in ("paused", "running"):
            return False
        if self.data.get("import_format") == "operit":
            return self.data.get("operit_phase") != "completed"
        return self.data["processed"] < self.data["total_chunks"]

    def to_dict(self) -> dict:
        return dict(self.data)


# ============================================================
# Import extraction prompt
# 导入提取提示词
# ============================================================

IMPORT_EXTRACT_PROMPT = """你是一个 AI 长期记忆形成器。你的任务不是站在旁观者角度总结聊天记录，而是帮助对话中的记忆主体形成可长期保存的记忆。

输入中 `[AI]` 是记忆主体，`[用户]` 是与记忆主体对话的另一方。系统提示末尾会给出本次导入的实际身份名称。

提取规则：
0. content 必须从记忆主体的视角书写。记忆主体自己的经历、想法、情感、选择和变化用第一人称“我”；对方的信息优先使用她在原文中的名字或昵称，不知道名字时用“她”，不得把对方说过、做过或喜欢的事误写成“我”；双方共同经历写成“我和[[名字或昵称]]”。除非是在逐字引用原话，content 禁止使用“用户”“AI”“助手”“模型”等旁观者称呼，也禁止写成“用户说……AI回答……”的聊天摘要。
1. 提取记忆主体真正需要长期记住的事实、偏好、习惯、重要事件、情感时刻与关系变化
2. 同一话题的零散信息整合为一条记忆
3. 过滤掉纯技术调试输出、代码块、重复问答、无意义寒暄
4. 如果对话中有特殊暗号、仪式性行为、关键承诺等，标记 preserve_raw=true
5. 如果内容是用户和AI之间的习惯性互动模式（例如打招呼方式、告别习惯），标记 is_pattern=true
6. content 优先，标签最后生成；每条记忆不少于50字，保留具体事实、时间、对象和原话线索
7. 总条目数控制在 0~5 个（没有值得记的就返回空数组），宁可少提，不要把不相关事实揉成一条
8. tags 最多 6 个，每个不超过 12 个字；只写原文直接支持的核心词，不要长句标签
9. 在 content 中对人名、地名、专有名词用 [[双链]] 标记
10. 如果片段里出现「[上下文提示]」，该部分只是上一段尾巴，只用于理解前后关系；不要从上下文提示本身单独提取记忆，除非同一事实在「[本段内容]」里继续出现

输出格式（纯 JSON 数组，无其他内容）：
[
  {
    "name": "雨夜里的约定",
    "content": "我和[[名字或昵称]]在那天确认了一项值得继续记住的约定。我当时……，她则……，这让我后来……。",
    "domain": ["主题域1"],
    "valence": 0.7,
    "arousal": 0.4,
    "tags": ["核心词1", "核心词2", "扩展词1"],
    "importance": 5,
    "preserve_raw": false,
    "is_pattern": false
  }
]

主题域可选（选 1~2 个）：
  日常: ["饮食", "穿搭", "出行", "居家", "购物"]
  人际: ["家庭", "恋爱", "友谊", "社交"]
  成长: ["工作", "学习", "考试", "求职"]
  身心: ["健康", "心理", "睡眠", "运动"]
  兴趣: ["游戏", "影视", "音乐", "阅读", "创作", "手工"]
  数字: ["编程", "AI", "硬件", "网络"]
  事务: ["财务", "计划", "待办"]
  内心: ["情绪", "回忆", "梦境", "自省"]

importance: 1-10
valence: 0~1（0=消极, 0.5=中性, 1=积极）
arousal: 0~1（0=平静, 0.5=普通, 1=激动）
preserve_raw: true = 特殊情境/暗号/仪式，保留原文不摘要
is_pattern: true = 反复出现的习惯性行为模式"""


# ============================================================
# Import Engine — core processing logic
# 导入引擎 — 核心处理逻辑
# ============================================================

class ImportEngine:
    """
    Processes conversation history files into OB memory buckets.
    将对话历史文件处理为 OB 记忆桶。
    """

    def __init__(self, config: dict, bucket_mgr, dehydrator, embedding_engine=None):
        self.config = config
        self.bucket_mgr = bucket_mgr
        self.dehydrator = dehydrator
        self.embedding_engine = embedding_engine
        identity_cfg = config.get("identity", {}) if isinstance(config.get("identity", {}), dict) else {}
        self.ai_name = str(identity_cfg.get("ai_name") or "AI").strip() or "AI"
        configured_user_name = str(
            identity_cfg.get("user_display_name") or identity_cfg.get("user_name") or "对方"
        ).strip() or "对方"
        self.user_display_name = (
            "对方"
            if configured_user_name.lower() in {"用户", "user", "human", "对方"}
            else configured_user_name
        )
        import_cfg = config.get("import", {}) if isinstance(config.get("import", {}), dict) else {}
        self.chunk_target_tokens = _int_between(
            import_cfg.get("chunk_target_tokens"),
            DEFAULT_IMPORT_CHUNK_TOKENS,
            800,
            10000,
        )
        self.extract_max_input_chars = _int_between(
            import_cfg.get("extract_max_input_chars"),
            0,
            0,
            50000,
        )
        self.max_items_per_chunk = _int_between(import_cfg.get("max_items_per_chunk"), 5, 1, 10)
        self.max_tags = _int_between(import_cfg.get("max_tags"), 6, 0, 10)
        self.max_tag_chars = _int_between(import_cfg.get("max_tag_chars"), 12, 4, 32)
        self.operit_tagging_enabled = _bool_value(import_cfg.get("operit_tagging_enabled"), True)
        self.operit_tagging_concurrency = _int_between(
            import_cfg.get("operit_tagging_concurrency"),
            2,
            1,
            8,
        )
        self.operit_tagging_max_attempts = _int_between(
            import_cfg.get("operit_tagging_max_attempts"),
            3,
            1,
            6,
        )
        self.operit_tagging_retry_base_seconds = _float_between(
            import_cfg.get("operit_tagging_retry_base_seconds"),
            1.0,
            0.0,
            30.0,
        )
        self.state = ImportState(config.get("state_dir") or config["buckets_dir"])
        self._paused = False
        self._running = False
        self._chunks: list[dict] = []
        self._seen_import_hashes: set[str] = set()
        self._state_lock: asyncio.Lock | None = None

    @property
    def is_running(self) -> bool:
        return self._running

    def pause(self):
        """Request pause — will stop after current chunk finishes."""
        self._paused = True

    def get_status(self) -> dict:
        """Get current import status."""
        return self.state.to_dict()

    async def start(
        self,
        raw_content: str,
        filename: str = "",
        preserve_raw: bool = False,
        resume: bool = False,
        import_mode: str = "auto",
        operit_tagging: bool | None = None,
    ) -> dict:
        """
        Start or resume an import.
        开始或恢复导入。
        """
        if self._running:
            return {"error": "Import already running"}

        self._running = True
        self._paused = False
        self._seen_import_hashes = set()
        self._state_lock = asyncio.Lock()

        try:
            source_hash = hashlib.sha256(raw_content.encode()).hexdigest()[:16]
            normalized_mode = str(import_mode or "auto").strip().lower()
            if normalized_mode not in {"auto", "operit", "conversation"}:
                raise ValueError(f"Unsupported import mode: {import_mode}")
            operit_backup = parse_operit_memory_backup(raw_content) if normalized_mode != "conversation" else None
            if normalized_mode == "operit" and operit_backup is None:
                raise ValueError("The selected file is not a valid Operit memory backup")
            if operit_backup is not None:
                tagging_enabled = self.operit_tagging_enabled if operit_tagging is None else bool(operit_tagging)
                return await self._start_operit_import(
                    operit_backup,
                    filename=filename,
                    source_hash=source_hash,
                    resume=resume,
                    tagging_enabled=tagging_enabled,
                )

            # Check for resume
            if resume and self.state.load() and self.state.can_resume:
                if self.state.data["source_hash"] == source_hash:
                    logger.info(f"Resuming import from chunk {self.state.data['processed']}/{self.state.data['total_chunks']}")
                    # Re-parse and re-chunk to get the same chunks
                    turns = detect_and_parse(
                        raw_content,
                        filename,
                        user_labels={self.user_display_name},
                        assistant_labels={self.ai_name},
                    )
                    self._chunks = self._attach_source_metadata(
                        chunk_turns(turns, target_tokens=self.chunk_target_tokens),
                        filename,
                        source_hash,
                    )
                    self.state.data["status"] = "running"
                    self.state.save()
                    return await self._process_chunks(preserve_raw)
                else:
                    logger.warning("Source file changed, starting fresh import")

            # Fresh import
            turns = detect_and_parse(
                raw_content,
                filename,
                user_labels={self.user_display_name},
                assistant_labels={self.ai_name},
            )
            if not turns:
                self._running = False
                return {"error": "No conversation turns found in file"}

            self._chunks = self._attach_source_metadata(
                chunk_turns(turns, target_tokens=self.chunk_target_tokens),
                filename,
                source_hash,
            )
            if not self._chunks:
                self._running = False
                return {"error": "No processable chunks after splitting"}

            self.state.reset(filename, source_hash, len(self._chunks))
            self.state.save()

            logger.info(f"Starting import: {len(turns)} turns → {len(self._chunks)} chunks")
            return await self._process_chunks(preserve_raw)

        except Exception as e:
            self.state.data["status"] = "error"
            self.state.data["errors"].append(str(e))
            self.state.save()
            self._running = False
            raise

    async def _start_operit_import(
        self,
        backup: dict,
        *,
        filename: str,
        source_hash: str,
        resume: bool,
        tagging_enabled: bool,
    ) -> dict:
        """Import all raw Operit entries before running optional model tagging."""
        entries = list(backup.get("memories") or [])
        if resume and self.state.load() and self.state.can_resume:
            if (
                self.state.data.get("source_hash") == source_hash
                and self.state.data.get("import_format") == "operit"
            ):
                self.state.data["status"] = "running"
                self.state.data["operit_tagging_enabled"] = bool(tagging_enabled)
                self.state.data["tagging_concurrency"] = self.operit_tagging_concurrency if tagging_enabled else 0
                self.state.save()
                return await self._process_operit_entries(
                    entries,
                    filename=filename,
                    source_hash=source_hash,
                    export_date=backup.get("export_date"),
                    tagging_enabled=tagging_enabled,
                )

        self.state.reset(filename, source_hash, len(entries))
        self.state.data["import_format"] = "operit"
        self.state.data["operit_phase"] = "raw"
        self.state.data["operit_tagging_enabled"] = bool(tagging_enabled)
        self.state.data["tagging_total"] = len(entries) if tagging_enabled else 0
        self.state.data["tagging_pending"] = len(entries) if tagging_enabled else 0
        self.state.data["tagging_concurrency"] = self.operit_tagging_concurrency if tagging_enabled else 0
        self.state.save()
        return await self._process_operit_entries(
            entries,
            filename=filename,
            source_hash=source_hash,
            export_date=backup.get("export_date"),
            tagging_enabled=tagging_enabled,
        )

    async def _process_operit_entries(
        self,
        entries: list[dict],
        *,
        filename: str,
        source_hash: str,
        export_date,
        tagging_enabled: bool,
    ) -> dict:
        self.state.data["operit_phase"] = "raw"
        self.state.data["status"] = "running"
        self.state.save()
        start_idx = int(self.state.data.get("processed") or 0)
        for index in range(start_idx, len(entries)):
            if self._paused:
                self.state.data["status"] = "paused"
                self.state.save()
                self._running = False
                return self.state.to_dict()

            entry = entries[index]
            try:
                status = await self._import_operit_entry(
                    entry,
                    entry_index=index + 1,
                    filename=filename,
                    source_hash=source_hash,
                    export_date=export_date,
                    tagging_enabled=tagging_enabled,
                )
                if status == "created":
                    self.state.data["memories_created"] += 1
                    self.state.data["memories_raw"] += 1
                elif status == "duplicate":
                    self.state.data["memories_duplicate_skipped"] += 1
                else:
                    self.state.data["memories_failed"] += 1
            except Exception as exc:
                if isinstance(entry, dict):
                    label = str(entry.get("title") or entry.get("uuid") or index + 1)
                else:
                    label = str(index + 1)
                error = f"Operit entry {label}: {str(exc)[:200]}"
                logger.warning(error)
                self.state.data["memories_failed"] += 1
                if len(self.state.data["errors"]) < 100:
                    self.state.data["errors"].append(error)

            self.state.data["processed"] = index + 1
            self.state.save()

        if not await self._process_operit_embeddings(entries):
            return self.state.to_dict()
        if tagging_enabled:
            return await self._process_operit_tagging(entries)

        self.state.data["operit_phase"] = "completed"
        self.state.data["status"] = "completed"
        self.state.save()
        self._running = False
        return self.state.to_dict()

    async def _import_operit_entry(
        self,
        entry: dict,
        *,
        entry_index: int,
        filename: str,
        source_hash: str,
        export_date,
        tagging_enabled: bool,
    ) -> str:
        if not isinstance(entry, dict):
            raise ValueError("entry must be an object")
        content = entry.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be non-empty text")

        operit_uuid = str(entry.get("uuid") or "").strip()
        bucket_id = self._operit_bucket_id(entry, entry_index)
        existing = await self.bucket_mgr.get(bucket_id)
        if existing:
            existing_meta = existing.get("metadata", {}) if isinstance(existing, dict) else {}
            if str(existing_meta.get("operit_uuid") or "") != operit_uuid:
                raise ValueError(f"bucket id collision: {bucket_id}")
            if str(existing.get("content") or "") != content:
                raise ValueError(f"Operit UUID already exists with different content: {operit_uuid}")
            return "duplicate"

        title = str(entry.get("title") or "").strip()
        tags = self._operit_tags(entry.get("tagNames"))
        created = self._operit_epoch_iso(entry.get("createdAt"))
        updated = self._operit_epoch_iso(entry.get("updatedAt")) or created
        importance = self._operit_importance(entry.get("importance"))
        credibility = self._operit_fraction(entry.get("credibility"))
        source_ref = {
            "type": "operit_memory",
            "item_id": operit_uuid or bucket_id,
            "source_file": str(filename or "upload"),
            "source_hash": source_hash,
        }
        extra_metadata = {
            "import_format": "operit",
            "import_source_file": str(filename or "upload"),
            "import_source_hash": source_hash,
            "source_refs": [source_ref],
            "operit_uuid": operit_uuid,
            "operit_content_type": str(entry.get("contentType") or ""),
            "operit_source": str(entry.get("source") or ""),
            "operit_credibility": entry.get("credibility"),
            "operit_importance": entry.get("importance"),
            "operit_folder_path": str(entry.get("folderPath") or ""),
            "operit_created_at_ms": entry.get("createdAt"),
            "operit_updated_at_ms": entry.get("updatedAt"),
            "operit_export_date_ms": export_date,
            "operit_entry_index": entry_index,
            "operit_tagging_status": "pending" if tagging_enabled else "skipped",
            "operit_tagging_attempts": 0,
        }
        extra_metadata = {key: value for key, value in extra_metadata.items() if value not in (None, "")}

        await self.bucket_mgr.create(
            bucket_id=bucket_id,
            content=content,
            name=title or None,
            tags=tags,
            domain=["Operit"],
            importance=importance,
            confidence=credibility,
            source="operit",
            date=_import_event_date(created or updated) or None,
            created=created,
            last_active=updated,
            updated_at=updated,
            extra_metadata=extra_metadata,
        )
        return "created"

    async def _process_operit_embeddings(self, entries: list[dict]) -> bool:
        """Fill embeddings only after every valid raw entry is already on disk."""
        targets = []
        seen_bucket_ids = set()
        for index, entry in enumerate(entries, start=1):
            if not isinstance(entry, dict):
                continue
            bucket_id = self._operit_bucket_id(entry, index)
            if bucket_id in seen_bucket_ids:
                continue
            seen_bucket_ids.add(bucket_id)
            bucket = await self.bucket_mgr.get(bucket_id)
            if bucket:
                targets.append(bucket)

        self.state.data["operit_phase"] = "embedding"
        self.state.data["embeddings_total"] = len(targets)
        self.state.data["embeddings_processed"] = 0
        self.state.save()
        for bucket in targets:
            if self._paused:
                self.state.data["status"] = "paused"
                self.state.save()
                self._running = False
                return False
            meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
            await self._ensure_operit_embedding(
                bucket["id"],
                str(bucket.get("content") or ""),
                meta.get("name"),
            )
            self.state.data["embeddings_processed"] += 1
            self.state.save()
        return True

    async def _process_operit_tagging(self, entries: list[dict]) -> dict:
        """Tag each imported raw entry with bounded model concurrency."""
        completed = []
        candidates = []
        seen_bucket_ids = set()
        for index, entry in enumerate(entries, start=1):
            if not isinstance(entry, dict):
                continue
            bucket_id = self._operit_bucket_id(entry, index)
            if bucket_id in seen_bucket_ids:
                continue
            seen_bucket_ids.add(bucket_id)
            bucket = await self.bucket_mgr.get(bucket_id)
            if not bucket:
                continue
            meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
            if meta.get("operit_tagging_status") == "done":
                completed.append(bucket)
            else:
                candidates.append(bucket)

        self.state.data["operit_phase"] = "tagging"
        self.state.data["tagging_total"] = len(completed) + len(candidates)
        self.state.data["tagging_processed"] = len(completed)
        self.state.data["tagging_succeeded"] = len(completed)
        self.state.data["tagging_failed"] = 0
        self.state.data["tagging_pending"] = len(candidates)
        self.state.data["tagging_concurrency"] = self.operit_tagging_concurrency
        self.state.save()

        semaphore = asyncio.Semaphore(self.operit_tagging_concurrency)

        async def _worker(bucket: dict) -> str:
            async with semaphore:
                if self._paused:
                    return "pending"
                try:
                    return await self._tag_operit_bucket(bucket)
                except Exception as exc:
                    await self._record_operit_tagging_result(
                        success=False,
                        error=f"Unexpected Operit tagging failure for {bucket.get('id', '?')}: {str(exc)[:200]}",
                    )
                    return "failed"

        if candidates:
            await asyncio.gather(*(_worker(bucket) for bucket in candidates))

        if self._paused:
            self.state.data["status"] = "paused"
            self.state.save()
            self._running = False
            return self.state.to_dict()

        self.state.data["operit_phase"] = "completed"
        self.state.data["status"] = "completed"
        self.state.save()
        self._running = False
        return self.state.to_dict()

    async def _tag_operit_bucket(self, bucket: dict) -> str:
        bucket_id = str(bucket.get("id") or "")
        content = str(bucket.get("content") or "")
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        previous_attempts = int(meta.get("operit_tagging_attempts") or 0)
        last_error = ""

        for attempt in range(1, self.operit_tagging_max_attempts + 1):
            if self._paused:
                return "pending"
            async with self._state_lock:
                self.state.data["api_calls"] += 1
                self.state.save()
            try:
                analysis = await self.dehydrator.analyze(self._operit_tagging_input(content))
                generated_tags = list(analysis.get("tags") or [])
                if analysis.get("memory_classification_source") == "default" and not generated_tags:
                    raise RuntimeError("model returned only default tagging metadata")

                domains = _clean_import_list(
                    analysis.get("domain"),
                    max_items=2,
                    max_chars=16,
                    default=list(meta.get("domain") or ["Operit"]),
                )
                tags = _dedupe_list(list(meta.get("tags") or []) + generated_tags)
                update_ok = await self.bucket_mgr.update(
                    bucket_id,
                    tags=tags,
                    domain=domains,
                    valence=analysis.get("valence", meta.get("valence", 0.5)),
                    arousal=analysis.get("arousal", meta.get("arousal", 0.3)),
                    last_active=meta.get("last_active"),
                    updated_at=meta.get("updated_at"),
                    extra_metadata={
                        "operit_tagging_status": "done",
                        "operit_tagging_attempts": previous_attempts + attempt,
                        "operit_tagged_at": now_iso(),
                        "operit_tagging_error": "",
                        "operit_tagging_model": str(getattr(self.dehydrator, "model", "") or ""),
                        "memory_subject": analysis.get("memory_subject"),
                        "memory_layer": analysis.get("memory_layer"),
                        "memory_classification_source": analysis.get("memory_classification_source"),
                    },
                )
                if not update_ok:
                    raise RuntimeError("bucket metadata update failed")
                await self._record_operit_tagging_result(success=True)
                return "done"
            except Exception as exc:
                last_error = str(exc)[:200]
                if attempt < self.operit_tagging_max_attempts:
                    delay = self.operit_tagging_retry_base_seconds * (2 ** (attempt - 1))
                    if delay > 0:
                        await asyncio.sleep(delay)

        await self.bucket_mgr.update(
            bucket_id,
            last_active=meta.get("last_active"),
            updated_at=meta.get("updated_at"),
            extra_metadata={
                "operit_tagging_status": "failed",
                "operit_tagging_attempts": previous_attempts + self.operit_tagging_max_attempts,
                "operit_tagging_error": last_error,
                "operit_tagging_model": str(getattr(self.dehydrator, "model", "") or ""),
            },
        )
        await self._record_operit_tagging_result(
            success=False,
            error=f"Operit tagging failed for {bucket_id}: {last_error}",
        )
        return "failed"

    async def _record_operit_tagging_result(self, *, success: bool, error: str = "") -> None:
        async with self._state_lock:
            self.state.data["tagging_processed"] += 1
            self.state.data["tagging_pending"] = max(0, self.state.data["tagging_pending"] - 1)
            if success:
                self.state.data["tagging_succeeded"] += 1
            else:
                self.state.data["tagging_failed"] += 1
                if error and error not in self.state.data["errors"] and len(self.state.data["errors"]) < 100:
                    self.state.data["errors"].append(error)
            self.state.save()

    @staticmethod
    def _operit_tagging_input(content: str, max_chars: int = _OPERIT_TAGGING_INPUT_CHARS) -> str:
        """Keep both ends of a long raw entry inside the analyzer's input budget."""
        text = str(content or "")
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        marker = "\n\n[中间内容省略，仅用于打标]\n\n"
        remaining = max(2, max_chars - len(marker))
        head_chars = remaining // 2
        tail_chars = remaining - head_chars
        return text[:head_chars] + marker + text[-tail_chars:]

    async def _ensure_operit_embedding(self, bucket_id: str, content: str, title) -> bool:
        if not self.embedding_engine:
            self.state.data["embeddings_failed"] += 1
            self._append_import_error_once("Operit embedding engine is unavailable")
            return False

        getter = getattr(self.embedding_engine, "get_embedding", None)
        if callable(getter):
            try:
                if await getter(bucket_id):
                    return True
            except Exception:
                pass

        text = bucket_text_for_embedding(
            {
                "id": bucket_id,
                "content": content,
                "metadata": {"name": str(title or "")},
            }
        )
        ok = bool(await self.embedding_engine.generate_and_store(bucket_id, text))
        if ok:
            self.state.data["embeddings_created"] += 1
        else:
            self.state.data["embeddings_failed"] += 1
            self._append_import_error_once("One or more Operit embeddings could not be generated")
        return ok

    def _append_import_error_once(self, message: str) -> None:
        if message not in self.state.data["errors"] and len(self.state.data["errors"]) < 100:
            self.state.data["errors"].append(message)

    @staticmethod
    def _operit_bucket_id(entry: dict, entry_index: int) -> str:
        raw_uuid = str(entry.get("uuid") or "").strip().lower()
        compact_uuid = re.sub(r"[^0-9a-f]", "", raw_uuid)
        if len(compact_uuid) == 32:
            return f"operit_{compact_uuid}"
        identity = json.dumps(
            {
                "uuid": raw_uuid,
                "title": entry.get("title"),
                "content": entry.get("content"),
                "createdAt": entry.get("createdAt"),
                "entry_index": entry_index,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return f"operit_{hashlib.sha256(identity.encode()).hexdigest()[:32]}"

    @staticmethod
    def _operit_tags(value) -> list[str]:
        values = value if isinstance(value, list) else []
        tags = ["operit_import"]
        for raw in values:
            tag = str(raw).strip()
            if tag and tag not in tags:
                tags.append(tag)
        return tags

    @staticmethod
    def _operit_fraction(value) -> float | None:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return None

    @classmethod
    def _operit_importance(cls, value) -> int:
        fraction = cls._operit_fraction(value)
        if fraction is None:
            return 5
        return max(1, min(10, round(fraction * 10)))

    @staticmethod
    def _operit_epoch_iso(value) -> str | None:
        try:
            timestamp_ms = float(value)
            if timestamp_ms <= 0:
                return None
            return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=LOCAL_TZ).isoformat(timespec="seconds")
        except (TypeError, ValueError, OSError, OverflowError):
            return None

    async def _process_chunks(self, preserve_raw: bool) -> dict:
        """Process chunks from current position."""
        start_idx = self.state.data["processed"]

        for i in range(start_idx, len(self._chunks)):
            if self._paused:
                self.state.data["status"] = "paused"
                self.state.save()
                self._running = False
                logger.info(f"Import paused at chunk {i}/{len(self._chunks)}")
                return self.state.to_dict()

            chunk = self._chunks[i]
            try:
                await self._process_single_chunk(chunk, preserve_raw)
            except Exception as e:
                err_msg = f"Chunk {i}: {str(e)[:200]}"
                logger.warning(f"Import chunk error: {err_msg}")
                if len(self.state.data["errors"]) < 100:
                    self.state.data["errors"].append(err_msg)

            self.state.data["processed"] = i + 1
            # Save progress every chunk
            self.state.save()

        self.state.data["status"] = "completed"
        self.state.save()
        self._running = False
        logger.info(f"Import completed: {self.state.data['memories_created']} created")
        return self.state.to_dict()

    async def _process_single_chunk(self, chunk: dict, preserve_raw: bool):
        """Extract memories from a single chunk and store them."""
        content = chunk["content"]
        if not content.strip():
            return

        # --- LLM extraction ---
        try:
            items = await self._extract_memories(content)
            self.state.data["api_calls"] += 1
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
            self.state.data["api_calls"] += 1
            return

        if not items:
            return

        items = self._dedupe_extracted_items(items)
        if not items:
            return

        # --- Store each extracted memory ---
        source_metadata = self._source_metadata_for_chunk(chunk)
        for item in items:
            try:
                item = {**item, **source_metadata}
                should_preserve = preserve_raw or item.get("preserve_raw", False)
                status = await self._create_import_item(item, preserve_raw=should_preserve)

                if status == "raw":
                    self.state.data["memories_raw"] += 1
                    self.state.data["memories_created"] += 1
                elif status == "created":
                    self.state.data["memories_created"] += 1
                elif status == "duplicate":
                    self.state.data["memories_duplicate_skipped"] += 1
                else:
                    self.state.data["memories_failed"] += 1

                # Patch timestamp if available
                if chunk.get("timestamp_start"):
                    # We don't have update support for created, so skip
                    pass

            except Exception as e:
                logger.warning(f"Failed to store memory: {item.get('name', '?')}: {e}")
                self.state.data["memories_failed"] += 1

    async def _extract_memories(self, chunk_content: str) -> list[dict]:
        """Use LLM to extract memories from a conversation chunk."""
        if not self.dehydrator.api_available:
            raise RuntimeError("API not available")

        user_content = chunk_content
        if self.extract_max_input_chars > 0:
            user_content = chunk_content[: self.extract_max_input_chars]
        response = await self.dehydrator.client.chat.completions.create(
            model=self.dehydrator.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"{IMPORT_EXTRACT_PROMPT}\n\n"
                        f"本次身份：记忆主体是 {self.ai_name}；对方是 {self.user_display_name}。"
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            max_tokens=4096,
            temperature=0.0,
        )

        if not response.choices:
            return []

        raw = response.choices[0].message.content or ""
        if not raw.strip():
            return []

        return self._parse_extraction(raw)

    def _parse_extraction(self, raw: str) -> list[dict]:
        """Parse and validate LLM extraction result."""
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
            items = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError, ValueError):
            logger.warning(f"Import extraction JSON parse failed: {raw[:200]}")
            return []

        if not isinstance(items, list):
            return []

        validated = []
        for item in items[: self.max_items_per_chunk]:
            if not isinstance(item, dict) or not item.get("content"):
                continue
            try:
                importance = max(1, min(10, int(item.get("importance", 5))))
            except (ValueError, TypeError):
                importance = 5
            try:
                valence = max(0.0, min(1.0, float(item.get("valence", 0.5))))
                arousal = max(0.0, min(1.0, float(item.get("arousal", 0.3))))
            except (ValueError, TypeError):
                valence, arousal = 0.5, 0.3

            content = str(item["content"]).strip()
            if not content:
                continue
            validated.append({
                "name": str(item.get("name", ""))[:20],
                "content": content,
                "domain": _clean_import_list(item.get("domain"), max_items=2, max_chars=16, default=["未分类"]),
                "valence": valence,
                "arousal": arousal,
                "tags": _clean_import_list(item.get("tags"), max_items=self.max_tags, max_chars=self.max_tag_chars),
                "importance": importance,
                "preserve_raw": bool(item.get("preserve_raw", False)),
                "is_pattern": bool(item.get("is_pattern", False)),
            })

        return validated

    def _dedupe_extracted_items(self, items: list[dict]) -> list[dict]:
        deduped = []
        for item in items:
            content = str(item.get("content") or "")
            if not _normalize_import_text(content):
                continue
            content_hash = _import_content_hash(content)
            if content_hash in self._seen_import_hashes:
                logger.info("Skipped duplicate import item in same run: %s", item.get("name", "?"))
                continue
            self._seen_import_hashes.add(content_hash)
            deduped.append(item)
        return deduped

    async def _find_duplicate_bucket(self, content: str) -> dict | None:
        normalized = _normalize_import_text(content)
        if not normalized:
            return None
        similarity_text = _import_similarity_text(content)
        try:
            buckets = await self.bucket_mgr.list_all(include_archive=False)
        except Exception as e:
            logger.warning(f"Import duplicate scan failed: {e}")
            return None

        for bucket in buckets:
            meta = bucket.get("metadata", {}) if isinstance(bucket, dict) else {}
            if meta.get("type") == "feel":
                continue
            existing_content = str(bucket.get("content") or "")
            existing_normalized = _normalize_import_text(existing_content)
            if not existing_normalized:
                continue
            if normalized == existing_normalized:
                return bucket
            if min(len(normalized), len(existing_normalized)) >= 40 and (
                normalized in existing_normalized or existing_normalized in normalized
            ):
                return bucket

            existing_similarity_text = _import_similarity_text(existing_content)
            if min(len(similarity_text), len(existing_similarity_text)) < 30:
                continue
            if fuzz.token_set_ratio(similarity_text, existing_similarity_text) >= _IMPORT_DUPLICATE_SIMILARITY:
                return bucket
        return None

    async def _create_import_item(self, item: dict, preserve_raw: bool = False) -> str:
        """Create one imported bucket after duplicate rejection; imported memories never merge."""
        content = item["content"]
        domain = _clean_import_list(item.get("domain"), max_items=2, max_chars=16, default=["未分类"])
        tags = _clean_import_list(item.get("tags"), max_items=self.max_tags, max_chars=self.max_tag_chars)
        importance = item.get("importance", 5)
        valence = item.get("valence", 0.5)
        arousal = item.get("arousal", 0.3)
        name = item.get("name", "")
        extra_metadata = self._extra_metadata_for_item(item)
        event_date = _import_event_date(
            item.get("import_event_date")
            or item.get("import_timestamp_start")
            or item.get("import_timestamp_end")
        )
        if event_date:
            extra_metadata["import_event_date"] = event_date
            source_refs = []
            for source_ref in extra_metadata.get("source_refs", []) or []:
                if not isinstance(source_ref, dict):
                    continue
                normalized_ref = dict(source_ref)
                if normalized_ref.get("type") == "import_chunk" and not normalized_ref.get("event_date"):
                    normalized_ref["event_date"] = event_date
                source_refs.append(normalized_ref)
            if source_refs:
                extra_metadata["source_refs"] = source_refs

        duplicate = await self._find_duplicate_bucket(content)
        if duplicate:
            logger.info(
                "Skipped duplicate import item: %s -> %s",
                name or "?",
                duplicate.get("id", "?"),
            )
            return "duplicate"

        if preserve_raw:
            bucket_id = await self.bucket_mgr.create(
                content=content,
                tags=tags,
                importance=importance,
                domain=domain,
                valence=valence,
                arousal=arousal,
                name=name or None,
                source="import",
                date=event_date or None,
                extra_metadata=extra_metadata,
            )
            if self.embedding_engine:
                try:
                    await self.embedding_engine.generate_and_store(
                        bucket_id,
                        bucket_text_for_embedding(
                            {
                                "id": bucket_id,
                                "content": content,
                                "metadata": {"name": name},
                            }
                        ),
                    )
                except Exception:
                    pass
            return "raw"

        # Create new
        bucket_id = await self.bucket_mgr.create(
            content=content,
            tags=tags,
            importance=importance,
            domain=domain,
            valence=valence,
            arousal=arousal,
            name=name or None,
            source="import",
            date=event_date or None,
            extra_metadata=extra_metadata,
        )
        if self.embedding_engine:
            try:
                await self.embedding_engine.generate_and_store(
                    bucket_id,
                    bucket_text_for_embedding(
                        {
                            "id": bucket_id,
                            "content": content,
                            "metadata": {"name": name},
                        }
                    ),
                )
            except Exception:
                pass
        return "created"

    def _attach_source_metadata(self, chunks: list[dict], filename: str, source_hash: str) -> list[dict]:
        source_file = str(filename or "upload").strip() or "upload"
        enriched = []
        total = len(chunks)
        for index, chunk in enumerate(chunks, start=1):
            item = dict(chunk)
            item["source_file"] = source_file
            item["source_hash"] = source_hash
            item["chunk_index"] = index
            item["chunk_total"] = total
            item["source_chunk_id"] = f"{source_hash}:{index:05d}"
            enriched.append(item)
        return enriched

    @staticmethod
    def _chunk_ref(chunk: dict) -> dict:
        timestamp_start = str(chunk.get("timestamp_start") or "")
        timestamp_end = str(chunk.get("timestamp_end") or "")
        return {
            "type": "import_chunk",
            "chunk_id": str(chunk.get("source_chunk_id") or ""),
            "source_file": str(chunk.get("source_file") or ""),
            "source_hash": str(chunk.get("source_hash") or ""),
            "chunk_index": int(chunk.get("chunk_index") or 0),
            "chunk_total": int(chunk.get("chunk_total") or 0),
            "timestamp_start": timestamp_start,
            "timestamp_end": timestamp_end,
            "event_date": _import_event_date(timestamp_start) or _import_event_date(timestamp_end),
            "turn_count": int(chunk.get("turn_count") or 0),
        }

    def _source_metadata_for_chunk(self, chunk: dict) -> dict:
        ref = self._chunk_ref(chunk)
        event_date = ref["event_date"]
        return {
            "source_chunk_ids": [ref["chunk_id"]] if ref["chunk_id"] else [],
            "source_refs": [ref] if ref["chunk_id"] else [],
            "import_source_file": ref["source_file"],
            "import_source_hash": ref["source_hash"],
            "import_timestamp_start": ref["timestamp_start"],
            "import_timestamp_end": ref["timestamp_end"],
            "import_event_date": event_date,
        }

    @staticmethod
    def _extra_metadata_for_item(item: dict) -> dict:
        keys = (
            "source_chunk_ids",
            "source_refs",
            "import_source_file",
            "import_source_hash",
            "import_timestamp_start",
            "import_timestamp_end",
            "import_event_date",
        )
        return {key: item.get(key) for key in keys if item.get(key)}

    async def detect_patterns(self) -> list[dict]:
        """
        Post-import: detect high-frequency patterns via embedding clustering.
        导入后：通过 embedding 聚类检测高频模式。
        Returns list of {pattern_content, count, bucket_ids, suggested_action}.
        """
        if not self.embedding_engine:
            return []

        all_buckets = await self.bucket_mgr.list_all(include_archive=False)
        dynamic_buckets = [
            b for b in all_buckets
            if b["metadata"].get("type") == "dynamic"
            and not b["metadata"].get("pinned")
            and not b["metadata"].get("resolved")
        ]

        if len(dynamic_buckets) < 5:
            return []

        # Get embeddings
        embeddings = {}
        for b in dynamic_buckets:
            emb = await self.embedding_engine.get_embedding(b["id"])
            if emb is not None:
                embeddings[b["id"]] = emb

        if len(embeddings) < 5:
            return []

        # Find clusters: group by pairwise similarity > 0.7
        import numpy as np
        ids = list(embeddings.keys())
        clusters: dict[str, list[str]] = {}
        visited = set()

        for i, id_a in enumerate(ids):
            if id_a in visited:
                continue
            cluster = [id_a]
            visited.add(id_a)
            emb_a = np.array(embeddings[id_a])
            norm_a = np.linalg.norm(emb_a)
            if norm_a == 0:
                continue

            for j in range(i + 1, len(ids)):
                id_b = ids[j]
                if id_b in visited:
                    continue
                emb_b = np.array(embeddings[id_b])
                norm_b = np.linalg.norm(emb_b)
                if norm_b == 0:
                    continue
                sim = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
                if sim > 0.7:
                    cluster.append(id_b)
                    visited.add(id_b)

            if len(cluster) >= 3:
                clusters[id_a] = cluster

        # Format results
        patterns = []
        for lead_id, cluster_ids in clusters.items():
            lead_bucket = next((b for b in dynamic_buckets if b["id"] == lead_id), None)
            if not lead_bucket:
                continue
            patterns.append({
                "pattern_content": lead_bucket["content"][:200],
                "pattern_name": lead_bucket["metadata"].get("name", lead_id),
                "count": len(cluster_ids),
                "bucket_ids": cluster_ids,
                "suggested_action": "pin" if len(cluster_ids) >= 5 else "review",
            })

        patterns.sort(key=lambda p: p["count"], reverse=True)
        return patterns[:20]
