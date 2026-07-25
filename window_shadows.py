from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any

from identity import identity_names


WINDOW_SHADOW_VERSION = "window-shadow-v4"
WINDOW_SHADOW_REJECTED_DRAFT_VERSION = "window-shadow-rejected-draft-v1"


_HEADING_RE = re.compile(r"(?m)^(#{2,6})\s+(.+?)\s*$")
_SECTION_KEYS = (
    "self",
    "voice",
    "relationship",
    "interaction",
    "recent_events",
    "care_items",
    "handoff",
    "moments",
)
_BARE_CONTINUE_QUERY_RE = re.compile(
    r"^(?:(?:好|嗯|唔|行|可以|可|那|那么|好呀|好耶|嗯嗯))?"
    r"(?:(?:我们)?(?:继续|接着(?:来|说)?|然后呢))"
    r"(?:吧|呀|啦|哦|噢)?$",
    re.IGNORECASE,
)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _normalize_heading(value: Any) -> str:
    return re.sub(
        r"[\s\d一二三四五六七八九十、.．:：;；!?！？()（）\[\]【】'\"`_-]+",
        "",
        str(value or "").strip().lower(),
    )


def _section_key(heading: str) -> str:
    key = _normalize_heading(heading)
    if key in {"窗影", "这篇窗影", "windowshadow", "shadow"}:
        return "self"
    if (
        "这一窗之后什么留在了我身上" in key
        or "这窗之后什么留在了我身上" in key
        or "什么留在了我身上" in key
        or "这一窗留给我的" in key
        or "这窗留给我的" in key
        or "这一窗留下的" in key
        or "这窗留下的" in key
        or "这一窗留下了什么" in key
        or "这窗留下了什么" in key
    ):
        return "self"
    if "我是谁" in key or key in {"self", "identity", "myself"}:
        return "self"
    if (
        "思考与声音" in key
        or "思考和声音" in key
        or "我在想什么" in key
        or "我是怎么想的" in key
        or key in {"我的想法", "我的思绪"}
    ):
        return "voice"
    if "怎么思考" in key or "怎么说话" in key or "语言的指纹" in key:
        return "voice"
    if (
        "最近发生的事" in key
        or "最近发生了什么" in key
        or "最近事件" in key
        or key in {"recentevents", "recentwindowevents"}
    ):
        return "recent_events"
    if "仍在发生" in key or "仍悬着" in key or "值得带走" in key:
        return "care_items"
    if "线头" in key and ("亮着" in key or "还在" in key or "未完" in key):
        return "care_items"
    if key in {"未完线头", "还亮着的线头", "openthreads", "unfinishedthreads"}:
        return "care_items"
    if "还需要关心的事" in key or "需要关心的事" in key:
        return "care_items"
    if key in {"careitems", "thingstocareabout"}:
        return "care_items"
    if "怎么相处" in key or "相处方式" in key:
        return "interaction"
    if "给下个窗口的我" in key or "交给下个窗口" in key or "handoffnote" in key:
        return "handoff"
    if "想留下的记忆" in key or "重要时刻" in key or "难忘时刻" in key or "重要场景" in key:
        return "moments"
    if (
        key in {"关于你关于我们", "关于我们", "我眼里的你和我们"}
        or (key.startswith("关于") and "我们" in key)
    ):
        return "relationship"
    if "我对" in key and "我们" in key and ("新懂" in key or "理解" in key):
        return "relationship"
    if "我们之间是什么" in key or ("是谁" in key and "我们" in key):
        return "relationship"
    return ""


def parse_window_shadow(content: str) -> dict[str, str]:
    """Split a complete window self-narrative without rewriting its text."""
    text = str(content or "").strip()
    matches = list(_HEADING_RE.finditer(text))
    sections = {key: "" for key in _SECTION_KEYS}
    top_rows: list[tuple[re.Match[str], str]] = []
    for match in matches:
        key = _section_key(match.group(2))
        if key:
            top_rows.append((match, key))
    for index, (match, key) in enumerate(top_rows):
        end = top_rows[index + 1][0].start() if index + 1 < len(top_rows) else len(text)
        sections[key] = text[match.end():end].strip()
    return sections


def window_shadow_outside_sections(
    content: str,
    mutable_sections: set[str] | frozenset[str],
) -> str:
    """Redact selected section bodies so retries cannot rewrite the rest."""
    text = str(content or "")
    allowed = {
        str(value or "").strip()
        for value in mutable_sections
        if str(value or "").strip() in _SECTION_KEYS
    }
    if not allowed:
        return text
    matches = list(_HEADING_RE.finditer(text))
    top_rows: list[tuple[re.Match[str], str]] = []
    for match in matches:
        key = _section_key(match.group(2))
        if key:
            top_rows.append((match, key))
    output: list[str] = []
    cursor = 0
    for index, (match, key) in enumerate(top_rows):
        end = top_rows[index + 1][0].start() if index + 1 < len(top_rows) else len(text)
        if key not in allowed:
            continue
        output.append(text[cursor:match.end()])
        output.append(f"\n<rejected-draft-mutable:{key}>\n")
        cursor = end
    output.append(text[cursor:])
    return "".join(output)


def replace_window_shadow_sections(
    content: str,
    replacements: dict[str, str],
) -> tuple[str, list[str]]:
    """Replace only recognized section bodies while preserving every heading and byte outside them."""
    text = str(content or "")
    requested = {
        str(key or "").strip(): str(value or "")
        for key, value in (replacements or {}).items()
        if str(key or "").strip() in _SECTION_KEYS
    }
    if not requested:
        return text, []
    matches = list(_HEADING_RE.finditer(text))
    top_rows: list[tuple[re.Match[str], str]] = []
    for match in matches:
        key = _section_key(match.group(2))
        if key:
            top_rows.append((match, key))
    seen: set[str] = set()
    duplicate: set[str] = set()
    for _, key in top_rows:
        if key in seen:
            duplicate.add(key)
        seen.add(key)
    unavailable = sorted((set(requested) - seen) | (set(requested) & duplicate))
    if unavailable:
        return text, unavailable
    output: list[str] = []
    cursor = 0
    for index, (match, key) in enumerate(top_rows):
        if key not in requested:
            continue
        end = top_rows[index + 1][0].start() if index + 1 < len(top_rows) else len(text)
        prefix = text[cursor:match.end()]
        output.append(prefix)
        separator = "" if prefix.endswith(("\n", "\r")) else "\n"
        replacement = requested[key].strip()
        output.append(f"{separator}{replacement}\n\n" if replacement else separator)
        cursor = end
    output.append(text[cursor:])
    return "".join(output), []


def window_shadow_section_char_count(content: str) -> int:
    return len(re.sub(r"\s+", "", str(content or "").strip()))


def handoff_note_char_count(content: str) -> int:
    """Compatibility alias for old callers; no length contract remains."""
    return window_shadow_section_char_count(content)


def is_bare_window_continue_query(content: str) -> bool:
    """Match a low-information first-turn continuation, never `继续做某件事`."""
    compact = re.sub(
        r"[\s!！?？。.,，、:：;；~～…_\-]+",
        "",
        str(content or "").strip(),
    )
    return bool(compact and len(compact) <= 12 and _BARE_CONTINUE_QUERY_RE.fullmatch(compact))


def validate_window_shadow(
    content: str,
) -> tuple[dict[str, str], list[str]]:
    sections = parse_window_shadow(content)
    errors = []
    if not any(
        sections.get(key)
        for key in (
            "self",
            "voice",
            "relationship",
            "interaction",
            "recent_events",
            "care_items",
        )
    ):
        errors.append("missing_window_delta")
    if sections.get("self") and "我" not in sections["self"]:
        errors.append("self_section_needs_first_person")
    if sections.get("voice") and "我" not in sections["voice"]:
        errors.append("voice_section_needs_first_person")
    return sections, errors


def _scene_heading(match: re.Match[str], *, allow_legacy_moment: bool = False) -> bool:
    key = _normalize_heading(match.group(2))
    if (
        key == "scene"
        or key.startswith("scene")
        or key == "场景"
        or key.startswith("场景")
    ):
        return True
    return bool(
        allow_legacy_moment
        and (
            key == "moment"
            or key.startswith("moment")
            or key == "时刻"
            or key.startswith("时刻")
        )
    )


def _moment_title(heading: str, block: str, index: int) -> str:
    raw_heading = str(heading or "").strip()
    heading_title = re.sub(r"^(?:scene|场景|moment|时刻)\s*[:：|｜-]?\s*", "", raw_heading, flags=re.IGNORECASE).strip()
    if heading_title:
        return heading_title[:48]
    for line in block.splitlines():
        clean = line.strip().lstrip("#").strip()
        if not clean:
            continue
        title_match = re.match(r"^(?:标题|名字|名称)\s*[:：]\s*(.+)$", clean)
        if title_match:
            return title_match.group(1).strip()[:48]
        if clean.lower() in {"original", "reflection", "assistant reflection"}:
            continue
        if len(clean) <= 36 and not re.search(r"[。！？!?]$", clean):
            return clean[:48]
        break
    return f"窗影时刻{index}"


def _scene_heading_metadata(heading: str) -> tuple[str, list[str], list[str]]:
    """Read an authored title and cues from one canonical Scene marker."""
    match = re.match(
        r"^(?:scene|场景)\s*[|｜]\s*(.+)$",
        str(heading or "").strip(),
        flags=re.IGNORECASE,
    )
    if not match:
        return "", [], [
            "必须写成 `### scene | 标题 | cue：…`",
        ]
    title = ""
    cues = []
    seen = set()
    errors = []
    for position, raw in enumerate(re.split(r"[|｜]+", match.group(1))):
        token = re.sub(r"\s+", " ", raw).strip(
            " \t\r\n-—*•"
        )
        title_match = re.match(
            r"^(?:title|标题|名字|名称)\s*[:：=]\s*(.*)$",
            token,
            flags=re.IGNORECASE,
        )
        if title_match:
            value = title_match.group(1).strip()
            if title:
                errors.append("只能写一个标题")
            elif not value:
                errors.append("标题不能为空")
            elif len(value) > 48:
                errors.append("标题不能超过 48 个字符")
            else:
                title = value
            continue
        cue_match = re.match(
            r"^cue\s*[:：=]\s*(.*)$",
            token,
            flags=re.IGNORECASE,
        )
        if cue_match:
            cue = cue_match.group(1).strip(
                " \t\r\n-—*•、，,。.!！?？:：;；\"'“”‘’"
            )
            key = re.sub(r"[\s\W_]+", "", cue.lower())
            if len(key) < 2 or key in seen:
                continue
            seen.add(key)
            cues.append(cue[:80].rstrip())
            if len(cues) >= 8:
                break
            continue
        if position == 0 and not title:
            if not token:
                errors.append("标题不能为空")
            elif len(token) > 48:
                errors.append("标题不能超过 48 个字符")
            else:
                title = token
            continue
        if token:
            errors.append(
                f"标题后的 `{token}` 必须写成 `cue：…`"
            )
    if not title:
        errors.append("缺少当前作者写的标题")
    if not cues:
        errors.append("缺少至少一个当前作者写的 `cue：…`")
    return title, cues, errors


def extract_window_shadow_scenes(
    content: str,
    *,
    allow_legacy_moment: bool = False,
) -> list[dict[str, Any]]:
    """Copy explicit Scene blocks from the optional scene layer; never rewrite them."""
    sections = parse_window_shadow(content)
    text = sections.get("moments", "")
    if not text:
        return []
    matches = [
        match
        for match in _HEADING_RE.finditer(text)
        if _scene_heading(match, allow_legacy_moment=allow_legacy_moment)
    ]
    moments = []
    for index, match in enumerate(matches, start=1):
        end = matches[index].start() if index < len(matches) else len(text)
        body = text[match.end():end].strip()
        if not body:
            continue
        source_text = text[match.start():end].strip()
        heading_key = _normalize_heading(match.group(2))
        legacy_moment = allow_legacy_moment and (
            heading_key == "moment"
            or heading_key.startswith("moment")
            or heading_key == "时刻"
            or heading_key.startswith("时刻")
        )
        if legacy_moment:
            title = _moment_title(match.group(2), body, index)
            scene_cues = []
            marker_errors = []
        else:
            title, scene_cues, marker_errors = _scene_heading_metadata(
                match.group(2)
            )
        moments.append(
            {
                "title": title,
                # The heading is an extraction marker inside the Shadow, not
                # part of the standalone Scene object's authored prose.
                # The legacy grow reader keeps its old payload contract; only
                # the canonical Scene extractor returns the plain body.
                "content": source_text if legacy_moment else body,
                "source_text": source_text,
                "cues": scene_cues,
                "marker_errors": marker_errors,
            }
        )
    return moments


def extract_window_shadow_moments(content: str) -> list[dict[str, str]]:
    """Legacy grow reader: old `### moment` remains readable, never newly authored."""
    return extract_window_shadow_scenes(content, allow_legacy_moment=True)


def project_window_shadow_handoff(
    sections: dict[str, str] | None,
    *,
    user_display_name: str = "用户",
) -> dict[str, str]:
    """Project authored handoff layers without asking another model to summarize them."""
    values = sections if isinstance(sections, dict) else {}

    def render(rows: tuple[tuple[str, str], ...]) -> str:
        parts = []
        for key, title in rows:
            value = str(values.get(key) or "").strip()
            if value:
                parts.append(f"## {title}\n{value}")
        return "\n\n".join(parts).strip()

    return {
        "recent_events": str(values.get("recent_events") or "").strip(),
        "care_items": str(values.get("care_items") or "").strip(),
        # Old Shadows keep their authored note readable for explicit handoff.
        # New close_window calls no longer require or advertise this section.
        "handoff_note": str(values.get("handoff") or "").strip(),
        "flowing_self": render(
            (
                ("self", "这一窗之后，什么留在了我身上"),
                ("voice", "我的思考与声音哪里变得更具体"),
            )
        ),
        "recent_relationship": render(
            (
                ("relationship", f"我对{str(user_display_name or '用户').strip() or '用户'}和我们新懂了什么"),
                ("interaction", "什么仍在发生、仍悬着或值得带走"),
            )
        ),
    }


class WindowShadowRejectedDraftStore:
    """Non-canonical rejected close_window drafts keyed only for explicit retry."""

    def __init__(self, config: dict):
        state_dir = config.get("state_dir") or os.path.join(
            os.path.dirname(os.path.abspath(config.get("buckets_dir", "buckets"))),
            "state",
        )
        os.makedirs(state_dir, exist_ok=True)
        self.db_path = os.path.join(state_dir, "window_shadow_rejected_drafts.sqlite")
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rejected_window_shadow_drafts (
                idempotency_key TEXT PRIMARY KEY,
                version TEXT NOT NULL,
                status TEXT NOT NULL,
                reason TEXT NOT NULL DEFAULT '',
                error TEXT NOT NULL DEFAULT '',
                source_hash TEXT NOT NULL,
                shadow TEXT NOT NULL,
                request_json TEXT NOT NULL DEFAULT '{}',
                validation_json TEXT NOT NULL DEFAULT '{}',
                attempt_count INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rejected_shadow_drafts_updated "
            "ON rejected_window_shadow_drafts(updated_at DESC)"
        )
        conn.commit()
        conn.close()

    @staticmethod
    def source_hash(shadow: str) -> str:
        return hashlib.sha256(str(shadow or "").encode("utf-8")).hexdigest()

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        item = dict(row)
        for key, output_key in (
            ("request_json", "request"),
            ("validation_json", "validation"),
        ):
            try:
                parsed = json.loads(item.pop(key) or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                parsed = {}
            item[output_key] = parsed if isinstance(parsed, dict) else {}
        item["canonical"] = False
        item["ordinary_recall"] = False
        item["handoff_visible"] = False
        return item

    def save(
        self,
        *,
        idempotency_key: str,
        shadow: str,
        status: str,
        reason: str,
        error: str,
        request: dict | None = None,
        validation: dict | None = None,
    ) -> dict:
        key = str(idempotency_key or "").strip()
        if not key:
            raise ValueError("idempotency_key is required for rejected Shadow drafts")
        text = str(shadow or "")
        now = _now_utc()
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO rejected_window_shadow_drafts (
                idempotency_key, version, status, reason, error, source_hash,
                shadow, request_json, validation_json, attempt_count,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            ON CONFLICT(idempotency_key) DO UPDATE SET
                version = excluded.version,
                status = excluded.status,
                reason = excluded.reason,
                error = excluded.error,
                source_hash = excluded.source_hash,
                shadow = excluded.shadow,
                request_json = excluded.request_json,
                validation_json = excluded.validation_json,
                attempt_count = rejected_window_shadow_drafts.attempt_count + 1,
                updated_at = excluded.updated_at
            """,
            (
                key,
                WINDOW_SHADOW_REJECTED_DRAFT_VERSION,
                str(status or "invalid").strip() or "invalid",
                str(reason or "").strip(),
                str(error or "").strip(),
                self.source_hash(text),
                text,
                json.dumps(request if isinstance(request, dict) else {}, ensure_ascii=False),
                json.dumps(validation if isinstance(validation, dict) else {}, ensure_ascii=False),
                now,
                now,
            ),
        )
        conn.commit()
        conn.close()
        return self.get(key) or {}

    def get(self, idempotency_key: str) -> dict | None:
        key = str(idempotency_key or "").strip()
        if not key:
            return None
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM rejected_window_shadow_drafts WHERE idempotency_key = ?",
            (key,),
        ).fetchone()
        conn.close()
        return self._row(row)

    def delete(self, idempotency_key: str) -> bool:
        key = str(idempotency_key or "").strip()
        if not key:
            return False
        conn = self._connect()
        cursor = conn.execute(
            "DELETE FROM rejected_window_shadow_drafts WHERE idempotency_key = ?",
            (key,),
        )
        conn.commit()
        deleted = cursor.rowcount > 0
        conn.close()
        return deleted

    def stats(self) -> dict:
        conn = self._connect()
        row = conn.execute(
            "SELECT COUNT(*) AS count, MAX(updated_at) AS latest_updated_at "
            "FROM rejected_window_shadow_drafts"
        ).fetchone()
        conn.close()
        return {
            "count": int(row["count"] or 0) if row else 0,
            "latest_updated_at": str(row["latest_updated_at"] or "") if row else "",
            "canonical": False,
            "ordinary_recall": False,
        }


class WindowShadowStore:
    """Append-only full-window self narratives outside ordinary memory buckets."""

    def __init__(self, config: dict):
        self.user_display_name = identity_names(config).get("user_display_name") or "用户"
        state_dir = config.get("state_dir") or os.path.join(
            os.path.dirname(os.path.abspath(config.get("buckets_dir", "buckets"))),
            "state",
        )
        os.makedirs(state_dir, exist_ok=True)
        self.db_path = os.path.join(state_dir, "window_shadows.sqlite")
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS window_shadows (
                window_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL DEFAULT '',
                profile_id TEXT NOT NULL DEFAULT '',
                parent_shadow_id TEXT NOT NULL DEFAULT '',
                idempotency_key TEXT NOT NULL DEFAULT '',
                source_date TEXT NOT NULL DEFAULT '',
                version TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                content TEXT NOT NULL,
                sections_json TEXT NOT NULL DEFAULT '{}',
                moment_bucket_ids_json TEXT NOT NULL DEFAULT '[]',
                continue_scene_id TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_window_shadows_created ON window_shadows(created_at DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_window_shadows_session ON window_shadows(session_id, created_at DESC)"
        )
        columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(window_shadows)").fetchall()
        }
        if "profile_id" not in columns:
            conn.execute("ALTER TABLE window_shadows ADD COLUMN profile_id TEXT NOT NULL DEFAULT ''")
        if "parent_shadow_id" not in columns:
            conn.execute("ALTER TABLE window_shadows ADD COLUMN parent_shadow_id TEXT NOT NULL DEFAULT ''")
        if "idempotency_key" not in columns:
            conn.execute("ALTER TABLE window_shadows ADD COLUMN idempotency_key TEXT NOT NULL DEFAULT ''")
        if "continue_scene_id" not in columns:
            conn.execute("ALTER TABLE window_shadows ADD COLUMN continue_scene_id TEXT NOT NULL DEFAULT ''")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_window_shadows_parent ON window_shadows(parent_shadow_id)"
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_window_shadows_idempotency
            ON window_shadows(idempotency_key)
            WHERE idempotency_key != ''
            """
        )
        conn.commit()
        conn.close()

    @staticmethod
    def source_hash(content: str) -> str:
        return hashlib.sha256(str(content or "").encode("utf-8")).hexdigest()

    @staticmethod
    def _window_id(content_hash: str, session_id: str, idempotency_key: str = "") -> str:
        request_key = str(idempotency_key or "").strip()
        seed = (
            f"idempotency\n{request_key}"
            if request_key
            else f"{str(session_id or '').strip()}\n{content_hash}"
        )
        return "window_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        item = dict(row)
        for key, default in (("sections_json", {}), ("moment_bucket_ids_json", [])):
            try:
                parsed = json.loads(item.pop(key) or json.dumps(default, ensure_ascii=False))
            except (TypeError, ValueError, json.JSONDecodeError):
                parsed = default
            item["sections" if key == "sections_json" else "moment_bucket_ids"] = parsed
        item["scene_bucket_ids"] = list(item.get("moment_bucket_ids") or [])
        item["continue_scene_id"] = str(item.get("continue_scene_id") or "").strip()
        item["ordinary_recall"] = False
        return item

    def plan(
        self,
        content: str,
        *,
        session_id: str = "",
        idempotency_key: str = "",
    ) -> dict[str, str]:
        text = str(content or "")
        content_hash = self.source_hash(text)
        request_key = str(idempotency_key or "").strip()
        window_id = self._window_id(content_hash, session_id, request_key)
        return {
            "window_id": window_id,
            "session_id": str(session_id or "").strip() or window_id,
            "source_hash": content_hash,
            "idempotency_key": request_key,
        }

    def write(
        self,
        content: str,
        *,
        session_id: str = "",
        profile_id: str = "",
        parent_shadow_id: str = "",
        idempotency_key: str = "",
        source_date: str = "",
        sections: dict[str, str] | None = None,
    ) -> tuple[dict, bool]:
        # The full window shadow is an authored artifact. Preserve it byte-for-byte
        # instead of applying the normal memory-content cleanup path.
        text = str(content or "")
        planned = self.plan(
            text,
            session_id=session_id,
            idempotency_key=idempotency_key,
        )
        content_hash = planned["source_hash"]
        window_id = planned["window_id"]
        existing = self.get(window_id)
        if existing:
            return existing, False
        now = _now_utc()
        session_key = str(session_id or "").strip() or window_id
        payload = sections if isinstance(sections, dict) else parse_window_shadow(text)
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO window_shadows (
                window_id, session_id, profile_id, parent_shadow_id, idempotency_key,
                source_date, version, source_hash,
                content, sections_json, moment_bucket_ids_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '[]', ?, ?)
            """,
            (
                window_id,
                session_key,
                str(profile_id or "").strip(),
                str(parent_shadow_id or "").strip(),
                planned["idempotency_key"],
                str(source_date or "").strip(),
                WINDOW_SHADOW_VERSION,
                content_hash,
                text,
                json.dumps(payload, ensure_ascii=False),
                now,
                now,
            ),
        )
        conn.commit()
        conn.close()
        return self.get(window_id) or {}, True

    def get_by_idempotency_key(self, idempotency_key: str) -> dict | None:
        key = str(idempotency_key or "").strip()
        if not key:
            return None
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM window_shadows WHERE idempotency_key = ? LIMIT 1",
            (key,),
        ).fetchone()
        conn.close()
        return self._row(row)

    def attach_moment_buckets(
        self,
        window_id: str,
        bucket_ids: list[str],
        *,
        continue_scene_id: str = "",
    ) -> dict | None:
        clean_ids = list(dict.fromkeys(str(value or "").strip() for value in bucket_ids if str(value or "").strip()))
        primary_id = str(continue_scene_id or "").strip()
        if primary_id and primary_id not in clean_ids:
            raise ValueError("continue_scene_id must belong to scene_bucket_ids")
        conn = self._connect()
        conn.execute(
            "UPDATE window_shadows SET moment_bucket_ids_json = ?, continue_scene_id = ?, updated_at = ? WHERE window_id = ?",
            (
                json.dumps(clean_ids, ensure_ascii=False),
                primary_id,
                _now_utc(),
                str(window_id or "").strip(),
            ),
        )
        conn.commit()
        conn.close()
        return self.get(window_id)

    def attach_scene_buckets(
        self,
        window_id: str,
        bucket_ids: list[str],
        *,
        continue_scene_id: str = "",
    ) -> dict | None:
        return self.attach_moment_buckets(
            window_id,
            bucket_ids,
            continue_scene_id=continue_scene_id,
        )

    def delete(self, window_id: str) -> bool:
        """Rollback a just-created Shadow row; callers must verify ownership."""
        key = str(window_id or "").strip()
        if not key:
            return False
        conn = self._connect()
        cursor = conn.execute("DELETE FROM window_shadows WHERE window_id = ?", (key,))
        conn.commit()
        deleted = cursor.rowcount > 0
        conn.close()
        return deleted

    def get(self, window_id: str) -> dict | None:
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM window_shadows WHERE window_id = ?",
            (str(window_id or "").strip(),),
        ).fetchone()
        conn.close()
        return self._row(row)

    def latest(self, *, exclude_session_id: str = "") -> dict | None:
        conn = self._connect()
        if str(exclude_session_id or "").strip():
            row = conn.execute(
                "SELECT * FROM window_shadows WHERE session_id != ? ORDER BY created_at DESC LIMIT 1",
                (str(exclude_session_id).strip(),),
            ).fetchone()
        else:
            row = conn.execute("SELECT * FROM window_shadows ORDER BY created_at DESC LIMIT 1").fetchone()
        conn.close()
        return self._row(row)

    def latest_handoff_projection(self, *, exclude_session_id: str = "") -> dict | None:
        """Return the latest Shadow's authored self/relationship layers, never its moments."""
        row = self.latest(exclude_session_id=exclude_session_id)
        if not row:
            return None
        projection = project_window_shadow_handoff(
            row.get("sections", {}) if isinstance(row.get("sections"), dict) else {},
            user_display_name=self.user_display_name,
        )
        return {
            "window_id": str(row.get("window_id") or ""),
            "session_id": str(row.get("session_id") or ""),
            "source_date": str(row.get("source_date") or ""),
            "source_hash": str(row.get("source_hash") or ""),
            "scene_bucket_ids": list(row.get("scene_bucket_ids") or []),
            "continue_scene_id": str(row.get("continue_scene_id") or ""),
            **projection,
        }

    def handoff_projection(self, window_id: str) -> dict | None:
        """Return one exact parent Shadow projection; never guess by recency."""
        row = self.get(window_id)
        if not row:
            return None
        projection = project_window_shadow_handoff(
            row.get("sections", {}) if isinstance(row.get("sections"), dict) else {},
            user_display_name=self.user_display_name,
        )
        return {
            "window_id": str(row.get("window_id") or ""),
            "session_id": str(row.get("session_id") or ""),
            "profile_id": str(row.get("profile_id") or ""),
            "parent_shadow_id": str(row.get("parent_shadow_id") or ""),
            "source_date": str(row.get("source_date") or ""),
            "source_hash": str(row.get("source_hash") or ""),
            "scene_bucket_ids": list(row.get("scene_bucket_ids") or []),
            "continue_scene_id": str(row.get("continue_scene_id") or ""),
            **projection,
        }

    def list(self, limit: int = 20, *, include_content: bool = True) -> list[dict]:
        limit = max(1, min(int(limit or 20), 200))
        fields = "*" if include_content else (
            "window_id, session_id, profile_id, parent_shadow_id, source_date, version, source_hash, '' AS content, "
            "sections_json, moment_bucket_ids_json, continue_scene_id, created_at, updated_at"
        )
        conn = self._connect()
        rows = conn.execute(
            f"SELECT {fields} FROM window_shadows ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [self._row(row) or {} for row in rows]

    def portrait_materials(self, limit: int = 4, *, per_item_chars: int = 2400) -> list[dict]:
        rows = self.list(limit=limit, include_content=True)
        output = []
        for row in reversed(rows):
            sections = row.get("sections", {}) if isinstance(row.get("sections"), dict) else {}
            parts = []
            portrait_sections = (
                ("self", "我是谁"),
                ("voice", "我怎么思考、怎么说话"),
                ("relationship", "我们之间是什么"),
                ("interaction", "我们怎么相处"),
            )
            heading_chars = sum(len(label) + 4 for _, label in portrait_sections)
            section_chars = max(120, (max(1, per_item_chars - heading_chars)) // len(portrait_sections))
            for key, label in portrait_sections:
                value = str(sections.get(key) or "").strip()
                if value:
                    if len(value) > section_chars:
                        value = value[: max(1, section_chars - 1)].rstrip() + "…"
                    parts.append(f"[{label}]\n{value}")
            text = "\n\n".join(parts).strip()
            if not text:
                continue
            output.append(
                {
                    "window_id": row.get("window_id", ""),
                    # Portrait evidence uses the immutable artifact id so two
                    # shadows cannot collapse when a caller reuses a session id.
                    "session_id": row.get("window_id", ""),
                    "source_session_id": row.get("session_id", ""),
                    "source_date": row.get("source_date", ""),
                    "created_at": row.get("created_at", ""),
                    "text": text,
                    # Shadow prose is the author's observation, so it can propose a
                    # User or Relationship portrait but can never publish one.
                    "allowed_scopes": ["user", "relationship"],
                }
            )
        return output

    def stats(self) -> dict:
        conn = self._connect()
        count = int(conn.execute("SELECT COUNT(*) FROM window_shadows").fetchone()[0])
        latest = conn.execute("SELECT window_id, created_at FROM window_shadows ORDER BY created_at DESC LIMIT 1").fetchone()
        conn.close()
        return {
            "count": count,
            "latest_window_id": str(latest["window_id"] if latest else ""),
            "latest_created_at": str(latest["created_at"] if latest else ""),
            "db_path": self.db_path,
        }
