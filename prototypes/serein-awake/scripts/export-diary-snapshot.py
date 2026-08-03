from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = ROOT / ".private" / "diary-vps.db"
OUTPUT_PATH = ROOT / "public" / "private" / "diary-snapshot.json"


def parse_json(value: str | None, fallback):
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def split_paragraphs(content: str) -> list[str]:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    paragraphs = [
        re.sub(r"\n+", "\n", paragraph).strip()
        for paragraph in re.split(r"\n\s*\n", normalized)
    ]
    return [paragraph for paragraph in paragraphs if paragraph]


def excerpt_from(paragraphs: list[str]) -> str:
    if not paragraphs:
        return ""
    plain = re.sub(r"^\s{0,3}#{1,6}\s+", "", paragraphs[0])
    plain = re.sub(r"[*_`>#-]+", "", plain)
    plain = re.sub(r"\s+", " ", plain).strip()
    return plain if len(plain) <= 92 else f"{plain[:91]}…"


def local_time(value: str | None) -> str:
    if not value:
        return "00:00"
    try:
        return datetime.fromisoformat(value).strftime("%H:%M")
    except ValueError:
        match = re.search(r"\b(\d{2}:\d{2})", value)
        return match.group(1) if match else "00:00"


def comment_time(value: str | None) -> str:
    if not value:
        return ""
    try:
        return datetime.fromisoformat(value).strftime("%m月%d日 %H:%M")
    except ValueError:
        return value


def identity(author: str | None) -> tuple[str, str]:
    if author == "user":
        return "Rain", "user"
    if author == "ai":
        return "Haven", "assistant"
    return author or "Haven", "assistant"


def export_snapshot() -> None:
    if not DATABASE_PATH.exists():
        raise SystemExit(f"Missing private diary snapshot: {DATABASE_PATH}")

    database_hash = hashlib.sha256(DATABASE_PATH.read_bytes()).hexdigest()
    connection = sqlite3.connect(
        f"file:{DATABASE_PATH.as_posix()}?mode=ro",
        uri=True,
    )
    connection.row_factory = sqlite3.Row

    comments_by_diary: dict[int, list[dict]] = {}
    for row in connection.execute(
        "SELECT id, diary_id, content, author, created_at "
        "FROM comments ORDER BY created_at, id"
    ):
        author, role = identity(row["author"])
        comments_by_diary.setdefault(row["diary_id"], []).append(
            {
                "id": f"diary-comment-vps-{row['id']}",
                "author": author,
                "role": role,
                "createdAt": comment_time(row["created_at"]),
                "content": row["content"],
            }
        )

    entries = []
    rows = connection.execute(
        "SELECT id, date, content, author, emotion_tags, created_at, title, "
        "entry_type, visibility, unlock_at, revision, source_id, metadata "
        "FROM diaries "
        "WHERE deleted_at = '' AND visibility = 'active' "
        "ORDER BY date DESC, created_at DESC, id DESC"
    )
    for row in rows:
        author, role = identity(row["author"])
        body = split_paragraphs(row["content"])
        metadata = parse_json(row["metadata"], {})
        entries.append(
            {
                "id": f"diary-vps-{row['id']}",
                "date": row["date"],
                "time": local_time(row["created_at"]),
                "author": author,
                "role": role,
                "darkroom": row["entry_type"] == "darkroom",
                "unlockAt": row["unlock_at"] or "",
                "title": row["title"] or f"{row['date']} 的日记",
                "excerpt": excerpt_from(body),
                "body": body,
                "references": [],
                "comments": comments_by_diary.get(row["id"], []),
                "revision": row["revision"],
                "sourceId": row["source_id"] or metadata.get("legacy_entry_id", ""),
                "emotionTags": parse_json(row["emotion_tags"], []),
            }
        )

    connection.close()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "snapshotId": database_hash,
        "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": {
            "host": "Hangzhou VPS",
            "service": "Ombre-Brain DiaryStore",
        },
        "entries": entries,
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    diary_count = sum(not entry["darkroom"] for entry in entries)
    darkroom_count = sum(entry["darkroom"] for entry in entries)
    comment_count = sum(len(entry["comments"]) for entry in entries)
    print(
        f"Exported {diary_count} diaries, {darkroom_count} darkroom entries, "
        f"and {comment_count} comments to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    export_snapshot()
