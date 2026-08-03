from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "public" / "private" / "window-shadows-snapshot.json"
OMBRE_HOST = os.environ.get("OMBRE_VPS_HOST", "8.136.154.242")
BRIDGE_HOST = os.environ.get("HAVEN_BRIDGE_VPS_HOST", "168.119.228.217")
BRIDGE_KEY = os.environ.get(
    "HAVEN_BRIDGE_SSH_KEY",
    str(Path.home() / ".ssh" / "id_ed25519"),
)
BRIDGE_HANDOFF_PATH = os.environ.get(
    "HAVEN_BRIDGE_HANDOFF_PATH",
    "/opt/haven_bridge/data/handoffs/20260718-181248-serein-before-window-switch.md",
)

OMBRE_SCRIPT = r"""
import json
import sqlite3

path = "/srv/ombre-brain/state/window_shadows.sqlite"
connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
connection.row_factory = sqlite3.Row
rows = connection.execute(
    "SELECT window_id, session_id, profile_id, source_date, created_at, "
    "source_hash, content FROM window_shadows ORDER BY created_at DESC"
).fetchall()
print(json.dumps([dict(row) for row in rows], ensure_ascii=False))
connection.close()
"""


def run(command: list[str], *, input_text: str | None = None) -> str:
    result = subprocess.run(
        command,
        input=input_text,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr.strip() or f"Command failed: {' '.join(command)}")
    return result.stdout


def fetch_ombre_shadows() -> list[dict]:
    output = run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "KexAlgorithms=curve25519-sha256,curve25519-sha256@libssh.org",
            f"root@{OMBRE_HOST}",
            "python3 -",
        ],
        input_text=OMBRE_SCRIPT,
    )
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        raise SystemExit("The Ombre VPS returned no Window Shadow projection.")
    rows = json.loads(lines[-1])
    if not isinstance(rows, list):
        raise SystemExit("The Ombre Window Shadow projection has an invalid shape.")
    return rows


def fetch_bridge_handoff() -> str:
    return run(
        [
            "ssh",
            "-i",
            BRIDGE_KEY,
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            f"root@{BRIDGE_HOST}",
            "cat",
            BRIDGE_HANDOFF_PATH,
        ],
    )


def normalize_newlines(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def strip_frontmatter(value: str) -> tuple[dict[str, str], str]:
    normalized = normalize_newlines(value)
    match = re.match(r"^---\n(?P<meta>.*?)\n---\n(?P<body>.*)$", normalized, re.DOTALL)
    if not match:
        return {}, normalized

    metadata: dict[str, str] = {}
    for line in match.group("meta").splitlines():
        key, separator, raw_value = line.partition(":")
        if separator:
            metadata[key.strip()] = raw_value.strip().strip('"')
    return metadata, match.group("body").strip()


def first_heading(value: str, fallback: str) -> str:
    headings = re.findall(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", value, re.MULTILINE)
    for heading in headings:
        cleaned = re.sub(r"[*_`]", "", heading).strip()
        if cleaned.lower() not in {"window shadow", "窗影"}:
            return cleaned
    return fallback


def excerpt(value: str, *, limit: int = 96) -> str:
    blocks = re.split(r"\n\s*\n", normalize_newlines(value))
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines or all(line.startswith("#") for line in lines):
            continue
        plain = " ".join(lines)
        plain = re.sub(r"^\s{0,3}#{1,6}\s+", "", plain)
        plain = re.sub(r"^\s*[-*+]\s+", "", plain)
        plain = re.sub(r"[*_`>#]+", "", plain)
        plain = re.sub(r"\s+", " ", plain).strip()
        if plain:
            return plain if len(plain) <= limit else f"{plain[:limit - 1]}…"
    return ""


def display_time(value: str) -> tuple[str, str, str]:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    local = parsed.astimezone(ZoneInfo("Asia/Shanghai"))
    return (
        local.isoformat(timespec="seconds"),
        local.strftime("%Y.%m.%d"),
        local.strftime("%H:%M"),
    )


def relative_label(date_label: str, time_label: str) -> str:
    return f"{date_label.replace('.', ' 年 ', 1).replace('.', ' 月 ', 1)} 日 {time_label}"


def project_ombre_shadow(row: dict) -> dict:
    text = str(row.get("content") or "")
    closed_at, created_date_label, time_label = display_time(str(row["created_at"]))
    source_date = str(row.get("source_date") or "").strip()
    date_label = source_date.replace("-", ".") if source_date else created_date_label
    if source_date and date_label != created_date_label:
        closed_at = f"{source_date}T00:00:00+08:00"
        time_label = "历史补录"
    return {
        "id": str(row["window_id"]),
        "closedAt": closed_at,
        "dateLabel": date_label,
        "timeLabel": time_label,
        "relativeLabel": relative_label(date_label, time_label),
        "title": first_heading(text, f"{date_label} 的窗影"),
        "summary": excerpt(text),
        "text": text,
        "scenes": [],
        "sourceLabel": "Ombre v1 · 杭州",
        "statusLabel": "已入库窗影",
        "sourceKind": "ombre-window-shadow",
        "documentOwnsTitle": True,
        "sourceId": str(row["window_id"]),
        "sourceSessionId": str(row.get("session_id") or ""),
        "contentHash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def project_bridge_handoff(raw_text: str) -> dict:
    metadata, body = strip_frontmatter(raw_text)
    created_at = metadata.get("created_at", "2026-07-18T18:12:48+08:00")
    closed_at, date_label, time_label = display_time(created_at)
    return {
        "id": "handoff-20260718-serein-before-window-switch",
        "closedAt": closed_at,
        "dateLabel": date_label,
        "timeLabel": time_label,
        "relativeLabel": relative_label(date_label, time_label),
        "title": "Serein 换窗前交接",
        "summary": excerpt(body),
        "text": body,
        "scenes": [],
        "sourceLabel": "Haven Bridge · 德国",
        "statusLabel": "交接稿 · 待迁移",
        "sourceKind": "haven-window-handoff",
        "documentOwnsTitle": False,
        "sourceId": BRIDGE_HANDOFF_PATH,
        "sourceSessionId": metadata.get("source_thread_id", ""),
        "contentHash": hashlib.sha256(normalize_newlines(raw_text).encode("utf-8")).hexdigest(),
    }


def fetch_snapshot() -> None:
    ombre_rows = fetch_ombre_shadows()
    shadows = [project_ombre_shadow(row) for row in ombre_rows]
    bridge_raw = fetch_bridge_handoff()
    bridge_metadata, _ = strip_frontmatter(bridge_raw)
    bridge_source_date = str(bridge_metadata.get("created_at") or "")[:10]
    bridge_already_admitted = any(
        str(row.get("source_date") or "") == bridge_source_date
        and re.search(r"历史补录.*handoff Markdown", str(row.get("content") or ""))
        for row in ombre_rows
    )
    if not bridge_already_admitted:
        shadows.append(project_bridge_handoff(bridge_raw))
    shadows.sort(key=lambda shadow: shadow["closedAt"], reverse=True)

    snapshot_seed = "\n".join(
        f"{shadow['sourceId']}:{shadow['contentHash']}" for shadow in shadows
    )
    payload = {
        "snapshotId": hashlib.sha256(snapshot_seed.encode("utf-8")).hexdigest(),
        "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": {
            "ombre": {
                "host": OMBRE_HOST,
                "state": "/srv/ombre-brain/state/window_shadows.sqlite",
                "mode": "read-only SQLite projection",
            },
            "bridge": {
                "host": BRIDGE_HOST,
                "path": BRIDGE_HANDOFF_PATH,
                "mode": "read-only Markdown projection",
            },
        },
        "shadows": shadows,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Fetched {len(shadows)} Window Shadow artifacts to {OUTPUT_PATH}")


if __name__ == "__main__":
    fetch_snapshot()
