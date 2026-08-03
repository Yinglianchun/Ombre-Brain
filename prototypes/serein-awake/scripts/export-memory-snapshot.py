from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = ROOT / ".private" / "serein-memory-rehearsal.sqlite"
REPORT_PATH = ROOT / ".private" / "serein-memory-import-report.json"
OUTPUT_PATH = ROOT / "public" / "private" / "memory-snapshot.json"
EDGE_SNAPSHOT_PATH = ROOT / ".private" / "ombre-scene-edges-vps.json"
PROJECTION_VERSION = "memory-projection-v5"


def split_paragraphs(content: str) -> list[str]:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    return [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", normalized)
        if paragraph.strip()
    ]


def excerpt_from(paragraphs: list[str]) -> str:
    if not paragraphs:
        return ""
    plain = re.sub(r"^\s{0,3}#{1,6}\s+", "", paragraphs[0])
    plain = re.sub(r"[*_`>#-]+", "", plain)
    plain = re.sub(r"\s+", " ", plain).strip()
    return plain if len(plain) <= 108 else f"{plain[:107]}…"


def scene_date(occurred_at: str | None, recorded_at: str | None) -> str:
    for value in (occurred_at, recorded_at):
        if value and re.match(r"^\d{4}-\d{2}-\d{2}", value):
            return value[:10]
    return "2026-01-01"


def annotation_identity(author: str | None) -> tuple[str, str]:
    if author == "Rain":
        return "Rain", "user"
    if author == "Haven":
        return "Haven", "assistant"
    return "旧记忆", "source"


def bucket_domain(source_path: str) -> str:
    parts = Path(source_path).parts
    for index, part in enumerate(parts):
        if part.lower() not in {"archive", "dynamic", "permanent"}:
            continue
        if index + 1 < len(parts) - 1:
            return parts[index + 1]
    return ""


def source_is_self_anchor(source_path: str) -> bool:
    try:
        source_text = Path(source_path).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return False
    frontmatter = source_text.split("---", 2)
    metadata = frontmatter[1] if len(frontmatter) == 3 else ""
    return bool(
        re.search(
            r"(?m)^(?:migration_source_)?self_anchor:\s*true\s*$",
            metadata,
        )
    )


def export_snapshot() -> None:
    if not DATABASE_PATH.exists() or not REPORT_PATH.exists():
        raise SystemExit("Missing private Serein import rehearsal artifacts.")

    database_hash = hashlib.sha256(DATABASE_PATH.read_bytes()).hexdigest()
    edge_snapshot_hash = (
        hashlib.sha256(EDGE_SNAPSHOT_PATH.read_bytes()).hexdigest()
        if EDGE_SNAPSHOT_PATH.exists()
        else "no-edge-snapshot"
    )
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    connection = sqlite3.connect(
        f"file:{DATABASE_PATH.as_posix()}?mode=ro",
        uri=True,
    )
    connection.row_factory = sqlite3.Row

    import_sources = {
        row["target_id"]: dict(row)
        for row in connection.execute(
            "SELECT target_id, source_id, source_path "
            "FROM import_items WHERE status = 'imported' AND target_type = 'scene'"
        )
    }
    scene_id_by_source_id = {
        str(row["source_id"]): str(row["target_id"])
        for row in import_sources.values()
        if row.get("source_id") and row.get("target_id")
    }
    curations = {
        row["scene_id"]: bool(row["favorite"])
        for row in connection.execute(
            "SELECT scene_id, favorite FROM scene_curations"
        )
    }
    annotations: dict[str, list[dict]] = {}
    for row in connection.execute(
        "SELECT annotation_id, target_id, text, author, created_at "
        "FROM annotations WHERE target_type = 'scene' AND status = 'active' "
        "ORDER BY created_at, annotation_id"
    ):
        author, role = annotation_identity(row["author"])
        annotations.setdefault(row["target_id"], []).append(
            {
                "id": row["annotation_id"],
                "author": author,
                "role": role,
                "createdAt": row["created_at"][:10],
                "content": row["text"],
            }
        )

    sources: dict[str, list[dict]] = {}
    for row in connection.execute(
        "SELECT scene_id, source_type, source_id, ordinal "
        "FROM scene_sources ORDER BY scene_id, ordinal"
    ):
        sources.setdefault(row["scene_id"], []).append(
            {
                "id": f"{row['source_type']}:{row['source_id']}",
                "kind": "Ombre v1 原文",
                "title": f"只读来源 · {row['source_id']}",
            }
        )

    related: dict[str, dict[str, list[dict]]] = {}

    def add_relation(
        source_scene_id: str,
        target_scene_id: str,
        *,
        edge_id: str,
        relation_type: str,
        directionality: str,
        confidence: float,
    ) -> None:
        if not source_scene_id or not target_scene_id or source_scene_id == target_scene_id:
            return
        source_direction = "symmetric" if directionality == "symmetric" else "outgoing"
        target_direction = "symmetric" if directionality == "symmetric" else "incoming"
        related.setdefault(source_scene_id, {}).setdefault(target_scene_id, []).append(
            {
                "edgeId": edge_id,
                "type": relation_type,
                "direction": source_direction,
                "confidence": confidence,
            }
        )
        related.setdefault(target_scene_id, {}).setdefault(source_scene_id, []).append(
            {
                "edgeId": edge_id,
                "type": relation_type,
                "direction": target_direction,
                "confidence": confidence,
            }
        )

    for row in connection.execute(
        "SELECT relation_id, source_scene_id, target_scene_id, relation_type, "
        "strength, created_by FROM scene_relations "
        "WHERE status = 'active'"
    ):
        add_relation(
            row["source_scene_id"],
            row["target_scene_id"],
            edge_id=row["relation_id"],
            relation_type=row["relation_type"],
            directionality="symmetric" if row["relation_type"] in {
                "echoes",
                "contrasts_with",
            } else "directed",
            confidence=float(row["strength"]),
        )

    projected_edge_count = 0
    if EDGE_SNAPSHOT_PATH.exists():
        edge_snapshot = json.loads(EDGE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
        for edge in edge_snapshot.get("edges", []):
            source_scene_id = scene_id_by_source_id.get(str(edge.get("sourceId") or ""))
            target_scene_id = scene_id_by_source_id.get(str(edge.get("targetId") or ""))
            if not source_scene_id or not target_scene_id:
                continue
            add_relation(
                source_scene_id,
                target_scene_id,
                edge_id=str(edge.get("edgeId") or ""),
                relation_type=str(edge.get("relationType") or "relates_to"),
                directionality=str(edge.get("directionality") or "directed"),
                confidence=float(edge.get("confidence") or 0),
            )
            projected_edge_count += 1

    scenes = []
    for row in connection.execute(
        "SELECT scene_id, text, title, occurred_at, recorded_at, author "
        "FROM scenes WHERE status = 'active' "
        "ORDER BY coalesce(occurred_at, recorded_at) DESC, scene_id"
    ):
        body = split_paragraphs(row["text"])
        source = import_sources.get(row["scene_id"], {})
        original_source_path = str(source.get("source_path") or "")
        source_path = original_source_path.replace("\\", "/").lower()
        source_domain = bucket_domain(original_source_path)
        scene_sources = sources.get(row["scene_id"], [])
        related_scene_map = related.get(row["scene_id"], {})
        related_scenes = [
            {
                "id": related_scene_id,
                "relations": sorted(
                    relations,
                    key=lambda relation: (
                        -float(relation["confidence"]),
                        relation["type"],
                        relation["edgeId"],
                    ),
                ),
            }
            for related_scene_id, relations in sorted(related_scene_map.items())
        ]
        related_ids = [item["id"] for item in related_scenes]
        scenes.append(
            {
                "id": row["scene_id"],
                "date": scene_date(row["occurred_at"], row["recorded_at"]),
                "title": row["title"] or "没有题目的一幕",
                "excerpt": excerpt_from(body),
                "body": body,
                "author": row["author"],
                "annotations": annotations.get(row["scene_id"], []),
                "sources": scene_sources,
                "sourceCount": len(scene_sources),
                "relatedScenes": related_scenes,
                "relatedSceneIds": related_ids,
                "relationCount": len(related_ids),
                "narrativeRefs": [],
                "favorite": curations.get(row["scene_id"], False),
                "status": "已沉底" if "/archive/" in source_path else "可浮现",
                "bucketDomain": source_domain,
                "selfAnchor": source_is_self_anchor(original_source_path),
                "sourceKind": "serein-import-rehearsal",
            }
        )

    connection.close()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "snapshotId": f"{database_hash}:{edge_snapshot_hash}:{PROJECTION_VERSION}",
        "projectionVersion": PROJECTION_VERSION,
        "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": {
            "host": "Hangzhou VPS",
            "system": "Ombre v1 read-only snapshot",
            "projection": "Serein import rehearsal",
            "manifestHash": report.get("source_manifest_hash", ""),
        },
        "rehearsal": {
            "scanned": report.get("scanned", 0),
            "imported": report.get("imported", 0),
            "skipped": report.get("skipped", 0),
            "failed": report.get("failed", 0),
            "validatedEdgesProjected": projected_edge_count,
        },
        "scenes": scenes,
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Exported {len(scenes)} rehearsed Scenes and "
        f"{sum(len(scene['annotations']) for scene in scenes)} annotations "
        f"to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    export_snapshot()
