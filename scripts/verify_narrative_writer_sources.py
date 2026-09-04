from __future__ import annotations

import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from narrative_writer_sources import conversation_source_messages, writer_material_ids


EVENT_A = "event_111111111111111111111111"
EVENT_B = "event_222222222222222222222222"
SCENE_A = "scene_a"


def ids(*, events=None, scenes=None, diaries=None, uploads=None):
    return {
        "event_ids": events or [],
        "scene_ids": scenes or [],
        "diary_ids": diaries or [],
        "darkroom_ids": [],
        "upload_ids": uploads or [],
    }


def main() -> None:
    content = "小雨：原句。"
    valid = conversation_source_messages([{
        "source_system": "haven_bridge",
        "session_id": "session-1",
        "message_id": "message-1",
        "role": "user",
        "created_at": "2026-09-04T00:00:00Z",
        "content": content,
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }])
    assert valid["status"] == "ok" and valid["messages"][0]["content"] == content, valid
    assert conversation_source_messages([])["status"] == "fallback"
    assert conversation_source_messages([{"content": "", "content_sha256": "a" * 64}])["status"] == "fallback"
    assert conversation_source_messages([{"content": content, "content_sha256": "a" * 64}])["status"] == "invalid"

    current = ids(events=[EVENT_A], scenes=[SCENE_A])
    proposed = ids(events=[EVENT_A, EVENT_B], scenes=[SCENE_A], diaries=[7], uploads=["upload_abc"])
    update = writer_material_ids("update", current, proposed)
    assert update["status"] == "ok" and update["scope"] == "newly_added", update
    assert update["material_ids"] == ids(events=[EVENT_B], diaries=[7], uploads=["upload_abc"]), update
    rewrite = writer_material_ids("rewrite", current, proposed)
    assert rewrite["status"] == "ok" and rewrite["material_ids"] == proposed, rewrite
    assert writer_material_ids("update", current, current)["reason"] == "no_new_materials"
    removed = ids(events=[EVENT_A])
    assert writer_material_ids("update", current, removed)["reason"] == "update_requires_rewrite_after_material_removal"

    server_text = (ROOT / "server.py").read_text(encoding="utf-8")
    assert '"source_mode": "conversation"' in server_text
    assert '"source_mode": "material"' in server_text
    assert '"source_mode": "direct"' in server_text
    assert "material_snapshot_sha256(sealed_materials)" in server_text
    print("NARRATIVE_WRITER_SOURCES_OK")


if __name__ == "__main__":
    main()
