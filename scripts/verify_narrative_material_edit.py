from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from narrative_materials import (
    material_delta,
    material_snapshot_sha256,
    narrative_preview_fingerprint,
    normalize_material_ids,
    render_material_snapshot,
)
from narrative_rolls import NarrativeRollStore


EVENT_A = "event_111111111111111111111111"
EVENT_B = "event_222222222222222222222222"
EVENT_C = "event_333333333333333333333333"


def _document() -> str:
    return (
        "# Material edit\n\n"
        "## 第一人称叙事\n\n"
        "旧正文。\n\n"
        "## 来源账\n\n"
        f"- {EVENT_A}\n"
        f"- {EVENT_B}\n"
    )


def _materials(event_ids: list[str], salt: str = "") -> dict:
    return {
        "status": "ok",
        "events": [
            {
                "event_id": event_id,
                "fingerprint": hashlib.sha256(f"{event_id}:{salt}".encode()).hexdigest(),
            }
            for event_id in event_ids
        ],
        "scenes": [],
        "diaries": [],
        "darkrooms": [],
    }


def main() -> None:
    current_ids = normalize_material_ids({
        "event_ids": [EVENT_A, EVENT_B],
        "scene_ids": [],
        "diary_ids": [],
        "darkroom_ids": [],
    })
    proposed_ids = normalize_material_ids({
        "event_ids": [EVENT_A, EVENT_C],
        "scene_ids": [],
        "diary_ids": [],
        "darkroom_ids": [],
    })
    delta = material_delta(current_ids, proposed_ids)
    assert delta["added"]["event_ids"] == [EVENT_C], delta
    assert delta["removed"]["event_ids"] == [EVENT_B], delta

    materials = _materials([EVENT_A, EVENT_C])
    snapshot_hash = material_snapshot_sha256(materials)
    base_document_hash = "a" * 64
    preview = narrative_preview_fingerprint(
        narrative_id="narrative_material_edit",
        revision=1,
        document_sha256=base_document_hash,
        body="新正文。",
        material_snapshot_sha256_value=snapshot_hash,
    )
    assert preview == narrative_preview_fingerprint(
        narrative_id="narrative_material_edit",
        revision=1,
        document_sha256=base_document_hash,
        body="新正文。",
        material_snapshot_sha256_value=snapshot_hash,
    )
    drifted_hash = material_snapshot_sha256(_materials([EVENT_A, EVENT_C], "drift"))
    assert drifted_hash != snapshot_hash
    assert preview != narrative_preview_fingerprint(
        narrative_id="narrative_material_edit",
        revision=1,
        document_sha256=base_document_hash,
        body="新正文。",
        material_snapshot_sha256_value=drifted_hash,
    )

    with tempfile.TemporaryDirectory(prefix="narrative-material-edit-") as temp_dir:
        store = NarrativeRollStore({"state_dir": str(Path(temp_dir) / "state")})
        created = store.publish(
            narrative_id="narrative_material_edit",
            document=_document(),
            expected_revision=0,
            title="Material edit",
            source_event_ids=[EVENT_A, EVENT_B],
        )
        assert created["status"] == "created", created
        saved = store.save_body(
            "narrative_material_edit",
            "新正文。",
            expected_revision=created["revision"],
            expected_document_sha256=created["document_sha256"],
            source_scene_ids=[],
            source_event_ids=[EVENT_A, EVENT_C],
            source_diary_ids=[],
            source_darkroom_ids=[],
            material_snapshot=render_material_snapshot(materials),
        )
        assert saved["status"] == "updated", saved
        assert saved["linked_event_ids"] == [EVENT_A, EVENT_C], saved
        assert saved["membership_changed"] is True, saved
        assert saved["excluded_event_ids"] == [EVENT_B], saved
        assert saved["history"][0]["linked_event_ids"] == [EVENT_A, EVENT_B], saved
        assert EVENT_B in saved["full_document"], "removal must not rewrite historical source text"
        assert "## 绑定材料快照" in saved["full_document"]

        stale = store.save_body(
            "narrative_material_edit",
            "不得写入。",
            expected_revision=1,
            expected_document_sha256=saved["document_sha256"],
        )
        assert stale["status"] == "conflict", stale
        current = store.read("narrative_material_edit")
        assert current["revision"] == 2 and current["body"] == "新正文。", current

    server_text = (ROOT / "server.py").read_text(encoding="utf-8")
    vite_text = (ROOT / "prototypes" / "serein-awake" / "vite.config.mjs").read_text(encoding="utf-8")
    ui_text = (ROOT / "prototypes" / "serein-awake" / "src" / "pages" / "NarrativePage.jsx").read_text(encoding="utf-8")
    for required in (
        "material_snapshot_changed",
        "preview_fingerprint_mismatch",
        "_materialize_narrative_writer_sources(proposed_narrative)",
        "render_material_snapshot(fresh_materials)",
    ):
        assert required in server_text, required
    assert "preview_fingerprint: String(body.previewFingerprint" in vite_text
    assert "本次材料变更" in ui_text and "确认保存" in ui_text

    print("NARRATIVE_MATERIAL_EDIT_OK")


if __name__ == "__main__":
    main()
