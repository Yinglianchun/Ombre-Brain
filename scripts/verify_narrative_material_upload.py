from __future__ import annotations

import hashlib
import sys
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from narrative_materials import material_snapshot_sha256, normalize_material_ids, render_material_snapshot
from narrative_rolls import NarrativeRollStore
from narrative_uploads import MAX_UPLOAD_BYTES, NarrativeUploadStore


EVENT_ID = "event_111111111111111111111111"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="narrative-upload-") as temp_dir:
        state_dir = Path(temp_dir) / "state"
        upload_store = NarrativeUploadStore({"state_dir": str(state_dir)})
        raw = "小雨从本地补来的一段材料。\n第二行。".encode("utf-8")
        created = upload_store.create(raw, filename="../补充材料.md", content_type="text/markdown")
        assert created["status"] == "ok" and created["created"] is True, created
        assert created["filename"] == "补充材料.md", created
        assert created["sha256"] == hashlib.sha256(raw).hexdigest(), created
        assert created["extraction_status"] == "extracted", created
        upload_id = created["upload_id"]
        stored = upload_store.read(upload_id)
        assert stored["status"] == "ok" and stored["extracted_text"] == raw.decode(), stored
        blob = next((state_dir / "narrative_rolls" / "uploads" / "blobs").iterdir())
        assert blob.read_bytes() == raw

        duplicate = upload_store.create(raw, filename="改名.md", content_type="text/plain")
        assert duplicate["status"] == "ok" and duplicate["created"] is False, duplicate
        assert duplicate["upload_id"] == upload_id
        assert len(list((state_dir / "narrative_rolls" / "uploads" / "blobs").iterdir())) == 1

        binary = upload_store.create(b"\x89PNG\r\n\x1a\n", filename="画面.png", content_type="image/png")
        assert binary["status"] == "ok" and binary["extraction_status"] == "metadata_only", binary
        assert upload_store.read(binary["upload_id"])["extracted_text"] == ""
        oversized = upload_store.create(b"x" * (MAX_UPLOAD_BYTES + 1), filename="too-big.bin")
        assert oversized["reason"] == "upload_too_large" and oversized["writes_performed"] == [], oversized

        docx_bytes = BytesIO()
        with zipfile.ZipFile(docx_bytes, "w") as archive:
            archive.writestr("word/document.xml", '<w:document><w:body><w:p><w:r><w:t>文档里的锚点</w:t></w:r></w:p></w:body></w:document>')
        docx = upload_store.create(
            docx_bytes.getvalue(),
            filename="锚点.docx",
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        assert docx["extraction_status"] == "extracted", docx
        assert upload_store.read(docx["upload_id"])["extracted_text"] == "文档里的锚点"

        ids = normalize_material_ids({
            "event_ids": [EVENT_ID],
            "scene_ids": [],
            "diary_ids": [],
            "darkroom_ids": [],
            "upload_ids": [upload_id],
        })
        assert ids["upload_ids"] == [upload_id]
        legacy_client_ids = normalize_material_ids({
            "event_ids": [EVENT_ID],
            "scene_ids": [],
            "diary_ids": [1],
            "darkroom_ids": [],
        }, fallback=ids)
        assert legacy_client_ids["upload_ids"] == [upload_id], legacy_client_ids
        materials = {
            "events": [{"event_id": EVENT_ID, "fingerprint": "a" * 64}],
            "scenes": [],
            "diaries": [],
            "darkrooms": [],
            "uploads": [{
                "upload_id": upload_id,
                "filename": created["filename"],
                "sha256": created["sha256"],
            }],
        }
        snapshot = render_material_snapshot(materials)
        assert upload_id in snapshot and "补充材料.md" in snapshot
        assert material_snapshot_sha256(materials) == material_snapshot_sha256(materials)

        roll_store = NarrativeRollStore({"state_dir": str(state_dir)})
        document = (
            "# Upload\n\n## 第一人称叙事\n\n正文。\n\n## 来源账\n\n"
            f"- {EVENT_ID}\n- {upload_id}\n"
        )
        first = roll_store.publish(
            narrative_id="narrative_upload_test",
            document=document,
            expected_revision=0,
            title="Upload",
            source_event_ids=[EVENT_ID],
            source_upload_ids=[upload_id],
        )
        assert first["status"] == "created" and first["linked_upload_ids"] == [upload_id], first
        revised = roll_store.save_body(
            "narrative_upload_test",
            "去掉本地材料后的正文。",
            expected_revision=first["revision"],
            expected_document_sha256=first["document_sha256"],
            source_event_ids=[EVENT_ID, "event_222222222222222222222222"],
            source_scene_ids=[],
            source_diary_ids=[],
            source_darkroom_ids=[],
            source_upload_ids=[],
            material_snapshot=render_material_snapshot({
                "events": [
                    {"event_id": EVENT_ID, "fingerprint": "a" * 64},
                    {"event_id": "event_222222222222222222222222", "fingerprint": "b" * 64},
                ],
                "scenes": [], "diaries": [], "darkrooms": [], "uploads": [],
            }),
        )
        assert revised["status"] == "updated" and revised["linked_upload_ids"] == [], revised
        assert upload_store.read(upload_id)["status"] == "ok", "unbinding must not delete the uploaded file"

    server = (ROOT / "server.py").read_text(encoding="utf-8")
    ui = (ROOT / "prototypes" / "serein-awake" / "src" / "pages" / "NarrativePage.jsx").read_text(encoding="utf-8")
    vite = (ROOT / "prototypes" / "serein-awake" / "vite.config.mjs").read_text(encoding="utf-8")
    assert "/api/narrative-rolls/material-uploads" in server
    assert '"linked_upload_ids": proposed_material_ids["upload_ids"]' in server
    assert "从本地上传材料" in ui and "uploadNarrativeMaterial" in ui
    assert "/__serein/narrative-material-upload" in vite
    print("NARRATIVE_MATERIAL_UPLOAD_OK")


if __name__ == "__main__":
    main()
