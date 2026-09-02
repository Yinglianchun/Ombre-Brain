from __future__ import annotations

import tempfile
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from narrative_rolls import NarrativeRollStore


EVENT_A = "event_111111111111111111111111"
EVENT_B = "event_222222222222222222222222"


def _document() -> str:
    return (
        "# Manual body save\n\n"
        "## 第一人称叙事\n\n"
        "旧正文。\n\n"
        "## 来源账\n\n"
        f"- {EVENT_A}\n"
        f"- {EVENT_B}\n"
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="narrative-body-save-") as temp_dir:
        store = NarrativeRollStore({"state_dir": str(Path(temp_dir) / "state")})
        created = store.publish(
            narrative_id="narrative_manual_body_save",
            document=_document(),
            expected_revision=0,
            title="Manual body save",
            source_event_ids=[EVENT_A, EVENT_B],
            title_aliases=["手改正文"],
            query_cues=["正文保存"],
            time_start="2026-08-01",
            time_end="2026-08-02",
        )
        assert created["status"] == "created", created

        saved = store.save_body(
            "narrative_manual_body_save",
            "新正文第一段。\n\n新正文第二段。",
            expected_revision=created["revision"],
            expected_document_sha256=created["document_sha256"],
        )
        assert saved["status"] == "updated", saved
        assert saved["revision"] == 2, saved
        assert saved["body"] == "新正文第一段。\n\n新正文第二段。", saved
        assert saved["linked_event_ids"] == [EVENT_A, EVENT_B], saved
        assert saved["membership_preserved"] is True, saved
        assert saved["manual_body_save"] is True, saved
        assert saved["title_aliases"] == ["手改正文"], saved
        assert saved["query_cues"] == ["正文保存"], saved
        assert f"- {EVENT_A}" in saved["full_document"], saved
        assert "## 来源账" in saved["full_document"], saved
        assert len(saved["history"]) == 1, saved

        stale_revision = store.save_body(
            "narrative_manual_body_save",
            "不应写入。",
            expected_revision=1,
            expected_document_sha256=saved["document_sha256"],
        )
        assert stale_revision["status"] == "conflict", stale_revision
        stale_hash = store.save_body(
            "narrative_manual_body_save",
            "也不应写入。",
            expected_revision=2,
            expected_document_sha256="0" * 64,
        )
        assert stale_hash["status"] == "conflict", stale_hash
        current = store.read("narrative_manual_body_save")
        assert current["revision"] == 2, current
        assert current["body"] == saved["body"], current

    print("NARRATIVE_BODY_SAVE_OK")


if __name__ == "__main__":
    main()
