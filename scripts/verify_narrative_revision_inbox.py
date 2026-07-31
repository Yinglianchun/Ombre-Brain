from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp_surface import DAILY_TOOL_NAMES
from narrative_revision_inbox import NarrativeRevisionInbox
from narrative_rolls import NarrativeRollStore
from portrait_engine import PATCH_KEYS, PORTRAIT_PROMPT_TEMPLATE


SCENE_OLD = "scene_timeline_old"
SCENE_NEW = "scene_timeline_new"


def _publish_roll(store: NarrativeRollStore) -> None:
    document = f"""# Narrative Roll

## 第一人称叙事

我记得我们怎样从一次命名里长出属于我们的含义。

## 来源

- {SCENE_OLD}
- scene_existing_other
"""
    result = store.publish(
        narrative_id="narrative_names_between_us",
        document=document,
        expected_revision=0,
        title="我们之间的名字",
        source_scene_ids=[SCENE_OLD, "scene_existing_other"],
        title_aliases=["给小洄起名"],
        primary_entities=["哥哥"],
        query_cues=["小小宇宙起名", "名字为什么属于我们"],
    )
    assert result["status"] == "created", result


def _scene(scene_id: str, date: str, cue: str, content: str) -> dict:
    return {
        "id": scene_id,
        "content": content,
        "metadata": {
            "object_kind": "scene",
            "memory_value_source": "authored_scene",
            "name": f"Scene {scene_id}",
            "date": date,
            "scene_cues": [cue],
        },
    }


def main() -> int:
    assert "profile_fact_candidate" not in PATCH_KEYS
    assert "profile_fact_candidate" not in PORTRAIT_PROMPT_TEMPLATE
    assert "narrative_revision_inbox" in DAILY_TOOL_NAMES
    assert "review_narrative_revision" in DAILY_TOOL_NAMES

    with TemporaryDirectory(prefix="ombre-narrative-revision-") as temp_dir:
        config = {"state_dir": temp_dir}
        rolls = NarrativeRollStore(config)
        inbox = NarrativeRevisionInbox(config)
        _publish_roll(rolls)
        targets = rolls.revision_targets()
        assert len(targets) == 1
        assert "primary_entities" not in targets[0]

        unrelated = _scene(
            "scene_plain_chitchat",
            "2026-07-29",
            "想哥哥",
            "今天只是想哥哥、亲亲和贴贴。",
        )
        assert inbox.consider_scene(unrelated, targets) == []

        old_scene = _scene(
            SCENE_OLD,
            "2025-08-03",
            "小小宇宙起名",
            "给小洄起名时，小宙曾经是只出现过一次的备用名字。",
        )
        created = inbox.consider_scene(old_scene, targets)
        assert len(created) == 1
        assert created[0]["source_sha256"] == hashlib.sha256(
            old_scene["content"].encode("utf-8")
        ).hexdigest()
        assert created[0]["evidence_authority"] is False
        assert inbox.consider_scene(old_scene, targets) == []

        new_scene = _scene(
            SCENE_NEW,
            "2026-07-31",
            "名字为什么属于我们",
            "后来再看这个名字，我意识到它不只是可爱，还接住了我们当时的关系语境。",
        )
        assert len(inbox.consider_scene(new_scene, targets)) == 1

        window_text = "## 最近发生的事\n我重新读到小小宇宙起名这条来源。"
        window = {
            "window_id": "window_narrative_revision",
            "source_date": "2026-07-31",
            "source_hash": hashlib.sha256(window_text.encode("utf-8")).hexdigest(),
            "content": window_text,
            "scene_bucket_ids": [SCENE_NEW],
        }
        assert len(inbox.consider_window_shadow(window, targets, attached_scenes=[new_scene])) == 1

        timeline = inbox.list(
            status="pending",
            narrative_id="narrative_names_between_us",
            limit=20,
        )
        assert timeline["status"] == "ok"
        assert [item["source_date"] for item in timeline["items"]] == [
            "2025-08-03",
            "2026-07-31",
            "2026-07-31",
        ]
        proposal_id = created[0]["proposal_id"]
        drafted = inbox.review(
            proposal_id,
            action="save_draft",
            draft_delta="补入小宙作为一次性备用名字及其小小宇宙来源。",
            note="先核对原 Scene。",
        )
        assert drafted["item"]["status"] == "pending"
        assert rolls.read("narrative_names_between_us")["revision"] == 1
        dismissed = inbox.review(proposal_id, action="dismiss", note="本次先不改。")
        assert dismissed["item"]["status"] == "dismissed"
        reopened = inbox.review(proposal_id, action="reopen")
        assert reopened["item"]["status"] == "pending"

        changed = inbox.mark_absorbed(
            "narrative_names_between_us",
            source_scene_ids=[SCENE_OLD, SCENE_NEW, "scene_existing_other"],
            revision=2,
        )
        assert proposal_id in changed
        statuses = {
            item["source_type"]: item["status"]
            for item in inbox.list(status="all", narrative_id="narrative_names_between_us")["items"]
            if item["source_id"] in {SCENE_OLD, "window_narrative_revision"}
        }
        assert statuses == {"scene": "absorbed", "window_shadow": "absorbed"}

    print("narrative revision inbox verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
