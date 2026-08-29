from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from narrative_rolls import NarrativeRollStore
from narrative_source_verification import verify_narrative_scene_sources


SCENE_A = "scene_mig2_legacy_a"
SCENE_B = "scene_mig2_legacy_b"
SCENE_NEW_INACTIVE = "scene_mig2_new_inactive"


def _document(*scene_ids: str) -> str:
    return (
        "# Legacy reviewed Arc\n\n"
        "## 第一人称叙事\n\n"
        "旧正文必须在修订时原样保留。\n\n"
        "## 来源账\n\n"
        + "\n".join(f"- {scene_id}" for scene_id in scene_ids)
        + "\n"
    )


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="narrative-legacy-scene-") as temp_dir:
        store = NarrativeRollStore({"state_dir": str(Path(temp_dir) / "state")})
        current_document = _document(SCENE_A, SCENE_B)
        created = store.publish(
            narrative_id="narrative_legacy_scene_revision",
            document=current_document,
            expected_revision=0,
            title="Legacy Scene revision",
            source_scene_ids=[SCENE_A, SCENE_B],
        )
        assert created["status"] == "created", created
        current_roll = store.read("narrative_legacy_scene_revision")

        async def get_scene(scene_id: str) -> dict | None:
            if scene_id == SCENE_NEW_INACTIVE:
                return {
                    "id": scene_id,
                    "content": "new inactive",
                    "metadata": {"active": False},
                }
            return None

        is_active = lambda scene: scene.get("metadata", {}).get("active") is not False
        appended_document = current_document + "\n## 新增目录说明\n\n只追加新说明。\n"
        resolved, errors = await verify_narrative_scene_sources(
            narrative_id="narrative_legacy_scene_revision",
            expected_revision=1,
            exact_document=appended_document,
            scene_ids=[SCENE_A, SCENE_B],
            current_roll=current_roll,
            get_scene=get_scene,
            is_active_canonical_scene=is_active,
        )
        assert errors == [], errors
        assert [row["scene_id"] for row in resolved] == [SCENE_A, SCENE_B], resolved
        assert all(row["verification"] == "preserved_existing_legacy" for row in resolved), resolved
        updated = store.publish(
            narrative_id="narrative_legacy_scene_revision",
            document=appended_document,
            expected_revision=1,
            title="Legacy Scene revision",
            source_scene_ids=[SCENE_A, SCENE_B],
        )
        assert updated["status"] == "updated", updated

        modified_document = current_document.replace("旧正文必须", "旧正文已经")
        _resolved, modified_errors = await verify_narrative_scene_sources(
            narrative_id="narrative_legacy_scene_revision",
            expected_revision=1,
            exact_document=modified_document,
            scene_ids=[SCENE_A, SCENE_B],
            current_roll=current_roll,
            get_scene=get_scene,
            is_active_canonical_scene=is_active,
        )
        assert {row["reason"] for row in modified_errors} == {"scene_not_found"}, modified_errors

        with_new_inactive = current_document + f"\n- {SCENE_NEW_INACTIVE}\n"
        preserved, new_errors = await verify_narrative_scene_sources(
            narrative_id="narrative_legacy_scene_revision",
            expected_revision=1,
            exact_document=with_new_inactive,
            scene_ids=[SCENE_A, SCENE_B, SCENE_NEW_INACTIVE],
            current_roll=current_roll,
            get_scene=get_scene,
            is_active_canonical_scene=is_active,
        )
        assert [row["scene_id"] for row in preserved] == [SCENE_A, SCENE_B], preserved
        assert new_errors == [
            {"scene_id": SCENE_NEW_INACTIVE, "reason": "not_active_canonical_scene"}
        ], new_errors

        _resolved, revision_errors = await verify_narrative_scene_sources(
            narrative_id="narrative_legacy_scene_revision",
            expected_revision=2,
            exact_document=appended_document,
            scene_ids=[SCENE_A, SCENE_B],
            current_roll=current_roll,
            get_scene=get_scene,
            is_active_canonical_scene=is_active,
        )
        assert {row["reason"] for row in revision_errors} == {"scene_not_found"}, revision_errors

    print("PASS: legacy Scene bindings are preserved only for exact append-only revisions")


if __name__ == "__main__":
    asyncio.run(main())
