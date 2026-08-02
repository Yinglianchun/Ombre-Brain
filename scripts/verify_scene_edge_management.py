#!/usr/bin/env python3
"""Verify exact, endpoint-bound soft removal of reviewed Scene edges."""

import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene_linker import SceneEdgeStore


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        config = {"state_dir": str(Path(temp_dir) / "state")}
        store = SceneEdgeStore(config)
        conn = sqlite3.connect(store.db_path)
        conn.execute(
            """
            INSERT INTO scene_edges (
                edge_id, source_scene_id, target_scene_id, relation_type,
                directionality, confidence, reason, source_evidence,
                target_evidence, source_hash, target_hash, proposal_id,
                linker_version, active, accepted_at, accepted_by, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
            """,
            (
                "scene_rel_test",
                "scene_source",
                "scene_target",
                "echoes",
                "symmetric",
                0.9,
                "test edge",
                "source evidence",
                "target evidence",
                "source hash",
                "target hash",
                "scene_edge_test",
                "scene-linker-test",
                "2026-08-02T00:00:00Z",
                "Rain",
                "2026-08-02T00:00:00Z",
            ),
        )
        conn.commit()
        conn.close()

        mismatch = store.deactivate_edge(
            "scene_rel_test",
            scene_id="scene_unrelated",
            reviewer="Rain",
        )
        assert mismatch["status"] == "invalid"
        assert store.list_edges()[0]["active"] is True

        removed = store.deactivate_edge(
            "scene_rel_test",
            scene_id="scene_source",
            reviewer="Rain",
            reason="manual_mistake",
        )
        assert removed["status"] == "deactivated"
        assert removed["edge"]["active"] is False
        assert removed["edge"]["deactivated_by"] == "Rain"
        assert removed["edge"]["deactivation_reason"] == "manual_mistake"
        assert store.list_edges() == []
        assert len(store.list_edges(include_inactive=True)) == 1

        repeated = store.deactivate_edge(
            "scene_rel_test",
            scene_id="scene_target",
            reviewer="Rain",
        )
        assert repeated["status"] == "unchanged"

    print("scene edge management verification passed")


if __name__ == "__main__":
    main()
