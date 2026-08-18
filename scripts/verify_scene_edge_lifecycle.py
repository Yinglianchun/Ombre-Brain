#!/usr/bin/env python3
"""Verify Scene-edge snapshot refresh, revalidation, cancellation, and restore."""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene_linker import SceneEdgeProposalStore, SceneEdgeStore, SceneLinker, _scene_hash


def scene(scene_id: str, content: str) -> dict:
    return {
        "id": scene_id,
        "content": content,
        "metadata": {
            "object_kind": "scene",
            "memory_value_source": "authored_scene",
            "active": True,
        },
    }


def edge(anchor_id: str, candidate_id: str, confidence: float) -> dict:
    return {
        "anchor_scene_id": anchor_id,
        "candidate_scene_id": candidate_id,
        "source_scene_id": "scene_source",
        "target_scene_id": "scene_target",
        "relation_type": "continues",
        "directionality": "directed",
        "confidence": confidence,
        "reason": "两段具体经历前后延续，并且保留同一个明确约定。",
        "source_evidence": "保留这个约定",
        "target_evidence": "后来真的继续了",
    }


class SceneManager:
    def __init__(self, scenes: list[dict]):
        self.scenes = {item["id"]: item for item in scenes}

    async def get(self, scene_id: str):
        return self.scenes.get(scene_id)

    async def list_all(self, include_archive: bool = False):
        _ = include_archive
        return list(self.scenes.values())


def verify_legacy_status_migration(temp_dir: str) -> None:
    state_dir = Path(temp_dir) / "legacy-state"
    state_dir.mkdir()
    db_path = state_dir / "scene_edge_proposals.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE scene_edges (
            edge_id TEXT PRIMARY KEY,
            source_scene_id TEXT NOT NULL,
            target_scene_id TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            directionality TEXT NOT NULL,
            confidence REAL NOT NULL,
            reason TEXT NOT NULL,
            source_evidence TEXT NOT NULL,
            target_evidence TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            target_hash TEXT NOT NULL,
            proposal_id TEXT NOT NULL UNIQUE,
            linker_version TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            accepted_at TEXT NOT NULL,
            accepted_by TEXT,
            deactivated_at TEXT,
            deactivated_by TEXT,
            deactivation_reason TEXT,
            updated_at TEXT NOT NULL
        )
        """
    )
    values = (
        "scene_source", "scene_target", "continues", "directed", 0.9,
        "legacy", "source", "target", "source-hash", "target-hash",
        "scene_edge_legacy", "scene-linker-v1", "2026-08-01T00:00:00Z",
        "Rain", "2026-08-02T00:00:00Z", "system", "2026-08-02T00:00:00Z",
    )
    conn.execute(
        """
        INSERT INTO scene_edges (
            edge_id, source_scene_id, target_scene_id, relation_type, directionality,
            confidence, reason, source_evidence, target_evidence, source_hash, target_hash,
            proposal_id, linker_version, active, accepted_at, accepted_by,
            deactivated_at, deactivated_by, deactivation_reason, updated_at
        ) VALUES ('scene_rel_legacy', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, NULL, ?)
        """,
        values,
    )
    conn.commit()
    conn.close()
    migrated = SceneEdgeStore({"state_dir": str(state_dir)}, create=False)
    edge = migrated.list_edges(include_inactive=True)[0]
    assert edge["lifecycle_status"] == "cancelled"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="scene-edge-lifecycle-") as temp_dir:
        verify_legacy_status_migration(temp_dir)
        config = {"state_dir": str(Path(temp_dir) / "state")}
        proposals = SceneEdgeProposalStore(config)
        reviewed = SceneEdgeStore(config)

        source = scene("scene_source", "我说要保留这个约定，等下一次继续。")
        target = scene("scene_target", "后来真的继续了，我们没有把它忘掉。")
        first = proposals.replace_for_anchor(
            source,
            {target["id"]: target},
            [edge(source["id"], target["id"], 0.96)],
            model="automatic-old",
        )[0]

        target["content"] += " 只是多补了一句背景。"
        refreshed = proposals.replace_for_anchor(
            target,
            {source["id"]: source},
            [edge(target["id"], source["id"], 0.80)],
            model="automatic-fresh",
        )[0]
        assert refreshed["proposal_id"] == first["proposal_id"]
        assert refreshed["confidence"] == 0.80
        assert refreshed["anchor_scene_id"] == target["id"]
        assert refreshed["anchor_hash"] == _scene_hash(target)
        assert refreshed["proposal_origin"] == "automatic"

        accepted = proposals.promote(refreshed["proposal_id"], reviewed_by="Rain")
        assert accepted and accepted["edge"]["active"] is True

        source["content"] += " 原来的证据还在。"
        lifecycle = proposals.refresh_after_scene_content_change(
            source["id"],
            {source["id"]: source, target["id"]: target},
            created_by="scene_edit",
        )
        assert lifecycle["formal_edges_needing_review"] == 1
        assert lifecycle["normal_relink_required"] is False
        assert len(lifecycle["revalidation_proposal_ids"]) == 1
        inactive = reviewed.list_edges(include_inactive=True)[0]
        assert inactive["active"] is False
        assert inactive["lifecycle_status"] == "needs_review"
        assert inactive["deactivation_reason"] == "scene_content_changed"
        revalidation = proposals.get(lifecycle["revalidation_proposal_ids"][0])
        assert revalidation["proposal_origin"] == "snapshot_revalidation"
        assert revalidation["supersedes_edge_id"] == inactive["edge_id"]

        reaccepted = proposals.promote(revalidation["proposal_id"], reviewed_by="Rain")
        assert reaccepted and reaccepted["edge"]["active"] is True
        assert reaccepted["edge"]["source_hash"] == _scene_hash(source)
        assert reaccepted["edge"]["lifecycle_status"] == "active"

        cancelled = reviewed.deactivate_edge(
            reaccepted["edge"]["edge_id"],
            scene_id=source["id"],
            reviewer="Rain",
            reason="manual_mistake",
        )
        assert cancelled["edge"]["lifecycle_status"] == "cancelled"
        restored = reviewed.restore_edge(
            cancelled["edge"]["edge_id"],
            {source["id"]: source, target["id"]: target},
            reviewer="Rain",
        )
        assert restored["status"] == "restored"
        assert restored["edge"]["restored_by"] == "Rain"

        target["content"] = "原来的目标证据已经被改掉。"
        missing_evidence = proposals.refresh_after_scene_content_change(
            target["id"],
            {source["id"]: source, target["id"]: target},
            created_by="scene_edit",
        )
        assert missing_evidence["formal_edges_needing_review"] == 1
        assert missing_evidence["revalidation_proposal_ids"] == []
        assert missing_evidence["normal_relink_required"] is True
        stale_restore = reviewed.restore_edge(
            restored["edge"]["edge_id"],
            {source["id"]: source, target["id"]: target},
            reviewer="Rain",
        )
        assert stale_restore["status"] == "stale"

        linker = SceneLinker(config)
        manager = SceneManager([source, target])
        manual = asyncio.run(linker.create_manual_proposal(
            source_scene_id=source["id"],
            target_scene_id=target["id"],
            relation_type="echoes",
            source_evidence="保留这个约定",
            target_evidence="原来的目标证据已经被改掉",
            reason="两段具体经历都保留了同一个约定，但落点发生了变化。",
            bucket_mgr=manager,
            created_by="Rain",
        ))
        assert manual["status"] == "pending"
        manual_proposal = manual["proposal"]
        assert manual_proposal["proposal_origin"] == "manual"
        manual_edge = proposals.promote(manual_proposal["proposal_id"], reviewed_by="Rain")["edge"]

        relink = asyncio.run(linker.create_manual_proposal(
            source_scene_id=source["id"],
            target_scene_id=target["id"],
            relation_type="continues",
            source_evidence="保留这个约定",
            target_evidence="原来的目标证据已经被改掉",
            reason="修订后更准确的关系是前一段约定在后一段继续发生。",
            bucket_mgr=manager,
            created_by="Rain",
            supersedes_edge_id=manual_edge["edge_id"],
        ))
        assert relink["status"] == "pending"
        assert relink["proposal"]["proposal_origin"] == "manual_relink"
        assert relink["proposal"]["supersedes_edge_id"] == manual_edge["edge_id"]
        relinked_edge = proposals.promote(relink["proposal"]["proposal_id"], reviewed_by="Rain")["edge"]
        history = {item["edge_id"]: item for item in reviewed.list_edges(include_inactive=True)}
        assert history[manual_edge["edge_id"]]["lifecycle_status"] == "replaced"
        assert history[manual_edge["edge_id"]]["replaced_by_edge_id"] == relinked_edge["edge_id"]
        assert relinked_edge["supersedes_edge_id"] == manual_edge["edge_id"]

        source["content"] += " 启动维护时这句新背景还没有进边快照。"
        maintenance = asyncio.run(linker.reconcile_lifecycle(manager))
        assert maintenance["formal_edges_needing_review"] == 1
        assert len(maintenance["revalidation_proposal_ids"]) == 1
        assert maintenance["normal_relink_scene_ids"] == []

    print("scene edge lifecycle verification passed")


if __name__ == "__main__":
    main()
