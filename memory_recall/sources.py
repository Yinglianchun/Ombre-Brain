from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from memory_moments import build_bucket_recall_item
from memory_relevance import MemoryRelevanceOptions


SCENE_NODE_KIND = "scene"
LEGACY_MOMENT_NODE_KIND = "legacy_moment"


@dataclass(frozen=True)
class RecallNodeRef:
    kind: str
    object_id: str

    @property
    def graph_id(self) -> str:
        if self.kind == SCENE_NODE_KIND:
            return f"scene:{self.object_id}"
        return self.object_id


@dataclass(frozen=True)
class MemoryRecallSourcePartition:
    scenes: list[dict]
    legacy_buckets: list[dict]

    @property
    def scene_map(self) -> dict[str, dict]:
        return {
            str(scene.get("id") or ""): scene
            for scene in self.scenes
            if str(scene.get("id") or "")
        }


class MemoryRecallSources:
    """Own the canonical Scene / legacy bucket source boundary for recall."""

    def __init__(
        self,
        *,
        is_canonical_scene: Callable[[dict | None], bool],
        relevance_options: MemoryRelevanceOptions | None = None,
        annotation_options: dict | None = None,
    ):
        self.is_canonical_scene = is_canonical_scene
        self.relevance_options = relevance_options
        self.annotation_options = dict(annotation_options or {})

    def partition(self, buckets: list[dict]) -> MemoryRecallSourcePartition:
        scenes: list[dict] = []
        legacy_buckets: list[dict] = []
        for bucket in buckets or []:
            if self.is_canonical_scene(bucket):
                scenes.append(bucket)
            else:
                legacy_buckets.append(bucket)
        return MemoryRecallSourcePartition(scenes=scenes, legacy_buckets=legacy_buckets)

    def scene_node(self, scene: dict) -> dict:
        scene_id = str(scene.get("id") or "").strip()
        if not scene_id or not self.is_canonical_scene(scene):
            raise ValueError("canonical authored Scene required")
        meta = scene.get("metadata", {}) if isinstance(scene.get("metadata"), dict) else {}
        canonical_metadata = {
            key: meta.get(key)
            for key in (
                "memory_value_source",
                "write_contract",
                "scene_cues",
                "source_refs",
                "source_turn_ids",
                "source_event_ids",
                "source_hashes",
                "linked_shadow_id",
                "shadow_id",
            )
            if key in meta
        }
        canonical_metadata["object_kind"] = SCENE_NODE_KIND
        return build_bucket_recall_item(
            scene,
            node_id=RecallNodeRef(SCENE_NODE_KIND, scene_id).graph_id,
            node_kind=SCENE_NODE_KIND,
            section="scene",
            source="scene",
            source_id=scene_id,
            extra_metadata=canonical_metadata,
            relevance_options=self.relevance_options,
            annotation_options=self.annotation_options,
        )

    @staticmethod
    def legacy_nodes(moments: list[dict]) -> list[dict]:
        nodes: list[dict] = []
        for moment in moments or []:
            moment_id = str(moment.get("moment_id") or "").strip()
            if not moment_id:
                continue
            node = dict(moment)
            metadata = dict(node.get("metadata", {}) or {})
            metadata["recall_node_kind"] = LEGACY_MOMENT_NODE_KIND
            node["metadata"] = metadata
            node["metadata_json"] = json.dumps(
                metadata,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            node["node_id"] = moment_id
            node["node_kind"] = LEGACY_MOMENT_NODE_KIND
            nodes.append(node)
        return nodes

    @staticmethod
    def scene_edges_as_recall_edges(scene_edges: list[dict]) -> list[dict]:
        projected: list[dict] = []
        for edge in scene_edges or []:
            source_id = str(edge.get("source") or "").strip()
            target_id = str(edge.get("target") or "").strip()
            if not source_id or not target_id:
                continue
            projected.append(
                {
                    **edge,
                    "source": RecallNodeRef(SCENE_NODE_KIND, source_id).graph_id,
                    "target": RecallNodeRef(SCENE_NODE_KIND, target_id).graph_id,
                    "source_kind": SCENE_NODE_KIND,
                    "target_kind": SCENE_NODE_KIND,
                    "source_object_id": source_id,
                    "target_object_id": target_id,
                    "graph_scope": "scene",
                }
            )
        return projected

    @staticmethod
    def decorate_same_event_clusters(nodes: list[dict], bridge_edges: list[dict]) -> list[dict]:
        cluster_by_node: dict[str, str] = {}
        for edge in bridge_edges or []:
            if str(edge.get("relation_type") or "") != "same_event":
                continue
            scene_id = str(edge.get("scene_id") or "").strip()
            legacy_moment_id = str(edge.get("legacy_moment_id") or "").strip()
            if not scene_id or not legacy_moment_id:
                continue
            cluster_by_node[RecallNodeRef(SCENE_NODE_KIND, scene_id).graph_id] = scene_id
            cluster_by_node[legacy_moment_id] = scene_id

        decorated: list[dict] = []
        for item in nodes or []:
            node_id = str(item.get("node_id") or item.get("moment_id") or "")
            cluster = cluster_by_node.get(node_id)
            if not cluster:
                decorated.append(item)
                continue
            node = dict(item)
            metadata = dict(node.get("metadata", {}) or {})
            metadata["same_event_cluster"] = cluster
            metadata["same_event_scene_id"] = cluster
            node["metadata"] = metadata
            node["metadata_json"] = json.dumps(
                metadata,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            node["same_event_cluster"] = cluster
            node["same_event_scene_id"] = cluster
            decorated.append(node)
        return decorated
