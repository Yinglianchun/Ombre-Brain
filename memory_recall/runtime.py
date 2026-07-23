from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any, Awaitable, Callable

from memory_edges import legacy_edges_for_recall, legacy_moment_edges_for_recall
from memory_moments import MemoryMomentStore
from .sources import MemoryRecallSources


@dataclass(frozen=True)
class MemoryRecallGraph:
    """One coherent moment-graph snapshot used by direct and diffused recall."""

    moments: list[dict]
    grouped_moments: dict[str, list[dict]]
    edges: list[dict]
    scene_native: bool = False

    @property
    def nodes(self) -> list[dict]:
        return self.moments

    @property
    def grouped_nodes(self) -> dict[str, list[dict]]:
        return self.grouped_moments

    def as_legacy_tuple(
        self,
    ) -> tuple[list[dict], dict[str, list[dict]], list[dict]]:
        return self.moments, self.grouped_moments, self.edges


@dataclass(frozen=True)
class MemoryRecallResult:
    """Direct selection and optional diffusion produced from one graph snapshot."""

    graph: MemoryRecallGraph
    direct_moments: list[dict]
    candidate_moments: list[dict]
    suppressed_moments: list[dict]
    suppressed_buckets: list[dict]
    query_planner_debug: dict[str, Any]
    related_context: str = ""
    diffused_debug: list[dict[str, Any]] = field(default_factory=list)

    def with_diffusion(
        self,
        related_context: str,
        diffused_debug: list[dict[str, Any]],
    ) -> "MemoryRecallResult":
        return replace(
            self,
            related_context=str(related_context or ""),
            diffused_debug=list(diffused_debug or []),
        )


@dataclass(frozen=True)
class MemoryDiffusionPlan:
    """Pure graph inputs and limits for one diffusion pass."""

    moment_map: dict[str, dict]
    seed_scores: dict[str, float]
    explore_options: Any
    explore_limit: int
    inject_limit: int
    same_event_seed_clusters: frozenset[str] = frozenset()


@dataclass(frozen=True)
class MemoryDiffusionSelection:
    """Ranked candidates and the subset allowed through the item limit."""

    candidates: list[dict[str, Any]]
    selected: list[dict[str, Any]]


class MemoryDiffusionCandidatePool:
    """Deduplicate, rank, and limit already-gated diffusion candidates."""

    _WHY_PRIORITY = {
        "same_topic": 5,
        "date_neighbor": 4,
        "semantic_neighbor": 3,
        "explicit_edge": 3,
    }

    def __init__(self, *, excluded_same_event_clusters: frozenset[str] = frozenset()) -> None:
        self._by_memory: dict[str, dict[str, Any]] = {}
        self._excluded_same_event_clusters = frozenset(excluded_same_event_clusters)

    def add(self, row: dict[str, Any]) -> None:
        bucket_id = str(row.get("bucket_id") or "")
        if not bucket_id:
            return
        cluster_id = self._same_event_cluster(row)
        if cluster_id and cluster_id in self._excluded_same_event_clusters:
            row["injectable"] = False
            row["gate_allowed"] = False
            row["gate_reason"] = "same_event_scene_primary"
            row["suppression_reason"] = "same_event_scene_primary"
        row["rank_key"] = self.rank_key(row)
        memory_key = f"same_event:{cluster_id}" if cluster_id else f"bucket:{bucket_id}"
        existing = self._by_memory.get(memory_key)
        if existing is None or row["rank_key"] > existing.get("rank_key", ()):
            self._by_memory[memory_key] = row

    def select(self, inject_limit: int) -> MemoryDiffusionSelection:
        candidates = sorted(
            self._by_memory.values(),
            key=lambda row: row.get("rank_key", ()),
            reverse=True,
        )
        selected = [row for row in candidates if row.get("injectable")][
            : max(0, int(inject_limit))
        ]
        selected_ids = {id(row) for row in selected}
        for row in candidates:
            if row.get("injectable") and id(row) not in selected_ids:
                row["suppression_reason"] = "inject_limit"
        return MemoryDiffusionSelection(candidates=candidates, selected=selected)

    @staticmethod
    def _same_event_cluster(row: dict[str, Any]) -> str:
        moment = row.get("moment") if isinstance(row.get("moment"), dict) else {}
        metadata = moment.get("metadata", {}) if isinstance(moment.get("metadata"), dict) else {}
        return str(
            row.get("same_event_cluster")
            or moment.get("same_event_cluster")
            or metadata.get("same_event_cluster")
            or ""
        ).strip()

    @classmethod
    def rank_key(cls, row: dict[str, Any]) -> tuple:
        why_priority = cls._WHY_PRIORITY.get(str(row.get("why") or ""), 1)
        path_len = int(row.get("path_len") or 0)
        return (
            1 if row.get("injectable") else 0,
            1 if row.get("chain_bundle") else 0,
            why_priority,
            1 if row.get("has_topic_evidence") else 0,
            _safe_float(row.get("confidence"), 0.0),
            _safe_float(row.get("activation"), 0.0),
            -path_len,
        )


class MemoryRecallRuntime:
    """Own moment graph refresh, cache stamps, and snapshot consistency.

    Domain policy remains outside this cache/runtime boundary.  Callbacks decide
    which buckets/moments are recallable and how reviewed bucket edges project
    into moment edges; the runtime only builds one internally consistent graph.
    """

    def __init__(
        self,
        *,
        moment_store: MemoryMomentStore,
        memory_edge_store: object,
        scene_edge_store: object,
        is_excluded_bucket: Callable[[dict | None], bool],
        bucket_edges_for_recall: Callable[[list[dict]], list[dict]],
        graph_signature: Callable[[list[dict], list[dict] | None], str],
        recallable_moments: Callable[[list[dict]], list[dict]],
        moments_by_bucket: Callable[[list[dict]], dict[str, list[dict]]],
        bucket_edges_as_moment_edges: Callable[
            [list[dict], dict[str, list[dict]]],
            list[dict],
        ],
        recall_sources: MemoryRecallSources | None = None,
        legacy_scene_bridge_store: object | None = None,
        scene_native_enabled: bool = False,
    ):
        self.moment_store = moment_store
        self.memory_edge_store = memory_edge_store
        self.scene_edge_store = scene_edge_store
        self.is_excluded_bucket = is_excluded_bucket
        self.bucket_edges_for_recall = bucket_edges_for_recall
        self.graph_signature = graph_signature
        self.recallable_moments = recallable_moments
        self.moments_by_bucket = moments_by_bucket
        self.bucket_edges_as_moment_edges = bucket_edges_as_moment_edges
        self.recall_sources = recall_sources
        self.legacy_scene_bridge_store = legacy_scene_bridge_store
        self.scene_native_enabled = bool(scene_native_enabled and recall_sources is not None)
        self.clear()

    def clear(self) -> None:
        self._cache_signature = ""
        self._cache_value: MemoryRecallGraph | None = None
        self._cache_bucket_list_id = 0
        self._cache_edge_stamp: tuple[Any, ...] = ()
        self._cache_store_stamp: tuple[int, int] = (0, 0)

    def refresh(self, all_buckets: list[dict]) -> MemoryRecallGraph:
        self._prune_excluded_moment_index(all_buckets)
        if self.scene_native_enabled:
            self._prune_native_scene_moment_index(all_buckets)
        bucket_list_id = id(all_buckets)
        edge_stamp = self._edge_store_stamp()
        store_stamp = self._moment_store_stamp()
        if (
            self._cache_value is not None
            and bucket_list_id == self._cache_bucket_list_id
            and edge_stamp == self._cache_edge_stamp
            and store_stamp == self._cache_store_stamp
        ):
            return self._cache_value

        recallable_buckets = [
            bucket
            for bucket in all_buckets
            if not self.is_excluded_bucket(bucket)
        ]
        recallable_bucket_ids = {
            str(bucket.get("id") or "")
            for bucket in recallable_buckets
            if str(bucket.get("id") or "")
        }
        if self.scene_native_enabled:
            return self._refresh_scene_native(
                recallable_buckets,
                bucket_list_id=bucket_list_id,
                edge_stamp=edge_stamp,
                store_stamp=store_stamp,
            )

        bucket_edges = self.bucket_edges_for_recall(recallable_buckets)
        signature = self.graph_signature(recallable_buckets, bucket_edges)
        if (
            signature
            and signature == self._cache_signature
            and self._cache_value is not None
            and store_stamp == self._cache_store_stamp
        ):
            return self._cache_value

        self.moment_store.bulk_upsert(recallable_buckets)
        moments = [
            moment
            for moment in self.recallable_moments(self.moment_store.list_all())
            if str(moment.get("bucket_id") or "") in recallable_bucket_ids
        ]
        grouped = self.moments_by_bucket(moments)
        moment_ids = {
            str(moment.get("moment_id") or "")
            for moment in moments
            if str(moment.get("moment_id") or "")
        }
        edges = [
            edge
            for edge in legacy_moment_edges_for_recall(self.moment_store.list_edges())
            if str(edge.get("source") or "") in moment_ids
            and str(edge.get("target") or "") in moment_ids
        ]
        edges.extend(self.bucket_edges_as_moment_edges(bucket_edges, grouped))
        graph = MemoryRecallGraph(
            moments=moments,
            grouped_moments=grouped,
            edges=edges,
            scene_native=False,
        )
        return self._cache_graph(graph, signature, bucket_list_id, edge_stamp)

    def _refresh_scene_native(
        self,
        recallable_buckets: list[dict],
        *,
        bucket_list_id: int,
        edge_stamp: tuple[Any, ...],
        store_stamp: tuple[int, int],
    ) -> MemoryRecallGraph:
        assert self.recall_sources is not None
        partition = self.recall_sources.partition(recallable_buckets)
        scene_map = partition.scene_map
        scene_ids = set(scene_map)
        scene_edges = self.scene_edge_store.recall_edges(scene_map)
        legacy_bucket_edges = legacy_edges_for_recall(
            self.memory_edge_store.list_edges(),
            scene_ids=scene_ids,
        )
        bridge_rows = (
            self.legacy_scene_bridge_store.list_edges()
            if self.legacy_scene_bridge_store is not None
            else []
        )
        signature_edges = [*scene_edges, *legacy_bucket_edges, *bridge_rows]
        signature = self.graph_signature(recallable_buckets, signature_edges)
        if (
            signature
            and signature == self._cache_signature
            and self._cache_value is not None
            and store_stamp == self._cache_store_stamp
        ):
            return self._cache_value

        self.moment_store.bulk_upsert(partition.legacy_buckets)
        legacy_bucket_ids = {
            str(bucket.get("id") or "")
            for bucket in partition.legacy_buckets
            if str(bucket.get("id") or "")
        }
        stored_legacy_moments = [
            moment
            for moment in self.moment_store.list_all()
            if str(moment.get("bucket_id") or "") in legacy_bucket_ids
        ]
        legacy_nodes = self.recall_sources.legacy_nodes(stored_legacy_moments)
        scene_nodes = [self.recall_sources.scene_node(scene) for scene in partition.scenes]
        nodes = self.recallable_moments([*scene_nodes, *legacy_nodes])
        grouped = self.moments_by_bucket(nodes)
        node_ids = {
            str(node.get("node_id") or node.get("moment_id") or "")
            for node in nodes
            if str(node.get("node_id") or node.get("moment_id") or "")
        }
        edges = [
            edge
            for edge in legacy_moment_edges_for_recall(self.moment_store.list_edges())
            if str(edge.get("source") or "") in node_ids
            and str(edge.get("target") or "") in node_ids
        ]
        edges.extend(self.bucket_edges_as_moment_edges(legacy_bucket_edges, grouped))
        edges.extend(self.recall_sources.scene_edges_as_recall_edges(scene_edges))

        legacy_moment_map = {
            str(node.get("moment_id") or ""): node
            for node in legacy_nodes
            if str(node.get("moment_id") or "")
        }
        bridge_edges = (
            self.legacy_scene_bridge_store.recall_edges(scene_map, legacy_moment_map)
            if self.legacy_scene_bridge_store is not None
            else []
        )
        edges.extend(bridge_edges)
        nodes = self.recall_sources.decorate_same_event_clusters(nodes, bridge_edges)
        grouped = self.moments_by_bucket(nodes)
        graph = MemoryRecallGraph(
            moments=nodes,
            grouped_moments=grouped,
            edges=edges,
            scene_native=True,
        )
        return self._cache_graph(graph, signature, bucket_list_id, edge_stamp)

    def _cache_graph(
        self,
        graph: MemoryRecallGraph,
        signature: str,
        bucket_list_id: int,
        edge_stamp: tuple[Any, ...],
    ) -> MemoryRecallGraph:
        self._cache_signature = signature
        self._cache_value = graph
        self._cache_bucket_list_id = bucket_list_id
        self._cache_edge_stamp = edge_stamp
        self._cache_store_stamp = self._moment_store_stamp()
        return graph

    async def select_direct(
        self,
        graph: MemoryRecallGraph,
        all_buckets: list[dict],
        *,
        query: str,
        session_id: str,
        search_query: str,
        select_dynamic_moments: Callable[..., Awaitable[tuple]],
    ) -> MemoryRecallResult:
        (
            direct_moments,
            candidate_moments,
            suppressed_moments,
            suppressed_buckets,
            query_planner_debug,
        ) = await select_dynamic_moments(
            query,
            session_id,
            all_buckets,
            graph.grouped_moments,
            all_moments=graph.moments,
            search_query=search_query,
            include_query_planner_debug=True,
        )
        direct_moments = list(direct_moments or [])
        candidate_moments = list(candidate_moments or [])
        suppressed_moments = list(suppressed_moments or [])
        if graph.scene_native:
            direct_moments, same_event_suppressed = self._collapse_same_event_items(
                graph,
                direct_moments,
                record_suppressed=True,
            )
            candidate_moments, _ = self._collapse_same_event_items(
                graph,
                candidate_moments,
                record_suppressed=False,
            )
            suppressed_moments.extend(same_event_suppressed)
        return MemoryRecallResult(
            graph=graph,
            direct_moments=direct_moments,
            candidate_moments=candidate_moments,
            suppressed_moments=suppressed_moments,
            suppressed_buckets=suppressed_buckets,
            query_planner_debug=query_planner_debug,
        )

    @staticmethod
    def _collapse_same_event_items(
        graph: MemoryRecallGraph,
        items: list[dict],
        *,
        record_suppressed: bool,
    ) -> tuple[list[dict], list[dict]]:
        scene_by_cluster: dict[str, dict] = {}
        for node in graph.moments:
            metadata = node.get("metadata", {}) if isinstance(node.get("metadata"), dict) else {}
            cluster = str(node.get("same_event_cluster") or metadata.get("same_event_cluster") or "")
            if cluster and str(node.get("node_kind") or "") == "scene":
                scene_by_cluster[cluster] = node

        collapsed: list[dict] = []
        suppressed: list[dict] = []
        seen_keys: set[str] = set()
        structural_keys = {
            "moment_id",
            "node_id",
            "node_kind",
            "bucket_id",
            "section",
            "text",
            "ordinal",
            "source",
            "source_id",
            "text_hash",
            "metadata",
            "metadata_json",
            "created_at",
            "updated_at",
        }
        for item in items or []:
            metadata = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
            cluster = str(item.get("same_event_cluster") or metadata.get("same_event_cluster") or "")
            chosen = item
            same_event_duplicate: dict | None = None
            scene = scene_by_cluster.get(cluster) if cluster else None
            if scene is not None and str(item.get("node_kind") or "") != "scene":
                chosen = dict(scene)
                for key, value in item.items():
                    if key not in structural_keys:
                        chosen[key] = value
                chosen["same_event_promoted_from"] = str(item.get("moment_id") or "")
                if record_suppressed:
                    same_event_duplicate = dict(item)
                    same_event_duplicate["suppression_reason"] = "same_event_scene_primary"
                    same_event_duplicate["canonical_scene_id"] = cluster

            dedupe_key = f"same_event:{cluster}" if cluster else str(
                chosen.get("node_id") or chosen.get("moment_id") or id(chosen)
            )
            if dedupe_key in seen_keys:
                if same_event_duplicate is not None:
                    suppressed.append(same_event_duplicate)
                continue
            seen_keys.add(dedupe_key)
            if same_event_duplicate is not None:
                suppressed.append(same_event_duplicate)
            collapsed.append(chosen)
        return collapsed, suppressed

    def plan_diffusion(
        self,
        moments: list[dict],
        seed_moments: list[dict],
        query_plan: Any,
        *,
        diffusion_options: Any,
        inject_max_cards: int,
        inject_max_items: int,
        explore_multiplier: int,
    ) -> MemoryDiffusionPlan:
        inject_limit = self._diffusion_inject_limit(
            query_plan,
            inject_max_cards=inject_max_cards,
            inject_max_items=inject_max_items,
        )
        base = max(1, int(getattr(diffusion_options, "top_k", 0) or 0))
        explore_limit = max(
            base,
            inject_limit,
            min(24, base * max(1, int(explore_multiplier))),
        )
        return MemoryDiffusionPlan(
            moment_map=self._moment_diffusion_map(moments),
            seed_scores=self._seed_scores_for_moments(seed_moments),
            explore_options=replace(
                diffusion_options,
                top_k=max(
                    int(getattr(diffusion_options, "top_k", 0) or 0),
                    explore_limit,
                ),
            ),
            explore_limit=explore_limit,
            inject_limit=inject_limit,
            same_event_seed_clusters=frozenset(
                cluster
                for moment in seed_moments
                if (
                    cluster := str(
                        moment.get("same_event_cluster")
                        or (
                            moment.get("metadata", {}).get("same_event_cluster")
                            if isinstance(moment.get("metadata"), dict)
                            else ""
                        )
                        or ""
                    ).strip()
                )
            ),
        )

    @staticmethod
    def new_candidate_pool(
        *,
        excluded_same_event_clusters: frozenset[str] = frozenset(),
    ) -> MemoryDiffusionCandidatePool:
        return MemoryDiffusionCandidatePool(
            excluded_same_event_clusters=excluded_same_event_clusters
        )

    @staticmethod
    def _diffusion_inject_limit(
        query_plan: Any,
        *,
        inject_max_cards: int,
        inject_max_items: int,
    ) -> int:
        if inject_max_cards <= 0:
            return 0
        if getattr(query_plan, "wants_body_chain", False):
            return max(0, min(5, inject_max_cards * 3))
        return max(0, min(2, inject_max_cards, inject_max_items))

    @staticmethod
    def _moment_diffusion_map(moments: list[dict]) -> dict[str, dict]:
        mapped: dict[str, dict] = {}
        for moment in moments:
            moment_id = moment.get("moment_id")
            if not moment_id:
                continue
            item = dict(moment)
            meta = dict(item.get("metadata", {}) or {})
            meta["importance"] = meta.get("bucket_importance", 5)
            meta["type"] = meta.get("bucket_type", "")
            meta["anchor"] = meta.get("bucket_anchor", False)
            meta["pinned"] = meta.get("bucket_pinned", False)
            meta["protected"] = meta.get("bucket_protected", False)
            meta["name"] = meta.get("bucket_name", "")
            meta["resolved"] = meta.get("bucket_resolved", False)
            meta["digested"] = meta.get("bucket_digested", False)
            item["metadata"] = meta
            mapped[str(moment_id)] = item
        return mapped

    @staticmethod
    def _seed_scores_for_moments(moments: list[dict]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for moment in moments:
            moment_id = str(moment.get("moment_id") or "")
            if not moment_id:
                continue
            scores[moment_id] = max(
                0.15,
                min(1.0, _safe_float(moment.get("score", 0.75), 0.75)),
            )
        return scores

    def _prune_excluded_moment_index(self, all_buckets: list[dict]) -> None:
        for bucket in all_buckets or []:
            if not self.is_excluded_bucket(bucket):
                continue
            bucket_id = str(bucket.get("id") or "").strip()
            if bucket_id:
                self.moment_store.delete_bucket(bucket_id)

    def _prune_native_scene_moment_index(self, all_buckets: list[dict]) -> None:
        if self.recall_sources is None:
            return
        partition = self.recall_sources.partition(
            [bucket for bucket in all_buckets or [] if not self.is_excluded_bucket(bucket)]
        )
        for scene in partition.scenes:
            scene_id = str(scene.get("id") or "").strip()
            if not scene_id:
                continue
            sync_alias_projection = getattr(
                self.moment_store,
                "sync_alias_projection",
                None,
            )
            if callable(sync_alias_projection):
                sync_alias_projection(
                    scene,
                    [self.recall_sources.scene_node(scene)],
                )
            elif self.moment_store.list_for_bucket(scene_id, limit=1):
                # Compatibility for old/injected stores that do not expose the
                # source-neutral alias index yet.
                self.moment_store.delete_bucket(scene_id)

    def _edge_store_stamp(
        self,
    ) -> tuple[Any, ...]:
        legacy_path = str(getattr(self.memory_edge_store, "path", "") or "")
        scene_path = str(getattr(self.scene_edge_store, "db_path", "") or "")
        # Scene review writes use SQLite WAL. Watching the main db alone can
        # miss a freshly accepted edge until a checkpoint happens.
        scene_stamp = (
            self._path_stamp(scene_path),
            self._path_stamp(scene_path + "-wal" if scene_path else ""),
        )
        bridge_path = str(getattr(self.legacy_scene_bridge_store, "db_path", "") or "")
        bridge_stamp = (
            self._path_stamp(bridge_path),
            self._path_stamp(bridge_path + "-wal" if bridge_path else ""),
        )
        return self._path_stamp(legacy_path), scene_stamp, bridge_stamp

    def _moment_store_stamp(self) -> tuple[int, int]:
        path = str(getattr(self.moment_store, "db_path", "") or "")
        return self._path_stamp(path)

    @staticmethod
    def _path_stamp(path: str) -> tuple[int, int]:
        if not path:
            return (0, 0)
        try:
            stat = os.stat(path)
        except OSError:
            return (0, 0)
        return (int(getattr(stat, "st_mtime_ns", 0)), int(stat.st_size))


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)
