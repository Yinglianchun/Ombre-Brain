#!/usr/bin/env python3
"""Plan or incrementally synchronize the Scene/Event observed-entity sidecar."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bucket_manager import BucketManager
from fact_events import FactEventStore
from memory_recall.observed_entities import ObservedEntityShadowIndex
from narrative_rolls import NarrativeRollStore
from scene_evidence import SceneEvidenceStore
from utils import bucket_text_for_embedding, load_config


def active_scene_ids(buckets: list[dict[str, Any]]) -> list[str]:
    output: list[str] = []
    for bucket in buckets:
        metadata = bucket.get("metadata") if isinstance(bucket.get("metadata"), dict) else {}
        authored = (
            str(metadata.get("memory_value_source") or "") == "authored_scene"
            or str(metadata.get("object_kind") or "").strip().lower() == "scene"
        )
        archived = (
            str(metadata.get("type") or "").strip().lower() == "archived"
            or str(metadata.get("scene_status") or "").strip().lower() == "archived"
            or metadata.get("active") is False
        )
        scene_id = str(bucket.get("id") or "").strip()
        content = bucket_text_for_embedding(bucket)
        if authored and not archived and scene_id and content.strip():
            output.append(scene_id)
    return list(dict.fromkeys(output))


def active_events(store: FactEventStore) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = store.list(
            item_type="event",
            status="active",
            limit=500,
            offset=offset,
            include_sources=True,
        )
        items = list(page.get("items") or [])
        output.extend(items)
        offset += len(items)
        if not items or offset >= int(page.get("count") or 0):
            return output


def recall_scope_profiles(
    narrative_store: NarrativeRollStore,
    fact_event_store: FactEventStore,
) -> list[dict[str, Any]]:
    profiles = narrative_store.recall_scope_profiles()
    for profile in profiles:
        arc_key = str(profile.get("arc_key") or "").strip()
        if not arc_key:
            continue
        members = list(profile.get("members") or [])
        for link in fact_event_store.arc_event_links(arc_key):
            event_id = str(link.get("event_id") or "").strip()
            member = {"owner_kind": "event", "owner_id": event_id}
            if event_id and member not in members:
                members.append(member)
        profile["members"] = members
    return profiles


async def run(args: argparse.Namespace) -> int:
    config = load_config(args.config or None)
    buckets = await BucketManager(config).list_all(include_archive=True)
    scene_ids = active_scene_ids(buckets)
    scene_refs = SceneEvidenceStore(config, create=False).list_active_for_scenes(scene_ids)
    fact_event_store = FactEventStore(config, create=False)
    events = active_events(fact_event_store)
    owners = [
        *(
            {
                "owner_kind": "scene",
                "owner_id": scene_id,
                "source_refs": scene_refs.get(scene_id) or [],
            }
            for scene_id in scene_ids
        ),
        *(
            {
                "owner_kind": "event",
                "owner_id": str(event.get("item_id") or ""),
                "source_refs": event.get("source_refs") or [],
            }
            for event in events
            if str(event.get("item_id") or "").strip()
        ),
    ]
    profiles = recall_scope_profiles(NarrativeRollStore(config), fact_event_store)
    index = ObservedEntityShadowIndex(config)
    plan = index.sync(owners=owners, arc_profiles=profiles, dry_run=True)
    print(
        json.dumps(
            {
                "mode": "plan",
                **plan,
                "sidecar_path": index.db_path,
                "source_text_queryable": False,
                "writes_performed": [],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not args.apply:
        return 0

    result = index.sync(owners=owners, arc_profiles=profiles)
    print(
        json.dumps(
            {
                "mode": "apply",
                **result,
                "sidecar_path": index.db_path,
                "writes_performed": [index.db_path],
                "live_injection_enabled": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plan or incrementally synchronize observed Scene/Event entities."
    )
    parser.add_argument("--config", default="")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write only the rebuildable observed_entity_shadow.sqlite sidecar.",
    )
    raise SystemExit(asyncio.run(run(parser.parse_args())))
