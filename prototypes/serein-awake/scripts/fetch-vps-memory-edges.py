from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / ".private" / "ombre-scene-edges-vps.json"
VPS_HOST = os.environ.get("OMBRE_VPS_HOST", "8.136.154.242")

REMOTE_SCRIPT = r"""
import asyncio
import json

from bucket_manager import BucketManager
from scene_linker import SceneEdgeStore
from utils import load_config

config = load_config()
buckets = asyncio.run(BucketManager(config).list_all(include_archive=True))
scene_map = {
    str(bucket.get("id") or ""): bucket
    for bucket in buckets
    if bucket.get("id")
}
edges = SceneEdgeStore(config, create=False).recall_edges(scene_map)
print(json.dumps([
    {
        "edgeId": edge.get("edge_id"),
        "sourceId": edge.get("source"),
        "targetId": edge.get("target"),
        "relationType": edge.get("relation_type"),
        "directionality": edge.get("directionality"),
        "confidence": edge.get("confidence"),
    }
    for edge in edges
], ensure_ascii=False))
"""


def fetch_edges() -> None:
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "KexAlgorithms=curve25519-sha256,curve25519-sha256@libssh.org",
        f"root@{VPS_HOST}",
        (
            "cd /opt/Ombre-Brain && "
            "docker compose -f compose.hk.yml exec -T ombre-brain python -"
        ),
    ]
    result = subprocess.run(
        command,
        input=REMOTE_SCRIPT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr.strip() or "Failed to read VPS Scene edges.")

    output_lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not output_lines:
        raise SystemExit("The VPS returned no Scene-edge projection.")
    edges = json.loads(output_lines[-1])
    if not isinstance(edges, list):
        raise SystemExit("The VPS Scene-edge projection has an invalid shape.")

    payload = {
        "source": {
            "host": VPS_HOST,
            "state": "/srv/ombre-brain/state/scene_edge_proposals.sqlite",
            "filter": "SceneEdgeStore.recall_edges",
        },
        "fetchedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "edges": edges,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Fetched {len(edges)} validated Scene edges to {OUTPUT_PATH}")


if __name__ == "__main__":
    fetch_edges()
