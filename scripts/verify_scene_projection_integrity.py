"""Verify Serein sees effective Scene status and dirty lifecycle metadata."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import server


dirty_projection = server._scene_projection_status({
    "metadata": {"type": "archived", "active": True},
})
assert dirty_projection == {
    "status": "已沉底",
    "storage_status": "archived",
    "type": "archived",
    "active": True,
    "scene_status": "",
    "status_consistent": False,
}

active_false_projection = server._scene_projection_status({
    "metadata": {"type": "dynamic", "active": False, "scene_status": ""},
})
assert active_false_projection["status"] == "已沉底"
assert active_false_projection["storage_status"] == "archived"
assert active_false_projection["status_consistent"] is False

clean_projection = server._scene_projection_status({
    "metadata": {"type": "dynamic", "active": True, "scene_status": "active"},
})
assert clean_projection["status"] == "可浮现"
assert clean_projection["status_consistent"] is True

print("SCENE_PROJECTION_INTEGRITY_OK")
