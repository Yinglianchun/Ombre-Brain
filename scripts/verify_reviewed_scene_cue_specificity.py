from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_recall_entry_evidence import build_service


service = build_service()
reviewed_scene = {
    "id": "scene-reviewed-cue",
    "content": "正文不参与这组词法证据测试。",
    "metadata": {
        "memory_value_source": "authored_scene",
        "scene_cues": ["不想模板化", "窗口连续性与关系归属"],
        "scene_cues_reviewed_at": "2026-08-03T00:00:00Z",
    },
}

# A single shared two-character fragment is not trusted direct evidence.
assert service._bucket_authored_cue_terms("不想吃药", reviewed_scene) == []

# Full cue containment remains trusted.
assert service._bucket_authored_cue_terms("我还是不想模板化", reviewed_scene) == [
    "不想模板化"
]

# A compound paraphrase with enough independent specific overlap remains trusted.
assert service._bucket_authored_cue_terms(
    "换窗口后关系的连续性和归属还在吗",
    reviewed_scene,
) == ["窗口连续性与关系归属"]

print("reviewed Scene cue specificity verification passed")
