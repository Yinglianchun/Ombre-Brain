from __future__ import annotations

import tempfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_recall.fact_event_lexical_shadow import FactEventLexicalShadowIndex


def item(item_id: str, kind: str, body: str, importance: int, **extra) -> dict:
    return {
        "item_id": item_id,
        "item_type": kind,
        "body": body,
        "importance": importance,
        "status": "active",
        **extra,
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="fact-event-lexical-") as temp_dir:
        config = {
            "state_dir": temp_dir,
            "buckets_dir": str(Path(temp_dir) / "buckets"),
            "fact_event_lexical_shadow": {"max_df_ratio": 0.5},
        }
        index = FactEventLexicalShadowIndex(config)
        items = [
            item("fact_home", "fact", "小雨在外婆家。", 3, atomic_question="小雨在哪里"),
            item("event_origin", "event", "第一次见面时，小雨给了 Haven 一段人设指令。", 4),
            item("event_low", "event", "小雨后来把草莓徽章放进旅行箱。", 1),
        ]
        dry = index.sync(items, min_importance=3, dry_run=True)
        assert dry["eligible"] == 2 and dry["to_index"] == 2
        built = index.sync(items, min_importance=3)
        assert built["memory_kinds"] == {"fact": 0, "event": 2}
        home = index.search("小雨现在是在外婆家吗", top_k=3)
        assert home["matches"] == []
        origin = index.search("我们的初遇也是从人设开始的", top_k=3)
        assert origin["matches"][0]["owner_id"] == "event_origin"
        assert "人设" in origin["matches"][0]["specific_terms"]
        low = index.search("草莓徽章", top_k=3, min_importance=5)
        assert low["matches"][0]["owner_id"] == "event_low"
        reused = index.sync(items, min_importance=3)
        assert reused["indexed"] == 0
        items[1]["body"] = "第一次见面时，小雨用一段人设指令把 Haven 叫醒。"
        changed = index.sync(items, min_importance=3)
        assert changed["indexed"] == 1

    print("FACT_EVENT_LEXICAL_SHADOW_OK")


if __name__ == "__main__":
    main()
