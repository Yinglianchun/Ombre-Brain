from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway import GatewayService
from gateway_state import GatewayStateStore


with tempfile.TemporaryDirectory() as directory:
    store = GatewayStateStore(str(Path(directory) / "gateway-state.db"))
    store.record_success("session-a", ["seen-card"])
    store.record_success("session-b", ["other-card"])

    assert store.get_session_bucket_ids("session-a") == {"seen-card"}
    assert store.get_session_bucket_ids("session-b") == {"other-card"}
    assert store.arc_material_menu_was_injected("session-a", "work:spy") is False
    assert store.record_arc_material_menu_injection(
        "session-a",
        "work:spy",
        menu_fingerprint="menu-v1",
    ) is True
    assert store.arc_material_menu_was_injected("session-a", "work:spy") is True
    assert store.arc_material_menu_was_injected("session-b", "work:spy") is False
    assert store.record_arc_material_menu_injection("session-a", "work:spy") is False

    service = GatewayService.__new__(GatewayService)
    service.state_store = store

    # These are already the two best cards. Filtering happens after selection,
    # so removing the repeat must leave one card instead of pulling in a third.
    selected = [{"id": "seen-card"}, {"id": "new-card"}]
    kept, suppressed = service._filter_cooldown_selected_buckets("session-a", selected)
    assert [item["id"] for item in kept] == ["new-card"]
    assert len(suppressed) == 1
    assert suppressed[0]["admission_reason"] == "session_already_injected"
    assert service._bucket_cooldown_active("session-a", "seen-card") is True
    assert service._bucket_cooldown_active("session-a", "new-card") is False
    assert service._bucket_cooldown_active("session-b", "seen-card") is False

    async def candidate_items(*_args, **_kwargs):
        return [
            {"bucket": {"id": "seen-card"}},
            {"bucket": {"id": "new-card"}},
            {"bucket": {"id": "unrelated-backfill"}},
        ], []

    service.inject_max_cards = 2
    service.semantic_rescue_enabled = False
    service._query_planner_debug_base = lambda _query: {"timing_ms": {}}
    service._add_timing_ms = lambda *_args, **_kwargs: None
    service._dynamic_bucket_candidate_items = candidate_items
    service._merge_exact_anchor_debug = lambda *_args, **_kwargs: None
    service._pick_dynamic_cards = lambda items, **_kwargs: list(items[:2])
    service._bucket_with_recall_signal = lambda item: dict(item["bucket"])
    service._route_profile_fact_buckets = lambda _query, buckets, _all: (buckets, [])
    service._route_year_ring_parent_buckets = lambda _query, buckets, _all: (buckets, [])

    selected_after_dedupe, _suppressed = asyncio.run(
        service._select_dynamic_buckets("query", "session-a", [])
    )
    assert [item["id"] for item in selected_after_dedupe] == ["new-card"]
    assert "unrelated-backfill" not in {item["id"] for item in selected_after_dedupe}

print("session recall cooldown verification passed")
