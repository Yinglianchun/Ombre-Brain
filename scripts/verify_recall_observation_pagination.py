"""Targeted stable-cursor checks for Gateway recall-observation pages."""

from __future__ import annotations

import asyncio
import json
import tempfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway import GatewayService
from gateway_state import GatewayStateStore


with tempfile.TemporaryDirectory() as directory:
    store = GatewayStateStore(str(Path(directory) / "gateway-state.db"))
    ids = [
        store.record_injection_debug(
            "session-a" if index % 2 else "session-b",
            index,
            {"query": f"query-{index}"},
            max_entries=20,
        )
        for index in range(1, 7)
    ]

    first_page = store.list_injection_debug(limit=3, include_context=False)
    assert [item["id"] for item in first_page] == list(reversed(ids[-3:]))

    cursor = first_page[-1]["id"]
    second_page = store.list_injection_debug(
        limit=3,
        include_context=False,
        before_id=cursor,
    )
    assert [item["id"] for item in second_page] == list(reversed(ids[:3]))
    assert set(item["id"] for item in first_page).isdisjoint(
        item["id"] for item in second_page
    )

    reviewed = store.list_injection_debug(
        limit=2,
        include_context=False,
        ids=[ids[0], ids[-1], ids[-1], -1],
    )
    assert [item["id"] for item in reviewed] == [ids[-1], ids[0]]

    assert store.update_injection_debug_payload(
        ids[-1],
        {"query": "query-6", "typed_event_scene_observation": {"status": "would_inject"}},
    ) is True
    updated = store.list_injection_debug(limit=1, include_context=False, ids=[ids[-1]])
    assert updated[0]["payload"]["typed_event_scene_observation"]["status"] == "would_inject"

    session_page = store.list_injection_debug(
        session_id="session-a",
        limit=10,
        include_context=False,
        before_id=ids[-1] + 1,
    )
    assert session_page
    assert all(item["session_id"] == "session-a" for item in session_page)

    service = GatewayService.__new__(GatewayService)
    service.state_store = store
    service._authorize = lambda _header: None

    class Request:
        headers = {"Authorization": "Bearer test"}
        query_params = {
            "limit": "2",
            "before_id": str(ids[-1] + 1),
            "review_ids": str(ids[0]),
            "include_context": "0",
        }

    response = asyncio.run(service.handle_injection_debug(Request()))
    payload = json.loads(response.body)
    assert [item["id"] for item in payload["items"]] == [ids[-1], ids[-2]]
    assert payload["has_more"] is True
    assert payload["next_before_id"] == ids[-2]
    assert [item["id"] for item in payload["reviewed_items"]] == [ids[0]]

print("recall observation pagination verification passed")
