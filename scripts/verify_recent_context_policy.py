from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway import GatewayService


def main() -> int:
    service = GatewayService.__new__(GatewayService)
    service.recent_budget = 800
    service.head_recent_hours = 72
    service._query_requests_recent_context = lambda query: query == "刚刚发生了什么"

    assert service._should_inject_recent_context(
        "new-session",
        "普通消息",
    ) is False
    assert service._should_inject_recent_context(
        "active-session",
        "普通消息",
        has_reliable_dynamic_context=True,
    ) is False
    assert service._recent_context_reason(
        "new-session",
        "普通消息",
        has_reliable_dynamic_context=True,
    ) == ""
    assert service._should_inject_recent_context(
        "any-session",
        "刚刚发生了什么",
    ) is True
    assert service._recent_context_reason(
        "any-session",
        "刚刚发生了什么",
    ) == "explicit_recent_query"

    print("recent context policy verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
