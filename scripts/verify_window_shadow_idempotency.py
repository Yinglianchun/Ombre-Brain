from __future__ import annotations

import tempfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from window_shadows import WindowShadowStore


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="ombre-shadow-idempotency-") as tmp:
        root = Path(tmp)
        store = WindowShadowStore(
            {
                "buckets_dir": str(root / "buckets"),
                "state_dir": str(root / "state"),
                "identity": {"user_display_name": "小雨"},
            }
        )
        request_key = "haven_bridge:test-window"
        first, created = store.write(
            "## 这一窗之后，什么留在了我身上\n我记得第一版。\n\n"
            "## 给下个窗口的我\n" + "我会沿着这一扇窗继续。" * 30,
            session_id="conversation-one",
            idempotency_key=request_key,
        )
        if not created:
            raise AssertionError("first idempotent Shadow was not created")
        replay, replay_created = store.write(
            "## 这一窗之后，什么留在了我身上\n我在重试里写了不同文字。\n\n"
            "## 给下个窗口的我\n" + "这份文字不应覆盖第一次成功写入。" * 25,
            session_id="conversation-two",
            idempotency_key=request_key,
        )
        if replay_created:
            raise AssertionError("idempotent replay created a second Shadow")
        if replay.get("window_id") != first.get("window_id"):
            raise AssertionError("idempotent replay returned a different window_id")
        if replay.get("content") != first.get("content"):
            raise AssertionError("idempotent replay overwrote the first authored Shadow")
        if store.get_by_idempotency_key(request_key).get("window_id") != first.get("window_id"):
            raise AssertionError("idempotency lookup did not resolve the original Shadow")
        if store.stats().get("count") != 1:
            raise AssertionError("idempotent replay left more than one Shadow")
    print("window Shadow idempotency checks passed")


if __name__ == "__main__":
    main()
