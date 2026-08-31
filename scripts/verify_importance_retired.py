from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bucket_manager import BucketManager
from decay_engine import DecayEngine


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="importance-retired-") as temp_dir:
        manager = BucketManager(
            {
                "buckets_dir": str(Path(temp_dir) / "buckets"),
                "scoring_weights": {
                    "topic_relevance": 4.0,
                    "time_proximity": 1.5,
                    "importance": 99.0,
                },
            }
        )
        assert manager.w_importance == 0.0

        bucket_id = await manager.create(
            content="一条用于验证旧字段退出的记忆。",
            name="字段退出验证",
            importance=10,
        )
        created = await manager.get(bucket_id)
        assert created is not None
        assert "importance" not in created["metadata"], created

        assert await manager.update(bucket_id, importance=1)
        updated = await manager.get(bucket_id)
        assert updated is not None
        assert "importance" not in updated["metadata"], updated

        decay = DecayEngine({"decay": {"lambda": 0.05}}, manager)
        shared = {
            "type": "dynamic",
            "activation_count": 2,
            "last_active": "2026-08-31T00:00:00",
        }
        assert decay.calculate_score({**shared, "importance": 1}) == decay.calculate_score(
            {**shared, "importance": 10}
        )

    print("IMPORTANCE_RETIRED_OK")


if __name__ == "__main__":
    asyncio.run(main())
