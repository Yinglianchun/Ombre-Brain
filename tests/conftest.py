# ============================================================
# Shared test fixtures — isolated temp environment for all tests
# 共享测试 fixtures —— 为所有测试提供隔离的临时环境
#
# IMPORTANT: All tests run against a temp directory.
# Your real /data or local buckets are NEVER touched.
# 重要：所有测试在临时目录运行，绝不触碰真实记忆数据。
# ============================================================

import os
import sys
import math
import pytest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path

# Ensure project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def test_config(tmp_path):
    """Minimal config pointing to a temp directory."""
    buckets_dir = str(tmp_path / "buckets")
    os.makedirs(os.path.join(buckets_dir, "permanent"), exist_ok=True)
    os.makedirs(os.path.join(buckets_dir, "dynamic"), exist_ok=True)
    os.makedirs(os.path.join(buckets_dir, "archive"), exist_ok=True)
    os.makedirs(os.path.join(buckets_dir, "feel"), exist_ok=True)

    return {
        "buckets_dir": buckets_dir,
        "matching": {"fuzzy_threshold": 50, "max_results": 10},
        "wikilink": {"enabled": False},
        "scoring_weights": {
            "topic_relevance": 4.0,
            "emotion_resonance": 2.0,
            "time_proximity": 2.5,
            "importance": 1.0,
            "content_weight": 3.0,
        },
        "decay": {
            "lambda": 0.05,
            "threshold": 0.3,
            "check_interval_hours": 24,
            "emotion_weights": {"base": 1.0, "arousal_boost": 0.8},
        },
        "dehydration": {
            "api_key": os.environ.get("OMBRE_API_KEY", ""),
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "model": "gemini-2.5-flash-lite",
        },
        "embedding": {
            "api_key": os.environ.get("OMBRE_API_KEY", ""),
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "model": "gemini-embedding-001",
        },
        "gateway": {
            "host": "127.0.0.1",
            "port": 8010,
            "upstream_base_url": "https://upstream.example/v1",
            "upstream_default_model": "gateway-default-model",
            "head_recent_hours": 72,
            "dynamic_top_k": 10,
            "inject_max_cards": 2,
            "skip_recent_rounds": 5,
            "cooldown_hours": 48,
            "cooldown_floor": 0.3,
            "inject_total_budget": 1200,
            "core_memory_budget": 500,
            "recent_context_budget": 300,
            "recalled_memory_budget": 400,
            "semantic_weight": 0.45,
            "keyword_weight": 0.35,
            "importance_weight": 0.10,
            "freshness_weight": 0.10,
            "first_card_min_score": 0.55,
            "second_card_min_score": 0.50,
            "second_card_relative_score": 0.85,
        },
        "persona": {
            "enabled": True,
            "profile_id": "haven_xiaoyu",
            "mode": "llm",
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat",
            "api_key": "",
            "temperature": 0.1,
            "max_tokens": 500,
            "global_decay_hours": 168,
            "session_mood_half_life_minutes": 90,
            "max_personality_delta": 0.01,
            "max_relationship_delta": 0.03,
            "max_affect_delta": 0.18,
            "initial_personality": {
                "openness": 0.56,
                "conscientiousness": 0.50,
                "extraversion": 0.44,
                "agreeableness": 0.66,
                "neuroticism": 0.36,
            },
            "initial_relationship": {
                "affinity": 0.86,
                "dominance": 0.38,
                "defensiveness": 0.12,
                "trust": 0.82,
            },
            "initial_affect": {
                "valence": 0.56,
                "arousal": 0.34,
                "mood_label": "warm_neutral",
                "session_defensiveness": 0.12,
            },
        },
    }


@pytest.fixture
def bucket_mgr(test_config):
    from bucket_manager import BucketManager
    return BucketManager(test_config)


@pytest.fixture
def decay_eng(test_config, bucket_mgr):
    from decay_engine import DecayEngine
    return DecayEngine(test_config, bucket_mgr)
