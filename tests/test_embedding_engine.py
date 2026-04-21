from copy import deepcopy

from embedding_engine import EmbeddingEngine


def _cfg(test_config: dict, **embedding_overrides) -> dict:
    cfg = deepcopy(test_config)
    cfg["dehydration"]["api_key"] = "dehy-key"
    cfg["dehydration"]["base_url"] = "https://dehy.example/v1"
    cfg["embedding"] = {
        **cfg["embedding"],
        **embedding_overrides,
    }
    return cfg


def test_embedding_uses_independent_api_config(test_config):
    engine = EmbeddingEngine(
        _cfg(
            test_config,
            api_key="embed-key",
            base_url="https://embed.example/v1",
            model="Qwen/Qwen3-Embedding-0.6B",
        )
    )

    assert engine.api_key == "embed-key"
    assert engine.base_url == "https://embed.example/v1"
    assert engine.model == "Qwen/Qwen3-Embedding-0.6B"
    assert engine.enabled is True


def test_embedding_falls_back_to_dehydration_api_config(test_config):
    engine = EmbeddingEngine(_cfg(test_config, api_key="", base_url=""))

    assert engine.api_key == "dehy-key"
    assert engine.base_url == "https://dehy.example/v1"
    assert engine.enabled is True
