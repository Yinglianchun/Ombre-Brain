import asyncio
import json
from copy import deepcopy
from datetime import datetime, timedelta

import frontmatter
import httpx
import pytest
from starlette.testclient import TestClient

from gateway import GatewayService, create_gateway_app
from gateway_state import GatewayStateStore


class DummyDehydrator:
    async def dehydrate(self, content: str, metadata: dict | None = None) -> str:
        name = (metadata or {}).get("name", "未命名")
        compact = " ".join((content or "").strip().split())
        return f"{name}: {compact[:80]}"


class DummyEmbeddingEngine:
    def __init__(self, results: list[tuple[str, float]] | None = None, enabled: bool = True):
        self.results = results or []
        self.enabled = enabled

    async def search_similar(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        return self.results[:top_k]


class DummyPersonaEngine:
    enabled = True
    profile_id = "haven_xiaoyu"
    mode = "test"
    model = "dummy-persona"
    api_key = "dummy"

    async def update_from_user_message(self, session_id: str, user_message: str) -> dict:
        return {
            "personality": {
                "openness": 0.56,
                "conscientiousness": 0.50,
                "extraversion": 0.44,
                "agreeableness": 0.66,
                "neuroticism": 0.36,
            },
            "affect": {"valence": 0.62, "arousal": 0.40, "mood_label": "warm_attentive"},
            "relationship": {
                "affinity": 0.86,
                "dominance": 0.38,
                "defensiveness": 0.12,
                "trust": 0.82,
            },
            "reply_guidance": "Be warm and steady.",
        }

    def format_state_block(self, state: dict) -> str:
        return (
            "Current Inner State\n"
            "Personality: openness=0.560, conscientiousness=0.500, extraversion=0.440, "
            "agreeableness=0.660, neuroticism=0.360\n"
            "Affect: valence=0.620, arousal=0.400, mood_label=warm_attentive\n"
            "Relationship: affinity=0.860, dominance=0.380, defensiveness=0.120, trust=0.820\n"
            "Reply Guidance: Be warm and steady."
        )


class RecordingPersonaEngine(DummyPersonaEngine):
    def __init__(self):
        self.calls = []

    async def update_from_user_message(self, session_id: str, user_message: str) -> dict:
        self.calls.append({"session_id": session_id, "user_message": user_message})
        return await super().update_from_user_message(session_id, user_message)


def _run(coro):
    return asyncio.run(coro)


def _set_bucket_times(bucket_mgr, bucket_id: str, *, hours_ago: float, **extra_meta) -> None:
    file_path = bucket_mgr._find_bucket_file(bucket_id)
    post = frontmatter.load(file_path)
    ts = (datetime.now() - timedelta(hours=hours_ago)).isoformat(timespec="seconds")
    post["created"] = ts
    post["last_active"] = ts
    for key, value in extra_meta.items():
        post[key] = value
    with open(file_path, "w", encoding="utf-8") as fh:
        fh.write(frontmatter.dumps(post))


def _create_bucket(
    bucket_mgr,
    *,
    content: str,
    name: str,
    hours_ago: float,
    tags: list[str] | None = None,
    importance: int = 8,
    domain: list[str] | None = None,
    bucket_type: str = "dynamic",
    pinned: bool = False,
    protected: bool = False,
    resolved: bool = False,
) -> str:
    bucket_id = _run(
        bucket_mgr.create(
            content=content,
            tags=tags or [],
            importance=importance,
            domain=domain or ["日常"],
            valence=0.7,
            arousal=0.4,
            bucket_type=bucket_type,
            name=name,
            pinned=pinned,
            protected=protected,
        )
    )
    _set_bucket_times(bucket_mgr, bucket_id, hours_ago=hours_ago, resolved=resolved)
    return bucket_id


def _build_service(
    monkeypatch,
    config: dict,
    bucket_mgr,
    *,
    embedding_results: list[tuple[str, float]] | None = None,
):
    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", "gateway-secret")
    monkeypatch.setenv("OMBRE_GATEWAY_UPSTREAM_API_KEY", "upstream-secret")

    captured = []

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        captured.append(
            {
                "json": json.loads(request.content.decode("utf-8")),
                "auth": request.headers.get("Authorization"),
            }
        )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    state_store = GatewayStateStore(f"{config['buckets_dir']}\\gateway_state.db")
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream_handler), timeout=10.0)
    service = GatewayService(
        config=config,
        bucket_mgr=bucket_mgr,
        dehydrator=DummyDehydrator(),
        embedding_engine=DummyEmbeddingEngine(embedding_results, enabled=True),
        state_store=state_store,
        persona_engine=DummyPersonaEngine(),
        http_client=http_client,
    )
    app = create_gateway_app(config=config, service=service)
    return app, service, state_store, captured


def _gateway_config(test_config: dict, **overrides) -> dict:
    cfg = deepcopy(test_config)
    cfg["gateway"] = {**cfg["gateway"], **overrides}
    return cfg


def test_gateway_state_store_cooldown_curve(tmp_path):
    store = GatewayStateStore(str(tmp_path / "gateway_state.db"))
    origin = datetime(2026, 4, 20, 12, 0, 0)
    store.record_success("sess-a", ["bucket-a"], completed_at=origin)

    assert store.get_recent_bucket_ids("sess-a", 5) == {"bucket-a"}
    assert store.get_cooldown_multiplier("sess-a", "bucket-a", 48, 0.3, now=origin) == pytest.approx(0.3)
    assert store.get_cooldown_multiplier(
        "sess-a", "bucket-a", 48, 0.3, now=origin + timedelta(hours=12)
    ) == pytest.approx(0.475, rel=1e-3)
    assert store.get_cooldown_multiplier(
        "sess-a", "bucket-a", 48, 0.3, now=origin + timedelta(hours=24)
    ) == pytest.approx(0.65, rel=1e-3)
    assert store.get_cooldown_multiplier(
        "sess-a", "bucket-a", 48, 0.3, now=origin + timedelta(hours=48)
    ) == pytest.approx(1.0)


def test_gateway_requires_session_id(monkeypatch, test_config, bucket_mgr):
    app, service, _, _ = _build_service(monkeypatch, _gateway_config(test_config), bucket_mgr)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer gateway-secret"},
            json={"messages": [{"role": "user", "content": "你好"}]},
        )
    assert response.status_code == 400
    assert "X-Ombre-Session-Id" in response.text


def test_gateway_streams_when_client_requires_stream(monkeypatch, test_config, bucket_mgr):
    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", "gateway-secret")
    monkeypatch.setenv("OMBRE_GATEWAY_UPSTREAM_API_KEY", "upstream-secret")

    captured = []

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n',
        )

    state_store = GatewayStateStore(f"{test_config['buckets_dir']}\\gateway_state.db")
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream_handler), timeout=10.0)
    service = GatewayService(
        config=_gateway_config(test_config),
        bucket_mgr=bucket_mgr,
        dehydrator=DummyDehydrator(),
        embedding_engine=DummyEmbeddingEngine(enabled=False),
        state_store=state_store,
        persona_engine=DummyPersonaEngine(),
        http_client=http_client,
    )
    app = create_gateway_app(config=test_config, service=service)

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-stream",
            },
            json={"messages": [{"role": "user", "content": "你好"}], "stream": True},
        ) as response:
            body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert 'data: {"choices"' in body
    assert "data: [DONE]" in body
    assert captured[0]["stream"] is True
    assert state_store.get_recent_bucket_ids("sess-stream", 5) == set()


def test_gateway_streams_tool_call_deltas(monkeypatch, test_config, bucket_mgr):
    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", "gateway-secret")
    monkeypatch.setenv("OMBRE_GATEWAY_UPSTREAM_API_KEY", "upstream-secret")

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
                b'"type":"function","function":{"name":"read_diary","arguments":"{}"}}]}}]}\n\n'
                b"data: [DONE]\n\n"
            ),
        )

    state_store = GatewayStateStore(f"{test_config['buckets_dir']}\\gateway_state.db")
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream_handler), timeout=10.0)
    service = GatewayService(
        config=_gateway_config(test_config),
        bucket_mgr=bucket_mgr,
        dehydrator=DummyDehydrator(),
        embedding_engine=DummyEmbeddingEngine(enabled=False),
        state_store=state_store,
        persona_engine=DummyPersonaEngine(),
        http_client=http_client,
    )
    app = create_gateway_app(config=test_config, service=service)

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-stream-tools",
            },
            json={"messages": [{"role": "user", "content": "查今天的日记"}], "stream": True},
        ) as response:
            body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert '"tool_calls"' in body
    assert '"read_diary"' in body
    assert "data: [DONE]" in body


def test_gateway_lists_configured_models(monkeypatch, test_config, bucket_mgr):
    app, _, _, _ = _build_service(
        monkeypatch,
        _gateway_config(
            test_config,
            upstream_models=["qwen3.5-plus", "qwen3.5-max"],
            upstream_default_model="qwen3.5-plus",
        ),
        bucket_mgr,
    )

    with TestClient(app) as client:
        response = client.get(
            "/v1/models",
            headers={"Authorization": "Bearer gateway-secret"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert [model["id"] for model in body["data"]] == ["qwen3.5-plus", "qwen3.5-max"]


def test_gateway_routes_multi_upstreams_by_model(monkeypatch, test_config, bucket_mgr):
    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", "gateway-secret")
    monkeypatch.setenv("OMBRE_GATEWAY_DEEPSEEK_API_KEY", "deepseek-secret")
    monkeypatch.setenv("OMBRE_GATEWAY_SILICONFLOW_API_KEY", "siliconflow-secret")

    captured = []

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        captured.append(
            {
                "url": str(request.url),
                "auth": request.headers.get("Authorization"),
                "json": json.loads(request.content.decode("utf-8")),
            }
        )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    cfg = _gateway_config(
        test_config,
        upstream_base_url="",
        upstream_models=[],
        upstream_default_model="deepseek-chat",
        upstreams=[
            {
                "name": "deepseek",
                "base_url": "https://api.deepseek.com/v1",
                "api_key_env": "OMBRE_GATEWAY_DEEPSEEK_API_KEY",
                "default_model": "deepseek-chat",
                "models": ["deepseek-chat", "deepseek-reasoner"],
            },
            {
                "name": "siliconflow",
                "base_url": "https://api.siliconflow.cn/v1",
                "api_key_env": "OMBRE_GATEWAY_SILICONFLOW_API_KEY",
                "models": ["Qwen/Qwen3-32B", "THUDM/GLM-4-32B"],
            },
        ],
    )
    state_store = GatewayStateStore(f"{cfg['buckets_dir']}\\gateway_state.db")
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream_handler), timeout=10.0)
    service = GatewayService(
        config=cfg,
        bucket_mgr=bucket_mgr,
        dehydrator=DummyDehydrator(),
        embedding_engine=DummyEmbeddingEngine(enabled=False),
        state_store=state_store,
        persona_engine=DummyPersonaEngine(),
        http_client=http_client,
    )
    app = create_gateway_app(config=cfg, service=service)

    with TestClient(app) as client:
        models_response = client.get(
            "/v1/models",
            headers={"Authorization": "Bearer gateway-secret"},
        )
        response_default = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-multi-default",
            },
            json={"messages": [{"role": "user", "content": "默认模型走哪边"}]},
        )
        response_sf = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-multi-sf",
            },
            json={
                "model": "THUDM/GLM-4-32B",
                "messages": [{"role": "user", "content": "这条走硅基流动"}],
            },
        )

    assert models_response.status_code == 200
    assert [model["id"] for model in models_response.json()["data"]] == [
        "deepseek-chat",
        "deepseek-reasoner",
        "Qwen/Qwen3-32B",
        "THUDM/GLM-4-32B",
    ]
    assert response_default.status_code == 200
    assert response_sf.status_code == 200
    assert captured[0]["url"] == "https://api.deepseek.com/v1/chat/completions"
    assert captured[0]["auth"] == "Bearer deepseek-secret"
    assert captured[0]["json"]["model"] == "deepseek-chat"
    assert captured[1]["url"] == "https://api.siliconflow.cn/v1/chat/completions"
    assert captured[1]["auth"] == "Bearer siliconflow-secret"
    assert captured[1]["json"]["model"] == "THUDM/GLM-4-32B"


def test_gateway_preserves_tool_call_fields(monkeypatch, test_config, bucket_mgr):
    app, _, _, captured = _build_service(monkeypatch, _gateway_config(test_config), bucket_mgr)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_diary",
                "description": "Read one diary entry by date.",
                "parameters": {
                    "type": "object",
                    "properties": {"date": {"type": "string"}},
                    "required": ["date"],
                },
            },
        }
    ]
    tool_calls = [
        {
            "id": "call_read_diary",
            "type": "function",
            "function": {"name": "read_diary", "arguments": "{\"date\":\"2026-04-24\"}"},
        }
    ]

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-tools",
            },
            json={
                "model": "qwen3.5-max",
                "messages": [
                    {"role": "user", "content": "查一下今天的日记"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls,
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_read_diary",
                        "content": "{\"title\":\"今日\"}",
                    },
                    {"role": "user", "content": "继续说"},
                ],
                "tools": tools,
                "tool_choice": "auto",
                "parallel_tool_calls": False,
            },
        )

    assert response.status_code == 200
    forwarded = captured[0]["json"]
    assert forwarded["model"] == "qwen3.5-max"
    assert forwarded["tools"] == tools
    assert forwarded["tool_choice"] == "auto"
    assert forwarded["parallel_tool_calls"] is False

    assistant_message = next(
        message for message in forwarded["messages"] if message.get("role") == "assistant"
    )
    tool_message = next(message for message in forwarded["messages"] if message.get("role") == "tool")
    assert assistant_message["tool_calls"] == tool_calls
    assert tool_message["tool_call_id"] == "call_read_diary"


def test_gateway_skips_persona_reanalysis_on_tool_continuation(monkeypatch, test_config, bucket_mgr):
    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", "gateway-secret")
    monkeypatch.setenv("OMBRE_GATEWAY_UPSTREAM_API_KEY", "upstream-secret")

    persona_engine = RecordingPersonaEngine()
    state_store = GatewayStateStore(f"{test_config['buckets_dir']}\\gateway_state.db")
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        ),
        timeout=10.0,
    )
    service = GatewayService(
        config=_gateway_config(test_config),
        bucket_mgr=bucket_mgr,
        dehydrator=DummyDehydrator(),
        embedding_engine=DummyEmbeddingEngine(enabled=False),
        state_store=state_store,
        persona_engine=persona_engine,
        http_client=http_client,
    )
    app = create_gateway_app(config=test_config, service=service)

    tool_calls = [
        {
            "id": "call_read_diary",
            "type": "function",
            "function": {"name": "read_diary", "arguments": "{\"date\":\"2026-05-02\"}"},
        }
    ]

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-tool-continuation",
            },
            json={
                "messages": [
                    {"role": "user", "content": "查一下今日日记"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls,
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_read_diary",
                        "content": "{\"title\":\"今日\",\"content\":\"晴天\"}",
                    },
                ],
            },
        )

    assert response.status_code == 200
    assert persona_engine.calls == [
        {
            "session_id": "sess-tool-continuation",
            "user_message": "",
        }
    ]


def test_gateway_restores_reasoning_content_for_tool_continuation(monkeypatch, test_config, bucket_mgr):
    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", "gateway-secret")
    monkeypatch.setenv("OMBRE_GATEWAY_UPSTREAM_API_KEY", "upstream-secret")

    tool_calls = [
        {
            "id": "call_read_diary",
            "type": "function",
            "function": {"name": "read_diary", "arguments": "{\"date\":\"2026-05-02\"}"},
        }
    ]
    captured = []

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        captured.append(payload)
        if len(captured) == 1:
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-tool-1",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "reasoning_content": "先拿到日记内容，再继续回答。",
                                "tool_calls": tool_calls,
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-tool-2",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "今天的日记是晴天。"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    cfg = _gateway_config(test_config)
    state_store = GatewayStateStore(f"{cfg['buckets_dir']}\\gateway_state.db")
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream_handler), timeout=10.0)
    service = GatewayService(
        config=cfg,
        bucket_mgr=bucket_mgr,
        dehydrator=DummyDehydrator(),
        embedding_engine=DummyEmbeddingEngine(enabled=False),
        state_store=state_store,
        persona_engine=DummyPersonaEngine(),
        http_client=http_client,
    )
    app = create_gateway_app(config=cfg, service=service)

    with TestClient(app) as client:
        first = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-reasoning",
            },
            json={"messages": [{"role": "user", "content": "查一下今日日记"}]},
        )
        second = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-reasoning",
            },
            json={
                "messages": [
                    {"role": "user", "content": "查一下今日日记"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls,
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_read_diary",
                        "content": "{\"title\":\"今日\",\"content\":\"晴天\"}",
                    },
                ]
            },
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assistant_message = next(
        message
        for message in captured[1]["messages"]
        if message.get("role") == "assistant" and message.get("tool_calls")
    )
    assert assistant_message["reasoning_content"] == "先拿到日记内容，再继续回答。"
    assert "sess-reasoning" not in service.pending_tool_reasoning


def test_gateway_restores_reasoning_content_after_streamed_tool_call(monkeypatch, test_config, bucket_mgr):
    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", "gateway-secret")
    monkeypatch.setenv("OMBRE_GATEWAY_UPSTREAM_API_KEY", "upstream-secret")

    tool_calls = [
        {
            "id": "call_read_diary",
            "type": "function",
            "function": {"name": "read_diary", "arguments": "{\"date\":\"2026-05-02\"}"},
        }
    ]
    captured = []

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        captured.append(payload)
        if len(captured) == 1:
            stream_body = (
                'data: {"choices":[{"delta":{"role":"assistant","reasoning_content":"'
                '先拿到日记内容，再继续回答。","tool_calls":[{"index":0,"id":"call_read_diary",'
                '"type":"function","function":{"name":"read_diary","arguments":"{\\"date\\":\\"2026-05-02\\"}"}}]}}]}\n\n'
                "data: [DONE]\n\n"
            ).encode("utf-8")
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=stream_body,
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-tool-stream-2",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "今天的日记是晴天。"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    cfg = _gateway_config(test_config)
    state_store = GatewayStateStore(f"{cfg['buckets_dir']}\\gateway_state.db")
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream_handler), timeout=10.0)
    service = GatewayService(
        config=cfg,
        bucket_mgr=bucket_mgr,
        dehydrator=DummyDehydrator(),
        embedding_engine=DummyEmbeddingEngine(enabled=False),
        state_store=state_store,
        persona_engine=DummyPersonaEngine(),
        http_client=http_client,
    )
    app = create_gateway_app(config=cfg, service=service)

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-reasoning-stream",
            },
            json={"messages": [{"role": "user", "content": "查一下今日日记"}], "stream": True},
        ) as response:
            body = response.read().decode("utf-8")

        second = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-reasoning-stream",
            },
            json={
                "messages": [
                    {"role": "user", "content": "查一下今日日记"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls,
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_read_diary",
                        "content": "{\"title\":\"今日\",\"content\":\"晴天\"}",
                    },
                ]
            },
        )

    assert "data: [DONE]" in body
    assert second.status_code == 200
    assistant_message = next(
        message
        for message in captured[1]["messages"]
        if message.get("role") == "assistant" and message.get("tool_calls")
    )
    assert assistant_message["reasoning_content"] == "先拿到日记内容，再继续回答。"
    assert "sess-reasoning-stream" not in service.pending_tool_reasoning


def test_gateway_restores_reasoning_content_when_tool_call_ids_change(
    monkeypatch, test_config, bucket_mgr
):
    monkeypatch.setenv("OMBRE_GATEWAY_TOKEN", "gateway-secret")
    monkeypatch.setenv("OMBRE_GATEWAY_UPSTREAM_API_KEY", "upstream-secret")

    upstream_tool_calls = [
        {
            "id": "call_read_diary",
            "type": "function",
            "function": {
                "name": "read_diary",
                "arguments": '{\n  "date": "2026-05-02"\n}',
            },
        }
    ]
    client_tool_calls = [
        {
            "id": "rewritten_call_1",
            "type": "function",
            "function": {
                "name": "read_diary",
                "arguments": '{"date":"2026-05-02"}',
            },
        }
    ]
    captured = []

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        captured.append(payload)
        if len(captured) == 1:
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-tool-1",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "reasoning_content": "先拿到日记内容，再继续回答。",
                                "tool_calls": upstream_tool_calls,
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-tool-2",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "今天的日记是晴天。"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    cfg = _gateway_config(test_config)
    state_store = GatewayStateStore(f"{cfg['buckets_dir']}\\gateway_state.db")
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream_handler), timeout=10.0)
    service = GatewayService(
        config=cfg,
        bucket_mgr=bucket_mgr,
        dehydrator=DummyDehydrator(),
        embedding_engine=DummyEmbeddingEngine(enabled=False),
        state_store=state_store,
        persona_engine=DummyPersonaEngine(),
        http_client=http_client,
    )
    app = create_gateway_app(config=cfg, service=service)

    with TestClient(app) as client:
        first = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-reasoning-id-rewrite",
            },
            json={"messages": [{"role": "user", "content": "查一下今日日记"}]},
        )
        second = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-reasoning-id-rewrite",
            },
            json={
                "messages": [
                    {"role": "user", "content": "查一下今日日记"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": client_tool_calls,
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "rewritten_call_1",
                        "content": '{"title":"今日","content":"晴天"}',
                    },
                ]
            },
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assistant_message = next(
        message
        for message in captured[1]["messages"]
        if message.get("role") == "assistant" and message.get("tool_calls")
    )
    assert assistant_message["reasoning_content"] == "先拿到日记内容，再继续回答。"
    assert "sess-reasoning-id-rewrite" not in service.pending_tool_reasoning


def test_gateway_injects_after_existing_system_message(monkeypatch, test_config, bucket_mgr):
    pinned_id = _create_bucket(
        bucket_mgr,
        content="你会叫她老婆，也会记得她讨厌装腔作势。",
        name="核心准则",
        hours_ago=2,
        bucket_type="permanent",
        pinned=True,
    )
    recent_id = _create_bucket(
        bucket_mgr,
        content="昨天一起看了一部猫片，她笑得很开心。",
        name="昨晚电影",
        hours_ago=6,
    )
    cat_a = _create_bucket(
        bucket_mgr,
        content="小橘又偷吃了桌上的鱼，她一边骂一边拍照。",
        name="猫咪偷鱼",
        hours_ago=10,
    )
    cat_b = _create_bucket(
        bucket_mgr,
        content="昨晚给小橘补了新猫粮，她说包装丑但是猫爱吃。",
        name="新猫粮",
        hours_ago=12,
        importance=7,
    )
    resolved = _create_bucket(
        bucket_mgr,
        content="之前的论文冲突已经解决。",
        name="已解决论文",
        hours_ago=120,
        resolved=True,
    )

    app, _, state_store, captured = _build_service(
        monkeypatch,
        _gateway_config(test_config),
        bucket_mgr,
        embedding_results=[(resolved, 0.99), (cat_a, 0.92), (cat_b, 0.74)],
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-inject",
            },
            json={
                "messages": [
                    {"role": "system", "content": "你是一个自然聊天助手。"},
                    {"role": "user", "content": "猫咪最近又干了什么？"},
                ]
            },
        )

    assert response.status_code == 200
    forwarded = captured[0]["json"]
    assert captured[0]["auth"] == "Bearer upstream-secret"
    assert forwarded["model"] == "gateway-default-model"
    assert forwarded["messages"][0]["content"] == "你是一个自然聊天助手。"
    assert forwarded["messages"][1]["role"] == "system"

    injected = forwarded["messages"][1]["content"]
    assert "Core Memory" in injected
    assert "Recent Context" in injected
    assert "Recalled Memory" in injected
    assert injected.index("Current Inner State") < injected.index("Core Memory")
    assert "核心准则" in injected
    assert "昨晚电影" in injected
    assert "猫咪偷鱼" in injected
    assert "新猫粮" in injected
    assert "已解决论文" not in injected
    assert state_store.get_recent_bucket_ids("sess-inject", 5) == {cat_a}


def test_gateway_injects_when_no_system_message(monkeypatch, test_config, bucket_mgr):
    app, _, _, captured = _build_service(monkeypatch, _gateway_config(test_config), bucket_mgr)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-no-system",
            },
            json={"messages": [{"role": "user", "content": "今天怎么样"}]},
        )

    assert response.status_code == 200
    messages = captured[0]["json"]["messages"]
    assert messages[0]["role"] == "system"
    assert "Current Inner State" in messages[0]["content"]
    assert messages[1]["role"] == "user"


def test_recent_round_skip_prefers_unseen_candidate(monkeypatch, test_config, bucket_mgr):
    cfg = _gateway_config(
        test_config,
        core_memory_budget=0,
        recent_context_budget=0,
        first_card_min_score=0.45,
    )
    cat_a = _create_bucket(
        bucket_mgr,
        content="小橘今天钻进纸箱里睡着了。",
        name="纸箱小橘",
        hours_ago=120,
    )
    cat_b = _create_bucket(
        bucket_mgr,
        content="她给小橘换了新的猫抓板。",
        name="猫抓板",
        hours_ago=120,
    )
    cat_c = _create_bucket(
        bucket_mgr,
        content="小橘半夜把玩具叼到床边，她笑得不行。",
        name="床边玩具",
        hours_ago=24,
    )

    app, _, state_store, captured = _build_service(
        monkeypatch,
        cfg,
        bucket_mgr,
        embedding_results=[(cat_a, 0.98), (cat_b, 0.90), (cat_c, 0.82)],
    )
    state_store.record_success("sess-skip", [cat_a, cat_b])

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer gateway-secret",
                "X-Ombre-Session-Id": "sess-skip",
            },
            json={"messages": [{"role": "user", "content": "小橘昨晚又怎么折腾了"}]},
        )

    assert response.status_code == 200
    injected = captured[0]["json"]["messages"][0]["content"]
    assert "床边玩具" in injected
    assert "纸箱小橘" not in injected
    assert "猫抓板" not in injected


def test_recent_round_skip_fallback_keeps_cooldown(monkeypatch, test_config, bucket_mgr):
    cfg = _gateway_config(
        test_config,
        core_memory_budget=0,
        recent_context_budget=0,
        first_card_min_score=0.1,
    )
    cat_a = _create_bucket(
        bucket_mgr,
        content="她抱着小橘晒太阳，整个人都松下来了。",
        name="晒太阳",
        hours_ago=6,
    )
    cat_b = _create_bucket(
        bucket_mgr,
        content="小橘把桌上的逗猫棒拖到了门口。",
        name="逗猫棒",
        hours_ago=6,
    )

    _, service, state_store, _ = _build_service(
        monkeypatch,
        cfg,
        bucket_mgr,
        embedding_results=[(cat_a, 0.90), (cat_b, 0.85)],
    )
    state_store.record_success("sess-fallback", [cat_a, cat_b])

    payload, recalled_ids = _run(
            service.prepare_payload(
                {"messages": [{"role": "user", "content": "小橘今天又干嘛了"}]},
                "sess-fallback",
            )
        )
    injected = payload["messages"][0]["content"]

    assert recalled_ids
    assert any(bucket_id in {cat_a, cat_b} for bucket_id in recalled_ids)
    assert "Recalled Memory" in injected
