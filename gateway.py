import logging
import os
import secrets
import json
import codecs
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from bucket_manager import BucketManager
from dehydrator import Dehydrator
from embedding_engine import EmbeddingEngine
from gateway_state import GatewayStateStore
from persona_engine import PersonaStateEngine
from utils import count_tokens_approx, load_config, setup_logging, strip_wikilinks

logger = logging.getLogger("ombre_brain.gateway")


class GatewayService:
    """
    OpenAI-compatible gateway that injects Ombre memory before forwarding
    chat completions upstream.
    """

    def __init__(
        self,
        config: dict,
        bucket_mgr: BucketManager | None = None,
        dehydrator: Dehydrator | None = None,
        embedding_engine: EmbeddingEngine | None = None,
        state_store: GatewayStateStore | None = None,
        persona_engine: PersonaStateEngine | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.config = config
        self.gateway_cfg = config.get("gateway", {})
        self.bucket_mgr = bucket_mgr or BucketManager(config)
        self.dehydrator = dehydrator or Dehydrator(config)
        self.embedding_engine = embedding_engine or EmbeddingEngine(config)
        self.state_store = state_store or GatewayStateStore(
            os.path.join(config["buckets_dir"], "gateway_state.db")
        )
        self.persona_engine = persona_engine or PersonaStateEngine(config)
        self.gateway_token = os.environ.get("OMBRE_GATEWAY_TOKEN", "")
        self.upstream_api_key = os.environ.get("OMBRE_GATEWAY_UPSTREAM_API_KEY", "")
        self.upstream_base_url = self.gateway_cfg.get("upstream_base_url", "").rstrip("/")
        self.upstream_default_model = self.gateway_cfg.get("upstream_default_model", "")
        self.upstream_models = self._normalize_model_list(
            self.gateway_cfg.get("upstream_models", []),
            self.upstream_default_model,
        )
        self.upstreams = self._load_upstreams()
        self.upstream_models = self._aggregate_upstream_models()
        if not self.upstream_default_model:
            for upstream in self.upstreams:
                default_model = upstream.get("default_model") or ""
                if default_model:
                    self.upstream_default_model = default_model
                    break

        self.head_recent_hours = int(self.gateway_cfg.get("head_recent_hours", 72))
        self.dynamic_top_k = int(self.gateway_cfg.get("dynamic_top_k", 10))
        self.inject_max_cards = max(0, min(2, int(self.gateway_cfg.get("inject_max_cards", 2))))
        self.skip_recent_rounds = max(0, int(self.gateway_cfg.get("skip_recent_rounds", 5)))
        self.cooldown_hours = float(self.gateway_cfg.get("cooldown_hours", 48))
        self.cooldown_floor = float(self.gateway_cfg.get("cooldown_floor", 0.3))

        self.inject_total_budget = int(self.gateway_cfg.get("inject_total_budget", 1200))
        self.core_budget = int(self.gateway_cfg.get("core_memory_budget", 500))
        self.recent_budget = int(self.gateway_cfg.get("recent_context_budget", 300))
        self.recalled_budget = int(self.gateway_cfg.get("recalled_memory_budget", 400))

        self.semantic_weight = float(self.gateway_cfg.get("semantic_weight", 0.45))
        self.keyword_weight = float(self.gateway_cfg.get("keyword_weight", 0.35))
        self.importance_weight = float(self.gateway_cfg.get("importance_weight", 0.10))
        self.freshness_weight = float(self.gateway_cfg.get("freshness_weight", 0.10))
        self.first_card_min_score = float(self.gateway_cfg.get("first_card_min_score", 0.55))
        self.second_card_min_score = float(self.gateway_cfg.get("second_card_min_score", 0.50))
        self.second_card_relative_score = float(
            self.gateway_cfg.get("second_card_relative_score", 0.85)
        )
        self.pending_tool_reasoning: dict[str, dict[tuple[str, ...], dict[str, Any]]] = {}

        self.http_client = http_client or httpx.AsyncClient(timeout=60.0)

    async def close(self) -> None:
        if self.http_client and not getattr(self.http_client, "is_closed", False):
            await self.http_client.aclose()

    async def health_payload(self) -> dict:
        stats = await self.bucket_mgr.get_stats()
        return {
            "status": "ok",
            "gateway": {
                "token_configured": bool(self.gateway_token),
                "upstream_ready": bool(self.upstreams) and all(
                    bool(upstream.get("base_url") and upstream.get("api_key"))
                    for upstream in self.upstreams
                ),
                "upstream_base_url": self.upstream_base_url
                or (self.upstreams[0]["base_url"] if len(self.upstreams) == 1 else ""),
                "upstream_default_model": self.upstream_default_model,
                "upstream_models": self.upstream_models,
                "upstreams": [
                    {
                        "name": upstream["name"],
                        "base_url": upstream["base_url"],
                        "default_model": upstream["default_model"],
                        "models": upstream["models"],
                        "ready": bool(upstream.get("base_url") and upstream.get("api_key")),
                    }
                    for upstream in self.upstreams
                ],
            },
            "persona": {
                "enabled": bool(self.persona_engine.enabled),
                "profile_id": self.persona_engine.profile_id,
                "mode": self.persona_engine.mode,
                "model": self.persona_engine.model,
                "api_ready": bool(self.persona_engine.api_key),
            },
            "buckets": stats,
        }

    async def handle_health(self, request: Request) -> JSONResponse:
        try:
            return JSONResponse(await self.health_payload())
        except Exception as exc:
            logger.exception("Gateway health check failed: %s", exc)
            return JSONResponse({"status": "error", "detail": str(exc)}, status_code=500)

    async def handle_chat(self, request: Request) -> Response:
        auth_result = self._authorize(request.headers.get("Authorization", ""))
        if auth_result is not None:
            return auth_result

        session_id = (request.headers.get("X-Ombre-Session-Id") or "").strip()
        if not session_id:
            return JSONResponse(
                {"error": {"message": "X-Ombre-Session-Id is required", "type": "invalid_request_error"}},
                status_code=400,
            )

        try:
            payload = await request.json()
        except Exception:
            return JSONResponse(
                {"error": {"message": "Request body must be valid JSON", "type": "invalid_request_error"}},
                status_code=400,
            )

        if not isinstance(payload, dict):
            return JSONResponse(
                {"error": {"message": "Request body must be a JSON object", "type": "invalid_request_error"}},
                status_code=400,
            )

        try:
            forward_payload, recalled_ids = await self.prepare_payload(payload, session_id)
        except ValueError as exc:
            return JSONResponse(
                {"error": {"message": str(exc), "type": "invalid_request_error"}},
                status_code=400,
            )
        except RuntimeError as exc:
            return JSONResponse(
                {"error": {"message": str(exc), "type": "server_error"}},
                status_code=503,
            )

        if forward_payload.get("stream") is True:
            try:
                return await self._stream_upstream(forward_payload, session_id, recalled_ids)
            except RuntimeError as exc:
                return JSONResponse(
                    {"error": {"message": str(exc), "type": "server_error"}},
                    status_code=503,
                )

        upstream_response = await self._forward_upstream(forward_payload)
        if 200 <= upstream_response.status_code < 300:
            self._capture_reasoning_from_response(session_id, upstream_response)
            await self._record_successful_round(session_id, recalled_ids)

        return self._proxy_response(upstream_response)

    async def handle_models(self, request: Request) -> Response:
        auth_result = self._authorize(request.headers.get("Authorization", ""))
        if auth_result is not None:
            return auth_result

        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": model,
                        "object": "model",
                        "created": 0,
                        "owned_by": "ombre-gateway",
                    }
                    for model in self.upstream_models
                ],
            }
        )

    async def prepare_payload(self, payload: dict, session_id: str) -> tuple[dict, list[str]]:
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")

        model = payload.get("model") or self.upstream_default_model
        if not model:
            raise ValueError("model is required when gateway.upstream_default_model is empty")
        self._get_upstream_for_model(model)

        all_buckets = await self.bucket_mgr.list_all(include_archive=False)
        query = self._extract_last_user_query(messages)
        persona_query = self._extract_current_turn_user_query(messages)
        persona_state = await self.persona_engine.update_from_user_message(session_id, persona_query)
        persona_block = self.persona_engine.format_state_block(persona_state)
        core_memory = await self._build_core_memory_block(all_buckets)
        recent_context = await self._build_recent_context_block(all_buckets)
        recalled_buckets = await self._select_dynamic_buckets(query, session_id, all_buckets)
        recalled_memory = await self._summarize_buckets(recalled_buckets, self.recalled_budget)
        injected_message = self._build_injected_memory_message(
            persona_block=persona_block,
            core_memory=core_memory,
            recent_context=recent_context,
            recalled_memory=recalled_memory,
        )

        forward_payload = deepcopy(payload)
        forward_payload["model"] = model
        self._restore_cached_reasoning_content(session_id, forward_payload.get("messages"))
        forward_payload["messages"] = self._inject_system_message(
            forward_payload["messages"],
            injected_message,
        )
        forward_payload["stream"] = payload.get("stream") is True
        return forward_payload, [bucket["id"] for bucket in recalled_buckets]

    def _authorize(self, auth_header: str) -> JSONResponse | None:
        if not self.gateway_token:
            return JSONResponse(
                {"error": {"message": "Gateway token is not configured", "type": "server_error"}},
                status_code=503,
            )

        scheme, _, token = auth_header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            return JSONResponse(
                {"error": {"message": "Authorization: Bearer token is required", "type": "authentication_error"}},
                status_code=401,
            )

        if not secrets.compare_digest(token, self.gateway_token):
            return JSONResponse(
                {"error": {"message": "Invalid gateway token", "type": "authentication_error"}},
                status_code=401,
            )
        return None

    async def _forward_upstream(self, payload: dict) -> httpx.Response:
        model = str(payload.get("model") or "").strip()
        upstream = self._get_upstream_for_model(model)
        url = f"{upstream['base_url']}/chat/completions"
        return await self.http_client.post(
            url,
            headers={
                "Authorization": f"Bearer {upstream['api_key']}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

    async def _stream_upstream(
        self,
        payload: dict,
        session_id: str,
        recalled_ids: list[str],
    ) -> Response:
        model = str(payload.get("model") or "").strip()
        upstream = self._get_upstream_for_model(model)
        url = f"{upstream['base_url']}/chat/completions"
        request = self.http_client.build_request(
            "POST",
            url,
            headers={
                "Authorization": f"Bearer {upstream['api_key']}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        upstream_response = await self.http_client.send(request, stream=True)
        content_type = upstream_response.headers.get("content-type", "text/event-stream")

        if not 200 <= upstream_response.status_code < 300:
            body = await upstream_response.aread()
            await upstream_response.aclose()
            return Response(
                content=body,
                status_code=upstream_response.status_code,
                media_type=content_type,
            )

        async def stream_body():
            completed = False
            stream_state = self._new_stream_capture_state()
            try:
                async for chunk in upstream_response.aiter_bytes():
                    if chunk:
                        self._consume_stream_capture_chunk(stream_state, chunk)
                        yield chunk
                self._consume_stream_capture_chunk(stream_state, b"", final=True)
                completed = True
            finally:
                await upstream_response.aclose()
                if completed:
                    self._capture_reasoning_from_stream_state(session_id, stream_state)
                    await self._record_successful_round(session_id, recalled_ids)

        return StreamingResponse(
            stream_body(),
            status_code=upstream_response.status_code,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    async def _record_successful_round(self, session_id: str, recalled_ids: list[str]) -> None:
        round_id = self.state_store.record_success(session_id, recalled_ids)
        for bucket_id in recalled_ids:
            await self.bucket_mgr.touch(bucket_id)
        logger.info(
            "Gateway round completed | session=%s round=%s recalled=%s",
            session_id,
            round_id,
            recalled_ids,
        )

    def _proxy_response(self, upstream_response: httpx.Response) -> Response:
        content_type = upstream_response.headers.get("content-type", "application/json")
        try:
            body = upstream_response.json()
            return JSONResponse(body, status_code=upstream_response.status_code)
        except ValueError:
            return Response(
                content=upstream_response.text,
                status_code=upstream_response.status_code,
                media_type=content_type,
            )

    def _extract_last_user_query(self, messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            if message.get("role") != "user":
                continue
            content = self._coerce_message_text(message.get("content"))
            if content.strip():
                return content.strip()
        return ""

    def _extract_current_turn_user_query(self, messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role == "system":
                continue
            if role != "user":
                return ""
            content = self._coerce_message_text(message.get("content"))
            return content.strip()
        return ""

    def _coerce_message_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type in {"text", "input_text"}:
                    text = item.get("text") or item.get("input_text") or ""
                    if text:
                        chunks.append(str(text))
            return "\n".join(chunks)
        return ""

    async def _build_core_memory_block(self, all_buckets: list[dict]) -> str:
        core_buckets = [
            bucket for bucket in all_buckets
            if bucket.get("metadata", {}).get("pinned") or bucket.get("metadata", {}).get("protected")
        ]
        core_buckets.sort(
            key=lambda bucket: (
                int(bucket.get("metadata", {}).get("importance", 0)),
                bucket.get("metadata", {}).get("last_active", ""),
            ),
            reverse=True,
        )
        return await self._summarize_buckets(core_buckets, self.core_budget)

    async def _build_recent_context_block(self, all_buckets: list[dict]) -> str:
        cutoff = datetime.now() - timedelta(hours=self.head_recent_hours)
        recent_buckets = []
        for bucket in all_buckets:
            meta = bucket.get("metadata", {})
            if meta.get("type") == "feel":
                continue
            if meta.get("pinned") or meta.get("protected"):
                continue
            created = self._parse_iso(meta.get("created") or meta.get("last_active"))
            if created and created >= cutoff:
                recent_buckets.append(bucket)

        recent_buckets.sort(
            key=lambda bucket: bucket.get("metadata", {}).get("created", ""),
            reverse=True,
        )
        return await self._summarize_buckets(recent_buckets[:6], self.recent_budget)

    async def _select_dynamic_buckets(
        self,
        query: str,
        session_id: str,
        all_buckets: list[dict],
    ) -> list[dict]:
        if not query or self.inject_max_cards <= 0:
            return []

        eligible = [bucket for bucket in all_buckets if self._is_dynamic_candidate(bucket)]
        if not eligible:
            return []

        bucket_map = {bucket["id"]: bucket for bucket in eligible}
        keyword_scores = self._get_keyword_candidates(query, eligible)
        semantic_scores = await self._get_semantic_candidates(query, set(bucket_map))
        candidate_ids = set(keyword_scores) | set(semantic_scores)
        if not candidate_ids:
            return []

        now = datetime.now()
        scored_candidates = []
        for bucket_id in candidate_ids:
            bucket = bucket_map.get(bucket_id)
            if not bucket:
                continue
            meta = bucket.get("metadata", {})
            freshness_score = self._clamp(self.bucket_mgr._calc_time_score(meta))
            importance_score = self._clamp(float(meta.get("importance", 5)) / 10.0)
            semantic_score = self._clamp(semantic_scores.get(bucket_id, 0.0))
            keyword_score = self._clamp(keyword_scores.get(bucket_id, 0.0))
            base_score = (
                semantic_score * self.semantic_weight
                + keyword_score * self.keyword_weight
                + importance_score * self.importance_weight
                + freshness_score * self.freshness_weight
            )
            cooldown_multiplier = self.state_store.get_cooldown_multiplier(
                session_id=session_id,
                bucket_id=bucket_id,
                cooldown_hours=self.cooldown_hours,
                cooldown_floor=self.cooldown_floor,
                now=now,
            )
            scored_candidates.append(
                {
                    "bucket": bucket,
                    "score": round(base_score * cooldown_multiplier, 4),
                    "semantic_score": semantic_score,
                    "keyword_score": keyword_score,
                    "importance_score": importance_score,
                    "freshness_score": freshness_score,
                    "cooldown_multiplier": cooldown_multiplier,
                }
            )

        scored_candidates.sort(key=lambda item: item["score"], reverse=True)
        recent_ids = self.state_store.get_recent_bucket_ids(session_id, self.skip_recent_rounds)
        filtered = [item for item in scored_candidates if item["bucket"]["id"] not in recent_ids]
        active_pool = filtered or scored_candidates
        selected = self._pick_dynamic_cards(active_pool)
        return [item["bucket"] for item in selected]

    def _get_keyword_candidates(self, query: str, buckets: list[dict]) -> dict[str, float]:
        scored = []
        for bucket in buckets:
            keyword_score = self._clamp(self.bucket_mgr._calc_topic_score(query, bucket))
            if keyword_score > 0:
                scored.append((bucket["id"], keyword_score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return {bucket_id: score for bucket_id, score in scored[: self.dynamic_top_k]}

    async def _get_semantic_candidates(self, query: str, eligible_ids: set[str]) -> dict[str, float]:
        if not getattr(self.embedding_engine, "enabled", False):
            return {}

        results = await self.embedding_engine.search_similar(query, top_k=self.dynamic_top_k)
        semantic_scores = {}
        for bucket_id, similarity in results:
            if bucket_id not in eligible_ids:
                continue
            semantic_scores[bucket_id] = self._clamp(similarity)
        return semantic_scores

    def _pick_dynamic_cards(self, scored_candidates: list[dict]) -> list[dict]:
        if not scored_candidates:
            return []

        chosen = []
        first = scored_candidates[0]
        if first["score"] < self.first_card_min_score:
            return []
        chosen.append(first)

        if self.inject_max_cards < 2 or len(scored_candidates) < 2:
            return chosen

        second = scored_candidates[1]
        if (
            second["score"] >= self.second_card_min_score
            and second["score"] >= first["score"] * self.second_card_relative_score
        ):
            chosen.append(second)
        return chosen

    async def _summarize_buckets(self, buckets: list[dict], budget: int) -> str:
        if budget <= 0 or not buckets:
            return ""

        remaining = budget
        parts = []
        for bucket in buckets:
            summary = await self._summarize_bucket(bucket)
            summary_tokens = count_tokens_approx(summary)
            if summary_tokens <= 0:
                continue
            if summary_tokens > remaining and parts:
                break
            if summary_tokens > remaining:
                summary = self._trim_text(summary, remaining)
                summary_tokens = count_tokens_approx(summary)
            if summary_tokens <= 0:
                continue
            parts.append(f"- {summary}")
            remaining -= summary_tokens
            if remaining <= 0:
                break
        return "\n".join(parts)

    async def _summarize_bucket(self, bucket: dict) -> str:
        metadata = {
            key: value
            for key, value in bucket.get("metadata", {}).items()
            if key != "tags"
        }
        cleaned = strip_wikilinks(bucket.get("content", ""))
        try:
            return await self.dehydrator.dehydrate(cleaned, metadata)
        except Exception as exc:
            logger.warning("Gateway summary fallback for %s: %s", bucket.get("id"), exc)
            title = metadata.get("name", bucket.get("id", "memory"))
            truncated = self._trim_text(cleaned, 90)
            return f"📌 记忆桶: {title}\n{truncated}"

    def _build_injected_memory_message(
        self,
        persona_block: str,
        core_memory: str,
        recent_context: str,
        recalled_memory: str,
    ) -> str:
        sections = [
            "Use the following private memory only when it fits naturally. "
            "Keep the reply seamless and do not mention memory lookup, search, or hidden context.",
            "",
            persona_block,
            "",
            "Core Memory",
            core_memory or "(none)",
            "",
            "Recent Context",
            recent_context or "(none)",
            "",
            "Recalled Memory",
            recalled_memory or "(none)",
        ]
        injected = "\n".join(sections).strip()
        if count_tokens_approx(injected) > self.inject_total_budget:
            injected = self._trim_text(injected, self.inject_total_budget)
        return injected

    def _inject_system_message(self, messages: list[dict], injected_message: str) -> list[dict]:
        new_messages = deepcopy(messages)
        memory_message = {"role": "system", "content": injected_message}
        if new_messages and isinstance(new_messages[0], dict) and new_messages[0].get("role") == "system":
            return [new_messages[0], memory_message, *new_messages[1:]]
        return [memory_message, *new_messages]

    def _restore_cached_reasoning_content(self, session_id: str, messages: Any) -> None:
        if not isinstance(messages, list) or not any(
            isinstance(message, dict) and message.get("role") == "tool"
            for message in messages
        ):
            return

        cache = self.pending_tool_reasoning.get(session_id)
        if not cache:
            return

        restored = 0
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            if message.get("reasoning_content"):
                continue
            signature = self._tool_call_signature(message)
            if not signature:
                continue
            cached_message = cache.get(signature)
            if not cached_message or not cached_message.get("reasoning_content"):
                continue
            message["reasoning_content"] = cached_message["reasoning_content"]
            restored += 1

        if restored:
            logger.info(
                "Gateway restored reasoning_content for %s assistant tool-call message(s) | session=%s",
                restored,
                session_id,
            )

    def _capture_reasoning_from_response(self, session_id: str, upstream_response: httpx.Response) -> None:
        try:
            body = upstream_response.json()
        except ValueError:
            return
        self._capture_reasoning_from_response_body(session_id, body)

    def _capture_reasoning_from_response_body(self, session_id: str, body: Any) -> None:
        message = self._extract_assistant_message_from_response_body(body)
        if message:
            self._update_reasoning_cache(session_id, message)

    def _extract_assistant_message_from_response_body(self, body: Any) -> dict[str, Any] | None:
        if not isinstance(body, dict):
            return None
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        choice = choices[0]
        if not isinstance(choice, dict):
            return None
        message = choice.get("message")
        if isinstance(message, dict) and message.get("role", "assistant") == "assistant":
            return message
        return None

    def _update_reasoning_cache(self, session_id: str, assistant_message: dict[str, Any]) -> None:
        signature = self._tool_call_signature(assistant_message)
        reasoning_content = assistant_message.get("reasoning_content")
        if signature and reasoning_content:
            cache = self.pending_tool_reasoning.setdefault(session_id, {})
            cache[signature] = {
                "reasoning_content": reasoning_content,
                "tool_calls": deepcopy(assistant_message.get("tool_calls", [])),
            }
            logger.info(
                "Gateway cached reasoning_content for tool continuation | session=%s tool_calls=%s",
                session_id,
                list(signature),
            )
            return

        if not signature:
            self.pending_tool_reasoning.pop(session_id, None)

    def _tool_call_signature(self, assistant_message: Any) -> tuple[str, ...]:
        if not isinstance(assistant_message, dict):
            return ()
        tool_calls = assistant_message.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            return ()

        signature = []
        for index, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                continue
            tool_id = tool_call.get("id")
            if tool_id:
                signature.append(str(tool_id))
                continue
            function = tool_call.get("function", {})
            if isinstance(function, dict):
                signature.append(
                    f"idx:{index}:{function.get('name', '')}:{function.get('arguments', '')}"
                )
        return tuple(signature)

    def _new_stream_capture_state(self) -> dict[str, Any]:
        return {
            "decoder": codecs.getincrementaldecoder("utf-8")(),
            "buffer": "",
            "message": {
                "role": "assistant",
                "content": "",
                "reasoning_content": "",
            },
            "tool_calls_by_index": {},
        }

    def _consume_stream_capture_chunk(
        self,
        stream_state: dict[str, Any],
        chunk: bytes,
        final: bool = False,
    ) -> None:
        decoder = stream_state["decoder"]
        if chunk:
            stream_state["buffer"] += decoder.decode(chunk)
        if final:
            stream_state["buffer"] += decoder.decode(b"", final=True)

        buffer = stream_state["buffer"].replace("\r\n", "\n")
        while "\n\n" in buffer:
            event_text, buffer = buffer.split("\n\n", 1)
            self._consume_sse_event(stream_state, event_text)

        if final and buffer.strip():
            self._consume_sse_event(stream_state, buffer)
            buffer = ""

        stream_state["buffer"] = buffer

    def _consume_sse_event(self, stream_state: dict[str, Any], event_text: str) -> None:
        data_lines = []
        for raw_line in event_text.split("\n"):
            line = raw_line.strip()
            if line.startswith("data:"):
                data_lines.append(line[5:].strip())

        if not data_lines:
            return
        payload = "\n".join(data_lines).strip()
        if not payload or payload == "[DONE]":
            return

        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            return

        if not isinstance(event, dict):
            return
        for choice in event.get("choices", []):
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict):
                self._merge_stream_message_delta(stream_state, delta)
            message = choice.get("message")
            if isinstance(message, dict):
                self._merge_complete_message(stream_state, message)

    def _merge_stream_message_delta(self, stream_state: dict[str, Any], delta: dict[str, Any]) -> None:
        message = stream_state["message"]
        if delta.get("role"):
            message["role"] = delta["role"]
        if isinstance(delta.get("content"), str):
            message["content"] += delta["content"]
        if isinstance(delta.get("reasoning_content"), str):
            message["reasoning_content"] += delta["reasoning_content"]

        tool_calls = delta.get("tool_calls")
        if not isinstance(tool_calls, list):
            return
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            index = int(tool_call.get("index", 0))
            target = stream_state["tool_calls_by_index"].setdefault(
                index,
                {"type": "function", "function": {"name": "", "arguments": ""}},
            )
            if tool_call.get("id"):
                target["id"] = tool_call["id"]
            if tool_call.get("type"):
                target["type"] = tool_call["type"]
            function = tool_call.get("function")
            if isinstance(function, dict):
                target_function = target.setdefault("function", {"name": "", "arguments": ""})
                if isinstance(function.get("name"), str):
                    target_function["name"] += function["name"]
                if isinstance(function.get("arguments"), str):
                    target_function["arguments"] += function["arguments"]

    def _merge_complete_message(self, stream_state: dict[str, Any], message: dict[str, Any]) -> None:
        target = stream_state["message"]
        if message.get("role"):
            target["role"] = message["role"]
        if isinstance(message.get("content"), str):
            target["content"] = message["content"]
        if isinstance(message.get("reasoning_content"), str):
            target["reasoning_content"] = message["reasoning_content"]
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            stream_state["tool_calls_by_index"] = {
                index: deepcopy(tool_call)
                for index, tool_call in enumerate(tool_calls)
                if isinstance(tool_call, dict)
            }

    def _capture_reasoning_from_stream_state(self, session_id: str, stream_state: dict[str, Any]) -> None:
        assistant_message = self._build_stream_assistant_message(stream_state)
        if assistant_message:
            self._update_reasoning_cache(session_id, assistant_message)

    def _build_stream_assistant_message(self, stream_state: dict[str, Any]) -> dict[str, Any] | None:
        message = deepcopy(stream_state.get("message", {}))
        tool_calls_by_index = stream_state.get("tool_calls_by_index", {})
        tool_calls = [
            deepcopy(tool_calls_by_index[index])
            for index in sorted(tool_calls_by_index)
            if isinstance(tool_calls_by_index[index], dict)
        ]

        content = message.get("content", "")
        reasoning_content = message.get("reasoning_content", "")
        if not (tool_calls or content or reasoning_content):
            return None

        assistant_message: dict[str, Any] = {"role": message.get("role", "assistant")}
        assistant_message["content"] = content if content else None
        if reasoning_content:
            assistant_message["reasoning_content"] = reasoning_content
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        return assistant_message

    def _is_dynamic_candidate(self, bucket: dict) -> bool:
        meta = bucket.get("metadata", {})
        if meta.get("type") in {"feel", "permanent", "archived"}:
            return False
        if meta.get("resolved"):
            return False
        if meta.get("pinned") or meta.get("protected"):
            return False
        return True

    def _trim_text(self, text: str, budget_tokens: int) -> str:
        if budget_tokens <= 0:
            return ""
        if count_tokens_approx(text) <= budget_tokens:
            return text
        trimmed = text
        while trimmed and count_tokens_approx(trimmed) > budget_tokens:
            cut = max(1, int(len(trimmed) * 0.85))
            trimmed = trimmed[:cut].rstrip()
        return trimmed

    def _parse_iso(self, value: Any) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value))
        except ValueError:
            return None

    def _clamp(self, value: float, lower: float = 0.0, upper: float = 1.0) -> float:
        return max(lower, min(upper, float(value)))

    def _load_upstreams(self) -> list[dict[str, Any]]:
        raw_upstreams = self.gateway_cfg.get("upstreams", [])
        if isinstance(raw_upstreams, list) and raw_upstreams:
            upstreams = []
            for index, raw in enumerate(raw_upstreams, start=1):
                if not isinstance(raw, dict):
                    continue
                name = str(raw.get("name") or f"upstream-{index}").strip() or f"upstream-{index}"
                base_url = str(raw.get("base_url") or "").rstrip("/")
                default_model = str(raw.get("default_model") or "").strip()
                api_key = str(raw.get("api_key") or "").strip()
                api_key_env = str(raw.get("api_key_env") or "").strip()
                if api_key_env and not api_key:
                    api_key = os.environ.get(api_key_env, "")
                upstreams.append(
                    {
                        "name": name,
                        "base_url": base_url,
                        "api_key": api_key,
                        "default_model": default_model,
                        "models": self._normalize_model_list(raw.get("models", []), default_model),
                    }
                )
            if upstreams:
                return upstreams

        return [
            {
                "name": "default",
                "base_url": self.upstream_base_url,
                "api_key": self.upstream_api_key,
                "default_model": self.upstream_default_model,
                "models": self._normalize_model_list(
                    self.gateway_cfg.get("upstream_models", []),
                    self.upstream_default_model,
                ),
            }
        ]

    def _aggregate_upstream_models(self) -> list[str]:
        models = []
        for upstream in self.upstreams:
            for model in upstream.get("models", []):
                if not model:
                    continue
                if model in models:
                    logger.warning(
                        'Duplicate gateway model "%s" found in upstream "%s"; first match wins',
                        model,
                        upstream.get("name", "unknown"),
                    )
                    continue
                models.append(model)
        return models

    def _get_upstream_for_model(self, model: str) -> dict[str, Any]:
        if not self.upstreams:
            raise RuntimeError("gateway upstream is not configured")

        if len(self.upstreams) == 1:
            upstream = self.upstreams[0]
        else:
            normalized_model = str(model or "").strip()
            if not normalized_model:
                raise ValueError("model is required when gateway has multiple upstreams")
            upstream = next(
                (
                    candidate
                    for candidate in self.upstreams
                    if normalized_model in candidate.get("models", [])
                ),
                None,
            )
            if upstream is None:
                raise ValueError(f'model "{normalized_model}" is not configured in gateway.upstreams')

        if not upstream.get("base_url"):
            raise RuntimeError(f'gateway upstream "{upstream["name"]}" base_url is not configured')
        if not upstream.get("api_key"):
            raise RuntimeError(f'gateway upstream "{upstream["name"]}" api_key is not configured')
        return upstream

    def _normalize_model_list(self, raw_models: Any, default_model: str) -> list[str]:
        if isinstance(raw_models, str):
            candidates = [item.strip() for item in raw_models.split(",")]
        elif isinstance(raw_models, list):
            candidates = [str(item).strip() for item in raw_models]
        else:
            candidates = []

        models = []
        for model in candidates:
            if model and model not in models:
                models.append(model)

        if default_model and default_model not in models:
            models.insert(0, default_model)
        return models


def create_gateway_app(
    config: dict | None = None,
    service: GatewayService | None = None,
) -> Starlette:
    config = config or load_config()
    service = service or GatewayService(config)

    @asynccontextmanager
    async def lifespan(app: Starlette):
        app.state.gateway_service = service
        yield
        await service.close()

    async def health(request: Request) -> JSONResponse:
        return await request.app.state.gateway_service.handle_health(request)

    async def chat_completions(request: Request) -> Response:
        return await request.app.state.gateway_service.handle_chat(request)

    async def models(request: Request) -> Response:
        return await request.app.state.gateway_service.handle_models(request)

    app = Starlette(
        debug=False,
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/v1/models", models, methods=["GET"]),
            Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        ],
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    return app


def main() -> None:
    config = load_config()
    setup_logging(config.get("log_level", "INFO"))
    gateway_cfg = config.get("gateway", {})
    app = create_gateway_app(config=config)
    host = gateway_cfg.get("host", "0.0.0.0")
    port = int(gateway_cfg.get("port", 8010))
    logger.info("Ombre Brain gateway starting | host=%s port=%s", host, port)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
