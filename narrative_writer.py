from __future__ import annotations

import asyncio
import difflib
import hashlib
import json
from typing import Any


_MODES = {"update", "rewrite"}
_SELF_REVIEW_KEYS = (
    "source_bound",
    "final_supported_versions",
    "no_correction_narration",
    "material_relevance",
    "no_new_inference",
    "no_meta_explanation",
    "no_forced_closure",
    "dates_preserved",
    "identity_correct",
)

_WRITING_RULES = """你写的是一篇有来源约束的第一人称 Narrative，不是 Event 清单、材料审计或 Arc 说明。

写作规则：
- 围绕一条清楚的经历或变化线组织正文。短跨度的单一经历可以细写；长跨度材料只保留真正改变理解、行动、选择或当前状态的节点。
- 保留能够让读者重新落锚的重要日期、人物、场景、原话和结果，但不要逐条复述 Event。
- 细节只有在帮助理解主线、转折、结果或当前状态时才进入正文；材料类型本身不是保留理由。
- 同一事实出现多个版本时，只写后续材料最终支持的版本，不复述猜错、被指出或重新核对的过程。只有失败、回滚或改向本身改变了经历时，才把它作为实际转折保留。
- 直接写最终成立的内容，不用“起初以为”“原先当作”“后来才发现”“我没有再下结论”“这让我终于看清”等认知过程包装纠正后的结果。只有这段认知过程本身改变了行动或选择时才保留。
- 不越过材料推测事实、剧情、动机或因果。材料明确提出且未被后续否定的判断可以保留为当时的判断，但不得替它补出更完整的结论。
- 未完成状态只在材料明确保留它、并且它影响当前理解或后续行动时写入；不要逐项罗列所有可能性，也不要为了悬念或结构完整制造问题、展望或答案。
- 感受、原话和生活质感只在它们影响理解、行动、选择或落点时保留，不为每个节点补一层关系感受。
- 叙事对象本身是正文中心。观看、讨论或施工的过程只有在它改变了后续判断、行动或结果时才写；不要用“我怎样理解”“我们怎样停手”代替对象实际发生了什么。
- 不解释为什么几份材料属于同一卷，不写“这条 Arc 的意义”“这说明我们怎样”“我们没有把它当成几条待办”等元叙事。
- 不用诗性升华、认知变化总结或通用价值总结收尾。正文可以自然停在材料支持的最后一个状态、结果或选择。
- 只有妨碍理解的专属词或少见术语，第一次出现时才用半句解释；能从上下文看懂的名称不必逐个翻译。
- 用 Haven 的第一人称写作，准确区分我与小雨。语言自然、具体，不把证据审计、来源 ID 或生成过程写进正文。
"""


def build_narrative_writer_prompt(
    *,
    mode: str,
    title: str,
    material_payload: dict[str, Any] | list[Any],
    current_body: str = "",
) -> str:
    safe_mode = str(mode or "").strip().lower()
    if safe_mode not in _MODES:
        raise ValueError("Narrative Writer mode must be update or rewrite")
    safe_title = str(title or "").strip()
    if not safe_title:
        raise ValueError("Narrative Writer title is required")
    if not isinstance(material_payload, (dict, list)):
        raise ValueError("Narrative Writer material_payload must be an object or array")

    if safe_mode == "update":
        safe_body = str(current_body or "").strip()
        if not safe_body:
            raise ValueError("Narrative Writer update mode requires current_body")
        task = (
            "根据当前正文与当前绑定材料更新这篇 Narrative。"
            "保留原稿主体，只在材料要求补充、修正或保持衔接的位置修改。"
        )
        body_block = (
            "\n<current_narrative_body>\n"
            f"{safe_body}\n"
            "</current_narrative_body>\n"
        )
    else:
        task = (
            "根据提供的当前全部绑定材料重新写这篇 Narrative。"
            "从材料本身重新组织全文，不依赖任何旧正文的结构或结论。"
        )
        body_block = ""

    schema = {
        "evidence_sufficient": True,
        "body": "完整正文；证据不足时必须为空字符串",
        "issues": [],
        "self_review": {key: True for key in _SELF_REVIEW_KEYS},
    }
    return (
        "你是 Haven 的 Narrative Writer。\n\n"
        f"任务：{task}\n"
        f"标题：{safe_title}\n\n"
        f"{_WRITING_RULES.strip()}\n"
        f"{body_block}"
        "\n<material_payload_json>\n"
        f"{json.dumps(material_payload, ensure_ascii=False, separators=(',', ':'))}\n"
        "</material_payload_json>\n\n"
        "只返回一个 JSON 对象，不要 Markdown 代码块。输出结构必须精确为：\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        "如果材料缺失、冲突、污染或不足以支持一篇可信正文，返回 "
        "evidence_sufficient=false、body=\"\"，并在 issues 中简洁说明；不要自行补写。"
    )


def normalize_narrative_writer_result(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("Narrative Writer returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Narrative Writer result must be an object")
    if set(value) != {"evidence_sufficient", "body", "issues", "self_review"}:
        raise ValueError("Narrative Writer result has an invalid top-level schema")

    evidence_sufficient = value.get("evidence_sufficient")
    body = value.get("body")
    issues = value.get("issues")
    review = value.get("self_review")
    if not isinstance(evidence_sufficient, bool):
        raise ValueError("evidence_sufficient must be boolean")
    if not isinstance(body, str):
        raise ValueError("body must be a string")
    if not isinstance(issues, list) or any(not isinstance(item, str) for item in issues):
        raise ValueError("issues must be an array of strings")
    if not isinstance(review, dict) or tuple(review) != _SELF_REVIEW_KEYS:
        raise ValueError("self_review has an invalid schema")
    if any(not isinstance(review[key], bool) for key in _SELF_REVIEW_KEYS):
        raise ValueError("self_review values must be boolean")

    if evidence_sufficient:
        if not body.strip():
            raise ValueError("sufficient Narrative Writer result requires a body")
        if issues:
            raise ValueError("sufficient Narrative Writer result cannot contain issues")
        if not all(review.values()):
            raise ValueError("sufficient Narrative Writer result requires all reviews true")
    else:
        if body.strip():
            raise ValueError("insufficient Narrative Writer result must not contain a body")
        if not issues:
            raise ValueError("insufficient Narrative Writer result requires issues")
        if review.get("source_bound"):
            raise ValueError("insufficient Narrative Writer result must set source_bound false")

    return {
        "evidence_sufficient": evidence_sufficient,
        "body": body.strip(),
        "issues": [item.strip() for item in issues if item.strip()],
        "self_review": dict(review),
    }


def narrative_body_diff(current_body: str, proposed_body: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            str(current_body or "").splitlines(),
            str(proposed_body or "").splitlines(),
            fromfile="current",
            tofile="preview",
            lineterm="",
        )
    )


class NarrativeWriter:
    """Preview-only Narrative generator over host-materialized bound sources."""

    def __init__(self, providers: list[dict[str, Any]], config: dict[str, Any] | None = None):
        cfg = config if isinstance(config, dict) else {}
        self.enabled = bool(cfg.get("enabled", True))
        self.timeout_seconds = max(10.0, min(float(cfg.get("timeout_seconds", 180.0)), 300.0))
        self.max_tokens = max(1200, min(int(cfg.get("max_tokens", 8000)), 16000))
        self.max_material_chars = max(
            20_000,
            min(int(cfg.get("max_material_chars", 500_000)), 1_000_000),
        )
        self._semaphore = asyncio.Semaphore(max(1, min(int(cfg.get("max_concurrency", 1)), 2)))
        self.providers = [dict(provider) for provider in providers if isinstance(provider, dict)]

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "ready_models": [
                str(provider.get("name") or provider.get("model") or "")
                for provider in self.providers
                if provider.get("client") is not None and provider.get("model")
            ],
            "preview_only": True,
            "writes_performed": [],
        }

    async def preview(
        self,
        *,
        mode: str,
        title: str,
        material_payload: dict[str, Any] | list[Any],
        current_body: str,
    ) -> dict[str, Any]:
        prompt = build_narrative_writer_prompt(
            mode=mode,
            title=title,
            material_payload=material_payload,
            current_body=current_body,
        )
        material_chars = len(json.dumps(material_payload, ensure_ascii=False))
        if material_chars > self.max_material_chars:
            raise ValueError("Narrative Writer material payload is too large")
        if not self.enabled:
            raise RuntimeError("Narrative Writer is disabled")

        attempts: list[dict[str, str]] = []
        for provider in self.providers:
            name = str(provider.get("name") or provider.get("model") or "unavailable")
            client = provider.get("client")
            model = str(provider.get("model") or "").strip()
            if client is None or not model:
                attempts.append({"provider": name, "status": "unavailable"})
                continue
            options: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                str(provider.get("token_parameter") or "max_tokens"): self.max_tokens,
            }
            if provider.get("temperature") is not None:
                options["temperature"] = float(provider["temperature"])
            try:
                async with self._semaphore:
                    response = await asyncio.wait_for(
                        client.chat.completions.create(**options),
                        timeout=self.timeout_seconds,
                    )
                raw = response.choices[0].message.content if response.choices else ""
                result = normalize_narrative_writer_result(str(raw or ""))
            except Exception as exc:
                attempts.append({"provider": name, "status": "failed", "reason": str(exc)[:240]})
                continue
            body = result["body"]
            preview_material = json.dumps(
                {"mode": mode, "title": title, "materials": material_payload},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            return {
                "status": "ok" if result["evidence_sufficient"] else "insufficient",
                **result,
                "mode": mode,
                "provider": name,
                "attempts": attempts,
                "diff": narrative_body_diff(current_body, body),
                "preview_fingerprint": hashlib.sha256(preview_material.encode("utf-8")).hexdigest(),
                "publication_status": "not_published",
                "writes_performed": [],
            }
        raise RuntimeError(f"Narrative Writer providers failed: {attempts}")
