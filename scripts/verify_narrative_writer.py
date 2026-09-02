from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from narrative_writer import NarrativeWriter, build_narrative_writer_prompt, normalize_narrative_writer_result


REVIEW = {
    "source_bound": True,
    "final_supported_versions": True,
    "no_correction_narration": True,
    "material_relevance": True,
    "no_new_inference": True,
    "no_meta_explanation": True,
    "no_forced_closure": True,
    "dates_preserved": True,
    "identity_correct": True,
}


class FakeCompletions:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(self.result, ensure_ascii=False)))]
        )


def _expect_value_error(callable_) -> None:
    try:
        callable_()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


async def _verify_service() -> None:
    result = {
        "evidence_sufficient": True,
        "body": "2026 年 8 月 22 日，我们看完了这一集。",
        "issues": [],
        "self_review": REVIEW,
    }
    completions = FakeCompletions(result)
    writer = NarrativeWriter(
        [{
            "name": "fake",
            "model": "fake-model",
            "client": SimpleNamespace(chat=SimpleNamespace(completions=completions)),
            "token_parameter": "max_tokens",
        }],
        {"max_tokens": 2400},
    )
    preview = await writer.preview(
        mode="update",
        title="一篇 Narrative",
        current_body="旧正文。",
        material_payload={"events": [{"title": "一次观看", "source_messages": ["最终剧情"]}]},
    )
    assert preview["status"] == "ok"
    assert preview["publication_status"] == "not_published"
    assert preview["writes_performed"] == []
    assert "-旧正文。" in preview["diff"]
    assert "+2026 年 8 月 22 日" in preview["diff"]
    assert completions.calls[0]["max_tokens"] == 2400


def main() -> None:
    materials = {
        "events": [{"title": "一次观看", "date": "2026-08-22", "source_messages": ["最终确认的剧情"]}],
        "scenes": [],
    }
    rewrite = build_narrative_writer_prompt(
        mode="rewrite",
        title="《时光代理人》共同观看与推演",
        material_payload=materials,
    )
    assert "当前全部绑定材料" in rewrite
    assert "后续材料最终支持的版本" in rewrite
    assert "不复述猜错" in rewrite
    assert "不用“起初以为”" in rewrite
    assert "不要逐项罗列所有可能性" in rewrite
    assert "叙事对象本身是正文中心" in rewrite
    assert "不越过材料推测" in rewrite
    assert "不解释为什么几份材料属于同一卷" in rewrite
    assert "不用诗性升华" in rewrite
    assert "<material_payload_json>" in rewrite
    assert "<current_narrative_body>" not in rewrite
    for host_term in ("按钮", "上传", "diff", "CAS", "revision", "membership", "发布确认"):
        assert host_term not in rewrite, host_term

    update = build_narrative_writer_prompt(
        mode="update",
        title="一篇 Narrative",
        current_body="旧正文主体。",
        material_payload={"current_materials": materials["events"]},
    )
    assert "保留原稿主体" in update
    assert "<current_narrative_body>\n旧正文主体。\n</current_narrative_body>" in update
    _expect_value_error(
        lambda: build_narrative_writer_prompt(
            mode="update",
            title="一篇 Narrative",
            material_payload=materials,
        )
    )

    assert normalize_narrative_writer_result(
        {
            "evidence_sufficient": True,
            "body": "有来源的正文。",
            "issues": [],
            "self_review": REVIEW,
        }
    )["body"] == "有来源的正文。"

    insufficient_review = dict(REVIEW)
    insufficient_review["source_bound"] = False
    insufficient = normalize_narrative_writer_result(
        {
            "evidence_sufficient": False,
            "body": "",
            "issues": ["一份绑定原文缺失"],
            "self_review": insufficient_review,
        }
    )
    assert insufficient["evidence_sufficient"] is False
    _expect_value_error(
        lambda: normalize_narrative_writer_result(
            {
                "evidence_sufficient": False,
                "body": "擅自补出的正文",
                "issues": ["证据不足"],
                "self_review": insufficient_review,
            }
        )
    )
    asyncio.run(_verify_service())
    print("NARRATIVE_WRITER_OK")


if __name__ == "__main__":
    main()
