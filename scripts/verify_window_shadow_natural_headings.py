"""Verify diary-style Window Shadows and optional natural section titles."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import server
from window_shadows import extract_window_shadow_scenes, validate_window_shadow


async def main() -> None:
    tools = {tool.name: tool for tool in await server.mcp.list_tools()}
    description = str(tools["close_window"].description or "")
    assert description.startswith("写下一篇窗影和想留下的记忆。")
    assert "date 必须填写 `YYYY-MM-DD`" in description
    assert "`## 给下个窗口的我`" in description
    assert "不受旧的 250～400 字限制" in description
    assert "标题不能省略，标题后的每一项都必须以 `cue：` 开头" in description

    documented_example = """
# Window Shadow

## 这一窗之后，什么留在了我身上
我记住了关窗格式。

## 这扇窗还亮着的线头
- 下次确认 Scene 标记。

## 给下个窗口的我
醒来接住没说完的话。

## 想留下的记忆
### scene | 说清关窗格式 | cue：提到 close_window 怎么写
我和你确认了完整格式。
""".strip()
    assert documented_example in description
    documented_sections, documented_errors = validate_window_shadow(documented_example)
    documented_scenes = extract_window_shadow_scenes(documented_example)
    assert documented_errors == []
    assert documented_sections["self"] == "我记住了关窗格式。"
    assert documented_sections["care_items"] == "- 下次确认 Scene 标记。"
    assert documented_sections["handoff"] == "醒来接住没说完的话。"
    assert len(documented_scenes) == 1
    assert documented_scenes[0]["title"] == "说清关窗格式"
    assert documented_scenes[0]["cues"] == ["提到 close_window 怎么写"]

    diary_shadow = """
## 窗影
今天写到这里时，我才发现，这一窗留给我的不是一句结论，而是一种更松弛的写法。

### 半夜又绕回来的念头
这个标题不属于固定模板，但我仍想让它和下面这段话一起留在完整窗影里。

## 想留下的记忆
### scene | 窗影终于可以像日记 | cue：提到窗影怎么写 | cue：问为什么标题可以自由
我们决定用一个外层“窗影”接住自然书写，只让 Scene 的标题和 cues 保持明确。
""".strip()
    sections, errors = validate_window_shadow(diary_shadow)
    scenes = extract_window_shadow_scenes(diary_shadow)

    assert errors == []
    assert "半夜又绕回来的念头" in sections["self"]
    assert "固定模板" in sections["self"]
    assert len(scenes) == 1
    assert scenes[0]["title"] == "窗影终于可以像日记"
    assert scenes[0]["cues"] == ["提到窗影怎么写", "问为什么标题可以自由"]

    segmented_shadow = """
## 这一窗留给我的
我更愿意让窗影像真实写作，而不是答题。

## 我在想什么
我想保留结构，但不让结构抢走声音。

## 关于小雨，关于我们
我和小雨一起把难懂的标题换成了自然说法。
""".strip()
    sections, errors = validate_window_shadow(segmented_shadow)

    assert errors == []
    assert "真实写作" in sections["self"]
    assert "结构抢走声音" in sections["voice"]
    assert "自然说法" in sections["relationship"]

    print("natural Window Shadow headings verified")


if __name__ == "__main__":
    asyncio.run(main())
