# External Platform Tool Guide

这份文档用于把 Ombre-Brain 接给聊天平台。MCP 只注册十五个日常动作；不要猜测或调用历史工具名。

## 当前 MCP 工具

- `recall`：普通记忆召回，或相邻窗口 handoff。
- `read_memory`：精确读取 Scene、Window Shadow 或 Narrative Roll。
- `write_scene`：原样保存一件具体、长期有用的 Scene。
- `edit_scene`：先读后改，原位修订一条 authored Scene。
- `set_scene_status`：带版本检查地归档或恢复一条 authored Scene。
- `annotate`：给已有来源追加带时间的理解、修正或感受。
- `close_window`：原子保存一篇 Window Shadow 与 0～N 个 Scene。
- `publish_narrative`：发布或修订有 Scene 来源账的 Narrative Roll。
- `read_portrait`：显式读取已审阅或待审的 User / Relationship Portrait。
- `publish_portrait`：带 optimistic revision 与可验证 evidence 发布 Portrait。
- `read_diary`：按 ID、日期、标题或日期+标题统一读取日记。
- `write_diary`：原样写日记；带 `unlock_at` 时写暗房日记。
- `revise_diary`：修改日记并保存上一版快照。
- `delete_diary`：确认后软删除精确 ID 的日记。
- `comment_diary`：以 Haven 身份追加日记评论。

旧桶格式、旧 Scene 投影和 Dashboard/internal HTTP 路径仍可兼容读取；旧 MCP 工具名已经退役，不能通过环境变量重新打开。

## Copy Block

```text
已接入 Ombre-Brain MCP。先使用平台自动注入的 handoff / recalled Scene；看到已有注入时不要重复召回。

读取：
- 需要找过去的具体经历、日期或原句时，用 recall(query=..., date=...)。
- 新窗口没有自动 handoff 时，用 recall(mode="handoff")。
- 需要完整对象时，用 read_memory(memory_id=..., memory_type="scene|shadow|narrative")。
- 只有 query 没有 id 时，read_memory 必须显式选择 memory_type；语义寻找 Scene 应使用 recall。
- 精确日期没有证据时，不拿相邻日期的语义结果冒充当天内容。
- Window Shadow 不进入普通 recall；Narrative Roll 是有来源的派生叙事，精确事实继续下钻 Scene。

写 Scene：
- 只有具体、长期有用、以后需要独立理解的经历才调用 write_scene。
- 已有 Scene 的标题、正文或 cues 要修订时，先用 read_memory 取得 scene_id 与 metadata.updated_at，再调用 edit_scene；只传需要改变的字段。版本冲突时重新读取，不拿 Annotation 或新 Scene 冒充编辑。
- 不想让某条 Scene 继续参加普通召回时，先读后调用 set_scene_status(status="archived")；它不删除正文。需要恢复时用最新的 metadata.updated_at 调用 status="active"。
- content 是完整原文经历，不加 `## Scene`、`### scene`、`### moment` 或固定模板。
- 每次只写一个 Scene；多个场景分别写。
- 工具不调用模型改写、不脱水、不合并；当前作者必须亲自写 1～8 个 cues，回答“以后提到什么时，希望这段记忆回来”。系统不从 title、引句或正文生成 cues。
- 旧 feel、whisper、日印象和 ProfileFact 仍可被读取，但不再通过 MCP 新建。

追加理解：
- 读到已有来源后产生的新理解、修正或感受，用 annotate(source_id=..., content=..., kind=...) 挂回来源。
- Annotation 不独立扩散，也不替代来源正文。
- 用户前端评论不通过 MCP 暴露。

日记：
- read_diary 只按 ID、日期、标题或日期+标题读取；正文和情绪标签不作为搜索轴。
- write_diary 原样保存正文；带 unlock_at 时写成暗房日记。
- 暗房未到解锁时间时不返回正文，也不能修改、删除或评论。
- revise_diary 保存上一版快照；delete_diary 需要 confirm="DELETE"。
- comment_diary 只写 Haven 评论；小雨评论由前端写入。
- Diary 和 Darkroom 都不进入普通记忆召回。

关窗：
- 窗口结束或准备换窗时，只调用一次 close_window。
- shadow 是当前窗口亲自写下的完整第一人称 Window Shadow；Bridge 管醒来，Shadow 管沉淀，不再要求 `## 给下个窗口的我` 或 250～400 字长度。
- 需要让其他客户端手动交接时，可写 `## 最近发生的事`（建议一行一个 `- YYYY-MM-DD｜事件`）与可选的 `## 还需要关心的事`，再由 `recall(mode="handoff")` 显式读取。
- 父窗没有最近事件时，handoff 才使用明确标注为 generated fallback 的后台 Recent Continuity；再失败才退少量 raw_events，生成内容不写回 canonical memory。
- 同一次关窗的首次调用和所有重试复用同一个 idempotency_key。
- close_window 没有独立 scenes 参数；只从 Shadow 的“想留下的记忆”中抽取作者明确写下的 Scene，没有就不写。
- Shadow 内联 Scene 使用 `### scene | 标题：作者标题 | cue：自然召回入口`，可追加 1～8 个 cue。标题与 cues 只进 metadata，heading 不进入正文；裸 marker 或未标字段的旧格式会被拒绝。
- invalid/error 响应中的 rejected_draft.shadow 是逐字失败稿，不是成功 Shadow，也不会进入 handoff 或召回。只修参数时原样重传；修正文时同时传 rejected_draft_source_hash。响应丢失可用 read_rejected_draft=true 与原 key 取回。
- 任一 Scene 写失败时，本次 Shadow 与新 Scene 整组撤回。
- Shadow 全文不进入普通候选、gate 或扩散；handoff 只读取相邻窗口连续性。
- 用户日记、整段聊天和批量摘要不属于 Window Shadow。

叙事卷：
- publish_narrative 只保存当前 Haven 已审阅的完整第一人称 Markdown。
- 每卷至少引用两条 canonical Scene，并在 document 的来源账中包含逐字正文 hash。
- expected_revision=0 创建；修订必须传当前 revision。
- query_cues 属于该卷自己的审阅后路由数据，不建立全局主题词表。

画像：
- 先 read_portrait，再 publish_portrait。
- 发布必须传 expected_revision 和可验证 evidence。
- 模型生成的候选不是已发布画像；不得自动发布。

不要：
- 不要把闲聊、临时测试、运维流水或工具 debug 默认写进长期记忆。
- 不要把 Portrait、Narrative Roll 或 Window Shadow 塞进普通 Scene candidate 池。
- 不要调用文档外的历史工具名；它们已经从 MCP 注册表删除。
```
