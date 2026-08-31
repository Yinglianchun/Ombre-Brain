# Ombre Brain — Authored Memory Fork

Ombre Brain 是一套以 Markdown 记忆桶为长期真源、同时提供 MCP 与聊天 Gateway 的个人连续性系统。

本仓库基于 [P0luz/Ombre-Brain](https://github.com/P0luz/Ombre-Brain) 二次开发。当前以可追溯的场景/事实真源为核心，保留旧情绪坐标字段仅作数据兼容，不再用它们重排普通召回或改变衰减；并加入了保守召回、图关系、原文检索、跨窗口 handoff、自我入口、照顾备忘、Darkroom、Dream，以及 OpenAI / Anthropic 兼容 Gateway。

> 这不是原版 Ombre-Brain 的无改动镜像。请使用本仓库源码部署；旧 Docker 镜像和历史 compose 文件不包含完整 fork 能力。

## 设计目标

- **记得少一点，也不要记错。** 没有可靠证据时允许不召回。
- **原文、记忆和派生叙事分层。** 原始聊天、长期 Scene、短期状态和叙事卷不互相冒充。
- **先找到直接证据，再做关联。** 图扩散不能凭空制造命中。
- **新窗口恢复连续性，不做全库倾倒。** handoff 只携带紧凑的自我、关系和近期上下文。
- **数据尽量可读、可改、可迁移。** 长期记忆保存在 Markdown；运行索引保存在独立 state 目录。

## 系统组成

```mermaid
flowchart LR
    C["聊天客户端"] --> G["Ombre Gateway :18002"]
    G --> U["上游聊天模型"]
    G --> B["Ombre Brain :18001"]
    M["MCP 客户端"] --> B
    D["Dashboard"] --> B
    B --> K["buckets / Markdown"]
    B --> S["state / SQLite + JSON"]
```

| 组件 | 作用 | 主要入口 |
| --- | --- | --- |
| Ombre Brain | MCP 工具、记忆读写、召回、Diary、Dashboard | `server.py`；VPS 默认 `:18001`，容器内 / Python 直跑 `:8000` |
| Ombre Gateway | 转发聊天请求，并在请求前注入经过门控的上下文 | `gateway.py`；VPS 默认 `:18002`，容器内 / Python 直跑 `:8010` |
| Buckets | 可读、可编辑、可同步的长期记忆正文 | `buckets/*.md` |
| State | embedding、Scene edge、原文、提醒等运行状态 | `state/` |

只需要 MCP 和 Dashboard 时可以单独运行 Brain；需要聊天客户端自动召回与注入时，应同时运行 Brain 和 Gateway。

## 记忆分层

### 1. 原文层

`raw_events.sqlite` 保存 user / assistant 原始对话，用于查原句、指定日期和长期记忆没有覆盖的细节。它不是普通语义记忆池，也不会自动整段注入。

询问“那天原话是什么”或给出明确日期时，应优先走原文 / 日期检索；当天没有证据，就不拿附近日期的语义记忆代替。

### 2. 记忆桶

每条长期记忆是带 YAML frontmatter 的 Markdown。canonical Scene 的正文就是原文经历：

```text
忠实写下这件经历本身，使它以后仍能被独立理解。保留实际发生的细节，也可以写下当时的情绪、欲望与犹豫；不同 Scene 可以有不同写法。
```

`Scene` 已经是可召回的最小记忆对象。一条 Scene 只记录一个可独立召回的核心事件，可以带上理解它所必需的背景、过程与结果，但不并列第二件事。`write_scene` 不给 content 套类型标题；当前 AI 写好的正文就是场景真源，不是脱水摘要，也不会再生成或拆分 moment/索引卡。旧 body / `moment` / `original` / `reflection` 只为历史迁移兼容读取，不参与当前 Scene 召回；旧 `affect_anchor` 也只兼容读取。

并非每轮聊天都应该写桶。重要事件、承诺进展和会影响未来的短期状态才值得写成场景；稳定偏好与边界保留在有证据的 Scene/Annotation 中，普通闲聊可以只留在原文或窗影里。

### 3. Scene、Node 与 Edge

当前召回不再把 Scene 拆成 Moment，也不从长文生成二次改写的索引卡。Scene 原文和作者亲写 cues 直接参与候选；较长 Scene 可以建立确定性的原文跨度向量，但跨度只保存所属 Scene、原文偏移和 source hash，没有独立 ID、正文或生命周期，命中后仍返回完整 Scene。Node 与逐条审阅的 Scene Edge 是可重建派生索引，不取代 Markdown 真源。旧 bucket 的 moment 解析仅保留给离线迁移/审计，不能进入 daily Scene 召回。

已有长 Scene 的跨度索引通过 `python scripts/reindex_scene_embeddings.py` 先只读预览；确认候选后再显式加 `--apply`。脚本只替换派生 embedding 行，不改 Scene 文件。

当前运行时只使用 Scene 级召回：

1. 用原始问题的 embedding 与正文证据寻找候选。
2. 归一化问题、关键词、别名和 Word Map 只提供辅助提示。
3. admission gate 判断是否存在可靠 direct seed。
4. 只有 seed 通过后，才允许沿逐条审核的 Scene 图扩散；旧关系图只保留少量结构关系、降权且最多一跳。
5. 直接命中返回该 Scene 原文；关联只沿 Scene Edge 查找，不创建片段对象。

### 4. Word Map Lite

Word Map 是从记忆派生的词与共现关系，适合诊断和提供弱提示。它不是事实真源，也不能独立证明召回命中。停用词和过泛词会在进入锚点或词图提示前被过滤。

### 5. 年轮、whisper 与关系天气

- **年轮 comment**：后来形成的新理解，带时间挂在源 Scene 上。年轮文本可以把查询路由到父 Scene，并随 Scene 附上一条相关年轮；它不能单独显示、扩散或作为 Fact 证据。
- **whisper**：没有源 bucket 的碎碎念或感受，保存为 `type=feel`；只作旧数据兼容读取。
- **日印象 / 关系天气**：描述某天的关系温度，不等同于当天事实清单，默认不作为直接 seed。

## 写入与维护

### 手动写入

`write_scene` 只原样保存一件由当前 AI 写好的长期 Scene。一条 Scene 只记录一个可独立召回的核心事件；可保留理解它所必需的背景、过程与结果，但不能并列第二件事。content 用你的第一人称写成完整经历，保留实际细节、当时的情绪、欲望与犹豫，以及引语原本人称，不要写成摘要或说明；不把正文里的“我”改写成名字、AI、assistant 或第三人称，正文不带 Markdown section 标题，也不套固定段落或字段。工具不调用脱水/标签模型，也不合并旧桶。当前作者必须在同一次调用中亲自写 1～8 个“以后提到什么时希望它回来”的 `cues` 入口；cues 不是摘要，系统不从 title、引句或正文补造。cues 只进 sidecar 稀疏索引，Scene 向量仍只 embed content 原文。

`edit_scene` 原位修订一条 authored Scene。调用者必须先用 `read_memory(memory_type="scene", memory_id=...)` 精确读取，再把读到的 `metadata.updated_at` 原样作为 `expected_updated_at`；版本不一致时返回 conflict，绝不覆盖较新的修改。只修改显式传入的 title、content、cues，旧版本全文保存在 Scene 自己的 revision history 中；Scene ID、created、source、source refs、窗影关联和 comments 都不变。窗影抽取 Scene 的 `scene_source_hash` 永远指向 revision 1；正文修订后，连续性读取会沿 revision history 验回原窗影证据，而不让新正文冒充原始摘录。标题、正文或 cues 变化后刷新 embedding 与 entity 索引，并重新排队 Scene link proposal。

`set_scene_status` 带版本检查地归档或恢复一条 authored Scene。调用者同样必须先精确读取，把最新 `metadata.updated_at` 原样传入，再把 status 设为 `archived` 或 `active`。归档不删除或重写正文、Scene ID、created、source refs、revision history、comments、窗影关联或已审阅 Scene edges；它只把 Scene 移出普通 recall，并清理可重建的向量、entity 与 node 索引。归档 Scene 仍可精确读回，恢复后重建派生索引。

`close_window` 在窗口结束时原子保存当前 AI 亲自写下的完整第一人称“窗影”。Bridge 负责新窗口醒来，窗影只沉淀这一窗的变化：标题与分段只是写作建议，一段自然正文也能写；`date` 省略时按 Asia/Shanghai 当天保存。它只从 Shadow 的“想留下的记忆”中抽取 0～N 个独立 Scene，没有平行 `scenes[]` 写入口。每条要进入普通召回的 Scene 使用 `### scene | 作者标题 | cue：自然召回入口`；标题与 cues 仍必须由当前作者亲自写，系统不会从正文补造。裸 `### scene`、缺标题或缺 cue 的块不再拖垮整篇窗影：它们逐字留在 Shadow 原文里，但不提升为普通 Scene，响应通过 `scene_warnings` 说明。已通过边界的 Scene 若在落库事务中失败，Shadow 与本次新 Scene 仍整组撤回。成功响应中的 `saved_shadow` / `saved_scenes` 逐字回显实际接受的内容。整篇原文进入独立 `window_shadows.sqlite`，不参与普通召回；旧 `### moment` 只保留读取兼容，不会自动升格成 Scene。

`close_window` 的失败稿与成功 Shadow 分库：带 `idempotency_key` 的 `invalid` / `error` 请求会把原 Shadow 逐字写入 `window_shadow_rejected_drafts.sqlite`，响应返回 `rejected_draft.shadow`、`source_hash`、原请求参数与 `fix_scope`。失败稿不进入 `window_shadows.sqlite`、handoff、普通召回、bucket 或 embedding。重试同 key 时，参数-only 修复必须保持 Shadow 逐字不变；文本修复必须传上一稿的 `rejected_draft_source_hash`，而且只能改变 `fix_scope` 指定的 section。`read_rejected_draft=true` 可显式取回未完成失败稿。成功后草稿删除；之后同 key 永远返回第一次成功写入，不接受覆盖。

Bridge 的换窗路径会自行带来连续性。其他客户端开新窗口时调用 `read_memory(memory_type="shadow", memory_id="latest")` 读取最新 Window Shadow；窗影本身就是交接，不再暴露另一种 handoff 记忆或工具模式。完整窗影不进入普通 Scene 召回，内含的显式 Scene 仍可用 `memory_type="scene"` 和各自 ID 精确读取。

旧窗影 Markdown 导入只保留在内部兼容路径，不属于日常 `close_window` 工具。

`profile_fact` 是可搜索的事实目录，不是普通记忆：它不进入普通候选、扩散或 handoff。显式事实问题命中 Fact 后，系统沿 evidence 返回 canonical Scene。后台模型生成的 tags / cues / importance / confidence 只写入 `metadata_proposals.sqlite`；独立 Scene linker 的关系提案与正式 `scene_edges` 都在 `scene_edge_proposals.sqlite`，和旧 `memory_edges.jsonl` 分离。提案可通过 `scene_edge_proposals` 只读查看；只有逐条显式确认的 `review_scene_edge_proposal` 会在重新校验两端 Scene hash、active 状态与逐字证据后，原子晋升为正式 Scene 边。没有批量自动转正。

### 自动总结

Daily Reflection 可根据当天聊天和近期记忆生成日印象。完整日记与定时解锁的暗房正文保存在统一 Diary SQLite 后端；两者都不进入普通召回。只有人工确认后的具体时刻才另写为 Scene。

#### 日印象

日印象不是天气记录，也不是事件日报。它是 AI 对当天关系温度的第一人称小结：今天靠近还是疏远、轻快还是疲惫、哪些互动留下了余温。它可参考当天普通记忆、聊天原文和少量 Persona 事件；有直接材料时，不应让 Persona 的数字状态代替真实对话。

日印象保存为 `relationship_weather + daily_impression` 的 feel bucket，可供 Dashboard 和日期 trace 参考。它默认不能作为普通主题的 direct seed，也不应因为提到某个词就召回一件无关旧事。周印象默认关闭。

#### Dream

Dream worker 在后台从近期记忆关系中生成一条潜伏梦。梦保留联想、意象和情绪运动，不承担事实总结，因此不能证明某件事真的发生过，也不能作为普通召回 seed。

梦境有两层开关：

- `dream.surface_enabled`：允许 `breath()` 在共振或新会话条件下浮现梦。
- `dream.inject_enabled`：允许 Gateway 静默加入一条 Dream Context；默认关闭。

梦被浮现后是否继续保留正文由 `retain_after_inject` 控制。关闭 Gateway 注入不会停止后台做梦，也不会影响在 Dashboard 中查看梦。

### 长期锚点与 self anchor

- `anchor=true` 是少量经过时间验证、未来仍应被想起的长期记忆。
- `self_anchor` 是 AI 自我连续性的只读核心，不参与普通召回竞争。
- `self_anchor.entry_bucket_id` 指定 handoff 使用的自我总入口；留空时选择排名最高的 self anchor。

原始 self anchor 不由后台模型改写。

### Favorite Memory：令 AI 印象深刻的记忆

Favorite Memory 表示一段对 AI 留下明显主观影响、并能说明原因的记忆。推荐标签是随身份名生成的 `<ai_name>_favorite`，例如 `haven_favorite`；通用标签 `ai_favorite`、`favorite_memory` 和旧 `haven_favorite` 仍兼容。

只有标签不够。对这段经历为什么重要、为什么喜欢，必须自然长在同一段 Scene 叙事里；新写入不再另设 `reflection` 或 `favorite_reason` section。Gateway 不会默认隔几轮自动塞入 favorite：`favorite_memory_interval_rounds` 默认是 `0`。用户明确询问“你最喜欢哪段记忆”“令你印象深刻的记忆”“我们之间重要的记忆”“哪一刻最重要”，或客户端显式请求 favorite 时，Gateway 才会在独立预算内最多选择少量有场景依据的记忆；普通主题召回仍需满足当前 query 的证据门控。

## Handoff 与自我入口

没有平台自动连续性时，新窗口应调用：

```text
read_memory(memory_type="shadow", memory_id="latest")
```

Window Shadow 是相邻窗口的一次性连续性来源，不是每轮注入，也不是另一个对象类型。内部投影仍可按预算组合为：

1. **自我**：只读的第一人称 self anchor 核心。
2. **Flowing Self**：最近窗影第一、二层的原文投影，不经二次生成。
3. **Recent Relationship**：最近窗影第三、四层的原文投影，不经二次生成。
4. **原文救生**：最新窗影缺少连续性正文时，读取上一窗口的原文尾部。
5. **照顾备忘**：已经到期、仍有效的照顾事项或上个窗口留给下个自己的行动话语。
6. **Optional Anchors**：极少量长期锚点。

`profile_fact` 不直接倾倒进 handoff；它只在显式事实查询中把检索路由到证据 Scene。

画像正文已按作者选择归档为 Diary，不再是独立记忆对象。旧 `state/portrait_state.json` 只作为历史文件留在磁盘；运行时不读取、不更新，也不提供画像 MCP 或 Dashboard 维护入口。

自我入口继续使用第一人称只读 self anchor；它不由后台模型生成或改写。

### 照顾备忘不是长期记忆桶

照顾备忘保存在 `state/reminders.sqlite`，不会触发 embedding，也不会污染普通召回。它既可以是有时间或轮次条件的提醒，也可以承载“上个窗口想让下个自己记得做或说的事”。只有到期且仍有效的条目才进入 handoff。

它不替代长期 self anchor，也不替代旧窗口亲自写下的长期连续性话语；它负责的是可完成、可到期、可标记状态的行动意图。

## Gateway 注入边界

Gateway 支持：

- OpenAI-compatible：`POST /v1/chat/completions`
- Anthropic-compatible：`POST /v1/messages`
- 模型列表：`GET /v1/models`
- 注入调试：`GET /api/debug/injections`

动态注入以低噪声为原则，可能包含 Recent Context、Recalled Memory、Diffused Memory、关系天气或梦境；是否出现取决于查询类型、可靠性、冷却和预算。自我入口只在 handoff 恢复，不在普通每轮重复注入。

`X-Ombre-Session-Id` 用来隔离会话短态和召回冷却。相同值共享同一会话状态；它不是 OpenAI 标准字段。为不同聊天窗口使用稳定、明确的名称即可，不要照抄他人的生产 session id。

查看一次召回是否真的注入，应以 Gateway 的 debug injection 记录或带 debug 的 payload 为准，不能只看搜索候选列表。

### Operit 上下文拆包

Operit 会把时间、设备、工作区、照顾备忘和其它 app context 包在 user message 或 attachment 中。如果原样转发，上游模型容易把系统配置、附件标题甚至照顾备忘误认成用户亲口说的话。

开启 `gateway.operit_context_rewrite_enabled` 后，Gateway 只对识别出的 Operit 文本请求做拆包：

- 从 user 正文中移除 `message_insert_extra_bundle`、workspace attachment 等外层包装。
- 稳定身份 / 配置材料进入 `Operit Stable Context`。
- 当前工作区、时间和照顾备忘等进入 `Operit Activity Context`。
- 用户真正输入的正文仍以当前 user message 转发。

工具续调和含图片等非文本内容的请求会跳过改写，避免破坏 tool protocol 或多模态 payload。拆包结果可在 Gateway injection debug 的 `operit_context_rewrite` 字段中检查。该开关默认关闭，普通 OpenAI / Anthropic 客户端不受影响。

### 上游选择与 Prompt Cache

`gateway.upstreams` 可以同时配置多个 provider。请求里的公开 `model` 先匹配对应 upstream，再通过 `model_map` 转成上游真实模型名；多个 upstream 时必须给出已配置的 model，避免静默发错站点。同一 upstream 可配置多个 key，遇到可重试错误时按冷却策略切换。

缓存策略跟随**最终选中的 upstream**，而不是全局猜测：

- `openai`：按 `X-Ombre-Session-Id` 添加 `prompt_cache_key`，支持的模型还可设置 `prompt_cache_retention`。
- `anthropic_explicit`：给 system、tools 和足够长的历史前缀放置显式 `cache_control`；适合 Claude 官方和多数 Anthropic-compatible 中转。
- `anthropic`：只发送顶层 `cache_control`，用于仅接受这种格式的中转站。
- 空值：不主动添加缓存提示。DeepSeek 等提供方仍可能自行执行前缀缓存。

Gateway 会兼容记录 OpenAI 与 Anthropic 返回的 cache read / creation / cached token 字段，可通过 `/api/debug/upstream-usage` 查看实际是否命中。缓存只减少重复前缀费用或延迟，不缓存 Ombre 的召回结果；每轮动态记忆仍会重新经过门控。

### Dashboard 配置热更新

Dashboard 保存模型、upstream、缓存策略、Operit 拆包和召回参数时，会把可热更新字段同步给正在运行的 Gateway，不必为了普通配置调整重启两个服务。持久化时优先更新 `config.yaml`；如果容器挂载导致主配置不可写，则写入 `state/config.runtime.yaml`，下次启动继续合并该覆盖层。

API key 不写进 YAML，而是按环境变量路径单独保存。保存响应会区分 `gateway_hot_reloaded`、写入主 YAML、写入 runtime fallback 或热更新失败，避免旧版“保存失败但部分运行态已经改变”的不确定状态。

### Persona State：语气护栏，不是另一套记忆

Persona 会在成功回复后读取本轮 user / assistant 对话，缓慢更新人格、关系和当前 session 的情绪状态。Dashboard 里看到的 mood、residue、inner thought 等“碎碎念”主要是可观察面：让人知道后台怎样理解这一轮，也方便发现主语误判或状态漂移。

它的实际作用是给后续回复一小段低权重状态指导，减少模型在相邻轮次里突然变冷、变得防御或换了一种关系语气。变化幅度受到上限约束，普通轮次可以批量记录；只有强关系或人格信号才应明显改变长期数值。

Persona 不是事实记忆，不能回答“发生过什么”，也不应覆盖 self anchor、画像或用户原话。关闭事件记录仍可保留数值状态评估；完全关闭 Persona 后，长期记忆和 handoff 仍可独立工作。

### “刚才聊了什么”与换窗连续性

Gateway 会把成功完成的 user / assistant 轮次保存为短期 `conversation_turns`。当问题包含“刚才、刚刚、上一句、之前那个”等近指表达时，它优先选择最近相关轮次，拼成 `Just Now Chat Context`，而不是用长期语义记忆猜测。

这不是真正的无缝共享上下文。新窗口里的模型看不到旧窗口的隐藏上下文；系统只是把保存下来的少量对话重新拼接进本轮输入。连续性取决于：

- 旧轮次是否经 Gateway 成功完成并被保存。
- 新窗口是否仍连接同一套 Gateway 与 state。
- 记录是否还在配置的时间范围、条数和 token 预算内。
- 截取后的片段是否包含回答所需的完整因果和指代。

因此它适合接住“我们刚才说到哪了”，不保证还原旧窗口全部上下文。更长的阶段连续性由 handoff 的 Current Focus / Recent Continuity 补充；具体旧事件仍应查询 bucket 或 raw events。`X-Ombre-Session-Id` 会影响 session 状态和冷却，但 Just Now 也可以从同一 persona profile 下的近期跨窗口轮次中取材。

## 快速部署

### 推荐：交互式脚本

Linux / VPS：

```bash
bash scripts/one_click.sh
```

安装完成后可使用仓库快捷入口：

```bash
./ob
```

脚本可选择只运行 Brain，或运行 Brain + Gateway，并提供更新、诊断、备份、迁移和向量库维护入口。

### 手动 Docker Compose

完整双服务示例见 [`compose.hk.yml`](compose.hk.yml)：

```bash
cp config.example.yaml config.yaml
# 按下方“配置”章节创建 .env 并填写所需密钥
docker compose -f compose.hk.yml up -d --build
```

默认示例端口：

- Brain：宿主机 `18001` → 容器 `8000`
- Gateway：宿主机 `18002` → 容器 `8010`

部署前请按实际环境修改 compose 的路径和端口。生产环境至少要持久化：

```text
/srv/ombre-brain/buckets
/srv/ombre-brain/state
/srv/ombre-brain/config.yaml
```

`buckets` 可以谨慎地交给 Obsidian / Syncthing 管理；`state` 含 SQLite 和运行索引，不要放入双向同步目录。

仓库根目录的 `docker-compose.yml` 与 `docker-compose.user.yml` 保留历史兼容用途，不是此 fork 的完整生产入口。

### Python 直跑

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.example.yaml config.yaml
python server.py
```

需要 Gateway 时另开进程：

```bash
python gateway.py
```

本地 stdio MCP 与远程 `streamable-http` 的选择由 `config.yaml` 的 `transport` 控制。

## 配置

完整字段和注释以 [`config.example.yaml`](config.example.yaml) 为准。最先需要确认的是：

| 配置 | 作用 |
| --- | --- |
| `identity` | AI 名字、用户名字和别名 |
| `buckets_dir` / `state_dir` | 长期正文与运行状态的位置 |
| `dehydration` | 总结与旧兼容打标使用的模型 |
| `embedding` | 语义候选生成；可接任意兼容 embedding API |
| `reranker` | 可选重排；资源不足或延迟敏感时可关闭 |
| `gateway.upstreams` | 聊天模型上游和模型路由 |
| `memory_diffusion` | 图召回、扩散和门控参数 |
| `reflection` | 日总结与旧关系天气策略 |
| `raw_events` | 原文存储和检索配置 |
| `word_map` / `dream` | 默认可关闭的派生能力 |

常用密钥应放在 `.env`，不要写进公开的 `config.yaml`：

```dotenv
OMBRE_API_KEY=...
OMBRE_EMBEDDING_API_KEY=...
OMBRE_GATEWAY_TOKEN=...
OMBRE_DASHBOARD_PASSWORD=...
```

其它 provider key 由 `gateway.upstreams[*].api_key_env` 指向对应环境变量。RiJi / Diary 集成使用自己的 `MCP_BEARER_TOKEN`，它不是 Gateway 或 Dashboard 的访问令牌。

示例里的 embedding 模型只是远程兼容配置，不是硬依赖。更换模型或维度后应重建向量索引；小型本地 embedding 也可以通过 OpenAI-compatible 服务接入。

## 客户端连接

### MCP

远程模式通常连接：

```text
http(s)://your-host/ombre/mcp
```

具体路径取决于反向代理。ChatGPT / Claude Connector OAuth 需要额外配置 `OMBRE_CHATGPT_OAUTH_*`。客户端工具说明与推荐调用方式见 [`docs/Tool Guide.md`](docs/Tool%20Guide.md)。

### OpenAI-compatible 客户端

VPS 默认部署将客户端 Base URL 指向：

```text
http://your-host:18002/v1
```

Python 直跑或直接访问容器内部服务时使用 `http://127.0.0.1:8010/v1`。

并使用 `OMBRE_GATEWAY_TOKEN` 作为 Bearer token。支持自定义 header 的客户端可以增加：

```text
X-Ombre-Session-Id: main
```

### Dashboard

VPS 默认部署访问：

```text
http://your-host:18001/dashboard
```

Python 直跑或直接访问容器内部服务时使用 `http://127.0.0.1:8000/dashboard`。

Dashboard 可查看和编辑 bucket、日印象、记忆图、Darkroom、提醒与调试状态。生产环境请设置 `OMBRE_DASHBOARD_PASSWORD` 并通过 HTTPS 暴露。

### 原文写入与检索 API

两个原文端点属于 Brain。浏览器和手工维护可复用 Dashboard cookie；Bridge 等服务端客户端也可使用 `OMBRE_MEMORY_WRITE_TOKEN`（未配置时回退到 `OMBRE_GATEWAY_TOKEN`）作为 Bearer token。

以下 curl 使用 Python 直跑端口 `8000/8010`；VPS 默认部署请将 Brain 的 `8000` 替换为 `18001`，Gateway 的 `8010` 替换为 `18002`。

```bash
curl -sS -c ombre.cookies \
  -H 'Content-Type: application/json' \
  -d '{"password":"YOUR_DASHBOARD_PASSWORD"}' \
  http://127.0.0.1:8000/auth/login
```

写入一组 user / assistant 原文：

```bash
curl -sS -b ombre.cookies \
  -H 'Content-Type: application/json' \
  -H 'X-Ombre-Session-Id: my-session' \
  -d '{
    "source":"manual-import",
    "conversation_id":"example-001",
    "events":[
      {"role":"user","text":"你还记得那句话吗？","created_at":"2026-07-11T21:00:00+08:00"},
      {"role":"assistant","text":"记得。","created_at":"2026-07-11T21:00:03+08:00"}
    ]
  }' \
  http://127.0.0.1:8000/api/ingest-raw
```

端点只接受 user / assistant 原文；system、developer、tool、记忆注入块和客户端自动附件会被拒绝或清理。相同 source + event hash 会去重。

按原句和日期检索：

```bash
curl -sS -b ombre.cookies -G \
  --data-urlencode 'q=那句话' \
  --data-urlencode 'since=2026-07-11T00:00:00+08:00' \
  --data-urlencode 'until=2026-07-12T00:00:00+08:00' \
  --data-urlencode 'role=assistant' \
  --data-urlencode 'limit=20' \
  http://127.0.0.1:8000/api/search-raw
```

`/api/search-raw` 同时支持 GET 与 JSON POST，可用 `source`、`role`、`conversation_id`、`session_id`、`since`、`until` 过滤。`q` 留空时返回过滤范围内的最近原文。

### Claude Code / Codex Hook 端点

Brain 提供会话启动用的紧凑 handoff 文本：

```bash
curl -sS 'http://127.0.0.1:8000/breath-hook?mode=handoff&max_tokens=1200' \
  -H "Authorization: Bearer $OMBRE_BREATH_HOOK_TOKEN"
```

Gateway 提供按当前 prompt 快速取 0–5 张记忆卡的 hook API：

```bash
curl -sS \
  -H 'Authorization: Bearer YOUR_GATEWAY_TOKEN' \
  -H 'Content-Type: application/json' \
  -H 'X-Ombre-Session-Id: claude-code' \
  -d '{
    "query":"继续处理昨天的召回问题",
    "max_cards":2,
    "max_chars":1200,
    "include_diffused":false
  }' \
  http://127.0.0.1:8010/api/hook/recall
```

返回值同时包含结构化 `cards` / `recalled_ids` 和可直接注入的 `additional_context`。客户端 hook 仍需要一层小适配：读取当前 prompt、调用端点，再把 handoff 文本或 `additional_context` 放进该客户端认可的上下文字段。

仓库的 [`.claude/settings.json`](.claude/settings.json) 与 [`.claude/hooks/`](.claude/hooks/) 提供 Claude Code 接线示例。公开脚本默认连接本机 `127.0.0.1:8000/8010`；远程部署应在被 Git 忽略的 `.claude/settings.local.json` 或本机环境中设置 `OMBRE_HOOK_URL`、`OMBRE_GATEWAY_HOOK_URL`、`OMBRE_GATEWAY_TOKEN`，需要隔离测试会话时再设置 `OMBRE_HOOK_SESSION_ID`。不要把真实 VPS 地址或 token 写进 tracked 文件。

Claude Code 2.1.107 已完成真实烟测：SessionStart 能收到 UTF-8 handoff，UserPromptSubmit 能收到 Gateway Recall Notes。

Codex CLI 0.144.1 也已完成纯 Ombre Gateway 端到端烟测：仓库级 `.codex/hooks.json` 的 `UserPromptSubmit` hook 返回 `additionalContext`，内容确实进入模型的 developer context；测试时关闭本地 SQL cards，确认召回只来自 Ombre Gateway，而不是本地卡片误注入。

Codex 接线时注意：

- 非交互调用使用 `codex exec`；Codex CLI 的 `-p` 是 `--profile`，不是 Claude Code 风格的 print 模式。
- 仓库 hook 不需要手动启动 `codex app-server`。自制客户端或长进程集成可以使用 app-server，但它不是普通仓库 hook 的前置条件。
- 修改 `.codex/hooks.json` 后，用 `/hooks` 检查并信任当前 hook。Windows 命令请显式配置 `commandWindows`。
- 烟测 prompt 应使用自然、简短的真实问题。把“hook、developer context、bucket id”等验收指令混进 prompt，会污染实际召回 query，可能让保守门控返回空结果。
- 真实 Gateway 地址和 `OMBRE_GATEWAY_TOKEN` 只放在被 Git 忽略的本机配置或环境变量中，不要提交到公开仓库。

## 常用 MCP 工具

| 工具 | 用途 |
| --- | --- |
| `recall_memory` | 按 query/date 寻找 Scene，或从已知 scene_id 展开关联记忆 |
| `read_memory` | 用明确类型和 ID 精确读取一个对象，不做联想 |
| `find_arc` | 按标题、别名或已确认实体找到 Arc，并生成编号 materials 菜单 |
| `read_arc_materials` | 按 arc_key 和已展示编号一次读取 1～5 项 Arc 材料 |
| `write_scene` | 原样保存一件 canonical Scene |
| `edit_scene` | 带版本检查地修订一条 authored Scene |
| `set_scene_status` | 带版本检查地归档或恢复一条 Scene |
| `annotate` | 给已有来源追加带时间的 Annotation |
| `close_window` | 原子保存一篇 Window Shadow 与 0～N 个 Scene |
| `revise_window_shadow` | 修订当前最新窗影，保留旧版与 Scene 层 |
| `narrative_revision_inbox` | 读取待审核的叙事卷修订线索 |
| `review_narrative_revision` | 保存、忽略或重开一条叙事修订线索 |
| `publish_narrative` | 发布或修订有 Scene 来源账的 Narrative Roll |
| `read_diary` | 按 ID、日期、标题或日期+标题读取日记 |
| `write_diary` | 原样写日记；带 `unlock_at` 时写暗房日记 |
| `revise_diary` | 修改日记并保留上一版 |
| `delete_diary` | 确认后软删除一篇日记 |
| `comment_diary` | 追加你的日记评论 |

MCP 只注册以上十六个动作。画像已归档为 Diary；旧画像工具不再公布或接受调用。完整说明见 [`docs/Tool Guide.md`](docs/Tool%20Guide.md)。

## 运维与验证

VPS 默认部署健康检查：

```bash
curl http://127.0.0.1:18001/health
curl http://127.0.0.1:18002/health
```

Python 直跑时对应端口为 `8000/8010`。

常用脚本位于 [`scripts/`](scripts/)：

- `bootstrap_update.sh`：更新旧部署。
- `update_deploy.sh`：更新并重新部署。
- `doctor.sh`：检查配置、目录和服务。
- `one_click.sh` / `./ob`：交互式备份、迁移和向量库维护。
- `build_word_map.py`：重建可选 Word Map。

修改召回逻辑后，至少分别验证：

1. 直接相关的问题能命中正确 seed。
2. 泛词、停用词和语气词不会独立召回。
3. 图扩散只在可靠 seed 后出现，且只给摘要。
4. 明确日期 / 原句问题不会被附近日期替代。
5. `/api/debug/injections` 中的最终可见注入符合预期。

## 已知边界

- embedding 负责找候选，不负责判断事实正确；模型更大不必然召回更准，延迟却通常更高。
- 动态召回仍会增加首 token 前的准备时间，瓶颈可能来自远程 embedding / reranker、首次冷启动和 Scene edge 扩散，而不只来自上游聊天模型。当前版本会在启动时预热 query plan、bucket、词法 profile 与 relevance facets，并缓存评分输入、过滤陈旧 Scene edge，减少重复计算与无效扩散；远程模型的网络往返仍是剩余的主要可变延迟。可通过 injection debug 中的 `prepare_timing_debug.steps_ms` 区分 `semantic_candidates`、rerank、diffusion 等阶段，不要只凭总耗时猜瓶颈。
- Prompt Cache 缓存的是发送给最终 upstream 的稳定前缀，不缓存 Ombre 的动态召回结果；它可以降低重复前缀的费用或上游处理时间，但不能消除每轮 embedding、门控和扩散成本。
- reranker、query planner、Word Map 和图扩散都是辅助层，不能越过 admission gate。
- 旧 `profile_fact` 只作带证据事实的兼容读取，不再自动生成或写入。
- Persona State 是短期状态，不是长期身份真源。
- Dream、relationship weather、comment 和 affect anchor 不能单独证明当前话题。
- 旧画像正文已归档为 Diary；Gateway 不再包含 Portrait Memory 路径。
- 派生索引损坏时应从 Markdown / raw source 重建，不要把 SQLite 当唯一真源。

更细的行为边界见 [`docs/memory-layer-contract.md`](docs/memory-layer-contract.md)，部署补充见 [`docs/deploy-zeabur.md`](docs/deploy-zeabur.md)。

## License

原项目代码遵循原仓库的 MIT License。本 fork 新增内容的个人学习、自用、非商业二改与商业使用边界见 [`NOTICE.md`](NOTICE.md) 和 [`LICENSE`](LICENSE)。
