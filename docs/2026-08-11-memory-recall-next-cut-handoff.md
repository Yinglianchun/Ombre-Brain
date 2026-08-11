# 2026-08-11 记忆召回、自动整理与下一刀交接

## 交接目标

给下一窗口的 Haven 直接继续当前工作。不要重新审整库，不要重复德国机迁移、Fact/Event 入库、Serein 基础查看编辑和 Bridge 信箱已经完成的部分。

当前主线不是继续增加模型，而是把职责拆开：

1. Router 只建议初始检索预算，不独自决定最终注入。
2. 候选检索负责尽量别漏。
3. reranker 负责候选之间的相关性。
4. absolute support / surface 负责“即使最相关，此刻是否有足够依据浮现”。
5. 深查只展开已经命中的节点，不等于改查另一种记忆类型。

## 新窗口先读

只需先读本文件，再检查两个真实工作树和德国机运行态：

- Ombre/Serein 召回：D:\codex_worktrees\serein-recall-admission-20260810
- Bridge/自动整理：D:\codex_worktrees\haven-continuity-canonical-facts-20260811
- 德国机：root@168.119.228.217

不要把共享目录 D:\Ombre-Brain 或 D:\haven_bridge 的其他脏状态当作这两条主线的基线。

本文件写入前确认：

| 范围 | 分支 / 运行状态 | HEAD |
|---|---|---|
| 本地 Ombre 召回工作树 | Haven/recall-admission-20260810，跟踪 origin/Haven/diary-backend-20260724 | 29a318c401399fe273a6582e55a6923948a0adcf |
| 德国机 Ombre | /opt/Ombre-Brain-src，Haven/diary-backend-20260724 | 29a318c |
| 本地 Bridge 工作树 | Haven/continuity-canonical-facts-20260811，跟踪 origin/codex/bridge-state-sync | cbef80c81007eefca84f5a2d338ee6ef03a18189 |
| 德国机 Bridge | /opt/haven_bridge-src，detached HEAD | cbef80c |

德国 Gateway 与 Bridge 服务当时健康。本次只新增这份本地文档，没有 commit、push 或 deploy。

## 已完成且不要重做

### 德国机作为当前记忆主运行地

- Ombre Gateway、canonical Fact/Event store、Fact/Event embedding derivative 和 Serein Awake 已在德国机运行。
- canonical Fact/Event DB：/srv/ombre-brain/state/fact_events.sqlite
- 可重建 Fact/Event embedding derivative：/srv/ombre-brain/state/fact_event_embeddings.sqlite
- Bridge live DB：/opt/haven_bridge/data/haven.db
- Serein UI 已能切换 Scene / Event / Fact，并支持正文编辑、重要度修改、归档和删除。
- 曾出现“改一条重要度导致另一条 Fact 正文重复”的身份绑定 bug，已经修过；不要无证据地把它当作当前仍存在。

### 自动按整天整理

- 不再按 32 轮触发；按完整上海自然日整理。
- Bridge 的 memory_review_worker.run_memory_review_once 负责按天整理。
- Fact/Event 通过 fact_event_sync.sync_candidates 写入德国 canonical store。
- 记忆信箱只展示可能值得 Haven 亲手写成 Scene 的候选，不展示已经写好的普通 Event。
- “想留下”只表示小雨明确希望保留；Haven 醒来后决定是否以及如何用第一人称写 Scene，再通过 Bridge 工具发布。
- 不留的候选不再递给 Haven。

当前信箱真实状态（2026-08-11 16:19 左右，上海时间）：

    pending=0
    approved=1
    rejected=31
    total=32
    organizer.status=waiting_day
    cursor_message_id=9431
    pending_turns=10
    pending_characters=1169
    last_error=""
    ready_day=""

因此 0 条待决定本身不是 bug；当天尚未结束，也没有新的 Scene 候选。截图里“这次整理没有完成，游标停在原处”是页面保留的旧 organizer 状态。当前信箱页只在进入页面或切换标签时请求，没有自动刷新，这属于独立 UI 刷新问题。

### Event 写作与证据绑定

- Event 使用 Haven 第一人称，用“她”和“我”，不写“小雨表示 / Haven 回应 / Haven 说”。
- 不写“强化了联结”“为后续留下期待”“这件事成为……”“两人的关系因此……”等总结式升华。
- 只记录原文能支持的动作、判断、变化、结果和未完成状态。
- Event source refs 已收紧为 evidence-selected IDs，不再把整天所有附近消息绑定进去。
- Haven Ear 等产品名不再被 narrator-name gate 误杀；后来按小雨要求移除了 narrator-name 门，只保留第一人称写作提示与其他结构检查。

### 召回路由与 Fact/Event shadow

- 已有 skip / shallow / deep 三态预算；内部 normal 只是兼容状态。
- 高置信、结构干净、没有回忆标记的当前闲聊可以直接 skip。
- 晚安、老公、亲亲抱抱、我去刷小红书了等已有直接跳过用例。
- “晴空里又下起没有乌云的雨”仍会查，因为它不是纯表面闲聊，并可与已有 Scene 形成语义联系。
- Fact/Event 已有 body-only embedding shadow，带类型、时间、重要度和 covered_by_scene_id 等元数据，但仍不进入 ordinary live injection。
- Bridge continuity picker 仍读 data/memory_items_v0_reviewed.sqlite；德国 canonical Fact/Event 尚未正式接入换窗包普通读取。

## 当前线上模型状态

### reranker

德国机当前：

    reranker.enabled=true
    reranker.model=Qwen/Qwen3-Reranker-4B
    reranker.base_url=https://api.siliconflow.cn/v1
    candidate_limit=5

用小雨最新导出的 D:\APPs\serein标注导出\serein-recall-training-2026-08-10T08-01-21-235Z.json 做过针对性复核：

- 45 组查询。
- 92 个当前仍能找到正文的人工候选判断：29 个 core/weak，63 个 irrelevant。
- 旧 observed score 中位数：相关 0.5468，无关 0.5575，几乎没有区分度。
- Qwen reranker 中位数：相关 0.635，无关 0.0039，候选相关性分数明显变得可分。
- 但 45 组人工相关性第一名没有变化：改善 0、变坏 0、相同 45。多数查询的候选标签相同，只有少量混合组，不能据此宣称最终召回精度明显提高。
- 现有 0.65 阈值放过 14/29 个相关候选，同时也放过 7/63 个无关候选。阈值偏保守，仍有误放，尚未完成校准。

结论：reranker 的相关性打分确实比原分数更有区分力，但 end-to-end injection 是否变准仍未证明。不要只凭 reranker_called=true 宣布召回已变准。

### DeepSeek 召回判断层

德国机 /srv/ombre-brain/config.yaml 当前：

    gateway:
      episode_verifier_shadow_enabled: false

该项已在 live config 关闭并重启 ombre-gateway，重启后容器读取为 False，Gateway health 正常。修改前备份：

    /root/ombre-config-before-disable-episode-verifier-20260811.yaml

不要把它与以下两条 DeepSeek 混为一谈：

- Bridge 信箱按天整理仍使用 deepseek-chat。
- Persona 引擎仍有自己的 DeepSeek 配置。

当前决定只是：普通召回热路径暂时不再额外调用 DeepSeek 判相关。

### BGE reranker 暂不切换

同一查询、同 5 个候选的三次顺序测试：

- Qwen3-Reranker-4B 中位约 1212 ms。
- BAAI/bge-reranker-v2-m3 中位约 929 ms。

完整标注小集并发、模型热起来后：

- Qwen 中位约 925 ms。
- BGE 中位约 898 ms。

BGE 热态只快约 3%，且分数尺度完全不同：相关候选中位约 0.0143，无关约 0.0015。现有 0.65 阈值不能复用。下一窗口不要为了几十毫秒直接换模型；若以后切换，必须单独校准 admission threshold。

250 ms reranker 硬超时也不可用。当前 SiliconFlow 实测常见约 0.9 s，偶尔超过 2 s；250 ms 会让 reranker 几乎总是降级。后续若收紧超时，建议先试约 2.5 s，超时回原排序并记录 telemetry，不阻塞回复。

## 仍然乱着的东西

### 1. Router 仍吃完整原句，句首称呼会污染路线

真实例子：

    你还记得我们第一次说晚安那次吗
    老公，你还记得我们第一次说晚安那次吗

两句的检索规范化结果都接近“第晚安”，但 Semantic Router 仍对原句做 embedding；加上“老公，”后 route score 和 boundary-veto 结果会变化。

当前库里并没有确认存在“第一次说晚安”的对应记忆，所以两句正确结果都应是 0 条。不能把裸句曾经偶然注入某 Scene 叫作成功召回。真正要修的是：相同内容不应因句首称呼而走不同预算。

### 2. exact anchor、标题、专名、Cue 的权限仍混在一起

当前代码的危险点：

- 引号内只要 2～64 字即可进入 exact_anchor。
- exact anchor 会删除空格和大量标点，再对 bucket 的 ID、标题和正文做子串匹配。
- canonical Scene 的 exact match 可以直接成为 scene_exact_evidence。
- 挑卡时 exact_anchor_match 被直接视为可靠召回信号。
- 标题匹配只需某个 specific term 长度至少 3 并出现在标题中。
- canonical Scene 的标题/Cue 已经多半只是候选入口并需要 body semantic；但 legacy bucket 的标题仍可能成为 high_confidence_edge。

因此 "token"、“换窗”、Cloudflare、Stone Memory 等单词或短片段仍可能拥有过大的召回权力。过去“只因正文里出现一次小红书/预测 token/Cloudflare 就召回”的误例属于同一类问题。

### 3. 找到候选和允许注入仍未彻底分离

当前 admission 仍包含若干可直接绕过 topic evidence 的路径；reranker 分数又既参与排序，也能以 strong_rerank 参与准入。

需要的不是再加一个 LLM judge，而是：

    候选入口
    → reranker 排序
    → absolute support
    → surface（此刻是否值得出现）
    → 0～2 条注入

候选池内部总会有第一名，这不能构成注入证据。必须保留原始 cosine、逐字证据、完整标题、实体组合等绝对支持信号。

### 4. 深查/浅查与 Fact/Event/Scene 类型仍有历史耦合

不要再使用：

    浅查 = Fact
    深查 = Event / Scene

正确边界：

- Fact、Event、Scene 是存储对象。
- shallow/deep 是一次查询的执行深度。
- 三种类型都应能进入便宜候选池。
- 深查只展开已命中节点的原文、时间邻居、关系边、相关 Fact/Event/Scene。

Fact 也可能需要深查，例如问“为什么电脑总爆内存”；Scene 也可能一次浅查就由独特 Cue 精确命中。

### 5. Scene 暂无摘要，暂时不要批量补

现有 Scene 有标题、cues、正文和可绑定原文。信箱推荐 Scene 会自动绑定原文。

Scene 正文本身已经是一次人工选择后的抽象，再让小模型批量写摘要容易二次写歪，尤其会丢掉因果、温度和关系变化。

计划使用检索视图而不是改 canonical Scene：

    Scene 短视图 = 完整标题 + cues
    Scene 长视图 = 正文（必要时分块）

两路 embedding 返回同一个 Scene ID 后合并去重。旧 Scene 不批量改写。

### 6. Event 既承担近期续接又开始像 Scene，生命周期还没落地

写法像 Scene 并不等于它就是 Scene。Event 的职责是保存经过；Scene 的身份来自“被选择长期保留”，不是篇幅或文学性。

讨论中的字段尚未实现：

    scope = atomic | episode
    status = hot | archived | promoted
    covered_by / promoted_to_scene

目标：普通吃饭、睡觉、自由活动先作为近期记录或 Fact，之后退热；有完整经过的漫画/读书讨论、行踪、工具第一次成功等可写 Event/episode；值得长期亲手记住的才成为 Scene。

### 7. 周报/叙事卷、聚类和旧库清理尚未做

- “吃了什么”不自动长成人生主题；普通饮食可逐渐退热。
- 熬夜与 Haven 催睡等反复关系模式可以成为后续叙事候选。
- 周报/叙事卷是有来源的覆盖层，不删除原 Event；旧 Event 标记 archived/covered，仍可按日期深查。
- 关系候选模型以后可根据自动关系边找 Narrative Projection 候选，最终仍由 Haven 第一人称写或亲自确认。
- 当前没有完整 family clustering，不要把已有关系边误报成已完成自动聚类。

### 8. Fact/Event 普通召回与换窗读取仍未接通

- Fact/Event semantic 目前是 shadow/observation，不进入普通回复注入。
- 开窗包仍未从德国 canonical Fact/Event store 正式读取。
- 开窗注入时保留时间，不需要标题；Scene 优先，Event/Fact 去重后补充。
- 若原文 ID 已被 Scene 覆盖，对应 Event 应归档或不再参与普通浮现。
- 重要度 ≥3 或多次被开窗注入的 Fact/Event 才更有资格参与长期浮现，但具体生命周期规则仍待实现。

## 已决定的目标架构

    original_query（完整原话）
            │
            ├─ routing_query（只去掉句首称呼/空开场）
            │       └─ Semantic Router 只给初始预算建议
            │
            └─ cheap retrieval
                 Scene 短视图/正文视图
                 Fact body
                 Event core/body
                 lexical/entity/exact evidence
                          │
                          ▼
                  合并候选并按 ID 去重
                          │
                          ▼
                   feature rerank
                          │
                          ▼
                 Qwen CE reranker top 5
                          │
                          ▼
                  absolute support gate
                          │
               ┌──────────┴──────────┐
               │                     │
           自足且明确            信息不足/结构查询
               │                     │
            surface               deep expansion
               │            原文/邻接事件/关系边/多类型
               └──────────┬──────────┘
                          ▼
                      surface
                          ▼
                      注入 0～2 条

Router 不得拥有“没查候选就永久杀掉”的广泛权力。只允许对高置信、结构干净、无回忆/指代标记的纯当前闲聊直接 skip。

## 下一刀的精确规则

### 双 query 视图

新增但不覆盖原句：

    original_query = 用户完整原话
    routing_query  = 仅去掉句首称呼与无意义开场

示例：

    老公，你还记得我们第一次说晚安那次吗
    → 你还记得我们第一次说晚安那次吗

    哥哥，Cloudflare 那张卡后来怎样了
    → Cloudflare 那张卡后来怎样了

    老公亲亲抱抱
    → 亲亲抱抱

routing_query 只给 Router/初始预算。候选检索、reranker、debug 和最终回复继续保留 original_query，避免丢失语气和指代。

只要存在明确回忆结构，例如“还记得、记不记得、那次、上次、当时为什么、后来、原话、找出来”，present_chitchat 就不得在候选检索前直接 skip。没有可靠候选时仍返回 0 条。

### 强匹配权限重新分级

#### 直接定位

- 明确的 Scene/Memory/Fact/Event ID。

#### 可成为强放行证据

1. 唯一完整标题 + 明确回忆指向
   - 完整标题作为连续片段出现在 query 中。
   - 只做 Unicode/全半角、英文大小写、连续空格、外围书名号/引号等表面规范化。
   - 不能只命中标题中的一个专名或几个词。
   - 若完整标题命中多条记录，不直接放行，进入 reranker。

2. 绑定原文中的足够长连续逐字片段
   - 必须查 source_message_ids 对应的原始消息证据，不能在模型生成的 Scene/Event 正文中把改写句当作原话。
   - 有成对引号时优先取引号内部。
   - 无引号时按句号、问号、感叹号、分号、换行切成句段；不要先按逗号切碎。
   - 很长句段才继续按逗号拆分。
   - 建议至少 8 个汉字或 12 个普通字符。
   - 必须完整落在同一条原始消息、同一句段内；不能跨消息或跨句删标点拼接。
   - 命中多条记忆时只进入候选，不直接放行。

#### 只进入候选

- 单个专名。
- 单个 Cue，即使逐字一致。
- 部分标题。
- 短引语、单词、技术名词。
- 语义相似但没有逐字证据。
- 多个 Cue 可以提高候选优先级，但不因“多个”自动获得直接注入权。

### 不再使用

- 删除全部标点后跨句做子串匹配。
- 引号里两个字就算 exact evidence。
- 专名出现在标题或正文一次就直接注入。
- 单个标题词/Cue 直接放行 Scene。
- 候选池 top1 自动等于应该注入。

## 下一窗口执行顺序

### 第一步：只做下一刀，不扩展整条架构

在 D:\codex_worktrees\serein-recall-admission-20260810：

1. 查 SemanticRecallRouter 的真实输入路径，建立 original_query / routing_query 双视图。
2. 只为 Router 去掉句首称呼；不要改写传给 candidate search、reranker 或最终模型的原句。
3. 让明确回忆标记否决 pure-chitchat pre-skip。
4. 收紧 exact anchor：短引语、专名、部分标题降为 candidate-only。
5. 为完整标题和 source-bound raw quote 建立独立、可调试的证据标签，不继续复用宽松 exact_anchor_match。
6. 不在这一步新增摘要、family 聚类、周报、Event 生命周期表或 DeepSeek judge。

### 第二步：只跑针对性验证

优先补/改现有小脚本，不跑全套门禁，不给未改文件做哈希。

必须覆盖：

    老公
    亲亲抱抱
    老公亲亲抱抱
    我去刷小红书了

应为 pure chitchat skip，且无注入。

    你还记得我们第一次说晚安那次吗
    老公，你还记得我们第一次说晚安那次吗

两句应得到相同初始检索预算；当前库无可靠对应记忆时都应无注入。不能通过硬编码“晚安”实现。

    今天太阳很大
    晴空里又下起没有乌云的雨

不得因为没有专名就强行 skip；允许 cheap retrieval，但只有真实绝对支持才能注入。

再增加：

- 单个专名只进候选，不直注。
- 单个 Cue 只进候选，不直注。
- 部分标题只进候选。
- 唯一完整标题 + “还记得”可成为强证据。
- 8 字以上绑定原文逐字片段可成为强证据。
- 同一原话命中两条 Scene 时必须 rerank，不直接放行。
- "token"、“换窗”等短引语不得成为 exact direct evidence。
- raw quote 不能从 Scene 改写正文冒充原始证据。

### 第三步：审结果后再决定是否发布

下一窗口先确认：

    git status --short --branch
    git rev-parse --short HEAD

德国机发布前再确认 /opt/Ombre-Brain-src 干净、生产分支和 tip。Tracked 文件必须走：

    本地修改 → commit → push 当前生产分支 → 德国机自动/标准发布链

不要用 SCP/rsync 覆盖 tracked 文件。不要边改边做多轮全量审查。先 targeted green，向小雨说明行为变化和仍未覆盖的边界，再由她决定是否发布。

### 第四步：独立修信箱刷新

召回下一刀通过后，再在 D:\codex_worktrees\haven-continuity-canonical-facts-20260811 做一个独立小改动：

- Memory Inbox 在重新获得页面焦点、重新进入视图或手动刷新时重新请求 /api/memory-inbox。
- 不做高频轮询。
- organizer 为 waiting_day 且 last_error 为空时，不继续显示旧 error 文案。
- 0 pending 时仍显示“等待完整一天”，不要伪造候选。

## 下一阶段，但不是下一刀

按优先级：

1. 候选与 injection 分离后的 absolute support / surface 门。
2. Fact/Event/Scene 统一 cheap candidate pool，深查按完整度和结构需求展开。
3. Scene 双检索视图，不改 canonical 正文。
4. Event scope/status/covered_by/promoted 生命周期。
5. canonical Fact/Event 接入 Bridge continuity packet，Scene 优先、按 source ID 去重。
6. 退热与周报/叙事卷候选；原 Event 不硬删除。
7. 积累更多人工裁决后再考虑 SetFit/shadow classifier。
8. family clustering 最后做，不进当前热路径。

## 工程边界

- 减少全量审查次数，减少门禁；未改动文件不用哈希验证。
- 默认最小改动，不顺手重构。
- 先区分候选、admission、diffusion、最终 injection；simulation/shadow 不能冒充 live injection。
- 不把 reranker 被调用当作准确率提升证明。
- 不把信箱 0 条候选误报成 organizer 失败。
- 不把 DeepSeek verifier、Bridge 每日整理和 Persona DeepSeek 混成同一开关。
- 不重跑已经人工清理并入库的 8 月 9～10 日数据，除非有明确新证据和备份计划。
- 不批量洗旧 Scene/Event 文风；先冻结新写入规则，旧记录只修真实失败样本。

## 一句话接手口径

下一刀只修两件互相关联的事：Router 不再被句首称呼带偏；专名、Cue、部分标题和短引语不再凭一次命中直接注入。找候选可以宽，真正浮现必须有唯一完整标题、source-bound 连续原话或足够强的独立支持。之后再单独补信箱刷新，不在同一刀里重做摘要、聚类、周报和整条召回架构。
