# Ombre Brain — 内部开发文档 / INTERNALS

> 本文档面向开发者和维护者。记录功能总览、环境变量、模块依赖、硬编码值和核心设计决策。
> 最后更新：2026-07-18（P1 场景真源与情绪权重退役）

---

## 0. 功能总览——这个系统到底做了什么

### 记忆能力

**存储与组织**
- 每条记忆 = 一个 Markdown 文件（YAML frontmatter 存元数据），直接兼容 Obsidian 浏览/编辑
- 四种桶类型：`dynamic`（普通，会衰减）、`permanent`（固化，不衰减）、`feel`（模型感受，不浮现）、`archived`（已遗忘）
- 按主题域分子目录：`dynamic/日常/`、`dynamic/情感/`、`dynamic/编程/` 等
- 钉选桶（pinned）：importance 锁 10，永不衰减/合并，始终浮现为「核心准则」

**每条记忆追踪的元数据**
- `id`（12位短UUID）、`name`（可读名≤80字）、`tags`（3~6个原文精确词）
- `domain`（1~2个主题域，从 8 大类 30+ 细分域选）
- `valence` / `arousal` / `model_valence`：旧数据和私密 feel 的兼容字段；不参与普通召回排序或衰减
- `importance`（1~10）、`activation_count`（被想起次数）
- `resolved`（已解决/沉底）、`digested`（已消化/写过 feel）、`pinned`（钉选）
- `created`、`last_active` 时间戳

**四种检索模式**
1. **自动浮现**（`breath()` 无参数）：按衰减分排序推送，钉选桶始终展示，Top-1 固定 + Top-20 随机打乱（引入多样性），有 token 预算（默认 10000）
2. **关键词+向量双通道搜索**（`breath(query=...)`）：rapidfuzz 模糊匹配 + Gemini embedding 余弦相似度，合并去重
3. **Feel 独立检索**（`breath(domain="feel")`）：按创建时间倒序返回所有 feel
4. **随机浮现**：搜索结果 <3 条时 40% 概率漂浮 1~3 条低权重旧桶（模拟人类随机联想）

**三维搜索评分**（归一化到 0~100）
- topic_relevance（权重 4.0）：name×3 + domain×2.5 + tags×2 + body
- time_proximity（权重 1.5）：`e^(-0.1×days)`
- importance（权重 1.0）：importance/10
- resolved 桶全局降权 ×0.3

**记忆随时间变化**
- **衰减引擎**：改进版艾宾浩斯遗忘曲线
  - 公式：`Score = Importance × activation_count^0.3 × e^(-λ×days) × freshness`
  - 新鲜度加成：`1.0 + e^(-t/36h)`，刚存入 ×2.0，~36h 半衰，72h 后 ≈×1.0
  - valence/arousal 不改变保留时长或紧迫度
  - resolved → ×0.05 沉底；resolved+digested → ×0.02 加速淡化
- **自动归档**：score 低于阈值(0.3) → 移入 archive
- **自动结案**：importance≤4 且 >30天 → 自动 resolved
- **永不衰减**：permanent / pinned / protected / feel

**记忆间交互**
- **只读重复提示**：普通 hold 会提示相关旧桶，但总是新建场景，不自动 LLM 合并或覆盖真源
- **时间涟漪**：touch 一个桶时，±48h 内创建的桶 activation_count +0.3（上限 5 桶/次）
- **向量相似网络**：embedding 余弦相似度 >0.5 建边
- **Feel 结晶化**：≥3 条相似 feel（相似度>0.7）→ 提示升级为钉选准则

**情绪边界**
- 情绪是对场景的后见理解，不是事实真源；reflection/comment/feel 不能成为普通直接召回 seed
- valence/arousal 参数仍可被旧客户端传递和读取，但不影响普通召回排名或衰减

**模型感受/反思系统**
- **Feel 写入**（`hold(feel=True)`）：存模型第一人称感受，标记源记忆为 digested
- **Dream 做梦**（`dream()`）：返回最近 10 条 + 自省引导 + 连接提示 + 结晶化提示
- **对话启动流程**：breath() → dream() → breath(domain="feel") → 开始对话

**新 Scene 写入**
- `write_scene` / `close_window` 的同步写入不调用 LLM：纯 Scene 正文先落盘，标题、显式 tags 与 scene_cues 作为 sidecar 元数据；向量只 embed content
- 每条新 canonical Scene 必须由当前作者显式给出至少一个 cue，表达“以后提到什么时希望这段记忆回来”；title、引句和正文都不作 cue 兜底
- 自动 enrich 只写 `metadata_proposals.sqlite` 待审 sidecar，不自动修改 tags / importance / confidence / embedding / edges
- 可选 Scene linker 在写入返回后异步运行，按顺序故障转移多个独立模型；两端逐字证据校验通过的结果先只写 `scene_edge_proposals.sqlite`。只读审核工具不改图；只有带精确确认词的单条接受操作在重验 Scene hash、active 状态与证据后，才在同一 SQLite 事务中写正式 `scene_edges` 并接受 proposal。它永不写旧 `memory_edges.jsonl`
- 脱水、日记拆分和旧自动摘记只保留给专项导入/旧兼容路径
- 普通召回先直接检索 Scene，再沿逐条审核的 Scene 图受限多跳；旧 JSONL 图仅白名单结构边、降权、最多一跳
- Wikilink `[[]]` 由 LLM 在内容中标记

---

### 技术能力

**9 个 MCP 工具**

| 工具 | 关键参数 | 功能 |
|---|---|---|
| `breath` | query, max_tokens, domain, valence, arousal, max_results | 检索/浮现记忆 |
| `hold` | content, cues, tags, importance, pinned, feel, whisper, source_bucket, valence, arousal | 原样保存一段纯 Scene 正文；旧 section 仅兼容读取 |
| `close_window` | shadow, idempotency_key, rejected_draft_source_hash, rejected_draft_section_patch, read_rejected_draft, continue_scene_index | 保存整窗窗影并只抽取内联 Scene；失败稿分库等待同 key 重试 |
| `grow` | content | `close_window` 兼容别名；不提升旧 `### moment` |
| `comment_bucket` | bucket_id, content, kind, valence, arousal | 给源 bucket 追加年轮 |
| `trace` | bucket_id, name, domain, valence, arousal, importance, tags, resolved, pinned, anchor, digested, content, delete | 修改元数据/内容/删除 |
| `pulse` | include_archive | 系统状态 |
| `dream` | （无） | 做梦自省 |
| `resurface` | max_results, include_archive, max_tokens | 只读浮现久未触碰的旧记忆 |
| `reflect` | period, force | 生成日印象 relationship_weather feel；周印象默认关闭 |

**工具详细行为**

**`breath`** — 两种模式：
- **浮现模式**（无 query）：无参调用，按衰减引擎活跃度排序返回 top 记忆，permanent/pinned 始终浮现
- **检索模式**（有 query）：关键词 + 向量双通道搜索，三维评分（topic×4 + time×2.5 + importance×1），阈值过滤
- **Feel 检索**（`domain="feel"`）：特殊通道，按创建时间倒序返回所有 feel 类型桶，不走评分逻辑
- valence/arousal 参数仅兼容读取，不参与普通排序

**`hold`** — 主要模式：
- **普通模式**（`feel=False`，默认）：当前 AI 先写一段完整 Scene 原文，content 不带 `## Scene` / `### scene` / `### moment`，并亲自写至少一个 sidecar `cues` → 工具只校验、原样新建；向量只取 content，不调用模型、不自动合并
- **Whisper 模式**（`whisper=True`）：无源碎碎念/悄悄话，独立保存为 `type=feel + whisper`，可用 `breath(domain="whisper")` 读取。
- **旧兼容 Feel 路径**（`feel=True`）：不建议新调用。带 `source_bucket` 时转为年轮 comment；不带源桶时转为 whisper。

**`comment_bucket`** — 年轮：
- 给已有 bucket 追加 `metadata.comments[]`，MCP 作者固定取 `identity.ai_name`
- 追加后 touch 源 bucket、刷新源 bucket embedding，不新建独立 feel 桶
- 年轮可以按文本把查询路由到父 Scene；只随父 Scene 附一条，不作为独立候选、扩散节点或 Fact 证据

**`dream`** — 做梦/自省触发器：
- 返回最近 10 条 dynamic 桶摘要 + 自省引导词
- 检测 feel 结晶化：≥3 条相似 feel（embedding 相似度>0.7）→ 提示升级为钉选准则
- 检测未消化记忆：列出 `digested=False` 的桶供模型反思

**`reflect`** — 关系天气：
- `period=daily` 生成当天 `daily_impression`
- `period=weekly` 默认 skipped；当前不把日印象压缩成周印象
- 结果存为 `type=feel`，带 `relationship_weather` 标签
- 默认复用 `persona` 模型配置和 key，可用 `OMBRE_REFLECTION_*` 单独覆盖

**Scene Edge** — canonical Scene 正式关系边：
- 表：`state/scene_edge_proposals.sqlite -> scene_edges`
- 只允许 `continues / echoes / resolves / contrasts_with / evidenced_by`，且两端必须有逐字正文证据
- proposal 与正式边原子晋升；普通召回可在可靠 direct Scene seed 后受限多跳

**Legacy Memory Edge** — 旧桶兼容关系：
- 文件：`state/memory_edges.jsonl`，保留原文件，不自动迁移、不由 Scene linker 改写
- 普通召回只读 `same_event / context_of / updates / evidenced_by / supports / precedes / triggers / contradicts / belongs_to`，统一降权且最多一跳
- `emotional_echo / reflects_on / relates_to / previous_context / next_context` 等旧泛关系不再参与普通扩散

**`trace`** — 记忆编辑：
- 修改任意元数据字段（name/domain/valence/arousal/importance/tags/resolved/pinned）
- `digested=0/1`：隐藏/取消隐藏记忆（控制是否在 dream 中出现）
- `content="..."`：替换正文内容并重新生成 embedding
- `delete=True`：删除桶文件

**`close_window`** — Window Shadow：
- 整篇第一人称窗影原样写入独立 SQLite，不进普通候选池
- 没有平行 `scenes[]` 写入口；“想留下的记忆”内联格式为 `### scene | 标题：作者标题 | cue：召回入口`，可追加 1～8 个 cue。标题和 cues 只进 metadata，heading 落库时去掉；不调用 LLM，也不从正文补造
- 写前完整校验，任一 Scene 失败就补偿删除本次新建 Scene 与 Shadow；成功后才排 embedding
- `invalid` / `error` 且有 idempotency key 时，原 Shadow 逐字写入单独的 `window_shadow_rejected_drafts.sqlite`；该库没有 ordinary recall、handoff、bucket、embedding 或 `read_memory` 入口
- 同 key 的 divergent Shadow 没有上一稿 hash 时拒绝；参数-only 修复必须逐字复用，文本修复必须带 `rejected_draft_source_hash` 且只能改变 `fix_scope` 指定的 section；成功写入后清除失败稿，成功幂等结果始终优先
- `grow` 仅为旧客户端兼容别名，不把旧 `### moment` 提升成 Scene

**`pulse`** — 系统状态：
- 返回各类型桶数量、衰减引擎状态、未解决/钉选/feel 统计

**REST API（17 个端点）**

| 端点 | 方法 | 功能 |
|---|---|---|
| `/health` | GET | 健康检查 |
| `/breath-hook` | GET | SessionStart 钩子 |
| `/dream-hook` | GET | Dream 钩子 |
| `/dashboard` | GET | Dashboard 页面 |
| `/api/buckets` | GET | 桶列表 |
| `/api/bucket/{id}` | GET | 桶详情 |
| `/api/search?q=` | GET | 搜索 |
| `/api/network` | GET | 向量相似网络 |
| `/api/breath-debug` | GET | 评分调试 |
| `/api/config` | GET | 配置查看（key 脱敏） |
| `/api/config` | POST | 热更新配置 |
| `/api/import/upload` | POST | 上传并启动历史对话导入 |
| `/api/import/status` | GET | 导入进度查询 |
| `/api/import/pause` | POST | 暂停/继续导入 |
| `/api/import/patterns` | GET | 导入完成后词频规律检测 |
| `/api/import/results` | GET | 已导入记忆桶列表 |
| `/api/import/review` | POST | 批量审阅/批准导入结果 |

**Dashboard（7 个 Tab）**
1. 记忆桶列表：6 种过滤器 + 主题域过滤 + 搜索 + 详情面板
2. Breath 模拟：输入参数 → 可视化五步流程 → 三维评分；旧 emotion 栏仅显示兼容诊断且权重为 0
3. 日印象：月历 + 单卡片详情，按日期显示完整 daily relationship weather；手动编辑仍走 bucket 详情面板
4. Persona：查看 Persona State sessions / events / 当前状态
5. 记忆网络：Canvas 力导向图（节点=桶，边=相似度）
6. 配置：热更新记忆分析、embedding、Gateway 记忆浮现、图扩散、反思、梦境和旧兼容参数
7. 导入：历史对话拖拽上传 → 分块处理进度条 → 词频规律分析 → 导入结果审阅

**部署选项**
1. 本地 stdio（`python server.py`）
2. Docker + Cloudflare Tunnel（`docker-compose.yml`）
3. Docker Hub 预构建镜像（`docker-compose.user.yml`，`p0luz/ombre-brain`）
4. Render.com 一键部署（`render.yaml`）
5. Zeabur 部署（`zbpack.json`）
6. GitHub Actions 自动构建推送 Docker Hub（`.github/workflows/docker-publish.yml`）

**迁移/批处理工具**：`migrate_to_domains.py`、`reclassify_domains.py`、`reclassify_api.py`、`backfill_embeddings.py`、`write_memory.py`、`check_buckets.py`、`import_memory.py`（历史对话导入引擎）

**降级策略**
- 记忆分析 API 不可用 → 本地精确打标；专项压缩路径降级到关键词提取 + 句子评分
- 向量搜索不可用 → 纯 fuzzy match
- `close_window` 使用整组失败边界；不允许只留下半篇 Shadow 或部分 Scene

**安全**：路径遍历防护（`safe_path()`）、API Key 脱敏、API Key 不持久化到 yaml、输入范围钳制

**监控**：结构化日志、Health 端点、Breath Debug 端点、Dashboard 统计栏、衰减周期日志

---

## 1. 环境变量清单

| 变量名 | 用途 | 必填 | 默认值 / 示例 |
|---|---|---|---|
| `OMBRE_API_KEY` | 脱水/打标/嵌入的 LLM API 密钥，覆盖 `config.yaml` 的 `dehydration.api_key` | 否（无则 API 功能降级到本地） | `""` |
| `OMBRE_BASE_URL` | API base URL，覆盖 `config.yaml` 的 `dehydration.base_url` | 否 | `""` |
| `OMBRE_TRANSPORT` | 传输模式：`stdio` / `sse` / `streamable-http` | 否 | `""` → 回退到 config 或 `"stdio"` |
| `OMBRE_BUCKETS_DIR` | 记忆桶存储目录路径 | 否 | `""` → 回退到 config 或 `./buckets` |
| `OMBRE_HOOK_URL` | SessionStart 钩子调用的服务器 URL | 否 | `"http://localhost:8000"` |
| `OMBRE_HOOK_SKIP` | 设为 `"1"` 跳过 SessionStart 钩子 | 否 | 未设置（不跳过） |

环境变量优先级：`环境变量 > config.yaml > 硬编码默认值`。所有环境变量在 `utils.py` 中读取并注入 config dict。

---

## 2. 模块结构与依赖关系

```
                    ┌──────────────┐
                    │  server.py   │  MCP 主入口，9 个工具 + Dashboard + Hook
                    └──────┬───────┘
           ┌───────────────┼───────────────┬────────────────┐
           ▼               ▼               ▼                ▼
   bucket_manager.py  dehydrator.py  decay_engine.py  embedding_engine.py
   记忆桶 CRUD+搜索   元数据分析/旧压缩  遗忘曲线+归档   向量化+语义检索
           │               │                                │
           └───────┬───────┘                                │
                   ▼                                        ▼
              utils.py ◄────────────────────────────────────┘
              配置/日志/ID/路径安全/token估算
```

| 文件 | 职责 | 依赖（项目内） | 被谁调用 |
|---|---|---|---|
| `server.py` | MCP 服务器主入口，注册工具 + Dashboard API + 钩子端点 | `bucket_manager`, `dehydrator`, `decay_engine`, `embedding_engine`, `utils` | `test_tools.py` |
| `bucket_manager.py` | 记忆桶 CRUD、多维索引搜索、wikilink 注入、激活更新 | `utils` | `server.py`, `check_buckets.py`, `backfill_embeddings.py` |
| `decay_engine.py` | 衰减引擎：遗忘曲线计算、自动归档、自动结案 | 无（接收 `bucket_mgr` 实例） | `server.py` |
| `dehydrator.py` | 普通 hold 的元数据分析，以及专项导入/旧路径的压缩与合并兼容（LLM API + 本地降级） | `utils` | `server.py` |
| `embedding_engine.py` | 向量化引擎：Gemini embedding API + SQLite + 余弦搜索 | `utils` | `server.py`, `backfill_embeddings.py` |
| `utils.py` | 配置加载、日志、路径安全、ID 生成、token 估算 | 无 | 所有模块 |
| `write_memory.py` | 手动写入记忆 CLI（绕过 MCP） | 无（独立脚本） | 无 |
| `backfill_embeddings.py` | 为存量桶批量生成 embedding | `utils`, `bucket_manager`, `embedding_engine` | 无 |
| `check_buckets.py` | 桶数据完整性检查 | `bucket_manager`, `utils` | 无 |
| `import_memory.py` | 历史对话导入引擎（支持 Claude JSON/ChatGPT/DeepSeek/Markdown/纯文本），分块处理+断点续传+词频分析 | `utils` | `server.py` |
| `reclassify_api.py` | 用 LLM API 重打标未分类桶 | 无（直接用 `openai`） | 无 |
| `reclassify_domains.py` | 基于关键词本地重分类 | 无 | 无 |
| `migrate_to_domains.py` | 平铺桶 → 域子目录迁移 | 无 | 无 |
| `test_smoke.py` | 冒烟测试 | `utils`, `bucket_manager`, `dehydrator`, `decay_engine` | 无 |
| `test_tools.py` | MCP 工具端到端测试 | `utils`, `server`, `bucket_manager` | 无 |

---

## 3. 硬编码值清单

### 3.1 固定分数 / 特殊返回值

| 值 | 位置 | 用途 |
|---|---|---|
| `999.0` | `decay_engine.py` calculate_score | pinned / protected / permanent 桶永不衰减 |
| `50.0` | `decay_engine.py` calculate_score | feel 桶固定活跃度分数 |
| `0.02` | `decay_engine.py` resolved_factor | resolved + digested 时的权重乘数（加速淡化） |
| `0.05` | `decay_engine.py` resolved_factor | 仅 resolved 时的权重乘数（沉底） |
| `1.5` | `decay_engine.py` urgency_boost | arousal > 0.7 且未解决时的紧迫度加成 |

### 3.2 衰减公式参数

| 值 | 位置 | 用途 |
|---|---|---|
| `36.0` | `decay_engine.py` _calc_time_weight | 新鲜度半衰期（小时），`1.0 + e^(-t/36)` |
| `0.3` (指数) | `decay_engine.py` calculate_score | `activation_count ** 0.3`（记忆巩固指数） |
| `3.0` (天) | `decay_engine.py` calculate_score | 短期/长期切换阈值 |
| `0.7 / 0.3` | `decay_engine.py` combined_weight | 短期权重分配：time×0.7 + emotion×0.3 |
| `0.7` | `decay_engine.py` urgency_boost | arousal 紧迫度触发阈值 |
| `4` / `30` (天) | `decay_engine.py` execute_cycle | 自动结案：importance≤4 且 >30天 |

### 3.3 搜索/评分参数

| 值 | 位置 | 用途 |
|---|---|---|
| `×3` / `×2.5` / `×2` | `bucket_manager.py` _calc_topic_score | 桶名 / 域名 / 标签的 topic 评分权重 |
| `1000` (字符) | `bucket_manager.py` _calc_topic_score | 正文截取长度 |
| `0.1` | `bucket_manager.py` _calc_time_score | 时间亲近度衰减系数 `e^(-0.1 × days)` |
| `0.3` | `bucket_manager.py` search_multi | resolved 桶的归一化分数乘数 |
| `0.5` | `server.py` breath/search | 向量搜索相似度下限 |
| `0.7` | `server.py` dream | feel 结晶相似度阈值 |

### 3.4 Token 限制 / 截断

| 值 | 位置 | 用途 |
|---|---|---|
| `10000` | `server.py` breath 默认 max_tokens | 浮现/搜索 token 预算 |
| `20000` | `server.py` breath 上限 | max_tokens 硬上限 |
| `50` / `20` | `server.py` breath | max_results 上限 / 默认值 |
| `3000` | `dehydrator.py` dehydrate | API 脱水内容截断 |
| `2000` | `dehydrator.py` merge | API 合并内容各截断 |
| `5000` | `dehydrator.py` digest | API 日记整理内容截断 |
| `2000` | `embedding_engine.py` | embedding 文本截断 |
| `100` | `dehydrator.py` | 内容 < 100 token 跳过脱水 |

### 3.5 时间/间隔/重试

| 值 | 位置 | 用途 |
|---|---|---|
| `60.0s` | `dehydrator.py` | OpenAI 客户端 timeout |
| `30.0s` | `embedding_engine.py` | Embedding API timeout |
| `60s` | `server.py` keepalive | 保活 ping 间隔 |
| `48.0h` | `bucket_manager.py` touch | 时间涟漪窗口 ±48h |
| `2s` | `backfill_embeddings.py` | 批次间等待 |

### 3.6 随机浮现

| 值 | 位置 | 用途 |
|---|---|---|
| `3` | `server.py` breath search | 结果不足 3 条时触发 |
| `0.4` | `server.py` breath search | 40% 概率触发随机浮现 |
| `2.0` | `server.py` breath search | 随机池：score < 2.0 的低权重桶 |
| `1~3` | `server.py` breath search | 随机浮现数量 |

### 3.7 旧情绪参数兼容

`valence` / `arousal` 仍可读取和传递；普通召回与衰减中的有效权重均为 `0`，不再做展示层偏移。

### 3.8 其他

| 值 | 位置 | 用途 |
|---|---|---|
| `12` | `utils.py` gen_id | bucket ID 长度（UUID hex[:12]） |
| `80` | `utils.py` sanitize_name | 桶名最大长度 |
| `1.5` / `1.3` | `utils.py` count_tokens_approx | 中文/英文 token 估算系数 |
| `8000` | `server.py` | MCP 服务器端口 |
| `30` 字符 | `server.py` grow | 短内容快速路径阈值 |
| `10` | `server.py` dream | 取最近 N 个桶 |

---

## 4. Config.yaml 完整键表

| 键路径 | 默认值 | 用途 |
|---|---|---|
| `transport` | `"stdio"` | 传输模式 |
| `log_level` | `"INFO"` | 日志级别 |
| `buckets_dir` | `"./buckets"` | 记忆桶目录 |
| `merge_threshold` | `90` | 合并相似度阈值 (0-100) |
| `dehydration.model` | `"deepseek-chat"` | 记忆元数据分析及旧兼容压缩所用 LLM |
| `dehydration.base_url` | `"https://api.deepseek.com/v1"` | API 地址 |
| `dehydration.api_key` | `""` | API 密钥 |
| `dehydration.max_tokens` | `1024` | 分析/旧兼容压缩返回 token 上限 |
| `dehydration.temperature` | `0.1` | 分析模型温度 |
| `embedding.enabled` | `true` | 启用向量检索 |
| `embedding.model` | `"gemini-embedding-001"` | Embedding 模型 |
| `decay.lambda` | `0.05` | 衰减速率 λ |
| `decay.threshold` | `0.3` | 归档分数阈值 |
| `decay.check_interval_hours` | `24` | 衰减扫描间隔（小时） |
| `decay.emotion_weights.base` | `1.0` | 旧配置兼容，占位但不参与衰减 |
| `decay.emotion_weights.arousal_boost` | `0.0` | 旧配置兼容，唤醒度不改变保留时长 |
| `matching.fuzzy_threshold` | `50` | 模糊匹配下限 |
| `matching.max_results` | `5` | 匹配返回上限 |
| `scoring_weights.topic_relevance` | `4.0` | 主题评分权重 |
| `scoring_weights.emotion_resonance` | `0.0` | 旧配置兼容，不参与普通召回排序 |
| `scoring_weights.time_proximity` | `1.5` | 时间评分权重 |
| `scoring_weights.importance` | `1.0` | 重要性评分权重 |
| `scoring_weights.content_weight` | `3.0` | 正文评分权重 |
| `wikilink.enabled` | `true` | 启用 wikilink 注入 |
| `wikilink.use_tags` | `false` | wikilink 包含标签 |
| `wikilink.use_domain` | `true` | wikilink 包含域名 |
| `wikilink.use_auto_keywords` | `true` | wikilink 自动关键词 |
| `wikilink.auto_top_k` | `8` | wikilink 取 Top-K 关键词 |
| `wikilink.min_keyword_len` | `2` | wikilink 最短关键词长度 |
| `wikilink.exclude_keywords` | `[]` | wikilink 排除关键词表 |

---

## 5. 核心设计决策记录

### 5.1 为什么用 Markdown + YAML frontmatter 而不是数据库？

**决策**：每个记忆桶 = 一个 `.md` 文件，元数据在 YAML frontmatter 里。

**理由**：
- 与 Obsidian 原生兼容——用户可以直接在 Obsidian 里浏览、编辑、搜索记忆
- 文件系统即数据库，天然支持 git 版本管理
- 无外部数据库依赖，部署简单
- wikilink 注入让记忆之间自动形成知识图谱

**放弃方案**：SQLite/PostgreSQL 全量存储。过于笨重，失去 Obsidian 可视化优势。

### 5.2 为什么 embedding 单独存 SQLite 而不放 frontmatter？

**决策**：向量存 `embeddings.db`（SQLite），与 Markdown 文件分离。

**理由**：
- 3072 维浮点向量无法合理存入 YAML frontmatter
- SQLite 支持批量查询和余弦相似度计算
- embedding 是派生数据，丢失可重新生成（`backfill_embeddings.py`）
- 不污染 Obsidian 可读性

### 5.3 为什么搜索用双通道（关键词 + 向量）而不是纯向量？

**决策**：关键词模糊匹配（rapidfuzz）+ 向量语义检索并联，结果去重合并。

**理由**：
- 纯向量在精确名词匹配上表现差（"2024年3月"这类精确信息）
- 纯关键词无法处理语义近似（"很累" → "身体不适"）
- 双通道互补，关键词保精确性，向量补语义召回
- 向量不可用时自动降级到纯关键词模式

### 5.4 为什么还保留 `dehydrator.py`？

**决策**：普通 `hold` 与 `close_window` 完全退出 Dehydrator/二级标签模型写入链路。`dehydrate()` / `analyze()` / `merge()` 只保留给专项导入和旧兼容路径；已写好的 Scene 正文永不交给它改写。后台 enrich 只能生成待审 sidecar proposal，不能自动改变正文或召回权重。

**理由**：
- 场景正文是事实真源，必须由当前窗口亲自写并原样保存
- 元数据分析仍能降低检索和管理成本，但结果必须先停在待审 proposal，不能替代或暗改场景本身
- 历史导入可能需要受控压缩，因此不在 P1 直接删除底层兼容能力

### 5.5 为什么 feel 和普通记忆分开？

**决策**：`feel=True` 的记忆存入独立 `feel/` 目录，不参与普通浮现、不衰减、不合并。

**理由**：
- feel 是模型的自省产物，不是事件记录——两者逻辑完全不同
- 事件记忆应该衰减遗忘，但"我从中学到了什么"不应该被遗忘
- feel 的 valence 是模型自身感受（不等于事件情绪），混在一起会污染情感检索
- feel 可以通过 `breath(domain="feel")` 单独读取

### 5.6 为什么 resolved 不删除记忆？

**决策**：`resolved=True` 让记忆"沉底"（权重 ×0.05），但保留在文件系统中，关键词搜索仍可触发。

**理由**：
- 模拟人类记忆：resolved 的事不会主动想起，但别人提到时能回忆
- 删除是不可逆的，沉底可随时 `resolved=False` 重新激活
- `resolved + digested` 进一步降权到 ×0.02（已消化 = 更释然）

**放弃方案**：直接删除。不可逆，且与人类记忆模型不符。

### 5.7 为什么衰减不再读取情绪坐标？

**决策**：保留 importance、activation、时间和 resolved/digested 生命周期；valence/arousal 不再改变保留时长或紧迫度。

**理由**：情绪是对场景的后见理解，不是事件本体。让一次模型打出的坐标长期控制记忆生死，会把不稳定推断固化成系统真相。

### 5.8 为什么 dream 设计成对话开头自动执行？

**决策**：每次新对话启动时，Claude 执行 `dream()` 消化最近记忆，有沉淀写 feel，能放下的 resolve。

**理由**：
- 模拟睡眠中的记忆整理——人在睡觉时大脑会重放和整理白天的经历
- 让 Claude 对过去的记忆有"第一人称视角"的自省，而不是冷冰冰地搬运数据
- 自动触发确保每次对话都"接续"上一次，而非从零开始

### 5.9 为什么新鲜度用连续指数衰减而不是分段阶梯？

**决策**：`bonus = 1.0 + e^(-t/36)`，t 为小时，36h 半衰。

**理由**：
- 分段阶梯（0-1天=1.0，第2天=0.9...）有不自然的跳变
- 连续指数更符合遗忘曲线的物理模型
- 36h 半衰期使新桶在前两天有明显优势，72h 后接近自然回归
- 值域 1.0~2.0 保证老记忆不被惩罚（×1.0），只是新记忆有额外加成（×2.0）

**放弃方案**：分段线性（原实现）。跳变点不自然，参数多且不直观。

### 5.10 为什么退役情感记忆重构？

**决策**：旧 `valence` / `arousal` 字段与接口参数继续兼容，但普通检索不按坐标重排，也不再修改展示值。

**理由**：同一场景可以在不同时间被重新理解；召回应先找对可追溯的场景，再由当前上下文理解它，而不是让旧坐标预先决定答案。

---

## 6. 目录结构约定

```
buckets/
├── permanent/       # pinned/protected 桶，importance=10，永不衰减
├── dynamic/
│   ├── 日常/        # domain 子目录
│   ├── 情感/
│   ├── 自省/
│   ├── 数字/
│   └── ...
├── archive/         # 衰减归档桶
└── feel/            # 模型自省 feel 桶
```

桶文件格式：
```markdown
---
id: 76237984fa5d
name: 桶名
domain: [日常, 情感]
tags: [关键词1, 关键词2]
importance: 5
valence: 0.6
arousal: 0.4
activation_count: 3
resolved: false
pinned: false
digested: false
created: 2026-04-17T10:00:00+08:00
last_active: 2026-04-17T14:00:00+08:00
type: dynamic
---

桶正文内容...
```
