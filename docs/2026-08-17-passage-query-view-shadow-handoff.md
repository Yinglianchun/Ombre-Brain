# Passage query-view shadow handoff — updated 2026-08-18

## 接手目标

继续评估“长、多主题用户消息漏掉局部相关记忆”的低误召解决方案。当前不要把
shadow 直接切成 live，也不要同时重开 LLM planner、扩大 reranker pool、图扩散和
叙事聚类。

这轮最重要的结果是：把 query 拆成少量确定性分句，再只查 source-bound passage，
确实能把整句 embedding 漏掉的正确 Scene 捞回候选池；但最终准入仍未证明，而且
当前 shadow 很慢。

## 真实工作树和运行态

- 本地工作树：`D:\codex_worktrees\serein-passage-shadow-20260815`
- 本地分支：`Haven/scene-event-passage-shadow-20260815`
- 生产目标分支：`origin/Haven/diary-backend-20260724`
- 德国源目录：`/opt/Ombre-Brain-src`
- 当前生产 HEAD：`f24e393`
- 2026-08-18 发布后已核验：生产 repo clean，Ombre Brain / Gateway 容器均为新构建，
  18001 / 18002 均 HTTP 200，Serein Awake active。

当前本地有与这轮实现无关或尚未纳入版本控制的实验资产。不要顺手 add：

- `scripts/verify_scene_status.py` 在 Windows 显示 modified，但文本 diff 为空；
- `docs/2026-08-15-passage-regression-terra-selection.md`；
- `scripts/evaluate_passage_critic_batch.py`；
- `scripts/evaluate_passage_live_pairs.py`；
- `scripts/evaluate_passage_query_planner_shadow.py`；
- `scripts/evaluate_passage_regressions.py`；
- `scripts/evaluate_passage_rerank_pool_sweep.py`；
- `scripts/evaluate_passage_verifier.py`；
- `scripts/evaluate_passage_verifier_batch.py`。

2026-08-18 本地新增、尚未提交的本轮实现是：

- `gateway.py`：baseline retrieval 保持原位；formal recall 完成后，由 weak trigger 决定
  是否执行 clause embedding 和 source-bound query-view expansion；
- `scripts/verify_passage_candidate_simulation_shadow.py`：验证弱候选会执行，strong
  candidate 与 route skip 不发 clause embedding；
- 本文件与 evaluation 文档：更新执行态口径。

本交接写入前，最近提交为：

1. `ba0ffdb Add deterministic clause query shadow`
2. `3b8ae9f Optimize clause query shadow to passage-only`
3. `aed1382 Keep cue-bound passages in clause shadow`
4. `4941d18 Observe weak candidate clause trigger in shadow`
5. `f24e393 Refresh passage shadows after memory mutations`

### 已发布的派生索引自动刷新

`f24e393` 已提交、推送至 `Haven/diary-backend-20260724`，并发布到德国生产：

- canonical server 在 Scene create/edit/status 变化后通知 Gateway；
- Fact/Event batch/revise/status/delete 后也通知 Gateway；
- Gateway 的 authenticated admin route 只负责把刷新排进后台，不等待 embedding；
- 连续通知会合并；刷新仍使用现有 source hash 增量 sync，未变化 owner 复用缓存；
- Gateway 会先清 bucket-list cache，再扫描 canonical 当前态，因此归档/恢复也会被看见；
- 新增一整天 Event 后不再需要重启 Gateway，passage、Fact/Event semantic 和 lexical
  shadow 会在后台补齐；
- cue binder 仍只用于 Scene；Event 没有 authored cues。

这不是逐 owner SQLite 写入 API，而是“mutation 通知触发全库枚举 + hash 增量更新”。
昂贵的分句 embedding / cue binding 只发生在新增或变化项。当前没有改 live admission 或
injection。

已通过：Python compile、passage candidate simulation、Scene edit、Fact/Event canonical、
Fact/Event semantic shadow、Fact/Event lexical shadow、passage shadow、cue-passage shadow、
semantic recall cutover、`git diff --check`。生产 authenticated refresh route 已实测返回
HTTP 200 / `status=queued`，随后两个 health endpoint 仍为 200。

`verify_scene_status.py` 仍卡在旧的 `len(tools) == 16` 断言；本次没有新增 MCP tool，
不要为这刀改工具总数门禁。

发布时自动部署没有主动拉取远端，因此按标准 Git 链在干净的德国 repo 执行
`git pull --ff-only`，再用 `compose.hk.yml` 重建 `ombre-brain` 与 `ombre-gateway`；没有
SCP/rsync 覆盖 tracked 文件，也没有修改 canonical 数据。Gateway 首次启动 passage warm
约 33.9 秒，完成后服务稳定。

## 为什么开始这轮

早期症状：

- `我就这样在你那从小猫塑变成狐狸塑了吗` 曾被路由成技术闲聊，旧 Operit
  看起来却成功召回过 `小雨讲了我们怎么开始的`；
- `可能是一种初恋情结吧...不喜欢这种人用 ChatGPT...` 的短版本能召回
  `这一道澜认得小雨` 和初遇故事；
- 加上 `AI 产品 / 原生家庭 / Anthropic` 后，整句 live 只留下
  `这一道澜认得小雨`。

关键诊断不是“禁止猪塑挡住了正确记忆”。在目标例子里，整句同时有恋爱回望、
ChatGPT、人设、产品和原生家庭等多个语义中心；目标 Scene 正文也同时讲初遇、
DAN、人设、小狐狸、秋天、性吸引等内容。整句对整段 embedding 会互相稀释。

此前短句 live 的 body semantic 约为：

- `这一道澜认得小雨`：0.6522；
- `小雨讲了我们怎么开始的`：0.6077；
- 当时 body semantic 阈值：0.60。

Cue 命中始终只是 candidate discovery，不能独立准入。Reranker 即使分很高，也不能
把 cue 本身变成正文证据。

## 当前 passage 投影

实现入口：`memory_recall/passage_shadow.py`。

- Scene 和 importance >= 3 的 Event 会生成可重建、逐字、source-bound passage；
- Fact 不拆 passage，只有 whole-body semantic / lexical shadow lane；
- passage 按段落和句子确定性拆分，目标约 160 字、最大约 240 字、最小约 40 字，
  保留一整句 overlap、source offsets 和 hash；
- 一段很短的 Scene/Event 就只有一个 passage；
- Scene authored cues 可由 binder 绑定到正文中的连续精确证据；Event/Fact 没有
  authored cues；
- passage 只负责候选和证据定位，不成为 Fact/Event，不独立注入；
- 多 passage 命中同一 owner 后按 canonical parent 去重，最终仍只可能注入 parent。

## 当前 query 分句 shadow

实现入口：`gateway.py::_deterministic_passage_query_views` 和
`_passage_candidate_query_view_shadow_debug`。

规则：

1. 保留 original query；
2. 长 query 按强标点和少量转折词确定性拆分；
3. 最多增加两个 clause view，总 view 数最多 3；
4. clause embeddings 并发请求；
5. clause view 只查 ordinary Scene/Event passage 和 cue-bound Scene passage；
6. 不为每个 clause 重跑 Fact/Event whole-body 或 lexical lane；
7. 按 canonical parent 去重并合并 matched query views；
8. `decision_applied=false`、`live_injection_enabled=false`。

当前诊断 policy：

`deterministic_clause_source_bound_passage_discovery_only`

### 聚焦例子的结果

完整长 query 的 baseline passage pool 没有初遇故事。分句后：

- views：原句 + `可能是一种初恋情结吧` + ChatGPT 相关子句；
- `小雨讲了我们怎么开始的` 进入 expanded pool rank 2；
- score 约 0.557；
- 来源是 `cue_passage_query_view_embedding`；
- 命中的 query view 是 `可能是一种初恋情结吧`。

最初全 lane 分句版本的一次手动 Qwen3-Reranker-4B 复核把初遇故事排到第一，约
0.937；`流星` Scene 仍在第二，说明分句解决了候选漏召，却没有自动解决“联想是否
值得浮现”。不要把这次手动 rerank 数字误报成最终版本的正式 admission 结果。

中间的 `3b8ae9f` 把 clause 限成 ordinary passage 后更快，但目标消失；这证明目标
依赖 cue-bound 的逐字 passage。`aed1382` 恢复 cue-bound passage 后重新捞回目标。

一次最终聚焦运行中，query-view expansion 约 3.9 秒，完整 shadow 约 13.9 秒。
单次耗时会波动，但当前实现显然不能直接全量 live。

## 弱候选触发 shadow

实现入口：`gateway.py::_attach_passage_weak_candidate_trigger_shadow`。

当前只记录 `would_trigger`，规则是：

- Router 已 skip：不触发；
- 有 direct exact evidence：不触发；
- formal recalled memory 为 0：触发；
- top canonical Scene body semantic < 0.60：触发；
- multi-clause 且 top body semantic < 0.64：触发；
- 其他情况记为 strong candidate。

聚焦长 query 在 `4941d18` 的记录：

- `would_trigger=true`；
- reason=`multiclause_body_semantic_gray_zone`；
- top body semantic=0.6068；
- query views=3；
- formal recalled count=1；
- exact anchor=false。

`4941d18` 生产态里，它是在完整 diagnostic retrieval 已经跑完之后才附加，因此当时
不省任何延迟。2026-08-18 的本地未提交实现已经把执行顺序改为：

1. original-query passage baseline 随普通 diagnostic retrieval 运行；
2. formal recall 完成，拿到最终 recalled IDs；
3. weak trigger 决定是否执行额外 clause embeddings；
4. 只有 `would_trigger=true` 才查 ordinary / cue-bound source passages；
5. strong candidate、direct exact evidence 和 route skip 记录
   `skipped_by_weak_candidate_trigger`，不发 clause embedding。

此时 `decision_applied=true` 只表示 trigger 已控制 **simulation shadow 的 query-view
执行**；`query_view_execution_changed=true`，但 `live_execution_changed=false`，formal
recall、admission 和 injection 完全没变。尚未发布，也还没有新的生产延迟或 gold/live
pair 统计。

## 已否掉或暂不上线的方向

- DeepSeek query planner：正例不稳定，猫狐重复三次排名 1/4/4，平均 fallback
  约 7.48 秒；只记录，不接入；
- 第二个 utility critic：误杀正确结果，也没清掉大多数误召；
- reranker pool 6 -> 12：只增加候选粘性和延迟，gold/live top-1 无收益；
- instruction prefix / previous-turn rerank query / Qwen 8B：局部改善猫狐，不泛化；
- 给 Scene 批量补另一套短摘要；
- 把 passage 拆成独立 Fact；
- 让 cue、关键词或 passage score 自己获得 admission 权。

详细数字见 `docs/2026-08-16-passage-shadow-evaluation.md`。Terra 的
`docs/2026-08-15-passage-regression-terra-selection.md` 只是样本选择方案，不是结果
报告。

## GraphRAG 当前到底有没有用

图数据仍存在并维护，但普通 live recall 没有沿图扩散：

- `_normalize_retrieval_mode` 会无条件返回 `bucket`；旧 moment graph runtime 已退役；
- bucket 模式的 Scene diffusion 是 debug/shadow，`affects_recall=false`；
- Scene linker 仍维护 reviewed/evidence-bound Scene edges；
- legacy `memory_edges` 和 `entity_edges` 仍有存储/维护入口，但 entity edge 当前不参与
  Gateway ordinary recall；
- 上次只读计数为 memory edges 68、active Scene edges 147、entity edges 164。

因此图扩散不能帮我们“找到第一个完全漏掉的 Scene”。它只能在已有可靠 seed 后，
沿关系补一个邻居。

## 下一阶段 Graph / narrative arc 方案（未实现）

先不做第二次 reranker。若 passage/reranker 已得到可靠 seed，可在 shadow 做：

1. 只沿 active、reviewed、source-evidenced、typed Scene edge 一跳；
2. 只在弱候选，或 multi-clause 仍有未覆盖 clause 时扩散；
3. exact anchor、短单句或候选已覆盖全部 clauses 时不扩散；
4. 用现有 query/clause embedding、edge type 和 evidence 便宜过滤邻居；
5. graph neighbour 仍只是 candidate，最多加 1 个，不直接 injection；
6. 优先 relation type：`CONTINUES`、`STATE_CHANGE`、`RESOLVES`、
   `CORRECTION`；不先放宽泛 `RELATES_TO`。

叙事弧不要先做自动总结。先建立可重建的 derived `arc_cluster`：

- 输入只用 reviewed Scene/Event typed relations、shared source message IDs、时间邻近、
  entities 和 embedding similarity；
- cluster 是导航/index，不是事实权威；
- 人工审过且稳定的 cluster 才可能发布成 Narrative Roll；
- ordinary chat 仍浮现具体 Scene/Event；只有宽泛回望，或至少多个独立 direct Scene
  hits 指向同一 published Roll 时，才考虑 Roll projection。

理想链路：

`query views -> passages -> parent candidates -> one reranker -> reliable seed ->`
`arc membership -> same-arc one-hop Scene edge -> uncovered-clause check -> max 1 shadow candidate`

Passage 找局部句子；reranker 选 seed；arc 限制故事线；Scene edge 只补遗漏的一环。

## 下一窗口精确执行顺序

1. 先读本文件和 `docs/2026-08-16-passage-shadow-evaluation.md`，不要重跑已经否掉的
   planner、critic、pool 12、8B 实验。
2. 已在本地完成 weak-trigger 控制执行的纯 shadow 版本；不要把它误报成已发布。
3. 下一步用 Terra 固定 gold suite 和 live probe/confounder pairs 统计：
   - 应触发而触发；
   - 不该触发却触发；
   - 新增正确 parent；
   - 新增明显无关 parent；
   - 端到端增量延迟。
4. 单独盯住聚焦长 query、猫狐纠正、token/byte current-state 反例、present chitchat
   和带上一轮共指的样本。
5. 只有弱触发 gate 稳定后，才另开 graph one-hop shadow；不要同一提交里加 arc cluster。
6. arc cluster 先离线产 proposal/review artifact，不进入 hot path，不生成新的 canonical
   summary。

## 验证和发布边界

最近一次 targeted green：

- `C:\Python313\python.exe -m py_compile gateway.py`
- `C:\Python313\python.exe scripts\verify_passage_candidate_simulation_shadow.py`
- `C:\Python313\python.exe scripts\verify_scene_edit.py`
- `C:\Python313\python.exe scripts\verify_passage_shadow.py`
- `C:\Python313\python.exe scripts\verify_cue_passage_shadow.py`
- `C:\Python313\python.exe scripts\verify_fact_events.py`
- `C:\Python313\python.exe scripts\verify_fact_event_semantic_shadow.py`
- `C:\Python313\python.exe scripts\verify_fact_event_lexical_shadow.py`
- `C:\Python313\python.exe scripts\verify_semantic_recall_cutover.py`
- Serein `npm run build`
- Serein `npm run test:recall-simulation`
- Serein `npm run test:sites`（4/4）
- `git diff --check`

所有 query-view、weak-trigger、passage expansion 和 Scene diffusion 都是 shadow。
不要把 expanded candidates 或 UI 诊断说成 live injection。

`f24e393` 已发布。若后续再改 tracked 文件，仍走唯一生产链：本地精确改动 -> commit ->
push 当前生产分支 -> 德国标准部署。不要 SCP/rsync 覆盖 tracked 运行文件。

## 一句话接手口径

分句 + source-bound passage 已证明能救回整句 embedding 漏掉的初遇 Scene；现在要做的
不是上线它，而是让它只在“候选真的弱”时才跑，并用固定正反例验证新增候选是否值得。
图只能从可靠 seed 补一跳，叙事弧只能先做限制图噪声的 derived index，二者都不能替代
第一跳检索，也不能直接获得注入权。
