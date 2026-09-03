# Narrative 编辑工作流 v1

这份文件描述 UI、host 与存储层的职责，不进入 Narrative Writer prompt。

## UI 操作

### 添加材料

- 编辑器可按精确 ID 增加或移除已有 Event、Scene、Diary、Darkroom，也可从本地上传一份文件并立即加入拟绑定材料；
- 材料 proposal 与正文一起进入预览，明确显示完整目标 membership 以及 added / removed；
- 删除只改变 Narrative membership，不删除或改写 canonical Event、Scene、Diary、Darkroom，也不删除已上传的原文件；
- 本地文件以内容哈希生成稳定 `upload_id`，保留原始字节、文件名、MIME、大小与 SHA-256。UTF-8 文本、Markdown、CSV、JSON、YAML、日志及 DOCX 会尽量提取文字供 Writer 阅读；其他格式只作为有来源的附件记录，不推断内容；
- 单个文件上限为 10 MB。上传完成本身只创建材料，不会改 Narrative；仍须经过预览并确认保存，才会绑定到当前卷。

### 保存正文

- 保存用户在编辑器中的手动修改；
- 材料选择不变时精确保留原 membership；
- 不调用 Writer。

### 更新叙事卷

- host 读取当前已发布正文，以及当前正文 revision 之后新增或变更的材料；
- Writer 以旧正文为主体，把新增内容整合到合适位置，只做衔接所需的局部调整；
- 生成更新预览与 diff，不直接发布。

### 全篇重写

- host 重新读取当前全部绑定材料；
- Writer 从头组织正文，旧正文不作为必须保留的结构；
- 生成完整重写预览与 diff，不直接发布。

## Host 合同

- 调用 Writer 前冻结 mode、Narrative revision、材料 membership 与每份材料的 revision、fingerprint 或 content hash；
- 重要内容优先物化绑定原文，不只把 Event 标题或旧正文摘要交给 Writer；
- Writer 只能返回正文草稿和证据问题；membership proposal/delta 由 host 明确附在 preview 外层，不由 Writer 藏进正文；
- 材料缺失、漂移、冲突或无法读取时停止生成预览；
- preview 不写线上。用户确认后，才以 expected revision、document hash 与 preview fingerprint CAS 发布新的 Narrative revision；
- 保存前重新读取目标 membership 的全部材料并核对 active 状态与内容快照；任何缺失、漂移或 fingerprint 不匹配都 fail closed；
- 发布正文不得隐式添加、移除或替换材料。

按钮、文件上传、diff、CAS、revision、材料核验和发布确认等工作流说明不进入 Writer prompt。

## 当前实现阶段

当前实现已经包含 `edit` / `update` / `rewrite` 的只读预览封印。UI 可为 Event、Scene、Diary、Darkroom 和本地上传材料提出精确增删；预览显式返回目标 IDs、delta 与材料 snapshot hash。保存会 fresh-read 全部目标材料、复算 preview fingerprint，并在同一 revision 中写正文与 membership。只改正文时目标 IDs 默认为当前绑定，因此不会意外清空或扩张材料。旧版只提交四类材料的客户端会保留既有上传材料，不会在滚动发布期间误删。

Writer 由 Serein 主机上的 Codex ephemeral 单次任务执行。`update` 与 `rewrite` 都使用 `gpt-5.6-sol`，reasoning effort 为 `medium`。Host 显式读取 `prototypes/serein-awake/codex_agents/narrative_writer/AGENTS.md` 并注入冻结 prompt；动态任务只携带 mode、标题、当前正文（仅 update）和后端冻结的当前绑定材料。任务以 read-only sandbox、空临时目录、`--ignore-rules` 和 `--ephemeral` 运行，不调用工具、不创建或归档持久 Codex thread；完成后删除临时输入输出文件。

这条链不会定时或随材料变化自动生成正文。只有用户点击“更新”或“重写”才会创建一次预览任务；预览仍然不写 Narrative registry。

修订箱每天 `04:00 Asia/Shanghai` 运行一次派生扫描。已有卷只用程序比较 `published_at` 与当前绑定材料的最新 `updated_at`，较新的材料会生成“需要更新”提示；未归卷的高重要度 Event 只把标题、摘要与 ID 交给现有外部模型，模型只能提出“可能成卷”的分组。两类结果都只是修订箱提示，不调用 Narrative Writer、不创建叙事卷，也不发布正文。修订箱里的“重写”仍是一次明确的用户点击，生成预览后还必须另点“保存”。
