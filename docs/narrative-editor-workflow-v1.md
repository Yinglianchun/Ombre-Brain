# Narrative 编辑工作流 v1

这份文件描述 UI、host 与存储层的职责，不进入 Narrative Writer prompt。

## UI 操作

### 添加材料

- 绑定已有 Event、Scene、Diary、Window Shadow 或补充材料；
- 支持从用户本地上传文件；
- 只修改材料目录，不调用 Writer，不改标题与正文；
- 显示哪些材料尚未进入当前正文 revision。

本地文件保留原文件，并记录稳定 material ID、原文件名、MIME、大小和内容 hash。可可靠提取的文字进入 Writer 阅读材料；无法提取时明确标记，不生成或猜测文件内容。上传文件不取得 Event 或聊天原文 ownership。

### 保存正文

- 保存用户在编辑器中的手动修改；
- 不添加或移除材料；
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
- Writer 只能返回正文草稿和证据问题，不提供 membership、Event ownership 或 Arc 边界字段；
- 材料缺失、漂移、冲突或无法读取时停止生成预览；
- preview 不写线上。用户确认后，才以 CAS 发布新的 Narrative revision；
- 发布正文不得隐式添加、移除或替换材料。

按钮、文件上传、diff、CAS、revision、材料核验和发布确认等工作流说明不进入 Writer prompt。

## 当前实现阶段

当前实现已经包含 `update` / `rewrite` 的只读预览，以及正文手改后的 `save-body`：保存会创建新的 Narrative revision，并精确保留当前 membership，不调用 Writer。材料 membership 基线与本地文件上传仍在后续阶段接入；UI 不伪装这些能力已经可用。

Writer 由 Serein 主机上的 Codex 单次任务执行。`update` 使用 `gpt-5.6-terra`，`rewrite` 使用 `gpt-5.6-sol`，两者 reasoning effort 均为 `medium`。角色规则只放在 `prototypes/serein-awake/codex_agents/narrative_writer/AGENTS.md`；动态任务只携带 mode、标题、当前正文（仅 update）和后端冻结的当前绑定材料。任务以 read-only sandbox 运行，不调用工具；完成后立即归档 Codex thread，并删除临时输入输出文件。

这条链不会定时或随材料变化自动生成正文。只有用户点击“更新”或“重写”才会创建一次预览任务；预览仍然不写 Narrative registry。

修订箱每天 `04:00 Asia/Shanghai` 运行一次派生扫描。已有卷只用程序比较 `published_at` 与当前绑定材料的最新 `updated_at`，较新的材料会生成“需要更新”提示；未归卷的高重要度 Event 只把标题、摘要与 ID 交给现有外部模型，模型只能提出“可能成卷”的分组。两类结果都只是修订箱提示，不调用 Narrative Writer、不创建叙事卷，也不发布正文。修订箱里的“重写”仍是一次明确的用户点击，生成预览后还必须另点“保存”。
