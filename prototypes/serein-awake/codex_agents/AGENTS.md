# Narrative job routing

叙事卷后台只允许两种彼此隔离的语义任务：

- `narrative_scout`：使用 `https://756777.xyz/v1` 的 `gpt-5.6-terra`。它只从 host 已完成的关键词 one-hop corridor 中提出新卷 review candidate，不写正文、不改材料、不发布。
- `narrative_writer`：使用 ephemeral Codex job 的 `gpt-5.6-sol`。`update` 与 `rewrite` 都读取冻结后的完整材料；它只产预览，不自行检索、增删材料或发布。

两条任务都 fail closed。不得在模型、JSON、超时或凭据失败时回退到普通上游、脱水模型、Scene Linker 或另一个角色。
