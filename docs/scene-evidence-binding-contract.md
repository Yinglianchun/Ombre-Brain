# Scene 原文证据绑定合同

Scene Markdown 仍是 canonical 正文。证据只写入独立的
`state/scene_evidence.sqlite`，不会重写正文、frontmatter、Scene embedding
或召回状态。

- `write_scene(..., evidence_refs=[])`：不传证据仍正常写 Scene，并返回真实
  `scene_id`、`evidence_status=unbound`；传入证据时只有 `bound` 才算完整成功。
- `bind_scene_evidence(scene_id, evidence_refs, bound_by="")`：只追加证据；相同
  消息键与相同原文幂等，相同键但正文或 hash 漂移时拒绝。
- `read_scene_evidence(scene_id)`：读取某张 Scene 的证据状态和精确原文。

`evidence_ref` 包含 `source_system`、`session_id`/`thread_id`、`message_id`、
`role`、`created_at`、精确 `content`、UTF-8 `content_sha256`、
`binding_method` 与 `evidence_kind`。`evidence_kind` 只允许 `primary`、
`supporting`、`adjacent_context`。

Bridge 和 Serein 服务端负责重新读取可见原文并计算 hash；浏览器不能提交正文
或 hash。旧的已同步候选只能调用 `bind_scene_evidence` 补绑，不能重放
`write_scene`。
