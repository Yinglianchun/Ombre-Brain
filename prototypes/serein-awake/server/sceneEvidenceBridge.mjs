import { createHash } from "node:crypto";

export const evidenceKinds = new Set(["primary", "supporting", "adjacent_context"]);

export function contentSha256(content) {
  if (typeof content !== "string") throw new Error("evidence_content_invalid");
  return createHash("sha256").update(content, "utf8").digest("hex");
}

export function buildSceneEvidenceRefs(messages, selections) {
  if (!Array.isArray(messages) || !Array.isArray(selections) || !selections.length) {
    throw new Error("evidence_selection_required");
  }
  const byId = new Map(messages.map((message) => [String(message?.id || ""), message]));
  const seen = new Set();
  return selections.map((selection) => {
    const messageId = String(selection?.messageId || "").trim();
    if (!messageId || seen.has(messageId)) throw new Error("evidence_selection_invalid");
    seen.add(messageId);
    const message = byId.get(messageId);
    if (!message) throw new Error("evidence_message_missing");
    const content = message.content;
    const sessionId = String(message.session_id || "").trim();
    const role = String(message.role || "").trim().toLowerCase();
    const createdAt = String(message.created_at || "").trim();
    if (typeof content !== "string" || !content.trim() || !sessionId || !createdAt) {
      throw new Error("evidence_message_invalid");
    }
    if (!new Set(["user", "assistant"]).has(role)) throw new Error("evidence_role_invalid");
    const evidenceKind = String(selection?.evidenceKind || "primary").trim().toLowerCase();
    if (!evidenceKinds.has(evidenceKind)) throw new Error("evidence_kind_invalid");
    return {
      source_system: "haven_bridge",
      session_id: sessionId,
      thread_id: String(message.thread_id || "").trim(),
      message_id: messageId,
      role,
      created_at: createdAt,
      content,
      content_sha256: contentSha256(content),
      evidence_kind: evidenceKind,
      binding_method: "serein_manual_selection",
    };
  });
}
