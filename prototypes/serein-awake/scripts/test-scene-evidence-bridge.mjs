import assert from "node:assert/strict";
import {
  buildSceneEvidenceRefs,
  contentSha256,
  normalizeEvidenceMessageId,
  normalizeEvidenceSearchQuery,
} from "../server/sceneEvidenceBridge.mjs";

const content = "原文保留空格  \n和换行。";
const refs = buildSceneEvidenceRefs([
  {
    id: 42,
    session_id: 7,
    thread_id: "thread-7",
    role: "user",
    created_at: "2026-08-04T03:00:00Z",
    content,
  },
], [{ messageId: 42, evidenceKind: "supporting" }]);

assert.equal(refs.length, 1);
assert.equal(refs[0].content, content);
assert.equal(refs[0].content_sha256, contentSha256(content));
assert.equal(refs[0].evidence_kind, "supporting");
assert.equal(refs[0].binding_method, "serein_manual_selection");
assert.equal(normalizeEvidenceSearchQuery("  原文关键词  "), "原文关键词");
assert.equal(normalizeEvidenceSearchQuery("x".repeat(160)).length, 120);
assert.equal(normalizeEvidenceMessageId("7675"), 7675);
assert.equal(normalizeEvidenceMessageId("nope"), 0);
assert.throws(() => buildSceneEvidenceRefs([], [{ messageId: 42 }]), /evidence_message_missing/);
assert.throws(
  () => buildSceneEvidenceRefs([{ id: 42, session_id: 7, role: "trace", created_at: "now", content }], [{ messageId: 42 }]),
  /evidence_role_invalid/,
);
assert.throws(
  () => buildSceneEvidenceRefs([{ id: 42, session_id: 7, role: "user", created_at: "now", content }], [{ messageId: 42 }, { messageId: 42 }]),
  /evidence_selection_invalid/,
);

console.log("scene evidence bridge verification passed");
