import assert from "node:assert/strict";

import {
  buildNarrativeTaskPrompt,
  narrativeBodyDiff,
  narrativeModelForMode,
  normalizeNarrativeWriterResult,
  parseCodexThreadId,
} from "../server/narrativeCodexRunner.mjs";

const review = {
  source_bound: true,
  final_supported_versions: true,
  no_correction_narration: true,
  material_relevance: true,
  no_new_inference: true,
  no_meta_explanation: true,
  no_forced_closure: true,
  dates_preserved: true,
  identity_correct: true,
};

assert.deepEqual(narrativeModelForMode("update"), {
  model: "gpt-5.6-terra",
  reasoningEffort: "medium",
});
assert.deepEqual(narrativeModelForMode("rewrite"), {
  model: "gpt-5.6-sol",
  reasoningEffort: "medium",
});

const updatePrompt = buildNarrativeTaskPrompt({
  mode: "update",
  title: "归航",
  currentBody: "旧正文",
  materials: { events: [{ event_id: "event_1" }] },
});
assert.match(updatePrompt, /"current_body":"旧正文"/);
const rewritePrompt = buildNarrativeTaskPrompt({
  mode: "rewrite",
  title: "归航",
  currentBody: "旧正文",
  materials: { events: [{ event_id: "event_1" }] },
});
assert.doesNotMatch(rewritePrompt, /current_body/);

assert.equal(normalizeNarrativeWriterResult({
  evidence_sufficient: true,
  body: " 新正文 ",
  issues: [],
  self_review: review,
}).body, "新正文");

assert.equal(normalizeNarrativeWriterResult({
  evidence_sufficient: false,
  body: "",
  issues: ["材料不足"],
  self_review: { ...review, source_bound: false },
}).evidence_sufficient, false);

assert.match(narrativeBodyDiff("第一行\n旧行\n末行", "第一行\n新行\n末行"), /-旧行\n\+新行/);
assert.equal(parseCodexThreadId([
  JSON.stringify({ type: "thread.started", thread_id: "019d-test" }),
  JSON.stringify({ type: "turn.completed" }),
].join("\n")), "019d-test");

console.log("NARRATIVE_CODEX_RUNNER_OK");

