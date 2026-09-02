import assert from "node:assert/strict";

import {
  buildNarrativeTaskPrompt,
  narrativeBodyDiff,
  narrativeCodexArgs,
  narrativeModelForMode,
  normalizeNarrativeWriterResult,
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

const codexArgs = narrativeCodexArgs({
  selection: narrativeModelForMode("update"),
  taskDir: "/tmp/task",
  schemaPath: "/roles/output.schema.json",
  outputPath: "/tmp/task/result.json",
});
assert.ok(codexArgs.includes("--ephemeral"));
assert.ok(codexArgs.includes("--ignore-rules"));
assert.equal(codexArgs[codexArgs.indexOf("--cd") + 1], "/tmp/task");
assert.ok(!codexArgs.includes("archive"));

const updatePrompt = buildNarrativeTaskPrompt({
  mode: "update",
  title: "归航",
  currentBody: "旧正文",
  materials: { events: [{ event_id: "event_1" }] },
  roleRules: "只使用绑定材料。",
});
assert.match(updatePrompt, /"current_body":"旧正文"/);
assert.match(updatePrompt, /<narrative_writer_role_rules>\n只使用绑定材料。/);
const rewritePrompt = buildNarrativeTaskPrompt({
  mode: "rewrite",
  title: "归航",
  currentBody: "旧正文",
  materials: { events: [{ event_id: "event_1" }] },
  roleRules: "只使用绑定材料。",
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
console.log("NARRATIVE_CODEX_RUNNER_OK");
