import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const reviewKeys = [
  "source_bound",
  "final_supported_versions",
  "no_correction_narration",
  "material_relevance",
  "no_new_inference",
  "no_meta_explanation",
  "no_forced_closure",
  "dates_preserved",
  "identity_correct",
];

export const narrativeModelForMode = (mode) => {
  if (!new Set(["update", "rewrite"]).has(mode)) throw new Error("invalid_narrative_writer_mode");
  return { model: "gpt-5.6-sol", reasoningEffort: "medium" };
};

export function buildNarrativeTaskPrompt({ mode, title, currentBody, materials, roleRules }) {
  if (!new Set(["update", "rewrite"]).has(mode)) throw new Error("invalid_narrative_writer_mode");
  const task = {
    mode,
    title: String(title || "").trim(),
    materials,
    ...(mode === "update" ? { current_body: String(currentBody || "") } : {}),
  };
  const rules = String(roleRules || "").trim();
  if (!task.title || !materials || typeof materials !== "object" || !rules) {
    throw new Error("invalid_narrative_writer_input");
  }
  return [
    "[Haven Internal]",
    "SYSTEM ACTION MODE: narrative_writer_preview, not user chat.",
    "The host supplied the complete role rules and frozen material below. Do not call tools or read files.",
    "只返回 output schema 要求的 JSON。",
    "",
    "<narrative_writer_role_rules>",
    rules,
    "</narrative_writer_role_rules>",
    "",
    "<narrative_writer_input_json>",
    JSON.stringify(task),
    "</narrative_writer_input_json>",
  ].join("\n");
}

export function normalizeNarrativeWriterResult(value) {
  const result = typeof value === "string" ? JSON.parse(value) : value;
  if (!result || typeof result !== "object" || Array.isArray(result)) {
    throw new Error("narrative_writer_result_not_object");
  }
  const keys = Object.keys(result).sort().join(",");
  if (keys !== ["body", "evidence_sufficient", "issues", "self_review"].sort().join(",")) {
    throw new Error("narrative_writer_result_schema_invalid");
  }
  if (typeof result.evidence_sufficient !== "boolean" || typeof result.body !== "string") {
    throw new Error("narrative_writer_result_types_invalid");
  }
  if (!Array.isArray(result.issues) || result.issues.some((item) => typeof item !== "string")) {
    throw new Error("narrative_writer_issues_invalid");
  }
  const review = result.self_review;
  if (!review || typeof review !== "object" || Array.isArray(review)) {
    throw new Error("narrative_writer_review_invalid");
  }
  if (Object.keys(review).sort().join(",") !== [...reviewKeys].sort().join(",")) {
    throw new Error("narrative_writer_review_schema_invalid");
  }
  if (reviewKeys.some((key) => typeof review[key] !== "boolean")) {
    throw new Error("narrative_writer_review_types_invalid");
  }
  const body = result.body.trim();
  const issues = result.issues.map((item) => item.trim()).filter(Boolean);
  if (result.evidence_sufficient && (!body || issues.length || reviewKeys.some((key) => !review[key]))) {
    throw new Error("narrative_writer_sufficient_result_invalid");
  }
  if (!result.evidence_sufficient && (body || !issues.length || review.source_bound)) {
    throw new Error("narrative_writer_insufficient_result_invalid");
  }
  return { ...result, body, issues, self_review: { ...review } };
}

export function narrativeBodyDiff(currentBody, proposedBody) {
  const before = String(currentBody || "").replace(/\r\n?/g, "\n").split("\n");
  const after = String(proposedBody || "").replace(/\r\n?/g, "\n").split("\n");
  if (before.join("\n") === after.join("\n")) return "";
  let prefix = 0;
  while (prefix < before.length && prefix < after.length && before[prefix] === after[prefix]) prefix += 1;
  let suffix = 0;
  while (
    suffix < before.length - prefix
    && suffix < after.length - prefix
    && before[before.length - 1 - suffix] === after[after.length - 1 - suffix]
  ) suffix += 1;
  const removed = before.slice(prefix, before.length - suffix);
  const added = after.slice(prefix, after.length - suffix);
  return [
    "--- current",
    "+++ preview",
    `@@ -${prefix + 1},${removed.length} +${prefix + 1},${added.length} @@`,
    ...removed.map((line) => `-${line}`),
    ...added.map((line) => `+${line}`),
  ].join("\n");
}

function workerEnvironment() {
  const allowed = new Set([
    "HOME", "LANG", "LC_ALL", "PATH", "SSL_CERT_DIR", "SSL_CERT_FILE", "TMPDIR", "CODEX_HOME",
  ]);
  const clean = Object.fromEntries(Object.entries(process.env).filter(([key]) => allowed.has(key)));
  clean.HOME = clean.HOME || "/root";
  clean.CODEX_HOME = clean.CODEX_HOME || "/root/.codex";
  clean.PATH = clean.PATH || "/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin";
  return clean;
}

function runProcess(command, args, { cwd, input = "", timeoutMs = 300_000 } = {}) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(command, args, {
      cwd,
      env: workerEnvironment(),
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true,
    });
    let stdout = "";
    let stderr = "";
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      child.kill("SIGKILL");
    }, timeoutMs);
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", (error) => { clearTimeout(timer); reject(error); });
    child.on("close", (code) => {
      clearTimeout(timer);
      resolvePromise({ code, stdout, stderr, timedOut });
    });
    child.stdin.end(input);
  });
}

export function narrativeCodexArgs({ selection, taskDir, schemaPath, outputPath }) {
  return [
    "exec",
    "--ephemeral",
    "--disable", "shell_tool",
    "--disable", "unified_exec",
    "--disable", "hooks",
    "--disable", "apps",
    "--ignore-user-config",
    "--ignore-rules",
    "--sandbox", "read-only",
    "--skip-git-repo-check",
    "--model", selection.model,
    "-c", `model_reasoning_effort="${selection.reasoningEffort}"`,
    "-c", 'web_search="disabled"',
    "--cd", taskDir,
    "--output-schema", schemaPath,
    "--output-last-message", outputPath,
    "--json",
    "-",
  ];
}

export async function runNarrativeCodexTask({
  mode,
  title,
  currentBody,
  materials,
  roleDir,
  codexCommand = process.env.SEREIN_CODEX_CLI || "/root/.local/bin/codex",
  tempRoot = process.env.SEREIN_CODEX_TASK_DIR || join(tmpdir(), "serein-codex-tasks"),
}) {
  const rolePath = resolve(roleDir);
  const rulesPath = join(rolePath, "AGENTS.md");
  const schemaPath = join(rolePath, "output.schema.json");
  if (!existsSync(rulesPath) || !existsSync(schemaPath)) {
    throw new Error("narrative_writer_role_not_installed");
  }
  const roleRules = readFileSync(rulesPath, "utf8").trim();
  if (!roleRules) throw new Error("narrative_writer_role_rules_empty");
  const roleRulesSha256 = createHash("sha256").update(roleRules, "utf8").digest("hex");
  mkdirSync(tempRoot, { recursive: true, mode: 0o700 });
  const taskDir = mkdtempSync(join(tempRoot, "narrative-writer-"));
  const outputPath = join(taskDir, "result.json");
  const selection = narrativeModelForMode(mode);
  const prompt = buildNarrativeTaskPrompt({ mode, title, currentBody, materials, roleRules });
  let result;
  try {
    const args = narrativeCodexArgs({ selection, taskDir, schemaPath, outputPath });
    const run = await runProcess(codexCommand, args, {
      cwd: taskDir,
      input: prompt,
      timeoutMs: mode === "rewrite" ? 300_000 : 180_000,
    });
    if (run.timedOut || run.code !== 0 || !existsSync(outputPath)) {
      throw new Error(run.timedOut ? "narrative_writer_timeout" : `narrative_writer_codex_${run.code}:${run.stderr.trim().slice(0, 300)}`);
    }
    const normalized = normalizeNarrativeWriterResult(readFileSync(outputPath, "utf8"));
    result = {
      status: normalized.evidence_sufficient ? "ok" : "insufficient",
      ...normalized,
      mode,
      provider: selection.model,
      reasoning_effort: selection.reasoningEffort,
      diff: narrativeBodyDiff(currentBody, normalized.body),
      publication_status: "not_published",
      writes_performed: [],
      execution_mode: "ephemeral",
      session_persisted: false,
      role_rules_sha256: roleRulesSha256,
    };
  } finally {
    rmSync(taskDir, { recursive: true, force: true });
  }
  return result;
}
