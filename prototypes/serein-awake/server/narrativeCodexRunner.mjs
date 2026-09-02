import { spawn } from "node:child_process";
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

export const narrativeModelForMode = (mode) => (
  mode === "rewrite"
    ? { model: "gpt-5.6-sol", reasoningEffort: "medium" }
    : { model: "gpt-5.6-terra", reasoningEffort: "medium" }
);

export function buildNarrativeTaskPrompt({ mode, title, currentBody, materials }) {
  if (!new Set(["update", "rewrite"]).has(mode)) throw new Error("invalid_narrative_writer_mode");
  const task = {
    mode,
    title: String(title || "").trim(),
    materials,
    ...(mode === "update" ? { current_body: String(currentBody || "") } : {}),
  };
  if (!task.title || !materials || typeof materials !== "object") {
    throw new Error("invalid_narrative_writer_input");
  }
  return [
    "[Haven Internal]",
    "SYSTEM ACTION MODE: narrative_writer_preview, not user chat.",
    "按照本目录 AGENTS.md 完成一次叙事卷预览。不要调用工具，不要读写文件。",
    "只返回 output schema 要求的 JSON。",
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

export function parseCodexThreadId(stdout) {
  for (const line of String(stdout || "").split(/\r?\n/)) {
    if (!line.trim()) continue;
    try {
      const event = JSON.parse(line);
      if (event?.type === "thread.started" && typeof event.thread_id === "string") return event.thread_id;
      if (event?.type === "thread.started" && typeof event.thread?.id === "string") return event.thread.id;
    } catch {
      // Ignore non-JSON diagnostics; the output file remains authoritative.
    }
  }
  return "";
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
  const schemaPath = join(rolePath, "output.schema.json");
  if (!existsSync(join(rolePath, "AGENTS.md")) || !existsSync(schemaPath)) {
    throw new Error("narrative_writer_role_not_installed");
  }
  mkdirSync(tempRoot, { recursive: true, mode: 0o700 });
  const taskDir = mkdtempSync(join(tempRoot, "narrative-writer-"));
  const outputPath = join(taskDir, "result.json");
  const selection = narrativeModelForMode(mode);
  const prompt = buildNarrativeTaskPrompt({ mode, title, currentBody, materials });
  let threadId = "";
  let archived = false;
  let result;
  try {
    const args = [
      "exec",
      "--disable", "shell_tool",
      "--disable", "unified_exec",
      "--disable", "hooks",
      "--disable", "apps",
      "--ignore-user-config",
      "--sandbox", "read-only",
      "--skip-git-repo-check",
      "--model", selection.model,
      "-c", `model_reasoning_effort="${selection.reasoningEffort}"`,
      "-c", 'web_search="disabled"',
      "--cd", rolePath,
      "--output-schema", schemaPath,
      "--output-last-message", outputPath,
      "--json",
      "-",
    ];
    const run = await runProcess(codexCommand, args, {
      cwd: rolePath,
      input: prompt,
      timeoutMs: mode === "rewrite" ? 300_000 : 180_000,
    });
    threadId = parseCodexThreadId(run.stdout);
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
    };
  } finally {
    if (threadId) {
      const archive = await runProcess(codexCommand, ["archive", threadId], { cwd: rolePath, timeoutMs: 30_000 });
      archived = archive.code === 0;
      if (!archived) console.error(`[serein] Failed to archive Narrative Writer thread ${threadId}: ${archive.stderr.trim()}`);
    }
    rmSync(taskDir, { recursive: true, force: true });
  }
  if (threadId && !archived) throw new Error("narrative_writer_thread_archive_failed");
  return { ...result, thread_archived: archived };
}
