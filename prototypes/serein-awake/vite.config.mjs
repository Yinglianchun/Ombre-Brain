import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { spawn } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import {
  buildSceneEvidenceRefs,
  normalizeEvidenceMessageId,
  normalizeEvidenceSearchQuery,
} from "./server/sceneEvidenceBridge.mjs";
import { narrativeBodyDiff, runNarrativeCodexTask } from "./server/narrativeCodexRunner.mjs";
import { buildNarrativePreviewFingerprint } from "./server/narrativeMaterialPreview.mjs";

const narrativeWriterRoleDir = fileURLToPath(new URL("./codex_agents/narrative_writer/", import.meta.url));

const canonicalSceneDomains = new Set([
  "relationship",
  "intimacy",
  "inner",
  "life",
  "tech",
  "project",
  "general",
]);

function secretValue(name) {
  const direct = String(process.env[name] || "").trim();
  if (direct) return direct;
  const file = String(process.env[`${name}_FILE`] || "").trim();
  if (!file) return "";
  try {
    return String(readFileSync(file, "utf8") || "").trim();
  } catch {
    return "";
  }
}

function semanticRouteDraftPath() {
  return String(
    process.env.SEREIN_ROUTE_DRAFT_FILE || "/var/lib/serein-awake/semantic-route-draft.json",
  ).trim();
}

function readServerSemanticRouteDraft() {
  const file = semanticRouteDraftPath();
  if (!file || !existsSync(file)) return null;
  try {
    const draft = JSON.parse(readFileSync(file, "utf8"));
    return Array.isArray(draft?.routes) ? draft : null;
  } catch {
    return null;
  }
}

function saveServerSemanticRouteDraft(body) {
  if (!Array.isArray(body?.routes) || body.routes.length < 1 || body.routes.length > 50) {
    throw new Error("route_draft_invalid");
  }
  const baseDatasetVersion = Number.parseInt(body.baseDatasetVersion, 10);
  if (!Number.isInteger(baseDatasetVersion) || baseDatasetVersion < 1) {
    throw new Error("route_draft_version_invalid");
  }
  const current = readServerSemanticRouteDraft();
  const expectedRevision = body.expectedRevision == null
    ? null
    : Number.parseInt(body.expectedRevision, 10);
  const currentRevision = Number.parseInt(current?.revision, 10) || 0;
  if (expectedRevision != null && expectedRevision !== currentRevision) {
    const error = new Error("route_draft_revision_conflict");
    error.current = current;
    throw error;
  }
  const record = {
    schemaVersion: 1,
    baseDatasetVersion,
    revision: currentRevision + 1,
    updatedAt: new Date().toISOString(),
    routes: body.routes,
  };
  const file = semanticRouteDraftPath();
  mkdirSync(dirname(file), { recursive: true, mode: 0o700 });
  const temporary = `${file}.${process.pid}.${randomUUID()}.tmp`;
  writeFileSync(temporary, `${JSON.stringify(record, null, 2)}\n`, { encoding: "utf8", mode: 0o600 });
  renameSync(temporary, file);
  return record;
}

function clearServerSemanticRouteDraft() {
  const file = semanticRouteDraftPath();
  if (file && existsSync(file)) rmSync(file);
}

async function readJsonBody(request, maxBytes = 32_768) {
  const chunks = [];
  let size = 0;
  for await (const chunk of request) {
    size += chunk.length;
    if (size > maxBytes) throw new Error("request_too_large");
    chunks.push(chunk);
  }
  return JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
}

export function normalizeRecallSimulationOptions(body = {}) {
  const recallAblation = String(body.recall_ablation || "normal").trim().toLowerCase();
  const simulationScope = String(body.simulation_scope || "live_mirror").trim().toLowerCase();
  const allowedRecallAblations = new Set(["normal", "without_cues", "without_embedding"]);
  const allowedSimulationScopes = new Set(["live_mirror", "full_shadow"]);
  const simulation = [true, 1, "1", "true", "yes", "on"].includes(body.simulation);
  if (!allowedSimulationScopes.has(simulationScope)) {
    return { ok: false, error: "invalid_simulation_scope", message: "模拟范围不正确。" };
  }
  if (!allowedRecallAblations.has(recallAblation)) {
    return { ok: false, error: "invalid_recall_ablation", message: "消融模式不正确。" };
  }
  if (recallAblation !== "normal" && !simulation) {
    return {
      ok: false,
      error: "recall_ablation_requires_simulation",
      message: "消融只允许从召回模拟发起。",
    };
  }
  if (recallAblation !== "normal" && simulationScope !== "full_shadow") {
    return {
      ok: false,
      error: "recall_ablation_requires_full_shadow",
      message: "消融只在完整 shadow 诊断里运行。",
    };
  }
  return { ok: true, recallAblation, simulation, simulationScope };
}

function parseMcpEvent(text) {
  const data = String(text || "")
    .split(/\r?\n/)
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trim())
    .join("\n");
  if (!data) throw new Error("mcp_response_missing");
  return JSON.parse(data);
}

async function callOmbreTool(name, args) {
  const gatewayToken = String(process.env.OMBRE_GATEWAY_TOKEN || "").trim();
  const ombreBase = String(process.env.OMBRE_MEMORY_URL || "http://8.136.154.242:18001").replace(/\/$/, "");
  if (!gatewayToken) throw new Error("memory_bridge_not_configured");

  const headers = {
    Accept: "application/json, text/event-stream",
    Authorization: `Bearer ${gatewayToken}`,
    "Content-Type": "application/json",
  };
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 20_000);
  let sessionId = "";

  try {
    const initialize = await fetch(`${ombreBase}/mcp`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        jsonrpc: "2.0",
        id: randomUUID(),
        method: "initialize",
        params: {
          protocolVersion: "2025-06-18",
          capabilities: {},
          clientInfo: { name: "serein-basement", version: "0.1" },
        },
      }),
      signal: controller.signal,
    });
    if (!initialize.ok) throw new Error(`mcp_initialize_${initialize.status}`);
    sessionId = String(initialize.headers.get("mcp-session-id") || "").trim();
    if (!sessionId) throw new Error("mcp_session_missing");
    parseMcpEvent(await initialize.text());

    const sessionHeaders = { ...headers, "Mcp-Session-Id": sessionId };
    await fetch(`${ombreBase}/mcp`, {
      method: "POST",
      headers: sessionHeaders,
      body: JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" }),
      signal: controller.signal,
    });

    const upstream = await fetch(`${ombreBase}/mcp`, {
      method: "POST",
      headers: sessionHeaders,
      body: JSON.stringify({
        jsonrpc: "2.0",
        id: randomUUID(),
        method: "tools/call",
        params: { name, arguments: args },
      }),
      signal: controller.signal,
    });
    if (!upstream.ok) throw new Error(`mcp_call_${upstream.status}`);
    const envelope = parseMcpEvent(await upstream.text());
    if (envelope.error) throw new Error(envelope.error.message || "mcp_tool_failed");
    const toolResult = envelope.result;
    if (toolResult?.isError) throw new Error(toolResult.content?.[0]?.text || "mcp_tool_failed");
    const text = toolResult?.content?.find((item) => item.type === "text")?.text;
    return text ? JSON.parse(text) : toolResult;
  } finally {
    clearTimeout(timer);
    if (sessionId) {
      fetch(`${ombreBase}/mcp`, {
        method: "DELETE",
        headers: { ...headers, "Mcp-Session-Id": sessionId },
      }).catch(() => {});
    }
  }
}

async function callOmbreDashboardOverSsh(path, { method = "GET", body } = {}) {
  const sshTarget = String(process.env.OMBRE_VPS_SSH_TARGET || "root@8.136.154.242").trim();
  const requestPayload = Buffer.from(JSON.stringify({ path, method, body }), "utf8").toString("base64");
  const script = `
import base64, http.cookiejar, json, subprocess, urllib.error, urllib.request
request = json.loads(base64.b64decode("${requestPayload}").decode("utf-8"))
password = subprocess.check_output(["docker", "exec", "ombre-brain", "printenv", "OMBRE_DASHBOARD_PASSWORD"], text=True).strip()
cookies = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookies))
login = urllib.request.Request("http://127.0.0.1:18001/auth/login", data=json.dumps({"password": password}).encode(), headers={"Content-Type": "application/json"}, method="POST")
opener.open(login, timeout=12).read()
data = None if request.get("body") is None else json.dumps(request["body"]).encode()
headers = {"Accept": "application/json"}
if data is not None:
    headers["Content-Type"] = "application/json"
upstream = urllib.request.Request("http://127.0.0.1:18001" + request["path"], data=data, headers=headers, method=request.get("method") or "GET")
try:
    response = opener.open(upstream, timeout=18)
    status = response.status
    text = response.read().decode("utf-8")
except urllib.error.HTTPError as error:
    status = error.code
    text = error.read().decode("utf-8")
try:
    payload = json.loads(text or "{}")
except Exception:
    payload = {"error": "dashboard_response_invalid"}
print(json.dumps({"status": status, "payload": payload}, ensure_ascii=False))
`;

  return new Promise((resolve, reject) => {
    const child = spawn("ssh", ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10", sshTarget, "python3", "-"], {
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true,
    });
    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => child.kill(), 25_000);
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", (error) => { clearTimeout(timer); reject(error); });
    child.on("close", (code) => {
      clearTimeout(timer);
      if (code !== 0) {
        reject(new Error(stderr.trim() || `dashboard_ssh_${code}`));
        return;
      }
      try {
        const result = JSON.parse(stdout.trim());
        resolve({ ok: result.status >= 200 && result.status < 300, status: result.status, payload: result.payload });
      } catch {
        reject(new Error("dashboard_ssh_response_invalid"));
      }
    });
    child.stdin.end(script);
  });
}

async function callOmbreDashboard(path, { method = "GET", body } = {}) {
  const password = secretValue("OMBRE_DASHBOARD_PASSWORD");
  const ombreBase = String(process.env.OMBRE_MEMORY_URL || "http://8.136.154.242:18001").replace(/\/$/, "");
  if (!password) {
    if (String(process.env.OMBRE_DISABLE_SSH_READS || "").trim() === "1") {
      throw new Error("dashboard_bridge_not_configured");
    }
    return callOmbreDashboardOverSsh(path, { method, body });
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 20_000);
  try {
    const login = await fetch(`${ombreBase}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password }),
      signal: controller.signal,
    });
    if (!login.ok) throw new Error(`dashboard_login_${login.status}`);
    const cookie = String(login.headers.get("set-cookie") || "").split(";")[0].trim();
    if (!cookie) throw new Error("dashboard_session_missing");

    const upstream = await fetch(`${ombreBase}${path}`, {
      method,
      headers: {
        Accept: "application/json",
        Cookie: cookie,
        ...(body === undefined ? {} : { "Content-Type": "application/json" }),
      },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal,
    });
    const text = await upstream.text();
    let payload;
    try {
      payload = JSON.parse(text || "{}");
    } catch {
      payload = { error: "dashboard_response_invalid" };
    }
    return { ok: upstream.ok, status: upstream.status, payload };
  } finally {
    clearTimeout(timer);
  }
}

function runOmbreReadScript(script, errorCode, timeoutMs = 30_000) {
  const sshTarget = String(process.env.OMBRE_VPS_SSH_TARGET || "root@8.136.154.242").trim();
  const localComposeFile = String(process.env.OMBRE_LOCAL_COMPOSE_FILE || "").trim();
  if (!localComposeFile && String(process.env.OMBRE_DISABLE_SSH_READS || "").trim() === "1") {
    return Promise.reject(new Error(`${errorCode}_not_configured`));
  }
  return new Promise((resolve, reject) => {
    const command = localComposeFile ? "docker" : "ssh";
    const args = localComposeFile
      ? ["compose", "-f", localComposeFile, "exec", "-T", "ombre-brain", "python", "-"]
      : [
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "KexAlgorithms=curve25519-sha256,curve25519-sha256@libssh.org",
        sshTarget,
        "python3",
        "-",
      ];
    const child = spawn(
      command,
      args,
      { stdio: ["pipe", "pipe", "pipe"], windowsHide: true },
    );
    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => child.kill(), timeoutMs);
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", (error) => { clearTimeout(timer); reject(error); });
    child.on("close", (code) => {
      clearTimeout(timer);
      if (code !== 0) {
        reject(new Error(stderr.trim() || `${errorCode}_${code}`));
        return;
      }
      try {
        resolve(JSON.parse(stdout.trim()));
      } catch {
        reject(new Error(`${errorCode}_invalid`));
      }
    });
    child.stdin.end(script);
  });
}

async function readLiveMemoryProjection(sourceIds) {
  const safeSourceIds = Array.from(new Set(
    (Array.isArray(sourceIds) ? sourceIds : [])
      .map((sourceId) => String(sourceId || "").trim())
      .filter((sourceId) => /^[A-Za-z0-9_.:#-]{1,160}$/.test(sourceId)),
  )).slice(0, 400);
  const upstream = await callOmbreDashboard("/api/serein/memory-projection", {
    method: "POST",
    body: { source_ids: safeSourceIds },
  });
  if (!upstream.ok) throw new Error(`memory_live_projection_${upstream.status}`);
  return upstream.payload;
}

async function readLiveDiaries() {
  const diaries = [];
  let offset = 0;
  while (offset < 500) {
    const upstream = await callOmbreDashboard("/diaries/search", {
      method: "POST",
      body: { keyword: "", limit: 100, offset },
    });
    if (!upstream.ok) throw new Error(`diary_live_${upstream.status}`);
    const page = Array.isArray(upstream.payload?.diaries) ? upstream.payload.diaries : [];
    diaries.push(...page);
    if (page.length < 100) break;
    offset += page.length;
  }
  const fingerprint = diaries.map((entry) => (
    `${entry.id}:${entry.updated_at || entry.created_at || ""}:${entry.revision || 0}`
  )).join("|");
  return {
    status: "ok",
    snapshotId: Buffer.from(fingerprint, "utf8").toString("base64url"),
    source: "Ombre DiaryStore live read-only projection",
    entries: diaries,
  };
}

function uniqueNarrativeSourceIds(items, key) {
  return Array.from(new Set(
    items.flatMap((item) => (Array.isArray(item?.[key]) ? item[key] : []))
      .map((value) => String(value ?? "").trim())
      .filter(Boolean),
  ));
}

function sourceDate(...values) {
  const value = values.find((candidate) => String(candidate || "").trim());
  return value ? String(value).slice(0, 10) : "";
}

export function buildNarrativeSourceLedgers(items, metadata = {}) {
  const eventById = new Map((metadata.events || []).map((item) => [String(item.item_id), item]));
  const sceneById = new Map((metadata.scenes || []).map((item) => [String(item.id), item]));
  const diaryById = new Map((metadata.diaries || []).map((item) => [String(item.id), item]));

  return items.map((item) => {
    const sourceLedger = [];
    for (const sourceId of item.linked_event_ids || []) {
      const id = String(sourceId);
      const source = eventById.get(id);
      sourceLedger.push({
        source_type: "event",
        source_id: id,
        title: String(source?.title || "未找到的 Event"),
        date: sourceDate(source?.local_date, source?.source_started_at, source?.created_at),
        status: String(source?.status || (source ? "active" : "missing")),
      });
    }
    for (const sourceId of item.linked_scene_ids || []) {
      const id = String(sourceId);
      const source = sceneById.get(id);
      sourceLedger.push({
        source_type: "scene",
        source_id: id,
        title: String(source?.name || source?.title || "未找到的 Scene"),
        date: sourceDate(source?.source_date, source?.created, source?.created_at, source?.updated_at),
        status: String(source?.status_view || source?.status || (source ? "active" : "missing")),
      });
    }
    for (const sourceId of item.linked_diary_ids || []) {
      const id = String(sourceId);
      const source = diaryById.get(id);
      sourceLedger.push({
        source_type: "diary",
        source_id: id,
        title: String(source?.title || `日记 ${id}`),
        date: sourceDate(source?.date, source?.created_at),
        status: String(source?.entry_type || source?.visibility || (source ? "active" : "missing")),
      });
    }
    for (const sourceId of item.linked_darkroom_ids || []) {
      const id = String(sourceId);
      const source = diaryById.get(id);
      sourceLedger.push({
        source_type: "darkroom",
        source_id: id,
        title: String(source?.title || `暗房 ${id}`),
        date: sourceDate(source?.date, source?.created_at),
        status: String(source?.entry_type || source?.visibility || (source ? "active" : "missing")),
      });
    }
    const uploadById = new Map((item.linked_uploads || []).map((upload) => [String(upload.upload_id), upload]));
    for (const sourceId of item.linked_upload_ids || []) {
      const id = String(sourceId);
      const source = uploadById.get(id);
      sourceLedger.push({
        source_type: "upload",
        source_id: id,
        title: String(source?.filename || id),
        date: sourceDate(source?.created_at),
        status: String(source?.extraction_status || (source ? "stored" : "missing")),
      });
    }
    return { ...item, source_ledger: sourceLedger };
  });
}

async function attachNarrativeSourceLedgers(items) {
  const eventIds = uniqueNarrativeSourceIds(items, "linked_event_ids");
  const sceneIds = uniqueNarrativeSourceIds(items, "linked_scene_ids");
  const diaryIds = new Set([
    ...uniqueNarrativeSourceIds(items, "linked_diary_ids"),
    ...uniqueNarrativeSourceIds(items, "linked_darkroom_ids"),
  ]);

  let events = [];
  if (eventIds.length) {
    const result = await callOmbreDashboard("/api/fact-events/read-many", {
      method: "POST",
      body: { item_ids: eventIds, include_sources: false, resolve_active_successors: false },
    });
    if (result.ok) events = Array.isArray(result.payload?.items) ? result.payload.items : [];
  }

  let scenes = [];
  if (sceneIds.length) {
    const result = await callOmbreDashboard("/api/buckets/light?include_archive=1&limit=2000");
    if (result.ok) scenes = Array.isArray(result.payload?.buckets) ? result.payload.buckets : [];
  }

  let diaries = [];
  if (diaryIds.size) {
    const result = await readLiveDiaries();
    diaries = result.entries.filter((entry) => diaryIds.has(String(entry.id)));
    const found = new Set(diaries.map((entry) => String(entry.id)));
    for (const diaryId of diaryIds) {
      if (found.has(diaryId)) continue;
      const exact = await callOmbreDashboard(`/diaries/${encodeURIComponent(diaryId)}`);
      if (exact.ok) diaries.push(exact.payload);
    }
  }

  return buildNarrativeSourceLedgers(items, { events, scenes, diaries });
}

export async function readLiveNarratives() {
  const index = await callOmbreDashboard("/api/narrative-rolls?limit=100");
  if (!index.ok) throw new Error(`narrative_live_projection_${index.status}`);
  const summaries = Array.isArray(index.payload?.items) ? index.payload.items : [];
  const items = [];
  for (const summary of summaries) {
    const narrativeId = String(summary?.narrative_id || "").trim();
    if (!narrativeId) continue;
    const result = await callOmbreDashboard(`/api/narrative-rolls?narrative_id=${encodeURIComponent(narrativeId)}`);
    if (!result.ok) throw new Error(`narrative_live_projection_${result.status}`);
    items.push(result.payload);
  }
  const enrichedItems = await attachNarrativeSourceLedgers(items);
  const fingerprint = enrichedItems.map((item) => (
    `${item?.narrative_id || ""}:${item?.revision || ""}:${item?.document_sha256 || ""}:${JSON.stringify(item.source_ledger)}`
  )).join("|");
  return {
    status: "ok",
    snapshotId: createHash("sha256").update(fingerprint).digest("hex"),
    source: "Ombre Narrative Roll registry live read-only projection",
    items: enrichedItems,
  };
}

async function readLiveWindowShadows() {
  const upstream = await callOmbreDashboard("/api/window-shadows?limit=100&include_content=1");
  if (!upstream.ok) throw new Error(`window_shadow_live_projection_${upstream.status}`);
  const windows = Array.isArray(upstream.payload?.windows) ? upstream.payload.windows : [];
  const shadows = windows.map((item) => {
    const text = String(item?.content || "").replace(/\r\n?/g, "\n").trim();
    const created = new Date(item?.created_at || 0);
    const createdDate = Number.isNaN(created.getTime())
      ? String(item?.source_date || "").slice(0, 10)
      : new Intl.DateTimeFormat("en-CA", { timeZone: "Asia/Shanghai" }).format(created);
    const sourceDate = String(item?.source_date || createdDate).slice(0, 10);
    const historical = sourceDate !== createdDate;
    const dateLabel = sourceDate.replaceAll("-", ".");
    const timeLabel = historical || Number.isNaN(created.getTime())
      ? "历史补录"
      : new Intl.DateTimeFormat("zh-CN", {
        timeZone: "Asia/Shanghai", hour: "2-digit", minute: "2-digit", hour12: false,
      }).format(created);
    const headings = Array.from(text.matchAll(/^\s{0,3}#{1,6}\s+(.+?)\s*$/gm))
      .map((match) => match[1].replace(/[*_`]/g, "").trim())
      .filter((heading) => !["window shadow", "窗影"].includes(heading.toLowerCase()));
    const summary = text.split(/\n\s*\n/).map((block) => block.split("\n").map((line) => line.trim()).filter(Boolean))
      .find((lines) => lines.length && !lines.every((line) => line.startsWith("#") || line.startsWith(">")))
      ?.join(" ").replace(/^\s{0,3}#{1,6}\s+|^\s*[-*+]\s+/g, "").replace(/[*_`>#]+/g, "").replace(/\s+/g, " ").trim() || "";
    const contentHash = createHash("sha256").update(text).digest("hex");
    return {
      id: String(item?.window_id || ""),
      closedAt: historical ? `${sourceDate}T00:00:00+08:00` : String(item?.created_at || ""),
      dateLabel,
      timeLabel,
      relativeLabel: `${dateLabel} · ${timeLabel}`,
      title: headings[0] || `${dateLabel} 的窗影`,
      summary: summary.length <= 96 ? summary : `${summary.slice(0, 95)}…`,
      text,
      scenes: [],
      sourceLabel: "Ombre v1 · 杭州",
      statusLabel: "已入库窗影",
      sourceKind: "ombre-window-shadow",
      documentOwnsTitle: true,
      sourceId: String(item?.window_id || ""),
      sourceSessionId: String(item?.session_id || ""),
      contentHash,
    };
  });
  const fingerprint = shadows.map((item) => `${item.sourceId}:${item.contentHash}`).join("\n");
  return {
    status: "ok",
    snapshotId: createHash("sha256").update(fingerprint).digest("hex"),
    source: "Ombre Window Shadow live read-only projection",
    shadows,
  };
}

export async function readHavenBridgeHookLedger(limit = 80, beforeId = 0, reviewIds = []) {
  const sshTarget = String(process.env.HAVEN_BRIDGE_VPS_SSH_TARGET || "root@168.119.228.217").trim();
  const identityFile = String(
    process.env.HAVEN_BRIDGE_VPS_IDENTITY_FILE || "C:\\Users\\86188\\.ssh\\id_ed25519",
  ).trim();
  const localDatabase = String(process.env.HAVEN_BRIDGE_LOCAL_DB || "").trim();
  const safeLimit = Math.max(1, Math.min(200, Number.parseInt(limit, 10) || 80));
  const safeBeforeId = Math.max(0, Number.parseInt(beforeId, 10) || 0);
  const safeReviewIds = [...new Set((Array.isArray(reviewIds) ? reviewIds : [])
    .map((item) => Number.parseInt(item, 10))
    .filter((item) => Number.isInteger(item) && item > 0))].slice(0, 500);
  const script = `
import json, sqlite3

conn = sqlite3.connect("/opt/haven_bridge/data/haven.db")
conn.row_factory = sqlite3.Row
where = [
    "role='user'",
    "CASE WHEN json_valid(metadata_json) THEN json_type(metadata_json, '$.hook_memory_outcome') ELSE NULL END IS NOT NULL",
]
params = []
before_id = ${safeBeforeId}
if before_id > 0:
    where.append("id < ?")
    params.append(before_id)
params.append(${safeLimit + 1})
rows = conn.execute(
    """
    SELECT id, session_id, created_at, content, metadata_json
    FROM messages
    WHERE """ + " AND ".join(where) + """
    ORDER BY id DESC
    LIMIT ?
    """,
    params,
).fetchall()
has_more = len(rows) > ${safeLimit}
page_rows = rows[:${safeLimit}]
page_ids = {int(row["id"]) for row in page_rows}
review_ids = ${JSON.stringify(safeReviewIds)}
review_rows = []
if review_ids:
    review_rows = conn.execute(
        """
        SELECT id, session_id, created_at, content, metadata_json
        FROM messages
        WHERE role='user'
          AND CASE WHEN json_valid(metadata_json) THEN json_type(metadata_json, '$.hook_memory_outcome') ELSE NULL END IS NOT NULL
          AND id IN (""" + ",".join("?" for _ in review_ids) + ") ORDER BY id DESC",
        review_ids,
    ).fetchall()

def compact_row(row):
    try:
        metadata = json.loads(row["metadata_json"] or "{}")
    except Exception:
        metadata = {}
    raw_ids = metadata.get("gateway_memory_injected_ids")
    injected_ids = [str(item).strip() for item in raw_ids] if isinstance(raw_ids, list) else []
    injected_ids = list(dict.fromkeys(item for item in injected_ids if item))
    raw_items = metadata.get("gateway_memory_items")
    memory_items = []
    for item in raw_items if isinstance(raw_items, list) else []:
        if not isinstance(item, dict):
            continue
        compact = {}
        for key in ("id", "title", "domain", "date", "moment_id", "source_kind"):
            value = str(item.get(key) or "").strip()
            if value:
                compact[key] = value[:200]
        score = item.get("score")
        if isinstance(score, (int, float)):
            compact["score"] = float(score)
        if compact:
            memory_items.append(compact)
    return {
        "id": int(row["id"]),
        "session_id": int(row["session_id"]) if row["session_id"] is not None else None,
        "created_at": str(row["created_at"] or ""),
        "query": str(row["content"] or "")[:500],
        "hook_memory_outcome": str(metadata.get("hook_memory_outcome") or ""),
        "hook_memory_sources": metadata.get("hook_memory_sources") if isinstance(metadata.get("hook_memory_sources"), list) else [],
        "gateway_memory_trigger": str(metadata.get("gateway_memory_trigger") or ""),
        "gateway_memory_route": str(metadata.get("gateway_memory_route") or ""),
        "gateway_memory_injected_ids": injected_ids,
        "gateway_memory_items": memory_items,
    }

items = [compact_row(row) for row in page_rows]
reviewed_items = [compact_row(row) for row in review_rows if int(row["id"]) not in page_ids]
next_before_id = int(page_rows[-1]["id"]) if page_rows else None
conn.close()
print(json.dumps({
    "status": "ok",
    "items": items,
    "reviewed_items": reviewed_items,
    "has_more": has_more,
    "next_before_id": next_before_id,
    "next_cursor": str(next_before_id) if next_before_id else None,
}, ensure_ascii=False))
`;

  return new Promise((resolve, reject) => {
    const command = localDatabase
      ? String(process.env.SEREIN_PYTHON_BIN || "python3").trim()
      : "ssh";
    const args = localDatabase
      ? ["-"]
      : ["-i", identityFile, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", sshTarget, "python3", "-"];
    const localScript = localDatabase
      ? script.replace('/opt/haven_bridge/data/haven.db', localDatabase.replaceAll('\\', '\\\\').replaceAll('"', '\\"'))
      : script;
    const child = spawn(
      command,
      args,
      { stdio: ["pipe", "pipe", "pipe"], windowsHide: true },
    );
    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => child.kill(), 25_000);
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", (error) => { clearTimeout(timer); reject(error); });
    child.on("close", (code) => {
      clearTimeout(timer);
      if (code !== 0) {
        reject(new Error(stderr.trim() || `haven_bridge_ssh_${code}`));
        return;
      }
      try {
        resolve(JSON.parse(stdout.trim()));
      } catch {
        reject(new Error("haven_bridge_hook_ledger_invalid"));
      }
    });
    child.stdin.end(localScript);
  });
}

async function readHavenBridgeEvidenceMessages({
  limit = 40,
  beforeId = 0,
  messageIds = [],
  query = "",
  contextMessageId = 0,
  contextRadius = 6,
} = {}) {
  const sshTarget = String(process.env.HAVEN_BRIDGE_VPS_SSH_TARGET || "root@168.119.228.217").trim();
  const identityFile = String(
    process.env.HAVEN_BRIDGE_VPS_IDENTITY_FILE || "C:\\Users\\86188\\.ssh\\id_ed25519",
  ).trim();
  const localDatabase = String(
    process.env.HAVEN_BRIDGE_LOCAL_DB
      || (existsSync("/opt/haven_bridge/data/haven.db") ? "/opt/haven_bridge/data/haven.db" : ""),
  ).trim();
  const safeLimit = Math.max(1, Math.min(80, Number.parseInt(limit, 10) || 40));
  const safeBeforeId = Math.max(0, Number.parseInt(beforeId, 10) || 0);
  const safeQuery = normalizeEvidenceSearchQuery(query);
  const safeContextMessageId = normalizeEvidenceMessageId(contextMessageId);
  const safeContextRadius = Math.max(1, Math.min(12, Number.parseInt(contextRadius, 10) || 6));
  const safeMessageIds = Array.isArray(messageIds)
    ? [...new Set(messageIds.map((value) => Number.parseInt(value, 10)).filter((value) => value > 0))].slice(0, 12)
    : [];
  const requestPayload = Buffer.from(JSON.stringify({
    limit: safeLimit,
    before_id: safeBeforeId,
    message_ids: safeMessageIds,
    query: safeQuery,
    context_message_id: safeContextMessageId,
    context_radius: safeContextRadius,
  }), "utf8").toString("base64");
  const script = `
import base64, json, sqlite3

request = json.loads(base64.b64decode("${requestPayload}").decode("utf-8"))
conn = sqlite3.connect("/opt/haven_bridge/data/haven.db")
conn.row_factory = sqlite3.Row
where = [
    "m.role IN ('user', 'assistant')",
    "COALESCE(json_extract(m.metadata_json, '$.draft'), 0)=0",
    "COALESCE(json_extract(m.metadata_json, '$.discarded'), 0)=0",
    "((m.role='user' AND m.source IN ('chat','codex_direct') AND (m.source!='chat' OR COALESCE(json_extract(m.metadata_json, '$.delivery_status'), '')='done')) OR (m.role='assistant' AND m.source IN ('codex','codex_direct') AND COALESCE(json_extract(m.metadata_json, '$.autonomy'), 0)=0 AND COALESCE(json_extract(m.metadata_json, '$.proactive'), 0)=0))",
]
params = []
message_ids = [int(item) for item in request.get("message_ids") or [] if int(item) > 0]
query = str(request.get("query") or "").strip()
exact_id = query[1:] if query.startswith("#") else ""
context_message_id = int(request.get("context_message_id") or 0)
if not context_message_id and exact_id.isdigit():
    context_message_id = int(exact_id)
context_radius = max(1, min(12, int(request.get("context_radius") or 6)))
select_sql = "SELECT m.id, m.session_id, m.role, m.source, m.created_at, m.content, s.external_thread_id AS thread_id FROM messages m LEFT JOIN sessions s ON s.id=m.session_id"
base_where = " AND ".join(where)

mode = "browse"
target_message_id = None
has_more = False
next_before_id = None

if context_message_id > 0:
    mode = "context"
    target = conn.execute(
        select_sql + " WHERE " + base_where + " AND m.id=?",
        (context_message_id,),
    ).fetchone()
    rows = []
    if target is not None:
        target_message_id = int(target["id"])
        before = conn.execute(
            select_sql + " WHERE " + base_where + " AND m.session_id=? AND m.id<? ORDER BY m.id DESC LIMIT ?",
            (target["session_id"], target_message_id, context_radius),
        ).fetchall()
        after = conn.execute(
            select_sql + " WHERE " + base_where + " AND m.session_id=? AND m.id>? ORDER BY m.id ASC LIMIT ?",
            (target["session_id"], target_message_id, context_radius),
        ).fetchall()
        rows = list(reversed(before)) + [target] + list(after)
elif message_ids:
    mode = "selected"
    where.append("m.id IN (" + ",".join("?" for _ in message_ids) + ")")
    params.extend(message_ids)
    params.append(len(message_ids) + 1)
    rows = conn.execute(
        select_sql + " WHERE " + " AND ".join(where) + " ORDER BY m.id DESC LIMIT ?",
        params,
    ).fetchall()
else:
    if query:
        mode = "search"
        where.append("instr(m.content, ?) > 0")
        params.append(query)
    before_id = int(request.get("before_id") or 0)
    if before_id > 0:
        where.append("m.id < ?")
        params.append(before_id)
    params.append(int(request.get("limit") or 40) + 1)
    rows = conn.execute(
        select_sql + " WHERE " + " AND ".join(where) + " ORDER BY m.id DESC LIMIT ?",
        params,
    ).fetchall()
    has_more = len(rows) > int(request.get("limit") or 40)
    if has_more:
        rows = rows[:-1]
    next_before_id = min((int(row["id"]) for row in rows), default=0) or None

items = []
for row in rows:
    item = dict(row)
    item["is_context_target"] = int(row["id"]) == target_message_id
    items.append(item)
conn.close()
print(json.dumps({
    "status": "ok",
    "items": items,
    "has_more": has_more,
    "next_before_id": next_before_id,
    "query": str(request.get("query") or ""),
    "mode": mode,
    "target_message_id": target_message_id,
    "context_radius": context_radius if mode == "context" else None,
}, ensure_ascii=False))
`;

  return new Promise((resolve, reject) => {
    const command = localDatabase ? "python3" : "ssh";
    const args = localDatabase
      ? ["-"]
      : ["-i", identityFile, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", sshTarget, "python3", "-"];
    const localScript = localDatabase
      ? script.replace('/opt/haven_bridge/data/haven.db', localDatabase.replaceAll('\\', '\\\\').replaceAll('"', '\\"'))
      : script;
    const child = spawn(command, args, { stdio: ["pipe", "pipe", "pipe"], windowsHide: true });
    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => child.kill(), 25_000);
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", (error) => { clearTimeout(timer); reject(error); });
    child.on("close", (code) => {
      clearTimeout(timer);
      if (code !== 0) {
        reject(new Error(stderr.trim() || `haven_bridge_evidence_${code}`));
        return;
      }
      try {
        resolve(JSON.parse(stdout.trim()));
      } catch {
        reject(new Error("haven_bridge_evidence_invalid"));
      }
    });
    child.stdin.end(localScript);
  });
}

function sereinGatewayBridge() {
  return {
    name: "serein-gateway-bridge",
    configureServer(server) {
      server.middlewares.use("/__serein/haven-bridge/hook-injections", async (request, response) => {
        response.setHeader("Content-Type", "application/json; charset=utf-8");
        if (request.method !== "POST") {
          response.statusCode = 405;
          response.end(JSON.stringify({ error: "method_not_allowed" }));
          return;
        }
        try {
          const body = await readJsonBody(request);
          response.statusCode = 200;
          response.end(JSON.stringify(await readHavenBridgeHookLedger(
            body.limit,
            body.beforeId,
            body.reviewIds,
          )));
        } catch (error) {
          response.statusCode = 502;
          response.end(JSON.stringify({
            status: "error",
            error: "haven_bridge_hook_ledger_failed",
            message: "本地桥没有读到 Haven Bridge hook 账本。",
            items: [],
          }));
        }
      });

      server.middlewares.use("/__serein/gateway/injections", async (request, response) => {
        response.setHeader("Content-Type", "application/json; charset=utf-8");
        if (request.method !== "POST") {
          response.statusCode = 405;
          response.end(JSON.stringify({ error: "method_not_allowed" }));
          return;
        }
        try {
          const body = await readJsonBody(request);
          const limit = Math.max(1, Math.min(100, Number.parseInt(body.limit, 10) || 50));
          const beforeId = Math.max(0, Number.parseInt(body.beforeId, 10) || 0);
          const reviewIds = [...new Set((Array.isArray(body.reviewIds) ? body.reviewIds : [])
            .map((item) => Number.parseInt(item, 10))
            .filter((item) => Number.isInteger(item) && item > 0))].slice(0, 500);
          const params = new URLSearchParams({ limit: String(limit), include_context: "0" });
          if (beforeId) params.set("before_id", String(beforeId));
          if (reviewIds.length) params.set("review_ids", reviewIds.join(","));
          const upstream = await callOmbreDashboard(`/api/gateway-injections?${params.toString()}`);
          response.statusCode = upstream.status;
          response.end(JSON.stringify(upstream.payload));
        } catch (error) {
          response.statusCode = error?.name === "AbortError" ? 504 : 502;
          response.end(JSON.stringify({
            status: "error",
            error: "gateway_observation_failed",
            message: error?.name === "AbortError" ? "最近注入读取超时。" : "本地桥没有读到最近注入。",
            items: [],
          }));
        }
      });

      server.middlewares.use("/__serein/gateway/recall", async (request, response) => {
        response.setHeader("Content-Type", "application/json; charset=utf-8");
        if (request.method !== "POST") {
          response.statusCode = 405;
          response.end(JSON.stringify({ error: "method_not_allowed" }));
          return;
        }

        const gatewayToken = String(process.env.OMBRE_GATEWAY_TOKEN || "").trim();
        const gatewayBase = String(process.env.OMBRE_GATEWAY_URL || "http://8.136.154.242:18002").replace(/\/$/, "");
        if (!gatewayToken) {
          response.statusCode = 503;
          response.end(JSON.stringify({
            error: "gateway_bridge_not_configured",
            message: "本地预览没有安全载入 Gateway 凭据。",
          }));
          return;
        }

        try {
          const body = await readJsonBody(request);
          const query = String(body.query || "").trim();
          if (!query) {
            response.statusCode = 400;
            response.end(JSON.stringify({ error: "query_required", message: "先写一句要测试的话。" }));
            return;
          }
          const simulationOptions = normalizeRecallSimulationOptions(body);
          if (!simulationOptions.ok) {
            response.statusCode = 400;
            response.end(JSON.stringify({
              error: simulationOptions.error,
              message: simulationOptions.message,
            }));
            return;
          }
          const { recallAblation, simulation, simulationScope } = simulationOptions;

          const controller = new AbortController();
          const timer = setTimeout(() => controller.abort(), 30_000);
          const upstream = await fetch(`${gatewayBase}/api/hook/recall`, {
            method: "POST",
            headers: {
              Authorization: `Bearer ${gatewayToken}`,
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              query,
              session_id: `serein-basement-${randomUUID()}`,
              recall_mode: "full",
              include_debug: true,
              simulation,
              simulation_scope: simulationScope,
              recall_ablation: recallAblation,
              include_context: false,
              include_recent_context: false,
              max_cards: 5,
              max_chars: 1200,
            }),
            signal: controller.signal,
          });
          clearTimeout(timer);
          response.statusCode = upstream.status;
          response.end(await upstream.text());
        } catch (error) {
          response.statusCode = error?.name === "AbortError" ? 504 : 502;
          response.end(JSON.stringify({
            error: "gateway_bridge_failed",
            message: error?.name === "AbortError" ? "Gateway 响应超时。" : "本地桥没有完成这次模拟。",
          }));
        }
      });

      server.middlewares.use("/__serein/gateway/semantic-route-draft", async (request, response) => {
        response.setHeader("Content-Type", "application/json; charset=utf-8");
        if (!["GET", "PUT", "DELETE"].includes(request.method)) {
          response.statusCode = 405;
          response.end(JSON.stringify({ error: "method_not_allowed" }));
          return;
        }
        try {
          if (request.method === "GET") {
            const draft = readServerSemanticRouteDraft();
            response.statusCode = 200;
            response.end(JSON.stringify({ status: draft ? "ok" : "empty", draft }));
            return;
          }
          if (request.method === "DELETE") {
            clearServerSemanticRouteDraft();
            response.statusCode = 200;
            response.end(JSON.stringify({ status: "cleared" }));
            return;
          }
          const body = await readJsonBody(request, 256_000);
          const draft = saveServerSemanticRouteDraft(body);
          response.statusCode = 200;
          response.end(JSON.stringify({ status: "saved", draft }));
        } catch (error) {
          const conflict = error?.message === "route_draft_revision_conflict";
          response.statusCode = conflict ? 409 : error?.message === "request_too_large" ? 413 : 400;
          response.end(JSON.stringify({
            error: error?.message || "route_draft_failed",
            ...(conflict ? { draft: error.current || null } : {}),
          }));
        }
      });

      server.middlewares.use("/__serein/gateway/semantic-routes", async (request, response) => {
        response.setHeader("Content-Type", "application/json; charset=utf-8");
        if (!["GET", "POST"].includes(request.method)) {
          response.statusCode = 405;
          response.end(JSON.stringify({ error: "method_not_allowed" }));
          return;
        }

        const gatewayToken = String(process.env.OMBRE_GATEWAY_TOKEN || "").trim();
        const gatewayBase = String(process.env.OMBRE_GATEWAY_URL || "http://8.136.154.242:18002").replace(/\/$/, "");
        if (!gatewayToken) {
          response.statusCode = 503;
          response.end(JSON.stringify({
            error: "gateway_bridge_not_configured",
            message: "本地预览没有安全载入 Gateway 凭据。",
          }));
          return;
        }

        try {
          const publishing = request.method === "POST";
          const body = publishing ? await readJsonBody(request, 128_000) : undefined;
          const controller = new AbortController();
          const timer = setTimeout(() => controller.abort(), publishing ? 120_000 : 20_000);
          const upstream = await fetch(
            `${gatewayBase}/api/semantic-recall/routes${publishing ? "/publish" : ""}`,
            {
              method: request.method,
              headers: {
                Authorization: `Bearer ${gatewayToken}`,
                ...(publishing ? { "Content-Type": "application/json" } : {}),
              },
              body: publishing ? JSON.stringify(body) : undefined,
              signal: controller.signal,
            },
          );
          clearTimeout(timer);
          response.statusCode = upstream.status;
          response.end(await upstream.text());
        } catch (error) {
          response.statusCode = error?.name === "AbortError" ? 504 : 502;
          response.end(JSON.stringify({
            error: "semantic_route_bridge_failed",
            message: error?.name === "AbortError" ? "Router 数据集操作超时。" : "本地桥没有完成 Router 数据集操作。",
          }));
        }
      });

      server.middlewares.use("/__serein/gateway/domain-policies", async (request, response) => {
        response.setHeader("Content-Type", "application/json; charset=utf-8");
        if (!["GET", "POST"].includes(request.method)) {
          response.statusCode = 405;
          response.end(JSON.stringify({ error: "method_not_allowed" }));
          return;
        }

        const gatewayToken = String(process.env.OMBRE_GATEWAY_TOKEN || "").trim();
        const gatewayBase = String(process.env.OMBRE_GATEWAY_URL || "http://8.136.154.242:18002").replace(/\/$/, "");
        if (!gatewayToken) {
          response.statusCode = 503;
          response.end(JSON.stringify({
            error: "gateway_bridge_not_configured",
            message: "本地预览没有安全载入 Gateway 凭据。",
          }));
          return;
        }

        try {
          const publishing = request.method === "POST";
          const body = publishing ? await readJsonBody(request, 32_000) : undefined;
          const controller = new AbortController();
          const timer = setTimeout(() => controller.abort(), 20_000);
          const upstream = await fetch(
            `${gatewayBase}/api/semantic-recall/domain-policies${publishing ? "/publish" : ""}`,
            {
              method: request.method,
              headers: {
                Authorization: `Bearer ${gatewayToken}`,
                ...(publishing ? { "Content-Type": "application/json" } : {}),
              },
              body: publishing ? JSON.stringify(body) : undefined,
              signal: controller.signal,
            },
          );
          clearTimeout(timer);
          response.statusCode = upstream.status;
          response.end(await upstream.text());
        } catch (error) {
          response.statusCode = error?.name === "AbortError" ? 504 : 502;
          response.end(JSON.stringify({
            error: "domain_policy_bridge_failed",
            message: error?.name === "AbortError" ? "主域策略操作超时。" : "本地桥没有完成主域策略操作。",
          }));
        }
      });
    },
  };
}

function sereinMemoryBridge() {
  return {
    name: "serein-memory-bridge",
    configureServer(server) {
      server.middlewares.use("/__serein/live/memory-scenes", async (request, response) => {
        response.setHeader("Content-Type", "application/json; charset=utf-8");
        if (request.method !== "POST") {
          response.statusCode = 405;
          response.end(JSON.stringify({ error: "method_not_allowed" }));
          return;
        }
        try {
          const body = await readJsonBody(request, 96_000);
          response.statusCode = 200;
          response.end(JSON.stringify(await readLiveMemoryProjection(body.sourceIds)));
        } catch (error) {
          console.error("[serein-memory-bridge] live Scene projection failed", error);
          response.statusCode = error?.name === "AbortError" ? 504 : 502;
          response.end(JSON.stringify({
            status: "error",
            error: "memory_live_projection_failed",
            message: "没有读到线上 Scene，已保留本地快照回退。",
            scenes: [],
            edges: [],
          }));
        }
      });

      server.middlewares.use("/__serein/live/fact-events", async (request, response) => {
        response.setHeader("Content-Type", "application/json; charset=utf-8");
        if (request.method !== "POST") {
          response.statusCode = 405;
          response.end(JSON.stringify({ error: "method_not_allowed" }));
          return;
        }
        try {
          const body = await readJsonBody(request);
          const type = body.type === "fact" ? "fact" : "event";
          const status = ["active", "archived", "superseded", "all"].includes(body.status)
            ? body.status
            : "all";
          const params = new URLSearchParams({
            type,
            status,
            include_sources: body.includeSources === true ? "1" : "0",
            limit: String(Math.max(1, Math.min(500, Number.parseInt(body.limit, 10) || 500))),
            offset: String(Math.max(0, Number.parseInt(body.offset, 10) || 0)),
          });
          const query = String(body.query || "").trim();
          if (query) params.set("query", query);
          const upstream = await callOmbreDashboard(`/api/fact-events?${params.toString()}`);
          response.statusCode = upstream.status;
          response.end(JSON.stringify(upstream.payload));
        } catch (error) {
          response.statusCode = error?.name === "AbortError" ? 504 : 502;
          response.end(JSON.stringify({
            error: "fact_events_read_failed",
            message: error?.name === "AbortError" ? "读取事实和事件超时。" : "暂时没有读到事实和事件。",
            items: [],
          }));
        }
      });

      server.middlewares.use("/__serein/memory/revise-fact-event", async (request, response) => {
        response.setHeader("Content-Type", "application/json; charset=utf-8");
        if (request.method !== "POST") {
          response.statusCode = 405;
          response.end(JSON.stringify({ error: "method_not_allowed" }));
          return;
        }
        try {
          const body = await readJsonBody(request);
          const upstream = await callOmbreDashboard("/api/fact-events/revise", {
            method: "POST",
            body: {
              item_id: String(body.itemId || "").trim(),
              title: body.title,
              body: body.body,
              ...(Object.prototype.hasOwnProperty.call(body, "recallable") ? { recallable: body.recallable } : {}),
            },
          });
          response.statusCode = upstream.status;
          response.end(JSON.stringify(upstream.payload));
        } catch (error) {
          response.statusCode = error?.name === "AbortError" ? 504 : 502;
          response.end(JSON.stringify({
            error: "fact_event_revision_failed",
            message: error?.name === "AbortError" ? "保存修订超时。" : "没有完成这次修订。",
          }));
        }
      });

      server.middlewares.use("/__serein/memory/set-fact-event-status", async (request, response) => {
        response.setHeader("Content-Type", "application/json; charset=utf-8");
        if (request.method !== "POST") {
          response.statusCode = 405;
          response.end(JSON.stringify({ error: "method_not_allowed" }));
          return;
        }
        try {
          const body = await readJsonBody(request);
          const upstream = await callOmbreDashboard("/api/fact-events/status", {
            method: "POST",
            body: {
              item_id: String(body.itemId || "").trim(),
              status: String(body.status || "").trim(),
            },
          });
          response.statusCode = upstream.status;
          response.end(JSON.stringify(upstream.payload));
        } catch (error) {
          response.statusCode = error?.name === "AbortError" ? 504 : 502;
          response.end(JSON.stringify({
            error: "fact_event_status_failed",
            message: error?.name === "AbortError" ? "更新状态超时。" : "没有完成这次状态修改。",
          }));
        }
      });

      server.middlewares.use("/__serein/memory/delete-fact-event", async (request, response) => {
        response.setHeader("Content-Type", "application/json; charset=utf-8");
        if (request.method !== "POST") {
          response.statusCode = 405;
          response.end(JSON.stringify({ error: "method_not_allowed" }));
          return;
        }
        try {
          const body = await readJsonBody(request);
          const upstream = await callOmbreDashboard("/api/fact-events/delete", {
            method: "POST",
            body: { item_id: String(body.itemId || "").trim() },
          });
          response.statusCode = upstream.status;
          response.end(JSON.stringify(upstream.payload));
        } catch (error) {
          response.statusCode = error?.name === "AbortError" ? 504 : 502;
          response.end(JSON.stringify({
            error: "fact_event_deletion_failed",
            message: error?.name === "AbortError" ? "永久删除超时。" : "没有完成永久删除。",
          }));
        }
      });

      server.middlewares.use("/__serein/live/diaries", async (request, response) => {
        response.setHeader("Content-Type", "application/json; charset=utf-8");
        const route = new URL(request.url || "/", "http://serein.local");
        const diaryId = route.pathname.match(/^\/(\d+)\/?$/u)?.[1];

        if (
          (request.method === "POST" && /^\/entry\/?$/u.test(route.pathname))
          || (request.method === "PUT" && diaryId)
        ) {
          try {
            const body = await readJsonBody(request, 256_000);
            const upstream = await callOmbreDashboard(
              diaryId ? `/diaries/${diaryId}` : "/diaries",
              { method: diaryId ? "PUT" : "POST", body },
            );
            response.statusCode = upstream.status;
            response.end(JSON.stringify(upstream.payload));
          } catch (error) {
            response.statusCode = error?.name === "AbortError" ? 504 : 502;
            response.end(JSON.stringify({
              error: "diary_save_failed",
              message: "这篇日记没有保存，请稍后再试。",
            }));
          }
          return;
        }

        if (request.method === "DELETE" && diaryId) {
          try {
            const upstream = await callOmbreDashboard(`/diaries/${diaryId}`, { method: "DELETE" });
            response.statusCode = upstream.status;
            response.end(JSON.stringify(upstream.payload));
          } catch (error) {
            response.statusCode = error?.name === "AbortError" ? 504 : 502;
            response.end(JSON.stringify({
              error: "diary_delete_failed",
              message: "这篇日记没有删掉，请稍后再试。",
            }));
          }
          return;
        }

        if (request.method !== "POST" || !/^\/?$/u.test(route.pathname)) {
          response.statusCode = 405;
          response.end(JSON.stringify({ error: "method_not_allowed" }));
          return;
        }
        try {
          response.statusCode = 200;
          response.end(JSON.stringify(await readLiveDiaries()));
        } catch (error) {
          response.statusCode = error?.name === "AbortError" ? 504 : 502;
          response.end(JSON.stringify({
            status: "error",
            error: "diary_live_projection_failed",
            message: "没有读到线上日记，已保留本地快照回退。",
            entries: [],
          }));
        }
      });

      server.middlewares.use("/__serein/live/narratives", async (request, response) => {
        response.setHeader("Content-Type", "application/json; charset=utf-8");
        if (request.method !== "POST") {
          response.statusCode = 405;
          response.end(JSON.stringify({ error: "method_not_allowed" }));
          return;
        }
        try {
          response.statusCode = 200;
          response.end(JSON.stringify(await readLiveNarratives()));
        } catch (error) {
          response.statusCode = error?.name === "AbortError" ? 504 : 502;
          response.end(JSON.stringify({
            status: "error",
            error: "narrative_live_projection_failed",
            message: "没有读到线上叙事卷，已保留本地样卷回退。",
            items: [],
          }));
        }
      });

      server.middlewares.use("/__serein/narrative-preview", async (request, response) => {
        response.setHeader("Content-Type", "application/json; charset=utf-8");
        if (request.method !== "POST") {
          response.statusCode = 405;
          response.end(JSON.stringify({ error: "method_not_allowed" }));
          return;
        }
        try {
          const body = await readJsonBody(request, 128_000);
          const narrativeId = String(body.narrativeId || "").trim();
          const mode = ["edit", "update", "rewrite"].includes(body.mode) ? body.mode : "";
          if (!/^[A-Za-z0-9_.:#-]{1,160}$/.test(narrativeId) || !mode) {
            response.statusCode = 400;
            response.end(JSON.stringify({ status: "invalid", reason: "invalid_preview_request", writes_performed: [] }));
            return;
          }
          const upstream = await callOmbreDashboard("/api/narrative-rolls/preview-input", {
            method: "POST",
            body: {
              narrative_id: narrativeId,
              mode,
              expected_revision: Number.parseInt(body.expectedRevision, 10) || undefined,
              expected_document_sha256: String(body.expectedDocumentSha256 || "").trim(),
              proposed_material_ids: body.proposedMaterialIds,
            },
          });
          if (!upstream.ok) {
            response.statusCode = upstream.status;
            response.end(JSON.stringify(upstream.payload));
            return;
          }
          const input = upstream.payload;
          const preview = mode === "edit"
            ? {
                status: "ok",
                evidence_sufficient: true,
                body: String(body.proposedBody || "").trim(),
                issues: [],
                diff: narrativeBodyDiff(input.current_body, String(body.proposedBody || "").trim()),
                mode,
                provider: "host_validation_only",
                publication_status: "not_published",
                writes_performed: [],
              }
            : await runNarrativeCodexTask({
                mode,
                title: input.title,
                currentBody: input.current_body,
                materials: input.materials,
                roleDir: narrativeWriterRoleDir,
              });
          if (!preview.body) {
            response.statusCode = mode === "edit" ? 400 : 200;
            response.end(JSON.stringify({
              ...preview,
              narrative_id: narrativeId,
              base_revision: input.base_revision,
              base_document_sha256: input.base_document_sha256,
              material_counts: input.material_counts,
              current_material_ids: input.current_material_ids,
              proposed_material_ids: input.proposed_material_ids,
              material_delta: input.material_delta,
              material_snapshot_sha256: input.material_snapshot_sha256,
              writes_performed: [],
            }));
            return;
          }
          const fingerprint = buildNarrativePreviewFingerprint({
            narrativeId,
            revision: input.base_revision,
            documentSha256: input.base_document_sha256,
            body: preview.body,
            materialSnapshotSha256: input.material_snapshot_sha256,
          });
          response.statusCode = 200;
          response.end(JSON.stringify({
            ...preview,
            narrative_id: narrativeId,
            base_revision: input.base_revision,
            base_document_sha256: input.base_document_sha256,
            material_counts: input.material_counts,
            current_material_ids: input.current_material_ids,
            proposed_material_ids: input.proposed_material_ids,
            material_delta: input.material_delta,
            material_snapshot_sha256: input.material_snapshot_sha256,
            preview_fingerprint: fingerprint,
          }));
        } catch (error) {
          console.error("[serein-memory-bridge] Narrative preview failed", error);
          response.statusCode = error?.name === "AbortError" ? 504 : 502;
          response.end(JSON.stringify({
            status: "error",
            reason: "narrative_preview_failed",
            message: error?.name === "AbortError" ? "这次预览生成超时了。" : "暂时没有生成叙事卷预览。",
            writes_performed: [],
          }));
        }
      });

      server.middlewares.use("/__serein/narrative-material-upload", async (request, response) => {
        response.setHeader("Content-Type", "application/json; charset=utf-8");
        if (request.method !== "POST") {
          response.statusCode = 405;
          response.end(JSON.stringify({ error: "method_not_allowed" }));
          return;
        }
        try {
          const body = await readJsonBody(request, 14_100_000);
          const filename = String(body.filename || "").trim();
          const contentType = String(body.contentType || "application/octet-stream").trim();
          const contentBase64 = String(body.contentBase64 || "");
          if (!filename || filename.length > 240 || !contentBase64) {
            response.statusCode = 400;
            response.end(JSON.stringify({ status: "invalid", reason: "invalid_upload_request", writes_performed: [] }));
            return;
          }
          const upstream = await callOmbreDashboard("/api/narrative-rolls/material-uploads", {
            method: "POST",
            body: {
              filename,
              content_type: contentType,
              content_base64: contentBase64,
            },
          });
          response.statusCode = upstream.status;
          response.end(JSON.stringify(upstream.payload));
        } catch (error) {
          response.statusCode = error?.message === "request_too_large" ? 413 : 502;
          response.end(JSON.stringify({
            status: "error",
            reason: error?.message === "request_too_large" ? "upload_too_large" : "narrative_material_upload_failed",
            writes_performed: [],
          }));
        }
      });

      server.middlewares.use("/__serein/narrative-save", async (request, response) => {
        response.setHeader("Content-Type", "application/json; charset=utf-8");
        if (request.method !== "POST") {
          response.statusCode = 405;
          response.end(JSON.stringify({ error: "method_not_allowed" }));
          return;
        }
        try {
          const body = await readJsonBody(request, 260_000);
          const narrativeId = String(body.narrativeId || "").trim();
          const narrativeBody = String(body.body || "");
          if (!/^[A-Za-z0-9_.:#-]{1,160}$/.test(narrativeId) || !narrativeBody.trim()) {
            response.statusCode = 400;
            response.end(JSON.stringify({ status: "invalid", reason: "invalid_body_save_request" }));
            return;
          }
          const upstream = await callOmbreDashboard("/api/narrative-rolls/save-body", {
            method: "POST",
            body: {
              narrative_id: narrativeId,
              body: narrativeBody,
              expected_revision: Number.parseInt(body.expectedRevision, 10),
              expected_document_sha256: String(body.expectedDocumentSha256 || "").trim(),
              proposed_material_ids: body.proposedMaterialIds,
              expected_material_snapshot_sha256: String(body.expectedMaterialSnapshotSha256 || "").trim(),
              preview_fingerprint: String(body.previewFingerprint || "").trim(),
            },
          });
          response.statusCode = upstream.status;
          response.end(JSON.stringify(upstream.payload));
        } catch (error) {
          console.error("[serein-memory-bridge] Narrative body save failed", error);
          response.statusCode = error?.name === "AbortError" ? 504 : 502;
          response.end(JSON.stringify({
            status: "error",
            reason: "narrative_body_save_failed",
            message: error?.name === "AbortError" ? "保存正文超时了。" : "这次正文没有保存。",
          }));
        }
      });

      server.middlewares.use("/__serein/live/window-shadows", async (request, response) => {
        response.setHeader("Content-Type", "application/json; charset=utf-8");
        if (request.method !== "POST") {
          response.statusCode = 405;
          response.end(JSON.stringify({ error: "method_not_allowed" }));
          return;
        }
        try {
          response.statusCode = 200;
          response.end(JSON.stringify(await readLiveWindowShadows()));
        } catch (error) {
          response.statusCode = error?.name === "AbortError" ? 504 : 502;
          response.end(JSON.stringify({
            status: "error",
            error: "window_shadow_live_projection_failed",
            message: "没有读到线上窗影，已保留本地快照回退。",
            shadows: [],
          }));
        }
      });

      server.middlewares.use("/__serein/live/dreams", async (request, response) => {
        response.setHeader("Content-Type", "application/json; charset=utf-8");
        if (request.method !== "POST") {
          response.statusCode = 405;
          response.end(JSON.stringify({ error: "method_not_allowed" }));
          return;
        }
        try {
          const body = await readJsonBody(request);
          const limit = Math.max(1, Math.min(Number(body.limit) || 30, 100));
          const upstream = await callOmbreDashboard(`/api/dreams?limit=${limit}`);
          response.statusCode = upstream.status;
          response.end(JSON.stringify(upstream.payload));
        } catch (error) {
          response.statusCode = error?.name === "AbortError" ? 504 : 502;
          response.end(JSON.stringify({
            error: "dream_bridge_failed",
            message: error?.name === "AbortError" ? "读取梦境超时。" : "没有读到梦境。",
          }));
        }
      });

      server.middlewares.use("/__serein/live/dream-detail", async (request, response) => {
        response.setHeader("Content-Type", "application/json; charset=utf-8");
        if (request.method !== "POST") {
          response.statusCode = 405;
          response.end(JSON.stringify({ error: "method_not_allowed" }));
          return;
        }
        try {
          const body = await readJsonBody(request);
          const dreamId = String(body.dreamId || "").trim();
          if (!dreamId) {
            response.statusCode = 400;
            response.end(JSON.stringify({ error: "dream_id_required" }));
            return;
          }
          const upstream = await callOmbreDashboard(`/api/dreams/${encodeURIComponent(dreamId)}`);
          response.statusCode = upstream.status;
          response.end(JSON.stringify(upstream.payload));
        } catch (error) {
          response.statusCode = error?.name === "AbortError" ? 504 : 502;
          response.end(JSON.stringify({
            error: "dream_bridge_failed",
            message: error?.name === "AbortError" ? "读取这场梦超时。" : "这场梦暂时翻不开。",
          }));
        }
      });

      server.middlewares.use("/__serein/memory/scene-evidence", async (request, response) => {
        response.setHeader("Content-Type", "application/json; charset=utf-8");
        if (request.method !== "POST") {
          response.statusCode = 405;
          response.end(JSON.stringify({ error: "method_not_allowed" }));
          return;
        }
        try {
          const body = await readJsonBody(request);
          const sceneId = String(body.sceneId || "").trim();
          if (!/^[A-Za-z0-9_.:#-]{1,160}$/.test(sceneId)) {
            response.statusCode = 400;
            response.end(JSON.stringify({ error: "invalid_scene_id", message: "这张 Scene 没有可读取的原文证据 ID。" }));
            return;
          }
          const result = await callOmbreTool("read_scene_evidence", { scene_id: sceneId });
          response.statusCode = result?.status === "invalid" ? 404 : result?.status === "error" ? 502 : 200;
          response.end(JSON.stringify(result));
        } catch (error) {
          response.statusCode = error?.name === "AbortError" ? 504 : 502;
          response.end(JSON.stringify({
            error: "scene_evidence_read_failed",
            message: error?.name === "AbortError" ? "读取原文证据超时。" : "暂时没有读到这张 Scene 的原文证据。",
          }));
        }
      });

      server.middlewares.use("/__serein/memory/bridge-source-messages", async (request, response) => {
        response.setHeader("Content-Type", "application/json; charset=utf-8");
        if (request.method !== "POST") {
          response.statusCode = 405;
          response.end(JSON.stringify({ error: "method_not_allowed" }));
          return;
        }
        try {
          const body = await readJsonBody(request);
          const result = await readHavenBridgeEvidenceMessages({
            limit: body.limit,
            beforeId: body.beforeId,
            query: body.query,
            contextMessageId: body.contextMessageId,
            contextRadius: body.contextRadius,
          });
          response.statusCode = 200;
          response.end(JSON.stringify(result));
        } catch (error) {
          console.error("[serein] bridge source messages failed", error);
          response.statusCode = 502;
          response.end(JSON.stringify({
            error: "bridge_source_messages_failed",
            message: "暂时没有读到 Haven Bridge 的原文表。",
            items: [],
          }));
        }
      });

      server.middlewares.use("/__serein/memory/bind-scene-evidence", async (request, response) => {
        response.setHeader("Content-Type", "application/json; charset=utf-8");
        if (request.method !== "POST") {
          response.statusCode = 405;
          response.end(JSON.stringify({ error: "method_not_allowed" }));
          return;
        }
        try {
          const body = await readJsonBody(request);
          const sceneId = String(body.sceneId || "").trim();
          const selections = Array.isArray(body.selections) ? body.selections.slice(0, 12) : [];
          const messageIds = selections.map((item) => Number.parseInt(item?.messageId, 10)).filter((item) => item > 0);
          if (!/^[A-Za-z0-9_.:#-]{1,160}$/.test(sceneId) || !messageIds.length || messageIds.length !== selections.length) {
            response.statusCode = 400;
            response.end(JSON.stringify({ error: "invalid_evidence_selection", message: "先选择要绑定的 Bridge 原文。" }));
            return;
          }
          const source = await readHavenBridgeEvidenceMessages({ messageIds });
          const evidenceRefs = buildSceneEvidenceRefs(source.items, selections);
          const result = await callOmbreTool("bind_scene_evidence", {
            scene_id: sceneId,
            evidence_refs: evidenceRefs,
            bound_by: "serein_memory_ui",
          });
          response.statusCode = result?.status === "invalid" ? 400 : result?.status === "error" ? 502 : 200;
          response.end(JSON.stringify(result));
        } catch (error) {
          const invalid = String(error?.message || "").startsWith("evidence_");
          response.statusCode = invalid ? 400 : error?.name === "AbortError" ? 504 : 502;
          response.end(JSON.stringify({
            error: invalid ? error.message : "scene_evidence_bind_failed",
            message: invalid ? "选择的原文已变化，请刷新后重选。" : "没有完成这次原文绑定。",
          }));
        }
      });

      server.middlewares.use("/__serein/memory/unbind-scene-evidence", async (request, response) => {
        response.setHeader("Content-Type", "application/json; charset=utf-8");
        if (request.method !== "POST") {
          response.statusCode = 405;
          response.end(JSON.stringify({ error: "method_not_allowed" }));
          return;
        }
        try {
          const body = await readJsonBody(request);
          const sceneId = String(body.sceneId || "").trim();
          const evidenceIds = Array.isArray(body.evidenceIds)
            ? [...new Set(body.evidenceIds.map((item) => Number.parseInt(item, 10)).filter((item) => item > 0))].slice(0, 12)
            : [];
          if (!/^[A-Za-z0-9_.:#-]{1,160}$/.test(sceneId) || !evidenceIds.length) {
            response.statusCode = 400;
            response.end(JSON.stringify({ error: "invalid_evidence_unbind", message: "没有找到要取消的原文绑定。" }));
            return;
          }
          const result = await callOmbreTool("unbind_scene_evidence", {
            scene_id: sceneId,
            evidence_ids: evidenceIds,
            unbound_by: "serein_memory_ui",
          });
          response.statusCode = result?.status === "invalid" ? 400 : result?.status === "error" ? 502 : 200;
          response.end(JSON.stringify(result));
        } catch (error) {
          response.statusCode = error?.name === "AbortError" ? 504 : 502;
          response.end(JSON.stringify({
            error: "scene_evidence_unbind_failed",
            message: "没有完成这次原文解绑。",
          }));
        }
      });

      server.middlewares.use("/__serein/memory/read-scene", async (request, response) => {
        response.setHeader("Content-Type", "application/json; charset=utf-8");
        if (request.method !== "POST") {
          response.statusCode = 405;
          response.end(JSON.stringify({ error: "method_not_allowed" }));
          return;
        }
        try {
          const body = await readJsonBody(request);
          const sceneId = String(body.sceneId || "").trim();
          if (!/^[A-Za-z0-9_.:#-]{1,160}$/.test(sceneId)) {
            response.statusCode = 400;
            response.end(JSON.stringify({ error: "invalid_scene_id", message: "Scene ID 不正确。" }));
            return;
          }
          const scene = await callOmbreTool("read_memory", { memory_type: "scene", memory_id: sceneId });
          response.statusCode = scene?.status === "not_found" ? 404 : 200;
          response.end(JSON.stringify(scene));
        } catch (error) {
          response.statusCode = error?.name === "AbortError" ? 504 : 502;
          response.end(JSON.stringify({
            error: "memory_bridge_failed",
            message: error?.name === "AbortError" ? "读取 Scene 超时。" : "没有读到这张 Scene 的维护信息。",
          }));
        }
      });

      server.middlewares.use("/__serein/memory/edit-scene-domain", async (request, response) => {
        response.setHeader("Content-Type", "application/json; charset=utf-8");
        if (request.method !== "POST") {
          response.statusCode = 405;
          response.end(JSON.stringify({ error: "method_not_allowed" }));
          return;
        }
        try {
          const body = await readJsonBody(request);
          const sourceId = String(body.sourceId || "").trim();
          const domain = String(body.domain || "").trim();
          if (!/^[A-Za-z0-9_.:#-]{1,160}$/.test(sourceId)) {
            response.statusCode = 400;
            response.end(JSON.stringify({ error: "invalid_scene_id", message: "这张 Scene 没有可写回的 Ombre 来源。" }));
            return;
          }
          if (!canonicalSceneDomains.has(domain)) {
            response.statusCode = 400;
            response.end(JSON.stringify({ error: "invalid_scene_domain", message: "主域不在允许范围内。" }));
            return;
          }

          const upstream = await callOmbreDashboard("/api/buckets/bulk-update", {
            method: "POST",
            body: { bucket_ids: [sourceId], domain },
          });
          response.statusCode = upstream.status;
          response.end(JSON.stringify(upstream.payload));
        } catch (error) {
          response.statusCode = error?.name === "AbortError" ? 504 : 502;
          response.end(JSON.stringify({
            error: "memory_domain_write_failed",
            message: error?.name === "AbortError" ? "修改主域超时。" : "没有把主域写回线上 Ombre。",
          }));
        }
      });

      server.middlewares.use("/__serein/memory/edit-scene-cues", async (request, response) => {
        response.setHeader("Content-Type", "application/json; charset=utf-8");
        if (request.method !== "POST") {
          response.statusCode = 405;
          response.end(JSON.stringify({ error: "method_not_allowed" }));
          return;
        }
        try {
          const body = await readJsonBody(request);
          const sceneId = String(body.sceneId || "").trim();
          const expectedUpdatedAt = String(body.expectedUpdatedAt || "").trim();
          const cues = Array.isArray(body.cues)
            ? body.cues.map((cue) => String(cue || "").trim()).filter(Boolean)
            : [];
          if (!/^[A-Za-z0-9_.:#-]{1,160}$/.test(sceneId) || !expectedUpdatedAt) {
            response.statusCode = 400;
            response.end(JSON.stringify({ error: "invalid_scene_revision", message: "缺少 Scene 或版本信息。" }));
            return;
          }
          if (!cues.length || cues.length > 8 || cues.some((cue) => cue.length > 80)) {
            response.statusCode = 400;
            response.end(JSON.stringify({ error: "invalid_scene_cues", message: "保留 1～8 条 cue，每条不超过 80 字。" }));
            return;
          }
          const result = await callOmbreTool("edit_scene", {
            scene_id: sceneId,
            expected_updated_at: expectedUpdatedAt,
            cues,
          });
          response.statusCode = result?.status === "conflict" ? 409 : 200;
          response.end(JSON.stringify(result));
        } catch (error) {
          response.statusCode = error?.name === "AbortError" ? 504 : 502;
          response.end(JSON.stringify({
            error: "memory_bridge_failed",
            message: error?.name === "AbortError" ? "保存 Scene 超时。" : "没有完成这次 Scene 修订。",
          }));
        }
      });

      server.middlewares.use("/__serein/memory/edit-scene", async (request, response) => {
        response.setHeader("Content-Type", "application/json; charset=utf-8");
        if (request.method !== "POST") {
          response.statusCode = 405;
          response.end(JSON.stringify({ error: "method_not_allowed" }));
          return;
        }
        try {
          const body = await readJsonBody(request);
          const sceneId = String(body.sceneId || "").trim();
          const expectedUpdatedAt = String(body.expectedUpdatedAt || "").trim();
          const title = String(body.title || "").trim();
          const content = String(body.content || "").trim();
          const date = String(body.date || "").trim();
          if (!/^[A-Za-z0-9_.:#-]{1,160}$/.test(sceneId) || !expectedUpdatedAt || !title || !content || !/^\d{4}-\d{2}-\d{2}$/.test(date)) {
            response.statusCode = 400;
            response.end(JSON.stringify({ error: "invalid_scene_revision", message: "日期、标题、正文或 Scene 版本不完整。" }));
            return;
          }
          const result = await callOmbreTool("edit_scene", {
            scene_id: sceneId,
            expected_updated_at: expectedUpdatedAt,
            title,
            content,
            date,
          });
          response.statusCode = result?.status === "conflict" ? 409 : result?.status === "invalid" ? 400 : 200;
          response.end(JSON.stringify(result));
        } catch (error) {
          response.statusCode = error?.name === "AbortError" ? 504 : 502;
          response.end(JSON.stringify({
            error: "scene_revision_failed",
            message: error?.name === "AbortError" ? "保存 Scene 超时。" : "没有完成这次 Scene 修订。",
          }));
        }
      });

      server.middlewares.use("/__serein/memory/set-scene-status", async (request, response) => {
        response.setHeader("Content-Type", "application/json; charset=utf-8");
        if (request.method !== "POST") {
          response.statusCode = 405;
          response.end(JSON.stringify({ error: "method_not_allowed" }));
          return;
        }
        try {
          const body = await readJsonBody(request);
          const sceneId = String(body.sceneId || "").trim();
          const expectedUpdatedAt = String(body.expectedUpdatedAt || "").trim();
          const status = body.status === "archived" ? "archived" : "active";
          if (!/^[A-Za-z0-9_.:#-]{1,160}$/.test(sceneId) || !expectedUpdatedAt) {
            response.statusCode = 400;
            response.end(JSON.stringify({ error: "invalid_scene_status", message: "缺少 Scene 或版本信息。" }));
            return;
          }
          const result = await callOmbreTool("set_scene_status", {
            scene_id: sceneId,
            expected_updated_at: expectedUpdatedAt,
            status,
          });
          response.statusCode = result?.status === "conflict" ? 409 : result?.status === "invalid" ? 400 : 200;
          response.end(JSON.stringify(result));
        } catch (error) {
          response.statusCode = error?.name === "AbortError" ? 504 : 502;
          response.end(JSON.stringify({
            error: "scene_status_failed",
            message: error?.name === "AbortError" ? "更新 Scene 状态超时。" : "没有完成这次 Scene 状态修改。",
          }));
        }
      });

      server.middlewares.use("/__serein/memory/delete-scenes", async (request, response) => {
        response.setHeader("Content-Type", "application/json; charset=utf-8");
        if (request.method !== "POST") {
          response.statusCode = 405;
          response.end(JSON.stringify({ error: "method_not_allowed" }));
          return;
        }
        try {
          const body = await readJsonBody(request);
          const sceneIds = Array.isArray(body.sceneIds)
            ? [...new Set(body.sceneIds.map((value) => String(value || "").trim()))]
              .filter((value) => /^[A-Za-z0-9_.:#-]{1,160}$/.test(value))
            : [];
          if (!sceneIds.length) {
            response.statusCode = 400;
            response.end(JSON.stringify({ error: "invalid_scene_ids", message: "没有找到要删除的 Scene。" }));
            return;
          }
          const upstream = await callOmbreDashboard("/api/buckets/delete", {
            method: "POST",
            body: { bucket_ids: sceneIds, confirm: "DELETE" },
          });
          response.statusCode = upstream.status;
          response.end(JSON.stringify(upstream.payload));
        } catch (error) {
          response.statusCode = error?.name === "AbortError" ? 504 : 502;
          response.end(JSON.stringify({
            error: "scene_delete_failed",
            message: error?.name === "AbortError" ? "删除 Scene 超时。" : "没有完成这次删除。",
          }));
        }
      });

      server.middlewares.use("/__serein/memory/narrative-revisions", async (request, response) => {
        response.setHeader("Content-Type", "application/json; charset=utf-8");
        if (request.method !== "POST") {
          response.statusCode = 405;
          response.end(JSON.stringify({ error: "method_not_allowed" }));
          return;
        }
        try {
          const body = await readJsonBody(request);
          const status = ["pending", "dismissed", "absorbed", "all"].includes(body.status)
            ? body.status
            : "pending";
          const narrativeId = String(body.narrativeId || "").trim();
          const result = await callOmbreTool("narrative_revision_inbox", {
            status,
            narrative_id: narrativeId,
            limit: Math.max(1, Math.min(Number(body.limit) || 50, 100)),
          });
          response.statusCode = result?.status === "invalid" ? 400 : 200;
          response.end(JSON.stringify(result));
        } catch (error) {
          response.statusCode = error?.name === "AbortError" ? 504 : 502;
          response.end(JSON.stringify({
            error: "revision_bridge_failed",
            message: error?.name === "AbortError" ? "读取修订箱超时。" : "没有读到修订箱。",
          }));
        }
      });

      server.middlewares.use("/__serein/memory/review-narrative-revision", async (request, response) => {
        response.setHeader("Content-Type", "application/json; charset=utf-8");
        if (request.method !== "POST") {
          response.statusCode = 405;
          response.end(JSON.stringify({ error: "method_not_allowed" }));
          return;
        }
        try {
          const body = await readJsonBody(request);
          const proposalId = String(body.proposalId || "").trim();
          const action = String(body.action || "").trim();
          if (!proposalId || !["save_draft", "dismiss", "reopen"].includes(action)) {
            response.statusCode = 400;
            response.end(JSON.stringify({ error: "invalid_revision_action", message: "修订动作不正确。" }));
            return;
          }
          const result = await callOmbreTool("review_narrative_revision", {
            proposal_id: proposalId,
            action,
            draft_delta: String(body.draftDelta || ""),
            note: String(body.note || ""),
          });
          response.statusCode = result?.status === "not_found" ? 404 : result?.status === "invalid" ? 400 : 200;
          response.end(JSON.stringify(result));
        } catch (error) {
          response.statusCode = error?.name === "AbortError" ? 504 : 502;
          response.end(JSON.stringify({
            error: "revision_bridge_failed",
            message: error?.name === "AbortError" ? "保存修订草稿超时。" : "没有完成这次修订箱操作。",
          }));
        }
      });

      server.middlewares.use("/__serein/memory/scene-edge-proposals", async (request, response) => {
        response.setHeader("Content-Type", "application/json; charset=utf-8");
        if (request.method !== "POST") {
          response.statusCode = 405;
          response.end(JSON.stringify({ error: "method_not_allowed" }));
          return;
        }
        try {
          const body = await readJsonBody(request);
          const allowedStatuses = ["pending", "accepted", "rejected", "superseded", "all"];
          const status = allowedStatuses.includes(body.status) ? body.status : "pending";
          const limit = Math.max(1, Math.min(Number(body.limit) || 30, 100));
          const query = new URLSearchParams({ status, limit: String(limit), include_context: "true" });
          const upstream = await callOmbreDashboard(`/api/scene-edge-proposals?${query}`);
          response.statusCode = upstream.status;
          response.end(JSON.stringify(upstream.payload));
        } catch (error) {
          const unconfigured = error?.message === "dashboard_bridge_not_configured";
          response.statusCode = unconfigured ? 503 : error?.name === "AbortError" ? 504 : 502;
          response.end(JSON.stringify({
            error: unconfigured ? "dashboard_bridge_not_configured" : "relationship_bridge_failed",
            message: unconfigured ? "本地预览没有安全载入 Dashboard 凭据。" : "没有读到关系提案。",
          }));
        }
      });

      server.middlewares.use("/__serein/memory/review-scene-edge-proposal", async (request, response) => {
        response.setHeader("Content-Type", "application/json; charset=utf-8");
        if (request.method !== "POST") {
          response.statusCode = 405;
          response.end(JSON.stringify({ error: "method_not_allowed" }));
          return;
        }
        try {
          const body = await readJsonBody(request);
          const proposalId = String(body.proposalId || "").trim();
          const decision = String(body.decision || "").trim();
          if (!proposalId || !["accept", "reject"].includes(decision)) {
            response.statusCode = 400;
            response.end(JSON.stringify({ error: "invalid_relationship_action", message: "关系审核动作不正确。" }));
            return;
          }
          const upstream = await callOmbreDashboard("/api/scene-edge-proposals/review", {
            method: "POST",
            body: {
              proposal_id: proposalId,
              decision,
              confirm: decision === "accept" ? "ACCEPT_SCENE_EDGE" : "REJECT_SCENE_EDGE",
            },
          });
          response.statusCode = upstream.status;
          response.end(JSON.stringify(upstream.payload));
        } catch (error) {
          const unconfigured = error?.message === "dashboard_bridge_not_configured";
          response.statusCode = unconfigured ? 503 : error?.name === "AbortError" ? 504 : 502;
          response.end(JSON.stringify({
            error: unconfigured ? "dashboard_bridge_not_configured" : "relationship_bridge_failed",
            message: unconfigured ? "本地预览没有安全载入 Dashboard 凭据。" : "没有完成这次关系审核。",
          }));
        }
      });

      server.middlewares.use("/__serein/memory/create-scene-edge-proposal", async (request, response) => {
        response.setHeader("Content-Type", "application/json; charset=utf-8");
        if (request.method !== "POST") {
          response.statusCode = 405;
          response.end(JSON.stringify({ error: "method_not_allowed" }));
          return;
        }
        try {
          const body = await readJsonBody(request);
          const safeId = /^[A-Za-z0-9_.:#-]{1,160}$/;
          const sourceSceneId = String(body.sourceSceneId || "").trim();
          const targetSceneId = String(body.targetSceneId || "").trim();
          if (!safeId.test(sourceSceneId) || !safeId.test(targetSceneId) || sourceSceneId === targetSceneId) {
            response.statusCode = 400;
            response.end(JSON.stringify({ error: "invalid_scene_edge_target", message: "需要两张不同的 Scene。" }));
            return;
          }
          const upstream = await callOmbreDashboard("/api/scene-edge-proposals/manual", {
            method: "POST",
            body: {
              source_scene_id: sourceSceneId,
              target_scene_id: targetSceneId,
              relation_type: String(body.relationType || "").trim(),
              source_evidence: String(body.sourceEvidence || ""),
              target_evidence: String(body.targetEvidence || ""),
              reason: String(body.reason || ""),
              supersedes_edge_id: String(body.supersedesEdgeId || "").trim(),
              confidence: 1,
              confirm: "CREATE_SCENE_EDGE_PROPOSAL",
            },
          });
          response.statusCode = upstream.status;
          response.end(JSON.stringify(upstream.payload));
        } catch (error) {
          const unconfigured = error?.message === "dashboard_bridge_not_configured";
          response.statusCode = unconfigured ? 503 : error?.name === "AbortError" ? 504 : 502;
          response.end(JSON.stringify({
            error: unconfigured ? "dashboard_bridge_not_configured" : "scene_edge_create_bridge_failed",
            message: unconfigured ? "本地预览没有安全载入 Dashboard 凭据。" : "没有创建这条关系提案。",
          }));
        }
      });

      server.middlewares.use("/__serein/memory/scene-edges", async (request, response) => {
        response.setHeader("Content-Type", "application/json; charset=utf-8");
        if (request.method !== "POST") {
          response.statusCode = 405;
          response.end(JSON.stringify({ error: "method_not_allowed" }));
          return;
        }
        try {
          const body = await readJsonBody(request);
          const query = new URLSearchParams({ include_inactive: "true" });
          if (body.sceneId) query.set("scene_id", String(body.sceneId).trim());
          const upstream = await callOmbreDashboard(`/api/scene-edges?${query}`);
          response.statusCode = upstream.status;
          response.end(JSON.stringify(upstream.payload));
        } catch (error) {
          const unconfigured = error?.message === "dashboard_bridge_not_configured";
          response.statusCode = unconfigured ? 503 : error?.name === "AbortError" ? 504 : 502;
          response.end(JSON.stringify({
            error: unconfigured ? "dashboard_bridge_not_configured" : "scene_edge_history_bridge_failed",
            message: unconfigured ? "本地预览没有安全载入 Dashboard 凭据。" : "没有读到关系边历史。",
          }));
        }
      });

      server.middlewares.use("/__serein/memory/delete-scene-edge", async (request, response) => {
        response.setHeader("Content-Type", "application/json; charset=utf-8");
        if (request.method !== "POST") {
          response.statusCode = 405;
          response.end(JSON.stringify({ error: "method_not_allowed" }));
          return;
        }
        try {
          const body = await readJsonBody(request);
          const edgeId = String(body.edgeId || "").trim();
          const sceneId = String(body.sceneId || "").trim();
          const safeId = /^[A-Za-z0-9_.:#-]{1,160}$/;
          if (!safeId.test(edgeId) || !safeId.test(sceneId)) {
            response.statusCode = 400;
            response.end(JSON.stringify({ error: "invalid_scene_edge_target", message: "关系边或 Scene ID 不正确。" }));
            return;
          }
          const upstream = await callOmbreDashboard(`/api/scene-edges/${encodeURIComponent(edgeId)}`, {
            method: "DELETE",
            body: {
              scene_id: sceneId,
              confirm: "DELETE_SCENE_EDGE",
              reason: "manual_ui_remove",
            },
          });
          response.statusCode = upstream.status;
          response.end(JSON.stringify(upstream.payload));
        } catch (error) {
          const unconfigured = error?.message === "dashboard_bridge_not_configured";
          response.statusCode = unconfigured ? 503 : error?.name === "AbortError" ? 504 : 502;
          response.end(JSON.stringify({
            error: unconfigured ? "dashboard_bridge_not_configured" : "scene_edge_delete_bridge_failed",
            message: unconfigured ? "本地预览没有安全载入 Dashboard 凭据。" : "没有完成这次关系边移除。",
          }));
        }
      });

      server.middlewares.use("/__serein/memory/restore-scene-edge", async (request, response) => {
        response.setHeader("Content-Type", "application/json; charset=utf-8");
        if (request.method !== "POST") {
          response.statusCode = 405;
          response.end(JSON.stringify({ error: "method_not_allowed" }));
          return;
        }
        try {
          const body = await readJsonBody(request);
          const edgeId = String(body.edgeId || "").trim();
          const safeId = /^[A-Za-z0-9_.:#-]{1,160}$/;
          if (!safeId.test(edgeId)) {
            response.statusCode = 400;
            response.end(JSON.stringify({ error: "invalid_scene_edge_target", message: "关系边 ID 不正确。" }));
            return;
          }
          const upstream = await callOmbreDashboard(`/api/scene-edges/${encodeURIComponent(edgeId)}/restore`, {
            method: "POST",
            body: { confirm: "RESTORE_SCENE_EDGE" },
          });
          response.statusCode = upstream.status;
          response.end(JSON.stringify(upstream.payload));
        } catch (error) {
          const unconfigured = error?.message === "dashboard_bridge_not_configured";
          response.statusCode = unconfigured ? 503 : error?.name === "AbortError" ? 504 : 502;
          response.end(JSON.stringify({
            error: unconfigured ? "dashboard_bridge_not_configured" : "scene_edge_restore_bridge_failed",
            message: unconfigured ? "本地预览没有安全载入 Dashboard 凭据。" : "没有恢复这条关系边。",
          }));
        }
      });
    },
  };
}

export default defineConfig({
  base: String(process.env.SEREIN_BASE_PATH || "/"),
  build: {
    outDir: "dist/client",
  },
  optimizeDeps: {
    include: ["react", "react-dom/client"],
  },
  server: {
    host: "0.0.0.0",
    allowedHosts: [
      "serein.rain4haven99.cyou",
      "haven.rain4haven99.cyou",
      "terminal.local",
      "23456544321123.asia",
      "8.136.154.242",
      "localhost",
    ],
    warmup: {
      clientFiles: ["./src/main.jsx"],
    },
  },
  plugins: [react(), sereinGatewayBridge(), sereinMemoryBridge()],
});
