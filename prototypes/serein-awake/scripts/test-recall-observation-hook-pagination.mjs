import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const directory = mkdtempSync(path.join(tmpdir(), "serein-hook-observation-"));
const databasePath = path.join(directory, "haven.db");
const pythonBin = process.env.SEREIN_TEST_PYTHON
  || (process.platform === "win32" ? "C:\\Python313\\python.exe" : "python3");
const rows = [
  [3, 1, "user", "2026-08-03 03:00:00", "old reviewed", { hook_memory_outcome: "injected", developer_context: "private" }],
  [4, 1, "user", "2026-08-03 04:00:00", "not a hook row", {}],
  [5, 1, "user", "2026-08-03 05:00:00", "older page", { hook_memory_outcome: "no_match" }],
  [6, 1, "assistant", "2026-08-03 06:00:00", "assistant row", { hook_memory_outcome: "injected" }],
  [7, 1, "user", "2026-08-03 07:00:00", "recent two", { hook_memory_outcome: "injected" }],
  [8, 1, "user", "2026-08-03 08:00:00", "most recent", {
    hook_memory_outcome: "injected",
    gateway_memory_injected_ids: ["scene-1"],
    gateway_memory_items: [{ id: "scene-1", title: "visible title", score: 0.8, body: "must not escape" }],
    additional_context: "private",
  }],
];
const setup = spawnSync(pythonBin, ["-"], {
  encoding: "utf8",
  input: `
import json, sqlite3
conn = sqlite3.connect(${JSON.stringify(databasePath)})
conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id INTEGER, role TEXT, created_at TEXT, content TEXT, metadata_json TEXT)")
rows = json.loads(${JSON.stringify(JSON.stringify(rows))})
conn.executemany("INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)", [(*row[:5], json.dumps(row[5], ensure_ascii=False)) for row in rows])
conn.commit()
conn.close()
`,
});
assert.equal(setup.status, 0, setup.stderr);

process.env.HAVEN_BRIDGE_LOCAL_DB = databasePath;
process.env.SEREIN_PYTHON_BIN = pythonBin;

try {
  const { readHavenBridgeHookLedger } = await import("../vite.config.mjs");
  const firstPage = await readHavenBridgeHookLedger(2, 0, [3]);
  assert.equal(firstPage.status, "ok");
  assert.deepEqual(firstPage.items.map((item) => item.id), [8, 7]);
  assert.equal(firstPage.has_more, true);
  assert.equal(firstPage.next_before_id, 7);
  assert.deepEqual(firstPage.reviewed_items.map((item) => item.id), [3]);
  assert.equal(Object.hasOwn(firstPage.items[0], "developer_context"), false);
  assert.equal(Object.hasOwn(firstPage.items[0], "additional_context"), false);
  assert.equal(Object.hasOwn(firstPage.items[0].gateway_memory_items[0], "body"), false);

  const secondPage = await readHavenBridgeHookLedger(2, firstPage.next_before_id);
  assert.deepEqual(secondPage.items.map((item) => item.id), [5, 3]);
  assert.equal(secondPage.has_more, false);
  assert.equal(secondPage.next_before_id, 3);
} finally {
  delete process.env.HAVEN_BRIDGE_LOCAL_DB;
  delete process.env.SEREIN_PYTHON_BIN;
  rmSync(directory, { recursive: true, force: true });
}

console.log("hook observation pagination checks: PASS");
