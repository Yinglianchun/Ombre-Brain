import assert from "node:assert/strict";

const storage = new Map();
globalThis.window = {
  localStorage: {
    getItem(key) {
      return storage.has(key) ? storage.get(key) : null;
    },
    setItem(key, value) {
      storage.set(key, String(value));
    },
    removeItem(key) {
      storage.delete(key);
    },
  },
};

const requests = [];
globalThis.fetch = async (url, options = {}) => {
  requests.push({ url: String(url), options });
  assert.equal(String(url), "/__serein/live/memory-scenes");
  return {
    ok: true,
    async json() {
      return {
        status: "ok",
        snapshotId: "live-only",
        scenes: [
          {
            source_id: "scene-live-1",
            date: "2026-08-10",
            title: "德国机真实 Scene",
            content: "这是从 canonical 读取的正文。",
            author: "Haven",
            favorite: true,
            status: "可浮现",
            bucket_domain: "relationship",
            self_anchor: false,
            updated_at: "2026-08-10T10:00:00Z",
            content_hash: "hash-live-1",
          },
        ],
        edges: [],
      };
    },
  };
};

const { loadMemorySnapshot } = await import("../src/storage/memoryStore.js");
const scenes = await loadMemorySnapshot();

assert.equal(requests.length, 1);
assert.deepEqual(JSON.parse(requests[0].options.body), { sourceIds: [] });
assert.equal(scenes.length, 1);
assert.equal(scenes[0].id, "scene-live-1");
assert.equal(scenes[0].title, "德国机真实 Scene");
assert.equal(scenes[0].sourceKind, "ombre-live-readonly");
assert.equal(scenes.some((scene) => scene.id === "first-door"), false);

const cachedScenes = await loadMemorySnapshot();
assert.equal(requests.length, 1);
assert.equal(cachedScenes.length, 1);
assert.equal(cachedScenes[0].id, "scene-live-1");

console.log("live-only memory projection checks: PASS");
