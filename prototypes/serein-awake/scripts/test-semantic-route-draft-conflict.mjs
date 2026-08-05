import assert from "node:assert/strict";
import { inspectServerSemanticRouteDraft } from "../src/storage/basementStore.js";

const publishedRoutes = [{ name: "present_chitchat" }, { name: "技术闲聊" }];
const staleDraft = {
  baseDatasetVersion: 31,
  revision: 8,
  routes: [{ name: "present_chitchat" }],
};

const conflict = inspectServerSemanticRouteDraft({ draft: staleDraft }, 32);
assert.equal(conflict.status, "conflict");
assert.equal(conflict.baseDatasetVersion, 31);
assert.equal(conflict.datasetVersion, 32);
assert.deepEqual(publishedRoutes.map((route) => route.name), ["present_chitchat", "技术闲聊"]);

const current = inspectServerSemanticRouteDraft({ draft: { ...staleDraft, baseDatasetVersion: 32 } }, 32);
assert.equal(current.status, "current");
assert.equal(current.draft.revision, 8);

assert.deepEqual(inspectServerSemanticRouteDraft({}, 32), { status: "none", draft: null });

console.log("semantic route draft conflict checks passed");
