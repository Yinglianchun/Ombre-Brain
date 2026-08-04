import assert from "node:assert/strict";

import { normalizeRecallSimulationOptions } from "../vite.config.mjs";


assert.deepEqual(normalizeRecallSimulationOptions({}), {
  ok: true,
  recallAblation: "normal",
  simulation: false,
});
assert.deepEqual(normalizeRecallSimulationOptions({
  simulation: true,
  recall_ablation: "without_cues",
}), {
  ok: true,
  recallAblation: "without_cues",
  simulation: true,
});
assert.equal(
  normalizeRecallSimulationOptions({ recall_ablation: "without_embedding" }).error,
  "recall_ablation_requires_simulation",
);
assert.equal(
  normalizeRecallSimulationOptions({ simulation: true, recall_ablation: "other" }).error,
  "invalid_recall_ablation",
);

console.log("recall simulation budget bridge checks: PASS");
