import assert from "node:assert/strict";

import { normalizeRecallSimulationOptions } from "../vite.config.mjs";


assert.deepEqual(normalizeRecallSimulationOptions({}), {
  ok: true,
  recallAblation: "normal",
  simulation: false,
  simulationScope: "live_mirror",
});
assert.deepEqual(normalizeRecallSimulationOptions({
  simulation: true,
  simulation_scope: "full_shadow",
  recall_ablation: "without_cues",
}), {
  ok: true,
  recallAblation: "without_cues",
  simulation: true,
  simulationScope: "full_shadow",
});
assert.equal(
  normalizeRecallSimulationOptions({ recall_ablation: "without_embedding" }).error,
  "recall_ablation_requires_simulation",
);
assert.equal(
  normalizeRecallSimulationOptions({ simulation: true, recall_ablation: "other" }).error,
  "invalid_recall_ablation",
);
assert.equal(
  normalizeRecallSimulationOptions({ simulation: true, recall_ablation: "without_cues" }).error,
  "recall_ablation_requires_full_shadow",
);
assert.equal(
  normalizeRecallSimulationOptions({ simulation: true, simulation_scope: "other" }).error,
  "invalid_simulation_scope",
);

console.log("recall simulation budget bridge checks: PASS");
