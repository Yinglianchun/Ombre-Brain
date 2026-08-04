import assert from "node:assert/strict";
import {
  buildRecallObservationTrainingExport,
  classifyObservationReview,
  observationGroup,
} from "../src/storage/recallObservationExport.js";
import { createRecallSimulationTrainingLabel } from "../src/storage/recallSimulationTraining.js";

const items = [
  {
    id: "hook-1",
    query: "最爱哥哥了",
    queryAvailable: true,
    observedAction: "",
    route: "present_chitchat",
    sessionId: "s-1",
    createdAt: "2026-08-03 12:00:00",
    source: "hook",
    injected: [
      { id: "scene-origin", title: "小雨讲了我们怎么开始的", scoreValue: 0.536 },
      { id: "scene-meteor", title: "我们关于流星的讨论", scoreValue: 0.565 },
    ],
  },
  {
    id: "gateway-2",
    query: "哥哥在吗",
    queryAvailable: true,
    observedAction: "skip",
    route: "present_chitchat",
    sessionId: "s-2",
    createdAt: "2026-08-03 12:01:00",
    source: "gateway",
  },
  {
    id: "hook-3",
    query: "不确定的一句",
    queryAvailable: true,
    observedAction: "recall",
    route: "recall_needed",
    sessionId: "s-3",
    createdAt: "2026-08-03 12:02:00",
    source: "hook",
  },
  {
    id: "gateway-4",
    query: "旧记录未保留原句",
    queryAvailable: false,
    observedAction: "recall",
    route: "recall_needed",
    sessionId: "s-4",
    createdAt: "2026-08-03 12:03:00",
    source: "gateway",
  },
  {
    id: "hook-5",
    query: "没有人工判断",
    queryAvailable: true,
    observedAction: "recall",
    route: "recall_needed",
    sessionId: "s-5",
    createdAt: "2026-08-03 12:04:00",
    source: "hook",
  },
];

const reviews = {
  "hook-1": {
    verdict: "false_positive",
    updatedAt: "2026-08-03T04:01:00Z",
    full_prompt: "must not export",
    candidateReviews: {
      "scene-origin": "core",
      "scene-meteor": "weak",
      "scene-unknown": "invalid",
    },
  },
  "gateway-2": {
    verdict: "correct",
    routeVerdict: "incorrect",
    expectedRoute: "present_chitchat",
    updatedAt: "2026-08-03T04:02:00Z",
  },
  "hook-3": { verdict: "uncertain", updatedAt: "2026-08-03T04:03:00Z" },
  "gateway-4": { verdict: "missed", updatedAt: "2026-08-03T04:04:00Z" },
};

const manualSimulation = createRecallSimulationTrainingLabel({
  query: "给予别人善意也是在善待自己",
  expectedAction: "recall",
  expectedRoute: "recall_needed",
  expectedMemoryIds: ["scene_b62357c38b2f41b58cb0fbbf3eef6820"],
  observedAction: "skip",
  observedRoute: "present_chitchat",
}, {
  id: "manual-simulation-1",
  batchId: "batch-1",
  now: "2026-08-03T04:04:30Z",
});

const payload = buildRecallObservationTrainingExport(
  items,
  reviews,
  "2026-08-03T04:05:00Z",
  [manualSimulation],
);
assert.equal(payload.schema_version, 4);
assert.equal(payload.export_type, "serein.basement.recall-observation-training-export");
assert.deepEqual(payload.summary, {
  total_observations: 5,
  total_manual_simulations: 1,
  total_cases: 6,
  available: 3,
  rejected: 2,
  missing: 1,
});
assert.equal(payload.observations[0].observation_id, "hook-1");
assert.equal(payload.observations[0].verdict, "false_positive");
assert.equal(payload.observations[0].group, "session:s-1");
assert.deepEqual(payload.observations[0].candidate_judgments, [
  { memory_id: "scene-origin", rank: 1, observed_score: 0.536, relevance: "core" },
  { memory_id: "scene-meteor", rank: 2, observed_score: 0.565, relevance: "weak" },
]);
assert.equal(Object.hasOwn(payload.observations[0].candidate_judgments[0], "title"), false);
assert.equal(payload.observations[1].observed_action, "skip");
assert.equal(payload.observations[1].action_verdict, "correct");
assert.equal(payload.observations[1].route_verdict, "incorrect");
assert.equal(payload.observations[1].expected_route, "present_chitchat");
assert.equal(payload.reviews["gateway-2"].route_verdict, "incorrect");
assert.equal(payload.observations[3].query, "");
assert.equal(payload.observations[4].verdict, null);
assert.equal(payload.observations[5].source, "manual_simulation");
assert.equal(payload.observations[5].verdict, "missed");
assert.equal(payload.observations[5].label_action, "recall");
assert.equal(payload.observations[5].label_route, "recall_needed");
assert.equal(payload.observations[5].observed_action, "skip");
assert.equal(payload.observations[5].observed_route, "present_chitchat");
assert.equal(payload.observations[5].group, "manual-simulation:batch-1");
assert.deepEqual(payload.observations[5].expected_memory_ids, ["scene_b62357c38b2f41b58cb0fbbf3eef6820"]);
assert.equal(payload.reviews["manual-simulation-1"].verdict, "missed");
assert.equal(Object.hasOwn(payload.observations[0], "injected"), false);
assert.equal(Object.hasOwn(payload.observations[0], "full_prompt"), false);
assert.equal(Object.hasOwn(payload.reviews["hook-1"], "full_prompt"), false);
assert.equal(classifyObservationReview(items[1], reviews["gateway-2"]).label, "skip");
assert.equal(classifyObservationReview(items[2], reviews["hook-3"]).kind, "rejected");
assert.deepEqual(observationGroup({ id: "gateway-6", createdAt: "2026-08-03T04:06:00Z" }), {
  key: "date:2026-08-03",
  basis: "created_at_date",
});
console.log("recall observation export checks: PASS");
