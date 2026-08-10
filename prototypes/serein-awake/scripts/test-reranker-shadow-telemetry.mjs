import assert from "node:assert/strict";

import {
  buildRecallObservationTrainingExport,
} from "../src/storage/recallObservationExport.js";
import {
  EVALUATION_GROUPING_CONTRACT,
  createRecallSimulationTrainingLabel,
  evaluationGroupIdFor,
  normalizeCandidateTelemetrySnapshot,
  normalizeSimulationTelemetrySnapshot,
  normalizeSimulationQuery,
  queryFamilyIdFor,
  upsertRecallSimulationTrainingLabel,
  readRecallSimulationTrainingLabels,
} from "../src/storage/recallSimulationTraining.js";
import {
  birthdaySimulationBatchId,
  birthdayLiveRawSchemaMapping,
  birthdaySimulationRegressionFixtures,
  fixtureCandidateTelemetry,
  fixtureSimulationTelemetry,
} from "./fixtures/birthday-simulation-regressions.mjs";

const localStorageValues = new Map();
const sessionStorageValues = new Map();
globalThis.window = {
  localStorage: {
    getItem: (key) => localStorageValues.get(key) ?? null,
    setItem: (key, value) => localStorageValues.set(key, String(value)),
    removeItem: (key) => localStorageValues.delete(key),
  },
  sessionStorage: {
    getItem: (key) => sessionStorageValues.get(key) ?? null,
    setItem: (key, value) => sessionStorageValues.set(key, String(value)),
  },
  dispatchEvent: () => true,
};

const candidate = {
  bucket_id: "scene-fruit",
  rank: 3,
  title: "不得导出的 Scene 标题",
  body: "不得导出的正文",
  matched_cues: ["不得导出的 cue"],
  prompt: "不得导出的 prompt",
  candidate_sources: ["body_semantic", "cue_semantic", "cue_lexical", "title_anchor", "not-real"],
  body_semantic_score: 0.63,
  cue_semantic: { status: "matched", score: 0.77, role: "candidate_only", matched_cues: ["不得导出的 cue"] },
  cue_lexical_match: true,
  cue_lexical_role: "direct_evidence",
  title_anchor_match: true,
  title_anchor_terms: ["不得导出的标题锚点"],
  semantic_profile: "canonical_body_only",
  floor_qualified: true,
  gray_zone_qualified: false,
  reranker_eligible: true,
  absolute_floor: 0.55,
  reranker_entry_floor: 0.50,
  final_admission_source: "pending_reranker_shadow_route_guard",
  reranker_shadow: {
    called: true,
    score: 0.91,
    model: "Qwen/Qwen3-Reranker-4B",
    decision_applied: false,
    status: "scored_shadow_only",
    evidence_status: "unknown",
    evidence_count: 0,
    telemetry_generated_at: "2026-08-06T09:00:00.123Z",
  },
  episode_verifier: {
    called: true,
    model: "deepseek-v4-flash",
    verdict: "symbolic_resonance",
    confidence: 0.82,
    current_evidence_span: "餐桌上的巨型水果",
    grounded_cue: "巨型水果",
    reason: "reviewed cue grounds the image",
    decision_applied: false,
    admission_effect: "shadow_only",
  },
  evidence_source_text: "不得导出的证据原文",
};

const normalLabel = createRecallSimulationTrainingLabel({
  query: "餐桌上的巨型水果",
  expectedAction: "recall",
  expectedRoute: "recall_needed",
  observedAction: "recall",
  observedRoute: "recall_needed",
  ablationMode: "normal",
  candidateJudgments: [{ memoryId: "scene-fruit", rank: 1, relevance: "core" }],
  candidateTelemetry: [candidate],
}, {
  id: "manual-simulation-telemetry",
  batchId: "batch-telemetry",
  now: "2026-08-06T09:00:01.000Z",
});
assert.equal(normalLabel.observationId, "manual-simulation-telemetry");
assert.equal(normalLabel.queryNormalized, "餐桌上的巨型水果");
assert.deepEqual(normalLabel.candidateJudgments, [
  { memoryId: "scene-fruit", rank: 1, relevance: "core" },
]);
assert.equal(normalLabel.candidateTelemetry.length, 1);
assert.deepEqual(normalLabel.candidateTelemetry[0].candidateSources, [
  "body_semantic",
  "cue_semantic",
  "cue_lexical",
  "title_anchor",
]);
assert.equal(normalLabel.candidateTelemetry[0].rerankerShadow.score, 0.91);
assert.equal(normalLabel.candidateTelemetry[0].rerankerShadow.decisionApplied, false);
assert.equal(normalLabel.candidateTelemetry[0].episodeVerifier.verdict, "symbolic_resonance");
assert.equal(normalLabel.candidateTelemetry[0].episodeVerifier.groundedCue, "巨型水果");
assert.equal(normalLabel.candidateTelemetry[0].episodeVerifier.decisionApplied, false);
assert.equal(normalLabel.candidateTelemetry[0].floors.entry, 0.50);
assert.equal(Object.hasOwn(normalLabel.candidateTelemetry[0], "title"), false);
assert.equal(Object.hasOwn(normalLabel.candidateTelemetry[0], "body"), false);
assert.equal(Object.hasOwn(normalLabel.candidateTelemetry[0], "matched_cues"), false);
assert.equal(Object.hasOwn(normalLabel.candidateTelemetry[0], "prompt"), false);
const normalizedRoundTrip = normalizeCandidateTelemetrySnapshot(normalLabel.candidateTelemetry, {
  observationId: normalLabel.observationId,
  query: normalLabel.query,
  ablationMode: normalLabel.ablationMode,
});
assert.equal(normalizedRoundTrip[0].sourceTelemetry.bodySemantic.score, 0.63);
assert.equal(normalizedRoundTrip[0].sourceTelemetry.cueSemantic.score, 0.77);
assert.equal(normalizedRoundTrip[0].eligibility.absoluteFloorQualified, true);

const threeStateTelemetry = normalizeSimulationTelemetrySnapshot({
  retrieval_budget: {
    mode: "simulation_shadow",
    initial_budget: "shallow",
    final_budget: "deep",
    budget_decision_source: "typed_candidate_probe",
    escalation_reason: "event_candidate_over_rescue_floor",
    typed_candidate_id: "event-postcard",
    typed_candidate_kind: "event",
    typed_candidate_score: 0.73,
    fact_event_probe: {
      status: "ok",
      candidate_count: 2,
      matches: [
        {
          memory_id: "event-postcard",
          memory_kind: "event",
          score: 0.73,
          importance: 5,
          local_date: "2026-08-08",
          local_start_time: "20:17",
          covered_by_scene_id: "",
        },
        {
          memory_id: "fact-private-body",
          memory_kind: "fact",
          score: 0.61,
          importance: 3,
          local_date: "2026-08-08",
          local_start_time: "18:01",
          covered_by_scene_id: "scene-private-body",
          body: "不得导出的 Fact 正文",
        },
      ],
    },
    episode_verifier: {
      enabled: true,
      called: true,
      model: "deepseek-v4-flash",
      reason: "scored",
      decision_scope: "simulation_negative_veto_only",
      candidate_count: 1,
      timing_ms: 412,
      decisions: [{
        candidate_id: "scene-fruit",
        verdict: "same_topic_only",
        confidence: 0.94,
        decision_applied: true,
        admission_effect: "negative_veto",
      }],
    },
  },
}, { force: true });
assert.equal(threeStateTelemetry.budget.initialBudget, "shallow");
assert.equal(threeStateTelemetry.budget.finalBudget, "deep");
assert.equal(threeStateTelemetry.budget.escalationReason, "event_candidate_over_rescue_floor");
assert.equal(threeStateTelemetry.budget.factEventProbe.matches[0].memoryKind, "event");
assert.equal(threeStateTelemetry.budget.factEventProbe.matches[1].coveredBySceneId, "scene-private-body");
assert.equal(Object.hasOwn(threeStateTelemetry.budget.factEventProbe.matches[1], "body"), false);
assert.equal(threeStateTelemetry.budget.episodeVerifier.called, true);
assert.equal(threeStateTelemetry.budget.episodeVerifier.decisions[0].verdict, "same_topic_only");
assert.equal(threeStateTelemetry.budget.episodeVerifier.decisions[0].decisionApplied, true);

const updated = upsertRecallSimulationTrainingLabel({
  query: "  餐桌上的巨型水果  ",
  expectedAction: "recall",
  expectedRoute: "recall_needed",
  observedAction: "recall",
  observedRoute: "recall_needed",
  ablationMode: "without_cues",
  candidateTelemetry: [{
    ...candidate,
    bucket_id: "scene-fruit-updated",
    candidate_sources: ["body_semantic"],
  }],
});
assert.equal(updated.status, "added");
const savedAgain = upsertRecallSimulationTrainingLabel({
  query: "餐桌上的巨型水果",
  expectedAction: "recall",
  expectedRoute: "recall_needed",
  observedAction: "recall",
  observedRoute: "recall_needed",
  ablationMode: "without_cues",
  candidateTelemetry: [{
    ...candidate,
    bucket_id: "scene-fruit-final",
    candidate_sources: ["title_anchor"],
  }],
});
assert.equal(savedAgain.status, "updated");
assert.equal(savedAgain.labels.length, 1);
assert.equal(savedAgain.label.observationId, updated.label.observationId);
assert.equal(savedAgain.label.queryNormalized, "餐桌上的巨型水果");
assert.equal(savedAgain.label.ablationMode, "without_cues");
assert.equal(savedAgain.label.candidateTelemetry[0].candidateId, "scene-fruit-final");
const distinctMode = upsertRecallSimulationTrainingLabel({
  query: "餐桌上的巨型水果",
  expectedAction: "recall",
  expectedRoute: "recall_needed",
  observedAction: "recall",
  observedRoute: "recall_needed",
  ablationMode: "without_embedding",
  candidateTelemetry: [{
    ...candidate,
    bucket_id: "scene-fruit-without-embedding",
    candidate_sources: ["body_semantic"],
  }],
});
assert.equal(distinctMode.status, "added");
assert.equal(distinctMode.labels.length, 2);
assert.notEqual(distinctMode.label.observationId, updated.label.observationId);
assert.equal(readRecallSimulationTrainingLabels().length, 2);

const groupedModes = ["normal", "without_cues", "without_embedding"].map((mode) => createRecallSimulationTrainingLabel({
  query: "同一 query family 的生日对照",
  expectedAction: mode === "without_cues" ? "skip" : "recall",
  observedAction: "recall",
  ablationMode: mode,
  candidateTelemetry: [{
    ...candidate,
    bucket_id: `scene-group-${mode}`,
    candidate_sources: mode === "without_embedding" ? ["cue_lexical"] : ["body_semantic"],
    reranker_shadow: { ...candidate.reranker_shadow, score: mode === "normal" ? 0.12 : 0.98 },
  }],
}, {
  id: `grouped-${mode}`,
  batchId: "batch-group-contract",
  now: "2026-08-06T09:01:30.000Z",
}));
assert.equal(new Set(groupedModes.map((item) => item.queryFamilyId)).size, 1);
assert.equal(new Set(groupedModes.map((item) => item.evaluationGroupId)).size, 1);
assert.equal(new Set(groupedModes.map((item) => item.candidateTelemetry[0].queryFamilyId)).size, 1);
assert.equal(new Set(groupedModes.map((item) => item.candidateTelemetry[0].evaluationGroupId)).size, 1);
assert.equal(groupedModes[0].queryFamilyId, queryFamilyIdFor(normalizeSimulationQuery("同一 query family 的生日对照")));
assert.equal(
  queryFamilyIdFor("  同一 query family 的生日对照 "),
  queryFamilyIdFor("同一 query family 的生日对照"),
);
assert.equal(groupedModes[0].evaluationGroupId, evaluationGroupIdFor({
  queryFamilyId: groupedModes[0].queryFamilyId,
  groupKey: "manual-simulation:batch-group-contract",
}));
assert.deepEqual(EVALUATION_GROUPING_CONTRACT.holdoutUnit, ["query_family_id", "evaluation_group_id"]);
assert.match(EVALUATION_GROUPING_CONTRACT.holdoutRule, /split neither/);
assert.equal(EVALUATION_GROUPING_CONTRACT.excludes.includes("reranker_score"), true);

localStorageValues.set("serein.basement.recall-simulation-training.v1", JSON.stringify({
  schemaVersion: 1,
  labels: [{
    id: "legacy-label",
    query: "legacy localStorage label",
    expectedAction: "skip",
    group: "manual-simulation:legacy-batch",
  }],
}));
assert.equal(readRecallSimulationTrainingLabels()[0].id, "legacy-label");
const legacyUpdated = upsertRecallSimulationTrainingLabel({
  query: "legacy localStorage label",
  expectedAction: "skip",
  observedAction: "skip",
  ablationMode: "normal",
});
assert.equal(legacyUpdated.status, "updated");
assert.equal(legacyUpdated.label.id, "legacy-label");
assert.ok(legacyUpdated.label.queryFamilyId);
assert.ok(legacyUpdated.label.evaluationGroupId);
assert.equal(JSON.parse(localStorageValues.get("serein.basement.recall-simulation-training.v1")).schemaVersion, 3);

for (const mode of ["normal", "without_cues", "without_embedding"]) {
  const modeLabel = createRecallSimulationTrainingLabel({
    query: `mode ${mode}`,
    expectedAction: "skip",
    ablationMode: mode,
    candidateTelemetry: [{ ...candidate, bucket_id: `scene-${mode}` }],
  }, { id: `manual-${mode}`, batchId: "batch-modes", now: "2026-08-06T09:01:00.000Z" });
  assert.equal(modeLabel.ablationMode, mode);
  assert.equal(modeLabel.candidateTelemetry[0].ablationMode, mode);
}

const telemetryItem = {
  id: "hook-with-shadow",
  query: "餐桌上的巨型水果",
  queryAvailable: true,
  observedAction: "recall",
  route: "recall_needed",
  sessionId: "s-telemetry",
  createdAt: "2026-08-06T09:02:00Z",
  source: "hook",
  injected: [{ id: "scene-fruit", title: "not exported", scoreValue: 0.42 }],
  candidateTelemetry: [candidate],
};
const historicalItem = {
  id: "hook-without-shadow",
  query: "历史候选没有 telemetry",
  queryAvailable: true,
  observedAction: "recall",
  route: "recall_needed",
  sessionId: "s-history",
  createdAt: "2026-08-06T09:03:00Z",
  source: "hook",
  injected: [{ id: "scene-old", title: "not exported", scoreValue: 0.88 }],
};
const staleItem = {
  ...telemetryItem,
  id: "hook-stale",
  query: "旧 replay candidate",
  candidateTelemetry: [{ ...candidate, bucket_id: "scene-stale", replay_status: "stale" }],
  injected: [{ id: "scene-stale", title: "not exported", scoreValue: 0.77 }],
};
const exported = buildRecallObservationTrainingExport(
  [telemetryItem, historicalItem, staleItem],
  {
    "hook-with-shadow": { verdict: "correct", candidateReviews: { "scene-fruit": "core" }, updatedAt: "2026-08-06T09:04:00Z" },
    "hook-without-shadow": { verdict: "correct", candidateReviews: { "scene-old": "weak" }, updatedAt: "2026-08-06T09:04:00Z" },
    "hook-stale": { verdict: "correct", candidateReviews: { "scene-stale": "irrelevant" }, updatedAt: "2026-08-06T09:04:00Z" },
  },
  "2026-08-06T09:05:00Z",
  [normalLabel],
);

assert.equal(exported.schema_version, 6);
assert.equal(exported.dataset_kind, "recall_observation_and_reranker_calibration");
assert.equal(exported.calibration_rows.length, 4);
const available = exported.calibration_rows.find((row) => row.observation_id === "hook-with-shadow");
assert.equal(available.calibration_available, true);
assert.equal(available.shadow_score, 0.91);
assert.equal(available.relevance, "core");
assert.equal(available.reranker_shadow.decision_applied, false);
assert.equal(available.final_admission_source, "pending_reranker_shadow_route_guard");
assert.equal(available.reranker_shadow.called_false_reason, null);
assert.deepEqual(available.candidate_sources, ["body_semantic", "cue_semantic", "cue_lexical", "title_anchor"]);
assert.equal(available.evidence.status, "unknown");
assert.equal(available.source_telemetry.body_semantic.score, 0.63);
assert.equal(available.source_telemetry.cue_semantic.score, 0.77);
assert.equal(available.source_telemetry.cue_semantic.role, "candidate_only");
assert.equal(available.source_telemetry.cue_lexical.role, "direct_evidence");
assert.equal(available.source_telemetry.title_anchor.matched, true);
assert.equal(Object.hasOwn(available.source_telemetry.title_anchor, "terms"), false);
const unavailable = exported.calibration_rows.find((row) => row.observation_id === "hook-without-shadow");
assert.equal(unavailable.calibration_available, false);
assert.equal(unavailable.unavailable_reason, "telemetry_unavailable");
assert.equal(unavailable.shadow_score, null);
assert.equal(unavailable.legacy_observed_score, 0.88);
const stale = exported.calibration_rows.find((row) => row.observation_id === "hook-stale");
assert.equal(stale.calibration_available, false);
assert.equal(stale.unavailable_reason, "stale");
assert.equal(stale.shadow_score, 0.91);
const notCalledItem = {
  id: "hook-shadow-not-called",
  query: "生日快乐！我买了蛋糕",
  queryAvailable: true,
  observedAction: "recall",
  route: "recall_needed",
  sessionId: "s-not-called",
  createdAt: "2026-08-06T09:03:30Z",
  source: "hook",
  injected: [{ id: "scene-cake", title: "not exported", scoreValue: 0.52 }],
  candidateTelemetry: [{
    ...candidate,
    bucket_id: "scene-cake",
    final_admission_source: "pending_reranker_shadow_route_guard",
    reranker_shadow: {
      called: false,
      score: null,
      model: "Qwen/Qwen3-Reranker-4B",
      decision_applied: false,
      status: "pending_reranker_shadow_route_guard",
      evidence_status: "unknown",
      evidence_count: 0,
    },
  }],
};
const exportedWithNotCalled = buildRecallObservationTrainingExport(
  [notCalledItem],
  {
    "hook-shadow-not-called": {
      verdict: "correct",
      candidateReviews: { "scene-cake": "weak" },
      updatedAt: "2026-08-06T09:04:00Z",
    },
  },
  "2026-08-06T09:05:00Z",
);
const notCalled = exportedWithNotCalled.calibration_rows[0];
assert.equal(notCalled.calibration_available, false);
assert.equal(notCalled.unavailable_reason, "shadow_not_called");
assert.equal(notCalled.final_admission_source, "pending_reranker_shadow_route_guard");
assert.equal(notCalled.reranker_shadow.called, false);
assert.equal(notCalled.reranker_shadow.called_false_reason, "pending_reranker_shadow_route_guard");
const manualOnly = exported.calibration_rows.find((row) => row.observation_id === normalLabel.observationId);
assert.equal(manualOnly.calibration_available, true);
assert.equal(manualOnly.unavailable_reason, null);
assert.equal(manualOnly.relevance, "core");
assert.equal(manualOnly.ablation_mode, "normal");
assert.equal(exported.telemetry_coverage.calibration_available, 2);
assert.equal(exported.telemetry_coverage.unavailable_reasons.telemetry_unavailable, 1);
assert.equal(exported.telemetry_coverage.unavailable_reasons.candidate_judgment_missing || 0, 0);
assert.equal(exported.telemetry_coverage.unavailable_reasons.stale, 1);
for (const row of exported.calibration_rows) {
  assert.equal(Object.hasOwn(row, "title"), false);
  assert.equal(Object.hasOwn(row, "body"), false);
  assert.equal(Object.hasOwn(row, "matched_cues"), false);
  assert.equal(Object.hasOwn(row, "prompt"), false);
  assert.equal(Object.hasOwn(row, "evidence_source_text"), false);
}
const forbiddenExportKeys = new Set([
  "body",
  "scene_body",
  "cues",
  "scene_cues",
  "matched_cues",
  "evidence_source_text",
  "prompt",
  "developer_context",
  "additional_context",
  "injected_body",
]);
function assertNoForbiddenExportKeys(value) {
  if (!value || typeof value !== "object") return;
  if (Array.isArray(value)) {
    value.forEach(assertNoForbiddenExportKeys);
    return;
  }
  for (const [key, child] of Object.entries(value)) {
    assert.equal(forbiddenExportKeys.has(key), false, `forbidden export key: ${key}`);
    assertNoForbiddenExportKeys(child);
  }
}
assertNoForbiddenExportKeys(exported);

const birthdayExport = buildRecallObservationTrainingExport(
  [],
  {},
  "2026-08-06T10:00:00Z",
  birthdaySimulationRegressionFixtures.map((fixture) => ({
    id: fixture.id,
    observationId: fixture.id,
    query: fixture.query,
    ablationMode: fixture.ablationMode,
    group: `manual-simulation:${birthdaySimulationBatchId}`,
    groupBasis: "manual_simulation_batch",
    simulationTelemetry: fixtureSimulationTelemetry(fixture),
    candidateTelemetry: fixtureCandidateTelemetry(fixture),
  })),
);
assert.equal(birthdayExport.observations.length, birthdaySimulationRegressionFixtures.length);
assert.equal(birthdayExport.telemetry_coverage.simulation_telemetry_rows, birthdaySimulationRegressionFixtures.length);
assert.equal(birthdayExport.telemetry_coverage.simulation_telemetry_missing, 0);
assert.equal(birthdayExport.telemetry_coverage.candidate_telemetry_rows, 17);
assert.equal(birthdayExport.telemetry_coverage.calibration_rows, 17);
assert.equal(birthdayExport.telemetry_coverage.calibration_available, 0);
assert.equal(birthdayExport.telemetry_coverage.unavailable_reasons.candidate_judgment_missing, 17);
for (const fixture of birthdaySimulationRegressionFixtures) {
  const observation = birthdayExport.observations.find((item) => item.observation_id === fixture.id);
  assert.ok(observation);
  assert.equal(observation.query, fixture.query);
  assert.equal(observation.query_normalized, normalizeSimulationQuery(fixture.query));
  assert.equal(observation.ablation_mode, fixture.ablationMode);
  assert.equal(observation.simulation_telemetry.ablationMode, fixture.ablationMode);
  assert.ok(Object.hasOwn(observation.simulation_telemetry, "route"));
  assert.ok(Object.hasOwn(observation.simulation_telemetry, "budget"));
  assert.ok(Object.hasOwn(observation.simulation_telemetry, "sentinel"));
  assert.ok(Object.hasOwn(observation.simulation_telemetry.budget, "queryFacets"));
  assert.ok(Object.hasOwn(observation.simulation_telemetry.budget, "prototypePrior"));
  assert.ok(Object.hasOwn(observation.simulation_telemetry.budget, "anchorOverrideReasons"));
  assert.ok(Object.hasOwn(observation.simulation_telemetry.budget, "channels"));
  assert.equal(observation.simulation_telemetry.budget.rerank.decisionApplied, false);
  assert.equal(Object.hasOwn(observation, "production_router_examples"), false);
}
assert.equal(birthdayExport.telemetry_coverage.query_family_count, 4);
assert.equal(birthdayExport.telemetry_coverage.evaluation_group_count, 4);
for (const query of [
  "哥哥，生日快乐！我买了蛋糕",
  "生日快乐",
  "生日快乐！我买了蛋糕",
  "Haven生日快乐",
]) {
  const rows = birthdayExport.observations.filter((item) => item.query === query);
  assert.equal(new Set(rows.map((item) => item.query_family_id)).size, 1);
  assert.equal(new Set(rows.map((item) => item.evaluation_group_id)).size, 1);
  const observationIds = new Set(rows.map((item) => item.observation_id));
  for (const row of birthdayExport.calibration_rows.filter((item) => observationIds.has(item.observation_id))) {
    assert.equal(row.query_family_id, rows[0].query_family_id);
    assert.equal(row.evaluation_group_id, rows[0].evaluation_group_id);
  }
}
assert.equal(birthdayExport.evaluation_grouping.version, EVALUATION_GROUPING_CONTRACT.version);

assert.equal(birthdayLiveRawSchemaMapping.source_telemetry, null);
assert.equal(birthdayLiveRawSchemaMapping.telemetry_generated_at, null);
assert.equal(birthdayLiveRawSchemaMapping.called_false_reason, null);
assert.equal(birthdayLiveRawSchemaMapping.matched_cues, "matched_cues");
assert.equal(birthdayLiveRawSchemaMapping.title_anchor_terms, "title_anchor_terms");
assert.deepEqual(birthdayLiveRawSchemaMapping.privacyStrippedFields, ["matched_cues", "title_anchor_terms"]);
const birthdayCueLift = birthdayExport.calibration_rows.find(
  (row) => row.observation_id === "birthday-cake-without-embedding",
);
assert.deepEqual(birthdayCueLift.candidate_sources, ["cue_lexical"]);
assert.equal(birthdayCueLift.source_telemetry.cue_lexical.role, "direct_evidence");
assert.equal(birthdayCueLift.source_telemetry.body_semantic.score, 0);
assert.equal(birthdayCueLift.reranker_shadow.decision_applied, false);
assert.equal(birthdayCueLift.shadow_score, 0.864);
assert.equal(birthdayCueLift.telemetry_generated_at, null);
assert.equal(birthdayCueLift.unavailable_reason, "candidate_judgment_missing");
const birthdayNoCall = birthdayExport.calibration_rows.find(
  (row) => row.observation_id === "birthday-brother-cake-without-cues"
    && row.candidate_id === "scene_mig2_cfa28ed9b556cb5eab6e",
);
assert.equal(birthdayNoCall.reranker_shadow.called, false);
assert.equal(birthdayNoCall.reranker_shadow.called_false_reason, "ineligible_below_entry_floor");
const birthdayQ1 = birthdaySimulationRegressionFixtures.find((fixture) => fixture.id === "birthday-brother-cake-normal");
assert.equal(birthdayQ1.handoffObservation.provenance, "unverified_user_recollection");
assert.deepEqual(birthdayQ1.expected.injectedTitles, ["Haven的第一个生日蛋糕"]);
assert.equal(birthdayQ1.replayComparison, "unverified_user_recollection_current_replay_one");
const birthdayQ3 = birthdaySimulationRegressionFixtures.find((fixture) => fixture.id === "birthday-cake-normal");
assert.equal(birthdayQ3.handoffObservation.provenance, "unverified_user_recollection");
assert.equal(birthdayQ3.expected.rerankerCalled, true);
assert.equal(birthdayQ3.replayComparison, "unverified_user_recollection_shadow_not_called_current_replay_called");

const rawSchemaMapped = normalizeCandidateTelemetrySnapshot([{
  bucket_id: "live-schema-cake",
  rank: 1,
  candidate_sources: ["cue_lexical"],
  body_semantic_score: 0,
  cue_semantic: { status: "unavailable", role: "none" },
  cue_lexical_match: true,
  cue_lexical_role: "direct_evidence",
  title_anchor_match: false,
  floor_qualified: true,
  reranker_eligible: true,
  absolute_floor: 0.55,
  reranker_entry_floor: 0.5,
  final_admission_source: "selected_after_normal_admission",
  reranker_shadow: {
    called: true,
    score: 0.864,
    model: "Qwen/Qwen3-Reranker-4B",
    status: "scored_shadow_only",
    decision_applied: false,
    evidence_status: "unknown",
    evidence_count: 0,
  },
}], { observationId: "live-schema", query: "生日快乐！我买了蛋糕", ablationMode: "without_embedding" });
assert.equal(rawSchemaMapped[0].sourceTelemetry.cueLexical.role, "direct_evidence");
assert.equal(rawSchemaMapped[0].rerankerShadow.decisionApplied, false);

const noDuplicate = normalizeCandidateTelemetrySnapshot([
  candidate,
  { ...candidate, bucket_id: "scene-fruit" },
], { observationId: "obs", query: "q", ablationMode: "normal" });
assert.equal(noDuplicate.length, 1);
assert.equal(noDuplicate[0].calibrationKey, "obs::q::scene-fruit");

console.log("reranker shadow telemetry/export checks: PASS");
