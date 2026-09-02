import assert from "node:assert/strict";
import {
  hasTypedExpectedMatch,
  normalizeTypedRecallObservation,
} from "../src/recallObservationTyped.js";

const normalized = normalizeTypedRecallObservation({
  typed_event_scene_observation: {
    status: "would_inject",
    selected_refs: ["event:event_chocolate"],
    candidate_count: 2,
    candidate_summaries: [
      {
        ref: "event:event_chocolate",
        owner_kind: "event",
        title: "讨论巧克蕾",
        candidate_score: 0.64,
        rerank_score: 0.92,
        disposition: "direct_evidence",
        reason: "reranker_direct_evidence",
      },
      {
        ref: "scene:scene_other",
        owner_kind: "scene",
        title: "无关 Scene",
        candidate_score: 0.31,
        rerank_score: 0.01,
        disposition: "reject",
        reason: "reranker_below_direct_threshold",
      },
    ],
    admission: { mode: "direct_evidence_rerank", intent: "entity_detail", operator: "none" },
    entity_scope: { scope_anchor: { arc_key: "topic:chocolate" } },
    timing_ms: 2310,
    candidate_timing_ms: 7,
    actual_injected_ids: ["legacy-scene"],
    simulation_only: true,
    decision_applied: false,
    live_injection_enabled: false,
    live_mode: "guarded_live",
    live_mode_gate: {
      mode: "guarded_live",
      guard_applied: true,
      guarded_live_allowed: true,
      reason: "memory_backed_entity_detail",
    },
    runs_after_response: true,
  },
});

assert.equal(normalized.status, "would_inject");
assert.equal(normalized.scopeLabel, "topic:chocolate");
assert.equal(normalized.candidates[0].selected, true);
assert.equal(normalized.candidates[1].selected, false);
assert.deepEqual(normalized.actualInjectedIds, ["legacy-scene"]);
assert.equal(normalized.candidateTimingMs, 7);
assert.equal(normalized.liveMode, "guarded_live");
assert.equal(normalized.guardApplied, true);
assert.equal(normalized.guardedLiveAllowed, true);
assert.equal(hasTypedExpectedMatch({ typedObservation: normalized }), true);

const guardedBlocked = normalizeTypedRecallObservation({
  typed_event_scene_observation: {
    status: "would_inject",
    selected_refs: ["scene:scene_other"],
    live_mode_gate: {
      mode: "guarded_live",
      guard_applied: true,
      guarded_live_allowed: false,
      reason: "global_semantic_without_guarded_intent",
    },
  },
});
assert.equal(hasTypedExpectedMatch({ typedObservation: guardedBlocked }), false);

const pending = normalizeTypedRecallObservation({
  typed_event_scene_observation: {
    status: "pending",
    candidate_count: 5,
    simulation_only: true,
    decision_applied: false,
  },
});
assert.equal(pending.status, "pending");
assert.equal(pending.candidateCount, 5);
assert.equal(hasTypedExpectedMatch({ typedObservation: pending }), false);
assert.equal(normalizeTypedRecallObservation({}), null);

console.log("TYPED_RECALL_OBSERVATION_UI_OK");
