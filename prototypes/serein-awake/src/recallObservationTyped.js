function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function finiteNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function normalizeTypedRecallObservation(semantic) {
  const raw = semantic?.typed_event_scene_observation;
  if (!raw || typeof raw !== "object") return null;
  const admission = raw.admission && typeof raw.admission === "object" ? raw.admission : {};
  const scope = raw.entity_scope && typeof raw.entity_scope === "object" ? raw.entity_scope : {};
  const scopeAnchor = scope.scope_anchor && typeof scope.scope_anchor === "object"
    ? scope.scope_anchor
    : {};
  const selectedRefs = asArray(raw.selected_refs).map(String);
  const selectedSet = new Set(selectedRefs);
  const candidateRows = asArray(raw.candidate_summaries).length
    ? asArray(raw.candidate_summaries)
    : asArray(admission.candidates);
  return {
    status: String(raw.status || "unknown"),
    reason: String(raw.reason || ""),
    candidateCount: Number.isFinite(Number(raw.candidate_count))
      ? Number(raw.candidate_count)
      : candidateRows.length,
    selectedRefs,
    candidates: candidateRows.map((candidate) => {
      const ref = String(candidate?.ref || "");
      return {
        ref,
        title: String(candidate?.title || ref || "未命名记忆"),
        ownerKind: String(candidate?.owner_kind || ""),
        candidateScore: finiteNumber(candidate?.candidate_score),
        rerankScore: finiteNumber(candidate?.rerank_score),
        disposition: String(candidate?.disposition || ""),
        reason: String(candidate?.reason || ""),
        selected: selectedSet.has(ref),
      };
    }),
    cards: asArray(raw.cards),
    admissionMode: String(admission.mode || ""),
    intent: String(admission.intent || scope.intent || ""),
    operator: String(admission.operator || scope.operator || ""),
    scopeLabel: String(scopeAnchor.title || scopeAnchor.arc_key || ""),
    timingMs: finiteNumber(raw.timing_ms),
    candidateTimingMs: finiteNumber(raw.candidate_timing_ms),
    actualInjectedIds: asArray(raw.actual_injected_ids).map(String),
    simulationOnly: raw.simulation_only === true,
    decisionApplied: raw.decision_applied === true,
    liveInjectionEnabled: raw.live_injection_enabled === true,
    runsAfterResponse: raw.runs_after_response === true,
  };
}

export function hasTypedExpectedMatch(item) {
  return item?.typedObservation?.status === "would_inject"
    && item.typedObservation.selectedRefs.length > 0;
}
