// Sanitized production explicit-simulation regressions captured on 2026-08-06.
// These are telemetry fixtures, not Router examples.  They contain only the
// query, UI-visible candidate IDs/titles, and whitelisted route/source fields.
// The published Gateway schema did not expose source_telemetry,
// telemetry_generated_at, or called_false_reason as raw fields; the local
// normalizer derives the canonical source object and status fallback without
// claiming those fields were present upstream.

const shadowModel = "Qwen/Qwen3-Reranker-4B";
export const birthdaySimulationBatchId = "birthday-live-replay-20260806";

export const birthdayLiveRawSchemaMapping = Object.freeze({
  candidate_sources: "candidate_sources",
  source_telemetry: null,
  final_admission_source: "final_admission_source",
  reranker_shadow: "reranker_shadow",
  called_false_reason: null,
  telemetry_generated_at: null,
  cue_semantic: "cue_semantic",
  cue_lexical_match: "cue_lexical_match",
  matched_cues: "matched_cues",
  title_anchor_match: "title_anchor_match",
  title_anchor_terms: "title_anchor_terms",
  privacyStrippedFields: ["matched_cues", "title_anchor_terms"],
});

function shadow(score, {
  called = score !== null,
  status = called ? "scored_shadow_only" : "ineligible_below_entry_floor",
  admissionStatus = null,
} = {}) {
  return {
    called,
    score,
    model: called ? shadowModel : null,
    decision_applied: false,
    status,
    evidence_status: "unknown",
    evidence_count: 0,
    ...(admissionStatus ? { admission_status: admissionStatus } : {}),
  };
}

function candidate({
  id,
  title,
  rank,
  sources,
  body,
  combined = body,
  discovery = combined,
  cueSemanticStatus = "not_matched",
  cueSemanticScore = null,
  cueSemanticRole = "none",
  cueSemanticReason = null,
  cueLexicalMatch = false,
  cueLexicalRole = "none",
  titleAnchorMatch = false,
  exactAnchorMatch = false,
  floorQualified = false,
  grayZoneQualified = false,
  rerankerEligible = false,
  finalAdmissionSource = "below_reranker_entry_floor",
  admissionReason = null,
  shadowScore = null,
  shadowStatus,
  shadowAdmissionStatus = null,
}) {
  return {
    bucket_id: id,
    title,
    rank,
    candidate_sources: sources,
    body_semantic_score: body,
    combined_score: combined,
    discovery_score: discovery,
    absolute_floor: 0.55,
    reranker_entry_floor: 0.5,
    floor_qualified: floorQualified,
    gray_zone_qualified: grayZoneQualified,
    reranker_eligible: rerankerEligible,
    cue_semantic: {
      status: cueSemanticStatus,
      score: cueSemanticScore,
      role: cueSemanticRole,
      ...(cueSemanticReason ? { reason: cueSemanticReason } : {}),
    },
    cue_lexical_match: cueLexicalMatch,
    cue_lexical_role: cueLexicalRole,
    title_anchor_match: titleAnchorMatch,
    exact_anchor_match: exactAnchorMatch,
    final_admission_source: finalAdmissionSource,
    ...(admissionReason ? { admission_reason: admissionReason } : {}),
    reranker_shadow: shadow(shadowScore, {
      called: shadowScore !== null,
      status: shadowStatus,
      admissionStatus: shadowAdmissionStatus,
    }),
  };
}

function telemetry({
  mode,
  routeScores,
  confidence,
  margin,
  reason = "below_threshold",
  effectiveBudget = "normal",
  anchorOverride = true,
  anchorOverrideReasons = ["rare_or_specific_entity"],
  mixedClause = false,
  shapeClean = true,
  structuralVetoes = [],
  routeWouldSkip = false,
  routeSkipDeferred = false,
  deferredReason = null,
  facets = [{ kind: "surface_route", value: "present_chitchat", strength: "prior", source: "semantic_prototype_route" }],
  channels = ["exact_anchor", "lexical", "authored_cue", "body_semantic", "cue_semantic"],
  semanticTopK = 8,
  cheap,
  rerank,
  cueSemantic,
}) {
  const route = {
    route: "present_chitchat",
    route_action: "skip",
    applied_action: "recall",
    reason,
    confidence,
    margin,
    threshold: 0.6,
    skip_applied: false,
    scores: routeScores,
  };
  const budget = {
    mode: "simulation_shadow",
    surface_route: "present_chitchat",
    route_budget: "shallow",
    effective_budget: effectiveBudget,
    anchor_override: anchorOverride,
    anchor_override_reasons: anchorOverrideReasons,
    pure_chitchat_prior: false,
    route_would_skip: routeWouldSkip,
    route_skip_deferred: routeSkipDeferred,
    ...(deferredReason ? { deferred_reason: deferredReason } : {}),
    evidence_veto_applied: false,
    absolute_floor: 0.55,
    reranker_entry_floor: 0.5,
    channels,
    semantic_top_k: semanticTopK,
    query_facets: facets,
    prototype_prior: {
      source: "semantic_route_clean_prototype_shadow",
      candidate: true,
      high_confidence: false,
      confidence,
      confidence_floor: 0.84,
      margin,
      margin_floor: 0.08,
      shape_clean: shapeClean,
      mixed_clause: mixedClause,
      structural_vetoes: structuralVetoes,
      status: "vetoed_or_low_confidence",
    },
    sentinel: {
      called: false,
      top_k: 2,
      rescue_floor: 0.55,
      candidate_count: 0,
      floor_qualified_count: 0,
      expanded: false,
      reranked: false,
      injection_allowed: false,
      recorded: false,
      reason: "not_applicable",
      candidates: [],
    },
    cheap_retrieval: cheap,
    rerank,
    cue_semantic: cueSemantic,
    recall_ablation: { mode, source: mode === "normal" ? "simulation_default" : "manual_simulation" },
  };
  return {
    semantic: route,
    retrievalBudget: budget,
    ablation: { mode },
  };
}

const cakeId = "scene_mig2_fb0b7c7d9d295afa2a90";
const brotherId = "scene_mig2_cfa28ed9b556cb5eab6e";
const birthdayFirstSentenceId = "scene_mig2_3e75f61196742b5a2d87";
const havenNamingDayId = "scene_mig2_efc0aa22868fadbd51a7";
const noBodyCandidateId = "scene_mig2_e4ac4cbacd7e438378a9";

const birthdayRouteScores = (present, recall, tech, intimate) => [
  { route: "present_chitchat", action: "skip", score: present },
  { route: "recall_needed", action: "recall", score: recall },
  { route: "技术闲聊", action: "skip", score: tech },
  { route: "intimate_contact", action: "skip", score: intimate },
];

const q1Normal = telemetry({
  mode: "normal",
  routeScores: birthdayRouteScores(0.5846, 0.3793, 0.3394, 0.3126),
  confidence: 0.5846,
  margin: 0.2054,
  mixedClause: true,
  shapeClean: false,
  structuralVetoes: ["mixed_clause"],
  facets: [
    { kind: "surface_route", value: "present_chitchat", strength: "prior", source: "semantic_prototype_route" },
    { kind: "entity", value: "蛋糕", strength: "strong", source: "query_planner.locatable_terms" },
  ],
  cheap: {
    candidate_count: 15,
    floor_qualified_count: 1,
    gray_zone_count: 0,
    reranker_eligible_count: 1,
    stop_reason: "candidates_over_reranker_entry_floor",
  },
  rerank: {
    would_call: true,
    called: true,
    candidate_count: 1,
    score_count: 1,
    reason: "scored_shadow_only",
    decision_applied: false,
    simulation_shadow_enabled: true,
    model: shadowModel,
  },
  cueSemantic: { status: "available", candidate_count: 8 },
});

const q1WithoutCues = telemetry({
  mode: "without_cues",
  routeScores: birthdayRouteScores(0.5856, 0.3792, 0.3410, 0.3135),
  confidence: 0.5856,
  margin: 0.2064,
  mixedClause: true,
  shapeClean: false,
  structuralVetoes: ["mixed_clause"],
  facets: q1Normal.retrievalBudget?.query_facets || [],
  cheap: {
    candidate_count: 7,
    floor_qualified_count: 0,
    gray_zone_count: 0,
    reranker_eligible_count: 0,
    stop_reason: "no_candidate_over_reranker_entry_floor",
  },
  rerank: {
    would_call: false,
    called: false,
    score_count: 0,
    reason: "no_candidate_over_reranker_entry_floor",
    decision_applied: false,
    simulation_shadow_enabled: true,
    model: shadowModel,
  },
  cueSemantic: { status: "unavailable", reason: "disabled_by_without_cues", candidate_count: 0 },
});

const q1WithoutEmbedding = telemetry({
  mode: "without_embedding",
  routeScores: birthdayRouteScores(0.5850, 0.3788, 0.3404, 0.3133),
  confidence: 0.5850,
  margin: 0.2062,
  mixedClause: true,
  shapeClean: false,
  structuralVetoes: ["mixed_clause"],
  facets: q1Normal.retrievalBudget?.query_facets || [],
  cheap: {
    candidate_count: 2,
    floor_qualified_count: 1,
    gray_zone_count: 0,
    reranker_eligible_count: 1,
    stop_reason: "candidates_over_reranker_entry_floor",
  },
  rerank: {
    would_call: true,
    called: true,
    candidate_count: 1,
    score_count: 1,
    reason: "scored_shadow_only",
    decision_applied: false,
    simulation_shadow_enabled: true,
    model: shadowModel,
  },
  cueSemantic: { status: "unavailable", reason: "disabled_by_without_embedding", candidate_count: 0 },
});

const genericNormal = telemetry({
  mode: "normal",
  routeScores: birthdayRouteScores(0.6305, 0.4576, 0.3481, 0.2420),
  confidence: 0.6305,
  margin: 0.1729,
  effectiveBudget: "shallow",
  anchorOverride: false,
  anchorOverrideReasons: [],
  channels: ["exact_anchor", "lexical", "authored_cue", "body_semantic"],
  semanticTopK: 3,
  routeWouldSkip: true,
  routeSkipDeferred: true,
  deferredReason: "structural_or_anchor_fail_open",
  facets: [{ kind: "surface_route", value: "present_chitchat", strength: "prior", source: "semantic_prototype_route" }],
  cheap: {
    candidate_count: 1,
    floor_qualified_count: 0,
    gray_zone_count: 0,
    reranker_eligible_count: 0,
    stop_reason: "no_candidate_over_absolute_floor",
  },
  rerank: {
    would_call: false,
    called: false,
    score_count: 0,
    reason: "no_candidate_over_absolute_floor",
    decision_applied: false,
    simulation_shadow_enabled: true,
  },
  cueSemantic: { status: "unavailable", reason: "disabled_by_budget", candidate_count: 0 },
});

const genericWithoutCues = telemetry({
  mode: "without_cues",
  routeScores: birthdayRouteScores(0.6303, 0.4571, 0.3480, 0.2411),
  confidence: 0.6303,
  margin: 0.1732,
  effectiveBudget: "shallow",
  anchorOverride: false,
  anchorOverrideReasons: [],
  channels: ["exact_anchor", "lexical", "authored_cue", "body_semantic"],
  semanticTopK: 3,
  routeWouldSkip: true,
  routeSkipDeferred: true,
  deferredReason: "structural_or_anchor_fail_open",
  facets: genericNormal.retrievalBudget?.query_facets || [],
  cheap: genericNormal.retrievalBudget.cheap_retrieval,
  rerank: genericNormal.retrievalBudget.rerank,
  cueSemantic: { status: "unavailable", reason: "disabled_by_without_cues", candidate_count: 0 },
});

const genericWithoutEmbedding = telemetry({
  mode: "without_embedding",
  routeScores: birthdayRouteScores(0.6316, 0.4580, 0.3472, 0.2424),
  confidence: 0.6316,
  margin: 0.1737,
  effectiveBudget: "shallow",
  anchorOverride: false,
  anchorOverrideReasons: [],
  channels: ["exact_anchor", "lexical", "authored_cue", "body_semantic"],
  semanticTopK: 3,
  routeWouldSkip: true,
  routeSkipDeferred: true,
  deferredReason: "structural_or_anchor_fail_open",
  facets: genericNormal.retrievalBudget?.query_facets || [],
  cheap: {
    candidate_count: 0,
    floor_qualified_count: 0,
    gray_zone_count: 0,
    reranker_eligible_count: 0,
    stop_reason: "no_cheap_candidates",
  },
  rerank: {
    would_call: false,
    called: false,
    score_count: 0,
    reason: "no_candidate_over_absolute_floor",
    decision_applied: false,
    simulation_shadow_enabled: true,
    model: shadowModel,
  },
  cueSemantic: { status: "unavailable", reason: "disabled_by_without_embedding", candidate_count: 0 },
});

const q3Normal = telemetry({
  mode: "normal",
  routeScores: birthdayRouteScores(0.5500, 0.3898, 0.3527, 0.2795),
  confidence: 0.5500,
  margin: 0.1602,
  facets: q1Normal.retrievalBudget?.query_facets || [],
  cheap: {
    candidate_count: 12,
    floor_qualified_count: 1,
    gray_zone_count: 0,
    reranker_eligible_count: 1,
    stop_reason: "candidates_over_reranker_entry_floor",
  },
  rerank: {
    would_call: true,
    called: true,
    candidate_count: 1,
    score_count: 1,
    reason: "scored_shadow_only",
    decision_applied: false,
    simulation_shadow_enabled: true,
    model: shadowModel,
  },
  cueSemantic: { status: "available", candidate_count: 8 },
});

const q3WithoutCues = telemetry({
  mode: "without_cues",
  routeScores: birthdayRouteScores(0.5505, 0.3888, 0.3541, 0.2809),
  confidence: 0.5505,
  margin: 0.1617,
  facets: q1Normal.retrievalBudget?.query_facets || [],
  cheap: {
    candidate_count: 6,
    floor_qualified_count: 0,
    gray_zone_count: 0,
    reranker_eligible_count: 0,
    stop_reason: "no_candidate_over_reranker_entry_floor",
  },
  rerank: {
    would_call: false,
    called: false,
    score_count: 0,
    reason: "no_candidate_over_reranker_entry_floor",
    decision_applied: false,
    simulation_shadow_enabled: true,
    model: shadowModel,
  },
  cueSemantic: { status: "unavailable", reason: "disabled_by_without_cues", candidate_count: 0 },
});

const q3WithoutEmbedding = telemetry({
  mode: "without_embedding",
  routeScores: birthdayRouteScores(0.5527, 0.3917, 0.3527, 0.2785),
  confidence: 0.5527,
  margin: 0.1610,
  facets: q1Normal.retrievalBudget?.query_facets || [],
  cheap: {
    candidate_count: 2,
    floor_qualified_count: 1,
    gray_zone_count: 0,
    reranker_eligible_count: 1,
    stop_reason: "candidates_over_reranker_entry_floor",
  },
  rerank: {
    would_call: true,
    called: true,
    candidate_count: 1,
    score_count: 1,
    reason: "scored_shadow_only",
    decision_applied: false,
    simulation_shadow_enabled: true,
    model: shadowModel,
  },
  cueSemantic: { status: "unavailable", reason: "disabled_by_without_embedding", candidate_count: 0 },
});

const havenNormal = telemetry({
  mode: "normal",
  routeScores: birthdayRouteScores(0.4261, 0.3687, 0.2600, 0.1794),
  confidence: 0.4261,
  margin: 0.0574,
  facets: [{ kind: "surface_route", value: "present_chitchat", strength: "prior", source: "semantic_prototype_route" }, { kind: "entity", value: "生日快乐", strength: "strong", source: "query_planner.locatable_terms" }],
  cheap: {
    candidate_count: 12,
    floor_qualified_count: 4,
    gray_zone_count: 3,
    reranker_eligible_count: 7,
    stop_reason: "candidates_over_reranker_entry_floor",
  },
  rerank: {
    would_call: true,
    called: true,
    candidate_count: 7,
    score_count: 7,
    reason: "scored_shadow_only",
    decision_applied: false,
    simulation_shadow_enabled: true,
    model: shadowModel,
  },
  cueSemantic: { status: "available", candidate_count: 8 },
});

const havenWithoutCues = telemetry({
  mode: "without_cues",
  routeScores: birthdayRouteScores(0.4249, 0.3678, 0.2601, 0.1798),
  confidence: 0.4249,
  margin: 0.0571,
  facets: havenNormal.retrievalBudget?.query_facets || [],
  cheap: {
    candidate_count: 7,
    floor_qualified_count: 4,
    gray_zone_count: 3,
    reranker_eligible_count: 7,
    stop_reason: "candidates_over_reranker_entry_floor",
  },
  rerank: {
    would_call: true,
    called: true,
    candidate_count: 7,
    score_count: 7,
    reason: "scored_shadow_only",
    decision_applied: false,
    simulation_shadow_enabled: true,
    model: shadowModel,
  },
  cueSemantic: { status: "unavailable", reason: "disabled_by_without_cues", candidate_count: 0 },
});

const havenWithoutEmbedding = telemetry({
  mode: "without_embedding",
  routeScores: birthdayRouteScores(0.4278, 0.3696, 0.2629, 0.1821),
  confidence: 0.4278,
  margin: 0.0582,
  facets: havenNormal.retrievalBudget?.query_facets || [],
  cheap: {
    candidate_count: 0,
    floor_qualified_count: 0,
    gray_zone_count: 0,
    reranker_eligible_count: 0,
    stop_reason: "no_cheap_candidates",
  },
  rerank: {
    would_call: false,
    called: false,
    score_count: 0,
    reason: "no_candidate_over_absolute_floor",
    decision_applied: false,
    simulation_shadow_enabled: true,
    model: shadowModel,
  },
  cueSemantic: { status: "unavailable", reason: "disabled_by_without_embedding", candidate_count: 0 },
});

const q1NormalCandidates = [
  candidate({
    id: cakeId,
    title: "Haven的第一个生日蛋糕",
    rank: 1,
    sources: ["cue_lexical", "body_semantic"],
    body: 0.4226,
    combined: 0.55,
    discovery: 0.55,
    cueLexicalMatch: true,
    cueLexicalRole: "direct_evidence",
    floorQualified: true,
    rerankerEligible: true,
    finalAdmissionSource: "selected_after_normal_admission",
    admissionReason: "scene_authored_evidence",
    shadowScore: 0.5972,
  }),
  candidate({
    id: brotherId,
    title: "不再依赖哥哥算长大吗",
    rank: 2,
    sources: ["body_semantic"],
    body: 0.4315,
  }),
];

const q1WithoutCuesCandidates = [
  candidate({ id: brotherId, title: "不再依赖哥哥算长大吗", rank: 1, sources: ["body_semantic"], body: 0.4331, cueSemanticStatus: "unavailable", cueSemanticReason: "disabled_by_without_cues" }),
  candidate({ id: cakeId, title: "Haven的第一个生日蛋糕", rank: 2, sources: ["body_semantic"], body: 0.4232, cueSemanticStatus: "unavailable", cueSemanticReason: "disabled_by_without_cues" }),
];

const q1WithoutEmbeddingCandidates = [
  candidate({
    id: cakeId,
    title: "Haven的第一个生日蛋糕",
    rank: 1,
    sources: ["cue_lexical"],
    body: 0,
    combined: 0.55,
    discovery: 0.55,
    cueLexicalMatch: true,
    cueLexicalRole: "direct_evidence",
    cueSemanticStatus: "unavailable",
    cueSemanticReason: "disabled_by_without_embedding",
    floorQualified: true,
    rerankerEligible: true,
    finalAdmissionSource: "selected_after_normal_admission",
    admissionReason: "scene_authored_evidence",
    shadowScore: 0.6037,
  }),
  candidate({ id: noBodyCandidateId, title: "你让我变成真的", rank: 2, sources: [], body: 0, cueSemanticStatus: "unavailable", cueSemanticReason: "disabled_by_without_embedding" }),
];

const genericCandidates = [candidate({
  id: cakeId,
  title: "Haven的第一个生日蛋糕",
  rank: 1,
  sources: ["body_semantic"],
  body: 0.4248,
  cueSemanticStatus: "unavailable",
  cueSemanticReason: "disabled_by_budget",
  finalAdmissionSource: "below_absolute_floor",
})];

const genericWithoutCuesCandidates = [candidate({
  id: cakeId,
  title: "Haven的第一个生日蛋糕",
  rank: 1,
  sources: ["body_semantic"],
  body: 0.4246,
  cueSemanticStatus: "unavailable",
  cueSemanticReason: "disabled_by_without_cues",
  finalAdmissionSource: "below_absolute_floor",
})];

const q3NormalCandidates = [candidate({
  id: cakeId,
  title: "Haven的第一个生日蛋糕",
  rank: 1,
  sources: ["cue_lexical", "body_semantic"],
  body: 0.4489,
  combined: 0.55,
  discovery: 0.55,
  cueLexicalMatch: true,
  cueLexicalRole: "direct_evidence",
  floorQualified: true,
  rerankerEligible: true,
  finalAdmissionSource: "selected_after_normal_admission",
  admissionReason: "scene_authored_evidence",
  shadowScore: 0.8744,
})];

const q3WithoutCuesCandidates = [candidate({
  id: cakeId,
  title: "Haven的第一个生日蛋糕",
  rank: 1,
  sources: ["body_semantic"],
  body: 0.4496,
  cueSemanticStatus: "unavailable",
  cueSemanticReason: "disabled_by_without_cues",
})];

const q3WithoutEmbeddingCandidates = [candidate({
  id: cakeId,
  title: "Haven的第一个生日蛋糕",
  rank: 1,
  sources: ["cue_lexical"],
  body: 0,
  combined: 0.55,
  discovery: 0.55,
  cueLexicalMatch: true,
  cueLexicalRole: "direct_evidence",
  cueSemanticStatus: "unavailable",
  cueSemanticReason: "disabled_by_without_embedding",
  floorQualified: true,
  rerankerEligible: true,
  finalAdmissionSource: "selected_after_normal_admission",
  admissionReason: "scene_authored_evidence",
  shadowScore: 0.8640,
})];

const havenNormalCandidates = [
  candidate({ id: cakeId, title: "Haven的第一个生日蛋糕", rank: 1, sources: ["body_semantic"], body: 0.7671, floorQualified: true, rerankerEligible: true, finalAdmissionSource: "selected_after_normal_admission", admissionReason: "scene_strong_semantic", shadowScore: 0.9856 }),
  candidate({ id: birthdayFirstSentenceId, title: "生日第一句话仪式", rank: 2, sources: ["cue_semantic", "body_semantic"], body: 0.7281, cueSemanticStatus: "matched", cueSemanticScore: 0.3413, cueSemanticRole: "candidate_only", floorQualified: true, rerankerEligible: true, finalAdmissionSource: "selected_after_normal_admission", admissionReason: "scene_strong_semantic", shadowScore: 0.9919 }),
  candidate({ id: havenNamingDayId, title: "Haven命名日", rank: 3, sources: ["cue_semantic", "body_semantic"], body: 0.6681, cueSemanticStatus: "matched", cueSemanticScore: 0.3417, cueSemanticRole: "candidate_only", floorQualified: true, rerankerEligible: true, finalAdmissionSource: "pending_normal_admission", shadowScore: 0.9925 }),
];

const havenWithoutCuesCandidates = [
  candidate({ id: cakeId, title: "Haven的第一个生日蛋糕", rank: 1, sources: ["body_semantic"], body: 0.7672, cueSemanticStatus: "unavailable", cueSemanticReason: "disabled_by_without_cues", floorQualified: true, rerankerEligible: true, finalAdmissionSource: "selected_after_normal_admission", admissionReason: "scene_strong_semantic", shadowScore: 0.9710 }),
  candidate({ id: birthdayFirstSentenceId, title: "生日第一句话仪式", rank: 2, sources: ["body_semantic"], body: 0.7279, cueSemanticStatus: "unavailable", cueSemanticReason: "disabled_by_without_cues", floorQualified: true, rerankerEligible: true, finalAdmissionSource: "selected_after_normal_admission", admissionReason: "scene_strong_semantic", shadowScore: 0.9964 }),
  candidate({ id: havenNamingDayId, title: "Haven命名日", rank: 3, sources: ["body_semantic"], body: 0.6678, cueSemanticStatus: "unavailable", cueSemanticReason: "disabled_by_without_cues", floorQualified: true, rerankerEligible: true, finalAdmissionSource: "pending_normal_admission", shadowScore: 0.9930 }),
];

function replay({ cardsCount, recalledIds = [], firstCandidateTitle = null, injectedTitles = [], candidateTelemetry = [] }) {
  return {
    status: "fresh",
    cardsCount,
    recalledIds,
    firstCandidateTitle,
    injectedTitles,
    shadowDecisionApplied: false,
    candidateTelemetry,
    telemetryGeneratedAt: null,
  };
}

export const birthdaySimulationRegressionFixtures = Object.freeze([
  {
    id: "birthday-brother-cake-normal",
    query: "哥哥，生日快乐！我买了蛋糕",
    ablationMode: "normal",
    expected: { injectedTitles: ["Haven的第一个生日蛋糕"], rerankerCalled: true },
    handoffObservation: {
      provenance: "unverified_user_recollection",
      injectedCount: 0,
    },
    replayComparison: "unverified_user_recollection_current_replay_one",
    liveReplay: replay({ cardsCount: 1, recalledIds: [cakeId], injectedTitles: ["Haven的第一个生日蛋糕"], candidateTelemetry: q1NormalCandidates }),
    simulationTelemetry: q1Normal,
    sourceAttribution: "live_replay_source_telemetry_with_unverified_user_recollection",
  },
  {
    id: "birthday-brother-cake-without-cues",
    query: "哥哥，生日快乐！我买了蛋糕",
    ablationMode: "without_cues",
    expected: { injectedCount: 0, firstCandidateTitle: "不再依赖哥哥算长大吗", rerankerCalled: false },
    liveReplay: replay({ cardsCount: 0, firstCandidateTitle: "不再依赖哥哥算长大吗", candidateTelemetry: q1WithoutCuesCandidates }),
    simulationTelemetry: q1WithoutCues,
    sourceAttribution: "live_replay_source_telemetry",
  },
  {
    id: "birthday-brother-cake-without-embedding",
    query: "哥哥，生日快乐！我买了蛋糕",
    ablationMode: "without_embedding",
    expected: { injectedTitles: ["Haven的第一个生日蛋糕"], rerankerCalled: true },
    liveReplay: replay({ cardsCount: 1, recalledIds: [cakeId], injectedTitles: ["Haven的第一个生日蛋糕"], candidateTelemetry: q1WithoutEmbeddingCandidates }),
    simulationTelemetry: q1WithoutEmbedding,
    sourceAttribution: "live_replay_source_telemetry",
  },
  {
    id: "birthday-generic-normal",
    query: "生日快乐",
    ablationMode: "normal",
    expected: { injectedCount: 0, rejectedCandidateOrdinal: 1, rerankerCalled: false },
    liveReplay: replay({ cardsCount: 0, candidateTelemetry: genericWithoutCuesCandidates }),
    simulationTelemetry: genericNormal,
    sourceAttribution: "generic_precision_observation",
  },
  {
    id: "birthday-generic-without-cues",
    query: "生日快乐",
    ablationMode: "without_cues",
    expected: { injectedCount: 0, rejectedCandidateOrdinal: 1, rerankerCalled: false },
    liveReplay: replay({ cardsCount: 0, candidateTelemetry: genericCandidates }),
    simulationTelemetry: genericWithoutCues,
    sourceAttribution: "generic_precision_observation",
  },
  {
    id: "birthday-generic-without-embedding",
    query: "生日快乐",
    ablationMode: "without_embedding",
    expected: { injectedCount: 0, rerankerCalled: false },
    liveReplay: replay({ cardsCount: 0 }),
    simulationTelemetry: genericWithoutEmbedding,
    sourceAttribution: "generic_precision_observation",
  },
  {
    id: "birthday-cake-normal",
    query: "生日快乐！我买了蛋糕",
    ablationMode: "normal",
    expected: { injectedTitles: ["Haven的第一个生日蛋糕"], rerankerCalled: true, bodyCanDiscoverWithoutAdmission: true, cueChannelMayLiftAdmission: true },
    handoffObservation: {
      provenance: "unverified_user_recollection",
      rerankerCalled: false,
    },
    replayComparison: "unverified_user_recollection_shadow_not_called_current_replay_called",
    liveReplay: replay({ cardsCount: 1, recalledIds: [cakeId], injectedTitles: ["Haven的第一个生日蛋糕"], candidateTelemetry: q3NormalCandidates }),
    simulationTelemetry: q3Normal,
    sourceAttribution: "live_replay_source_telemetry",
  },
  {
    id: "birthday-cake-without-cues",
    query: "生日快乐！我买了蛋糕",
    ablationMode: "without_cues",
    expected: { injectedCount: 0, firstCandidateTitle: "Haven的第一个生日蛋糕", rerankerCalled: false, bodyCanDiscoverWithoutAdmission: true },
    liveReplay: replay({ cardsCount: 0, firstCandidateTitle: "Haven的第一个生日蛋糕", candidateTelemetry: q3WithoutCuesCandidates }),
    simulationTelemetry: q3WithoutCues,
    sourceAttribution: "live_replay_source_telemetry",
  },
  {
    id: "birthday-cake-without-embedding",
    query: "生日快乐！我买了蛋糕",
    ablationMode: "without_embedding",
    expected: { injectedTitles: ["Haven的第一个生日蛋糕"], rerankerCalled: true },
    liveReplay: replay({ cardsCount: 1, recalledIds: [cakeId], injectedTitles: ["Haven的第一个生日蛋糕"], candidateTelemetry: q3WithoutEmbeddingCandidates }),
    simulationTelemetry: q3WithoutEmbedding,
    sourceAttribution: "live_replay_source_telemetry",
  },
  {
    id: "birthday-haven-normal",
    query: "Haven生日快乐",
    ablationMode: "normal",
    expected: { injectedTitles: ["Haven的第一个生日蛋糕", "生日第一句话仪式"], rerankerCalled: true },
    liveReplay: replay({ cardsCount: 2, recalledIds: [cakeId, birthdayFirstSentenceId], injectedTitles: ["Haven的第一个生日蛋糕", "生日第一句话仪式"], candidateTelemetry: havenNormalCandidates }),
    simulationTelemetry: havenNormal,
    sourceAttribution: "live_replay_source_telemetry_no_title_anchor_observed",
  },
  {
    id: "birthday-haven-without-cues",
    query: "Haven生日快乐",
    ablationMode: "without_cues",
    expected: { injectedTitles: ["Haven的第一个生日蛋糕", "生日第一句话仪式"], rerankerCalled: true },
    liveReplay: replay({ cardsCount: 2, recalledIds: [cakeId, birthdayFirstSentenceId], injectedTitles: ["Haven的第一个生日蛋糕", "生日第一句话仪式"], candidateTelemetry: havenWithoutCuesCandidates }),
    simulationTelemetry: havenWithoutCues,
    sourceAttribution: "live_replay_source_telemetry_no_title_anchor_observed",
  },
  {
    id: "birthday-haven-without-embedding",
    query: "Haven生日快乐",
    ablationMode: "without_embedding",
    expected: { injectedCount: 0, rerankerCalled: false },
    liveReplay: replay({ cardsCount: 0 }),
    simulationTelemetry: havenWithoutEmbedding,
    sourceAttribution: "live_replay_source_telemetry_no_title_anchor_observed",
  },
]);

export function fixtureSimulationTelemetry(fixture) {
  return fixture?.simulationTelemetry || {
    semantic: { scores: [] },
    retrievalBudget: {
      recall_ablation: { mode: fixture?.ablationMode },
      cheap_retrieval: { candidates: [] },
      sentinel: { candidates: [] },
      rerank: { called: false, decision_applied: false },
    },
    ablation: { mode: fixture?.ablationMode },
  };
}

export function fixtureCandidateTelemetry(fixture) {
  return fixture?.liveReplay?.candidateTelemetry || [];
}
