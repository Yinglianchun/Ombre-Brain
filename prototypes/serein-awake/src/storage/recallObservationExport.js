import {
  EVALUATION_GROUPING_CONTRACT,
  evaluationGroupIdFor,
  normalizeCandidateTelemetrySnapshot,
  normalizeSimulationQuery,
  normalizeSimulationTelemetrySnapshot,
  queryFamilyIdFor,
} from "./recallSimulationTraining.js";

export const RECALL_OBSERVATION_EXPORT_SCHEMA_VERSION = 6;
export const RECALL_OBSERVATION_EXPORT_TYPE = "serein.basement.recall-observation-training-export";
export const RECALL_OBSERVATION_DATASET_KIND = "recall_observation_and_reranker_calibration";

const validActions = new Set(["skip", "recall"]);
const validCandidateRelevances = new Set(["core", "weak", "irrelevant"]);
const placeholderQueries = new Set(["旧记录未保留原句", "原句未记录"]);

function cleanText(value) {
  return String(value || "").trim();
}

function cleanQuery(item) {
  const query = cleanText(item?.query);
  if (item?.queryAvailable === false || placeholderQueries.has(query)) return "";
  return query;
}

function cleanNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function dateGroup(value) {
  const raw = cleanText(value);
  const directDate = raw.match(/^(\d{4}-\d{2}-\d{2})/);
  if (directDate) return directDate[1];
  const timestamp = Date.parse(raw);
  if (!Number.isFinite(timestamp)) return "";
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Shanghai",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(new Date(timestamp));
}

export function observationGroup(item) {
  const sessionId = cleanText(item?.sessionId || item?.session_id);
  if (sessionId) return { key: `session:${sessionId}`, basis: "session_id" };
  const date = dateGroup(item?.createdAt || item?.created_at);
  if (date) return { key: `date:${date}`, basis: "created_at_date" };
  return {
    key: `observation:${cleanText(item?.id) || "unknown"}`,
    basis: "observation_id_fallback",
  };
}

function evaluationGroupingFor(item, group, query) {
  const reviewBatchId = cleanText(
    item?.reviewBatchId
      || item?.review_batch_id
      || item?.reviewBatch
      || item?.review_batch,
  );
  const sessionId = cleanText(item?.sessionId || item?.session_id);
  const source = cleanText(item?.source).toLowerCase() || "unknown";
  const observationId = cleanText(item?.id || item?.observationId || item?.observation_id);
  const sourceGroupKey = reviewBatchId
    ? `review-batch:${reviewBatchId}`
    : sessionId
      ? `session:${sessionId}`
      : source && observationId
        ? `source:${source}:${observationId}`
        : group.key;
  const queryFamilyId = queryFamilyIdFor(query);
  return {
    queryFamilyId,
    evaluationGroupId: evaluationGroupIdFor({
      queryFamilyId,
      groupKey: sourceGroupKey,
      source,
      observationId,
    }),
    sourceGroupKey,
  };
}

export function classifyObservationReview(item, review) {
  const verdict = cleanText(review?.verdict).toLowerCase();
  const query = cleanQuery(item);
  if (!review) return { kind: "missing", reason: "review_missing", label: null };
  if (!query) return { kind: "rejected", reason: "query_missing", label: null };
  if (verdict === "false_positive") {
    return { kind: "available", reason: "human_verdict_false_positive", label: "skip" };
  }
  if (verdict === "missed") {
    return { kind: "available", reason: "human_verdict_missed", label: "recall" };
  }
  if (verdict === "correct") {
    const action = cleanText(item?.observedAction || item?.action).toLowerCase();
    if (validActions.has(action)) {
      return {
        kind: "available",
        reason: "human_verdict_correct_plus_observed_action",
        label: action,
      };
    }
    return { kind: "rejected", reason: "correct_action_missing", label: null };
  }
  if (verdict === "uncertain") {
    return { kind: "rejected", reason: "uncertain_verdict", label: null };
  }
  return { kind: "rejected", reason: "verdict_invalid", label: null };
}

function candidateJudgmentsFor(item, review) {
  const candidateReviews = review?.candidateReviews && typeof review.candidateReviews === "object"
    ? review.candidateReviews
    : {};
  return (Array.isArray(item?.injected) ? item.injected : [])
    .map((memory, index) => {
      const score = memory?.scoreValue;
      return {
        memory_id: cleanText(memory?.id),
        rank: index + 1,
        observed_score: score !== null && score !== undefined && score !== "" && Number.isFinite(Number(score))
          ? Number(score)
          : null,
        relevance: cleanText(candidateReviews[memory?.id]).toLowerCase(),
      };
    })
    .filter((candidate) => candidate.memory_id && validCandidateRelevances.has(candidate.relevance));
}

function exportSourceTelemetry(source) {
  if (!source || typeof source !== "object") return null;
  return {
    body_semantic: source.bodySemantic
      ? { matched: source.bodySemantic.matched, score: source.bodySemantic.score }
      : null,
    cue_semantic: source.cueSemantic
      ? {
        matched: source.cueSemantic.matched,
        status: source.cueSemantic.status,
        score: source.cueSemantic.score,
        role: source.cueSemantic.role,
      }
      : null,
    cue_lexical: source.cueLexical
      ? { matched: source.cueLexical.matched, role: source.cueLexical.role }
      : null,
    title_anchor: source.titleAnchor
      ? { matched: source.titleAnchor.matched, count: source.titleAnchor.count }
      : null,
    exact_anchor: source.exactAnchor
      ? { matched: source.exactAnchor.matched }
      : null,
    lexical: source.lexical
      ? { matched: source.lexical.matched }
      : null,
    retrieval_alias: source.retrievalAlias
      ? { matched: source.retrievalAlias.matched }
      : null,
  };
}

function exportObservation(item, review, telemetry = [], simulationTelemetry = null, grouping = null) {
  const group = observationGroup(item);
  const evaluationGrouping = grouping || evaluationGroupingFor(item, group, cleanQuery(item));
  const verdict = cleanText(review?.verdict).toLowerCase();
  const routeVerdict = cleanText(review?.routeVerdict).toLowerCase();
  const query = cleanQuery(item);
  const candidateJudgments = candidateJudgmentsFor(item, review);
  return {
    id: cleanText(item?.id),
    observation_id: cleanText(item?.id),
    query,
    query_normalized: normalizeSimulationQuery(query),
    verdict: verdict || null,
    action_verdict: verdict || null,
    route_verdict: routeVerdict || null,
    expected_route: routeVerdict === "incorrect" ? cleanText(review?.expectedRoute) || null : null,
    observed_action: cleanText(item?.observedAction || item?.action).toLowerCase() || null,
    observed_route: cleanText(item?.route) || null,
    ablation_mode: cleanText(item?.ablationMode || simulationTelemetry?.ablationMode) || null,
    session_id: cleanText(item?.sessionId || item?.session_id) || null,
    created_at: cleanText(item?.createdAt || item?.created_at) || null,
    group: group.key,
    group_basis: group.basis,
    query_family_id: evaluationGrouping.queryFamilyId,
    evaluation_group_id: evaluationGrouping.evaluationGroupId,
    source: cleanText(item?.source) || "unknown",
    review_updated_at: cleanText(review?.updatedAt) || null,
    candidate_judgments: candidateJudgments,
    candidate_telemetry_count: telemetry.length,
    simulation_telemetry: simulationTelemetry || null,
  };
}

function exportReview(review) {
  const verdict = cleanText(review?.verdict).toLowerCase();
  const routeVerdict = cleanText(review?.routeVerdict).toLowerCase();
  return {
    verdict: verdict || null,
    action_verdict: verdict || null,
    route_verdict: routeVerdict || null,
    expected_route: routeVerdict === "incorrect" ? cleanText(review?.expectedRoute) || null : null,
    query: cleanQuery(review) || null,
    observedAt: cleanText(review?.observedAt) || null,
    updatedAt: cleanText(review?.updatedAt) || null,
  };
}

function manualSimulationEntry(label) {
  const id = cleanText(label?.observationId || label?.observation_id || label?.id);
  const query = cleanText(label?.query);
  const manualGroup = cleanText(label?.group) || `manual-simulation:${id}`;
  const queryFamilyId = queryFamilyIdFor(query);
  const evaluationGroupId = evaluationGroupIdFor({
    queryFamilyId,
    groupKey: manualGroup,
    source: "manual_simulation",
    observationId: id,
  });
  const expectedAction = cleanText(label?.expectedAction).toLowerCase();
  const expectedRoute = cleanText(label?.expectedRoute);
  const observedAction = cleanText(label?.observedAction).toLowerCase();
  const observedRoute = cleanText(label?.observedRoute);
  const verdict = cleanText(label?.actionVerdict).toLowerCase();
  const routeVerdict = cleanText(label?.routeVerdict).toLowerCase();
  const item = {
    id,
    query,
    queryAvailable: true,
    observedAction,
    action: observedAction,
    route: observedRoute,
    createdAt: cleanText(label?.createdAt),
    source: "manual_simulation",
    ablationMode: cleanText(label?.ablationMode),
    sessionId: manualGroup,
    candidateTelemetry: label?.candidateTelemetry,
    simulationTelemetry: label?.simulationTelemetry,
  };
  const review = {
    verdict,
    routeVerdict,
    expectedRoute: routeVerdict === "incorrect" ? expectedRoute : null,
    query: item.query,
    observedAt: item.createdAt,
    updatedAt: cleanText(label?.updatedAt),
  };
  const telemetry = normalizeCandidateTelemetrySnapshot(label?.candidateTelemetry, {
    observationId: id,
    query: item.query,
    queryFamilyId,
    evaluationGroupId,
    groupKey: manualGroup,
    source: "manual_simulation",
    ablationMode: label?.ablationMode,
  });
  const simulationTelemetry = label?.simulationTelemetry?.schemaVersion === 1
    ? label.simulationTelemetry
    : normalizeSimulationTelemetrySnapshot(label?.simulationTelemetry, {
      ablationMode: label?.ablationMode,
    });
  const observation = {
    ...exportObservation(item, review, telemetry, simulationTelemetry, {
      queryFamilyId,
      evaluationGroupId,
    }),
    group: manualGroup,
    group_basis: cleanText(label?.groupBasis) || "manual_simulation_batch",
    source: "manual_simulation",
    label_action: validActions.has(expectedAction) ? expectedAction : null,
    label_route: expectedRoute || null,
    expected_memory_ids: [...new Set((Array.isArray(label?.expectedMemoryIds) ? label.expectedMemoryIds : [])
      .map(cleanText)
      .filter(Boolean))],
  };
  return { item, review, observation, telemetry, simulationTelemetry };
}

function calibrationAvailability(telemetry, judgment) {
  if (!telemetry) return { available: false, reason: "telemetry_unavailable" };
  if (telemetry.telemetryStatus === "stale") return { available: false, reason: "stale" };
  if (telemetry.telemetryStatus === "unavailable") return { available: false, reason: "telemetry_unavailable" };
  if (!judgment) return { available: false, reason: "candidate_judgment_missing" };
  if (!telemetry.rerankerShadow) return { available: false, reason: "shadow_telemetry_missing" };
  if (!telemetry.rerankerShadow.called) return { available: false, reason: "shadow_not_called" };
  if (telemetry.rerankerShadow.score === null) return { available: false, reason: "shadow_score_missing" };
  if (!telemetry.candidateSources.length) return { available: false, reason: "candidate_source_missing" };
  if (!telemetry.evidence || !["bound", "unknown"].includes(telemetry.evidence.status)) {
    return { available: false, reason: "evidence_status_missing" };
  }
  if (telemetry.rerankerShadow.decisionApplied !== false) {
    return { available: false, reason: "decision_applied_not_false" };
  }
  return { available: true, reason: null };
}

function calibrationRow(entry, candidateId, judgment, telemetry) {
  const availability = calibrationAvailability(telemetry, judgment);
  const normalizedQuery = normalizeSimulationQuery(entry.observation.query);
  const observationId = entry.observation.observation_id;
  const eligibility = telemetry?.eligibility
    ? {
      reranker_eligible: telemetry.eligibility.rerankerEligible,
      absolute_floor_qualified: telemetry.eligibility.absoluteFloorQualified,
      gray_zone_qualified: telemetry.eligibility.grayZoneQualified,
    }
    : null;
  const rerankerShadow = telemetry?.rerankerShadow
    ? {
      called: telemetry.rerankerShadow.called,
      score: telemetry.rerankerShadow.score,
      model: telemetry.rerankerShadow.model,
      decision_applied: telemetry.rerankerShadow.decisionApplied,
      status: telemetry.rerankerShadow.status,
      reason: telemetry.rerankerShadow.reason,
      called_false_reason: telemetry.rerankerShadow.calledFalseReason,
      admission_status: telemetry.rerankerShadow.admissionStatus,
    }
    : null;
  return {
    calibration_key: telemetry?.calibrationKey
      || [observationId, normalizedQuery, candidateId].map(cleanText).join("::"),
    observation_id: observationId,
    query: entry.observation.query,
    query_normalized: telemetry?.queryNormalized || normalizedQuery,
    source: entry.observation.source,
    group: entry.observation.group,
    group_basis: entry.observation.group_basis,
    query_family_id: entry.observation.query_family_id,
    evaluation_group_id: entry.observation.evaluation_group_id,
    ablation_mode: telemetry?.ablationMode || null,
    candidate_id: candidateId,
    rank: telemetry?.rank || judgment?.rank || null,
    relevance: judgment?.relevance || null,
    legacy_observed_score: cleanNumber(judgment?.observed_score),
    candidate_sources: telemetry?.candidateSources || [],
    source_telemetry: exportSourceTelemetry(telemetry?.sourceTelemetry),
    eligibility,
    floors: telemetry?.floors || null,
    final_admission_source: telemetry?.finalAdmissionSource || null,
    admission_status: telemetry?.admissionStatus || null,
    reranker_shadow: rerankerShadow,
    shadow_score: telemetry?.rerankerShadow?.score ?? null,
    model: telemetry?.rerankerShadow?.model || null,
    evidence: telemetry?.evidence || null,
    telemetry_generated_at: telemetry?.telemetryGeneratedAt || null,
    semantic_profile: telemetry?.semanticProfile || null,
    telemetry_status: telemetry?.telemetryStatus || "unavailable",
    calibration_available: availability.available,
    unavailable_reason: availability.reason,
  };
}

function buildCalibrationRows(entries) {
  return entries.flatMap((entry) => {
    const judgments = new Map(
      (entry.observation.candidate_judgments || []).map((item) => [item.memory_id, item]),
    );
    const telemetry = new Map((entry.telemetry || []).map((item) => [item.candidateId, item]));
    const ids = [...new Set([...telemetry.keys(), ...judgments.keys()])];
    return ids.map((candidateId) => calibrationRow(
      entry,
      candidateId,
      judgments.get(candidateId) || null,
      telemetry.get(candidateId) || null,
    ));
  });
}

function countBy(items, field) {
  return items.reduce((counts, item) => {
    const key = cleanText(item?.[field]) || "unknown";
    counts[key] = (counts[key] || 0) + 1;
    return counts;
  }, {});
}

export function buildRecallObservationTrainingExport(
  items,
  reviews,
  generatedAt = new Date().toISOString(),
  manualSimulations = [],
) {
  const observationEntries = (Array.isArray(items) ? items : [])
    .filter((item) => item && cleanText(item.id))
    .map((item) => {
      const review = reviews?.[item.id] || null;
      const query = cleanQuery(item);
      const group = observationGroup(item);
      const grouping = evaluationGroupingFor(item, group, query);
      const telemetry = normalizeCandidateTelemetrySnapshot(item?.candidateTelemetry, {
        observationId: cleanText(item.id),
        query,
        queryFamilyId: grouping.queryFamilyId,
        evaluationGroupId: grouping.evaluationGroupId,
        groupKey: grouping.sourceGroupKey,
        source: item?.source,
        ablationMode: item?.ablationMode,
      });
      const simulationTelemetry = item?.simulationTelemetry?.schemaVersion === 1
        ? item.simulationTelemetry
        : normalizeSimulationTelemetrySnapshot(item?.simulationTelemetry, {
          ablationMode: item?.ablationMode,
        });
      return {
        item,
        review,
        telemetry,
        simulationTelemetry,
        observation: exportObservation(item, review, telemetry, simulationTelemetry, grouping),
      };
    });
  const manualEntries = (Array.isArray(manualSimulations) ? manualSimulations : [])
    .filter((item) => item && cleanText(item.id || item.observationId || item.observation_id))
    .map(manualSimulationEntry);
  const entries = [...observationEntries, ...manualEntries];
  const observations = entries.map((entry) => entry.observation);
  const calibrationRows = buildCalibrationRows(entries);
  const loadedSourceCounts = observationEntries.reduce((summary, entry) => {
    const source = cleanText(entry.item?.source).toLowerCase();
    if (source === "hook") summary.hook += 1;
    if (source === "gateway") summary.gateway += 1;
    return summary;
  }, { hook: 0, gateway: 0 });
  const counts = entries.reduce((summary, entry) => {
    const classification = classifyObservationReview(entry.item, entry.review);
    summary[classification.kind] += 1;
    return summary;
  }, { available: 0, rejected: 0, missing: 0 });
  const alignedReviews = Object.fromEntries(
    entries
      .filter((entry) => entry.review)
      .map((entry) => [entry.observation.observation_id, exportReview(entry.review)]),
  );
  const queryFamilyCount = new Set(
    observations.map((item) => item.query_family_id).filter(Boolean),
  ).size;
  const evaluationGroupCount = new Set(
    observations.map((item) => item.evaluation_group_id).filter(Boolean),
  ).size;
  const calibrationCoverage = {
    simulation_telemetry_rows: entries.filter((entry) => entry.simulationTelemetry).length,
    simulation_telemetry_missing: entries.filter((entry) => !entry.simulationTelemetry).length,
    candidate_judgment_rows: calibrationRows.filter((row) => row.relevance).length,
    candidate_telemetry_rows: entries.reduce((count, entry) => count + entry.telemetry.length, 0),
    calibration_rows: calibrationRows.length,
    calibration_available: calibrationRows.filter((row) => row.calibration_available).length,
    query_family_count: queryFamilyCount,
    evaluation_group_count: evaluationGroupCount,
    unavailable: calibrationRows.filter((row) => row.unavailable_reason && row.unavailable_reason !== "stale").length,
    stale: calibrationRows.filter((row) => row.unavailable_reason === "stale").length,
    unavailable_reasons: countBy(
      calibrationRows.filter((row) => !row.calibration_available),
      "unavailable_reason",
    ),
  };
  return {
    schema_version: RECALL_OBSERVATION_EXPORT_SCHEMA_VERSION,
    export_type: RECALL_OBSERVATION_EXPORT_TYPE,
    dataset_kind: RECALL_OBSERVATION_DATASET_KIND,
    generated_at: generatedAt,
    review_storage_key: "serein.basement.recall-observation-review.v1",
    privacy: {
      includes: [
        "user_query",
        "human_verdict",
        "separate_action_and_route_verdicts",
        "observed_route_and_action",
        "source_and_time_group",
        "query_family_and_evaluation_group_ids",
        "reviewed_candidate_id_rank_score_and_relevance",
        "candidate_source_floor_admission_and_shadow_telemetry",
        "simulation_route_budget_sentinel_and_channel_telemetry",
        "manual_simulation_expected_action_route_and_memory_ids",
      ],
      excludes: [
        "full_prompt",
        "developer_context",
        "injected_memory_body",
        "additional_context",
        "scene_body",
        "scene_cues",
        "evidence_source_text",
      ],
    },
    partitions: {
      recall_observations: "observations",
      reranker_calibration: "calibration_rows",
    },
    evaluation_grouping: EVALUATION_GROUPING_CONTRACT,
    scope: {
      coverage: "currently_loaded_observation_window",
      full_history: false,
      loaded_hook_observations: loadedSourceCounts.hook,
      loaded_gateway_observations: loadedSourceCounts.gateway,
      loaded_manual_simulations: manualEntries.length,
    },
    telemetry_coverage: calibrationCoverage,
    summary: {
      total_observations: observationEntries.length,
      total_manual_simulations: manualEntries.length,
      total_cases: observations.length,
      loaded_hook_observations: loadedSourceCounts.hook,
      loaded_gateway_observations: loadedSourceCounts.gateway,
      simulation_telemetry_rows: calibrationCoverage.simulation_telemetry_rows,
      simulation_telemetry_missing: calibrationCoverage.simulation_telemetry_missing,
      candidate_judgment_rows: calibrationCoverage.candidate_judgment_rows,
      candidate_telemetry_rows: calibrationCoverage.candidate_telemetry_rows,
      calibration_rows: calibrationCoverage.calibration_rows,
      calibration_available: calibrationCoverage.calibration_available,
      calibration_unavailable: calibrationCoverage.unavailable,
      calibration_stale: calibrationCoverage.stale,
      query_family_count: calibrationCoverage.query_family_count,
      evaluation_group_count: calibrationCoverage.evaluation_group_count,
      ...counts,
    },
    observations,
    calibration_rows: calibrationRows,
    reviews: alignedReviews,
  };
}
