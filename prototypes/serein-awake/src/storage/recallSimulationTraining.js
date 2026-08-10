export const recallSimulationTrainingStorageKey = "serein.basement.recall-simulation-training.v1";

const batchStorageKey = "serein.basement.recall-simulation-training-batch.v1";
const validActions = new Set(["skip", "recall"]);
const validAblationModes = new Set(["normal", "without_cues", "without_embedding"]);
const validTelemetryStatuses = new Set(["fresh", "stale", "unavailable"]);
const validCandidateRelevances = new Set(["core", "weak", "irrelevant"]);
export const EVALUATION_GROUPING_CONTRACT_VERSION = "serein.evaluation-group.v1";
export const EVALUATION_GROUPING_HASH_ALGORITHM = "fnv1a32";
const candidateSourceEnums = new Set([
  "exact_anchor",
  "title_anchor",
  "lexical",
  "cue_lexical",
  "cue_semantic",
  "body_semantic",
  "retrieval_alias",
]);

const cleanText = (value) => String(value || "").trim();

export function normalizeSimulationQuery(value) {
  return cleanText(value)
    .normalize("NFKC")
    .replace(/\s+/g, " ")
    .trim();
}

function stableHash(value) {
  let hash = 2166136261;
  for (const character of String(value)) {
    hash ^= character.codePointAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

export function queryFamilyIdFor(value) {
  const normalizedQuery = normalizeSimulationQuery(value);
  return normalizedQuery
    ? `qf-v1-${stableHash(normalizedQuery)}`
    : null;
}

export function evaluationGroupIdFor({
  queryFamilyId,
  groupKey,
  source,
  observationId,
} = {}) {
  const family = cleanText(queryFamilyId);
  const sourceGroup = cleanText(groupKey)
    || [cleanText(source), cleanText(observationId)].filter(Boolean).join(":");
  if (!family && !sourceGroup) return null;
  return `eg-v1-${stableHash(`${family || "no-query"}|${sourceGroup || "unknown"}`)}`;
}

export const EVALUATION_GROUPING_CONTRACT = Object.freeze({
  version: EVALUATION_GROUPING_CONTRACT_VERSION,
  hashAlgorithm: EVALUATION_GROUPING_HASH_ALGORITHM,
  queryFamilyId: "qf-v1-fnv1a32(normalizeSimulationQuery(query))",
  evaluationGroupId: "eg-v1-fnv1a32(query_family_id|source_group_key)",
  sourceGroupPrecedence: [
    "review_batch_id",
    "session_id",
    "manual_simulation_batch_id",
    "observation_source_and_id",
  ],
  holdoutUnit: ["query_family_id", "evaluation_group_id"],
  holdoutRule: "rows sharing either id form one holdout component; split neither",
  candidateRows: "all candidate rows for one observation share both ids",
  excludes: [
    "candidate_relevance",
    "reranker_score",
    "admission_result",
    "route_and_router_example_fields",
  ],
});

function nullableNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function nullableBoolean(value) {
  return typeof value === "boolean" ? value : null;
}

function nullableNonNegativeInteger(value) {
  const number = Number.parseInt(value, 10);
  return Number.isInteger(number) && number >= 0 ? number : null;
}

function normalizedAblationMode(value) {
  const mode = cleanText(value).toLowerCase();
  return validAblationModes.has(mode) ? mode : null;
}

function normalizedCandidateJudgments(values) {
  const seen = new Set();
  return (Array.isArray(values) ? values : []).map((item, index) => {
    const memoryId = cleanText(item?.memoryId || item?.memory_id || item?.candidateId || item?.candidate_id);
    const relevance = cleanText(item?.relevance).toLowerCase();
    if (!memoryId || seen.has(memoryId) || !validCandidateRelevances.has(relevance)) return null;
    seen.add(memoryId);
    return {
      memoryId,
      rank: nullableNonNegativeInteger(item?.rank) || index + 1,
      relevance,
    };
  }).filter(Boolean);
}

function normalizedTelemetryStatus(value, fallback = "unavailable") {
  const status = cleanText(value).toLowerCase();
  return validTelemetryStatuses.has(status) ? status : fallback;
}

function candidateSources(raw) {
  const sources = Array.isArray(raw) ? raw : [];
  return [...new Set(sources
    .map(cleanText)
    .filter((source) => candidateSourceEnums.has(source)))];
}

function nullableSourceMatch(value, fallback = null) {
  return typeof value === "boolean" ? value : fallback;
}

function sourceRole(value) {
  const role = cleanText(value).toLowerCase();
  return ["candidate_only", "direct_evidence", "none"].includes(role) ? role : null;
}

function sourceStatus(value) {
  const status = cleanText(value).toLowerCase();
  return ["matched", "not_matched", "unavailable", "unknown"].includes(status)
    ? status
    : null;
}

function normalizeCandidateSourceTelemetry(raw, sources) {
  const sourceSet = new Set(sources);
  const cueSemantic = raw.cueSemantic && typeof raw.cueSemantic === "object"
    ? raw.cueSemantic
    : raw.cue_semantic && typeof raw.cue_semantic === "object"
      ? raw.cue_semantic
      : {};
  const cueLexicalMatch = nullableSourceMatch(
    raw.cueLexicalMatch ?? raw.cue_lexical_match,
    sourceSet.has("cue_lexical") ? true : null,
  );
  const cueSemanticMatch = nullableSourceMatch(
    raw.cueSemanticMatch ?? raw.cue_semantic_match,
    sourceSet.has("cue_semantic") ? true : null,
  );
  const titleAnchorMatch = nullableSourceMatch(
    raw.titleAnchorMatch ?? raw.title_anchor_match,
    sourceSet.has("title_anchor") ? true : null,
  );
  const bodySemanticScore = nullableNumber(
    raw.bodySemanticScore ?? raw.body_semantic_score ?? raw.semanticScore ?? raw.semantic_score,
  );
  const lexicalMatch = nullableSourceMatch(
    raw.lexicalMatch ?? raw.lexical_match,
    sourceSet.has("lexical") ? true : null,
  );
  return {
    bodySemantic: {
      matched: bodySemanticScore === null ? (sourceSet.has("body_semantic") ? true : null) : true,
      score: bodySemanticScore,
    },
    cueSemantic: {
      matched: cueSemanticMatch,
      status: sourceStatus(cueSemantic.status)
        || (cueSemanticMatch === true ? "matched" : null),
      score: nullableNumber(cueSemantic.score ?? raw.cueSemanticScore ?? raw.cue_semantic_score),
      role: sourceRole(cueSemantic.role || raw.cueSemanticRole || raw.cue_semantic_role),
    },
    cueLexical: {
      matched: cueLexicalMatch,
      role: sourceRole(raw.cueLexicalRole || raw.cue_lexical_role),
    },
    titleAnchor: {
      matched: titleAnchorMatch,
      count: Array.isArray(raw.titleAnchorTerms || raw.title_anchor_terms)
        ? (raw.titleAnchorTerms || raw.title_anchor_terms).length
        : null,
    },
    exactAnchor: {
      matched: nullableSourceMatch(
        raw.exactAnchorMatch ?? raw.exact_anchor_match,
        sourceSet.has("exact_anchor") ? true : null,
      ),
    },
    lexical: {
      matched: lexicalMatch,
    },
    retrievalAlias: {
      matched: nullableSourceMatch(
        raw.retrievalAliasMatch ?? raw.retrieval_alias_match,
        sourceSet.has("retrieval_alias") ? true : null,
      ),
    },
  };
}

function normalizeCanonicalSourceTelemetry(source) {
  if (!source || typeof source !== "object") return null;
  const normalizePart = (part) => (part && typeof part === "object" ? part : null);
  return {
    bodySemantic: normalizePart(source.bodySemantic),
    cueSemantic: normalizePart(source.cueSemantic),
    cueLexical: normalizePart(source.cueLexical),
    titleAnchor: normalizePart(source.titleAnchor),
    exactAnchor: normalizePart(source.exactAnchor),
    lexical: normalizePart(source.lexical),
    retrievalAlias: normalizePart(source.retrievalAlias),
  };
}

export function candidateTelemetryKey(observationId, queryNormalized, candidateId) {
  return [observationId, queryNormalized, candidateId].map(cleanText).join("::");
}

export function normalizeCandidateTelemetrySnapshot(values, options = {}) {
  const observationId = cleanText(options.observationId);
  const queryNormalized = normalizeSimulationQuery(options.queryNormalized || options.query);
  const queryFamilyId = cleanText(options.queryFamilyId) || queryFamilyIdFor(queryNormalized);
  const evaluationGroupId = cleanText(options.evaluationGroupId) || evaluationGroupIdFor({
    queryFamilyId,
    groupKey: options.groupKey,
    source: options.source,
    observationId,
  });
  const fallbackAblationMode = normalizedAblationMode(options.ablationMode);
  const rows = Array.isArray(values) ? values : [];
  const seen = new Set();
  return rows.map((raw, index) => {
    if (!raw || typeof raw !== "object") return null;
    const candidateId = cleanText(raw.candidateId || raw.candidate_id || raw.bucket_id);
    if (!candidateId || seen.has(candidateId)) return null;
    seen.add(candidateId);
    const shadow = raw.rerankerShadow && typeof raw.rerankerShadow === "object"
      ? raw.rerankerShadow
      : raw.reranker_shadow && typeof raw.reranker_shadow === "object"
        ? raw.reranker_shadow
        : null;
    const rawEligibility = raw.eligibility && typeof raw.eligibility === "object"
      ? raw.eligibility
      : {};
    const rawFloors = raw.floors && typeof raw.floors === "object" ? raw.floors : {};
    const rawEvidence = raw.evidence && typeof raw.evidence === "object" ? raw.evidence : {};
    const hasTelemetryShape = Boolean(
      shadow
      || raw.candidateSources
      || raw.candidate_sources
      || raw.rerankerEligible !== undefined
      || raw.reranker_eligible !== undefined
      || raw.finalAdmissionSource
      || raw.final_admission_source,
    );
    const telemetryStatus = normalizedTelemetryStatus(
      raw.telemetryStatus || raw.telemetry_status || raw.replayStatus || raw.replay_status,
      hasTelemetryShape ? "fresh" : "unavailable",
    );
    const evidenceStatus = cleanText(
      rawEvidence.status || raw.evidenceStatus || raw.evidence_status || shadow?.evidence_status,
    ).toLowerCase();
    const normalizedEvidenceStatus = ["bound", "unknown"].includes(evidenceStatus)
      ? evidenceStatus
      : null;
    const absoluteFloor = nullableNumber(
      rawFloors.absolute ?? raw.absoluteFloor ?? raw.absolute_floor,
    );
    const entryFloor = nullableNumber(
      rawFloors.entry
        ?? raw.entryFloor
        ?? raw.entry_floor
        ?? raw.rerankerEntryFloor
        ?? raw.reranker_entry_floor,
    );
    const absoluteFloorPassed = nullableBoolean(
      rawEligibility.absoluteFloorPassed
        ?? rawEligibility.absolute_floor_passed
        ?? rawEligibility.absoluteFloorQualified
        ?? rawEligibility.absolute_floor_qualified
        ?? raw.floorQualified
        ?? raw.floor_qualified,
    );
    const grayZoneQualified = nullableBoolean(
      rawEligibility.grayZoneQualified
        ?? rawEligibility.gray_zone_qualified
        ?? raw.grayZoneQualified
        ?? raw.gray_zone_qualified,
    );
    const rerankerEligible = nullableBoolean(
      rawEligibility.rerankerEligible
        ?? rawEligibility.reranker_eligible
        ?? raw.rerankerEligible
        ?? raw.reranker_eligible,
    );
    const finalAdmissionSource = cleanText(
      raw.finalAdmissionSource || raw.final_admission_source,
    ) || null;
    const shadowReason = cleanText(
      shadow?.reason
        || raw.rerankerShadowReason
        || raw.reranker_shadow_reason
        || shadow?.status,
    ) || null;
    const shadowCalled = Boolean(shadow?.called);
    return {
      observationId,
      queryNormalized,
      queryFamilyId,
      evaluationGroupId,
      calibrationKey: candidateTelemetryKey(observationId, queryNormalized, candidateId),
      candidateId,
      rank: nullableNonNegativeInteger(raw.rank) || index + 1,
      ablationMode: normalizedAblationMode(raw.ablationMode || raw.ablation_mode) || fallbackAblationMode,
      candidateSources: candidateSources(raw.candidateSources || raw.candidate_sources),
      eligibility: {
        rerankerEligible,
        absoluteFloorQualified: absoluteFloorPassed,
        grayZoneQualified,
      },
      sourceTelemetry: normalizeCanonicalSourceTelemetry(raw.sourceTelemetry || raw.source_telemetry)
        || normalizeCandidateSourceTelemetry(raw, candidateSources(raw.candidateSources || raw.candidate_sources)),
      floors: {
        absolute: absoluteFloor,
        entry: entryFloor,
      },
      finalAdmissionSource,
      admissionStatus: cleanText(
        raw.admissionStatus
          || raw.admission_status
          || finalAdmissionSource,
      ) || null,
      rerankerShadow: shadow ? {
        called: shadowCalled,
        score: nullableNumber(shadow.score),
        model: cleanText(shadow.model) || null,
        decisionApplied: nullableBoolean(shadow.decisionApplied ?? shadow.decision_applied),
        status: cleanText(shadow.status) || null,
        reason: shadowReason,
        calledFalseReason: !shadowCalled
          ? cleanText(
            shadow.calledFalseReason
              || shadow.called_false_reason
              || shadowReason,
          ) || null
          : null,
        admissionStatus: cleanText(shadow.admissionStatus || shadow.admission_status) || null,
      } : null,
      evidence: normalizedEvidenceStatus ? {
        status: normalizedEvidenceStatus,
        count: nullableNonNegativeInteger(
          rawEvidence.count ?? raw.evidenceCount ?? raw.evidence_count ?? shadow?.evidence_count,
        ),
      } : null,
      telemetryStatus,
      telemetryGeneratedAt: cleanText(
        raw.telemetryGeneratedAt
          || raw.telemetry_generated_at
          || shadow?.telemetryGeneratedAt
          || shadow?.telemetry_generated_at,
      ) || null,
      semanticProfile: cleanText(raw.semanticProfile || raw.semantic_profile) || null,
    };
  }).filter(Boolean);
}

function normalizedDebugBoolean(value) {
  return nullableBoolean(value);
}

function normalizedRouteScores(values) {
  return (Array.isArray(values) ? values : [])
    .map((item) => ({
      route: cleanText(item?.route) || null,
      action: cleanText(item?.action).toLowerCase() || null,
      score: nullableNumber(item?.score),
    }))
    .filter((item) => item.route);
}

function normalizedSentinelSources(values) {
  const aliases = {
    title: "title_anchor",
    authored_cue: "cue_lexical",
    cue_semantic: "cue_semantic",
    body_semantic: "body_semantic",
    exact_anchor: "exact_anchor",
    lexical: "lexical",
  };
  return [...new Set((Array.isArray(values) ? values : [])
    .map(cleanText)
    .map((value) => aliases[value] || value)
    .filter((value) => candidateSourceEnums.has(value)))];
}

function normalizedQueryFacets(values) {
  return (Array.isArray(values) ? values : [])
    .map((item) => ({
      kind: cleanText(item?.kind) || null,
      value: cleanText(item?.value) || null,
      strength: cleanText(item?.strength) || null,
      source: cleanText(item?.source) || null,
      matched: nullableBoolean(item?.matched),
      score: nullableNumber(item?.score),
    }))
    .filter((item) => item.kind || item.value || item.source);
}

function normalizedPrototypePrior(value) {
  if (!value || typeof value !== "object") return null;
  return {
    source: cleanText(value.source) || null,
    candidate: nullableBoolean(value.candidate),
    highConfidence: nullableBoolean(value.highConfidence ?? value.high_confidence),
    confidence: nullableNumber(value.confidence),
    confidenceFloor: nullableNumber(value.confidenceFloor ?? value.confidence_floor),
    margin: nullableNumber(value.margin),
    marginFloor: nullableNumber(value.marginFloor ?? value.margin_floor),
    shapeClean: nullableBoolean(value.shapeClean ?? value.shape_clean),
    mixedClause: nullableBoolean(value.mixedClause ?? value.mixed_clause),
    structuralVetoes: (Array.isArray(value.structuralVetoes)
      ? value.structuralVetoes
      : value.structural_vetoes)
      ?.map(cleanText)
      .filter(Boolean)
      .slice(0, 12) || [],
    status: cleanText(value.status) || null,
  };
}

function normalizeSimulationTelemetrySnapshot(input, options = {}) {
  const root = input && typeof input === "object" ? input : {};
  const semantic = root.semantic && typeof root.semantic === "object"
    ? root.semantic
    : root.semantic_recall_debug && typeof root.semantic_recall_debug === "object"
      ? root.semantic_recall_debug
      : {};
  const budget = root.retrievalBudget && typeof root.retrievalBudget === "object"
    ? root.retrievalBudget
    : root.retrieval_budget && typeof root.retrieval_budget === "object"
      ? root.retrieval_budget
      : semantic.retrieval_budget && typeof semantic.retrieval_budget === "object"
        ? semantic.retrieval_budget
        : {};
  const sentinel = root.sentinel && typeof root.sentinel === "object"
    ? root.sentinel
    : budget.sentinel && typeof budget.sentinel === "object"
      ? budget.sentinel
      : {};
  const ablation = root.ablation && typeof root.ablation === "object"
    ? root.ablation
    : root.recall_ablation && typeof root.recall_ablation === "object"
      ? root.recall_ablation
      : budget.recall_ablation && typeof budget.recall_ablation === "object"
        ? budget.recall_ablation
        : semantic.recall_ablation && typeof semantic.recall_ablation === "object"
          ? semantic.recall_ablation
          : {};
  const cheap = budget.cheapRetrieval && typeof budget.cheapRetrieval === "object"
    ? budget.cheapRetrieval
    : budget.cheap_retrieval && typeof budget.cheap_retrieval === "object"
      ? budget.cheap_retrieval
      : {};
  const cueSemantic = budget.cueSemantic && typeof budget.cueSemantic === "object"
    ? budget.cueSemantic
    : budget.cue_semantic && typeof budget.cue_semantic === "object"
      ? budget.cue_semantic
      : {};
  const factEventProbe = budget.factEventProbe && typeof budget.factEventProbe === "object"
    ? budget.factEventProbe
    : budget.fact_event_probe && typeof budget.fact_event_probe === "object"
      ? budget.fact_event_probe
      : {};
  const rerank = budget.rerank && typeof budget.rerank === "object" ? budget.rerank : {};
  const routeScores = normalizedRouteScores(semantic.scores);
  const hasShape = Boolean(
    Object.keys(semantic).length
      || Object.keys(budget).length
      || Object.keys(sentinel).length
      || Object.keys(ablation).length
      || options.force,
  );
  if (!hasShape) return null;
  const rerankCalled = normalizedDebugBoolean(rerank.called);
  const rerankReason = cleanText(rerank.reason) || null;
  return {
    schemaVersion: 1,
    ablationMode: normalizedAblationMode(ablation.mode || options.ablationMode),
    route: {
      name: cleanText(semantic.route) || null,
      action: cleanText(semantic.route_action) || null,
      appliedAction: cleanText(semantic.applied_action) || null,
      recommendedAction: cleanText(semantic.recommended_action) || null,
      confidence: nullableNumber(semantic.confidence),
      margin: nullableNumber(semantic.margin),
      reason: cleanText(semantic.reason) || null,
      skipApplied: normalizedDebugBoolean(semantic.skip_applied),
      routeSkipReason: cleanText(semantic.route_skip_reason) || null,
      scores: routeScores,
    },
    budget: {
      mode: cleanText(budget.mode) || null,
      surfaceRoute: cleanText(budget.surface_route) || null,
      routeBudget: cleanText(budget.route_budget) || null,
      effectiveBudget: cleanText(budget.effective_budget) || null,
      initialBudget: cleanText(budget.initial_budget) || null,
      finalBudget: cleanText(budget.final_budget) || null,
      decisionSource: cleanText(budget.budget_decision_source) || null,
      escalationReason: cleanText(budget.escalation_reason) || null,
      typedCandidateId: cleanText(budget.typed_candidate_id) || null,
      typedCandidateKind: cleanText(budget.typed_candidate_kind) || null,
      typedCandidateScore: nullableNumber(budget.typed_candidate_score),
      selectedMemoryId: cleanText(budget.selected_memory_id) || null,
      selectedMemoryKind: cleanText(budget.selected_memory_kind) || null,
      anchorOverride: normalizedDebugBoolean(budget.anchor_override),
      anchorOverrideReasons: (Array.isArray(budget.anchor_override_reasons)
        ? budget.anchor_override_reasons
        : [])
        .map(cleanText)
        .filter(Boolean)
        .slice(0, 12),
      pureChitchatPrior: normalizedDebugBoolean(budget.pure_chitchat_prior),
      routeWouldSkip: normalizedDebugBoolean(budget.route_would_skip),
      budgetSkipApplied: normalizedDebugBoolean(budget.budget_skip_applied),
      routeSkipDeferred: normalizedDebugBoolean(budget.route_skip_deferred),
      evidenceVetoApplied: normalizedDebugBoolean(budget.evidence_veto_applied),
      deferredReason: cleanText(budget.deferred_reason) || null,
      prototypePrior: normalizedPrototypePrior(budget.prototype_prior),
      queryFacets: normalizedQueryFacets(budget.query_facets),
      channels: normalizedSentinelSources(budget.channels),
      semanticTopK: nullableNonNegativeInteger(budget.semantic_top_k),
      dateOnly: normalizedDebugBoolean(budget.date_only),
      cheapRetrieval: {
        candidateCount: nullableNonNegativeInteger(cheap.candidate_count),
        floorQualifiedCount: nullableNonNegativeInteger(cheap.floor_qualified_count),
        grayZoneCount: nullableNonNegativeInteger(cheap.gray_zone_count),
        rerankerEligibleCount: nullableNonNegativeInteger(cheap.reranker_eligible_count),
        absoluteFloor: nullableNumber(budget.absolute_floor),
        rerankerEntryFloor: nullableNumber(budget.reranker_entry_floor),
        stopReason: cleanText(cheap.stop_reason) || null,
      },
      cueSemantic: {
        status: cleanText(cueSemantic.status) || null,
        reason: cleanText(cueSemantic.reason) || null,
        candidateCount: nullableNonNegativeInteger(cueSemantic.candidate_count),
        datasetVersion: cleanText(cueSemantic.dataset_version) || null,
      },
      factEventProbe: {
        status: cleanText(factEventProbe.status) || null,
        reason: cleanText(factEventProbe.reason) || null,
        candidateCount: nullableNonNegativeInteger(factEventProbe.candidate_count),
        matches: (Array.isArray(factEventProbe.matches) ? factEventProbe.matches : [])
          .map((item) => ({
            memoryId: cleanText(item?.memory_id || item?.memoryId) || null,
            memoryKind: cleanText(item?.memory_kind || item?.memoryKind) || null,
            score: nullableNumber(item?.score),
            importance: nullableNonNegativeInteger(item?.importance),
            localDate: cleanText(item?.local_date || item?.localDate) || null,
            localStartTime: cleanText(item?.local_start_time || item?.localStartTime) || null,
            coveredBySceneId: cleanText(
              item?.covered_by_scene_id || item?.coveredBySceneId,
            ) || null,
          }))
          .filter((item) => item.memoryId && ["fact", "event"].includes(item.memoryKind))
          .slice(0, 12),
      },
      rerank: {
        wouldCall: normalizedDebugBoolean(rerank.would_call),
        called: rerankCalled,
        candidateCount: nullableNonNegativeInteger(rerank.candidate_count),
        scoreCount: nullableNonNegativeInteger(rerank.score_count),
        reason: rerankReason,
        calledFalseReason: rerankCalled === false
          ? cleanText(rerank.called_false_reason || rerankReason) || null
          : null,
        model: cleanText(rerank.model) || null,
        decisionApplied: normalizedDebugBoolean(rerank.decision_applied),
        telemetryGeneratedAt: cleanText(rerank.telemetry_generated_at) || null,
      },
    },
    sentinel: {
      called: normalizedDebugBoolean(sentinel.called),
      topK: nullableNonNegativeInteger(sentinel.top_k),
      rescueFloor: nullableNumber(sentinel.rescue_floor),
      candidateCount: nullableNonNegativeInteger(sentinel.candidate_count),
      floorQualifiedCount: nullableNonNegativeInteger(sentinel.floor_qualified_count),
      expanded: normalizedDebugBoolean(sentinel.expanded),
      reranked: normalizedDebugBoolean(sentinel.reranked),
      injectionAllowed: normalizedDebugBoolean(sentinel.injection_allowed),
      recorded: normalizedDebugBoolean(sentinel.recorded),
      reason: cleanText(sentinel.reason) || null,
      candidates: (Array.isArray(sentinel.candidates) ? sentinel.candidates : [])
        .map((item) => ({
          candidateId: cleanText(item?.candidateId || item?.candidate_id || item?.bucket_id) || null,
          semanticScore: nullableNumber(item?.semanticScore ?? item?.semantic_score),
          rescueScore: nullableNumber(item?.rescueScore ?? item?.rescue_score),
          sources: normalizedSentinelSources(item?.sources),
          floorQualified: normalizedDebugBoolean(item?.floorQualified ?? item?.floor_qualified),
          wouldInject: normalizedDebugBoolean(item?.wouldInject ?? item?.would_inject),
        }))
        .filter((item) => item.candidateId),
    },
  };
}

export { normalizeSimulationTelemetrySnapshot };

function uniqueTexts(values) {
  return [...new Set((Array.isArray(values) ? values : [])
    .map(cleanText)
    .filter(Boolean))];
}

function randomId(prefix) {
  const uuid = globalThis.crypto?.randomUUID?.();
  return `${prefix}-${uuid || `${Date.now()}-${Math.random().toString(16).slice(2)}`}`;
}

function activeBatchId() {
  try {
    const saved = window.sessionStorage.getItem(batchStorageKey);
    if (saved) return saved;
    const created = randomId("batch");
    window.sessionStorage.setItem(batchStorageKey, created);
    return created;
  } catch {
    return `date-${new Date().toISOString().slice(0, 10)}`;
  }
}

export function createRecallSimulationTrainingLabel(input, options = {}) {
  const query = cleanText(input?.query);
  const expectedAction = cleanText(input?.expectedAction).toLowerCase();
  if (!query || !validActions.has(expectedAction)) return null;
  const observedAction = cleanText(input?.observedAction).toLowerCase();
  const observedRoute = cleanText(input?.observedRoute);
  const expectedRoute = cleanText(input?.expectedRoute);
  const now = cleanText(options.now) || new Date().toISOString();
  const batchId = cleanText(options.batchId) || activeBatchId();
  const id = cleanText(options.id) || randomId("manual-simulation");
  const observationId = cleanText(options.observationId || input?.observationId) || id;
  const group = cleanText(options.groupKey) || `manual-simulation:${batchId}`;
  const queryNormalized = normalizeSimulationQuery(query);
  const queryFamilyId = queryFamilyIdFor(queryNormalized);
  const evaluationGroupId = evaluationGroupIdFor({
    queryFamilyId,
    groupKey: group,
    source: "manual_simulation",
    observationId,
  });
  const actionVerdict = observedAction === expectedAction
    ? "correct"
    : expectedAction === "recall" ? "missed" : "false_positive";
  const routeVerdict = expectedRoute
    ? observedRoute === expectedRoute ? "correct" : "incorrect"
    : null;
  const ablationMode = normalizedAblationMode(input?.ablationMode) || "normal";
  const candidateTelemetry = normalizeCandidateTelemetrySnapshot(input?.candidateTelemetry, {
    observationId,
    queryNormalized,
    queryFamilyId,
    evaluationGroupId,
    groupKey: group,
    source: "manual_simulation",
    ablationMode,
  });
  const simulationTelemetry = normalizeSimulationTelemetrySnapshot(
    input?.simulationTelemetry || input?.simulation_telemetry,
    { ablationMode },
  );
  return {
    id,
    observationId,
    query,
    queryNormalized,
    expectedAction,
    expectedRoute: expectedRoute || null,
    expectedMemoryIds: uniqueTexts(input?.expectedMemoryIds),
    observedAction: validActions.has(observedAction) ? observedAction : null,
    observedRoute: observedRoute || null,
    actionVerdict,
    routeVerdict,
    group,
    groupBasis: "manual_simulation_batch",
    queryFamilyId,
    evaluationGroupId,
    source: "manual_simulation",
    ablationMode,
    candidateJudgments: normalizedCandidateJudgments(input?.candidateJudgments),
    candidateTelemetry,
    simulationTelemetry,
    createdAt: now,
    updatedAt: now,
  };
}

export function readRecallSimulationTrainingLabels() {
  try {
    const saved = JSON.parse(window.localStorage.getItem(recallSimulationTrainingStorageKey));
    return Array.isArray(saved?.labels) ? saved.labels.filter((item) => item?.query && item?.expectedAction) : [];
  } catch {
    return [];
  }
}

export function upsertRecallSimulationTrainingLabel(input) {
  const labels = readRecallSimulationTrainingLabels();
  const query = cleanText(input?.query);
  const normalizedQuery = normalizeSimulationQuery(query);
  const ablationMode = normalizedAblationMode(input?.ablationMode) || "normal";
  const index = labels.findIndex(
    (item) => normalizeSimulationQuery(item?.query) === normalizedQuery
      && (normalizedAblationMode(item?.ablationMode) || "normal") === ablationMode,
  );
  const existing = index >= 0 ? labels[index] : null;
  const next = createRecallSimulationTrainingLabel(input, {
    id: existing?.id,
    observationId: existing?.observationId || existing?.observation_id || existing?.id,
    groupKey: existing?.group,
  });
  if (!next) return { status: "invalid", labels, label: null };
  const status = index >= 0 ? "updated" : "added";
  if (index >= 0) {
    next.id = existing.id || next.id;
    next.observationId = existing.observationId || existing.observation_id || next.observationId || next.id;
    next.createdAt = existing.createdAt || next.createdAt;
    labels[index] = next;
  } else {
    labels.push(next);
  }
  window.localStorage.setItem(recallSimulationTrainingStorageKey, JSON.stringify({
    schemaVersion: 3,
    updatedAt: next.updatedAt,
    labels,
  }));
  window.dispatchEvent(new CustomEvent("serein:recall-simulation-training-updated"));
  return { status, labels, label: next };
}
