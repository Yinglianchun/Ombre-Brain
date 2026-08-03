export const RECALL_OBSERVATION_EXPORT_SCHEMA_VERSION = 3;
export const RECALL_OBSERVATION_EXPORT_TYPE = "serein.basement.recall-observation-training-export";

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

function exportObservation(item, review) {
  const group = observationGroup(item);
  const verdict = cleanText(review?.verdict).toLowerCase();
  const routeVerdict = cleanText(review?.routeVerdict).toLowerCase();
  const candidateReviews = review?.candidateReviews && typeof review.candidateReviews === "object"
    ? review.candidateReviews
    : {};
  const candidateJudgments = (Array.isArray(item?.injected) ? item.injected : [])
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
  return {
    id: cleanText(item?.id),
    observation_id: cleanText(item?.id),
    query: cleanQuery(item),
    verdict: verdict || null,
    action_verdict: verdict || null,
    route_verdict: routeVerdict || null,
    expected_route: routeVerdict === "incorrect" ? cleanText(review?.expectedRoute) || null : null,
    observed_action: cleanText(item?.observedAction || item?.action).toLowerCase() || null,
    observed_route: cleanText(item?.route) || null,
    session_id: cleanText(item?.sessionId || item?.session_id) || null,
    created_at: cleanText(item?.createdAt || item?.created_at) || null,
    group: group.key,
    group_basis: group.basis,
    source: cleanText(item?.source) || "unknown",
    review_updated_at: cleanText(review?.updatedAt) || null,
    candidate_judgments: candidateJudgments,
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

export function buildRecallObservationTrainingExport(
  items,
  reviews,
  generatedAt = new Date().toISOString(),
) {
  const observations = (Array.isArray(items) ? items : [])
    .filter((item) => item && cleanText(item.id))
    .map((item) => exportObservation(item, reviews?.[item.id] || null));
  const counts = observations.reduce((summary, observation) => {
    const item = items.find((candidate) => cleanText(candidate?.id) === observation.observation_id);
    const review = reviews?.[observation.observation_id] || null;
    const classification = classifyObservationReview(item, review);
    summary[classification.kind] += 1;
    return summary;
  }, { available: 0, rejected: 0, missing: 0 });
  const alignedReviews = Object.fromEntries(
    observations
      .filter((observation) => reviews?.[observation.observation_id])
      .map((observation) => [observation.observation_id, exportReview(reviews[observation.observation_id])]),
  );
  return {
    schema_version: RECALL_OBSERVATION_EXPORT_SCHEMA_VERSION,
    export_type: RECALL_OBSERVATION_EXPORT_TYPE,
    generated_at: generatedAt,
    review_storage_key: "serein.basement.recall-observation-review.v1",
    privacy: {
      includes: [
        "user_query",
        "human_verdict",
        "separate_action_and_route_verdicts",
        "observed_route_and_action",
        "source_and_time_group",
        "reviewed_candidate_id_rank_score_and_relevance",
      ],
      excludes: ["full_prompt", "developer_context", "injected_memory_body", "additional_context"],
    },
    summary: {
      total_observations: observations.length,
      ...counts,
    },
    observations,
    reviews: alignedReviews,
  };
}
