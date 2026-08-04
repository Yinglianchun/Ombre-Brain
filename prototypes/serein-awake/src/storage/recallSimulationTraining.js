export const recallSimulationTrainingStorageKey = "serein.basement.recall-simulation-training.v1";

const batchStorageKey = "serein.basement.recall-simulation-training-batch.v1";
const validActions = new Set(["skip", "recall"]);

const cleanText = (value) => String(value || "").trim();

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
  const actionVerdict = observedAction === expectedAction
    ? "correct"
    : expectedAction === "recall" ? "missed" : "false_positive";
  const routeVerdict = expectedRoute
    ? observedRoute === expectedRoute ? "correct" : "incorrect"
    : null;
  return {
    id: cleanText(options.id) || randomId("manual-simulation"),
    query,
    expectedAction,
    expectedRoute: expectedRoute || null,
    expectedMemoryIds: uniqueTexts(input?.expectedMemoryIds),
    observedAction: validActions.has(observedAction) ? observedAction : null,
    observedRoute: observedRoute || null,
    actionVerdict,
    routeVerdict,
    group: `manual-simulation:${batchId}`,
    groupBasis: "manual_simulation_batch",
    source: "manual_simulation",
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
  const next = createRecallSimulationTrainingLabel(input);
  if (!next) return { status: "invalid", labels: readRecallSimulationTrainingLabels(), label: null };
  const labels = readRecallSimulationTrainingLabels();
  const index = labels.findIndex((item) => cleanText(item?.query) === next.query);
  const status = index >= 0 ? "updated" : "added";
  if (index >= 0) {
    next.id = labels[index].id || next.id;
    next.createdAt = labels[index].createdAt || next.createdAt;
    labels[index] = next;
  } else {
    labels.push(next);
  }
  window.localStorage.setItem(recallSimulationTrainingStorageKey, JSON.stringify({
    schemaVersion: 1,
    updatedAt: next.updatedAt,
    labels,
  }));
  window.dispatchEvent(new CustomEvent("serein:recall-simulation-training-updated"));
  return { status, labels, label: next };
}
