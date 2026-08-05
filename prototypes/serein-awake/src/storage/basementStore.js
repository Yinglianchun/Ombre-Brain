import { canonicalDomainPolicies, semanticRouteSnapshot } from "../data/basement.js";

const draftStorageKey = "serein.basement.semantic-route-draft.v1";
const domainPolicyStorageKey = "serein.basement.domain-policy-draft.v1";
export const recallObservationStorageKey = "serein.basement.recall-observation-review.v1";

const cloneRoutes = (routes) => JSON.parse(JSON.stringify(routes));

const baselineExampleKeys = new Set(
  semanticRouteSnapshot.routes.flatMap((route) => (
    route.utterances.map((item) => `${route.name}\n${String(item.text || "").trim()}`)
  )),
);

function normalizeExample(item, routeName) {
  const legacySource = String(item?.source || "").trim();
  const text = String(item?.text || "").trim();
  const baselineKey = `${routeName}\n${text}`;
  return {
    text,
    role: ["typical", "boundary"].includes(item?.role)
      ? item.role
      : legacySource === "hard_negative" ? "boundary" : "typical",
    origin: ["manual", "online_false_positive", "online_false_negative", "import"].includes(item?.origin)
      ? item.origin
      : legacySource === "historical_false_positive" ? "online_false_positive" : "import",
    status: ["draft", "published", "retired"].includes(item?.status)
      ? item.status
      : baselineExampleKeys.has(baselineKey) ? "published" : "draft",
  };
}

function normalizeRoutes(routes) {
  return cloneRoutes(routes).map((route) => ({
    ...route,
    utterances: Array.isArray(route.utterances)
      ? route.utterances.map((item) => normalizeExample(item, route.name)).filter((item) => item.text)
      : [],
  }));
}

export function readSemanticRouteDraft(snapshot = semanticRouteSnapshot) {
  try {
    const saved = JSON.parse(window.localStorage.getItem(draftStorageKey));
    if (!Array.isArray(saved?.routes)) return normalizeRoutes(snapshot.routes);
    if (saved.baseDatasetVersion !== snapshot.datasetVersion) {
      const currentRoutes = normalizeRoutes(snapshot.routes);
      const drafts = normalizeRoutes(saved.routes).flatMap((route) => (
        route.utterances
          .filter((item) => item.status === "draft")
          .map((item) => ({
            routeName: route.name === "simple_contact" ? "present_chitchat" : route.name,
            item,
          }))
      ));
      for (const { routeName, item } of drafts) {
        const target = currentRoutes.find((route) => route.name === routeName && route.enabled !== false);
        if (target && !currentRoutes.some((route) => route.utterances.some((existing) => existing.text === item.text))) {
          target.utterances.push(item);
        }
      }
      return currentRoutes;
    }
    return normalizeRoutes(saved.routes);
  } catch {
    return normalizeRoutes(snapshot.routes);
  }
}

export function saveSemanticRouteDraft(routes, baseDatasetVersion = semanticRouteSnapshot.datasetVersion) {
  window.localStorage.setItem(draftStorageKey, JSON.stringify({
    schemaVersion: 2,
    baseDatasetVersion,
    updatedAt: new Date().toISOString(),
    routes: normalizeRoutes(routes),
  }));
}

export function clearSemanticRouteDraft(snapshot = semanticRouteSnapshot) {
  window.localStorage.removeItem(draftStorageKey);
  return normalizeRoutes(snapshot.routes);
}

async function readRouteDraftResponse(response) {
  const payload = await response.json();
  if (!response.ok) {
    const error = new Error(payload?.error || "route_draft_request_failed");
    error.draft = payload?.draft || null;
    throw error;
  }
  return payload;
}

export async function readServerSemanticRouteDraft() {
  const response = await fetch("/__serein/gateway/semantic-route-draft");
  return readRouteDraftResponse(response);
}

export function inspectServerSemanticRouteDraft(serverState, publishedDatasetVersion) {
  const draft = serverState?.draft;
  if (!draft) return { status: "none", draft: null };
  const baseDatasetVersion = Number(draft.baseDatasetVersion);
  const datasetVersion = Number(publishedDatasetVersion);
  return {
    status: baseDatasetVersion === datasetVersion ? "current" : "conflict",
    draft,
    baseDatasetVersion,
    datasetVersion,
  };
}

export async function saveServerSemanticRouteDraft(
  routes,
  baseDatasetVersion,
  expectedRevision = null,
) {
  const response = await fetch("/__serein/gateway/semantic-route-draft", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ routes: normalizeRoutes(routes), baseDatasetVersion, expectedRevision }),
  });
  return readRouteDraftResponse(response);
}

export async function clearServerSemanticRouteDraft() {
  const response = await fetch("/__serein/gateway/semantic-route-draft", { method: "DELETE" });
  return readRouteDraftResponse(response);
}

export function appendSemanticRouteDraftExample({
  routeName,
  text,
  role,
  origin,
  routes: currentRoutes,
  baseDatasetVersion = semanticRouteSnapshot.datasetVersion,
}) {
  const normalizedText = String(text || "").trim();
  const routes = Array.isArray(currentRoutes)
    ? normalizeRoutes(currentRoutes)
    : readSemanticRouteDraft();
  if (!normalizedText || !routes.some((route) => route.name === routeName)) return { status: "invalid", routes };
  if (routes.some((route) => route.utterances.some((item) => item.text === normalizedText))) {
    return { status: "duplicate", routes };
  }
  const nextRoutes = routes.map((route) => route.name === routeName ? {
    ...route,
    utterances: [...route.utterances, {
      text: normalizedText,
      role: role === "boundary" ? "boundary" : "typical",
      origin: ["online_false_positive", "online_false_negative"].includes(origin) ? origin : "manual",
      status: "draft",
    }],
  } : route);
  saveSemanticRouteDraft(nextRoutes, baseDatasetVersion);
  return { status: "added", routes: nextRoutes };
}

export function readRecallObservationReviews() {
  try {
    const saved = JSON.parse(window.localStorage.getItem(recallObservationStorageKey));
    return saved && typeof saved === "object" && !Array.isArray(saved) ? saved : {};
  } catch {
    return {};
  }
}

export function saveRecallObservationReviews(reviews) {
  window.localStorage.setItem(recallObservationStorageKey, JSON.stringify(reviews));
}

export function hasDomainPolicyDraft() {
  return window.localStorage.getItem(domainPolicyStorageKey) != null;
}

export function readDomainPolicyDraft(baseline = canonicalDomainPolicies) {
  try {
    const saved = JSON.parse(window.localStorage.getItem(domainPolicyStorageKey));
    if (!Array.isArray(saved)) return cloneRoutes(baseline);
    const savedByKey = new Map(saved.map((item) => [item?.key, item?.policy]));
    return baseline.map((domain) => ({
      ...domain,
      policy: ["normal", "explicit_only", "excluded"].includes(savedByKey.get(domain.key))
        ? savedByKey.get(domain.key)
        : domain.policy,
    }));
  } catch {
    return cloneRoutes(canonicalDomainPolicies);
  }
}

export function saveDomainPolicyDraft(domains) {
  window.localStorage.setItem(domainPolicyStorageKey, JSON.stringify(
    domains.map(({ key, policy }) => ({ key, policy })),
  ));
}

export function clearDomainPolicyDraft(baseline = canonicalDomainPolicies) {
  window.localStorage.removeItem(domainPolicyStorageKey);
  return cloneRoutes(baseline);
}
