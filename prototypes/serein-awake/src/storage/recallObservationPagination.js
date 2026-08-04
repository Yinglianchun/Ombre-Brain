export const recallObservationPageLimits = Object.freeze({
  hook: 100,
  gateway: 50,
});

function cleanId(value) {
  return String(value ?? "").trim();
}

function numericId(value) {
  const parsed = Number.parseInt(cleanId(value), 10);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : 0;
}

export function mergeObservationRows(currentRows, incomingRows) {
  const byId = new Map();
  for (const row of [...(Array.isArray(currentRows) ? currentRows : []), ...(Array.isArray(incomingRows) ? incomingRows : [])]) {
    const id = cleanId(row?.id);
    if (id) byId.set(id, row);
  }
  return [...byId.values()].sort((left, right) => {
    const idDifference = numericId(right?.id) - numericId(left?.id);
    if (idDifference) return idDifference;
    const createdDifference = cleanId(right?.created_at).localeCompare(cleanId(left?.created_at));
    if (createdDifference) return createdDifference;
    return cleanId(right?.id).localeCompare(cleanId(left?.id));
  });
}

export function normalizeObservationPage(payload) {
  const items = Array.isArray(payload?.items) ? payload.items : [];
  const reviewedItems = Array.isArray(payload?.reviewed_items) ? payload.reviewed_items : [];
  const rawBeforeId = payload?.next_before_id ?? payload?.next_cursor;
  const nextBeforeId = numericId(rawBeforeId) || null;
  return {
    items,
    reviewedItems,
    rows: mergeObservationRows(items, reviewedItems),
    hasMore: Boolean(payload?.has_more && nextBeforeId),
    nextBeforeId,
  };
}

export function reviewedObservationIds(reviews, source) {
  const keys = reviews && typeof reviews === "object" && !Array.isArray(reviews)
    ? Object.keys(reviews)
    : [];
  const ids = keys.flatMap((key) => {
    if (source === "hook") {
      const match = key.match(/^hook-(\d+)$/);
      return match ? [Number.parseInt(match[1], 10)] : [];
    }
    return /^\d+$/.test(key) ? [Number.parseInt(key, 10)] : [];
  });
  return [...new Set(ids.filter((id) => Number.isInteger(id) && id > 0))].slice(0, 500);
}
