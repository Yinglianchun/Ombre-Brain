const noMatchTriggers = new Set([
  "no_match",
  "gateway_no_match",
  "below_threshold",
  "insufficient_margin",
]);

export function resolveBridgeObservationOutcome({ injected = [], hookOutcome = "", trigger = "", routeAction = "" } = {}) {
  if (injected.length || hookOutcome === "injected") return "injected";
  if (noMatchTriggers.has(trigger)) return "no_match";
  return routeAction === "skip" ? "skip" : "skip";
}
