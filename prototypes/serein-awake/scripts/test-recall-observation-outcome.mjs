import assert from "node:assert/strict";

import { resolveBridgeObservationOutcome } from "../src/recallObservationOutcome.js";

assert.equal(resolveBridgeObservationOutcome({ routeAction: "skip", trigger: "no_match" }), "no_match");
assert.equal(resolveBridgeObservationOutcome({ routeAction: "skip", trigger: "gateway_no_match" }), "no_match");
assert.equal(resolveBridgeObservationOutcome({ routeAction: "skip" }), "skip");
assert.equal(resolveBridgeObservationOutcome({
  injected: [{ id: "scene-a" }],
  routeAction: "skip",
  trigger: "no_match",
}), "injected");

console.log("recall observation outcome checks: PASS");
