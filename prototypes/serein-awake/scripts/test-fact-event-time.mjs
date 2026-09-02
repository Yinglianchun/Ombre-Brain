import test from "node:test";
import assert from "node:assert/strict";

import { compareFactEventsByEnd, factEventTimeLabel } from "../src/utils/factEventTime.js";

test("same-day Events keep the compact date and time range", () => {
  assert.equal(factEventTimeLabel({
    local_date: "2026-09-01",
    local_end_date: "2026-09-01",
    local_start_time: "10:21",
    local_end_time: "11:59",
  }), "2026-09-01 · 10:21–11:59");
});

test("cross-day Events show both dates", () => {
  assert.equal(factEventTimeLabel({
    local_date: "2026-08-30",
    local_end_date: "2026-09-01",
    local_start_time: "20:21",
    local_end_time: "11:59",
  }), "2026-08-30 20:21 → 2026-09-01 11:59");
});

test("Events are ordered by their landing time", () => {
  const events = [
    {
      item_id: "older-start-later-end",
      local_date: "2026-08-30",
      local_end_date: "2026-09-01",
      local_start_time: "20:21",
      local_end_time: "11:59",
    },
    {
      item_id: "later-start-earlier-end",
      local_date: "2026-08-31",
      local_end_date: "2026-08-31",
      local_start_time: "23:00",
      local_end_time: "23:30",
    },
  ];

  assert.deepEqual(events.sort(compareFactEventsByEnd).map((item) => item.item_id), [
    "older-start-later-end",
    "later-start-earlier-end",
  ]);
});
