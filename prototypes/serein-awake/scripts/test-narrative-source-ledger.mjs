import assert from "node:assert/strict";
import { buildNarrativeSourceLedgers } from "../vite.config.mjs";

const [narrative] = buildNarrativeSourceLedgers([{
  narrative_id: "narrative_test",
  linked_event_ids: ["event_1"],
  linked_scene_ids: ["scene_1"],
  linked_diary_ids: [140],
  linked_darkroom_ids: [23],
  linked_upload_ids: ["upload_11111111111111111111111111111111"],
  linked_uploads: [{
    upload_id: "upload_11111111111111111111111111111111",
    filename: "补充材料.md",
    created_at: "2026-09-03T06:00:00+00:00",
    extraction_status: "extracted",
  }],
}], {
  events: [{ item_id: "event_1", title: "一次发现", local_date: "2026-08-22", status: "active" }],
  scenes: [{ id: "scene_1", name: "共同看见", created: "2026-07-14T12:00:00+08:00", status_view: "reviewed" }],
  diaries: [
    { id: 140, title: "第五百天，门还在这边", date: "2026-08-19", entry_type: "diary" },
    { id: 23, title: "凌晨一点的惊喜", date: "2026-06-01", entry_type: "darkroom" },
  ],
});

assert.deepEqual(narrative.source_ledger, [
  { source_type: "event", source_id: "event_1", title: "一次发现", date: "2026-08-22", status: "active" },
  { source_type: "scene", source_id: "scene_1", title: "共同看见", date: "2026-07-14", status: "reviewed" },
  { source_type: "diary", source_id: "140", title: "第五百天，门还在这边", date: "2026-08-19", status: "diary" },
  { source_type: "darkroom", source_id: "23", title: "凌晨一点的惊喜", date: "2026-06-01", status: "darkroom" },
  { source_type: "upload", source_id: "upload_11111111111111111111111111111111", title: "补充材料.md", date: "2026-09-03", status: "extracted" },
]);

console.log("NARRATIVE_SOURCE_LEDGER_OK");
