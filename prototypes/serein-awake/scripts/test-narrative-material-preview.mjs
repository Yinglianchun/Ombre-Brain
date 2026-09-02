import assert from "node:assert/strict";

import { buildNarrativePreviewFingerprint } from "../server/narrativeMaterialPreview.mjs";


assert.equal(
  buildNarrativePreviewFingerprint({
    narrativeId: "narrative_test",
    revision: 7,
    documentSha256: "a".repeat(64),
    body: "  body text  ",
    materialSnapshotSha256: "b".repeat(64),
  }),
  "be08d28d190d16be6c7941141e6c712f28892e6c9af1fd44169dfa3ccd090e66",
);

console.log("NARRATIVE_MATERIAL_PREVIEW_OK");
