import { createHash } from "node:crypto";


export function buildNarrativePreviewFingerprint({
  narrativeId,
  revision,
  documentSha256,
  body,
  materialSnapshotSha256,
}) {
  const bodyHash = createHash("sha256").update(String(body || "").trim()).digest("hex");
  const payload = [
    "narrative-material-preview-v1",
    String(narrativeId || "").trim(),
    String(Number.parseInt(revision, 10)),
    String(documentSha256 || "").trim().toLowerCase(),
    bodyHash,
    String(materialSnapshotSha256 || "").trim().toLowerCase(),
  ].join("\n");
  return createHash("sha256").update(payload).digest("hex");
}
