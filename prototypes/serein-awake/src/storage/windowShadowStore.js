import { defaultWindowShadows } from "../data/windowShadows.js";

function normalizeShadow(shadow) {
  if (
    !shadow
    || typeof shadow.id !== "string"
    || typeof shadow.closedAt !== "string"
    || typeof shadow.dateLabel !== "string"
    || typeof shadow.timeLabel !== "string"
    || typeof shadow.title !== "string"
    || typeof shadow.summary !== "string"
    || typeof shadow.text !== "string"
  ) return null;

  return {
    ...shadow,
    relativeLabel: typeof shadow.relativeLabel === "string"
      ? shadow.relativeLabel
      : `${shadow.dateLabel} ${shadow.timeLabel}`,
    scenes: Array.isArray(shadow.scenes)
      ? shadow.scenes.filter((scene) => (
        scene
        && typeof scene.id === "string"
        && typeof scene.title === "string"
      ))
      : [],
    sourceLabel: typeof shadow.sourceLabel === "string" ? shadow.sourceLabel : "",
    statusLabel: typeof shadow.statusLabel === "string" ? shadow.statusLabel : "",
    documentOwnsTitle: shadow.documentOwnsTitle === true,
  };
}

async function readWindowShadowResponse(response) {
  if (!response.ok) return null;
  const payload = await response.json();
  if (
    !payload
    || typeof payload.snapshotId !== "string"
    || !Array.isArray(payload.shadows)
  ) return null;

  const shadows = payload.shadows.map(normalizeShadow).filter(Boolean);
  return shadows.length ? shadows : null;
}

export async function loadWindowShadows() {
  try {
    const liveResponse = await fetch("/__serein/live/window-shadows", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    const liveShadows = await readWindowShadowResponse(liveResponse);
    if (liveShadows) return liveShadows;
  } catch {}

  try {
    const snapshotResponse = await fetch(`${import.meta.env.BASE_URL}private/window-shadows-snapshot.json`, { cache: "no-store" });
    return await readWindowShadowResponse(snapshotResponse);
  } catch {
    return null;
  }
}

export function readFallbackWindowShadows() {
  return defaultWindowShadows;
}
