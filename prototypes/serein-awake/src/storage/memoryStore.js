const memoryScenesStorageKey = "serein.memory.scene-records.v1";
const retiredMemorySnapshotStorageKey = "serein.memory.snapshot.v1";
const memoryLiveCacheAtStorageKey = "serein.memory.live-cache-at.v1";
const memorySceneTombstonesStorageKey = "serein.memory.scene-tombstones.v1";
const memoryLiveCacheMaxAgeMs = 5 * 60 * 1000;
let memorySnapshotLoadInFlight = null;

function splitParagraphs(content) {
  return String(content || "")
    .replace(/\r\n?/g, "\n")
    .trim()
    .split(/\n\s*\n/)
    .map((paragraph) => paragraph.trim())
    .filter(Boolean);
}

function excerptFrom(body) {
  const plain = String(body[0] || "")
    .replace(/^\s{0,3}#{1,6}\s+/u, "")
    .replace(/[*_`>#-]+/gu, "")
    .replace(/\s+/gu, " ")
    .trim();
  return plain.length <= 108 ? plain : `${plain.slice(0, 107)}…`;
}

function sourceIdFromScene(scene) {
  const source = scene?.sources?.find((candidate) => typeof candidate?.id === "string");
  if (!source) return "";
  const separator = source.id.indexOf(":");
  return separator >= 0 ? source.id.slice(separator + 1) : source.id;
}

function relationDirection(edge, endpoint) {
  if (edge.directionality === "symmetric") return "symmetric";
  return endpoint === "source" ? "outgoing" : "incoming";
}

function mergeLiveMemoryProjection(snapshotScenes, liveProjection, { includeSnapshotOnly = true } = {}) {
  if (!Array.isArray(liveProjection?.scenes) || !liveProjection.scenes.length) {
    return snapshotScenes;
  }

  const snapshotBySourceId = new Map();
  snapshotScenes.forEach((scene) => {
    const sourceId = sourceIdFromScene(scene);
    if (sourceId) snapshotBySourceId.set(sourceId, scene);
  });

  const sceneIdBySourceId = new Map();
  const mergedById = new Map(
    includeSnapshotOnly ? snapshotScenes.map((scene) => [scene.id, scene]) : [],
  );
  liveProjection.scenes.forEach((liveScene) => {
    const sourceId = String(liveScene?.source_id || "").trim();
    if (!sourceId) return;
    const fallback = snapshotBySourceId.get(sourceId);
    const sceneId = fallback?.id || sourceId;
    const body = splitParagraphs(liveScene.content);
    const sources = fallback?.sources?.length
      ? fallback.sources
      : [{ id: `manual_source:${sourceId}`, kind: "Ombre v1 原文", title: `只读来源 · ${sourceId}` }];
    sceneIdBySourceId.set(sourceId, sceneId);
    mergedById.set(sceneId, {
      ...fallback,
      id: sceneId,
      date: String(liveScene.date || fallback?.date || "2026-01-01").slice(0, 10),
      title: String(liveScene.title || fallback?.title || "没有题目的一幕"),
      excerpt: excerptFrom(body) || fallback?.excerpt || "",
      body: body.length ? body : fallback?.body || [],
      author: String(liveScene.author || fallback?.author || "legacy_unknown"),
      annotations: [],
      sources,
      sourceCount: sources.length,
      relatedScenes: fallback?.relatedScenes || [],
      relatedSceneIds: fallback?.relatedSceneIds || [],
      relationCount: fallback?.relationCount || 0,
      narrativeRefs: fallback?.narrativeRefs || [],
      favorite: liveScene.favorite === true,
      status: liveScene.status === "已沉底" ? "已沉底" : "可浮现",
      bucketDomain: String(liveScene.bucket_domain || fallback?.bucketDomain || ""),
      selfAnchor: liveScene.self_anchor === true,
      sourceKind: "ombre-live-readonly",
      canonicalSceneId: sourceId,
      sourceUpdatedAt: String(liveScene.updated_at || "").trim(),
      sourceRevision: [liveScene.updated_at, liveScene.content_hash]
        .map((value) => String(value || "").trim())
        .filter(Boolean)
        .join(":"),
    });
  });

  snapshotScenes.forEach((scene) => {
    const sourceId = sourceIdFromScene(scene);
    if (sourceId && !sceneIdBySourceId.has(sourceId)) sceneIdBySourceId.set(sourceId, scene.id);
    sceneIdBySourceId.set(scene.id, scene.id);
  });

  const relationsByScene = new Map();
  (Array.isArray(liveProjection.edges) ? liveProjection.edges : []).forEach((edge) => {
    if (edge?.active === false) return;
    const sourceSceneId = sceneIdBySourceId.get(String(edge?.source || ""));
    const targetSceneId = sceneIdBySourceId.get(String(edge?.target || ""));
    if (!sourceSceneId || !targetSceneId || sourceSceneId === targetSceneId) return;
    const add = (ownerId, relatedId, endpoint) => {
      const related = relationsByScene.get(ownerId) || new Map();
      const relations = related.get(relatedId) || [];
      relations.push({
        edgeId: String(edge.edge_id || ""),
        type: String(edge.relation_type || "relates_to"),
        direction: relationDirection(edge, endpoint),
        confidence: Number(edge.confidence || 0),
      });
      related.set(relatedId, relations);
      relationsByScene.set(ownerId, related);
    };
    add(sourceSceneId, targetSceneId, "source");
    add(targetSceneId, sourceSceneId, "target");
  });

  for (const [sceneId, related] of relationsByScene) {
    const scene = mergedById.get(sceneId);
    if (!scene) continue;
    const relatedScenes = [...related.entries()].map(([id, relations]) => ({ id, relations }));
    mergedById.set(sceneId, {
      ...scene,
      relatedScenes,
      relatedSceneIds: relatedScenes.map((item) => item.id),
      relationCount: relatedScenes.length,
    });
  }

  return [...mergedById.values()].sort((left, right) => (
    `${right.date}:${right.id}`.localeCompare(`${left.date}:${left.id}`)
  ));
}

function readMemorySceneTombstones() {
  try {
    const saved = JSON.parse(window.localStorage.getItem(memorySceneTombstonesStorageKey));
    return new Set(
      Array.isArray(saved)
        ? saved.filter((sceneId) => typeof sceneId === "string" && sceneId)
        : [],
    );
  } catch {
    return new Set();
  }
}

function normalizeMemoryScene(savedScene, fallbackScene) {
  if (!savedScene || typeof savedScene !== "object") return fallbackScene;
  return {
    ...fallbackScene,
    ...savedScene,
    title: typeof savedScene.title === "string" ? savedScene.title : fallbackScene.title,
    excerpt: typeof savedScene.excerpt === "string" ? savedScene.excerpt : fallbackScene.excerpt,
    body: Array.isArray(savedScene.body) && savedScene.body.every((item) => typeof item === "string")
      ? savedScene.body
      : fallbackScene.body,
    favorite: typeof savedScene.favorite === "boolean" ? savedScene.favorite : fallbackScene.favorite,
    status: savedScene.status === "已沉底" || savedScene.status === "可浮现"
      ? savedScene.status
      : fallbackScene.status,
    bucketDomain: typeof savedScene.bucketDomain === "string"
      ? savedScene.bucketDomain
      : fallbackScene.bucketDomain ?? "",
    selfAnchor: typeof savedScene.selfAnchor === "boolean"
      ? savedScene.selfAnchor
      : fallbackScene.selfAnchor === true,
    annotations: Array.isArray(savedScene.annotations)
      ? savedScene.annotations.filter((annotation) => (
        annotation
        && typeof annotation.id === "string"
        && typeof annotation.author === "string"
        && typeof annotation.role === "string"
        && typeof annotation.content === "string"
      ))
      : fallbackScene.annotations,
    sources: fallbackScene.sources,
    sourceKind: fallbackScene.sourceKind,
    canonicalSceneId: fallbackScene.canonicalSceneId,
    sourceUpdatedAt: fallbackScene.sourceUpdatedAt,
    sourceRevision: fallbackScene.sourceRevision,
    relatedScenes: fallbackScene.relatedScenes ?? [],
    relatedSceneIds: fallbackScene.relatedSceneIds,
    relationCount: fallbackScene.relationCount,
    narrativeRefs: fallbackScene.narrativeRefs,
  };
}

export function readMemoryScenes() {
  try {
    window.localStorage.removeItem?.(retiredMemorySnapshotStorageKey);
    const tombstonedSceneIds = readMemorySceneTombstones();
    const saved = JSON.parse(window.localStorage.getItem(memoryScenesStorageKey));
    if (!Array.isArray(saved)) return [];
    return saved
      .filter((scene) => scene?.sourceKind === "ombre-live-readonly")
      .map((scene) => normalizeMemoryScene(scene, scene))
      .filter((scene) => (
        scene?.id
        && scene?.title
        && Array.isArray(scene?.body)
        && !tombstonedSceneIds.has(scene.id)
      ));
  } catch {
    return [];
  }
}

async function loadMemorySnapshotOnce() {
  try {
    const cachedAt = Number(window.localStorage.getItem(memoryLiveCacheAtStorageKey) || 0);
    const cachedScenes = readMemoryScenes();
    if (cachedScenes.length && Date.now() - cachedAt < memoryLiveCacheMaxAgeMs) return cachedScenes;
  } catch {
    // Continue to the live projection when cache metadata is unavailable.
  }

  let liveProjection = null;
  try {
    const liveResponse = await fetch("/__serein/live/memory-scenes", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sourceIds: [] }),
    });
    if (liveResponse.ok) {
      const payload = await liveResponse.json();
      if (payload?.status === "ok" && Array.isArray(payload.scenes) && payload.scenes.length) {
        liveProjection = payload;
      }
    }
  } catch {
    liveProjection = null;
  }

  if (!liveProjection) return null;

  try {
    const snapshotScenes = [];
    const projectedScenes = mergeLiveMemoryProjection(snapshotScenes, liveProjection, { includeSnapshotOnly: false });
    const saved = JSON.parse(window.localStorage.getItem(memoryScenesStorageKey));
    const savedById = new Map(
      Array.isArray(saved) ? saved.map((scene) => [scene?.id, scene]) : [],
    );
    const explicitSelfAnchorIds = new Set(snapshotScenes
      .filter((scene) => scene.selfAnchor === true)
      .map((scene) => scene.id));

    const tombstonedSceneIds = readMemorySceneTombstones();
    const scenes = projectedScenes
      .filter((scene) => (
        scene.sourceKind === "ombre-live-readonly" || !tombstonedSceneIds.has(scene.id)
      ))
      .map((scene) => {
        const storedScene = savedById.get(scene.id);
        if (scene.sourceKind === "ombre-live-readonly") {
          return normalizeMemoryScene(
            {
              ...scene,
              annotations: storedScene?.annotations,
            },
            scene,
          );
        }
        const storedOverlay = storedScene || null;
        const normalized = !storedScene
          ? normalizeMemoryScene(scene, scene)
          : normalizeMemoryScene(
          {
            ...scene,
            ...storedOverlay,
            bucketDomain: scene.bucketDomain,
            selfAnchor: scene.selfAnchor,
          },
          scene,
        );
        return {
          ...normalized,
          selfAnchor: scene.selfAnchor === true || explicitSelfAnchorIds.has(scene.id),
        };
      })
      .filter((scene) => scene?.id && scene?.title && Array.isArray(scene?.body));
    if (!scenes.length) return null;

    storeMemoryScenes(scenes);
    window.localStorage.setItem(memoryLiveCacheAtStorageKey, String(Date.now()));
    return scenes;
  } catch {
    return null;
  }
}

export function loadMemorySnapshot() {
  if (!memorySnapshotLoadInFlight) {
    memorySnapshotLoadInFlight = loadMemorySnapshotOnce()
      .finally(() => {
        memorySnapshotLoadInFlight = null;
      });
  }
  return memorySnapshotLoadInFlight;
}

export function storeMemoryScenes(sceneRecords) {
  try {
    window.localStorage.setItem(memoryScenesStorageKey, JSON.stringify(sceneRecords));
  } catch {
    // The local prototype remains editable when persistence is unavailable.
  }
}

export function tombstoneMemoryScenes(sceneIds) {
  try {
    const tombstonedSceneIds = readMemorySceneTombstones();
    sceneIds.forEach((sceneId) => {
      if (typeof sceneId === "string" && sceneId) tombstonedSceneIds.add(sceneId);
    });
    window.localStorage.setItem(
      memorySceneTombstonesStorageKey,
      JSON.stringify([...tombstonedSceneIds]),
    );
  } catch {
    // The in-memory deletion still works when persistence is unavailable.
  }
}
