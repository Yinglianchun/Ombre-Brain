import { defaultMemoryScenes } from "../data/memory.js";

const memoryScenesStorageKey = "serein.memory.scene-records.v1";
const memorySnapshotStorageKey = "serein.memory.snapshot.v1";
const memorySceneTombstonesStorageKey = "serein.memory.scene-tombstones.v1";

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

function mergeLiveMemoryProjection(snapshotScenes, liveProjection) {
  if (!Array.isArray(liveProjection?.scenes) || !liveProjection.scenes.length) {
    return snapshotScenes;
  }

  const snapshotBySourceId = new Map();
  snapshotScenes.forEach((scene) => {
    const sourceId = sourceIdFromScene(scene);
    if (sourceId) snapshotBySourceId.set(sourceId, scene);
  });

  const sceneIdBySourceId = new Map();
  const mergedById = new Map(snapshotScenes.map((scene) => [scene.id, scene]));
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
      annotations: fallback?.annotations || [],
      sources,
      sourceCount: sources.length,
      relatedScenes: fallback?.relatedScenes || [],
      relatedSceneIds: fallback?.relatedSceneIds || [],
      relationCount: fallback?.relationCount || 0,
      narrativeRefs: fallback?.narrativeRefs || [],
      favorite: liveScene.favorite === true || fallback?.favorite === true,
      status: liveScene.status === "已沉底" ? "已沉底" : "可浮现",
      bucketDomain: String(liveScene.bucket_domain || fallback?.bucketDomain || ""),
      selfAnchor: liveScene.self_anchor === true || fallback?.selfAnchor === true,
      sourceKind: "ombre-live-readonly",
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
    const tombstonedSceneIds = readMemorySceneTombstones();
    const saved = JSON.parse(window.localStorage.getItem(memoryScenesStorageKey));
    if (!Array.isArray(saved)) {
      return defaultMemoryScenes.filter((scene) => !tombstonedSceneIds.has(scene.id));
    }
    if (saved.some((scene) => scene?.sourceKind === "serein-import-rehearsal")) {
      return saved
        .map((scene) => normalizeMemoryScene(scene, scene))
        .filter((scene) => (
          scene?.id
          && scene?.title
          && Array.isArray(scene?.body)
          && !tombstonedSceneIds.has(scene.id)
        ));
    }
    const savedById = new Map(saved.map((scene) => [scene?.id, scene]));
    return defaultMemoryScenes
      .filter((scene) => !tombstonedSceneIds.has(scene.id))
      .map((scene) => normalizeMemoryScene(savedById.get(scene.id), scene));
  } catch {
    return defaultMemoryScenes;
  }
}

export async function loadMemorySnapshot() {
  try {
    const response = await fetch(`${import.meta.env.BASE_URL}private/memory-snapshot.json`, { cache: "no-store" });
    if (!response.ok) return null;
    const snapshot = await response.json();
    if (
      !snapshot
      || typeof snapshot.snapshotId !== "string"
      || !Array.isArray(snapshot.scenes)
    ) return null;

    let liveProjection = null;
    try {
      const sourceIds = snapshot.scenes.map(sourceIdFromScene).filter(Boolean);
      const liveResponse = await fetch("/__serein/live/memory-scenes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sourceIds }),
      });
      if (liveResponse.ok) {
        const payload = await liveResponse.json();
        if (payload?.status === "ok") liveProjection = payload;
      }
    } catch {
      liveProjection = null;
    }

    const projectedScenes = mergeLiveMemoryProjection(snapshot.scenes, liveProjection);
    const saved = JSON.parse(window.localStorage.getItem(memoryScenesStorageKey));
    const savedById = new Map(
      Array.isArray(saved) ? saved.map((scene) => [scene?.id, scene]) : [],
    );
    const explicitSelfAnchorIds = new Set(
      snapshot.scenes.filter((scene) => scene.selfAnchor === true).map((scene) => scene.id),
    );

    const tombstonedSceneIds = readMemorySceneTombstones();
    const scenes = projectedScenes
      .filter((scene) => !tombstonedSceneIds.has(scene.id))
      .map((scene) => {
        const storedScene = savedById.get(scene.id);
        const sourceRevisionMatches = (
          scene.sourceKind !== "ombre-live-readonly"
          || (
            Boolean(storedScene?.sourceRevision)
            && storedScene.sourceRevision === scene.sourceRevision
          )
        );
        const storedOverlay = !storedScene
          ? null
          : sourceRevisionMatches
            ? storedScene
            : {
              annotations: storedScene.annotations,
            };
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

    window.localStorage.setItem(
      memorySnapshotStorageKey,
      `${snapshot.snapshotId}:${liveProjection?.snapshotId || "fallback"}`,
    );
    return scenes;
  } catch {
    return null;
  }
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
