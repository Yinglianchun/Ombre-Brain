import { defaultDiaryEntries } from "../data/diary.js";
import { people } from "../data/awake.js";
import { readLocalPreference } from "./awakeStore.js";

const diaryStorageKey = "serein.diary.entries.v1";
const diarySnapshotStorageKey = "serein.diary.snapshot.v1";
const supplementalSeedIds = new Set(["diary-morning-before-rain"]);

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
  return plain.length <= 92 ? plain : `${plain.slice(0, 91)}…`;
}

function localTime(value) {
  const match = String(value || "").match(/T(\d{2}:\d{2})/u);
  return match?.[1] || "00:00";
}

function projectLiveEntry(entry) {
  const bodyAvailable = entry?.body_available !== false && entry?.locked !== true;
  const body = bodyAvailable ? splitParagraphs(entry.content) : [];
  const author = entry?.author === "user" ? "Rain" : entry?.author === "ai" ? "Haven" : String(entry?.author || "Haven");
  const role = entry?.author === "user" ? "user" : "assistant";
  return {
    id: `diary-vps-${entry.id}`,
    date: String(entry.date || ""),
    time: localTime(entry.created_at),
    author,
    role,
    darkroom: entry.entry_type === "darkroom",
    locked: entry.locked === true,
    unlockAt: String(entry.unlock_at || ""),
    title: String(entry.title || `${entry.date || ""} 的日记`),
    excerpt: bodyAvailable ? excerptFrom(body) : "门还没有开。",
    body,
    references: [],
    comments: (Array.isArray(entry.comments) ? entry.comments : []).map((comment) => ({
      id: `diary-comment-vps-${comment.id}`,
      author: comment.author === "user" ? "Rain" : comment.author === "ai" ? "Haven" : String(comment.author || "Haven"),
      role: comment.author === "user" ? "user" : "assistant",
      createdAt: String(comment.created_at || ""),
      content: String(comment.content || ""),
    })),
    revision: Number(entry.revision || 1),
    sourceId: String(entry.source_id || entry.metadata?.legacy_entry_id || ""),
    emotionTags: Array.isArray(entry.emotion_tags) ? entry.emotion_tags : [],
    sourceKind: "ombre-diary-live-readonly",
  };
}

function mergeSavedDiaryState(liveEntries, savedEntries) {
  const saved = Array.isArray(savedEntries) ? savedEntries : [];
  const savedById = new Map(saved.map((entry) => [entry?.id, entry]));
  const merged = liveEntries.map((entry) => {
    const local = savedById.get(entry.id);
    if (!local) return entry;
    const localComments = Array.isArray(local.comments) ? local.comments : [];
    const serverCommentIds = new Set(entry.comments.map((comment) => comment.id));
    const comments = [
      ...entry.comments,
      ...localComments.filter((comment) => !serverCommentIds.has(comment.id) && !String(comment.id).startsWith("diary-comment-vps-")),
    ];
    if (Number(local.revision || 0) > Number(entry.revision || 0)) {
      return { ...entry, ...local, comments, sourceKind: entry.sourceKind };
    }
    return { ...entry, comments };
  });
  const liveIds = new Set(liveEntries.map((entry) => entry.id));
  return [
    ...merged,
    ...saved.filter((entry) => entry?.id && !liveIds.has(entry.id) && !String(entry.id).startsWith("diary-vps-")),
  ];
}

export function readDiaryUserIdentity() {
  return {
    author: readLocalPreference("serein.awake.name.xiaoyu", people[0].name),
    role: "user",
  };
}

function normalizeEntry(entry, fallbackIdentity) {
  if (!entry || typeof entry !== "object") return null;
  if (
    typeof entry.id !== "string"
    || typeof entry.date !== "string"
    || typeof entry.time !== "string"
    || typeof entry.title !== "string"
    || typeof entry.excerpt !== "string"
    || !Array.isArray(entry.body)
  ) return null;

  return {
    ...entry,
    author: typeof entry.author === "string" && entry.author.trim()
      ? entry.author
      : fallbackIdentity.author,
    role: typeof entry.role === "string" && entry.role.trim()
      ? entry.role
      : fallbackIdentity.role,
    darkroom: typeof entry.darkroom === "boolean"
      ? entry.darkroom
      : fallbackIdentity.darkroom === true,
    body: entry.body.filter((paragraph) => typeof paragraph === "string"),
    references: Array.isArray(entry.references)
      ? entry.references.filter((reference) => (
        reference
        && typeof reference.id === "string"
        && typeof reference.kind === "string"
        && typeof reference.title === "string"
      ))
      : [],
    comments: Array.isArray(entry.comments)
      ? entry.comments.filter((comment) => (
        comment
        && typeof comment.id === "string"
        && typeof comment.author === "string"
        && typeof comment.role === "string"
        && typeof comment.content === "string"
      ))
      : [],
  };
}

export async function loadDiarySnapshot() {
  try {
    let response = await fetch("/__serein/live/diaries", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    let snapshot = response.ok ? await response.json() : null;
    const isLive = snapshot?.status === "ok" && Array.isArray(snapshot.entries);
    if (!isLive) {
      response = await fetch(`${import.meta.env.BASE_URL}private/diary-snapshot.json`, { cache: "no-store" });
      if (!response.ok) return null;
      snapshot = await response.json();
    }
    if (!response.ok) return null;
    if (
      !snapshot
      || typeof snapshot.snapshotId !== "string"
      || !Array.isArray(snapshot.entries)
    ) return null;

    const storedEntries = JSON.parse(window.localStorage.getItem(diaryStorageKey));

    const userIdentity = readDiaryUserIdentity();
    const projectedEntries = isLive
      ? snapshot.entries.map(projectLiveEntry)
      : snapshot.entries;
    const entries = mergeSavedDiaryState(projectedEntries, storedEntries)
      .map((entry) => normalizeEntry(entry, userIdentity))
      .filter(Boolean);
    if (!entries.length) return null;

    window.localStorage.setItem(diarySnapshotStorageKey, snapshot.snapshotId);
    return entries;
  } catch {
    return null;
  }
}

export function readDiaryEntries() {
  try {
    const saved = JSON.parse(window.localStorage.getItem(diaryStorageKey));
    if (!Array.isArray(saved)) return defaultDiaryEntries;
    const userIdentity = readDiaryUserIdentity();
    const entries = saved.map((entry) => {
      const seededEntry = defaultDiaryEntries.find((candidate) => candidate.id === entry?.id);
      return normalizeEntry(entry, seededEntry ?? userIdentity);
    }).filter(Boolean);
    if (!entries.length) return defaultDiaryEntries;

    const entryIds = new Set(entries.map((entry) => entry.id));
    const hasOriginalSeed = entryIds.has("diary-rain-stopped");
    const supplementalEntries = hasOriginalSeed
      ? defaultDiaryEntries.filter((entry) => supplementalSeedIds.has(entry.id) && !entryIds.has(entry.id))
      : [];
    return [...entries, ...supplementalEntries];
  } catch {
    return defaultDiaryEntries;
  }
}

export function storeDiaryEntries(entries) {
  try {
    window.localStorage.setItem(diaryStorageKey, JSON.stringify(entries));
  } catch {
    // Keep the local prototype usable when browser storage is unavailable.
  }
}
