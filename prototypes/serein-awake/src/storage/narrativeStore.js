import { narrativeRolls as fallbackNarrativeRolls } from "../data/narrative.js";

const romanVolumes = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"];
const visualTones = ["paper", "mist", "white", "fog", "ink"];
const visualSizes = ["tall", "medium", "tall", "medium", "tall"];

function splitParagraphs(content) {
  return String(content || "")
    .replace(/\r\n?/g, "\n")
    .trim()
    .split(/\n\s*\n/)
    .map((paragraph) => paragraph.trim())
    .filter(Boolean);
}

function parseTitle(value) {
  const [title, ...subtitleParts] = String(value || "").split(/\s+\/\s+/);
  return {
    title: title?.trim() || "未命名叙事卷",
    subtitle: subtitleParts.join(" / ").trim(),
  };
}

function cleanTableCell(value) {
  return String(value || "")
    .replace(/^`|`$/g, "")
    .replace(/\*\*/g, "")
    .trim();
}

function parseTableRows(section) {
  const rows = String(section || "")
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.startsWith("|"))
    .map((line) => line.split("|").slice(1, -1).map(cleanTableCell));
  if (rows.length < 2) return { headers: [], rows: [] };
  return {
    headers: rows[0],
    rows: rows.slice(2),
  };
}

function parseSceneDateHints(document) {
  const dates = new Map();
  String(document || "")
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.startsWith("|") && !/^\|\s*[-:]+/u.test(line))
    .forEach((line) => {
      const cells = line.split("|").slice(1, -1).map(cleanTableCell);
      const id = cells.join(" ").match(/scene_[A-Za-z0-9_.:-]+/u)?.[0];
      const date = cells.find((cell) => /\d{4}-\d{2}-\d{2}/u.test(cell))?.match(/\d{4}-\d{2}-\d{2}/u)?.[0];
      if (id && date && !dates.has(id)) dates.set(id, date);
    });
  return dates;
}

function parseSourceLedger(document) {
  const section = String(document || "").match(
    /(?:^|\n)##\s+(?:(?:Canonical\s+)?Scene\s+)?来源账\s*\n([\s\S]*?)(?=\n##\s+|$)/iu,
  )?.[1] || "";
  const table = parseTableRows(section);
  const sceneIndex = table.headers.findIndex((header) => /Scene/iu.test(header));
  const purposeIndex = table.headers.findIndex((header) => /用途/u.test(header));
  const dateIndex = table.headers.findIndex((header) => /日期|时间/u.test(header));
  const dateHints = parseSceneDateHints(document);
  if (sceneIndex < 0 || purposeIndex < 0) return [];
  return table.rows
    .filter((cells) => cells.length > Math.max(sceneIndex, purposeIndex))
    .map((cells) => {
      const sceneCell = cells[sceneIndex];
      const id = sceneCell.match(/(?:scene|window)_[A-Za-z0-9_.:-]+/u)?.[0] || sceneCell;
      const title = sceneCell.replace(/`/g, "").replace(id, "").trim() || id;
      const date = dateIndex >= 0 ? cells[dateIndex] : dateHints.get(id) || "";
      return { date, id, title, purpose: cells[purposeIndex] };
    });
}

const sourceTypeLabels = {
  event: "Event",
  scene: "Scene",
  diary: "日记",
  darkroom: "暗房",
  upload: "本地文件",
};

function materialIdsFromSources(sources = []) {
  const ids = { event_ids: [], scene_ids: [], diary_ids: [], darkroom_ids: [], upload_ids: [] };
  const keyByType = { event: "event_ids", scene: "scene_ids", diary: "diary_ids", darkroom: "darkroom_ids", upload: "upload_ids" };
  for (const source of sources) {
    const key = keyByType[source?.type || "scene"];
    if (!key) continue;
    const value = ["diary_ids", "darkroom_ids"].includes(key) ? Number(source.id) : String(source.id || "");
    if (value && !ids[key].some((item) => String(item) === String(value))) ids[key].push(value);
  }
  return ids;
}

function withMaterialIds(roll) {
  const inferred = materialIdsFromSources(roll?.sources);
  return {
    ...roll,
    materialIds: {
      ...inferred,
      ...(roll?.materialIds || {}),
      upload_ids: roll?.materialIds?.upload_ids || inferred.upload_ids,
    },
  };
}

function projectSources(item, fallback) {
  const writtenNotes = new Map(parseSourceLedger(item.full_document).map((source) => [source.id, source]));
  const structured = Array.isArray(item.source_ledger) ? item.source_ledger : [];
  if (structured.length) {
    return structured.map((source) => {
      const id = String(source.source_id || "");
      const note = writtenNotes.get(id);
      const type = String(source.source_type || "scene");
      return {
        type,
        typeLabel: sourceTypeLabels[type] || type,
        id,
        title: String(source.title || note?.title || id),
        date: String(source.date || note?.date || ""),
        purpose: String(note?.purpose || ""),
        status: String(source.status || ""),
      };
    });
  }
  const written = Array.from(writtenNotes.values()).map((source) => ({
    ...source,
    type: "scene",
    typeLabel: "Scene",
    status: "",
  }));
  return written.length ? written : fallback?.sources || [];
}

function projectLiveRoll(item, index, fallback) {
  const parsedTitle = parseTitle(item.title);
  const displayTitle = fallback?.displayTitle || parsedTitle.title;
  const sources = projectSources(item, fallback);
  const paragraphs = splitParagraphs(item.body);
  return {
    ...fallback,
    id: String(item.narrative_id || fallback?.id || `narrative-live-${index + 1}`),
    volume: fallback?.volume || romanVolumes[index] || String(index + 1),
    title: displayTitle,
    subtitle: parsedTitle.subtitle || fallback?.subtitle || "",
    spineTitle: displayTitle,
    spineSubtitle: parsedTitle.subtitle || fallback?.spineSubtitle || "",
    description: String(item.current_status_cue || fallback?.description || ""),
    timeStart: String(item.time_start || fallback?.timeStart || ""),
    timeEnd: String(item.time_end || fallback?.timeEnd || ""),
    sceneCount: Number(item.linked_scene_count || item.linked_scene_ids?.length || fallback?.sceneCount || 0),
    sourceCount: sources.length,
    size: fallback?.size || visualSizes[index % visualSizes.length],
    tone: fallback?.tone || visualTones[index % visualTones.length],
    status: item.lifecycle === "active" ? "仍在生长" : String(item.lifecycle || "已审阅"),
    scope: String(item.scope || fallback?.scope || "arc"),
    projectionStatus: String(item.publication_status || fallback?.projectionStatus || "reviewed"),
    current: String(item.current_status_cue || fallback?.current || ""),
    paragraphs: paragraphs.length ? paragraphs : fallback?.paragraphs || [],
    body: String(item.body || ""),
    sources,
    materialIds: {
      event_ids: (item.direct_linked_event_ids || item.linked_event_ids || []).map(String),
      scene_ids: (item.linked_scene_ids || []).map(String),
      diary_ids: (item.linked_diary_ids || []).map(Number),
      darkroom_ids: (item.linked_darkroom_ids || []).map(Number),
      upload_ids: (item.linked_upload_ids || []).map(String),
    },
    sceneNames: sources.length
      ? sources.map((source) => source.title)
      : fallback?.sceneNames || item.linked_scene_ids || [],
    revision: Number(item.revision || 1),
    documentHash: String(item.document_sha256 || ""),
    sourceKind: "ombre-narrative-live-readonly",
  };
}

export async function previewNarrativeRoll(roll, mode, proposedMaterialIds, proposedBody = "") {
  const response = await fetch("/__serein/narrative-preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      narrativeId: roll.id,
      mode,
      expectedRevision: roll.revision,
      expectedDocumentSha256: roll.documentHash,
      proposedMaterialIds,
      proposedBody,
    }),
  });
  const payload = await response.json().catch(() => ({
    status: "error",
    message: "预览返回了无法读取的内容。",
    writes_performed: [],
  }));
  if (!response.ok || !["ok", "insufficient"].includes(payload?.status)) {
    const error = new Error(payload?.message || payload?.reason || "没有生成这次预览。");
    error.payload = payload;
    throw error;
  }
  return payload;
}

export async function saveNarrativeRollBody(roll, body, preview) {
  const response = await fetch("/__serein/narrative-save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      narrativeId: roll.id,
      body,
      expectedRevision: roll.revision,
      expectedDocumentSha256: roll.documentHash,
      proposedMaterialIds: preview.proposed_material_ids,
      expectedMaterialSnapshotSha256: preview.material_snapshot_sha256,
      previewFingerprint: preview.preview_fingerprint,
    }),
  });
  const payload = await response.json().catch(() => ({
    status: "error",
    message: "保存返回了无法读取的内容。",
  }));
  if (!response.ok || !["created", "updated"].includes(payload?.status)) {
    const error = new Error(payload?.message || payload?.reason || "这次正文没有保存。");
    error.payload = payload;
    throw error;
  }
  return payload;
}

export async function loadNarrativeRolls() {
  try {
    const response = await fetch("/__serein/live/narratives", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    if (!response.ok) return fallbackNarrativeRolls.map(withMaterialIds);
    const payload = await response.json();
    if (payload?.status !== "ok" || !Array.isArray(payload.items) || !payload.items.length) {
      return fallbackNarrativeRolls.map(withMaterialIds);
    }
    const fallbackById = new Map(fallbackNarrativeRolls.map((roll) => [roll.id, roll]));
    return payload.items
      .filter((item) => String(item.body || "").trim())
      .map((item, index) => (
        projectLiveRoll(item, index, fallbackById.get(item.narrative_id))
      ));
  } catch {
    return fallbackNarrativeRolls.map(withMaterialIds);
  }
}

export function readFallbackNarrativeRolls() {
  return fallbackNarrativeRolls.map(withMaterialIds);
}

export async function uploadNarrativeMaterial(file) {
  if (!(file instanceof File) || file.size < 1) throw new Error("请选择一个本地文件。");
  if (file.size > 10 * 1024 * 1024) throw new Error("单个材料不能超过 10 MB。");
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
  }
  const response = await fetch("/__serein/narrative-material-upload", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      filename: file.name,
      contentType: file.type || "application/octet-stream",
      contentBase64: btoa(binary),
    }),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok || payload?.status !== "ok") {
    throw new Error(payload?.message || payload?.reason || "这个文件没有上传成功。");
  }
  return payload;
}
