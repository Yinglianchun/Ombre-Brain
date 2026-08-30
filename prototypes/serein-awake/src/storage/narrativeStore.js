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
};

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
    sources,
    sceneNames: sources.length
      ? sources.map((source) => source.title)
      : fallback?.sceneNames || item.linked_scene_ids || [],
    revision: Number(item.revision || 1),
    documentHash: String(item.document_sha256 || ""),
    sourceKind: "ombre-narrative-live-readonly",
  };
}

export async function loadNarrativeRolls() {
  try {
    const response = await fetch("/__serein/live/narratives", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    if (!response.ok) return fallbackNarrativeRolls;
    const payload = await response.json();
    if (payload?.status !== "ok" || !Array.isArray(payload.items) || !payload.items.length) {
      return fallbackNarrativeRolls;
    }
    const fallbackById = new Map(fallbackNarrativeRolls.map((roll) => [roll.id, roll]));
    return payload.items
      .filter((item) => String(item.body || "").trim())
      .map((item, index) => (
        projectLiveRoll(item, index, fallbackById.get(item.narrative_id))
      ));
  } catch {
    return fallbackNarrativeRolls;
  }
}

export function readFallbackNarrativeRolls() {
  return fallbackNarrativeRolls;
}
