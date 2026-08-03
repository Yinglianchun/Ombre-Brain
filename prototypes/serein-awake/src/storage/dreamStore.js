const statusLabels = {
  latent: "潜伏中",
  surfaced: "已浮现",
  forgotten: "已散去",
};

const formatDreamDate = (value, fallback = "") => {
  const source = String(value || fallback || "").trim();
  if (!source) return "日期不详";
  const date = new Date(source);
  if (Number.isNaN(date.getTime())) return source.replaceAll("-", ".");
  return new Intl.DateTimeFormat("zh-CN", {
    timeZone: "Asia/Shanghai",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(date).replaceAll("/", ".");
};

const formatDreamTime = (value) => {
  const date = new Date(String(value || ""));
  if (Number.isNaN(date.getTime())) return "";
  return new Intl.DateTimeFormat("zh-CN", {
    timeZone: "Asia/Shanghai",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(date);
};

const compactDreamText = (value, maxLength = 150) => {
  const text = String(value || "")
    .replace(/^#{1,6}\s+/gm, "")
    .replace(/[*_>`~\[\]()]/g, "")
    .replace(/\s+/g, " ")
    .trim();
  if (!text) return "";
  return text.length > maxLength ? `${text.slice(0, maxLength).trim()}…` : text;
};

function normalizeDream(record) {
  const generatedAt = String(record.generated_at || "").trim();
  return {
    id: String(record.dream_id || "").trim(),
    generatedAt,
    localDate: String(record.local_date || "").trim(),
    aiName: String(record.ai_name || "Haven").trim() || "Haven",
    status: String(record.status || "latent").trim(),
    statusLabel: statusLabels[record.status] || record.status || "潜伏中",
    hasBody: Boolean(record.has_body ?? record.body),
    body: typeof record.body === "string" ? record.body : "",
    dateLabel: formatDreamDate(generatedAt, record.local_date),
    timeLabel: formatDreamTime(generatedAt),
  };
}

async function requestDream(path, body = {}) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.message || payload.error || "梦境没有回应");
  return payload;
}

export async function loadDreams() {
  const payload = await requestDream("/__serein/live/dreams", { limit: 50 });
  const dreams = Array.isArray(payload.records)
    ? payload.records.map(normalizeDream).filter((dream) => dream.id)
    : [];
  const latestReadable = dreams.find((dream) => dream.hasBody);
  if (!latestReadable) return dreams;
  try {
    const detail = await loadDreamDetail(latestReadable.id);
    return dreams.map((dream) => dream.id === detail.id ? detail : dream);
  } catch {
    return dreams;
  }
}

export async function loadDreamDetail(dreamId) {
  const payload = await requestDream("/__serein/live/dream-detail", { dreamId });
  return normalizeDream({ ...payload, has_body: true });
}

export function dreamExcerpt(dream) {
  if (!dream) return "昨夜没有留下梦。";
  if (!dream.hasBody) return "这场梦已经散去了。";
  return compactDreamText(dream.body) || "梦还没有被翻开。";
}
