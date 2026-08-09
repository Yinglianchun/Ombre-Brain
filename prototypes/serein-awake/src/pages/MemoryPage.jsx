import { useEffect, useMemo, useRef, useState } from "react";
import {
  Archive,
  ArrowCounterClockwise,
  BookOpenText,
  CalendarBlank,
  CaretRight,
  ChatCircleText,
  Check,
  CheckSquare,
  Heart,
  LinkSimple,
  ListBullets,
  MagnifyingGlass,
  PencilSimple,
  Plus,
  ShareNetwork,
  Square,
  Sparkle,
  Trash,
  UserFocus,
  X,
} from "@phosphor-icons/react";
import { defaultAnnotationIdentity } from "../data/memory.js";
import { canonicalDomainPolicies } from "../data/basement.js";
import {
  loadMemorySnapshot,
  readMemoryScenes,
  storeMemoryScenes,
  tombstoneMemoryScenes,
} from "../storage/memoryStore.js";
import { MarkdownProjection } from "../components/MarkdownProjection.jsx";
import { SceneCueEditor } from "../components/SceneCueEditor.jsx";
import { SceneEvidenceEditor } from "../components/SceneEvidenceEditor.jsx";

const relationLabels = {
  continues: {
    outgoing: "后来继续",
    incoming: "接着它发生",
    symmetric: "继续展开",
  },
  echoes: {
    outgoing: "彼此回响",
    incoming: "彼此回响",
    symmetric: "彼此回响",
  },
  resolves: {
    outgoing: "后来落地",
    incoming: "让它落地",
    symmetric: "让它落地",
  },
  contrasts_with: {
    outgoing: "形成对照",
    incoming: "形成对照",
    symmetric: "形成对照",
  },
  evidenced_by: {
    outgoing: "被它印证",
    incoming: "印证了它",
    symmetric: "彼此印证",
  },
};

function relatedScenesFor(scene) {
  if (Array.isArray(scene.relatedScenes) && scene.relatedScenes.length) {
    return scene.relatedScenes;
  }
  return (scene.relatedSceneIds ?? []).map((id) => ({ id, relations: [] }));
}

function relationLabel(relation) {
  const labels = relationLabels[relation.type];
  return labels?.[relation.direction] ?? labels?.symmetric ?? relation.type ?? "有关联";
}

function ombreSourceIdForScene(scene) {
  if (scene?.sourceKind !== "ombre-live-readonly") return "";
  if (typeof scene.canonicalSceneId === "string" && scene.canonicalSceneId) {
    return scene.canonicalSceneId;
  }
  const source = scene.sources?.find((candidate) => (
    typeof candidate?.id === "string" && candidate.id.startsWith("manual_source:")
  ));
  return source ? source.id.slice("manual_source:".length) : "";
}

const memoryTypeLabels = { scene: "Scene", event: "事件", fact: "事实" };

function sceneExcerpt(body) {
  const plain = String(body[0] || "").replace(/[*_`>#-]+/gu, "").replace(/\s+/gu, " ").trim();
  return plain.length <= 108 ? plain : `${plain.slice(0, 107)}…`;
}

function sourceTimeLabel(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value || "").replace("T", " ").slice(0, 16);
  return new Intl.DateTimeFormat("zh-CN", {
    timeZone: "Asia/Shanghai",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(date).replaceAll("/", "-");
}

function FactEventDetail({ item, onClose, onRevised, onStatusChanged, onDeleted }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(() => ({
    title: item.title || "",
    body: item.body || "",
    importance: Number(item.importance) || 1,
  }));
  const [state, setState] = useState("idle");
  const [message, setMessage] = useState("");

  const save = async () => {
    if (!draft.body.trim() || (item.item_type === "event" && !draft.title.trim())) return;
    setState("saving");
    setMessage("");
    try {
      const response = await fetch("/__serein/memory/revise-fact-event", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          itemId: item.item_id,
          ...(draft.title.trim() !== (item.title || "") ? { title: draft.title.trim() } : {}),
          ...(draft.body.trim() !== item.body ? { body: draft.body.trim() } : {}),
          ...(draft.importance !== Number(item.importance) ? { importance: draft.importance } : {}),
        }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok || !payload.item) throw new Error(payload.message || payload.error || "保存失败");
      onRevised(item.item_id, payload.item);
      setEditing(false);
      setState("saved");
      setMessage(payload.status === "superseded" ? "已保存为新的修订版" : "重要度已更新");
    } catch (error) {
      setState("error");
      setMessage(error.message || "保存失败");
    }
  };

  const setItemStatus = async (status) => {
    setState("saving");
    setMessage("");
    try {
      const response = await fetch("/__serein/memory/set-fact-event-status", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ itemId: item.item_id, status }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok || !payload.item) throw new Error(payload.message || payload.error || "操作失败");
      onStatusChanged(item.item_id, payload.item);
    } catch (error) {
      setState("error");
      setMessage(error.message || "操作失败");
    }
  };

  const deleteItem = async () => {
    if (!window.confirm(`永久删除这条${memoryTypeLabels[item.item_type]}？它的历史修订和来源副本也会删除，且无法撤销。Haven Bridge 原始聊天不受影响。`)) return;
    setState("saving");
    setMessage("");
    try {
      const response = await fetch("/__serein/memory/delete-fact-event", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ itemId: item.item_id }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok || !Number(payload.deleted)) throw new Error(payload.message || payload.error || "永久删除失败");
      onDeleted(payload.item_ids || [item.item_id], payload.item_type || item.item_type);
    } catch (error) {
      setState("error");
      setMessage(error.message || "永久删除失败");
    }
  };
  const timeLabel = item.local_start_time === item.local_end_time
    ? item.local_start_time
    : `${item.local_start_time}–${item.local_end_time}`;

  return (
    <div className="scene-detail__content fact-event-detail" key={item.item_id}>
      <button className="scene-detail__close" type="button" aria-label="关闭详情" onClick={onClose}>
        <X size={24} weight="light" aria-hidden="true" />
      </button>
      <header className="scene-detail__header">
        <div className="scene-detail__header-top">
          <time dateTime={item.local_date}>{item.local_date} · {timeLabel}</time>
          <div className="scene-detail__edit-actions">
            {editing ? (
              <>
                <button type="button" onClick={() => setEditing(false)} disabled={state === "saving"}>取消</button>
                <button className="is-primary" type="button" onClick={save} disabled={state === "saving"}>
                  <Check size={14} weight="light" aria-hidden="true" />
                  {state === "saving" ? "保存中…" : "保存修订"}
                </button>
              </>
            ) : (
              <>
                <button type="button" onClick={() => setEditing(true)} disabled={state === "saving"}>
                  <PencilSimple size={14} weight="light" aria-hidden="true" />编辑正文
                </button>
                <button type="button" onClick={() => setItemStatus(item.status === "archived" ? "active" : "archived")} disabled={state === "saving"}>
                  {item.status === "archived"
                    ? <><ArrowCounterClockwise size={14} weight="light" aria-hidden="true" />恢复</>
                    : <><Archive size={14} weight="light" aria-hidden="true" />归档</>}
                </button>
                <button className="is-danger" type="button" onClick={deleteItem} disabled={state === "saving"}>
                  <Trash size={14} weight="light" aria-hidden="true" />删除
                </button>
              </>
            )}
          </div>
        </div>
        {editing && item.item_type === "event" ? (
          <input className="scene-editor__title" value={draft.title} maxLength={160} onChange={(event) => setDraft({ ...draft, title: event.target.value })} />
        ) : <h2>{item.title || item.body}</h2>}
        <div className="scene-detail__meta">
          <span>{memoryTypeLabels[item.item_type]}</span>
          <span><LinkSimple size={15} weight="light" aria-hidden="true" />{item.source_refs?.length || 0} 条原文</span>
          <label className="fact-event-importance">
            重要度
            <select value={draft.importance} disabled={!editing} onChange={(event) => setDraft({ ...draft, importance: Number(event.target.value) })}>
              {[1, 2, 3, 4, 5].map((value) => <option value={value} key={value}>{value}</option>)}
            </select>
          </label>
          {item.status !== "active" ? <span>{item.status === "archived" ? "已归档" : "旧版本"}</span> : null}
        </div>
        {message ? <p className={`fact-event-detail__message${state === "error" ? " is-error" : ""}`} role={state === "error" ? "alert" : "status"}>{message}</p> : null}
      </header>
      {editing ? (
        <div className="scene-editor fact-event-editor">
          <label>
            <span>{item.item_type === "event" ? "经过" : "事实"}</span>
            <textarea rows={item.item_type === "event" ? 9 : 4} maxLength={item.item_type === "event" ? 1600 : 500} value={draft.body} onChange={(event) => setDraft({ ...draft, body: event.target.value })} />
          </label>
          <p>修改正文会留下旧版本，并继续沿用已绑定的原文。</p>
        </div>
      ) : item.item_type === "event" ? (
        <div className="scene-detail__body"><p>{item.body}</p></div>
      ) : null}
      <section className="fact-event-sources">
        <header><h3>原文</h3><span>{item.source_refs?.length || 0}</span></header>
        {(item.source_refs || []).map((source) => (
          <article key={`${source.source_system}:${source.session_id}:${source.message_id}`}>
            <div><strong>{source.role === "user" ? "小雨" : "Haven"}</strong><time>{sourceTimeLabel(source.created_at)}</time></div>
            <p>{source.content || `原文 #${source.message_id}`}</p>
          </article>
        ))}
      </section>
    </div>
  );
}

function SceneDomainEditor({ scene, onSaved }) {
  const currentDomain = canonicalDomainPolicies.find((item) => item.key === scene.bucketDomain);
  const sourceId = ombreSourceIdForScene(scene);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(scene.bucketDomain || "general");
  const [state, setState] = useState("idle");
  const [message, setMessage] = useState("");

  const save = async (event) => {
    event.preventDefault();
    if (!sourceId || state === "saving") return;
    const target = canonicalDomainPolicies.find((item) => item.key === draft);
    if (!target) return;
    if (!window.confirm(`把「${scene.title}」的主域改为「${target.label}」？\n\n这会写回线上 Ombre。`)) return;

    setState("saving");
    setMessage("");
    try {
      const response = await fetch("/__serein/memory/edit-scene-domain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sourceId, domain: draft }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.message || payload.error || "主域保存失败");
      onSaved(draft);
      setEditing(false);
      setState("saved");
      setMessage(payload.changed_count ? "已写回线上" : "线上已经是这个主域");
    } catch (error) {
      setState("error");
      setMessage(error.message || "主域保存失败");
    }
  };

  return (
    <div className={`scene-domain${state === "error" ? " is-error" : ""}`}>
      <span className="scene-domain__label">主域</span>
      {editing ? (
        <form onSubmit={save}>
          <select value={draft} onChange={(event) => setDraft(event.target.value)} disabled={state === "saving"}>
            {canonicalDomainPolicies.map((domain) => (
              <option value={domain.key} key={domain.key}>{domain.label}</option>
            ))}
          </select>
          <button type="button" onClick={() => {
            setDraft(scene.bucketDomain || "general");
            setEditing(false);
            setState("idle");
            setMessage("");
          }} disabled={state === "saving"}>取消</button>
          <button className="is-primary" type="submit" disabled={state === "saving" || draft === scene.bucketDomain}>
            {state === "saving" ? "写入中…" : "保存"}
          </button>
        </form>
      ) : (
        <>
          <strong>{currentDomain?.label || scene.bucketDomain || "未标记"}</strong>
          {sourceId ? (
            <button type="button" onClick={() => {
              setDraft(scene.bucketDomain || "general");
              setEditing(true);
              setState("idle");
              setMessage("");
            }}>
              <PencilSimple size={13} weight="light" aria-hidden="true" />
              修改
            </button>
          ) : <small>只读投影</small>}
        </>
      )}
      {message ? <small role={state === "error" ? "alert" : "status"}>{message}</small> : null}
    </div>
  );
}

export function MemoryPage() {
  const [sceneRecords, setSceneRecords] = useState(readMemoryScenes);
  const [memoryType, setMemoryType] = useState("scene");
  const [factEvents, setFactEvents] = useState({ fact: [], event: [] });
  const [factEventLoadState, setFactEventLoadState] = useState("loading");
  const [factEventError, setFactEventError] = useState("");
  const [query, setQuery] = useState("");
  const [view, setView] = useState("all");
  const [selectedSceneId, setSelectedSceneId] = useState(null);
  const [selectedFactEventId, setSelectedFactEventId] = useState(null);
  const [editingSceneId, setEditingSceneId] = useState(null);
  const [editDraft, setEditDraft] = useState(null);
  const [annotationDraft, setAnnotationDraft] = useState("");
  const [annotationComposerOpen, setAnnotationComposerOpen] = useState(false);
  const [selectionMode, setSelectionMode] = useState(false);
  const [selectedSceneIds, setSelectedSceneIds] = useState(() => new Set());
  const [deleteConfirmationOpen, setDeleteConfirmationOpen] = useState(false);
  const [sceneActionState, setSceneActionState] = useState("idle");
  const [sceneActionMessage, setSceneActionMessage] = useState("");
  const [pendingSourceSceneId, setPendingSourceSceneId] = useState(() => (
    window.localStorage.getItem("serein.memory.open-source-id") || ""
  ));
  const detailRef = useRef(null);

  useEffect(() => {
    storeMemoryScenes(sceneRecords);
  }, [sceneRecords]);

  useEffect(() => {
    let cancelled = false;
    loadMemorySnapshot().then((snapshotScenes) => {
      if (cancelled || !snapshotScenes?.length) return;
      setSceneRecords(snapshotScenes);
      setSelectedSceneId(null);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setFactEventLoadState("loading");
      setFactEventError("");
      try {
        const entries = await Promise.all(["fact", "event"].map(async (type) => {
          const response = await fetch("/__serein/live/fact-events", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ type, status: "all" }),
          });
          const payload = await response.json().catch(() => ({}));
          if (!response.ok) throw new Error(payload.message || payload.error || "读取失败");
          return [type, Array.isArray(payload.items) ? payload.items : []];
        }));
        if (!cancelled) {
          setFactEvents(Object.fromEntries(entries));
          setFactEventLoadState("ready");
        }
      } catch (error) {
        if (!cancelled) {
          setFactEventLoadState("error");
          setFactEventError(error.message || "暂时没有读到事实和事件。");
        }
      }
    };
    load();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    const openFromRecall = (event) => {
      const sourceId = String(event.detail?.sourceId || "").trim();
      if (sourceId) setPendingSourceSceneId(sourceId);
    };
    window.addEventListener("serein:open-memory-scene", openFromRecall);
    return () => window.removeEventListener("serein:open-memory-scene", openFromRecall);
  }, []);

  useEffect(() => {
    if (!pendingSourceSceneId) return;
    const matchingScene = sceneRecords.find((scene) => (
      scene.sources?.some((source) => source.id === `manual_source:${pendingSourceSceneId}`)
    ));
    if (!matchingScene) return;
    setSelectedSceneId(matchingScene.id);
    setPendingSourceSceneId("");
    window.localStorage.removeItem("serein.memory.open-source-id");
  }, [pendingSourceSceneId, sceneRecords]);

  const filteredScenes = useMemo(() => {
    const normalizedQuery = query.trim().toLocaleLowerCase("zh-CN");
    return sceneRecords.filter((scene) => {
      const matchesView = view === "all"
        || (view === "self" && scene.selfAnchor)
        || (view === "favorite" && scene.favorite)
        || (view === "sunken" && scene.status === "已沉底")
        || (view === "emergent" && scene.status === "可浮现");
      const haystack = [
        scene.title,
        scene.excerpt,
        ...scene.body,
        ...scene.annotations.map((annotation) => `${annotation.author} ${annotation.content}`),
      ].join(" ").toLocaleLowerCase("zh-CN");
      return matchesView && (!normalizedQuery || haystack.includes(normalizedQuery));
    });
  }, [
    query,
    sceneRecords,
    view,
  ]);

  const activeFactEvents = useMemo(() => {
    if (memoryType === "scene") return [];
    const normalizedQuery = query.trim().toLocaleLowerCase("zh-CN");
    return (factEvents[memoryType] || []).filter((item) => {
      const matchesView = view === "archived" ? item.status === "archived" : item.status === "active";
      const haystack = `${item.title || ""} ${item.body || ""}`.toLocaleLowerCase("zh-CN");
      return matchesView && (!normalizedQuery || haystack.includes(normalizedQuery));
    });
  }, [factEvents, memoryType, query, view]);

  const selectedScene = sceneRecords.find((scene) => scene.id === selectedSceneId) ?? null;
  const selectedFactEvent = memoryType === "scene"
    ? null
    : (factEvents[memoryType] || []).find((item) => item.item_id === selectedFactEventId) ?? null;
  const selectedSceneCount = selectedSceneIds.size;
  const allFilteredScenesSelected = Boolean(filteredScenes.length)
    && filteredScenes.every((scene) => selectedSceneIds.has(scene.id));

  useEffect(() => {
    if (!selectedScene) return undefined;
    const closeOnEscape = (event) => {
      if (event.key === "Escape") setSelectedSceneId(null);
    };
    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [selectedScene]);

  useEffect(() => {
    if (!deleteConfirmationOpen) return undefined;
    const closeOnEscape = (event) => {
      if (event.key === "Escape") setDeleteConfirmationOpen(false);
    };
    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [deleteConfirmationOpen]);

  useEffect(() => {
    setEditingSceneId(null);
    setEditDraft(null);
    setAnnotationDraft("");
    setAnnotationComposerOpen(false);
    detailRef.current?.scrollTo({ top: 0, behavior: "smooth" });
  }, [selectedSceneId]);

  const updateScene = (sceneId, update) => {
    setSceneRecords((current) => current.map((scene) => (
      scene.id === sceneId ? { ...scene, ...update(scene) } : scene
    )));
  };

  const toggleFavorite = (sceneId) => {
    updateScene(sceneId, (scene) => ({ favorite: !scene.favorite }));
  };

  const enterSelectionMode = () => {
    setSelectedSceneId(null);
    setSelectionMode(true);
  };

  const exitSelectionMode = () => {
    setSelectionMode(false);
    setSelectedSceneIds(new Set());
    setDeleteConfirmationOpen(false);
  };

  const toggleSceneSelection = (sceneId) => {
    setSelectedSceneIds((current) => {
      const next = new Set(current);
      if (next.has(sceneId)) next.delete(sceneId);
      else next.add(sceneId);
      return next;
    });
  };

  const selectAllFilteredScenes = () => {
    setSelectedSceneIds((current) => {
      const next = new Set(current);
      filteredScenes.forEach((scene) => next.add(scene.id));
      return next;
    });
  };

  const updateSelectedSceneStatus = async (status) => {
    if (!selectedSceneCount) return;
    const targets = sceneRecords.filter((scene) => selectedSceneIds.has(scene.id));
    if (targets.some((scene) => !ombreSourceIdForScene(scene) || !scene.sourceUpdatedAt)) {
      setSceneActionState("error");
      setSceneActionMessage("所选内容里有尚未接入德国机写入的 Scene。");
      return;
    }
    setSceneActionState("saving");
    setSceneActionMessage("");
    try {
      const responses = await Promise.all(targets.map(async (scene) => {
        const response = await fetch("/__serein/memory/set-scene-status", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sceneId: ombreSourceIdForScene(scene),
            expectedUpdatedAt: scene.sourceUpdatedAt,
            status: status === "已沉底" ? "archived" : "active",
          }),
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok || !["updated", "unchanged"].includes(payload.status)) {
          throw new Error(payload.message || payload.reason || "Scene 状态修改失败");
        }
        return [scene.id, payload.updated_at || scene.sourceUpdatedAt];
      }));
      const revisions = new Map(responses);
      setSceneRecords((current) => current.map((scene) => (
        revisions.has(scene.id)
          ? { ...scene, status, sourceUpdatedAt: revisions.get(scene.id) }
          : scene
      )));
      setSelectedSceneIds(new Set());
      setSceneActionState("saved");
      setSceneActionMessage(status === "已沉底" ? "已归档" : "已恢复可浮现");
    } catch (error) {
      setSceneActionState("error");
      setSceneActionMessage(error.message || "Scene 状态修改失败");
    }
  };

  const confirmSelectedSceneDeletion = async () => {
    if (!selectedSceneCount) return;
    const deletedIds = new Set(selectedSceneIds);
    const sourceIds = sceneRecords
      .filter((scene) => deletedIds.has(scene.id))
      .map(ombreSourceIdForScene)
      .filter(Boolean);
    if (sourceIds.length !== deletedIds.size) {
      setSceneActionState("error");
      setSceneActionMessage("所选内容里有尚未接入德国机删除的 Scene。");
      setDeleteConfirmationOpen(false);
      return;
    }
    setSceneActionState("saving");
    try {
      const response = await fetch("/__serein/memory/delete-scenes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sceneIds: sourceIds }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok || Number(payload.deleted) !== sourceIds.length) {
        throw new Error(payload.message || payload.error || "Scene 删除未全部完成");
      }
      tombstoneMemoryScenes(deletedIds);
      setSceneRecords((current) => current
        .filter((scene) => !deletedIds.has(scene.id))
        .map((scene) => {
          const relatedScenes = relatedScenesFor(scene)
            .filter((relatedScene) => !deletedIds.has(relatedScene.id));
          return {
            ...scene,
            relatedScenes,
            relatedSceneIds: relatedScenes.map((relatedScene) => relatedScene.id),
            relationCount: relatedScenes.length,
          };
        }));
      setSelectedSceneId(null);
      setSelectedSceneIds(new Set());
      setDeleteConfirmationOpen(false);
      setSceneActionState("saved");
      setSceneActionMessage("已删除");
    } catch (error) {
      setSceneActionState("error");
      setSceneActionMessage(error.message || "Scene 删除失败");
    }
  };

  const beginEditingScene = (scene) => {
    setEditingSceneId(scene.id);
    setEditDraft({
      title: scene.title,
      bodyText: scene.body.join("\n\n"),
    });
    setSceneActionState("idle");
    setSceneActionMessage("");
  };

  const cancelEditingScene = () => {
    setEditingSceneId(null);
    setEditDraft(null);
  };

  const saveEditedScene = async () => {
    if (!selectedScene || !editDraft) return;
    const title = editDraft.title.trim();
    const body = editDraft.bodyText
      .split(/\n\s*\n/)
      .map((paragraph) => paragraph.trim())
      .filter(Boolean);
    const sourceId = ombreSourceIdForScene(selectedScene);
    if (!title || !body.length || !sourceId || !selectedScene.sourceUpdatedAt) return;
    setSceneActionState("saving");
    setSceneActionMessage("");
    try {
      const response = await fetch("/__serein/memory/edit-scene", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sceneId: sourceId,
          expectedUpdatedAt: selectedScene.sourceUpdatedAt,
          title,
          content: body.join("\n\n"),
        }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok || !["updated", "unchanged"].includes(payload.status)) {
        throw new Error(payload.message || payload.reason || "Scene 保存失败");
      }
      updateScene(selectedScene.id, () => ({
        title,
        excerpt: sceneExcerpt(body),
        body,
        sourceUpdatedAt: payload.updated_at || selectedScene.sourceUpdatedAt,
      }));
      cancelEditingScene();
      setSceneActionState("saved");
      setSceneActionMessage(payload.status === "updated" ? "已写回德国机" : "内容没有变化");
    } catch (error) {
      setSceneActionState("error");
      setSceneActionMessage(error.message || "Scene 保存失败");
    }
  };

  const deleteAnnotation = (sceneId, annotationId) => {
    updateScene(sceneId, (scene) => ({
      annotations: scene.annotations.filter((annotation) => annotation.id !== annotationId),
    }));
  };

  const addAnnotation = (event) => {
    event.preventDefault();
    if (!selectedScene || !annotationDraft.trim()) return;
    const content = annotationDraft.trim();
    updateScene(selectedScene.id, (scene) => ({
      annotations: [
        ...scene.annotations,
        {
          id: `annotation-${scene.id}-${Date.now()}`,
          ...defaultAnnotationIdentity,
          createdAt: new Date().toLocaleString("zh-CN", {
            month: "2-digit",
            day: "2-digit",
            hour: "2-digit",
            minute: "2-digit",
          }),
          content,
        },
      ],
    }));
    setAnnotationDraft("");
    setAnnotationComposerOpen(false);
  };

  const switchMemoryType = (type) => {
    setMemoryType(type);
    setView("all");
    setQuery("");
    setSelectedSceneId(null);
    setSelectedFactEventId(null);
    exitSelectionMode();
  };

  const acceptFactEventRevision = (previousId, revisedItem) => {
    setFactEvents((current) => ({
      ...current,
      [revisedItem.item_type]: current[revisedItem.item_type]
        .filter((item) => item.item_id !== previousId && item.item_id !== revisedItem.item_id)
        .concat(revisedItem),
    }));
    setSelectedFactEventId(revisedItem.item_id);
  };

  const acceptFactEventStatus = (previousId, updatedItem) => {
    setFactEvents((current) => ({
      ...current,
      [updatedItem.item_type]: current[updatedItem.item_type]
        .filter((item) => item.item_id !== previousId)
        .concat(updatedItem),
    }));
    setSelectedFactEventId(null);
  };

  const acceptFactEventDeletion = (deletedIds, itemType) => {
    const deleted = new Set(deletedIds);
    setFactEvents((current) => ({
      ...current,
      [itemType]: current[itemType].filter((item) => !deleted.has(item.item_id)),
    }));
    setSelectedFactEventId(null);
  };

  const selectedSceneIsEditing = selectedScene && editingSceneId === selectedScene.id;
  const detailOpen = Boolean(selectedScene || selectedFactEvent);

  return (
    <div className="memory-layout">
      <aside className="memory-filters" aria-label="记忆筛选">
        <label className="memory-search">
          <MagnifyingGlass size={18} weight="light" aria-hidden="true" />
          <span className="sr-only">搜索{memoryTypeLabels[memoryType]}</span>
          <input
            type="search"
            value={query}
            placeholder={`搜索${memoryTypeLabels[memoryType]}`}
            onChange={(event) => setQuery(event.target.value)}
          />
          {query ? (
            <button type="button" aria-label="清空搜索" onClick={() => setQuery("")}>
              <X size={14} weight="light" aria-hidden="true" />
            </button>
          ) : null}
        </label>

        <div className="memory-type-switch" aria-label="记忆类型">
          {["scene", "event", "fact"].map((type) => (
            <button
              className={memoryType === type ? "is-active" : ""}
              type="button"
              aria-pressed={memoryType === type}
              onClick={() => switchMemoryType(type)}
              key={type}
            >
              {memoryTypeLabels[type]}
            </button>
          ))}
        </div>

        <div className="memory-filter-group">
          <h2>视图</h2>
          {memoryType === "scene" ? (
            <>
          <button
            className={view === "all" ? "is-active" : ""}
            type="button"
            aria-pressed={view === "all"}
            onClick={() => setView("all")}
          >
            <ListBullets size={17} weight="light" aria-hidden="true" />
            <span>全部 Scene</span>
            <small>{sceneRecords.length}</small>
          </button>
          <button
            className={view === "self" ? "is-active" : ""}
            type="button"
            aria-pressed={view === "self"}
            onClick={() => setView("self")}
          >
            <UserFocus size={17} weight="light" aria-hidden="true" />
            <span>自我</span>
            <small>{sceneRecords.filter((scene) => scene.selfAnchor).length}</small>
          </button>
          <button
            className={view === "favorite" ? "is-active" : ""}
            type="button"
            aria-pressed={view === "favorite"}
            onClick={() => setView("favorite")}
          >
            <Heart size={17} weight="light" aria-hidden="true" />
            <span>舍不得丢的</span>
            <small>{sceneRecords.filter((scene) => scene.favorite).length}</small>
          </button>
          <button
            className={view === "sunken" ? "is-active" : ""}
            type="button"
            aria-pressed={view === "sunken"}
            onClick={() => setView("sunken")}
          >
            <Archive size={17} weight="light" aria-hidden="true" />
            <span>已沉底</span>
            <small>{sceneRecords.filter((scene) => scene.status === "已沉底").length}</small>
          </button>
          <button
            className={view === "emergent" ? "is-active" : ""}
            type="button"
            aria-pressed={view === "emergent"}
            onClick={() => setView("emergent")}
          >
            <Sparkle size={17} weight="light" aria-hidden="true" />
            <span>可浮现</span>
            <small>{sceneRecords.filter((scene) => scene.status === "可浮现").length}</small>
          </button>
            </>
          ) : (
            <>
              <button className={view === "all" ? "is-active" : ""} type="button" aria-pressed={view === "all"} onClick={() => setView("all")}>
                <ListBullets size={17} weight="light" aria-hidden="true" />
                <span>正在使用</span>
                <small>{(factEvents[memoryType] || []).filter((item) => item.status === "active").length}</small>
              </button>
              <button className={view === "archived" ? "is-active" : ""} type="button" aria-pressed={view === "archived"} onClick={() => setView("archived")}>
                <Archive size={17} weight="light" aria-hidden="true" />
                <span>已归档</span>
                <small>{(factEvents[memoryType] || []).filter((item) => item.status === "archived").length}</small>
              </button>
            </>
          )}
        </div>

      </aside>

      <div className={`memory-workspace${detailOpen ? " is-detail-open" : ""}`}>
        <section className="memory-stream" aria-labelledby="memory-title">
          <header className="memory-stream__header">
            <div>
              <h1 id="memory-title">记忆</h1>
              <p>{memoryType === "scene"
                ? "Scene 按发生时间排列，保留它们原来的语气。"
                : memoryType === "event"
                  ? "事件只记录经过，不替我们判断它意味着什么。"
                  : "事实保持短小、独立，并带着它来自哪一刻。"}</p>
            </div>
            <div className="memory-stream__header-actions">
              <span aria-live="polite">{memoryType === "scene" ? filteredScenes.length : activeFactEvents.length} 个{memoryTypeLabels[memoryType]}</span>
              {memoryType === "scene" && !selectionMode ? (
                <button type="button" onClick={enterSelectionMode}>
                  <CheckSquare size={15} weight="light" aria-hidden="true" />
                  批量整理
                </button>
              ) : null}
            </div>
          </header>
          {memoryType === "scene" && sceneActionMessage ? (
            <p className={`memory-action-message${sceneActionState === "error" ? " is-error" : ""}`} role={sceneActionState === "error" ? "alert" : "status"}>
              {sceneActionMessage}
            </p>
          ) : null}

          {memoryType === "scene" && selectionMode ? (
            <div className="memory-batch-toolbar" aria-label="批量整理 Scene">
              <div>
                <button type="button" onClick={exitSelectionMode}>退出批量</button>
                <button
                  type="button"
                  onClick={selectAllFilteredScenes}
                  disabled={allFilteredScenesSelected || !filteredScenes.length}
                >
                  全选当前筛选
                </button>
                <button
                  type="button"
                  onClick={() => setSelectedSceneIds(new Set())}
                  disabled={!selectedSceneCount}
                >
                  清空
                </button>
              </div>
              <div>
                <span aria-live="polite">已选 {selectedSceneCount} 条</span>
                <button
                  type="button"
                  onClick={() => updateSelectedSceneStatus("已沉底")}
                  disabled={!selectedSceneCount || sceneActionState === "saving"}
                >
                  <Archive size={15} weight="light" aria-hidden="true" />
                  归档
                </button>
                <button
                  type="button"
                  onClick={() => updateSelectedSceneStatus("可浮现")}
                  disabled={!selectedSceneCount || sceneActionState === "saving"}
                >
                  <ArrowCounterClockwise size={15} weight="light" aria-hidden="true" />
                  恢复可浮现
                </button>
                <button
                  className="is-danger"
                  type="button"
                  onClick={() => setDeleteConfirmationOpen(true)}
                  disabled={!selectedSceneCount || sceneActionState === "saving"}
                >
                  <Trash size={15} weight="light" aria-hidden="true" />
                  删除
                </button>
              </div>
            </div>
          ) : null}

          {memoryType === "scene" ? (filteredScenes.length ? (
            <ol className={`scene-timeline${selectionMode ? " is-batch-mode" : ""}`}>
              {filteredScenes.map((scene, index) => {
                const isSelected = selectedSceneId === scene.id;
                const isBatchSelected = selectedSceneIds.has(scene.id);
                return (
                  <li
                    className={[
                      "scene-entry",
                      isSelected ? "is-selected" : "",
                      isBatchSelected ? "is-batch-selected" : "",
                    ].filter(Boolean).join(" ")}
                    key={scene.id}
                    style={{ "--scene-index": index }}
                  >
                    {selectionMode ? (
                      <span className="scene-entry__selection-marker" aria-hidden="true">
                        {isBatchSelected
                          ? <CheckSquare size={17} weight="fill" />
                          : <Square size={17} weight="light" />}
                      </span>
                    ) : (
                      <span className="scene-entry__marker" aria-hidden="true" />
                    )}
                    <button
                      className="scene-entry__button"
                      type="button"
                      aria-pressed={selectionMode ? isBatchSelected : isSelected}
                      aria-controls={selectionMode ? undefined : "scene-detail"}
                      onClick={() => {
                        if (selectionMode) toggleSceneSelection(scene.id);
                        else setSelectedSceneId(scene.id);
                      }}
                    >
                      <time dateTime={scene.date}>{scene.date}</time>
                      <span className="scene-entry__title">
                        <strong>{scene.title}</strong>
                        {scene.favorite ? <Heart size={15} weight="fill" aria-label="已收藏" /> : null}
                      </span>
                      <p className="scene-entry__excerpt">{scene.excerpt}</p>
                      <span className="scene-entry__meta" aria-label="Scene 信息">
                        <span><LinkSimple size={15} weight="light" aria-hidden="true" />{scene.sourceCount} 条原文</span>
                        <span><ShareNetwork size={15} weight="light" aria-hidden="true" />{scene.relationCount} 个关联</span>
                        <span>
                          {scene.status === "已沉底"
                            ? <Archive size={15} weight="light" aria-hidden="true" />
                            : <Sparkle size={15} weight="light" aria-hidden="true" />}
                          {scene.status}
                        </span>
                      </span>
                      {!selectionMode ? (
                        <CaretRight className="scene-entry__chevron" size={18} weight="light" aria-hidden="true" />
                      ) : null}
                    </button>
                  </li>
                );
              })}
            </ol>
          ) : (
            <div className="memory-empty" role="status">
              <MagnifyingGlass size={24} weight="light" aria-hidden="true" />
              <h2>没有找到这一段</h2>
              <p>换一个词，或者把时间放宽一点。</p>
              <button
                type="button"
                onClick={() => {
                  setQuery("");
                  setView("all");
                }}
              >
                查看全部 Scene
              </button>
            </div>
          )) : factEventLoadState === "loading" ? (
            <div className="memory-empty" role="status"><CalendarBlank size={24} weight="light" /><h2>正在翻这一页</h2></div>
          ) : factEventLoadState === "error" ? (
            <div className="memory-empty" role="alert"><h2>暂时没有读到</h2><p>{factEventError}</p></div>
          ) : activeFactEvents.length ? (
            <ol className="scene-timeline fact-event-timeline">
              {activeFactEvents.map((item, index) => {
                const isSelected = selectedFactEventId === item.item_id;
                const timeLabel = item.local_start_time === item.local_end_time
                  ? item.local_start_time
                  : `${item.local_start_time}–${item.local_end_time}`;
                return (
                  <li className={`scene-entry${isSelected ? " is-selected" : ""}`} key={item.item_id} style={{ "--scene-index": index }}>
                    <span className="scene-entry__marker" aria-hidden="true" />
                    <button className="scene-entry__button" type="button" aria-pressed={isSelected} aria-controls="scene-detail" onClick={() => setSelectedFactEventId(item.item_id)}>
                      <time dateTime={item.local_date}>{item.local_date} · {timeLabel}</time>
                      {item.item_type === "event" ? (
                        <span className="scene-entry__title"><strong>{item.title}</strong></span>
                      ) : null}
                      <p className={`scene-entry__excerpt${item.item_type === "fact" ? " is-fact" : ""}`}>{item.body}</p>
                      <span className="scene-entry__meta">
                        <span><LinkSimple size={15} weight="light" aria-hidden="true" />{item.source_refs?.length || 0} 条原文</span>
                        <span>重要度 {item.importance}</span>
                        {item.injection_count ? <span>已注入 {item.injection_count} 次</span> : null}
                      </span>
                      <CaretRight className="scene-entry__chevron" size={18} weight="light" aria-hidden="true" />
                    </button>
                  </li>
                );
              })}
            </ol>
          ) : (
            <div className="memory-empty" role="status">
              <MagnifyingGlass size={24} weight="light" aria-hidden="true" />
              <h2>这一页还是空的</h2>
              <p>换一天、换一个月，或者清空搜索。</p>
              <button type="button" onClick={() => { setQuery(""); setView("all"); }}>查看全部</button>
            </div>
          )}
        </section>

        <button
          className="scene-detail__veil"
          type="button"
          aria-label="关闭记忆详情"
          tabIndex={detailOpen ? 0 : -1}
          onClick={() => { setSelectedSceneId(null); setSelectedFactEventId(null); }}
        />

        <aside
          ref={detailRef}
          className="scene-detail"
          id="scene-detail"
          aria-label="记忆详情"
          aria-hidden={!detailOpen}
          inert={!detailOpen}
        >
          {selectedScene ? (
            <div className="scene-detail__content" key={selectedScene.id}>
              <button
                className="scene-detail__close"
                type="button"
                aria-label="关闭记忆详情"
                onClick={() => setSelectedSceneId(null)}
              >
                <X size={24} weight="light" aria-hidden="true" />
              </button>

              <header className="scene-detail__header">
                <div className="scene-detail__header-top">
                  <time dateTime={selectedScene.date}>{selectedScene.date}</time>
                  <div className="scene-detail__edit-actions">
                    {selectedSceneIsEditing ? (
                      <>
                        <button type="button" onClick={cancelEditingScene} disabled={sceneActionState === "saving"}>取消</button>
                        <button className="is-primary" type="button" onClick={saveEditedScene} disabled={sceneActionState === "saving"}>
                          <Check size={14} weight="light" aria-hidden="true" />
                          {sceneActionState === "saving" ? "保存中…" : "保存"}
                        </button>
                      </>
                    ) : (
                      <button type="button" onClick={() => beginEditingScene(selectedScene)} disabled={!ombreSourceIdForScene(selectedScene) || !selectedScene.sourceUpdatedAt}>
                        <PencilSimple size={14} weight="light" aria-hidden="true" />
                        编辑记忆
                      </button>
                    )}
                  </div>
                </div>
                {selectedSceneIsEditing ? (
                  <input
                    className="scene-editor__title"
                    aria-label="记忆标题"
                    value={editDraft.title}
                    onChange={(event) => setEditDraft((current) => ({ ...current, title: event.target.value }))}
                  />
                ) : (
                  <h2>{selectedScene.title}</h2>
                )}
                <div className="scene-detail__meta" aria-label="Scene 信息">
                  <span><LinkSimple size={15} weight="light" aria-hidden="true" />{selectedScene.sourceCount} 条原文</span>
                  <span><ShareNetwork size={15} weight="light" aria-hidden="true" />{selectedScene.relationCount} 个关联</span>
                  {selectedScene.narrativeRefs.length ? (
                    <span><BookOpenText size={15} weight="light" aria-hidden="true" />{selectedScene.narrativeRefs.length} 个叙事卷引用</span>
                  ) : null}
                  <button
                    className={`scene-detail__favorite${selectedScene.favorite ? " is-active" : ""}`}
                    type="button"
                    aria-pressed={selectedScene.favorite}
                    onClick={() => toggleFavorite(selectedScene.id)}
                  >
                    <Heart size={15} weight={selectedScene.favorite ? "fill" : "light"} aria-hidden="true" />
                    {selectedScene.favorite ? "已收藏" : "收藏"}
                  </button>
                </div>
                {sceneActionMessage ? <p className={`fact-event-detail__message${sceneActionState === "error" ? " is-error" : ""}`}>{sceneActionMessage}</p> : null}
                <SceneDomainEditor
                  key={`scene-domain-${selectedScene.id}-${selectedScene.bucketDomain}`}
                  scene={selectedScene}
                  onSaved={(bucketDomain) => updateScene(selectedScene.id, () => ({ bucketDomain }))}
                />
              </header>

              {selectedSceneIsEditing ? (
                <div className="scene-editor">
                  <label>
                    <span>正文</span>
                    <textarea
                      rows={10}
                      value={editDraft.bodyText}
                      onChange={(event) => setEditDraft((current) => ({ ...current, bodyText: event.target.value }))}
                    />
                  </label>
                  <p>用空行分开段落。保存后会写回德国机，并保留上一版。</p>
                </div>
              ) : (
                <MarkdownProjection
                  className="scene-detail__body"
                  content={selectedScene.body}
                />
              )}

              <SceneCueEditor
                key={`scene-cues-${selectedScene.id}`}
                scene={selectedScene}
                onSaved={(cues) => updateScene(selectedScene.id, () => ({ cues }))}
              />

              <SceneEvidenceEditor
                key={`scene-evidence-${selectedScene.id}`}
                sceneId={ombreSourceIdForScene(selectedScene)}
                sceneTitle={selectedScene.title}
              />

              <section className="scene-annotations" aria-labelledby={`scene-annotations-${selectedScene.id}`}>
                <header className="scene-annotations__header">
                  <div>
                    <ChatCircleText size={19} weight="light" aria-hidden="true" />
                    <h3 id={`scene-annotations-${selectedScene.id}`}>记忆注脚</h3>
                    <span>{selectedScene.annotations.length}</span>
                  </div>
                  <button type="button" onClick={() => setAnnotationComposerOpen((current) => !current)}>
                    <Plus size={15} weight="light" aria-hidden="true" />
                    写下注脚
                  </button>
                </header>

                {selectedScene.annotations.length ? (
                  <div className="scene-annotations__list">
                    {selectedScene.annotations.map((annotation) => (
                      <article className="scene-annotation" key={annotation.id}>
                        <header>
                          <div>
                            <strong>{annotation.author}</strong>
                            <span>{annotation.role}</span>
                            {annotation.createdAt ? <time>{annotation.createdAt}</time> : null}
                          </div>
                          <button
                            type="button"
                            aria-label={`删除 ${annotation.author} 的注脚`}
                            onClick={() => deleteAnnotation(selectedScene.id, annotation.id)}
                          >
                            <Trash size={15} weight="light" aria-hidden="true" />
                          </button>
                        </header>
                        <MarkdownProjection
                          className="scene-annotation__content"
                          content={annotation.content}
                        />
                      </article>
                    ))}
                  </div>
                ) : (
                  <p className="scene-annotations__empty">这里还没有注脚。</p>
                )}

                {annotationComposerOpen ? (
                  <form className="scene-annotation-composer" onSubmit={addAnnotation}>
                    <textarea
                      autoFocus
                      rows={4}
                      value={annotationDraft}
                      placeholder="写下你想留在这段记忆旁边的话…"
                      onChange={(event) => setAnnotationDraft(event.target.value)}
                    />
                    <footer>
                      <span>{defaultAnnotationIdentity.author} · {defaultAnnotationIdentity.role}</span>
                      <div>
                        <button type="button" onClick={() => {
                          setAnnotationDraft("");
                          setAnnotationComposerOpen(false);
                        }}>
                          取消
                        </button>
                        <button className="is-primary" type="submit" disabled={!annotationDraft.trim()}>
                          添加注脚
                        </button>
                      </div>
                    </footer>
                  </form>
                ) : null}
              </section>

              <div className="scene-reference-stack">
                <section className="scene-reference-section" aria-labelledby={`scene-related-${selectedScene.id}`}>
                  <h3 id={`scene-related-${selectedScene.id}`}>关联 Scene</h3>
                  {relatedScenesFor(selectedScene).length ? (
                    <div className="scene-related-list">
                      {relatedScenesFor(selectedScene).map((relatedSceneRecord) => {
                      const relatedScene = sceneRecords.find((scene) => scene.id === relatedSceneRecord.id);
                      return relatedScene ? (
                        <div className="scene-related-edge" key={relatedScene.id}>
                          <button
                            className="scene-related-edge__open"
                            type="button"
                            onClick={() => setSelectedSceneId(relatedScene.id)}
                          >
                            <span>{relatedScene.title}</span>
                            {relatedSceneRecord.relations?.length ? (
                              <small>
                                {[...new Set(relatedSceneRecord.relations.map(relationLabel))].join(" · ")}
                              </small>
                            ) : null}
                          </button>
                        </div>
                      ) : null;
                    })}
                    </div>
                  ) : (
                    <p className="scene-reference-section__empty">还没有通过审核的关系边。</p>
                  )}
                </section>

                {selectedScene.narrativeRefs.length ? (
                  <section className="scene-reference-section" aria-labelledby={`scene-narrative-${selectedScene.id}`}>
                    <h3 id={`scene-narrative-${selectedScene.id}`}>被叙事卷引用</h3>
                    <div className="scene-narrative-list">
                      {selectedScene.narrativeRefs.map((reference) => (
                        <div key={reference.id}>
                          <LinkSimple size={16} weight="light" aria-hidden="true" />
                          <span>{reference.roll} · {reference.chapter}</span>
                          <strong>{reference.title}</strong>
                        </div>
                      ))}
                    </div>
                  </section>
                ) : null}
              </div>
            </div>
          ) : selectedFactEvent ? (
            <FactEventDetail
              key={selectedFactEvent.item_id}
              item={selectedFactEvent}
              onClose={() => setSelectedFactEventId(null)}
              onRevised={acceptFactEventRevision}
              onStatusChanged={acceptFactEventStatus}
              onDeleted={acceptFactEventDeletion}
            />
          ) : null}
        </aside>
      </div>

      {deleteConfirmationOpen ? (
        <div className="memory-delete-dialog__veil">
          <section
            className="memory-delete-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="memory-delete-dialog-title"
            aria-describedby="memory-delete-dialog-description"
          >
            <span>DELETE SCENE</span>
            <h2 id="memory-delete-dialog-title">删除这 {selectedSceneCount} 条记忆？</h2>
            <p id="memory-delete-dialog-description">
              删除后，这些 Scene 会从搜索、关联和召回中消失；正式数据层会同时清除 Scene 与 cue 的 embedding。此操作不能在这里恢复。
            </p>
            <div>
              <button type="button" onClick={() => setDeleteConfirmationOpen(false)} disabled={sceneActionState === "saving"}>取消</button>
              <button className="is-danger" type="button" onClick={confirmSelectedSceneDeletion} disabled={sceneActionState === "saving"}>
                {sceneActionState === "saving" ? "删除中…" : `删除 ${selectedSceneCount} 条`}
              </button>
            </div>
          </section>
        </div>
      ) : null}
    </div>
  );
}
