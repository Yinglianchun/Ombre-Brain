import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowLeft, CaretDown, Check, Crosshair, LinkSimple, MagnifyingGlass, Plus, Quotes, X } from "@phosphor-icons/react";

function evidenceKey(item) {
  return `${item?.source_system || ""}:${item?.session_id || ""}:${item?.message_id || ""}`;
}

function formatEvidenceTime(value) {
  if (!value) return "时间未记录";
  const raw = String(value).trim();
  const hasZone = /(?:Z|[+-]\d{2}:?\d{2})$/i.test(raw);
  const normalized = !hasZone && /^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}/.test(raw)
    ? `${raw.replace(" ", "T")}Z`
    : raw;
  const parsed = new Date(normalized);
  if (Number.isNaN(parsed.getTime())) return value;
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(parsed);
}

export function SceneEvidenceEditor({ sceneId, sceneTitle }) {
  const [evidence, setEvidence] = useState([]);
  const [state, setState] = useState("loading");
  const [message, setMessage] = useState("");
  const [pickerOpen, setPickerOpen] = useState(false);
  const [sourceMessages, setSourceMessages] = useState([]);
  const [sourceState, setSourceState] = useState("idle");
  const [selectedIds, setSelectedIds] = useState(() => new Set());
  const [nextBeforeId, setNextBeforeId] = useState(null);
  const [hasMore, setHasMore] = useState(false);
  const [searchDraft, setSearchDraft] = useState("");
  const [activeQuery, setActiveQuery] = useState("");
  const [sourceMode, setSourceMode] = useState("browse");
  const [contextTargetId, setContextTargetId] = useState(null);
  const [returnQuery, setReturnQuery] = useState("");
  const contextTargetRef = useRef(null);

  const loadEvidence = useCallback(async () => {
    if (!sceneId) {
      setState("readonly");
      setEvidence([]);
      return;
    }
    setState("loading");
    setMessage("");
    try {
      const response = await fetch("/__serein/memory/scene-evidence", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sceneId }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.message || payload.error || "原文证据读取失败");
      setEvidence(Array.isArray(payload.evidence_refs) ? payload.evidence_refs : []);
      setState("ready");
    } catch (error) {
      setState("error");
      setMessage(error.message || "原文证据读取失败");
    }
  }, [sceneId]);

  useEffect(() => {
    loadEvidence();
  }, [loadEvidence]);

  useEffect(() => {
    if (sourceMode !== "context" || sourceState !== "ready" || !contextTargetId) return;
    contextTargetRef.current?.scrollIntoView({ block: "center" });
  }, [contextTargetId, sourceMessages, sourceMode, sourceState]);

  const loadSourceMessages = async ({ append = false, query = activeQuery, contextMessageId = 0 } = {}) => {
    if (sourceState === "loading") return;
    const normalizedQuery = String(query || "").trim();
    setActiveQuery(normalizedQuery);
    setSourceState("loading");
    setMessage("");
    try {
      const response = await fetch("/__serein/memory/bridge-source-messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          limit: 40,
          beforeId: append ? nextBeforeId : null,
          query: normalizedQuery,
          contextMessageId,
          contextRadius: 6,
        }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.message || payload.error || "Bridge 原文读取失败");
      const items = Array.isArray(payload.items) ? payload.items : [];
      setSourceMessages((current) => append ? [...current, ...items] : items);
      setNextBeforeId(payload.next_before_id || null);
      setHasMore(Boolean(payload.has_more));
      setSourceMode(payload.mode || (normalizedQuery ? "search" : "browse"));
      setContextTargetId(payload.target_message_id || null);
      setSourceState("ready");
    } catch (error) {
      setSourceState("error");
      setMessage(error.message || "Bridge 原文读取失败");
    }
  };

  const openPicker = () => {
    setPickerOpen(true);
    setSelectedIds(new Set());
    if (!sourceMessages.length) loadSourceMessages();
  };

  const searchSourceMessages = (event) => {
    event.preventDefault();
    setSelectedIds(new Set());
    setReturnQuery("");
    loadSourceMessages({ query: searchDraft });
  };

  const clearSearch = () => {
    setSearchDraft("");
    setSelectedIds(new Set());
    setReturnQuery("");
    loadSourceMessages({ query: "" });
  };

  const locateEvidence = (item) => {
    const query = `#${item.message_id}`;
    setPickerOpen(true);
    setSearchDraft(query);
    setSelectedIds(new Set());
    setReturnQuery("");
    loadSourceMessages({ query });
  };

  const locateSourceMessage = (item) => {
    setReturnQuery(sourceMode === "search" ? activeQuery : "");
    setSelectedIds(new Set());
    loadSourceMessages({ query: "", contextMessageId: item.id });
  };

  const returnFromContext = () => {
    setSearchDraft(returnQuery);
    setSelectedIds(new Set());
    loadSourceMessages({ query: returnQuery });
  };

  const toggleSelection = (id) => {
    setSelectedIds((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else if (next.size < 12) next.add(id);
      return next;
    });
  };

  const bindSelected = async () => {
    if (!selectedIds.size || state === "saving") return;
    if (!window.confirm(`把选中的 ${selectedIds.size} 条 Bridge 原文绑定到「${sceneTitle}」？\n\n原文会写入独立证据索引，不修改 Scene 正文和向量。`)) return;
    setState("saving");
    setMessage("");
    try {
      const response = await fetch("/__serein/memory/bind-scene-evidence", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sceneId,
          selections: [...selectedIds].map((messageId) => ({ messageId, evidenceKind: "primary" })),
        }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.message || payload.error || "原文绑定失败");
      setEvidence(Array.isArray(payload.evidence_refs) ? payload.evidence_refs : evidence);
      setSelectedIds(new Set());
      setPickerOpen(false);
      setState("ready");
      setMessage(payload.bound_count ? `已绑定 ${payload.bound_count} 条原文` : "这些原文已经绑定过了");
    } catch (error) {
      setState("error");
      setMessage(error.message || "原文绑定失败");
    }
  };

  const unbindEvidence = async (item) => {
    const evidenceId = Number.parseInt(item?.id, 10);
    if (!evidenceId || state === "saving") return;
    if (!window.confirm(`取消这条原文与「${sceneTitle}」的绑定？\n\n原文仍保留在 Haven Bridge，Scene 正文也不会改变。`)) return;
    setState("saving");
    setMessage("");
    try {
      const response = await fetch("/__serein/memory/unbind-scene-evidence", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sceneId, evidenceIds: [evidenceId] }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.message || payload.error || "原文解绑失败");
      setEvidence(Array.isArray(payload.evidence_refs) ? payload.evidence_refs : []);
      setState("ready");
      setMessage(payload.unbound_count ? "已取消这条原文绑定" : "这条原文已经取消绑定了");
    } catch (error) {
      setState("error");
      setMessage(error.message || "原文解绑失败");
    }
  };

  return (
    <section className="scene-evidence" aria-labelledby={`scene-evidence-${sceneId || "snapshot"}`}>
      <header className="scene-evidence__header">
        <div>
          <Quotes size={19} weight="light" aria-hidden="true" />
          <h3 id={`scene-evidence-${sceneId || "snapshot"}`}>原文证据</h3>
          <span>{evidence.length}</span>
        </div>
        {sceneId ? (
          <button type="button" onClick={openPicker} disabled={state === "saving"}>
            <Plus size={15} weight="light" aria-hidden="true" />
            绑定原文
          </button>
        ) : <small>快照 Scene 暂不可写回</small>}
      </header>

      {state === "loading" ? <p className="scene-evidence__empty">正在核对证据索引…</p> : null}
      {state !== "loading" && evidence.length ? (
        <div className="scene-evidence__list">
          {evidence.map((item) => (
            <article key={item.id || evidenceKey(item)}>
              <header>
                <div>
                  <span>{item.role === "user" ? "小雨" : "Haven"}</span>
                  <time>{formatEvidenceTime(item.created_at)}</time>
                  <small>{item.evidence_kind === "supporting" ? "补充证据" : item.evidence_kind === "adjacent_context" ? "相邻上下文" : "主要证据"}</small>
                </div>
                {item.source_system === "haven_bridge" && item.message_id ? (
                  <button type="button" onClick={() => locateEvidence(item)} disabled={state === "saving"}>
                    <MagnifyingGlass size={13} weight="light" aria-hidden="true" />
                    定位原文
                  </button>
                ) : null}
              </header>
              <p>{item.content || "这条证据只保存了外部快照引用。"}</p>
            </article>
          ))}
        </div>
      ) : null}
      {state !== "loading" && !evidence.length ? (
        <p className="scene-evidence__empty">这张 Scene 还没有绑定可核对的聊天原文。</p>
      ) : null}

      {pickerOpen ? (
        <div className="scene-evidence-picker">
          <header>
            <div>
              <strong>从 Haven Bridge 选择</strong>
              <span>按消息时间倒序，只列出真实可见的聊天原文。</span>
            </div>
            <button type="button" onClick={() => setPickerOpen(false)}>收起</button>
          </header>
          <form className="scene-evidence-picker__search" onSubmit={searchSourceMessages}>
            <MagnifyingGlass size={15} weight="light" aria-hidden="true" />
            <input
              type="search"
              value={searchDraft}
              placeholder="搜索原文关键词，或输入 #消息ID"
              aria-label="搜索 Haven Bridge 原文"
              onChange={(event) => setSearchDraft(event.target.value)}
            />
            {activeQuery ? (
              <button type="button" aria-label="清除原文搜索" onClick={clearSearch}>
                <X size={13} weight="light" aria-hidden="true" />
                清除
              </button>
            ) : null}
            <button className="is-primary" type="submit" disabled={sourceState === "loading"}>搜索</button>
          </form>
          {sourceMode === "context" ? (
            <div className="scene-evidence-picker__context-status">
              <button type="button" onClick={returnFromContext} disabled={sourceState === "loading"}>
                <ArrowLeft size={13} weight="light" aria-hidden="true" />
                {returnQuery ? "返回搜索结果" : "返回最近原文"}
              </button>
              <span>已定位 #{contextTargetId} · 同一会话前后各 6 条</span>
            </div>
          ) : sourceMode === "search" && activeQuery ? (
            <p className="scene-evidence-picker__search-summary">搜索命中；点“看前后”回到它在原会话中的位置。</p>
          ) : null}
          {sourceMessages.length ? (
            <div className="scene-evidence-picker__list">
              {sourceMessages.map((item) => {
                const id = String(item.id);
                const boundEvidence = evidence.find((evidenceItem) => evidenceKey(evidenceItem) === `haven_bridge:${item.session_id}:${id}`);
                const alreadyBound = Boolean(boundEvidence);
                const selected = selectedIds.has(id);
                const contextTarget = Boolean(item.is_context_target) || String(contextTargetId || "") === id;
                return (
                  <article
                    className={`${selected ? "is-selected" : ""}${alreadyBound ? " is-bound" : ""}${contextTarget ? " is-context-target" : ""}`}
                    key={id}
                    ref={contextTarget ? contextTargetRef : undefined}
                  >
                    <label>
                      <input
                        type="checkbox"
                        checked={alreadyBound || selected}
                        disabled={state === "saving"}
                        onChange={() => alreadyBound ? unbindEvidence(boundEvidence) : toggleSelection(id)}
                      />
                      <span className="scene-evidence-picker__check">
                        {alreadyBound || selected ? <Check size={13} weight="bold" aria-hidden="true" /> : null}
                      </span>
                      <span className="scene-evidence-picker__copy">
                        <span>
                          <strong>{item.role === "user" ? "小雨" : "Haven"}</strong>
                          <time>{formatEvidenceTime(item.created_at)}</time>
                          <small>#{id}</small>
                          {contextTarget ? <small>当前定位</small> : null}
                          {alreadyBound ? <small>已绑定 · 取消勾选可解绑</small> : null}
                        </span>
                        <p>{item.content}</p>
                      </span>
                    </label>
                    {sourceMode !== "context" ? (
                      <button className="scene-evidence-picker__locate" type="button" onClick={() => locateSourceMessage(item)} disabled={sourceState === "loading"}>
                        <Crosshair size={13} weight="light" aria-hidden="true" />
                        看前后
                      </button>
                    ) : null}
                  </article>
                );
              })}
            </div>
          ) : sourceState === "loading" ? (
            <p className="scene-evidence__empty">{activeQuery ? "正在搜索原文…" : "正在翻原文表…"}</p>
          ) : (
            <p className="scene-evidence__empty">{activeQuery ? `没有找到包含「${activeQuery}」的原文。` : "没有读到可绑定的聊天原文。"}</p>
          )}
          <footer>
            {sourceMode === "context" ? (
              <span>目标句已放回原会话时间线</span>
            ) : (
              <button type="button" onClick={() => loadSourceMessages({ append: true })} disabled={!hasMore || sourceState === "loading"}>
                <CaretDown size={14} weight="light" aria-hidden="true" />
                {hasMore ? (activeQuery ? "更早的结果" : "更早的原文") : "已经到底了"}
              </button>
            )}
            <span>已选 {selectedIds.size} 条</span>
            <button className="is-primary" type="button" onClick={bindSelected} disabled={!selectedIds.size || state === "saving"}>
              <LinkSimple size={14} weight="light" aria-hidden="true" />
              {state === "saving" ? "绑定中…" : "确认绑定"}
            </button>
          </footer>
        </div>
      ) : null}

      {message ? <p className={`scene-evidence__message${state === "error" || sourceState === "error" ? " is-error" : ""}`} role={state === "error" || sourceState === "error" ? "alert" : "status"}>{message}</p> : null}
    </section>
  );
}
