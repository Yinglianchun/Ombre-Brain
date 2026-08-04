import { useCallback, useEffect, useMemo, useState } from "react";
import { CaretDown, Check, LinkSimple, Plus, Quotes } from "@phosphor-icons/react";

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

  const boundKeys = useMemo(() => new Set(evidence.map(evidenceKey)), [evidence]);

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

  const loadSourceMessages = async ({ append = false } = {}) => {
    if (sourceState === "loading") return;
    setSourceState("loading");
    setMessage("");
    try {
      const response = await fetch("/__serein/memory/bridge-source-messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ limit: 40, beforeId: append ? nextBeforeId : null }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.message || payload.error || "Bridge 原文读取失败");
      const items = Array.isArray(payload.items) ? payload.items : [];
      setSourceMessages((current) => append ? [...current, ...items] : items);
      setNextBeforeId(payload.next_before_id || null);
      setHasMore(Boolean(payload.has_more));
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
                <span>{item.role === "user" ? "小雨" : "Haven"}</span>
                <time>{formatEvidenceTime(item.created_at)}</time>
                <small>{item.evidence_kind === "supporting" ? "补充证据" : item.evidence_kind === "adjacent_context" ? "相邻上下文" : "主要证据"}</small>
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
          {sourceMessages.length ? (
            <div className="scene-evidence-picker__list">
              {sourceMessages.map((item) => {
                const id = String(item.id);
                const alreadyBound = boundKeys.has(`haven_bridge:${item.session_id}:${id}`);
                const selected = selectedIds.has(id);
                return (
                  <label className={`${selected ? "is-selected" : ""}${alreadyBound ? " is-bound" : ""}`} key={id}>
                    <input
                      type="checkbox"
                      checked={selected}
                      disabled={alreadyBound || state === "saving"}
                      onChange={() => toggleSelection(id)}
                    />
                    <span className="scene-evidence-picker__check">
                      {alreadyBound || selected ? <Check size={13} weight="bold" aria-hidden="true" /> : null}
                    </span>
                    <span className="scene-evidence-picker__copy">
                      <span>
                        <strong>{item.role === "user" ? "小雨" : "Haven"}</strong>
                        <time>{formatEvidenceTime(item.created_at)}</time>
                        {alreadyBound ? <small>已绑定</small> : null}
                      </span>
                      <p>{item.content}</p>
                    </span>
                  </label>
                );
              })}
            </div>
          ) : sourceState === "loading" ? (
            <p className="scene-evidence__empty">正在翻原文表…</p>
          ) : (
            <p className="scene-evidence__empty">没有读到可绑定的聊天原文。</p>
          )}
          <footer>
            <button type="button" onClick={() => loadSourceMessages({ append: true })} disabled={!hasMore || sourceState === "loading"}>
              <CaretDown size={14} weight="light" aria-hidden="true" />
              {hasMore ? "更早的原文" : "已经到底了"}
            </button>
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
