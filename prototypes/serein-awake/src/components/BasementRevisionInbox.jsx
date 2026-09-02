import { useEffect, useMemo, useState } from "react";
import { Archive, ArrowClockwise, Check, NotePencil, WarningCircle } from "@phosphor-icons/react";

const statusLabels = {
  pending: "待判断",
  dismissed: "已忽略",
  absorbed: "已吸收",
  all: "全部",
};

const sourceLabels = {
  scene: "Scene",
  window_shadow: "窗影",
  material_freshness: "材料时间",
  event_group: "成卷候选",
};

const proposalKindLabels = {
  existing_roll_update: "需要更新",
  new_roll_candidate: "可能成卷",
};

async function requestRevisionInbox(status) {
  const response = await fetch("/__serein/memory/narrative-revisions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status, limit: 100 }),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload?.message || payload?.error || "没有读到修订箱");
  return payload;
}

export function BasementRevisionInbox() {
  const [filter, setFilter] = useState("pending");
  const [state, setState] = useState({ status: "loading", payload: null, error: "" });
  const [editingId, setEditingId] = useState("");
  const [draftDelta, setDraftDelta] = useState("");
  const [reviewNote, setReviewNote] = useState("");
  const [savingId, setSavingId] = useState("");

  const load = async (nextFilter = filter) => {
    setState((current) => ({ ...current, status: "loading", error: "" }));
    try {
      const payload = await requestRevisionInbox(nextFilter);
      setState({ status: "done", payload, error: "" });
    } catch (error) {
      setState({ status: "error", payload: null, error: error instanceof Error ? error.message : "没有读到修订箱" });
    }
  };

  useEffect(() => { load(filter); }, [filter]);

  const items = state.payload?.items ?? [];
  const scan = state.payload?.scan ?? {};
  const groups = useMemo(() => {
    const grouped = new Map();
    items.forEach((item) => {
      const key = item.narrative_id || "unknown";
      if (!grouped.has(key)) grouped.set(key, []);
      grouped.get(key).push(item);
    });
    return [...grouped.entries()];
  }, [items]);

  const openEditor = (item) => {
    setEditingId(item.proposal_id);
    setDraftDelta(item.draft_delta || "");
    setReviewNote(item.review_note || "");
  };

  const openNarrativeRewrite = (item) => {
    if (!String(item.narrative_id || "").startsWith("narrative_")) return;
    window.sessionStorage.setItem("serein:narrative-rewrite-intent", item.narrative_id);
    window.location.hash = "#narrative";
  };

  const review = async (item, action) => {
    const prompts = {
      dismiss: "把这条来源标成“本次不影响这卷”？以后仍可重新打开。",
      reopen: "把这条来源重新放回待判断？",
    };
    if (prompts[action] && !window.confirm(prompts[action])) return;
    setSavingId(item.proposal_id);
    try {
      const response = await fetch("/__serein/memory/review-narrative-revision", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          proposalId: item.proposal_id,
          action,
          draftDelta: action === "save_draft" ? draftDelta : "",
          note: action === "save_draft" ? reviewNote : "",
        }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload?.message || payload?.error || "没有保存这次判断");
      setEditingId("");
      await load(filter);
    } catch (error) {
      window.alert(error instanceof Error ? error.message : "没有保存这次判断");
    } finally {
      setSavingId("");
    }
  };

  return (
    <section className="basement-workbench" aria-labelledby="revision-inbox-title">
      <header className="basement-workbench__header">
        <div>
          <span className="basement-kicker">来源发生变化之后</span>
          <h2 id="revision-inbox-title">修订箱</h2>
          <p>凌晨四点只检查与标记：已有卷看材料时间，可能成卷的 Event 交给外部模型判断。</p>
        </div>
        <div className="basement-live-note">
          <i aria-hidden="true" />
          <span>真实修订队列 · {items.length} 条</span>
        </div>
      </header>

      {scan.last_scan_at && (
        <p className="basement-workbench__scan-note">
          上次扫描 {scan.last_scan_at} · {scan.external_model || "未调用外部模型"} · 正文写入 0
        </p>
      )}

      <div className="review-toolbar">
        <label>查看
          <select value={filter} onChange={(event) => setFilter(event.target.value)}>
            {Object.entries(statusLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select>
        </label>
        <button type="button" onClick={() => load(filter)} disabled={state.status === "loading"}>
          <ArrowClockwise size={15} className={state.status === "loading" ? "is-spinning" : ""} aria-hidden="true" />刷新
        </button>
      </div>

      <aside className="review-boundary-note">
        <WarningCircle size={17} aria-hidden="true" />
        <p><strong>这里只是修订线索，不是证据。</strong>保存差分只会留下一份作者草稿，不会改写或发布叙事卷；正式修订前仍要重读完整来源和当前卷。</p>
      </aside>

      {state.status === "error" && <div className="basement-error" role="alert"><WarningCircle size={19} /><div><strong>没有读到修订箱</strong><p>{state.error}</p></div></div>}

      {state.status !== "error" && state.status !== "loading" && groups.length === 0 && (
        <div className="basement-empty-state">
          <Archive size={22} weight="light" aria-hidden="true" />
          <span>{filter === "pending" ? "现在没有待判断的修订" : `没有${statusLabels[filter]}记录`}</span>
          <p>新来源只有确实命中叙事卷的 authored cue 或已审核锚点时，才会来到这里。</p>
        </div>
      )}

      <div className="revision-group-list">
        {groups.map(([narrativeId, groupItems]) => (
          <section className="revision-group" key={narrativeId}>
            <header>
              <div>
                <span>{groupItems[0]?.proposal_kind === "new_roll_candidate" ? "NEW ROLL CANDIDATE" : "NARRATIVE ROLL"}</span>
                <h3>{groupItems[0]?.narrative_title || narrativeId}</h3>
              </div>
              <small>{groupItems.length} 条来源</small>
            </header>
            {groupItems.map((item) => (
              <article className="revision-card" key={item.proposal_id}>
                <div className="revision-card__meta">
                  <span>{sourceLabels[item.source_type] || item.source_type || "来源"}</span>
                  <time>{item.source_date || "日期未写"}</time>
                  {item.proposal_kind && <b>{proposalKindLabels[item.proposal_kind] || item.proposal_kind}</b>}
                  <em>{statusLabels[item.status] || item.status}</em>
                </div>
                <h4>{item.source_title || item.source_id}</h4>
                {item.source_excerpt && <blockquote>{item.source_excerpt}</blockquote>}
                {item.proposal_kind === "existing_roll_update" && (
                  <p className="revision-card__freshness">
                    卷最后修订：{item.narrative_published_at || "未知"}<br />
                    最新材料：{item.latest_material_at || item.source_date || "未知"}
                  </p>
                )}
                {item.proposal_kind === "new_roll_candidate" && (
                  <details className="review-technical-details">
                    <summary>{item.source_event_ids?.length || 0} 条候选 Event</summary>
                    <p>{item.source_event_ids?.join(" · ")}</p>
                  </details>
                )}
                {(item.matched_anchors?.length ?? 0) > 0 && (
                  <div className="revision-anchor-list">
                    {item.matched_anchors.map((anchor, index) => (
                      <span key={`${anchor.anchor || anchor.source_cue}-${index}`}>
                        {anchor.reason === "authored_scene_cue" ? "cue" : "审核锚点"} · {anchor.source_cue || anchor.anchor}
                      </span>
                    ))}
                  </div>
                )}
                <details className="review-technical-details">
                  <summary>来源校验信息</summary>
                  <dl>
                    <div><dt>来源 ID</dt><dd>{item.source_id}</dd></div>
                    <div><dt>来源 SHA-256</dt><dd>{item.source_sha256 || "—"}</dd></div>
                    <div><dt>基线 revision</dt><dd>{item.baseline_revision ?? "—"}</dd></div>
                    <div><dt>基线 SHA-256</dt><dd>{item.baseline_document_sha256 || "—"}</dd></div>
                  </dl>
                </details>

                {editingId === item.proposal_id ? (
                  <div className="revision-draft-editor">
                    <label>我想怎样改这卷
                      <textarea rows={5} value={draftDelta} onChange={(event) => setDraftDelta(event.target.value)} placeholder="只写这次要补、要改或要撤掉的内容。" />
                    </label>
                    <label>留给下次复核的注记
                      <input value={reviewNote} onChange={(event) => setReviewNote(event.target.value)} placeholder="可选" />
                    </label>
                    <div>
                      <button type="button" onClick={() => setEditingId("")}>取消</button>
                      <button type="button" className="basement-primary-action" disabled={!draftDelta.trim() || savingId === item.proposal_id} onClick={() => review(item, "save_draft")}><Check size={15} />保存差分草稿</button>
                    </div>
                  </div>
                ) : (
                  <footer className="review-card-actions">
                    {item.status === "pending" && <>
                      <button type="button" onClick={() => review(item, "dismiss")} disabled={savingId === item.proposal_id}>本次不影响</button>
                      {item.proposal_kind !== "new_roll_candidate" && <>
                        <button type="button" onClick={() => openEditor(item)}><NotePencil size={15} />写修订草稿</button>
                        <button type="button" className="basement-primary-action" onClick={() => openNarrativeRewrite(item)}><NotePencil size={15} />重写</button>
                      </>}
                    </>}
                    {item.status === "dismissed" && <button type="button" onClick={() => review(item, "reopen")} disabled={savingId === item.proposal_id}>重新打开</button>}
                    {item.status === "absorbed" && <span>已由叙事卷 revision {item.absorbed_revision ?? "—"} 吸收</span>}
                  </footer>
                )}
              </article>
            ))}
          </section>
        ))}
      </div>
    </section>
  );
}
