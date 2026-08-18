import { useEffect, useState } from "react";
import { ArrowClockwise, ArrowCounterClockwise, Check, LinkSimple, Plus, Trash, WarningCircle, X } from "@phosphor-icons/react";
import { MarkdownProjection } from "./MarkdownProjection.jsx";

const statusLabels = {
  pending: "待判断",
  accepted: "已接受",
  rejected: "已拒绝",
  superseded: "已过期",
  all: "全部",
};

const relationLabels = {
  evidenced_by: "彼此印证",
  echoes: "彼此回响",
  follows: "后来发生",
  continues: "延续",
  contrasts_with: "形成对照",
  caused_by: "由此发生",
  resolves: "回应 / 化解",
};

const lifecycleLabels = {
  active: "正在使用",
  needs_review: "Scene 改过，待重验",
  cancelled: "已取消",
  archived: "随 Scene 归档",
  replaced: "已被新关系替代",
};

const emptyDraft = () => ({
  sourceSceneId: "",
  targetSceneId: "",
  relationType: "continues",
  sourceEvidence: "",
  targetEvidence: "",
  reason: "",
  supersedesEdgeId: "",
});

export function BasementRelationshipProposals() {
  const [filter, setFilter] = useState("pending");
  const [state, setState] = useState({ status: "loading", payload: null, error: "" });
  const [reviewingId, setReviewingId] = useState("");
  const [scenePreview, setScenePreview] = useState(null);
  const [edgeActionId, setEdgeActionId] = useState("");
  const [composerOpen, setComposerOpen] = useState(false);
  const [draft, setDraft] = useState(emptyDraft);

  const load = async (nextFilter = filter) => {
    setState((current) => ({ ...current, status: "loading", error: "" }));
    try {
      const [proposalResponse, edgeResponse] = await Promise.all([
        fetch("/__serein/memory/scene-edge-proposals", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ status: nextFilter, limit: 50 }),
        }),
        fetch("/__serein/memory/scene-edges", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: "{}",
        }),
      ]);
      const [proposalPayload, edgePayload] = await Promise.all([
        proposalResponse.json(),
        edgeResponse.json(),
      ]);
      if (!proposalResponse.ok) throw new Error(proposalPayload?.message || proposalPayload?.error || "没有读到关系提案");
      if (!edgeResponse.ok) throw new Error(edgePayload?.message || edgePayload?.error || "没有读到关系边历史");
      setState({ status: "done", payload: { ...proposalPayload, edges: edgePayload.edges || [] }, error: "" });
    } catch (error) {
      setState({ status: "error", payload: null, error: error instanceof Error ? error.message : "没有读到关系提案" });
    }
  };

  useEffect(() => { load(filter); }, [filter]);

  useEffect(() => {
    if (!scenePreview) return undefined;
    const closeOnEscape = (event) => {
      if (event.key === "Escape") setScenePreview(null);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [scenePreview]);

  const proposals = state.payload?.proposals ?? [];
  const edges = state.payload?.edges ?? [];
  const openScenePreview = async (sceneId, fallback = {}) => {
    if (!sceneId) return;
    setScenePreview({
      id: sceneId,
      title: fallback.name || sceneId,
      date: fallback.date || "",
      content: fallback.content || "",
      domain: "",
      status: "loading",
      error: "",
    });
    try {
      const response = await fetch("/__serein/memory/read-scene", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sceneId }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload?.message || payload?.error || "没有读到这张 Scene");
      setScenePreview({
        id: payload.id || sceneId,
        title: payload.metadata?.name || fallback.name || sceneId,
        date: payload.metadata?.date || fallback.date || "",
        content: payload.content || fallback.content || "",
        domain: payload.domain_label || payload.canonical_domain || "",
        status: "done",
        error: "",
      });
    } catch (error) {
      setScenePreview((current) => current?.id === sceneId ? {
        ...current,
        status: current.content ? "done" : "error",
        error: error instanceof Error ? error.message : "没有读到这张 Scene",
      } : current);
    }
  };
  const review = async (proposal, decision) => {
    const verb = decision === "accept" ? "接受" : "拒绝";
    const consequence = decision === "accept"
      ? "接受后会重新核验两端 Scene、hash 和逐字证据，并写入一条正式关系边。"
      : "拒绝只会关闭这条候选，不会改 Scene，也不会写正式关系边。";
    if (!window.confirm(`${verb}「${proposal.anchor_scene?.name || proposal.source_scene_id} → ${proposal.candidate_scene?.name || proposal.target_scene_id}」？\n\n${consequence}`)) return;
    setReviewingId(proposal.proposal_id);
    try {
      const response = await fetch("/__serein/memory/review-scene-edge-proposal", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ proposalId: proposal.proposal_id, decision }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload?.message || payload?.error || "没有完成关系审核");
      await load(filter);
    } catch (error) {
      window.alert(error instanceof Error ? error.message : "没有完成关系审核");
    } finally {
      setReviewingId("");
    }
  };

  const submitManualProposal = async (event) => {
    event.preventDefault();
    setEdgeActionId(draft.supersedesEdgeId || "manual-proposal");
    try {
      const response = await fetch("/__serein/memory/create-scene-edge-proposal", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(draft),
      });
      const payload = await response.json();
      if (!response.ok || !["pending", "already_active"].includes(payload.status)) {
        throw new Error(payload?.message || payload?.reason || payload?.error || "没有创建这条关系提案");
      }
      setDraft(emptyDraft());
      setComposerOpen(false);
      setFilter("pending");
      await load("pending");
    } catch (error) {
      window.alert(error instanceof Error ? error.message : "没有创建这条关系提案");
    } finally {
      setEdgeActionId("");
    }
  };

  const cancelEdge = async (edge) => {
    if (!window.confirm(`取消「${edge.source_title || edge.source} → ${edge.target_title || edge.target}」？\n\n关系历史会保留，也可以在证据仍有效时恢复。`)) return;
    setEdgeActionId(edge.edge_id);
    try {
      const response = await fetch("/__serein/memory/delete-scene-edge", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ edgeId: edge.edge_id, sceneId: edge.source }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload?.message || payload?.reason || payload?.error || "没有取消这条关系边");
      await load(filter);
    } catch (error) {
      window.alert(error instanceof Error ? error.message : "没有取消这条关系边");
    } finally {
      setEdgeActionId("");
    }
  };

  const restoreEdge = async (edge) => {
    setEdgeActionId(edge.edge_id);
    try {
      const response = await fetch("/__serein/memory/restore-scene-edge", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ edgeId: edge.edge_id }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload?.message || payload?.reason || payload?.error || "当前证据不足，不能恢复");
      await load(filter);
    } catch (error) {
      window.alert(error instanceof Error ? error.message : "当前证据不足，不能恢复");
    } finally {
      setEdgeActionId("");
    }
  };

  const beginRelink = (edge) => {
    setDraft({
      ...emptyDraft(),
      sourceSceneId: edge.source,
      targetSceneId: edge.target,
      relationType: edge.relation_type,
      sourceEvidence: edge.source_evidence,
      targetEvidence: edge.target_evidence,
      reason: edge.reason,
      supersedesEdgeId: edge.edge_id,
    });
    setComposerOpen(true);
  };

  return (
    <section className="basement-workbench" aria-labelledby="relationship-proposals-title">
      <header className="basement-workbench__header">
        <div>
          <span className="basement-kicker">先看证据，再让边存在</span>
          <h2 id="relationship-proposals-title">关系提案</h2>
          <p>模型只负责把可能相关的两张 Scene 放到桌上；关系是否成立，由我们逐条判断。</p>
        </div>
        <div className="basement-live-note">
          <i aria-hidden="true" />
          <span>真实提案库 · {state.payload?.count ?? proposals.length} 条</span>
        </div>
      </header>

      <div className="review-toolbar">
        <label>查看
          <select value={filter} onChange={(event) => setFilter(event.target.value)}>
            {Object.entries(statusLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select>
        </label>
        <button type="button" onClick={() => load(filter)} disabled={state.status === "loading"}>
          <ArrowClockwise size={15} className={state.status === "loading" ? "is-spinning" : ""} aria-hidden="true" />刷新
        </button>
        <button type="button" onClick={() => { setDraft(emptyDraft()); setComposerOpen((open) => !open); }}>
          <Plus size={15} aria-hidden="true" />手动提案
        </button>
      </div>

      <aside className="review-boundary-note">
        <WarningCircle size={17} aria-hidden="true" />
        <p><strong>相似，不等于有关联。</strong>接受前会再验两端 active 状态、内容 hash 与逐字证据；只有通过审核的边，才会出现在记忆卡的“关联 Scene”。</p>
      </aside>

      {composerOpen && (
        <form className="relationship-manual-form" onSubmit={submitManualProposal}>
          <header>
            <div>
              <strong>{draft.supersedesEdgeId ? "重新连接这段关系" : "手动提出一段关系"}</strong>
              <p>仍然只生成待审核提案；两段证据都必须逐字存在于当前 Scene 正文。</p>
            </div>
            <button type="button" onClick={() => { setComposerOpen(false); setDraft(emptyDraft()); }}><X size={15} />取消</button>
          </header>
          <div className="relationship-manual-form__grid">
            <label>起点 Scene ID<input value={draft.sourceSceneId} onChange={(event) => setDraft((current) => ({ ...current, sourceSceneId: event.target.value }))} required /></label>
            <label>终点 Scene ID<input value={draft.targetSceneId} onChange={(event) => setDraft((current) => ({ ...current, targetSceneId: event.target.value }))} required /></label>
            <label>关系
              <select value={draft.relationType} onChange={(event) => setDraft((current) => ({ ...current, relationType: event.target.value }))}>
                <option value="continues">延续</option>
                <option value="echoes">彼此回响</option>
                <option value="resolves">回应 / 化解</option>
                <option value="contrasts_with">形成对照</option>
                <option value="evidenced_by">被另一幕印证</option>
              </select>
            </label>
            <label className="is-wide">为什么成立<input value={draft.reason} onChange={(event) => setDraft((current) => ({ ...current, reason: event.target.value }))} minLength={12} required /></label>
            <label>起点逐字证据<textarea rows={3} value={draft.sourceEvidence} onChange={(event) => setDraft((current) => ({ ...current, sourceEvidence: event.target.value }))} required /></label>
            <label>终点逐字证据<textarea rows={3} value={draft.targetEvidence} onChange={(event) => setDraft((current) => ({ ...current, targetEvidence: event.target.value }))} required /></label>
          </div>
          <footer><button className="basement-primary-action" type="submit" disabled={Boolean(edgeActionId)}><Check size={15} />放进待审核箱</button></footer>
        </form>
      )}

      {state.status === "error" && <div className="basement-error" role="alert"><WarningCircle size={19} /><div><strong>没有读到关系提案</strong><p>{state.error}</p></div></div>}
      {state.status !== "error" && state.status !== "loading" && proposals.length === 0 && (
        <div className="basement-empty-state"><LinkSimple size={22} weight="light" /><span>这里没有{statusLabels[filter]}提案</span><p>模型找到候选也不会直接改关系图；只有明确审核后才会落边。</p></div>
      )}

      <div className="relationship-proposal-list">
        {proposals.map((proposal) => {
          const source = proposal.anchor_scene || {};
          const target = proposal.candidate_scene || {};
          const sourceSceneId = source.scene_id || proposal.anchor_scene_id || proposal.source_scene_id;
          const targetSceneId = target.scene_id || proposal.candidate_scene_id || proposal.target_scene_id;
          const evidenceFor = (sceneId) => {
            if (sceneId === proposal.source_scene_id) return proposal.source_evidence;
            if (sceneId === proposal.target_scene_id) return proposal.target_evidence;
            return "";
          };
          const ready = proposal.review_state === "ready";
          return (
            <article className="relationship-proposal-card" key={proposal.proposal_id}>
              <header>
                <div><span>{relationLabels[proposal.relation_type] || proposal.relation_type}</span><strong>{Math.round(Number(proposal.confidence || 0) * 100)}%</strong></div>
                <em>{statusLabels[proposal.status] || proposal.status}</em>
              </header>

              <div className="relationship-pair">
                <button type="button" aria-haspopup="dialog" onClick={() => openScenePreview(sourceSceneId, source)}>
                  <small>{source.date || "日期未写"}</small><strong>{source.name || sourceSceneId}</strong>
                </button>
                <span aria-hidden="true">→</span>
                <button type="button" aria-haspopup="dialog" onClick={() => openScenePreview(targetSceneId, target)}>
                  <small>{target.date || "日期未写"}</small><strong>{target.name || targetSceneId}</strong>
                </button>
              </div>

              {proposal.reason && <p className="relationship-reason">{proposal.reason}</p>}
              <div className="relationship-evidence-grid">
                <blockquote><span>左侧原句</span>{evidenceFor(sourceSceneId) || "没有逐字证据"}</blockquote>
                <blockquote><span>右侧原句</span>{evidenceFor(targetSceneId) || "没有逐字证据"}</blockquote>
              </div>

              <details className="review-technical-details">
                <summary>展开校验信息</summary>
                <dl>
                  <div><dt>review state</dt><dd>{proposal.review_state || "—"}</dd></div>
                  <div><dt>来源</dt><dd>{proposal.proposal_origin || "automatic"}</dd></div>
                  <div><dt>提出者</dt><dd>{proposal.created_by || "scene_linker"}</dd></div>
                  <div><dt>模型</dt><dd>{proposal.model || "—"}</dd></div>
                  <div><dt>起点 hash</dt><dd>{proposal.anchor_hash || "—"}</dd></div>
                  <div><dt>终点 hash</dt><dd>{proposal.candidate_hash || "—"}</dd></div>
                </dl>
              </details>

              {proposal.status === "pending" && (
                <footer className="review-card-actions">
                  {!ready && <span className="review-state-warning">当前不可接受：{proposal.review_state || "需要重新核验"}</span>}
                  <button type="button" onClick={() => review(proposal, "reject")} disabled={reviewingId === proposal.proposal_id}><X size={15} />拒绝</button>
                  <button type="button" className="basement-primary-action" onClick={() => review(proposal, "accept")} disabled={!ready || reviewingId === proposal.proposal_id}><Check size={15} />接受关系</button>
                </footer>
              )}
            </article>
          );
        })}
      </div>

      <section className="relationship-history" aria-labelledby="relationship-history-title">
        <header>
          <div><span className="basement-kicker">可撤回，也留痕</span><h3 id="relationship-history-title">正式关系与历史</h3></div>
          <strong>{edges.filter((edge) => edge.active).length} 条正在使用 · {edges.length} 条历史</strong>
        </header>
        {edges.length ? (
          <div className="relationship-history__list">
            {edges.map((edge) => (
              <article className={`relationship-history__row${edge.active ? " is-active" : ""}`} key={edge.edge_id}>
                <div>
                  <span>{relationLabels[edge.relation_type] || edge.relation_type}</span>
                  <strong>{edge.source_title || edge.source} → {edge.target_title || edge.target}</strong>
                  <small>{lifecycleLabels[edge.lifecycle_status] || edge.lifecycle_status || (edge.active ? "正在使用" : "已停用")}{edge.deactivation_reason ? ` · ${edge.deactivation_reason}` : ""}</small>
                </div>
                <footer>
                  <button type="button" onClick={() => beginRelink(edge)} disabled={Boolean(edgeActionId)}>重新连接</button>
                  {edge.active ? (
                    <button type="button" onClick={() => cancelEdge(edge)} disabled={edgeActionId === edge.edge_id}><Trash size={14} />取消</button>
                  ) : edge.lifecycle_status !== "replaced" ? (
                    <button type="button" onClick={() => restoreEdge(edge)} disabled={edgeActionId === edge.edge_id}><ArrowCounterClockwise size={14} />验证并恢复</button>
                  ) : null}
                </footer>
              </article>
            ))}
          </div>
        ) : <div className="basement-empty-state"><LinkSimple size={22} weight="light" /><span>还没有正式关系历史</span></div>}
      </section>

      {scenePreview && (
        <div className="basement-scene-preview__veil" role="presentation" onMouseDown={(event) => {
          if (event.target === event.currentTarget) setScenePreview(null);
        }}>
          <article className="basement-scene-preview" role="dialog" aria-modal="true" aria-labelledby="basement-scene-preview-title">
            <button type="button" className="basement-scene-preview__close" aria-label="关闭记忆卡" onClick={() => setScenePreview(null)}>
              <X size={19} weight="light" aria-hidden="true" />
            </button>
            <header>
              <span>SCENE · 只读</span>
              <time>{scenePreview.date || "日期未写"}</time>
              <h3 id="basement-scene-preview-title">{scenePreview.title}</h3>
              <p>{scenePreview.domain || "正在读取线上记忆"}</p>
            </header>
            <div className="basement-scene-preview__body">
              {scenePreview.status === "loading" && !scenePreview.content && <p className="basement-scene-preview__loading">正在把这张记忆拿过来……</p>}
              {scenePreview.status === "error" && <div className="basement-scene-preview__error"><WarningCircle size={16} />{scenePreview.error}</div>}
              {scenePreview.content && <MarkdownProjection content={scenePreview.content} />}
              {scenePreview.error && scenePreview.content && <small className="basement-scene-preview__stale">线上重读失败，暂时显示关系提案随附的只读内容。</small>}
            </div>
            <footer><code>{scenePreview.id}</code><span>Esc 关闭</span></footer>
          </article>
        </div>
      )}
    </section>
  );
}
