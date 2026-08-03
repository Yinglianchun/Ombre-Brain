import { useEffect, useState } from "react";
import { ArrowClockwise, Check, LinkSimple, WarningCircle, X } from "@phosphor-icons/react";
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
  follows: "后来发生",
  continues: "延续",
  contrasts_with: "形成对照",
  caused_by: "由此发生",
  resolves: "回应 / 化解",
};

export function BasementRelationshipProposals() {
  const [filter, setFilter] = useState("pending");
  const [state, setState] = useState({ status: "loading", payload: null, error: "" });
  const [reviewingId, setReviewingId] = useState("");
  const [scenePreview, setScenePreview] = useState(null);

  const load = async (nextFilter = filter) => {
    setState((current) => ({ ...current, status: "loading", error: "" }));
    try {
      const response = await fetch("/__serein/memory/scene-edge-proposals", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status: nextFilter, limit: 50 }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload?.message || payload?.error || "没有读到关系提案");
      setState({ status: "done", payload, error: "" });
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
      </div>

      <aside className="review-boundary-note">
        <WarningCircle size={17} aria-hidden="true" />
        <p><strong>相似，不等于有关联。</strong>接受前会再验两端 active 状态、内容 hash 与逐字证据；只有通过审核的边，才会出现在记忆卡的“关联 Scene”。</p>
      </aside>

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
