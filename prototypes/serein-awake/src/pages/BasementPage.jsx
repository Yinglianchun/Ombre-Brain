import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowClockwise,
  Check,
  Eye,
  Flask,
  GitDiff,
  PencilSimple,
  Plus,
  ShareNetwork,
  Trash,
  WarningCircle,
  X,
} from "@phosphor-icons/react";
import { BasementRecallObservation } from "../components/BasementRecallObservation.jsx";
import { BasementRelationshipProposals } from "../components/BasementRelationshipProposals.jsx";
import { BasementRevisionInbox } from "../components/BasementRevisionInbox.jsx";
import {
  basementRecallExamples,
  canonicalDomainPolicies,
  semanticRouteSnapshot,
} from "../data/basement.js";
import {
  clearDomainPolicyDraft,
  clearServerSemanticRouteDraft,
  clearSemanticRouteDraft,
  hasDomainPolicyDraft,
  readDomainPolicyDraft,
  readServerSemanticRouteDraft,
  readSemanticRouteDraft,
  saveServerSemanticRouteDraft,
  saveDomainPolicyDraft,
  saveSemanticRouteDraft,
} from "../storage/basementStore.js";
import { upsertRecallSimulationTrainingLabel } from "../storage/recallSimulationTraining.js";

const routeLabels = {
  simple_contact: "陪伴与贴近",
  present_chitchat: "此刻闲聊",
  intimate_contact: "亲密联系",
  recall_needed: "明确回望",
};

const roleLabels = {
  typical: "典型例句",
  boundary: "边界例句",
};

const originLabels = {
  manual: "人工添加",
  online_false_positive: "线上误召",
  online_false_negative: "线上漏召",
  import: "导入",
};

const exampleStatusLabels = {
  draft: "草稿",
  published: "已发布",
  retired: "已停用",
};

const reasonLabels = {
  recall_route_won: "召回路线胜出",
  matched_skip_route: "命中不召回路线",
  boundary_veto: "边界护栏撤销跳过",
  below_threshold: "最高分未过阈值，继续召回",
  insufficient_margin: "路线差距不足，继续召回",
  route_index_stale: "例句已变化，向量需要重建",
};

const actionLabel = (action) => (action === "skip" ? "no-recall" : "recall");
const percent = (value) => `${(Number(value || 0) * 100).toFixed(1)}%`;
const clone = (value) => JSON.parse(JSON.stringify(value));
const defaultRouteThreshold = 0.72;

function recommendedRouteThreshold(route) {
  const name = String(route?.name || "").trim();
  const label = String(route?.label || "").trim();
  if (name === "present_chitchat" || label === "此刻闲聊") return 0.60;
  if (name === "技术闲聊" || label === "技术闲聊") return 0.58;
  return null;
}

function applyRecommendedRouteThresholds(routes) {
  return routes.map((route) => {
    const recommended = recommendedRouteThreshold(route);
    return recommended == null || route.threshold != null ? route : { ...route, threshold: recommended };
  });
}

function routeSnapshotFromApi(payload) {
  if (!payload || !Array.isArray(payload.routes)) throw new Error("route_dataset_invalid");
  return {
    datasetVersion: Number(payload.dataset_version),
    deploymentState: payload.deployment_state || "production",
    capturedAt: payload.published?.published_at || new Date().toISOString(),
    model: payload.embedding?.model || semanticRouteSnapshot.model,
    boundaryExampleCount: Number(payload.boundary_example_count || payload.published?.boundary_example_count || 0),
    indexedBoundaryExampleCount: Number(payload.indexed_boundary_example_count || 0),
    boundaryIndexReady: payload.boundary_index_ready !== false,
    routes: payload.routes.map((route) => ({
      ...route,
      label: route.label || routeLabels[route.name] || route.name,
      enabled: route.enabled !== false,
      utterances: Array.isArray(route.utterances) ? route.utterances.map((item) => ({
        text: String(item.text || "").trim(),
        role: item.role === "boundary" ? "boundary" : "typical",
        origin: ["manual", "online_false_positive", "online_false_negative", "import"].includes(item.origin)
          ? item.origin
          : "import",
        status: item.status === "retired" ? "retired" : "published",
      })) : [],
    })),
  };
}

function routePublishError(error) {
  if (error.startsWith("route_publish_version_conflict:")) return "线上数据集已经变化。请重新载入后检查草稿，再发布。";
  if (error.startsWith("route_source_active_center_missing:")) return "每条启用路线都至少需要一条典型例句；边界例句不会单独形成路线中心。";
  if (error.startsWith("route_source_utterance_duplicate:")) return "同一句不能同时出现在两条路线里。";
  if (error === "route_publish_confirmation_required") return "这次发布缺少确认标记。";
  return error || "发布没有完成。";
}

async function readRouteApiResponse(response) {
  const text = await response.text();
  try {
    return JSON.parse(text || "{}");
  } catch {
    if (response.status === 404) {
      throw new Error("Gateway 还没有部署 Router 发布接口；本机草稿仍然保留。");
    }
    throw new Error(text || `Router 接口返回 ${response.status}`);
  }
}

function openSceneInMemory(item) {
  const sceneId = item.bucket_id || item.moment_id || String(item.id || "").split("#").pop();
  if (!sceneId) return;
  window.localStorage.setItem("serein.memory.open-source-id", sceneId);
  window.location.hash = "#memory";
  window.dispatchEvent(new CustomEvent("serein:open-memory-scene", { detail: { sourceId: sceneId } }));
}

const recallAblationOptions = [
  { value: "normal", label: "正常", description: "cues + 正文 embedding" },
  { value: "without_cues", label: "关闭 cues", description: "只看正文 embedding" },
  { value: "without_embedding", label: "关闭正文 embedding", description: "只看 cues / 词面" },
];

const recallCandidateSourceLabels = {
  exact_anchor: "精确锚点",
  title_anchor: "标题锚点",
  lexical: "正文词面",
  cue_lexical: "cue 词面",
  body_semantic: "正文 embedding",
  retrieval_alias: "检索别名",
};

const rerankerShadowLabels = {
  eligible_not_called: "达到 floor，shadow 未调用",
  ineligible_below_floor: "低于 floor，不进入 reranker",
};

function RecallSimulator() {
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("idle");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [recallAblation, setRecallAblation] = useState("normal");
  const [trainingForm, setTrainingForm] = useState({ expectedAction: "", expectedRoute: "", memoryIds: "" });
  const [trainingNotice, setTrainingNotice] = useState("");

  const runSimulation = async (event) => {
    event?.preventDefault();
    const text = query.trim();
    if (!text || status === "loading") return;
    setStatus("loading");
    setError("");
    try {
      const response = await fetch("/__serein/gateway/recall", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: text,
          simulation: true,
          include_debug: true,
          recall_mode: "full",
          recall_ablation: recallAblation,
        }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload?.message || payload?.error || "Gateway 没有返回结果");
      setResult(payload);
      setTrainingForm({ expectedAction: "", expectedRoute: "", memoryIds: "" });
      setTrainingNotice("");
      setStatus("done");
    } catch (requestError) {
      setResult(null);
      setError(requestError instanceof Error ? requestError.message : "无法连接 Gateway");
      setStatus("error");
    }
  };

  const debug = result?.debug ?? {};
  const semantic = debug.semantic_recall_debug ?? {};
  const routeScores = semantic.scores ?? [];
  const boundaryVeto = semantic.boundary_veto ?? {};
  const boundaryCandidate = boundaryVeto.candidate ?? null;
  const injected = debug.recall_why_summary?.injected ?? [];
  const suppressed = debug.recall_why_summary?.suppressed ?? [];
  const cards = result?.cards ?? [];
  const retrievalBudget = semantic.retrieval_budget ?? {};
  const prototypePrior = retrievalBudget.prototype_prior ?? {};
  const sentinel = retrievalBudget.sentinel ?? {};
  const cheapRetrieval = retrievalBudget.cheap_retrieval ?? {};
  const rerankerShadow = retrievalBudget.rerank ?? {};
  const ablationDebug = retrievalBudget.recall_ablation ?? semantic.recall_ablation ?? {
    mode: recallAblation,
  };
  const candidateEvidence = Array.isArray(cheapRetrieval.candidates)
    ? cheapRetrieval.candidates
    : [];
  const momentDebugByBucketId = new Map(
    (debug.recalled_moment_debug ?? []).map((item) => [item.bucket_id, item]),
  );
  const trainingRouteOptions = [...new Map([
    ...semanticRouteSnapshot.routes.map((route) => [route.name, route.label || routeLabels[route.name] || route.name]),
    ...routeScores.map((route) => [route.route, routeLabels[route.route] || route.route]),
  ].filter(([name]) => name)).entries()];

  const chooseExpectedAction = (expectedAction) => {
    setTrainingForm((current) => ({
      ...current,
      expectedAction,
      expectedRoute: current.expectedRoute
        || (expectedAction === "recall" ? "recall_needed" : semantic.route || "present_chitchat"),
    }));
    setTrainingNotice("");
  };

  const saveTrainingLabel = () => {
    const saved = upsertRecallSimulationTrainingLabel({
      query: result?.query || query,
      expectedAction: trainingForm.expectedAction,
      expectedRoute: trainingForm.expectedRoute,
      expectedMemoryIds: trainingForm.memoryIds.split(/[\s,，]+/),
      observedAction: semantic.applied_action,
      observedRoute: semantic.route,
    });
    setTrainingNotice(saved.status === "added"
      ? "已保存为人工模拟训练标注；下次导出会与真实观察合并，并保留来源。"
      : saved.status === "updated"
        ? "这句的人工模拟标注已更新，不会重复堆一条。"
        : "先选择这句话应该召回还是应该跳过。");
  };

  return (
    <section className="basement-workbench" aria-labelledby="recall-simulator-title">
      <header className="basement-workbench__header">
        <div>
          <span className="basement-kicker">真实 Gateway 路径</span>
          <h2 id="recall-simulator-title">召回模拟</h2>
          <p>输入原句，查看 Router 决定、候选记忆与最终注入。测试不会留下正式注入记录。</p>
        </div>
        <div className="basement-live-note">
          <i aria-hidden="true" />
          <span>{semantic.model || semanticRouteSnapshot.model}</span>
        </div>
      </header>

      <form className="recall-simulator-form" onSubmit={runSimulation}>
        <label htmlFor="recall-simulator-query">原句</label>
        <textarea
          id="recall-simulator-query"
          value={query}
          rows={3}
          placeholder="把这一轮真正会说的话放进来。"
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={(event) => {
            if ((event.metaKey || event.ctrlKey) && event.key === "Enter") runSimulation(event);
          }}
        />
        <div className="recall-simulator-form__footer">
          <div className="recall-example-prompts" aria-label="测试原句">
            {basementRecallExamples.map((example) => (
              <button type="button" key={example} onClick={() => setQuery(example)}>{example}</button>
            ))}
          </div>
          <button className="basement-primary-action" type="submit" disabled={!query.trim() || status === "loading"}>
            {status === "loading" ? <ArrowClockwise size={17} className="is-spinning" aria-hidden="true" /> : <Flask size={17} aria-hidden="true" />}
            {status === "loading" ? "正在走一遍" : "现场模拟"}
          </button>
        </div>
        <fieldset className="recall-ablation-control">
          <legend>消融观察</legend>
          <div className="recall-ablation-control__options">
            {recallAblationOptions.map((option) => (
              <label key={option.value} className={recallAblation === option.value ? "is-active" : ""}>
                <input
                  type="radio"
                  name="recall-ablation"
                  value={option.value}
                  checked={recallAblation === option.value}
                  onChange={(event) => setRecallAblation(event.target.value)}
                />
                <span><strong>{option.label}</strong><small>{option.description}</small></span>
              </label>
            ))}
          </div>
          <p>Route 与 evidence veto 保持不变；这里只切换 simulation 候选通道，不把 cue 混进 Scene 正文向量。</p>
        </fieldset>
      </form>

      {status === "idle" && (
        <div className="basement-empty-state">
          <span>这里显示真实结果</span>
          <p>Route 只是入口判断。最终有没有记忆出现，还要继续经过候选、证据与放行。</p>
        </div>
      )}

      {status === "error" && (
        <div className="basement-error" role="alert">
          <WarningCircle size={19} aria-hidden="true" />
          <div><strong>没有走到 Gateway</strong><p>{error}</p></div>
        </div>
      )}

      {status === "done" && (
        <div className="recall-result" aria-live="polite">
          <section className="recall-decision">
            <div className="recall-decision__route">
              <span>ROUTE</span>
              <strong>{routeLabels[semantic.route] || semantic.route || "未匹配"}</strong>
              <em className={`route-action route-action--${semantic.applied_action || "recall"}`}>
                {actionLabel(semantic.applied_action)}
              </em>
            </div>
            <dl className="recall-decision__facts">
              <div><dt>置信度</dt><dd>{percent(semantic.confidence)}</dd></div>
              <div><dt>候选记忆</dt><dd>{debug.candidate_count ?? 0}</dd></div>
              <div><dt>最终注入</dt><dd>{debug.injected_bucket_ids?.length ?? result.recalled_ids?.length ?? 0}</dd></div>
              <div><dt>原因</dt><dd>{reasonLabels[semantic.reason] || semantic.reason || "继续按证据判断"}</dd></div>
            </dl>
          </section>

          <section className="recall-result-section">
            <div className="recall-result-section__heading">
              <h3>预算 Router（simulation shadow）</h3>
              <span>{retrievalBudget.effective_budget || "未返回"}</span>
            </div>
            <dl className="recall-decision__facts">
              <div><dt>surface_route</dt><dd>{retrievalBudget.surface_route || "—"}</dd></div>
              <div><dt>route_budget</dt><dd>{retrievalBudget.route_budget || "—"}</dd></div>
              <div><dt>effective_budget</dt><dd>{retrievalBudget.effective_budget || "—"}</dd></div>
              <div><dt>anchor_override</dt><dd>{retrievalBudget.anchor_override ? "是" : "否"}</dd></div>
              <div><dt>pure chitchat prior</dt><dd>{retrievalBudget.pure_chitchat_prior ? "高置信候选" : "否"}</dd></div>
              <div><dt>prototype confidence</dt><dd>{prototypePrior.confidence == null ? "—" : percent(prototypePrior.confidence)}</dd></div>
              <div><dt>sentinel top1/2</dt><dd>{sentinel.called ? `${sentinel.floor_qualified_count ?? 0} / ${sentinel.candidate_count ?? 0}` : sentinel.reason || "未运行"}</dd></div>
              <div><dt>absolute floor</dt><dd>{cheapRetrieval.floor_qualified_count ?? 0} / {cheapRetrieval.candidate_count ?? 0}</dd></div>
              <div><dt>reranker shadow</dt><dd>{rerankerShadow.would_call ? "有资格，未调用生产模型" : rerankerShadow.reason || "未进入"}</dd></div>
              <div><dt>query_facets</dt><dd>{(retrievalBudget.query_facets || []).map((facet) => `${facet.kind}:${facet.value}`).join(" · ") || "—"}</dd></div>
            </dl>
            <p className="recall-evidence-decomposition__note">
              sentinel 只复用现有 query vector 做 top1/2 救援检查，不扩图、不 rerank、不注入、不写正式记录；异常一律 fail-open。
            </p>
          </section>

          <section className="recall-result-section">
            <div className="recall-result-section__heading"><h3>路线对照</h3><span>同一原句只做一次 query embedding</span></div>
            <div className="route-score-list">
              {routeScores.map((score) => (
                <div className="route-score" key={score.route}>
                  <div><strong>{routeLabels[score.route] || score.route}</strong><span>{actionLabel(score.action)}</span></div>
                  <output>{percent(score.score)}</output>
                  <small>最近例句：{score.top_examples?.[0]?.text || "无"}</small>
                </div>
              ))}
            </div>
          </section>

          <section className="recall-result-section recall-evidence-decomposition">
            <div className="recall-result-section__heading">
              <h3>候选证据拆解</h3>
              <span>{candidateEvidence.length} 条 · {recallAblationOptions.find((option) => option.value === ablationDebug.mode)?.label || ablationDebug.mode || "正常"}</span>
            </div>
            <p className="recall-evidence-decomposition__note">
              canonical Scene 的 body semantic 仍是正文原文向量；cue semantic 尚无独立索引时显示 unavailable，不用 0 冒充。候选发现与最终放行依据分开显示。
            </p>
            {candidateEvidence.length ? (
              <div className="recall-evidence-list">
                {candidateEvidence.map((candidate) => (
                  <article className={`recall-evidence-row ${candidate.floor_qualified ? "is-qualified" : "is-suppressed"}`} key={candidate.bucket_id}>
                    <header>
                      <strong>{candidate.title || candidate.bucket_id}</strong>
                      <span>{candidate.final_admission_source || "pending"}</span>
                    </header>
                    <dl>
                      <div><dt>body semantic</dt><dd>{candidate.body_semantic_score == null ? "—" : percent(candidate.body_semantic_score)}</dd></div>
                      <div><dt>semantic profile</dt><dd>{candidate.semantic_profile || "unknown"}</dd></div>
                      <div><dt>cue semantic</dt><dd>{candidate.cue_semantic?.status || "unknown"}</dd></div>
                      <div><dt>cue lexical</dt><dd>{candidate.cue_lexical_match ? candidate.matched_cues?.join(" · ") || "命中" : "未命中"}</dd></div>
                      <div><dt>title anchor</dt><dd>{candidate.title_anchor_match ? candidate.title_anchor_terms?.join(" · ") || "命中" : "未命中"}</dd></div>
                      <div><dt>候选来源</dt><dd>{(candidate.candidate_sources || []).map((source) => recallCandidateSourceLabels[source] || source).join(" · ") || "未记录"}</dd></div>
                      <div><dt>combined / floor</dt><dd>{percent(candidate.combined_score)} / {percent(candidate.absolute_floor)}</dd></div>
                      <div><dt>reranker shadow</dt><dd>{candidate.reranker_shadow?.score == null ? rerankerShadowLabels[candidate.reranker_shadow?.status] || candidate.reranker_shadow?.status || "未调用" : percent(candidate.reranker_shadow.score)}</dd></div>
                    </dl>
                    <button type="button" className="recall-card__memory-link" onClick={() => openSceneInMemory(candidate)}>
                      在记忆卡里查看召回入口
                    </button>
                  </article>
                ))}
              </div>
            ) : <p className="recall-none">本轮没有进入廉价候选池的记忆。</p>}
          </section>

          {boundaryCandidate && (
            <section className="recall-result-section">
              <div className="recall-result-section__heading">
                <h3>边界护栏</h3>
                <span>{boundaryVeto.applied ? "已撤销 skip" : "本轮未触发"}</span>
              </div>
              <div className="route-score-list">
                <div className="route-score">
                  <div>
                    <strong>{routeLabels[boundaryCandidate.route] || boundaryCandidate.route}</strong>
                    <span>{actionLabel(boundaryCandidate.action)}</span>
                  </div>
                  <output>{percent(boundaryCandidate.score)}</output>
                  <small>边界例句：{boundaryCandidate.text || "无"}</small>
                  <small>
                    {boundaryCandidate.passes_threshold ? "已过护栏阈值" : `未过护栏阈值 ${percent(boundaryVeto.threshold)}`}
                    {" · "}
                    {boundaryCandidate.beats_skip
                      ? "强于 skip 路线"
                      : boundaryCandidate.within_deficit
                        ? `落后 ${percent(boundaryCandidate.deficit)}，仍在护栏差值内`
                        : `落后 ${percent(boundaryCandidate.deficit)}，超过护栏差值 ${percent(boundaryVeto.max_deficit)}`}
                  </small>
                </div>
              </div>
            </section>
          )}

          <section className="recall-result-section">
            <div className="recall-result-section__heading"><h3>最终放行</h3><span>{cards.length || injected.length} 条</span></div>
            {(cards.length || injected.length) ? (
              <div className="recall-card-list">
                {(cards.length ? cards : injected).map((item, index) => (
                  <article className="recall-card" key={item.id || item.bucket_id || index}>
                    <div><strong>{item.title || item.bucket_name || item.id || item.bucket_id}</strong><span>{item.source || item.final_status || "direct"}</span></div>
                    {(item.text || item.content) && <p>{item.text || item.content}</p>}
                    {(item.admission_reasons?.length > 0) && <small>{item.admission_reasons.join(" / ")}</small>}
                    {momentDebugByBucketId.get(item.bucket_id)?.authored_cue_match ? (
                      <small className="recall-card__cue-hit">
                        cue 命中：{momentDebugByBucketId.get(item.bucket_id).authored_cue_terms.join(" · ")}
                      </small>
                    ) : null}
                    <button type="button" className="recall-card__memory-link" onClick={() => openSceneInMemory(item)}>
                      在记忆卡里查看召回入口
                    </button>
                  </article>
                ))}
              </div>
            ) : <p className="recall-none">Router 允许继续寻找，但这一句没有记忆通过证据门。</p>}
          </section>

          <section className="recall-training-label">
            <div className="recall-result-section__heading">
              <h3>保存为训练标注</h3>
              <span>人工模拟 · 不冒充真实 Hook</span>
            </div>
            <p>判断这一句本来应该做什么。重复保存同一句会更新原标注，不会越堆越多。</p>
            <div className="recall-training-label__actions" role="group" aria-label="预期召回动作">
              <button
                type="button"
                className={trainingForm.expectedAction === "recall" ? "is-active" : ""}
                onClick={() => chooseExpectedAction("recall")}
              >应该召回</button>
              <button
                type="button"
                className={trainingForm.expectedAction === "skip" ? "is-active" : ""}
                onClick={() => chooseExpectedAction("skip")}
              >应该跳过</button>
            </div>
            <div className="recall-training-label__fields">
              <label>
                <span>预期路线</span>
                <select
                  value={trainingForm.expectedRoute}
                  onChange={(event) => setTrainingForm((current) => ({ ...current, expectedRoute: event.target.value }))}
                >
                  <option value="">不标路线</option>
                  {trainingRouteOptions.map(([name, label]) => <option value={name} key={name}>{label}</option>)}
                </select>
              </label>
              <label>
                <span>目标记忆 ID（可选）</span>
                <input
                  value={trainingForm.memoryIds}
                  placeholder="scene_…；多个可用空格分开"
                  onChange={(event) => setTrainingForm((current) => ({ ...current, memoryIds: event.target.value }))}
                />
              </label>
              <button
                className="basement-primary-action"
                type="button"
                disabled={!trainingForm.expectedAction}
                onClick={saveTrainingLabel}
              ><Check size={16} aria-hidden="true" />保存标注</button>
            </div>
            {trainingNotice && <small className="recall-training-label__notice" role="status">{trainingNotice}</small>}
          </section>

          {suppressed.length > 0 && (
            <details className="recall-suppressed">
              <summary>被拒绝的候选 <span>{suppressed.length}</span></summary>
              <div>
                {suppressed.slice(0, 12).map((item) => (
                  <p key={item.bucket_id}>
                    <strong>{item.bucket_name || item.bucket_id}</strong>
                    <span>{item.admission_reasons?.join(" / ") || "未通过当前证据门"}</span>
                    <button type="button" onClick={() => openSceneInMemory(item)}>打开记忆卡</button>
                  </p>
                ))}
              </div>
            </details>
          )}
        </div>
      )}
    </section>
  );
}

const domainPolicyLabels = {
  normal: "正常召回",
  explicit_only: "仅明确召回",
  excluded: "完全排除",
};

const domainPolicyDescriptions = {
  normal: "可以参与普通候选、证据放行与关系扩散。",
  explicit_only: "只认 authored cue、标题、ID 或明确锚点，不接受纯 embedding 泛化。",
  excluded: "候选、加分、直召回与扩散全部禁止，恢复正常前不会进入注入。",
};

function DomainPolicyEditor() {
  const [snapshot, setSnapshot] = useState({ datasetVersion: 1, active: false, domains: canonicalDomainPolicies });
  const [domains, setDomains] = useState(() => readDomainPolicyDraft(canonicalDomainPolicies));
  const [datasetState, setDatasetState] = useState({ status: "loading", message: "正在核对线上主域策略……" });
  const [publishState, setPublishState] = useState({ status: "idle", message: "" });
  const baseline = JSON.stringify(snapshot.domains.map(({ key, policy }) => ({ key, policy })));
  const current = JSON.stringify(domains.map(({ key, policy }) => ({ key, policy })));
  const contentDirty = current !== baseline;
  const dirty = contentDirty || !snapshot.active;
  const excludedCount = domains.filter((domain) => domain.policy === "excluded").length;
  const explicitCount = domains.filter((domain) => domain.policy === "explicit_only").length;

  const setPolicy = (key, policy) => {
    const nextDomains = domains.map((domain) => domain.key === key ? { ...domain, policy } : domain);
    setDomains(nextDomains);
    saveDomainPolicyDraft(nextDomains);
  };

  const loadPublishedPolicies = async () => {
    setDatasetState({ status: "loading", message: "正在核对线上主域策略……" });
    try {
      const response = await fetch("/__serein/gateway/domain-policies");
      const payload = await response.json();
      if (!response.ok || !Array.isArray(payload.policies)) {
        throw new Error(String(payload?.error || "domain_policy_dataset_unavailable"));
      }
      const policyByKey = new Map(payload.policies.map((item) => [item?.key, item?.policy]));
      const publishedDomains = canonicalDomainPolicies.map((domain) => ({
        ...domain,
        policy: ["normal", "explicit_only", "excluded"].includes(policyByKey.get(domain.key))
          ? policyByKey.get(domain.key)
          : domain.policy,
      }));
      const nextSnapshot = {
        datasetVersion: Number(payload.dataset_version) || 1,
        active: Boolean(payload.active),
        domains: publishedDomains,
      };
      const keepDraft = hasDomainPolicyDraft();
      setSnapshot(nextSnapshot);
      setDomains(keepDraft ? readDomainPolicyDraft(publishedDomains) : publishedDomains);
      setDatasetState({ status: "ready", message: `已核对线上 v${nextSnapshot.datasetVersion}` });
    } catch (error) {
      setDatasetState({ status: "error", message: error.message || "没有读到线上主域策略。" });
    }
  };

  useEffect(() => {
    loadPublishedPolicies();
  }, []);

  const resetDraft = () => {
    if (contentDirty && !window.confirm("放弃本机所有主域策略草稿？")) return;
    setDomains(clearDomainPolicyDraft(snapshot.domains));
    setPublishState({ status: "idle", message: "" });
  };

  const publishPolicies = async () => {
    const nextVersion = snapshot.datasetVersion + 1;
    if (!window.confirm(
      `把完整主域策略从 v${snapshot.datasetVersion} 发布为 v${nextVersion}？\n\n`
      + "完全排除会阻止该主域的候选、明确 ID 与关系扩散；仅明确召回只接受 authored cue、标题或 ID。Scene 本身不会被修改。",
    )) return;
    setPublishState({ status: "publishing", message: `正在发布 v${nextVersion}……` });
    try {
      const response = await fetch("/__serein/gateway/domain-policies", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          expected_dataset_version: snapshot.datasetVersion,
          confirm: "PUBLISH_DOMAIN_RECALL_POLICIES",
          policies: domains.map(({ key, policy }) => ({ key, policy })),
        }),
      });
      const payload = await response.json();
      if (!response.ok) {
        const raw = String(payload?.error || "domain_policy_publish_failed");
        if (raw.startsWith("domain_policy_publish_version_conflict:")) {
          throw new Error("线上主域策略已经变化。请重新核对后再发布。");
        }
        throw new Error(raw);
      }
      const policyByKey = new Map(payload.policies.map((item) => [item?.key, item?.policy]));
      const publishedDomains = canonicalDomainPolicies.map((domain) => ({
        ...domain,
        policy: policyByKey.get(domain.key) || domain.policy,
      }));
      const nextSnapshot = {
        datasetVersion: Number(payload.dataset_version),
        active: true,
        domains: publishedDomains,
      };
      clearDomainPolicyDraft(publishedDomains);
      setSnapshot(nextSnapshot);
      setDomains(publishedDomains);
      setDatasetState({ status: "ready", message: `已核对线上 v${nextSnapshot.datasetVersion}` });
      setPublishState({ status: "success", message: `v${nextSnapshot.datasetVersion} 已切换生效。` });
    } catch (error) {
      setPublishState({ status: "error", message: error.message || "发布没有完成，线上策略未改变。" });
    }
  };

  return (
    <section className="basement-workbench" aria-labelledby="domain-policy-title">
      <header className="basement-workbench__header">
        <div>
          <span className="basement-kicker">只改变召回资格</span>
          <h2 id="domain-policy-title">主域边界</h2>
          <p>按 Scene 自己保存的 canonical_domain 决定召回方式。这里不会改动任何 Scene。</p>
        </div>
        <div className="domain-policy-summary" aria-label="主域策略摘要">
          <span><strong>{explicitCount}</strong> 仅明确</span>
          <span><strong>{excludedCount}</strong> 已排除</span>
        </div>
      </header>

      <div className="domain-policy-list">
        {domains.map((domain) => (
          <article className={`domain-policy-row domain-policy-row--${domain.policy}`} key={domain.key}>
            <div className="domain-policy-row__identity">
              <span>{domain.key}</span>
              <h3>{domain.label}</h3>
              <p>{domain.description}</p>
            </div>
            <div className="domain-policy-controls" role="group" aria-label={`${domain.label}主域策略`}>
              {Object.entries(domainPolicyLabels).map(([policy, label]) => (
                <button
                  type="button"
                  className={domain.policy === policy ? "is-active" : ""}
                  key={policy}
                  onClick={() => setPolicy(domain.key, policy)}
                >
                  {label}
                </button>
              ))}
            </div>
            <p className="domain-policy-row__explanation">
              {domainPolicyDescriptions[domain.policy]}
              {domain.policy === "explicit_only" && <code>domain_explicit_only</code>}
              {domain.policy === "excluded" && <code>domain_excluded</code>}
            </p>
          </article>
        ))}
      </div>

      <footer className="route-editor-footer">
        <div>
          <strong>{!snapshot.active
            ? "生产主域策略尚未启用"
            : dirty ? "策略草稿只在这台电脑上" : `与生产 v${snapshot.datasetVersion} 一致`}</strong>
          <span>{publishState.message || datasetState.message || "发布后，召回模拟与真实 Gateway 会使用同一套主域边界。"}</span>
        </div>
        <div>
          <button type="button" onClick={loadPublishedPolicies} disabled={datasetState.status === "loading" || publishState.status === "publishing"}>核对线上</button>
          <button type="button" onClick={resetDraft} disabled={!contentDirty}>撤销草稿</button>
          <button
            type="button"
            className="basement-primary-action"
            onClick={publishPolicies}
            disabled={!dirty || datasetState.status !== "ready" || publishState.status === "publishing"}
            title={datasetState.status === "ready" ? "确认后原子切换生产主域策略" : "先核对线上主域策略版本"}
          >{publishState.status === "publishing" ? "正在发布" : "应用到生产"}</button>
        </div>
      </footer>
    </section>
  );
}

function RouteExampleEditor() {
  const [routeSnapshot, setRouteSnapshot] = useState(semanticRouteSnapshot);
  const [routes, setRoutes] = useState(() => readSemanticRouteDraft(semanticRouteSnapshot));
  const [selectedRouteName, setSelectedRouteName] = useState("recall_needed");
  const [creatingRoute, setCreatingRoute] = useState(false);
  const [newRouteLabel, setNewRouteLabel] = useState("");
  const [newRouteAction, setNewRouteAction] = useState("skip");
  const [editingIndex, setEditingIndex] = useState(null);
  const [editingText, setEditingText] = useState("");
  const [newText, setNewText] = useState("");
  const [newRole, setNewRole] = useState("typical");
  const [datasetState, setDatasetState] = useState({ status: "loading", message: "正在核对线上版本……" });
  const [publishState, setPublishState] = useState({ status: "idle", message: "" });
  const [draftSyncState, setDraftSyncState] = useState({ status: "idle", message: "" });
  const draftRevisionRef = useRef(0);
  const saveQueueRef = useRef(Promise.resolve());

  const selectedRoute = routes.find((route) => route.name === selectedRouteName) ?? routes[0];
  const baseline = JSON.stringify(routeSnapshot.routes);
  const dirty = JSON.stringify(routes) !== baseline;
  const boundaryRebuildNeeded = routeSnapshot.boundaryExampleCount > 0 && !routeSnapshot.boundaryIndexReady;
  const exampleCount = useMemo(
    () => routes.reduce((total, route) => total + route.utterances.length, 0),
    [routes],
  );

  const loadPublishedDataset = async () => {
    setDatasetState({ status: "loading", message: "正在核对线上版本……" });
    try {
      const response = await fetch("/__serein/gateway/semantic-routes");
      const payload = await readRouteApiResponse(response);
      if (!response.ok) throw new Error(payload.message || payload.error || "route_dataset_unavailable");
      const nextSnapshot = routeSnapshotFromApi(payload);
      const localRoutes = readSemanticRouteDraft(nextSnapshot);
      const serverState = await readServerSemanticRouteDraft();
      let nextRoutes = localRoutes;
      if (serverState.draft) {
        if (serverState.draft.baseDatasetVersion !== nextSnapshot.datasetVersion) {
          throw new Error(`服务器草稿基于 v${serverState.draft.baseDatasetVersion}，线上已是 v${nextSnapshot.datasetVersion}，先不要覆盖。`);
        }
        nextRoutes = serverState.draft.routes;
        draftRevisionRef.current = serverState.draft.revision || 0;
        saveSemanticRouteDraft(nextRoutes, nextSnapshot.datasetVersion);
        setDraftSyncState({ status: "saved", message: "服务器草稿已载入" });
      } else if (JSON.stringify(localRoutes) !== JSON.stringify(nextSnapshot.routes)) {
        const migrated = await saveServerSemanticRouteDraft(localRoutes, nextSnapshot.datasetVersion, 0);
        draftRevisionRef.current = migrated.draft.revision;
        setDraftSyncState({ status: "saved", message: "原浏览器草稿已迁到服务器" });
      } else {
        nextRoutes = nextSnapshot.routes;
        draftRevisionRef.current = 0;
        setDraftSyncState({ status: "saved", message: "服务器暂无草稿" });
      }
      const routesWithRecommendedThresholds = applyRecommendedRouteThresholds(nextRoutes);
      if (JSON.stringify(routesWithRecommendedThresholds) !== JSON.stringify(nextRoutes)) {
        const saved = await saveServerSemanticRouteDraft(
          routesWithRecommendedThresholds,
          nextSnapshot.datasetVersion,
          draftRevisionRef.current,
        );
        draftRevisionRef.current = saved.draft.revision;
        nextRoutes = routesWithRecommendedThresholds;
        saveSemanticRouteDraft(nextRoutes, nextSnapshot.datasetVersion);
        setDraftSyncState({ status: "saved", message: "已加入闲聊路线的建议阈值，等待发布" });
      }
      setRouteSnapshot(nextSnapshot);
      setRoutes(nextRoutes);
      setDatasetState({ status: "ready", message: `已核对线上 v${nextSnapshot.datasetVersion}` });
    } catch (error) {
      setDatasetState({ status: "error", message: error.message || "没有读到线上 Router 数据集。" });
    }
  };

  useEffect(() => {
    loadPublishedDataset();
  }, []);

  const persistRoutes = (nextRoutes) => {
    setRoutes(nextRoutes);
    saveSemanticRouteDraft(nextRoutes, routeSnapshot.datasetVersion);
    setDraftSyncState({ status: "saving", message: "正在保存到服务器……" });
    const task = saveQueueRef.current.catch(() => {}).then(async () => {
      const saved = await saveServerSemanticRouteDraft(
        nextRoutes,
        routeSnapshot.datasetVersion,
        draftRevisionRef.current,
      );
      draftRevisionRef.current = saved.draft.revision;
      setDraftSyncState({ status: "saved", message: "草稿已保存在德国机" });
      return saved;
    });
    saveQueueRef.current = task;
    task.catch((error) => {
      setDraftSyncState({
        status: "error",
        message: error.message === "route_draft_revision_conflict"
          ? "服务器草稿已在别处变化，重新核对后再编辑。"
          : "草稿没有保存到服务器。",
      });
    });
    return task;
  };

  const commitRoutes = (nextRoutes) => { persistRoutes(nextRoutes); };

  const addRoute = (event) => {
    event.preventDefault();
    const label = newRouteLabel.trim();
    if (!label) return;
    if (routes.some((route) => (route.label || route.name).trim() === label || route.name === label)) {
      window.alert("这个类别已经存在了。");
      return;
    }
    const route = {
      name: label,
      label,
      action: newRouteAction === "recall" ? "recall" : "skip",
      threshold: newRouteAction === "recall" ? defaultRouteThreshold : 0.60,
      enabled: true,
      utterances: [],
    };
    commitRoutes([...routes, route]);
    setSelectedRouteName(route.name);
    setNewRouteLabel("");
    setNewRouteAction("skip");
    setCreatingRoute(false);
  };

  const deleteRoute = () => {
    if (!selectedRoute || routes.length <= 1) return;
    const label = selectedRoute.label || routeLabels[selectedRoute.name] || selectedRoute.name;
    const detail = selectedRoute.utterances.length
      ? `，连同其中 ${selectedRoute.utterances.length} 条例句`
      : "";
    if (!window.confirm(`从发布草稿中删除类别「${label}」${detail}？`)) return;
    const selectedIndex = routes.findIndex((route) => route.name === selectedRoute.name);
    const nextRoutes = routes.filter((route) => route.name !== selectedRoute.name);
    const nextSelected = nextRoutes[Math.min(selectedIndex, nextRoutes.length - 1)];
    commitRoutes(nextRoutes);
    setSelectedRouteName(nextSelected.name);
    setEditingIndex(null);
  };

  const startEditing = (index, text) => {
    setEditingIndex(index);
    setEditingText(text);
  };

  const saveEdit = () => {
    const text = editingText.trim();
    if (!text || editingIndex == null) return;
    const nextRoutes = clone(routes);
    const edited = nextRoutes.find((route) => route.name === selectedRoute.name).utterances[editingIndex];
    edited.text = text;
    edited.status = "draft";
    commitRoutes(nextRoutes);
    setEditingIndex(null);
    setEditingText("");
  };

  const deleteExample = (index) => {
    const item = selectedRoute.utterances[index];
    if (!window.confirm(`从草稿中删除「${item.text}」？`)) return;
    const nextRoutes = clone(routes);
    nextRoutes.find((route) => route.name === selectedRoute.name).utterances.splice(index, 1);
    commitRoutes(nextRoutes);
    setEditingIndex(null);
  };

  const addExample = (event) => {
    event.preventDefault();
    const text = newText.trim();
    if (!text) return;
    const duplicate = routes.some((route) => route.utterances.some((item) => item.text.trim() === text));
    if (duplicate) {
      window.alert("这句已经在例句库里了。");
      return;
    }
    const nextRoutes = clone(routes);
    nextRoutes.find((route) => route.name === selectedRoute.name).utterances.push({
      text,
      role: newRole,
      origin: "manual",
      status: "draft",
    });
    commitRoutes(nextRoutes);
    setNewText("");
  };

  const updateRouteThreshold = (rawValue) => {
    const parsed = Number(rawValue);
    const current = Number(selectedRoute.threshold ?? defaultRouteThreshold);
    if (!Number.isFinite(parsed)) return current;
    const threshold = Math.round(Math.max(0.40, Math.min(0.95, parsed)) * 100) / 100;
    if (threshold === current && selectedRoute.threshold != null) return threshold;
    const nextRoutes = clone(routes);
    nextRoutes.find((route) => route.name === selectedRoute.name).threshold = threshold;
    commitRoutes(nextRoutes);
    return threshold;
  };

  const resetDraft = async () => {
    if (dirty && !window.confirm("放弃本机所有例句草稿？")) return;
    try {
      await saveQueueRef.current.catch(() => {});
      await clearServerSemanticRouteDraft();
      draftRevisionRef.current = 0;
      setRoutes(clearSemanticRouteDraft(routeSnapshot));
      setDraftSyncState({ status: "saved", message: "服务器草稿已清除" });
      setEditingIndex(null);
    } catch {
      setDraftSyncState({ status: "error", message: "服务器草稿没有清除。" });
    }
  };

  const publishRoutes = async () => {
    const boundaryCount = routes.reduce((total, route) => (
      total + route.utterances.filter((item) => item.role === "boundary" && item.status !== "retired").length
    ), 0);
    const nextVersion = routeSnapshot.datasetVersion + 1;
    const thresholdSummary = routes
      .filter((route) => route.enabled !== false)
      .map((route) => `${route.label || route.name} ${percent(route.threshold ?? defaultRouteThreshold)}`)
      .join(" · ");
    const confirmed = window.confirm(
      `把完整 Router 数据集从 v${routeSnapshot.datasetVersion} 发布为 v${nextVersion}？\n\n`
      + `路线阈值：${thresholdSummary}\n\n`
      + `将重建全部启用路线的典型例句向量；${boundaryCount} 条边界例句会建立独立护栏向量，但不进入等权路线中心。强反向边界命中只会撤销错误 skip，不会强制注入。构建或校验失败时，当前版本保持不变。`,
    );
    if (!confirmed) return;
    setPublishState({ status: "publishing", message: `正在构建 v${nextVersion} 的全部向量……` });
    try {
      await saveQueueRef.current;
      const response = await fetch("/__serein/gateway/semantic-routes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          expected_dataset_version: routeSnapshot.datasetVersion,
          confirm: "PUBLISH_SEMANTIC_ROUTES",
          routes,
        }),
      });
      const payload = await readRouteApiResponse(response);
      if (!response.ok) throw new Error(routePublishError(String(payload.error || payload.message || "")));
      const nextSnapshot = routeSnapshotFromApi(payload);
      await clearServerSemanticRouteDraft();
      draftRevisionRef.current = 0;
      clearSemanticRouteDraft(nextSnapshot);
      setRouteSnapshot(nextSnapshot);
      setRoutes(readSemanticRouteDraft(nextSnapshot));
      setPublishState({ status: "success", message: `v${nextSnapshot.datasetVersion} 已完整重建并切换生效。` });
      setDraftSyncState({ status: "saved", message: "服务器草稿已在发布后清除" });
      setDatasetState({ status: "ready", message: `已核对线上 v${nextSnapshot.datasetVersion}` });
    } catch (error) {
      setPublishState({ status: "error", message: error.message || "发布没有完成，线上版本未改变。" });
    }
  };

  return (
    <section className="basement-workbench" aria-labelledby="route-editor-title">
      <header className="basement-workbench__header">
        <div>
          <span className="basement-kicker">人工审核后才生效</span>
          <h2 id="route-editor-title">例句维护</h2>
          <p>分别维护样本角色与来源。这里只整理人工审核草稿，生成模型不进入在线请求路径。</p>
        </div>
        <div className="route-draft-state">
          <strong>{exampleCount}</strong>
          <span>{dirty
            ? "服务器草稿未发布"
            : `${routeSnapshot.deploymentState === "production" ? "生产快照" : "本地候选"} v${routeSnapshot.datasetVersion}`}</span>
          {datasetState.status === "ready" && (
            <span>
              边界护栏 {routeSnapshot.indexedBoundaryExampleCount ?? 0}/{routeSnapshot.boundaryExampleCount ?? 0}
              {routeSnapshot.boundaryIndexReady ? " 已索引" : " 等待重建"}
            </span>
          )}
        </div>
      </header>

      <div className="route-editor-layout">
        <nav className="route-family-list" aria-label="召回路线">
          {routes.map((route) => (
            <button
              type="button"
              className={route.name === selectedRoute.name ? "is-active" : ""}
              key={route.name}
              onClick={() => { setSelectedRouteName(route.name); setEditingIndex(null); }}
            >
              <span><strong>{route.label || routeLabels[route.name] || route.name}</strong><small>{route.enabled ? actionLabel(route.action) : "未启用"}</small></span>
              <em>{route.utterances.length}</em>
            </button>
          ))}
          {creatingRoute ? (
            <form className="route-family-create" onSubmit={addRoute}>
              <input
                autoFocus
                value={newRouteLabel}
                onChange={(event) => setNewRouteLabel(event.target.value)}
                placeholder="类别名称"
                aria-label="类别名称"
              />
              <select value={newRouteAction} onChange={(event) => setNewRouteAction(event.target.value)} aria-label="召回行为">
                <option value="skip">no-recall · 直接 skip</option>
                <option value="recall">recall · 继续召回</option>
              </select>
              <div>
                <button type="submit" disabled={!newRouteLabel.trim()}><Check size={15} aria-hidden="true" />保存</button>
                <button type="button" onClick={() => { setCreatingRoute(false); setNewRouteLabel(""); }}><X size={15} aria-hidden="true" />取消</button>
              </div>
            </form>
          ) : (
            <button type="button" className="route-family-add" onClick={() => setCreatingRoute(true)}>
              <Plus size={16} aria-hidden="true" />
              <span><strong>新增类别</strong><small>选择 skip 或 recall</small></span>
            </button>
          )}
        </nav>

        <div className="route-example-editor">
          <div className="route-example-editor__heading">
            <div><h3>{selectedRoute.label || routeLabels[selectedRoute.name]}</h3><p>{selectedRoute.name}</p></div>
            <div className="route-example-editor__heading-actions">
              <label className="route-threshold-control">
                <span>命中阈值</span>
                <input
                  key={`${selectedRoute.name}-${selectedRoute.threshold ?? "default"}`}
                  type="number"
                  min="0.40"
                  max="0.95"
                  step="0.01"
                  defaultValue={Number(selectedRoute.threshold ?? defaultRouteThreshold).toFixed(2)}
                  onBlur={(event) => { event.target.value = updateRouteThreshold(event.target.value).toFixed(2); }}
                  onKeyDown={(event) => { if (event.key === "Enter") event.currentTarget.blur(); }}
                  aria-label={`${selectedRoute.label || selectedRoute.name}命中阈值`}
                />
              </label>
              <span className={`route-action route-action--${selectedRoute.action}`}>{actionLabel(selectedRoute.action)}</span>
              <button type="button" onClick={deleteRoute} disabled={routes.length <= 1} aria-label={`删除类别${selectedRoute.label || selectedRoute.name}`} title="从发布草稿中删除整个类别">
                <Trash size={15} aria-hidden="true" />删除类别
              </button>
            </div>
          </div>

          <form className="route-example-add" onSubmit={addExample}>
            <input value={newText} onChange={(event) => setNewText(event.target.value)} placeholder="新增一条经过判断的原句" />
            <select value={newRole} onChange={(event) => setNewRole(event.target.value)} aria-label="样本角色">
              <option value="typical">典型例句</option>
              <option value="boundary">边界例句</option>
            </select>
            <button type="submit" disabled={!newText.trim()}><Plus size={16} aria-hidden="true" />加入草稿</button>
          </form>

          <div className="route-example-list">
            {selectedRoute.utterances.length ? selectedRoute.utterances.map((item, index) => (
              <div className="route-example-row" key={`${item.text}-${index}`}>
                {editingIndex === index ? (
                  <div className="route-example-row__edit">
                    <input autoFocus value={editingText} onChange={(event) => setEditingText(event.target.value)} onKeyDown={(event) => {
                      if (event.key === "Enter") saveEdit();
                      if (event.key === "Escape") setEditingIndex(null);
                    }} />
                    <button type="button" aria-label="保存修改" onClick={saveEdit}><Check size={16} aria-hidden="true" /></button>
                    <button type="button" aria-label="取消修改" onClick={() => setEditingIndex(null)}><X size={16} aria-hidden="true" /></button>
                  </div>
                ) : (
                  <>
                    <div>
                      <p>{item.text}</p>
                      <span>{roleLabels[item.role] || item.role} · {originLabels[item.origin] || item.origin} · {exampleStatusLabels[item.status] || item.status}</span>
                    </div>
                    <div className="route-example-row__actions">
                      <button type="button" aria-label={`编辑${item.text}`} onClick={() => startEditing(index, item.text)}><PencilSimple size={15} aria-hidden="true" /></button>
                      <button type="button" aria-label={`删除${item.text}`} onClick={() => deleteExample(index)}><Trash size={15} aria-hidden="true" /></button>
                    </div>
                  </>
                )}
              </div>
            )) : <p className="route-example-empty">{selectedRoute.enabled === false
              ? "这条路线还没有例句。未启用路线可以先留空。"
              : "这条路线还没有例句。发布前至少补一条经过判断的原句。"}</p>}
          </div>
        </div>
      </div>

      <div className="route-example-contract">
        <strong>典型例句</strong>描述路线中心；<strong>边界例句</strong>使用独立护栏向量，不进入等权路线中心。no-recall 路线达到阈值并领先时准备 skip；若反向边界分数更强，护栏只撤销这次 skip，仍由正常检索与证据门决定是否注入。
      </div>

      <footer className="route-editor-footer">
        <div>
          <strong>{dirty
            ? draftSyncState.status === "saving" ? "正在保存服务器草稿" : "草稿保存在私有服务器"
            : boundaryRebuildNeeded
              ? "生产数据未变，但边界护栏等待重建"
              : `与${routeSnapshot.deploymentState === "production" ? "生产快照" : "本地候选"}一致`}</strong>
          <span>{publishState.message || (boundaryRebuildNeeded ? `当前 ${routeSnapshot.boundaryExampleCount} 条边界例句尚未进入独立护栏索引。` : "") || draftSyncState.message || datasetState.message || "发布时校验全部类别、递增版本，并原子重建所有已启用类别的 Router 向量；no-recall 命中后直接 skip。"}</span>
        </div>
        <div>
          <button type="button" onClick={loadPublishedDataset} disabled={datasetState.status === "loading" || publishState.status === "publishing"} title="重新读取线上完整数据集"><ArrowClockwise size={14} className={datasetState.status === "loading" ? "is-spinning" : ""} />核对线上</button>
          <button type="button" onClick={resetDraft} disabled={!dirty || publishState.status === "publishing"}>撤销草稿</button>
          <button
            type="button"
            className="basement-primary-action"
            onClick={publishRoutes}
            disabled={(!dirty && !boundaryRebuildNeeded) || datasetState.status !== "ready" || publishState.status === "publishing"}
            title={datasetState.status === "ready" ? "校验完整数据集、重建全部典型例句向量并原子切换" : "先核对线上 Router 版本"}
          >{publishState.status === "publishing" ? <><ArrowClockwise size={14} className="is-spinning" />正在重建</> : boundaryRebuildNeeded && !dirty ? "重建并发布边界护栏" : "发布并重建向量"}</button>
        </div>
      </footer>
    </section>
  );
}

export function BasementPage() {
  const [activeTool, setActiveTool] = useState("recall");

  useEffect(() => {
    document.querySelector(".basement-experience")?.scrollTo({ top: 0 });
  }, [activeTool]);

  return (
    <div className="basement-experience">
      <header className="basement-page-header">
        <div>
          <span>不常开灯的地方</span>
          <h1>地下室</h1>
        </div>
        <p>先把记忆入口看清楚，再决定要不要改变它。</p>
      </header>

      <div className="basement-layout">
        <aside className="basement-tool-index" aria-label="地下室工具">
          <button type="button" className={activeTool === "recall" ? "is-active" : ""} onClick={() => setActiveTool("recall")}>
            <Flask size={19} weight="light" aria-hidden="true" />
            <span><strong>召回模拟</strong><small>走一遍真实入口</small></span>
          </button>
          <button type="button" className={activeTool === "observations" ? "is-active" : ""} onClick={() => setActiveTool("observations")}>
            <Eye size={19} weight="light" aria-hidden="true" />
            <span><strong>召回观察</strong><small>看真实运行与误差</small></span>
          </button>
          <button type="button" className={activeTool === "domains" ? "is-active" : ""} onClick={() => setActiveTool("domains")}>
            <WarningCircle size={19} weight="light" aria-hidden="true" />
            <span><strong>主域边界</strong><small>单独排除或恢复</small></span>
          </button>
          <button type="button" className={activeTool === "examples" ? "is-active" : ""} onClick={() => setActiveTool("examples")}>
            <PencilSimple size={19} weight="light" aria-hidden="true" />
            <span><strong>例句维护</strong><small>审核 Router 边界</small></span>
          </button>
          <button type="button" className={activeTool === "revisions" ? "is-active" : ""} onClick={() => setActiveTool("revisions")}>
            <GitDiff size={19} weight="light" aria-hidden="true" />
            <span><strong>修订箱</strong><small>来源与叙事卷之间</small></span>
          </button>
          <button type="button" className={activeTool === "relationships" ? "is-active" : ""} onClick={() => setActiveTool("relationships")}>
            <ShareNetwork size={19} weight="light" aria-hidden="true" />
            <span><strong>关系提案</strong><small>审核 Scene 之间的边</small></span>
          </button>
        </aside>

        {activeTool === "recall"
          ? <RecallSimulator />
          : activeTool === "observations"
            ? <BasementRecallObservation />
            : activeTool === "domains"
              ? <DomainPolicyEditor />
              : activeTool === "examples"
                ? <RouteExampleEditor />
                : activeTool === "revisions"
                  ? <BasementRevisionInbox />
                  : <BasementRelationshipProposals />}
      </div>
    </div>
  );
}
