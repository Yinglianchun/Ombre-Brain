import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowClockwise,
  Check,
  DownloadSimple,
  Eye,
  Plus,
  Question,
  WarningCircle,
  X,
} from "@phosphor-icons/react";
import { semanticRouteSnapshot } from "../data/basement.js";
import {
  appendSemanticRouteDraftExample,
  readRecallObservationReviews,
  readSemanticRouteDraft,
  readServerSemanticRouteDraft,
  saveRecallObservationReviews,
  saveSemanticRouteDraft,
  saveServerSemanticRouteDraft,
} from "../storage/basementStore.js";
import { buildRecallObservationTrainingExport } from "../storage/recallObservationExport.js";

const snapshotRouteLabels = Object.fromEntries(
  semanticRouteSnapshot.routes.map((route) => [route.name, route.label || route.name]),
);
const snapshotRouteActions = Object.fromEntries(
  semanticRouteSnapshot.routes.map((route) => [route.name, route.action || ""]),
);

const outcomeLabels = {
  injected: "已注入",
  no_match: "未命中",
  skip: "已跳过",
};

const sourceLabels = {
  hook: "Hook 实际送入",
  gateway: "Gateway 准备注入",
};

const verdicts = [
  { key: "correct", label: "正确", icon: Check },
  { key: "false_positive", label: "误召", icon: X },
  { key: "missed", label: "漏召", icon: Plus },
  { key: "uncertain", label: "不确定", icon: Question },
];

const candidateRelevances = [
  { key: "core", label: "核心相关" },
  { key: "weak", label: "弱相关" },
  { key: "irrelevant", label: "无关" },
];

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function normalizeScore(item) {
  const raw = item?.score?.final ?? item?.score?.semantic ?? item?.score?.keyword ?? item?.score;
  const score = Number(raw);
  if (!Number.isFinite(score)) return "";
  return score <= 1 ? `${(score * 100).toFixed(1)}%` : score.toFixed(2);
}

function normalizeScoreValue(item) {
  const raw = item?.score?.final ?? item?.score?.semantic ?? item?.score?.keyword ?? item?.score;
  const score = Number(raw);
  return Number.isFinite(score) ? score : null;
}

function normalizeObservation(row) {
  const payload = row?.payload && typeof row.payload === "object" ? row.payload : {};
  const semantic = payload.semantic_recall_debug && typeof payload.semantic_recall_debug === "object"
    ? payload.semantic_recall_debug
    : {};
  const why = payload.recall_why_summary && typeof payload.recall_why_summary === "object"
    ? payload.recall_why_summary
    : {};
  const injectedDetails = asArray(why.injected);
  const injectedIds = asArray(payload.injected_bucket_ids);
  const injected = injectedDetails.length
    ? injectedDetails.map((item) => ({
      id: item.bucket_id || item.id || "",
      title: item.bucket_name || item.title || item.bucket_id || item.id || "未命名记忆",
      score: normalizeScore(asArray(item.evidence)[0] || item),
      scoreValue: normalizeScoreValue(asArray(item.evidence)[0] || item),
    }))
    : injectedIds.map((id) => ({ id, title: id, score: "", scoreValue: null }));
  const action = String(semantic.applied_action || semantic.action || "").trim();
  const query = String(payload.query || payload.query_preview || payload.original_query || payload.user_query || "").trim();
  const outcome = action === "skip" ? "skip" : injected.length ? "injected" : "no_match";
  const confidence = Number(semantic.confidence);
  return {
    id: String(row?.id ?? `${row?.session_id || "session"}-${row?.round_id || "round"}`),
    createdAt: row?.created_at || "",
    query: query || "旧记录未保留原句",
    queryAvailable: Boolean(query),
    route: String(semantic.route || "").trim(),
    action,
    observedAction: action,
    sessionId: row?.session_id ?? row?.sessionId ?? payload.session_id ?? "",
    confidence: Number.isFinite(confidence) ? `${(confidence * 100).toFixed(1)}%` : "",
    outcome,
    injected,
    source: "gateway",
    trigger: String(semantic.reason || "").trim(),
    hookOutcome: "",
  };
}

function normalizeBridgeObservation(row, routeActions = snapshotRouteActions) {
  const injectedIds = asArray(row?.gateway_memory_injected_ids).map((item) => String(item || "").trim()).filter(Boolean);
  const injectedDetails = asArray(row?.gateway_memory_items);
  const detailsById = new Map(injectedDetails.map((item) => [String(item?.id || "").trim(), item]));
  const injected = injectedIds.map((id) => {
    const item = detailsById.get(id) || {};
    return {
      id,
      title: String(item.title || id),
      score: normalizeScore(item),
      scoreValue: normalizeScoreValue(item),
    };
  });
  const hookOutcome = String(row?.hook_memory_outcome || "").trim();
  const trigger = String(row?.gateway_memory_trigger || "").trim();
  const route = String(row?.gateway_memory_route || "").trim();
  const routeAction = routeActions[route] || "";
  const query = String(row?.query || "").trim();
  const outcome = injected.length || hookOutcome === "injected"
    ? "injected"
    : routeAction === "skip" ? "skip"
      : ["no_match", "gateway_no_match", "below_threshold", "insufficient_margin"].includes(trigger) ? "no_match" : "skip";
  return {
    id: `hook-${row?.id}`,
    createdAt: row?.created_at || "",
    query: query || "原句未记录",
    queryAvailable: Boolean(query),
    route,
    action: routeAction,
    observedAction: routeAction,
    sessionId: row?.session_id ?? row?.sessionId ?? "",
    confidence: "",
    outcome,
    injected,
    source: "hook",
    trigger,
    hookOutcome,
    messageId: row?.id,
  };
}

function formatObservedAt(value) {
  if (!value) return "时间未记录";
  const raw = String(value).trim();
  const sqliteUtc = /^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?$/.test(raw);
  const date = new Date(sqliteUtc ? `${raw.replace(" ", "T")}Z` : raw);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat("zh-CN", {
    timeZone: "Asia/Shanghai",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(date);
}

export function BasementRecallObservation() {
  const [draftRoutes, setDraftRoutes] = useState(readSemanticRouteDraft);
  const [publishedRoutes, setPublishedRoutes] = useState(semanticRouteSnapshot.routes);
  const [draftDatasetVersion, setDraftDatasetVersion] = useState(semanticRouteSnapshot.datasetVersion);
  const draftRevisionRef = useRef(0);
  const [status, setStatus] = useState("loading");
  const [errors, setErrors] = useState({});
  const [datasets, setDatasets] = useState({ hook: [], gateway: [] });
  const [source, setSource] = useState("hook");
  const [filter, setFilter] = useState("injected");
  const [reviews, setReviews] = useState(readRecallObservationReviews);
  const [draftForms, setDraftForms] = useState({});
  const [draftNotices, setDraftNotices] = useState({});
  const [exportNotice, setExportNotice] = useState("");

  const load = useCallback(async () => {
    setStatus("loading");
    setErrors({});
    const requestJson = async (url, limit) => {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ limit }),
      });
      const payload = await response.json();
      if (!response.ok || payload?.status !== "ok") {
        throw new Error(payload?.message || payload?.error || "没有读到记录");
      }
      return asArray(payload.items);
    };
    const [hookResult, gatewayResult] = await Promise.allSettled([
      requestJson("/__serein/haven-bridge/hook-injections", 100),
      requestJson("/__serein/gateway/injections", 50),
    ]);
    const nextErrors = {};
    const hookItems = hookResult.status === "fulfilled"
      ? hookResult.value
      : (nextErrors.hook = hookResult.reason?.message || "无法读取 Hook 账本", []);
    const gatewayItems = gatewayResult.status === "fulfilled"
      ? gatewayResult.value
      : (nextErrors.gateway = gatewayResult.reason?.message || "无法读取 Gateway 记录", []);
    setDatasets({ hook: hookItems, gateway: gatewayItems });
    setErrors(nextErrors);
    setStatus(Object.keys(nextErrors).length === 2 ? "error" : "done");
  }, []);

  useEffect(() => { load(); }, [load]);

  useEffect(() => {
    let cancelled = false;
    const hydrateDraftRoutes = async () => {
      try {
        const [publishedResponse, serverState] = await Promise.all([
          fetch("/__serein/gateway/semantic-routes"),
          readServerSemanticRouteDraft(),
        ]);
        const published = await publishedResponse.json();
        if (!publishedResponse.ok || !Array.isArray(published.routes)) throw new Error("route_dataset_unavailable");
        const datasetVersion = Number(published.dataset_version);
        const publishedRoutes = published.routes.map((route) => ({
          ...route,
          label: route.label || route.name,
          enabled: route.enabled !== false,
          utterances: Array.isArray(route.utterances) ? route.utterances : [],
        }));
        if (!cancelled) setPublishedRoutes(publishedRoutes);
        let nextRoutes;
        if (serverState.draft?.baseDatasetVersion === datasetVersion) {
          nextRoutes = serverState.draft.routes;
          draftRevisionRef.current = serverState.draft.revision || 0;
        } else {
          const snapshot = { ...semanticRouteSnapshot, datasetVersion, routes: publishedRoutes };
          nextRoutes = readSemanticRouteDraft(snapshot);
          if (JSON.stringify(nextRoutes) !== JSON.stringify(publishedRoutes)) {
            const migrated = await saveServerSemanticRouteDraft(nextRoutes, datasetVersion, 0);
            draftRevisionRef.current = migrated.draft.revision;
          } else {
            draftRevisionRef.current = 0;
          }
        }
        if (cancelled) return;
        saveSemanticRouteDraft(nextRoutes, datasetVersion);
        setDraftDatasetVersion(datasetVersion);
        setDraftRoutes(nextRoutes);
      } catch {
        // Keep the local draft available when the private server bridge is temporarily unreachable.
      }
    };
    hydrateDraftRoutes();
    return () => { cancelled = true; };
  }, []);

  const publishedRouteActions = useMemo(() => ({
    ...snapshotRouteActions,
    ...Object.fromEntries(publishedRoutes.map((route) => [route.name, route.action || ""])),
  }), [publishedRoutes]);
  const publishedRouteLabels = useMemo(() => ({
    ...snapshotRouteLabels,
    ...Object.fromEntries(publishedRoutes.map((route) => [route.name, route.label || route.name])),
  }), [publishedRoutes]);
  const normalizedDatasets = useMemo(() => ({
    hook: asArray(datasets.hook).map((row) => normalizeBridgeObservation(row, publishedRouteActions)),
    gateway: asArray(datasets.gateway).map(normalizeObservation),
  }), [datasets, publishedRouteActions]);
  const items = normalizedDatasets[source];
  const exportItems = useMemo(
    () => [...normalizedDatasets.hook, ...normalizedDatasets.gateway],
    [normalizedDatasets],
  );
  const exportPayload = useMemo(
    () => buildRecallObservationTrainingExport(exportItems, reviews),
    [exportItems, reviews],
  );
  const exportSummary = exportPayload.summary;
  const filteredItems = useMemo(
    () => filter === "all" ? items : items.filter((item) => item.outcome === filter),
    [filter, items],
  );

  const setVerdict = (item, verdict) => {
    const nextReviews = {
      ...reviews,
      [item.id]: {
        ...(reviews[item.id] || {}),
        verdict,
        observedAt: item.createdAt,
        query: item.query,
        updatedAt: new Date().toISOString(),
      },
    };
    setReviews(nextReviews);
    saveRecallObservationReviews(nextReviews);
    if (["false_positive", "missed"].includes(verdict)) {
      setDraftForms((current) => ({
        ...current,
        [item.id]: current[item.id] || { routeName: "", role: "typical" },
      }));
    }
  };

  const setCandidateRelevance = (item, memoryId, relevance) => {
    if (!memoryId) return;
    const currentReview = reviews[item.id] || {};
    const candidateReviews = { ...(currentReview.candidateReviews || {}) };
    if (candidateReviews[memoryId] === relevance) delete candidateReviews[memoryId];
    else candidateReviews[memoryId] = relevance;
    const nextReviews = {
      ...reviews,
      [item.id]: {
        ...currentReview,
        candidateReviews,
        observedAt: item.createdAt,
        query: item.query,
        updatedAt: new Date().toISOString(),
      },
    };
    setReviews(nextReviews);
    saveRecallObservationReviews(nextReviews);
  };

  const addDraft = async (item) => {
    const form = draftForms[item.id] || {};
    if (!form.routeName) {
      setDraftNotices((current) => ({ ...current, [item.id]: "先选这句话应该属于哪条路线。" }));
      return;
    }
    const verdict = reviews[item.id]?.verdict;
    const result = appendSemanticRouteDraftExample({
      routeName: form.routeName,
      text: item.query,
      role: form.role,
      origin: verdict === "missed" ? "online_false_negative" : "online_false_positive",
      routes: draftRoutes,
      baseDatasetVersion: draftDatasetVersion,
    });
    setDraftRoutes(result.routes);
    let status = result.status;
    if (status === "added") {
      try {
        const saved = await saveServerSemanticRouteDraft(
          result.routes,
          draftDatasetVersion,
          draftRevisionRef.current,
        );
        draftRevisionRef.current = saved.draft.revision;
      } catch (error) {
        status = error.message === "route_draft_revision_conflict" ? "conflict" : "save_failed";
      }
    }
    const notice = status === "added"
      ? "已进入服务器例句草稿；发布前不会改变线上 Router。"
      : status === "duplicate" ? "这句已经在例句草稿或生产快照里。"
        : status === "conflict" ? "服务器草稿刚在别处变化，重新打开召回观察后再转入。"
          : status === "save_failed" ? "只保住了当前浏览器草稿，服务器暂时没有保存成功。"
            : "这条记录暂时不能转成草稿。";
    setDraftNotices((current) => ({ ...current, [item.id]: notice }));
  };

  const counts = useMemo(() => ({
    injected: items.filter((item) => item.outcome === "injected").length,
    no_match: items.filter((item) => item.outcome === "no_match").length,
    skip: items.filter((item) => item.outcome === "skip").length,
  }), [items]);

  const downloadTrainingExport = useCallback(() => {
    if (!exportSummary.total_observations) {
      setExportNotice("当前没有已加载的观察记录，暂时没有可导出的内容。");
      return;
    }
    const blob = new Blob([JSON.stringify(exportPayload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    anchor.href = url;
    anchor.download = `serein-recall-training-${stamp}.json`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
    setExportNotice(`已导出 ${exportSummary.total_observations} 条观察；准备脚本会再次复核可用、拒绝与缺失。`);
  }, [exportPayload, exportSummary]);

  return (
    <section className="basement-workbench" aria-labelledby="recall-observation-title">
      <header className="basement-workbench__header">
        <div>
          <span className="basement-kicker">真实运行，只读观察</span>
          <h2 id="recall-observation-title">召回观察</h2>
          <p>先看 Gateway 准备了什么，再核对 Haven Bridge hook 真正送入模型的内容。</p>
        </div>
        <div className="observation-header-actions">
          <button className="observation-refresh" type="button" onClick={load} disabled={status === "loading"}>
            <ArrowClockwise size={16} className={status === "loading" ? "is-spinning" : ""} aria-hidden="true" />
            刷新
          </button>
          <button
            className="observation-export-button"
            type="button"
            onClick={downloadTrainingExport}
            disabled={!exportSummary.total_observations || status === "loading"}
          >
            <DownloadSimple size={16} aria-hidden="true" />
            导出训练标注
          </button>
        </div>
      </header>

      <div className="observation-toolbar" role="group" aria-label="召回结果筛选">
        <div className="observation-source-switch" role="group" aria-label="观测来源">
          {Object.entries(sourceLabels).map(([key, label]) => (
            <button type="button" className={source === key ? "is-active" : ""} key={key} onClick={() => setSource(key)}>
              {label}<span>{datasets[key].length}</span>
            </button>
          ))}
        </div>
        <i aria-hidden="true" />
        {[
          ["injected", "已注入", counts.injected],
          ["no_match", "未命中", counts.no_match],
          ["skip", "已跳过", counts.skip],
          ["all", "全部", items.length],
        ].map(([key, label, count]) => (
          <button type="button" className={filter === key ? "is-active" : ""} key={key} onClick={() => setFilter(key)}>
            {label}<span>{count}</span>
          </button>
        ))}
        <p>{source === "hook" ? "这里只显示真正送入模型的 hook 账本。" : "这里是 Gateway 自己记录的准备结果。"}</p>
      </div>

      <div className="observation-export-summary" aria-live="polite">
        <div className="observation-export-summary__copy">
          <span>训练标注导出</span>
          <p>只含原句、整轮判断、已评单卡的 ID/相关度、路线/动作、来源与时间分组；不含完整 prompt、注入正文或上下文。</p>
        </div>
        <dl>
          <div><dt>可用</dt><dd>{exportSummary.available}</dd></div>
          <div><dt>拒绝</dt><dd>{exportSummary.rejected}</dd></div>
          <div><dt>缺失</dt><dd>{exportSummary.missing}</dd></div>
        </dl>
        {exportNotice && <p className="observation-export-summary__notice" role="status">{exportNotice}</p>}
      </div>

      {status === "loading" && (
        <div className="observation-loading" aria-live="polite">
          {[0, 1, 2].map((item) => <i key={item} />)}
        </div>
      )}

      {status === "error" && (
        <div className="basement-error" role="alert">
          <WarningCircle size={19} aria-hidden="true" />
          <div><strong>两个观测入口都没有读到</strong><p>{Object.values(errors).join(" / ")}</p></div>
        </div>
      )}

      {status === "done" && errors[source] && (
        <div className="basement-error" role="alert">
          <WarningCircle size={19} aria-hidden="true" />
          <div><strong>{sourceLabels[source]}暂时不可用</strong><p>{errors[source]}</p></div>
        </div>
      )}

      {status === "done" && !filteredItems.length && (
        <div className="basement-empty-state">
          <Eye size={23} weight="light" aria-hidden="true" />
          <span>{source === "hook" && !items.length ? "还没有 hook 注入记录" : "这个筛选里还没有记录"}</span>
          <p>{source === "hook" && !items.length
            ? "审计合同已经在线。下一条真实用户消息经过 Haven Bridge hook 后会出现在这里。"
            : "可以换一个结果类型，或等下一轮真实聊天经过对应入口。"}</p>
        </div>
      )}

      {status === "done" && filteredItems.length > 0 && (
        <div className="observation-list">
          {filteredItems.map((item) => {
            const review = reviews[item.id] || {};
            const showDraft = ["false_positive", "missed"].includes(review.verdict);
            const form = draftForms[item.id] || { routeName: "", role: "typical" };
            return (
              <article className="observation-card" key={item.id}>
                <header>
                  <div>
                    <time>{formatObservedAt(item.createdAt)} · {sourceLabels[item.source]}</time>
                    <h3>{item.query}</h3>
                  </div>
                  <span className={`observation-outcome observation-outcome--${item.outcome}`}>{outcomeLabels[item.outcome]}</span>
                </header>

                <dl className="observation-route-facts">
                  <div><dt>Router 路线</dt><dd>{item.route ? publishedRouteLabels[item.route] || item.route : "旧记录未写入"}</dd></div>
                  <div><dt>{item.source === "hook" ? "Gateway trigger" : "动作"}</dt><dd>{item.source === "hook" ? item.trigger || "未记录" : item.action || "旧记录未写入"}</dd></div>
                  <div><dt>{item.source === "hook" ? "Hook outcome" : "置信度"}</dt><dd>{item.source === "hook" ? item.hookOutcome || "未记录" : item.confidence || "旧记录未写入"}</dd></div>
                </dl>

                <div className="observation-injections">
                  <span>{item.source === "hook" ? "真正送入模型" : "Gateway 准备注入"}</span>
                  {item.injected.length ? item.injected.map((memory) => (
                    <div className="observation-memory-row" key={`${item.id}-${memory.id}`}>
                      <strong>{memory.title}</strong>
                      <code>{memory.id || "ID 未记录"}</code>
                      <em>{memory.score || "score 未记录"}</em>
                      {item.source === "hook" && item.outcome === "injected" && memory.id && (
                        <div className="observation-memory-review" role="group" aria-label={`记忆相关度：${memory.title}`}>
                          <span>单卡相关度</span>
                          {candidateRelevances.map(({ key, label }) => (
                            <button
                              type="button"
                              className={review.candidateReviews?.[memory.id] === key ? "is-active" : ""}
                              key={key}
                              onClick={() => setCandidateRelevance(item, memory.id, key)}
                            >
                              {label}
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                  )) : <p>没有记忆进入这一轮上下文。</p>}
                </div>

                <div className="observation-review">
                  <span>人工判断</span>
                  <div role="group" aria-label={`判断：${item.query}`}>
                    {verdicts.map(({ key, label, icon: Icon }) => (
                      <button type="button" className={review.verdict === key ? "is-active" : ""} key={key} onClick={() => setVerdict(item, key)}>
                        <Icon size={14} aria-hidden="true" />{label}
                      </button>
                    ))}
                  </div>
                </div>

                {showDraft && (
                  <div className="observation-draft">
                    <div>
                      <label>
                        应属路线
                        <select value={form.routeName} onChange={(event) => setDraftForms((current) => ({
                          ...current,
                          [item.id]: { ...form, routeName: event.target.value },
                        }))}>
                          <option value="">选择路线</option>
                          {draftRoutes.map((route) => <option value={route.name} key={route.name}>{route.label || route.name}</option>)}
                        </select>
                      </label>
                      <label>
                        样本角色
                        <select value={form.role} onChange={(event) => setDraftForms((current) => ({
                          ...current,
                          [item.id]: { ...form, role: event.target.value },
                        }))}>
                          <option value="typical">典型例句</option>
                          <option value="boundary">边界例句</option>
                        </select>
                      </label>
                    </div>
                    <button type="button" onClick={() => addDraft(item)}>转为例句草稿</button>
                    {draftNotices[item.id] && <p>{draftNotices[item.id]}</p>}
                  </div>
                )}
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}
