import { useEffect, useMemo, useRef, useState } from "react";
import { flushSync } from "react-dom";
import {
  ArrowLeft,
  ArrowRight,
  CaretDown,
  CircleNotch,
  PencilSimple,
  X,
} from "@phosphor-icons/react";
import {
  loadNarrativeRolls,
  previewNarrativeRoll,
  readFallbackNarrativeRolls,
  saveNarrativeRollBody,
} from "../storage/narrativeStore.js";

const transitionTo = (update) => {
  if (!document.startViewTransition) {
    update();
    return;
  }

  document.startViewTransition(() => {
    flushSync(update);
  });
};

const formatRange = (roll) => `${roll.timeStart} — ${roll.timeEnd}`;

const turnShelfWithWheel = (event) => {
  const shelf = event.currentTarget;
  if (shelf.scrollWidth <= shelf.clientWidth) return;

  const wheelDelta = Math.abs(event.deltaY) >= Math.abs(event.deltaX)
    ? event.deltaY
    : event.deltaX;
  const distance = event.deltaMode === 1
    ? wheelDelta * 40
    : event.deltaMode === 2
      ? wheelDelta * shelf.clientWidth
      : wheelDelta;
  const nextScrollLeft = Math.max(
    0,
    Math.min(shelf.scrollWidth - shelf.clientWidth, shelf.scrollLeft + distance),
  );

  if (nextScrollLeft === shelf.scrollLeft) return;
  event.preventDefault();
  shelf.scrollLeft = nextScrollLeft;
};

export function NarrativePage() {
  const shelfRef = useRef(null);
  const [narrativeRolls, setNarrativeRolls] = useState(readFallbackNarrativeRolls);
  const [selectedRollId, setSelectedRollId] = useState(null);
  const [editorOpen, setEditorOpen] = useState(false);
  const [previewBody, setPreviewBody] = useState("");
  const [previewDiff, setPreviewDiff] = useState("");
  const [previewMode, setPreviewMode] = useState("");
  const [previewError, setPreviewError] = useState("");
  const [previewing, setPreviewing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState("");
  const selectedRoll = useMemo(
    () => narrativeRolls.find((roll) => roll.id === selectedRollId) ?? null,
    [narrativeRolls, selectedRollId],
  );

  useEffect(() => {
    let cancelled = false;
    loadNarrativeRolls().then((rolls) => {
      if (!cancelled && rolls?.length) setNarrativeRolls(rolls);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedRoll) return undefined;
    const closeOnEscape = (event) => {
      if (event.key === "Escape") {
        if (editorOpen) {
          setEditorOpen(false);
        } else {
          transitionTo(() => setSelectedRollId(null));
        }
      }
    };
    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [editorOpen, selectedRoll]);

  useEffect(() => {
    const shelf = shelfRef.current;
    if (!shelf) return undefined;
    shelf.addEventListener("wheel", turnShelfWithWheel, { passive: false });
    return () => shelf.removeEventListener("wheel", turnShelfWithWheel);
  }, [selectedRoll]);

  const openRoll = (rollId) => {
    transitionTo(() => setSelectedRollId(rollId));
  };

  const closeRoll = () => {
    setEditorOpen(false);
    setPreviewBody("");
    setPreviewDiff("");
    setPreviewMode("");
    setPreviewError("");
    setSaveMessage("");
    transitionTo(() => setSelectedRollId(null));
  };

  const openEditor = () => {
    setPreviewBody(selectedRoll?.body || selectedRoll?.paragraphs?.join("\n\n") || "");
    setPreviewDiff("");
    setPreviewMode("");
    setPreviewError("");
    setSaveMessage("");
    setEditorOpen(true);
  };

  const closeEditor = () => {
    setEditorOpen(false);
    setPreviewBody("");
    setPreviewDiff("");
    setPreviewMode("");
    setPreviewError("");
    setSaveMessage("");
  };

  const generatePreview = async (mode) => {
    if (!selectedRoll || previewing) return;
    setEditorOpen(true);
    setPreviewBody(selectedRoll.body || selectedRoll.paragraphs.join("\n\n"));
    setPreviewDiff("");
    setPreviewMode(mode);
    setPreviewing(true);
    setPreviewError("");
    setSaveMessage("");
    try {
      const result = await previewNarrativeRoll(selectedRoll, mode);
      if (result.status === "insufficient") {
        setPreviewBody(selectedRoll.body || selectedRoll.paragraphs.join("\n\n"));
        setPreviewDiff("");
        setPreviewMode(mode);
        setPreviewError(result.issues?.join("；") || "当前绑定材料不足以生成可信正文。");
        return;
      }
      setPreviewBody(result.body);
      setPreviewDiff(result.diff || "");
      setPreviewMode(mode);
    } catch (error) {
      setPreviewError(error.message || "没有生成这次预览。");
    } finally {
      setPreviewing(false);
    }
  };

  const saveBody = async () => {
    if (!selectedRoll || saving || previewing || !previewBody.trim()) return;
    setSaving(true);
    setPreviewError("");
    setSaveMessage("");
    try {
      const result = await saveNarrativeRollBody(selectedRoll, previewBody);
      const paragraphs = previewBody
        .replace(/\r\n?/g, "\n")
        .trim()
        .split(/\n\s*\n/)
        .map((paragraph) => paragraph.trim())
        .filter(Boolean);
      setNarrativeRolls((rolls) => rolls.map((roll) => (
        roll.id === selectedRoll.id
          ? {
              ...roll,
              body: previewBody.trim(),
              paragraphs,
              revision: Number(result.revision || roll.revision + 1),
              documentHash: String(result.document_sha256 || roll.documentHash),
            }
          : roll
      )));
      setPreviewDiff("");
      setPreviewMode("");
      setSaveMessage(`已保存为 revision ${result.revision}`);
    } catch (error) {
      setPreviewError(error.message || "这次正文没有保存。");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className={`narrative-experience${selectedRoll ? " is-reading" : ""}`}>
      {!selectedRoll ? (
        <section className="narrative-library" aria-labelledby="narrative-title">
          <header className="narrative-library__header">
            <div>
              <span className="narrative-kicker">NARRATIVE ROLLS</span>
              <h1 id="narrative-title">叙事卷</h1>
              <p>把散落的 Scene 放回时间里，看它们怎样慢慢长成一条路。</p>
            </div>
            <div className="narrative-library__count" aria-label="目前共有五卷">
              <strong>{String(narrativeRolls.length).padStart(2, "0")}</strong>
              <span>卷，仍在生长</span>
            </div>
          </header>

          <div className="narrative-shelf-shell">
            <div
              className="narrative-shelf"
              role="list"
              aria-label="叙事卷书架"
              ref={shelfRef}
            >
              {narrativeRolls.map((roll, index) => (
                <button
                  className={`narrative-book narrative-book--${roll.tone}`}
                  key={roll.id}
                  type="button"
                  role="listitem"
                  style={{
                    "--roll-index": index,
                    viewTransitionName: `roll-${roll.id}`,
                  }}
                  onClick={() => openRoll(roll.id)}
                  aria-label={`打开第 ${roll.volume} 卷：${roll.title}，${roll.subtitle}`}
                >
                  <span className="narrative-book__page narrative-book__page--one" aria-hidden="true" />
                  <span className="narrative-book__page narrative-book__page--two" aria-hidden="true" />
                  <span className="narrative-book__page narrative-book__page--three" aria-hidden="true" />
                  <span className="narrative-book__cover">
                    <span className="narrative-book__volume">VOL. {roll.volume}</span>
                    <span
                      className={`narrative-book__title${/[A-Za-z]/u.test(roll.spineTitle) ? " narrative-book__title--latin" : ""}`}
                    >
                      {roll.spineTitle}
                    </span>
                    <span className="narrative-book__subtitle">{roll.spineSubtitle}</span>
                    <span className="narrative-book__footer">
                      <span>{roll.timeStart}</span>
                      <ArrowRight size={15} weight="light" aria-hidden="true" />
                    </span>
                  </span>
                </button>
              ))}

              <div className="narrative-book narrative-book--future" role="listitem" aria-label="未来的未命名叙事卷">
                <CircleNotch size={17} weight="light" aria-hidden="true" />
                <span>尚未命名</span>
                <small>仍在长出来</small>
              </div>
            </div>
            <div className="narrative-shelf__line" aria-hidden="true" />
            <p className="narrative-shelf__hint">滚轮向后翻 · 悬停翻开一点 · 点击阅读</p>
          </div>
        </section>
      ) : (
        <article className="narrative-reader" aria-labelledby="narrative-reader-title">
          <nav className="narrative-reader__topbar" aria-label="叙事卷阅读导航">
            <button className="narrative-reader__back" type="button" onClick={closeRoll}>
              <ArrowLeft size={18} weight="light" aria-hidden="true" />
              回到书架
            </button>
            <div className="narrative-reader__tools">
              <span>
                VOL. {selectedRoll.volume}
                <i aria-hidden="true">·</i>
                {selectedRoll.status}
              </span>
              <div className="narrative-reader__actions" role="group" aria-label="叙事卷操作">
                <button
                  className={editorOpen && !previewMode ? "is-active" : ""}
                  type="button"
                  disabled={previewing || saving}
                  onClick={openEditor}
                >
                  <PencilSimple size={15} weight="light" aria-hidden="true" />
                  编辑
                </button>
                <button
                  className={previewMode === "update" ? "is-active" : ""}
                  type="button"
                  disabled={previewing || saving}
                  onClick={() => generatePreview("update")}
                >
                  {previewing && previewMode === "update" ? <CircleNotch className="is-spinning" size={15} aria-hidden="true" /> : null}
                  更新
                </button>
                <button
                  className={previewMode === "rewrite" ? "is-active" : ""}
                  type="button"
                  disabled={previewing || saving}
                  onClick={() => generatePreview("rewrite")}
                >
                  {previewing && previewMode === "rewrite" ? <CircleNotch className="is-spinning" size={15} aria-hidden="true" /> : null}
                  重写
                </button>
                {editorOpen ? (
                  <button className="is-close" type="button" aria-label="关闭编辑区" onClick={closeEditor}>
                    <X size={15} weight="light" aria-hidden="true" />
                  </button>
                ) : null}
              </div>
            </div>
          </nav>

          <section
            className="narrative-reader__article"
            style={{ viewTransitionName: `roll-${selectedRoll.id}` }}
          >
            <header className="narrative-reader__article-header">
              <span>
                NARRATIVE PROJECTION · {selectedRoll.scope?.toUpperCase() ?? "ARC"} · VOL. {selectedRoll.volume}
              </span>
              <h1 id="narrative-reader-title">{selectedRoll.title}</h1>
              <p>{selectedRoll.subtitle}</p>
              <small>{selectedRoll.description}</small>
              <dl className="narrative-reader__facts">
                <div>
                  <dt>状态</dt>
                  <dd>{selectedRoll.projectionStatus ?? selectedRoll.status}</dd>
                </div>
                <div>
                  <dt>证据范围</dt>
                  <dd>{formatRange(selectedRoll)}</dd>
                </div>
                <div>
                  <dt>来源</dt>
                  <dd>{selectedRoll.sourceCount ?? selectedRoll.sceneCount} 条</dd>
                </div>
              </dl>
            </header>

            {editorOpen ? (
              <section className="narrative-editor" aria-label="叙事卷编辑预览">
                <header>
                  <div>
                    <span>{previewMode ? "WRITER PREVIEW" : "BODY EDITOR"}</span>
                    <h2>{previewMode ? (previewMode === "update" ? "更新预览" : "重写预览") : "手改正文"}</h2>
                  </div>
                  <small>{previewMode ? "预览不会自动发布，确认后再保存" : "保存只改正文，不调用 Writer"}</small>
                </header>
                <div className="narrative-editor__workspace">
                  <div className="narrative-editor__paper">
                    <textarea
                      aria-label="叙事卷正文预览"
                      value={previewBody}
                      onChange={(event) => {
                        setPreviewBody(event.target.value);
                        setSaveMessage("");
                      }}
                      spellCheck="false"
                    />
                    {previewError ? <p className="narrative-editor__error" role="alert">{previewError}</p> : null}
                    {saveMessage ? <p className="narrative-editor__saved" role="status">{saveMessage}</p> : null}
                  </div>
                  <aside className="narrative-editor__materials" aria-label="当前绑定材料">
                    <span>当前材料</span>
                    <strong>{selectedRoll.sources?.length || 0} 条</strong>
                    <ol>
                      {(selectedRoll.sources || []).map((source) => (
                        <li key={`editor:${source.type || "scene"}:${source.id}`}>
                          <em>{source.typeLabel || "Scene"}</em>
                          <span>{source.title}</span>
                        </li>
                      ))}
                    </ol>
                  </aside>
                </div>
                {previewDiff ? (
                  <details className="narrative-editor__diff">
                    <summary>查看与当前正文的差异</summary>
                    <pre>{previewDiff}</pre>
                  </details>
                ) : null}
                <footer>
                  <button className="is-primary" type="button" disabled={saving || previewing || !previewBody.trim()} onClick={saveBody}>
                    {saving ? <CircleNotch className="is-spinning" size={16} aria-hidden="true" /> : null}
                    保存
                  </button>
                </footer>
              </section>
            ) : (
              <section className="narrative-manuscript__body" aria-label="叙事正文">
                {selectedRoll.paragraphs.map((paragraph) => <p key={paragraph}>{paragraph}</p>)}
              </section>
            )}

            {!editorOpen && selectedRoll.sources ? (
              <details className="narrative-ledger">
                <summary>
                  <span>
                    <strong>来源账</strong>
                    <small>{selectedRoll.sources.length} 条绑定材料</small>
                  </span>
                  <CaretDown size={17} weight="light" aria-hidden="true" />
                </summary>
                <ol>
                  {selectedRoll.sources.map((source) => (
                    <li key={`${source.type || "scene"}:${source.id}`}>
                      <time>{source.date || "—"}</time>
                      <span>
                        <span className="narrative-ledger__source-title">
                          <em>{source.typeLabel || "Scene"}</em>
                          <strong>{source.title}</strong>
                        </span>
                        <small>{source.purpose || source.id}</small>
                      </span>
                    </li>
                  ))}
                </ol>
              </details>
            ) : !editorOpen ? (
              <section className="narrative-scenes" aria-labelledby="narrative-scenes-title">
                <div>
                  <h2 id="narrative-scenes-title">卷中的 Scene</h2>
                  <span>{selectedRoll.sourceCount ?? selectedRoll.sceneCount} 条来源</span>
                </div>
                <ul>
                  {selectedRoll.sceneNames.map((scene) => <li key={scene}>{scene}</li>)}
                </ul>
              </section>
            ) : null}
          </section>
        </article>
      )}
    </div>
  );
}
