import { useEffect, useRef, useState } from "react";
import { ArrowLeft, X } from "@phosphor-icons/react";
import { MarkdownProjection } from "./MarkdownProjection.jsx";
import { loadDreamDetail } from "../storage/dreamStore.js";

export function DreamReader({ open, dreams, selectedDreamId, onSelect, onDreamLoaded, onClose }) {
  const closeButtonRef = useRef(null);
  const articleRef = useRef(null);
  const [mobileDetailOpen, setMobileDetailOpen] = useState(false);
  const [loadingId, setLoadingId] = useState(null);
  const [error, setError] = useState("");
  const selectedDream = dreams.find((dream) => dream.id === selectedDreamId) ?? dreams[0];

  useEffect(() => {
    if (!open) {
      setMobileDetailOpen(false);
      setError("");
      return undefined;
    }
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeButtonRef.current?.focus();
    const handleKeyDown = (event) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [open, onClose]);

  useEffect(() => {
    articleRef.current?.scrollTo({ top: 0, behavior: "auto" });
    if (!open || !selectedDream?.hasBody || selectedDream.body) return;
    let active = true;
    setLoadingId(selectedDream.id);
    setError("");
    loadDreamDetail(selectedDream.id)
      .then((dream) => {
        if (active) onDreamLoaded(dream);
      })
      .catch((requestError) => {
        if (active) setError(requestError instanceof Error ? requestError.message : "这场梦暂时翻不开。");
      })
      .finally(() => {
        if (active) setLoadingId(null);
      });
    return () => {
      active = false;
    };
  }, [onDreamLoaded, open, selectedDream?.body, selectedDream?.hasBody, selectedDream?.id]);

  if (!open || !selectedDream) return null;

  const selectDream = (dreamId) => {
    onSelect(dreamId);
    setMobileDetailOpen(true);
    setError("");
  };

  const readingState = loadingId === selectedDream.id
    ? "正在想起这场梦……"
    : error || (selectedDream.hasBody ? "梦还没有被翻开。" : "这场梦已经散去了。");

  return (
    <div className="window-shadow-layer dream-layer" role="dialog" aria-modal="true" aria-labelledby="dream-reader-title">
      <div className="window-shadow-layer__veil" aria-hidden="true" onClick={onClose} />

      <section className={`window-shadow-reader dream-reader${mobileDetailOpen ? " is-mobile-detail-open" : ""}`}>
        <header className="window-shadow-reader__topbar">
          <div>
            <span>DREAMS</span>
            <h2 id="dream-reader-title">梦境</h2>
          </div>
          <button ref={closeButtonRef} type="button" aria-label="关闭梦境" onClick={onClose}>
            <X size={22} weight="light" aria-hidden="true" />
          </button>
        </header>

        <div className="window-shadow-reader__body">
          <aside className="window-shadow-index dream-index" aria-label="历次梦境">
            <div className="window-shadow-index__heading">
              <span>做过的梦</span>
              <small>{dreams.length} 场</small>
            </div>
            <ol>
              {dreams.map((dream, index) => {
                const active = dream.id === selectedDream.id;
                return (
                  <li key={dream.id}>
                    <button
                      type="button"
                      className={active ? "is-active" : ""}
                      aria-pressed={active}
                      onClick={() => selectDream(dream.id)}
                    >
                      <span>{String(dreams.length - index).padStart(2, "0")}</span>
                      <span>
                        <time dateTime={dream.generatedAt}>{dream.dateLabel}</time>
                        <strong>{dream.aiName} 做了一个梦</strong>
                        <em>{dream.statusLabel}</em>
                      </span>
                    </button>
                  </li>
                );
              })}
            </ol>
          </aside>

          <article ref={articleRef} className="window-shadow-reading dream-reading">
            <button className="window-shadow-reading__mobile-back" type="button" onClick={() => setMobileDetailOpen(false)}>
              <ArrowLeft size={17} weight="light" aria-hidden="true" />
              所有梦境
            </button>

            <header className="window-shadow-reading__header">
              <time dateTime={selectedDream.generatedAt}>
                {selectedDream.dateLabel}{selectedDream.timeLabel ? ` · ${selectedDream.timeLabel}` : ""}
              </time>
              <div className="window-shadow-reading__provenance">
                <span>{selectedDream.statusLabel}</span>
              </div>
              <h1>{selectedDream.aiName} 做了一个梦</h1>
            </header>

            {selectedDream.body ? (
              <MarkdownProjection content={selectedDream.body} className="window-shadow-reading__text" />
            ) : (
              <p className="dream-reading__state">{readingState}</p>
            )}
          </article>
        </div>
      </section>
    </div>
  );
}
