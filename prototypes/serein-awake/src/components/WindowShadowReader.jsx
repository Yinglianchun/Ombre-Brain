import { useEffect, useRef, useState } from "react";
import { ArrowLeft, X } from "@phosphor-icons/react";
import { MarkdownProjection } from "./MarkdownProjection.jsx";

export function WindowShadowReader({
  open,
  shadows,
  selectedShadowId,
  onSelect,
  onClose,
}) {
  const closeButtonRef = useRef(null);
  const articleRef = useRef(null);
  const [mobileDetailOpen, setMobileDetailOpen] = useState(false);
  const selectedShadow = shadows.find((shadow) => shadow.id === selectedShadowId) ?? shadows[0];

  useEffect(() => {
    if (!open) {
      setMobileDetailOpen(false);
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
  }, [selectedShadow?.id]);

  if (!open || !selectedShadow) return null;

  const selectShadow = (shadowId) => {
    onSelect(shadowId);
    setMobileDetailOpen(true);
  };

  return (
    <div className="window-shadow-layer" role="dialog" aria-modal="true" aria-labelledby="window-shadow-reader-title">
      <div
        className="window-shadow-layer__veil"
        aria-hidden="true"
        onClick={onClose}
      />

      <section className={`window-shadow-reader${mobileDetailOpen ? " is-mobile-detail-open" : ""}`}>
        <header className="window-shadow-reader__topbar">
          <div>
            <span>WINDOW SHADOWS</span>
            <h2 id="window-shadow-reader-title">窗影</h2>
          </div>
          <button ref={closeButtonRef} type="button" aria-label="关闭窗影" onClick={onClose}>
            <X size={22} weight="light" aria-hidden="true" />
          </button>
        </header>

        <div className="window-shadow-reader__body">
          <aside className="window-shadow-index" aria-label="历次窗影">
            <div className="window-shadow-index__heading">
              <span>历次窗影</span>
              <small>{shadows.length} 窗</small>
            </div>
            <ol>
              {shadows.map((shadow, index) => {
                const active = shadow.id === selectedShadow.id;
                return (
                  <li key={shadow.id}>
                    <button
                      type="button"
                      className={active ? "is-active" : ""}
                      aria-pressed={active}
                      onClick={() => selectShadow(shadow.id)}
                    >
                      <span>{String(shadows.length - index).padStart(2, "0")}</span>
                      <span>
                        <time dateTime={shadow.closedAt}>{shadow.dateLabel}</time>
                        <strong>{shadow.title}</strong>
                        <small>{shadow.summary}</small>
                        {shadow.statusLabel ? <em>{shadow.statusLabel}</em> : null}
                      </span>
                    </button>
                  </li>
                );
              })}
            </ol>
          </aside>

          <article ref={articleRef} className="window-shadow-reading">
            <button
              className="window-shadow-reading__mobile-back"
              type="button"
              onClick={() => setMobileDetailOpen(false)}
            >
              <ArrowLeft size={17} weight="light" aria-hidden="true" />
              所有窗影
            </button>

            <header className={`window-shadow-reading__header${selectedShadow.documentOwnsTitle ? " is-document-led" : ""}`}>
              <time dateTime={selectedShadow.closedAt}>
                {selectedShadow.dateLabel} · {selectedShadow.timeLabel}
              </time>
              {selectedShadow.sourceLabel || selectedShadow.statusLabel ? (
                <div className="window-shadow-reading__provenance">
                  {selectedShadow.sourceLabel ? <span>{selectedShadow.sourceLabel}</span> : null}
                  {selectedShadow.statusLabel ? <span>{selectedShadow.statusLabel}</span> : null}
                </div>
              ) : null}
              {!selectedShadow.documentOwnsTitle ? (
                <>
                  <h1>{selectedShadow.title}</h1>
                  <p>{selectedShadow.summary}</p>
                </>
              ) : null}
            </header>

            <MarkdownProjection
              content={selectedShadow.text}
              className="window-shadow-reading__text"
            />

            {selectedShadow.scenes?.length ? (
              <footer className="window-shadow-reading__scenes">
                <span>这一窗留下的 Scene</span>
                <ul>
                  {selectedShadow.scenes.map((scene) => (
                    <li key={scene.id}>{scene.title}</li>
                  ))}
                </ul>
              </footer>
            ) : null}
          </article>
        </div>
      </section>
    </div>
  );
}
