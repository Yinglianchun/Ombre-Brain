import { useEffect, useMemo, useState } from "react";
import { flushSync } from "react-dom";
import {
  ArrowLeft,
  ArrowRight,
  CaretDown,
  CircleNotch,
} from "@phosphor-icons/react";
import { loadNarrativeRolls, readFallbackNarrativeRolls } from "../storage/narrativeStore.js";

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

export function NarrativePage() {
  const [narrativeRolls, setNarrativeRolls] = useState(readFallbackNarrativeRolls);
  const [selectedRollId, setSelectedRollId] = useState(null);
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
        transitionTo(() => setSelectedRollId(null));
      }
    };
    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [selectedRoll]);

  const openRoll = (rollId) => {
    transitionTo(() => setSelectedRollId(rollId));
  };

  const closeRoll = () => {
    transitionTo(() => setSelectedRollId(null));
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
            <div className="narrative-shelf" role="list" aria-label="叙事卷书架">
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
                    <span className="narrative-book__title">{roll.spineTitle}</span>
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
            <p className="narrative-shelf__hint">悬停翻开一点 · 点击阅读</p>
          </div>
        </section>
      ) : (
        <article className="narrative-reader" aria-labelledby="narrative-reader-title">
          <nav className="narrative-reader__topbar" aria-label="叙事卷阅读导航">
            <button className="narrative-reader__back" type="button" onClick={closeRoll}>
              <ArrowLeft size={18} weight="light" aria-hidden="true" />
              回到书架
            </button>
            <span>
              VOL. {selectedRoll.volume}
              <i aria-hidden="true">·</i>
              {selectedRoll.status}
            </span>
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
                  <dd>{selectedRoll.sceneCount} Scene</dd>
                </div>
              </dl>
            </header>

            <section className="narrative-manuscript__body" aria-label="叙事正文">
              {selectedRoll.paragraphs.map((paragraph) => <p key={paragraph}>{paragraph}</p>)}
            </section>

            {selectedRoll.sources ? (
              <details className="narrative-ledger">
                <summary>
                  <span>
                    <strong>来源账</strong>
                    <small>{selectedRoll.sources.length} 条 Scene 原文</small>
                  </span>
                  <CaretDown size={17} weight="light" aria-hidden="true" />
                </summary>
                <ol>
                  {selectedRoll.sources.map((source) => (
                    <li key={source.id}>
                      <time>{source.date}</time>
                      <span>
                        <strong>{source.title}</strong>
                        <small>{source.purpose}</small>
                      </span>
                    </li>
                  ))}
                </ol>
              </details>
            ) : (
              <section className="narrative-scenes" aria-labelledby="narrative-scenes-title">
                <div>
                  <h2 id="narrative-scenes-title">卷中的 Scene</h2>
                  <span>{selectedRoll.sceneCount} 条来源</span>
                </div>
                <ul>
                  {selectedRoll.sceneNames.map((scene) => <li key={scene}>{scene}</li>)}
                </ul>
              </section>
            )}
          </section>
        </article>
      )}
    </div>
  );
}
