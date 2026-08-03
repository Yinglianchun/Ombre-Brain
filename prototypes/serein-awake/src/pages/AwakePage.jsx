import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { MoonStars, PencilSimple } from "@phosphor-icons/react";
import { Avatar } from "../components/Avatar.jsx";
import { SettingsPanel } from "../components/SettingsPanel.jsx";
import { Sidebar } from "../components/Sidebar.jsx";
import { DreamReader } from "../components/DreamReader.jsx";
import { WindowShadowReader } from "../components/WindowShadowReader.jsx";
import {
  compositionPresets,
  coverSettingStorageKeys,
  defaultCompositionItems,
  defaultCoverSettings,
  people,
} from "../data/awake.js";
import { defaultWindowShadows } from "../data/windowShadows.js";
import { useReplaceableImages } from "../hooks/useReplaceableImages.js";
import {
  readCompositionItems,
  readLocalPreference,
  storeCompositionItems,
  storeLocalPreference,
} from "../storage/awakeStore.js";
import {
  loadWindowShadows,
  readFallbackWindowShadows,
} from "../storage/windowShadowStore.js";
import { dreamExcerpt, loadDreams } from "../storage/dreamStore.js";

gsap.registerPlugin(ScrollTrigger);

export function AwakePage({
  activeArea,
  scopeRef,
  onNavigate,
  onUnavailable,
  settingsOpen,
  onSettingsOpenChange,
}) {
  const compositionRef = useRef(null);
  const shadowTriggerRef = useRef(null);
  const dreamTriggerRef = useRef(null);
  const activeDrag = useRef(null);
  const activeResize = useRef(null);
  const { images, replace } = useReplaceableImages();
  const [compositionEditing, setCompositionEditing] = useState(false);
  const [compositionElements, setCompositionElements] = useState(readCompositionItems);
  const [selectedCompositionId, setSelectedCompositionId] = useState(null);
  const [newCompositionKind, setNewCompositionKind] = useState("black-block");
  const [shadowReaderOpen, setShadowReaderOpen] = useState(false);
  const [dreamReaderOpen, setDreamReaderOpen] = useState(false);
  const [windowShadows, setWindowShadows] = useState(readFallbackWindowShadows);
  const [selectedShadowId, setSelectedShadowId] = useState(defaultWindowShadows[0]?.id ?? null);
  const [dreams, setDreams] = useState([]);
  const [dreamsStatus, setDreamsStatus] = useState("loading");
  const [selectedDreamId, setSelectedDreamId] = useState(null);
  const [coverSettings, setCoverSettings] = useState(() => ({
    togetherText: readLocalPreference("serein.awake.togetherText", defaultCoverSettings.togetherText),
    tagline: readLocalPreference("serein.awake.tagline", defaultCoverSettings.tagline),
    fadeStart: Math.min(82, Math.max(48, Number(
      readLocalPreference("serein.awake.fadeStart", defaultCoverSettings.fadeStart),
    ) || defaultCoverSettings.fadeStart)),
    portraitHazeEnabled:
      readLocalPreference("serein.awake.portraitHaze", defaultCoverSettings.portraitHazeEnabled ? "on" : "off") !== "off",
    compositionEnabled:
      readLocalPreference("serein.awake.composition", defaultCoverSettings.compositionEnabled ? "on" : "off") !== "off",
  }));
  const [identityNames, setIdentityNames] = useState(() => ({
    xiaoyu: readLocalPreference("serein.awake.name.xiaoyu", people[0].name),
    haven: readLocalPreference("serein.awake.name.haven", people[1].name),
  }));
  const resolvedPeople = people.map((person) => ({ ...person, name: identityNames[person.key] }));
  const latestShadow = windowShadows[0];
  const latestDream = dreams.find((dream) => dream.hasBody) ?? dreams[0] ?? null;

  useEffect(() => {
    let active = true;
    loadWindowShadows().then((snapshotShadows) => {
      if (!active || !snapshotShadows?.length) return;
      setWindowShadows(snapshotShadows);
      setSelectedShadowId(snapshotShadows[0].id);
    });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    let active = true;
    loadDreams()
      .then((records) => {
        if (!active) return;
        setDreams(records);
        setSelectedDreamId(records.find((dream) => dream.hasBody)?.id ?? records[0]?.id ?? null);
        setDreamsStatus("ready");
      })
      .catch(() => {
        if (active) setDreamsStatus("error");
      });
    return () => {
      active = false;
    };
  }, []);

  useLayoutEffect(() => {
    if (activeArea !== "醒来") return undefined;

    const mm = gsap.matchMedia();
    const context = gsap.context(() => {
      mm.add(
        {
          motion: "(prefers-reduced-motion: no-preference)",
          reduced: "(prefers-reduced-motion: reduce)",
        },
        ({ conditions }) => {
          if (conditions.reduced) {
            gsap.set(".cover", { autoAlpha: 0 });
            gsap.set([".white-veil", ".content-view", ".awake-stage .sidebar"], { autoAlpha: 1 });
            return;
          }

          const timeline = gsap.timeline({
            defaults: { ease: "none" },
            scrollTrigger: {
              trigger: ".awake-experience",
              start: "top top",
              end: "+=165%",
              scrub: 0.65,
              pin: ".awake-stage",
              anticipatePin: 1,
            },
          });

          timeline
            .to(".cover__composition", { autoAlpha: 0, duration: 0.18 }, 0)
            .to(".cover__portrait-haze", { autoAlpha: 0, duration: 0.34 }, 0.06)
            .to(".cover__image", { scale: 1.045, autoAlpha: 0.12, duration: 0.46 }, 0)
            .to(".cover__identity", { yPercent: -7, autoAlpha: 0, duration: 0.34 }, 0.14)
            .to(".white-veil", { autoAlpha: 1, duration: 0.48 }, 0.08)
            .fromTo(
              ".content-view",
              { autoAlpha: 0, y: 28 },
              { autoAlpha: 1, y: 0, duration: 0.42 },
              0.46,
            )
            .fromTo(
              ".content-view .reveal",
              { autoAlpha: 0, y: 18 },
              { autoAlpha: 1, y: 0, stagger: 0.035, duration: 0.28 },
              0.52,
            )
            .fromTo(
              ".awake-stage .sidebar",
              { autoAlpha: 0, x: -18 },
              { autoAlpha: 1, x: 0, duration: 0.25 },
              0.66,
            );
        },
      );
    }, scopeRef.current);

    document.fonts?.ready.then(() => ScrollTrigger.refresh());

    return () => {
      context.revert();
      mm.revert();
    };
  }, [activeArea, scopeRef]);

  const navigateFromAwake = (label) => {
    setShadowReaderOpen(false);
    setDreamReaderOpen(false);
    setCompositionEditing(false);
    setSelectedCompositionId(null);
    onSettingsOpenChange(false);
    onNavigate(label);
  };

  const closeShadowReader = useCallback(() => {
    setShadowReaderOpen(false);
    window.setTimeout(() => shadowTriggerRef.current?.focus(), 0);
  }, []);

  const closeDreamReader = useCallback(() => {
    setDreamReaderOpen(false);
    window.setTimeout(() => dreamTriggerRef.current?.focus(), 0);
  }, []);

  const storeLoadedDream = useCallback((loadedDream) => {
    setDreams((current) => current.map((dream) => dream.id === loadedDream.id ? loadedDream : dream));
  }, []);

  const beginCompositionDrag = (event, id) => {
    if (!compositionEditing || !compositionRef.current) return;
    event.preventDefault();
    setSelectedCompositionId(id);
    event.currentTarget.setPointerCapture(event.pointerId);
    const item = compositionElements.find((element) => element.id === id);
    activeDrag.current = { id, x: item.x, y: item.y };
  };

  const moveCompositionItem = (event, id) => {
    if (!compositionEditing || activeDrag.current?.id !== id || !compositionRef.current) return;
    const rect = compositionRef.current.getBoundingClientRect();
    const x = Math.min(98, Math.max(2, ((event.clientX - rect.left) / rect.width) * 100));
    const y = Math.min(96, Math.max(4, ((event.clientY - rect.top) / rect.height) * 100));
    activeDrag.current = { id, x, y };
    event.currentTarget.style.left = `${x}%`;
    event.currentTarget.style.top = `${y}%`;
  };

  const finishCompositionDrag = (event, id) => {
    if (activeDrag.current?.id !== id) return;
    const { x, y } = activeDrag.current;
    activeDrag.current = null;
    setCompositionElements((current) => {
      const next = current.map((item) => item.id === id ? { ...item, x, y } : item);
      storeCompositionItems(next);
      return next;
    });
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  const beginCompositionResize = (event, item) => {
    if (!compositionEditing) return;
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    activeResize.current = {
      id: item.id,
      startX: event.clientX,
      startY: event.clientY,
      width: item.width,
      height: item.height,
      scale: compositionRef.current ? compositionRef.current.clientWidth / 1100 : 1,
    };
  };

  const resizeCompositionItem = (event, id) => {
    if (!compositionEditing || activeResize.current?.id !== id) return;
    event.preventDefault();
    event.stopPropagation();
    const scale = Math.max(0.1, activeResize.current.scale);
    const width = Math.min(360, Math.max(8, activeResize.current.width + (event.clientX - activeResize.current.startX) / scale));
    const height = Math.min(240, Math.max(6, activeResize.current.height + (event.clientY - activeResize.current.startY) / scale));
    activeResize.current = { ...activeResize.current, nextWidth: width, nextHeight: height };
    const shape = event.currentTarget.parentElement;
    shape.style.width = `${(width / 1100) * 100}%`;
    shape.style.height = "auto";
    shape.style.aspectRatio = `${width} / ${height}`;
  };

  const finishCompositionResize = (event, id) => {
    if (activeResize.current?.id !== id) return;
    event.preventDefault();
    event.stopPropagation();
    const width = activeResize.current.nextWidth ?? activeResize.current.width;
    const height = activeResize.current.nextHeight ?? activeResize.current.height;
    activeResize.current = null;
    setCompositionElements((current) => {
      const next = current.map((item) => item.id === id ? { ...item, width, height } : item);
      storeCompositionItems(next);
      return next;
    });
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  const nudgeCompositionItem = (event, id) => {
    if (compositionEditing && ["Delete", "Backspace"].includes(event.key)) {
      event.preventDefault();
      deleteCompositionItem(id);
      return;
    }
    if (!compositionEditing || !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) return;
    event.preventDefault();

    if (event.altKey) {
      const delta = event.shiftKey ? 10 : 2;
      setCompositionElements((items) => {
        const next = items.map((item) => {
          if (item.id !== id) return item;
          const horizontal = event.key === "ArrowLeft" ? -delta : event.key === "ArrowRight" ? delta : 0;
          const vertical = event.key === "ArrowUp" ? -delta : event.key === "ArrowDown" ? delta : 0;
          return {
            ...item,
            width: Math.min(360, Math.max(8, item.width + horizontal)),
            height: Math.min(240, Math.max(6, item.height + vertical)),
          };
        });
        storeCompositionItems(next);
        return next;
      });
      return;
    }

    const delta = event.shiftKey ? 1 : 0.25;
    setCompositionElements((items) => {
      const next = items.map((item) => item.id === id ? {
        ...item,
        x: Math.min(98, Math.max(2, item.x + (event.key === "ArrowRight" ? delta : event.key === "ArrowLeft" ? -delta : 0))),
        y: Math.min(96, Math.max(4, item.y + (event.key === "ArrowDown" ? delta : event.key === "ArrowUp" ? -delta : 0))),
      } : item);
      storeCompositionItems(next);
      return next;
    });
  };

  const resetComposition = () => {
    setCompositionElements(defaultCompositionItems);
    setSelectedCompositionId(null);
    storeCompositionItems(defaultCompositionItems);
  };

  const addCompositionItem = () => {
    const preset = compositionPresets[newCompositionKind];
    const id = `custom-${Date.now()}`;
    const item = {
      id,
      label: `新增${preset.label}`,
      kind: newCompositionKind,
      layer: "front",
      x: 50,
      y: 50,
      width: preset.width,
      height: preset.height,
    };
    setCompositionElements((current) => {
      const next = [...current, item];
      storeCompositionItems(next);
      return next;
    });
    setSelectedCompositionId(id);
  };

  const deleteCompositionItem = (id = selectedCompositionId) => {
    if (!id) return;
    setCompositionElements((current) => {
      const next = current.filter((item) => item.id !== id);
      storeCompositionItems(next);
      return next;
    });
    setSelectedCompositionId((current) => current === id ? null : current);
  };

  const setCompositionLayer = (layer) => {
    if (!selectedCompositionId) return;
    setCompositionElements((current) => {
      const next = current.map((item) => item.id === selectedCompositionId ? { ...item, layer } : item);
      storeCompositionItems(next);
      return next;
    });
  };

  const updateCoverSetting = (key, value) => {
    setCoverSettings((current) => ({ ...current, [key]: value }));
    const storesAsToggle = key === "compositionEnabled" || key === "portraitHazeEnabled";
    storeLocalPreference(
      coverSettingStorageKeys[key],
      storesAsToggle ? (value ? "on" : "off") : value,
    );
  };

  const updateIdentityName = (key, value) => {
    setIdentityNames((current) => ({ ...current, [key]: value }));
    storeLocalPreference(`serein.awake.name.${key}`, value);
  };

  const startCompositionEditing = () => {
    if (!coverSettings.compositionEnabled) updateCoverSetting("compositionEnabled", true);
    onSettingsOpenChange(false);
    setSelectedCompositionId(null);
    setCompositionEditing(true);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <>
      <section
        className="awake-experience"
        aria-label="醒来"
        aria-hidden={shadowReaderOpen || dreamReaderOpen ? "true" : undefined}
        hidden={activeArea !== "醒来"}
      >
        <div className="awake-stage">
          <div
            className={`cover${coverSettings.compositionEnabled ? "" : " composition-is-off"}${compositionEditing ? " composition-is-editing" : ""}`}
          >
            <img
              className="cover__image"
              src={images.hero}
              alt="雨后窗外的树影"
              style={{ "--cover-fade-start": `${coverSettings.fadeStart}%` }}
            />
            <div
              className={`cover__portrait-haze-shell${coverSettings.portraitHazeEnabled ? "" : " is-off"}`}
              aria-hidden="true"
            >
              <span className="cover__portrait-haze" />
            </div>
            <div
              ref={compositionRef}
              className="cover__composition"
              aria-label={compositionEditing ? "黑白构成层调整画板" : undefined}
              aria-hidden={compositionEditing ? undefined : "true"}
            >
              {compositionElements.map((item) => (
                <span
                  className={`composition-shape composition-shape--${item.kind} is-${item.layer}${selectedCompositionId === item.id ? " is-selected" : ""}`}
                  key={item.id}
                  role={compositionEditing ? "button" : undefined}
                  tabIndex={compositionEditing ? 0 : -1}
                  aria-pressed={compositionEditing ? selectedCompositionId === item.id : undefined}
                  aria-label={compositionEditing ? `移动${item.label}` : undefined}
                  style={{
                    left: `${item.x}%`,
                    top: `${item.y}%`,
                    width: `${(item.width / 1100) * 100}%`,
                    height: "auto",
                    aspectRatio: `${item.width} / ${item.height}`,
                  }}
                  onPointerDown={(event) => beginCompositionDrag(event, item.id)}
                  onPointerMove={(event) => moveCompositionItem(event, item.id)}
                  onPointerUp={(event) => finishCompositionDrag(event, item.id)}
                  onPointerCancel={(event) => finishCompositionDrag(event, item.id)}
                  onKeyDown={(event) => nudgeCompositionItem(event, item.id)}
                >
                  {compositionEditing && selectedCompositionId === item.id ? (
                    <i
                      className="composition-resize-handle"
                      aria-hidden="true"
                      onPointerDown={(event) => beginCompositionResize(event, item)}
                      onPointerMove={(event) => resizeCompositionItem(event, item.id)}
                      onPointerUp={(event) => finishCompositionResize(event, item.id)}
                      onPointerCancel={(event) => finishCompositionResize(event, item.id)}
                    />
                  ) : null}
                </span>
              ))}
            </div>
            {compositionEditing ? (
              <div className="composition-editor" aria-label="构成层工具">
                <label>
                  <span>新增元素</span>
                  <select value={newCompositionKind} onChange={(event) => setNewCompositionKind(event.target.value)}>
                    {Object.entries(compositionPresets).map(([value, preset]) => (
                      <option value={value} key={value}>{preset.label}</option>
                    ))}
                  </select>
                </label>
                <button type="button" onClick={addCompositionItem}>增加</button>
                <button type="button" disabled={!selectedCompositionId} onClick={() => deleteCompositionItem()}>删除</button>
                <button type="button" disabled={!selectedCompositionId} onClick={() => setCompositionLayer("behind")}>头像下方</button>
                <button type="button" disabled={!selectedCompositionId} onClick={() => setCompositionLayer("front")}>置顶</button>
                <button type="button" onClick={resetComposition}>重置</button>
                <button
                  type="button"
                  onClick={() => {
                    setCompositionEditing(false);
                    setSelectedCompositionId(null);
                  }}
                >
                  <PencilSimple size={14} weight="light" aria-hidden="true" />
                  完成
                </button>
              </div>
            ) : null}
            <div className="cover__identity" aria-label="小雨和 Haven">
              <div className="cover-person">
                <Avatar person={resolvedPeople[0]} src={images.xiaoyu} onReplace={(file) => replace("xiaoyu", file)} />
                <h1>{resolvedPeople[0].name}</h1>
              </div>

              <div className="together-mark">
                <span className="together-mark__line" aria-hidden="true" />
                <MoonStars size={24} weight="light" aria-hidden="true" />
                <span className="together-mark__line" aria-hidden="true" />
                <strong>{coverSettings.togetherText}</strong>
              </div>

              <div className="cover-person">
                <Avatar person={resolvedPeople[1]} src={images.haven} onReplace={(file) => replace("haven", file)} />
                <h1>{resolvedPeople[1].name}</h1>
              </div>

              <div className="cover__tagline">
                <p>{coverSettings.tagline}</p>
              </div>
            </div>
          </div>

          <div className="white-veil" aria-hidden="true" />

          <div className="content-view">
            <header className="content-header reveal">
              <div>
                <span className="content-header__date">2026.07.23</span>
                <h2>醒来</h2>
              </div>
              <p>先认出彼此，再看看有什么还留在心里。</p>
            </header>

            <section className="portrait-section" aria-labelledby="portrait-title">
              <div className="section-heading reveal">
                <h3 id="portrait-title">画像</h3>
                <span>上次更新 2026.07.21</span>
              </div>
              <div className="portrait-pair">
                {resolvedPeople.map((person) => (
                  <article className="portrait reveal" key={person.key}>
                    <Avatar
                      person={person}
                      src={images[person.key]}
                      size="small"
                      editable
                      onReplace={(file) => replace(person.key, file)}
                    />
                    <div className="portrait__copy">
                      <div className="portrait__name-row">
                        <h4>{person.name}</h4>
                        <button type="button" aria-label={`编辑${person.name}的画像`} onClick={() => onUnavailable("画像编辑")}>
                          <PencilSimple size={15} weight="light" aria-hidden="true" />
                          编辑
                        </button>
                      </div>
                      <p>{person.detail}</p>
                    </div>
                  </article>
                ))}
              </div>
            </section>

            <div className="lower-grid">
              <section className="window-shadow reveal" aria-labelledby="shadow-title">
                <div className="section-heading">
                  <h3 id="shadow-title">上一窗影</h3>
                  <span>{latestShadow.relativeLabel}</span>
                </div>
                <p>{latestShadow.summary}</p>
                <button
                  ref={shadowTriggerRef}
                  type="button"
                  onClick={() => {
                    setSelectedShadowId(latestShadow.id);
                    setShadowReaderOpen(true);
                  }}
                >
                  读那一窗
                </button>
              </section>

              <section className="dream-preview reveal" aria-labelledby="dream-preview-title">
                <div className="section-heading">
                  <h3 id="dream-preview-title">梦境</h3>
                  <span>{latestDream ? latestDream.dateLabel : ""}</span>
                </div>
                <p>{dreamsStatus === "loading"
                  ? "梦还在雾里。"
                  : dreamsStatus === "error"
                    ? "梦境暂时没有接通。"
                    : dreamExcerpt(latestDream)}</p>
                {latestDream ? (
                  <button
                    ref={dreamTriggerRef}
                    type="button"
                    onClick={() => {
                      setSelectedDreamId(latestDream.id);
                      setDreamReaderOpen(true);
                    }}
                  >
                    翻开梦境
                  </button>
                ) : null}
              </section>
            </div>
          </div>

          <Sidebar
            activeArea={activeArea}
            onNavigate={navigateFromAwake}
            onUnavailable={onUnavailable}
            onOpenSettings={() => {
              setCompositionEditing(false);
              setSelectedCompositionId(null);
              onSettingsOpenChange(true);
            }}
          />
        </div>
      </section>

      <SettingsPanel
        open={settingsOpen}
        onClose={() => onSettingsOpenChange(false)}
        coverSettings={coverSettings}
        onCoverSetting={updateCoverSetting}
        identityNames={identityNames}
        onNameChange={updateIdentityName}
        images={images}
        onReplace={replace}
        onEditComposition={() => {
          if (activeArea === "醒来") {
            startCompositionEditing();
            return;
          }
          navigateFromAwake("醒来");
          window.setTimeout(startCompositionEditing, 0);
        }}
        onResetComposition={resetComposition}
      />

      <WindowShadowReader
        open={shadowReaderOpen}
        shadows={windowShadows}
        selectedShadowId={selectedShadowId}
        onSelect={setSelectedShadowId}
        onClose={closeShadowReader}
      />

      <DreamReader
        open={dreamReaderOpen}
        dreams={dreams}
        selectedDreamId={selectedDreamId}
        onSelect={setSelectedDreamId}
        onDreamLoaded={storeLoadedDream}
        onClose={closeDreamReader}
      />
    </>
  );
}
