import { useEffect, useRef, useState } from "react";
import { Sidebar } from "./components/Sidebar.jsx";
import { AwakePage } from "./pages/AwakePage.jsx";
import { MemoryPage } from "./pages/MemoryPage.jsx";
import { NarrativePage } from "./pages/NarrativePage.jsx";
import { DiaryPage } from "./pages/DiaryPage.jsx";
import { BasementPage } from "./pages/BasementPage.jsx";
import { UniversePage } from "./pages/UniversePage.jsx";

const availableAreas = new Set(["醒来", "记忆", "叙事卷", "日记", "地下室", "宇宙"]);
const areaHashes = {
  醒来: "",
  记忆: "#memory",
  叙事卷: "#narrative",
  日记: "#diary",
  地下室: "#basement",
  宇宙: "#universe",
};

const readAreaFromHash = () => {
  if (window.location.hash === "#memory") return "记忆";
  if (window.location.hash === "#narrative") return "叙事卷";
  if (window.location.hash === "#diary") return "日记";
  if (window.location.hash === "#basement") return "地下室";
  if (window.location.hash === "#universe") return "宇宙";
  return "醒来";
};

export function App() {
  const root = useRef(null);
  const [activeArea, setActiveArea] = useState(readAreaFromHash);
  const [notice, setNotice] = useState("");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [universeNavEntering, setUniverseNavEntering] = useState(false);

  useEffect(() => {
    const syncAreaFromHash = () => setActiveArea(readAreaFromHash());
    window.addEventListener("hashchange", syncAreaFromHash);
    return () => window.removeEventListener("hashchange", syncAreaFromHash);
  }, []);

  useEffect(() => {
    if (activeArea !== "宇宙") {
      setUniverseNavEntering(false);
      return undefined;
    }
    setUniverseNavEntering(true);
    const timer = window.setTimeout(() => setUniverseNavEntering(false), 1450);
    return () => window.clearTimeout(timer);
  }, [activeArea]);

  const showUnavailable = (label) => {
    setNotice(`${label}还没有展开。我们先把这一页做好。`);
    window.setTimeout(() => setNotice(""), 2400);
  };

  const navigateTo = (label) => {
    if (!availableAreas.has(label)) {
      showUnavailable(label);
      return;
    }

    setSettingsOpen(false);
    setActiveArea(label);
    window.history.replaceState(
      null,
      "",
      `${window.location.pathname}${window.location.search}${areaHashes[label]}`,
    );
    window.scrollTo({ top: 0, behavior: "auto" });
  };

  return (
    <main
      ref={root}
      className={`app-shell app-shell--${
        activeArea === "醒来"
          ? "awake"
          : activeArea === "记忆"
            ? "memory"
            : activeArea === "叙事卷"
              ? "narrative"
              : activeArea === "日记"
                ? "diary"
                : activeArea === "地下室"
                  ? "basement"
                  : "universe"
      }`}
    >
      <AwakePage
        activeArea={activeArea}
        scopeRef={root}
        onNavigate={navigateTo}
        onUnavailable={showUnavailable}
        settingsOpen={settingsOpen}
        onSettingsOpenChange={setSettingsOpen}
      />

      <section className="memory-page" aria-label="记忆" hidden={activeArea !== "记忆"}>
        {activeArea === "记忆" ? <MemoryPage /> : null}
        <Sidebar
          activeArea={activeArea}
          onNavigate={navigateTo}
          onUnavailable={showUnavailable}
          onOpenSettings={() => setSettingsOpen(true)}
        />
      </section>

      <section className="narrative-page" aria-label="叙事卷" hidden={activeArea !== "叙事卷"}>
        <NarrativePage />
        <Sidebar
          activeArea={activeArea}
          onNavigate={navigateTo}
          onUnavailable={showUnavailable}
          onOpenSettings={() => setSettingsOpen(true)}
        />
      </section>

      <section className="diary-page" aria-label="日记" hidden={activeArea !== "日记"}>
        <DiaryPage />
        <Sidebar
          activeArea={activeArea}
          onNavigate={navigateTo}
          onUnavailable={showUnavailable}
          onOpenSettings={() => setSettingsOpen(true)}
        />
      </section>

      <section className="basement-page" aria-label="地下室" hidden={activeArea !== "地下室"}>
        <BasementPage />
        <Sidebar
          activeArea={activeArea}
          onNavigate={navigateTo}
          onUnavailable={showUnavailable}
          onOpenSettings={() => setSettingsOpen(true)}
        />
      </section>

      <section
        className={`universe-page${universeNavEntering ? " is-entering" : ""}`}
        aria-label="宇宙"
        hidden={activeArea !== "宇宙"}
      >
        <UniversePage />
        <button className="universe-nav-sensor" type="button" aria-label="显示导航" />
        <Sidebar
          activeArea={activeArea}
          onNavigate={navigateTo}
          onUnavailable={showUnavailable}
          onOpenSettings={() => setSettingsOpen(true)}
        />
      </section>

      <div className={`notice${notice ? " is-visible" : ""}`} role="status" aria-live="polite">{notice}</div>
    </main>
  );
}
