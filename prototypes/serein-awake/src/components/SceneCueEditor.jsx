import { useEffect, useMemo, useState } from "react";
import {
  ArrowClockwise,
  Check,
  LinkSimple,
  PencilSimple,
  Plus,
  Trash,
  X,
} from "@phosphor-icons/react";

function ombreSceneId(scene) {
  const source = scene?.sources?.find((item) => String(item?.id || "").startsWith("manual_source:"));
  return source ? source.id.slice("manual_source:".length) : "";
}

export function SceneCueEditor({ scene, onSaved }) {
  const sourceSceneId = useMemo(() => ombreSceneId(scene), [scene]);
  const [loadState, setLoadState] = useState("idle");
  const [saveState, setSaveState] = useState("idle");
  const [liveScene, setLiveScene] = useState(null);
  const [draft, setDraft] = useState([]);
  const [editing, setEditing] = useState(false);
  const [message, setMessage] = useState("");

  const storedCues = liveScene?.metadata?.scene_cues ?? [];
  const expectedUpdatedAt = liveScene?.metadata?.updated_at || "";
  const cleanDraft = draft.map((cue) => cue.trim()).filter(Boolean);
  const dirty = JSON.stringify(cleanDraft) !== JSON.stringify(storedCues);
  const invalid = !cleanDraft.length || cleanDraft.length > 8 || cleanDraft.some((cue) => cue.length > 80);

  const readScene = async () => {
    if (!sourceSceneId) return;
    setLoadState("loading");
    setMessage("");
    try {
      const response = await fetch("/__serein/memory/read-scene", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sceneId: sourceSceneId }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload?.message || payload?.error || "没有读到 Scene");
      const cues = Array.isArray(payload?.metadata?.scene_cues) ? payload.metadata.scene_cues : [];
      setLiveScene(payload);
      setDraft(cues);
      setLoadState("done");
      setSaveState("idle");
      setMessage("");
    } catch (error) {
      setLoadState("error");
      setMessage(error instanceof Error ? error.message : "没有读到 Scene");
    }
  };

  useEffect(() => {
    setLoadState("idle");
    setSaveState("idle");
    setLiveScene(null);
    setDraft([]);
    setEditing(false);
    setMessage("");
    if (sourceSceneId) readScene();
    // readScene intentionally follows the selected source Scene only.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourceSceneId]);

  const updateCue = (index, value) => {
    setDraft((current) => current.map((cue, cueIndex) => cueIndex === index ? value : cue));
    setSaveState("idle");
    setMessage("");
  };

  const removeCue = (index) => {
    setDraft((current) => current.filter((_, cueIndex) => cueIndex !== index));
    setSaveState("idle");
    setMessage("");
  };

  const cancelEditing = () => {
    setDraft(storedCues);
    setEditing(false);
    setSaveState("idle");
    setMessage("");
  };

  const saveCues = async () => {
    if (!dirty || invalid || saveState === "saving") return;
    setSaveState("saving");
    setMessage("");
    try {
      const response = await fetch("/__serein/memory/edit-scene-cues", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sceneId: sourceSceneId,
          expectedUpdatedAt,
          cues: cleanDraft,
        }),
      });
      const payload = await response.json();
      if (response.status === 409 || payload?.status === "conflict") {
        setSaveState("conflict");
        setMessage("这张 Scene 在打开后又被修改了。重新读取后再改，不覆盖新版本。");
        return;
      }
      if (!response.ok || !["updated", "unchanged"].includes(payload?.status)) {
        throw new Error(payload?.message || payload?.reason || payload?.error || "没有保存成功");
      }
      const nextScene = payload.scene || {
        ...liveScene,
        metadata: {
          ...liveScene.metadata,
          scene_cues: cleanDraft,
          updated_at: payload.updated_at || expectedUpdatedAt,
        },
      };
      const nextCues = nextScene?.metadata?.scene_cues ?? cleanDraft;
      setLiveScene(nextScene);
      setDraft(nextCues);
      setEditing(false);
      setSaveState("saved");
      setMessage(payload.status === "unchanged" ? "没有变化。" : "已保存，并保留修改前的版本。");
      onSaved?.(nextCues);
    } catch (error) {
      setSaveState("error");
      setMessage(error instanceof Error ? error.message : "没有保存成功");
    }
  };

  if (!sourceSceneId) return null;

  return (
    <section className="scene-cues" aria-labelledby={`scene-cues-${scene.id}`}>
      <header className="scene-cues__header">
        <div>
          <LinkSimple size={19} weight="light" aria-hidden="true" />
          <div>
            <h3 id={`scene-cues-${scene.id}`}>召回入口</h3>
            <span>authored cues</span>
          </div>
        </div>
        {loadState === "done" && !editing ? (
          <button type="button" onClick={() => { setEditing(true); setMessage(""); }}>
            <PencilSimple size={14} weight="light" aria-hidden="true" />
            修改
          </button>
        ) : null}
      </header>

      {loadState === "loading" ? (
        <p className="scene-cues__state"><ArrowClockwise className="is-spinning" size={15} aria-hidden="true" />正在读取真实 Scene……</p>
      ) : null}

      {loadState === "error" ? (
        <div className="scene-cues__state is-error">
          <span>{message}</span>
          <button type="button" onClick={readScene}>重试</button>
        </div>
      ) : null}

      {loadState === "done" && !editing ? (
        <>
          {storedCues.length ? (
            <ol className="scene-cues__list">
              {storedCues.map((cue) => <li key={cue}>{cue}</li>)}
            </ol>
          ) : (
            <p className="scene-cues__empty">还没有人工写下召回入口。</p>
          )}
          <p className="scene-cues__boundary">只认这里明确保存的 cue；不会拿 tags、词图或正文自动补齐。</p>
          {message ? <strong className="scene-cues__message">{message}</strong> : null}
        </>
      ) : null}

      {loadState === "done" && editing ? (
        <div className="scene-cues__editor">
          <p>以后提到什么，希望这段记忆回来？一行一个，最多 8 条。</p>
          <div className="scene-cues__rows">
            {draft.map((cue, index) => (
              <label key={`${sourceSceneId}-cue-${index}`}>
                <span>{index + 1}</span>
                <input
                  value={cue}
                  maxLength={80}
                  aria-label={`召回入口 ${index + 1}`}
                  onChange={(event) => updateCue(index, event.target.value)}
                />
                <button type="button" aria-label={`删除召回入口 ${index + 1}`} onClick={() => removeCue(index)}>
                  <Trash size={14} weight="light" aria-hidden="true" />
                </button>
              </label>
            ))}
          </div>
          {draft.length < 8 ? (
            <button type="button" className="scene-cues__add" onClick={() => setDraft((current) => [...current, ""])}>
              <Plus size={14} weight="light" aria-hidden="true" />
              增加一条
            </button>
          ) : null}
          <footer>
            <div>
              <span>保存使用当前 updated_at 做版本校验；冲突时不会覆盖。</span>
              {invalid ? <strong>至少保留 1 条非空 cue。</strong> : null}
              {message ? <strong className={`is-${saveState}`}>{message}</strong> : null}
            </div>
            <div>
              {saveState === "conflict" ? (
                <button type="button" onClick={readScene}><ArrowClockwise size={14} aria-hidden="true" />重新读取</button>
              ) : null}
              <button type="button" onClick={cancelEditing}><X size={14} aria-hidden="true" />取消</button>
              <button className="is-primary" type="button" onClick={saveCues} disabled={!dirty || invalid || saveState === "saving"}>
                <Check size={14} aria-hidden="true" />
                {saveState === "saving" ? "正在保存" : "保存"}
              </button>
            </div>
          </footer>
        </div>
      ) : null}
    </section>
  );
}
