import { ImageSquare, X } from "@phosphor-icons/react";
import { people } from "../data/awake.js";

export function SettingsPanel({
  open,
  onClose,
  coverSettings,
  onCoverSetting,
  identityNames,
  onNameChange,
  images,
  onReplace,
  onEditComposition,
  onResetComposition,
}) {
  return (
    <>
      <button
        className={`settings-scrim${open ? " is-open" : ""}`}
        type="button"
        aria-label="关闭设置"
        tabIndex={open ? 0 : -1}
        onClick={onClose}
      />
      <section
        className={`settings-panel${open ? " is-open" : ""}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-title"
        aria-hidden={!open}
        inert={!open}
      >
        <header className="settings-panel__header">
          <div>
            <span>醒来</span>
            <h2 id="settings-title">设置</h2>
          </div>
          <button type="button" aria-label="关闭设置" onClick={onClose}>
            <X size={19} weight="light" aria-hidden="true" />
          </button>
        </header>

        <div className="settings-panel__body">
          <section className="settings-group" aria-labelledby="settings-cover-title">
            <div className="settings-group__heading">
              <h3 id="settings-cover-title">封面</h3>
              <p>照片、名字与留在下面的那句话。</p>
            </div>

            <label className="settings-upload settings-upload--cover">
              <img src={images.hero} alt="当前封面" />
              <span><ImageSquare size={17} weight="light" aria-hidden="true" />更换封面</span>
              <input type="file" accept="image/*" onChange={(event) => onReplace("hero", event.target.files?.[0])} />
            </label>

            <div className="settings-people">
              {people.map((person) => (
                <div className="settings-person" key={person.key}>
                  <label className="settings-upload settings-upload--avatar">
                    <img src={images[person.key]} alt={`${identityNames[person.key]}的头像`} />
                    <span>更换头像</span>
                    <input type="file" accept="image/*" onChange={(event) => onReplace(person.key, event.target.files?.[0])} />
                  </label>
                  <label className="settings-field">
                    <span>{person.key === "xiaoyu" ? "你的名字" : "他的名字"}</span>
                    <input
                      type="text"
                      value={identityNames[person.key]}
                      maxLength={18}
                      onChange={(event) => onNameChange(person.key, event.target.value)}
                    />
                  </label>
                </div>
              ))}
            </div>

            <label className="settings-field">
              <span>纪念日文字</span>
              <input
                type="text"
                value={coverSettings.togetherText}
                maxLength={30}
                onChange={(event) => onCoverSetting("togetherText", event.target.value)}
              />
            </label>

            <label className="settings-range">
              <span>
                <strong>白色过渡位置</strong>
                <output>{coverSettings.fadeStart}%</output>
              </span>
              <input
                type="range"
                min="48"
                max="82"
                step="1"
                value={coverSettings.fadeStart}
                onInput={(event) => onCoverSetting("fadeStart", Number(event.currentTarget.value))}
              />
              <small>越往左，白色越早出现。</small>
            </label>

            <label className="settings-toggle settings-toggle--cover">
              <span>
                <strong>头像柔光</strong>
                <small>在两个人周围叠一层很淡的白色径向磨砂。</small>
              </span>
              <input
                type="checkbox"
                checked={coverSettings.portraitHazeEnabled}
                onChange={(event) => onCoverSetting("portraitHazeEnabled", event.target.checked)}
              />
            </label>

            <label className="settings-field">
              <span>写在我们下面的话</span>
              <textarea
                value={coverSettings.tagline}
                rows={3}
                maxLength={80}
                onChange={(event) => onCoverSetting("tagline", event.target.value)}
              />
            </label>
          </section>

          <section className="settings-group" aria-labelledby="settings-composition-title">
            <div className="settings-group__heading">
              <h3 id="settings-composition-title">黑白构成</h3>
              <p>装饰会随封面一起收缩，并在画像退场前先淡去。</p>
            </div>
            <label className="settings-toggle">
              <span>
                <strong>显示构成元素</strong>
                <small>随时可以关掉，不会删除已经摆好的位置。</small>
              </span>
              <input
                type="checkbox"
                checked={coverSettings.compositionEnabled}
                onChange={(event) => onCoverSetting("compositionEnabled", event.target.checked)}
              />
            </label>
            <div className="settings-actions">
              <button type="button" onClick={onEditComposition}>编辑位置与大小</button>
              <button type="button" onClick={onResetComposition}>恢复默认构图</button>
            </div>
          </section>
        </div>
      </section>
    </>
  );
}
