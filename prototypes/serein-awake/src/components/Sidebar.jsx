import {
  BookOpenText,
  Books,
  GearFine,
  House,
  Planet,
  SlidersHorizontal,
  Sparkle,
} from "@phosphor-icons/react";

const navItems = [
  { label: "醒来", icon: House },
  { label: "记忆", icon: Sparkle },
  { label: "叙事卷", icon: Books },
  { label: "日记", icon: BookOpenText },
  { label: "地下室", icon: SlidersHorizontal },
];

export function Sidebar({ activeArea, onNavigate, onUnavailable, onOpenSettings }) {
  return (
    <aside className="sidebar" aria-label="主要导航">
      <div className="sidebar__brand" aria-hidden="true">s</div>
      <nav className="sidebar__nav">
        {navItems.map(({ label, icon: Icon }) => {
          const active = label === activeArea;
          return (
            <button
              className={`nav-item${active ? " is-active" : ""}`}
              key={label}
              type="button"
              aria-current={active ? "page" : undefined}
              onClick={() => !active && onNavigate(label)}
            >
              <Icon size={20} weight={active ? "fill" : "light"} aria-hidden="true" />
              <span>{label}</span>
            </button>
          );
        })}
      </nav>
      <div className="sidebar__footer">
        <button className="nav-item sidebar__universe" type="button" onClick={() => onUnavailable("宇宙")}>
          <Planet size={20} weight="light" aria-hidden="true" />
          <span>宇宙</span>
        </button>
        <button className="nav-item" type="button" onClick={onOpenSettings}>
          <GearFine size={20} weight="light" aria-hidden="true" />
          <span>设置</span>
        </button>
      </div>
    </aside>
  );
}
