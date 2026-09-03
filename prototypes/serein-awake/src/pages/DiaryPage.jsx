import { useEffect, useMemo, useRef, useState } from "react";
import {
  BookOpenText,
  CalendarBlank,
  CaretLeft,
  CaretRight,
  ChatCircleText,
  Check,
  ListBullets,
  LockSimple,
  MagnifyingGlass,
  PencilSimpleLine,
  Plus,
  Quotes,
  Trash,
  X,
} from "@phosphor-icons/react";
import { defaultDarkroom } from "../data/diary.js";
import {
  deleteDiaryEntry,
  loadDiarySnapshot,
  readDiaryEntries,
  readDiaryUserIdentity,
  storeDiaryEntries,
} from "../storage/diaryStore.js";
import { MarkdownProjection } from "../components/MarkdownProjection.jsx";

const dayNames = ["日", "一", "二", "三", "四", "五", "六"];

const sortEntries = (entries) => [...entries].sort((a, b) => (
  `${b.date}T${b.time}`.localeCompare(`${a.date}T${a.time}`)
));

const formatDiaryDate = (date) => new Intl.DateTimeFormat("zh-CN", {
  year: "numeric",
  month: "long",
  day: "numeric",
  weekday: "long",
}).format(new Date(`${date}T12:00:00`));

const formatMonth = (month) => {
  const [year, monthNumber] = month.split("-").map(Number);
  return `${year}年${monthNumber}月`;
};

const shiftMonth = (month, delta) => {
  const [year, monthNumber] = month.split("-").map(Number);
  const next = new Date(year, monthNumber - 1 + delta, 1);
  return `${next.getFullYear()}-${String(next.getMonth() + 1).padStart(2, "0")}`;
};

const getMonthCells = (month) => {
  const [year, monthNumber] = month.split("-").map(Number);
  const firstDay = new Date(year, monthNumber - 1, 1).getDay();
  const days = new Date(year, monthNumber, 0).getDate();
  return [
    ...Array.from({ length: firstDay }, (_, index) => ({ id: `blank-${index}`, day: null })),
    ...Array.from({ length: days }, (_, index) => {
      const day = index + 1;
      return {
        id: `${month}-${String(day).padStart(2, "0")}`,
        day,
        date: `${month}-${String(day).padStart(2, "0")}`,
      };
    }),
  ];
};

const formatDarkroomCountdown = (totalSeconds) => {
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return [hours, minutes, seconds].map((value) => String(value).padStart(2, "0")).join(":");
};

const isDarkroomEntryLocked = (entry, now) => {
  const unlockTime = new Date(entry.unlockAt).getTime();
  return Number.isFinite(unlockTime) ? now < unlockTime : entry.locked === true;
};

export function DiaryPage() {
  const readerRef = useRef(null);
  const [diaryUserIdentity, setDiaryUserIdentity] = useState(readDiaryUserIdentity);
  const [entries, setEntries] = useState(() => sortEntries(readDiaryEntries()));
  const [selectedEntryId, setSelectedEntryId] = useState(() => readDiaryEntries()[0]?.id ?? null);
  const [query, setQuery] = useState("");
  const [view, setView] = useState("list");
  const [month, setMonth] = useState("2026-07");
  const [calendarDate, setCalendarDate] = useState(() => readDiaryEntries()[0]?.date ?? null);
  const [composerOpen, setComposerOpen] = useState(false);
  const [editingEntryId, setEditingEntryId] = useState(null);
  const [darkroomOpen, setDarkroomOpen] = useState(false);
  const [darkroomEntryId, setDarkroomEntryId] = useState(null);
  const [darkroomClock, setDarkroomClock] = useState(() => Date.now());
  const [darkroomLineCount, setDarkroomLineCount] = useState(0);
  const [commentDraft, setCommentDraft] = useState("");
  const [deletingEntryId, setDeletingEntryId] = useState(null);
  const [draft, setDraft] = useState({
    date: new Date().toISOString().slice(0, 10),
    title: "",
    body: "",
  });

  useEffect(() => {
    storeDiaryEntries(entries);
  }, [entries]);

  useEffect(() => {
    let cancelled = false;
    loadDiarySnapshot().then((snapshotEntries) => {
      if (cancelled || !snapshotEntries?.length) return;
      const nextEntries = sortEntries(snapshotEntries);
      const firstEntry = nextEntries.find((entry) => !entry.darkroom) ?? nextEntries[0];
      setEntries(nextEntries);
      setSelectedEntryId(firstEntry?.id ?? null);
      setCalendarDate(firstEntry?.date ?? null);
      setMonth(firstEntry?.date?.slice(0, 7) ?? "2026-07");
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const refreshUserIdentity = (event) => {
      if (event.detail?.key !== "serein.awake.name.xiaoyu") return;
      setDiaryUserIdentity(readDiaryUserIdentity());
    };
    window.addEventListener("serein:preference-change", refreshUserIdentity);
    return () => window.removeEventListener("serein:preference-change", refreshUserIdentity);
  }, []);

  useEffect(() => {
    if (!composerOpen && !darkroomOpen) return undefined;
    const closeOnEscape = (event) => {
      if (event.key !== "Escape") return;
      setComposerOpen(false);
      setEditingEntryId(null);
      setDarkroomOpen(false);
    };
    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [composerOpen, darkroomOpen]);

  const diaryEntries = useMemo(() => entries.filter((entry) => !entry.darkroom), [entries]);
  const darkroomEntries = useMemo(() => entries.filter((entry) => entry.darkroom), [entries]);

  const filteredEntries = useMemo(() => {
    const normalized = query.trim().toLocaleLowerCase("zh-CN");
    if (!normalized) return diaryEntries;
    return diaryEntries.filter((entry) => [
      entry.title,
      entry.excerpt,
      ...entry.body,
    ].join(" ").toLocaleLowerCase("zh-CN").includes(normalized));
  }, [diaryEntries, query]);

  const selectedEntry = filteredEntries.find((entry) => entry.id === selectedEntryId)
    ?? filteredEntries[0]
    ?? null;

  useEffect(() => {
    readerRef.current?.scrollTo({ top: 0, behavior: "smooth" });
  }, [selectedEntry?.id]);

  const entriesByDate = useMemo(() => {
    const grouped = new Map();
    diaryEntries.forEach((entry) => {
      const existing = grouped.get(entry.date) ?? [];
      grouped.set(entry.date, [...existing, entry]);
    });
    return grouped;
  }, [diaryEntries]);

  const monthCells = useMemo(() => getMonthCells(month), [month]);
  const calendarEntries = calendarDate ? entriesByDate.get(calendarDate) ?? [] : [];
  const selectedDayEntries = selectedEntry ? entriesByDate.get(selectedEntry.date) ?? [] : [];
  const selectedDayIndex = selectedDayEntries.findIndex((entry) => entry.id === selectedEntry?.id);
  const lockedDarkroomEntry = darkroomEntries.find((entry) => isDarkroomEntryLocked(entry, darkroomClock));
  const darkroomUnlockAt = lockedDarkroomEntry?.unlockAt || defaultDarkroom.unlockAt;
  const darkroomUnlockTime = new Date(darkroomUnlockAt).getTime();
  const darkroomLocked = lockedDarkroomEntry != null;
  const darkroomRemainingSeconds = Math.max(0, Math.ceil((darkroomUnlockTime - darkroomClock) / 1000));
  const selectedDarkroomEntry = darkroomEntries.find((entry) => entry.id === darkroomEntryId)
    ?? darkroomEntries[0]
    ?? null;

  useEffect(() => {
    if (!darkroomOpen || !darkroomLocked) {
      setDarkroomLineCount(0);
      return undefined;
    }

    setDarkroomLineCount(0);
    const lineTimers = [
      window.setTimeout(() => setDarkroomLineCount(1), 500),
      window.setTimeout(() => setDarkroomLineCount(2), 2000),
      window.setTimeout(() => setDarkroomLineCount(3), 4500),
      window.setTimeout(() => setDarkroomLineCount(4), 6500),
    ];
    const clockTimer = window.setInterval(() => setDarkroomClock(Date.now()), 1000);

    return () => {
      lineTimers.forEach((timer) => window.clearTimeout(timer));
      window.clearInterval(clockTimer);
    };
  }, [darkroomLocked, darkroomOpen]);

  useEffect(() => {
    if (!darkroomOpen || darkroomLocked || !darkroomEntries.some((entry) => entry.locked === true)) {
      return undefined;
    }

    let cancelled = false;
    loadDiarySnapshot().then((snapshotEntries) => {
      if (!cancelled && snapshotEntries?.length) setEntries(sortEntries(snapshotEntries));
    });
    return () => {
      cancelled = true;
    };
  }, [darkroomLocked, darkroomOpen]);

  const openDarkroom = () => {
    setDarkroomEntryId((current) => current ?? darkroomEntries[0]?.id ?? null);
    setDarkroomClock(Date.now());
    setDarkroomOpen(true);
  };

  const updateEntry = (entryId, update) => {
    setEntries((current) => current.map((entry) => (
      entry.id === entryId ? { ...entry, ...update(entry) } : entry
    )));
  };

  const selectEntry = (entry) => {
    setSelectedEntryId(entry.id);
    setCalendarDate(entry.date);
    setCommentDraft("");
  };

  const selectSameDayEntry = (offset) => {
    const nextEntry = selectedDayEntries[selectedDayIndex + offset];
    if (nextEntry) selectEntry(nextEntry);
  };

  const openNewComposer = () => {
    setEditingEntryId(null);
    setDraft({
      date: new Date().toISOString().slice(0, 10),
      title: "",
      body: "",
    });
    setComposerOpen(true);
  };

  const openEditComposer = (entry) => {
    setEditingEntryId(entry.id);
    setDraft({
      date: entry.date,
      title: entry.title,
      body: entry.body.join("\n\n"),
    });
    setComposerOpen(true);
  };

  const closeComposer = () => {
    setComposerOpen(false);
    setEditingEntryId(null);
  };

  const saveDiary = (event) => {
    event.preventDefault();
    const title = draft.title.trim();
    const body = draft.body
      .split(/\n\s*\n/)
      .map((paragraph) => paragraph.trim())
      .filter(Boolean);
    if (!title || !body.length || !draft.date) return;

    const now = new Date();
    if (editingEntryId) {
      setEntries((current) => sortEntries(current.map((entry) => (
        entry.id === editingEntryId
          ? {
              ...entry,
              date: draft.date,
              title,
              excerpt: body[0].slice(0, 64),
              body,
              revision: (entry.revision ?? 1) + 1,
              updatedAt: now.toISOString(),
            }
          : entry
      ))));
      setSelectedEntryId(editingEntryId);
      setMonth(draft.date.slice(0, 7));
      setCalendarDate(draft.date);
      setQuery("");
      closeComposer();
      return;
    }

    const entry = {
      id: `diary-${Date.now()}`,
      date: draft.date,
      time: `${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}`,
      ...diaryUserIdentity,
      title,
      excerpt: body[0].slice(0, 64),
      body,
      references: [],
      comments: [],
    };
    setEntries((current) => sortEntries([entry, ...current]));
    setSelectedEntryId(entry.id);
    setMonth(entry.date.slice(0, 7));
    setCalendarDate(entry.date);
    setView("list");
    setDraft({ date: new Date().toISOString().slice(0, 10), title: "", body: "" });
    closeComposer();
  };

  const addComment = (event) => {
    event.preventDefault();
    if (!selectedEntry || !commentDraft.trim()) return;
    const now = new Date();
    updateEntry(selectedEntry.id, (entry) => ({
      comments: [
        ...entry.comments,
        {
          id: `diary-comment-${Date.now()}`,
          ...diaryUserIdentity,
          createdAt: now.toLocaleString("zh-CN", {
            month: "2-digit",
            day: "2-digit",
            hour: "2-digit",
            minute: "2-digit",
          }),
          content: commentDraft.trim(),
        },
      ],
    }));
    setCommentDraft("");
  };

  const deleteComment = (commentId) => {
    if (!selectedEntry) return;
    updateEntry(selectedEntry.id, (entry) => ({
      comments: entry.comments.filter((comment) => comment.id !== commentId),
    }));
  };

  const deleteDiary = async () => {
    if (!selectedEntry || deletingEntryId) return;
    const deletionDetail = selectedEntry.sourceKind === "ombre-diary-live-readonly"
      ? "删除后，它会从日记和搜索里消失；系统会保留删除前的修订快照。"
      : "这篇只保存在当前浏览器，删除后无法从这里恢复。";
    const confirmed = window.confirm(
      `删除「${selectedEntry.title}」？\n\n${deletionDetail}`,
    );
    if (!confirmed) return;

    const entryId = selectedEntry.id;
    setDeletingEntryId(entryId);
    try {
      await deleteDiaryEntry(selectedEntry);
      const nextEntries = sortEntries(entries.filter((entry) => entry.id !== entryId));
      const nextSelected = nextEntries.find((entry) => !entry.darkroom) ?? null;
      setEntries(nextEntries);
      setSelectedEntryId(nextSelected?.id ?? null);
      setCalendarDate(nextSelected?.date ?? null);
      if (nextSelected?.date) setMonth(nextSelected.date.slice(0, 7));
    } catch (error) {
      window.alert(error?.message || "这篇日记没有删掉，请稍后再试。");
    } finally {
      setDeletingEntryId(null);
    }
  };

  return (
    <div className="diary-layout">
      <header className="diary-toolbar">
        <div className="diary-toolbar__title">
          <span>DIARY</span>
          <h1>日记</h1>
        </div>

        <label className="diary-search">
          <MagnifyingGlass size={19} weight="light" aria-hidden="true" />
          <span className="sr-only">搜索日记标题或正文</span>
          <input
            type="search"
            value={query}
            placeholder="搜索日记标题或正文"
            onChange={(event) => {
              setQuery(event.target.value);
              if (event.target.value) setView("list");
            }}
          />
          {query ? (
            <button type="button" aria-label="清空搜索" onClick={() => setQuery("")}>
              <X size={14} weight="light" aria-hidden="true" />
            </button>
          ) : null}
        </label>

        <div className="diary-toolbar__actions">
          <button className="diary-action diary-action--quiet" type="button" onClick={openDarkroom}>
            <LockSimple size={18} weight="light" aria-hidden="true" />
            进入暗房
          </button>
          <button className="diary-action diary-action--primary" type="button" onClick={openNewComposer}>
            <PencilSimpleLine size={18} weight="light" aria-hidden="true" />
            写一篇
          </button>
        </div>
      </header>

      <div className="diary-workspace">
        <aside className={`diary-browser diary-browser--${view}`} aria-label="日记索引">
          <div className="diary-view-switcher" aria-label="日记浏览方式">
            <button
              className={view === "list" ? "is-active" : ""}
              type="button"
              aria-pressed={view === "list"}
              onClick={() => setView("list")}
            >
              <ListBullets size={16} weight="light" aria-hidden="true" />
              列表
            </button>
            <button
              className={view === "calendar" ? "is-active" : ""}
              type="button"
              aria-pressed={view === "calendar"}
              onClick={() => setView("calendar")}
            >
              <CalendarBlank size={16} weight="light" aria-hidden="true" />
              月历
            </button>
          </div>

          {view === "list" ? (
            <div className="diary-entry-list">
              <p>{query ? `找到 ${filteredEntries.length} 篇` : "按日期排列"}</p>
              {filteredEntries.length ? filteredEntries.map((entry) => (
                <button
                  className={selectedEntry?.id === entry.id ? "is-active" : ""}
                  key={entry.id}
                  type="button"
                  onClick={() => selectEntry(entry)}
                >
                  <time dateTime={entry.date}>
                    <strong>{entry.date.slice(8, 10)}</strong>
                    <span>{Number(entry.date.slice(5, 7))}月</span>
                  </time>
                  <span>
                    <strong>{entry.title}</strong>
                    <small>{entry.excerpt}</small>
                    <em>{entry.author} · {entry.role}</em>
                  </span>
                </button>
              )) : (
                <div className="diary-browser__empty">
                  <p>没有找到这篇日记。</p>
                  <button type="button" onClick={() => setQuery("")}>清空搜索</button>
                </div>
              )}
            </div>
          ) : (
            <div className="diary-calendar">
              <header>
                <button type="button" aria-label="上个月" onClick={() => setMonth((current) => shiftMonth(current, -1))}>
                  <CaretLeft size={17} weight="light" aria-hidden="true" />
                </button>
                <strong>{formatMonth(month)}</strong>
                <button type="button" aria-label="下个月" onClick={() => setMonth((current) => shiftMonth(current, 1))}>
                  <CaretRight size={17} weight="light" aria-hidden="true" />
                </button>
              </header>
              <div className="diary-calendar__weekdays" aria-hidden="true">
                {dayNames.map((day) => <span key={day}>{day}</span>)}
              </div>
              <div className="diary-calendar__days">
                {monthCells.map((cell) => {
                  if (!cell.day) return <span className="is-blank" key={cell.id} />;
                  const dayEntries = entriesByDate.get(cell.date) ?? [];
                  const active = calendarDate === cell.date;
                  return (
                    <button
                      className={`${dayEntries.length ? "has-entry" : ""}${active ? " is-active" : ""}`}
                      key={cell.id}
                      type="button"
                      aria-label={`${cell.date}${dayEntries.length ? `，${dayEntries.length} 篇日记` : "，没有日记"}`}
                      disabled={!dayEntries.length}
                      onClick={() => {
                        setCalendarDate(cell.date);
                        if (dayEntries.length === 1) selectEntry(dayEntries[0]);
                        if (dayEntries.length > 1 && !dayEntries.some((entry) => entry.id === selectedEntry?.id)) {
                          selectEntry(dayEntries[0]);
                        }
                      }}
                    >
                      {cell.day}
                      {dayEntries.length > 1 ? (
                        <span className="diary-calendar__count" aria-hidden="true">{dayEntries.length}</span>
                      ) : dayEntries.length ? <i aria-hidden="true" /> : null}
                    </button>
                  );
                })}
              </div>
              <p>
                {calendarEntries.length > 1
                  ? `这一天有 ${calendarEntries.length} 篇，正文末尾可以翻阅。`
                  : "有墨点的日子写过东西。"}
              </p>
            </div>
          )}
        </aside>

        <article className="diary-reader" aria-live="polite" ref={readerRef}>
          {selectedEntry ? (
            <div className="diary-reader__inner" key={selectedEntry.id}>
              <header className="diary-reader__header">
                <div className="diary-reader__meta">
                  <time dateTime={selectedEntry.date}>
                    {formatDiaryDate(selectedEntry.date)} · {selectedEntry.time} · {selectedEntry.author}
                  </time>
                  <div className="diary-reader__actions">
                    <button type="button" onClick={() => openEditComposer(selectedEntry)}>
                      <PencilSimpleLine size={14} weight="light" aria-hidden="true" />
                      编辑
                    </button>
                    <button
                      className="diary-reader__delete"
                      type="button"
                      disabled={deletingEntryId === selectedEntry.id}
                      onClick={deleteDiary}
                    >
                      <Trash size={14} weight="light" aria-hidden="true" />
                      {deletingEntryId === selectedEntry.id ? "删除中…" : "删除"}
                    </button>
                  </div>
                </div>
                <h2>{selectedEntry.title}</h2>
                {selectedEntry.references.length ? (
                  <div className="diary-reference-list" aria-label="引用关系">
                    {selectedEntry.references.map((reference) => (
                      <span key={reference.id}>
                        <Quotes size={15} weight="light" aria-hidden="true" />
                        被 {reference.kind}「{reference.title}」引用
                      </span>
                    ))}
                  </div>
                ) : null}
              </header>

              <MarkdownProjection
                className="diary-reader__body"
                content={selectedEntry.body}
              />

              {selectedDayEntries.length > 1 ? (
                <nav className="diary-day-navigation" aria-label="翻阅当天的其他日记">
                  <button
                    type="button"
                    disabled={selectedDayIndex <= 0}
                    onClick={() => selectSameDayEntry(-1)}
                  >
                    <span>上一篇</span>
                    <small>{selectedDayEntries[selectedDayIndex - 1]?.title}</small>
                  </button>
                  <span>{selectedDayIndex + 1} / {selectedDayEntries.length}</span>
                  <button
                    type="button"
                    disabled={selectedDayIndex >= selectedDayEntries.length - 1}
                    onClick={() => selectSameDayEntry(1)}
                  >
                    <span>下一篇</span>
                    <small>{selectedDayEntries[selectedDayIndex + 1]?.title}</small>
                  </button>
                </nav>
              ) : null}

              <section className="diary-comments" aria-labelledby={`diary-comments-${selectedEntry.id}`}>
                <header>
                  <div>
                    <ChatCircleText size={18} weight="light" aria-hidden="true" />
                    <h3 id={`diary-comments-${selectedEntry.id}`}>评论</h3>
                    <span>{selectedEntry.comments.length}</span>
                  </div>
                </header>

                {selectedEntry.comments.length ? (
                  <div className="diary-comments__list">
                    {selectedEntry.comments.map((comment) => (
                      <article key={comment.id}>
                        <header>
                          <div>
                            <strong>{comment.author}</strong>
                            <span>{comment.role}</span>
                            <time>{comment.createdAt}</time>
                          </div>
                          <button
                            type="button"
                            aria-label={`删除 ${comment.author} 的评论`}
                            onClick={() => deleteComment(comment.id)}
                          >
                            <Trash size={14} weight="light" aria-hidden="true" />
                          </button>
                        </header>
                        <MarkdownProjection
                          className="diary-comment__content"
                          content={comment.content}
                        />
                      </article>
                    ))}
                  </div>
                ) : (
                  <p className="diary-comments__empty">还没有人在这页纸边写字。</p>
                )}

                <form className="diary-comment-composer" onSubmit={addComment}>
                  <textarea
                    rows={3}
                    value={commentDraft}
                    placeholder="在这篇日记旁边留一句话…"
                    onChange={(event) => setCommentDraft(event.target.value)}
                  />
                  <footer>
                    <span>{diaryUserIdentity.author} · {diaryUserIdentity.role}</span>
                    <button type="submit" disabled={!commentDraft.trim()}>
                      <Plus size={14} weight="light" aria-hidden="true" />
                      添加评论
                    </button>
                  </footer>
                </form>
              </section>
            </div>
          ) : (
            <div className="diary-reader__empty">
              <BookOpenText size={25} weight="light" aria-hidden="true" />
              <p>这里还没有翻开的日记。</p>
            </div>
          )}
        </article>
      </div>

      {composerOpen ? (
        <div className="diary-modal-layer" role="presentation">
          <button className="diary-modal-layer__veil" type="button" aria-label="关闭写作空间" onClick={closeComposer} />
          <section className="diary-composer" role="dialog" aria-modal="true" aria-labelledby="diary-composer-title">
            <header>
              <div>
                <span>{editingEntryId ? "EDIT ENTRY" : "NEW ENTRY"}</span>
                <h2 id="diary-composer-title">{editingEntryId ? "编辑日记" : "写一篇"}</h2>
              </div>
              <button type="button" aria-label="关闭写作空间" onClick={closeComposer}>
                <X size={22} weight="light" aria-hidden="true" />
              </button>
            </header>
            <form onSubmit={saveDiary}>
              <p className="diary-composer__identity">
                {editingEntryId
                  ? `原署名 ${selectedEntry?.author} · ${selectedEntry?.role}，修改不会改变署名`
                  : `署名为 ${diaryUserIdentity.author} · ${diaryUserIdentity.role}`}
              </p>
              <label>
                <span>日期</span>
                <input
                  type="date"
                  value={draft.date}
                  onChange={(event) => setDraft((current) => ({ ...current, date: event.target.value }))}
                />
              </label>
              <label>
                <span>标题</span>
                <input
                  autoFocus
                  type="text"
                  value={draft.title}
                  placeholder="给今天留一个名字"
                  onChange={(event) => setDraft((current) => ({ ...current, title: event.target.value }))}
                />
              </label>
              <label className="diary-composer__body">
                <span>正文</span>
                <textarea
                  rows={10}
                  value={draft.body}
                  placeholder="从这一刻开始写。空行会成为新的段落。"
                  onChange={(event) => setDraft((current) => ({ ...current, body: event.target.value }))}
                />
              </label>
              <footer>
                <button type="button" onClick={closeComposer}>取消</button>
                <button className="is-primary" type="submit" disabled={!draft.title.trim() || !draft.body.trim()}>
                  <Check size={15} weight="light" aria-hidden="true" />
                  {editingEntryId ? "保存修改" : "保存日记"}
                </button>
              </footer>
            </form>
          </section>
        </div>
      ) : null}

      {darkroomOpen ? (
        <div
          className={`diary-modal-layer diary-modal-layer--darkroom diary-modal-layer--darkroom-${darkroomLocked ? "locked" : "open"}`}
          role="presentation"
        >
          <div className="diary-modal-layer__veil" aria-hidden="true" />
          {darkroomLocked ? (
            <section className="darkroom-door" role="dialog" aria-modal="true" aria-labelledby="darkroom-title">
              <div className="darkroom-door__content">
                <LockSimple size={32} weight="light" aria-hidden="true" />
                <div className="darkroom-door__dialogue">
                  <h2
                    className={`darkroom-door__line ${darkroomLineCount >= 1 ? "is-visible" : ""}`}
                    id="darkroom-title"
                  >
                    {defaultDarkroom.lockedTitle}
                  </h2>
                  <p className={`darkroom-door__line ${darkroomLineCount >= 2 ? "is-visible" : ""}`}>
                    {defaultDarkroom.lockedQuestion}
                  </p>
                  <p className={`darkroom-door__line ${darkroomLineCount >= 3 ? "is-visible" : ""}`}>
                    {defaultDarkroom.lockedCopy}
                  </p>
                  <p className={`darkroom-door__line ${darkroomLineCount >= 4 ? "is-visible" : ""}`}>
                    你可以在门口等一会儿。
                  </p>
                </div>
                <div className={`darkroom-door__countdown ${darkroomLineCount >= 4 ? "is-visible" : ""}`}>
                  距离开门还有 {formatDarkroomCountdown(darkroomRemainingSeconds)}
                </div>
                <div className={`darkroom-door__actions ${darkroomLineCount >= 4 ? "is-visible" : ""}`}>
                  <button type="button" onClick={() => setDarkroomOpen(false)}>在外面等</button>
                </div>
              </div>
            </section>
          ) : (
            <section className="darkroom-open-room" role="dialog" aria-modal="true" aria-labelledby="darkroom-open-title">
              <header>
                <div>
                  <span>OPEN</span>
                  <h2 id="darkroom-open-title">{defaultDarkroom.title}</h2>
                </div>
                <button type="button" onClick={() => setDarkroomOpen(false)}>
                  <X size={18} weight="light" aria-hidden="true" />
                  回到日记
                </button>
              </header>

              <div className="darkroom-open-room__body">
                <aside aria-label="暗房日记">
                  <p>门后的日记</p>
                  {darkroomEntries.map((entry) => (
                    <button
                      className={selectedDarkroomEntry?.id === entry.id ? "is-active" : ""}
                      key={entry.id}
                      type="button"
                      onClick={() => setDarkroomEntryId(entry.id)}
                    >
                      <time dateTime={entry.date}>{entry.date.replaceAll("-", ".")}</time>
                      <strong>{entry.title}</strong>
                      <span>{entry.author} · {entry.role}</span>
                    </button>
                  ))}
                </aside>

                <article>
                  {selectedDarkroomEntry ? (
                    <div className="darkroom-open-room__page" key={selectedDarkroomEntry.id}>
                      <header>
                        <time dateTime={selectedDarkroomEntry.date}>
                          {formatDiaryDate(selectedDarkroomEntry.date)} · {selectedDarkroomEntry.time} · {selectedDarkroomEntry.author}
                        </time>
                        <h3>{selectedDarkroomEntry.title}</h3>
                      </header>
                      <MarkdownProjection
                        className="darkroom-open-room__content"
                        content={selectedDarkroomEntry.body}
                      />
                    </div>
                  ) : (
                    <div className="darkroom-open-room__empty">
                      <BookOpenText size={25} weight="light" aria-hidden="true" />
                      <p>门后还没有日记。</p>
                    </div>
                  )}
                </article>
              </div>
            </section>
          )}
        </div>
      ) : null}
    </div>
  );
}
