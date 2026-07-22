import os
import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any


class GatewayStateStore:
    """
    Tracks successful gateway rounds and which dynamic buckets were injected
    per session, so cooldown and recent-round skipping can work.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS request_rounds (
                session_id TEXT NOT NULL,
                round_id INTEGER NOT NULL,
                completed_at TEXT NOT NULL,
                PRIMARY KEY (session_id, round_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS injected_buckets (
                session_id TEXT NOT NULL,
                round_id INTEGER NOT NULL,
                bucket_id TEXT NOT NULL,
                injected_at TEXT NOT NULL,
                PRIMARY KEY (session_id, round_id, bucket_id)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_injected_lookup
            ON injected_buckets (session_id, bucket_id, injected_at DESC)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS injection_debug (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                round_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_injection_debug_lookup
            ON injection_debug (session_id, id DESC)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recent_context_injections (
                session_id TEXT NOT NULL,
                round_id INTEGER NOT NULL,
                injected_at TEXT NOT NULL,
                PRIMARY KEY (session_id, round_id)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_recent_context_lookup
            ON recent_context_injections (session_id, injected_at DESC)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                round_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                user_text TEXT NOT NULL DEFAULT '',
                assistant_text TEXT NOT NULL DEFAULT '',
                model TEXT NOT NULL DEFAULT '',
                client TEXT NOT NULL DEFAULT '',
                route TEXT NOT NULL DEFAULT '',
                UNIQUE(profile_id, session_id, round_id)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_conversation_turns_recent
            ON conversation_turns (profile_id, created_at DESC, id DESC)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_conversation_turns_session
            ON conversation_turns (profile_id, session_id, created_at DESC)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS upstream_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                round_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                model TEXT NOT NULL DEFAULT '',
                route TEXT NOT NULL DEFAULT '',
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                prompt_cache_hit_tokens INTEGER,
                prompt_cache_miss_tokens INTEGER,
                cached_tokens INTEGER,
                cache_read_input_tokens INTEGER,
                cache_creation_input_tokens INTEGER,
                usage_json TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_upstream_usage_lookup
            ON upstream_usage (session_id, id DESC)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS window_sessions (
                conversation_id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL,
                previous_conversation_id TEXT NOT NULL DEFAULT '',
                parent_shadow_id TEXT NOT NULL DEFAULT '',
                closed_shadow_id TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'open',
                first_hook_prompt_at TEXT NOT NULL DEFAULT '',
                opened_at TEXT NOT NULL,
                closed_at TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_window_sessions_profile_recent
            ON window_sessions (profile_id, opened_at DESC)
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_window_sessions_parent_claim
            ON window_sessions (parent_shadow_id)
            WHERE parent_shadow_id != ''
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS window_session_aliases (
                profile_id TEXT NOT NULL,
                transport_session_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (profile_id, transport_session_id)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_window_session_aliases_conversation
            ON window_session_aliases (conversation_id)
            """
        )
        window_columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(window_sessions)").fetchall()
        }
        if "first_hook_prompt_at" not in window_columns:
            conn.execute(
                "ALTER TABLE window_sessions ADD COLUMN first_hook_prompt_at TEXT NOT NULL DEFAULT ''"
            )
        conn.commit()
        conn.close()

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")

    @staticmethod
    def _window_session_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        item = dict(row)
        item["handoff_ready"] = bool(str(item.get("parent_shadow_id") or "").strip())
        item["hook_first_prompt_seen"] = bool(
            str(item.get("first_hook_prompt_at") or "").strip()
        )
        return item

    def get_window_session(
        self,
        conversation_id: str,
        *,
        profile_id: str = "",
    ) -> dict[str, Any] | None:
        safe_conversation_id = str(conversation_id or "").strip()
        if not safe_conversation_id:
            return None
        conn = self._connect()
        if str(profile_id or "").strip():
            row = conn.execute(
                "SELECT * FROM window_sessions WHERE conversation_id = ? AND profile_id = ?",
                (safe_conversation_id, str(profile_id).strip()),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM window_sessions WHERE conversation_id = ?",
                (safe_conversation_id,),
            ).fetchone()
        conn.close()
        return self._window_session_row(row)

    def resolve_window_conversation(
        self,
        *,
        profile_id: str,
        transport_session_id: str,
        rotate_closed: bool = True,
    ) -> dict[str, Any]:
        """Resolve a client-stable session id to one internal window epoch.

        Closing the active epoch leaves its authored Shadow on that row.  The
        next valid chat request atomically creates (or reuses) exactly one child
        epoch and binds the stable transport id to it.
        """
        safe_profile_id = str(profile_id or "default").strip() or "default"
        safe_transport_id = str(transport_session_id or "main").strip() or "main"
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            alias = conn.execute(
                """
                SELECT conversation_id
                FROM window_session_aliases
                WHERE profile_id = ? AND transport_session_id = ?
                """,
                (safe_profile_id, safe_transport_id),
            ).fetchone()
            active = None
            if alias is not None:
                active = conn.execute(
                    """
                    SELECT * FROM window_sessions
                    WHERE conversation_id = ? AND profile_id = ?
                    """,
                    (str(alias["conversation_id"] or ""), safe_profile_id),
                ).fetchone()
                if active is None:
                    conn.execute(
                        """
                        DELETE FROM window_session_aliases
                        WHERE profile_id = ? AND transport_session_id = ?
                        """,
                        (safe_profile_id, safe_transport_id),
                    )

            resolution = "existing"
            if active is None:
                direct = conn.execute(
                    "SELECT * FROM window_sessions WHERE conversation_id = ?",
                    (safe_transport_id,),
                ).fetchone()
                if direct is not None and str(direct["profile_id"] or "") == safe_profile_id:
                    active = direct
                    resolution = "bound"
                else:
                    conversation_id = (
                        safe_transport_id
                        if direct is None
                        else f"conversation_{uuid.uuid4().hex}"
                    )
                    now = self._utc_now()
                    conn.execute(
                        """
                        INSERT INTO window_sessions (
                            conversation_id, profile_id, previous_conversation_id,
                            parent_shadow_id, closed_shadow_id, status,
                            opened_at, closed_at, updated_at
                        ) VALUES (?, ?, '', '', '', 'open', ?, '', ?)
                        """,
                        (conversation_id, safe_profile_id, now, now),
                    )
                    active = conn.execute(
                        "SELECT * FROM window_sessions WHERE conversation_id = ?",
                        (conversation_id,),
                    ).fetchone()
                    resolution = "opened"

                now = self._utc_now()
                conn.execute(
                    """
                    INSERT INTO window_session_aliases (
                        profile_id, transport_session_id, conversation_id,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(profile_id, transport_session_id) DO UPDATE SET
                        conversation_id = excluded.conversation_id,
                        updated_at = excluded.updated_at
                    """,
                    (
                        safe_profile_id,
                        safe_transport_id,
                        str(active["conversation_id"] or ""),
                        now,
                        now,
                    ),
                )

            rotated = False
            visited_ids: set[str] = set()
            while (
                rotate_closed
                and active is not None
                and str(active["status"] or "") == "closed"
                and str(active["closed_shadow_id"] or "").strip()
            ):
                previous_id = str(active["conversation_id"] or "").strip()
                if previous_id in visited_ids:
                    conn.rollback()
                    return {
                        "status": "conflict",
                        "reason": "window_chain_cycle",
                        "transport_session_id": safe_transport_id,
                        "conversation_id": previous_id,
                    }
                visited_ids.add(previous_id)
                parent_shadow_id = str(active["closed_shadow_id"] or "").strip()
                child = conn.execute(
                    "SELECT * FROM window_sessions WHERE parent_shadow_id = ?",
                    (parent_shadow_id,),
                ).fetchone()
                if child is not None and str(child["profile_id"] or "") != safe_profile_id:
                    conn.rollback()
                    return {
                        "status": "conflict",
                        "reason": "parent_shadow_profile_mismatch",
                        "transport_session_id": safe_transport_id,
                        "conversation_id": previous_id,
                    }
                if child is None:
                    child_id = f"conversation_{uuid.uuid4().hex}"
                    now = self._utc_now()
                    conn.execute(
                        """
                        INSERT INTO window_sessions (
                            conversation_id, profile_id, previous_conversation_id,
                            parent_shadow_id, closed_shadow_id, status,
                            opened_at, closed_at, updated_at
                        ) VALUES (?, ?, ?, ?, '', 'open', ?, '', ?)
                        """,
                        (
                            child_id,
                            safe_profile_id,
                            previous_id,
                            parent_shadow_id,
                            now,
                            now,
                        ),
                    )
                    child = conn.execute(
                        "SELECT * FROM window_sessions WHERE conversation_id = ?",
                        (child_id,),
                    ).fetchone()
                active = child
                rotated = True

            if rotated and active is not None:
                now = self._utc_now()
                conn.execute(
                    """
                    UPDATE window_session_aliases
                    SET conversation_id = ?, updated_at = ?
                    WHERE profile_id = ? AND transport_session_id = ?
                    """,
                    (
                        str(active["conversation_id"] or ""),
                        now,
                        safe_profile_id,
                        safe_transport_id,
                    ),
                )
                conn.commit()
                item = self._window_session_row(active) or {}
                item.update(
                    {
                        "resolution": "rotated",
                        "rotated": True,
                        "transport_session_id": safe_transport_id,
                    }
                )
                return item

            conn.commit()
            item = self._window_session_row(active) or {}
            item.update(
                {
                    "resolution": resolution,
                    "rotated": False,
                    "transport_session_id": safe_transport_id,
                }
            )
            return item
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def open_window(
        self,
        *,
        profile_id: str,
        conversation_id: str = "",
        previous_conversation_id: str = "",
        parent_shadow_id: str = "",
    ) -> dict[str, Any]:
        """Declare one API/UI window and claim at most one exact parent Shadow."""
        safe_profile_id = str(profile_id or "default").strip() or "default"
        safe_conversation_id = str(conversation_id or "").strip() or f"conversation_{uuid.uuid4().hex}"
        safe_previous_id = str(previous_conversation_id or "").strip()
        requested_parent_id = str(parent_shadow_id or "").strip()

        conn = self._connect()
        existing = conn.execute(
            "SELECT * FROM window_sessions WHERE conversation_id = ?",
            (safe_conversation_id,),
        ).fetchone()
        if existing is not None:
            conn.close()
            item = self._window_session_row(existing) or {}
            if str(item.get("profile_id") or "") != safe_profile_id:
                return {
                    "status": "conflict",
                    "reason": "conversation_profile_mismatch",
                    "conversation_id": safe_conversation_id,
                }
            if safe_previous_id and str(item.get("previous_conversation_id") or "") != safe_previous_id:
                return {
                    "status": "conflict",
                    "reason": "previous_conversation_mismatch",
                    "conversation_id": safe_conversation_id,
                }
            if requested_parent_id and str(item.get("parent_shadow_id") or "") != requested_parent_id:
                return {
                    "status": "conflict",
                    "reason": "parent_shadow_mismatch",
                    "conversation_id": safe_conversation_id,
                }
            item["status"] = "existing"
            return item

        resolved_parent_id = requested_parent_id
        if safe_previous_id:
            previous = conn.execute(
                "SELECT * FROM window_sessions WHERE conversation_id = ? AND profile_id = ?",
                (safe_previous_id, safe_profile_id),
            ).fetchone()
            if previous is None:
                conn.close()
                return {
                    "status": "invalid",
                    "reason": "previous_conversation_not_found",
                    "conversation_id": safe_conversation_id,
                    "previous_conversation_id": safe_previous_id,
                }
            previous_shadow_id = str(previous["closed_shadow_id"] or "").strip()
            if not previous_shadow_id:
                conn.close()
                return {
                    "status": "invalid",
                    "reason": "previous_conversation_not_closed",
                    "conversation_id": safe_conversation_id,
                    "previous_conversation_id": safe_previous_id,
                }
            if resolved_parent_id and resolved_parent_id != previous_shadow_id:
                conn.close()
                return {
                    "status": "conflict",
                    "reason": "parent_shadow_not_from_previous_conversation",
                    "conversation_id": safe_conversation_id,
                }
            resolved_parent_id = previous_shadow_id

        now = self._utc_now()
        try:
            conn.execute(
                """
                INSERT INTO window_sessions (
                    conversation_id, profile_id, previous_conversation_id,
                    parent_shadow_id, closed_shadow_id, status,
                    opened_at, closed_at, updated_at
                ) VALUES (?, ?, ?, ?, '', 'open', ?, '', ?)
                """,
                (
                    safe_conversation_id,
                    safe_profile_id,
                    safe_previous_id,
                    resolved_parent_id,
                    now,
                    now,
                ),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            return {
                "status": "conflict",
                "reason": "parent_shadow_already_claimed" if resolved_parent_id else "conversation_conflict",
                "conversation_id": safe_conversation_id,
                "parent_shadow_id": resolved_parent_id,
            }
        row = conn.execute(
            "SELECT * FROM window_sessions WHERE conversation_id = ?",
            (safe_conversation_id,),
        ).fetchone()
        conn.close()
        item = self._window_session_row(row) or {}
        item["status"] = "opened"
        return item

    def close_window_session(
        self,
        *,
        profile_id: str,
        conversation_id: str,
        shadow_id: str,
    ) -> dict[str, Any]:
        """Bind one authored Shadow to the conversation that produced it."""
        safe_profile_id = str(profile_id or "default").strip() or "default"
        safe_conversation_id = str(conversation_id or "").strip()
        safe_shadow_id = str(shadow_id or "").strip()
        if not safe_conversation_id or not safe_shadow_id:
            return {"status": "invalid", "reason": "conversation_and_shadow_required"}

        conn = self._connect()
        existing = conn.execute(
            "SELECT * FROM window_sessions WHERE conversation_id = ?",
            (safe_conversation_id,),
        ).fetchone()
        now = self._utc_now()
        if existing is None:
            conn.execute(
                """
                INSERT INTO window_sessions (
                    conversation_id, profile_id, previous_conversation_id,
                    parent_shadow_id, closed_shadow_id, status,
                    opened_at, closed_at, updated_at
                ) VALUES (?, ?, '', '', ?, 'closed', ?, ?, ?)
                """,
                (safe_conversation_id, safe_profile_id, safe_shadow_id, now, now, now),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM window_sessions WHERE conversation_id = ?",
                (safe_conversation_id,),
            ).fetchone()
            conn.close()
            item = self._window_session_row(row) or {}
            item["status"] = "closed"
            return item

        if str(existing["profile_id"] or "") != safe_profile_id:
            conn.close()
            return {
                "status": "conflict",
                "reason": "conversation_profile_mismatch",
                "conversation_id": safe_conversation_id,
            }
        existing_shadow_id = str(existing["closed_shadow_id"] or "").strip()
        if existing_shadow_id and existing_shadow_id != safe_shadow_id:
            conn.close()
            return {
                "status": "conflict",
                "reason": "conversation_already_closed_with_other_shadow",
                "conversation_id": safe_conversation_id,
                "closed_shadow_id": existing_shadow_id,
            }
        if existing_shadow_id == safe_shadow_id and str(existing["status"] or "") == "closed":
            conn.close()
            item = self._window_session_row(existing) or {}
            item["status"] = "existing"
            return item

        conn.execute(
            """
            UPDATE window_sessions
            SET closed_shadow_id = ?, status = 'closed', closed_at = ?, updated_at = ?
            WHERE conversation_id = ?
            """,
            (safe_shadow_id, now, now, safe_conversation_id),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM window_sessions WHERE conversation_id = ?",
            (safe_conversation_id,),
        ).fetchone()
        conn.close()
        item = self._window_session_row(row) or {}
        item["status"] = "closed"
        return item

    def claim_first_hook_prompt(
        self,
        *,
        conversation_id: str,
        profile_id: str = "",
    ) -> dict[str, Any]:
        """Atomically mark the first UserPromptSubmit-like hook for one window."""
        safe_conversation_id = str(conversation_id or "").strip()
        safe_profile_id = str(profile_id or "").strip()
        if not safe_conversation_id:
            return {"status": "invalid", "reason": "conversation_id_required", "is_first": False}
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            if safe_profile_id:
                row = conn.execute(
                    "SELECT * FROM window_sessions WHERE conversation_id = ? AND profile_id = ?",
                    (safe_conversation_id, safe_profile_id),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM window_sessions WHERE conversation_id = ?",
                    (safe_conversation_id,),
                ).fetchone()
            if row is None:
                conn.rollback()
                return {"status": "invalid", "reason": "window_not_found", "is_first": False}
            observed_at = str(row["first_hook_prompt_at"] or "").strip()
            if observed_at:
                conn.commit()
                return {
                    "status": "existing",
                    "conversation_id": safe_conversation_id,
                    "is_first": False,
                    "observed_at": observed_at,
                }
            observed_at = self._utc_now()
            conn.execute(
                "UPDATE window_sessions SET first_hook_prompt_at = ?, updated_at = ? WHERE conversation_id = ?",
                (observed_at, observed_at, safe_conversation_id),
            )
            conn.commit()
            return {
                "status": "claimed",
                "conversation_id": safe_conversation_id,
                "is_first": True,
                "observed_at": observed_at,
            }
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def record_success(
        self,
        session_id: str,
        bucket_ids: list[str],
        completed_at: datetime | None = None,
    ) -> int:
        completed_at = completed_at or datetime.now()
        completed_iso = completed_at.isoformat(timespec="seconds")
        conn = self._connect()
        row = conn.execute(
            "SELECT COALESCE(MAX(round_id), 0) AS current_round FROM request_rounds WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        next_round = int(row["current_round"]) + 1
        conn.execute(
            "INSERT INTO request_rounds (session_id, round_id, completed_at) VALUES (?, ?, ?)",
            (session_id, next_round, completed_iso),
        )
        for bucket_id in bucket_ids:
            conn.execute(
                """
                INSERT OR REPLACE INTO injected_buckets
                (session_id, round_id, bucket_id, injected_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, next_round, bucket_id, completed_iso),
            )
        conn.commit()
        conn.close()
        return next_round

    def get_current_round(self, session_id: str) -> int:
        conn = self._connect()
        row = conn.execute(
            "SELECT COALESCE(MAX(round_id), 0) AS current_round FROM request_rounds WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        conn.close()
        return int(row["current_round"]) if row else 0

    def get_last_success_at(self, session_id: str) -> datetime | None:
        conn = self._connect()
        row = conn.execute(
            """
            SELECT completed_at
            FROM request_rounds
            WHERE session_id = ?
            ORDER BY round_id DESC
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
        conn.close()
        if not row:
            return None
        try:
            return datetime.fromisoformat(str(row["completed_at"]))
        except ValueError:
            return None

    def get_recent_bucket_ids(self, session_id: str, recent_rounds: int) -> set[str]:
        if recent_rounds <= 0:
            return set()
        conn = self._connect()
        current_round = self.get_current_round(session_id)
        if current_round <= 0:
            conn.close()
            return set()
        min_round = max(1, current_round - recent_rounds + 1)
        rows = conn.execute(
            """
            SELECT DISTINCT bucket_id
            FROM injected_buckets
            WHERE session_id = ? AND round_id >= ?
            """,
            (session_id, min_round),
        ).fetchall()
        conn.close()
        return {row["bucket_id"] for row in rows}

    def get_last_injected_at(self, session_id: str, bucket_id: str) -> datetime | None:
        conn = self._connect()
        row = conn.execute(
            """
            SELECT injected_at
            FROM injected_buckets
            WHERE session_id = ? AND bucket_id = ?
            ORDER BY injected_at DESC
            LIMIT 1
            """,
            (session_id, bucket_id),
        ).fetchone()
        conn.close()
        if not row:
            return None
        try:
            return datetime.fromisoformat(str(row["injected_at"]))
        except ValueError:
            return None

    def record_recent_context_injection(
        self,
        session_id: str,
        round_id: int,
        injected_at: datetime | None = None,
    ) -> None:
        injected_at = injected_at or datetime.now()
        conn = self._connect()
        conn.execute(
            """
            INSERT OR REPLACE INTO recent_context_injections
            (session_id, round_id, injected_at)
            VALUES (?, ?, ?)
            """,
            (session_id, int(round_id), injected_at.isoformat(timespec="seconds")),
        )
        conn.commit()
        conn.close()

    def get_last_recent_context_at(self, session_id: str) -> datetime | None:
        conn = self._connect()
        row = conn.execute(
            """
            SELECT injected_at
            FROM recent_context_injections
            WHERE session_id = ?
            ORDER BY injected_at DESC
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
        conn.close()
        if not row:
            return None
        try:
            return datetime.fromisoformat(str(row["injected_at"]))
        except ValueError:
            return None

    def record_injection_debug(
        self,
        session_id: str,
        round_id: int,
        payload: dict[str, Any],
        *,
        max_entries: int = 80,
    ) -> int:
        created_at = datetime.now().isoformat(timespec="seconds")
        body = json.dumps(payload, ensure_ascii=False)
        conn = self._connect()
        cursor = conn.execute(
            """
            INSERT INTO injection_debug (session_id, round_id, created_at, payload_json)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, int(round_id), created_at, body),
        )
        debug_id = int(cursor.lastrowid or 0)
        conn.execute(
            """
            DELETE FROM injection_debug
            WHERE id NOT IN (
                SELECT id FROM injection_debug ORDER BY id DESC LIMIT ?
            )
            """,
            (max(1, int(max_entries)),),
        )
        conn.commit()
        conn.close()
        return debug_id

    def record_conversation_turn(
        self,
        *,
        profile_id: str,
        session_id: str,
        round_id: int,
        user_text: str,
        assistant_text: str = "",
        model: str = "",
        client: str = "",
        route: str = "",
        created_at: datetime | None = None,
        max_entries: int = 500,
    ) -> int:
        created_at = created_at or datetime.now(timezone.utc)
        created_iso = created_at.isoformat(timespec="seconds")
        safe_profile_id = str(profile_id or "default").strip() or "default"
        safe_session_id = str(session_id or "default").strip() or "default"
        conn = self._connect()
        cursor = conn.execute(
            """
            INSERT OR REPLACE INTO conversation_turns
            (profile_id, session_id, round_id, created_at, user_text, assistant_text, model, client, route)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                safe_profile_id,
                safe_session_id,
                int(round_id),
                created_iso,
                str(user_text or ""),
                str(assistant_text or ""),
                str(model or ""),
                str(client or ""),
                str(route or ""),
            ),
        )
        if max_entries > 0:
            conn.execute(
                """
                DELETE FROM conversation_turns
                WHERE profile_id = ?
                  AND id NOT IN (
                    SELECT id FROM conversation_turns
                    WHERE profile_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                  )
                """,
                (safe_profile_id, safe_profile_id, max(1, int(max_entries))),
            )
        conn.commit()
        turn_id = int(cursor.lastrowid or 0)
        conn.close()
        return turn_id

    def list_recent_conversation_turns(
        self,
        *,
        profile_id: str,
        session_id: str | None = None,
        limit: int = 10,
        hours: float = 6.0,
    ) -> list[dict[str, Any]]:
        safe_limit = max(1, min(50, int(limit or 10)))
        safe_profile_id = str(profile_id or "default").strip() or "default"
        safe_session_id = str(session_id or "").strip()
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max(0.0, float(hours or 0)))
        conn = self._connect()
        where_clause = "profile_id = ? AND created_at >= ?"
        params: list[Any] = [safe_profile_id, cutoff.isoformat(timespec="seconds")]
        if safe_session_id:
            where_clause += " AND session_id = ?"
            params.append(safe_session_id)
        params.append(safe_limit)
        rows = conn.execute(
            f"""
            SELECT id, profile_id, session_id, round_id, created_at,
                   user_text, assistant_text, model, client, route
            FROM conversation_turns
            WHERE {where_clause}
            ORDER BY id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        conn.close()
        return [
            {
                "id": row["id"],
                "profile_id": row["profile_id"],
                "session_id": row["session_id"],
                "round_id": row["round_id"],
                "created_at": row["created_at"],
                "user_text": row["user_text"] or "",
                "assistant_text": row["assistant_text"] or "",
                "model": row["model"] or "",
                "client": row["client"] or "",
                "route": row["route"] or "",
            }
            for row in rows
        ]

    def list_conversation_turns_between(
        self,
        *,
        profile_id: str,
        start_at: datetime,
        end_at: datetime,
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        safe_limit = max(1, min(80, int(limit or 12)))
        safe_profile_id = str(profile_id or "default").strip() or "default"
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT id, profile_id, session_id, round_id, created_at,
                   user_text, assistant_text, model, client, route
            FROM conversation_turns
            WHERE profile_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (safe_profile_id, max(safe_limit, 500)),
        ).fetchall()
        conn.close()

        compare_tz = start_at.tzinfo or end_at.tzinfo

        def parse_local(value: Any) -> datetime | None:
            try:
                parsed = datetime.fromisoformat(str(value or ""))
            except ValueError:
                return None
            if compare_tz is None:
                return parsed.replace(tzinfo=None) if parsed.tzinfo is not None else parsed
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=compare_tz)
            return parsed.astimezone(compare_tz)

        start = start_at
        end = end_at
        if compare_tz is not None:
            if start.tzinfo is None:
                start = start.replace(tzinfo=compare_tz)
            else:
                start = start.astimezone(compare_tz)
            if end.tzinfo is None:
                end = end.replace(tzinfo=compare_tz)
            else:
                end = end.astimezone(compare_tz)
        elif start.tzinfo is not None:
            start = start.replace(tzinfo=None)
        elif end.tzinfo is not None:
            end = end.replace(tzinfo=None)

        filtered = []
        for row in rows:
            created = parse_local(row["created_at"])
            if created is None or not (start <= created < end):
                continue
            filtered.append(row)
            if len(filtered) >= safe_limit:
                break

        return [
            {
                "id": row["id"],
                "profile_id": row["profile_id"],
                "session_id": row["session_id"],
                "round_id": row["round_id"],
                "created_at": row["created_at"],
                "user_text": row["user_text"] or "",
                "assistant_text": row["assistant_text"] or "",
                "model": row["model"] or "",
                "client": row["client"] or "",
                "route": row["route"] or "",
            }
            for row in filtered
        ]

    def list_injection_debug(
        self,
        *,
        session_id: str = "",
        limit: int = 20,
        include_context: bool = True,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(100, int(limit)))
        conn = self._connect()
        if session_id:
            rows = conn.execute(
                """
                SELECT id, session_id, round_id, created_at, payload_json
                FROM injection_debug
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, session_id, round_id, created_at, payload_json
                FROM injection_debug
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        conn.close()

        items: list[dict[str, Any]] = []
        for row in rows:
            try:
                payload = json.loads(row["payload_json"])
            except json.JSONDecodeError:
                payload = {"raw": row["payload_json"]}
            if isinstance(payload, dict) and not include_context:
                payload = dict(payload)
                payload.pop("stable_context", None)
                payload.pop("dynamic_context", None)
            items.append(
                {
                    "id": row["id"],
                    "session_id": row["session_id"],
                    "round_id": row["round_id"],
                    "created_at": row["created_at"],
                    "payload": payload,
                }
            )
        return items

    def record_upstream_usage(
        self,
        *,
        session_id: str,
        round_id: int,
        model: str,
        route: str,
        usage: dict[str, Any],
        max_entries: int = 200,
    ) -> int:
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        safe_usage = dict(usage or {})
        prompt_tokens = safe_usage.get("prompt_tokens") or safe_usage.get("input_tokens")
        completion_tokens = safe_usage.get("completion_tokens") or safe_usage.get("output_tokens")
        prompt_details = safe_usage.get("prompt_tokens_details")
        cached_tokens = None
        if isinstance(prompt_details, dict):
            cached_tokens = prompt_details.get("cached_tokens")

        conn = self._connect()
        cursor = conn.execute(
            """
            INSERT INTO upstream_usage (
                session_id, round_id, created_at, model, route,
                prompt_tokens, completion_tokens,
                prompt_cache_hit_tokens, prompt_cache_miss_tokens,
                cached_tokens, cache_read_input_tokens, cache_creation_input_tokens,
                usage_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(session_id or "default"),
                int(round_id),
                created_at,
                str(model or ""),
                str(route or ""),
                _optional_int(prompt_tokens),
                _optional_int(completion_tokens),
                _optional_int(safe_usage.get("prompt_cache_hit_tokens")),
                _optional_int(safe_usage.get("prompt_cache_miss_tokens")),
                _optional_int(cached_tokens),
                _optional_int(safe_usage.get("cache_read_input_tokens")),
                _optional_int(safe_usage.get("cache_creation_input_tokens")),
                json.dumps(safe_usage, ensure_ascii=False),
            ),
        )
        usage_id = int(cursor.lastrowid or 0)
        if max_entries > 0:
            conn.execute(
                """
                DELETE FROM upstream_usage
                WHERE id NOT IN (
                    SELECT id FROM upstream_usage ORDER BY id DESC LIMIT ?
                )
                """,
                (max(1, int(max_entries)),),
            )
        conn.commit()
        conn.close()
        return usage_id

    def list_upstream_usage(
        self,
        *,
        session_id: str = "",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        safe_limit = max(1, min(100, int(limit or 20)))
        conn = self._connect()
        if session_id:
            rows = conn.execute(
                """
                SELECT id, session_id, round_id, created_at, model, route,
                       prompt_tokens, completion_tokens,
                       prompt_cache_hit_tokens, prompt_cache_miss_tokens,
                       cached_tokens, cache_read_input_tokens, cache_creation_input_tokens,
                       usage_json
                FROM upstream_usage
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, safe_limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, session_id, round_id, created_at, model, route,
                       prompt_tokens, completion_tokens,
                       prompt_cache_hit_tokens, prompt_cache_miss_tokens,
                       cached_tokens, cache_read_input_tokens, cache_creation_input_tokens,
                       usage_json
                FROM upstream_usage
                ORDER BY id DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        conn.close()

        items: list[dict[str, Any]] = []
        for row in rows:
            try:
                usage = json.loads(row["usage_json"] or "{}")
            except json.JSONDecodeError:
                usage = {}
            items.append(
                {
                    "id": row["id"],
                    "session_id": row["session_id"],
                    "round_id": row["round_id"],
                    "created_at": row["created_at"],
                    "model": row["model"] or "",
                    "route": row["route"] or "",
                    "prompt_tokens": row["prompt_tokens"],
                    "completion_tokens": row["completion_tokens"],
                    "prompt_cache_hit_tokens": row["prompt_cache_hit_tokens"],
                    "prompt_cache_miss_tokens": row["prompt_cache_miss_tokens"],
                    "cached_tokens": row["cached_tokens"],
                    "cache_read_input_tokens": row["cache_read_input_tokens"],
                    "cache_creation_input_tokens": row["cache_creation_input_tokens"],
                    "usage": usage,
                }
            )
        return items

    def get_cooldown_multiplier(
        self,
        session_id: str,
        bucket_id: str,
        cooldown_hours: float,
        cooldown_floor: float,
        now: datetime | None = None,
    ) -> float:
        if cooldown_hours <= 0:
            return 1.0
        now = now or datetime.now()
        last_injected = self.get_last_injected_at(session_id, bucket_id)
        if not last_injected:
            return 1.0
        elapsed_hours = max(0.0, (now - last_injected).total_seconds() / 3600)
        if elapsed_hours >= cooldown_hours:
            return 1.0
        progress = elapsed_hours / cooldown_hours
        return round(cooldown_floor + (1.0 - cooldown_floor) * progress, 4)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
