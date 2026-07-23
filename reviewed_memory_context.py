from __future__ import annotations

import re
from typing import Any, Callable, Mapping

from utils import count_tokens_approx


class ReviewedMemoryContextProvider:
    """Consume reviewed Narrative Roll and Portrait heads for one query.

    The provider does not retrieve ordinary Scene candidates or assemble the
    final prompt.  It only applies the reviewed-derived-object visibility and
    whole-body budget rules that must not live in Gateway transport code.
    """

    def __init__(
        self,
        *,
        narrative_store_getter: Callable[[], Any],
        portrait_store_getter: Callable[[], Any],
        identity: Mapping[str, Any],
        budget_getter: Callable[[], int],
    ) -> None:
        self._narrative_store_getter = narrative_store_getter
        self._portrait_store_getter = portrait_store_getter
        self.identity = identity
        self._budget_getter = budget_getter

    def narrative_debug(self, query: str) -> dict[str, Any]:
        return self._narrative_store_getter().shadow_match(query, [])

    def prepare_narrative(
        self,
        query: str,
        direct_first_hop_scene_ids: list[str],
    ) -> tuple[str, str, dict[str, Any]]:
        body, compact_index, debug = self._narrative_store_getter().prepare_injection(
            query,
            direct_first_hop_scene_ids,
        )
        if not body:
            return "", "", debug

        body_section_tokens = count_tokens_approx(f"Narrative Roll\n{body}")
        if body_section_tokens <= max(1, int(self._budget_getter())):
            debug.update(
                {
                    "status": "injected",
                    "visible_injection": True,
                    "injection_mode": "body",
                    "injected_narrative_ids": [
                        str(debug.get("admitted_narrative_id") or "")
                    ],
                    "injected_tokens": body_section_tokens,
                }
            )
            return body, "Narrative Roll", debug

        index_section_tokens = count_tokens_approx(
            f"Narrative Roll Index\n{compact_index}"
        )
        debug.update(
            {
                "status": "index_injected",
                "visible_injection": bool(compact_index),
                "injection_mode": "index_only",
                "reason": "full_body_exceeds_total_injection_budget",
                "injected_narrative_ids": (
                    [str(debug.get("admitted_narrative_id") or "")]
                    if compact_index
                    else []
                ),
                "injected_tokens": index_section_tokens if compact_index else 0,
            }
        )
        return compact_index, "Narrative Roll Index", debug

    def explicit_portrait_scopes(self, query: str) -> list[str]:
        compact = self._compact(query)
        if not compact:
            return []
        user_subjects = list(
            dict.fromkeys(
                self._compact(value)
                for value in (
                    "我",
                    self.identity.get("user_display_name"),
                    self.identity.get("user_name"),
                    *(self.identity.get("user_aliases") or []),
                )
                if self._compact(value)
            )
        )
        user_subject_pattern = "|".join(
            re.escape(value) for value in sorted(user_subjects, key=len, reverse=True)
        )
        user_requested = any(
            marker in compact
            for marker in (
                "我的画像",
                "用户画像",
                "你眼里的我",
                "你眼中的我",
                "你怎么看我",
                "userportrait",
                *(f"{subject}画像" for subject in user_subjects if subject != "我"),
            )
        ) or bool(
            user_subject_pattern
            and re.search(
                rf"(?:{user_subject_pattern})(?:现在|到底|在你眼里|在你眼中)?是(?:个|一个)?什么样(?:的)?人?",
                compact,
            )
        )
        relationship_requested = any(
            marker in compact
            for marker in (
                "关系画像",
                "你怎么看我们的关系",
                "你怎么看我俩的关系",
                "relationshipportrait",
            )
        ) or bool(
            re.search(
                r"(?:我们|我俩)(?:现在|到底)?是(?:什么|怎样|怎么)(?:样的)?关系",
                compact,
            )
            or re.search(
                r"(?:我们|我俩)的关系(?:现在)?(?:是什么样|怎么样|如何|是什么)",
                compact,
            )
        )
        return [
            scope
            for scope, requested in (
                ("user", user_requested),
                ("relationship", relationship_requested),
            )
            if requested
        ]

    def prepare_portrait(self, query: str) -> tuple[str, dict[str, Any]]:
        requested = self.explicit_portrait_scopes(query)
        debug: dict[str, Any] = {
            "status": "skipped",
            "reason": "query_not_explicit_portrait",
            "requested_scopes": requested,
            "injected_scopes": [],
        }
        if not requested:
            return "", debug
        result = self._portrait_store_getter().read_reviewed_portrait()
        if str(result.get("status") or "") != "ok":
            debug.update(
                {"status": "unavailable", "reason": "portrait_state_unavailable"}
            )
            return "", debug
        sections = []
        unavailable = []
        scope_rows = result.get("scopes", {}) if isinstance(result.get("scopes"), dict) else {}
        for scope in requested:
            row = scope_rows.get(scope, {}) if isinstance(scope_rows.get(scope), dict) else {}
            if not bool(row.get("published")):
                unavailable.append(
                    {
                        "scope": scope,
                        "source": str(row.get("source") or ""),
                        "reason": (
                            "evidence_invalid"
                            if row.get("evidence_invalid")
                            else "no_reviewed_manual_head"
                        ),
                    }
                )
                continue
            stable = str(row.get("stable") or "").strip()
            if not stable:
                unavailable.append({"scope": scope, "reason": "empty_head"})
                continue
            title = "Full User Portrait" if scope == "user" else "Full Relationship Portrait"
            sections.append(f"{title}\n{stable}")
            debug["injected_scopes"].append(scope)
        context = "\n\n".join(sections)
        debug.update(
            {
                "status": "injected" if context else "unavailable",
                "reason": "explicit_reviewed_head" if context else "no_reviewed_manual_head",
                "unavailable_scopes": unavailable,
                "context_chars": len(context),
            }
        )
        return context, debug

    @staticmethod
    def _compact(value: object) -> str:
        return re.sub(
            r"[^0-9a-z\u4e00-\u9fff]+",
            "",
            str(value or "").strip().lower(),
        )
