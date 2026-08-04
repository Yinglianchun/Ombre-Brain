# Scene evidence picker design QA

Date: 2026-08-04

Reference: `C:\Users\86188\AppData\Local\Temp\codex-clipboard-a54209c2-f4d6-48d4-9041-bcc7089c27e9.png`

## Visual comparison

- Kept the existing Serein typography, borders, colors, and editorial density.
- Increased the gap between source evidence and memory annotations from the cramped baseline to a measured 38 px.
- Search controls and locate actions use the existing Phosphor icon set and button treatment.

## Interaction checks

- Keyword search returned real Haven Bridge source messages and preserved pagination.
- Exact `#7675` lookup returned message 7675 inside its original session, with six visible messages before and after it.
- `定位原文` opened the original session timeline, highlighted message 7675, and centered it in the scroll area.
- Keyword results kept their search ordering; `看前后` moved a hit into its original-session context and `返回搜索结果` restored the same query.
- An already-bound source rendered as a checked, enabled checkbox with `已绑定 · 取消勾选可解绑`.
- Reversible unbind and rebind passed the SQLite contract tests; a production-safe invalid unbind probe returned HTTP 400 without changing evidence.
- Existing production evidence remained 26/26 active after the sidecar migration.

## Browser QA

- Verified in the Codex in-app browser at desktop width.
- Confirmed the evidence-to-annotation gap from DOM geometry: 38 px.
- Confirmed the search field, contextual locate result, bound checkbox state, and responsive layout render without overlap or clipping.
- Confirmed target centering from DOM geometry (`centerDelta = 0`), 13 same-session messages for a 6 + target + 6 context window, and no console errors.

final result: passed
