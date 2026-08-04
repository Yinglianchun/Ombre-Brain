# Retrieval-budget transplant map

This map records how to replay the retrieval-budget / facet / sentinel / ablation work without merging its old worktree. It is a migration guide, not deployment authorization.

## Locked baselines

- Target: `Haven/diary-backend-20260724` at `e817a81`, plus the local Recall Observation pagination patch built on that tip.
- Source: preserved worktree `D:\codex_worktrees\serein-recall-ablation-20260804` at `355465d`.
- The committed `gateway.py` is identical at `355465d` and `e817a81`. The old local retrieval-budget diff already called `_apply_semantic_scene_evidence_veto()`; the migration requirement is to preserve its existing order and rerun the contract tests, not to add a missing veto.
- The actual Gateway overlap is between the old worktree's uncommitted retrieval-budget diff and this worktree's uncommitted pagination diff. Vite, CSS, and Basement UI additionally overlap evidence binding and pagination work.
- Source pure-module fingerprint: `memory_recall/retrieval_budget.py` SHA-256 `D0EEE5A59AD079118EB63198A8751B968064D552489AF37B5A5AFCB9AE129727`.
- Never merge or cherry-pick the whole source worktree. Replay one block at a time and verify each block before continuing.

## Transplant classification

| Source material | Treatment on the target baseline | Required verification |
| --- | --- | --- |
| `memory_recall/retrieval_budget.py` | Copy byte-for-byte as the first isolated block. It is a pure module and does not import Gateway or UI code. Keep it simulation/shadow-only. | Pure function tests for facets, budget ordering, sentinel finalization, and absolute-floor partitioning. |
| Pure assertions in `scripts/verify_retrieval_budget_router.py` | Reuse as behavioral fixtures. Split pure-module assertions from Gateway integration assertions. | The pure suite must run before any Gateway wiring. |
| `normalize_recall_ablation_mode()` and `recall_ablation_debug_payload()` currently embedded in old `gateway.py` | Recreate as a small isolated helper or a narrow current-file addition; do not copy the old 482-line Gateway diff. | Invalid modes rejected; non-normal ablation accepted only for full debug simulation. |
| Old sentinel candidate expectations | Reuse as behavior: top 1–2, reuses route query vector, no graph expansion, no reranker, never injects. | Query-vector reuse, below-floor skip, over-floor fail-open, and unavailable-search fail-open tests. |
| Old `BasementPage.jsx`, `styles.css`, and `vite.config.mjs` edits | Rewrite against the current files. These files now contain newer evidence binding and Recall Observation pagination work. | Existing evidence, observation pagination/export, build, and Sites tests remain green. |
| Old Gateway integration assertions | Use as test-source only. Rewrite fixtures around the post-pagination files while preserving the already-existing evidence veto order. | Current evidence veto tests plus new simulation-only budget integration tests. |

## Gateway replay points after pagination

The current call order is authoritative. Re-entry must preserve this contract:

1. `handle_hook_recall()` parses an explicit simulation flag and a simulation-only ablation mode.
2. `semantic_recall_router.route_with_vector(query)` produces both route debug and the reusable query vector.
3. Build `retrieval_budget` from the current query planner and anchor planner. This is still a shadow decision.
4. Run the sentinel only for a high-confidence pure-chitchat prior. The sentinel may change budget/skip readiness but may not admit a Scene.
5. Compute the Router's tentative skip, then run the current `_apply_semantic_scene_evidence_veto()`. Budget wiring must not bypass, replace, or weaken this veto.
6. Resolve the simulation action: only pure-chitchat plus no structural/explicit anchor plus sentinel below rescue floor may become budget `skip=0`; every unavailable or ambiguous path fails open.
7. If retrieval continues, pass budget limits into current candidate generation. Apply the cheap absolute floor before any reranker eligibility decision.
8. Keep the configured production reranker state unchanged. Simulation may report `would_call`; it must not silently enable or call the production reranker.

Specific current functions to adapt, rather than replacing, are:

- `GatewayService.handle_hook_recall`
- `GatewayService._dynamic_bucket_candidate_items`
- `GatewayService._get_semantic_candidates`
- `GatewayService._rerank_scored_bucket_candidates`
- `GatewayService._apply_semantic_scene_evidence_veto`
- current recall debug formatters and `recall_why` summary builders

## UI and bridge rewrite points

- Rebuild Recall Simulation controls in the current `BasementPage.jsx`; do not copy the old component diff.
- Extend the current `/__serein/gateway/recall` bridge narrowly with `simulation: true` and validated ablation. Preserve the current evidence bridge and observation pagination routes in the same Vite config.
- Render `query_facets`, `surface_route`, `route_budget`, `anchor_override`, `effective_budget`, sentinel result, absolute-floor counts, and reranker-shadow eligibility from debug only.
- Candidate evidence must be decomposed as body semantic, cue semantic, cue lexical, title anchor, final admission source, and reranker shadow score. Until an independent cue index exists, `cue_semantic` is `unavailable`, not zero.
- Canonical authored Scene semantic scores are body-only. Legacy vector rows must be labeled legacy/unknown rather than presented as canonical body-only evidence.
- Keep the new Hook/Gateway pagination state, loaded-window export scope, stable review restoration, and privacy whitelist intact.

## Source assumptions to discard

- Old line numbers and function signatures are not migration instructions.
- A route top-1 result, route confidence, or route margin alone never authorizes a real-user-message skip.
- Structural marker lists are veto/facet aids, not an exhaustive chitchat phrase dictionary or a replacement for the frozen semantic prior.
- A date alone does not raise the budget to recall.
- Cue semantic similarity is candidate discovery only and cannot inject by itself.
- A generic `semantic` debug label is not enough; canonical body semantic and future cue semantic must remain distinguishable.
- `authored_cue_match` or title discovery must not accidentally become an unconditional final admission while applying the absolute floor.
- The Qwen cross-encoder returns a scalar relevance score only. It does not return relation/evidence/source-supported JSON.
- Missing raw evidence is `unknown`, not `unsupported`; legacy/manual Scenes must not be blanket-demoted.
- The old worktree's configured state says nothing about current production enablement. Reranker remains disabled until separately calibrated and authorized.
- Simulation success, local build success, and shadow metrics do not authorize commit, push, deployment, or production activation.

## Replay sequence and gates

1. **Pure module:** copy `retrieval_budget.py`; run pure tests. No Gateway/UI edits.
2. **Debug-only planner:** attach budget debug to explicit simulation requests; verify default Hook behavior is byte-for-byte equivalent at the response contract level.
3. **Sentinel:** reuse the route vector; verify no expansion, rerank, injection, or record write.
4. **Candidate budget and absolute floor:** adapt current candidate functions; preserve evidence veto and fail-open behavior.
5. **Ablation:** add normal / without-cues / without-embedding only for full debug simulation.
6. **Current UI rewrite:** render budget and source decomposition on top of the post-pagination Serein files.
7. **Reranker shadow visibility:** only after candidate/floor traces are available; keep production disabled.
8. **Calibration and release review:** grouped action labels and candidate hard negatives remain separate. Any deployment requires a new explicit authorization and the production clean-worktree release process.
