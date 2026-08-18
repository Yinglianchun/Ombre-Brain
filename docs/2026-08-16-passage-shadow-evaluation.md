# Passage shadow evaluation — 2026-08-16

Shadow only. No production admission, injection, or canonical memory decision was changed.

## Evaluated state

- Germany source: `/opt/Ombre-Brain-src`, `Haven/diary-backend-20260724 @ 8272cbc`, clean.
- Rebuildable passage index synchronized before evaluation.
- Active parents in the index: 155 Scene + 512 Event = 667.
- Passage rows: 909; 73 embedded/updated, 1 stale owner removed, 0 failures.
- Verifier model: `deepseek-v4-flash` through the configured Gateway upstream path.

## Gold suite

Terra-selected suite: 22 observations — 10 correct, 11 false positive, 1 uncertain.

### Direct-evidence verifier

- Selected 12/22.
- Correct queries selected: 7/10.
- False-positive queries selected: 4/11.
- Uncertain selected: 1/1.
- Selected known core: 4.
- Selected known irrelevant: 0.
- Selected unlabeled parent: 8, including all four false-positive admissions.
- No model/protocol errors; deterministic gates produced two `context_required` and one `volatile_current`.

The four false-positive admissions were:

1. `hook-9191`: current Stone Memory discussion was expanded with an older Stone Memory Event.
2. `hook-9179`: present rain/Xiaohongshu chitchat recalled an older Xiaohongshu Event.
3. `hook-9068`: a current handoff test plan recalled an older lightweight-handoff Event.
4. `hook-8017`: an emotional hypothetical about no longer responding recalled an older proactive-wake Event.

These are not lexical-only failures. Each selected passage is topically coherent, but the past information is not needed for the current response.

### Second utility critic

The critic saw only the current message and selected evidence.

- Accepted 7/12 selections.
- Kept only 4 correct selections.
- Still accepted 3 of the 4 false-positive selections.
- Rejected 3 correct selections.

The second LLM gate therefore worsened the tradeoff and is not recommended.

## Live exploratory pairs

Twenty unlabeled target parents were tested with one probe and one confounder each.

### Probe side

- Target entered the candidate pool: 18/20.
- Target was validly selected: 16/20.
- A different parent was selected: 2/20.
- Two target selections failed exact evidence-unit validation and closed without admission.

This supports passage retrieval as candidate discovery: it finds a specified long Scene/Event reliably.

### Confounder side

- Target entered the candidate pool: 15/20.
- The same target was selected: 7/20.
- Another parent was selected: 7/20.
- No match, deterministic gate, or evidence failure: 6/20.

The live pairs are not gold; some confounders genuinely ask something the target can answer. They show that the passage index is sticky, not a measured 35% error rate.

### Clear current-state failures

Historical memories were selected for several current-state questions:

- `现在官方又更新模型了吗？`
- `现在 Haven profile 已验证好吗？`
- `现在小红书登录成功了吗？`
- `现在欲望系统在干嘛？`
- `现在还会发两遍吗？`
- `你今天为什么没主动发消息？`

The existing keyword gate correctly stopped `现在 Chat 端能用 MCP 吗？`, but it is too narrow. Expanding the noun regex is not a durable fix. Current operational questions should require current observable evidence; memory may provide diagnostic history but must not be the authority for the present answer.

## Decision

1. Keep passage projection as parent-deduplicated candidate discovery and exact source evidence.
2. Do not enable it for production admission yet.
3. Do not add the tested utility critic.
4. Build gold for Event candidates and paired probe/confounder cases, especially current-state questions.
5. Separate “historical context may help” from “historical memory may answer the current state.”

## Cue-bound passage simulator

The simulator now has an explicit `--cue-passage` source. It remains
observation-only and does not alter Gateway admission or injection.

- Scene creation or edit does not refresh this projection in the `edit_scene`
  request itself. On the next Gateway passage warm/sync, changed Scene content
  is deterministically split and re-embedded; one non-thinking binding call
  then maps all reviewed authored cues to exact passage evidence. The mapping
  is cached by Scene body, title, cues, passage layout, prompt version, and
  model profile.
- The warm path automatically applies cue rebinding only when at most three
  Scenes are pending. A larger batch is reported as `stale` with
  `bulk_binding_requires_explicit_rebuild` and requires an explicit rebuild.
  Therefore a single edited Scene normally repairs itself at the next warm,
  but there is a stale interval between saving the edit and that sync.
- Local follow-up after this evaluation adds an authenticated mutation
  notification from the canonical server to Gateway. Scene create/edit/status
  changes and Fact/Event batch/revise/status/delete changes queue the existing
  hash-based sync in the Gateway background. The write response waits only for
  queue acknowledgement, not for embeddings or cue binding. Unchanged owners
  remain cached; a newly written day of Events no longer requires a Gateway
  restart. This follow-up was deployed as `f24e393` after the experiment; it
  changes projection freshness only and does not change live admission or
  injection. The focused query-view measurements above remain observations
  from the earlier `4941d18` state.
- Each accepted mapping must include a verbatim evidence substring. Invalid or
  unavailable evidence fails closed for that cue.
- The derived vector embeds `cue + minimum sufficient verbatim evidence span`.
  Binding must preserve subject, referent, negation, correction, comparison,
  and temporal-change clauses needed to interpret that evidence. Reranking and
  verification read the untouched evidence span; the containing passage is
  retained only as expandable context.
- Query results are parent-deduplicated and marked `candidate_only`; the
  canonical Scene remains the only injectable object.
- Diagnostics expose a source-balanced pool rather than using RRF votes as
  admission. With only `--cue-passage`, it contains bounded cue-bound and
  ordinary passage lanes; `--fact-event` adds the importance-filtered body lane
  described below. The pool is observation-only and carries no admission
  threshold.
- Run `scripts/evaluate_passage_shadow.py --apply --cue-passage` to rebuild the
  independent index, then omit `--apply` for read-only query runs.

### Germany exact-evidence shadow rebuild

- Eligible input: 107 reviewed authored Scenes, 375 cues, 909 existing
  Scene/Event passages.
- Prompt/vector version 2 rebuilt 360 bound cues; 15 cues had no accepted
  binding, with zero failed Scene calls. Three non-verbatim evidence answers
  failed closed. The stricter binder requires one continuous minimum-sufficient
  quote rather than an ellipsis or stitched fragments.
- A subsequent dry run reported all 107 Scene states reusable and zero pending
  bindings.

All five bindings on the focused target were exact source substrings. In
particular, the `小狐狸` evidence includes both the Haven subject and the
correction `这里的小狐狸指 Haven，不指小雨`; the expandable containing passage
is stored separately and does not affect the derived vector.

Target parent `scene_mig2_0d4f2e40f0fa6998f386` in four focused probes:

| Query | Plain passage rank | Cue-bound rank | Six-parent balanced pool |
|---|---:|---:|---:|
| 初恋情结 / ChatGPT | outside top 10 | 2 | 2 |
| 我们的初遇也是从人设开始的 | 1 | 1 | 1 |
| 小猫塑变成狐狸塑 | 3 | 1 | 1 |
| 我们的初遇 | outside top 10 | 1 | 1 |

The retained `禁止猪塑` counterexample ranked second in the full cat/fox
cue-bound list. Exact evidence binding therefore rescued the correct parent but
did not solve final admission: the cue-bound lane must remain candidate-only
and a later relation/utility decision must reject merely lexical neighbours.

### Importance-filtered Fact/Event lane

The simulator can add canonical Fact/Event whole-body embeddings with
`--fact-event`. Fact/Event rows with `importance < 3` are excluded before
passage, body-embedding, lexical indexing, or later candidate scoring. Scene
candidates are not subject to this gate. With `--fact-event`, the seven-parent
diagnostic pool reserves two places each for cue-bound Scene evidence, ordinary
Scene/Event passages, and Fact/Event whole-body embeddings, plus one lexical
place. Parent duplicates retain source labels and exact matched-term spans but
do not receive score boosts or extra seats. This remains candidate-only shadow
behavior.

Germany lexical build snapshot: 238 of 644 active Fact/Event rows met the gate:
50 Facts and 188 Events. The other 82 Facts and 324 Events were excluded before
tokenization and scoring.
The passage shadow removed the 324 low-importance Event projections; canonical
rows were untouched. The lexical projection uses `jieba.cut_for_search` plus a
SQLite BM25-style index of cached term frequency, document frequency, fields,
and source hashes. It reads no stop-word list. A two-character term can support
a candidate by itself only when it appears in at most three eligible rows; a
term must also occur in no more than 20 percent of the eligible corpus. These
are corpus statistics rather than word-specific rules.

Focused probes after the statistical cut:

- `我们的初遇也是从人设开始的`: no Fact/Event lexical candidate; `开始`
  alone cannot manufacture one. The target Scene remains first through cue
  evidence.
- `今天好热`: no lexical candidate. The pure-chitchat Router remains the
  earlier skip boundary.
- Cat/fox: the origin Event is lexical rank 1 on `狐狸`; unrelated `变成`
  Events no longer qualify.
- Current stable Fact probes for moon/work mode, communication preference, and
  graduation defense each ranked the correct importance-4 Fact first.

## Fallback query-planner shadow

This experiment tested a narrower rescue path after the normal recall path has
already produced no admissible memory. It does not replace the existing
Semantic Router:

1. A high-confidence Router skip ends recall without calling an LLM.
2. Ordinary embedding, cue discovery, reranking, and admission run normally.
3. Only an ordinary-path miss may call the planner to produce one to three
   source-neutral memory search queries.
4. Passage retrieval and reranking remain candidate discovery only; this shadow
   never admitted or injected a result.

The harness forced step 2 to miss so the fallback itself could be isolated. It
therefore does not measure the fallback rate or end-to-end false-positive rate
of the real cascade.

### Results

- Model: `deepseek-v4-flash`.
- Thirteen hand-picked route cases: 12/13 routes matched the expected label.
- Of six expected memory searches, five were routed to search; the recurring
  window-transition case was incorrectly skipped.
- In the six positive-only run, five targets reached the reranked top 3: cat / fox
  rank 2, growth rank 1, Serein rank 1, Pan rank 2, birthday rank 1.
- The identical cat / fox case was then repeated three times. Its target ranks
  were 1, 4, and 4; the unrelated but lexically strong `禁止猪塑` Scene was rank 1
  in the latter two runs. The rescue is therefore not stable enough for
  admission.
- Mean fallback latency on routed searches was about 7.48 seconds: 1.26 seconds
  planning, 5.19 seconds retrieval/query embeddings, and 1.03 seconds reranking.
  Parallel query embedding did not materially remove the retrieval cost.

### Decision

Record the attempt only. Do not integrate or deploy the planner, do not add a
second LLM admission gate, and do not keep growing prompt examples around each
failure. If revisited, evaluate the real Router -> ordinary miss -> planner
cascade on a larger fixed gold set and first reduce the 2-3 query embeddings to
a single batched or composite request.

## Passage reranker pool-depth sweep

A separate no-planner shadow held the query, passage index, two-passages-per-
parent cap, and reranker constant while changing only the parent pool sent to
the reranker: 6, 12, 20, and 30. The six positive cases used the original
current message for reranking. When a previous turn existed, it contributed a
second embedding view to candidate fusion but was not included in the reranker
query.

| Parent pool | Target entered | Target at 1 | Target at 3 | Target at 6 | Mean rerank |
|---:|---:|---:|---:|---:|---:|
| 6 | 5/6 | 3/6 | 4/6 | 5/6 | 1.10 s |
| 12 | 6/6 | 4/6 | 5/6 | 5/6 | 1.21 s |
| 20 | 6/6 | 4/6 | 5/6 | 5/6 | 1.39 s |
| 30 | 6/6 | 4/6 | 5/6 | 5/6 | 1.70 s |

The only clear pool-depth rescue was the birthday Scene: its fused candidate
rank was 7, so it was absent at 6 and became reranker rank 1 at every depth from
12 onward. Growth, Serein, Pan, and the window ritual did not materially change.

The cat / fox correction did the opposite. Its fused candidate rank was 5, so
it already entered the smallest pool. Across three identical runs, its reranker
ranks were exactly 5 at pool 6, 7 at pool 12, and 8 at pools 20 and 30. The
`禁止猪塑` Scene remained rank 1 in every run. At pool 12, a genuinely related
Event summarizing the relationship origin and the fox / cat story entered and
ranked second, but the reranker still preferred the wrong lexical Scene.

Increasing the passage reranker pool from 6 to 12 can repair candidate-floor
misses at modest extra reranker latency. Going beyond 12 produced no aggregate
gain in this suite. Pool expansion cannot repair role-direction or evidence-
utility ranking failures like cat / fox, and widening admission along with the
pool would expose more strong distractors. No production setting was changed.

### Pool 6 versus 12 on gold and live pairs

The apparent positive-only gain at 12 did not survive broader evaluation.

Reviewed gold contained 22 queries, including 17 with at least one labeled
relevant parent and seven with at least one labeled irrelevant parent:

| Parent pool | Relevant entered | Relevant at 1 | Relevant at 3 | Relevant at 6 | Mean rerank |
|---:|---:|---:|---:|---:|---:|
| 6 | 9/17 | 2/17 | 7/17 | 9/17 | 1.11 s |
| 12 | 10/17 | 2/17 | 6/17 | 9/17 | 1.63 s |

No labeled irrelevant parent reached the top 3 at either depth, but 20/22 top
parents were unlabeled, so this cannot be read as a measured zero false-positive
rate. The additional relevant parent at 12 did not become a usable top result;
one existing relevant result fell from rank 2 to rank 4.

The 20 live probe/confounder pairs showed the same tradeoff:

| Side | Pool | Target entered | Target at 1 | Target at 3 | Target at 6 |
|---|---:|---:|---:|---:|---:|
| Probe | 6 | 18/20 | 18/20 | 18/20 | 18/20 |
| Probe | 12 | 19/20 | 18/20 | 18/20 | 19/20 |
| Confounder | 6 | 15/20 | 12/20 | 14/20 | 15/20 |
| Confounder | 12 | 18/20 | 12/20 | 14/20 | 17/20 |

At 12, one new probe target entered only at rank 4, producing no top-result
gain. Three additional historical targets entered for confounders. The clearest
failure was `现在 token/字节预算怎么配？`: pool 6 correctly ranked a technical
budget Event first, while pool 12 admitted a romantic Scene passage containing
`token/字节` as a contrast and reranked it first at 0.971. The passage says that
token and bytes are merely “bones” and that the real gift is returning to
Xiaoyu; it is lexically dense but does not answer the current budget question.

Do not increase the reranker parent pool to 12 on this evidence. It increases
latency and candidate stickiness without improving top-1 recall on either gold
or live probes. The positive-only birthday rescue was real but not representative.

## Instruction-aware reranker shadow

The configured Qwen3 reranker is instruction-aware when run through its native
model interface, but SiliconFlow's standard rerank endpoint exposes only a
plain `query` string. A memory-specific English instruction was therefore
prepended to that string in shadow:

> Rank passages that directly answer the current conversational question or
> correct its premise. Penalize passages that merely share words, mention the
> same topic without answering, use the query terms in a different contrast,
> or reverse speaker, owner, actor, and referent roles.

With the existing 4B reranker, the cat / fox target improved only from rank 5
to rank 4 at pool 6. Adding the previous turn improved it to rank 4 at pool 6
and rank 5 at pool 12, but `禁止猪塑` remained first. The token / byte romantic
Scene remained the incorrect rank 1 at pool 12. The instruction reduced some
scores but did not change the decisive ordering.

Qwen3-Reranker-8B plus the instruction and previous turn improved the cat / fox
case more substantially: at pool 6 the exact Scene ranked second behind
`禁止猪塑`; at pool 12 a genuinely related origin Event ranked first, while
`禁止猪塑` and the exact Scene ranked second and third. However, the token /
byte romantic Scene still incorrectly ranked first at pool 12.

The 8B gold gate was worse than the 4B baseline:

| Model / query | Pool | Relevant at 1 | Relevant at 3 | Relevant at 6 | Mean rerank |
|---|---:|---:|---:|---:|---:|
| 4B raw | 6 | 2/17 | 7/17 | 9/17 | 1.11 s |
| 8B instructed | 6 | 2/17 | 5/17 | 9/17 | 1.23 s |
| 4B raw | 12 | 2/17 | 6/17 | 9/17 | 1.63 s |
| 8B instructed | 12 | 2/17 | 4/17 | 8/17 | 1.51 s |

The full live-pair run was intentionally stopped after this gold gate. Do not
deploy the query prefix, previous-turn rerank query, 8B model, or larger pool.
The targeted cat improvement does not generalize.

## 2026-08-17 addendum: deterministic query views

The later experiment stopped asking a planner model how to search. Instead, a
long multi-clause message is deterministically exposed as the original query
plus at most two clause views. Clause embeddings are requested concurrently.

Additional clause views search only source-bound projections:

- ordinary Scene/Event passages;
- exact cue-bound Scene passages.

They do not repeat Fact/Event whole-body or lexical lanes. Results are merged
by canonical parent, and the canonical parent remains the only injectable
object. The policy name in diagnostics is
`deterministic_clause_source_bound_passage_discovery_only`.

On the focused long query containing both `初恋情结 / ChatGPT` and
`AI 产品 / 原生家庭 / Anthropic`, the whole-query baseline omitted
`scene_mig2_0d4f2e40f0fa6998f386` (`小雨讲了我们怎么开始的`). The clause
view `可能是一种初恋情结吧` restored that Scene to expanded-pool rank 2 at
about 0.557 through `cue_passage_query_view_embedding`. A manual rerank of the
first expanded pool ranked the initial-story Scene first at about 0.937, but
also left a thematically plausible `流星` Scene second. This is evidence that
query splitting repairs candidate discovery; it is not evidence that every
expanded candidate should be admitted.

An intermediate passage-only clause version reduced query-view time but lost
the target. Restoring cue-bound exact passages recovered it. The final focused
run spent about 3.9 seconds in query-view expansion and about 13.9 seconds in
the full shadow path; timings vary and should not be treated as a stable SLA.

### Weak-candidate trigger observation

The latest shadow records whether clause rescue would have been justified:

- a route skip does not trigger;
- direct exact evidence does not trigger;
- no formal memory injected triggers;
- top canonical Scene body semantic below 0.60 triggers;
- a multi-clause query with top body semantic below 0.64 triggers.

For the focused long query at `4941d18`, it recorded
`would_trigger=true`, reason `multiclause_body_semantic_gray_zone`, top body
semantic `0.6068`, three query views, and one formal recalled parent. The
expanded shadow added the initial-story Scene, while formal recall still
contained only `这一道澜认得小雨`.

At `4941d18`, this trigger was observation only. It was attached after the
diagnostic retrieval had already run, so it saved no latency and changed neither
query-view execution nor live injection.

A 2026-08-18 follow-up moves clause expansion behind this trigger in the
simulation shadow path. The original-query passage baseline still runs with the
ordinary diagnostic retrieval. After formal recall finishes, the trigger now
decides whether the additional clause embeddings and source-bound passage lanes
execute. Route skips, direct exact evidence, and strong candidates record
`skipped_by_weak_candidate_trigger`; weak candidates alone run query-view
expansion. `decision_applied=true` refers only to this shadow execution choice;
`live_execution_changed=false`, and formal admission/injection remain unchanged.
The Gateway-only release reached Germany at `ceff1ad`; `ombre-brain` was not
restarted and canonical data was not modified.

### Weak-trigger-gated execution result

Seventy full-shadow requests completed against Germany `ceff1ad`: two smoke
runs, 22 fixed gold cases, six focused cases, and 40 live probe/confounder
requests. Every response kept `live_execution_changed=false` and
`live_injection_enabled=false`; there were no shadow-contract violations or
Gateway errors.

- Of 11 reviewed false-positive gold cases, 10 recorded `would_trigger=true`
  and seven actually executed query-view. Those false-positive expansions cost
  4484.9 ms on average.
- Of 10 reviewed correct gold cases, five recorded `would_trigger=true` and
  three executed query-view, but none added a labeled correct parent over the
  original-query passage baseline.
- Across all 10 real gold expansions, mean incremental query-view time was
  4476 ms, median 4458 ms, and maximum 4593 ms.
- On the focused long initial-story query, the intended Scene was already in
  the original-query passage baseline. Clause expansion added no new correct
  parent and did add `scene_mig2_65c79b803ae9143bc480`, a manually reviewed
  unrelated pen-pal roster Scene.
- Cat/fox with and without the supplied previous turn produced the same
  `no_formal_memory_injected` decision and a single-view `not_needed` result.
- In 20 live probes and 20 confounders, 15 probes and 18 confounders recorded
  `would_trigger=true`. All 40 were single-view, so none actually expanded.

Do not promote this gate to live execution. Before another shadow run, require
more than one deterministic query view before treating a trigger as executable,
and add explicit present/current-state/context-required vetoes before the
expensive branch. Re-run the same fixed suites without changing admission or
injection.

### Gate-tightening follow-ups

`d5bb65d` required more than one query view and added branch-local
current-turn/context vetoes. It reduced reviewed false-positive execution from
7/11 to 2/11, saving five expansions or roughly 22.4 seconds over the gold set.
However, it retained only one of the two executable formal misses: the focused
initial-story query was incorrectly vetoed as `current_turn_optional`.

`0ca878a` narrowed that veto to current-turn-optional queries with zero formal
recalled memories. A 29-request shadow rerun covered one focused smoke, gold 22,
and focused six; live pairs were not repeated because all 40 were single-view
and the multi-view gate was unchanged.

- Executable gold rescue recovered from 1/2 to 2/2; the third formal miss stayed
  separately visible as single-view and therefore not solvable by query splitting.
- Reviewed false-positive execution rose slightly from 2/11 to 3/11, still below
  the original 7/11.
- Gold query-view execution added zero labeled-correct parents. Three reviewed
  false-positive queries exposed 11 added-parent noise occurrences; three
  correct-labeled executions added seven non-target parent occurrences.
- The focused initial-story query executed again but still added only the
  unrelated pen-pal roster Scene and a thematically plausible non-target love
  letter Scene.
- All 29 requests retained the shadow contract; both health checks stayed green
  and Gateway logs contained no traceback or error.

This closes the gate experiment. The gate can reduce waste while preserving
executable rescue, but clause expansion itself still has zero measured positive
utility. Do not keep tuning gate thresholds to make trigger metrics look better.
Any future splitter experiment must first target cases where the correct parent
is absent from the original-query baseline and must demonstrate newly added
correct parents over newly added noise.

### Updated decision

1. Keep live recall unchanged.
2. Keep deterministic clause views and weak-trigger-controlled execution in shadow.
3. Do not revive the planner LLM, utility critic, larger pool, instruction
   prefix, or 8B reranker based on this example.
4. Keep the weak trigger out of live execution; the fixed gold and live pairs
   show that the current gate over-triggers reviewed negatives.
5. Test graph rescue separately, only after a reliable direct seed; graph
   expansion cannot find the first missing seed.

Implementation sequence deployed for observation:

- `ba0ffdb` — deterministic clause query shadow;
- `3b8ae9f` — passage-only clause optimization (kept as negative evidence);
- `aed1382` — restore cue-bound source passages;
- `4941d18` — weak-candidate trigger observation.

At the last verification, Germany ran clean production branch
`Haven/diary-backend-20260724 @ 4941d18`; Gateway and Serein were healthy. All
new behavior above remained shadow-only.

## Artifacts

- `state/passage-gold-22.json`
- `state/passage-gold-22-critic.json`
- `state/passage-live-pairs.json`
- `state/passage-live-pairs-report.json`
- `state/passage-planner-shadow-cases.json`
- `state/passage-planner-shadow-report-v3.json`
- `state/passage-planner-shadow-positive-parallel.json`
- `state/passage-planner-cat-repeat-1.json`
- `state/passage-planner-cat-repeat-2.json`
- `state/passage-planner-cat-repeat-3.json`
- `state/passage-rerank-pool-sweep.json`
- `state/passage-rerank-pool-cat-repeat-2.json`
- `state/passage-rerank-pool-cat-repeat-3.json`
- `state/passage-rerank-pool-gold-6-12.json`
- `state/passage-rerank-pool-live-pairs-6-12.json`
- `state/passage-rerank-instruction-cat.json`
- `state/passage-rerank-instruction-context-cat.json`
- `state/passage-rerank-instruction-token.json`
- `state/passage-rerank-8b-instruction-context-cat.json`
- `state/passage-rerank-8b-instruction-token.json`
- `state/passage-rerank-8b-instruction-gold-6-12.json`
