# Event / Scene typed recall and entity scope shadow

Status: local implementation only. No deploy, canonical mutation, Arc mutation,
cursor advance, live injection, raw query route, or model installation.

## Candidate contract

- Every active Event is eligible for the Event shadow lane. Event importance is
  retained as diagnostics but is not an eligibility or scoring gate.
- Scene and Event each have two embedding routes: whole-object and, only for a
  long object that splits into multiple spans, passage embedding. An owner's
  score is `max(whole, passage)`; the scores are never added. Scores are ordered
  only inside their object lane, and the flattened debug pool does not compare
  Event cosine directly with Scene cosine.
- Scene cues add a Scene owner to the candidate lane. Cue similarity cannot
  replace passage similarity, cannot add to the score, and cue-bound passage
  text is stripped from cue-only candidates.
- Event passage text is derived only from Event body text. Event/Scene objects
  with at most 200 body characters use only their existing whole embedding.
  Longer objects enter the deterministic passage splitter, and still remain
  whole-only when the splitter yields at most one span. No duplicate one-passage
  vector is stored.
- Fact is not part of this new typed recall path. Existing Fact storage and
  compatibility APIs are unchanged, but the shadow pool contains only Event
  and Scene owners.
- Freshness is a bounded within-lane rerank feature. The original embedding
  score remains unchanged; ordinary queries receive at most a `0.015` prior,
  while explicit recent-language queries receive at most `0.05`. Freshness is
  never compared across Event and Scene lanes and cannot admit a candidate.

Gateway startup warm only plans passage and observed-entity changes. It reports
the deltas but starts neither passage embedding nor entity sidecar writes.
Canonical mutation refreshes may apply a bounded passage delta (three passages
by default) and the matching fingerprint-based entity delta; larger passage
deltas remain stale until an explicit backfill.

The explicit backfill command is plan-only unless `--apply` is present:

```powershell
C:\Python313\python.exe scripts\backfill_passage_shadow.py
C:\Python313\python.exe scripts\backfill_passage_shadow.py --apply
```

Apply mode uses the lower backfill concurrency and request delay from
`passage_shadow`. It builds in a temporary SQLite file beside the live sidecar.
Any failed owner discards the temporary file and preserves the current index;
only a fully successful build is atomically activated.

## Entity sidecar contract

`observed_entity_shadow.sqlite` is a rebuildable, incrementally synchronized
sidecar and stores:

- repeated or explicit entity surfaces from exact bound Event/Scene sources;
- source IDs, exact entity spans, counts, and extraction basis, but not a
  queryable copy of the source transcript;
- Arc-level entity aggregation from already-confirmed direct members;
- owner-to-Arc link candidates with receipts;
- deterministic scope anchors.

Each Event/Scene owner has a source fingerprint. An unchanged warm reuses its
stored observations without rerunning entity extraction. Changed and new owners
are extracted individually; removed owners are deleted individually. Arc
aggregates, scope anchors, and link candidates are reconciled by row diff, so an
unchanged warm performs no derived inserts, updates, or deletes. Arc membership
changes recompute only the lightweight derived rows. A change to the authored
Arc entity vocabulary invalidates owner extraction because it changes which
single-occurrence known titles are admissible.

The standalone synchronizer is plan-only unless `--apply` is present:

```powershell
C:\Python313\python.exe scripts\backfill_observed_entity_shadow.py
C:\Python313\python.exe scripts\backfill_observed_entity_shadow.py --apply
```

Apply mode writes only `observed_entity_shadow.sqlite`. It reads exact bound
source snapshots for extraction but never exposes them as a query-time route.

Observed entities can propose an Arc link but never admit or write one. Arc
aggregation excludes the candidate owner when producing the
`repeated_observed_entity` signal, preventing a fragment from proving its own
membership.

Titles and title aliases are scope anchors. Primary/supporting entities are
anchors only when their Arc mapping is unique. Derived observed entities need
support from at least two confirmed Arc members and a unique Arc mapping.
Only entity-shaped observations can become a derived scope or cross-member link
signal: an actually mentioned known title, an explicit work title, a short
quoted entity, or a jieba person/place/organization name. Broader repeated
`nz`/Latin terms remain auditable observations but receive no scope authority.

Entity resolution and retrieval intent form a hard conjunction:

- entity only -> `scope_only`, no retrieval;
- scoped progress/timeline/narrative/evidence intent -> `scoped_recall`;
- deictic intent such as `看到哪了` without scope -> `insufficient_scope`;
- ambiguous entity -> `ambiguous_scope`, no global fallback.

A unique entity-to-Arc match is applied before candidate search: whole and
passage searches, plus lexical/cue candidate entry, are restricted to confirmed
Event/Scene members of that Arc. Entity-only, ambiguous, and unscoped deictic
queries return zero candidates. Queries without an Arc scope may use the global
typed pool only when the existing recall policy finds specific query residue;
generic terms do not launch a global vector sweep.

The resolver emits an operator (`latest_relevant_member`, `timeline`,
`member_search`, `narrative_read`, `exact_evidence`, or `arc_index`). It does
not ask a reranker to implement chronology or reading depth.

## Pull-based Arc reading

Narrative prose is not added to the ordinary candidate pool. A confirmed Arc
member may carry a body-free card containing `arc_key`, `narrative_id`, title,
revision, member count, Narrative availability, latest member date when known,
and the fixed hint `可按需读取`. Owners without a confirmed Arc carry no card.

The read-only `find_arc(query, limit)` tool accepts title text, title aliases,
confirmed registry entities, and bounded fuzzy title matches. It returns only
light cards. The caller may then use the existing exact
`read_memory(memory_type="narrative", memory_id=...)` tool when the current
question genuinely needs the authored Narrative. Neither operation enables
Gateway Narrative injection.

## Fixed verification

Gold cases live in `resources/typed_recall_scope_gold_v1.json` and cover
entity-only veto, scoped progress, bare progress, ambiguous characters,
cross-member observed entities, self-proof rejection, generic relationship
names, Narrative reads, and exact evidence routes.

Run:

```powershell
C:\Python313\python.exe scripts\verify_typed_event_scene_entity_shadow.py
```

Expected receipt: `TYPED_EVENT_SCENE_ENTITY_SHADOW_OK GOLD=9/9`.
