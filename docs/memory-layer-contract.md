# Memory Layer Contract

This document describes how Ombre-Brain should treat different memory layers.
It is a behavior contract, not a new storage schema.

The goal is simple: a memory layer must know when it can appear, how much text it can carry, and whether it is allowed to affect recall.

## Principles

- Not every chat line becomes memory. A turn should be judged as a whole before writing.
- Stable facts, short-term states, and process events do not belong in the same bucket role.
- User-side memory and relationship-side memory are different. Store both when both are durable.
- Direct recall must be stricter than related context.
- Emotional temperature is not a solved/unsolved flag. It is valence and arousal: sweet/painful, cold/burning.
- Raw detail should stay available somewhere, but not every layer should inject raw detail.

## User Side And Relationship Side

One interaction can produce two different memories.

User side:

- The user's state, preference, boundary, habit, need, pain point, current difficulty.
- Example: "I have been sleeping badly this week."

Relationship or AI side:

- What the assistant did, promised, learned to notice, or should carry next time.
- Example: "When the user says she is tired, first check sleep and recent overload before giving advice."

Do not collapse these into one generic summary. If only the user state matters, write only that. If only the relationship lesson matters, write only that. If neither will matter later, write nothing.

## Layer Table

| Layer | Storage signal | Can be direct seed | Render shape | Gateway injection | Cooldown | Can diffuse | Original detail |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Core memory | `pinned`, `protected`, important `permanent` | Yes, but usually requested or interval gated | Short stable rule or original if explicitly read | `Core Memory`, normally off by budget | Rare | Can be source, but carefully | Keep exact wording |
| Long-term anchor | `anchor=true`, old enough, scarce | Yes | Dehydrated or original when explicitly asked | Breath surfacing; gateway only if recalled | Normal recall cooldown | Yes | Keep exact wording |
| Direct recalled bucket | admitted body/title moment | Yes | Short bucket: `bucket_original`; long bucket: `bucket_window`; high value/detail query: `bucket_capsule` | `Recalled Memory` | `skip_recent_rounds` + `cooldown_hours` | Source for diffusion | Keep original in bucket |
| Diffused related memory | reached from reliable direct seed through memory edges | No | Summary only | `Diffused Memory` | follows source/normal ranking | May continue only through reliable chain rules | Do not inject full bucket |
| Recent context | recent non-core bucket, or recent context store | No, unless user explicitly asks recent | Compact recent block | Only on explicit recent query, 24h re-entry, or reliable dynamic context; has cooldown | `recent_context_cooldown_hours` | No | Keep original elsewhere |
| Relationship weather | `type=feel`, `relationship_weather`, `daily_impression`, `weekly_impression` | No | Short weather/temperature block | Separate section; default mostly quiet | Interval/config gated | No direct seed; can color response | Keep affect text, not as topic proof |
| Affect anchor / favorite reason / comments | section text in bucket or metadata comments | No | Auxiliary context only | Only attached to reliable target or favorite block | follows parent | No seed | Keep exact wording when intimate |
| Favorite memory | `ai_favorite`, legacy `haven_favorite`, or `<ai_name>_favorite`, with reason | Not by tag alone | Small favorite card | Manual marker/header or configured interval | Separate budget | Can be source if also directly recalled | Keep reason |
| Dream | dream engine latent/surfaced item | Not normal recall seed | Original dream text | Surfaces once when dream resonance passes | Dream-level surface rules | Can later create edges/rings | Do not truncate in memory layer |
| Diary / raw chat source | external diary, local chat log, imported source | No | Not injected as memory by default | Not direct gateway context | N/A | Extracted segments only | Preserve raw source outside buckets |
| Archive / digested old memory | archived, `resolved`, `digested`, low activity | Usually no | Only on explicit search or resurfacing | Not automatic | Sinks strongly | Rare | Keep for lookup |

## Write Decision

After a meaningful interaction, decide in this order:

1. Is there a stable user preference, boundary, identity fact, habit, or recurring need?
2. Is there a relationship-side lesson: how the assistant should respond, what agreement changed, what promise exists?
3. Is this a short-term state that should influence the next few turns or days, but not become a permanent fact?
4. Is this a process event with emotional history that may matter later?
5. Is it just ordinary chat noise?

Only write what survives that test.

Newly written buckets should carry a writer-side first pass:

```yaml
memory_subject: user | relationship | event
memory_layer: stable_boundary | short_state | process_event | relationship_lesson
memory_classification_source: model | model_adjusted | rule | default
```

This first pass answers what the writer thought the memory was. Runtime recall may still apply stricter layer policy from bucket type, tags, pinned/protected, archive state, and context-only sections.

Runtime uses the writer pass as a hint only after stronger storage signals have been checked. `stable_boundary` and `relationship_lesson` use long-term anchor policy unless the bucket is manually pinned/protected/favorited/archived. `short_state` and `process_event` use dynamic memory policy.

Examples:

- "I dislike being lectured." Long-term user boundary. It may become core or anchor if repeated or central.
- "I have a headache today." Short-term state. Prefer recent context or diary; only bucket it if it affects a larger event.
- "We fought a few days ago, then made up." Process event. Store the event shape and emotional temperature; do not call it solved just because arousal cooled.
- "You answered me softly after I said I was tired." Relationship-side learning. Store how the assistant should carry it, not only that the user was tired.

## Direct And Related Rules

Direct means the current query has reliable evidence in the bucket body/title/summary/tags or a high-confidence admitted moment.

Direct return:

- Short bucket: return original bucket body.
- Long bucket: return matched moment plus nearby original window. Prefer `source_ref` line windows when available; otherwise use the inline original text/window.
- High-value bucket or detail query: return dehydrated bucket capsule.

Related means the bucket was reached from a reliable direct seed.

Related return:

- Always summary.
- Include path/relation when it helps the model understand why it appeared.
- Never let related memory pretend to be the current fact.
- When reliable chain walk is enabled and the path has at least two steps, Gateway and MCP breath may render a compact `Chain Bundle`: seed, chain path, target summary, and 1-2 temperature context items. This is still summary-only related memory, not raw bucket injection.

Retrieval mode:

- `graph` is the default runtime: direct recall uses admitted moments and related memory can diffuse through approved edges.
- `bucket` is an opt-in comparison mode: direct recall uses reliable bucket candidates and the same direct render rules, but skips moment graph refresh and does not inject related memory.

Context-only sections:

- `comment`
- `affect_anchor`
- `favorite_reason`

These can color a reliable memory, but they cannot prove a direct hit by themselves.

## Runtime Gates

The runtime answers "which layer is this?" with `memory_layers.py`, then applies three separate gates:

- Direct seed gate: only admitted bucket body/title moments can prove a direct recall. Dream resonance, source records, relationship weather, and context-only sections cannot.
- Recall context gate: comments, affect anchors, and favorite reasons may stay indexed as context for their parent bucket, so they can appear beside a reliable direct hit.
- Related target gate: diffused memory must be summary-only and must pass the target layer policy. Archive/resolved/digested buckets stay hidden in normal related recall, but can return as old-memory summary when the query explicitly asks for old, archived, conflict, or resolved material.
- Recent context gate: automatic `Recent Context` only uses dynamic memory. Writer-classified stable boundaries and relationship lessons do not appear just because they were written recently; they can still appear when directly recalled or when the user explicitly asks for recent memory.

Query-level gates are planned once by `RecallPolicy.plan_query()`. The plan carries whether the query wants a body chain, whether topic evidence is enforced, whether old/archive material is explicitly requested, and whether cautious diffusion is allowed by repair context.

This separation is important. A moment can be searchable context without being allowed to prove the current topic.

Debug surfaces should expose the same runtime decision. `inspect_moments`, `/api/moments`, `inspect_diffusion`, `/api/diffusion-debug`, `/api/recall-debug`, `/api/breath-debug`, and Gateway injection debug include:

- `layer_debug`: the inferred layer, writer hint, and static layer policy.
- `runtime_gate`: the per-query decision for direct seed, related injection, recent context, topic evidence, and the reason an item would or would not appear.

## Injection Order

Gateway dynamic context should stay quiet and ordered:

1. `Recent Context`
2. `Context Mode`
3. `Recalled Memory`
4. `Diffused Memory`
5. Persona state
6. `Relationship Weather`
7. `<identity.ai_name> Favorite Memory`

Core memory is stable context and should not compete with dynamic memory budget.

## What This Contract Prevents

- Treating a temporary state as a permanent preference.
- Letting affect anchors become keyword bait.
- Returning whole related buckets just because they are nearby.
- Losing exact intimate wording from true direct hits.
- Recording only "what the user said" while forgetting "what the assistant learned to do."
