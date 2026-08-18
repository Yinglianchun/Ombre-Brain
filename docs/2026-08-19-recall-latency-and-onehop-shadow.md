# Recall latency and reviewed Scene one-hop shadow — 2026-08-19

This report is separate from the closed passage/query-view experiment. It
records Germany runtime optimization and a candidate-only reviewed Scene graph
experiment. No graph candidate affected formal recall, admission, context, or
injection.

## Runtime baseline

- Germany source: `/opt/Ombre-Brain-src`
- Production branch: `Haven/diary-backend-20260724`
- Initial latency audit source/runtime: `be712a2`
- Matrix/cue optimizations: `0bc6bf4`, `df19a8b`, `6e6d315`
- Reviewed one-hop shadow: `9828dac`
- Only `ombre-gateway` was rebuilt for these releases. Ombre Brain and canonical
  data were not restarted or modified.

## Latency diagnosis

Eleven gold live-mirror responses were individually verified before the cache
changes. The remaining server-side batch completed but its transport output was
not retained, so it is not reported as a 22-case result.

| Timing | Median | p95 |
|---|---:|---:|
| Request wall | 1181 ms | 3185 ms |
| `prepare_payload` | 766 ms | 2787 ms |
| Outside prepare | 400 ms | 1093 ms |
| Dynamic bucket select | 740 ms | 2752 ms |
| Semantic candidates | 382 ms | 391 ms |
| Keyword candidates | 12 ms | 37 ms |
| Admission | 47 ms | 79 ms |

`list_all_buckets` was cached at 0 ms. Semantic rescue was disabled/not called
in all 11 cases. Long tails came from authored-cue processing and the live
reranker, not from bucket loading, graph traversal, or semantic rescue.

### Normalized embedding matrix cache

The previous semantic search reopened SQLite, decoded all vector JSON, and
recomputed every document norm on every query. Germany held 230 vectors at 2560
dimensions.

A read-only production-index equivalence benchmark over eight deterministic
query vectors measured:

- old implementation total: 2840.8 ms;
- normalized matrix build: 309.7 ms once;
- eight matrix queries total: 3.2 ms;
- top-10 ID order identical for every query;
- maximum absolute score difference: `1.1e-15`.

`0bc6bf4` caches a normalized NumPy matrix and checks a SQLite count/max-update
signature before reuse. Gateway startup preloads the matrix; local writes clear
the cache immediately, and external committed writes are detected by signature.

Production focused-query verification reduced semantic candidate search from
about 382 ms to 4–5 ms. Four stable warm requests were 911–934 ms wall and kept
the exact same recalled Scene ID as before the release.

### Authored cue term reuse

The old path repeatedly tokenized the same query for every Scene and rebuilt
generic-term sets. `df19a8b` computes query terms once and reuses cue term keys.
`6e6d315` moves the one-time reviewed-cue warmup to Gateway startup.

For the previous authored-cue outlier `hook-8374`:

- old authored-cue step: 1228 ms;
- first request before startup warm: 1331 ms;
- stable cached requests: 17–21 ms;
- after startup warm, the first user request measured 14 ms;
- recalled Scene ID remained identical.

The remaining stable long tail is the external reranker: about 1.0 seconds per
call, with a 2.5-second cold/provider sample. No reranker bypass was introduced;
the named-identity example was supported by body semantic plus reranking rather
than a unique exact anchor, so skipping it was not proven behavior-preserving.

## Reviewed Scene one-hop evaluation

The Scene edge store contained 146 active rows at the read-only audit. Only 54
were currently recall-valid:

- source missing/inactive: 37;
- target missing/inactive: 15;
- source content hash stale: 26;
- target content hash stale: 14;
- valid: 54.

Edge lifecycle repair is a separate workstream. No edge or Scene was modified.

### Offline gold and paired-query sweep

For each query, formal recalled Scene IDs were treated as reliable seeds. One-hop
neighbors had to be present in the ordinary cheap candidate pool and pass a
body-semantic floor.

Gold 22:

- correct formal-miss cases: 3;
- floors 0.45–0.52: one target rescued, zero false-positive expansion, zero
  correct non-target additions;
- floor 0.54: the correct target was lost;
- the successful edge was `echoes`; a directed-only relation filter could not
  recover it.

Live probe/confounder pairs (20 + 20):

- Scene probe formal misses: 6;
- floors 0.45–0.52: zero target rescue and zero confounder historical-target or
  noise exposure;
- floor 0.50/0.52: one unnecessary non-target neighbor on a probe whose target
  was already formally recalled; its relation was `evidenced_by`;
- floor 0.54: no additions.

The narrow policy selected for continued shadow observation is therefore:

```text
formal recalled canonical Scene seed
  -> current valid reviewed `echoes` edge, one hop
  -> neighbor already in ordinary cheap candidate pool
  -> body semantic >= 0.50
  -> at most one candidate-only neighbor
```

Across the fixed gold plus live pairs, this policy added only the correct
initial-story target and no other candidate.

### Production shadow smoke

`9828dac` exposes the policy only in full-shadow retrieval-budget debug as
`reviewed_scene_onehop_shadow`.

- focused initial-story query: one correct candidate, semantic `0.5361`, edge
  confidence `0.91`;
- present Xiaohongshu chitchat: no candidate;
- paired-query `evidenced_by` non-target: no candidate;
- one-hop computation: 59–62 ms in the three smoke requests;
- every result retained `decision_applied=false`, `affects_recall=false`, and
  `live_injection_enabled=false`.

## Decision

Keep the reviewed one-hop policy in shadow. The fixed suite has one positive
addition and no observed noise after the `echoes + semantic >= 0.50` gate, but
one positive is not enough to change live admission or injection. Continue
collecting candidate-only observations on broader real queries.

Do not reopen query splitting, do not switch Germany back to the legacy moment
graph, and do not implement edge review/lifecycle UX in this workstream. Reranker
latency should be audited separately, with behavior equivalence demonstrated
before any bypass.

## Artifacts

- `state/mainland-control-1125a70-20260818.json`
- `state/graph-onehop-semantic-gate-6e6d315.json`
- `state/graph-onehop-semantic-gate-6e6d315-summary.md`
- `state/graph-onehop-live-pairs-6e6d315.json`
- `state/graph-onehop-live-pairs-6e6d315-summary.md`
