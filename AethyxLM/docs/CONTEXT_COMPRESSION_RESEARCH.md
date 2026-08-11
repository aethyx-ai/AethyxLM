# Local Context Compilation Research

## Purpose

AethyxLM's research question is: **what representation preserves the same useful context with less model-side cost?** Text, graphs, visual pages, learned latents, and hybrids are candidates. None is assumed to win in advance.

The present harness is an isolated research system. It does not change the pretrained transformer and it does not claim that the proposed 70% target has been achieved.

## Reference analysis

Two public projects informed the research direction. No source code from either project is included in AethyxLM.

### Graphify

Reference: <https://github.com/Graphify-Labs/graphify>

Useful principles identified from its implementation and documentation:

- deterministic local extraction before any semantic model pass;
- real nodes and traversable relationships rather than an opaque vector-only index;
- source IDs and explicit `EXTRACTED` versus `INFERRED` provenance;
- query-scoped subgraphs instead of repeatedly sending a complete corpus;
- incremental manifests, deduplication, and stable identifiers;
- graph retrieval evaluated on downstream recall and QA, not graph size alone.

AethyxLM applies these principles to conversational/tool/document context, not codebase mapping. Its graph implementation is original and intentionally small.

### PxPipe

Reference: <https://github.com/teamchong/pxpipe>

Useful principles identified from its implementation and documentation:

- local request transformation before transmission;
- only compress content when a measured profitability gate predicts a benefit;
- preserve recent turns, user instructions, IDs, hashes, secrets, and other precision-critical values as native text;
- separate dense historical material from a live text tail;
- use stable chunks and page budgets;
- measure a counterfactual text cost beside compressed cost;
- treat visual representation as lossy and expose failures honestly.

AethyxLM does not proxy Anthropic/OpenAI traffic and does not assume image-token economics transfer to our model. Visual pages remain one benchmark candidate requiring a compatible encoder.

## Current Aethyx design

The harness has five stages:

```text
Local raw context
      |
Verbatim-risk classification
      |---------------------> protected native text
      v
Deterministic graph extraction
      v
Query-scoped source selection
      |---------------------> compact graph/text envelope
      v
Optional visual-page planning (vision-capable target only)
      v
Retention + cost benchmark
```

Every item has a stable `source_id`. Graph edges carry provenance and confidence. System prompts, tool schemas, agent state, recent conversation turns, code blocks, long identifiers, URLs, paths, and likely secrets are conservatively protected.

The current 31M model is text-only. Therefore visual mode refuses to emit pages unless the caller explicitly declares a vision input contract. Learned latent representations remain covered by the separate adapter pilots and require training before meaningful comparison.

## Benchmark rules

`context_lab/benchmark.py` records:

- baseline tokenizer-v2 tokens;
- serialized compiled-envelope tokens;
- relevant-source recall;
- exact-value retention;
- visual page count;
- warnings and omitted sources.

`model_task_accuracy` remains `null` until the representation is actually consumed by a compatible model. Visual pixel area is not mislabeled as an equivalent number of text tokens.

Run the included smoke benchmark:

```powershell
python scripts/context_lab.py benchmark `
  context_lab/examples/retrieval_cases.json `
  --mode graph `
  --output evaluation/results/context-graph-smoke.json
```

## How local compilation works in a future website

Our own Aethyx chat website can compile locally because the browser already holds the user's conversation and explicitly selected documents:

1. The UI sends context to a dedicated Web Worker.
2. The worker classifies exact-risk content and builds/retrieves the graph.
3. Raw source material remains in browser memory or IndexedDB.
4. The network request contains the current query plus the compiled envelope and any protected text required for correctness.
5. The server records the compiler version and representation type for evaluation.

Reference browser modules are under `context_lab/web/`. They perform no network calls themselves. `website_adapter.js` demonstrates that compilation completes before `fetch` constructs the API request.

Correctness mode allows an explicit raw fallback when compression would cost more or violate preservation rules. Privacy-strict mode sets `requireCompression: true`; it fails locally instead of silently transmitting raw context when no safe representation is available.

Important privacy limit: encoded graph or visual content still communicates information to the server. Local compilation avoids transmitting omitted raw material and repeated full context; it is not encryption. A served website can also change its JavaScript, so a stronger trust model should use a signed/version-pinned compiler bundle, strict CSP, disabled raw telemetry, transparent network inspection, and eventually an optional open local companion application.

## ChatGPT and Claude websites

A normal Aethyx webpage cannot intercept or rewrite requests made by `chatgpt.com` or `claude.ai` because browser same-origin protections isolate those sites. Integration with third-party websites would require one of:

- an explicit browser extension;
- a user-controlled local proxy where permitted;
- manual image/file export and upload;
- an official provider extension/API.

Those routes also depend on provider terms and accepted input formats. Custom learned latents cannot be sent through a third-party website unless that provider explicitly supports the representation. Our primary product path should therefore be first-party local compilation inside the future Aethyx website.

## Next experiments

1. Establish raw-text accuracy on long-context retrieval, tool use, and multi-turn consistency.
2. Evaluate query-scoped graph envelopes at several source budgets.
3. Train a small graph/context adapter and compare against serialized graph text.
4. Add visual pages only after a vision encoder exists; measure exact-string failures separately.
5. Compare hybrids that keep a protected text tail and retrieve graph nodes or visual pages.
6. Report bandwidth, latency, GPU memory, attention compute, and task accuracy together.
