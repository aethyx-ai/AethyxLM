/**
 * Browser-side reference compiler for the future Aethyx chat client.
 * It has no network access and is intended to run inside a Web Worker.
 */

const WORD = /[\p{L}\p{N}]{3,}/gu;
const PRECISION = [
  /\b[0-9a-fA-F]{16,}\b/,
  /\b[0-9a-fA-F]{8}-[0-9a-fA-F-]{27,}\b/,
  /\b(?:sk|api|token|secret|password)[-_A-Za-z0-9]{8,}\b/i,
  /https?:\/\/\S+/,
  /\b\d{8,}\b/,
];

function terms(text) {
  return new Set((text.match(WORD) || []).map((value) => value.toLocaleLowerCase()));
}

function mustPreserve(item) {
  if (item.protected) return true;
  if (["system", "tool_schema", "agent_state"].includes(item.kind)) return true;
  if (item.kind === "conversation" && Number(item.recency || 0) >= 2) return true;
  if (item.text.includes("```")) return true;
  return PRECISION.some((pattern) => pattern.test(item.text));
}

async function sha256(text) {
  const bytes = new TextEncoder().encode(text);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, "0")).join("");
}

export async function compileContext(request, options = {}) {
  const maxSelectedItems = Number(options.maxSelectedItems || 12);
  const queryTerms = terms(request.query);
  const protectedItems = [];
  const candidates = [];
  for (const item of request.items) {
    if (mustPreserve(item)) {
      protectedItems.push(item);
      continue;
    }
    const itemTerms = terms(item.text);
    let overlap = 0;
    for (const term of queryTerms) if (itemTerms.has(term)) overlap += 1;
    const score = 0.65 * overlap / Math.max(queryTerms.size, 1)
      + 0.25 * Number(item.priority || 0.5)
      + 0.10 * Math.min(Number(item.recency || 0), 3);
    candidates.push({ item, score });
  }
  candidates.sort((left, right) => right.score - left.score || left.item.source_id.localeCompare(right.item.source_id));
  const selectedItems = candidates.slice(0, maxSelectedItems).map(({ item }) => item);
  const represented = new Set([...protectedItems, ...selectedItems].map((item) => item.source_id));
  const sourceDigest = await sha256(JSON.stringify(request.items.map((item) => [item.source_id, item.kind, item.text])));
  const compiled = {
    version: "aethyx-context-web/0.1",
    mode: "graph",
    query: request.query,
    protected_items: protectedItems,
    selected_items: selectedItems,
    graph: {
      nodes: selectedItems.map((item) => ({
        node_id: `source:${item.source_id}`,
        label: item.text.slice(0, 160).replace(/\s+/g, " "),
        kind: item.kind,
        source_ids: [item.source_id],
      })),
      edges: selectedItems.slice(1).map((item, index) => ({
        source: `source:${selectedItems[index].source_id}`,
        target: `source:${item.source_id}`,
        relation: "followed_by",
        provenance: "EXTRACTED",
        confidence: 1.0,
      })),
    },
    omitted_source_ids: request.items.filter((item) => !represented.has(item.source_id)).map((item) => item.source_id),
    source_digest: sourceDigest,
    warnings: [],
  };
  const rawChars = request.query.length + request.items.reduce((total, item) => total + item.text.length, 0);
  const compiledChars = JSON.stringify(compiled).length;
  if (compiledChars >= rawChars * 0.95) {
    if (options.allowRawFallback === false) {
      throw new Error("No safe, profitable compression was found and raw fallback is disabled.");
    }
    return {
      version: "aethyx-context-web/0.1",
      mode: "raw",
      query: request.query,
      raw_items: request.items,
      source_digest: sourceDigest,
      warnings: ["Compression bypassed because the local size gate predicted no benefit."],
    };
  }
  return compiled;
}
