/** Compile in a browser worker, then send only the compiled envelope to AethyxLM. */
export function createLocalCompiler(workerUrl) {
  const worker = new Worker(workerUrl, { type: "module" });
  let nextId = 0;
  const pending = new Map();
  worker.onmessage = ({ data }) => {
    const entry = pending.get(data.id);
    if (!entry) return;
    pending.delete(data.id);
    data.ok ? entry.resolve(data.compiled) : entry.reject(new Error(data.error));
  };
  return {
    compile(request, options = {}) {
      const id = ++nextId;
      return new Promise((resolve, reject) => {
        pending.set(id, { resolve, reject });
        worker.postMessage({ id, request, options });
      });
    },
    close() { worker.terminate(); },
  };
}

export async function sendCompiledChat({ compiler, endpoint, query, localContext, signal, requireCompression = false }) {
  const compiledContext = await compiler.compile(
    { query, items: localContext },
    { allowRawFallback: !requireCompression },
  );
  return fetch(endpoint, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ query, compiled_context: compiledContext }),
    signal,
  });
}
