import { compileContext } from "./context_compiler.js";

self.onmessage = async (event) => {
  const id = event.data.id;
  try {
    const compiled = await compileContext(event.data.request, event.data.options || {});
    self.postMessage({ id, ok: true, compiled });
  } catch (error) {
    self.postMessage({ id, ok: false, error: String(error?.message || error) });
  }
};
