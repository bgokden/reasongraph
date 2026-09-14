/**
 * ReasonGraph client for JavaScript and TypeScript.
 *
 * No dependencies: it uses the platform's fetch, so it runs on Node 18+, Bun, Deno, Cloudflare
 * Workers and the browser. Types ship alongside in index.d.ts.
 *
 *     import { Memory } from "reasongraph";
 *     const mem = new Memory({ url: "https://memory.primaxiom.ai", apiKey: "rgm_..." });
 *     await mem.remember("scout", "TSMC is building a fab in Phoenix.");
 *     const facts = await mem.discover("What could disrupt the Phoenix fab?");
 */

export class MemoryError extends Error {
  constructor(message, status, body) {
    super(message);
    this.name = "MemoryError";
    this.status = status;
    this.body = body;
  }
}

const DEFAULTS = { timeoutMs: 60_000, retries: 2 };

export class Memory {
  /**
   * @param {{url?: string, apiKey?: string, timeoutMs?: number, retries?: number, fetch?: typeof fetch}} [options]
   * url and apiKey fall back to REASONGRAPH_URL and REASONGRAPH_API_KEY where an environment exists.
   */
  constructor(options = {}) {
    const env = typeof process !== "undefined" && process.env ? process.env : {};
    this.url = (options.url || env.REASONGRAPH_URL || "https://memory.primaxiom.ai").replace(/\/+$/, "");
    this.apiKey = options.apiKey || env.REASONGRAPH_API_KEY || "";
    this.timeoutMs = options.timeoutMs ?? DEFAULTS.timeoutMs;
    this.retries = options.retries ?? DEFAULTS.retries;
    this._fetch = options.fetch || globalThis.fetch;
    if (!this._fetch) throw new Error("no fetch available; pass one via options.fetch");
  }

  async _call(method, path, body) {
    let lastError;
    for (let attempt = 0; attempt <= this.retries; attempt++) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeoutMs);
      try {
        const res = await this._fetch(this.url + path, {
          method,
          signal: controller.signal,
          headers: {
            "content-type": "application/json",
            ...(this.apiKey ? { authorization: `Bearer ${this.apiKey}` } : {}),
          },
          ...(body === undefined ? {} : { body: JSON.stringify(body) }),
        });
        if (res.status === 429) {
          // The service reports the plan limit in headers; surface both, do not retry.
          throw new MemoryError(
            `quota reached (${res.headers.get("X-Quota-Used") || "?"} of ${res.headers.get("X-Quota-Limit") || "?"} requests)`,
            429, await safeBody(res));
        }
        // A deploy answers 502/503 for a few seconds. Those are worth retrying; 4xx never is.
        if (res.status >= 500 && attempt < this.retries) {
          lastError = new MemoryError(`server returned ${res.status}`, res.status, await safeBody(res));
          await sleep(400 * (attempt + 1));
          continue;
        }
        if (!res.ok) throw new MemoryError(`${method} ${path} failed with ${res.status}`, res.status, await safeBody(res));
        return await safeBody(res);
      } catch (err) {
        if (err instanceof MemoryError) throw err;
        // Network failure or timeout: retry, then give up with something readable.
        lastError = err;
        if (attempt >= this.retries) {
          throw new MemoryError(err.name === "AbortError" ? `timed out after ${this.timeoutMs}ms` : String(err.message || err), 0, null);
        }
        await sleep(400 * (attempt + 1));
      } finally {
        clearTimeout(timer);
      }
    }
    throw lastError;
  }

  /** Store one sentence (or a paragraph, which the service splits) in a session. */
  remember(session, text, options = {}) {
    return this._call("POST", `/sessions/${encodeURIComponent(session)}/memory`,
      { text, ...(options.split === undefined ? {} : { split: options.split }) });
  }

  /** Store many at once: one round trip instead of n. */
  rememberMany(session, texts, options = {}) {
    return this._call("POST", `/sessions/${encodeURIComponent(session)}/memory/batch`,
      { texts, ...(options.split === undefined ? {} : { split: options.split }) });
  }

  /** Facts that answer a question, each with the session it came from. */
  recall(query, options = {}) {
    return this._call("POST", "/query", { query, top_k: options.topK ?? 5, hops: options.hops ?? 4, ...pickSession(options) });
  }

  /** Like recall, but returns how each fact connects back to the question. */
  discover(query, options = {}) {
    return this._call("POST", "/discover", {
      query, top_k: options.topK ?? 5, hops: options.hops ?? 4,
      max_results: options.maxResults ?? 10, ...pickSession(options),
    });
  }

  /** The cause-and-effect chain between two facts. */
  causalChain(from, to, options = {}) {
    return this._call("POST", "/causal_chain", { from_content: from, to_content: to, max_depth: options.maxDepth ?? 6 });
  }

  /** What a fact led to, walking forward. */
  traceEffects(content, options = {}) {
    return this._call("POST", "/trace", { content, direction: "effects", max_depth: options.maxDepth ?? 6 });
  }

  /** What led to a fact, walking backward. */
  traceCauses(content, options = {}) {
    return this._call("POST", "/trace", { content, direction: "causes", max_depth: options.maxDepth ?? 6 });
  }

  /** One turn with memory recalled, injected, and the exchange remembered. */
  chat(messages, options = {}) {
    return this._call("POST", "/chat", {
      messages, session: options.session ?? "chat",
      ...(options.system ? { system: options.system } : {}),
      ...(options.maxHistoryTokens ? { max_history_tokens: options.maxHistoryTokens } : {}),
      ...(options.keepTailTokens ? { keep_tail_tokens: options.keepTailTokens } : {}),
    });
  }

  /** Replace a fact, keeping the old one readable in its history. */
  correct(session, oldText, newText) {
    return this._call("POST", "/supersede", { session, old_text: oldText, new_text: newText });
  }

  /** What this fact replaced, and when. */
  history(text) {
    return this._call("POST", "/history", { text });
  }

  /** Erase one session: its facts, and the entities only it linked. */
  forgetSession(session) {
    return this._call("DELETE", `/sessions/${encodeURIComponent(session)}`);
  }

  /** Fact and entity counts for this workspace. */
  stats() {
    return this._call("GET", "/stats");
  }
}

function pickSession(options) {
  return options.session ? { session: options.session } : {};
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function safeBody(res) {
  const text = await res.text();
  if (!text) return null;
  try { return JSON.parse(text); } catch { return text; }
}

/**
 * Tool definitions for plain function calling, in the shape OpenAI, Anthropic-compatible
 * gateways and most agent frameworks accept. Each returned tool carries an `execute` that
 * runs it, so a caller can dispatch by name without writing any glue.
 *
 *     const tools = memoryTools(mem, "research-agent");
 *     const defs = tools.map(t => t.definition);          // send these to the model
 *     const tool = tools.find(t => t.name === call.name); // run what it asked for
 *     const result = await tool.execute(call.arguments);
 */
export function memoryTools(memory, session = "agent") {
  return [
    {
      name: "remember",
      definition: {
        type: "function",
        function: {
          name: "remember",
          description: "Store a fact you have learned so that you and other agents can recall it later.",
          parameters: {
            type: "object",
            properties: { text: { type: "string", description: "One plain sentence stating the fact." } },
            required: ["text"],
          },
        },
      },
      execute: async ({ text }) => {
        const r = await memory.remember(session, text);
        return `remembered (${r?.stored ?? 1} fact)`;
      },
    },
    {
      name: "recall",
      definition: {
        type: "function",
        function: {
          name: "recall",
          description: "Look up facts that answer a question, from anything any agent has stored.",
          parameters: {
            type: "object",
            properties: { query: { type: "string", description: "The question, in plain words." } },
            required: ["query"],
          },
        },
      },
      execute: async ({ query }) => {
        const facts = await memory.recall(query, { topK: 5 });
        const list = Array.isArray(facts) ? facts : facts?.facts || [];
        return list.length
          ? list.map((f) => `- ${f.content}${f.scopes ? ` [${f.scopes.join(", ")}]` : ""}`).join("\n")
          : "nothing on record about that";
      },
    },
    {
      name: "why",
      definition: {
        type: "function",
        function: {
          name: "why",
          description: "Trace what caused something, following recorded cause-and-effect links.",
          parameters: {
            type: "object",
            properties: { fact: { type: "string", description: "The thing whose causes you want." } },
            required: ["fact"],
          },
        },
      },
      execute: async ({ fact }) => {
        const trace = await memory.traceCauses(fact);
        const hops = trace?.chain || [];
        return hops.length
          ? hops.map((h) => `${h.cause} -> ${h.effect}`).join("\n")
          : "no recorded causes for that";
      },
    },
  ];
}

export default Memory;
