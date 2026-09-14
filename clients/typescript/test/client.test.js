import { test } from "node:test";
import assert from "node:assert/strict";
import { Memory, MemoryError, memoryTools } from "../src/index.js";

function stub(handler) {
  const calls = [];
  const fetch = async (url, init) => {
    calls.push({ url, init, body: init.body ? JSON.parse(init.body) : undefined });
    return handler(calls.length, { url, init });
  };
  return { fetch, calls };
}

const ok = (body, status = 200, headers = {}) => ({
  ok: status >= 200 && status < 300, status,
  headers: { get: (k) => headers[k] ?? headers[k.toLowerCase()] ?? null },
  text: async () => JSON.stringify(body),
});

test("it sends the key and the path the service expects", async () => {
  const s = stub(() => ok({ stored: 1 }));
  const mem = new Memory({ url: "https://m.example/", apiKey: "rgm_x", fetch: s.fetch });
  const r = await mem.remember("scout", "TSMC is building a fab in Phoenix.");

  assert.deepEqual(r, { stored: 1 });
  assert.equal(s.calls[0].url, "https://m.example/sessions/scout/memory");
  assert.equal(s.calls[0].init.headers.authorization, "Bearer rgm_x");
  assert.equal(s.calls[0].body.text, "TSMC is building a fab in Phoenix.");
});

test("a session name with a slash cannot escape its path", async () => {
  const s = stub(() => ok({}));
  await new Memory({ url: "https://m.example", fetch: s.fetch }).remember("a/../b", "x");
  assert.ok(s.calls[0].url.endsWith("/sessions/a%2F..%2Fb/memory"), s.calls[0].url);
});

test("a 502 during a deploy is retried, and the retry's answer is returned", async () => {
  const s = stub((n) => (n === 1 ? ok({ detail: "bad gateway" }, 502) : ok([{ content: "back" }])));
  const mem = new Memory({ url: "https://m.example", fetch: s.fetch, retries: 2 });
  assert.deepEqual(await mem.recall("anything?"), [{ content: "back" }]);
  assert.equal(s.calls.length, 2);
});

test("a quota refusal is not retried and says what the limit was", async () => {
  const s = stub(() => ok({ detail: "limit" }, 429, { "X-Quota-Used": "10001", "X-Quota-Limit": "10000" }));
  const mem = new Memory({ url: "https://m.example", fetch: s.fetch, retries: 3 });
  await assert.rejects(() => mem.recall("x"), (e) => {
    assert.ok(e instanceof MemoryError);
    assert.equal(e.status, 429);
    assert.match(e.message, /10,?001 of 10,?000/);
    return true;
  });
  assert.equal(s.calls.length, 1, "a quota error must not be retried");
});

test("a 4xx is reported rather than retried", async () => {
  const s = stub(() => ok({ detail: "no" }, 422));
  const mem = new Memory({ url: "https://m.example", fetch: s.fetch, retries: 3 });
  await assert.rejects(() => mem.recall("x"), /422/);
  assert.equal(s.calls.length, 1);
});

test("a hanging request times out with a readable message", async () => {
  const fetch = (url, init) => new Promise((_, reject) =>
    init.signal.addEventListener("abort", () => reject(Object.assign(new Error("aborted"), { name: "AbortError" }))));
  const mem = new Memory({ url: "https://m.example", fetch, timeoutMs: 20, retries: 0 });
  await assert.rejects(() => mem.recall("x"), /timed out after 20ms/);
});

test("camelCase options become the wire names the service reads", async () => {
  const s = stub(() => ok([]));
  const mem = new Memory({ url: "https://m.example", fetch: s.fetch });
  await mem.discover("why?", { topK: 3, hops: 2, maxResults: 7, session: "notes" });
  assert.deepEqual(s.calls[0].body, { query: "why?", top_k: 3, hops: 2, max_results: 7, session: "notes" });
});

test("chat passes the folding budget through", async () => {
  const s = stub(() => ok({ reply: "ok" }));
  const mem = new Memory({ url: "https://m.example", fetch: s.fetch });
  await mem.chat([{ role: "user", content: "hi" }], { maxHistoryTokens: 4000, keepTailTokens: 1500 });
  assert.equal(s.calls[0].body.max_history_tokens, 4000);
  assert.equal(s.calls[0].body.keep_tail_tokens, 1500);
});

test("the tools carry schemas a model can be given, and run themselves", async () => {
  const s = stub((n) => (n === 1 ? ok({ stored: 1 }) : ok([{ content: "the fab needs water", scopes: ["scout"] }])));
  const mem = new Memory({ url: "https://m.example", fetch: s.fetch });
  const tools = memoryTools(mem, "research-agent");

  assert.deepEqual(tools.map((t) => t.name), ["remember", "recall", "why"]);
  const def = tools[0].definition.function;
  assert.equal(def.name, "remember");
  assert.deepEqual(def.parameters.required, ["text"]);

  assert.match(await tools[0].execute({ text: "a fact" }), /remembered/);
  assert.equal(s.calls[0].url, "https://m.example/sessions/research-agent/memory");
  assert.match(await tools[1].execute({ query: "water?" }), /the fab needs water \[scout\]/);
});

test("recall says so plainly when nothing is on record", async () => {
  const s = stub(() => ok([]));
  const tools = memoryTools(new Memory({ url: "https://m.example", fetch: s.fetch }));
  assert.match(await tools[1].execute({ query: "anything?" }), /nothing on record/);
});
