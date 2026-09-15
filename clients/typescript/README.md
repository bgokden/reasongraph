# reasongraph (JavaScript / TypeScript)

Graph memory for AI agents: remember facts, recall them by meaning, and trace what caused what.

No dependencies. Uses the platform `fetch`, so it runs on Node 18+, Bun, Deno, Cloudflare Workers
and in the browser.

Not on npm yet. It is plain JavaScript with types alongside and no build step, so copy this
folder into your project, or install it by path from a checkout:

```bash
npm install ./path/to/reasongraph/clients/typescript
```

```ts
import { Memory } from "reasongraph";

const mem = new Memory({ apiKey: process.env.REASONGRAPH_API_KEY });

await mem.remember("scout", "TSMC is building a chip fab in Phoenix, Arizona.");
await mem.remember("analyst", "Arizona declared a water emergency after a three-year drought.");

const facts = await mem.discover("What could disrupt the Phoenix fab?");
for (const f of facts) console.log(f.content, f.scopes);
```

Two agents wrote one sentence each. Neither wrote the answer.

## Giving it to an agent as tools

For agents that call functions directly, without a framework:

```ts
import { Memory, memoryTools } from "reasongraph";

const tools = memoryTools(new Memory(), "research-agent");

const response = await openai.chat.completions.create({
  model: "...",
  messages,
  tools: tools.map((t) => t.definition),
});

for (const call of response.choices[0].message.tool_calls ?? []) {
  const tool = tools.find((t) => t.name === call.function.name);
  const result = await tool!.execute(JSON.parse(call.function.arguments));
}
```

Three tools: `remember` stores a fact, `recall` answers a question from anything any agent stored,
and `why` traces what caused something.

## What you get back

`recall` and `discover` return facts with the sessions they came from, so an answer can always be
traced to the note behind it. `discover` also returns the path it walked to reach each fact.

## Errors

Everything throws `MemoryError` with a `status`. Server errors and network failures are retried
twice; a quota refusal (429) and any 4xx are not, because retrying those only wastes time. A quota
error's message states the limit and what you used.

## Long conversations

`chat` takes an optional token budget. Past it, the newest messages stay verbatim and older ones are
folded into a rolling summary. Nothing is deleted: folded-out turns stay in memory and come back by
recall.

```ts
await mem.chat(messages, { maxHistoryTokens: 6000, keepTailTokens: 2000 });
```
