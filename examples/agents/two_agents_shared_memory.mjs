/**
 * Two agents, one memory, an answer neither of them could give alone.
 *
 * The Python version of this example is two_agents_shared_memory.py; this is the same story
 * for a JavaScript or TypeScript stack.
 *
 *     REASONGRAPH_API_KEY=rgm_... node examples/agents/two_agents_shared_memory.mjs
 */
import { Memory } from "../../clients/typescript/src/index.js";

const mem = new Memory({
  url: process.env.REASONGRAPH_URL || "https://memory.primaxiom.ai",
  apiKey: process.env.REASONGRAPH_API_KEY,
});

// A scout agent, reading supply-chain news. It never sees the water story.
await mem.rememberMany("scout", [
  "TSMC is building a chip fab in Phoenix, Arizona.",
  "The Phoenix fab is scheduled to start production next year.",
]);

// An analyst agent, reading climate reports. It has never heard of the fab.
await mem.rememberMany("analyst", [
  "Arizona declared a water emergency after a three-year drought.",
  "Chip fabrication uses large volumes of ultrapure water.",
]);

// Neither agent wrote this answer. The memory walks from one to the other.
const facts = await mem.discover("What could disrupt the Phoenix fab?", { maxResults: 6 });

console.log(`\n${facts.length} facts, from ${new Set(facts.flatMap((f) => f.scopes ?? [])).size} agents:\n`);
for (const f of facts) {
  const from = (f.scopes ?? []).join(", ");
  console.log(`  ${f.content}${from ? `  [${from}]` : ""}`);
}
