/** ReasonGraph client for JavaScript and TypeScript. No dependencies; uses the platform fetch. */

export interface Fact {
  content: string;
  scopes?: string[];
  /** How this fact was reached from the question: alternating facts and the entities bridging them. */
  path?: Array<{ content?: string; entity?: string; scopes?: string[] }>;
  causes?: string[];
  /** True when the fact came from a session other than the one asked about. */
  cross_session?: boolean;
}

export interface Hop { cause: string; effect: string; fact?: string; depth?: number; scopes?: string[] }
export interface Trace { origin: string | null; chain: Hop[]; terminals: string[] }
export interface ChatMessage { role: "system" | "user" | "assistant"; content: string }
export interface ChatReply { reply: string; session: string; narrative: string; context: Fact[]; stored?: unknown }
export interface Stats { tenant: string; facts: number; entities: number; sessions: string[] }

export interface MemoryOptions {
  /** Defaults to REASONGRAPH_URL, then https://memory.primaxiom.ai */
  url?: string;
  /** Defaults to REASONGRAPH_API_KEY */
  apiKey?: string;
  /** Per request, including retries. Default 60000. */
  timeoutMs?: number;
  /** Retries for 5xx and network failures only; 4xx and quota refusals are never retried. Default 2. */
  retries?: number;
  fetch?: typeof fetch;
}

export interface RecallOptions { topK?: number; hops?: number; session?: string }
export interface DiscoverOptions extends RecallOptions { maxResults?: number }
export interface ChatOptions {
  session?: string;
  system?: string;
  /** Fold the transcript once it passes this many tokens. */
  maxHistoryTokens?: number;
  /** How much of the newest conversation stays word for word. */
  keepTailTokens?: number;
}

/** Raised for every failure, with the HTTP status (0 for network failures and timeouts). */
export declare class MemoryError extends Error {
  status: number;
  body: unknown;
  constructor(message: string, status: number, body: unknown);
}

export declare class Memory {
  url: string;
  apiKey: string;
  constructor(options?: MemoryOptions);
  remember(session: string, text: string, options?: { split?: boolean }): Promise<{ stored?: number }>;
  rememberMany(session: string, texts: string[], options?: { split?: boolean }): Promise<{ stored?: number }>;
  recall(query: string, options?: RecallOptions): Promise<Fact[]>;
  discover(query: string, options?: DiscoverOptions): Promise<Fact[]>;
  causalChain(from: string, to: string, options?: { maxDepth?: number }): Promise<Trace>;
  traceEffects(content: string, options?: { maxDepth?: number }): Promise<Trace>;
  traceCauses(content: string, options?: { maxDepth?: number }): Promise<Trace>;
  chat(messages: ChatMessage[], options?: ChatOptions): Promise<ChatReply>;
  correct(session: string, oldText: string, newText: string): Promise<unknown>;
  history(text: string): Promise<unknown>;
  forgetSession(session: string): Promise<unknown>;
  stats(): Promise<Stats>;
}

export interface MemoryTool {
  name: "remember" | "recall" | "why";
  /** OpenAI-style function definition; most frameworks and gateways accept this shape. */
  definition: {
    type: "function";
    function: { name: string; description: string; parameters: Record<string, unknown> };
  };
  /** Runs the tool and returns text to hand back to the model. */
  execute(args: Record<string, any>): Promise<string>;
}

/** Tool definitions plus their executors, for agents that call functions directly. */
export declare function memoryTools(memory: Memory, session?: string): MemoryTool[];

export default Memory;
