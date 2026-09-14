"""ReasonGraphMemory as a classic LangChain memory, showing what the agent sees each turn.

    pip install "reasongraph[langchain]" langchain-openai
    OPENAI_API_KEY=... python langchain_memory_class.py

Two memory variables reach the model every turn: ``memory`` (the facts the graph connects
to the input, with sources and cause->effect links) and ``history`` (the transcript folded
to a token budget: a rolling summary, then the newest turns verbatim). Folding deletes
nothing; a turn that left the window comes back through ``memory`` when it is relevant.
Swap ReasonGraph() for MemoryClient(api_key=...) to run it against the hosted service.
"""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from reasongraph import ReasonGraph
from reasongraph.integrations.langchain import ReasonGraphMemory
from reasongraph.loop import make_summarizer

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
graph = ReasonGraph()
graph.initialize_sync()
# what the agent knew before this chat: an earlier session with the same team
graph.add_texts_sync(["Mara's team freezes deploys on Fridays.", "Jonas owns the staging cluster.",
                      "Staging runs Kubernetes 1.29."], scopes={"team-notes"})

memory = ReasonGraphMemory(target=graph, session="ops-chat",
                           max_history_tokens=110, keep_tail_tokens=55,   # tiny, so folding shows up
                           summarizer=make_summarizer(lambda msgs: llm.invoke(msgs).content))
SYSTEM = "You are an on-call assistant. Two sentences at most. Use what you remember when it applies."

for user in ["The staging deploy failed last night because the TLS certificate expired, so I rolled back to 2.3.1.",
             "How can we renew that certificate automatically next time?",
             "What should I check before re-running the deploy?",
             "Any tips for making rollbacks faster?",
             "Remind me: which version is staging on now, and why did we roll back?"]:
    seen = memory.load_memory_variables({"input": user})
    context = "\n\n".join(part for part in (seen["memory"], seen["history"] and f"Conversation so far:\n{seen['history']}") if part)
    reply = llm.invoke([SystemMessage(content=SYSTEM), SystemMessage(content=context), HumanMessage(content=user)]).content
    memory.save_context({"input": user}, {"output": reply})
    print(f"\n=== {user}\n[memory]\n{seen['memory'] or '(nothing relevant)'}\n[history]\n{seen['history'] or '(empty)'}\n[reply] {reply}")

print("\n[rolling summary now]", memory.summary)
graph.close_sync()
