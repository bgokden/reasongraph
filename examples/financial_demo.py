"""Financial crisis analysis with reasongraph.

This demo loads the built-in financial dataset and runs queries that
an analyst or agent might ask when researching economic crises.
"""

import textwrap
from reasongraph import ReasonGraph


def print_results(question: str, results: list[str]) -> None:
    print(f"\n{'=' * 70}")
    print(f"Q: {question}")
    print(f"{'=' * 70}")
    for i, text in enumerate(results, 1):
        wrapped = textwrap.fill(text, width=66, initial_indent="  ", subsequent_indent="  ")
        print(f"  [{i}] {text}")
    print()


def main():
    graph = ReasonGraph()
    graph.initialize_sync()
    graph.load_dataset_sync("financial")

    # An analyst starts by asking about the 2008 crisis.
    results = graph.query_sync(
        "What caused the 2008 financial crisis?",
        search_mode="hybrid",
        top_k=5,
        hops=2,
    )
    print_results("What caused the 2008 financial crisis?", results)

    # They want to understand how inflation spiraled in 2021-2022.
    results = graph.query_sync(
        "How did inflation surge after the pandemic?",
        search_mode="hybrid",
        top_k=5,
        hops=2,
    )
    print_results("How did inflation surge after the pandemic?", results)

    # Next: how does the Fed respond to crises?
    results = graph.query_sync(
        "What did the Federal Reserve do about interest rates?",
        search_mode="hybrid",
        top_k=5,
        hops=2,
    )
    print_results("What did the Federal Reserve do about interest rates?", results)

    # The dot-com bubble -- is there a pattern?
    results = graph.query_sync(
        "What happened during the dot-com bubble?",
        search_mode="hybrid",
        top_k=5,
        hops=2,
    )
    print_results("What happened during the dot-com bubble?", results)

    # European sovereign debt -- contagion across borders.
    results = graph.query_sync(
        "How did the Greek debt crisis spread to other countries?",
        search_mode="hybrid",
        top_k=3,
        hops=3,
    )
    print_results("How did the Greek debt crisis spread to other countries?", results)

    graph.close_sync()


if __name__ == "__main__":
    main()
