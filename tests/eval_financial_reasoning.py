"""Mixed-domain reasoning evaluation for reasongraph.

Loads ALL built-in datasets (syllogisms, causal, taxonomy, financial) into a
single graph and tests whether the library can still reason correctly across
domains.  This is the realistic scenario: an agent's memory contains diverse
knowledge, and queries must find the right reasoning chain without being
distracted by unrelated facts.

Graph size: ~80 text nodes, ~72 entity nodes, ~150 edges across 4 domains.

Metrics:
    Chain Completeness -- fraction of expected chain nodes found in results
    Recall@K           -- fraction of expected nodes in top K results
    Precision@K        -- fraction of top K results that are expected nodes
    Domain Accuracy    -- results come from the correct domain, not noise

Run:
    uv run python tests/eval_financial_reasoning.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from reasongraph import ReasonGraph
from reasongraph.datasets import AVAILABLE_DATASETS


@dataclass
class ReasoningCase:
    name: str
    domain: str
    agent_thought: str
    expected_chain: list[str]
    search_mode: str = "hybrid"
    hops: int = 3
    top_k: int = 5


# -- Test cases across all four domains --

CASES = [
    # ── Financial: 2008 subprime crisis ──
    ReasoningCase(
        name="2008 crisis: subprime to bailout",
        domain="financial",
        agent_thought="I need to understand what caused the 2008 financial crisis",
        expected_chain=[
            "Banks issued subprime mortgages to borrowers with poor credit histories.",
            "Loose lending standards fueled a housing price bubble across the United States.",
            "Mortgage-backed securities built on subprime loans collapsed when defaults surged.",
            "Lehman Brothers filed for bankruptcy in September 2008 after massive MBS losses.",
            "Lehman's collapse triggered a global credit freeze as interbank lending stopped.",
            "The U.S. government enacted TARP, a $700 billion bailout to stabilize the financial system.",
        ],
    ),
    # ── Financial: dot-com bubble ──
    ReasoningCase(
        name="Dot-com bubble: speculation to market crash",
        domain="financial",
        agent_thought="What happened during the dot-com bubble and how bad was the crash?",
        expected_chain=[
            "Speculative investment poured into internet startups during the late 1990s.",
            "Dozens of unprofitable dot-com companies launched IPOs at extreme valuations.",
            "The NASDAQ index crashed 78% from its March 2000 peak as the bubble burst.",
            "Technology companies laid off hundreds of thousands of workers after the crash.",
            "The market correction wiped out $5 trillion in market value between 2000 and 2002.",
        ],
    ),
    # ── Financial: eurozone debt crisis ──
    ReasoningCase(
        name="Eurozone crisis: Greek debt contagion",
        domain="financial",
        agent_thought="How did the Greek debt crisis spread to other European countries?",
        expected_chain=[
            "Greece revealed its budget deficit was far larger than previously reported.",
            "Credit rating agencies downgraded Greek sovereign debt to junk status.",
            "Investor panic spread to Portugal, Spain, and Italy as borrowing costs spiked.",
            "The ECB launched bond-buying programs to prevent eurozone fragmentation.",
            "Bailout agreements imposed strict austerity measures on affected countries.",
        ],
    ),
    # ── Financial: Lehman entity query ──
    ReasoningCase(
        name="Entity query: Lehman Brothers",
        domain="financial",
        agent_thought="What happened to Lehman Brothers and what were the consequences?",
        expected_chain=[
            "Lehman Brothers filed for bankruptcy in September 2008 after massive MBS losses.",
            "Lehman's collapse triggered a global credit freeze as interbank lending stopped.",
            "The U.S. government enacted TARP, a $700 billion bailout to stabilize the financial system.",
        ],
    ),
    # ── Financial: cross-chain rate hikes ──
    ReasoningCase(
        name="Cross-chain: Federal Reserve rate hikes",
        domain="financial",
        agent_thought="When and why did the Federal Reserve raise interest rates?",
        expected_chain=[
            "The Fed began aggressive rate hikes in 2022, raising rates at the fastest pace in decades.",
            "The Fed raised the federal funds rate from near zero to over 5% within eighteen months.",
        ],
    ),
    # ── Causal: rainfall -> flooding -> evacuation ──
    ReasoningCase(
        name="Causal: heavy rainfall chain",
        domain="causal",
        agent_thought="What are the consequences of heavy rainfall on communities?",
        expected_chain=[
            "Heavy rainfall caused the river to overflow.",
            "The river overflowed its banks, causing flooding in nearby villages.",
            "Flooding submerged roads and houses, forcing residents to evacuate.",
        ],
    ),
    # ── Causal: greenhouse gases -> sea level ──
    ReasoningCase(
        name="Causal: climate change chain",
        domain="causal",
        agent_thought="How do greenhouse gas emissions lead to sea level rise?",
        expected_chain=[
            "Rising greenhouse gas emissions trap more heat in the atmosphere.",
            "Trapped heat causes global average temperatures to rise.",
            "Rising temperatures accelerate polar ice cap melting.",
            "Polar ice melting causes sea levels to rise.",
        ],
    ),
    # ── Causal: antibiotic resistance ──
    ReasoningCase(
        name="Causal: antibiotic resistance",
        domain="causal",
        agent_thought="How does antibiotic overuse lead to higher mortality?",
        expected_chain=[
            "Overuse of antibiotics promotes the survival of resistant bacteria.",
            "Resistant bacteria spread, making standard treatments ineffective.",
            "Ineffective treatments lead to longer hospital stays and higher mortality.",
        ],
    ),
    # ── Syllogisms: Socrates mortality ──
    ReasoningCase(
        name="Syllogism: is Socrates mortal?",
        domain="syllogisms",
        agent_thought="Is Socrates mortal? What is the logical reasoning?",
        expected_chain=[
            "All humans are mortal.",
            "Socrates is a human.",
            "Therefore, Socrates is mortal.",
        ],
    ),
    # ── Syllogisms: dogs warm-blooded ──
    ReasoningCase(
        name="Syllogism: are dogs warm-blooded?",
        domain="syllogisms",
        agent_thought="Are dogs warm-blooded? Show me the logical chain.",
        expected_chain=[
            "All mammals are warm-blooded.",
            "All dogs are mammals.",
            "Therefore, all dogs are warm-blooded.",
        ],
    ),
    # ── Syllogisms: penguins and feathers ──
    ReasoningCase(
        name="Syllogism: do penguins have feathers?",
        domain="syllogisms",
        agent_thought="Do penguins have feathers? What is the deductive proof?",
        expected_chain=[
            "All birds have feathers.",
            "Penguins are birds.",
            "Therefore, penguins have feathers.",
        ],
    ),
    # ── Taxonomy: canine hierarchy ──
    ReasoningCase(
        name="Taxonomy: dog classification",
        domain="taxonomy",
        agent_thought="How are dogs classified in the animal kingdom?",
        expected_chain=[
            "Animals are multicellular organisms that consume organic material.",
            "Mammals are warm-blooded animals that nurse their young with milk.",
            "Canines are mammals in the family Canidae.",
            "Domestic dogs are canines bred as companions.",
        ],
    ),
    # ── Taxonomy: plant hierarchy ──
    ReasoningCase(
        name="Taxonomy: rose classification",
        domain="taxonomy",
        agent_thought="What kind of plant is a rose?",
        expected_chain=[
            "Plants are organisms that produce energy through photosynthesis.",
            "Flowering plants produce seeds enclosed in fruit.",
            "Roses are flowering plants of the genus Rosa.",
        ],
    ),
    # ── Taxonomy: feline species ──
    ReasoningCase(
        name="Taxonomy: cats and lions",
        domain="taxonomy",
        agent_thought="What do domestic cats and lions have in common?",
        expected_chain=[
            "Felines are mammals in the family Felidae.",
            "Domestic cats are small felines kept as pets.",
            "Lions are large felines that live in prides.",
        ],
    ),
    # ── Cross-domain: financial crisis NOT returning biology ──
    ReasoningCase(
        name="Domain isolation: crisis query ignores biology",
        domain="financial",
        agent_thought="What caused the credit freeze in 2008?",
        expected_chain=[
            "Lehman's collapse triggered a global credit freeze as interbank lending stopped.",
            "Lehman Brothers filed for bankruptcy in September 2008 after massive MBS losses.",
        ],
    ),
]


def chain_completeness(results: list[str], expected: list[str]) -> float:
    if not expected:
        return 1.0
    found = sum(1 for node in expected if node in results)
    return found / len(expected)


def recall_at_k(results: list[str], expected: list[str], k: int) -> float:
    if not expected:
        return 1.0
    top_k = set(results[:k])
    found = sum(1 for node in expected if node in top_k)
    return found / len(expected)


def precision_at_k(results: list[str], expected: list[str], k: int) -> float:
    top_k = results[:k]
    if not top_k:
        return 0.0
    relevant = sum(1 for node in top_k if node in expected)
    return relevant / len(top_k)


# All text nodes per domain for domain accuracy checking
DOMAIN_TEXTS = {
    "financial": {
        "Banks issued subprime mortgages to borrowers with poor credit histories.",
        "Loose lending standards fueled a housing price bubble across the United States.",
        "Mortgage-backed securities built on subprime loans collapsed when defaults surged.",
        "Lehman Brothers filed for bankruptcy in September 2008 after massive MBS losses.",
        "Lehman's collapse triggered a global credit freeze as interbank lending stopped.",
        "The U.S. government enacted TARP, a $700 billion bailout to stabilize the financial system.",
        "The Federal Reserve held interest rates near zero for years following the 2008 crisis.",
        "Cheap credit inflated asset prices in equities, real estate, and bonds.",
        "Rising asset prices and fiscal stimulus pushed consumer inflation above target.",
        "The Fed began aggressive rate hikes in 2022, raising rates at the fastest pace in decades.",
        "Higher interest rates tightened credit conditions and slowed corporate borrowing.",
        "Speculative investment poured into internet startups during the late 1990s.",
        "Dozens of unprofitable dot-com companies launched IPOs at extreme valuations.",
        "The NASDAQ index crashed 78% from its March 2000 peak as the bubble burst.",
        "Technology companies laid off hundreds of thousands of workers after the crash.",
        "The market correction wiped out $5 trillion in market value between 2000 and 2002.",
        "Pandemic stimulus checks and expanded unemployment benefits boosted consumer spending.",
        "Global supply chains broke down due to factory shutdowns and shipping bottlenecks.",
        "Surging demand against constrained supply created a demand-supply imbalance.",
        "U.S. consumer inflation reached 9.1% in June 2022, a 40-year high.",
        "The Fed raised the federal funds rate from near zero to over 5% within eighteen months.",
        "Rising yields caused the worst bond market selloff in a generation.",
        "Greece revealed its budget deficit was far larger than previously reported.",
        "Credit rating agencies downgraded Greek sovereign debt to junk status.",
        "Investor panic spread to Portugal, Spain, and Italy as borrowing costs spiked.",
        "The ECB launched bond-buying programs to prevent eurozone fragmentation.",
        "Bailout agreements imposed strict austerity measures on affected countries.",
    },
    "causal": {
        "Heavy rainfall caused the river to overflow.",
        "The river overflowed its banks, causing flooding in nearby villages.",
        "Flooding submerged roads and houses, forcing residents to evacuate.",
        "Deforestation reduced the tree cover on hillsides.",
        "Reduced tree cover caused increased soil erosion.",
        "Increased soil erosion led to landslides during heavy rains.",
        "Rising greenhouse gas emissions trap more heat in the atmosphere.",
        "Trapped heat causes global average temperatures to rise.",
        "Rising temperatures accelerate polar ice cap melting.",
        "Polar ice melting causes sea levels to rise.",
        "A factory released untreated waste into the river.",
        "Untreated waste contaminated the river water supply.",
        "Contaminated water caused illness among downstream communities.",
        "Overuse of antibiotics promotes the survival of resistant bacteria.",
        "Resistant bacteria spread, making standard treatments ineffective.",
        "Ineffective treatments lead to longer hospital stays and higher mortality.",
    },
    "syllogisms": {
        "All humans are mortal.",
        "Socrates is a human.",
        "Therefore, Socrates is mortal.",
        "All mammals are warm-blooded.",
        "All dogs are mammals.",
        "Therefore, all dogs are warm-blooded.",
        "All birds have feathers.",
        "Penguins are birds.",
        "Therefore, penguins have feathers.",
        "All planets orbit a star.",
        "Earth is a planet.",
        "Therefore, Earth orbits a star.",
        "All even numbers are divisible by two.",
        "Four is an even number.",
        "Therefore, four is divisible by two.",
    },
    "taxonomy": {
        "A living organism is any entity that exhibits life processes.",
        "Animals are multicellular organisms that consume organic material.",
        "Plants are organisms that produce energy through photosynthesis.",
        "Mammals are warm-blooded animals that nurse their young with milk.",
        "Reptiles are cold-blooded animals with scales.",
        "Birds are warm-blooded animals with feathers that lay eggs.",
        "Canines are mammals in the family Canidae.",
        "Felines are mammals in the family Felidae.",
        "Domestic dogs are canines bred as companions.",
        "Wolves are wild canines that live in packs.",
        "Domestic cats are small felines kept as pets.",
        "Lions are large felines that live in prides.",
        "Flowering plants produce seeds enclosed in fruit.",
        "Conifers are non-flowering plants that bear cones.",
        "Roses are flowering plants of the genus Rosa.",
        "Oak trees are flowering plants of the genus Quercus.",
        "Pine trees are conifers of the genus Pinus.",
    },
}


def domain_accuracy(results: list[str], domain: str) -> float:
    """Fraction of returned results that belong to the expected domain."""
    if not results:
        return 0.0
    correct_domain = DOMAIN_TEXTS.get(domain, set())
    in_domain = sum(1 for r in results if r in correct_domain)
    return in_domain / len(results)


def run_evaluation():
    # Load all datasets into one graph
    graph = ReasonGraph()
    graph.initialize_sync()
    for ds in AVAILABLE_DATASETS:
        graph.load_dataset_sync(ds)

    all_nodes = graph._run(graph.get_all_nodes())
    text_count = sum(1 for n in all_nodes if n.type == "text")
    entity_count = sum(1 for n in all_nodes if n.type == "entity")
    all_edges = graph._run(graph.get_all_edges())
    print(f"Graph loaded: {text_count} text nodes, {entity_count} entity nodes, {len(all_edges)} edges")
    print(f"Datasets: {', '.join(AVAILABLE_DATASETS)}")
    print()

    results_table = []
    totals = {"completeness": 0, "r5": 0, "r3": 0, "p5": 0, "domain": 0}

    for case in CASES:
        results = graph.query_sync(
            case.agent_thought,
            search_mode=case.search_mode,
            top_k=case.top_k,
            hops=case.hops,
        )

        comp = chain_completeness(results, case.expected_chain)
        r5 = recall_at_k(results, case.expected_chain, 5)
        r3 = recall_at_k(results, case.expected_chain, 3)
        p5 = precision_at_k(results, case.expected_chain, 5)
        da = domain_accuracy(results, case.domain)

        totals["completeness"] += comp
        totals["r5"] += r5
        totals["r3"] += r3
        totals["p5"] += p5
        totals["domain"] += da

        results_table.append({
            "name": case.name,
            "domain": case.domain,
            "chain_completeness": comp,
            "recall@5": r5,
            "recall@3": r3,
            "precision@5": p5,
            "domain_accuracy": da,
            "returned": len(results),
            "expected": len(case.expected_chain),
            "found": sum(1 for n in case.expected_chain if n in results),
        })

    graph.close_sync()

    n = len(CASES)

    # Detailed results
    print(f"{'=' * 80}")
    print(f"  Mixed-Domain Reasoning Evaluation -- {n} cases, 4 domains")
    print(f"{'=' * 80}")
    print()

    current_domain = None
    for r in results_table:
        if r["domain"] != current_domain:
            current_domain = r["domain"]
            print(f"  --- {current_domain.upper()} ---")

        status = "PASS" if r["chain_completeness"] >= 0.5 else "FAIL"
        print(f"  [{status}] {r['name']}")
        print(f"         Chain: {r['found']}/{r['expected']} ({r['chain_completeness']:.0%})"
              f"  R@5: {r['recall@5']:.0%}"
              f"  P@5: {r['precision@5']:.0%}"
              f"  Domain: {r['domain_accuracy']:.0%}")
    print()

    # Per-domain averages
    domains = sorted(set(r["domain"] for r in results_table))
    print(f"{'=' * 80}")
    print(f"  Per-Domain Averages")
    print(f"{'=' * 80}")
    print(f"  {'Domain':<14s} {'Cases':>5s} {'Chain':>7s} {'R@5':>7s} {'P@5':>7s} {'Domain%':>8s}")
    print(f"  {'-'*14} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")

    for domain in domains:
        dr = [r for r in results_table if r["domain"] == domain]
        dn = len(dr)
        print(f"  {domain:<14s} {dn:>5d}"
              f" {sum(r['chain_completeness'] for r in dr)/dn:>6.0%}"
              f" {sum(r['recall@5'] for r in dr)/dn:>6.0%}"
              f" {sum(r['precision@5'] for r in dr)/dn:>6.0%}"
              f" {sum(r['domain_accuracy'] for r in dr)/dn:>7.0%}")
    print()

    # Overall averages
    print(f"{'=' * 80}")
    print(f"  Overall Averages ({n} cases across {len(domains)} domains)")
    print(f"{'=' * 80}")
    print(f"  Chain Completeness : {totals['completeness']/n:.1%}")
    print(f"  Recall@5           : {totals['r5']/n:.1%}")
    print(f"  Recall@3           : {totals['r3']/n:.1%}")
    print(f"  Precision@5        : {totals['p5']/n:.1%}")
    print(f"  Domain Accuracy    : {totals['domain']/n:.1%}")
    print(f"  Cases >= 50%       : {sum(1 for r in results_table if r['chain_completeness'] >= 0.5)}/{n}")
    print(f"{'=' * 80}")

    # Search mode comparison
    print()
    print(f"{'=' * 80}")
    print(f"  Search Mode Comparison")
    print(f"{'=' * 80}")

    graph2 = ReasonGraph()
    graph2.initialize_sync()
    for ds in AVAILABLE_DATASETS:
        graph2.load_dataset_sync(ds)

    print(f"  {'Mode':<12s} {'Chain':>7s} {'R@5':>7s} {'P@5':>7s} {'Domain%':>8s}")
    print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")

    for mode in ("embedding", "keyword", "hybrid"):
        m_comp, m_r5, m_p5, m_da = 0.0, 0.0, 0.0, 0.0
        for case in CASES:
            results = graph2.query_sync(
                case.agent_thought,
                search_mode=mode,
                top_k=case.top_k,
                hops=case.hops,
            )
            m_comp += chain_completeness(results, case.expected_chain)
            m_r5 += recall_at_k(results, case.expected_chain, 5)
            m_p5 += precision_at_k(results, case.expected_chain, 5)
            m_da += domain_accuracy(results, case.domain)

        print(f"  {mode:<12s}"
              f" {m_comp/n:>6.0%}"
              f" {m_r5/n:>6.0%}"
              f" {m_p5/n:>6.0%}"
              f" {m_da/n:>7.0%}")

    graph2.close_sync()
    print(f"{'=' * 80}")


if __name__ == "__main__":
    run_evaluation()
