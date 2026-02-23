from __future__ import annotations

import json
from importlib import resources

AVAILABLE_DATASETS = ("syllogisms", "causal", "taxonomy", "financial", "medical", "analysis_patterns")


def load_dataset(name: str) -> dict:
    """Load a built-in dataset by name.

    Args:
        name: One of 'syllogisms', 'causal', 'taxonomy', 'financial', 'medical', 'analysis_patterns'.

    Returns:
        Dict with 'name', 'description', 'nodes', and 'edges' keys.

    Raises:
        ValueError: If the dataset name is not recognized.
    """
    if name not in AVAILABLE_DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {', '.join(AVAILABLE_DATASETS)}"
        )

    ref = resources.files("reasongraph.datasets").joinpath(f"{name}.json")
    data = json.loads(ref.read_text(encoding="utf-8"))
    return data
