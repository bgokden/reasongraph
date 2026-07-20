from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Node:
    """A node in the reason graph.

    ``scopes`` is a set of free-text tags (e.g. ``{"user-123", "topic-economy"}``).
    Scopes are labels, not partitions: the graph is shared and traversal crosses
    scopes freely, so a scope only narrows where a query starts, not what it can
    reach. A node may carry any number of scopes, or none.
    """

    content: str
    type: str = "text"
    embedding: list[float] | None = None
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    scopes: set[str] = field(default_factory=set)


@dataclass
class Edge:
    """A directed edge between two nodes."""

    from_content: str
    to_content: str
    last_accessed: datetime = field(default_factory=datetime.now)
